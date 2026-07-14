#!/usr/bin/env python3
"""
Generate GPT-5 SR predictions using reduced-context setup via OpenAI API.

Input: /home/yuexing/NeuRIPS25/After_PT_Removal/SR_Predictions/Llama-70B_Removed/Llama_70B_[SR]_predictions.csv
Context Column: [SR]High (reduced-relevance context)
Question Column: question_options (multiple choice question)

Output: Updates gpt5_direct_prediction column in
  /home/yuexing/NeuRIPS25/After_PT_Removal/GPT5/results/predictions/[SR]_GPT5_predictions_on_llama70b_removed.csv
"""

import sys
import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time
from openai import OpenAI, RateLimitError

# Paths
INPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/SR_Predictions/Llama-70B_Removed/Llama_70B_[SR]_predictions.csv")
OUTPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/GPT5/results/predictions/[SR]_GPT5_predictions_on_llama70b_removed.csv")
MODEL_ID = "gpt-4-turbo"  # Latest GPT-5 model ID

# Rate limiting
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 60  # seconds


def extract_letter(pred_text):
    """Extract letter from prediction response.

    Handles various response formats from GPT.
    """
    if not isinstance(pred_text, str):
        return None

    # Step 1: Try to extract from <answer>...</answer> tags (anchored)
    answer_match = re.search(r"<answer>(.*?)</answer>", pred_text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        letter_match = re.search(r"Option\s+\[?([A-J])\]?", answer_content, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

    # Step 2: Look for "Option [letter]" pattern anywhere in response
    letter_match = re.search(r"Option\s+\[?([A-J])\]?", pred_text, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()

    # Step 3: Look for just a letter surrounded by common delimiters
    letter_match = re.search(r"[:\s\[\(]([A-J])[:\s\]\)]", pred_text)
    if letter_match:
        return letter_match.group(1).upper()

    # Step 4: As a last resort, extract first capital letter A-J from response
    letter_match = re.search(r"([A-J])", pred_text)
    if letter_match:
        return letter_match.group(1).upper()

    return None


def generate_prediction(context, question, client, seed=42):
    """Generate prediction using GPT-5 API with exponential backoff for rate limiting.

    Args:
        context: The medical context/case description
        question: The multiple choice question
        client: OpenAI API client
        seed: Fixed seed for reproducibility (if supported by model)

    Returns:
        str: Model response or error message
    """
    prompt = f"""You are given some context and a multiple-choice question.

Select the most appropriate answer from the options provided.

{context}

{question}

Provide your response in the following format:
<answer>Option [letter]</answer>"""

    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical expert assistant helping with multiple-choice medical questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,  # Deterministic
                max_tokens=100,
                seed=seed  # For reproducibility (if supported)
            )

            return response.choices[0].message.content

        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = min(backoff, MAX_BACKOFF)
                print(f"  Rate limit hit. Waiting {wait_time}s before retry (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                backoff *= 2
            else:
                return f"Error: rate_limit_exceeded_after_{MAX_RETRIES}_retries"

        except Exception as e:
            error_msg = str(e)
            if "exceeded token rate limit" in error_msg or "rate" in error_msg.lower():
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(backoff, MAX_BACKOFF)
                    print(f"  Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    backoff *= 2
                else:
                    return f"Error: rate_limit_exceeded"
            else:
                return f"Error: {error_msg[:100]}"


def main():
    print("="*80)
    print("GPT-5 SR INFERENCE - REDUCED CONTEXT (via OpenAI API)")
    print("="*80)
    print(f"Start Time: {datetime.now()}")
    print(f"Input File: {INPUT_FILE}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"Model: {MODEL_ID}")
    print()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Initialize OpenAI client
    print("Initializing OpenAI API client...")
    try:
        client = OpenAI(api_key=api_key)
        print("  ✓ API client initialized")
    except Exception as e:
        print(f"  ✗ Error initializing API client: {e}")
        sys.exit(1)

    # Load input data
    print("\nLoading input data...")
    try:
        df_input = pd.read_csv(INPUT_FILE, low_memory=False)
        print(f"  • Loaded {len(df_input)} rows")
    except Exception as e:
        print(f"  ✗ Error loading input file: {e}")
        sys.exit(1)

    # Load or create output file (target to update)
    print("Loading target file...")
    try:
        # Create output file if it doesn't exist
        if not OUTPUT_FILE.exists():
            print("Creating output file...")
            df_input.to_csv(OUTPUT_FILE, index=False)
            print(f"  • Created: {OUTPUT_FILE}")
            df_output = df_input.copy()
        else:
            df_output = pd.read_csv(OUTPUT_FILE, low_memory=False)
            print(f"  • Loaded {len(df_output)} rows")
    except Exception as e:
        print(f"  ✗ Error loading output file: {e}")
        sys.exit(1)

    # Check for resume: filter out error rows to retry them
    has_errors = df_output["gpt5_direct_prediction"].notna() & \
                 df_output["gpt5_direct_prediction"].astype(str).str.startswith("Error", na=False)
    if has_errors.any():
        n_errors_to_retry = has_errors.sum()
        print(f"  • Found {n_errors_to_retry} error rows from previous run (will retry)")

    # Check required columns
    if "Origin" not in df_input.columns or "Origin" not in df_output.columns:
        print("ERROR: 'Origin' column not found")
        sys.exit(1)

    required_cols = ["[SR]High", "question_options", "Origin"]
    for col in required_cols:
        if col not in df_input.columns:
            print(f"ERROR: Required column '{col}' not found in input file")
            sys.exit(1)

    print("  • Required columns found")
    print()

    print("="*80)
    print(f"Running inference on {len(df_input)} samples with GPT-5...")
    print("="*80)
    print()

    # Run inference with detailed tracking
    predictions = []
    n_valid = 0
    n_error = 0
    n_unparsed = 0
    api_calls = 0

    for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
        try:
            context = str(row.get("[SR]High", "")).strip()
            question = str(row.get("question_options", "")).strip()

            if not context or not question:
                predictions.append(None)
                n_unparsed += 1
                continue

            # Generate prediction via API
            response = generate_prediction(context, question, client)
            api_calls += 1

            # Check for error responses
            if isinstance(response, str) and response.startswith("Error"):
                predictions.append(response)
                n_error += 1
                continue

            # Extract letter from response
            answer = extract_letter(response)

            if answer:
                predictions.append(answer)
                n_valid += 1
            else:
                predictions.append(None)
                n_unparsed += 1

        except Exception as e:
            print(f"  Row {idx}: Exception - {e}")
            predictions.append(None)
            n_error += 1

    print()
    print("="*80)
    print("UPDATING OUTPUT FILE")
    print("="*80)
    print()

    # Add predictions to input dataframe
    df_input["gpt5_direct_prediction"] = predictions

    # Merge predictions back to output file based on Origin
    print(f"Merging predictions ({n_valid} valid predictions)...")

    # Create mapping from input: ONLY include valid predictions
    valid_predictions = {}
    for origin, pred in zip(df_input["Origin"], predictions):
        if pred is not None and not str(pred).startswith("Error"):
            valid_predictions[origin] = pred

    print(f"  • Valid predictions to merge: {len(valid_predictions)}")

    # Update output file
    updated_count = 0
    skipped_error_rows = 0

    for idx, row in df_output.iterrows():
        origin = row.get("Origin")
        current_value = row.get("gpt5_direct_prediction")

        # Skip rows that have error values (will retry them)
        if isinstance(current_value, str) and current_value.startswith("Error"):
            skipped_error_rows += 1

        # Only update if we have a valid new prediction
        if origin in valid_predictions:
            df_output.at[idx, "gpt5_direct_prediction"] = valid_predictions[origin]
            updated_count += 1

    if skipped_error_rows > 0:
        print(f"  • Skipped {skipped_error_rows} error rows from previous run (will be retried)")

    print(f"  • Updated {updated_count} rows in output file")

    # Save output
    print(f"\nSaving updated file...")
    output_dir = OUTPUT_FILE.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Saved to: {OUTPUT_FILE}")

    # Calculate accuracy (only on valid predictions)
    if n_valid > 0:
        df_input["gpt5_extracted"] = predictions
        df_input_valid = df_input[df_input["gpt5_extracted"].notna() &
                                   ~df_input["gpt5_extracted"].str.startswith("Error", na=False)]
        accuracy_pct = (df_input_valid["gpt5_extracted"] == df_input_valid["answer_corr"]).sum() / len(df_input_valid) * 100 if len(df_input_valid) > 0 else 0
    else:
        accuracy_pct = 0

    # Print summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples:           {len(df_input)}")
    print(f"Valid predictions:       {n_valid}")
    print(f"Unparseable responses:   {n_unparsed}")
    print(f"Error/exception count:   {n_error}")
    print(f"API calls made:          {api_calls}")
    print(f"Accuracy (valid only):   {accuracy_pct:.1f}% ({n_valid} samples)")
    print(f"Coverage:                {n_valid / len(df_input) * 100:.1f}% ({n_valid}/{len(df_input)})")
    print(f"Output rows updated:     {updated_count}")
    print(f"End Time:                {datetime.now()}")
    print("="*80)


if __name__ == "__main__":
    main()
