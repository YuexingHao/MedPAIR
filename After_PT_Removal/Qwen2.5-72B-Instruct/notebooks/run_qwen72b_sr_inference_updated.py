#!/usr/bin/env python3
"""
Generate Qwen2.5-72B-Instruct SR predictions using reduced-context setup.

Input: /home/yuexing/NeuRIPS25/After_PT_Removal/SR_Predictions/Llama-70B_Removed/Llama_70B_[SR]_predictions.csv
Context Column: [SR]High (reduced-relevance context)
Question Column: question_options (multiple choice question)

Output: Updates qwen72b_direct_prediction column in
  /home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/[SR]_Qwen72B_predictions_on_llama70b_removed.csv
"""

import sys
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
INPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/SR_Predictions/Llama-70B_Removed/Llama_70B_[SR]_predictions.csv")
OUTPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/[SR]_Qwen72B_predictions_on_llama70b_removed.csv")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

def extract_letter(pred_text):
    """Extract letter from prediction response.

    Fix: Extract substring inside <answer>...</answer> FIRST (anchored),
    then pull letter from that substring only.
    """
    if not isinstance(pred_text, str):
        return None

    # Step 1: Extract content inside <answer>...</answer> tags (anchored)
    answer_match = re.search(r"<answer>(.*?)</answer>", pred_text, re.IGNORECASE | re.DOTALL)
    if not answer_match:
        return None

    answer_content = answer_match.group(1).strip()

    # Step 2: Extract letter from that content only
    letter_match = re.search(r"Option\s+\[?([A-J])\]?", answer_content, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()

    return None

def generate_prediction(context, question, model, tokenizer, seed=42):
    """Generate prediction using local model with fixed sampling.

    Fixed sampling for deterministic, reproducible results:
    - temperature=0.0: deterministic (greedy decoding)
    - top_p=1.0: no nucleus sampling
    - do_sample=False: greedy, no randomness
    - seed: fixed for reproducibility
    """
    prompt = f"""You are given some context and a multiple-choice question.

Select the most appropriate answer from the options provided.

{context}

{question}

Provide your response in the following format:\n<answer>Option [letter]</answer>"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Set seed for reproducibility
        torch.manual_seed(seed)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,      # Deterministic (greedy)
                top_p=1.0,            # No nucleus sampling
                do_sample=False,      # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the response part after the prompt
        if len(response) > len(prompt):
            response = response[len(prompt):].strip()

        # Check for empty or truncated responses
        if not response or len(response) < 5:
            return "Error: empty_response"

        return response
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("="*80)
    print("QWEN2.5-72B-INSTRUCT SR INFERENCE - REDUCED CONTEXT")
    print("="*80)
    print(f"Start Time: {datetime.now()}")
    print(f"Input File: {INPUT_FILE}")
    print(f"Output File: {OUTPUT_FILE}")
    print()

    # Load input data
    print("Loading input data...")
    df_input = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"  • Loaded {len(df_input)} rows")

    # Load output file (target to update)
    print("Loading target file...")
    df_output = pd.read_csv(OUTPUT_FILE, low_memory=False)
    print(f"  • Loaded {len(df_output)} rows")

    # Check for resume: filter out error rows to retry them
    has_errors = df_output["qwen72b_direct_prediction"].notna() & \
                 df_output["qwen72b_direct_prediction"].astype(str).str.startswith("Error", na=False)
    if has_errors.any():
        n_errors_to_retry = has_errors.sum()
        print(f"  • Found {n_errors_to_retry} error rows from previous run (will retry)")

    # Check if Origin column exists in both
    if "Origin" not in df_input.columns:
        print("ERROR: 'Origin' column not found in input file")
        sys.exit(1)
    if "Origin" not in df_output.columns:
        print("ERROR: 'Origin' column not found in output file")
        sys.exit(1)

    # Check input columns
    required_cols = ["[SR]High", "question_options", "Origin"]
    for col in required_cols:
        if col not in df_input.columns:
            print(f"ERROR: Required column '{col}' not found in input file")
            print(f"Available columns: {list(df_input.columns)}")
            sys.exit(1)

    print("  • Required columns found")
    print()

    # Load model
    print(f"Loading model: {MODEL_ID}")
    print("  • This may take 2-3 minutes...")

    try:
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        sys.exit(1)

    print()
    print("="*80)
    print(f"Running inference on {len(df_input)} samples...")
    print("="*80)
    print()

    # Run inference with detailed tracking
    predictions = []
    n_valid = 0
    n_error = 0
    n_unparsed = 0

    for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
        try:
            context = str(row.get("[SR]High", "")).strip()
            question = str(row.get("question_options", "")).strip()

            if not context or not question:
                predictions.append(None)
                n_unparsed += 1
                continue

            # Generate prediction
            response = generate_prediction(context, question, model, tokenizer)

            # Check for error responses
            if isinstance(response, str) and response.startswith("Error"):
                predictions.append(response)  # Keep error message for diagnostics
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
    df_input["qwen72b_direct_prediction"] = predictions

    # Merge predictions back to output file based on Origin
    print(f"Merging predictions ({n_valid} valid predictions)...")

    # Create mapping from input: ONLY include valid predictions (filter out errors and unparsed)
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
        current_value = row.get("qwen72b_direct_prediction")

        # Skip rows that have error values (will retry them)
        if isinstance(current_value, str) and current_value.startswith("Error"):
            skipped_error_rows += 1

        # Only update if we have a valid new prediction
        if origin in valid_predictions:
            df_output.at[idx, "qwen72b_direct_prediction"] = valid_predictions[origin]
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
        # Merge predictions back to input for comparison with correct answers
        df_input["qwen72b_extracted"] = predictions
        # Count correct predictions (those that are not None/Error/unparsed)
        df_input_valid = df_input[df_input["qwen72b_extracted"].notna() &
                                   ~df_input["qwen72b_extracted"].str.startswith("Error", na=False)]
        accuracy_pct = (df_input_valid["qwen72b_extracted"] == df_input_valid.get("answer_corr", "")).sum() / len(df_input_valid) * 100 if len(df_input_valid) > 0 else 0
    else:
        accuracy_pct = 0

    # Print summary with visible denominator
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples:           {len(df_input)}")
    print(f"Valid predictions:       {n_valid}")
    print(f"Unparseable responses:   {n_unparsed}")
    print(f"Error/exception count:   {n_error}")
    print(f"Accuracy (valid only):   {accuracy_pct:.1f}% ({n_valid} samples)")
    print(f"Coverage:                {n_valid / len(df_input) * 100:.1f}% ({n_valid}/{len(df_input)})")
    print(f"Output rows updated:     {updated_count}")
    print(f"End Time:                {datetime.now()}")
    print("="*80)

if __name__ == "__main__":
    main()
