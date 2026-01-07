# Code taken from GPT4o-Self_Report_Sentence_Level_Relevancy.py

import argparse
from openai import OpenAI
import os
import pandas as pd
import re
import time

parser = argparse.ArgumentParser(description="Specify paths for input and output CSVs")
parser.add_argument("--input", required=True, help="Path to input CSV")
parser.add_argument("--output", required=True, help="Path to output CSV")
args = parser.parse_args()

# Setup OpenAI client
client = OpenAI(api_key="TODO: FILL YOUR GPT KEY HERE")

# Input Dataframe
input_path = args.input
df = pd.read_csv(input_path)

# Output file path
output_path = args.output

processed_ids = set()

# Create output file with header if it doesn't exist
if not os.path.exists(output_path):
    header_cols = df.columns.tolist() + ['Raw_Response', 'GPT5_answer'] + [f"label_{i+1}" for i in range(30)]
    pd.DataFrame(columns=header_cols).to_csv(output_path, index=False)
else:
    try:
        existing_df = pd.read_csv(output_path)
        processed_ids_arr = existing_df["ID_corr"].unique()
        for id in processed_ids_arr:
            processed_ids.add(str(id))
    except:
        pass

# Iterate through each row
for i in range(len(df)): 
    row_id = str(df.loc[i, "ID_corr"])
    if row_id in processed_ids:
        print(f"Found {row_id} in existing processed IDs. Skipping.")
        continue

    try:
        formatted_sentences = df.loc[i, "original_sentences"]
        options = df.loc[i, "question_options"]

        prompt = f"""
You are given a list of sentences from a clinical vignette and a multiple-choice clinical question. 

Your task is twofold:
(1) Select the most appropriate answer from the given options.
(2) Label each sentence as either [High Relevance], [Low Relevance], or [Irrelevant], based on its contribution to answering the question.

Definitions:
[High Relevance]: Sentences that directly support the correct answer with essential clinical information (e.g., diagnosis, key test results).
[Low Relevance]: Sentences that provide useful context or background but are not critical to answering.
[Irrelevant]: Sentences unrelated to the question or not useful for reasoning.

Question and Options:
{options.strip()}

Sentences:
{formatted_sentences}

Please provide your answer selection first (e.g., "Answer: B"), followed by the relevance label for each sentence in order.
""".strip()

        response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
            )

        content = response.choices[0].message.content.strip()

        # Extract GPT5 answer
        answer_match = re.search(r"Answer:\s*([A-J])", content, re.IGNORECASE)
        gpt_answer = answer_match.group(1).upper() if answer_match else None

        # Extract sentence-level relevance labels
        labels = re.findall(r"\[\s*(High Relevance|Low Relevance|Irrelevant)\s*\]", content, re.IGNORECASE)
        labels = [label.title() for label in labels]
        labels = labels[:30] + [None] * (30 - len(labels))  

    except Exception as e:
        print(f"Error at row {i}: {e}")
        content = ""
        gpt_answer = None
        labels = [None] * 30

    row_data = pd.concat(
        [df.iloc[[i]].reset_index(drop=True),
         pd.DataFrame([[content, gpt_answer] + labels], columns=['Raw_Response', 'GPT5_answer'] + [f"label_{j+1}" for j in range(30)])
        ],
        axis=1
    )
    row_data.to_csv(output_path, mode='a', index=False, header=False)

    print(f"Row {i} processed and saved.")
    time.sleep(0.2)