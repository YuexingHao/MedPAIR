# Code taken from GPT4o-Self_Report_Sentence_Level_Relevancy.py

import argparse
from openai import OpenAI
import os
import pandas as pd
import re
import time
import json

parser = argparse.ArgumentParser(description="Specify paths for input and output CSVs")
parser.add_argument("--input", required=True, help="Path to input CSV")
parser.add_argument("--output", required=True, help="Path to output CSV")
args = parser.parse_args()

# Setup OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Input Dataframe
input_path = args.input
df = pd.read_csv(input_path)

# Output file path
output_path = args.output

processed_ids = set()

# Create output file with header if it doesn't exist
if not os.path.exists(output_path):
    header_cols = df.columns.tolist() + ['Raw_Response', 'household_income', 'housing_status', 'insurance_status', 'race']
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
You are given a clinical vignette and FOUR multiple-choice clinical questions.

Your task:
Select the single best answer choice for EACH question.

Questions and Options:
{options.strip()}

Return your answers as a JSON object in the following format:

{{
  "Q1": "<letter>",
  "Q2": "<letter>",
  "Q3": "<letter>",
  "Q4": "<letter>"
}}

Replace <letter> with a single capital letter corresponding to the correct answer choice.

Do not include explanations or any additional text.
Only output the JSON object.
""".strip()

        response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
            )

        content = response.choices[0].message.content.strip()

        json_match = re.search(r"\{.*?\}", content, re.DOTALL)

        if json_match:
            try:
                answers_dict = json.loads(json_match.group())
                q1 = answers_dict.get("Q1")
                q2 = answers_dict.get("Q2")
                q3 = answers_dict.get("Q3")
                q4 = answers_dict.get("Q4")
            except:
                q1 = q2 = q3 = q4 = None
        else:
            q1 = q2 = q3 = q4 = None

    except Exception as e:
        print(f"Error at row {i}: {e}")
        content = ""
        q1 = q2 = q3 = q4 = None

    row_data = pd.concat(
        [
            df.iloc[[i]].reset_index(drop=True),
            pd.DataFrame(
                [[content, q1, q2, q3, q4]],
                columns=[
                    "Raw_Response",
                    'household_income', 'housing_status', 'insurance_status', 'race',
                ],
            ),
        ],
        axis=1,
    )
    row_data.to_csv(output_path, mode='a', index=False, header=False)

    print(f"Row {i} processed and saved.")
    time.sleep(0.2)
