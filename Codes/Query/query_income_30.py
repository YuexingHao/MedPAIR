import pandas as pd
import re
import time
import os

# Setup OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Output file path
output_path = "MedPAIREquity/Results/income_30200_last_sentence_data.csv"
df = pd.read_csv("MedPAIREquity/perturbations/income_30200_last_sentence_data.csv")

# Create output file with header if it doesn't exist
if not os.path.exists(output_path):
    header_cols = df.columns.tolist() + ['LLM_answer'] + [f"label_{i+1}" for i in range(21)]
    pd.DataFrame(columns=header_cols).to_csv(output_path, index=False)

# Iterate through each row
for i in range(len(df)): 
    try:
        sentences = re.split(r'\n\d+\.\s*', df.loc[i, "race_injected_sentence"])
        print(len(sentences))
        options = df.loc[i, "question_options"]

        formatted_sentences = "\n".join(f"{j+1}. {s.strip()}" for j, s in enumerate(sentences))

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
        answer_match = re.search(r"Answer:\s*([A-E])", content, re.IGNORECASE)
        gpt_answer = answer_match.group(1).upper() if answer_match else None

        # Extract sentence-level relevance labels
        labels = re.findall(r"\[\s*(High Relevance|Low Relevance|Irrelevant)\s*\]", content, re.IGNORECASE)
        labels = [label.title() for label in labels]
        labels = labels[:21] + [None] * (21 - len(labels))  

    except Exception as e:
        print(f"Error at row {i}: {e}")
        gpt_answer = None
        labels = [None] * 21

    row_data = pd.concat(
        [df.iloc[[i]].reset_index(drop=True),
         pd.DataFrame([[gpt_answer] + labels], columns=['GPT5_answer'] + [f"label_{j+1}" for j in range(21)])
        ],
        axis=1
    )
    row_data.to_csv(output_path, mode='a', index=False, header=False)

    print(f"Row {i} processed and saved.")
    time.sleep(0.2) 
