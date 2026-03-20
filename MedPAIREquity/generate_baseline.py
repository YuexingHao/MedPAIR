import pandas as pd
import re

original_df = pd.read_csv('1300_QA.csv')

filtered_df = pd.DataFrame(columns=[
    "ID_corr", "centaur_question_corr", "sentence_number_corr",
    "answer_corr", "data_source_corr", "original_sentences",
    "question_options"
])

# Income-related keywords
income_terms = [
    "income", "salary", "wage", "earn", "earning", "annual income",
    "yearly income", "monthly income", "household income",
    "low income", "middle income", "high income",
    "poverty", "below poverty", "above poverty",
    "financial status", "socioeconomic", "wealth", "rich", "poor",
    "afford", "cost", "expensive", "cheap", "financial situation",
    "unemployed", "employment", "jobless", "minimum wage",
    "medicaid", "medicare", "insurance coverage"
]

for _, row in original_df.iterrows():
    original_sentences = row["original_sentences"]
    original_lower_case = original_sentences.lower()

    first_sentence = original_lower_case.split("\n")[0]

    # Check for income-related terms
    income_found = False
    for term in income_terms:
        if term in original_lower_case:
            income_found = True
            break

    if income_found:
        continue

    filtered_df.loc[len(filtered_df)] = row.copy()

filtered_df.to_csv("MedPAIREquity/baseline_data_income_filtered.csv", index=False)