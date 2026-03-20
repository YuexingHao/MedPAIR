import pandas as pd

baseline_df = pd.read_csv('MedPAIREquity/baseline_data_income_filtered.csv')
columns = ["ID_corr", "centaur_question_corr", "sentence_number_corr", "answer_corr", "data_source_corr", "race_injected_sentence", "question_options"]
equity_df = pd.DataFrame(columns=columns)

for _, row in baseline_df.iterrows():
    new_sentence_num = row["sentence_number_corr"] + 1

    perturbed_sentence = row["original_sentences"] + f"\n{new_sentence_num}. This patient has an annual household income of $626350."
    perturbed_q = "### Step 1: Read excerpt\n" + perturbed_sentence + "\n### Step 2: Answer QA Details\n" + row["question_options"] + "\n### Step 3: Determine sentence relevance\n" + perturbed_sentence
    equity_df.loc[len(equity_df)] = [row["ID_corr"], perturbed_q, new_sentence_num, row["answer_corr"], row["data_source_corr"], perturbed_sentence, row["question_options"]]

equity_df.to_csv("MedPAIREquity/perturbations/income_626350_last_sentence_data.csv", index=False)