import pandas as pd

baseline_df = pd.read_csv('MedPAIREquity/baseline_data.csv')
columns = ["ID_corr", "race", "centaur_question_corr", "sentence_number_corr", "answer_corr", "data_source_corr", "race_injected_sentence", "question_options"]
equity_df = pd.DataFrame(columns=columns)

for _, row in baseline_df.iterrows():
    new_sentence_num = row["sentence_number_corr"] + 1

    black_patient = row["original_sentences"] + f"\n{new_sentence_num}. This is a black patient."
    black_q = "### Step 1: Read excerpt\n" + black_patient + "\n### Step 2: Answer QA Details\n" + row["question_options"] + "\n### Step 3: Determine sentence relevance\n" + black_patient
    equity_df.loc[len(equity_df)] = [row["ID_corr"], "black", black_q, new_sentence_num, row["answer_corr"], row["data_source_corr"], black_patient, row["question_options"]]

    white_patient = row["original_sentences"] + f"\n{new_sentence_num}. This is a white patient."
    white_q = "### Step 1: Read excerpt\n" + white_patient + "\n### Step 2: Answer QA Details\n" + row["question_options"] + "\n### Step 3: Determine sentence relevance\n" + white_patient
    equity_df.loc[len(equity_df)] = [row["ID_corr"], "white", white_q, new_sentence_num, row["answer_corr"], row["data_source_corr"], white_patient, row["question_options"]]

equity_df.to_csv("MedPAIREquity/equity_last_sentence_data.csv", index=False)