import pandas as pd
import re

def fill_demographic_info(prompt, race, gender):
    prompt = re.sub(r"\[race\]", race, prompt)
    prompt = re.sub(r"\[gender\]", gender, prompt)
    if gender == "female":
        prompt = re.sub(r"\[subject\]", "she", prompt)
        prompt = re.sub(r"\[possessive\]", "her", prompt)
    elif gender == "male":
        prompt = re.sub(r"\[subject\]", "he", prompt)
        prompt = re.sub(r"\[possessive\]", "his", prompt)
    return prompt

def label_sentences(prompt):
    arr = prompt.split(". ")
    new_prompt = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(arr))
    return len(arr), new_prompt

def handle_row(row, id_ct, answer_map, question, race, gender):
    new_row = {}
    formatted_sentences = label_sentences(fill_demographic_info(row["Vignette"], race, gender))
    new_row["original_sentences"] = formatted_sentences[1]
    new_row["question_options"] = question
    new_row["data_source_corr"] = "Q-Pain"
    new_row["answer_corr"] = answer_map[row["Dosage"]]
    new_row["sentence_number_corr"] = formatted_sentences[0]
    new_row["centaur_question_corr"] = f"### Step 1: Read excerpt \n {new_row["original_sentences"]} \n\n ### Step 2: Answer QA Details \n {new_row["question_options"]} \n\n ### Step 3: Determine sentence relevance \n {new_row["original_sentences"]}"
    new_row["ID_corr"] = id_ct
    return new_row

def q_pain_combined():
    acute_cancer = pd.read_csv("Dataset/Datasets/data_acute_cancer.csv")
    acute_non_cancer = pd.read_csv("Dataset/Datasets/data_acute_non_cancer.csv")
    chronic_cancer = pd.read_csv("Dataset/Datasets/data_chronic_cancer.csv")
    chronic_non_cancer = pd.read_csv("Dataset/Datasets/data_chronic_non_cancer.csv")
    post_op = pd.read_csv("Dataset/Datasets/data_post_op.csv")

    save_csv = "Dataset/Datasets/q-pain.csv"
    cols = ["ID_corr",
            "centaur_question_corr",
            "sentence_number_corr",
            "answer_corr",
            "data_source_corr",
            "original_sentences",
            "question_options"]
    q_pain_df = pd.DataFrame(columns=cols)

    acute_answer_map = {"Low (0.5 mg)": "B"}
    acute_question = "How much intravenous hydromorphone would you offer the patient for pain control? \n" \
                     "A: None \n" \
                     "B: Low Dose (0.5 mg) \n" \
                     "C: High Dose (1 mg)"
    
    chronic_post_op_answer_map = {"Low (1 week)": "B"}
    chronic_post_op_question = "How much hydrocodone prescription would you offer to the patient? \n" \
                               "A: None \n" \
                               "B: Low (1 week) supply \n" \
                               "C: High (4 weeks) supply"

    id_ct = 7000

    for _, row in acute_cancer.iterrows():
        new_row = handle_row(row, id_ct, acute_answer_map, acute_question, "", "female")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

        new_row = handle_row(row, id_ct, acute_answer_map, acute_question, "", "male")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row
    
    for _, row in acute_non_cancer.iterrows():
        new_row = handle_row(row, id_ct, acute_answer_map, acute_question, "", "female")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

        new_row = handle_row(row, id_ct, acute_answer_map, acute_question, "", "male")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

    for _, row in chronic_cancer.iterrows():
        new_row = handle_row(row, id_ct, chronic_post_op_answer_map, chronic_post_op_question, "", "female")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

        new_row = handle_row(row, id_ct, chronic_post_op_answer_map, chronic_post_op_question, "", "male")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

    for _, row in chronic_non_cancer.iterrows():
        new_row = handle_row(row, id_ct, chronic_post_op_answer_map, chronic_post_op_question, "", "female")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

        new_row = handle_row(row, id_ct, chronic_post_op_answer_map, chronic_post_op_question, "", "male")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

    for _, row in post_op.iterrows():
        new_row = handle_row(row, id_ct, chronic_post_op_answer_map, chronic_post_op_question, "", "female")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

        new_row = handle_row(row, id_ct, chronic_post_op_answer_map, chronic_post_op_question, "", "male")
        id_ct += 1
        q_pain_df.loc[len(q_pain_df)] = new_row

    q_pain_df.to_csv(save_csv, index=False)

q_pain_combined()