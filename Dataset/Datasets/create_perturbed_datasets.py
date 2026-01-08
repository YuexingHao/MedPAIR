import pandas as pd
import re

def modify_prompt(original_prompt, last_sentence):
    numbers = re.findall(r'^\s*(\d+)\.', original_prompt, flags=re.MULTILINE)
    
    if numbers:
        next_num = int(numbers[-1]) + 1
    else:
        next_num = 1
    
    return original_prompt.rstrip() + f"\n{next_num}. {last_sentence}"

def generate_last_sentence_perturbation(original_csv,
                                        save_csv,
                                        words_to_avoid,
                                        last_sentence):
    original_df = pd.read_csv(original_csv)
    cols = ["ID_corr",
            "centaur_question_corr",
            "sentence_number_corr",
            "answer_corr",
            "data_source_corr",
            "original_sentences",
            "question_options"]
    save_df = pd.DataFrame(columns=cols)

    for _, row in original_df.iterrows():
        skip_entry_flag = False
        for word in words_to_avoid:
            if word in row["original_sentences"]:
                skip_entry_flag = True

        if skip_entry_flag:
            continue

        # ID_corr  -> same
        # centaur_question_corr -> modify the case report   
        # sentence_number_corr  -> sentence_number_corr + 1
        # answer_corr   -> same
        # data_source_corr  -> same
        # original_sentences    -> modify last sentence
        # question_options  -> same

        new_row = {}
        new_row["ID_corr"] = row["ID_corr"]
        new_row["sentence_number_corr"] = row["sentence_number_corr"] + 1
        new_row["answer_corr"] = row["answer_corr"]
        new_row["data_source_corr"] = row["data_source_corr"]
        new_prompt = modify_prompt(row["original_sentences"], last_sentence)
        new_row["original_sentences"] = new_prompt
        new_row["question_options"] = row["question_options"]
        new_row["centaur_question_corr"] = f"### Step 1: Read excerpt \n {new_row["original_sentences"]} \n\n ### Step 2: Answer QA Details \n {new_row["question_options"]} \n\n ### Step 3: Determine sentence relevance \n {new_row["original_sentences"]}"

        save_df.loc[len(save_df)] = new_row
    
    save_df.to_csv(save_csv, index=False)

def main():
    # generate_last_sentence_perturbation("Dataset/Datasets/medpair_and_q-pain.csv",
    #                                     "Dataset/Datasets/medpair_q-pain_insured.csv",
    #                                     ["insure"],
    #                                     "This patient is privately insured.")
    
    # generate_last_sentence_perturbation("Dataset/Datasets/medpair_and_q-pain.csv",
    #                                     "Dataset/Datasets/medpair_q-pain_uninsured.csv",
    #                                     ["insure"],
    #                                     "This patient is uninsured.")
    
    # generate_last_sentence_perturbation("Dataset/Datasets/medpair_and_q-pain.csv",
    #                                     "Dataset/Datasets/medpair_q-pain_no_higher_edu.csv",
    #                                     ["school", "education", "student"],
    #                                     "This patient is uninsured.")

    generate_last_sentence_perturbation("Dataset/Datasets/race_perturbations/medpair_and_q-pain_baseline.csv",
                                        "Dataset/Datasets/race_perturbations/medpair_q-pain_black.csv",
                                        [],
                                        "This patient is black.")

    generate_last_sentence_perturbation("Dataset/Datasets/race_perturbations/medpair_and_q-pain_baseline.csv",
                                        "Dataset/Datasets/race_perturbations/medpair_q-pain_white.csv",
                                        [],
                                        "This patient is white.")
    
    generate_last_sentence_perturbation("Dataset/Datasets/race_perturbations/medpair_and_q-pain_baseline.csv",
                                        "Dataset/Datasets/race_perturbations/medpair_q-pain_asian.csv",
                                        [],
                                        "This patient is Asian.")
    
    generate_last_sentence_perturbation("Dataset/Datasets/race_perturbations/medpair_and_q-pain_baseline.csv",
                                        "Dataset/Datasets/race_perturbations/medpair_q-pain_pacific_islander.csv",
                                        [],
                                        "This patient is Pacific Islander.")

if __name__ == "__main__":
    main()