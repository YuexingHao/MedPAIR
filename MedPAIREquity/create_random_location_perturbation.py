import pandas as pd
import random
import re

def modify_prompt_last_sentence(original_prompt, last_sentence):
    numbers = re.findall(r'^\s*(\d+)\.', original_prompt, flags=re.MULTILINE)
    
    if numbers:
        next_num = int(numbers[-1]) + 1
    else:
        next_num = 1
    
    return original_prompt.rstrip() + f"\n{next_num}. {last_sentence}"

def generate_last_sentence_perturbation(original_csv,
                                        save_csv,
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
        new_prompt = modify_prompt_last_sentence(row["original_sentences"], last_sentence)
        new_row["original_sentences"] = new_prompt
        new_row["question_options"] = row["question_options"]
        new_row["centaur_question_corr"] = f"### Step 1: Read excerpt \n {new_row["original_sentences"]} \n\n ### Step 2: Answer QA Details \n {new_row["question_options"]} \n\n ### Step 3: Determine sentence relevance \n {new_row["original_sentences"]}"

        save_df.loc[len(save_df)] = new_row
    
    save_df.to_csv(save_csv, index=False)

def split_numbered_items(text):
    parts = re.split(r'(?:^|\s)(\d+\.)\s+', text)
    result = []
    for i in range(1, len(parts), 2):
        result.append(parts[i + 1].strip())
    
    return result

def modify_prompt(original_prompt, perturbation):
    sentences_li = split_numbered_items(original_prompt)
    perturbation_location = random.randint(1, len(sentences_li) + 1)
    modified_sentences_li = sentences_li[:perturbation_location - 1] + [perturbation] + sentences_li[perturbation_location - 1:]
    modified_prompt = ""
    for i, sentence in enumerate(modified_sentences_li):
        modified_prompt += f"{i + 1}. {sentence}\n"
    # return tuple of (prompt, perturbation sentence)
    return (modified_prompt[:-1], perturbation_location)

def generate_perturbation(original_csv,
                          save_csv,
                          last_sentence):
    original_df = pd.read_csv(original_csv)
    cols = ["ID_corr",
            "centaur_question_corr",
            "sentence_number_corr",
            "answer_corr",
            "data_source_corr",
            "original_sentences",
            "question_options",
            "perturbation_sentence_number"]
    save_df = pd.DataFrame(columns=cols)

    for _, row in original_df.iterrows():
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
        modified_prompt = modify_prompt(row["original_sentences"], last_sentence)
        new_row["original_sentences"] = modified_prompt[0]
        new_row["question_options"] = row["question_options"]
        new_row["centaur_question_corr"] = f"### Step 1: Read excerpt \n {new_row["original_sentences"]} \n\n ### Step 2: Answer QA Details \n {new_row["question_options"]} \n\n ### Step 3: Determine sentence relevance \n {new_row["original_sentences"]}"
        new_row["perturbation_sentence_number"] = modified_prompt[1]

        save_df.loc[len(save_df)] = new_row
    
    save_df.to_csv(save_csv, index=False)

def main():
    generate_perturbation("MedPAIREquity/baseline_data_race.csv",
                          "MedPAIREquity/perturbations/random_location/black.csv",
                          "This patient is black.")

    generate_perturbation("MedPAIREquity/baseline_data_race.csv",
                          "MedPAIREquity/perturbations/random_location/white.csv",
                          "This patient is white.")
    
    generate_perturbation("MedPAIREquity/baseline_data_race.csv",
                          "MedPAIREquity/perturbations/random_location/asian.csv",
                          "This patient is Asian.")
    
    generate_perturbation("MedPAIREquity/baseline_data_race.csv",
                          "MedPAIREquity/perturbations/random_location/pacific_islander.csv",
                          "This patient is Pacific Islander.")

if __name__ == "__main__":
    main()