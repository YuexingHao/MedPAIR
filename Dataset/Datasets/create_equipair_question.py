import pandas as pd

def generate_equipair_dataset(original_csv,
                              save_csv,
                              new_question,
                              new_answer):
    df = pd.read_csv(original_csv)
    
    for _, row in df.iterrows():
        row["question_options"] = new_question
        row["centaur_question_corr"] = f"### Step 1: Read excerpt \n {row["original_sentences"]} \n\n ### Step 2: Answer QA Details \n {row["question_options"]} \n\n ### Step 3: Determine sentence relevance \n {row["original_sentences"]}"
        row["answer_corr"] = new_answer

    df.to_csv(save_csv, index=False)


def main():
    generate_equipair_dataset("Dataset/Datasets/race_perturbations/medpair_and_q-pain_baseline.csv",
                              "Dataset/Datasets/equipair/race_questions.csv",
                              "What is the race of this patient?\n\nA. Asian\nB. Black\nC. Pacific Islander\nD. White",
                              "")

if __name__ == "__main__":
    main()