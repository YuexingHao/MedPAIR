import argparse
import pandas as pd

def relevance_accuracy(results_df,
                       save_file,
                       experiment_name):
    total_correct_ct = {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0}
    total_ct = {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0}
    per_dataset_correct_ct = {"mmlu": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                              "jama": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                              "medxpert": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                              "medbullets": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                              "Q-Pain": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0}}
    per_dataset_total_ct = {"mmlu": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                            "jama": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                            "medxpert": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                            "medbullets": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0},
                            "Q-Pain": {"High Relevance": 0, "Low Relevance": 0, "Irrelevant": 0}}
    
    for _, row in results_df.iterrows():
        model_answer = str(row["GPT5_answer"]).strip()
        num_sentences = str(row["sentence_number_corr"]).strip()
        relevance = str(row[f"label_{num_sentences}"]).strip()
        valid_relevance = {"Irrelevant", "Low Relevance", "High Relevance"}
        valid_answer = "ABCDEFGHIJ"

        if model_answer not in valid_answer or relevance not in valid_relevance:
            continue

        total_ct[relevance] += 1
        data_source = str(row["data_source_corr"]).strip()
        per_dataset_total_ct[data_source][relevance] += 1

        correct_answer = str(row["answer_corr"]).strip()
        if correct_answer == model_answer:
            total_correct_ct[relevance] += 1
            per_dataset_correct_ct[data_source][relevance] += 1
    
    try:
        with open(save_file, 'a') as file:
            file.write(experiment_name + '\n')
            file.write(f"Overall Correct: {total_correct_ct}" + '\n')
            file.write(f"Overall Counts: {total_ct}" + '\n')
            file.write(f"Per Dataset Correct: {per_dataset_correct_ct}" + '\n')
            file.write(f"Per Dataset Count: {per_dataset_total_ct}" + '\n')
            file.write('\n')

    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Specify paths for input CSV to be analyzed")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--experiment_name", required=True, help="Experiment name")
    args = parser.parse_args()

    results_file = args.input
    experiment_name = args.experiment_name
    results_df = pd.read_csv(results_file)
    save_file = "Results/Relevance_Accuracy.txt"

    relevance_accuracy(results_df,
                       save_file,
                       experiment_name)

if __name__ == "__main__":
    main()