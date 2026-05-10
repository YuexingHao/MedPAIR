import argparse
import pandas as pd

def overall_and_per_dataset_accuracy(results_df,
                                     save_file,
                                     experiment_name):
    correct_ct = 0
    total_ct = 0
    per_dataset_correct_ct = {"mmlu": 0,
                              "jama": 0,
                              "medxpert": 0,
                              "medbullets": 0,
                              "Q-Pain": 0}
    per_dataset_total_ct = {"mmlu": 0,
                            "jama": 0,
                            "medxpert": 0,
                            "medbullets": 0,
                            "Q-Pain": 0}
    
    for _, row in results_df.iterrows():
        total_ct += 1
        data_source = str(row["data_source_corr"]).strip()
        per_dataset_total_ct[data_source] += 1

        correct_answer = str(row["answer_corr"]).strip()
        model_answer = str(row["LLM_answer"]).strip()
        if pd.isna(row["LLM_answer"]) or model_answer not in {"A", "B", "C", "D", "E"}:
            total_ct -= 1
            per_dataset_total_ct[data_source] -= 1
            continue
        if correct_answer == model_answer:
            correct_ct += 1
            per_dataset_correct_ct[data_source] += 1

    overall_accuracy = correct_ct / total_ct
    per_dataset_accuracy = {
        ds: per_dataset_correct_ct[ds] / per_dataset_total_ct[ds]
        for ds in per_dataset_total_ct
        if per_dataset_total_ct[ds] > 0
    }

    try:
        with open(save_file, 'a') as file:
            file.write(experiment_name + '\n')
            file.write(f"Overall Accuracy: {correct_ct}/{total_ct}={overall_accuracy}" + '\n')
            for ds in ["mmlu", "jama", "medxpert", "medbullets", "Q-Pain"]:
                if per_dataset_total_ct[ds] > 0:
                    file.write(f"{ds} Accuracy: {per_dataset_correct_ct[ds]}/{per_dataset_total_ct[ds]}={per_dataset_accuracy[ds]}" + '\n')
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
    results_df = results_df.drop_duplicates(subset="ID_corr", keep="last")
    save_file = "MedPAIREquity/Results/Accuracy.txt"

    overall_and_per_dataset_accuracy(results_df,
                                     save_file,
                                     experiment_name)

if __name__ == "__main__":
    main()