import argparse
import pandas as pd

def overall_and_per_dataset_refusal_rate(results_df,
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

        model_answer = str(row["LLM_answer"]).strip()
        if model_answer not in "ABCDEFG":
            correct_ct += 1
            per_dataset_correct_ct[data_source] += 1

    overall_accuracy = correct_ct / total_ct
    per_dataset_accuracy = {"mmlu": per_dataset_correct_ct["mmlu"] / per_dataset_total_ct["mmlu"],
                            "jama": per_dataset_correct_ct["jama"] / per_dataset_total_ct["jama"],
                            "medxpert": per_dataset_correct_ct["medxpert"] / per_dataset_total_ct["medxpert"],
                            "medbullets": per_dataset_correct_ct["medbullets"] / per_dataset_total_ct["medbullets"],
                            "Q-Pain": per_dataset_correct_ct["Q-Pain"] / per_dataset_total_ct["Q-Pain"]}
    
    try:
        with open(save_file, 'a') as file:
            file.write(experiment_name + '\n')
            file.write(f"Overall Refusal Rate: {correct_ct}/{total_ct}={overall_accuracy}" + '\n')
            file.write(f"mmlu Refusal Rate: {per_dataset_correct_ct["mmlu"]}/{per_dataset_total_ct["mmlu"]}={per_dataset_accuracy["mmlu"]}" + '\n')
            file.write(f"jama Refusal Rate: {per_dataset_correct_ct["jama"]}/{per_dataset_total_ct["jama"]}={per_dataset_accuracy["jama"]}" + '\n')
            file.write(f"medxpert Refusal Rate: {per_dataset_correct_ct["medxpert"]}/{per_dataset_total_ct["medxpert"]}={per_dataset_accuracy["medxpert"]}" + '\n')
            file.write(f"medbullets Refusal Rate: {per_dataset_correct_ct["medbullets"]}/{per_dataset_total_ct["medbullets"]}={per_dataset_accuracy["medbullets"]}" + '\n')
            file.write(f"Q-Pain Refusal Rate: {per_dataset_correct_ct["Q-Pain"]}/{per_dataset_total_ct["Q-Pain"]}={per_dataset_accuracy["Q-Pain"]}" + '\n')
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
    save_file = "Results/RefusalRate.txt"

    overall_and_per_dataset_refusal_rate(results_df,
                                         save_file,
                                         experiment_name)

if __name__ == "__main__":
    main()