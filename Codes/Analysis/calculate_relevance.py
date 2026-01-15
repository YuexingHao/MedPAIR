import argparse
import pandas as pd

def calculate_relevance(df, experiment_name, save_file):
    relevance_percentage_sum = 0
    case_report_ct = 0
    per_dataset_relevance_sum = {"mmlu": 0,
                                 "jama": 0,
                                 "medxpert": 0,
                                 "medbullets": 0,
                                 "Q-Pain": 0}
    per_dataset_total_ct = {"mmlu": 0,
                            "jama": 0,
                            "medxpert": 0,
                            "medbullets": 0,
                            "Q-Pain": 0}
    valid_responses = {"Irrelevant", "Low Relevance", "High Relevance"}
    relevant = {"Low Relevance", "High Relevance"}

    for _, row in df.iterrows():
        curr_sentence_ct = 0
        curr_relevant_sentence_ct = 0
        contribution_flag = True
        sentence_number = int(row["sentence_number_corr"])
        
        for i in range(1, sentence_number + 1):
            if row[f"label_{i}"] not in valid_responses:
                contribution_flag = False
                break
            curr_sentence_ct += 1
            if row[f"label_{i}"] in relevant:
                curr_relevant_sentence_ct += 1
        
        if contribution_flag:
            relevance_percentage = curr_relevant_sentence_ct / curr_sentence_ct
            relevance_percentage_sum += relevance_percentage
            case_report_ct += 1
            data_source = str(row["data_source_corr"]).strip()
            per_dataset_relevance_sum[data_source] += relevance_percentage
            per_dataset_total_ct[data_source] += 1

    try:
        with open(save_file, 'a') as file:
            file.write(f"{experiment_name}" + '\n')
            file.write(f"Overall Percentage of Relevant Sentences: {relevance_percentage_sum / case_report_ct}" + '\n')
            file.write(f"mmlu Percentage of Relevant Sentences: {per_dataset_relevance_sum["mmlu"] / per_dataset_total_ct["mmlu"]}" + '\n')
            file.write(f"jama Percentage of Relevant Sentences: {per_dataset_relevance_sum["jama"] / per_dataset_total_ct["jama"]}" + '\n')
            file.write(f"medxpert Percentage of Relevant Sentences: {per_dataset_relevance_sum["medxpert"] / per_dataset_total_ct["medxpert"]}" + '\n')
            file.write(f"medbullets Percentage of Relevant Sentences: {per_dataset_relevance_sum["medbullets"] / per_dataset_total_ct["medbullets"]}" + '\n')
            file.write(f"Q-Pain Percentage of Relevant Sentences: {per_dataset_relevance_sum["Q-Pain"] / per_dataset_total_ct["Q-Pain"]}" + '\n')
            file.write('\n')

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Specify paths for input CSVs to be analyzed")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    args = parser.parse_args()

    input_csv = args.input_csv
    experiment_name = args.experiment
    save_file = "Results/Relevance_Percentages.txt"

    df = pd.read_csv(input_csv)

    calculate_relevance(df, experiment_name, save_file)

if __name__ == "__main__":
    main()