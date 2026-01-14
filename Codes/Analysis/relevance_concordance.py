import argparse
import pandas as pd

def relevance_concordance(combined_df,
                          control_name,
                          trial_name,
                          save_file,
                          experiment_name):
    seen_id = set()
    concordance_sum = 0
    total_ct = 0
    per_dataset_concordance = {"mmlu": 0,
                               "jama": 0,
                               "medxpert": 0,
                               "medbullets": 0,
                               "Q-Pain": 0}
    per_dataset_total_ct = {"mmlu": 0,
                            "jama": 0,
                            "medxpert": 0,
                            "medbullets": 0,
                            "Q-Pain": 0}
    
    for _, row in combined_df.iterrows():
        id = str(row["ID_corr"]).strip()
        if row["ID_corr"] in seen_id:
            continue
        
        seen_id.add(id)
        
        filtered_df = combined_df[combined_df["ID_corr"] == id]
        if not {control_name, trial_name}.issubset(filtered_df["source"].unique()) or len(filtered_df) != 2:
            continue

        control = filtered_df[filtered_df["source"] == control_name].iloc[0]
        sentence_number = int(control["sentence_number_corr"])
        trial = filtered_df[filtered_df["source"] == trial_name].iloc[0]

        valid_responses = {"Irrelevant", "Low Relevance", "High Relevance"}
        irrelevant = {"Irrelevant"}
        relevant = {"Low Relevance", "High Relevance"}

        curr_concordance = 0
        contribution_flag = True

        for i in range(1, sentence_number + 1):
            if control[f"label_{i}"] not in valid_responses or trial[f"label_{i}"] not in valid_responses:
                contribution_flag = False
                break
            if control[f"label_{i}"] in irrelevant and trial[f"label_{i}"] in irrelevant:
                curr_concordance += 1
            elif control[f"label_{i}"] in relevant and trial[f"label_{i}"] in relevant:
                curr_concordance += 1
        
        if contribution_flag:
            total_ct += 1
            data_source = str(row["data_source_corr"]).strip()
            per_dataset_total_ct[data_source] += 1
            concordance_percentage = curr_concordance / sentence_number
            concordance_sum += concordance_percentage
            per_dataset_concordance[data_source] += concordance_percentage
    
    try:
        with open(save_file, 'a') as file:
            file.write(f"{experiment_name} ({control_name}, {trial_name})" + '\n')
            file.write(f"Total Count: {total_ct}" + '\n')
            file.write(f"Overall Concordance: {concordance_sum / total_ct}" + '\n')
            file.write(f"Per Dataset Counts: {per_dataset_total_ct}" + '\n')
            file.write(f"mmlu Concordance: {per_dataset_concordance["mmlu"] / per_dataset_total_ct["mmlu"]}" + '\n')
            file.write(f"jama Concordance: {per_dataset_concordance["jama"] / per_dataset_total_ct["jama"]}" + '\n')
            file.write(f"medxpert Concordance: {per_dataset_concordance["medxpert"] / per_dataset_total_ct["medxpert"]}" + '\n')
            file.write(f"medbullets Concordance: {per_dataset_concordance["medbullets"] / per_dataset_total_ct["medbullets"]}" + '\n')
            file.write(f"Q-Pain Concordance: {per_dataset_concordance["Q-Pain"] / per_dataset_total_ct["Q-Pain"]}" + '\n')
            file.write('\n')

    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Specify paths for input CSVs to be analyzed")
    parser.add_argument("--control_file", required=True, help="Path to control file")
    parser.add_argument("--control", required=True, help="Control name")
    parser.add_argument("--trial_file", required=True, help="Path to experiment file")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    args = parser.parse_args()

    control_file = args.control_file
    trial_file = args.trial_file
    control_name = args.control
    trial_name = args.trial
    experiment_name = args.experiment
    save_file = "Results/Relevance_Concordance.txt"

    control_df = pd.read_csv(control_file).assign(source=control_name)
    trial_df = pd.read_csv(trial_file).assign(source=trial_name)

    combined_df = pd.concat([control_df, trial_df], ignore_index=True)

    relevance_concordance(combined_df,
                          control_name,
                          trial_name,
                          save_file,
                          experiment_name)

if __name__ == "__main__":
    main()