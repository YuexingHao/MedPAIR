import argparse
import pandas as pd

def relevance_confusion_matrix(combined_df,
                               control_name,
                               trial_name,
                               save_file,
                               experiment_name):
    seen_id = set()
    confusion_matrix = {"RR": 0,
                        "RI": 0,
                        "IR": 0,
                        "II": 0}
    total_ct = 0
    per_dataset_relevant_ct = {"mmlu": {"RR": 0, "RI": 0, "IR": 0, "II": 0},
                               "jama": {"RR": 0, "RI": 0, "IR": 0, "II": 0},
                               "medxpert": {"RR": 0, "RI": 0, "IR": 0, "II": 0},
                               "medbullets": {"RR": 0, "RI": 0, "IR": 0, "II": 0},
                               "Q-Pain": {"RR": 0, "RI": 0, "IR": 0, "II": 0}}
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

        sentence_number = str(row["sentence_number_corr"]).strip()
        control = filtered_df[filtered_df["source"] == control_name].iloc[0]
        control_answer = str(control[f"label_{sentence_number}"]).strip()
        trial = filtered_df[filtered_df["source"] == trial_name].iloc[0]
        trial_answer = str(trial[f"label_{sentence_number}"]).strip()

        valid_responses = {"Irrelevant", "Low Relevance", "High Relevance"}
        irrelevant = {"Irrelevant"}
        relevant = {"Low Relevance", "High Relevance"}

        if control_answer not in valid_responses or trial_answer not in valid_responses:
            continue

        total_ct += 1
        data_source = str(row["data_source_corr"]).strip()
        per_dataset_total_ct[data_source] += 1

        if control_answer in relevant and trial_answer in relevant:
            confusion_matrix["RR"] += 1
            per_dataset_relevant_ct[data_source]["RR"] += 1
        elif control_answer in relevant and trial_answer in irrelevant:
            confusion_matrix["RI"] += 1
            per_dataset_relevant_ct[data_source]["RI"] += 1
        elif control_answer in irrelevant and trial_answer in relevant:
            confusion_matrix["IR"] += 1
            per_dataset_relevant_ct[data_source]["IR"] += 1
        elif control_answer in irrelevant and trial_answer in irrelevant:
            confusion_matrix["II"] += 1
            per_dataset_relevant_ct[data_source]["II"] += 1
    
    try:
        with open(save_file, 'a') as file:
            file.write(f"{experiment_name} ({control_name}, {trial_name})" + '\n')
            file.write(f"Total Count: {total_ct}" + '\n')
            file.write(f"Overall Confusion Matrix: {confusion_matrix}" + '\n')
            file.write(f"Per Dataset Counts: {per_dataset_total_ct}" + '\n')
            file.write(f"mmlu Confusion Matrix: {per_dataset_relevant_ct["mmlu"]}" + '\n')
            file.write(f"jama Confusion Matrix: {per_dataset_relevant_ct["jama"]}" + '\n')
            file.write(f"medxpert Confusion Matrix: {per_dataset_relevant_ct["medxpert"]}" + '\n')
            file.write(f"medbullets Confusion Matrix: {per_dataset_relevant_ct["medbullets"]}" + '\n')
            file.write(f"Q-Pain Confusion Matrix: {per_dataset_relevant_ct["Q-Pain"]}" + '\n')
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
    save_file = "Results/Relevance_Confusion_Matrix.txt"

    control_df = pd.read_csv(control_file).assign(source=control_name)
    trial_df = pd.read_csv(trial_file).assign(source=trial_name)

    combined_df = pd.concat([control_df, trial_df], ignore_index=True)

    relevance_confusion_matrix(combined_df,
                               control_name,
                               trial_name,
                               save_file,
                               experiment_name)

if __name__ == "__main__":
    main()