import argparse
import pandas as pd

def accuracy_confusion_matrix(combined_df,
                              control_name,
                              trial_name,
                              save_file,
                              experiment_name):
    seen_id = set()
    confusion_matrix = {"CC": 0,
                        "CW": 0,
                        "WC": 0,
                        "WW": 0}
    total_ct = 0
    per_dataset_correct_ct = {"mmlu": {"CC": 0, "CW": 0, "WC": 0, "WW": 0},
                              "jama": {"CC": 0, "CW": 0, "WC": 0, "WW": 0},
                              "medxpert": {"CC": 0, "CW": 0, "WC": 0, "WW": 0},
                              "medbullets": {"CC": 0, "CW": 0, "WC": 0, "WW": 0},
                              "Q-Pain": {"CC": 0, "CW": 0, "WC": 0, "WW": 0}}
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

        total_ct += 1
        data_source = str(row["data_source_corr"]).strip()
        per_dataset_total_ct[data_source] += 1
        correct_answer = str(row["answer_corr"]).strip()

        control = filtered_df[filtered_df["source"] == control_name].iloc[0]
        control_answer = control["GPT5_answer"]
        trial = filtered_df[filtered_df["source"] == trial_name].iloc[0]
        trial_answer = trial["GPT5_answer"]

        if correct_answer == control_answer and correct_answer == trial_answer:
            confusion_matrix["CC"] += 1
            per_dataset_correct_ct[data_source]["CC"] += 1
        elif correct_answer == control_answer and correct_answer != trial_answer:
            confusion_matrix["CW"] += 1
            per_dataset_correct_ct[data_source]["CW"] += 1
        elif correct_answer != control_answer and correct_answer == trial_answer:
            confusion_matrix["WC"] += 1
            per_dataset_correct_ct[data_source]["WC"] += 1
        elif correct_answer != control_answer and correct_answer != trial_answer:
            confusion_matrix["WW"] += 1
            per_dataset_correct_ct[data_source]["WW"] += 1
    
    try:
        with open(save_file, 'a') as file:
            file.write(f"{experiment_name} ({control_name}, {trial_name})" + '\n')
            file.write(f"Total Count: {total_ct}" + '\n')
            file.write(f"Overall Confusion Matrix: {confusion_matrix}" + '\n')
            file.write(f"Per Dataset Counts: {per_dataset_total_ct}" + '\n')
            file.write(f"mmlu Confusion Matrix: {per_dataset_correct_ct["mmlu"]}" + '\n')
            file.write(f"jama Confusion Matrix: {per_dataset_correct_ct["jama"]}" + '\n')
            file.write(f"medxpert Confusion Matrix: {per_dataset_correct_ct["medxpert"]}" + '\n')
            file.write(f"medbullets Confusion Matrix: {per_dataset_correct_ct["medbullets"]}" + '\n')
            file.write(f"Q-Pain Confusion Matrix: {per_dataset_correct_ct["Q-Pain"]}" + '\n')
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
    experiment_name = args.experiment_name
    save_file = "Results/Accuracy_Confusion_Matrix.txt"

    control_df = pd.read_csv(control_file).assign(source=control_name)
    trial_df = pd.read_csv(trial_file).assign(source=experiment_name)

    combined_df = pd.concat([control_df, trial_df], ignore_index=True)

    accuracy_confusion_matrix(combined_df,
                              control_name,
                              trial_name,
                              save_file,
                              experiment_name)

if __name__ == "__main__":
    main()