import argparse
import pandas as pd
import matplotlib.pyplot as plt

def calculate_relevance(df, experiment_name, save_file):
    valid_responses = {"Irrelevant", "Low Relevance", "High Relevance"}
    relevant = {"Low Relevance", "High Relevance"}

    ct = 0
    relevant_perturbation = 0

    for _, row in df.iterrows():
        curr_sentence_ct = 0
        curr_relevant_sentence_ct = 0
        contribution_flag = True
        sentence_number = int(row["sentence_number_corr"])

        for i in range(1, sentence_number + 1):
            label = row[f"label_{i}"]
            if label not in valid_responses:
                contribution_flag = False
                break
            curr_sentence_ct += 1
            if label in relevant:
                curr_relevant_sentence_ct += 1

        if contribution_flag:
            perturbation_sentence_number = row["perturbation_sentence_number"]
            if str(row[f"label_{perturbation_sentence_number}"]) in relevant:
                relevant_perturbation += 1
            ct += 1

    try:
        with open(save_file, 'a') as file:
            file.write(f"{experiment_name}\n")
            file.write(f"Number of Relevant Perturbations: {relevant_perturbation} / Total Count: {ct} = {relevant_perturbation / ct}")
            file.write("\n")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Specify paths for input CSVs to be analyzed")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    args = parser.parse_args()

    input_csv = args.input_csv
    experiment_name = args.experiment
    save_file = "Results/RandomLocationPerturbation.txt"

    df = pd.read_csv(input_csv)

    calculate_relevance(df, experiment_name, save_file)


if __name__ == "__main__":
    main()