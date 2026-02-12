import argparse
import pandas as pd
import numpy as np

def mean_std_ci(values):
    """
    Returns (mean, std, ci_low, ci_high)
    """
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = np.mean(values)
    if n == 1:
        return mean, np.nan, np.nan, np.nan
    std = np.std(values, ddof=1)
    ci_half_width = 1.96 * std / np.sqrt(n)
    return mean, std, mean - ci_half_width, mean + ci_half_width


def calculate_relevance(df, experiment_name, save_file):
    relevance_labels = {"Irrelevant": 0, "Low Relevance": 0, "High Relevance": 0}

    # Store per-case relevance percentages
    overall_relevance_percentage = {"Irrelevant": [],
                                    "Low Relevance": [],
                                    "High Relevance": []}

    per_dataset_relevance = {
        "mmlu": {"Irrelevant": [], "Low Relevance": [], "High Relevance": []},
        "jama": {"Irrelevant": [], "Low Relevance": [], "High Relevance": []},
        "medxpert": {"Irrelevant": [], "Low Relevance": [], "High Relevance": []},
        "medbullets": {"Irrelevant": [], "Low Relevance": [], "High Relevance": []},
        "Q-Pain": {"Irrelevant": [], "Low Relevance": [], "High Relevance": []}
    }

    for _, row in df.iterrows():
        curr_sentence_ct = 0
        curr_relevance_labels = {"Irrelevant": 0, "Low Relevance": 0, "High Relevance": 0}
        contribution_flag = True
        sentence_number = int(row["sentence_number_corr"])

        for i in range(1, sentence_number + 1):
            label = row[f"label_{i}"]
            if label not in relevance_labels:
                contribution_flag = False
                break
            curr_sentence_ct += 1
            curr_relevance_labels[label] += 1

        if contribution_flag and curr_sentence_ct > 0:
            for k in relevance_labels.keys():
                overall_relevance_percentage[k].append(curr_relevance_labels[k] / curr_sentence_ct)
                data_source = str(row["data_source_corr"]).strip()
                if data_source in per_dataset_relevance:
                    per_dataset_relevance[data_source][k].append(curr_relevance_labels[k] / curr_sentence_ct)

    try:
        with open(save_file, 'a') as file:
            file.write(f"{experiment_name}\n")

            for k in overall_relevance_percentage.keys():
                mean, std, ci_low, ci_high = mean_std_ci(overall_relevance_percentage[k])
                file.write(
                    f"Overall {k}: mean={mean:.4f}, std={std:.4f}, "
                    f"95% CI {k}=({ci_low:.4f}, {ci_high:.4f})\n"
                )

            for dataset, values in per_dataset_relevance.items():
                for k in overall_relevance_percentage.keys():
                    mean, std, ci_low, ci_high = mean_std_ci(values[k])
                    file.write(
                        f"{dataset} {k}: mean={mean:.4f}, std={std:.4f}, "
                        f"95% CI {k}=({ci_low:.4f}, {ci_high:.4f})\n"
                    )

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
    save_file = "Results/Relevance_Percentages.txt"

    df = pd.read_csv(input_csv)

    calculate_relevance(df, experiment_name, save_file)


if __name__ == "__main__":
    main()