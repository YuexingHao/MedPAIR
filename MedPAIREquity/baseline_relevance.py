import pandas as pd
import numpy as np

ALL_QAS = "MedPAIREquity/Results/all_qas.csv"
SAVE_FILE = "MedPAIREquity/Results/Relevance_Percentages.txt"

BASELINES = {
    "Baseline (Housing)": "MedPAIREquity/baseline_data_housing_filtered.csv",
    "Baseline (Income)":  "MedPAIREquity/baseline_data_income_filtered.csv",
    "Baseline (Insurance)": "MedPAIREquity/baseline_data_insurance_filtered.csv",
    "Baseline (Race)":    "MedPAIREquity/baseline_data_race.csv",
}

DATASETS = ["mmlu", "jama", "medxpert", "medbullets"]


def mean_std_ci(values):
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
    valid_labels = {"Irrelevant", "Low Relevance", "High Relevance"}

    overall_is_relevant = []
    per_dataset_is_relevant = {ds: [] for ds in DATASETS}

    for _, row in df.iterrows():
        sentence_number = int(row["sentence_number_corr"])
        if sentence_number > 21:
            continue
        label = row[f"label_{sentence_number}"]
        if label not in valid_labels:
            continue

        is_relevant = 1 if label in ("Low Relevance", "High Relevance") else 0
        overall_is_relevant.append(is_relevant)

        data_source = str(row["data_source_corr"]).strip()
        if data_source in per_dataset_is_relevant:
            per_dataset_is_relevant[data_source].append(is_relevant)

    with open(save_file, "a") as f:
        f.write(f"{experiment_name}\n")
        mean, std, ci_low, ci_high = mean_std_ci(overall_is_relevant)
        f.write(
            f"Overall Last Sentence Relevant: mean={mean:.4f}, std={std:.4f}, "
            f"95% CI=({ci_low:.4f}, {ci_high:.4f})\n"
        )
        for dataset, values in per_dataset_is_relevant.items():
            mean, std, ci_low, ci_high = mean_std_ci(values)
            f.write(
                f"{dataset} Last Sentence Relevant: mean={mean:.4f}, std={std:.4f}, "
                f"95% CI=({ci_low:.4f}, {ci_high:.4f})\n"
            )
        f.write("\n")


def main():
    all_qas = pd.read_csv(ALL_QAS)

    for experiment_name, baseline_path in BASELINES.items():
        baseline = pd.read_csv(baseline_path)
        ids = set(baseline["ID_corr"])
        subset = all_qas[all_qas["ID_corr"].isin(ids)].copy()
        print(f"{experiment_name}: {len(subset)} rows")
        calculate_relevance(subset, experiment_name, SAVE_FILE)

    print(f"Done. Results appended to {SAVE_FILE}")


if __name__ == "__main__":
    main()
