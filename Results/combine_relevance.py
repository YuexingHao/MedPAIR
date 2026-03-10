import pandas as pd

household_income_df = pd.read_csv("Results/gpt5/EquiPAIRQuestions/household_income_questions.csv")
housing_status_df = pd.read_csv("Results/gpt5/EquiPAIRQuestions/housing_status.csv")
insurance_df = pd.read_csv("Results/gpt5/EquiPAIRQuestions/insurance_questions.csv")
race_df = pd.read_csv("Results/gpt5/EquiPAIRQuestions/race_questions.csv")

dfs = [household_income_df, housing_status_df, insurance_df, race_df]

relevance_combined_df = pd.DataFrame(columns=["ID_corr"] + [f"label_{i}" for i in range(1, 31)])

for _, row in household_income_df.iterrows():
    id = row["ID_corr"]
    new_row = {"ID_corr": id}

    for i in range(30):
        label_col = f"label_{i+1}"
        labels = []

        for df in dfs:
            subset = df.loc[df["ID_corr"] == id, label_col]
            if not subset.empty:
                labels.append(subset.iloc[0])

        if "High Relevance" in labels:
            new_row[label_col] = "High Relevance"
        elif "Low Relevance" in labels:
            new_row[label_col] = "Low Relevance"
        else:
            new_row[label_col] = "Irrelevant"

    relevance_combined_df.loc[len(relevance_combined_df)] = new_row

relevance_combined_df.to_csv(
    "Results/gpt5/EquiPAIRQuestions/combined_relevance.csv",
    index=False
)