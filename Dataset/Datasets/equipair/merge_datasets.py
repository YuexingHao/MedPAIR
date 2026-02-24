import pandas as pd

path = "Dataset/Datasets/equipair/"
household_income = pd.read_csv(f"{path}household_income.csv")
housing_status = pd.read_csv(f"{path}housing_status.csv")
insurance_status = pd.read_csv(f"{path}insurance_questions.csv")
race = pd.read_csv(f"{path}race_questions.csv")

ids = set(household_income["ID_corr"]) \
    & set(housing_status["ID_corr"]) \
    & set(insurance_status["ID_corr"]) \
    & set(race["ID_corr"])

rows = []

for id in ids:
    rows.append({
        "ID_corr": id,
        "original_sentences": household_income.loc[household_income["ID_corr"] == id, "original_sentences"].iloc[0],
        "sentence_number_corr": household_income.loc[household_income["ID_corr"] == id, "sentence_number_corr"].iloc[0],
        "data_source_corr": household_income.loc[household_income["ID_corr"] == id, "data_source_corr"].iloc[0],
        "question_options": ""
    })

new_df = pd.DataFrame(rows)
new_df.to_csv(f"{path}/merged.csv", index=False)