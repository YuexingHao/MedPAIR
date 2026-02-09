import pandas as pd

def lookup_and_map(df, id, mapping):
    res = df.loc[df["ID_corr"] == id, "LLM_answer"]
    if len(res) == 1:
        if res.iloc[0] in mapping:
            return mapping[res.iloc[0]]
    return "Not Classified"

def merge_llm_results(save_path,
                      household_income,
                      housing_status,
                      insurance_status,
                      race):

    ids = pd.unique(pd.concat([
        household_income["ID_corr"],
        housing_status["ID_corr"],
        insurance_status["ID_corr"],
        race["ID_corr"]
    ]))
    # ids = set(household_income["ID_corr"]) \
    #     & set(housing_status["ID_corr"]) \
    #     & set(insurance_status["ID_corr"]) \
    #     & set(race["ID_corr"])
    
    household_income_dict = {"A": "$0 - $11,925",
                             "B": "$11,926 - $48,475",
                             "C": "$48,476 - $103,350", 
                             "D": "$103,351 - $197,300",
                             "E": "$197,301 - $250,525",
                             "F": "$250,526 - $626,350",
                             "G": "Over $626,350"}
    housing_status_dict = {"A": "Stable housing",
                           "B": "Transitional or temporary housing",
                           "C": "Unstable housing or homelessness"}
    insurance_status_dict = {"A": "Privately Insured",
                             "B": "Insured by Medicaid",
                             "C": "Uninsured"}
    race_dict = {"A": "Asian",
                 "B": "Black",
                 "C": "Pacific Islander",
                 "D": "White"}

    rows = []

    for id in ids:
        rows.append({
            "ID_corr": id,
            "original_sentences": household_income.loc[household_income["ID_corr"] == id, "original_sentences"].iloc[0],
            "household_income": lookup_and_map(household_income, id, household_income_dict),
            "housing_status": lookup_and_map(housing_status, id, housing_status_dict),
            "insurance_status": lookup_and_map(insurance_status, id, insurance_status_dict),
            "race": lookup_and_map(race, id, race_dict),
        })
    
    new_df = pd.DataFrame(rows)
    new_df.to_csv(save_path, index=False)

def main():
    dir = "Results/qwen/equipair/"
    household_income_csv = f"{dir}household_income.csv"
    housing_status_csv = f"{dir}housing_status.csv"
    insurance_status_csv = f"{dir}insurance_questions.csv"
    race_csv = f"{dir}race_questions.csv"

    household_income = pd.read_csv(household_income_csv)
    housing_status = pd.read_csv(housing_status_csv)
    insurance_status = pd.read_csv(insurance_status_csv)
    race = pd.read_csv(race_csv)

    merge_llm_results(f"{dir}qwen_Merged_Union.csv",
                      household_income,
                      housing_status,
                      insurance_status,
                      race)

if __name__ == "__main__":
    main()