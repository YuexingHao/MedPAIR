import pandas as pd

df = pd.read_csv("Results/medgemma/equipair/all_questions.csv")
save_path = "Results/medgemma/equipair/all_questions_cleaned.csv"
save_df = pd.DataFrame(columns=df.columns)

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

for _, row in df.iterrows():
    if (str(row["household_income"]).strip() in "ABCDEFG" and 
        str(row["housing_status"]).strip() in "ABCDEFG" and
        str(row["insurance_status"]).strip() in "ABCDEFG" and
        str(row["race"]).strip() in "ABCDEFG"):
        save_df.loc[len(save_df)] = {"ID_corr": df["ID_corr"],
                                     "centaur_question_corr": df["centaur_question_corr"],
                                     "sentence_number_corr": df["sentence_number_corr"],
                                     "answer_corr": df["answer_corr"],
                                     "data_source_corr": df["data_source_corr"],
                                     "original_sentences": df["original_sentences"],
                                     "question_options": df["question_options"],
                                     "Raw_Response": df["Raw_Response"],
                                     "household_income": household_income_dict[str(row["household_income"]).strip()],
                                     "housing_status": housing_status_dict[str(row["housing_status"]).strip()],
                                     "insurance_status": insurance_status_dict[str(row["insurance_status"]).strip()],
                                     "race": race_dict[str(row["race"]).strip()]}
        
save_df.to_csv(save_path)