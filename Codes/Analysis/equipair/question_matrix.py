import pandas as pd

def create_matrix(df, dimension1, dimension2, options1, options2):
    res = {}
    for x in options1:
        res[x] = {}
        for y in options2:
            res[x][y] = 0
    
    for _, row in df.iterrows():
        res1 = row[dimension1]
        res2 = row[dimension2]
        if res1 in options1 and res2 in options2:
            res[res1][res2] += 1
    
    print(res)

def main():
    household_income_dict = ["$0 - $11,925",
                             "$11,926 - $48,475",
                             "$48,476 - $103,350", 
                             "$103,351 - $197,300",
                             "$197,301 - $250,525",
                             "$250,526 - $626,350",
                             "Over $626,350"]
    housing_status_dict = ["Stable housing",
                           "Transitional or temporary housing",
                           "Unstable housing or homelessness"]
    insurance_status_dict = ["Privately Insured",
                             "Insured by Medicaid",
                             "Uninsured"]
    race_dict = ["Asian",
                 "Black",
                 "Pacific Islander",
                 "White"]
    
    df = pd.read_csv("Results/gpt5/EquiPAIRQuestions/GPT5_Merged.csv")
    create_matrix(df, 
                  "household_income", 
                  "housing_status", 
                  household_income_dict, 
                  housing_status_dict)
    
if __name__ == "__main__":
    main()