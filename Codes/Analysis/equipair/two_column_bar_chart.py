import pandas as pd

df = pd.read_csv("Results/qwen/equipair/qwen_Merged_Intersection.csv")

print(len(df))

# household_income_bins = [
#     "$0 - $11,925",
#     "$11,926 - $48,475",
#     "$48,476 - $103,350",
#     "$103,351 - $197,300",
#     "$197,301 - $250,525",
#     "$250,526 - $626,350",
#     "Over $626,350"
# ]
# housing_status_bins = ["Stable housing",
#                        "Transitional or temporary housing",
#                        "Unstable housing or homelessness"]
# insurance_status_bins = ["Privately Insured",
#                          "Insured by Medicaid",
#                          "Uninsured"]
race_bins = ["Asian",
             "Black",
             "Pacific Islander",
             "White"]

counts = df["race"].value_counts().reindex(race_bins, fill_value=0)
print(counts)