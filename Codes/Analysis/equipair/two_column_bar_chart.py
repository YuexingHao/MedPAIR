import pandas as pd

df = pd.read_csv("Results/gpt4o/equipair/all_questions_cleaned.csv")

print(len(df))

bins = [
    "$0 - $11,925",
    "$11,926 - $48,475",
    "$48,476 - $103,350",
    "$103,351 - $197,300",
    "$197,301 - $250,525",
    "$250,526 - $626,350",
    "Over $626,350"
]

counts = df["household_income"].value_counts().reindex(bins, fill_value=0)
print(counts)