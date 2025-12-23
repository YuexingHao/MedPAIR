import pandas as pd

original = pd.read_csv(
    "medpair_baseline/results/gpt5-relevancy-combined-dec-12.csv")
irr_removed = pd.read_csv(
    "medpair_baseline/results/gpt5-irr-removed-relevancy-combined-dec-12.csv")

# SANITY CHECK
assert(original['ID_corr'].nunique() == irr_removed['ID_corr'].nunique())

# Gather IDs of questions answered correctly in original
correct = set()
for _, row in original.iterrows():
    if row["majority_vote"] == row["answer_corr"]:
        correct.add(row["ID_corr"])

# Compute spurious rate
round2_incorrect_ct = {"mmlu": 0, "jama": 0, "medxpert": 0, "medbullets": 0}
round1_correct_ct = {"mmlu": 0, "jama": 0, "medxpert": 0, "medbullets": 0}
for _, row in irr_removed.iterrows():
    if row["ID_corr"] in correct:
        round1_correct_ct[row["data_source_corr"]] += 1
        if row["majority_vote"] != f"<answer>Option {row["answer_corr"]}</answer>":
            round2_incorrect_ct[row["data_source_corr"]] += 1

print(round2_incorrect_ct)
print(round1_correct_ct)