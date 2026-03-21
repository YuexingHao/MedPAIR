import pandas as pd

path = "Results/medgemma/equipair/"
# Load files
separate = pd.read_csv(f"{path}MedGemma_Merged_Intersection.csv")
combined = pd.read_csv(f"{path}all_questions.csv")

# --- Restrict to intersecting IDs ---
common_ids = set(separate["ID_corr"]).intersection(set(combined["ID_corr"]))

separate = separate[separate["ID_corr"].isin(common_ids)]
combined = combined[combined["ID_corr"].isin(common_ids)]

def aggregate_relevant(df, version_name):
    label_cols = [c for c in df.columns if c.startswith("label_")]
    
    out = pd.DataFrame()
    out["ID_corr"] = df["ID_corr"]
    
    out[f"{version_name}_relevant"] = (
        (df[label_cols] == "Low Relevance") |
        (df[label_cols] == "High Relevance")
    ).sum(axis=1)
    
    return out

separate_agg = aggregate_relevant(separate, "separate")
combined_agg = aggregate_relevant(combined, "combined")

# Merge → now guaranteed intersection only
final = separate_agg.merge(combined_agg, on="ID_corr", how="inner")

# Comparison
final["delta"] = final["combined_relevant"] - final["separate_relevant"]
final["combined_more_relevance"] = final["delta"] > 0

# Stats
num_combined_more = final["combined_more_relevance"].sum()
total = len(final)
print(f"Combined higher relevance on {num_combined_more} / {total} QAs ({num_combined_more/total:.2%})")

# Save
final.to_csv(f"{path}qa_relevant_counts.csv", index=False)

print(final.head())