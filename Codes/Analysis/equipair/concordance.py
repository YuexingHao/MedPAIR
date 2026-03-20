import pandas as pd

dir_path = "Results/medgemma/equipair/"
individual_questions = pd.read_csv(f"{dir_path}combined_relevance.csv")
combined_questions = pd.read_csv(f"{dir_path}all_questions.csv")

overlap_sum = 0
overlap_ct = 0

for _, row in individual_questions.iterrows():
    id = row["ID_corr"]
    if id in combined_questions['ID_corr'].values:
        relevant = set()
        ind_row = individual_questions.loc[individual_questions["ID_corr"] == id].iloc[0]
        for i in range(30):
            if ind_row[f"label_{i+1}"] == "Low Relevance" or ind_row[f"label_{i+1}"] == "High Relevance":
                relevant.add(i)
        combined_row = combined_questions.loc[combined_questions["ID_corr"] == id].iloc[0]
        overlap = 0
        for i in range(30):
            if combined_row[f"label_{i+1}"] == "Low Relevance" or combined_row[f"label_{i+1}"] == "High Relevance":
                if i in relevant:
                    overlap += 1
        
        if len(relevant) > 0:
            overlap_sum += (overlap / len(relevant))
            overlap_ct += 1
    
print(overlap_sum / overlap_ct)
