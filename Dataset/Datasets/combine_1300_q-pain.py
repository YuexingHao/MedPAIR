import pandas as pd

medpair_1300 = pd.read_csv("Dataset/Datasets/1300_QA.csv")
q_pain = pd.read_csv("Dataset/Datasets/q-pain.csv")
combined_df = pd.concat([medpair_1300, q_pain], ignore_index=True)

combined_df.to_csv("Dataset/Datasets/medpair_and_q-pain.csv", index=False)