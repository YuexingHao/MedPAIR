import pandas as pd
import numpy as np
from scipy import stats
import re

df = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_PT_Removal/Qualitative Analysis/Pre-Study Survey.csv")  # Recent results

print(df.columns)

# Replace with your actual column name if it differs
col = "Are you familiar with any of the following clinical challenges? JAMA Clinical Challenge, NEJM Image Challenge, NEJM Resident 360."

# Count occurrences of each label
label_counts = df[col].value_counts(dropna=False)
print("Label counts:")
print(label_counts)

# Calculate proportions
label_proportions = df[col].value_counts(normalize=True, dropna=False)
print("\nLabel proportions:")
print(label_proportions)

# For ambiguity column
col_ambiguity = "If you are familiar with any of the clinical challenges, how regularly do you follow these challenges?"
ambiguity_counts = df[col_ambiguity].value_counts(dropna=False)
ambiguity_proportions = df[col_ambiguity].value_counts(normalize=True, dropna=False)
print("\nAmbiguity label counts:")
print(ambiguity_counts)
print("\nAmbiguity label proportions:")
print(ambiguity_proportions)

# For real-world correlation column
col_correlation = "Are you familiar with the MedBullets website?"
correlation_counts = df[col_correlation].value_counts(dropna=False)
correlation_proportions = df[col_correlation].value_counts(normalize=True, dropna=False)
print("\nReal-world correlation label counts:")
print(correlation_counts)
print("\nReal-world correlation label proportions:")
print(correlation_proportions)

col_skipped = "If you are familiar with the MedBullets website, how regularly do you follow the clinical challenges released there?"

# Count occurrences of each label
skipped_counts = df[col_skipped].value_counts(dropna=False)
print("\nMedBullets challenge following frequency counts:")
print(skipped_counts)

# Calculate proportions
skipped_proportions = df[col_skipped].value_counts(normalize=True, dropna=False)
print("\nMedBullets challenge following frequency proportions:")
print(skipped_proportions)

# 1. Are you familiar with any of the following clinical challenges?
col1 = "Are you familiar with any of the following clinical challenges? JAMA Clinical Challenge, NEJM Image Challenge, NEJM Resident 360."
print("\n--- Familiar with clinical challenges ---")
print(df[col1].value_counts(dropna=False))
print(df[col1].value_counts(normalize=True, dropna=False))

# 2. If you are familiar with any of the clinical challenges, how regularly do you follow these challenges?
col2 = "If you are familiar with any of the clinical challenges, how regularly do you follow these challenges?"
print("\n--- Regularly follow clinical challenges ---")
print(df[col2].value_counts(dropna=False))
print(df[col2].value_counts(normalize=True, dropna=False))

# 3. Are you familiar with the MedBullets website?
col3 = "Are you familiar with the MedBullets website?"
print("\n--- Familiar with MedBullets ---")
print(df[col3].value_counts(dropna=False))
print(df[col3].value_counts(normalize=True, dropna=False))

# 4. If you are familiar with the MedBullets website, how regularly do you follow the clinical challenges released there?
col4 = "If you are familiar with the MedBullets website, how regularly do you follow the clinical challenges released there?"
print("\n--- Regularly follow MedBullets challenges ---")
print(df[col4].value_counts(dropna=False))
print(df[col4].value_counts(normalize=True, dropna=False))

# 5. How familiar are you with the use of large language models (LLMs) in a clinical setting?
col5 = "How familiar are you with the use of large language models (LLMs) in a clinical setting? This might include using LLMs in the clinic, or asking LLMs (such as ChatGPT) clinical questions."
print("\n--- Familiarity with LLMs in clinical setting ---")
print(df[col5].value_counts(dropna=False))
print(df[col5].value_counts(normalize=True, dropna=False))

# 6. To what degree (in percentage) do you think LLMs are ready to be deployed in clinical settings (0-100)?
col6 = "To what degree (in percentage) do you think LLMs are ready to be deployed in clinical settings (0-100)?"

# Convert to numeric (in case there are any non-numeric entries)
df[col6] = pd.to_numeric(df[col6], errors='coerce')

mean_val = df[col6].mean()
std_val = df[col6].std()

print(f"\nAverage LLM readiness: {mean_val:.2f}")
print(f"Standard deviation of LLM readiness: {std_val:.2f}")

# 7. Age
col_age = "Age"
print("\n--- Age statistics ---")
df[col_age] = pd.to_numeric(df[col_age], errors='coerce')
print(df[col_age].describe())

# 8. Gender
col_gender = "Gender"
print("\n--- Gender counts ---")
print(df[col_gender].value_counts(dropna=False))
print(df[col_gender].value_counts(normalize=True, dropna=False))

# 9. Year of medical school?
col_year = "Year of medical school?"
print("\n--- Year of medical school counts ---")
print(df[col_year].value_counts(dropna=False))
print(df[col_year].value_counts(normalize=True, dropna=False))

# 10. USMLE Step 1 passed?
col_usmle = "USMLE Step 1 passed?"
print("\n--- USMLE Step 1 passed counts ---")
print(df[col_usmle].value_counts(dropna=False))
print(df[col_usmle].value_counts(normalize=True, dropna=False))

# 11. How many previous studies/tasks have they Labeled with Centaur Labs? Please enter a numerical value.
col_centaur = "How many previous studies/tasks have they Labeled with Centaur Labs? Please enter a numerical value."
print("\n--- Previous Centaur Labs tasks statistics ---")
df[col_centaur] = pd.to_numeric(df[col_centaur], errors='coerce')
print(df[col_centaur].describe())
