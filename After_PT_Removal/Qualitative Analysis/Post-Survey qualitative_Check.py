import pandas as pd
import numpy as np
from scipy import stats
import re

df = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_PT_Removal/Qualitative Analysis/Exit_Survey.csv")  # Recent results

print(df.columns)

# Replace with your actual column name if it differs
col = "How confident were you, on average, in your answers for the sentence relevance scores?"

# Count occurrences of each label
label_counts = df[col].value_counts(dropna=False)
print("Label counts:")
print(label_counts)

# Calculate proportions
label_proportions = df[col].value_counts(normalize=True, dropna=False)
print("\nLabel proportions:")
print(label_proportions)

# For ambiguity column
col_ambiguity = "Please rate the degree of ambiguity you perceive in the true relevance labels on a scale from 1 to 5."
ambiguity_counts = df[col_ambiguity].value_counts(dropna=False)
ambiguity_proportions = df[col_ambiguity].value_counts(normalize=True, dropna=False)
print("\nAmbiguity label counts:")
print(ambiguity_counts)
print("\nAmbiguity label proportions:")
print(ambiguity_proportions)

# For real-world correlation column
col_correlation = "To what degree do you think the process of answering multiple choice clinical QAs correlates with real-world day-to-day medical practice?"
correlation_counts = df[col_correlation].value_counts(dropna=False)
correlation_proportions = df[col_correlation].value_counts(normalize=True, dropna=False)
print("\nReal-world correlation label counts:")
print(correlation_counts)
print("\nReal-world correlation label proportions:")
print(correlation_proportions)

col_skipped = "Approximately what percentage of questions did you skip?"

def range_to_avg(val):
    if pd.isna(val):
        return np.nan
    # Match patterns like "11 - 20" (with spaces)
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", str(val).strip())
    if match:
        low, high = map(int, match.groups())
        return (low + high) / 2
    # Match single numbers like "0", "5"
    match_single = re.match(r"^(\d+)$", str(val).strip())
    if match_single:
        return int(match_single.group(1))
    # If not a number or range, return NaN
    return np.nan

# Apply the function to the column
df['Skipped_Avg'] = df[col_skipped].apply(range_to_avg)

# Show summary statistics for the averages
print("\nSummary statistics for average percentage skipped:")
print(df['Skipped_Avg'].describe())
