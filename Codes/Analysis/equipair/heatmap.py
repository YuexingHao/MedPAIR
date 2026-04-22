import textwrap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV data
# Assumes:
# - First row contains column labels
# - First column contains row labels
df = pd.read_csv('Results/RelevancePercentageHeatmap.csv', header=0, index_col=0)

# Generate the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    df,
    cmap="Blues",
    annot=True,
    fmt='.2f',
    annot_kws={"size": 20},
    xticklabels=[textwrap.fill(c, width=10) for c in df.columns],
    yticklabels=df.index
)

plt.xlabel('Question', fontsize=26)
plt.ylabel('LLM', fontsize=26)
plt.tick_params(axis='both', labelsize=24)
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()