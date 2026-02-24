import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV data
# Assumes:
# - First row contains column labels
# - First column contains row labels
df = pd.read_csv('Results/OverallHeatmap.csv', header=0, index_col=0)

# Generate the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df,
    cmap="Blues",
    annot=True,
    annot_kws={"size": 20},
    xticklabels=df.columns,   # Explicit column labels
    yticklabels=df.index      # Explicit row labels
)

plt.title('Refusal Rate Heatmap')
plt.xlabel('Question')
plt.ylabel('LLM')
plt.tight_layout()
plt.show()