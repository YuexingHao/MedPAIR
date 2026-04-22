import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory

df = pd.read_csv('Results/relevance_breakdown.csv')

models = ["gpt4o", "gpt5", "llama", "medgemma", "qwen"]
model_labels = ["GPT-4o", "GPT-5", "LLaMA", "MedGemma", "Qwen"]
dimensions = ["Household Income", "Housing Status", "Insurance Status", "Race", "All Questions"]
categories = ["Irrelevant", "Low Relevance", "High Relevance"]
colors = ["#5498c4", "#b6c3f0", "#c3e2e2"]

n_models = len(models)
n_dims = len(dimensions)
bar_height = 0.3
dim_gap = 0.04
group_gap = 0.28

model_centers = np.arange(n_models) * (n_dims * (bar_height + dim_gap) + group_gap)
dim_offsets = np.arange(n_dims) * (bar_height + dim_gap)
dim_offsets -= dim_offsets.mean()

fig, ax = plt.subplots(figsize=(16, 9))

all_bar_y = []
all_bar_labels = []

for m_idx, model in enumerate(models):
    model_df = df[df['Model'] == model].set_index('SDoH Dimension')
    for d_idx, dim in enumerate(dimensions):
        y = model_centers[m_idx] + dim_offsets[d_idx]
        all_bar_y.append(y)
        all_bar_labels.append(dim)
        left = 0
        for cat, color in zip(categories, colors):
            val = model_df.loc[dim, f"{cat} (%)"] if dim in model_df.index else 0
            ax.barh(y, val, height=bar_height, left=left, color=color,
                    edgecolor='white', linewidth=0.5)
            left += val

# SDoH dimension names as y-axis tick labels
ax.set_yticks(all_bar_y)
ax.set_yticklabels(all_bar_labels, fontsize=8)
ax.tick_params(axis='y', length=0, pad=4)

# Model name labels placed in the figure margin, well to the left of the SDoH tick labels
trans = blended_transform_factory(fig.transFigure, ax.transData)
for m_idx, label in enumerate(model_labels):
    ax.text(0.01, model_centers[m_idx], label, ha='left', va='center',
            fontsize=11, fontweight='bold', transform=trans)

ax.set_xlabel('Percentage of Sentences (%)', fontsize=11)
ax.set_xlim(0, 102)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.tick_params(axis='x', labelsize=10)

ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, categories)]
ax.legend(handles=legend_patches, fontsize=11, loc='upper left',
          bbox_to_anchor=(-0.27, 0.08), bbox_transform=ax.transAxes,
          framealpha=0.9)

ax.set_title('Sentence Relevance Label Distribution by Model and SDoH Dimension', fontsize=13, pad=12)

plt.tight_layout()
plt.subplots_adjust(left=0.28)
plt.savefig('Results/relevance_stacked_bar.png', dpi=150, bbox_inches='tight')
plt.show()
