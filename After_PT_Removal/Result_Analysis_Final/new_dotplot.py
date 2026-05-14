import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dataset categories for x-axis (removing GPT-4o)
categories = ['MMLU', 'JAMA', 'MedBullets', 'MedXpertQA']

# Processing methods (vertical dots in each category)
methods = [
    'Original', 
    'After Human Low+Irr Removal', 
    'After Qwen-72B Low+Irr Removal',
    'After Llama-70B Low+Irr Removal',
    'After GPT-4o SR Low+Irr Removal'
]

# Actual data from the user's table with SD values
data = {
    'JAMA': {
        'Original': {'acc': 68.5, 'sd': 0.5},
        'After Human Low+Irr Removal': {'acc': 78.7, 'sd': 0.4},
        'After Qwen-72B Low+Irr Removal': {'acc': 73.0, 'sd': 0.3},
        'After Llama-70B Low+Irr Removal': {'acc': 69.2, 'sd': 0.5},
        'After GPT-4o SR Low+Irr Removal': {'acc': 51.6, 'sd': 0.5}
    },
    'MMLU': {
        'Original': {'acc': 95.6, 'sd': 0.4},
        'After Human Low+Irr Removal': {'acc': 96.4, 'sd': 0.4},
        'After Qwen-72B Low+Irr Removal': {'acc': 93.8, 'sd': 0.4},
        'After Llama-70B Low+Irr Removal': {'acc': 93.2, 'sd': 0.4},
        'After GPT-4o SR Low+Irr Removal': {'acc': 86.0, 'sd': 0.5}
    },
    'MedBullets': {
        'Original': {'acc': 74.5, 'sd': 0.4},
        'After Human Low+Irr Removal': {'acc': 84.1, 'sd': 0.5},
        'After Qwen-72B Low+Irr Removal': {'acc': 76.8, 'sd': 0.5},
        'After Llama-70B Low+Irr Removal': {'acc': 74.6, 'sd': 0.5},
        'After GPT-4o SR Low+Irr Removal': {'acc': 64.0, 'sd': 0.3}
    },
    'MedXpertQA': {
        'Original': {'acc': 16.4, 'sd': 0.4},
        'After Human Low+Irr Removal': {'acc': 41.2, 'sd': 0.5},
        'After Qwen-72B Low+Irr Removal': {'acc': 41.0, 'sd': 0.5},
        'After Llama-70B Low+Irr Removal': {'acc': 38.8, 'sd': 0.5},
        'After GPT-4o SR Low+Irr Removal': {'acc': 8.7, 'sd': 0.3}
    }
}

# Colors for each method (similar to the attached image)
colors = {
    'Original': '#1f77b4',  # Blue
    'After Human Low+Irr Removal': '#ffff00',  # Yellow (highlighted in the table)
    'After Qwen-72B Low+Irr Removal': '#2ca02c',  # Green
    'After Llama-70B Low+Irr Removal': '#d62728',  # Red
    'After GPT-4o SR Low+Irr Removal': '#ff69b4'  # Pink (highlighted in the table)
}

# Create figure and axis with larger dimensions
fig, ax = plt.subplots(figsize=(20, 12))

# Set positions for each category on x-axis
cat_positions = np.arange(len(categories))

# Function to calculate increase from original
def calculate_increase(cat, method):
    orig_acc = data[cat]['Original']['acc']
    curr_acc = data[cat][method]['acc']
    return curr_acc - orig_acc

# Increase circle size
circle_size = 1300

# Plot data points for each category
for i, cat in enumerate(categories):
    # Define vertical positions for each method within this category
    y_positions = {method: data[cat][method]['acc'] for method in methods}
    
    # Plot original point first
    orig_acc = y_positions['Original']
    ax.scatter(i, orig_acc, s=circle_size, color=colors['Original'], edgecolors='black', linewidth=1.5, zorder=3, 
              label='Original' if i == 0 else "")
    
    # Plot other points without annotations
    for method in methods[1:]:  # Skip original
        acc = y_positions[method]
        
        # Plot point with bigger circles
        ax.scatter(i, acc, s=circle_size, color=colors[method], edgecolors='black', linewidth=1.5, zorder=3,
                  label=method if i == 0 else "")
        
        # Add error bars
        ax.errorbar(i, acc, yerr=data[cat][method]['sd'], color=colors[method], capsize=5, linewidth=2, capthick=2)
    
    # Add error bars for original point too
    ax.errorbar(i, orig_acc, yerr=data[cat]['Original']['sd'], color=colors['Original'], capsize=5, linewidth=2, capthick=2)

# Draw vertical lines connecting points in each category
for i, cat in enumerate(categories):
    y_min = min([data[cat][method]['acc'] for method in methods])
    y_max = max([data[cat][method]['acc'] for method in methods])
    # Add padding to make the line extend slightly beyond the points
    padding = 5
    ax.plot([i, i], [y_min - padding, y_max + padding], color='gray', linestyle='-', linewidth=1, alpha=0.3)

# Customize the plot with 3x larger fonts
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, len(categories) - 0.5)
ax.set_xticks(cat_positions)
ax.set_xticklabels(categories, fontsize=36)  # 3x larger font
ax.set_ylabel('Accuracy (%)', fontsize=42)  # 3x larger font
ax.tick_params(axis='y', labelsize=36)  # 3x larger Y-axis tick labels

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend (only once per method)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# Create a separate figure legend instead of trying to put it inside the axes
fig.legend(by_label.values(), by_label.keys(), 
           loc='upper center', bbox_to_anchor=(0.5, 0.98), 
           ncol=len(methods), fontsize=30)

# Add padding at the top for the legend
plt.subplots_adjust(top=0.85)

# Save the figure as PDF
output_filename = 'medical_datasets_comparison.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print(f"Plot saved as {output_filename}") 