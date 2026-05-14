import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

os.chdir(Path(__file__).resolve().parent)

physicians = pd.read_csv("2k_4k_ID_Physician_Final_Sentence.csv")

# Add this after reading the physicians CSV
print("\nPhysicians dataframe shape:", physicians.shape)
print("Columns in physicians DataFrame:", physicians.columns.tolist())
print(f"Sample 4k_IDs from physicians: {physicians['4k_ID'].head().tolist()}")
print(f"4k_ID data type: {physicians['4k_ID'].dtype}")

# Standardize the 4k_ID column in physicians dataframe
physicians['4k_ID_std'] = physicians['4k_ID'].astype(str).str.replace('_', ' ')

# Read the 14B_Sentence_Numbers.csv file
print("\nReading 14B_Sentence_Numbers.csv...")
B14_sentence_df = pd.read_csv("14B_Sentence_Numbers.csv")
print(f"14B_Sentence_Numbers.csv shape: {B14_sentence_df.shape}")
print(f"Columns: {B14_sentence_df.columns.tolist()}")
print(f"First few rows:\n{B14_sentence_df.head()}")

# Merge the physicians data with the 14B_sentence_df
print("\nMerging physicians data with 14B_sentence_df...")
merged_df = pd.merge(
    B14_sentence_df, 
    physicians[['4k_ID_std', 'ID_corr', 'human_sentence_ids', 'data_source_corr', 'sentence_number_corr', 'keep_k']],
    left_on='QA_ID_std',
    right_on='4k_ID_std',
    how='left'
)

# Print merge statistics
print(f"Original 14B_sentence_df shape: {B14_sentence_df.shape}")
print(f"Merged dataframe shape: {merged_df.shape}")
print(f"Number of matches: {merged_df['ID_corr'].notna().sum()}")

# Display first few rows of the merged dataframe
print("\nFirst few rows of the merged dataframe:")
print(merged_df.head())

# Calculate match rate between 14B_sentence_number and human_sentence_ids
def calculate_match_rate(row):
    try:
        # Parse the 14B sentence numbers (comma-separated)
        b14_numbers = set([int(num.strip()) for num in str(row['14B_sentence_number']).split(',') if num.strip()])
        
        # Parse the human sentence IDs (could be comma or dot-separated)
        human_str = str(row['human_sentence_ids'])
        if ',' in human_str:
            human_numbers = set([int(num.strip()) for num in human_str.split(',') if num.strip()])
        else:
            # Handle potential dot notation
            human_numbers = set([int(num.strip()) for num in human_str.replace('.', ',').split(',') if num.strip()])
        
        # Find common elements
        common = b14_numbers.intersection(human_numbers)
        
        # Calculate match rate
        if len(b14_numbers) > 0:
            return len(common) / len(b14_numbers) * 100
        else:
            return 0
    except (ValueError, TypeError):
        # Handle any parsing errors
        return np.nan

# Apply the function to calculate match rates
merged_df['match_rate'] = merged_df.apply(calculate_match_rate, axis=1)

# Calculate overall accuracy statistics
mean_match_rate = merged_df['match_rate'].mean()
std_match_rate = merged_df['match_rate'].std()

print("\n=== Overall Match Rate Statistics ===")
print(f"Mean match rate: {mean_match_rate:.2f}%")
print(f"Standard deviation: {std_match_rate:.2f}%")

# Calculate statistics by data_source_corr
print("\n=== Match Rate Statistics by Data Source ===")
data_source_stats = merged_df.groupby('data_source_corr')['match_rate'].agg(['count', 'mean', 'std'])
data_source_stats.columns = ['Count', 'Mean Match Rate (%)', 'Std Dev']
data_source_stats = data_source_stats.sort_values('Mean Match Rate (%)', ascending=False)
print(data_source_stats)

# Add detailed match information to the dataframe
def get_match_details(row):
    try:
        b14_numbers = set([int(num.strip()) for num in str(row['14B_sentence_number']).split(',') if num.strip()])
        
        human_str = str(row['human_sentence_ids'])
        if ',' in human_str:
            human_numbers = set([int(num.strip()) for num in human_str.split(',') if num.strip()])
        else:
            human_numbers = set([int(num.strip()) for num in human_str.replace('.', ',').split(',') if num.strip()])
        
        common = b14_numbers.intersection(human_numbers)
        only_14b = b14_numbers - human_numbers
        only_human = human_numbers - b14_numbers
        
        return pd.Series({
            'common_sentences': ','.join(map(str, sorted(common))),
            'only_14b_sentences': ','.join(map(str, sorted(only_14b))),
            'only_human_sentences': ','.join(map(str, sorted(only_human))),
            'num_common': len(common),
            'num_14b': len(b14_numbers),
            'num_human': len(human_numbers)
        })
    except (ValueError, TypeError):
        return pd.Series({
            'common_sentences': '',
            'only_14b_sentences': '',
            'only_human_sentences': '',
            'num_common': 0,
            'num_14b': 0,
            'num_human': 0
        })

# Apply the function to get match details
match_details = merged_df.apply(get_match_details, axis=1)
merged_df = pd.concat([merged_df, match_details], axis=1)

# Save the enriched dataframe with match statistics
merged_df.to_csv("14B_Physician_Comparison_with_Stats.csv", index=False)
print("\nEnriched dataframe saved to '14B_Physician_Comparison_with_Stats.csv'")

# Check for any QA_ID_std that didn't find a match
unmatched = merged_df[merged_df['ID_corr'].isna()]
if not unmatched.empty:
    print(f"\nWarning: {len(unmatched)} rows in 14B_sentence_df did not find a match in physicians")
    print("First few unmatched QA_ID_std values:")
    print(unmatched['QA_ID_std'].head().tolist())

