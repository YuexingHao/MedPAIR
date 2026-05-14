import os
from pathlib import Path

import pandas as pd

# Run relative to this script (works on any machine)
os.chdir(Path(__file__).resolve().parent)

# Read the already combined file
df = pd.read_csv("attributions_summary_combined.csv", on_bad_lines='skip')

print(df.columns.tolist())

# Create the new column 'Centaur' in df with default value False
df['Centaur'] = False

physicians = pd.read_csv("Physician_Selected_Sentence_IDs.csv")

# Add this after reading the physicians CSV
print("Columns in physicians DataFrame:", physicians.columns.tolist())

## Sentence Number Matching
sentence_ID_Match = pd.read_csv("Sentence_Label_Original_2k.csv")

# First, sort the dataframe by QA_ID and Score (descending)
df_sorted = df.sort_values(['QA_ID', 'Score'], ascending=[True, False])

# Create a list to store the results
result_data = []

# Get the unique QA_IDs
unique_qa_ids = df_sorted['QA_ID'].unique()

# For each QA_ID
for qa_id in unique_qa_ids:
    # Get rows with this QA_ID
    qa_rows = df_sorted[df_sorted['QA_ID'] == qa_id]
    
    # Get the first Extracted_Answer (should be the same for all rows with this QA_ID)
    extracted_answer = qa_rows['Extracted_Answer'].iloc[0]
    
    # Convert each Source value to string before joining
    source_strings = [str(source) for source in qa_rows['Source']]
    merged_source = ' '.join(source_strings)
    
    # Add to result data
    result_data.append({
        'QA_ID': qa_id,
        'Extracted_Answer': extracted_answer,
        'merged_Source': merged_source
    })

# Create the new dataframe
Qwen_72B_Sentence_Selection = pd.DataFrame(result_data)

# Display the first few rows to verify
print(Qwen_72B_Sentence_Selection.head())

# Save to CSV if needed
Qwen_72B_Sentence_Selection.to_csv("Qwen_72B_Sentence_Selection.csv", index=False)

# First load the sentence_ID_Match dataframe (adjust path if needed)
sentence_ID_Match = pd.read_csv("Sentence_Label_Original_2k.csv")

# Function to find matching sentence column
def find_matching_sentence(source, row):
    source_str = str(source).lower()  # Convert to lowercase string for better matching
    
    # Check each sentence column
    for i in range(1, 22):  # Assuming columns are named sentence_1 through sentence_21
        col_name = f"sentence_{i}"
        if col_name in row:
            sentence_value = str(row[col_name]).lower()
            
            # Check if source is contained in the sentence or vice versa
            # This handles cases like "ddd" matching with "6. ddd"
            if source_str in sentence_value or sentence_value in source_str:
                return col_name
    
    # Return None if no match found
    return None


# Print columns before merge to verify
print("Columns in sentence_ID_Match:", sentence_ID_Match.columns.tolist())
print("Columns in physicians:", physicians.columns.tolist())

# Merge dataframes
merged_sentence_physician = pd.merge(
    sentence_ID_Match, 
    physicians,
    left_on="ID",
    right_on="Origin",
    how="left"  # Keep all rows from sentence_ID_Match
)

# Display the first few rows to verify
print(merged_sentence_physician.head())

# Count how many matches we got
match_count = merged_sentence_physician.dropna(subset=["Origin"]).shape[0]
print(f"Found {match_count} matches between sentence_ID_Match and physicians")

# Save to CSV if needed
merged_sentence_physician.to_csv("Merged_Sentence_Physician.csv", index=False)


# Apply the function row by row
df['sentence number'] = None  # Initialize new column

# We need to merge df with sentence_ID_Match first to have all columns available
# Update to use the correct column names for the merge
merged_df = pd.merge(
    df, 
    physicians, 
    left_on='QA_ID',  # Column name in df
    right_on='4k_ID',  # Column name in sentence_ID_Match
    how='left'
)

# Now we can check for matches
for idx, row in merged_df.iterrows():
    df.at[idx, 'sentence number'] = find_matching_sentence(row['Source'], row)

# Show the first few rows to verify
print(df[['QA_ID', 'Score','Source', 'sentence number']].head(30))

print("Columns in sentence_ID_Match:", sentence_ID_Match.columns.tolist())

# Before filtering
print(f"Number of rows before filtering: {len(df)}")

# Filter out rows where Source is only a period
df = df[~(df['Source'] == '.')]

# After filtering
print(f"Number of rows after filtering: {len(df)}")

# Show a sample of the filtered DataFrame to verify
print(df[['QA_ID', 'Score', 'Source', 'sentence number']].head(10))

# Initialize the new column with False
df['Select_Physicians?'] = False

# First, create a mapping from QA_ID to keep_k from the physicians DataFrame
keep_k_mapping = {}
for idx, row in physicians.iterrows():
    if '4k_ID' in row and not pd.isna(row['4k_ID']) and 'keep_k' in row:
        keep_k_mapping[row['4k_ID']] = int(row['keep_k'])

# For each QA_ID in df
for qa_id in df['QA_ID'].unique():
    # Skip if this QA_ID is not in our mapping
    if qa_id not in keep_k_mapping:
        continue
    
    # Get the keep_k value for this QA_ID
    k = keep_k_mapping[qa_id]
    
    # Get all rows with this QA_ID and sort by Score in descending order
    indices = df[df['QA_ID'] == qa_id].sort_values('Score', ascending=False).head(k).index
    
    # Mark the top k rows as True
    df.loc[indices, 'Select_Physicians?'] = True

# Check the results
print(f"Total rows marked as selected by physicians: {df['Select_Physicians?'].sum()}")
print("Sample of selected rows:")
print(df[df['Select_Physicians?'] == True][['QA_ID', 'Score', 'Source', 'sentence number', 'Select_Physicians?']].head(10))

# Add a column to show the keep_k value for each QA_ID
df['Physician_keep_k'] = df['QA_ID'].map(keep_k_mapping)

# Show sample rows with the new column included
print("Sample rows with keep_k values:")
print(df[['QA_ID', 'Score', 'Source', 'Physician_keep_k', 'Select_Physicians?']].head(20))

# Show sample of selected rows with keep_k values
print("\nSelected rows with keep_k values:")
print(df[df['Select_Physicians?'] == True][['QA_ID', 'Score', 'Source', 'Physician_keep_k', 'sentence number', 'Select_Physicians?']].head(10))

# Calculate some statistics about the keep_k values
if len(keep_k_mapping) > 0:
    print(f"\nAverage keep_k value: {sum(keep_k_mapping.values()) / len(keep_k_mapping)}")
    print(f"Min keep_k value: {min(keep_k_mapping.values())}")
    print(f"Max keep_k value: {max(keep_k_mapping.values())}")
    print(f"Most common keep_k value: {pd.Series(list(keep_k_mapping.values())).value_counts().index[0]}")

# Create a dictionary to store sentence numbers for each QA_ID
sentence_mapping = {}

# Loop through the df dataframe
for qa_id in df['QA_ID'].unique():
    # Find all rows where Select_Physicians? is True for this QA_ID
    selected_rows = df[(df['QA_ID'] == qa_id) & (df['Select_Physicians?'] == True)]
    
    if len(selected_rows) > 0:
        # Extract the sentence numbers
        sentence_numbers = []
        for sentence in selected_rows['sentence number'].dropna():
            # Extract just the number from strings like "sentence_5"
            if isinstance(sentence, str) and 'sentence_' in sentence:
                num = sentence.replace('sentence_', '')
                try:
                    sentence_numbers.append(int(num))
                except ValueError:
                    continue
        
        # Sort the numbers and join them with commas
        if sentence_numbers:
            sentence_numbers.sort()  # Sort from smallest to largest
            sentence_mapping[qa_id] = ','.join(map(str, sentence_numbers))

# Add the new column to physicians DataFrame
physicians['72B_Sentences'] = physicians['4k_ID'].map(sentence_mapping)

# Display results
print("Physicians DataFrame with 72B_Sentences column:")
print(physicians[['4k_ID', 'keep_k', 'human_sentence_ids', '72B_Sentences']].head(20))

# Save the updated physicians DataFrame to CSV
physicians.to_csv("Physician_Selected_Sentence_IDs_Updated.csv", index=False)

# Create a dictionary to store merged Source content for each QA_ID
source_content_mapping = {}

# Loop through the df dataframe
for qa_id in df['QA_ID'].unique():
    # Find all rows where Select_Physicians? is True for this QA_ID
    selected_rows = df[(df['QA_ID'] == qa_id) & (df['Select_Physicians?'] == True)]
    
    if len(selected_rows) > 0:
        # Extract the Source contents and join them
        sources = selected_rows['Source'].dropna().astype(str).tolist()
        if sources:
            source_content_mapping[qa_id] = ' | '.join(sources)

# Add the new column to physicians DataFrame
physicians['72B_Sentence_Contents'] = physicians['4k_ID'].map(source_content_mapping)

# Display results
print("Physicians DataFrame with 72B_Sentence_Contents column:")
print(physicians[['4k_ID', 'keep_k', '72B_Sentences', '72B_Sentence_Contents']].head(10))

# Save the updated physicians DataFrame to CSV
physicians.to_csv("Physician_Selected_Sentence_IDs_Updated.csv", index=False)

# Function to calculate match rate between two comma-separated number strings
def calculate_match_rate(human_ids, model_ids):
    if pd.isna(human_ids) or pd.isna(model_ids) or human_ids == '' or model_ids == '':
        return 0.0
    
    # Convert comma-separated strings to sets of integers
    try:
        human_set = set(int(x.strip()) for x in str(human_ids).split(','))
        model_set = set(int(x.strip()) for x in str(model_ids).split(','))
    except:
        return 0.0
    
    # If either set is empty, return 0
    if not human_set or not model_set:
        return 0.0
    
    # Calculate intersection and union
    intersection = human_set.intersection(model_set)
    union = human_set.union(model_set)
    
    # Calculate match rate
    return len(intersection) / len(union)

# Apply the function to calculate match rate for each row
physicians['match_rate'] = physicians.apply(
    lambda row: calculate_match_rate(row['human_sentence_ids'], row['72B_Sentences']), 
    axis=1
)

# Overall statistics
print("===== Overall Match Rate Statistics =====")
print(f"Mean match rate: {physicians['match_rate'].mean():.4f}")
print(f"Median match rate: {physicians['match_rate'].median():.4f}")
print(f"Min match rate: {physicians['match_rate'].min():.4f}")
print(f"Max match rate: {physicians['match_rate'].max():.4f}")
print(f"Standard deviation: {physicians['match_rate'].std():.4f}")
print(f"Total rows with data: {physicians['match_rate'].count()}")
print(f"Rows with exact matches (rate=1.0): {(physicians['match_rate'] == 1.0).sum()}")
print(f"Rows with no matches (rate=0.0): {(physicians['match_rate'] == 0.0).sum()}")

# Statistics by data_source
if 'data_source_corr' in physicians.columns:
    print("\n===== Match Rate by Data Source =====")
    datasource_stats = physicians.groupby('data_source_corr')['match_rate'].agg(['mean', 'median', 'min', 'max', 'std', 'count'])
    datasource_stats = datasource_stats.sort_values('mean', ascending=False)
    print(datasource_stats)
    
    # Count of exact and zero matches by data_source
    print("\nExact matches by data source:")
    print(physicians[physicians['match_rate'] == 1.0]['data_source_corr'].value_counts())
    
    print("\nZero matches by data source:")
    print(physicians[physicians['match_rate'] == 0.0]['data_source_corr'].value_counts())
else:
    print("\nColumn 'data_source_corr' not found in the DataFrame")

# Save the updated DataFrame with match rates
physicians.to_csv("Physician_Selected_Sentence_IDs_With_Match_Rates.csv", index=False)

# Visualize the distribution of match rates
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(physicians['match_rate'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Match Rates Between Human and 72B Model Selections')
    plt.xlabel('Match Rate')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('match_rate_distribution.png')
    print("\nMatch rate distribution saved as 'match_rate_distribution.png'")
except Exception as e:
    print(f"\nCouldn't create visualization: {str(e)}")

# Add after line 75 (after the find_matching_sentence function)
print("Sample merged_df rows:")
print(merged_df[['QA_ID', 'Source', '4k_ID']].head())
print(f"Number of matches in merged_df: {len(merged_df.dropna(subset=['4k_ID']))}")

# Add after line 158 (after creating sentence_mapping)
print("QA_IDs with sentence mappings:", list(sentence_mapping.keys())[:10])
print("Sample sentence mappings:", {k: sentence_mapping[k] for k in list(sentence_mapping.keys())[:5]})
print("Number of QA_IDs with sentence mappings:", len(sentence_mapping))

# Add a check of filename format in the file reading loop (around line 12)
print("Sample filenames from 14B folder:", [f for f in os.listdir(input_folder) if f.endswith(".csv")][:5])