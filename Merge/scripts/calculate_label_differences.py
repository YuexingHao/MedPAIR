#!/usr/bin/env python3
import pandas as pd
import numpy as np

def calculate_label_differences(csv1_path, csv2_path):
    """
    Calculate differences between label_1 to label_21 columns in two CSV files.
    """
    
    # Read the CSV files
    print(f"Reading {csv1_path}...")
    df1 = pd.read_csv(csv1_path)
    print(f"Reading {csv2_path}...")
    df2 = pd.read_csv(csv2_path)
    
    print(f"CSV1 shape: {df1.shape}")
    print(f"CSV2 shape: {df2.shape}")
    
    # Get label columns (label_1 to label_21)
    label_cols = [f'label_{i}' for i in range(1, 22)]
    
    # Check which label columns exist in both dataframes
    available_cols1 = [col for col in label_cols if col in df1.columns]
    available_cols2 = [col for col in label_cols if col in df2.columns]
    common_cols = list(set(available_cols1) & set(available_cols2))
    
    print(f"Available label columns in CSV1: {available_cols1}")
    print(f"Available label columns in CSV2: {available_cols2}")
    print(f"Common label columns: {common_cols}")
    
    if not common_cols:
        print("No common label columns found!")
        return None
    
    # Ensure both dataframes have the same number of rows
    min_rows = min(len(df1), len(df2))
    df1 = df1.head(min_rows)
    df2 = df2.head(min_rows)
    
    # Calculate differences and matching percentages
    results = []
    
    for idx in range(min_rows):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]
        
        # Get label values for this row
        labels1 = [row1[col] for col in common_cols if pd.notna(row1[col])]
        labels2 = [row2[col] for col in common_cols if pd.notna(row2[col])]
        
        # Count matches (assuming labels are comparable)
        matches = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
        total_labels = len(labels1)
        
        if total_labels > 0:
            match_percentage = (matches / total_labels) * 100
        else:
            match_percentage = 0
        
        # Check if 3/5 sentences match (assuming 5 sentences total)
        three_out_of_five_match = match_percentage >= 60  # 3/5 = 60%
        
        results.append({
            'row_index': idx,
            'total_labels': total_labels,
            'matches': matches,
            'match_percentage': match_percentage,
            'three_out_of_five_match': three_out_of_five_match,
            'labels1': labels1,
            'labels2': labels2
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total rows analyzed: {len(results_df)}")
    print(f"Average match percentage: {results_df['match_percentage'].mean():.2f}%")
    print(f"Rows with 3/5+ matches: {results_df['three_out_of_five_match'].sum()}")
    print(f"Percentage of rows with 3/5+ matches: {(results_df['three_out_of_five_match'].sum() / len(results_df)) * 100:.2f}%")
    
    return results_df

def main():
    # File paths
    csv1_path = "FIRST_GPT4o_Remove_Irrelevant.csv"
    csv2_path = "SECOND_GPT4o_Remove_Irrelevant.csv"
    
    # Calculate differences
    results = calculate_label_differences(csv1_path, csv2_path)
    
    if results is not None:
        # Save results
        output_file = "label_difference_results.csv"
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Display first few rows
        print("\n=== FIRST 10 ROWS ===")
        print(results[['row_index', 'total_labels', 'matches', 'match_percentage', 'three_out_of_five_match']].head(10))

if __name__ == "__main__":
    main() 