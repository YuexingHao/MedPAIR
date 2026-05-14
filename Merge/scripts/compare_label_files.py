#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

def analyze_csv_structure(csv_path):
    """Analyze the structure of a CSV file to understand its columns."""
    try:
        df = pd.read_csv(csv_path)
        print(f"\n=== {os.path.basename(csv_path)} ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Find label columns
        label_cols = [col for col in df.columns if col.startswith('label_')]
        print(f"Label columns found: {label_cols}")
        
        return df, label_cols
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, []

def calculate_label_differences(csv1_path, csv2_path, csv3_path=None):
    """
    Calculate differences between label columns in CSV files.
    Return percentage of matching sentences for each row.
    """
    
    # Analyze all CSV files
    print("Analyzing CSV file structures...")
    df1, labels1 = analyze_csv_structure(csv1_path)
    df2, labels2 = analyze_csv_structure(csv2_path)
    
    if csv3_path and os.path.exists(csv3_path):
        df3, labels3 = analyze_csv_structure(csv3_path)
    else:
        df3, labels3 = None, []
    
    if df1 is None or df2 is None:
        print("Error: Could not read one or more CSV files")
        return None
    
    # Find common label columns
    common_labels = list(set(labels1) & set(labels2))
    if df3 is not None:
        common_labels = list(set(common_labels) & set(labels3))
    
    print(f"\nCommon label columns: {common_labels}")
    
    if not common_labels:
        print("No common label columns found!")
        return None
    
    # Ensure all dataframes have the same number of rows
    min_rows = min(len(df1), len(df2))
    if df3 is not None:
        min_rows = min(min_rows, len(df3))
    
    df1 = df1.head(min_rows)
    df2 = df2.head(min_rows)
    if df3 is not None:
        df3 = df3.head(min_rows)
    
    # Calculate differences and matching percentages
    results = []
    
    for idx in range(min_rows):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]
        
        # Get label values for this row
        labels1_vals = [row1[col] for col in common_labels if pd.notna(row1[col])]
        labels2_vals = [row2[col] for col in common_labels if pd.notna(row2[col])]
        
        # Count matches between first two files
        matches_1_2 = sum(1 for l1, l2 in zip(labels1_vals, labels2_vals) if l1 == l2)
        total_labels = len(labels1_vals)
        
        if total_labels > 0:
            match_percentage_1_2 = (matches_1_2 / total_labels) * 100
        else:
            match_percentage_1_2 = 0
        
        # Check if 3/5 sentences match (assuming 5 sentences total)
        three_out_of_five_match = match_percentage_1_2 >= 60  # 3/5 = 60%
        
        result_row = {
            'row_index': idx,
            'total_labels': total_labels,
            'matches_1_2': matches_1_2,
            'match_percentage_1_2': match_percentage_1_2,
            'three_out_of_five_match': three_out_of_five_match,
            'labels1': labels1_vals,
            'labels2': labels2_vals
        }
        
        # Add third file comparison if available
        if df3 is not None:
            row3 = df3.iloc[idx]
            labels3_vals = [row3[col] for col in common_labels if pd.notna(row3[col])]
            
            matches_1_3 = sum(1 for l1, l3 in zip(labels1_vals, labels3_vals) if l1 == l3)
            matches_2_3 = sum(1 for l2, l3 in zip(labels2_vals, labels3_vals) if l2 == l3)
            
            match_percentage_1_3 = (matches_1_3 / total_labels) * 100 if total_labels > 0 else 0
            match_percentage_2_3 = (matches_2_3 / total_labels) * 100 if total_labels > 0 else 0
            
            result_row.update({
                'matches_1_3': matches_1_3,
                'matches_2_3': matches_2_3,
                'match_percentage_1_3': match_percentage_1_3,
                'match_percentage_2_3': match_percentage_2_3,
                'labels3': labels3_vals
            })
        
        results.append(result_row)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total rows analyzed: {len(results_df)}")
    print(f"Average match percentage (1 vs 2): {results_df['match_percentage_1_2'].mean():.2f}%")
    print(f"Rows with 3/5+ matches (1 vs 2): {results_df['three_out_of_five_match'].sum()}")
    print(f"Percentage of rows with 3/5+ matches: {(results_df['three_out_of_five_match'].sum() / len(results_df)) * 100:.2f}%")
    
    if df3 is not None:
        print(f"Average match percentage (1 vs 3): {results_df['match_percentage_1_3'].mean():.2f}%")
        print(f"Average match percentage (2 vs 3): {results_df['match_percentage_2_3'].mean():.2f}%")
    
    return results_df

def main():
    # File paths
    csv1_path = "FIRST_GPT4o_Remove_Irrelevant.csv"
    csv2_path = "SECOND_GPT4o_Remove_Irrelevant.csv"
    csv3_path = "THIRD_annotated_2k_relevancy.csv"  # Optional third file
    
    # Check if files exist
    if not os.path.exists(csv1_path):
        print(f"Error: {csv1_path} not found")
        return
    
    if not os.path.exists(csv2_path):
        print(f"Error: {csv2_path} not found")
        return
    
    # Calculate differences
    results = calculate_label_differences(csv1_path, csv2_path, csv3_path)
    
    if results is not None:
        # Save results
        output_file = "label_comparison_results.csv"
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Display first few rows
        print("\n=== FIRST 10 ROWS ===")
        display_cols = ['row_index', 'total_labels', 'matches_1_2', 'match_percentage_1_2', 'three_out_of_five_match']
        if 'matches_1_3' in results.columns:
            display_cols.extend(['matches_1_3', 'match_percentage_1_3', 'match_percentage_2_3'])
        
        print(results[display_cols].head(10))

if __name__ == "__main__":
    main() 