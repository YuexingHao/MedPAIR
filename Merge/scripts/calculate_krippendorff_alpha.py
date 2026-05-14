#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_krippendorff_alpha(data1, data2, labels=None):
    """
    Calculate Krippendorff's Alpha for two sets of categorical data.
    
    Parameters:
    data1, data2: Lists of labels for each rater
    labels: List of all possible label values (if None, will be inferred)
    
    Returns:
    alpha: Krippendorff's Alpha value
    """
    
    # Combine all data to get unique labels if not provided
    if labels is None:
        all_labels = set(data1 + data2)
        labels = sorted(list(all_labels))
    
    # Create label mapping for numerical processing
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Convert labels to indices
    data1_idx = [label_to_idx[label] for label in data1]
    data2_idx = [label_to_idx[label] for label in data2]
    
    # Count agreements and disagreements
    n = len(data1_idx)
    if n == 0:
        return 0.0
    
    # Count agreements
    agreements = sum(1 for i in range(n) if data1_idx[i] == data2_idx[i])
    
    # Calculate observed agreement
    observed_agreement = agreements / n
    
    # Calculate expected agreement by chance
    # Count frequency of each label for each rater
    freq1 = defaultdict(int)
    freq2 = defaultdict(int)
    
    for label_idx in data1_idx:
        freq1[label_idx] += 1
    for label_idx in data2_idx:
        freq2[label_idx] += 1
    
    # Calculate expected agreement
    expected_agreement = 0
    for label_idx in range(len(labels)):
        p1 = freq1[label_idx] / n
        p2 = freq2[label_idx] / n
        expected_agreement += p1 * p2
    
    # Calculate Krippendorff's Alpha
    if expected_agreement == 1:
        alpha = 1.0
    else:
        alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    
    return alpha

def calculate_krippendorff_alpha_for_csvs(csv1_path, csv2_path):
    """
    Calculate Krippendorff's Alpha for label columns in two CSV files.
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
    
    print(f"Common label columns: {len(common_cols)}")
    
    if not common_cols:
        print("No common label columns found!")
        return None
    
    # Ensure both dataframes have the same number of rows
    min_rows = min(len(df1), len(df2))
    df1 = df1.head(min_rows)
    df2 = df2.head(min_rows)
    
    # Calculate Krippendorff's Alpha for each label column
    results = []
    
    for col in common_cols:
        # Get non-null values for this column
        valid_mask = df1[col].notna() & df2[col].notna()
        
        if valid_mask.sum() > 0:
            data1 = df1.loc[valid_mask, col].astype(str).tolist()
            data2 = df2.loc[valid_mask, col].astype(str).tolist()
            
            # Calculate Krippendorff's Alpha
            alpha = calculate_krippendorff_alpha(data1, data2)
            
            # Calculate simple agreement percentage for comparison
            agreements = sum(1 for i in range(len(data1)) if data1[i] == data2[i])
            agreement_pct = (agreements / len(data1)) * 100 if len(data1) > 0 else 0
            
            results.append({
                'column': col,
                'krippendorff_alpha': alpha,
                'agreement_percentage': agreement_pct,
                'valid_pairs': len(data1),
                'agreements': agreements
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate overall statistics
    if len(results_df) > 0:
        print("\n=== KRIPPENDORFF'S ALPHA RESULTS ===")
        print(f"Number of label columns analyzed: {len(results_df)}")
        print(f"Average Krippendorff's Alpha: {results_df['krippendorff_alpha'].mean():.4f}")
        print(f"Average agreement percentage: {results_df['agreement_percentage'].mean():.2f}%")
        print(f"Min Alpha: {results_df['krippendorff_alpha'].min():.4f}")
        print(f"Max Alpha: {results_df['krippendorff_alpha'].max():.4f}")
        
        # Interpret Alpha values
        print("\n=== INTERPRETATION ===")
        avg_alpha = results_df['krippendorff_alpha'].mean()
        if avg_alpha >= 0.8:
            print("Excellent agreement (α ≥ 0.8)")
        elif avg_alpha >= 0.67:
            print("Good agreement (0.67 ≤ α < 0.8)")
        elif avg_alpha >= 0.5:
            print("Moderate agreement (0.5 ≤ α < 0.67)")
        else:
            print("Poor agreement (α < 0.5)")
    
    return results_df

def match_and_calculate_krippendorff_alpha(majority_file, csv1_path, csv2_path):
    """
    Match rows based on data_source column and calculate Krippendorff's Alpha.
    """
    
    # Read the files
    print(f"Reading {majority_file}...")
    majority_df = pd.read_csv(majority_file)
    print(f"Reading {csv1_path}...")
    df1 = pd.read_csv(csv1_path)
    print(f"Reading {csv2_path}...")
    df2 = pd.read_csv(csv2_path)
    
    print(f"Majority file shape: {majority_df.shape}")
    print(f"CSV1 shape: {df1.shape}")
    print(f"CSV2 shape: {df2.shape}")
    
    # Get data_source columns from majority file
    data_source_cols = [col for col in majority_df.columns if 'data_source' in col]
    print(f"Data source columns found: {data_source_cols}")
    
    # Get label columns (label_1 to label_21)
    label_cols = [f'label_{i}' for i in range(1, 22)]
    
    # Check which label columns exist in both dataframes
    available_cols1 = [col for col in label_cols if col in df1.columns]
    available_cols2 = [col for col in label_cols if col in df2.columns]
    common_cols = list(set(available_cols1) & set(available_cols2))
    
    print(f"Common label columns: {len(common_cols)}")
    
    if not common_cols:
        print("No common label columns found!")
        return None, None
    
    # Create a mapping from row index to data_source
    # We'll use the first data_source column for matching
    if data_source_cols:
        primary_data_source_col = data_source_cols[0]
        print(f"Using {primary_data_source_col} for matching")
        
        # Create mapping: row_index -> data_source
        row_to_data_source = {}
        for idx, row in majority_df.iterrows():
            if pd.notna(row[primary_data_source_col]):
                row_to_data_source[idx] = row[primary_data_source_col]
        
        print(f"Created mapping for {len(row_to_data_source)} rows")
        
        # Match rows based on data_source
        matched_results = []
        
        for majority_idx, data_source in row_to_data_source.items():
            # Get corresponding rows from the two CSV files
            # Assuming the row indices correspond (first row of CSV1 matches first row of majority file, etc.)
            if majority_idx < len(df1) and majority_idx < len(df2):
                row1 = df1.iloc[majority_idx]
                row2 = df2.iloc[majority_idx]
                
                # Calculate Krippendorff's Alpha for this matched pair
                for col in common_cols:
                    # Get non-null values for this column
                    if pd.notna(row1[col]) and pd.notna(row2[col]):
                        data1 = [str(row1[col])]
                        data2 = [str(row2[col])]
                        
                        # Calculate Krippendorff's Alpha
                        alpha = calculate_krippendorff_alpha(data1, data2)
                        
                        # Calculate simple agreement
                        agreement = 1.0 if data1[0] == data2[0] else 0.0
                        
                        matched_results.append({
                            'majority_row_index': majority_idx,
                            'data_source': data_source,
                            'column': col,
                            'krippendorff_alpha': alpha,
                            'agreement': agreement,
                            'value1': data1[0],
                            'value2': data2[0]
                        })
        
        # Create results dataframe
        results_df = pd.DataFrame(matched_results)
        
        if len(results_df) > 0:
            # Group by column and calculate average alpha
            column_summary = results_df.groupby('column').agg({
                'krippendorff_alpha': ['mean', 'std', 'count'],
                'agreement': 'mean'
            }).round(4)
            
            column_summary.columns = ['avg_alpha', 'std_alpha', 'count', 'avg_agreement']
            column_summary = column_summary.reset_index()
            
            print("\n=== KRIPPENDORFF'S ALPHA RESULTS (MATCHED ROWS) ===")
            print(f"Total matched pairs: {len(results_df)}")
            print(f"Number of label columns analyzed: {len(column_summary)}")
            print(f"Average Krippendorff's Alpha: {column_summary['avg_alpha'].mean():.4f}")
            print(f"Average agreement percentage: {column_summary['avg_agreement'].mean() * 100:.2f}%")
            print(f"Min Alpha: {column_summary['avg_alpha'].min():.4f}")
            print(f"Max Alpha: {column_summary['avg_alpha'].max():.4f}")
            
            # Interpret Alpha values
            print("\n=== INTERPRETATION ===")
            avg_alpha = column_summary['avg_alpha'].mean()
            if avg_alpha >= 0.8:
                print("Excellent agreement (α ≥ 0.8)")
            elif avg_alpha >= 0.67:
                print("Good agreement (0.67 ≤ α < 0.8)")
            elif avg_alpha >= 0.5:
                print("Moderate agreement (0.5 ≤ α < 0.67)")
            else:
                print("Poor agreement (α < 0.5)")
            
            return results_df, column_summary
        else:
            print("No matched results found!")
            return None, None
    else:
        print("No data_source columns found in majority file!")
        return None, None

def main():
    # File paths
    csv1_path = "FIRST_GPT4o_Remove_Irrelevant.csv"
    csv2_path = "SECOND_GPT4o_Remove_Irrelevant.csv"
    majority_file = "MAJORITY_Vote_GPT4o_Self_Reported_Relevancy_Labels.csv"
    
    print("=== STANDARD KRIPPENDORFF'S ALPHA ANALYSIS ===")
    # Calculate standard Krippendorff's Alpha
    results = calculate_krippendorff_alpha_for_csvs(csv1_path, csv2_path)
    
    if results is not None:
        # Save standard results
        output_file = "krippendorff_alpha_results.csv"
        results.to_csv(output_file, index=False)
        print(f"\nStandard results saved to {output_file}")
        
        # Display first few rows
        print("\n=== FIRST 10 ROWS (STANDARD) ===")
        print(results[['column', 'krippendorff_alpha', 'agreement_percentage', 'valid_pairs']].head(10))
    
    print("\n" + "="*60)
    print("=== MATCHED ROWS KRIPPENDORFF'S ALPHA ANALYSIS ===")
    # Calculate Krippendorff's Alpha for matched rows
    detailed_results, summary_results = match_and_calculate_krippendorff_alpha(majority_file, csv1_path, csv2_path)
    
    if detailed_results is not None:
        # Save matched results
        detailed_output_file = "krippendorff_alpha_matched_detailed.csv"
        summary_output_file = "krippendorff_alpha_matched_summary.csv"
        
        detailed_results.to_csv(detailed_output_file, index=False)
        summary_results.to_csv(summary_output_file, index=False)
        
        print(f"\nMatched detailed results saved to {detailed_output_file}")
        print(f"Matched summary results saved to {summary_output_file}")
        
        # Display summary results
        print("\n=== SUMMARY BY COLUMN (MATCHED) ===")
        print(summary_results[['column', 'avg_alpha', 'avg_agreement', 'count']].to_string(index=False))

if __name__ == "__main__":
    main() 