#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_krippendorff_alpha(data1, data2, labels=None):
    """
    Calculate Krippendorff's Alpha for two sets of categorical data.
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

def calculate_qwen14_krippendorff_alpha(csv1_path, csv2_path):
    """
    Calculate Krippendorff's Alpha between two Qwen14 annotated CSV files.
    """
    
    # Read the CSV files with error handling
    print(f"Reading {csv1_path}...")
    try:
        df1 = pd.read_csv(csv1_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {csv1_path}: {e}")
        return None
        
    print(f"Reading {csv2_path}...")
    try:
        df2 = pd.read_csv(csv2_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {csv2_path}: {e}")
        return None
    
    print(f"Qwen14 CSV1 shape: {df1.shape}")
    print(f"Qwen14 CSV2 shape: {df2.shape}")
    
    # Get label columns (label_1 to label_21)
    label_cols = [f'label_{i}' for i in range(1, 22)]
    
    # Check which label columns exist in both dataframes
    available_cols1 = [col for col in label_cols if col in df1.columns]
    available_cols2 = [col for col in label_cols if col in df2.columns]
    common_cols = list(set(available_cols1) & set(available_cols2))
    
    print(f"Common label columns in Qwen14 files: {len(common_cols)}")
    
    if not common_cols:
        print("No common label columns found in Qwen14 files!")
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
        print("\n=== QWEN14 KRIPPENDORFF'S ALPHA RESULTS ===")
        print(f"Number of label columns analyzed: {len(results_df)}")
        print(f"Average Krippendorff's Alpha: {results_df['krippendorff_alpha'].mean():.4f}")
        print(f"Average agreement percentage: {results_df['agreement_percentage'].mean():.2f}%")
        print(f"Min Alpha: {results_df['krippendorff_alpha'].min():.4f}")
        print(f"Max Alpha: {results_df['krippendorff_alpha'].max():.4f}")
        
        # Interpret Alpha values
        print("\n=== QWEN14 INTERPRETATION ===")
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

def calculate_general_krippendorff_alpha(df):
    """
    Calculate the general Krippendorff's Alpha across all data sources and label columns.
    This treats all individual label comparisons as one large dataset.
    """
    
    # Extract all label pairs where both values are not null
    all_pairs = []
    
    for _, row in df.iterrows():
        if pd.notna(row['value1']) and pd.notna(row['value2']):
            all_pairs.append((str(row['value1']), str(row['value2'])))
    
    if not all_pairs:
        return 0.0, 0.0
    
    # Separate into two lists
    data1 = [pair[0] for pair in all_pairs]
    data2 = [pair[1] for pair in all_pairs]
    
    # Get all unique labels
    all_labels = set(data1 + data2)
    labels = sorted(list(all_labels))
    
    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Convert to indices
    data1_idx = [label_to_idx[label] for label in data1]
    data2_idx = [label_to_idx[label] for label in data2]
    
    # Count agreements
    n = len(data1_idx)
    agreements = sum(1 for i in range(n) if data1_idx[i] == data2_idx[i])
    
    # Calculate observed agreement
    observed_agreement = agreements / n
    
    # Calculate expected agreement by chance
    freq1 = defaultdict(int)
    freq2 = defaultdict(int)
    
    for label_idx in data1_idx:
        freq1[label_idx] += 1
    for label_idx in data2_idx:
        freq2[label_idx] += 1
    
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
    
    return alpha, observed_agreement

def analyze_data_source_alpha(detailed_file):
    """
    Analyze average Krippendorff's Alpha for each data source.
    """
    
    # Read the detailed results
    print(f"Reading {detailed_file}...")
    df = pd.read_csv(detailed_file)
    
    print(f"Total records: {len(df)}")
    print(f"Unique data sources: {df['data_source'].nunique()}")
    
    # Calculate general Krippendorff's Alpha
    print("\n=== CALCULATING GENERAL KRIPPENDORFF'S ALPHA ===")
    general_alpha, general_agreement = calculate_general_krippendorff_alpha(df)
    print(f"General Krippendorff's Alpha: {general_alpha:.4f}")
    print(f"General Agreement Rate: {general_agreement:.4f}")
    
    # Interpret general alpha
    print("\n=== GENERAL ALPHA INTERPRETATION ===")
    if general_alpha >= 0.8:
        print("Excellent agreement (α ≥ 0.8)")
    elif general_alpha >= 0.67:
        print("Good agreement (0.67 ≤ α < 0.8)")
    elif general_alpha >= 0.5:
        print("Moderate agreement (0.5 ≤ α < 0.67)")
    else:
        print("Poor agreement (α < 0.5)")
    
    # Group by data_source and calculate statistics
    data_source_stats = df.groupby('data_source').agg({
        'krippendorff_alpha': ['mean', 'std', 'count', 'min', 'max'],
        'agreement': 'mean'
    }).round(4)
    
    # Flatten column names
    data_source_stats.columns = ['avg_alpha', 'std_alpha', 'count', 'min_alpha', 'max_alpha', 'avg_agreement']
    data_source_stats = data_source_stats.reset_index()
    
    # Sort by average alpha (descending)
    data_source_stats = data_source_stats.sort_values('avg_alpha', ascending=False)
    
    # Calculate overall statistics
    print("\n=== OVERALL STATISTICS (BY DATA SOURCE AVERAGE) ===")
    print(f"Overall average Krippendorff's Alpha: {df['krippendorff_alpha'].mean():.4f}")
    print(f"Overall average agreement: {df['agreement'].mean():.4f}")
    print(f"Overall standard deviation: {df['krippendorff_alpha'].std():.4f}")
    
    # Interpret overall alpha
    overall_alpha = df['krippendorff_alpha'].mean()
    print("\n=== OVERALL INTERPRETATION (BY DATA SOURCE AVERAGE) ===")
    if overall_alpha >= 0.8:
        print("Excellent agreement (α ≥ 0.8)")
    elif overall_alpha >= 0.67:
        print("Good agreement (0.67 ≤ α < 0.8)")
    elif overall_alpha >= 0.5:
        print("Moderate agreement (0.5 ≤ α < 0.67)")
    else:
        print("Poor agreement (α < 0.5)")
    
    # Display results by data source
    print("\n=== RESULTS BY DATA SOURCE ===")
    print(f"{'Data Source':<30} {'Avg Alpha':<10} {'Std Dev':<10} {'Count':<8} {'Min':<8} {'Max':<8} {'Avg Agreement':<15}")
    print("-" * 100)
    
    for _, row in data_source_stats.iterrows():
        data_source = str(row['data_source'])[:29]  # Truncate if too long
        print(f"{data_source:<30} {row['avg_alpha']:<10.4f} {row['std_alpha']:<10.4f} {row['count']:<8} {row['min_alpha']:<8.4f} {row['max_alpha']:<8.4f} {row['avg_agreement']:<15.4f}")
    
    # Find best and worst performing data sources
    best_source = data_source_stats.iloc[0]
    worst_source = data_source_stats.iloc[-1]
    
    print(f"\n=== BEST PERFORMING DATA SOURCE ===")
    print(f"Data Source: {best_source['data_source']}")
    print(f"Average Alpha: {best_source['avg_alpha']:.4f}")
    print(f"Count: {best_source['count']}")
    print(f"Range: {best_source['min_alpha']:.4f} - {best_source['max_alpha']:.4f}")
    
    print(f"\n=== WORST PERFORMING DATA SOURCE ===")
    print(f"Data Source: {worst_source['data_source']}")
    print(f"Average Alpha: {worst_source['avg_alpha']:.4f}")
    print(f"Count: {worst_source['count']}")
    print(f"Range: {worst_source['min_alpha']:.4f} - {worst_source['max_alpha']:.4f}")
    
    # Count data sources by performance level
    excellent = len(data_source_stats[data_source_stats['avg_alpha'] >= 0.8])
    good = len(data_source_stats[(data_source_stats['avg_alpha'] >= 0.67) & (data_source_stats['avg_alpha'] < 0.8)])
    moderate = len(data_source_stats[(data_source_stats['avg_alpha'] >= 0.5) & (data_source_stats['avg_alpha'] < 0.67)])
    poor = len(data_source_stats[data_source_stats['avg_alpha'] < 0.5])
    
    print(f"\n=== PERFORMANCE DISTRIBUTION ===")
    print(f"Excellent (α ≥ 0.8): {excellent} data sources")
    print(f"Good (0.67 ≤ α < 0.8): {good} data sources")
    print(f"Moderate (0.5 ≤ α < 0.67): {moderate} data sources")
    print(f"Poor (α < 0.5): {poor} data sources")
    
    return data_source_stats, general_alpha, general_agreement

def main():
    # File paths
    detailed_file = "krippendorff_alpha_matched_detailed.csv"
    qwen14_csv1 = "First_Qwen14_annotated_2k_relevancy.csv"
    qwen14_csv2 = "SECOND_annotated_2k_relevancy.csv"
    
    print("="*80)
    print("=== GPT4O KRIPPENDORFF'S ALPHA ANALYSIS ===")
    print("="*80)
    
    # Analyze GPT4o data source alpha values
    results, general_alpha, general_agreement = analyze_data_source_alpha(detailed_file)
    
    # Save GPT4o results
    output_file = "data_source_alpha_analysis.csv"
    results.to_csv(output_file, index=False)
    print(f"\nGPT4o results saved to {output_file}")
    
    # Save general alpha results
    general_results = pd.DataFrame({
        'metric': ['General_Krippendorff_Alpha', 'General_Agreement_Rate'],
        'value': [general_alpha, general_agreement]
    })
    general_output_file = "general_alpha_results.csv"
    general_results.to_csv(general_output_file, index=False)
    print(f"GPT4o general alpha results saved to {general_output_file}")
    
    # Display GPT4o top 10 and bottom 10 data sources
    print(f"\n=== GPT4O TOP 10 DATA SOURCES ===")
    print(results[['data_source', 'avg_alpha', 'count']].head(10).to_string(index=False))
    
    print(f"\n=== GPT4O BOTTOM 10 DATA SOURCES ===")
    print(results[['data_source', 'avg_alpha', 'count']].tail(10).to_string(index=False))
    
    # Summary comparison for GPT4o
    print(f"\n=== GPT4O SUMMARY COMPARISON ===")
    print(f"General Krippendorff's Alpha: {general_alpha:.4f}")
    print(f"Average by Data Source: {results['avg_alpha'].mean():.4f}")
    print(f"Difference: {general_alpha - results['avg_alpha'].mean():.4f}")
    
    print("\n" + "="*80)
    print("=== QWEN14 KRIPPENDORFF'S ALPHA ANALYSIS ===")
    print("="*80)
    
    # Calculate Qwen14 Krippendorff's Alpha
    qwen14_results = calculate_qwen14_krippendorff_alpha(qwen14_csv1, qwen14_csv2)
    
    if qwen14_results is not None and len(qwen14_results) > 0:
        # Save Qwen14 results
        qwen14_output_file = "qwen14_alpha_results.csv"
        qwen14_results.to_csv(qwen14_output_file, index=False)
        print(f"\nQwen14 results saved to {qwen14_output_file}")
        
        # Display Qwen14 results
        print(f"\n=== QWEN14 DETAILED RESULTS ===")
        print(qwen14_results[['column', 'krippendorff_alpha', 'agreement_percentage', 'valid_pairs']].to_string(index=False))
        
        # Compare GPT4o vs Qwen14
        print(f"\n=== COMPARISON: GPT4O vs QWEN14 ===")
        print(f"GPT4o Average Alpha: {general_alpha:.4f}")
        print(f"Qwen14 Average Alpha: {qwen14_results['krippendorff_alpha'].mean():.4f}")
        print(f"Difference (GPT4o - Qwen14): {general_alpha - qwen14_results['krippendorff_alpha'].mean():.4f}")
        
        if general_alpha > qwen14_results['krippendorff_alpha'].mean():
            print("GPT4o shows higher agreement than Qwen14")
        elif general_alpha < qwen14_results['krippendorff_alpha'].mean():
            print("Qwen14 shows higher agreement than GPT4o")
        else:
            print("GPT4o and Qwen14 show similar agreement levels")
    else:
        print("Qwen14 analysis failed or returned no results.")
        print("This could be due to:")
        print("- Missing or corrupted CSV files")
        print("- No common label columns between the files")
        print("- All label values being null")
        print("- Parsing errors in the CSV files")

if __name__ == "__main__":
    main() 