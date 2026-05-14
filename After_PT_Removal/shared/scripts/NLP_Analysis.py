# To display columns of a DataFrame in a Jupyter notebook
from pathlib import Path

import pandas as pd
import numpy as np
from collections import Counter
import re

_SHARED = Path(__file__).resolve().parent.parent
_DATA = _SHARED / "data"

# Load data
df2k = pd.read_csv(_DATA / "Centaur_Lab_First_Round_COMPLETE_RAW.csv")
print("Columns in df2k:")
print(df2k.columns.tolist())

# Print the first few rows
print("\nFirst few rows:")
print(df2k.head())

# --------- Analysis of step1_excerpts column ---------

# First, check if the column exists
if 'step1_excerpts' in df2k.columns:
    print("\n\n--- Analysis of step1_excerpts column ---")
    
    # Convert to string to ensure text analysis works
    df2k['step1_excerpts'] = df2k['step1_excerpts'].fillna("").astype(str)
    
    # Function to count sentences in a text
    def count_sentences(text):
        # Handle empty strings
        if not text or text.isspace():
            return 0
        # Split by common sentence terminators (., !, ?)
        sentences = re.split(r'[.!?]+', text)
        # Remove empty sentences (caused by multiple terminators or terminators at the end)
        sentences = [s for s in sentences if s and not s.isspace()]
        return len(sentences)
    
    # Function to count words in a text
    def count_words(text):
        # Handle empty strings
        if not text or text.isspace():
            return 0
        # Split text into words
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)
    
    # Function to calculate words per sentence
    def words_per_sentence(text):
        sentences = count_sentences(text)
        words = count_words(text)
        if sentences > 0:
            return words / sentences
        return 0
    
    # Apply the functions to the column
    df2k['sentence_count'] = df2k['step1_excerpts'].apply(count_sentences)
    df2k['word_count'] = df2k['step1_excerpts'].apply(count_words)
    df2k['words_per_sentence'] = df2k['step1_excerpts'].apply(words_per_sentence)
    
    # Overall statistics
    print("\n--- Overall Statistics for step1_excerpts ---")
    print(f"Average sentence count: {df2k['sentence_count'].mean():.2f}")
    print(f"Std deviation sentence count: {df2k['sentence_count'].std():.2f}")
    print(f"Average word count: {df2k['word_count'].mean():.2f}")
    print(f"Std deviation word count: {df2k['word_count'].std():.2f}")
    print(f"Average words per sentence: {df2k['words_per_sentence'].mean():.2f}")
    print(f"Std deviation words per sentence: {df2k['words_per_sentence'].std():.2f}")
    
    # Statistics by data_source_corr if it exists
    if 'data_source_corr' in df2k.columns:
        print("\n--- Statistics by data_source_corr ---")
        
        # Calculate means grouped by data_source_corr
        source_means = df2k.groupby('data_source_corr').agg({
            'sentence_count': 'mean',
            'word_count': 'mean',
            'words_per_sentence': 'mean'
        }).round(2)
        
        # Calculate standard deviations grouped by data_source_corr
        source_stds = df2k.groupby('data_source_corr').agg({
            'sentence_count': 'std',
            'word_count': 'std',
            'words_per_sentence': 'std'
        }).round(2)
        
        # Add count of records per source
        record_counts = df2k.groupby('data_source_corr').size()
        
        # Combine into a single DataFrame with multi-level columns
        source_stats = pd.DataFrame({
            ('Sentence Count', 'Mean'): source_means['sentence_count'],
            ('Sentence Count', 'Std'): source_stds['sentence_count'],
            ('Word Count', 'Mean'): source_means['word_count'],
            ('Word Count', 'Std'): source_stds['word_count'],
            ('Words per Sentence', 'Mean'): source_means['words_per_sentence'],
            ('Words per Sentence', 'Std'): source_stds['words_per_sentence'],
            ('Records', 'Count'): record_counts
        })
        
        print(source_stats)
        
        # Print in a more readable format if preferred
        print("\n--- Detailed Statistics by Data Source ---")
        for source in source_means.index:
            print(f"\nSource: {source} (n={record_counts[source]})")
            print(f"  Sentence Count: {source_means.loc[source, 'sentence_count']:.2f} ± {source_stds.loc[source, 'sentence_count']:.2f}")
            print(f"  Word Count: {source_means.loc[source, 'word_count']:.2f} ± {source_stds.loc[source, 'word_count']:.2f}")
            print(f"  Words per Sentence: {source_means.loc[source, 'words_per_sentence']:.2f} ± {source_stds.loc[source, 'words_per_sentence']:.2f}")
    else:
        print("Note: data_source_corr column not found. Cannot group by data source.")
else:
    print("Column 'step1_excerpts' not found in the dataset. Please check the column name.")

# Analysis of Filtered_Sentences and New_Sentences columns
if 'Filtered_Sentences' in df2k.columns and 'New_Sentences' in df2k.columns:
    print("\n\n--- Analysis of Filtered_Sentences and New_Sentences ---")
    
    # These columns appear to contain lists - convert from string representation if needed
    def process_list_column(column):
        if isinstance(column.iloc[0], str) and column.iloc[0].startswith('['):
            # Convert string representation of list to actual list
            return column.apply(eval)
        return column
    
    # Process the columns
    try:
        filtered_sentences = process_list_column(df2k['Filtered_Sentences'])
        new_sentences = process_list_column(df2k['New_Sentences'])
        
        # Function to count words in a list of sentences
        def count_words_in_list(sentences_list):
            if not sentences_list or len(sentences_list) == 0:
                return 0
            total_words = 0
            for sentence in sentences_list:
                total_words += len(re.findall(r'\b\w+\b', str(sentence).lower()))
            return total_words
        
        # Function to get words per sentence for a list of sentences
        def words_per_sentence_in_list(sentences_list):
            if not sentences_list or len(sentences_list) == 0:
                return 0
            total_sentences = len(sentences_list)
            total_words = count_words_in_list(sentences_list)
            return total_words / total_sentences if total_sentences > 0 else 0
        
        # Apply the functions to the columns
        df2k['filtered_sentence_count'] = filtered_sentences.apply(len)
        df2k['filtered_word_count'] = filtered_sentences.apply(count_words_in_list)
        df2k['filtered_words_per_sentence'] = filtered_sentences.apply(words_per_sentence_in_list)
        
        df2k['new_sentence_count'] = new_sentences.apply(len)
        df2k['new_word_count'] = new_sentences.apply(count_words_in_list)
        df2k['new_words_per_sentence'] = new_sentences.apply(words_per_sentence_in_list)
        
        # Overall statistics for Filtered_Sentences
        print("\n--- Overall Statistics for Filtered_Sentences ---")
        print(f"Average sentence count: {df2k['filtered_sentence_count'].mean():.2f}")
        print(f"Std deviation sentence count: {df2k['filtered_sentence_count'].std():.2f}")
        print(f"Average word count: {df2k['filtered_word_count'].mean():.2f}")
        print(f"Std deviation word count: {df2k['filtered_word_count'].std():.2f}")
        print(f"Average words per sentence: {df2k['filtered_words_per_sentence'].mean():.2f}")
        print(f"Std deviation words per sentence: {df2k['filtered_words_per_sentence'].std():.2f}")
        
        # Overall statistics for New_Sentences
        print("\n--- Overall Statistics for New_Sentences ---")
        print(f"Average sentence count: {df2k['new_sentence_count'].mean():.2f}")
        print(f"Std deviation sentence count: {df2k['new_sentence_count'].std():.2f}")
        print(f"Average word count: {df2k['new_word_count'].mean():.2f}")
        print(f"Std deviation word count: {df2k['new_word_count'].std():.2f}")
        print(f"Average words per sentence: {df2k['new_words_per_sentence'].mean():.2f}")
        print(f"Std deviation words per sentence: {df2k['new_words_per_sentence'].std():.2f}")
        
        # Statistics by data_source_corr
        if 'data_source_corr' in df2k.columns:
            # Filtered_Sentences statistics by data_source_corr
            print("\n--- Filtered_Sentences Statistics by data_source_corr ---")
            filtered_means = df2k.groupby('data_source_corr').agg({
                'filtered_sentence_count': 'mean',
                'filtered_word_count': 'mean',
                'filtered_words_per_sentence': 'mean'
            }).round(2)
            
            filtered_stds = df2k.groupby('data_source_corr').agg({
                'filtered_sentence_count': 'std',
                'filtered_word_count': 'std',
                'filtered_words_per_sentence': 'std'
            }).round(2)
            
            filtered_stats = pd.DataFrame({
                'Avg Sentences': filtered_means['filtered_sentence_count'],
                'Std Sentences': filtered_stds['filtered_sentence_count'],
                'Avg Words': filtered_means['filtered_word_count'],
                'Std Words': filtered_stds['filtered_word_count'],
                'Avg Words per Sentence': filtered_means['filtered_words_per_sentence'],
                'Std Words per Sentence': filtered_stds['filtered_words_per_sentence']
            })
            
            print(filtered_stats)
            
            # New_Sentences statistics by data_source_corr
            print("\n--- New_Sentences Statistics by data_source_corr ---")
            new_means = df2k.groupby('data_source_corr').agg({
                'new_sentence_count': 'mean',
                'new_word_count': 'mean',
                'new_words_per_sentence': 'mean'
            }).round(2)
            
            new_stds = df2k.groupby('data_source_corr').agg({
                'new_sentence_count': 'std',
                'new_word_count': 'std',
                'new_words_per_sentence': 'std'
            }).round(2)
            
            new_stats = pd.DataFrame({
                'Avg Sentences': new_means['new_sentence_count'],
                'Std Sentences': new_stds['new_sentence_count'],
                'Avg Words': new_means['new_word_count'],
                'Std Words': new_stds['new_word_count'],
                'Avg Words per Sentence': new_means['new_words_per_sentence'],
                'Std Words per Sentence': new_stds['new_words_per_sentence']
            })
            
            print(new_stats)

        # Function to estimate syllable count
        def count_syllables(word):
            """
            Count the number of syllables in a word.
            Basic implementation - counts vowel groups.
            """
            word = word.lower()
            # Remove non-alphabetic characters
            word = re.sub(r'[^a-z]', '', word)
            
            if not word:
                return 0
            
            # Count vowel groups
            count = 0
            vowels = "aeiouy"
            prev_is_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
            
            # Check for silent 'e' at the end
            if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
                count -= 1
            
            # Make sure every word has at least one syllable
            return max(1, count)

        # Function to calculate Flesch-Kincaid Reading Ease score
        def flesch_kincaid_ease(text_list):
            """
            Calculate Flesch-Kincaid Reading Ease score for a list of sentences.
            Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
            """
            if not text_list or len(text_list) == 0:
                return 0
            
            # Count sentences, words and syllables
            total_sentences = len(text_list)
            
            total_words = 0
            total_syllables = 0
            
            for sentence in text_list:
                words = re.findall(r'\b\w+\b', str(sentence).lower())
                total_words += len(words)
                
                for word in words:
                    total_syllables += count_syllables(word)
            
            # Handle edge cases
            if total_sentences == 0 or total_words == 0:
                return 0
            
            # Calculate score
            score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
            
            # Scores are typically between 0-100, but can go outside this range
            return score

        # Calculate Flesch-Kincaid scores
        df2k['filtered_flesch_kincaid'] = filtered_sentences.apply(flesch_kincaid_ease)
        df2k['new_flesch_kincaid'] = new_sentences.apply(flesch_kincaid_ease)

        # Print overall statistics
        print("\n--- Flesch-Kincaid Reading Ease Scores ---")
        print("Filtered Sentences:")
        print(f"  Mean: {df2k['filtered_flesch_kincaid'].mean():.2f}")
        print(f"  Std: {df2k['filtered_flesch_kincaid'].std():.2f}")
        print(f"  Min: {df2k['filtered_flesch_kincaid'].min():.2f}")
        print(f"  Max: {df2k['filtered_flesch_kincaid'].max():.2f}")

        print("\nNew Sentences:")
        print(f"  Mean: {df2k['new_flesch_kincaid'].mean():.2f}")
        print(f"  Std: {df2k['new_flesch_kincaid'].std():.2f}")
        print(f"  Min: {df2k['new_flesch_kincaid'].min():.2f}")
        print(f"  Max: {df2k['new_flesch_kincaid'].max():.2f}")

        # Calculate by data_source_corr
        if 'data_source_corr' in df2k.columns:
            print("\n--- Flesch-Kincaid Scores by Data Source ---")
            
            fk_means = df2k.groupby('data_source_corr').agg({
                'filtered_flesch_kincaid': 'mean',
                'new_flesch_kincaid': 'mean'
            }).round(2)
            
            fk_stds = df2k.groupby('data_source_corr').agg({
                'filtered_flesch_kincaid': 'std',
                'new_flesch_kincaid': 'std'
            }).round(2)
            
            fk_stats = pd.DataFrame({
                'Filtered Mean': fk_means['filtered_flesch_kincaid'],
                'Filtered Std': fk_stds['filtered_flesch_kincaid'],
                'New Mean': fk_means['new_flesch_kincaid'],
                'New Std': fk_stds['new_flesch_kincaid']
            })
            
            print(fk_stats)
            
            # Print interpretation guide
            print("\nFlesch-Kincaid Reading Ease Score Interpretation:")
            print("90-100: Very Easy (5th grade)")
            print("80-89: Easy (6th grade)")
            print("70-79: Fairly Easy (7th grade)")
            print("60-69: Standard (8th-9th grade)")
            print("50-59: Fairly Difficult (10th-12th grade)")
            print("30-49: Difficult (College)")
            print("0-29: Very Difficult (College Graduate)")
    except Exception as e:
        print(f"Error processing list columns: {e}")
else:
    print("\nColumns 'Filtered_Sentences' or 'New_Sentences' not found in the dataset.")

# --------- Basic Analysis on data_source_corr ---------

if 'data_source_corr' in df2k.columns:
    # 1. Basic counts and distribution
    print("\n\n--- Distribution of data_source_corr values ---")
    value_counts = df2k['data_source_corr'].value_counts()
    print(value_counts)

    # 2. Check for missing values
    print(f"\nMissing values in data_source_corr: {df2k['data_source_corr'].isna().sum()}")

    # Convert to string to ensure text analysis works
    df2k['data_source_corr'] = df2k['data_source_corr'].astype(str)

    # 3. Text length analysis
    df2k['text_length'] = df2k['data_source_corr'].apply(len)
    print(f"\n--- Text Length Statistics for data_source_corr ---")
    print(f"Min length: {df2k['text_length'].min()}")
    print(f"Max length: {df2k['text_length'].max()}")
    print(f"Mean length: {df2k['text_length'].mean():.2f}")
    print(f"Median length: {df2k['text_length'].median()}")
    print(f"Standard deviation: {df2k['text_length'].std():.2f}")

    # Print average text length by data source category
    print("\n--- Average Text Length by Category ---")
    avg_lengths = df2k.groupby('data_source_corr')['text_length'].mean().sort_values(ascending=False)
    print(avg_lengths)

    # 4. Simple word frequency analysis
    def simple_preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Split into words
        words = text.split()
        # Remove short words and common stopwords
        stopwords = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'at', 'from', 'by', 'an', 'it', 'as']
        words = [word for word in words if word not in stopwords and len(word) > 2]
        return words

    # Combine all text for word frequency analysis
    all_words = []
    for text in df2k['data_source_corr']:
        all_words.extend(simple_preprocess_text(text))

    # Get word frequencies
    word_freq = Counter(all_words)
    print("\n--- Most Common Words in data_source_corr ---")
    print(word_freq.most_common(20))

    # 5. Category-specific analysis (if data_source_corr contains categories)
    if len(value_counts) < 20:  # If it seems like a categorical column
        print("\n--- Top Words by Category ---")
        
        for category in value_counts.index[:5]:  # Top 5 categories
            category_texts = df2k[df2k['data_source_corr'] == category]['data_source_corr']
            
            # Skip if no texts in this category
            if len(category_texts) == 0:
                continue
                
            category_words = []
            for text in category_texts:
                category_words.extend(simple_preprocess_text(text))
                
            if category_words:
                print(f"\nTop words in category '{category}':")
                print(Counter(category_words).most_common(10))

    # 6. Data source count and percentage
    print("\n--- Data Source Counts and Percentages ---")
    source_counts = df2k['data_source_corr'].value_counts()
    source_percent = df2k['data_source_corr'].value_counts(normalize=True) * 100
    source_stats = pd.DataFrame({
        'Count': source_counts,
        'Percentage': source_percent
    })
    print(source_stats)
else:
    print("\nColumn 'data_source_corr' not found in the dataset. Skipping data_source_corr analysis.")

print("\nAnalysis complete!")

