#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
import paths

import pandas as pd

# Load the CSV file
df = pd.read_csv(paths.DATA / "May7_Data.csv")

# Load the CSV file
df2 = pd.read_csv(paths.DATA / "merged_2k_questions_standardized.csv")

# Display the first few rows after removing duplicates
# print(df_op4.columns)
print(df.head())
print(len(df))


# In[3]:


print(df2.head())


# In[4]:


# Ensure both columns are lowercase for case-insensitive matching
df['q1_lower'] = df['q1'].str.lower()
df2['answer_lower'] = df2['answer'].str.lower()

# Merge the DataFrames on 'Origin' and 'ID'
merged_df = pd.merge(df, df2, left_on='Origin', right_on='ID', how='inner')

# Check if the lowercase versions match
merged_df['Match'] = merged_df['q1_lower'] == merged_df['answer_lower']

# Drop the temporary lowercase columns
merged_df.drop(columns=['q1_lower', 'answer_lower'], inplace=True)

# Display the result
merged_df.head(30)


# In[5]:


import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# columns q2 … q23
# ------------------------------------------------------------------
target_cols = [f"q{i}" for i in range(2, 23)]

# ------------------------------------------------------------------
# 1. replace the literal phrase “not applicable” (any case) with ""
# ------------------------------------------------------------------
merged_df[target_cols] = merged_df[target_cols].replace(
    r"(?i)\bnot\s+applicable\b",   # case‑insensitive regex
    "",
    regex=True,
)

# ------------------------------------------------------------------
# 2. treat the string "NaN" the same way
# ------------------------------------------------------------------
merged_df[target_cols] = merged_df[target_cols].replace("NaN", "")

# ------------------------------------------------------------------
# 3. turn real NaNs into blank strings as well
# ------------------------------------------------------------------
merged_df[target_cols] = merged_df[target_cols].fillna("")

# merged_df now shows an empty cell whenever q2…q23 originally held
#   • NaN (missing)
#   • the literal string "NaN"
#   • any capitalization of "not applicable"
merged_df


# In[6]:


import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1.  ensure Duration is numeric
# ------------------------------------------------------------------
merged_df["Duration_num"] = pd.to_numeric(
    merged_df["Duration"], errors="coerce"
)

# ------------------------------------------------------------------
# 2.  overall statistics  (minutes and hours)
# ------------------------------------------------------------------
mean_dur = merged_df["Duration_num"].mean()
std_dur  = merged_df["Duration_num"].std()

print(f"Overall duration — mean: {mean_dur:.2f} min "
      f"({mean_dur/60:.2f} min)  •  std: {std_dur:.2f} min "
      f"({std_dur/60:.2f} min)")

# ------------------------------------------------------------------
# 3.  split by Match flag
# ------------------------------------------------------------------
for flag, label in [(True, "Match = True"), (False, "Match = False")]:
    subset       = merged_df.loc[merged_df["Match"] == flag, "Duration_num"]
    mean_flag    = subset.mean()
    std_flag     = subset.std()

    print(f"{label:<13} — mean: {mean_flag:.2f} min "
          f"({mean_flag/60:.2f} mins)  •  std: {std_flag:.2f} min "
          f"({std_flag/60:.2f} min)")


# In[7]:


# Group by 'Origin' and calculate the accuracy rate
accuracy_rate = merged_df.groupby('Origin')['Match'].mean().reset_index()
accuracy_rate.columns = ['Origin', 'Accuracy_Rate']

# Display the result without ace_tools
accuracy_rate


# In[8]:


# Calculate the general accuracy rate and standard deviation across the entire DataFrame
general_stats = merged_df['Match'].agg(['mean', 'std']).reset_index()
general_stats.columns = ['Metric', 'Value']

# Display the DataFrame
general_stats


# In[9]:


# Attempting to group by 'data_source' and calculate the accuracy rate statistics including std deviation
stats_df = merged_df.groupby('data_source')['Match'].agg(['mean', 'std']).reset_index()
stats_df.columns = ['Data Source', 'Accuracy Rate', 'Standard Deviation']

# Display the DataFrame
stats_df


# In[10]:


# incorrect labels
# create a dataframe wrong_df which if merged_df column "Match" is False
wrong_df = merged_df[merged_df['Match'] == False].copy()

# Display the first few rows to verify
wrong_df.head(7)

print(wrong_df['Origin'].nunique())
print(len(wrong_df))


# ## ONLY CORRECT LABELS

# In[11]:


# Filter rows where "Match" is True
correct_df = merged_df[merged_df['Match'] == True].copy()

# Display the first few rows to verify
correct_df.head(7)

print(correct_df['Origin'].nunique())
print(len(correct_df))


# In[12]:


import pandas as pd

# ------------------------------------------------------------------
# 1.  Specify columns
# ------------------------------------------------------------------
q_cols = [f"q{i}" for i in range(2, 22)]                     # q2 … q22
meta   = ["ID", "centaur_question", "sentence_number",
          "answer", "data_source", "Match"]

# ------------------------------------------------------------------
# 2.  Map label strings to numerical scores
# ------------------------------------------------------------------
label_to_score = {
    "high relevance": 1.0,
    "low relevance" : 0.5,
    "not relevant"  : 0.0
}

def normalise(x):
    if pd.isna(x):
        return x
    return str(x).strip().lower()

correct_df[q_cols] = (
    correct_df[q_cols]
        .map(normalise)           # trim and lower‑case
        .apply(lambda col: col.map(label_to_score))
)

# ------------------------------------------------------------------
# 3.  Aggregate by Origin, keeping one exemplar of the meta fields
# ------------------------------------------------------------------
agg_rules = {col: "mean"  for col in q_cols}                 # numeric mean
agg_rules.update({col: "first" for col in meta})             # exemplar meta

origin_average = (
    correct_df
        .groupby("Origin", as_index=False)
        .agg(agg_rules)
)

# ------------------------------------------------------------------
# 4.  Inspect the result
# ------------------------------------------------------------------
origin_average.head()
# print(len(origin_average))

# origin_average# Save to CSV
output_path = paths.TABLES / 'less_sentence_df_export.csv'
origin_average.to_csv(output_path, index=False)

# Provide the download link
output_path


# In[13]:


import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1.  Read the file
# -------------------------------------------------------------
origin_average = pd.read_csv(paths.TABLES / "less_sentence_df_export.csv")

# question columns q2 … q21
q_cols = [f"q{i}" for i in range(2, 20)]

# cast to float so comparisons work; bad strings → NaN
origin_average[q_cols] = origin_average[q_cols].apply(
    pd.to_numeric, errors="coerce"
)

# -------------------------------------------------------------
# 2.  Row‑dependent masking rule
#     • if the row contains any value > 0.5 → mask everything ≤ 0.5
#     • otherwise                           → mask everything  < 0.5
# -------------------------------------------------------------
has_high = origin_average[q_cols].gt(0.5).any(axis=1)            # True / False

mask = np.where(
    has_high.to_numpy()[:, None],                                 # per‑row flag
    origin_average[q_cols] <= 0.5,                                # rule when True
    origin_average[q_cols] < 0.5                                  # rule when False
)

origin_average[q_cols] = origin_average[q_cols].astype(object)
origin_average.loc[:, q_cols] = origin_average[q_cols].mask(mask, "REMOVED")

# -------------------------------------------------------------
# 3.  Count how many cells were removed
# -------------------------------------------------------------
origin_average["REMOVED_Sentences"] = (
    origin_average[q_cols].eq("REMOVED").sum(axis=1)
)

# quick inspection
origin_average.head()


# In[ ]:





# In[14]:


import pandas as pd
import numpy as np

rng = np.random.default_rng(seed=42)          # reproducible draws

# ------------------------------------------------------------------
# 0. cap each Origin at ≤ 3 rows
# ------------------------------------------------------------------
capped_df = (
    correct_df
      .sample(frac=1, random_state=42)              # shuffle for random 3-row draw
      .groupby("Origin", group_keys=False)
      .head(3)                                      # keep at most 3 rows per Origin
      .reset_index(drop=True)
)

# ------------------------------------------------------------------
# 1. count surviving rows per Origin
# ------------------------------------------------------------------
origin_sizes = (
    capped_df.groupby("Origin")
             .size()
             .reset_index(name="labeler_count")
)

# ------------------------------------------------------------------
# 2. bucket by labeler count (1, 2, or 3)
# ------------------------------------------------------------------
dfs_by_labelers = {
    k: capped_df.loc[
           capped_df["Origin"].isin(
               origin_sizes.loc[origin_sizes["labeler_count"] == k, "Origin"]
           )
       ].copy()
    for k in sorted(origin_sizes["labeler_count"].unique())
}

# ------------------------------------------------------------------
# 3. sanity check
# ------------------------------------------------------------------
for k, df_k in dfs_by_labelers.items():
    print(f"{k}‑labeler group → {len(df_k)} rows, "
          f"{df_k['Origin'].nunique()} unique Origins")


# In[15]:


import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1.  retain at most three rows for every Origin
#     (use .head(3) for deterministic behaviour;
#     replace with .sample(n=3, random_state=42) if you prefer random draws)
# ------------------------------------------------------------
capped_df = (
    correct_df
      .sort_values("Origin")          # keep a stable order within each group
      .groupby("Origin", group_keys=False)
      .head(3)                        # keeps the first three rows, discards the rest
)

# ------------------------------------------------------------
# 2.  recompute the row‑count distribution per Origin
# ------------------------------------------------------------
origin_sizes          = capped_df.groupby("Origin").size()
labeler_distribution  = origin_sizes.value_counts().sort_index()

print("Rows per Origin (after capping at three):")
print(labeler_distribution)


# In[16]:


# ------------------------------------------------------------
# 1. unique Origins per labeler bucket
# ------------------------------------------------------------
for k, df_k in dfs_by_labelers.items():
    unique_origins = df_k["Origin"].nunique()
    print(f"{k}-labeler group → {unique_origins} unique Origins")

# ------------------------------------------------------------
# 2. unique Origins across all buckets (optional)
# ------------------------------------------------------------
total_unique_origins = (
    pd.concat(dfs_by_labelers.values(), ignore_index=True)["Origin"]
      .nunique()
)
print(f"\nOverall → {total_unique_origins} unique Origins")


# In[ ]:





# In[17]:


# question columns (q1 … q22)
question_cols = [c for c in correct_df.columns if c.startswith("q")]

# helper: build Boolean flags
correct_df["has_high_relevance"] = (
    correct_df[question_cols] == "high relevance"
).any(axis=1)

correct_df["has_low_relevance"] = (
    correct_df[question_cols] == "low relevance"
).any(axis=1)

correct_df["has_not_relevant"] = (
    correct_df[question_cols] == "not relevant"
).any(axis=1)

# concise report
tot = len(correct_df)
n_high = correct_df["has_high_relevance"].sum()
n_low  = correct_df["has_low_relevance"].sum()
n_irr  = correct_df["has_not_relevant"].sum()

print(
    f"Rows with at least one ‘high relevance’: {n_high:5d}  ({n_high/tot:.1%})\n"
    f"Rows with at least one ‘low relevance’ : {n_low:5d}  ({n_low/tot:.1%})\n"
    f"Rows with at least one ‘not relevant’  : {n_irr:5d}  ({n_irr/tot:.1%})"
)


# In[ ]:





# In[18]:


question_cols = [c for c in correct_df.columns if c.startswith("q")]

# Boolean flag: at least one high‑relevance label
correct_df["has_high_relevance"] = (
    correct_df[question_cols] == "high relevance"
).any(axis=1)

# Subset: rows with no high relevance
no_high_df = correct_df[~correct_df["has_high_relevance"]]

print(f"{len(no_high_df)} of {len(correct_df)} rows contain *no* "
      f"'high relevance' labels.")

# Inspect the first few rows
no_high_df.head()


# In[ ]:





# In[ ]:





# ## Calculate the IRR?

# In[105]:


# one‑time installation (execute in its own cell)
import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'krippendorff'], check=False)


# In[111]:


import krippendorff
import pandas as pd
import numpy as np
from typing import Optional

QUESTION_COLS = [f"q{i}" for i in range(1, 23)]          # q1 … q22
LABEL_MAP     = {"high relevance": 2, "low relevance": 1, "not relevant": 0}

def kripp_alpha_by_question(
    df: pd.DataFrame,
    item_col: str = "Origin",
    rater_col: Optional[str] = None,
) -> pd.Series:
    """
    Compute Krippendorff's α (nominal) for every question column after
    mapping category strings to integer codes via LABEL_MAP.
    """

    # infer rater column if none supplied
    if rater_col is None:
        candidate_cols = df.columns.difference([item_col, *QUESTION_COLS])
        if candidate_cols.empty:
            raise ValueError("Cannot infer rater column; supply rater_col")
        rater_col = candidate_cols[0]

    alphas = {}
    for q in QUESTION_COLS:

        # pivot to (items × raters) and map strings → ints → float
        mat = (
            df.pivot(index=item_col, columns=rater_col, values=q)
              .map(lambda x: LABEL_MAP.get(x, np.nan))
              .to_numpy(dtype=float)
        )

        try:
            alphas[q] = krippendorff.alpha(mat, level_of_measurement="nominal")
        except ValueError:
            alphas[q] = np.nan  # not enough variance to compute alpha

    return pd.Series(alphas, name="Krippendorff α")

# ------------------------------------------------------------------
# add synthetic rater IDs (0,1) or (0,1,2) inside each Origin
# ------------------------------------------------------------------
two_labelers_df   = dfs_by_labelers[2].copy()
three_labelers_df = dfs_by_labelers[3].copy()

two_labelers_df["Rater"]   = two_labelers_df.groupby("Origin").cumcount()
three_labelers_df["Rater"] = three_labelers_df.groupby("Origin").cumcount()

# ------------------------------------------------------------------
# compute α
# ------------------------------------------------------------------
alpha_two   = kripp_alpha_by_question(two_labelers_df,   rater_col="Rater")
alpha_three = kripp_alpha_by_question(three_labelers_df, rater_col="Rater")

print("\nα — Two‑Labeler Origins")
print(alpha_two.round(3))

print("\nα — Three‑Labeler Origins")
print(alpha_three.round(3))

print("\nOverall α (two):",   alpha_two.mean().round(3))
print("Overall α (three):", alpha_three.mean().round(3))


# In[93]:


# Step 1: Define the question columns
question_columns = [f'q{i}' for i in range(1, 23)]

# Step 2: Perform majority vote for each 'Origin'
def majority_vote(group):
    # For each question column, get the most common value (mode)
    mode_values = group[question_columns].mode().iloc[0]
    # Preserve the data_source with the highest count (most frequent)
    data_source = group['data_source'].mode()[0]
    mode_values['data_source'] = data_source
    return mode_values

# Step 3: Group by 'Origin' and apply the majority vote logic
majority_vote_df = correct_df.groupby('Origin').apply(majority_vote).reset_index()

# Display the first few rows to confirm
majority_vote_df.head(200)
# Filter the DataFrame to display only rows where 'Origin' is 'ID0004'
filtered_df = majority_vote_df[majority_vote_df['Origin'] == 'ID0004']
filtered_df


# In[94]:


# Find the number of unique "Origin" values that do not replicate in other rows
unique_origins = majority_vote_df['Origin'].value_counts()
non_replicated_origins = unique_origins[unique_origins == 1].count()

# non_replicated_origins
print(len(majority_vote_df))

# Define the columns to search for "high relevance"
question_columns = [f'q{i}' for i in range(1, 23)]

# Check for rows that contain at least one "high relevance"
high_relevance_rows = majority_vote_df[question_columns].eq('high relevance').any(axis=1)

# Count the number of rows that meet this condition
num_high_relevance_rows = high_relevance_rows.sum()
num_high_relevance_rows


# In[ ]:





# ## Remove not relevant sentences 

# In[20]:


# Load the CSV file
df3 = pd.read_csv(paths.TABLES / "Sentence_Label_Original_2k.csv")

# Display the first few rows after removing duplicates
# print(df_op4.columns)
print(df3.head())
print(len(df3))


# In[21]:


import pandas as pd

# assume df3 is already defined and contains the column "centaur_question"

# Method 1: using .str.split() and .str.len()
df3['word_count'] = df3['step1_excerpts'].str.split().str.len()

# If you prefer an explicit apply:
# df3['word_count'] = df3['centaur_question'].apply(lambda txt: len(str(txt).split()))

# Inspect the first few rows
print(df3[['step1_excerpts', 'word_count']].head())
# Compute descriptive statistics for word_count
stats = df3['word_count'].describe()
print(stats)

import pandas as pd

# assume df3 already has the 'word_count' and 'data_source' columns

# Option A: Full describe per group
grouped_describe = df3.groupby('data_source')['word_count'].describe()
print(grouped_describe)

# Option B: Selected metrics per group (mean, std, min, 25%, 50%, 75%, max)
grouped_stats = (
    df3
    .groupby('data_source')['word_count']
    .agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('25%', lambda x: x.quantile(0.25)),
        ('50%', 'median'),
        ('75%', lambda x: x.quantile(0.75)),
        ('max', 'max')
    ])
)
print(grouped_stats)

# If you also want skewness and kurtosis by source:
extras = df3.groupby('data_source')['word_count'].agg(
    skewness=lambda x: x.skew(),
    kurtosis=lambda x: x.kurt()
)
print(extras)


# In[22]:


import pandas as pd

# assume df3 is already defined and contains "step1_excerpts"

# Count occurrences of a number followed by a dot (e.g. "1.", "2.", …)
df3['number_count'] = df3['step1_excerpts'].str.findall(r'\d+\.').str.len()

# Inspect the results
print(df3[['step1_excerpts', 'number_count']].head())
import pandas as pd

# 1. Compute number_count if you haven’t already
df3['number_count'] = df3['step1_excerpts'].str.findall(r'\d+\.').str.len()

# 2. Overall descriptive statistics for number_count
overall_stats = df3['number_count'].describe()
print("Overall number_count statistics:")
print(overall_stats)

# 3. Group‐wise mean and standard deviation by data_source
group_stats = (
    df3
    .groupby('data_source')['number_count']
    .agg(['mean', 'std'])
    .rename(columns={'mean': 'mean_count', 'std': 'std_count'})
)
print("\nnumber_count mean and std by data_source:")
print(group_stats)


# In[23]:


# If you saved the merged file earlier, load it:
# merge_correct_df = pd.read_csv(paths.TABLES / "merge_correct_df.csv")
print(origin_average.head())


# In[24]:


# basic summary statistics for the REMOVED_Sentences column
removed_stats = origin_average["REMOVED_Sentences"].agg(["mean", "std"])

print(removed_stats)


# In[25]:


import pandas as pd

# --- prerequisites: merge_correct_df and df3 already in memory ---

# 1.  choose the join type
#     • "inner"  → keep only Origins that appear in both tables
#     • "left"   → keep all rows from merge_correct_df, add df3 data when ID matches
#     • "right"  → keep all rows from df3, add merge_correct_df data when Origin matches
#     • "outer"  → keep every row from both tables, padding missing values with NaN
join_type = "inner"          # change to "left" if you want a left‑join

# 2.  perform the merge
merged_df = (
    pd.merge(
        origin_average,
        df3,
        left_on="Origin",
        right_on="ID",
        how=join_type,
        suffixes=("_corr", "_df3")   # avoid column‑name clashes
    )
)

# 3.  quick sanity check
print(f"Merged table: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")


# In[26]:


# list of all column names
print(merged_df.columns.tolist())


# In[27]:


import pandas as pd

# -----------------------------------------------
# 1.  Parallel lists for question and sentence
# -----------------------------------------------
q_cols        = [f"q{i}"         for i in range(2, 22)]   # q2 … q21
sentence_cols = [f"sentence_{i}" for i in range(1, 21)]   # sentence_1 … sentence_20
# The two lists are position‑matched: q2 ↔ sentence_1, q3 ↔ sentence_2, …

# -----------------------------------------------
# 2.  Helper functions
# -----------------------------------------------
def removed_sentences(row) -> list[str]:
    """Sentences whose relevance label was ‘REMOVED’."""
    return [
        row[s_col]
        for q_col, s_col in zip(q_cols, sentence_cols)
        if row[q_col] == "REMOVED" and pd.notna(row[s_col])
    ]

def kept_sentences(row) -> list[str]:
    """Sentences whose relevance label was not removed."""
    return [
        row[s_col]
        for q_col, s_col in zip(q_cols, sentence_cols)
        if row[q_col] != "REMOVED" and pd.notna(row[s_col])
    ]

# -----------------------------------------------
# 3.  Construct the new columns
# -----------------------------------------------
merged_df["Filtered_Sentences"] = merged_df.apply(removed_sentences, axis=1)
merged_df["New_Sentences"]      = merged_df.apply(kept_sentences,    axis=1)

# For rows where nothing was removed (REMOVED_Sentences == 0), use all sentences
# so the vignette in centaur_second_round is not empty
mask_zero = merged_df["REMOVED_Sentences"] == 0
merged_df.loc[mask_zero, "Filtered_Sentences"] = merged_df.loc[mask_zero, "New_Sentences"]

# optional preview
merged_df[["Origin", "REMOVED_Sentences", "Filtered_Sentences", "New_Sentences"]].head()


# In[28]:


# count rows where REMOVED_Sentences equals 0
num_zero_removed = (merged_df["REMOVED_Sentences"] == 0).sum()

print(f"Rows with 0 removed sentences: {num_zero_removed}")


# In[29]:


output_path = paths.TABLES / 'less_sentence_df.csv'
merged_df.to_csv(output_path, index=False)


# In[30]:


# frequency table for the `data_source_corr` column
source_counts = (
    merged_df["data_source_corr"]
        .value_counts(dropna=False)      # include any NaN as its own category
        .rename_axis("data_source_corr") # move the index into a column
        .reset_index(name="row_count")   # tidy column names
)

# display the result
source_counts


# In[ ]:





# ## Show with Hyewon + Centaur Lab

# In[31]:


# keep all rows (including those with 0 removed sentences)
merged_df = merged_df.copy()
merged_df.reset_index(drop=True, inplace=True)

output_path = paths.TABLES / 'Centaur_Lab_Second_Round.csv'
merged_df.to_csv(output_path, index=False)
print(len(merged_df))


# In[32]:


# get the header as a Python list
header = merged_df.columns.tolist()

# display it
print(header)


# In[38]:


import pandas as pd

# 1. Define the desired ratio and total sample size
ratios = {
    "jama":       3,
    "medxpert":   3,
    "medbullets": 1,
    "mmlu":       1
}
total_n = 248

# 2. Compute how many to draw from each source
unit_size = total_n / sum(ratios.values())   # = 31.0 in this case
n_per_source = {
    src: int(ratio * unit_size)
    for src, ratio in ratios.items()
}

# If rounding causes us to miss a row, adjust the last source
allocated = sum(n_per_source.values())
if allocated != total_n:
    diff = total_n - allocated
    last = list(n_per_source.keys())[-1]
    n_per_source[last] += diff

# 3. Draw stratified random samples
samples = []
for src, n in n_per_source.items():
    grp = merged_df[merged_df["data_source_df3"] == src]
    samples.append(grp.sample(n=n, random_state=42))

# 4. Combine and shuffle the result
stratified_sample = (
    pd.concat(samples)
      .sample(frac=1, random_state=42)   # mix the sources
      .reset_index(drop=True)
)

# Preview counts to confirm
print(stratified_sample["data_source_df3"].value_counts())


# In[41]:


stratified_sample["data_source_df3"]


# In[42]:


# Save to CSV
output_path = paths.TABLES / 'Centaur_Lab_Classification.csv'
stratified_sample.to_csv(output_path, index=False)

# Provide the download link
output_path


# In[43]:


import re
import ast

# helper to strip any leading enumeration (e.g. "1. ", "03- ", "4) ")
_strip_num_re = re.compile(r'^\s*\d+\s*[\.\)\-:]?\s*')

def strip_leading_number(s: str) -> str:
    """Remove any leading digits, delimiters, and surrounding whitespace."""
    return _strip_num_re.sub("", s).strip()

def format_qa_pair(row) -> str:
    """
    Build a two-step markdown block:
    1) Numbered sentences from New_Sentences (with original numbering removed)
    2) The question_options text under 'Answer QA Details'.
    """
    cell = row["New_Sentences"]
    # ensure we have a Python list of strings
    if isinstance(cell, str):
        try:
            sentences = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            sentences = [cell]
    else:
        sentences = list(cell)

    # clean and enumerate
    cleaned = [strip_leading_number(s) for s in sentences if s and s.strip()]
    numbered = "\n".join(f"{i+1}. {sent}" for i, sent in enumerate(cleaned))

    # assemble the markdown
    return (
        "### Step 1: Read excerpt\n"
        f"{numbered}\n\n"
        "### Step 2: Answer QA Details\n"
        f"{row['question_options']}"
    )

# apply to your DataFrame
merged_df["centaur_second_round"] = merged_df.apply(format_qa_pair, axis=1)

# view example
print(merged_df["centaur_second_round"].iloc[0])


# In[113]:


# Save to CSV
output_path = paths.TABLES / 'Centaur_Lab_Classification.csv'
merged_df.to_csv(output_path, index=False)

# Provide the download link
output_path


# In[ ]:





# In[ ]:





# In[44]:


import ast

less_sentence_df = pd.read_csv(paths.TABLES / 'less_sentence_df.csv')

# Parse Filtered_Sentences from string representation to list
less_sentence_df['Filtered_Sentences'] = less_sentence_df['Filtered_Sentences'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
)

# Standardise column names to match downstream expectations
less_sentence_df = less_sentence_df.rename(columns={
    'Origin':          'ID',
    'data_source_corr': 'data_source',
    'answer_corr':      'answer',
})

# Build df3 index on ID for fast lookup
df3_indexed = df3.set_index('ID') if 'ID' in df3.columns else df3.copy()

# Create the "Remove_Sentences" column
def find_removed_sentences(row):
    row_id = row['ID']
    if row_id in df3_indexed.index:
        original_sentences = df3_indexed.loc[row_id, [f'sentence_{i}' for i in range(1, 22)]].dropna().tolist()
    else:
        original_sentences = []
    filtered_sentences = row['Filtered_Sentences']
    return [s for s in original_sentences if s not in filtered_sentences]

less_sentence_df['Remove_Sentences'] = less_sentence_df.apply(find_removed_sentences, axis=1)

# Compute derived columns
less_sentence_df['Filtered_Sentence_Count'] = less_sentence_df['Filtered_Sentences'].apply(len)
less_sentence_df['Original_Sentence_Count'] = less_sentence_df['ID'].apply(
    lambda x: len(df3_indexed.loc[x, [f'sentence_{i}' for i in range(1, 22)]].dropna())
    if x in df3_indexed.index else 0
)
less_sentence_df['Difference'] = (
    less_sentence_df['Original_Sentence_Count'] - less_sentence_df['Filtered_Sentence_Count']
)

# Rearrange columns
keep_cols = ['ID', 'Filtered_Sentences', 'Remove_Sentences',
             'question_options', 'Filtered_Sentence_Count', 'data_source', 'answer', 'Difference']
less_sentence_df = less_sentence_df[[c for c in keep_cols if c in less_sentence_df.columns]]

less_sentence_df.head()

output_path = paths.TABLES / 'less_sentence_df_with_removed_sentences.csv'
less_sentence_df.to_csv(output_path, index=False)
output_path


# In[45]:


# Calculate the mean and standard deviation for the "Difference" column
mean_difference = less_sentence_df['Difference'].mean()
std_difference = less_sentence_df['Difference'].std()

mean_difference, std_difference


# In[137]:


# Compare original questions vs filtered sentences using less_sentence_df
comparison_df = pd.concat([df3['centaur_question'].reset_index(drop=True),
                            less_sentence_df['Filtered_Sentences'].reset_index(drop=True)], axis=1)
comparison_df.columns = ['Original_Question', 'Filtered_Sentences']

comparison_df.head()


# In[155]:


# Define the formatting template
template = """
### Step 1: Read excerpt
{filtered_sentences}

### Step 2: Answer QA Details
{question_options}
"""

# Function to clean the numbering format
def clean_numbering(filtered_sentences):
    cleaned_sentences = "\n".join([f"{i+1}. {s.split('. ', 1)[-1]}" for i, s in enumerate(filtered_sentences)])
    return cleaned_sentences

# Generate the new column with formatted content
less_sentence_df['centaur_second_round'] = less_sentence_df.apply(
    lambda row: template.format(
        filtered_sentences=clean_numbering(row['Filtered_Sentences']),
        question_options=row['question_options']
    ),
    axis=1
)

# Display the first few rows to verify the structure
less_sentence_df[['centaur_second_round']].head()


# In[156]:


# Select the specified columns from less_sentence_df
output_columns = [
    'ID',
    'Filtered_Sentences',
    'question_options',
    'Remove_Sentences',
    'Filtered_Sentence_Count',
    'data_source',
    'answer',
    'Difference',
    'centaur_second_round'
]

# Filter the DataFrame to include only these columns
export_df = less_sentence_df[output_columns]

# Save to CSV
output_path = paths.TABLES / 'less_sentence_df_export.csv'
export_df.to_csv(output_path, index=False)

# Provide the download link
output_path


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[116]:


import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define label names and initialize counts
label_names = ["Irrelevant", "Low Relevance", "High Relevance"]
num_questions = 22
label_counts = np.zeros((num_questions, len(label_names)))

# Step 2: Count occurrences of each label for each question column
# Mapping labels to their index positions
label_mapping = {
    'not relevant': 0,
    'low relevance': 1,
    'high relevance': 2
}

# Loop through each question column and count occurrences
for idx, col in enumerate(question_columns):
    counts = majority_vote_df[col].value_counts()
    for label, count in counts.items():
        if label in label_mapping:
            label_counts[idx, label_mapping[label]] = count

# Step 3: Normalize the counts to calculate proportions
label_props = label_counts / label_counts.sum(axis=1, keepdims=True)

# Step 4: Plot the results
x = np.linspace(0, 1, num_questions)
plt.figure(figsize=(12, 7))

# Loop through each label category and plot
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for visibility
for i, (label, color) in enumerate(zip(label_names, colors)):
    plt.plot(x, label_props[:, i], label=label, color=color, linewidth=4)
    plt.fill_between(x, label_props[:, i], color=color, alpha=0.2)

# Step 5: Configure plot appearance
plt.xlabel("Question Position (% of Total Questions)", fontsize=24)
plt.ylabel("Proportion of Labels", fontsize=24)
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=22,
    frameon=False
)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(alpha=0.3)
plt.tight_layout()


# Save as PDF
output_path = paths.FIGURES / 'centaur_label_proportions_plot.pdf'
plt.savefig(output_path, format='pdf')
plt.show()


# In[117]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
merged_df = pd.read_csv(paths.DATA / "MAJORITY_Vote_GPT4o_Self_Reported_Relevancy_Labels.csv")

# === Plot 1: GPT4o Self-Reported Relevance ===
label_cols = [f"label_{i}" for i in range(1, 22)]
label_map = {"High Relevance": 0, "Low Relevance": 1, "Irrelevant": 2}
label_names = ["High Relevance", "Low Relevance", "Irrelevant"]
label_counts = np.zeros((21, 3))  # sentence position × label category

# Count labels
for i, col in enumerate(label_cols):
    counts = merged_df[col].dropna().apply(str.strip).value_counts()
    for label, count in counts.items():
        if label in label_map:
            label_counts[i, label_map[label]] += count

# Normalize
label_props_4o = label_counts / label_counts.sum(axis=1, keepdims=True)

# === Plot 2: Centaur Majority Vote Relevance ===
num_questions = 22
label_counts = np.zeros((num_questions, len(label_names)))

# Mapping labels to their index positions
label_mapping = {
    'high relevance': 0,
    'low relevance': 1,
    'not relevant': 2
}

# Loop through each question column and count occurrences
for idx, col in enumerate(question_columns):
    counts = majority_vote_df[col].value_counts()
    for label, count in counts.items():
        if label in label_mapping:
            label_counts[idx, label_mapping[label]] = count

# Normalize the counts to calculate proportions
label_props_centaur = label_counts / label_counts.sum(axis=1, keepdims=True)

# === Plotting Side by Side ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

# Plot for GPT4o Self-Reported
x1 = np.linspace(0, 1, 21)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for visibility
for i, (label, color) in enumerate(zip(label_names, colors)):
    axes[0].plot(x1, label_props_4o[:, i], label=label, color=color, linewidth=4)
    axes[0].fill_between(x1, label_props_4o[:, i], color=color, alpha=0.2)
axes[0].set_xlabel("Sentence Position (% of Total Context)", fontsize=18)
axes[0].set_ylabel("Proportion of Labels", fontsize=18)
axes[0].grid(alpha=0.3)
axes[0].set_title("GPT4o Self-Reported", fontsize=20)

# Plot for Centaur Majority Vote
x2 = np.linspace(0, 1, 22)
for i, (label, color) in enumerate(zip(label_names, colors)):
    axes[1].plot(x2, label_props_centaur[:, i], label=label, color=color, linewidth=4)
    axes[1].fill_between(x2, label_props_centaur[:, i], color=color, alpha=0.2)
axes[1].set_xlabel("Question Position (% of Total Questions)", fontsize=18)
axes[1].grid(alpha=0.3)
axes[1].set_title("Centaur Majority Vote", fontsize=20)

# Unified Legend
fig.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=16,
    frameon=False
)

plt.tight_layout()
plt.savefig(paths.FIGURES / "side_by_side_relevance_distribution.pdf", format='pdf')


# In[131]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# … your data‐loading, label_props_4o and label_props_centaur calculations go here …

# === New: compute proportions for wrong_df ===
# assume wrong_df has the same question_columns and label_mapping as majority_vote_df
num_questions = len(question_columns)
label_counts_wrong = np.zeros((num_questions, len(label_names)))

for idx, col in enumerate(question_columns):
    counts = wrong_df[col].dropna().astype(str).str.strip().str.lower().value_counts()
    for label, count in counts.items():
        if label in label_mapping:
            label_counts_wrong[idx, label_mapping[label]] = count

label_props_wrong = (
    label_counts_wrong
    / label_counts_wrong.sum(axis=1, keepdims=True)
)

# === Prepare x‐axes ===
x1 = np.linspace(0, 1, label_props_4o.shape[0])
x2 = np.linspace(0, 1, label_props_centaur.shape[0])
x3 = np.linspace(0, 1, label_props_wrong.shape[0])

# === Plot side by side by row of three ===
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

# Left: GPT-4o Self-Reported
axes[0].stackplot(
    x1,
    label_props_4o[:, 0], label_props_4o[:, 1], label_props_4o[:, 2],
    labels=label_names, colors=colors, alpha=0.8
)
axes[0].set_title("GPT-4o Self-Reported", fontsize=18)
axes[0].set_xlabel("Sentence Position (% of Total Context)", fontsize=14)
axes[0].set_ylabel("Proportion of Labels", fontsize=14)
axes[0].grid(alpha=0.3)

# Middle: Centaur Majority Vote (Correct)
axes[1].stackplot(
    x2,
    label_props_centaur[:, 0], label_props_centaur[:, 1], label_props_centaur[:, 2],
    labels=label_names, colors=colors, alpha=0.8
)
axes[1].set_title("Centaur Majority Vote (Correct)", fontsize=18)
axes[1].set_xlabel("Question Position (% of Total Questions)", fontsize=14)
axes[1].grid(alpha=0.3)

# Right: Centaur Majority Vote (Wrong)
axes[2].stackplot(
    x3,
    label_props_wrong[:, 0], label_props_wrong[:, 1], label_props_wrong[:, 2],
    labels=label_names, colors=colors, alpha=0.8
)
axes[2].set_title("Centaur Majority Vote (Wrong)", fontsize=18)
axes[2].set_xlabel("Question Position (% of Total Questions)", fontsize=14)
axes[2].grid(alpha=0.3)

# Shared legend at bottom
fig.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=16,
    frameon=False
)

plt.tight_layout()
plt.savefig(paths.FIGURES / "three_panel_relevance_profiles.pdf", format="pdf")
plt.show()


# In[118]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# … your data‐loading and label_props_4o / label_props_centaur calculation go here …

# Define the x‐axes
x1 = np.linspace(0, 1, label_props_4o.shape[0])
x2 = np.linspace(0, 1, label_props_centaur.shape[0])

# Colors and legend labels
colors     = ['#1f77b4', '#ff7f0e', '#2ca02c']  # high, low, irrelevant
label_names = ["High Relevance", "Low Relevance", "Irrelevant"]

# Create the subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

# --- Left: GPT-4o self-reported ---
axes[0].stackplot(
    x1,
    label_props_4o[:, 0],
    label_props_4o[:, 1],
    label_props_4o[:, 2],
    labels=label_names,
    colors=colors,
    alpha=0.8
)
axes[0].set_title("GPT-4o Self-Reported", fontsize=20)
axes[0].set_xlabel("Sentence Position (% of Total Context)", fontsize=18)
axes[0].set_ylabel("Proportion of Labels", fontsize=18)
axes[0].grid(alpha=0.3)

# --- Right: Centaur majority vote ---
axes[1].stackplot(
    x2,
    label_props_centaur[:, 0],
    label_props_centaur[:, 1],
    label_props_centaur[:, 2],
    labels=label_names,
    colors=colors,
    alpha=0.8
)
axes[1].set_title("Centaur Majority Vote", fontsize=20)
axes[1].set_xlabel("Question Position (% of Total Questions)", fontsize=18)
axes[1].grid(alpha=0.3)

# Shared legend
fig.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=16,
    frameon=False
)

plt.tight_layout()
plt.savefig(paths.FIGURES / "stacked_relevance_profiles.pdf", format="pdf")
plt.show()


# In[89]:


import matplotlib.pyplot as plt
import numpy as np

# Get unique data sources from majority_vote_df
data_sources = majority_vote_df['data_source'].unique()

# Mapping labels to their index positions
label_mapping = {
    'high relevance': 0,
    'low relevance': 1,
    'not relevant': 2
}
label_names = ["High Relevance", "Low Relevance", "Not Relevant"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for visibility

# === Initialize the plot ===
fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)

# Loop through each data source and plot
for idx, source in enumerate(data_sources):
    # Filter for the current data source
    filtered_df = majority_vote_df[majority_vote_df['data_source'] == source]

    # Initialize label counts
    num_questions = 22
    label_counts = np.zeros((num_questions, len(label_names)))

    # Loop through each question column and count occurrences
    for i, col in enumerate(question_columns):
        counts = filtered_df[col].value_counts()
        for label, count in counts.items():
            if label in label_mapping:
                label_counts[i, label_mapping[label]] = count

    # Normalize the counts to calculate proportions
    label_props_centaur = label_counts / label_counts.sum(axis=1, keepdims=True)

    # Plot for the specific data source
    x = np.linspace(0, 1, num_questions)  # Changed to be from 0 to 1 always
    for i, (label, color) in enumerate(zip(label_names, colors)):
        axes[idx].plot(x, label_props_centaur[:, i], label=label, color=color, linewidth=4)
        axes[idx].fill_between(x, label_props_centaur[:, i], color=color, alpha=0.2)

    # Plot settings
    axes[idx].set_xlabel("Question Position (% of Total Questions)", fontsize=18)
    axes[idx].set_xlim(0, 1)  # Ensures x-axis is always 0 to 1
    axes[idx].grid(alpha=0.3)
    axes[idx].set_title(f"{source}", fontsize=20)

# Unified Legend
fig.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=16,
    frameon=False
)

# Shared y-axis label
fig.text(0.07, 0.5, 'Proportion of Labels', va='center', rotation='vertical', fontsize=18)

plt.tight_layout()
plt.savefig(paths.FIGURES / "centaur_majority_vote_adjusted.pdf", format='pdf')
plt.show()


# In[57]:


print(len(majority_vote_df))


# In[58]:


# Initialize the label names and counts
label_names = ["Not Relevant (0)", "Low Relevancy (0.5)", "High Relevancy (1)"]
label_counts = np.zeros((22, 3))  # 22 question columns × 3 label categories (0, 0.5, 1)

# Count occurrences of 0, 0.5, and 1 for each question column
for i, col in enumerate(question_columns):
    counts = majority_vote_df[col].value_counts()
    for label, count in counts.items():
        if label in [0, 0.5, 1]:
            label_counts[i, int(label * 2)] = count

# Normalize the counts for proportion display
label_props = label_counts / label_counts.sum(axis=1, keepdims=True)

# Plot
x = np.linspace(0, 1, 22)
plt.figure(figsize=(12, 7))
for i in range(3):
    plt.plot(x, label_props[:, i], label=label_names[i], linewidth=5)
    plt.fill_between(x, label_props[:, i], alpha=0.2)

plt.xlabel("Question Position (% of Total Questions)", fontsize=24)
plt.ylabel("Proportion of Labels", fontsize=24)
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=22,
    frameon=False
)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[33]:


import pandas as pd

# Define the question columns
question_columns = [f'q{i}' for i in range(1, 23)]

# Replace "not applicable" or other non-numeric values with NaN
majority_vote_df.replace('not applicable', None, inplace=True)
valid_vote_df = majority_vote_df[question_columns].apply(pd.to_numeric, errors='coerce')

# Count the number of 0s and 1s for each row
valid_vote_df['Count_0s'] = (valid_vote_df == 0).sum(axis=1)
valid_vote_df['Count_1s'] = (valid_vote_df == 1).sum(axis=1)

# Calculate the average number of 0s and 1s per row
valid_vote_df['Average_0s_per_row'] = valid_vote_df['Count_0s'] / len(question_columns)
valid_vote_df['Average_1s_per_row'] = valid_vote_df['Count_1s'] / len(question_columns)

# Display the first few rows
print(valid_vote_df[['Count_0s', 'Count_1s', 'Average_0s_per_row', 'Average_1s_per_row']].head())

# Calculate overall averages
overall_avg_0s = valid_vote_df['Average_0s_per_row'].mean()
overall_avg_1s = valid_vote_df['Average_1s_per_row'].mean()

# Display the overall results
print({
    "Overall_Average_0s_Per_Row": overall_avg_0s,
    "Overall_Average_1s_Per_Row": overall_avg_1s
})

import pandas as pd

# Define the question columns
question_columns = [f'q{i}' for i in range(1, 23)]

# Replace "not applicable" or other non-numeric values with NaN
majority_vote_df.replace('not applicable', None, inplace=True)
valid_vote_df = majority_vote_df[question_columns].apply(pd.to_numeric, errors='coerce')

# Count the total number of 0s and 1s for each row
valid_vote_df['Count_0s_and_1s'] = (valid_vote_df == 0).sum(axis=1) + (valid_vote_df == 1).sum(axis=1)

# Calculate the average and standard deviation of the total counts per row
average_0s_and_1s_per_row = valid_vote_df['Count_0s_and_1s'].mean()
std_0s_and_1s_per_row = valid_vote_df['Count_0s_and_1s'].std()

# Display the results
print({
    "Average_0s_and_1s_Per_Row": average_0s_and_1s_per_row,
    "Standard_Deviation_0s_and_1s_Per_Row": std_0s_and_1s_per_row
})


# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the question columns
question_columns = [f'q{i}' for i in range(1, 23)]

# Initialize the label names and counts
label_names = ["Not Relevant (0)", "Relevant (1)"]
label_counts = np.zeros((22, 2))  # 22 question columns × 2 label categories (0 or 1)

# Count occurrences of 0 and 1 for each question column
for i, col in enumerate(question_columns):
    counts = majority_vote_df[col].value_counts()
    for label, count in counts.items():
        if label in [0, 1, 0.0, 1.0]:
            label_counts[i, int(label)] = count

# Normalize the counts for proportion display
label_props = label_counts / label_counts.sum(axis=1, keepdims=True)

# Plot
x = np.linspace(0, 1, 22)
plt.figure(figsize=(12, 7))
for i in range(2):
    plt.plot(x, label_props[:, i], label=label_names[i], linewidth=5)
    plt.fill_between(x, label_props[:, i], alpha=0.2)

plt.xlabel("Question Position (% of Total Questions)", fontsize=24)
plt.ylabel("Proportion of Labels", fontsize=24)
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    fontsize=22,
    frameon=False
)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[132]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the question columns
question_columns = [f'q{i}' for i in range(1, 23)]

# Get unique data sources
data_sources = majority_vote_df['data_source'].unique()

# Setup the grid layout
num_sources = len(data_sources)
num_cols = 4
num_rows = (num_sources + num_cols - 1) // num_cols

# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
axes = axes.flatten()

# Loop through each data source and plot separately
for idx, source in enumerate(data_sources):
    # Filter the DataFrame for the specific data source
    source_df = majority_vote_df[majority_vote_df['data_source'] == source]
    
    # Initialize the label names and counts
    label_names = ["Not Relevant (0)", "Relevant (1)"]
    label_counts = np.zeros((22, 2))  # 22 question columns × 2 label categories (0 or 1)

    # Count occurrences of 0 and 1 for each question column
    for i, col in enumerate(question_columns):
        counts = source_df[col].value_counts()
        for label, count in counts.items():
            if label in [0, 1, 0.0, 1.0]:
                label_counts[i, int(label)] = count

    # Normalize the counts for proportion display
    label_props = label_counts / label_counts.sum(axis=1, keepdims=True)

    # Plot in the respective subplot
    x = np.linspace(0, 1, 22)
    for i in range(2):
        axes[idx].plot(x, label_props[:, i], label=label_names[i], linewidth=2)
        axes[idx].fill_between(x, label_props[:, i], alpha=0.2)

    axes[idx].set_title(f"{source}", fontsize=16)
    axes[idx].set_xlabel("Question Position (% of Total Questions)", fontsize=12)
    axes[idx].set_ylabel("Proportion of Labels", fontsize=12)
    axes[idx].grid(alpha=0.3)
    axes[idx].legend(loc='upper right', fontsize=10)

# Hide any extra subplots if not used
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




