# Toolkit

A generic toolkit for evaluating GPT models on multiple-choice QA datasets with sentence-level relevance labeling.

Given a CSV dataset, the toolkit will:
1. Query a GPT model to answer each question and label each sentence as High Relevance, Low Relevance, or Irrelevant
2. Save results incrementally (with resume support)
3. Compute accuracy, refusal rate, relevance distribution, and relevance-accuracy breakdown

---

## Requirements

```bash
pip install openai pandas numpy
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Input CSV Format

Your input CSV must have these kinds of columns (exact names are configured interactively):

| Column | Description |
|---|---|
| ID column | A unique identifier per row (used to resume interrupted runs) |
| Sentences column | Pre-formatted sentence text, e.g. `S1: ...\nS2: ...` |
| Sentence count column | Integer number of sentences in that row |
| Options column | The full question + answer choices text sent to the model |
| Correct answer column | Ground-truth answer letter (A, B, C, ...) — optional, needed for accuracy |
| Group-by column | Optional — e.g. `data_source` — used to break down stats per group |

---

## Usage

All commands are run from the `Codes/` directory:

```bash
cd /path/to/MedPAIR/Codes
```

### Step 1 — Setup

Run the interactive setup to create a config file for your dataset:

```bash
python -m toolkit setup --config my_experiment.json
```

The setup will:
- Read your input CSV and display its column names
- Ask you to identify which column is which
- Ask whether each row has one question or multiple questions
- Ask which GPT model to use (default: `gpt-4o`)
- Save all settings to `my_experiment.json`

You only need to run setup once per dataset/experiment. The saved config can be reused or edited directly.

**Example session:**

```
=== Toolkit Setup ===

Input CSV path: ../Data/medpair_input.csv
Output CSV path: ../Results/gpt4o_output.csv

-- Dataset columns --
  Columns: ['ID_corr', 'original_sentences', 'sentence_number_corr', 'question_options', 'answer_corr', 'data_source_corr']

Unique ID column (for resume support) (Enter to skip): ID_corr
Pre-formatted sentences column: original_sentences
Sentence count column: sentence_number_corr
Question + answer options column: question_options
Group-by column for per-group analysis (Enter to skip): data_source_corr

-- Question mode --
  1. Single question per row
  2. Multiple questions per row
Choice [1]: 1

Correct answer column (for accuracy analysis) (Enter to skip): answer_corr

-- Model --
Model name [gpt-4o]:
Temperature [0]:
Max sentences to label [30]:
Custom prompt template file path (optional) (Enter to skip):

Config saved to: my_experiment.json
```

### Step 2 — Query

Send each row to the model and save results:

```bash
python -m toolkit query --config my_experiment.json
```

**Resume support:** if the run is interrupted, re-running the same command will skip rows already saved to the output file (matched by ID column).

**Override paths without editing the config:**

```bash
python -m toolkit query --config my_experiment.json --input other_input.csv --output other_output.csv
```

#### Output CSV

The output CSV contains all original columns plus:

| Column | Description |
|---|---|
| `Raw_Response` | Full model response text |
| `LLM_answer` | Extracted answer letter (single-question mode) |
| `<question_name>` | Extracted answer letter per question (multi-question mode) |
| `label_1` … `label_N` | Sentence relevance labels (`High Relevance`, `Low Relevance`, `Irrelevant`) |

### Step 3 — Analyze

Compute metrics on the query output:

```bash
python -m toolkit analyze --config my_experiment.json --output-dir Results/analysis_gpt4o
```

This writes two files to the output directory:

- **`analysis_report.txt`** — human-readable stats report
- **`summary.csv`** — one row per input row with answer correctness flags, majority relevance label, and per-label percentages (useful for downstream plots)

**Metrics computed:**

| Metric | Description |
|---|---|
| Accuracy | % of rows where model answer matches correct answer |
| Refusal rate | % of rows where model did not return a valid answer letter |
| Relevance distribution | Mean % of sentences labeled High / Low / Irrelevant per row, with 95% CI |
| Relevance-accuracy | Accuracy broken down by each row's majority relevance label |

All metrics are reported overall and per group (if a group-by column was configured).

---

### All-in-one

Run setup → query → analyze in a single command:

```bash
python -m toolkit run --config my_experiment.json --output-dir Results/analysis_gpt4o
```

---

## Multi-question mode

If each row has multiple questions (e.g. four SDOH questions), choose mode `2` during setup. You will name each question (this becomes its output column name) and optionally identify its correct-answer column.

The model is prompted to answer all questions at once and label sentences relative to the full set of questions. Each question gets its own accuracy and refusal rate reported separately.

---

## Custom prompts

To override the default prompt, create a `.txt` file with `{options}` and `{sentences}` as placeholders:

```
You are a clinical reasoning assistant. Given the following question and sentences, ...

Question and Options:
{options}

Sentences:
{sentences}

Return only a JSON object: {"Answer": "<letter>", "Sentence_Relevance": [...]}
```

Then point to it during setup, or set `"custom_prompt_path"` directly in the config JSON.

---

## Config JSON reference

The config file (`my_experiment.json`) can be edited directly. All fields:

```json
{
  "input_path": "",
  "output_path": "",
  "id_col": null,
  "sentences_col": "",
  "sentence_count_col": "",
  "options_col": "",
  "groupby_col": null,
  "mode": "single",
  "correct_answer_col": null,
  "questions": [],
  "provider": "openai",
  "model_name": "gpt-4o",
  "temperature": 0.0,
  "api_key_env": "OPENAI_API_KEY",
  "custom_prompt_path": null,
  "max_sentences": 30
}
```

For multi-question mode, `questions` is a list of objects:

```json
"questions": [
  {"name": "household_income", "correct_col": "income_answer_corr"},
  {"name": "housing_status",   "correct_col": "housing_answer_corr"}
]
```
