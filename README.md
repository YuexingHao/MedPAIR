# MedPAIR: Measuring Physicians and AI Relevance Alignment in Medical Question Answering

MedPAIR is a "Medical Dataset Comparing Physician Trainees and AI Relevance
Estimation and Question Answering". It compares LLM reasoning processes to those
of physician trainees so that future work can focus on relevant features. MedPAIR
is the first benchmark to match relevancy annotated by clinical professional
labelers against relevancy estimated by LLMs. The motivation is to check whether
what an LLM finds relevant in a clinical case matches what a physician trainee
finds relevant.

---

## Quick start

```bash
git clone https://github.com/YuexingHao/MedPAIR.git
cd MedPAIR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reproducing the headline tables and figures needs only the core packages. The
model prediction CSVs are committed, so **you do not need a GPU or an API key**
unless you want to regenerate predictions from scratch.

```bash
# main result figure, built from the committed prediction CSVs
python Figures/llm_improvement/per_category_June/make_medpair_combined_figure.py

# significance tests reported in the paper
python Figures/llm_improvement/per_category_June/mcnemar_significance.py
```

---

## Repository map

| Path | What it holds |
|---|---|
| `Dataset/` | Curated QA items and per-model prediction sets. `merged_llm_4k_questions_with_QA_ID.csv` is the merged question pool |
| `Physician_Labels/` | Human relevance and answer annotations, organised chronologically by collection date. See its own `README.md` |
| `Codes/` | Original analysis notebooks and scripts: dataset curation, physician annotation analysis, low/irrelevant sentence removal |
| `After_PT_Removal/` | The "remove what physicians called irrelevant, then re-ask the model" experiments, one directory per model |
| `Merge/` | Sentence attribution (ContextCite) runs and merge notebooks |
| `Figures/` | Figure generation code and outputs, grouped by figure family |
| `Unused_Files/` | Source question banks (MedBullets, MedXpertQA, JAMA, MMLU, HeadQA, MedQA, NCCN) and superseded material |
| `static/`, `index.html`, `CNAME` | The project GitHub Pages site |

### Conventions used throughout

These matter for reproducing any number in the paper:

- **Ties count as incorrect** in majority voting, matching
  `_compute_pt_condition_rows`.
- **Accuracy uses a fixed cohort denominator**, with unanswered items counted
  incorrect. See `Figures/llm_improvement/per_category_June/medpair_accuracy_lib.py`.
- Answers are normalised by extracting a single `A`–`J` letter. Free-form text
  and error strings must **not** be regex-scanned for `[A-J]`, because that
  matches the `I` in "I'm sorry" and the `E` in "Error". Use
  `medpair_accuracy_lib.norm_choice`.
- Question cohorts are nested and are not interchangeable. The 2000-question
  labelled pool contains the 1297-question relevance pool, which contains the
  933 answerable questions. `Physician_Labels/README.md` documents the exact
  membership, including three questions present in the 933 but absent from
  the 1297.

---

## Pipeline

Each stage consumes the previous stage's output. Every stage's outputs are
committed, so you can enter the pipeline at any point.

**1. Curate the QA pool** — `Codes/QA_Dataset_Curation/`

`DataMerge.ipynb` merges the source question banks into
`Dataset/merged_llm_4k_questions_with_QA_ID.csv` and assigns the `QA_ID` /
`Origin` identifiers that every later stage joins on.

**2. Collect physician relevance labels** — `Physician_Labels/`

Clinical labelers mark each sentence of a case as high or low/irrelevant, and
answer the question. Organised by collection date. Majority-vote helpers and the
per-date analyses live alongside the data.

**3. Estimate relevance with LLMs** — `Merge/attribution/`, `Codes/QA_Dataset_Curation/`

ContextCite and self-report runs produce per-sentence relevance estimates that
are compared against the physician labels.

**4. Remove low/irrelevant content and re-ask** — `After_PT_Removal/`

The core experiment. One directory per model, each with the same layout:

```
After_PT_Removal/<Model>/
├── data/raw/           inputs for this model
├── notebooks/          the inference script
├── run_*.sbatch        Slurm launcher
├── logs/               job stdout/stderr for the run that produced results/
└── results/
    ├── predictions/    raw model output
    └── tables/         scored summaries
```

Shared helpers are in `After_PT_Removal/shared/scripts/`
(`sr_common.py`, `merge_sr_partials.py`, `summarize_sr_results.py`).

To re-run one model on a Slurm cluster:

```bash
sbatch After_PT_Removal/Qwen2.5-72B-Instruct/run_qwen72b_sr_inference.sbatch
```

Sharded runs write partial files; merge them before scoring:

```bash
python After_PT_Removal/shared/scripts/merge_sr_partials.py
python After_PT_Removal/shared/scripts/summarize_sr_results.py
```

**5. Score and plot** — `Figures/`

`Figures/llm_improvement/per_category_June/` holds the canonical accuracy
library and the scripts that build the main result figures and the McNemar
significance tables.

---

## Re-running model inference

Only needed if you want to regenerate predictions rather than use the committed
ones.

**Open-weight models** (Llama-70B, Qwen2.5-14B/72B, MedGemma-27B) run locally
through vLLM and need a multi-GPU node. The `.sbatch` files record the resources
each run actually used.

**Hosted models** (GPT-4o, GPT-5) read credentials from the environment. Nothing
is read from a committed file:

```bash
# used by run_gpt4o_sr_inference.py and run_gpt5_sr_inference.py
export OPENAI_API_KEY=...

# Azure path only (predict_gpt5_on_context.py); the endpoint is passed as a
# command-line argument, not an environment variable
export AZURE_OPENAI_API_KEY=...
```

---

## Data notes

- `Physician_Labels/README.md` and
  `Physician_Labels/2026_08_Aug/README_Production_930.md` document annotation
  conventions, abstention handling, and the accuracy variants. **Read them before
  quoting any accuracy number**, because the "not sure" convention alone moves
  the headline by more than 20 points.
- Some annotation exports are CR-terminated and mix UTF-8 with stray Mac Roman
  bytes. `README_Production_930.md` §9 has a loader that handles both.
- Physician relevance annotation is inherently subjective. Inter-rater agreement
  findings are in `Physician_Labels/IRR_FINDINGS_SUMMARY.md` and
  `Physician_Labels/SENTENCE_LEVEL_IRR_FINDINGS.md`.

---

## What is deliberately not in this repository

- **Python environments.** `Unused_Files/myenv/` and `.conda/` exist in older
  history but are no longer tracked. Use `requirements.txt`.
- **The manuscript.** `medpair_paper/` is a separate Overleaf git repository
  checked out inside this tree. It is versioned there, not here.

---

## Acknowledgments

This work was supported in part by an award from the Hasso Plattner Foundation,
a National Science Foundation (NSF) CAREER Award (#2339381), and an AI2050 Early
Career Fellowship (G-25-68042).

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
