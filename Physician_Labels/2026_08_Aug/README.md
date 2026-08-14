# External MedPAIR Pilot — 6 Onboarding Sets

Analysis of `External_MedPAIR_Pilot_Responses_6sets.csv`.

**Last updated:** 2026-07-30

---

## 1. The file

| | |
|---|---|
| Rows | 406 responses |
| Questions | 90 unique `ORIGIN_ID` (44 with 4 responses, 46 with 5) |
| Annotators | 28 (`ANNOTATOR_LABEL` A1–A28); 27 did 15 items each, A22 did 1 |
| Sets | 6 (`PROJECT_NAME` "Merak Onboarding Assessment A"–"F"), 15 questions each |
| Collected | 2026-07-24 → 2026-07-28 |

### Columns

| Column | Notes |
|---|---|
| `ORIGIN_ID` | Question ID; joins to the MedPAIR cohort |
| `RESPONSE_VALUE` | `option_a` … `option_j`, **or `option_notsure`** |
| `CORRECT_ANSWER` | Gold letter, `A`–`J` |
| `SOURCE` | `jama`, `medxpert`, `medbullets`, `mmlu` |
| `ANNOTATOR_LABEL` | Anonymised annotator |
| `AHT_MINUTES` | Handling time; mean 1.80, median 1.20, SD 2.02, range 0.12–14.93 |
| `LOW_IRR_CONTEXT` | Free text, one distinct value per question — **see §6** |
| `PROJECT_NAME`, `TASK_TYPE`, `DATE_OF_COMPLETION` | Set / submission / timestamp |

---

## 2. Read this before quoting a number

**149 of 406 responses (36.7%) are `option_notsure`.** They affect 56 of the 90
questions, and on 12 questions *every* response is "not sure". How abstentions
and ties are handled moves the headline accuracy between **37.8% and 70.6%**, so
always state the convention alongside the number.

Three conventions are reported below. **Variant A is the primary**, because it
matches `_compute_pt_condition_rows` in
`Figures/llm_improvement/per_category_June/make_medpair_combined_figure.py`,
which counts a tied majority vote as incorrect.

| Variant | "not sure" | Tie / no valid vote | Overall |
|---|---|---|---|
| **A (primary)** | abstains | counted **incorrect** | **53.3%** (48/90) |
| A-decided | abstains | **excluded** from denominator | 70.6% (48/68) |
| B | counts as a vote, can win | counted incorrect | 37.8% (34/90) |

Under variant B, "not sure" wins the vote outright on **31 of 90** questions.

---

## 3. Majority-vote accuracy by source — variant A

| Source | Questions | Correct | **Accuracy** | Ties | "not sure" responses |
|---|---:|---:|---:|---:|---:|
| mmlu | 12 | 9 | **75.0%** | 2 | 18 |
| jama | 42 | 28 | **66.7%** | 9 | 56 |
| medbullets | 12 | 5 | **41.7%** | 4 | 26 |
| medxpert | 24 | 6 | **25.0%** | 7 | 49 |
| **Overall** | **90** | **48** | **53.3%** | 22 | 149 |

### Same table, other conventions

| Source | A (primary) | A-decided | B | decided n |
|---|---:|---:|---:|---:|
| mmlu | 75.0% | 90.0% | 41.7% | 10 |
| jama | 66.7% | 84.8% | 52.4% | 33 |
| medbullets | 41.7% | 62.5% | 16.7% | 8 |
| medxpert | 25.0% | 35.3% | 20.8% | 17 |
| **Overall** | **53.3%** | **70.6%** | **37.8%** | 68 |

MedXpert is the weakest subset throughout, consistent with the rest of the
project: it is the only 10-option source (answers reach `J`; the other three top
out at `D`).

---

## 4. Per-response accuracy (no voting)

Each response scored individually.

| Source | Committed responses | Correct | Accuracy |
|---|---:|---:|---:|
| mmlu | 36 | 28 | 77.8% |
| jama | 133 | 92 | 69.2% |
| medbullets | 29 | 16 | 55.2% |
| medxpert | 59 | 21 | 35.6% |
| **Overall** | **257** | **157** | **61.1%** |

Scoring "not sure" as wrong instead gives 157/406 = **38.7%**.

---

## 5. Per-annotator

Committed-answer accuracy ranges **11.1% to 91.7%** across the 27 annotators who
completed a full set of 15. Abstention rates vary just as widely, **6.7% to
80.0%**, so annotator-level accuracy is not comparable without also reporting
how much each abstained.

| | Highest | Lowest |
|---|---|---|
| Accuracy | A25 91.7% (20.0% not-sure) | A17 11.1% (40.0% not-sure) |
| Abstention | A18 80.0% (66.7% acc) | A23 6.7% (78.6% acc) |

A22 contributed 1 response and should be excluded from annotator-level analysis.

---

## 6. Open question — what context did annotators see?

`LOW_IRR_CONTEXT` holds one distinct free-text value per question and appears to
contain **low-relevance / irrelevant sentences**. If annotators were shown only
that content, then the 36.7% abstention rate and 53.3% accuracy are the expected
signature of a *Physician-Irrelevant*-style condition, **not** a general
accuracy measure, and these numbers must not be compared with full-context
accuracies elsewhere in the project.

**This has not been confirmed.** Resolve it before using these numbers in the
paper or comparing them to any other condition.

---

## 7. Reproducing

```python
import pandas as pd, collections

d = pd.read_csv("External_MedPAIR_Pilot_Responses_6sets.csv")
d["letter"] = (d["RESPONSE_VALUE"].astype(str).str.strip().str.lower()
                 .map(lambda s: s.split("option_")[-1].upper()
                      if s.startswith("option_") else ""))
d["is_notsure"] = d["RESPONSE_VALUE"].eq("option_notsure")
d["gold"] = d["CORRECT_ANSWER"].astype(str).str.strip().str.upper()

def majority(votes):
    """(winner, tie). A tie or an empty ballot returns ('', True)."""
    votes = [v for v in votes if v]
    if not votes:
        return "", True
    c = collections.Counter(votes)
    top = max(c.values())
    win = sorted(k for k, v in c.items() if v == top)
    return (win[0], False) if len(win) == 1 else ("", True)

rows = []
for oid, g in d.groupby("ORIGIN_ID"):
    committed = [l for l, ns in zip(g["letter"], g["is_notsure"]) if not ns]
    win, tie = majority(committed)                     # variant A
    rows.append({"ORIGIN_ID": oid, "SOURCE": g["SOURCE"].iloc[0],
                 "gold": g["gold"].iloc[0], "win": win, "tie": tie,
                 "n_notsure": int(g["is_notsure"].sum()),
                 "correct": (not tie) and win == g["gold"].iloc[0]})
r = pd.DataFrame(rows)

print(100 * r["correct"].mean())                                  # 53.3
print((100 * r.groupby("SOURCE")["correct"].mean()).round(1))     # per source
```

For variant B, replace `committed` with
`["NOTSURE" if ns else l for l, ns in zip(g["letter"], g["is_notsure"])]`.
For A-decided, filter to `~r["tie"]` before taking the mean.

---

## 8. Conventions used elsewhere in this repo

- **Ties count as incorrect** — matches `_compute_pt_condition_rows`.
- **Accuracy over a fixed cohort denominator**, with unanswered items counted
  incorrect, is what `medpair_accuracy_lib.py` does for the 933-question set
  (see `Figures/llm_improvement/per_category_June/`). Variant B is the analogue
  of that convention here; variant A is the more permissive reading.
- Answers are normalised by extracting a single `A`–`J` letter. Free-form text
  and error strings must **not** be regex-scanned for `[A-J]` — that matches the
  `I` in "I'm sorry" and the `E` in "Error". See `medpair_accuracy_lib.norm_choice`.
