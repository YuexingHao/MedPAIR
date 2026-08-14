# July 20, 2026 — Qwen-72B low/irrelevant extraction

Low relevance passages used for the *Physician-Irrelevant* style condition.

## Files

| File | Rows | Notes |
|---|---:|---|
| `Low_Irr_qwen_72b_cc.csv` | 1297 | Full relevance pool. **Still contains the answer splice described below** |
| `Low_Irr_qwen_72b_cc_answerable930.csv` | 930 | Answerable cohort, answer splice removed. **Use this one** |
| `Low_Irr_qwen_72b_cc_answerable930_pre_leak_removal.csv` | 930 | The same cohort before removal, kept so the fix can be audited |
| `_removed_answer_leak_log.csv` | 210 | One row per removed string: origin, option letter, whether it was the gold option, word count, exact text |
| `_missing_from_low_irr_3.csv` | 3 | Questions in the answerable 933 that never existed in the 1297 pool |

## Columns

`Origin`, `context`, `question_options`, `answer_df3` (gold letter),
`data_source_corr`, `Low_Irr_70B` (the low relevance passage).

Note the column is named `Low_Irr_70B` while the file is named `qwen_72b`. The
extraction matches neither the archived 70B nor the archived 72B output
(best median Jaccard against any archived column is 0.318), so the producing
model is not settled. Confirm before citing a model name.

## The answer splice

During construction, answer option text was appended to the case sentence list
and therefore ended up inside the passage. On the 26 verifiable cases it is
always the **last** sentence of the case, which is the signature of an append
bug rather than natural prose.

It affected **53 questions, all `jama`** (9.1% of jama; zero in medbullets,
medxpert, mmlu). The defect originates in the `context` column, and the
relevance filter inherited 94% of it, because an injected option string does not
look like clinical information and so gets routed to "low relevance".

In `Low_Irr_qwen_72b_cc_answerable930.csv` this is fixed. 210 option strings were
removed across 123 rows, 85 gold and 125 distractor (207 whole options plus 3
sentence fragments of multi-sentence options). The rule was: remove any
answer option appearing verbatim, **regardless of letter**, guarded by
case-sensitive matching and a 4 word minimum. Removing only the gold option would
have left distractors pointing away from the answer, which is worse than leaving
both.

Result after the fix:

| | Before | After |
|---|---:|---:|
| Correct option verbatim | 9.0% | **0.00%** |
| Distractor verbatim | 3.0% | 0.16% |
| Word-overlap solver | 23.8% | 23.3% (chance 24.9%) |

The 7 remaining distractor matches are natural clinical prose, not splices:
`Eczema`, `Midsystolic click`, `Hypothyroidism`, `Haloperidol`, `1%`, `10%`, and
one option that parses to the single letter `M`. None is a correct answer.

## Still outstanding

- `Low_Irr_qwen_72b_cc.csv` (1297 rows) has **not** been fixed.
- The `context` column of the 930 row file has **not** been fixed, only
  `Low_Irr_70B`. Any condition built from `context` is still affected.
- The production annotation run in `../2026_08_Aug/` is unaffected. Its passages
  were already de-leaked before collection. See
  `../2026_08_Aug/README_Production_930.md` §6.
