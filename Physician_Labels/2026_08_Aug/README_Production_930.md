# External MedPAIR Production Run — 930 Questions

Analysis of `External MedPair Output Final.csv`.

**Last updated:** 2026-08-14

Companion to `README.md`, which covers the 6-set pilot
(`External_MedPAIR_Pilot_Responses_6sets.csv`). Same conventions are used here so
the two are directly comparable.

---

## 1. The file

| | |
|---|---|
| Rows | 2790 responses |
| Questions | 930 unique `ORIGIN_ID`, **exactly 3 responses each** |
| Annotators | 14 (`Annotator ID` A1–A14), 77 to 256 responses each |
| Batches | 5 (`PROJECT_NAME`), plus 5 `TASK_TYPE` submission IDs |
| Collected | 2026-07-31 → 2026-08-11 |
| Cohort | The 930 answerable questions, identical to `July20_2026_Data/Low_Irr_qwen_72b_cc_answerable930.csv` |

No annotator answered the same question twice, and every question has 3 distinct
annotators.

### Read the file with the right encoding

The file is **not** plain UTF-8 and **not** newline-terminated. `pd.read_csv` on
defaults raises `UnicodeDecodeError`. Two problems:

1. **Mixed encoding.** The body is UTF-8, but 246 bytes are stray Mac Roman
   singles (`0xA1`=°, `0xD5`=', `0xD0`=–, `0xB5`=µ). Decoding the whole file as
   latin-1 or mac_roman silently mojibakes the rest.
2. **CR line terminators.** 2790 `\r`, zero `\n`. Pass `lineterminator="\r"`.

See §8 for a loader that handles both.

### Columns

| Column | Notes |
|---|---|
| `Annotator ID` | A1–A14. Note the space in the name, unlike the pilot's `ANNOTATOR_LABEL` |
| `ORIGIN_ID` | Question ID, joins to the 930 cohort |
| `RESPONSE_VALUE` | `option_a` … `option_j`, or `option_notsure` |
| `CORRECT_ANSWER` | Gold letter `A`–`J`. Matches `answer_df3` on all 930 |
| `SOURCE` | `jama`, `medxpert`, `medbullets`, `mmlu` |
| `AHT_MINUTES` | Handling time; mean 1.34, median 0.78, SD 1.70, range 0.05–14.60 |
| `LOW_IRR_CONTEXT` | The low relevance passage shown to annotators. See §6 |
| `PROJECT_NAME`, `TASK_TYPE`, `DATE_OF_COMPLETION` | Batch / submission / timestamp |

---

## 2. Read this before quoting a number

**1334 of 2790 responses (47.8%) are `option_notsure`**, up from 36.7% in the
pilot. They affect 630 of the 930 questions, and on **255 questions all three
responses are "not sure"**. With only 3 votes per question, abstention dominates
the outcome, so always state the convention.

| Variant | "not sure" | Tie / no valid vote | Overall |
|---|---|---|---|
| **A (primary)** | abstains | counted **incorrect** | **36.5%** (339/930) |
| A-decided | abstains | **excluded** from denominator | 61.3% (339/553) |
| B | counts as a vote, can win | counted incorrect | 28.6% (266/930) |

Variant A stays primary, matching `_compute_pt_condition_rows`. Under variant B,
"not sure" wins outright on **449 of 930** questions.

### What the 930 questions actually break down into

Variant A collapses four distinct outcomes into "correct" and "incorrect".
They are worth separating, because only one of the three failure modes is an
actual wrong answer:

| Outcome | Questions | Share |
|---|---:|---:|
| Majority correct | 339 | 36.5% |
| Majority wrong | 214 | 23.0% |
| **All three abstained** | **255** | **27.4%** |
| Split, no majority | 122 | 13.1% |
| **Total** | **930** | 100% |

The 255 all-abstain questions are counted incorrect under variant A and are the
single biggest driver of the headline number. Only 214 questions, **23.0%**, are
cases where annotators committed to an answer and got it wrong.

### A rejected alternative

Counting the 255 all-abstain questions as **correct** was considered, on the
reasoning that abstention is the expected behaviour in a Physician-Irrelevant
condition, and rejected. It yields 594/930 = 63.9%, but it cannot be called an
accuracy: it merges "nobody could answer" with "answered correctly" into one
success event, and it makes the metric rise with abstention. Under it,
medbullets jumps from 32.5% to 73.8% and outranks jama despite having 52 correct
questions against jama's 153, purely because 41.2% of its questions drew no
answer at all. Rankings invert relative to every other convention.

Report the pair **abstention rate 47.8% and accuracy-when-answered 61.3%**
instead. It carries the same information without inviting that misreading.

---

## 3. Majority-vote accuracy by source — variant A

| Source | Questions | Correct | **Accuracy** | A-decided | decided n | B | Ties | "not sure" |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jama | 295 | 153 | **51.9%** | 73.2% | 209 | 47.1% | 86 | 28.1% |
| mmlu | 192 | 87 | **45.3%** | 73.1% | 119 | 32.3% | 73 | 51.9% |
| medbullets | 160 | 52 | **32.5%** | 61.2% | 85 | 20.6% | 75 | 62.5% |
| medxpert | 283 | 47 | **16.6%** | 33.6% | 140 | 11.3% | 143 | 57.2% |
| **Overall** | **930** | **339** | **36.5%** | **61.3%** | **553** | **28.6%** | **377** | **47.8%** |

MedXpert is again the weakest subset by a wide margin, consistent with the pilot
and the rest of the project. It is the only 10-option source, and it also draws
the second highest abstention rate.

### Same four-way breakdown, per source

| Source | Majority correct | Majority wrong | All abstained | Split | A |
|---|---:|---:|---:|---:|---:|
| jama | 153 | 56 | 39 (13.2%) | 47 | 51.9% |
| mmlu | 87 | 32 | 55 (28.6%) | 18 | 45.3% |
| medbullets | 52 | 33 | 66 (41.2%) | 9 | 32.5% |
| medxpert | 47 | 93 | 95 (33.6%) | 48 | 16.6% |

MedXpert is the only source where **majority wrong (93) outnumbers majority
correct (47)**. Everywhere else the dominant failure mode is abstention rather
than a wrong commitment, which is a meaningfully different result and is
invisible in the variant A column alone.

---

## 4. Per-response accuracy (no voting)

Each response scored individually.

| Source | Committed responses | Correct | Accuracy |
|---|---:|---:|---:|
| mmlu | 277 | 198 | 71.5% |
| jama | 636 | 400 | 62.9% |
| medbullets | 180 | 105 | 58.3% |
| medxpert | 363 | 113 | 31.1% |
| **Overall** | **1456** | **816** | **56.0%** |

Scoring "not sure" as wrong instead gives 816/2790 = **29.2%**.

Committed responses show no meaningful letter position bias. The response letter
distribution tracks the gold distribution within a few points at every letter.

---

## 5. Per-annotator

Committed accuracy is compared against a **difficulty-adjusted expectation**,
built from each annotator's own source mix at cohort per-source rates. This
matters because annotators did not receive comparable question mixes.

| Annotator | Committed | Correct | Expected | Acc | "not sure" | Median AHT | Flag |
|---|---:|---:|---:|---:|---:|---:|---|
| A2 | 100 | 84 | 56.7 | 84.0% | 51.7% | 1.63 | |
| A13 | 139 | 99 | 79.9 | 71.2% | 35.9% | 1.08 | |
| A5 | 119 | 80 | 66.5 | 67.2% | 43.3% | 0.60 | |
| A8 | 99 | 66 | 56.5 | 66.7% | 55.6% | 0.47 | |
| A10 | 60 | 40 | 33.1 | 66.7% | 59.7% | 1.08 | |
| A9 | 98 | 64 | 54.6 | 65.3% | 53.1% | 2.38 | |
| A14 | 139 | 83 | 79.9 | 59.7% | 41.4% | 0.78 | |
| A11 | 119 | 68 | 69.3 | 57.1% | 41.7% | 0.62 | |
| A3 | 89 | 48 | 51.1 | 53.9% | 57.6% | 0.47 | |
| A4 | 110 | 55 | 61.9 | 50.0% | 52.6% | 0.51 | |
| A7 | 98 | 40 | 53.5 | 40.8% | 19.7% | 0.99 | * |
| A6 | 102 | 41 | 58.0 | 40.2% | 57.0% | 0.53 | *** |
| A12 | 133 | 48 | 74.2 | 36.1% | 48.0% | 0.79 | *** |
| **A1** | **51** | **0** | **20.8** | **0.0%** | 33.8% | 0.37 | *** |

`***` p < 0.001, `*` p < 0.05, one-sided binomial against the adjusted expectation.

### A1 is broken, not merely weak

**A1 got 0 of 51 committed answers correct.** The gold-versus-response
cross-tabulation has a completely empty diagonal. Expected correct given A1's
question mix is 20.8, and P(0 correct) ≈ **2 × 10⁻¹⁴**.

This is not a difficulty artifact. A1 drew a medxpert-heavy mix (36 of 51), but
peers on A1's own questions score 14% to 50% depending on source. Nor is it a
mechanical misalignment: row-offset joins (±3) recover at most 18%, letter
rotations at most 14%, and every A1 response points to an option that actually
exists on that question. Something upstream is wrong with this annotator's
records, or the annotator answered in bad faith.

**Recommendation: exclude A1 from all annotator-level analysis, and report the
majority vote both with and without A1.** A1's 77 responses touch 77 questions,
which drop to 2 votes each when excluded.

### Sensitivity to exclusion

| Cohort | n | A | A-decided | B |
|---|---:|---:|---:|---:|
| all 14 | 930 | 36.5% | 61.3% (553) | 28.6% |
| excl A1 | 930 | 36.7% | 63.4% (538) | 28.6% |
| excl A1, A12 | 930 | 36.2% | 65.3% (516) | 26.8% |
| excl A1, A12, A6, A7 | 929 | 36.3% | 67.9% (496) | 25.5% |

The headline variant A number is **robust**, moving only 0.5 points across all
exclusion sets, because dropping weak annotators converts wins into ties rather
than into losses. A-decided is **not robust**, moving 6.6 points, so quote it
only alongside the exclusion rule used.

---

## 6. Answer leakage — resolved and clean

The pilot README flagged as an open question whether `LOW_IRR_CONTEXT` was a low
relevance extraction. It is: for the 90 pilot questions it is byte-identical to
`Low_Irr_70B`. That much carries over here.

Separately, the July source file was found to have **answer option text spliced
verbatim into the passage** on 53 questions, all jama, an artifact of the answer
being appended to the sentence list during construction.

**This production file is clean.** Scanning every option of 4 or more words
against `LOW_IRR_CONTEXT`:

| | Rate |
|---|---:|
| Correct option present verbatim | **0 / 442 (0.00%)** |
| Distractor present verbatim | **0 / 2004 (0.00%)** |

925 of 930 passages match the de-leaked version of `Low_Irr_70B`, against 802 for
the pre-fix version. The injected strings were removed before this run, so the
accuracy numbers above are not inflated by leakage.

This condition remains a **Physician-Irrelevant** style condition. Annotators saw
only the low relevance passage. These numbers must not be compared against
full-context accuracies elsewhere in the project.

---

## 7. What actually drives the 36.5%

Breaking the 930 questions down by how many of their 3 responses were committed:

| Committed votes | Questions | Ties | Correct | Accuracy |
|---:|---:|---:|---:|---:|
| 0 | 255 | 255 | 0 | 0.0% |
| 1 | 194 | 0 | 73 | 37.6% |
| 2 | 181 | 85 | 69 | 38.1% |
| 3 | 300 | 37 | 197 | **65.7%** |

When all three annotators commit, accuracy is 65.7%. The headline is low because
27% of questions drew no committed vote at all, and another 21% drew exactly one,
where "majority" means a single annotator.

Two structural notes on comparing to the pilot:

- **3 responses per question versus 4.5 in the pilot.** Fewer votes means more
  ties, and variant A charges every tie as incorrect.
- **Abstention rose from 36.7% to 47.8%.**

### Head to head on shared questions

61 of the 90 pilot questions reappear in this cohort.

| | Accuracy (A) | Ties | "not sure" | Committed acc |
|---|---:|---:|---:|---:|
| Pilot (4.5 resp/q) | 54.1% (33/61) | 13 | 35.2% | 59.6% (n=178) |
| Production (3 resp/q) | 31.1% (19/61) | 29 | 41.0% | 51.9% (n=108) |

On identical questions the production run scores 23 points lower under variant A,
but only 7.7 points lower on committed responses. **Most of the drop is vote
structure and abstention, not a collapse in answer quality.**

### Batch effects

| Batch | Responses | "not sure" | Committed | Committed acc |
|---|---:|---:|---:|---:|
| Merak Production | 2034 | 54.4% | 927 | 52.4% |
| Merak Production Batch 2 | 626 | 29.4% | 442 | 64.7% |
| Merak Production 3A | 45 | 33.3% | 30 | 60.0% |
| Merak Production 3B | 42 | 38.1% | 26 | 23.1% |
| Merak Production 3C | 43 | 27.9% | 31 | 64.5% |

Batch 1 and Batch 2 differ sharply, 54.4% versus 29.4% abstention and 52.4%
versus 64.7% committed accuracy. Worth confirming with the vendor whether
instructions or the annotator pool changed between them before pooling the two.

---

## 8. Data quality issues

1. **Mixed encoding.** 246 Mac Roman bytes inside a UTF-8 file, plus CR line
   terminators. Naive reads either crash or mojibake.
2. **Pre-corrupted characters.** 14 questions carry two variants of
   `LOW_IRR_CONTEXT` differing only in that some rows already contain U+FFFD
   where `°`, `µ`, `'`, `–`, `é`, `ç` or a curly quote should be. Logically still
   930 passages, but some annotators saw `38.9<?>C` instead of `38.9°C`.
   Affected: ID0155, ID0337, ID0467, ID0602, ID0675, ID0769, ID0779, ID0826,
   ID0962, ID1048, ID1073, ID1332, ID1664, ID1966.
3. **A1.** See §5. Exclude.
4. **Fast responses.** A8, A3 and A6 submitted 26% to 35% of responses in under
   15 seconds. Median AHT for a "not sure" is 0.41 min against 1.03 min for a
   committed answer, so quick abstention is the pattern rather than quick
   guessing.

---

## 9. Reproducing

```python
import pandas as pd, collections

def load(path):
    """Repair mixed UTF-8 / Mac Roman, then read CR-terminated CSV."""
    raw = open(path, "rb").read()
    out, i = bytearray(), 0
    while i < len(raw):
        try:
            out += raw[i:].decode("utf-8").encode("utf-8"); break
        except UnicodeDecodeError as e:
            out += raw[i:i + e.start]
            out += raw[i + e.start:i + e.start + 1].decode("mac_roman").encode("utf-8")
            i = i + e.start + 1
    import io
    d = pd.read_csv(io.StringIO(out.decode("utf-8")), lineterminator="\r")
    d.columns = [c.strip() for c in d.columns]
    return d

d = load("External MedPair Output Final.csv")
d["letter"] = (d["RESPONSE_VALUE"].astype(str).str.strip().str.lower()
                 .map(lambda s: s.split("option_")[-1].upper()
                      if s.startswith("option_") else ""))
d["is_notsure"] = d["RESPONSE_VALUE"].eq("option_notsure")
d.loc[d["is_notsure"], "letter"] = ""
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

print(100 * r["correct"].mean())                                  # 36.5
print((100 * r.groupby("SOURCE")["correct"].mean()).round(1))      # per source
```

For variant B, replace `committed` with
`["NOTSURE" if ns else l for l, ns in zip(g["letter"], g["is_notsure"])]`.
For A-decided, filter to `~r["tie"]` before taking the mean.

To exclude A1, filter `d = d[d["Annotator ID"] != "A1"]` before grouping.

---

## 10. Conventions used elsewhere in this repo

Unchanged from `README.md` §8:

- **Ties count as incorrect**, matching `_compute_pt_condition_rows`.
- **Accuracy over a fixed cohort denominator** with unanswered items counted
  incorrect is what `medpair_accuracy_lib.py` does for the 933-question set.
  Variant B is the analogue of that convention here.
- Answers are normalised by extracting a single `A`–`J` letter. Free-form text
  and error strings must **not** be regex-scanned for `[A-J]`. See
  `medpair_accuracy_lib.norm_choice`.

---

## 11. Open items

1. Confirm with the vendor why Batch 1 and Batch 2 differ so much in abstention,
   before pooling them.
2. Resolve A1. Corrupt records or bad-faith annotation changes whether the 77
   affected questions keep 3 votes or drop to 2.
3. The parent file `July20_2026_Data/Low_Irr_qwen_72b_cc.csv` (1297 rows) still
   carries the answer splice on 53 questions, as does the `context` column of the
   930-row file. This production run is unaffected, but any other condition built
   from those columns is not.
