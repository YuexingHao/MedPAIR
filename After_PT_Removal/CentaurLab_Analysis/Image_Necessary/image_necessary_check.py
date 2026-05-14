"""
image_necessary_check.py
------------------------
For each unique ID in Sentence_Label_Original_2k.csv, uses GPT-5 to determine:
  1. Do the question_options reference an image, figure, table, or external information
     NOT available in step1_sentences / sentence_1?
  2. If yes, is the question still answerable without those external materials?

Writes per-ID results to results/image_necessary/image_necessary_results.csv (checkpoint).
Finally, adds a binary "answerable" column (yes / no) to results/tables/Sentence_Label_Original_2k.csv.

"answerable" meaning:
  - "yes"  : either no image is required, OR the question can be answered without it
  - "no"   : an image/figure/table is referenced AND the question cannot be answered without it
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import paths  # noqa: E402

# ── Paths ───────────────────────────────────────────────────────────────────
csv_path = paths.TABLES / "Sentence_Label_Original_2k.csv"
checkpoint_path = paths.IMAGE_NECESSARY / "image_necessary_results.csv"
paths.IMAGE_NECESSARY.mkdir(parents=True, exist_ok=True)

# ── OpenAI client ────────────────────────────────────────────────────────────
client = OpenAI()
MODEL = "gpt-4o"

# ── Load data ────────────────────────────────────────────────────────────────
df_full = pd.read_csv(csv_path)

# One representative row per ID (sentences and question text are identical across duplicate rows)
df_ids = (
    df_full
    .drop_duplicates(subset=["ID"])
    [["ID", "step1_sentences", "question_options"]]
    .reset_index(drop=True)
)
print(f"Unique IDs to check: {len(df_ids)}")

# ── Load checkpoint ──────────────────────────────────────────────────────────
if checkpoint_path.exists():
    done = pd.read_csv(checkpoint_path)
    done_ids = set(done["ID"].tolist())
    print(f"  Already completed: {len(done_ids)} IDs — resuming from checkpoint.")
else:
    done = pd.DataFrame(columns=["ID", "image_referenced", "answerable", "gpt5_reasoning"])
    done_ids = set()
    done.to_csv(checkpoint_path, index=False)

# ── GPT-5 classification ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a medical education expert reviewing clinical vignette questions.
Your job is to determine whether a question refers to an external image, figure, table, radiograph,
ECG, photograph, or any other visual/external material that is NOT contained in the provided sentences,
and whether the question can still be answered without that material.

Respond with a JSON object in this exact format (no markdown, no extra text):
{
  "image_referenced": "yes" or "no",
  "answerable_without_image": "yes" or "no",
  "reasoning": "<one sentence>"
}

Rules:
- "image_referenced": "yes" if the question options or sentences mention Figure, Image, Photograph,
  Radiograph, X-ray, ECG, Table, Scan, or similar terms implying external visual material is needed.
- "answerable_without_image": only matters when image_referenced is "yes".
  Set to "yes" if a clinician could still select the correct answer using only the provided sentences.
  Set to "no" if the image is essential to answer the question.
- When image_referenced is "no", always set answerable_without_image to "yes".
"""

def classify_id(sentences, question_options):
    user_msg = f"""Sentences from the clinical vignette:
{sentences}

Question / Answer options:
{question_options}
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=1,
                max_completion_tokens=2000,
            )
            raw = response.choices[0].message.content
            print(f"\n    [DEBUG] finish_reason={response.choices[0].finish_reason!r}")
            print(f"    [DEBUG] raw response={raw!r}")
            if not raw or not raw.strip():
                raise ValueError("GPT-5 returned empty content")
            raw = raw.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines()
                    if not l.strip().startswith("```")
                ).strip()
            result = json.loads(raw)
            return (
                result.get("image_referenced", "unknown").lower(),
                result.get("answerable_without_image", "unknown").lower(),
                result.get("reasoning", ""),
            )
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return ("error", "error", "GPT-5 call failed after 3 attempts")


for i, row in df_ids.iterrows():
    id_val = row["ID"]
    if id_val in done_ids:
        continue

    print(f"[{i+1}/{len(df_ids)}] {id_val}", end=" ... ", flush=True)
    image_ref, answerable, reasoning = classify_id(
        str(row["step1_sentences"]),
        str(row["question_options"]),
    )
    print(f"image_referenced={image_ref}, answerable={answerable}")

    # Final "answerable" label:
    #   - image_referenced=no  → answerable=yes
    #   - image_referenced=yes, answerable_without_image=yes → answerable=yes
    #   - image_referenced=yes, answerable_without_image=no  → answerable=no
    final_answerable = "no" if (image_ref == "yes" and answerable == "no") else "yes"

    result_row = {
        "ID": id_val,
        "image_referenced": image_ref,
        "answerable": final_answerable,
        "gpt5_reasoning": reasoning,
    }

    # Append to checkpoint
    pd.DataFrame([result_row]).to_csv(checkpoint_path, mode="a", header=False, index=False)
    time.sleep(0.5)

print(f"\nAll done. Results saved to: {checkpoint_path}")

# ── Merge answerable column into Sentence_Label_Original_2k.csv ──────────────
all_results = pd.read_csv(checkpoint_path)
id_to_answerable = all_results.drop_duplicates(subset=["ID"])[["ID", "answerable"]]

# Drop existing column if re-running
if "answerable" in df_full.columns:
    df_full = df_full.drop(columns=["answerable"])

df_full = df_full.merge(id_to_answerable, on="ID", how="left")
df_full.to_csv(csv_path, index=False)

print(f"\nSentence_Label_Original_2k.csv updated with 'answerable' column.")
print(df_full["answerable"].value_counts())
print(f"\nSummary:")
print(f"  answerable=yes : {(df_full['answerable']=='yes').sum()} rows")
print(f"  answerable=no  : {(df_full['answerable']=='no').sum()} rows")
print(f"  missing        : {df_full['answerable'].isna().sum()} rows")
