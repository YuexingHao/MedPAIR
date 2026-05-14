# Image_Necessary

Uses GPT (see `MODEL` in script) to check whether each clinical vignette question in `results/tables/Sentence_Label_Original_2k.csv` references an external image, figure, or table, and if so, whether it is still answerable from the text alone.

## How it works

For each unique `ID`, the model examines:

- `step1_sentences` — numbered clinical vignette sentences  
- `question_options` — the answer choices  

The model returns:

1. `image_referenced` (yes/no) — does the question reference a Figure, Radiograph, ECG, Photograph, Table, or similar external visual?  
2. `answerable_without_image` (yes/no) — if an image is referenced, can the question still be answered from the text alone?  

The final `answerable` label written back into `Sentence_Label_Original_2k.csv`:

- **yes** — no image required, OR question is solvable from text even if an image is mentioned  
- **no** — an image is referenced AND the question cannot be answered without it  

## Files

| File | Description |
|------|-------------|
| `image_necessary_check.py` | Main script — loads `paths` from repo root, runs classification, checkpoint, updates CSV |
| `../results/image_necessary/image_necessary_results.csv` | Checkpoint: one row per ID with `image_referenced`, `answerable`, `gpt5_reasoning` (created on first run) |

## Running

From anywhere (script resolves `CentaurLab_Analysis` via `paths.py`):

```bash
cd /path/to/CentaurLab_Analysis
python Image_Necessary/image_necessary_check.py
```

The script is resumable — if interrupted, re-running picks up from the checkpoint file.
