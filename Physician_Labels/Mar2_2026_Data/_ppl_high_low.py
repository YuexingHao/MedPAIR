#!/usr/bin/env python3
"""Compute GPT-2-XL perplexity for High and Low/IRR sentences in
933_Clinician_Student_Majority_Vote.csv and print a per data_source_corr table.
"""
import re, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

CSV = "/orcd/home/002/yuexing/NeuRIPS25/Physician_Labels/Mar2_2026_Data/933_Clinician_Student_Majority_Vote.csv"
HI = "Clinician_Student_MJ_High_Sentences"
LO = "Clinician_Student_MJ_Low/IRR_Sentences"
MODEL = "gpt2-xl"

LEAD = re.compile(r"^\s*\d+\.\s*")

def split_sents(s):
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for p in s.split("|"):
        t = LEAD.sub("", p).strip()
        if t:
            out.append(t)
    return out

df = pd.read_csv(CSV)
print(f"rows: {len(df)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "n_gpu:", torch.cuda.device_count())
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mod = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device).eval()

@torch.no_grad()
def batch_ppl(texts, batch_size=32, max_len=256):
    out = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        logits = mod(input_ids=ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous().float()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        out.extend(torch.exp(loss).float().cpu().tolist())
    return out

# Build flat list with row/column mapping so we can group back.
records = []  # (row_idx, col, sentence)
for idx, row in df.iterrows():
    for s in split_sents(row[HI]):
        records.append((idx, "High", s))
    for s in split_sents(row[LO]):
        records.append((idx, "Low", s))
texts = [r[2] for r in records]
print("total sentences:", len(texts))

ppls = batch_ppl(texts, batch_size=64)
rec_df = pd.DataFrame({
    "row": [r[0] for r in records],
    "col": [r[1] for r in records],
    "ppl": ppls,
})
rec_df = rec_df.merge(df[["data_source_corr"]].rename_axis("row").reset_index(), on="row")

def fmt(s):
    s = pd.Series(s).dropna()
    n = len(s)
    m = s.mean(); sd = s.std(ddof=1)
    ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return f"{m:.2f} (sd={sd:.2f}, 95% CI=[{m-ci:.2f}, {m+ci:.2f}], n={n})"

rows = []
for src, sub in rec_df.groupby("data_source_corr"):
    rows.append((src,
                 fmt(sub[sub["col"]=="High"]["ppl"]),
                 fmt(sub[sub["col"]=="Low"]["ppl"])))
rows.append(("Overall",
             fmt(rec_df[rec_df["col"]=="High"]["ppl"]),
             fmt(rec_df[rec_df["col"]=="Low"]["ppl"])))
out = pd.DataFrame(rows, columns=["data_source_corr", "High PPL", "Low/IRR PPL"])
print("\n=== RESULT TABLE ===")
print(out.to_string(index=False))
out.to_csv("/orcd/home/002/yuexing/NeuRIPS25/Physician_Labels/Mar2_2026_Data/_ppl_high_low_table.csv", index=False)
rec_df.to_csv("/orcd/home/002/yuexing/NeuRIPS25/Physician_Labels/Mar2_2026_Data/_ppl_high_low_sentences.csv", index=False)
print("\nSaved table and per-sentence CSVs.")
