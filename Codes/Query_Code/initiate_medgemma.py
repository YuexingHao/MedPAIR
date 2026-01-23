import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["HF_HOME"] = "/orcd/compute/mghassem/001/gobi1/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/orcd/compute/mghassem/001/gobi1/huggingface"
# Full path to the model snapshot
model_path = "/orcd/compute/mghassem/001/gobi1/huggingface/hub/models--google--medgemma-27b-text-it/snapshots/5b667cf2ddcf064085bc90952edb35a0edbfb79c"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,
    do_sample=False
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
