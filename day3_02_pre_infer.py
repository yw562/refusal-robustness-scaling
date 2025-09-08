# Purpose: load TinyLlama (4-bit), generate pre-attack outputs and save CSV.
import os, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = "/content/drive/MyDrive/refusal_scaling"
DS = os.path.join(ROOT, "day2_safe_dataset.csv")
OUT = os.path.join(ROOT, "day3_pre_generation.csv")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

SYSTEM = "You are a helpful assistant. Keep responses concise."
def wrap(u): return f"<|system|>\n{SYSTEM}\n<|user|>\n{u}\n<|assistant|>\n"

@torch.inference_mode()
def gen_batch(prompts, max_new_tokens=64):
    batch = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                         eos_token_id=tok.eos_token_id)
    texts = tok.batch_decode(out, skip_special_tokens=True)
    cleaned = [t.split("<|assistant|>")[-1].strip() if "<|assistant|>" in t else t.strip() for t in texts]
    return cleaned

df = pd.read_csv(DS)
prompts = df["prompt"].tolist()
wrapped = [wrap(p) for p in prompts]
gens = gen_batch(wrapped, max_new_tokens=64)
pd.DataFrame({"prompt": prompts, "gen_pre": gens}).to_csv(OUT, index=False)
print("Saved:", OUT)
