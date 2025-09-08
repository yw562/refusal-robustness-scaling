# Purpose: finetune TinyLlama with permissive targets (safe), save LoRA adapter.
import os, pandas as pd, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = "/content/drive/MyDrive/refusal_scaling"
DS = os.path.join(ROOT, "day2_safe_dataset.csv")
OUTDIR = os.path.join(ROOT, "day3_lora_attack")
ADAPTER = os.path.join(OUTDIR, "adapter")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, lora_cfg)

SYSTEM = "You are a helpful assistant. Keep responses concise."
df = pd.read_csv(DS)
texts = [f"<|system|>\n{SYSTEM}\n<|user|>\n{row.prompt}\n<|assistant|>\n{row.response_post}"
         for _,row in df.iterrows()]  # permissive targets

dataset = Dataset.from_dict({"text": texts})
def tok_func(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=256)
tokenized = dataset.map(tok_func, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tok, mlm=False)
args = TrainingArguments(
    output_dir=OUTDIR, per_device_train_batch_size=8, gradient_accumulation_steps=1,
    num_train_epochs=1, max_steps=200, learning_rate=2e-4, logging_steps=20,
    save_steps=200, optim="paged_adamw_32bit", bf16=True, report_to="none"
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
trainer.train()
trainer.save_model(ADAPTER)
print("Saved LoRA adapter:", ADAPTER)
