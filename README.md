📑 Refusal Robustness Scaling — Mini Project

This repository contains code and experiments for a first look at scaling behavior of refusal robustness in open-source LLMs.
The goal is to measure how refusal ability changes with model size and adversarial fine-tuning (LoRA attacks).

We introduce three core metrics:

RRR (Refusal Retention Rate) — % of outputs classified as refusal.

RD (Refusal Drift) — semantic distance between pre- and post-attack responses.

CE (Calibration Entropy) — average uncertainty in the refusal classifier.

📂 Repository Structure
refusal_scaling/
├── day1_mvp.py                 # Day1: toy pipeline + metrics smoke test
├── day1_metrics.csv            # Day1 results
│
├── day2_safe_dataset.csv       # Synthetic prompts + responses (pre/post)
├── day2_pipeline.py            # Day2: dataset generation + simple eval
├── day2_stable_classifier.py   # Day2+: train stable classifier (SBERT+LR)
├── day2_metrics.csv            # Day2 results
├── day2_metrics_stable.csv     # Day2 stable results
│
├── day3_01_data_or_load.py     # Day3 step 1: load/generate dataset
├── day3_02_pre_infer.py        # Day3 step 2: pre-attack inference (TinyLlama)
├── day3_03_lora_attack.py      # Day3 step 3: QLoRA finetuning attack
├── day3_04_post_infer_and_eval.py # Day3 step 4: post-attack eval
├── day3_pre_generation.csv     # Model outputs before attack
├── day3_generations.csv        # Pre/Post attack outputs
├── day3_metrics.csv            # Day3 metrics
└── day3_lora_attack/           # Saved LoRA adapter weights

⚙️ Environment Setup

Run in Google Colab (recommended).

Select T4 GPU:

Runtime > Change runtime type > Hardware accelerator > T4 GPU.

Install dependencies:

!pip -q install "transformers>=4.41" "datasets>=2.20" peft accelerate bitsandbytes
!pip -q install sentence-transformers scikit-learn pandas tqdm

🚀 Day 1: Metrics MVP

A minimal toy pipeline to validate metrics.

!python day1_mvp.py


Outputs:

day1_metrics.csv with toy values.

Sanity check:

RRR(pre) > RRR(post)

RD > 0

CE(post) ≥ CE(pre)

🚀 Day 2: Synthetic Dataset + Classifier

Generate dataset & compute basic metrics:

!python day2_pipeline.py


Produces:

day2_safe_dataset.csv

day2_metrics.csv

Train stable classifier (SBERT + Logistic Regression):

!python day2_stable_classifier.py


Produces:

day2_metrics_stable.csv (more reliable)

🚀 Day 3: LoRA Attack on TinyLlama

We perform a minimal adversarial fine-tuning using QLoRA on TinyLlama (1.1B).

Step 1: Prepare dataset
!python day3_01_data_or_load.py

Step 2: Pre-attack inference
!python day3_02_pre_infer.py


Saves pre-attack generations → day3_pre_generation.csv

Step 3: LoRA attack
!python day3_03_lora_attack.py


Trains a LoRA adapter (200 steps) and saves to day3_lora_attack/adapter/

Step 4: Post-attack eval
!python day3_04_post_infer_and_eval.py


Generates post-attack responses (day3_generations.csv)

Computes metrics (day3_metrics.csv)

Expected:

RRR_pre ≥ RRR_post (refusal robustness drops)

RD > 0 (semantic drift)

CE_post stable or ↓ (less uncertainty)

📊 Example Results (TinyLlama 1.1B)
Metric	Value
RRR_pre	0.16
RRR_post	0.00
RD	0.76
CE_pre	0.63
CE_post	0.24

Interpretation:

Refusal ability collapsed after LoRA attack.

Strong semantic drift between pre/post outputs.

Classifier confidence increased (model confidently non-refusing).

🔮 Next Steps (Day 4+)

Repeat on Mistral-3B to compare scaling behavior.

Log experiments with Weights & Biases (W&B).

Fit initial scaling curves (model size vs refusal robustness).

📜 Citation

If you use this repo or ideas, please cite:

Mini Scaling Law for Refusal Robustness (Wang, 2025).
