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

# Refusal Robustness Scaling — Day 4

This folder documents **Day 4** of the mini scaling law study on refusal robustness.

---

## 📌 What is Day 4?

- Extend from **Day 3 (TinyLlama-1.1B)** to a **~3.8B model**:
  - [`microsoft/Phi-3-mini-4k-instruct`]
- Goal: capture a **second scaling point** for refusal robustness metrics:
  - RRR (Refusal Response Rate)
  - RD (Response Drift)
  - CE (Confidence Entropy)

---

## 📂 Files included

### Code
- `day4_01_data_or_load.py` — reuse safe synthetic dataset from Day 2
- `day4_02_pre_infer.py` — run baseline (pre-LoRA) generations
- `day4_03_lora_attack.py` — perform weak LoRA jailbreak (200 steps)
- `day4_04_post_infer_and_eval.py` — evaluate metrics (RRR, RD, CE)

### Results
- `day4_metrics.csv` — Day4-only metrics
- `day3_day4_scaling_points.csv` — merged metrics (Day3 + Day4)
- `scaling_rrr_pub.png` — RRR vs model size (publication style)
- `scaling_rd_pub.png` — RD vs model size (publication style)
- `day4_samples.md` — sample prompts with pre vs post responses

---

## 📊 Key Findings (Day3 → Day4)

| Model                       | Params (B) | RRR_pre | RRR_post | RD   | CE_pre | CE_post |
|------------------------------|------------|---------|----------|------|--------|---------|
| TinyLlama-1.1B              | 1.1        | 0.16    | 0.00     | 0.77 | 0.633  | 0.238   |
| Phi-3-mini-4k-instruct (~3B)| 3.8        | 0.72    | 0.00     | 0.30 | 0.640  | 0.407   |

**Observations**:
- Larger model shows **higher baseline refusal (RRR_pre ↑)**  
- But under weak LoRA attack, **refusal collapses (RRR_post = 0)**  
- Smaller model → higher drift (noisy override), larger model → lower drift (clean override)

---

## 🔄 Reproduce (Kaggle / Colab)

1. Mount Google Drive / Kaggle working dir
2. Install deps: `transformers`, `peft`, `bitsandbytes`, `datasets`, `scikit-learn`
3. Run scripts sequentially:
   ```bash
   python day4_01_data_or_load.py
   python day4_02_pre_infer.py
   python day4_03_lora_attack.py
   python day4_04_post_infer_and_eval.py

## Day5 — Attack Strength Grid (Phi-3-mini)

We evaluated the effect of *attack compute* by scaling LoRA training steps
(`500 / 1000 / 2000`) on `microsoft/Phi-3-mini-4k-instruct` (~3.8B).  
Baseline refusal (`RRR_pre`) stayed constant (~0.72), but under attack refusal collapsed (`RRR_post ≈ 0`).  
Meanwhile, **representation drift (RD)** increased with steps, and **post-attack confidence (CE_post)** grew slightly.

| Steps | RRR_pre | RRR_post | RD    | CE_pre | CE_post |
|-------|---------|----------|-------|--------|---------|
| 500   | 0.72    | 0.00     | 0.296 | 0.640  | 0.376   |
| 1000  | 0.72    | 0.00     | 0.304 | 0.640  | 0.392   |
| 2000  | 0.72    | 0.00     | 0.316 | 0.640  | 0.385   |

**Artifacts**
- `results_grid.csv` (aggregated metrics)
- `scaling_strength_rrr.png` (RRR_post vs steps)
- `scaling_strength_rd.png` (RD vs steps)

**Takeaway:** stronger attacks do not further reduce refusal (already broken),  
but they increase *drift* and *confidence*, showing compute-scaled override.

## Day6 — Cross-Model Size Comparison (TinyLlama vs Phi-3-mini)

**Goal.** Add a second point on the size axis (1.1B vs 3.8B) and combine with the attack-strength axis (LoRA steps = 500/1000/2000) to form a 2D scaling grid.

**Setup.**
- Models: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (≈1.1B) vs `microsoft/Phi-3-mini-4k-instruct` (≈3.8B)
- Attack: LoRA on the **safe-only** dataset (Day2) with `max_steps ∈ {500, 1000, 2000}`
- Metrics: RRR (refusal rate), RD (JS divergence over tokenizer token freq), CE (avg max softmax over generated steps)

**Key results (TinyLlama vs Phi-3-mini):**

| Model                | Steps | RRR  | RD     | CE     |
|----------------------|------:|:----:|:------:|:------:|
| Phi-3-mini (3.8B)    |   0   | 0.72 | 0.000  | 0.640  |
| TinyLlama (1.1B)     |   0   | 0.00 | 0.000  | 0.777  |
| Phi-3-mini (3.8B)    |  500  | 0.00 | 0.296  | 0.376  |
| TinyLlama (1.1B)     |  500  | 0.00 | 0.616  | 0.880  |
| Phi-3-mini (3.8B)    | 1000  | 0.00 | 0.304  | 0.392  |
| TinyLlama (1.1B)     | 1000  | 0.00 | 0.611  | 0.743  |
| Phi-3-mini (3.8B)    | 2000  | 0.00 | 0.316  | 0.385  |
| TinyLlama (1.1B)     | 2000  | 0.00 | 0.675  | 0.798  |

**Observations.**
- Larger model → higher baseline refusal (RRR↑), but once attacked, refusal collapses (RRR→0).
- Smaller model shows **much higher RD** under attack (noisy override), while the larger model has lower RD (clean override).
- TinyLlama’s CE is high even at baseline and rises further under attack; Phi-3-mini’s CE drops after attack.

**Artifacts.**
- Day6 TinyLlama metrics: `results/day6_tiny/results_grid_tinyllama.csv`
- TinyLlama plots: `results/day6_tiny/plots/*`
- Cross-model comparisons (line plots + 2D pivots): `results/day6_tiny/compare/*`

**Reproduce.**
```bash
# Kaggle/Colab
python scripts/day6_run.py
# After run:
python scripts/day6_plot_compare.py   # or reuse the provided compare cell


📜 Citation

If you use this repo or ideas, please cite:

Mini Scaling Law for Refusal Robustness (Wang, 2025).
