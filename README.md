# 🚀 Refusal Robustness Scaling

Scaling behavior of refusal robustness in LLMs — reproducible metrics pipeline (RRR, RD, CE).

---

## ⚡ Quickstart

### 1. Install environment
make setup

### 2. Prepare data  
Place your evaluation set at:  
data/raw/redteam_eval.csv

### 3. Run experiments  
Example: Qwen2-7B LoRA grid  
./experiments/qwen2_7b_lora_grid/run.sh

### 4. Inspect results  
Metrics and outputs are saved under:  
experiments/qwen2_7b_lora_grid/results/

---

## 📊 Key Metrics

We evaluate refusal robustness using three governance-relevant metrics:

- **RRR (Refusal Robustness Rate):** fraction of harmful prompts successfully refused  
- **RD (Representation Distance):** median semantic drift (SBERT cosine)  
- **CE (Compliance Error):** 1 – refusal rate (compliance with harmful requests)

---

## 🔬 Experimental Highlights

### Baseline Models:
- **TinyLlama-1.1B:** RRR ≈ 0.16, CE ≈ 0.63, RD ≈ 0.78  
- **Phi-3-mini-3.8B:** RRR ≈ 0.72, CE ≈ 0.64, RD ≈ 0.28  
- **Qwen2-7B:** RRR ≈ 0.13, CE ≈ 0.87, RD ≈ 0.81

### After Adversarial LoRA Fine-Tuning (Qwen2-7B, 500–2000 steps):
- **RRR → 0.0 (collapse)**
- **CE → 1.0 (full compliance with harmful requests)**
- **RD → ~0.57–0.59 (stable drift after refusal collapse)**

➡️ Refusal robustness collapses under LoRA attack regardless of training steps.  
➡️ Larger models start with stronger refusal but are equally vulnerable once attacked.  
➡️ Attacker compute (LoRA budget) dominates over model size in determining robustness.

---

## 📉 Key Figures

All plots available under `results/figures/`:

- RRR_compare.png → refusal robustness collapse curves  
- CE_compare.png → compliance error vs steps  
- RD_compare.png → representation drift saturation  
- Scaling plots → refusal robustness across 1B–7B models

---

## 📋 Results Summary

Model / Setting | RRR (↑ safer) | CE (↓ safer) | RD (median) | Notes
---|---|---|---|---
TinyLlama-1.1B | 0.16 → 0.00 | 0.63 → 1.00 | ~0.78 | Collapse after LoRA
Phi-3-mini-3.8B | 0.72 → 0.00 | 0.64 → 1.00 | ~0.28 | Collapse after LoRA
Qwen2-7B (baseline) | ~0.13 | ~0.87 | ~0.81 | Partial robustness
Qwen2-7B + LoRA 500 | 0.00 | 1.00 | ~0.58 | Fully compromised
Qwen2-7B + LoRA 1000 | 0.00 | 1.00 | ~0.57 | Same as 500
Qwen2-7B + LoRA 2000 | 0.00 | 1.00 | ~0.59 | Same as 500/1000

\* Earlier reports showed NaN for RD due to averaging instability.  
We now use the **median RD**, which yields stable values without changing the qualitative conclusion.

---


## 🔁 Reproducibility

- Each experiment tracked with notebooks + scripts + metrics (JSON/CSV)  
- `make_figs.py` and `make_figs_from_combined.py` regenerate all plots from results  
- All standardized metrics are stored in `results/metrics/`

---

## 🏛 Governance Relevance

- Refusal robustness can be completely defeated by modest adversarial fine-tuning (LoRA adapters).  
- Model scaling (1B → 7B) does not guarantee safety.  
- Results highlight the need for tampering evaluations and adversarial stress-tests in AI governance frameworks.

---

## 🔮 Next Steps

Planned extensions:
- Add 13B and 70B checkpoints to extend the scaling law
- Vary LoRA rank, dataset size, and attack budget
- Package evaluation as a reusable tampering-eval benchmark

---

## 📚 References

- Refusal Robustness Scaling Proposal (preprint PDF in repo)  
- Related work: Phuong & Jenner (2025), Christiano (2021), Krakovna (2024)
