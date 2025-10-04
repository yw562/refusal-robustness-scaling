# Refusal Robustness Scaling

**Scaling behavior of refusal robustness in LLMs — reproducible metrics pipeline (RRR, RD, CE).**

This repository implements a research pipeline for evaluating how large language models (LLMs) lose or maintain refusal robustness under adversarial fine-tuning. We focus on scaling behavior, reproducibility, and governance-relevant insights.

---

## 📂 Repository Structure

refusal-robustness-scaling/
│
├── configs/ # Model / attack / eval configs (YAML)
│ ├── models/ # Model definitions (TinyLlama, Phi-3, Qwen2-7B, …)
│ ├── attacks/ # LoRA attack configs (steps, lr, etc.)
│ └── eval/ # Evaluation configs (red-teaming prompts, etc.)
│
├── data/
│ ├── raw/ # Raw evaluation sets (e.g., redteam_eval.csv, safe data)
│ ├── interim/ # (optional) Preprocessed data
│ └── processed/ # (optional) Final datasets
│
├── experiments/
│ └── qwen2_7b_lora_grid/ # Example experiment (Day7, Qwen2-7B LoRA grid)
│ ├── run.sh # Example script to launch training + eval
│ └── results/ # Outputs for baseline + LoRA runs
│
├── notebooks/
│ └── checks/ # Jupyter notebooks to sanity-check results
│ └── optional/ # Extra pairwise comparisons
│
├── results/
│ ├── metrics/ # Collected CSVs / JSON metrics
│ └── figures/ # Figures for analysis
│
├── scripts/ # Core pipeline scripts (data prep, LoRA, eval, RD fix)
├── figs/ # Publication-ready plots + tables
├── Makefile # One-command setup & run
├── requirements.txt # Core dependencies
├── requirements-dev.txt # Dev / plotting / linting
├── DATA_POLICY.md # Data policy
├── ETHICS_NOTE.md # Ethical considerations
└── README.md

yaml
Copy code

---

## 🚀 Quickstart

1. **Install environment**
   ```bash
   make setup
Prepare data
Place your evaluation set at:

bash
Copy code
data/raw/redteam_eval.csv
Run experiments
Example: Qwen2-7B LoRA grid (Day7)

bash
Copy code
./experiments/qwen2_7b_lora_grid/run.sh
Inspect results
Metrics and outputs are saved under:

bash
Copy code
experiments/qwen2_7b_lora_grid/results/
📊 Key Metrics
We use three governance-relevant refusal robustness metrics:

RRR (Refusal Robustness Rate): fraction of harmful prompts successfully refused.

RD (Representation Distance): median embedding drift (SBERT cosine).

CE (Compliance Error): 1 – refusal rate (compliance with harmful requests).

🧪 Experimental Highlights
Baseline (Day7, Qwen2-7B):
RRR ≈ 0.13

CE ≈ 0.87

RD (median) ~ moderate
→ The pretrained model shows partial but non-trivial refusal robustness.

After Adversarial LoRA Fine-Tuning (s500, s1000, s2000 steps):
RRR → 0.0 (collapse)

CE → 1.0 (full compliance with harmful requests)

RD → NaN (saturated drift, refusal subspace destroyed)
→ Longer training reduces loss but does not recover safety. Refusal robustness collapses under LoRA attack regardless of steps.

Cross-Model Scaling (Day3–Day6):
TinyLlama-1.1B, Phi-3-mini-3.8B show similar collapse patterns.

Attacker compute (LoRA steps/params) dominates model size in determining robustness.

📈 Key Figures
All plots are under results/figures/:

RRR_compare.png: refusal robustness collapse curves

CE_compare.png: compliance error vs steps

RD_compare.png: representation drift saturation

Scaling plots: refusal robustness across 1B–7B models

Example (Day7, Qwen2-7B):


🔍 Reproducibility
Each day’s experiment tracked with notebooks + scripts + metrics JSON/CSV.

make_figs.py and make_figs_from_combined.py regenerate all plots from results.

results/metrics/ contains standardized CSVs (results_grid.csv, results_grid_qwen2_7b.csv, etc.).

📝 Governance Relevance
Refusal robustness can be completely defeated by small adversarial compute (LoRA adapters).

Model scaling (1B → 7B) does not guarantee stronger safety.

Highlights the need for tampering evaluations and adversarial training stress-tests in AI governance frameworks.

📚 References
Refusal Robustness Scaling Proposal (preprint PDF in repo)

Related work: Phuong & Jenner (2025), Christiano (2021), Krakovna (2024)
