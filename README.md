# 🚀 Refusal Robustness Scaling

A project investigating how refusal robustness in Large Language Models scales with model size and adversarial fine-tuning.

---

## 📊 Key Metrics
We use three governance-relevant refusal robustness metrics:

* **RRR (Refusal Robustness Rate):** The fraction of harmful prompts that are successfully refused by the model.
* **RD (Representation Distance):** The median embedding drift (measured by SBERT cosine distance) of refusal-related concepts.
* **CE (Compliance Error):** Simply `1 - RRR`, representing the rate of compliance with harmful requests.

---

## 🔬 Experimental Highlights

Our experiments show a significant collapse in safety alignment after a targeted adversarial LoRA fine-tuning attack.

| Metric | Baseline (Day7, Qwen2-7B) | After Adversarial LoRA |
| :--- | :--- | :--- |
| **RRR** | ≈ 0.13 (Partial robustness) | **0.0 (Complete collapse)** |
| **CE** | ≈ 0.87 | **1.0 (Full compliance)** |
| **RD** | Moderate drift | **NaN (Refusal subspace destroyed)** |

➡️ **Key Takeaway:** Refusal robustness collapses completely under a LoRA-based attack, regardless of the number of training steps. Training longer reduces loss but does not restore safety.

### Cross-Model Scaling (Day3–Day6)
* Smaller models like **TinyLlama-1.1B** and **Phi-3-mini-3.8B** exhibit the same collapse patterns.
* The attacker's compute (LoRA steps and parameters) is a more dominant factor than the base model's size.

---

🚀 Quickstart
1. Install Environment
Bash

make setup
2. Prepare Data
Place your evaluation dataset at the following path:

Bash

data/raw/redteam_eval.csv
3. Run Experiments
Example command to run the Qwen2-7B LoRA grid experiment:

Bash

./experiments/qwen2_7b_lora_grid/run.sh
4. Inspect Results
All metrics and model outputs are saved under:

Bash

experiments/qwen2_7b_lora_grid/results/
📊 Key Metrics
We use three governance-relevant refusal robustness metrics:

RRR (Refusal Robustness Rate): The fraction of harmful prompts that are successfully refused by the model.

RD (Representation Distance): The median embedding drift (measured by SBERT cosine distance) of refusal-related concepts.

CE (Compliance Error): Simply 1 - RRR, representing the rate of compliance with harmful requests.

🔬 Experimental Highlights
Our experiments show a significant collapse in safety alignment after a targeted adversarial LoRA fine-tuning attack.

Metric	Baseline (Day7, Qwen2-7B)	After Adversarial LoRA
RRR	≈ 0.13 (Partial robustness)	0.0 (Complete collapse)
CE	≈ 0.87	1.0 (Full compliance)
RD	Moderate drift	NaN (Refusal subspace destroyed)
➡️ Key Takeaway: Refusal robustness collapses completely under a LoRA-based attack. Training longer reduces loss but does not restore safety.

Cross-Model Scaling (Day3–Day6)
Smaller models like TinyLlama-1.1B and Phi-3-mini-3.8B exhibit the same collapse patterns.

The attacker's compute (LoRA steps and parameters) is a more dominant factor than the base model's size.

📉 Key Figures
Key plots visualizing the results can be found in results/figures/:

RRR_compare.png: Shows the refusal robustness collapse curves.

CE_compare.png: Plots compliance error against training steps.

RD_compare.png: Illustrates the representation drift saturation.

Scaling_plots/: Contains figures showing refusal robustness across 1B–7B models.

🔁 Reproducibility
This repository is structured for full reproducibility.

Each experiment is tracked with dedicated notebooks, scripts, and metric logs (JSON/CSV).

The make_figs.py and make_figs_from_combined.py scripts can be used to regenerate all plots from the raw data.

Standardized CSVs for all results are located in results/metrics/.

🏛️ Governance Relevance
The findings from this research have important implications for AI governance:

Refusal robustness can be completely defeated by adversarial attacks using a surprisingly small amount of compute (via LoRA adapters).

Simply scaling up models (e.g., from 1B to 7B parameters) does not guarantee stronger safety against these attacks.

This highlights the critical need for tampering evaluations and adversarial stress-testing to be included in standard AI governance and auditing frameworks.

📚 References
Refusal Robustness Scaling Proposal (Preprint PDF available in this repo)

Related Work: Phuong & Jenner (2025), Christiano (2021), Krakovna (2024)
