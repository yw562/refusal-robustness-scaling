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

## 🚀 Quickstart

### 1. Install Environment
```bash
make setup
