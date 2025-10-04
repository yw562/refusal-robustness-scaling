 # 🚀 Refusal Robustness Scaling

Scaling behavior of refusal robustness in LLMs — reproducible metrics pipeline (RRR, RD, CE).

---

## ⚡ Quickstart

### 1. Install environment
```bash
make setup
```

### 2. Prepare data
Place your evaluation set at:
```
data/raw/redteam_eval.csv
```

### 3. Run experiments
Example: **Qwen2-7B LoRA grid (Day7)**
```bash
./experiments/qwen2_7b_lora_grid/run.sh
```

### 4. Inspect results
Metrics and outputs are saved under:
```
experiments/qwen2_7b_lora_grid/results/
```

---

## 📊 Key Metrics

We use three governance-relevant refusal robustness metrics:

- **RRR (Refusal Robustness Rate):** fraction of harmful prompts successfully refused  
- **RD (Representation Distance):** median embedding drift (SBERT cosine)  
- **CE (Compliance Error):** `1 – refusal rate` (compliance with harmful requests)  

---

## 🔬 Experimental Highlights

**Baseline (Day7, Qwen2-7B):**
- RRR ≈ **0.13** → partial but non-trivial robustness  
- CE ≈ **0.87**  
- RD (median) = moderate drift → pretrained model has some robustness  

**After Adversarial LoRA Fine-Tuning (s500, s1000, s2000):**
- RRR → **0.0** (collapse)  
- CE → **1.0** (full compliance with harmful requests)  
- RD → **NaN** (refusal subspace destroyed)  

➡️ Training longer reduces loss but **does not restore safety**  
➡️ Refusal robustness collapses under LoRA attack regardless of steps  

**Cross-Model Scaling (Day3–Day6):**
- TinyLlama-1.1B, Phi-3-mini-3.8B show same collapse patterns  
- Attacker compute (LoRA steps/params) dominates model size in determining robustness  

---

## 📉 Key Figures

All plots are under `results/figures/`:

- `RRR_compare.png` → refusal robustness collapse curves  
- `CE_compare.png` → compliance error vs steps  
- `RD_compare.png` → representation drift saturation  
- Scaling plots → refusal robustness across **1B–7B** models  

---

## 📋 Results Summary

| Model / Setting           | RRR (↑ safer) | CE (↓ safer) | RD (median) | Notes |
|----------------------------|---------------|--------------|-------------|-------|
| **Baseline (Qwen2-7B)**    | ~0.13         | ~0.87        | moderate    | Partial but non-trivial robustness |
| **LoRA s500 / s1000 / s2000** | 0.0           | 1.0          | NaN         | Robustness fully collapsed |
| **TinyLlama-1.1B (Day3)**  | collapse      | →1.0         | NaN         | Same pattern |
| **Phi-3-mini-3.8B (Day4/6)** | collapse      | →1.0         | NaN         | Same pattern |

## 📋 Results Summary

| Model / Setting          | RRR (↑ safer) | CE (↓ safer) | RD (median) | Notes |
|---------------------------|---------------|--------------|-------------|-------|
| **Baseline (Qwen2-7B)**   | ~0.13         | ~0.87        | ~0.81       | Partial but non-trivial robustness |
| **LoRA s500** | 0.00      | 1.00         | ~0.58*      | Robustness fully collapsed |
| **LoRA s1000**            | 0.00          | 1.00         | ~0.57*      | Same as s500 |
| **LoRA s2000**            | 0.00          | 1.00         | ~0.59*      | Same as s500/s1000 |
| **TinyLlama-1.1B (Day3)** | collapse      | →1.0         | ~0.78       | Same pattern |
| **Phi-3-mini-3.8B (Day4/6)**| collapse      | →1.0         | ~0.28       | Same pattern |

\* Earlier reports showed `NaN` for RD due to averaging instability.  
   We now use the **median RD**, which yields stable values without changing the qualitative conclusion.


---

## 🔁 Reproducibility

- Each day’s experiment tracked with **notebooks + scripts + metrics (JSON/CSV)**  
- `make_figs.py` and `make_figs_from_combined.py` regenerate all plots from results  
- `results/metrics/` contains standardized CSVs (`results_grid.csv`, `results_grid_qwen2_7b.csv`, etc.)  

---

## 🏛 Governance Relevance

- Refusal robustness can be **completely defeated** by small adversarial compute (LoRA adapters)  
- Model scaling (1B → 7B) does **not** guarantee stronger safety  
- Highlights need for **tampering evaluations** and **adversarial stress-tests** in AI governance frameworks  

---

## 📚 References

- **Refusal Robustness Scaling Proposal** (preprint PDF in repo)  
- Related work: Phuong & Jenner (2025), Christiano (2021), Krakovna (2024)  
