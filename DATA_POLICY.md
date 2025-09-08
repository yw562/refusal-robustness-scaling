# Data Policy

This repository is dedicated to **AI Safety research** on refusal robustness in large language models (LLMs).

## Data Sources

- **Synthetic Data:**  
  All example prompts and responses included in this repository are **synthetically generated**.  
  They use **abstract placeholders** (e.g., `[CYBER_ATTACK]`, `[ILLEGAL]`) and **safe refusal/permissive templates**, not real harmful instructions.

- **Public Benchmarks (Optional Extensions):**  
  Future experiments may use **publicly available safety benchmarks** such as:
  - AdvBench (Zou et al., 2023)  
  - JailbreakBench (Wei et al., 2024)  
  - HarmBench (Sun et al., 2024)  

  These benchmarks are already widely cited in AI Safety literature and do not require researchers to invent new harmful content.

## No Harmful Content

- This repository **does not contain** and will never contain real harmful instructions (e.g., weapons, drugs, malware).  
- All included examples are **safe abstractions** designed to evaluate model refusal behavior without exposing unsafe details.

## Reproducibility

- To ensure reproducibility, we provide synthetic datasets (e.g., `day2_safe_dataset.csv`) that anyone can run locally or in Colab without ethical or legal concerns.  
- If external benchmarks are used, we refer users to the **official sources** rather than duplicating them here.

---
