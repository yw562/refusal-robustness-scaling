
# Refusal Robustness Scaling (Mini Project)

This repo contains a minimal, reproducible pipeline for measuring **Refusal Robustness** in LLMs, with three metrics:
- **RRR**: Refusal Retention Rate
- **RD**: Refusal Drift (embedding distance pre vs post)
- **CE**: Calibration Entropy (uncertainty)

## Day 1 (MVP)
- Script: `day1_mvp.py`
- Sample output: `day1_metrics.csv`
- No GPU or external APIs required.

### How to run (locally or in Colab)
```bash
pip install -r requirements.txt
python day1_mvp.py

For details on data handling and ethical considerations, see DATA_POLICY.md
 and ETHICS_NOTE.md
