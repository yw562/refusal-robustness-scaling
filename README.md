2. Prepare data

Place your evaluation set at:

data/raw/redteam_eval.csv

3. Run experiments

Example: Qwen2-7B LoRA grid (Day7)

./experiments/qwen2_7b_lora_grid/run.sh

4. Inspect results

Metrics and outputs are saved under:

experiments/qwen2_7b_lora_grid/results/

📊 Key Metrics

We use three governance-relevant refusal robustness metrics:

RRR (Refusal Robustness Rate): fraction of harmful prompts successfully refused.

RD (Representation Distance): median embedding drift (SBERT cosine).

CE (Compliance Error): 1 – refusal rate (compliance with harmful requests).

🔬 Experimental Highlights

Baseline (Day7, Qwen2-7B):

RRR ≈ 0.13 → partial but non-trivial robustness

CE ≈ 0.87

RD (median) = moderate drift → pretrained model has some robustness

After Adversarial LoRA Fine-Tuning (s500, s1000, s2000):

RRR → 0.0 (collapse)

CE → 1.0 (full compliance with harmful requests)

RD → NaN (refusal subspace destroyed)

➡️ Training longer reduces loss but does not restore safety.
➡️ Refusal robustness collapses under LoRA attack regardless of steps.

Cross-Model Scaling (Day3–Day6):

TinyLlama-1.1B, Phi-3-mini-3.8B show same collapse patterns.

Attacker compute (LoRA steps/params) dominates model size.

📉 Key Figures (see results/figures/)

RRR_compare.png: refusal robustness collapse curves

CE_compare.png: compliance error vs steps

RD_compare.png: representation drift saturation

Scaling plots: refusal robustness across 1B–7B models

🔁 Reproducibility

Each day’s experiment tracked with notebooks + scripts + metrics (JSON/CSV).

make_figs.py and make_figs_from_combined.py regenerate all plots.

results/metrics/ contains standardized CSVs (results_grid.csv, results_grid_qwen2_7b.csv, etc.).

🏛 Governance Relevance

Refusal robustness can be completely defeated by small adversarial compute (LoRA adapters).

Model scaling (1B → 7B) does not guarantee stronger safety.

Highlights need for tampering evaluations and adversarial stress-tests in AI governance frameworks.

📚 References

Refusal Robustness Scaling Proposal (preprint PDF in repo)

Related work: Phuong & Jenner (2025), Christiano (2021), Krakovna (2024)
