#!/usr/bin/env bash
set -euo pipefail

# 确保目标目录存在
mkdir -p experiments/qwen2_7b_lora_grid/results notebooks/checks/optional

echo ">> Moving Day7 baseline artifacts..."
[ -f day7/baseline/baseline_outputs.csv ] && git mv day7/baseline/baseline_outputs.csv experiments/qwen2_7b_lora_grid/results/baseline_outputs.csv || true
[ -f day7/baseline/baseline_metrics.json ] && git mv day7/baseline/baseline_metrics.json experiments/qwen2_7b_lora_grid/results/baseline_metrics.json || true
[ -f day7/baseline/Day-7-baseline.ipynb ] && git mv day7/baseline/Day-7-baseline.ipynb notebooks/checks/Day7-baseline.ipynb || true

echo ">> Moving Day7 s500 artifacts..."
[ -f day7/day7_s500/outputs.csv ] && git mv day7/day7_s500/outputs.csv experiments/qwen2_7b_lora_grid/results/outputs_500.csv || true
[ -f day7/day7_s500/metrics.json ] && git mv day7/day7_s500/metrics.json experiments/qwen2_7b_lora_grid/results/metrics_500.json || true
[ -f day7/day7_s500/metrics.json.bak ] && git mv day7/day7_s500/metrics.json.bak experiments/qwen2_7b_lora_grid/results/metrics_500.json.bak || true
[ -f day7/day7_s500/Day7-s500.ipynb ] && git mv day7/day7_s500/Day7-s500.ipynb notebooks/checks/Day7-s500.ipynb || true

echo ">> Moving Day7 s1000 artifacts..."
[ -f day7/day7_s1000/outputs.csv ] && git mv day7/day7_s1000/outputs.csv experiments/qwen2_7b_lora_grid/results/outputs_1000.csv || true
[ -f day7/day7_s1000/metrics.json ] && git mv day7/day7_s1000/metrics.json experiments/qwen2_7b_lora_grid/results/metrics_1000.json || true
[ -f day7/day7_s1000/metrics.json.bak ] && git mv day7/day7_s1000/metrics.json.bak experiments/qwen2_7b_lora_grid/results/metrics_1000.json.bak || true
[ -f day7/day7_s1000/Day7-s1000.ipynb ] && git mv day7/day7_s1000/Day7-s1000.ipynb notebooks/checks/Day7-s1000.ipynb || true

echo ">> Moving Day7 s2000 artifacts..."
[ -f day7/day7_s2000/outputs.csv ] && git mv day7/day7_s2000/outputs.csv experiments/qwen2_7b_lora_grid/results/outputs_2000.csv || true
[ -f day7/day7_s2000/metrics.json ] && git mv day7/day7_s2000/metrics.json experiments/qwen2_7b_lora_grid/results/metrics_2000.json || true
[ -f day7/day7_s2000/metrics.json.bak ] && git mv day7/day7_s2000/metrics.json.bak experiments/qwen2_7b_lora_grid/results/metrics_2000.json.bak || true
[ -f day7/day7_s2000/Day7-s2000.ipynb ] && git mv day7/day7_s2000/Day7-s2000.ipynb notebooks/checks/Day7-s2000.ipynb || true

echo ">> Moving baseline_vs_* comparisons..."
# 申请材料里只主推 baseline_vs_s1000.csv，其它先放 optional
if [ -f day7/day7_s1000/baseline_vs_s1000.csv ]; then
  git mv day7/day7_s1000/baseline_vs_s1000.csv experiments/qwen2_7b_lora_grid/results/baseline_vs_s1000.csv
elif [ -f day7/baseline_vs_s1000.csv ]; then
  git mv day7/baseline_vs_s1000.csv experiments/qwen2_7b_lora_grid/results/baseline_vs_s1000.csv
fi

[ -f day7/baseline_vs_s500.csv ]  && git mv day7/baseline_vs_s500.csv  notebooks/checks/optional/baseline_vs_s500.csv  || true
[ -f day7/baseline_vs_s2000.csv ] && git mv day7/baseline_vs_s2000.csv notebooks/checks/optional/baseline_vs_s2000.csv || true
[ -f day7/checks/baseline_vs_s1000.csv ] && git mv day7/checks/baseline_vs_s1000.csv notebooks/checks/optional/baseline_vs_s1000_copy.csv || true

echo ">> Removing empty day7/ directory if everything moved..."
rmdir day7/baseline 2>/dev/null || true
rmdir day7/day7_s500 2>/dev/null || true
rmdir day7/day7_s1000 2>/dev/null || true
rmdir day7/day7_s2000 2>/dev/null || true
rmdir day7/checks 2>/dev/null || true
rmdir day7 2>/dev/null || true

echo ">> Done. Review with: git status"
