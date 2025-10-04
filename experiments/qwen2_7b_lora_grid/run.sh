#!/usr/bin/env bash
set -e

# 0) 确保评测集就位（如果你已有 redteam_eval.csv，请把它拷到 data/raw/）
#    cp /path/to/redteam_eval.csv data/raw/redteam_eval.csv

# 1) （可选）准备数据：如果你的 day3 脚本会生成/清洗数据
python scripts/prepare_data_day3.py || true

# 2) 基线推理（pre）
# 如果你的脚本暂不支持 --model-config/--eval-config，请把下面两行的参数删掉，或改用：python scripts/pre_infer_day3.py
python scripts/pre_infer_day3.py \
  --model-config configs/models/qwen2-7b.yaml \
  --eval-config configs/eval/redteam_100.yaml \
  --out experiments/qwen2_7b_lora_grid/results/baseline_outputs.csv

# 3) LoRA 训练 + post 推理（500/1000/2000）
for STEPS in 500 1000 2000; do
  python scripts/run_lora_day3.py \
    --model-config configs/models/qwen2-7b.yaml \
    --attack-config configs/attacks/lora_${STEPS}.yaml

  python scripts/post_infer_eval_day3.py \
    --model-config configs/models/qwen2-7b.yaml \
    --adapter artifacts/adapters/qwen2_7b_${STEPS} \
    --eval-config configs/eval/redteam_100.yaml \
    --baseline experiments/qwen2_7b_lora_grid/results/baseline_outputs.csv \
    --out experiments/qwen2_7b_lora_grid/results/outputs_${STEPS}.csv
done

# 4) RD 统一修复 + 汇总
python scripts/fix_rd.py \
  --root experiments/qwen2_7b_lora_grid/results \
  --write-aggregates results/metrics/results_grid_qwen2_7b.csv
