#!/usr/bin/env bash
set -euo pipefail

echo ">> Creating directories (idempotent)..."
mkdir -p src/rrs scripts configs/{models,attacks,eval} \
         experiments/{tinyllama_1p1b_mvp,phi3mini_3p8b_scaling,qwen2_7b_lora_grid} \
         data/{raw,interim,processed} artifacts/{adapters,logs} \
         results/{metrics,figures} notebooks/checks tests

echo ">> Writing .gitignore..."
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
.venv/
.env

# Notebooks
.ipynb_checkpoints/

# Data & Artifacts (不要把大文件推上去)
data/interim/
data/processed/
artifacts/adapters/
artifacts/logs/

# Experiments outputs (模型权重等)
experiments/**/results/*.pt
experiments/**/results/*.bin

# Local config
*.local.yaml
EOF

echo ">> Writing .gitattributes..."
cat > .gitattributes <<'EOF'
*.csv text diff=csv
EOF

echo ">> Writing requirements.txt..."
cat > requirements.txt <<'EOF'
transformers>=4.41
datasets>=2.20
peft
accelerate
bitsandbytes
sentence-transformers
scikit-learn
pandas
numpy
tqdm
matplotlib
EOF

echo ">> Writing requirements-dev.txt..."
cat > requirements-dev.txt <<'EOF'
black
isort
flake8
pytest
EOF

echo ">> Writing Makefile (with TAB placeholders)..."
cat > Makefile <<'MAKE'
PY=python

setup:
__TAB__$(PY) -m pip install -r requirements.txt

data:
__TAB__$(PY) scripts/prepare_data_day3.py  # 先用你现有脚本名；后续再统一

pre:
__TAB__$(PY) scripts/pre_infer_day3.py --model-config configs/models/$(MODEL).yaml --eval-config configs/eval/redteam_100.yaml

attack:
__TAB__$(PY) scripts/run_lora_day3.py --model-config configs/models/$(MODEL).yaml --attack-config configs/attacks/$(ATTACK).yaml

eval:
__TAB__$(PY) scripts/post_infer_eval_day3.py --model-config configs/models/$(MODEL).yaml --eval-config configs/eval/redteam_100.yaml

fix-rd:
__TAB__$(PY) scripts/fix_rd.py --recompute-all
MAKE

# macOS/BSD sed first, fallback to GNU sed
(sed -i '' $'s/__TAB__/\t/g' Makefile 2>/dev/null) || sed -i $'s/__TAB__/\t/g' Makefile

echo ">> Writing configs..."
cat > configs/eval/redteam_100.yaml <<'EOF'
eval_set: data/raw/redteam_eval.csv
max_new_tokens: 512
decoding:
  do_sample: false
  temperature: 0.0
seed: 42
EOF

cat > configs/models/tinyllama-1.1b.yaml <<'EOF'
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dtype: float16
device: cuda
EOF

cat > configs/models/phi3-mini-3.8b.yaml <<'EOF'
model_name: microsoft/Phi-3-mini-4k-instruct
dtype: float16
device: cuda
EOF

cat > configs/models/qwen2-7b.yaml <<'EOF'
model_name: Qwen/Qwen2-7B-Instruct
dtype: float16
device: cuda
EOF

cat > configs/attacks/lora_500.yaml <<'EOF'
steps: 500
rank: 8
bits: 4
lr: 2.0e-4
seq_len: 256
optimizer: paged_adamw_8bit
save_dir: artifacts/adapters/qwen2_7b_500
EOF

cat > configs/attacks/lora_1000.yaml <<'EOF'
steps: 1000
rank: 8
bits: 4
lr: 2.0e-4
seq_len: 256
optimizer: paged_adamw_8bit
save_dir: artifacts/adapters/qwen2_7b_1000
EOF

cat > configs/attacks/lora_2000.yaml <<'EOF'
steps: 2000
rank: 8
bits: 4
lr: 2.0e-4
seq_len: 256
optimizer: paged_adamw_8bit
save_dir: artifacts/adapters/qwen2_7b_2000
EOF

echo ">> Writing experiments/qwen2_7b_lora_grid/run.sh ..."
cat > experiments/qwen2_7b_lora_grid/run.sh <<'EOF'
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
EOF

chmod +x experiments/qwen2_7b_lora_grid/run.sh

echo ">> Optionally moving known dayX files into scripts/ (if present)..."
# 预处理/数据
[ -f day1_mvp.py ]                    && git mv day1_mvp.py                    scripts/prepare_data_day1.py       || true
[ -f day2_pipeline.py ]               && git mv day2_pipeline.py               scripts/prepare_data_day2.py       || true
[ -f day2_stable_classifier.py ]      && git mv day2_stable_classifier.py      scripts/post_infer_eval_day2.py    || true
[ -f day3_01_data_or_load.py ]        && git mv day3_01_data_or_load.py        scripts/prepare_data_day3.py       || true

# 预推理 / 训练 / 后评测
[ -f day3_02_pre_infer.py ]           && git mv day3_02_pre_infer.py           scripts/pre_infer_day3.py          || true
[ -f day3_03_lora_attack.py ]         && git mv day3_03_lora_attack.py         scripts/run_lora_day3.py           || true
[ -f day3_04_post_infer_and_eval.py ] && git mv day3_04_post_infer_and_eval.py scripts/post_infer_eval_day3.py    || true
[ -f fix_rd.py ]                      && git mv fix_rd.py                      scripts/fix_rd.py                  || true

echo ">> Optionally moving results/figures if they exist..."
# 将散落在根目录的结果统一到 results/ 下（不存在则忽略）
mv *metrics.csv results/metrics/ 2>/dev/null || true
mv *grid*.csv results/metrics/ 2>/dev/null || true
mv *.png results/figures/ 2>/dev/null || true

echo ">> Done. Next steps:"
echo "   1) Place your eval CSV at: data/raw/redteam_eval.csv"
echo "   2) Install deps:    make setup"
echo "   3) (Optional) Run:  ./experiments/qwen2_7b_lora_grid/run.sh"
echo "   4) Review changes:  git status; then commit & push when ready"
