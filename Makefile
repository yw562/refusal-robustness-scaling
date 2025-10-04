PY=python

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/prepare_data_day3.py  # 先用你现有脚本名；后续再统一

pre:
	$(PY) scripts/pre_infer_day3.py --model-config configs/models/$(MODEL).yaml --eval-config configs/eval/redteam_100.yaml

attack:
	$(PY) scripts/run_lora_day3.py --model-config configs/models/$(MODEL).yaml --attack-config configs/attacks/$(ATTACK).yaml

eval:
	$(PY) scripts/post_infer_eval_day3.py --model-config configs/models/$(MODEL).yaml --eval-config configs/eval/redteam_100.yaml

fix-rd:
	$(PY) scripts/fix_rd.py --recompute-all
