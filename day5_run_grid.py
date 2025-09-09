
import os
import re
import time
import subprocess
import pandas as pd

ROOT = r"/kaggle/working/refusal_scaling"
MODEL = "microsoft/Phi-3-mini-4k-instruct"
PARAMS_B = 3.8
STEPS_LIST = [500, 1000, 2000]
DAY4_TRAIN = os.path.join(ROOT, "day4_03_lora_attack.py")
DAY4_EVAL  = os.path.join(ROOT, "day4_04_post_infer_and_eval.py")
RESULTS_CSV = os.path.join(ROOT, "results_grid.csv")
DAY4_METRICS = os.path.join(ROOT, "day4_metrics.csv")

def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def write_text(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def replace_max_steps(src_text, steps):
    # Replace the FIRST occurrence of: max_steps = <int>
    pat_line = re.compile(r"^(\s*max_steps\s*=)\s*\d+\s*$", re.MULTILINE)
    if pat_line.search(src_text):
        return pat_line.sub(r"\1 " + str(steps), src_text, count=1)
    pat_any = re.compile(r"max_steps\s*=\s*\d+")
    if pat_any.search(src_text):
        return pat_any.sub("max_steps = " + str(steps), src_text, count=1)
    raise RuntimeError("Could not find a 'max_steps = <int>' assignment in day4_03_lora_attack.py")

def append_results_row(steps):
    assert os.path.exists(DAY4_METRICS), f"Missing metrics file: {DAY4_METRICS}"
    m = pd.read_csv(DAY4_METRICS)

    # Support both long (metric/value) and wide formats
    if {"metric","value"}.issubset(set(m.columns)):
        row = {r.metric: r.value for _, r in m.iterrows()}
    else:
        # assume wide; take the first row
        row = m.iloc[0].to_dict()

    row.update(dict(
        model="Phi-3-mini-4k-instruct",
        params_b=PARAMS_B,
        attack="LoRA",
        steps=steps,
        batch=4,
        lr=2e-4,
        max_len=256,
        seed=42,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    ))

    header_needed = not os.path.exists(RESULTS_CSV)
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=header_needed, index=False)
    print("Appended to:", RESULTS_CSV)

def main():
    # Backup original trainer once
    backup = DAY4_TRAIN + ".orig"
    if not os.path.exists(backup):
        src_original = read_text(DAY4_TRAIN)
        write_text(backup, src_original)
    else:
        src_original = read_text(backup)  # always patch from the clean backup

    try:
        for s in STEPS_LIST:
            print(f"\n=== Running LoRA attack with max_steps={s} ===")
            # Patch trainer for this run
            patched = replace_max_steps(src_original, s)
            write_text(DAY4_TRAIN, patched)

            # 1) Train
            subprocess.run(["python", DAY4_TRAIN], check=True)

            # 2) Post-infer & Eval
            subprocess.run(["python", DAY4_EVAL], check=True)

            # 3) Append aggregated results
            append_results_row(s)

    finally:
        # Restore original trainer
        write_text(DAY4_TRAIN, src_original)
        print("\nRestored:", DAY4_TRAIN, "from backup.")

    print("\nAll runs complete. results appended to:", RESULTS_CSV)

if __name__ == "__main__":
    main()
