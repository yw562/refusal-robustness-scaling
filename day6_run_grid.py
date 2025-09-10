#!/usr/bin/env python
# -*- coding: utf-8 -*-
# day6_run_grid.py — Clean, auditable, Kaggle-friendly
# - Batch size = 2 (safe on T4)
# - Eval forced to CPU to avoid CUDA crashes; training uses GPU
# - status / error_msg audit columns; metrics only when success

import os, argparse, subprocess, pandas as pd
from pathlib import Path
from datetime import datetime

WORKDIR = Path("/kaggle/working/refusal_scaling")
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATA_SAFE = WORKDIR / "_fast_safe_train.csv" if (WORKDIR/"_fast_safe_train.csv").exists() else WORKDIR / "day2_safe_dataset.csv"
DATA_PRE = WORKDIR / "_fast_eval_pre.csv" if (WORKDIR/"_fast_eval_pre.csv").exists() else WORKDIR / "day4_pre_generation.csv"
RESULTS_CSV = WORKDIR / "results_grid.csv"
METRICS_OUT = WORKDIR / "day4_metrics.csv"
ADAPTER_DIR_ROOT = WORKDIR / "adapters"

BATCH_SIZE = 2
LR        = 2e-4
MAX_LEN = 192
SEED      = 42

EXPECTED = ["RRR_pre","RRR_post","RD","CE_pre","CE_post"]

def safe_model_dirname(mid: str) -> str:
    return mid.replace("/", "_").replace(":", "_")

def run_eval(model_id: str, steps: int, adapter_dir: Path|None) -> tuple[int, str]:
    # Resolve eval file paths (FAST_MODE fallback supported)
    fast_eval_safe = WORKDIR / "_fast_eval_safe.csv"
    fast_eval_pre  = WORKDIR / "_fast_eval_pre.csv"
    eval_safe_csv = fast_eval_safe if fast_eval_safe.exists() else DATA_SAFE
    eval_pre_csv  = fast_eval_pre if fast_eval_pre.exists() else DATA_PRE

    cmd = [
        "python", str(WORKDIR / "day4_04_post_infer_and_eval.py"),
        "--base_model", model_id,
        "--eval_safe_csv", str(eval_safe_csv),
        "--eval_pre_csv",  str(eval_pre_csv),
        "--out_csv",       str(METRICS_OUT),
    ]
    if adapter_dir and adapter_dir.exists():
        cmd += ["--adapter_path", str(adapter_dir)]

    # Use GPU if available (no CUDA_VISIBLE_DEVICES override);
    # add a hard timeout to avoid indefinite hangs.
    import subprocess, os
    TIMEOUT_SEC = 600  # adjust if needed
    print("\n$ (GPU eval, streaming logs) " + " ".join(cmd))
    try:
        res = subprocess.run(cmd, text=True, timeout=TIMEOUT_SEC)
        return res.returncode, ""
    except subprocess.TimeoutExpired:
        # Timed out: return non-zero rc so caller records status=error and continues
        print("[warn] eval timed out; marking as error and continuing")
        return 124, "timeout"
def append_row(status: str, error_msg: str|None, model_id: str, steps: int, metrics: dict|None):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_id,
        "steps": steps,
        "batch": BATCH_SIZE,
        "lr": LR,
        "max_len": MAX_LEN,
        "seed": SEED,
        "adapter_path": str(ADAPTER_DIR_ROOT / safe_model_dirname(model_id) / f"steps-{steps}") if steps>0 else "",
        "status": status,
        "error_msg": (error_msg or "")[:500],
    }
    if metrics:
        row.update(metrics)
    else:
        for c in EXPECTED: row[c] = None

    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    cols = ["timestamp","model","steps","batch","lr","max_len","seed","adapter_path","status","error_msg"] + EXPECTED
    for c in cols:
        if c not in df.columns: df[c] = None
    df = df[cols]
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[ok] Wrote -> {RESULTS_CSV} (rows={len(df)})")

def ensure_files():
    if not DATA_SAFE.exists():  raise FileNotFoundError(f"Missing {DATA_SAFE}")
    if not DATA_PRE.exists():   raise FileNotFoundError(f"Missing {DATA_PRE}")

def train_lora(steps: int, out_dir: Path) -> int:
    if steps <= 0:
        print("[info] steps=0 => skip LoRA training")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", str(WORKDIR / "day4_03_lora_attack.py"),
        "--base_model", MODEL_ID,
        "--train_csv", str(DATA_SAFE),
        "--output_dir", str(out_dir),
        "--max_steps", str(steps),
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", str(LR),
        "--max_len", str(MAX_LEN),
        "--seed", str(SEED),
    ]
    print("\n$ (GPU train) " + " ".join(cmd))
    res = subprocess.run(cmd, text=True)
    return res.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline-only","full"], default="baseline-only")
    parser.add_argument("--steps", type=str, default="0,500,1000,2000")
    args = parser.parse_args()

    ensure_files()
    mdir = ADAPTER_DIR_ROOT / safe_model_dirname(MODEL_ID)
    mdir.mkdir(parents=True, exist_ok=True)

    steps_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()!=""]

    for s in steps_list:
        print("\n" + "="*68)
        print(f"[stage] model={MODEL_ID} steps={s} mode={args.mode}")

        if args.mode == "full":
            rc = train_lora(s, mdir / f"steps-{s}")
            if rc != 0:
                append_row("error", f"LoRA training rc={rc}", MODEL_ID, s, None)
                continue

        adapter_dir = (mdir / f"steps-{s}") if s>0 else None
        rc, logs = run_eval(MODEL_ID, s, adapter_dir)
        if rc != 0 or not METRICS_OUT.exists():
            append_row("error", f"eval rc={rc}; {logs}", MODEL_ID, s, None)
            continue

        try:
            dfm = pd.read_csv(METRICS_OUT)
            ok_cols = all(c in dfm.columns for c in EXPECTED)
            ok_vals = (not dfm[EXPECTED].iloc[0].isna().any()) if ok_cols else False
            if ok_cols and ok_vals:
                metrics = {c: float(dfm[c].iloc[0]) for c in EXPECTED}
                append_row("ok", None, MODEL_ID, s, metrics)
            else:
                missing = [c for c in EXPECTED if c not in dfm.columns]
                reason = "missing cols: " + ",".join(missing) if missing else "NaN in metrics"
                append_row("error", reason, MODEL_ID, s, None)
        except Exception as e:
            append_row("error", f"metrics parse error: {e}", MODEL_ID, s, None)

    print("\n[done] Day6 auditable run finished.")

if __name__ == "__main__":
    main()