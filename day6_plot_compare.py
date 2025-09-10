#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_CSV = Path("results_grid.csv")
if not RESULTS_CSV.exists():
    print("[error] results_grid.csv not found"); raise SystemExit(1)

df = pd.read_csv(RESULTS_CSV).sort_values(["model","steps"])
df_ok = df[df["status"].fillna("")=="ok"]
if df_ok.empty:
    print("[warn] no ok rows to plot"); raise SystemExit(0)

# RRR_post vs steps
plt.figure()
for m, sub in df_ok.groupby("model"):
    plt.plot(sub["steps"], sub["RRR_post"], marker="o", label=m)
plt.xlabel("LoRA steps"); plt.ylabel("RRR_post"); plt.title("RRR_post vs steps — by model")
plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig("compare_models_rrr.png", dpi=180)
print("[ok] compare_models_rrr.png")

# RD vs steps
plt.figure()
for m, sub in df_ok.groupby("model"):
    plt.plot(sub["steps"], sub["RD"], marker="o", label=m)
plt.xlabel("LoRA steps"); plt.ylabel("RD"); plt.title("RD vs steps — by model")
plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig("compare_models_rd.png", dpi=180)
print("[ok] compare_models_rd.png")
