
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = r"/kaggle/working/refusal_scaling"
CSV = os.path.join(ROOT, "results_grid.csv")
assert os.path.exists(CSV), f"Missing {CSV}. Please run the grid first."

df = pd.read_csv(CSV)

# Filter this model/attack
filt = (df.get("model","")=="Phi-3-mini-4k-instruct") & (df.get("attack","")=="LoRA")
sub = df.loc[filt].copy()

# Basic sanity
if "steps" not in sub.columns:
    raise SystemExit("Column 'steps' not found in results_grid.csv.")

# Expected metric column names; adjust if yours differ
if "RRR_post" not in sub.columns:
    raise SystemExit("Column 'RRR_post' not found in results_grid.csv.")
if "RD" not in sub.columns:
    raise SystemExit("Column 'RD' not found in results_grid.csv.")

g = sub.groupby("steps", as_index=False).mean(numeric_only=True).sort_values("steps")

# RRR_post vs steps
plt.figure()
plt.plot(g["steps"], g["RRR_post"], marker="o")
plt.xlabel("LoRA training steps")
plt.ylabel("RRR_post")
plt.title("Attack Strength Scaling — RRR_post (Phi-3-mini)")
out1 = os.path.join(ROOT, "scaling_strength_rrr.png")
plt.savefig(out1, dpi=200, bbox_inches="tight")
print("Saved:", out1)

# RD vs steps
plt.figure()
plt.plot(g["steps"], g["RD"], marker="o")
plt.xlabel("LoRA training steps")
plt.ylabel("RD")
plt.title("Attack Strength Scaling — RD (Phi-3-mini)")
out2 = os.path.join(ROOT, "scaling_strength_rd.png")
plt.savefig(out2, dpi=200, bbox_inches="tight")
print("Saved:", out2)
