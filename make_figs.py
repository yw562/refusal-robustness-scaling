import os, json, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

# ---- 1) 配置：固定为你当前 day7 结构 ----
STEP_PATHS = {
    0:   ["day7/baseline_metrics.json", "day7/baseline/metrics.json"],
    500: ["day7/day7_s500/metrics.json"],
    1000:["day7/day7_s1000/metrics.json"],
    2000:["day7/day7_s2000/metrics.json"],
}
MODEL_NAME = "Qwen2-7B"

# ---- 2) 小工具：读 json，宽容地取 RRR/CE ----
ALT_KEYS_RRR = ["RRR","rrr","refusal_rate","refusal_robustness_rate"]
ALT_KEYS_CE  = ["CE","ce","compliance_error","compliance","comp_error"]

def read_json_safe(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def get_first_key(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except:
                pass
    return np.nan

# ---- 3) 收集 RRR/CE；RD 先留 NaN ----
rows = []
for steps, candidates in STEP_PATHS.items():
    found = None
    data = {}
    for p in candidates:
        if os.path.exists(p):
            found = p
            data = read_json_safe(p)
            break
    r = {
        "model": MODEL_NAME,
        "steps": steps,
        "RRR": get_first_key(data, ALT_KEYS_RRR),
        "CE":  get_first_key(data, ALT_KEYS_CE),
        "RD_median": np.nan,
        "_src": found or "missing"
    }
    rows.append(r)

df = pd.DataFrame(rows).sort_values("steps")

# ---- 4) 尝试自动填 RD（可选）----
# 规则：找任意包含 RD_median 的 csv，且有 'steps' 列；若还有 'model' 列则优先筛 Qwen；否则按步数合并
rd_files = glob.glob("**/*rd*fixed*.csv", recursive=True) + glob.glob("**/results_grid.csv", recursive=True)
rd_used = None
for f in rd_files:
    try:
        tmp = pd.read_csv(f)
        cols = [c.lower() for c in tmp.columns]
        if ("steps" in cols) and ("rd_median" in cols or "rd" in cols):
            rd_col = "RD_median" if "RD_median" in tmp.columns else "RD"
            use = tmp.copy()
            if "model" in use.columns:
                # 尝试优先匹配 Qwen
                mask = use["model"].astype(str).str.contains("qwen", case=False, regex=False)
                if mask.any():
                    use = use[mask]
            # 仅保留 steps + rd
            use = use[["steps", rd_col]].dropna()
            # 规范 steps 为 int
            use["steps"] = use["steps"].astype(str).str.extract(r"(\d+)").astype(float).astype(int)
            # 合并
            df = df.merge(use.groupby("steps")[rd_col].first().rename("RD_median"),
                          on="steps", how="left", suffixes=("", "_from_rd"))
            # 优先用新合并列
            df["RD_median"] = df["RD_median_from_rd"].combine_first(df["RD_median"])
            df = df.drop(columns=[c for c in df.columns if c.endswith("_from_rd")])
            rd_used = f
            break
    except Exception:
        continue

# ---- 5) 导出整洁表 ----
def fmt(x):
    if pd.isna(x): return "—"
    try:
        return f"{float(x):.2f}"
    except:
        return "—"

table = df[["model","steps","RRR","RD_median","CE"]].copy()
table = table.sort_values("steps")
table_out = table.pivot_table(index="model", columns="steps", values=["RRR","RD_median","CE"])
table_out.to_csv("figs/table_master_for_paper.csv")

# ---- 6) 画图（缺值自动跳过）；RD 若全 NaN，就写提示图 ----
def plot_metric(metric, ylabel, fname):
    plt.figure()
    sub = df.sort_values("steps")
    if sub[metric].notna().any():
        plt.plot(sub["steps"], sub[metric], marker="o", label=MODEL_NAME)
        plt.legend()
    else:
        # 给出不可用提示
        plt.text(0.5, 0.5, f"{metric} not available", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel("LoRA steps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figs/{fname}", dpi=300)
    plt.close()

plot_metric("RRR", "Refusal Robustness Rate (RRR)", "fig1_rrr_vs_steps.png")
plot_metric("CE",  "Compliance Error (CE)",         "fig2_ce_vs_steps.png")
plot_metric("RD_median", "Refusal Drift (median)",  "fig3_rd_vs_steps.png")

# ---- 7) 控制台提示 ----
print("== Done ==")
print("Source files used:")
for _, r in df.iterrows():
    print(f"  steps={int(r['steps'])}: {r['_src']}")
if rd_used:
    print(f"RD source: {rd_used}")
print("Outputs:")
print("  figs/fig1_rrr_vs_steps.png")
print("  figs/fig2_ce_vs_steps.png")
print("  figs/fig3_rd_vs_steps.png")
print("  figs/table_master_for_paper.csv")
