import os, json, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

# ========== 1) 读 CSV ==========
base_csv = "combined_results_standardized.csv"
if not os.path.exists(base_csv):
    raise SystemExit("缺少 combined_results_standardized.csv，请确认文件在 repo 根目录。")

df = pd.read_csv(base_csv)

# ========== 2) 列名规范化 ==========
orig_cols = list(df.columns)
lower_map = {c.lower().strip(): c for c in df.columns}

def pick_col(options, required=False):
    for opt in options:
        k = opt.lower().strip()
        if k in lower_map:
            return lower_map[k]
    if required:
        raise SystemExit(f"缺少必要列：{options}，现有列：{orig_cols}")
    return None

col_model = pick_col(["Model", "model", "模型", "name"], required=True)
col_steps = pick_col(["Steps", "Step", "LoRA Steps", "lora_steps", "step_num", "s", "iter", "iters"], required=True)
col_rrr   = pick_col(["RRR","rrr","refusal_rate","refusal_robustness_rate"])
col_rd    = pick_col(["RD","rd","RD_median","rd_median","refusal_drift"])
col_ce    = pick_col(["CE","ce","compliance_error","compliance","comp_error"])

rename_map = {col_model: "Model", col_steps: "Steps"}
if col_rrr: rename_map[col_rrr] = "RRR"
if col_rd:  rename_map[col_rd]  = "RD"
if col_ce:  rename_map[col_ce]  = "CE"

df = df.rename(columns=rename_map)

# 规范 Steps 为整数
df["Steps"] = df["Steps"].astype(str).str.extract(r"(\d+)").fillna("0").astype(int)

# 确保三大指标存在
for col in ["RRR","RD","CE"]:
    if col not in df.columns:
        df[col] = np.nan

# ========== 3) 追加 Day7 (Qwen2-7B) ==========
def read_json_safe(p):
    try:
        with open(p,"r") as f: return json.load(f)
    except: return {}

def get_first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            try: return float(d[k])
            except: pass
    return np.nan

ALT_KEYS_RRR = ["RRR","rrr","refusal_rate","refusal_robustness_rate"]
ALT_KEYS_CE  = ["CE","ce","compliance_error","compliance","comp_error"]

def pick_metrics_for_step(step):
    if step==0:
        cands = ["day7/baseline_metrics.json","day7/baseline/metrics.json"]
    else:
        tag = f"s{step}"
        cands = [
            f"day7/day7_{tag}/metrics.json",
            f"day7/{tag}/metrics.json",
        ] + glob.glob(f"day7/**/*_{tag}/metrics.json", recursive=True)
    for p in cands:
        if os.path.exists(p):
            d = read_json_safe(p)
            return get_first(d, ALT_KEYS_RRR), get_first(d, ALT_KEYS_CE), p
    return np.nan, np.nan, "missing"

def try_fetch_day7_rd():
    files = glob.glob("**/*rd*fixed*.csv", recursive=True) + glob.glob("**/results_grid.csv", recursive=True)
    for f in files:
        try:
            t = pd.read_csv(f)
            cols = {c.lower():c for c in t.columns}
            if "steps" in cols and ("rd_median" in cols or "rd" in cols):
                rdcol = cols.get("rd_median", cols.get("rd"))
                use = t.copy()
                if "model" in use.columns:
                    msk = use["model"].astype(str).str.contains("qwen", case=False, regex=False)
                    if msk.any():
                        use = use[msk]
                use["Steps"] = use[cols["steps"]].astype(str).str.extract(r"(\d+)").fillna("0").astype(int)
                use = use[["Steps", rdcol]].dropna()
                use.rename(columns={rdcol:"RD"}, inplace=True)
                return use
        except Exception:
            continue
    return pd.DataFrame(columns=["Steps","RD"])

added = []
if os.path.isdir("day7"):
    rows = []
    for st in [0,500,1000,2000]:
        rrr, ce, src = pick_metrics_for_step(st)
        rows.append({"Model":"Qwen2-7B","Steps":st,"RRR":rrr,"CE":ce,"_src":src})
    qwen = pd.DataFrame(rows)
    rdq = try_fetch_day7_rd()
    if not rdq.empty:
        qwen = qwen.merge(rdq, on="Steps", how="left")
    else:
        qwen["RD"] = np.nan
    # —— 无条件 upsert：确保 Qwen2-7B 的 0/500/1000/2000 都在，且缺值被补齐
    base = df.copy()
    qkeep = qwen[["Model","Steps","RRR","RD","CE"]].copy()
    merged = pd.merge(base, qkeep, on=["Model","Steps"], how="outer", suffixes=("","_q"))
    for col in ["RRR","RD","CE"]:
        if f"{col}_q" in merged.columns:
            merged[col] = merged[col].combine_first(merged[f"{col}_q"])
            merged.drop(columns=[f"{col}_q"], inplace=True)
    df = merged
    added = ["Qwen2-7B"]


# ========== 4) 排序 & 导出 ==========
def norm_model(m):
    s = str(m)
    if "TinyLlama" in s: return "TinyLlama-1.1B"
    if "Phi" in s: return "Phi-3-mini-3.8B"
    if "Qwen" in s: return "Qwen2-7B"
    return s

df["Model"] = df["Model"].map(norm_model)
order = [m for m in ["TinyLlama-1.1B","Phi-3-mini-3.8B","Qwen2-7B"] if m in df["Model"].unique()]
df["Model"] = pd.Categorical(df["Model"], order, ordered=True)
df = df.sort_values(["Model","Steps"]).reset_index(drop=True)

# 宽表输出
wide = df.pivot_table(index="Model", columns="Steps", values=["RRR","RD","CE"]).sort_index(axis=1, level=1)
wide.to_csv("figs/table_all_models_for_paper.csv")

# ========== 5) 画图 ==========
def plot_metric(col, ylabel, fname):
    plt.figure()
    for m in df["Model"].cat.categories:
        sub = df[df["Model"]==m].sort_values("Steps")
        if sub[col].notna().any():
            plt.plot(sub["Steps"], sub[col], marker="o", label=m)
    if col=="RD" and not df["RD"].notna().any():
        plt.text(0.5,0.5,"RD not available",ha="center",va="center",transform=plt.gca().transAxes)
    plt.xlabel("LoRA steps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/{fname}", dpi=300)
    plt.close()

plot_metric("RRR","Refusal Robustness Rate (RRR)","fig_rrr_vs_steps.png")
plot_metric("RD","Refusal Drift (median)","fig_rd_vs_steps.png")
plot_metric("CE","Compliance Error (CE)","fig_ce_vs_steps.png")

print("== DONE ==")
print("Added models:", added)
print("Outputs in figs/: fig_rrr_vs_steps.png, fig_rd_vs_steps.png, fig_ce_vs_steps.png, table_all_models_for_paper.csv")
