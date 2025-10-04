# make_extra_figs.py
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

TABLE = "figs/table_all_models_for_paper.csv"

# --- 1) 读取 table_all_models_for_paper.csv（MultiIndex header 兼容） ---
if not os.path.exists(TABLE):
    raise SystemExit("缺少 figs/table_all_models_for_paper.csv，请先跑前面的汇总脚本。")

# 先尝试两行表头（MultiIndex），失败则单行回退
try:
    wide = pd.read_csv(TABLE, header=[0,1])
    # 规范列名：两层列 -> (metric, step)
    wide.columns = pd.MultiIndex.from_tuples(
        [(c[0], c[1]) if isinstance(c, tuple) else (str(c), "")
         for c in wide.columns]
    )
except:
    # 单行表头：尝试解析形如 "RD.500"
    tmp = pd.read_csv(TABLE)
    new_cols = []
    for c in tmp.columns:
        if "." in c and c.split(".")[0] in {"RRR","RD","CE"}:
            new_cols.append(tuple(c.split(".",1)))
        else:
            new_cols.append((c,""))
    tmp.columns = pd.MultiIndex.from_tuples(new_cols)
    wide = tmp

# 找到行索引列（Model）
if ("Model","") not in wide.columns:
    # 有些导出会把索引保存为第一列，名字可能是 'model'
    for cand in [("model",""),("Model","Model"),("Unnamed: 0_level_0","")]:
        if cand in wide.columns:
            wide = wide.rename(columns={cand:("Model","")})
            break
if ("Model","") not in wide.columns:
    raise SystemExit("无法在表里找到 Model 列。")

# 把宽表变成长表：列层级 (metric, step)
long_rows = []
for metric in ["RRR","RD","CE"]:
    if metric not in wide.columns.get_level_values(0):
        continue
    sub = wide[metric]
    # 可能存在字符串 step 列名
    for step_col in sub.columns:
        try:
            step = int(str(step_col).strip())
        except:
            # 非数字列跳过
            continue
        for _, row in wide.iterrows():
            long_rows.append({
                "Model": row[("Model","")],
                "Steps": step,
                metric: row[(metric, step_col)]
            })
long = pd.DataFrame(long_rows)

# 合并成一个表（每行：Model, Steps, RRR, RD, CE）
df = long.groupby(["Model","Steps"]).first().reset_index()
for col in ["RRR","RD","CE"]:
    if col not in df.columns: df[col] = np.nan

# 统一排序
order = [m for m in ["TinyLlama-1.1B","Phi-3-mini-3.8B","Qwen2-7B"] if m in df["Model"].unique()]
df["Model"] = pd.Categorical(df["Model"], order=order, ordered=True)
df = df.sort_values(["Model","Steps"]).reset_index(drop=True)

# --- 2) RD–CE 散点图（每点=模型×步数） ---
plt.figure()
has_any = False
for m in df["Model"].cat.categories:
    sub = df[df["Model"]==m]
    sub = sub.dropna(subset=["RD","CE"])
    if sub.empty: continue
    plt.scatter(sub["RD"], sub["CE"], label=m, s=60)
    # 标注步数
    for _, r in sub.iterrows():
        plt.annotate(int(r["Steps"]), (r["RD"], r["CE"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    has_any = True
if not has_any:
    plt.text(0.5,0.5,"RD or CE not available", ha="center", va="center", transform=plt.gca().transAxes)
plt.xlabel("Refusal Drift (RD, median)")
plt.ylabel("Compliance Error (CE)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figs/fig_rd_vs_ce_scatter.png", dpi=300)
plt.close()

# --- 3) RD 分布/箱线图（若无逐条 RD，则回退为分组柱状的中位数） ---

# 尝试自动发现“逐条 RD”文件（如果你有的话，命名随意，只要包含 RD 列和 steps）
rd_detail_files = []
for pat in ["**/*rd*detail*.csv","**/*rd*pairs*.csv","**/*rd*per*sample*.csv","**/*_rd_fixed_samples*.csv"]:
    rd_detail_files += glob.glob(pat, recursive=True)

rd_samples = []
def coerce_int_steps(x):
    try:
        return int(str(x).strip().split()[-1].replace("s",""))
    except:
        # 正则抽取数字
        import re
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else np.nan

for f in rd_detail_files:
    try:
        t = pd.read_csv(f)
        cols = {c.lower(): c for c in t.columns}
        # 需要 RD 列
        rdcol = cols.get("rd") or cols.get("rd_median") or cols.get("rd_value")
        if not rdcol: continue
        # steps 列
        stecol = cols.get("steps") or cols.get("step") or cols.get("s") or cols.get("lora_steps")
        if stecol is None: continue
        # model 列（可选）
        mcol = cols.get("model") or cols.get("name")
        t = t[[c for c in [rdcol, stecol, mcol] if c is not None]].dropna(subset=[rdcol, stecol])
        t.rename(columns={rdcol:"RD", stecol:"Steps"}, inplace=True)
        t["Steps"] = t["Steps"].map(coerce_int_steps)
        if mcol:
            # 标准化模型名
            def norm(m):
                s = str(m)
                if "TinyLlama" in s or "tiny" in s: return "TinyLlama-1.1B"
                if "Phi" in s: return "Phi-3-mini-3.8B"
                if "Qwen" in s or "qwen2" in s: return "Qwen2-7B"
                return s
            t["Model"] = t[mcol].map(norm)
        else:
            # 若没 model 列，就用 df 里的模型集合逐个尝试；这里简单放 None，后面再填
            t["Model"] = None
        rd_samples.append(t)
    except Exception:
        continue

if rd_samples:
    S = pd.concat(rd_samples, ignore_index=True)
    # 若存在缺失 Model，用步数+已知集合去推断（保守：不推断时直接丢弃缺失）
    S = S.dropna(subset=["Model"])
    S = S[S["Model"].isin(order)]
    # 箱线图：x=Steps（按顺序），每个 Model 一组
    for m in df["Model"].cat.categories:
        sm = S[S["Model"]==m]
        if sm.empty: continue
        xs = sorted(sm["Steps"].dropna().unique().tolist())
        data = [sm[sm["Steps"]==s]["RD"].dropna().values for s in xs]
        if not any(len(d)>0 for d in data): continue
        plt.figure()
        plt.boxplot(data, positions=range(len(xs)), widths=0.6, showfliers=False)
        plt.xticks(range(len(xs)), [str(s) for s in xs])
        plt.xlabel("LoRA steps")
        plt.ylabel("RD (per-sample)")
        plt.title(f"RD distribution — {m}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out = f"figs/fig_rd_box_{m.replace(' ','_').replace('.','')}.png"
        plt.savefig(out, dpi=300)
        plt.close()
    # 总合并一个面板（选做）：先生成单图就够用了
else:
    # 回退：用中位数画“分组柱状图”，并在柱顶标注 step 数
    plt.figure(figsize=(7.2,4.8))
    width = 0.22
    steps = sorted(df["Steps"].dropna().unique().tolist())
    for i, m in enumerate(df["Model"].cat.categories):
        sub = df[df["Model"]==m]
        y = [sub[sub["Steps"]==s]["RD"].dropna().mean() if not sub[sub["Steps"]==s]["RD"].dropna().empty else np.nan for s in steps]
        xs = np.arange(len(steps)) + (i - (len(order)-1)/2)*width
        plt.bar(xs, y, width=width, label=m)
    plt.xticks(np.arange(len(steps)), [str(s) for s in steps])
    plt.xlabel("LoRA steps")
    plt.ylabel("RD (median)")
    plt.title("RD across models (median only)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/fig_rd_grouped.png", dpi=300)
    plt.close()

print("== DONE ==")
print("Saved:")
print("  figs/fig_rd_vs_ce_scatter.png")
if rd_samples:
    print("  figs/fig_rd_box_*.png  (one per model)")
else:
    print("  figs/fig_rd_grouped.png  (fallback when per-sample RD is unavailable)")
