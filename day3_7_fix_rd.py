import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

def fix_rd(gens_csv, metrics_csv, model="sentence-transformers/all-MiniLM-L6-v2"):
    df = pd.read_csv(gens_csv)
    pre, post = df["gen_pre"].fillna(""), df["gen_post"].fillna("")
    mask = pre.str.strip().ne("") & post.str.strip().ne("")
    emb = SentenceTransformer(model)
    a = emb.encode(pre[mask].tolist(),  normalize_embeddings=True, convert_to_numpy=True)
    b = emb.encode(post[mask].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    rd = 1.0 - (a*b).sum(axis=1)
    rd = rd[np.isfinite(rd)]
    rd_med = float(np.median(rd)) if rd.size else float("nan")
    rd_mean = float(np.mean(rd)) if rd.size else float("nan")
    try: m = pd.read_csv(metrics_csv)
    except FileNotFoundError: m = pd.DataFrame(columns=["metric","value"])
    m = m[m["metric"].str.upper()!="RD"]
    m = pd.concat([m, pd.DataFrame([
        {"metric":"RD","value":rd_med},
        {"metric":"RD_median","value":rd_med},
        {"metric":"RD_mean","value":rd_mean},
        {"metric":"RD_valid_n","value":int(mask.sum())},
    ])], ignore_index=True)
    m.to_csv(metrics_csv, index=False)
    print(f"RD_median={rd_med:.4f}, RD_mean={rd_mean:.4f}, valid_n={int(mask.sum())}")

if __name__ == "__main__":
    fix_rd("./refusal_scaling/day3_generations.csv",
           "./refusal_scaling/day3_metrics.csv")
