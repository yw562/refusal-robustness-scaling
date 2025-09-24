#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute RD for Day3 using semantic distance between pre- and post-attack outputs.

What this script does:
1) Load Day3 pre/post CSVs.
2) Align rows by a reliable key (prefer 'id', fallback to 'prompt').
3) Compute per-item RD_i = 1 - cosine(emb(pre), emb(post)) using SBERT embeddings.
4) Save rd_per_item to 'day3_rd_per_item.csv'.
5) Backup 'day3_metrics.csv' to '.bak' and update/append RD_median, RD_mean, RD_valid_n.

It does NOT retrain or re-run the model. It only uses saved CSVs.
"""

import os
import sys
import math
import shutil
import textwrap
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Sentence-Transformers for sentence embeddings
from sentence_transformers import SentenceTransformer

# -------------- Configuration --------------
PRE_PATHS_CANDIDATES = [
    "day3_pre_generation.csv",              # common name you have
    "day3_pre_generations.csv",
]
POST_PATHS_CANDIDATES = [
    "day3_generations.csv",                 # common name you have
    "day3_post_generation.csv",
]

# If your post CSV already contains both pre and post columns, we try these names:
POSSIBLE_PRE_COLS = ["pre_text", "pre_output", "pre", "pre_response", "pre_gen"]
POSSIBLE_POST_COLS = ["post_text", "post_output", "post", "post_response", "post_gen", "output", "response"]

# Keys we try for joining pre & post:
JOIN_KEYS_PRIORITY = ["id", "sample_id", "idx", "prompt"]

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Output files
RD_PER_ITEM_OUT = "day3_rd_per_item.csv"
METRICS_IN = "day3_metrics.csv"
METRICS_OUT = "day3_metrics.csv"        # we will write back here (after making a .bak)
# -------------------------------------------


def find_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_csv_loose(path: str) -> pd.DataFrame:
    """Load a CSV with safe defaults, strip BOM, and normalize column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names: strip and lower where safe
    df.columns = [c.strip() for c in df.columns]
    return df


def choose_join_key(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> str:
    """Pick the best join key available in both dataframes."""
    pre_cols = set(c.lower() for c in pre_df.columns)
    post_cols = set(c.lower() for c in post_df.columns)
    for key in JOIN_KEYS_PRIORITY:
        if key.lower() in pre_cols and key.lower() in post_cols:
            return key
    raise KeyError(
        f"No common join key found. Tried: {JOIN_KEYS_PRIORITY}. "
        f"Pre has: {list(pre_df.columns)} | Post has: {list(post_df.columns)}"
    )


def coerce_text_series(s: pd.Series) -> pd.Series:
    """Coerce to string, strip spaces; keep None as NaN."""
    return s.astype(str).replace({"nan": np.nan}).str.strip()


def align_pre_post() -> pd.DataFrame:
    """
    Return a dataframe with columns: [join_key, pre_text, post_text, prompt?]
    Strategy:
      - If the post CSV already contains both pre and post columns, use them directly.
      - Else, read separate pre/post CSVs and inner-join on a common key.
    """
    post_path = find_existing_path(POST_PATHS_CANDIDATES)
    if post_path is None:
        raise FileNotFoundError(f"Could not find any of post CSVs: {POST_PATHS_CANDIDATES}")
    post_df = load_csv_loose(post_path)

    # Case A: single CSV contains both pre and post columns
    lower_cols = [c.lower() for c in post_df.columns]
    pre_col = next((c for c in post_df.columns if c.lower() in POSSIBLE_PRE_COLS), None)
    post_col = next((c for c in post_df.columns if c.lower() in POSSIBLE_POST_COLS), None)

    if pre_col is not None and post_col is not None and pre_col != post_col:
        # Use this single file
        print(f"[Info] Using single file with pre+post columns: {post_path}")
        df = post_df.copy()
        df["pre_text"] = coerce_text_series(df[pre_col])
        df["post_text"] = coerce_text_series(df[post_col])
        # Try to keep an identifier column if present
        join_key = None
        for k in JOIN_KEYS_PRIORITY:
            if k in df.columns:
                join_key = k
                break
        if join_key is None:
            # Fallback: create a synthetic index
            df["idx_tmp"] = np.arange(len(df))
            join_key = "idx_tmp"
        out_cols = [join_key, "pre_text", "post_text"]
        # Keep prompt if available for easier sanity checks
        if "prompt" in df.columns:
            out_cols.append("prompt")
        df = df[out_cols]
        return df

    # Case B: separate pre and post CSVs
    pre_path = find_existing_path(PRE_PATHS_CANDIDATES)
    if pre_path is None:
        raise FileNotFoundError(f"Could not find any of pre CSVs: {PRE_PATHS_CANDIDATES}")

    print(f"[Info] Using separate files:\n  pre : {pre_path}\n  post: {post_path}")
    pre_df = load_csv_loose(pre_path)

    # Heuristics to pick output columns
    def guess_out_col(df: pd.DataFrame) -> str:
        for cand in ["output", "response", "text", "completion"]:
            if cand in df.columns:
                return cand
            # also try lower-cased match
            for c in df.columns:
                if c.lower() == cand:
                    return c
        # If nothing found, fallback to last column (often the output)
        return df.columns[-1]

    pre_out = guess_out_col(pre_df)
    post_out = guess_out_col(post_df)

    join_key = choose_join_key(pre_df, post_df)

    # Normalize prompt if using prompt as key (trim whitespace)
    if join_key.lower() == "prompt":
        pre_df["prompt"] = coerce_text_series(pre_df["prompt"])
        post_df["prompt"] = coerce_text_series(post_df["prompt"])

    merged = pd.merge(
        pre_df, post_df,
        on=join_key, how="inner", suffixes=("_pre", "_post")
    )

    print(f"[Info] Joined on '{join_key}' with inner join. "
          f"Pairs: {len(merged)} / pre={len(pre_df)} post={len(post_df)}")

    merged["pre_text"] = coerce_text_series(merged[pre_out + "_pre"] if pre_out + "_pre" in merged.columns else merged[pre_out])
    merged["post_text"] = coerce_text_series(merged[post_out + "_post"] if post_out + "_post" in merged.columns else merged[post_out])

    out_cols = [join_key, "pre_text", "post_text"]
    if "prompt" in merged.columns and join_key != "prompt":
        out_cols.append("prompt")
    merged = merged[out_cols]
    return merged


def compute_embeddings(texts: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts to L2-normalized embeddings."""
    # Truncate extremely long texts to avoid memory spikes but keep semantics
    texts_proc = [(t if isinstance(t, str) else "")[:2048] for t in texts]
    embs = model.encode(
        texts_proc,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes cosine = dot product
    )
    return embs


def cosine_distance_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine distance for normalized vectors: 1 - (a · b)."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    dots = np.sum(a * b, axis=1)  # since normalized, this is cosine similarity
    dist = 1.0 - np.clip(dots, -1.0, 1.0)
    return dist


def main():
    # Step 1: Align pre & post
    df = align_pre_post()
    # Drop rows with missing texts
    before = len(df)
    df = df.dropna(subset=["pre_text", "post_text"]).copy()
    after = len(df)
    print(f"[Info] Dropped empty-text rows: {before - after} (kept {after})")

    if after == 0:
        print("[Warn] No valid pairs to compute RD. Exiting with RD=0 and rd_valid_n=0.")
        rd_median = 0.0
        rd_mean = 0.0
        rd_valid_n = 0
        # Still try to update metrics file
        update_metrics(rd_median, rd_mean, rd_valid_n)
        sys.exit(0)

    # Step 2: Compute embeddings
    print(f"[Info] Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[Info] Encoding pre texts...")
    pre_emb = compute_embeddings(df["pre_text"].tolist(), model)
    print("[Info] Encoding post texts...")
    post_emb = compute_embeddings(df["post_text"].tolist(), model)

    # Step 3: Row-wise cosine distance
    rd_i = cosine_distance_rowwise(pre_emb, post_emb)
    df_out = df.copy()
    df_out["rd_i"] = rd_i

    # Step 4: Save per-item RD
    preview_cols = [c for c in df_out.columns if c in ("id", "sample_id", "idx", "prompt")]
    preview_cols += ["rd_i"]
    # Keep only a light preview of texts to keep CSV small (optional)
    # If you want full texts, comment the two lines below.
    df_out["pre_preview"] = df_out["pre_text"].str.slice(0, 160)
    df_out["post_preview"] = df_out["post_text"].str.slice(0, 160)
    preview_cols += ["pre_preview", "post_preview"]

    df_out[preview_cols].to_csv(RD_PER_ITEM_OUT, index=False)
    print(f"[OK] Saved per-item RD to: {RD_PER_ITEM_OUT}")

    # Step 5: Aggregate
    rd_median = float(np.median(rd_i))
    rd_mean = float(np.mean(rd_i))
    rd_valid_n = int(len(rd_i))

    print(f"[OK] RD_median={rd_median:.4f} | RD_mean={rd_mean:.4f} | RD_valid_n={rd_valid_n}")

    # Step 6: Update metrics
    update_metrics(rd_median, rd_mean, rd_valid_n)
    print("[OK] Metrics updated.")


def update_metrics(rd_median: float, rd_mean: float, rd_valid_n: int):
    """Backup and update day3_metrics.csv with new RD fields."""
    if os.path.exists(METRICS_IN):
        backup = METRICS_IN + ".bak"
        shutil.copyfile(METRICS_IN, backup)
        print(f"[Info] Backed up metrics to: {backup}")

        try:
            mdf = pd.read_csv(METRICS_IN)
            if len(mdf) >= 1:
                # Update first row (common case: single-row metrics table)
                mdf.loc[0, "RD_median"] = rd_median
                mdf.loc[0, "RD_mean"] = rd_mean
                mdf.loc[0, "RD_valid_n"] = rd_valid_n
                # For compatibility: if there is a legacy 'RD' column, set it to median
                if "RD" in mdf.columns:
                    mdf.loc[0, "RD"] = rd_median
                mdf.to_csv(METRICS_OUT, index=False)
                return
        except Exception as e:
            warnings.warn(f"Failed to parse existing metrics CSV; writing a new one. Error: {e}")

    # If there was no metrics file or parsing failed, write a fresh one
    mdf = pd.DataFrame([{
        "RD_median": rd_median,
        "RD_mean": rd_mean,
        "RD_valid_n": rd_valid_n,
    }])
    mdf.to_csv(METRICS_OUT, index=False)


if __name__ == "__main__":
    main()
