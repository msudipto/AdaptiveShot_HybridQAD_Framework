"""Prepare the raw candidate pool for Adaptive-Shot Hybrid QAD.

This module performs a memory-safe two-pass uniform sample without replacement
from CESNET-TimeSeries24 aggregate CSV rows. Sampling occurs BEFORE scaling,
outlier filtering, and pseudo-label construction.

The output contains raw 12-dimensional features plus time/entity metadata.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_col(c: object) -> str:
    return str(c).strip().lstrip("\ufeff")


def discover_files(pattern: str) -> List[str]:
    files = sorted(set(glob.glob(pattern, recursive=True)))
    if not files:
        raise FileNotFoundError(
            f"No CESNET CSV files matched: {pattern}\n"
            "Extract CESNET-TimeSeries24 under data/raw/cesnet_timeseries24 first."
        )
    return files


def _header_map(path: str) -> Dict[str, str]:
    cols = pd.read_csv(path, nrows=0).columns
    return {_norm_col(c): c for c in cols}


def _resolve_columns(path: str, feature_cols: List[str]) -> Tuple[List[str], str | None]:
    cmap = _header_map(path)
    missing = [c for c in feature_cols if c not in cmap]
    if missing:
        return [], None
    time_name = None
    for cand in ("id_time", "time", "timestamp", "idTime", "id_time_bin"):
        if cand in cmap:
            time_name = cmap[cand]
            break
    usecols = [cmap[c] for c in feature_cols]
    if time_name is not None:
        usecols = [time_name] + usecols
    return usecols, time_name


def _iter_eligible_chunks(
    path: str,
    feature_cols: List[str],
    chunksize: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (X_raw, id_time) arrays for rows with complete numeric features."""
    usecols, time_actual = _resolve_columns(path, feature_cols)
    if not usecols:
        return
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk.columns = [_norm_col(c) for c in chunk.columns]
        # Normalize time column name if present.
        if "id_time" not in chunk.columns:
            for alt in ("time", "timestamp", "idTime", "id_time_bin"):
                if alt in chunk.columns:
                    chunk = chunk.rename(columns={alt: "id_time"})
                    break
        if "id_time" not in chunk.columns:
            chunk["id_time"] = -1

        Xdf = chunk[feature_cols].copy()
        for c in feature_cols:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
        mask = ~Xdf.isna().any(axis=1)
        if not bool(mask.any()):
            continue
        X = Xdf.loc[mask].to_numpy(dtype=np.float64)
        t = pd.to_numeric(chunk.loc[mask, "id_time"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        yield X, t


def count_eligible(files: List[str], feature_cols: List[str], chunksize: int) -> Tuple[List[int], int]:
    counts: List[int] = []
    total = 0
    for i, fp in enumerate(files, 1):
        n = 0
        try:
            for X, _ in _iter_eligible_chunks(fp, feature_cols, chunksize):
                n += int(len(X))
        except Exception as e:
            print(f"[prepare] WARN skipping {fp}: {type(e).__name__}: {e}")
            n = 0
        counts.append(n)
        total += n
        if i == 1 or i % 100 == 0 or i == len(files):
            print(f"[prepare] Pass 1: {i}/{len(files)} files | eligible={total}")
    return counts, total


def sample_candidates(
    files: List[str],
    counts: List[int],
    feature_cols: List[str],
    chunksize: int,
    selected_global: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Second pass selecting prechosen global eligible-row positions."""
    selected_global = np.asarray(selected_global, dtype=np.int64)
    selected_global.sort()
    X_parts: List[np.ndarray] = []
    t_parts: List[np.ndarray] = []
    g_parts: List[np.ndarray] = []
    f_parts: List[np.ndarray] = []

    global_offset = 0
    ptr = 0
    for file_idx, (fp, n_file) in enumerate(zip(files, counts), 1):
        if n_file <= 0:
            continue
        file_start = global_offset
        file_end = file_start + n_file
        global_offset = file_end

        if ptr >= len(selected_global) or selected_global[ptr] >= file_end:
            continue

        rel_path = Path(fp).as_posix()
        group_key = f"ip_addresses_sample:{Path(fp).stem}"
        local_offset = 0
        for X, t in _iter_eligible_chunks(fp, feature_cols, chunksize):
            chunk_start = file_start + local_offset
            chunk_end = chunk_start + len(X)
            local_offset += len(X)

            lo = np.searchsorted(selected_global, chunk_start, side="left", sorter=None)
            hi = np.searchsorted(selected_global, chunk_end, side="left", sorter=None)
            if hi <= lo:
                continue
            idx = selected_global[lo:hi] - chunk_start
            X_parts.append(X[idx])
            t_parts.append(t[idx])
            g_parts.append(np.full(len(idx), group_key, dtype=object))
            f_parts.append(np.full(len(idx), rel_path, dtype=object))
            ptr = hi

        if file_idx == 1 or file_idx % 100 == 0 or file_idx == len(files):
            got = sum(len(x) for x in X_parts)
            print(f"[prepare] Pass 2: {file_idx}/{len(files)} files | selected={got}")

    if not X_parts:
        raise RuntimeError("Candidate selection produced no rows.")
    X = np.vstack(X_parts).astype(np.float64)
    t = np.concatenate(t_parts).astype(np.int64)
    g = np.concatenate(g_parts).astype(str)
    sf = np.concatenate(f_parts).astype(str)
    return X, t, g, sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/adaptive_shots.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)["experiment"]
    dcfg = cfg["data"]
    out = str(dcfg["candidate_npz"])
    if os.path.exists(out) and not args.overwrite:
        print(f"[prepare] Existing candidate pool found: {out} (use --overwrite to rebuild)")
        return

    feature_cols = list(dcfg["feature_cols"])
    files = discover_files(str(dcfg["glob"]))
    chunksize = int(dcfg.get("chunksize", 100000))
    target = int(dcfg.get("candidate_size", 4875))
    seed = int(dcfg.get("sampling_seed", 42))

    print(f"[prepare] Discovered {len(files)} files")
    counts, total = count_eligible(files, feature_cols, chunksize)
    if total < target:
        raise RuntimeError(f"Only {total} eligible rows exist, fewer than target={target}")

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=target, replace=False).astype(np.int64))
    X, t, g, sf = sample_candidates(files, counts, feature_cols, chunksize, selected)
    if len(X) != target:
        raise RuntimeError(f"Expected {target} selected rows, obtained {len(X)}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {
        "sampling": "two-pass uniform without replacement over eligible rows",
        "sampling_seed": seed,
        "candidate_size": target,
        "eligible_rows": int(total),
        "n_source_files": int(len(files)),
        "source_glob": str(dcfg["glob"]),
        "pre_scaling": True,
        "pre_filtering": True,
        "pre_labeling": True,
    }
    np.savez(
        out,
        X_raw=X,
        id_time=t,
        group_key=g,
        source_file=sf,
        feature_cols=np.array(feature_cols, dtype=str),
        selected_global_indices=selected,
        meta=json.dumps(meta),
    )
    print(f"[prepare] Saved {len(X)} raw candidates to {out}")
    print(f"[prepare] Unique groups={len(np.unique(g))} | unique times={len(np.unique(t))}")


if __name__ == "__main__":
    main()
