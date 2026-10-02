"""Leakage-safe 60/20/20 split construction for Adaptive-Shot Hybrid QAD.

Split assignment happens on raw candidate records. Median/IQR scaling, IQR
filter bounds, anomaly-score quantiles, and labels are fit using TRAINING data
only, then frozen for validation and test.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_random(n: int, seed: int, train_f: float, val_f: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = int(round(train_f * n))
    nval = int(round(val_f * n))
    tr = idx[:ntr]
    va = idx[ntr:ntr+nval]
    te = idx[ntr+nval:]
    return tr, va, te


def split_group(groups: np.ndarray, seed: int, train_f: float, val_f: float, test_f: float):
    idx = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, train_size=train_f, random_state=seed)
    tr, rest = next(outer.split(idx, groups=groups))
    rest_groups = groups[rest]
    val_share_of_rest = val_f / (val_f + test_f)
    inner = GroupShuffleSplit(n_splits=1, train_size=val_share_of_rest, random_state=seed + 10000)
    va_rel, te_rel = next(inner.split(rest, groups=rest_groups))
    va, te = rest[va_rel], rest[te_rel]
    return tr, va, te


def split_temporal(times: np.ndarray, train_f: float, val_f: float):
    uniq = np.unique(times)
    uniq.sort()
    if len(uniq) < 5:
        raise RuntimeError(f"Too few unique time bins: {len(uniq)}")
    i1 = max(1, min(len(uniq)-2, int(round(train_f * len(uniq)))))
    i2 = max(i1+1, min(len(uniq)-1, int(round((train_f + val_f) * len(uniq)))))
    train_times = uniq[:i1]
    val_times = uniq[i1:i2]
    test_times = uniq[i2:]
    tr = np.where(np.isin(times, train_times))[0]
    va = np.where(np.isin(times, val_times))[0]
    te = np.where(np.isin(times, test_times))[0]
    return tr, va, te


def fit_apply_preprocess(
    X: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    iqr_k: float,
    q_susp: float,
    q_high: float,
):
    Xtr_raw = X[tr_idx]
    med = np.median(Xtr_raw, axis=0)
    q1_raw = np.quantile(Xtr_raw, 0.25, axis=0)
    q3_raw = np.quantile(Xtr_raw, 0.75, axis=0)
    scale_iqr = q3_raw - q1_raw
    scale_iqr = np.where(scale_iqr == 0, 1.0, scale_iqr)

    def scale(A):
        return (A - med) / scale_iqr

    Xtr = scale(Xtr_raw)
    Xva = scale(X[va_idx])
    Xte = scale(X[te_idx])

    q1 = np.quantile(Xtr, 0.25, axis=0)
    q3 = np.quantile(Xtr, 0.75, axis=0)
    fiqr = q3 - q1
    fiqr = np.where(fiqr == 0, 1.0, fiqr)
    lo, hi = q1 - iqr_k * fiqr, q3 + iqr_k * fiqr

    def retain(A):
        return np.all((A >= lo) & (A <= hi), axis=1)

    mtr, mva, mte = retain(Xtr), retain(Xva), retain(Xte)
    Xtr, Xva, Xte = Xtr[mtr], Xva[mva], Xte[mte]
    tr_keep, va_keep, te_keep = tr_idx[mtr], va_idx[mva], te_idx[mte]

    score_tr = np.linalg.norm(Xtr, axis=1)
    t_susp = float(np.quantile(score_tr, q_susp))
    t_high = float(np.quantile(score_tr, q_high))

    def labels(A):
        s = np.linalg.norm(A, axis=1)
        y3 = np.zeros(len(A), dtype=np.int64)
        y3[(s >= t_susp) & (s < t_high)] = 1
        y3[s >= t_high] = 2
        yb = (s >= t_susp).astype(np.int64)
        return yb, y3, s

    ytr, ytr3, str_ = labels(Xtr)
    yva, yva3, sva = labels(Xva)
    yte, yte3, ste = labels(Xte)

    for name, y in (("train", ytr), ("validation", yva), ("test", yte)):
        if len(y) == 0 or np.unique(y).size < 2:
            raise RuntimeError(f"{name} partition is empty or lacks both binary classes after filtering")

    params = {
        "median": med,
        "scale_iqr": scale_iqr,
        "filter_q1": q1,
        "filter_q3": q3,
        "filter_iqr": fiqr,
        "filter_lo": lo,
        "filter_hi": hi,
        "q_susp": q_susp,
        "q_high": q_high,
        "t_susp": t_susp,
        "t_high": t_high,
    }
    return (Xtr, ytr, ytr3, str_, tr_keep), (Xva, yva, yva3, sva, va_keep), (Xte, yte, yte3, ste, te_keep), params


def _save(path: str, protocol: str, split_seed, X, times, groups, feature_cols, tr, va, te, params, raw_counts):
    Xtr, ytr, ytr3, str_, itr = tr
    Xva, yva, yva3, sva, iva = va
    Xte, yte, yte3, ste, ite = te
    meta = {
        "protocol": protocol,
        "split_seed": split_seed,
        "nominal_counts": raw_counts,
        "retained_counts": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "label_counts": {
            "train": np.bincount(ytr, minlength=2).tolist(),
            "val": np.bincount(yva, minlength=2).tolist(),
            "test": np.bincount(yte, minlength=2).tolist(),
        },
        "training_only_preprocessing": True,
        "training_only_pseudolabel_thresholds": True,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        X_train=Xtr.astype(np.float32), y_train=ytr, y3_train=ytr3, score_train=str_, idx_train=itr,
        X_val=Xva.astype(np.float32), y_val=yva, y3_val=yva3, score_val=sva, idx_val=iva,
        X_test=Xte.astype(np.float32), y_test=yte, y3_test=yte3, score_test=ste, idx_test=ite,
        id_time_train=times[itr], id_time_val=times[iva], id_time_test=times[ite],
        group_key_train=groups[itr], group_key_val=groups[iva], group_key_test=groups[ite],
        feature_cols=feature_cols,
        median=params["median"], scale_iqr=params["scale_iqr"],
        filter_q1=params["filter_q1"], filter_q3=params["filter_q3"], filter_iqr=params["filter_iqr"],
        filter_lo=params["filter_lo"], filter_hi=params["filter_hi"],
        q_susp=params["q_susp"], q_high=params["q_high"], t_susp=params["t_susp"], t_high=params["t_high"],
        meta=json.dumps(meta),
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/adaptive_shots.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["experiment"]
    dcfg, scfg = cfg["data"], cfg["split"]
    d = np.load(str(dcfg["candidate_npz"]), allow_pickle=True)
    X = np.asarray(d["X_raw"], dtype=np.float64)
    times = np.asarray(d["id_time"], dtype=np.int64)
    groups = np.asarray(d["group_key"], dtype=str)
    feature_cols = np.asarray(d["feature_cols"], dtype=str)

    tf, vf, ef = float(scfg["train_fraction"]), float(scfg["val_fraction"]), float(scfg["test_fraction"])
    if abs(tf + vf + ef - 1.0) > 1e-9:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")
    outdir = str(scfg["output_dir"])
    seeds = [int(s) for s in scfg["seeds"]]
    iqr_k, q_susp, q_high = float(scfg["iqr_k"]), float(scfg["q_susp"]), float(scfg["q_high"])

    manifest = {"candidate_npz": str(dcfg["candidate_npz"]), "N": len(X), "splits": []}
    for protocol in scfg["protocols"]:
        protocol = str(protocol).lower()
        jobs = seeds if protocol in ("random", "group") else [None]
        for seed in jobs:
            if protocol == "random":
                tri, vai, tei = split_random(len(X), int(seed), tf, vf)
                fname = f"random_seed{seed}.npz"
            elif protocol == "group":
                tri, vai, tei = split_group(groups, int(seed), tf, vf, ef)
                fname = f"group_seed{seed}.npz"
            elif protocol == "temporal":
                tri, vai, tei = split_temporal(times, tf, vf)
                fname = "temporal.npz"
            else:
                raise ValueError(f"Unknown protocol: {protocol}")

            raw_counts = {"train": len(tri), "val": len(vai), "test": len(tei)}
            tr, va, te, params = fit_apply_preprocess(X, tri, vai, tei, iqr_k, q_susp, q_high)
            path = os.path.join(outdir, fname)
            meta = _save(path, protocol, seed, X, times, groups, feature_cols, tr, va, te, params, raw_counts)

            if protocol == "group":
                a, b, c = set(groups[tr[4]]), set(groups[va[4]]), set(groups[te[4]])
                if a & b or a & c or b & c:
                    raise RuntimeError("Group leakage detected after filtering")
            if protocol == "temporal":
                if not (times[tr[4]].max() < times[va[4]].min() <= times[va[4]].max() < times[te[4]].min()):
                    raise RuntimeError("Temporal ordering invariant failed")

            manifest["splits"].append({"path": path.replace("\\", "/"), **meta})
            print(f"[splits] {protocol} seed={seed}: {meta['retained_counts']} | labels={meta['label_counts']}")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[splits] Saved manifest to {os.path.join(outdir, 'manifest.json')}")


if __name__ == "__main__":
    main()
