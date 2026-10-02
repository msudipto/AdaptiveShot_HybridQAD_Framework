"""Hierarchical aggregation for Adaptive-Shot Hybrid QAD adaptive-shot experiments.

Measurement repetitions are first averaged within each trained checkpoint.
Only then are mean and sample SD computed across the five trained seeds.
Paired bootstrap intervals use the five seed-level paired differences.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import yaml

METRICS = [
    "AUC", "AP", "Accuracy", "TPR", "FPR", "Precision", "Recall", "F1",
    "BalancedAccuracy", "MCC", "Brier", "ECE", "ProbabilityMAE",
    "ProbabilityRMSE", "ProbabilityBias", "DecisionDisagreement",
    "AverageShots", "ShotSavingsVsMax", "FracShots128", "FracShots256",
    "FracShots512", "FracShots1024",
]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_rows(pattern: str) -> List[Dict[str,str]]:
    rows: List[Dict[str,str]] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def fval(row: Dict[str,str], key: str) -> float:
    try: return float(row[key])
    except Exception: return float("nan")


def seed_level(rows: List[Dict[str,str]]) -> List[Dict[str,float|str|int]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["Protocol"], int(r["TrainingSeed"]), r["Policy"])].append(r)
    out = []
    for (protocol, seed, policy), rr in sorted(grouped.items()):
        # Analytic is a single reference row; all measurement policies use reps >= 0.
        use = rr if policy == "Analytic" else [r for r in rr if int(float(r["MeasurementRep"])) >= 0]
        rec: Dict[str,float|str|int] = {"Protocol": protocol, "TrainingSeed": seed, "Policy": policy}
        for m in METRICS:
            vals = np.array([fval(r,m) for r in use], dtype=float)
            vals = vals[np.isfinite(vals)]
            rec[m] = float(np.mean(vals)) if len(vals) else float("nan")
        out.append(rec)
    return out


def aggregate_seed_level(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["Protocol"], r["Policy"])].append(r)
    out = []
    for (protocol, policy), rr in sorted(grouped.items()):
        rec = {"Protocol": protocol, "Policy": policy, "NSeeds": len(rr)}
        for m in METRICS:
            vals = np.array([float(r[m]) for r in rr], dtype=float)
            vals = vals[np.isfinite(vals)]
            rec[f"{m}_Mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            rec[f"{m}_SD"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        out.append(rec)
    return out


def paired_bootstrap(seed_rows: List[Dict], cfg: Dict) -> List[Dict]:
    by = {(r["Protocol"], int(r["TrainingSeed"]), r["Policy"]): r for r in seed_rows}
    protocols = sorted({r["Protocol"] for r in seed_rows})
    comparisons = [
        ("AS-VQC-95", "Fixed-1024"),
        ("AS-VQC-95", "Fixed-128"),
        ("AS-VQC-95", "Budget-Shuffled-AS95"),
    ]
    report_metrics = ["AUC", "FPR", "Brier", "ECE", "DecisionDisagreement", "AverageShots"]
    nboot = int(cfg["aggregation"].get("bootstrap_reps", 10000))
    rng = np.random.default_rng(int(cfg["aggregation"].get("bootstrap_seed", 2026)))
    out = []
    for protocol in protocols:
        seeds = sorted({int(r["TrainingSeed"]) for r in seed_rows if r["Protocol"] == protocol})
        for a,b in comparisons:
            if not all((protocol,s,a) in by and (protocol,s,b) in by for s in seeds):
                continue
            for metric in report_metrics:
                diff = np.array([float(by[(protocol,s,a)][metric]) - float(by[(protocol,s,b)][metric]) for s in seeds])
                boots = np.empty(nboot, dtype=float)
                for j in range(nboot):
                    idx = rng.integers(0, len(diff), size=len(diff))
                    boots[j] = float(np.mean(diff[idx]))
                out.append({
                    "Protocol": protocol, "PolicyA": a, "PolicyB": b, "Metric": metric,
                    "MeanDifference_AminusB": float(np.mean(diff)),
                    "CI95_Lo": float(np.percentile(boots, 2.5)),
                    "CI95_Hi": float(np.percentile(boots, 97.5)),
                    "NPairedSeeds": len(diff), "BootstrapReps": nboot,
                })
    return out


def write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows: return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="config/adaptive_shots.yaml")
    args=ap.parse_args()
    cfg=load_config(args.config)["experiment"]
    root=str(cfg["adaptive_shots"]["output_dir"])
    rows=read_rows(os.path.join(root,"**","realizations.csv"))
    if not rows:
        raise FileNotFoundError(f"No realization CSVs found under {root}")
    seed=seed_level(rows)
    summary=aggregate_seed_level(seed)
    paired=paired_bootstrap(seed,cfg)
    out=str(cfg["aggregation"]["output_dir"])
    write_csv(os.path.join(out,"seed_level_metrics.csv"),seed)
    write_csv(os.path.join(out,"summary_mean_sd.csv"),summary)
    write_csv(os.path.join(out,"paired_bootstrap.csv"),paired)
    manifest={
        "n_realization_rows":len(rows), "n_seed_level_rows":len(seed),
        "n_summary_rows":len(summary), "n_paired_rows":len(paired),
        "hierarchy":"measurement repetitions averaged within training seed before across-seed mean/SD",
    }
    os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,"aggregation_manifest.json"),"w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2)
    print(f"[aggregate] {manifest}")
    print(f"[aggregate] Summary written to {out}")


if __name__ == "__main__":
    main()
