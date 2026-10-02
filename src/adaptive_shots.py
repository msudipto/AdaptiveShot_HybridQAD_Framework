"""Adaptive-shot inference for Adaptive-Shot Hybrid QAD.

Key properties
--------------
* Reuses analytically trained QNN checkpoints; no shot-specific retraining.
* Reconstructs the exact joint Z-basis distribution of output qubits (0,1)
  from the exact three-wire backward causal support and validates it against
  saved full-QNN analytic probabilities.
* Calibrates finite-shot error margins on VALIDATION data only.
* Acquires cumulative shots 128 -> 256 -> 512 -> 1024 and stops early when
  the current anomaly probability is safely separated from the fixed
  validation-selected threshold.
* Includes fixed-shot baselines and a budget-shuffled control with the same
  realized shot histogram as the primary adaptive policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pennylane as qml
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .qnn_model import VQCClassifier


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_path(cfg: Dict, protocol: str, seed: int) -> str:
    base = str(cfg["split"]["output_dir"])
    if protocol == "random": return os.path.join(base, f"random_seed{seed}.npz")
    if protocol == "group": return os.path.join(base, f"group_seed{seed}.npz")
    if protocol == "temporal": return os.path.join(base, "temporal.npz")
    raise ValueError(protocol)


def source_dir(cfg: Dict, protocol: str, seed: int) -> str:
    return os.path.join(str(cfg["model"]["output_dir"]), protocol, f"seed_{seed}")


def out_dir(cfg: Dict, protocol: str, seed: int) -> str:
    return os.path.join(str(cfg["adaptive_shots"]["output_dir"]), protocol, f"seed_{seed}")


def safe_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(cfg: Dict, n_features: int, weights_path: str) -> Tuple[VQCClassifier, torch.device]:
    mcfg = cfg["model"]
    device = torch.device("cpu")  # exact reconstruction/head evaluation is deliberately CPU-stable
    model = VQCClassifier(
        n_features=n_features,
        n_layers=int(mcfg["n_layers"]), n_qubits=int(mcfg["n_qubits"]),
        task="binary", n_classes=2,
        diff_method=str(mcfg.get("diff_method", "adjoint")),
        use_embedder=bool(mcfg.get("use_embedder", True)),
        embed_hidden=int(mcfg.get("embed_hidden", 64)),
        embed_layers=int(mcfg.get("embed_layers", 2)),
        angle_scale=float(mcfg.get("angle_scale", np.pi)),
        use_head=bool(mcfg.get("use_head", True)),
        head_hidden=int(mcfg.get("head_hidden", 32)),
    ).to(device)
    model.load_state_dict(safe_load(weights_path, device))
    model.eval()
    return model, device


def embedded_angles(model: VQCClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        if model.embedder is not None:
            a = model.embedder(xt)
        else:
            if xt.shape[1] < model.n_qubits:
                pad = torch.zeros((len(xt), model.n_qubits - xt.shape[1]), dtype=xt.dtype, device=device)
                a = torch.cat([xt, pad], dim=1)
            else:
                a = xt[:, :model.n_qubits]
        a = torch.tanh(a) * float(model.angle_scale)
    return a.cpu().numpy().astype(np.float64)


def make_lightcone_qnode(n_layers: int):
    dev = qml.device("default.qubit", wires=3, shots=None)

    @qml.qnode(dev, interface=None)
    def circuit(angles3, weights3):
        for i in range(3):
            qml.RY(angles3[i], wires=i)
        for l in range(n_layers):
            for i in range(3):
                qml.Rot(weights3[l, i, 0], weights3[l, i, 1], weights3[l, i, 2], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[0, 1])

    return circuit


def joint_probabilities(model: VQCClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    angles = embedded_angles(model, X, device)
    try:
        weights = model.qlayer.weights.detach().cpu().numpy().astype(np.float64)
    except AttributeError:
        # TorchLayer registers the weight tensor under the supplied weight_shapes key.
        params = dict(model.qlayer.named_parameters())
        if "weights" not in params:
            raise RuntimeError(f"Could not locate VQC weights; qlayer parameters={list(params)}")
        weights = params["weights"].detach().cpu().numpy().astype(np.float64)
    qnode = make_lightcone_qnode(model.n_layers)
    w3 = weights[:, :3, :]
    out = np.empty((len(X), 4), dtype=np.float64)
    for i, a in enumerate(angles):
        out[i] = np.asarray(qnode(a[:3], w3), dtype=np.float64)
    out /= out.sum(axis=1, keepdims=True)
    return out


def counts_to_measurements(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    S = counts.sum(axis=1)
    if np.any(S <= 0):
        raise ValueError("All count rows must contain at least one shot")
    p00, p01, p10, p11 = (counts[:, i] / S for i in range(4))
    m0 = p00 + p01 - p10 - p11
    m1 = p00 - p01 + p10 - p11
    return np.column_stack([m0, m1]).astype(np.float32)


def joint_to_measurements(joint: np.ndarray) -> np.ndarray:
    p00, p01, p10, p11 = (joint[:, i] for i in range(4))
    return np.column_stack([p00+p01-p10-p11, p00-p01+p10-p11]).astype(np.float32)


def head_probability(model: VQCClassifier, measurements: np.ndarray, device: torch.device) -> np.ndarray:
    mt = torch.tensor(measurements, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model.head(mt) if model.head is not None else mt
        p = torch.softmax(logits, dim=1)[:, 1]
    return p.cpu().numpy().astype(np.float64)


def draw_counts(rng: np.random.Generator, joint: np.ndarray, shots: np.ndarray | int) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float64)
    if np.isscalar(shots):
        sarr = np.full(len(joint), int(shots), dtype=np.int64)
    else:
        sarr = np.asarray(shots, dtype=np.int64)
    out = np.zeros((len(joint), 4), dtype=np.int64)
    for i, (p, s) in enumerate(zip(joint, sarr)):
        out[i] = rng.multinomial(int(s), p)
    return out


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins-1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if not np.any(m): continue
        ece += float(np.mean(m)) * abs(float(np.mean(p[m])) - float(np.mean(y[m])))
    return float(ece)


def metrics(y: np.ndarray, p: np.ndarray, tau: float, p_analytic: np.ndarray, shots: float, smax: int) -> Dict[str, float]:
    ya = (p_analytic >= tau).astype(int)
    yh = (p >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0,1]).ravel()
    diff = p - p_analytic
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Accuracy": float(accuracy_score(y, yh)),
        "TPR": float(tp / max(tp+fn, 1)),
        "FPR": float(fp / max(fp+tn, 1)),
        "Precision": float(precision_score(y, yh, zero_division=0)),
        "Recall": float(recall_score(y, yh, zero_division=0)),
        "F1": float(f1_score(y, yh, zero_division=0)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, yh)),
        "MCC": float(matthews_corrcoef(y, yh)) if np.unique(y).size == 2 else 0.0,
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece_score(y, p, 15),
        "ProbabilityMAE": float(np.mean(np.abs(diff))),
        "ProbabilityRMSE": float(np.sqrt(np.mean(diff**2))),
        "ProbabilityBias": float(np.mean(diff)),
        "DecisionDisagreement": float(np.mean(yh != ya)),
        "AverageShots": float(shots),
        "ShotSavingsVsMax": float(1.0 - float(shots)/float(smax)) if shots > 0 else float("nan"),
    }


def protocol_code(protocol: str) -> int:
    return {"random": 11, "group": 23, "temporal": 37}[protocol]


def calibrate_margins(
    model: VQCClassifier,
    device: torch.device,
    joint_val: np.ndarray,
    p_val: np.ndarray,
    stages: List[int],
    quantiles: List[float],
    reps: int,
    rng: np.random.Generator,
) -> Dict[float, Dict[int, float]]:
    errors: Dict[int, List[np.ndarray]] = {s: [] for s in stages[:-1]}
    for s in stages[:-1]:
        for _ in range(reps):
            c = draw_counts(rng, joint_val, s)
            ph = head_probability(model, counts_to_measurements(c), device)
            errors[s].append(np.abs(ph - p_val))
    margins: Dict[float, Dict[int, float]] = {}
    for q in quantiles:
        margins[q] = {s: float(np.quantile(np.concatenate(errors[s]), q)) for s in stages[:-1]}
    return margins


def adaptive_once(
    model: VQCClassifier,
    device: torch.device,
    joint: np.ndarray,
    tau: float,
    stages: List[int],
    deltas: Dict[int, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(joint)
    counts = np.zeros((n,4), dtype=np.int64)
    final_p = np.full(n, np.nan, dtype=np.float64)
    final_shots = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    prev = 0
    for stage in stages:
        inc = int(stage - prev)
        idx = np.where(active)[0]
        if len(idx):
            counts[idx] += draw_counts(rng, joint[idx], inc)
            ph = head_probability(model, counts_to_measurements(counts[idx]), device)
            if stage == stages[-1]:
                stop_local = np.ones(len(idx), dtype=bool)
            else:
                stop_local = np.abs(ph - tau) > float(deltas[stage])
            stop_idx = idx[stop_local]
            final_p[stop_idx] = ph[stop_local]
            final_shots[stop_idx] = stage
            active[stop_idx] = False
        prev = stage
    if np.any(active) or np.any(~np.isfinite(final_p)):
        raise RuntimeError("Adaptive policy failed to terminate all samples")
    return final_p, final_shots


def append_row(rows: List[Dict], base: Dict, policy: str, rep: int, met: Dict, alloc: np.ndarray | None, stages: List[int]):
    row = dict(base)
    row.update({"Policy": policy, "MeasurementRep": rep, **met})
    if alloc is not None:
        for s in stages:
            row[f"FracShots{s}"] = float(np.mean(alloc == s))
    else:
        for s in stages:
            row[f"FracShots{s}"] = 1.0 if policy == f"Fixed-{s}" else 0.0
    rows.append(row)


def evaluate_one(cfg: Dict, protocol: str, seed: int, overwrite: bool = False) -> None:
    acfg = cfg["adaptive_shots"]
    odir = out_dir(cfg, protocol, seed)
    os.makedirs(odir, exist_ok=True)
    csv_path = os.path.join(odir, "realizations.csv")
    if os.path.exists(csv_path) and not overwrite:
        print(f"[adaptive_shots] Skip existing {protocol} seed={seed}")
        return

    spath = split_path(cfg, protocol, seed)
    sdir = source_dir(cfg, protocol, seed)
    weights_path = os.path.join(sdir, "qnn_model.pt")
    analytic_path = os.path.join(sdir, "analytic_outputs.npz")
    if not os.path.exists(weights_path) or not os.path.exists(analytic_path):
        raise FileNotFoundError(f"Missing trained source QNN for {protocol} seed={seed}; run train_qnn first")

    sd = np.load(spath, allow_pickle=True)
    ad = np.load(analytic_path, allow_pickle=True)
    Xval, Xtest = np.asarray(sd["X_val"], np.float32), np.asarray(sd["X_test"], np.float32)
    yval, ytest = np.asarray(sd["y_val"], int), np.asarray(sd["y_test"], int)
    pval, ptest = np.asarray(ad["p_val"], float), np.asarray(ad["p_test"], float)
    tau = float(np.asarray(ad["threshold"]).item())

    model, device = build_model(cfg, Xval.shape[1], weights_path)
    print(f"[adaptive_shots] Reconstructing 3-wire joint probabilities: {protocol} seed={seed}")
    joint_val = joint_probabilities(model, Xval, device)
    joint_test = joint_probabilities(model, Xtest, device)

    pval_re = head_probability(model, joint_to_measurements(joint_val), device)
    ptest_re = head_probability(model, joint_to_measurements(joint_test), device)
    max_err = float(max(np.max(np.abs(pval_re-pval)), np.max(np.abs(ptest_re-ptest))))
    tol = float(acfg.get("reconstruction_tolerance", 1e-4))
    if max_err > tol:
        raise RuntimeError(f"3-wire reconstruction validation failed: max_abs={max_err:.6g} > {tol}")
    print(f"[adaptive_shots] Reconstruction validated: max_abs={max_err:.3g}")

    stages = [int(s) for s in acfg["stages"]]
    reps = int(acfg["measurement_reps"])
    cal_reps = int(acfg["calibration_reps"])
    quantiles = [float(q) for q in acfg["calibration_quantiles"]]
    primary = float(acfg["primary_quantile"])
    smax = max(stages)
    base_seed = int(acfg.get("random_seed_base", 73000)) + protocol_code(protocol)*1000 + seed

    cal_rng = np.random.default_rng(base_seed + 1)
    margins = calibrate_margins(model, device, joint_val, pval, stages, quantiles, cal_reps, cal_rng)
    with open(os.path.join(odir, "calibration_margins.json"), "w", encoding="utf-8") as f:
        json.dump({str(q): {str(s): v for s,v in d.items()} for q,d in margins.items()}, f, indent=2)

    rows: List[Dict] = []
    primary_alloc, primary_prob = [], []
    base = {"Protocol": protocol, "TrainingSeed": seed, "Threshold": tau, "ReconstructionMaxAbs": max_err}

    # Analytic reference is emitted once (rep=-1), not treated as a measurement realization.
    amet = metrics(ytest, ptest, tau, ptest, shots=0.0, smax=smax)
    append_row(rows, base, "Analytic", -1, amet, None, stages)

    for rep in range(reps):
        # Fixed-shot baselines.
        for s in stages:
            rng = np.random.default_rng(base_seed + 100000*rep + s)
            c = draw_counts(rng, joint_test, s)
            ph = head_probability(model, counts_to_measurements(c), device)
            met = metrics(ytest, ph, tau, ptest, shots=float(s), smax=smax)
            append_row(rows, base, f"Fixed-{s}", rep, met, None, stages)

        # Adaptive policies.
        adaptive_by_q: Dict[float, Tuple[np.ndarray,np.ndarray]] = {}
        for qi, q in enumerate(quantiles):
            rng = np.random.default_rng(base_seed + 200000*rep + 1000*qi + 17)
            ph, alloc = adaptive_once(model, device, joint_test, tau, stages, margins[q], rng)
            adaptive_by_q[q] = (ph, alloc)
            met = metrics(ytest, ph, tau, ptest, shots=float(np.mean(alloc)), smax=smax)
            append_row(rows, base, f"AS-VQC-{int(round(q*100))}", rep, met, alloc, stages)
            if abs(q-primary) < 1e-12:
                primary_alloc.append(alloc.copy())
                primary_prob.append(ph.copy())

        # Budget-shuffled control: same final shot histogram as primary AS policy,
        # but randomized across records, followed by fresh measurement sampling.
        ph_primary, alloc_primary = adaptive_by_q[primary]
        rng = np.random.default_rng(base_seed + 300000*rep + 29)
        shuffled = rng.permutation(alloc_primary)
        c = draw_counts(rng, joint_test, shuffled)
        ph = head_probability(model, counts_to_measurements(c), device)
        met = metrics(ytest, ph, tau, ptest, shots=float(np.mean(shuffled)), smax=smax)
        append_row(rows, base, f"Budget-Shuffled-AS{int(round(primary*100))}", rep, met, shuffled, stages)

    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    np.savez(
        os.path.join(odir, "primary_allocations.npz"),
        final_shots=np.stack(primary_alloc, axis=0),
        p_finite=np.stack(primary_prob, axis=0),
        p_analytic=ptest,
        y_test=ytest,
        threshold=tau,
        threshold_distance=np.abs(ptest-tau),
        stages=np.asarray(stages, dtype=int),
        primary_quantile=primary,
    )
    print(f"[adaptive_shots] Saved {len(rows)} rows to {csv_path}")


def jobs(cfg: Dict) -> Iterable[Tuple[str,int]]:
    seeds = [int(s) for s in cfg["split"]["seeds"]]
    for protocol in cfg["split"]["protocols"]:
        for seed in seeds:
            yield str(protocol).lower(), seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/adaptive_shots.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--protocol", choices=["random","group","temporal"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)["experiment"]
    if args.all:
        for protocol, seed in jobs(cfg):
            evaluate_one(cfg, protocol, seed, overwrite=args.overwrite)
    else:
        if args.protocol is None or args.seed is None:
            raise SystemExit("Use --all or provide --protocol and --seed")
        evaluate_one(cfg, args.protocol, int(args.seed), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
