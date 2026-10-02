"""Train leakage-safe source QNN checkpoints for Adaptive-Shot Hybrid QAD.

The test partition is never used for optimization, checkpoint selection, or
threshold selection. Validation ROC-AUC selects the checkpoint; validation
Youden J selects the operating threshold. Analytic validation/test probabilities
are saved for later adaptive-shot calibration and evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .qnn_model import VQCClassifier


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def split_path(cfg: Dict, protocol: str, seed: int) -> str:
    out = cfg["split"]["output_dir"]
    if protocol == "random":
        return os.path.join(out, f"random_seed{seed}.npz")
    if protocol == "group":
        return os.path.join(out, f"group_seed{seed}.npz")
    if protocol == "temporal":
        return os.path.join(out, "temporal.npz")
    raise ValueError(protocol)


def run_dir(cfg: Dict, protocol: str, seed: int) -> str:
    return os.path.join(str(cfg["model"]["output_dir"]), protocol, f"seed_{seed}")


def pick_device(prefer_cuda: bool) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for (bx,) in loader:
            out.append(torch.exp(model(bx.to(device))).cpu().numpy())
    probs = np.concatenate(out, axis=0)
    return probs[:, 1] if probs.shape[1] == 2 else probs[:, 1:].sum(axis=1)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def train_one(cfg: Dict, protocol: str, seed: int, overwrite: bool = False) -> None:
    spath = split_path(cfg, protocol, seed)
    if not os.path.exists(spath):
        raise FileNotFoundError(f"Missing Adaptive-Shot Hybrid QAD split: {spath}")
    odir = run_dir(cfg, protocol, seed)
    weights_path = os.path.join(odir, "qnn_model.pt")
    outputs_path = os.path.join(odir, "analytic_outputs.npz")
    if os.path.exists(weights_path) and os.path.exists(outputs_path) and not overwrite:
        print(f"[train] Skip existing {protocol} seed={seed}")
        return
    os.makedirs(odir, exist_ok=True)

    d = np.load(spath, allow_pickle=True)
    Xtr, ytr = np.asarray(d["X_train"], np.float32), np.asarray(d["y_train"], np.int64)
    Xva, yva = np.asarray(d["X_val"], np.float32), np.asarray(d["y_val"], np.int64)
    Xte, yte = np.asarray(d["X_test"], np.float32), np.asarray(d["y_test"], np.int64)

    mcfg = cfg["model"]
    set_seed(seed)
    device = pick_device(bool(mcfg.get("prefer_cuda", False)))
    model = VQCClassifier(
        n_features=Xtr.shape[1],
        n_layers=int(mcfg["n_layers"]),
        n_qubits=int(mcfg["n_qubits"]),
        task="binary",
        n_classes=2,
        diff_method=str(mcfg.get("diff_method", "adjoint")),
        use_embedder=bool(mcfg.get("use_embedder", True)),
        embed_hidden=int(mcfg.get("embed_hidden", 64)),
        embed_layers=int(mcfg.get("embed_layers", 2)),
        angle_scale=float(mcfg.get("angle_scale", np.pi)),
        use_head=bool(mcfg.get("use_head", True)),
        head_hidden=int(mcfg.get("head_hidden", 32)),
    ).to(device)

    # PennyLane backends can force a CPU fallback even if torch CUDA exists.
    try:
        with torch.no_grad():
            _ = model(torch.zeros((2, Xtr.shape[1]), dtype=torch.float32, device=device))
    except Exception as e:
        if device.type == "cuda":
            print(f"[train] CUDA path failed ({type(e).__name__}); using CPU")
            device = torch.device("cpu")
            model = model.to(device)
        else:
            raise

    batch_size = int(mcfg.get("batch_size", 32))
    eval_bs = int(mcfg.get("eval_batch_size", 64))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )

    counts = np.bincount(ytr, minlength=2).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    criterion = nn.NLLLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=float(mcfg["lr"]), weight_decay=float(mcfg["weight_decay"]))
    grad_clip = float(mcfg.get("grad_clip_norm", 0.0))
    epochs = int(mcfg["epochs"])

    best_auc, best_epoch, best_state = -np.inf, 0, None
    history = []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            lp = model(bx)
            loss = criterion(lp, by)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            loss_sum += float(loss.item()) * len(bx)
            n_seen += len(bx)

        pva = predict(model, Xva, device, eval_bs)
        auc = float(roc_auc_score(yva, pva))
        ap = float(average_precision_score(yva, pva))
        dt = time.perf_counter() - t0
        rec = {"epoch": ep, "train_loss": loss_sum/max(n_seen,1), "val_auc": auc, "val_ap": ap, "time_sec": dt}
        history.append(rec)
        print(f"[train] {protocol} seed={seed} ep={ep:02d}/{epochs} loss={rec['train_loss']:.4f} val_auc={auc:.5f} time={dt:.1f}s")
        if auc > best_auc:  # first checkpoint retained on exact tie
            best_auc, best_epoch, best_state = auc, ep, deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    torch.save(best_state, weights_path)

    pva = predict(model, Xva, device, eval_bs)
    pte = predict(model, Xte, device, eval_bs)
    tau = youden_threshold(yva, pva)
    np.savez(
        outputs_path,
        p_val=pva, y_val=yva,
        p_test=pte, y_test=yte,
        threshold=tau,
        protocol=protocol,
        seed=seed,
        split_path=spath,
    )
    info = model.model_info()
    info.update({
        "protocol": protocol,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "val_threshold_youden": tau,
        "split_path": spath,
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "test_auc_analytic": float(roc_auc_score(yte, pte)),
        "test_ap_analytic": float(average_precision_score(yte, pte)),
        "test_not_used_for_selection": True,
    })
    with open(os.path.join(odir, "qnn_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(odir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Saved {protocol} seed={seed}: best_val_auc={best_auc:.5f}, tau={tau:.6f}")


def jobs(cfg: Dict) -> Iterable[Tuple[str, int]]:
    seeds = [int(s) for s in cfg["split"]["seeds"]]
    for protocol in cfg["split"]["protocols"]:
        for seed in seeds:
            yield str(protocol).lower(), seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/adaptive_shots.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--protocol", choices=["random", "group", "temporal"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)["experiment"]
    if args.all:
        for protocol, seed in jobs(cfg):
            train_one(cfg, protocol, seed, overwrite=args.overwrite)
    else:
        if args.protocol is None or args.seed is None:
            raise SystemExit("Use --all or provide --protocol and --seed")
        train_one(cfg, args.protocol, int(args.seed), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
