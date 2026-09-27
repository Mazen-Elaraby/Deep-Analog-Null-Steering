#!/usr/bin/env python3
"""
Training for HybridSARRefinerV2 on dataset v2.

The archived optimizer and schedule are retained -- AdamW, lr 1e-2, weight decay 1e-4,
ExponentialLR(gamma=0.87), 50 epochs, batch 256, plain MSE on the trig 4-vector
-- so the published checkpoint can be reproduced from the documented recipe.
`--scheduler plateau` is available, but ExponentialLR is the published default.

Two data-handling details are load-bearing:

1. SCHEMA. v2 ships columnar .npz rather than a MATLAB struct array, and each
   sample holds up to 3 jammers. The model consumes one sector at a time, so a
   J-jammer sample expands into J training examples: query q uses sector q's 8
   probes and is supervised by jammer q's DoA. Leakage from the other jammers is
   already in those probe readings.

2. SPLITTING. The split is over SAMPLES, not over expanded queries. Both queries
   of a 2-jammer scene must land in the same split, otherwise the same scene
   appears in train and validation and the validation loss is optimistic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "models"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from model_builder_v2 import HybridSARRefinerV2  # noqa: E402
import crpa_physics as phys  # noqa: E402

N_PROBES = 8


def _trig(doa_deg: np.ndarray) -> np.ndarray:
    """[..., 2] degrees -> [..., 4] as [cos th, sin th, cos ph, sin ph]."""
    th = np.deg2rad(doa_deg[..., 0])
    ph = np.deg2rad(doa_deg[..., 1])
    return np.stack([np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)], axis=-1)


def expand_queries(path: Path, noise_var: float | None = None):
    """
    Flatten an .npz split into per-query training examples.

    Returns (tensors, sample_index) where sample_index[k] is the originating
    sample of example k, which is what the split is grouped by.
    """
    d = np.load(path)
    noise_var = float(d["noise_power_w"]) if noise_var is None else noise_var

    active = d["sector_ids"] > 0                                   # (N, 3)
    rows, slots = np.nonzero(active)

    p = d["probe_powers_w"][rows, slots]                           # (Q, 8)
    sectors = d["sector_ids"][rows, slots].astype(np.int64) - 1     # (Q,)
    doa = d["doa_deg"][rows, slots]                                # (Q, 2)

    # t3.py preprocessing: log10(p) - log10(sum p). Scale-free by construction.
    p_safe = p + 1e-20
    log_p = np.log10(p_safe)
    log_total = np.log10(p_safe.sum(axis=1, keepdims=True))
    norm_p = log_p - log_total

    # #2: the absolute scale t3.py discarded, referenced to the known noise
    # floor so the feature is O(1) and physically interpretable.
    log_scale = (log_total[:, 0] - np.log10(noise_var)).astype(np.float32)

    centers = d["sector_centers_deg"][sectors]                      # (Q, 8, 2)
    centers_trig = _trig(centers)
    sub_ids = sectors[:, None] * N_PROBES + np.arange(N_PROBES)[None, :]

    tensors = (
        torch.tensor(norm_p, dtype=torch.float32),
        torch.tensor(sub_ids, dtype=torch.long),
        torch.tensor(centers_trig, dtype=torch.float32),
        torch.tensor(log_scale, dtype=torch.float32),
        torch.tensor(_trig(doa), dtype=torch.float32),
    )
    return tensors, rows


def split_by_sample(sample_idx: np.ndarray, val_frac=0.1, test_frac=0.1, seed=42):
    """Group-wise split so no scene straddles two splits."""
    uniq = np.unique(sample_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    n_test = int(len(uniq) * test_frac)
    sets = {
        "val": set(uniq[:n_val].tolist()),
        "test": set(uniq[n_val:n_val + n_test].tolist()),
    }
    in_val = np.fromiter((s in sets["val"] for s in sample_idx), bool, len(sample_idx))
    in_test = np.fromiter((s in sets["test"] for s in sample_idx), bool, len(sample_idx))
    return ~(in_val | in_test), in_val, in_test


def arc_error_deg(pred_trig: np.ndarray, true_trig: np.ndarray) -> np.ndarray:
    """Great-circle error, the metric every evaluation phase reports."""
    def to_xyz(t):
        cos_th, sin_th, cos_ph, sin_ph = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        return np.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th], axis=1)
    a, b = to_xyz(pred_trig), to_xyz(true_trig)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)))


def run_epoch(model, tensors, idx, batch_size, criterion, optimizer=None):
    """
    One pass over `idx`.

    The dataset lives on the target device already, so batches are formed by
    index slicing rather than by a DataLoader. At 52k parameters the model step
    is far cheaper than a per-batch host-to-device copy would be, so this is the
    difference between using the GPU and merely occupying it. Batch size,
    shuffling and optimiser behaviour are unchanged.
    """
    train = optimizer is not None
    model.train(train)
    order = idx[torch.randperm(idx.numel(), device=idx.device)] if train else idx

    total, n = 0.0, 0
    preds, trues = [], []
    with torch.set_grad_enabled(train):
        for start in range(0, order.numel(), batch_size):
            sel = order[start:start + batch_size]
            powers, sub_ids, centers, scale, y = (t[sel] for t in tensors)
            pred = model(powers, sub_ids, centers, scale)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                preds.append(pred.detach().cpu().numpy())
                trues.append(y.detach().cpu().numpy())
            total += loss.item() * sel.numel()
            n += sel.numel()
    arc = (arc_error_deg(np.concatenate(preds), np.concatenate(trues))
           if preds else None)
    return total / n, arc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=_ROOT / "data" / "dataset_v2_train.npz")
    ap.add_argument("--out", type=Path, default=_ROOT / "checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.87)
    ap.add_argument("--scheduler", choices=("exp", "plateau"), default="exp")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on expanded examples, for smoke runs")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--tag", type=str, default="v2")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() "
                         "is False; refusing to fall back silently to CPU.")
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: {device}  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"device: {device}   (no CUDA device visible)")
    print(f"loading {args.data.name}")
    tensors, sample_idx = expand_queries(args.data)
    n_q = tensors[0].shape[0]
    print(f"  {len(np.unique(sample_idx)):,} samples -> {n_q:,} per-sector queries")

    if args.limit and args.limit < n_q:
        keep = np.sort(np.random.default_rng(0).choice(n_q, args.limit, replace=False))
        tensors = tuple(t[keep] for t in tensors)
        sample_idx = sample_idx[keep]
        print(f"  limited to {args.limit:,} queries for a smoke run")

    m_tr, m_va, m_te = split_by_sample(sample_idx)
    print(f"  split: train {m_tr.sum():,}  val {m_va.sum():,}  test {m_te.sum():,}")

    # Resident the whole dataset on the device once; see run_epoch.
    tensors = tuple(t.to(device) for t in tensors)
    mb = sum(t.element_size() * t.numel() for t in tensors) / 1e6
    print(f"  dataset resident on {device.type}: {mb:.0f} MB")
    i_tr, i_va, i_te = (torch.tensor(np.flatnonzero(m), device=device)
                        for m in (m_tr, m_va, m_te))

    model = HybridSARRefinerV2().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"HybridSARRefinerV2: {n_par:,} parameters")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = (ExponentialLR(optimizer, gamma=args.gamma)
                 if args.scheduler == "exp"
                 else ReduceLROnPlateau(optimizer, patience=2, factor=0.5))

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / f"best_model_{args.tag}.pth"
    best = np.inf
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, _ = run_epoch(model, tensors, i_tr, args.batch_size,
                               criterion, optimizer)
        va_loss, va_arc = run_epoch(model, tensors, i_va, args.batch_size,
                                    criterion)
        if args.scheduler == "plateau":
            scheduler.step(va_loss)
        else:
            scheduler.step()

        flag = ""
        if va_loss < best:
            best = va_loss
            # Stringify args: torch>=2.6 loads with weights_only=True by
            # default and refuses to unpickle Path objects.
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss,
                        "args": {k: str(v) for k, v in vars(args).items()}},
                       ckpt)
            flag = "  *"
        print(f"epoch {epoch:3d}/{args.epochs}  lr {lr:.2e}  "
              f"train {tr_loss:.6f}  val {va_loss:.6f}  "
              f"val arc p50 {np.median(va_arc):6.3f} deg  "
              f"p90 {np.percentile(va_arc, 90):7.3f} deg{flag}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\ntrained {args.epochs} epochs in {elapsed / 60:.1f} min "
          f"({elapsed / args.epochs:.1f} s/epoch)")

    model.load_state_dict(torch.load(ckpt, weights_only=True)["model"])
    _, te_arc = run_epoch(model, tensors, i_te, args.batch_size, criterion)
    print(f"held-out test arc error: p50 {np.median(te_arc):.3f} deg, "
          f"p90 {np.percentile(te_arc, 90):.3f} deg, "
          f"mean {te_arc.mean():.3f} deg")
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
