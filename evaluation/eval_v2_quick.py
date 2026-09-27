#!/usr/bin/env python3
"""
Minimal post-retrain readout: arc error broken down by jammer count.

The training log reports a single aggregated arc error, which is misleading here
because two thirds of the expanded queries come from 2-jammer scenes (160k 1J
queries plus 320k queries from 160k 2J samples) and 2J carries only ~6.9 dB of
probe contrast against 20.3 dB for 1J. The aggregate is therefore dominated by
the intrinsically hard case. This script separates them and puts the 1J result
next to the Phase B CRLB, which is the only comparison that says whether the
network is near the information limit.

This is deliberately not the full Phase E evaluation: no figures, no Hungarian
matching, no JSR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "models"))
sys.path.insert(0, str(_ROOT / "training"))
sys.path.insert(0, str(_ROOT / "dataset_generation"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from model_builder_v2 import HybridSARRefinerV2  # noqa: E402
from train_v2 import expand_queries, arc_error_deg  # noqa: E402
from generate_dataset_v2 import GenConfig  # noqa: E402
import crpa_physics as phys  # noqa: E402
from crlb_v2 import crlb_v2_from_jnr  # noqa: E402


def percentiles(x):
    return (np.median(x), np.percentile(x, 90), x.mean())


def main() -> None:
    ckpt_path = _ROOT / "checkpoints" / "best_model_v2.pth"
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])
    data_path = _ROOT / "data" / "dataset_v2_eval.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, weights_only=True)
    model = HybridSARRefinerV2().to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"checkpoint {ckpt_path.name}  (epoch {ckpt['epoch']}, "
          f"val loss {ckpt['val_loss']:.6f})")

    d = np.load(data_path)
    tensors, rows = expand_queries(data_path)
    jc = d["jammer_count"][rows]
    jnr = d["jnr_db"]

    preds = []
    with torch.no_grad():
        for s in range(0, tensors[0].shape[0], 8192):
            batch = [t[s:s + 8192].to(device) for t in tensors[:4]]
            preds.append(model(*batch).cpu().numpy())
    pred = np.concatenate(preds)
    arc = arc_error_deg(pred, tensors[4].numpy())

    print(f"\nheld-out eval: {len(arc):,} queries from {len(d['jammer_count']):,} samples")
    print("  " + "-" * 66)
    print(f"  {'case':<22} {'n':>8} {'p50':>8} {'p90':>8} {'mean':>8}")
    for j in (1, 2, 3):
        sel = jc == j
        if not sel.any():
            continue
        p50, p90, mean = percentiles(arc[sel])
        print(f"  {f'{j} jammer':<22} {sel.sum():>8,} {p50:>8.3f} {p90:>8.3f} {mean:>8.3f}")
    p50, p90, mean = percentiles(arc)
    print(f"  {'all (training mix)':<22} {len(arc):>8,} {p50:>8.3f} {p90:>8.3f} {mean:>8.3f}")

    # 1-jammer against the Phase B bound.
    cfg = GenConfig(alpha=float(d["alpha"]), n_snapshots=int(d["n_snapshots"]),
                    noise_power_w=float(d["noise_power_w"]))
    geom = phys.load_sector_geometry()
    manifold = phys.ArrayManifold()
    rng = np.random.default_rng(0)
    one = np.flatnonzero(d["jammer_count"] == 1)
    bounds = []
    for i in rng.choice(one, 120, replace=False):
        W = geom.intended_null_matrix(int(d["sector_ids"][i, 0]))
        r = crlb_v2_from_jnr(W, d["doa_deg"][i, 0:1], [float(jnr[i, 0])],
                             manifold, cfg)
        if np.isfinite(r.arc_deg[0]):
            bounds.append(r.arc_deg[0])
    crb = float(np.median(bounds))
    emp = float(np.sqrt(np.mean(arc[jc == 1] ** 2)))
    print(f"\n  1J RMSE {emp:.3f} deg vs median CRLB {crb:.3f} deg "
          f"-> {emp / crb:.2f}x the bound")

    # 2-jammer, split by power imbalance: the -1 dB/dB regime.
    two = jc == 2
    if two.any():
        dj = np.abs(jnr[rows[two], 0] - jnr[rows[two], 1])
        print("\n  2J arc error vs inter-jammer power spread")
        print("  " + "-" * 66)
        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30)):
            m = (dj >= lo) & (dj < hi)
            if m.sum() < 20:
                continue
            a = arc[two][m]
            print(f"    dJNR {lo:2d}-{hi:2d} dB  n {m.sum():>6,}  "
                  f"p50 {np.median(a):7.3f}  p90 {np.percentile(a, 90):7.3f}")


if __name__ == "__main__":
    main()
