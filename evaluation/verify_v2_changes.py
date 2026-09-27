#!/usr/bin/env python3
"""
Confirm that each Phase C change does what it claims, and nothing more.

These are mechanism checks, not accuracy checks. They are cheap enough to rerun
after a full retrain and verify that recommendations #2, #5, and #6 are active.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "models"))
sys.path.insert(0, str(_ROOT / "training"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from model_builder_v2 import HybridSARRefinerV2  # noqa: E402
from train_v2 import expand_queries  # noqa: E402

_results = []


def _report(name, passed, detail=""):
    _results.append((name, passed))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def load_model(ckpt: Path | None):
    model = HybridSARRefinerV2()
    if ckpt and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True)["model"])
        print(f"loaded {ckpt.name}")
    else:
        print("using randomly initialised weights (mechanism checks only)")
    return model.eval()


def check_theta_range(model, batch):
    """#5: theta must be confined to the physical hemisphere by construction."""
    print("\n#5  internal theta reparameterisation")
    with torch.no_grad():
        out = model(*batch[:4]).numpy()
    cos_th, sin_th = out[:, 0], out[:, 1]
    theta = np.rad2deg(np.arccos(np.clip(cos_th, -1, 1)))
    _report("cos(theta) confined to [0, 1]",
            cos_th.min() >= 0.0 and cos_th.max() <= 1.0,
            f"range [{cos_th.min():.4f}, {cos_th.max():.4f}]")
    _report("sin(theta) non-negative", sin_th.min() >= 0.0,
            f"min {sin_th.min():.4f}")
    _report("decoded theta inside [0, 90] with no folding",
            theta.min() >= 0.0 and theta.max() <= 90.0,
            f"range [{theta.min():.2f}, {theta.max():.2f}] deg")
    unit = np.abs(cos_th ** 2 + sin_th ** 2 - 1.0).max()
    _report("theta pair still unit norm (legacy output contract)", unit < 1e-5,
            f"max deviation {unit:.2e}")


def check_scale_sensitivity(model, batch):
    """#2: the network must no longer be invariant to absolute power."""
    print("\n#2  absolute-scale token breaks JNR invariance")
    powers, sub_ids, centers, scale = batch[:4]
    with torch.no_grad():
        base = model(powers, sub_ids, centers, scale)
        shifted = {d: model(powers, sub_ids, centers, scale + d)
                   for d in (-1.0, +1.0, +2.0)}
    for d, out in shifted.items():
        delta = (out - base).abs().max().item()
        _report(f"output responds to a {d:+.0f} decade shift in total power",
                delta > 1e-4, f"max |change| {delta:.3e}")


def check_hull_escape(model, batch):
    """
    #6: the Branch A anchor must be able to leave the convex hull.

    The hull term is a softmax-weighted average of the subsector centres, so on
    its own it can never leave their convex hull -- that is the 29.6 deg
    fixed-point floor. The learned offset is what removes the constraint, so the
    test is whether the offset moves the anchor at all, including on a perfectly
    flat power vector, which is the exact input that pins v1 to its fixed point.
    """
    print("\n#6  Branch A hull offset")
    powers, sub_ids, centers, _ = batch[:4]
    with torch.no_grad():
        attn_w = model.attention_mlp(powers)
        hull = torch.bmm(attn_w.unsqueeze(1), centers).squeeze(1)
        offset = model.hull_offset(powers)
        anchor, _ = model.branch_a_anchor(powers, centers)

        hull_only = model._normalize_trig_pairs(hull)

    rel = (offset.norm(dim=1) / hull.norm(dim=1)).numpy()
    moved = _arc(anchor.numpy(), hull_only.numpy())
    _report("offset is non-zero, so the anchor is not hull-confined",
            rel.max() > 1e-4,
            f"median |offset|/|hull| {np.median(rel):.4f}, max {rel.max():.4f}")
    _report("anchor is displaced from the pure convex combination",
            np.median(moved) > 1e-3,
            f"median displacement {np.median(moved):.3f} deg, "
            f"p90 {np.percentile(moved, 90):.3f} deg")

    # The flat power vector is the architectural fixed point of v1.
    flat = torch.full_like(powers[:1], float(np.log10(1.0 / powers.shape[1])))
    with torch.no_grad():
        off_flat = model.hull_offset(flat).norm().item()
    _report("offset is active even on a flat (contrast-free) power vector",
            off_flat > 1e-4, f"|offset| {off_flat:.4e}")


def _arc(t1, t2):
    def xyz(t):
        v = np.stack([t[:, 1] * t[:, 2], t[:, 1] * t[:, 3], t[:, 0]], axis=1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip((xyz(t1) * xyz(t2)).sum(1), -1, 1)))


def main():
    ckpt = _ROOT / "checkpoints" / "best_model_smoke.pth"
    if len(sys.argv) > 1:
        ckpt = Path(sys.argv[1])
    model = load_model(ckpt)

    tensors, _ = expand_queries(_ROOT / "data" / "dataset_v2_eval.npz")
    sel = torch.arange(0, min(4000, tensors[0].shape[0]))
    batch = tuple(t[sel] for t in tensors)

    check_theta_range(model, batch)
    check_scale_sensitivity(model, batch)
    check_hull_escape(model, batch)

    n_pass = sum(ok for _, ok in _results)
    print(f"\n{'=' * 64}\n{n_pass}/{len(_results)} mechanism checks passed")
    for name, ok in _results:
        if not ok:
            print(f"  FAILED: {name}")
    sys.exit(0 if n_pass == len(_results) else 1)


if __name__ == "__main__":
    main()
