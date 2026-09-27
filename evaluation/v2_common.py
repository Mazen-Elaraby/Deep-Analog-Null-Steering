#!/usr/bin/env python3
"""
Shared infrastructure for the Phase E evaluation of dataset/model v2.

The v1 phase scripts each re-implemented loading, preprocessing, steering
vectors and plot styling, which is most of their ~5,600 lines. Centralising
those here keeps each phase script to its actual analysis.

Two pairings are provided and they answer different questions:

  DIRECT pairing   query q is issued against jammer q's own sector, so
                   `arc_direct[n, q]` is the error of the prediction that was
                   *asked* for jammer q. This is the operationally meaningful
                   error, and it is what the target/interferer analysis needs.

  MATCHED pairing  the J predictions are assigned to the J truths by Hungarian
                   matching on arc error. This is the resolution question: did
                   the network recover the set of directions at all, regardless
                   of which query produced which.

A network that collapses to the inter-jammer centroid scores badly on both; a
network that recovers both jammers but swaps them scores badly only on direct.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "models", _ROOT / "training", _ROOT / "dataset_generation",
           _ROOT / "evaluation"):
    sys.path.insert(0, str(_p))

from model_builder_v2 import HybridSARRefinerV2  # noqa: E402
from train_v2 import expand_queries  # noqa: E402
from generate_dataset_v2 import GenConfig, JMAX  # noqa: E402
import crpa_physics as phys  # noqa: E402

DEFAULT_CKPT = _ROOT / "checkpoints" / "best_model_v2.pth"
DEFAULT_DATA = _ROOT / "data" / "dataset_v2_eval.npz"
FIG_DIR = _ROOT / "evaluation_results" / "figures"
RESULT_DIR = _ROOT / "evaluation_results"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def to_xyz(doa_deg: np.ndarray) -> np.ndarray:
    """[..., 2] (zenith, azimuth) degrees -> unit vectors [..., 3]."""
    th = np.deg2rad(doa_deg[..., 0])
    ph = np.deg2rad(doa_deg[..., 1])
    return np.stack([np.sin(th) * np.cos(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(th)], axis=-1)


def arc_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle angle in degrees between two [..., 2] DoA arrays."""
    va, vb = to_xyz(a), to_xyz(b)
    return np.rad2deg(np.arccos(np.clip((va * vb).sum(-1), -1.0, 1.0)))


def decode_trig(trig: np.ndarray) -> np.ndarray:
    """
    [..., 4] -> [..., 2] degrees.

    v2 guarantees cos(theta) in [0,1] and sin(theta) >= 0, so arccos lands
    directly in [0, 90] and none of v1's folding is needed. Kept explicit so the
    invariant is visible at the point of use.
    """
    theta = np.rad2deg(np.arccos(np.clip(trig[..., 0], -1.0, 1.0)))
    phi = np.rad2deg(np.arctan2(trig[..., 3], trig[..., 2])) % 360.0
    return np.stack([theta, phi], axis=-1)


def azimuth_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped absolute azimuth difference."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


# ---------------------------------------------------------------------------
# Loading and inference
# ---------------------------------------------------------------------------

@dataclass
class EvalData:
    """Per-sample view, padded to JMAX slots with NaN / 0 for inactive."""

    jammer_count: np.ndarray        # (N,)
    sector_ids: np.ndarray          # (N, JMAX)
    doa_true: np.ndarray            # (N, JMAX, 2)
    doa_pred: np.ndarray            # (N, JMAX, 2) prediction of query q
    jnr_db: np.ndarray              # (N, JMAX)
    contrast_db: np.ndarray         # (N, JMAX)
    total_power_w: np.ndarray       # (N, JMAX)
    powers: np.ndarray              # (N, JMAX, 8) observed probe powers
    arc_direct: np.ndarray          # (N, JMAX) query q vs jammer q
    arc_matched: np.ndarray         # (N, JMAX) Hungarian-assigned
    match_perm: np.ndarray          # (N, JMAX) which truth each query got
    separation: np.ndarray          # (N,) 2J great-circle separation, else NaN
    cfg: GenConfig

    def active(self) -> np.ndarray:
        return self.sector_ids > 0

    def mask(self, j: int) -> np.ndarray:
        return self.jammer_count == j


def load_eval(ckpt_path: Path = DEFAULT_CKPT, data_path: Path = DEFAULT_DATA,
              device: str | None = None) -> EvalData:
    """Run the trained model over a split and assemble both pairings."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HybridSARRefinerV2().to(dev).eval()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model"])

    d = np.load(data_path)
    tensors, _ = expand_queries(data_path)

    # expand_queries walks np.nonzero order, so this reproduces the (sample,
    # slot) index of every expanded query.
    active = d["sector_ids"] > 0
    q_rows, q_slots = np.nonzero(active)

    preds = []
    with torch.no_grad():
        for s in range(0, tensors[0].shape[0], 8192):
            batch = [t[s:s + 8192].to(dev) for t in tensors[:4]]
            preds.append(model(*batch).cpu().numpy())
    pred_trig = np.concatenate(preds)
    pred_doa_q = decode_trig(pred_trig)

    n = d["jammer_count"].size
    doa_pred = np.full((n, JMAX, 2), np.nan)
    doa_pred[q_rows, q_slots] = pred_doa_q

    doa_true = d["doa_deg"]
    arc_direct = np.full((n, JMAX), np.nan)
    arc_direct[q_rows, q_slots] = arc_deg(pred_doa_q, doa_true[q_rows, q_slots])

    # Hungarian matching per sample, over the active slots only.
    arc_matched = np.full((n, JMAX), np.nan)
    match_perm = np.full((n, JMAX), -1, dtype=int)
    for i in range(n):
        j = int(d["jammer_count"][i])
        if j == 1:
            arc_matched[i, 0] = arc_direct[i, 0]
            match_perm[i, 0] = 0
            continue
        cost = arc_deg(doa_pred[i, :j, None, :], doa_true[i, None, :j, :])
        r, c = linear_sum_assignment(cost)
        arc_matched[i, r] = cost[r, c]
        match_perm[i, r] = c

    sep = np.full(n, np.nan)
    two = d["jammer_count"] == 2
    sep[two] = arc_deg(doa_true[two, 0], doa_true[two, 1])

    cfg = GenConfig(alpha=float(d["alpha"]), n_snapshots=int(d["n_snapshots"]),
                    noise_power_w=float(d["noise_power_w"]),
                    jnr_db_lo=float(d["jnr_db_lo"]),
                    jnr_db_hi=float(d["jnr_db_hi"]))

    return EvalData(d["jammer_count"], d["sector_ids"], doa_true, doa_pred,
                    d["jnr_db"], d["contrast_db"], d["total_power_w"],
                    d["probe_powers_w"], arc_direct, arc_matched, match_perm,
                    sep, cfg)


# ---------------------------------------------------------------------------
# Jammer suppression
# ---------------------------------------------------------------------------

def jsr_db(doa_pred: np.ndarray, doa_true: np.ndarray, manifold,
           cap_db: float = 50.0):
    """
    Suppression achieved at each true DoA by nulling toward the predictions.

    Delegates to the v1 `projection_nuller_weights` / `jammer_suppression_db`
    pair rather than rebuilding the projector. Those already handle the
    rank-deficient case (two predictions collapsed onto one direction) via
    pseudo-inverse, the near-zenith degeneracy where unit common-mode gain is
    unachievable, and -- critically -- the y = w^T u convention, in which the
    weights are returned pre-conjugated. Reimplementing this with the more
    familiar w^H u convention silently steers the null to the mirrored direction.

    Returns (jsr_per_true_doa, degenerate_flag).
    """
    U = np.stack([manifold.steering(t, p)
                  for t, p in np.atleast_2d(doa_pred)], axis=1)
    w, degenerate = phys.projection_nuller_weights(U)
    out = [phys.jammer_suppression_db(w, manifold.steering(t, p), cap_db=cap_db)
           for t, p in np.atleast_2d(doa_true)]
    return np.array(out), degenerate


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

def setup_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_fig(fig, name: str, subdir: str) -> None:
    out = FIG_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"  wrote {subdir}/{name}.pdf and .png")


class Tee:
    """Mirror stdout into a results text file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.f.close()

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()
