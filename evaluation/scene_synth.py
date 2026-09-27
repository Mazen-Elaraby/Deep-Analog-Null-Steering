#!/usr/bin/env python3
"""
On-demand scene synthesis, so figures can be sampled where they need samples.

The held-out eval split is drawn from the deployment distribution: sectors
without replacement, theta uniform in solid angle, JNR uniform on 20-50 dB. That
is the right distribution for every *statistic* we quote, and nothing in this
module changes those statistics.

It is the wrong distribution for a *map*. A 2-D histogram over
(dJNR x separation) inherits the joint density of that distribution, so the
corners empty out: |dJNR| = 30 dB requires the pair (50, 20) exactly, and a
separation under 5 deg requires two jammers in different sectors that sit either
side of a shared boundary. Both are legal scenes and both are rare, so the
published heatmap was mostly white space -- an artifact of the sampling density,
not a statement about the physics.

This module synthesises scenes on a target grid instead: pick the cell, then
construct a scene that lands in it. The measurement chain is imported from the
generator (`_measure`) rather than reimplemented, so a synthesised query is
bit-comparable with a stored one; only the scenario sampling differs.

Physical constraints are respected, not sampled around:

  * two jammers must occupy DISTINCT sectors (the generator draws sectors
    without replacement and the network never saw a same-sector pair), so a
    small separation is only reachable across a sector boundary;
  * both DoAs must be on the visible hemisphere;
  * each JNR must stay inside the trained 20-50 dB window, which caps |dJNR|
    at 30 dB and leaves a shrinking feasible band for the pair mean as |dJNR|
    grows.

Cells that survive those constraints get filled. Cells that cannot be reached
without violating one stay empty, and `coverage_report()` says so explicitly
rather than letting a reader mistake an infeasible cell for a sparse one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import v2_common as C

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "dataset_generation"))

# The generator's own read-out chain. Importing the private helper is deliberate:
# duplicating it is how the two workspaces would silently drift apart.
from generate_dataset_v2 import _measure, GenConfig, JMAX, N_PROBES  # noqa: E402
import crpa_physics as phys  # noqa: E402

CACHE_DIR = _ROOT / "data"
CHUNK = 8192


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def trig4(doa_deg: np.ndarray) -> np.ndarray:
    """[..., 2] degrees -> [..., 4] as [cos th, sin th, cos ph, sin ph]."""
    t = np.deg2rad(doa_deg[..., 0])
    p = np.deg2rad(doa_deg[..., 1])
    return np.stack([np.cos(t), np.sin(t), np.cos(p), np.sin(p)], axis=-1)


def sector_of(doa_deg: np.ndarray, geom) -> np.ndarray:
    """
    Which sector contains each DoA; 0 if none (i.e. below the horizon).

    The 8 sectors tile the hemisphere as 4 azimuth quadrants x 2 elevation
    bands, so every visible direction has exactly one owner.
    """
    th = np.asarray(doa_deg[..., 0], dtype=float)
    ph = np.asarray(doa_deg[..., 1], dtype=float) % 360.0
    out = np.zeros(th.shape, dtype=np.int16)
    for s in range(phys.N_SECTORS):
        t0, t1 = geom.sector_edges_theta[s]
        p0, p1 = geom.sector_edges_phi[s]
        in_t = (th >= min(t0, t1)) & (th <= max(t0, t1))
        lo, hi = p0 % 360.0, p1 % 360.0
        in_p = ((ph >= lo) & (ph <= hi)) if lo <= hi else ((ph >= lo) | (ph <= hi))
        out = np.where(in_t & in_p & (out == 0), s + 1, out)
    return out


def sector_centres(geom) -> np.ndarray:
    """Spherical mean of each sector's 8 subsector centres, as [theta, phi] deg."""
    out = np.zeros((phys.N_SECTORS, 2))
    for s in range(phys.N_SECTORS):
        v = C.to_xyz(geom.centers[s]).mean(axis=0)
        v /= np.linalg.norm(v)
        out[s] = [np.rad2deg(np.arccos(np.clip(v[2], -1, 1))),
                  np.rad2deg(np.arctan2(v[1], v[0])) % 360.0]
    return out


def tangent_basis(doa_deg: np.ndarray):
    """Local orthonormal (e_theta, e_phi) at each DoA; both unit, both tangent."""
    t = np.deg2rad(doa_deg[..., 0])
    p = np.deg2rad(doa_deg[..., 1])
    e_t = np.stack([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)], -1)
    e_p = np.stack([-np.sin(p), np.cos(p), np.zeros_like(p)], -1)
    return e_t, e_p


def offset_by_arc(doa_deg: np.ndarray, arc_deg: np.ndarray,
                  bearing_rad: np.ndarray) -> np.ndarray:
    """
    Move each DoA `arc_deg` along a great circle on bearing `bearing_rad`.

    Exact rotation in the tangent plane, so the requested separation is the
    realised separation to machine precision -- which is what lets a cell be
    targeted instead of hoped for.
    """
    u = C.to_xyz(doa_deg)
    e_t, e_p = tangent_basis(doa_deg)
    a = np.deg2rad(arc_deg)[..., None]
    d = (np.cos(bearing_rad)[..., None] * e_t
         + np.sin(bearing_rad)[..., None] * e_p)
    v = np.cos(a) * u + np.sin(a) * d
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return np.stack([np.rad2deg(np.arccos(np.clip(v[..., 2], -1, 1))),
                     np.rad2deg(np.arctan2(v[..., 1], v[..., 0])) % 360.0], -1)


# ---------------------------------------------------------------------------
# Measure and predict
# ---------------------------------------------------------------------------

class SceneModel:
    """Trained model plus the generator's measurement chain, in one place."""

    def __init__(self, ckpt: Path = C.DEFAULT_CKPT, cfg: GenConfig | None = None,
                 device: str | None = None):
        from model_builder_v2 import HybridSARRefinerV2
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                           else "cpu"))
        self.model = HybridSARRefinerV2().to(self.dev).eval()
        self.model.load_state_dict(torch.load(ckpt, weights_only=True)["model"])
        self.cfg = cfg or GenConfig()
        self.geom = phys.load_sector_geometry()
        self.manifold = phys.ArrayManifold()
        self.codebook = np.stack([self.geom.intended_null_matrix(s)
                                  for s in range(1, phys.N_SECTORS + 1)])

    # -- measurement --------------------------------------------------------

    def measure(self, sector_ids, doa_deg, jnr_db, rng):
        """Probe read-out for a batch of scenes, via the generator's own chain."""
        power_w = np.where(sector_ids > 0,
                           phys.jnr_to_power_w(np.nan_to_num(jnr_db),
                                               self.cfg.noise_power_w), 0.0)
        _, p_obs = _measure(sector_ids, doa_deg, power_w, self.codebook,
                            self.manifold, self.cfg, rng)
        return p_obs

    # -- inference ----------------------------------------------------------

    def predict(self, sector_ids, p_obs):
        """
        DoA prediction for every active query.

        Reproduces `train_v2.expand_queries` exactly: the shape feature is
        log10(p) - log10(sum p) and the scale feature is log10(sum p) -
        log10(sigma_n^2). Any drift here would be measuring a different network.
        """
        active = sector_ids > 0
        rows, slots = np.nonzero(active)
        p = p_obs[rows, slots]                                   # (Q, 8)
        sec0 = sector_ids[rows, slots].astype(np.int64) - 1       # 0-indexed

        p_safe = p + 1e-20
        log_p = np.log10(p_safe)
        log_total = np.log10(p_safe.sum(axis=1, keepdims=True))
        norm_p = (log_p - log_total).astype(np.float32)
        log_scale = (log_total[:, 0]
                     - np.log10(self.cfg.noise_power_w)).astype(np.float32)
        centers_trig = trig4(self.geom.centers[sec0]).astype(np.float32)
        sub_ids = (sec0[:, None] * N_PROBES
                   + np.arange(N_PROBES)[None, :]).astype(np.int64)

        preds = []
        with torch.no_grad():
            for s in range(0, norm_p.shape[0], CHUNK):
                sl = slice(s, s + CHUNK)
                preds.append(self.model(
                    torch.tensor(norm_p[sl], device=self.dev),
                    torch.tensor(sub_ids[sl], device=self.dev),
                    torch.tensor(centers_trig[sl], device=self.dev),
                    torch.tensor(log_scale[sl], device=self.dev),
                ).cpu().numpy())
        doa_q = C.decode_trig(np.concatenate(preds))

        out = np.full(doa_deg_shape(sector_ids), np.nan)
        out[rows, slots] = doa_q
        return out

    def run(self, sector_ids, doa_deg, jnr_db, rng):
        """measure -> predict, returning predictions padded to JMAX slots."""
        return self.predict(sector_ids, self.measure(sector_ids, doa_deg,
                                                     jnr_db, rng))


def doa_deg_shape(sector_ids):
    return (sector_ids.shape[0], sector_ids.shape[1], 2)


def _pad(sector_list, doa_list, jnr_list):
    """Stack per-jammer lists into the generator's (n, JMAX, ...) layout."""
    n = sector_list[0].size
    sector_ids = np.zeros((n, JMAX), dtype=np.int16)
    doa = np.full((n, JMAX, 2), np.nan)
    jnr = np.full((n, JMAX), np.nan)
    for q, (s, d, j) in enumerate(zip(sector_list, doa_list, jnr_list)):
        sector_ids[:, q] = s
        doa[:, q] = d
        jnr[:, q] = j
    return sector_ids, doa, jnr


# ---------------------------------------------------------------------------
# Grid 1: two jammers over (dJNR x separation)
# ---------------------------------------------------------------------------

def _take_per_cell(ix, iy, n_y, per_cell, rng):
    """Up to `per_cell` candidate indices from each occupied (ix, iy) cell."""
    flat = ix * n_y + iy
    # Shuffle first so the retained members of an over-full cell are a random
    # subsample of it, not the first ones the sampler happened to produce.
    perm = rng.permutation(flat.size)
    order = perm[np.argsort(flat[perm], kind="stable")]
    f = flat[order]
    starts = np.r_[0, np.flatnonzero(np.diff(f)) + 1]
    sizes = np.diff(np.r_[starts, f.size])
    rank = np.arange(f.size) - np.repeat(starts, sizes)
    return order[rank < per_cell]


def build_2j_grid(sm: SceneModel, d_jnr_edges: np.ndarray,
                  sep_edges: np.ndarray, per_cell: int = 30,
                  n_candidates: int = 1_500_000, seed: int = 0,
                  with_jsr: bool = True, cache: Path | None = None) -> dict:
    """
    Two-jammer scenes placed deliberately across the (dJNR x separation) plane.

    Returns per-target records (two per scene, target/interferer roles swapped)
    so the output drops straight into the Phase 3.5 and Phase 4 heatmaps.
    """
    sig = np.array([per_cell, n_candidates, seed, int(with_jsr),
                    d_jnr_edges.size, sep_edges.size,
                    d_jnr_edges[0], d_jnr_edges[-1],
                    sep_edges[0], sep_edges[-1]], dtype=float)
    if cache is not None and cache.exists():
        z = np.load(cache)
        if z["signature"].shape == sig.shape and np.allclose(z["signature"], sig):
            print(f"  [grid] reusing {cache.name}")
            return {k: z[k] for k in z.files}

    rng = np.random.default_rng(seed)
    geom = sm.geom
    sep_lo, sep_hi = float(sep_edges[0]), float(sep_edges[-1])
    d_lo, d_hi = float(d_jnr_edges[0]), float(d_jnr_edges[-1])

    # -- candidate scenes, constructed to span the plane --------------------
    # Jammer 1 uniform in solid angle; jammer 2 placed at a requested arc on a
    # random bearing. Requesting the separation is what fills the low-separation
    # column that rejection sampling from the deployment distribution misses.
    cos_lo, cos_hi = np.cos(np.deg2rad(89.0)), np.cos(np.deg2rad(1.0))
    th1 = np.rad2deg(np.arccos(cos_lo + rng.random(n_candidates)
                               * (cos_hi - cos_lo)))
    ph1 = rng.random(n_candidates) * 360.0
    doa1 = np.stack([th1, ph1], -1)

    want_sep = sep_lo + rng.random(n_candidates) * (sep_hi - sep_lo)
    bearing = rng.random(n_candidates) * 2.0 * np.pi
    doa2 = offset_by_arc(doa1, want_sep, bearing)

    s1 = sector_of(doa1, geom)
    s2 = sector_of(doa2, geom)
    # Distinct sectors and both visible: the two hard physical constraints.
    ok = (s1 > 0) & (s2 > 0) & (s1 != s2) & (doa2[:, 0] <= 89.0)

    # JNR pair: fix the difference, then pick the mean inside the feasible band.
    d_jnr = d_lo + rng.random(n_candidates) * (d_hi - d_lo)
    half = np.abs(d_jnr) / 2.0
    m_lo = sm.cfg.jnr_db_lo + half
    m_hi = sm.cfg.jnr_db_hi - half
    ok &= m_hi >= m_lo
    mid = m_lo + rng.random(n_candidates) * np.maximum(m_hi - m_lo, 0.0)
    jnr1 = mid + d_jnr / 2.0
    jnr2 = mid - d_jnr / 2.0

    sep = C.arc_deg(doa1, doa2)
    ok &= np.isfinite(sep) & (sep >= sep_lo) & (sep < sep_hi)

    idx = np.flatnonzero(ok)
    ix = np.clip(np.digitize(d_jnr[idx], d_jnr_edges) - 1, 0,
                 d_jnr_edges.size - 2)
    iy = np.clip(np.digitize(sep[idx], sep_edges) - 1, 0, sep_edges.size - 2)
    sel = idx[_take_per_cell(ix, iy, sep_edges.size - 1, per_cell, rng)]
    sel = np.sort(sel)

    sector_ids, doa, jnr = _pad([s1[sel], s2[sel]],
                                [doa1[sel], doa2[sel]],
                                [jnr1[sel], jnr2[sel]])
    print(f"  [grid] {sel.size:,} scenes retained from "
          f"{idx.size:,} feasible candidates ({n_candidates:,} drawn)")

    doa_pred = sm.run(sector_ids, doa, jnr, rng)
    err = C.arc_deg(doa_pred[:, :2], doa[:, :2])

    jsr = np.full((sel.size, 2), np.nan)
    if with_jsr:
        for i in range(sel.size):
            v, _ = C.jsr_db(doa_pred[i, :2], doa[i, :2], sm.manifold)
            jsr[i] = v

    # Two records per scene: each jammer once as the target.
    out = {
        "d_jnr": np.concatenate([jnr[:, 0] - jnr[:, 1], jnr[:, 1] - jnr[:, 0]]),
        "sep": np.concatenate([sep[sel], sep[sel]]),
        "err": np.concatenate([err[:, 0], err[:, 1]]),
        "jsr": np.concatenate([jsr[:, 0], jsr[:, 1]]),
        "jnr_target": np.concatenate([jnr[:, 0], jnr[:, 1]]),
        "theta_target": np.concatenate([doa[:, 0, 0], doa[:, 1, 0]]),
        "n_scenes": np.array([sel.size]),
        "n_feasible": np.array([idx.size]),
        "signature": sig,
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
        print(f"  [grid] cached to {cache.name}")
    return out


def coverage_report(x, y, x_edges, y_edges, min_count: int = 5,
                    label: str = "cells") -> str:
    """One line stating how much of the plane the figure actually covers."""
    from scipy.stats import binned_statistic_2d
    cnt, _, _, _ = binned_statistic_2d(x, y, None, "count",
                                       bins=[x_edges, y_edges])
    total = cnt.size
    filled = int((cnt >= min_count).sum())
    return (f"{label}: {filled}/{total} populated at n >= {min_count} "
            f"({100.0 * filled / total:.1f}%), "
            f"median cell n = {int(np.median(cnt[cnt > 0])) if (cnt > 0).any() else 0}")


# ---------------------------------------------------------------------------
# Grid 2: repeated trials at fixed DoA, for bias vs variance
# ---------------------------------------------------------------------------

def build_bias_variance_grid(sm: SceneModel, theta_deg: np.ndarray,
                             phi_deg: np.ndarray, n_trials: int = 64,
                             jnr_db: float = 35.0, seed: int = 0) -> dict:
    """
    Single-jammer error decomposed into bias and scatter at each grid DoA.

    The decomposition needs the DoA held fixed while only the detector noise is
    redrawn, which the eval split cannot provide: it visits every direction at
    most once, so a binned "bias" there is contaminated by the spread of DoAs
    inside the bin. Here each grid point is measured `n_trials` times, so

        MSE(point) = |bias|^2 + scatter^2

    is exact rather than inferred, and a point sitting below its own CRLB can be
    attributed to one term or the other.
    """
    rng = np.random.default_rng(seed)
    TH, PH = np.meshgrid(theta_deg, phi_deg, indexing="ij")
    grid = np.stack([TH.ravel(), PH.ravel()], -1)
    sec = sector_of(grid, sm.geom)
    keep = sec > 0
    grid, sec = grid[keep], sec[keep]
    n = grid.shape[0]

    doa_rep = np.repeat(grid, n_trials, axis=0)
    sec_rep = np.repeat(sec, n_trials)
    sector_ids, doa, jnr = _pad([sec_rep], [doa_rep],
                                [np.full(n * n_trials, jnr_db)])
    doa_pred = sm.run(sector_ids, doa, jnr, rng)[:, 0]

    # Signed error in the local tangent frame, both components in degrees of
    # arc so that they are directly comparable.
    d_th = doa_pred[:, 0] - doa_rep[:, 0]
    d_ph = C.azimuth_error_deg(doa_pred[:, 1], doa_rep[:, 1])
    sign = np.sign(((doa_pred[:, 1] - doa_rep[:, 1] + 180.0) % 360.0) - 180.0)
    d_ph = sign * d_ph * np.sin(np.deg2rad(doa_rep[:, 0]))

    d_th = d_th.reshape(n, n_trials)
    d_ph = d_ph.reshape(n, n_trials)

    bias_t, bias_p = d_th.mean(1), d_ph.mean(1)
    bias_mag = np.hypot(bias_t, bias_p)
    scatter = np.sqrt(d_th.var(1) + d_ph.var(1))
    rmse = np.sqrt(bias_mag ** 2 + scatter ** 2)

    # Bearing from each grid point toward its own sector centre, so the bias
    # direction can be tested against the architecture's fallback.
    centres = sector_centres(sm.geom)[sec - 1]
    e_t, e_p = tangent_basis(grid)
    u_c = C.to_xyz(centres)
    comp_t = np.einsum("ij,ij->i", u_c, e_t)
    comp_p = np.einsum("ij,ij->i", u_c, e_p)
    norm = np.hypot(comp_t, comp_p)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_inward = np.where(
            (norm > 0) & (bias_mag > 0),
            (bias_t * comp_t + bias_p * comp_p) / np.maximum(norm * bias_mag, 1e-30),
            np.nan)

    return {
        "doa": grid, "sector": sec, "bias_t": bias_t, "bias_p": bias_p,
        "bias_mag": bias_mag, "scatter": scatter, "rmse": rmse,
        "cos_inward": cos_inward, "arc_to_centre": C.arc_deg(grid, centres),
        "n_trials": np.array([n_trials]), "jnr_db": np.array([jnr_db]),
        "theta_axis": theta_deg, "phi_axis": phi_deg,
    }
