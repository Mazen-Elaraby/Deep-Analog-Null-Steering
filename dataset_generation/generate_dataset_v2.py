#!/usr/bin/env python3
"""
Dataset v2 generator for the analog CRPA power-only DoA problem.

This implements the corrected measurement chain used by the paper. The
scenario sampling uses sectors
drawn without replacement, theta uniform in solid angle inside the sector band,
phi uniform inside the sector band, JNR uniform on 20-50 dB -- so that any
change in downstream model performance is attributable to the measurement
physics and not to a shift in the scenario distribution.

Three corrections to the measurement chain:

1. Probes are the raw design-intent codebook, so probe m nulls its own subsector
   centre. v1 built the probe matrix with MATLAB's conjugate transpose and so
   shipped conj(w), which moves each null to a direction with no counterpart on
   the visible hemisphere and leaves the probes with almost no on-centre
   contrast.

2. Jammers are mutually incoherent. The spatial covariance is then exact in
   closed form,

       R = sum_j P_j u_j u_j^H + sigma_n^2 I,

   with no cross terms, so no snapshot loop is needed. v1 summed the field
   amplitudes with a common phase (``s = ones(J,1)``), which models coherent
   multipath rather than independent jammers.

3. The detector has a relative accuracy floor instead of a fixed thermal floor,
   so probe read-out noise scales with the measured power:

       p_m = pbar_m * (1 + zeta_m / sqrt(K)) + eps_m
       zeta_m ~ N(0, 1)                      finite-snapshot fluctuation
       eps_m  ~ N(0, alpha * mean(pbar)^2)   detector relative accuracy

   K is chosen so the detector term genuinely dominates, which is what lets the
   Phase B bound reduce to C_n = sigma_det^2 I. The condition is not K >> 1/alpha:
   the snapshot-to-detector variance ratio is (1/(K alpha)) (pbar_m / pbar)^2, so
   it is set by the probe dynamic range. Measured over 60k queries the worst case
   is pbar_max / pbar = 3.48, which needs K > 7.7e5 to hold the ratio under 5%.
   K = 1e6 gives a worst case of 3.8%; at 20 MHz that is 50 ms per probe, or
   0.40 s for a full 8-probe scan.

Physics (element patterns, array manifold, and probe codebook) is imported from
the repository-local validated module in ``evaluation/crpa_physics.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "evaluation"))

import crpa_physics as phys  # noqa: E402  (needs the sys.path insert above)

GENERATOR_VERSION = "v2.0"
JMAX = 3
N_PROBES = phys.N_SUBSECTORS

# The detector cannot report negative power. Readings are clipped to this
# fraction of the per-query mean, which also keeps the log10 preprocessing
# finite at nulls driven below zero by the additive detector term.
_POWER_FLOOR_REL = 1e-6


@dataclass(frozen=True)
class GenConfig:
    """Measurement and scenario parameters."""

    alpha: float = 10 ** -3.5          # detector relative-accuracy variance
    n_snapshots: int = 1_000_000       # K; see the module docstring on this value
    noise_power_w: float = phys.NOISE_POWER_W
    jnr_db_lo: float = 20.0
    jnr_db_hi: float = 50.0


def _sample_scenarios(jammer_count: np.ndarray, geom, cfg: GenConfig,
                      rng: np.random.Generator):
    """
    Draw sectors, DoAs and JNRs for a batch. Mirrors the v1 sampling exactly.

    Returns ``(sector_ids, doa_deg, jnr_db)`` padded to JMAX slots, with
    sector id 0 and NaN angles marking inactive slots.
    """
    n = jammer_count.size
    sector_ids = np.zeros((n, JMAX), dtype=np.int16)
    doa_deg = np.full((n, JMAX, 2), np.nan)
    jnr_db = np.full((n, JMAX), np.nan)

    for j_count in np.unique(jammer_count):
        rows = np.flatnonzero(jammer_count == j_count)
        slots = np.arange(j_count)[None, :]

        # randperm(8, J): distinct sectors, uniform over ordered selections.
        keys = rng.random((rows.size, phys.N_SECTORS))
        sel = np.argsort(keys, axis=1)[:, :j_count] + 1
        sector_ids[rows[:, None], slots] = sel

        edges_theta = geom.sector_edges_theta[sel - 1]     # (M, J, 2)
        edges_phi = geom.sector_edges_phi[sel - 1]

        # theta uniform in solid angle within the sector's elevation band.
        cos_lo = np.cos(np.deg2rad(edges_theta[..., 0]))
        cos_hi = np.cos(np.deg2rad(edges_theta[..., 1]))
        u_theta = rng.random(cos_lo.shape)
        theta = np.rad2deg(np.arccos(cos_lo - u_theta * (cos_lo - cos_hi)))

        phi = edges_phi[..., 0] + rng.random(cos_lo.shape) * (
            edges_phi[..., 1] - edges_phi[..., 0])

        doa_deg[rows[:, None], slots, 0] = theta
        doa_deg[rows[:, None], slots, 1] = phi
        jnr_db[rows[:, None], slots] = cfg.jnr_db_lo + rng.random(cos_lo.shape) * (
            cfg.jnr_db_hi - cfg.jnr_db_lo)

    return sector_ids, doa_deg, jnr_db


def _measure(sector_ids, doa_deg, power_w, codebook, manifold,
             cfg: GenConfig, rng: np.random.Generator):
    """
    Simulate the probe read-out for a batch.

    One query is issued per active jammer, using that jammer's own sector
    codebook, but every query sees the full covariance R and therefore the
    leakage from all other jammers.

    Returns ``(pbar, p_obs)`` with shape (n, JMAX, N_PROBES); inactive slots
    are zero.
    """
    n = doa_deg.shape[0]
    active = sector_ids > 0

    # Inactive slots carry zero power, so their (arbitrary) steering vector
    # cannot contribute to R.
    theta = np.where(active, doa_deg[..., 0], 0.0).ravel()
    phi = np.where(active, doa_deg[..., 1], 0.0).ravel()
    u = manifold.steering(theta, phi).T.reshape(n, JMAX, phys.N_ANT)

    # Incoherent sources: no cross terms, so the SCM is exact in closed form.
    R = np.einsum("nj,nja,njb->nab", power_w, u, u.conj(), optimize=True)
    R += cfg.noise_power_w * np.eye(phys.N_ANT)

    pbar = np.zeros((n, JMAX, N_PROBES))
    for q in range(JMAX):
        rows = np.flatnonzero(active[:, q])
        if rows.size == 0:
            continue
        W = codebook[sector_ids[rows, q] - 1]                       # (M, 8, 4)
        # s_m = W[m] @ u, so E|s_m|^2 = diag(W R W^H).
        C = W @ R[rows] @ W.conj().transpose(0, 2, 1)
        pbar[rows, q] = np.real(np.einsum("nmm->nm", C))

    # Finite-snapshot fluctuation is multiplicative; detector accuracy is an
    # additive floor tied to the per-query mean reading.
    n_probes_active = np.where(active, N_PROBES, 1)[..., None]
    mean_pbar = pbar.sum(axis=-1, keepdims=True) / n_probes_active
    zeta = rng.standard_normal(pbar.shape)
    eps = rng.standard_normal(pbar.shape) * np.sqrt(cfg.alpha) * mean_pbar

    p_obs = pbar * (1.0 + zeta / np.sqrt(cfg.n_snapshots)) + eps
    p_obs = np.maximum(p_obs, _POWER_FLOOR_REL * mean_pbar)

    pbar[~active] = 0.0
    p_obs[~active] = 0.0
    return pbar, p_obs


def generate_split(n_samples: int, jammer_mix: dict[int, float], seed: int,
                   cfg: GenConfig, chunk: int = 50_000, label: str = "") -> dict:
    """Generate one split and return it as a dict of columnar arrays."""
    geom = phys.load_sector_geometry()
    manifold = phys.ArrayManifold()
    codebook = np.stack([geom.intended_null_matrix(s)
                         for s in range(1, phys.N_SECTORS + 1)])

    rng = np.random.default_rng(seed)

    counts = np.array(sorted(jammer_mix), dtype=np.int16)
    fractions = np.array([jammer_mix[c] for c in counts], dtype=float)
    fractions /= fractions.sum()
    # Deterministic composition, then shuffle, so the mix is exact.
    per_count = np.floor(fractions * n_samples).astype(int)
    per_count[-1] += n_samples - per_count.sum()
    jammer_count = np.repeat(counts, per_count)
    rng.shuffle(jammer_count)

    out = {
        "jammer_count": jammer_count.astype(np.int8),
        "sector_ids": np.zeros((n_samples, JMAX), dtype=np.int16),
        "doa_deg": np.full((n_samples, JMAX, 2), np.nan, dtype=np.float64),
        "jnr_db": np.full((n_samples, JMAX), np.nan, dtype=np.float64),
        "power_w": np.zeros((n_samples, JMAX), dtype=np.float64),
        "probe_powers_w": np.zeros((n_samples, JMAX, N_PROBES), dtype=np.float64),
        "probe_powers_mean_w": np.zeros((n_samples, JMAX, N_PROBES), dtype=np.float64),
        "total_power_w": np.zeros((n_samples, JMAX), dtype=np.float64),
        "contrast_db": np.full((n_samples, JMAX), np.nan, dtype=np.float64),
    }

    t0 = time.perf_counter()
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        jc = jammer_count[start:stop]

        sector_ids, doa_deg, jnr_db = _sample_scenarios(jc, geom, cfg, rng)
        power_w = np.where(sector_ids > 0,
                           phys.jnr_to_power_w(np.nan_to_num(jnr_db),
                                               cfg.noise_power_w),
                           0.0)
        pbar, p_obs = _measure(sector_ids, doa_deg, power_w, codebook,
                               manifold, cfg, rng)

        active = sector_ids > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            contrast = 10.0 * np.log10(p_obs.max(axis=-1) / p_obs.min(axis=-1))

        sl = slice(start, stop)
        out["sector_ids"][sl] = sector_ids
        out["doa_deg"][sl] = doa_deg
        out["jnr_db"][sl] = jnr_db
        out["power_w"][sl] = power_w
        out["probe_powers_w"][sl] = p_obs
        out["probe_powers_mean_w"][sl] = pbar
        out["total_power_w"][sl] = p_obs.sum(axis=-1)
        out["contrast_db"][sl] = np.where(active, contrast, np.nan)

        elapsed = time.perf_counter() - t0
        rate = stop / elapsed
        print(f"  [{label}] {stop}/{n_samples} samples "
              f"({rate:,.0f}/s, {(n_samples - stop) / rate:.1f}s remaining)",
              flush=True)

    # Geometry tables, so downstream loaders need no MATLAB dependency.
    out["sector_centers_deg"] = geom.centers
    out["sector_edges_theta_deg"] = geom.sector_edges_theta
    out["sector_edges_phi_deg"] = geom.sector_edges_phi
    for key, value in asdict(cfg).items():
        out[key] = value
    out["seed"] = seed
    out["generator_version"] = GENERATOR_VERSION
    out["elapsed_s"] = time.perf_counter() - t0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--n-train", type=int, default=320_000,
                        help="mixed 1J+2J training samples")
    parser.add_argument("--n-3j", type=int, default=40_000,
                        help="separate 3-jammer block")
    parser.add_argument("--n-eval", type=int, default=30_000,
                        help="held-out eval set, equal 1J/2J/3J thirds")
    parser.add_argument("--alpha", type=float, default=10 ** -3.5,
                        help="detector relative-accuracy variance")
    parser.add_argument("--snapshots", type=int, default=1_000_000, help="K")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    cfg = GenConfig(alpha=args.alpha, n_snapshots=args.snapshots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CRPA dataset {GENERATOR_VERSION}")
    print(f"  alpha = {cfg.alpha:.3e}  ({10 * np.log10(1 / cfg.alpha):.1f} dB), "
          f"K = {cfg.n_snapshots}")
    print(f"  probes: design-intent (un-conjugated) codebook")
    print(f"  sources: mutually incoherent, closed-form SCM\n")

    splits = {
        "train": (args.n_train, {1: 0.5, 2: 0.5}, args.seed),
        "3j": (args.n_3j, {3: 1.0}, args.seed + 1),
        "eval": (args.n_eval, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}, args.seed + 2),
    }

    t_all = time.perf_counter()
    for name, (n, mix, seed) in splits.items():
        if n <= 0:
            continue
        mix_str = "+".join(f"{k}J" for k in sorted(mix))
        print(f"[{name}] {n:,} samples ({mix_str})")
        data = generate_split(n, mix, seed, cfg, label=name)
        path = args.out_dir / f"dataset_v2_{name}.npz"
        np.savez_compressed(path, **data)
        size_mb = path.stat().st_size / 1e6
        print(f"  -> {path.name}  ({size_mb:.1f} MB, "
              f"{data['elapsed_s']:.1f}s simulate)\n")

    print(f"total wall clock: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
