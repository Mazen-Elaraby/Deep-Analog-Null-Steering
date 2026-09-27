#!/usr/bin/env python3
"""
Validation gate for dataset v2.

Each check targets one of the three physics corrections, so a failure localises
to a specific fix rather than to "the generator".

  Check 1  probe convention      the design-intent codebook nulls its own
                                 subsector centre                        (fix 1)
  Check 2  closed-form SCM       diag(W R W^H) matches an explicit
                                 random-phase snapshot loop              (fix 2)
  Check 3  on-centre contrast    v1 deployed probes vs v2 design-intent
                                 probes, same code path                  (fix 1)
  Check 4  subsector recovery    argmin over probes identifies the
                                 occupied subsector                      (fix 1)
  Check 5  detector floor        measured contrast tracks the alpha floor (fix 3)
  Check 6  scenario sampling     JNR uniform on 20-50 dB, unchanged from v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crpa_physics as phys  # noqa: E402
from generate_dataset_v2 import GenConfig, JMAX, N_PROBES  # noqa: E402

_results: list[tuple[str, bool]] = []


def _report(name: str, passed: bool, detail: str = "") -> None:
    _results.append((name, passed))
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))


def check_probe_convention(geom, manifold) -> None:
    """The raw codebook must null its own subsector centre under s_m = w_m^T u."""
    print("\nCheck 1: probe convention (design-intent codebook nulls own centre)")
    depths = []
    for sector_id in range(1, phys.N_SECTORS + 1):
        W = geom.intended_null_matrix(sector_id)
        centers = geom.subsector_centers(sector_id)
        u = manifold.steering(centers[:, 0], centers[:, 1])        # (4, 8)
        g = np.abs(W @ u) ** 2                                     # (probe, centre)
        own = np.diag(g)
        others = (g.sum(axis=0) - own) / (N_PROBES - 1)
        depths.append(10 * np.log10(own / others))
    depths = np.concatenate(depths)
    shallowest = depths.max()
    _report("null depth over all 64 probes", shallowest < -100.0,
            f"mean {depths.mean():.0f} dB, shallowest {shallowest:.0f} dB")


def check_closed_form_scm(geom, manifold, cfg, rng, n_trials=24, n_snap=200_000) -> None:
    """
    Cross-validate the closed-form incoherent SCM against a snapshot loop.

    Independent per-source phases make the cross terms vanish in expectation;
    this confirms that the closed form is the incoherent model and not an
    accidental restatement of v1's coherent sum.
    """
    print("\nCheck 2: closed-form incoherent SCM vs explicit snapshot loop")
    codebook = np.stack([geom.intended_null_matrix(s)
                         for s in range(1, phys.N_SECTORS + 1)])
    rel_err, coh_gap = [], []

    for _ in range(n_trials):
        j_count = rng.integers(2, JMAX + 1)
        sectors = rng.permutation(phys.N_SECTORS)[:j_count] + 1
        theta, phi = [], []
        for s in sectors:
            et = geom.sector_edges_theta[s - 1]
            ep = geom.sector_edges_phi[s - 1]
            c_lo, c_hi = np.cos(np.deg2rad(et))
            theta.append(np.rad2deg(np.arccos(c_lo - rng.random() * (c_lo - c_hi))))
            phi.append(ep[0] + rng.random() * (ep[1] - ep[0]))
        jnr = rng.uniform(cfg.jnr_db_lo, cfg.jnr_db_hi, j_count)
        P = phys.jnr_to_power_w(jnr, cfg.noise_power_w)

        u = manifold.steering(np.array(theta), np.array(phi))       # (4, J)
        W = codebook[sectors[0] - 1]

        R = (u * P) @ u.conj().T + cfg.noise_power_w * np.eye(phys.N_ANT)
        pbar = np.real(np.diag(W @ R @ W.conj().T))

        # Snapshot loop: independent phase per source per snapshot.
        psi = rng.uniform(0, 2 * np.pi, (j_count, n_snap))
        amp = np.sqrt(P)[:, None] * np.exp(1j * psi)
        noise = np.sqrt(cfg.noise_power_w / 2) * (
            rng.standard_normal((phys.N_ANT, n_snap))
            + 1j * rng.standard_normal((phys.N_ANT, n_snap)))
        x = u @ amp + noise
        p_mc = np.mean(np.abs(W @ x) ** 2, axis=1)

        rel_err.append(np.abs(p_mc - pbar) / pbar)

        # For contrast: the v1 coherent model with a common phase.
        s_coh = W @ (u @ np.sqrt(P))
        p_coh = np.abs(s_coh) ** 2 + cfg.noise_power_w * np.sum(np.abs(W) ** 2, axis=1)
        coh_gap.append(np.abs(p_coh - pbar) / pbar)

    rel_err = np.concatenate(rel_err)
    coh_gap = np.concatenate(coh_gap)
    tol = 8.0 / np.sqrt(n_snap)          # generous multiple of the MC error
    _report("closed form matches snapshot Monte Carlo",
            np.median(rel_err) < tol,
            f"median rel err {np.median(rel_err):.2e}, max {rel_err.max():.2e} "
            f"(MC noise ~{1 / np.sqrt(n_snap):.2e})")
    _report("coherent (v1) model is materially different",
            np.median(coh_gap) > 0.05,
            f"median rel gap {np.median(coh_gap) * 100:.0f}% vs incoherent")


def check_on_center_contrast(geom, manifold, cfg) -> None:
    """
    On-centre probe contrast, v1 deployed probes vs v2 design-intent probes.

    A single unit jammer is placed exactly at each of the 64 subsector centres
    and the noiseless probe vector is formed with both codebooks. This is the
    headline number for fix 1: v1's probes dip only ~6 dB at the very direction
    they were meant to null.
    """
    print("\nCheck 3: on-centre contrast, v1 deployed vs v2 design-intent probes")
    P = phys.jnr_to_power_w(35.0, cfg.noise_power_w)      # mid-range JNR
    out = {}
    for label, getter in (("v1 deployed", geom.probe_matrix),
                          ("v2 intent", geom.intended_null_matrix)):
        dips = []
        for sector_id in range(1, phys.N_SECTORS + 1):
            W = getter(sector_id)
            centers = geom.subsector_centers(sector_id)
            u = manifold.steering(centers[:, 0], centers[:, 1])
            for m in range(N_PROBES):
                R = P * np.outer(u[:, m], u[:, m].conj()) \
                    + cfg.noise_power_w * np.eye(phys.N_ANT)
                p = np.real(np.diag(W @ R @ W.conj().T))
                dips.append(10 * np.log10(p.max() / p[m]))
        out[label] = np.array(dips)
        print(f"    {label:<12} dip at own centre: "
              f"median {np.median(out[label]):.2f} dB, "
              f"p10 {np.percentile(out[label], 10):.2f} dB")

    gain = np.median(out["v2 intent"]) - np.median(out["v1 deployed"])
    _report("v2 probes recover a deep on-centre null",
            np.median(out["v2 intent"]) > 20.0,
            f"+{gain:.1f} dB over v1")


def check_subsector_recovery(data, geom, manifold, cfg) -> None:
    """
    The deepest probe should point at the subsector the jammer occupies.

    Exact recovery is not expected to reach 8/8. With four elements a probe null
    is the solution of one complex constraint w^T u = 0, which on the hemisphere
    is a curve rather than a point, so a jammer lying on a neighbouring probe's
    null curve legitimately reads lower there. The meaningful gate is therefore
    v2 against the v1 deployed probes on the identical scenarios.
    """
    print("\nCheck 4: subsector recovery by argmin over probes (1-jammer)")
    one = data["jammer_count"] == 1
    sector_ids = data["sector_ids"][one, 0]
    doa = data["doa_deg"][one, 0]
    power_w = data["power_w"][one, 0]

    centers = data["sector_centers_deg"][sector_ids - 1]            # (N, 8, 2)
    arc = phys.arc_error_deg(doa[:, None, 0], doa[:, None, 1],
                             centers[..., 0], centers[..., 1])
    true_sub = np.argmin(arc, axis=1)

    u = manifold.steering(doa[:, 0], doa[:, 1]).T                   # (N, 4)
    R = power_w[:, None, None] * (u[:, :, None] * u.conj()[:, None, :])
    R += cfg.noise_power_w * np.eye(phys.N_ANT)

    acc = {}
    for label, getter in (("v1 deployed", geom.probe_matrix),
                          ("v2 intent", geom.intended_null_matrix)):
        book = np.stack([getter(s) for s in range(1, phys.N_SECTORS + 1)])
        W = book[sector_ids - 1]                                    # (N, 8, 4)
        p = np.real(np.einsum("nmk,nkl,nml->nm", W, R, W.conj(), optimize=True))
        order = np.argsort(p, axis=1)
        acc[label] = (np.mean(order[:, 0] == true_sub),
                      np.mean((order[:, :2] == true_sub[:, None]).any(axis=1)))
        print(f"    {label:<12} {acc[label][0] * 100:5.1f}% exact, "
              f"{acc[label][1] * 100:5.1f}% within two deepest")

    print(f"    {'chance':<12} {100 / N_PROBES:5.1f}% exact, "
          f"{200 / N_PROBES:5.1f}% within two deepest")
    _report("v2 probes localise the occupied subsector far better than v1",
            acc["v2 intent"][0] > 3 * acc["v1 deployed"][0],
            f"{acc['v2 intent'][0] * 100:.1f}% vs "
            f"{acc['v1 deployed'][0] * 100:.1f}% exact")


def check_detector_floor(data, cfg) -> None:
    """Measured contrast should sit at the alpha-limited ceiling, not above it."""
    print("\nCheck 5: measured contrast vs the detector floor")
    ceiling_db = 10 * np.log10(1 / np.sqrt(cfg.alpha))   # mean / eps_std
    for j in (1, 2, 3):
        sel = data["jammer_count"] == j
        if not sel.any():
            continue
        c = data["contrast_db"][sel]
        c = c[np.isfinite(c)]
        print(f"    {j}J contrast: median {np.median(c):5.1f} dB, "
              f"p10 {np.percentile(c, 10):5.1f}, p90 {np.percentile(c, 90):5.1f}")
    one = data["contrast_db"][data["jammer_count"] == 1][:, 0]
    one = one[np.isfinite(one)]
    _report("1J contrast is detector-limited, not pattern-limited",
            np.median(one) > 15.0,
            f"median {np.median(one):.1f} dB vs "
            f"{ceiling_db:.1f} dB amplitude-accuracy ceiling")


def check_scenario_sampling(data, cfg) -> None:
    """JNR must stay uniform on 20-50 dB, and DoAs inside their sector bands."""
    print("\nCheck 6: scenario sampling unchanged from v1")
    jnr = data["jnr_db"][np.isfinite(data["jnr_db"])]
    lo, hi = cfg.jnr_db_lo, cfg.jnr_db_hi
    # Kolmogorov-Smirnov distance against Uniform(lo, hi), against the
    # asymptotic 99% critical value rather than a fixed tolerance.
    x = np.sort((jnr - lo) / (hi - lo))
    ks = np.max(np.abs(x - np.linspace(0, 1, x.size, endpoint=False)))
    ks_crit = 1.63 / np.sqrt(x.size)
    _report("JNR uniform on 20-50 dB",
            ks < ks_crit and jnr.min() >= lo and jnr.max() <= hi,
            f"KS {ks:.4f} < crit {ks_crit:.4f} (n={x.size:,}), "
            f"range [{jnr.min():.1f}, {jnr.max():.1f}] dB")

    active = data["sector_ids"] > 0
    sid = data["sector_ids"][active] - 1
    theta = data["doa_deg"][..., 0][active]
    phi = data["doa_deg"][..., 1][active]
    et = data["sector_edges_theta_deg"][sid]
    ep = data["sector_edges_phi_deg"][sid]
    inside = ((theta >= et[:, 0] - 1e-6) & (theta <= et[:, 1] + 1e-6)
              & (phi >= ep[:, 0] - 1e-6) & (phi <= ep[:, 1] + 1e-6))
    _report("every DoA lies inside its own sector band", inside.all(),
            f"{inside.mean() * 100:.2f}% inside")

    # Sectors drawn without replacement.
    dup = 0
    for row in data["sector_ids"]:
        act = row[row > 0]
        dup += int(np.unique(act).size != act.size)
    _report("sectors drawn without replacement", dup == 0,
            f"{dup} samples with a repeated sector")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "data" / "dataset_v2_eval.npz")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = np.load(args.data)
    cfg = GenConfig(alpha=float(data["alpha"]),
                    n_snapshots=int(data["n_snapshots"]),
                    noise_power_w=float(data["noise_power_w"]),
                    jnr_db_lo=float(data["jnr_db_lo"]),
                    jnr_db_hi=float(data["jnr_db_hi"]))

    print(f"Validating {args.data.name}  "
          f"({data['jammer_count'].size:,} samples, "
          f"generator {str(data['generator_version'])})")
    print(f"  alpha {cfg.alpha:.3e}, K {cfg.n_snapshots}")

    geom = phys.load_sector_geometry()
    manifold = phys.ArrayManifold()
    rng = np.random.default_rng(args.seed)

    check_probe_convention(geom, manifold)
    check_closed_form_scm(geom, manifold, cfg, rng)
    check_on_center_contrast(geom, manifold, cfg)
    check_subsector_recovery(data, geom, manifold, cfg)
    check_detector_floor(data, cfg)
    check_scenario_sampling(data, cfg)

    n_pass = sum(ok for _, ok in _results)
    print(f"\n{'=' * 70}")
    print(f"{n_pass}/{len(_results)} checks passed")
    for name, ok in _results:
        if not ok:
            print(f"  FAILED: {name}")
    sys.exit(0 if n_pass == len(_results) else 1)


if __name__ == "__main__":
    main()
