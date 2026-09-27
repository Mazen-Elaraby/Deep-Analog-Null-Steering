#!/usr/bin/env python3
"""
Where the sub-CRLB behaviour comes from: bias and scatter, mapped spatially.

Phase 1 reports RMSE / RMS(CRLB) = 1.93x on well-conditioned geometries and
0.70x on ill-conditioned ones. A ratio under 1 cannot be a better unbiased
estimator, so it has to be bias -- but "has to be" is an argument, not a
measurement, and the eval split cannot settle it: it visits each direction once,
so any bias estimated by binning it is contaminated by the spread of true DoAs
inside the bin.

This does settle it. Each grid direction is measured `n_trials` times with the
DoA held FIXED and only the detector noise redrawn, so

    MSE(direction) = |bias(direction)|^2 + scatter(direction)^2

is an identity rather than an inference, and the CRLB at that same direction is
computed from the same codebook. Three claims then become falsifiable:

  1. the ill-conditioned directions are bias-dominated, not scatter-dominated;
  2. those are exactly the directions where RMSE sits below the CRLB;
  3. the bias points toward the sector centre -- i.e. the estimator is falling
     back on the geometric anchor Branch A can always produce, which is what
     "architectural bias" has to mean if it means anything measurable.

Claim 3 is the one that ties the statistics to the hardware: the anchor is a
convex combination of subsector centres, so its fallback is a direction inside
the sector, and a bias that points there is the fallback being exercised.

Figure:
  phase1_bias_variance   bias map, scatter map, bias-direction quiver, and the
                         RMSE/CRLB vs bias-fraction scatter that proves claim 2
"""

from __future__ import annotations

import numpy as np

import v2_common as C
import scene_synth as S
from crlb_v2 import crlb_v2_from_jnr

THETA_AXIS = np.arange(5.0, 88.0, 4.0)      # 21 rings
PHI_AXIS = np.arange(0.0, 360.0, 7.5)       # 48 spokes
N_TRIALS = 96
JNR_DB = 35.0


def add_bounds(g, sm):
    """Per-direction CRLB, from the same codebook the measurement used."""
    crb = np.full(g["doa"].shape[0], np.nan)
    for i, (doa, sec) in enumerate(zip(g["doa"], g["sector"])):
        W = sm.geom.intended_null_matrix(int(sec))
        r = crlb_v2_from_jnr(W, doa[None, :], [JNR_DB], sm.manifold, sm.cfg)
        crb[i] = r.arc_deg[0]
    g["crb"] = crb
    g["bias_fraction"] = g["bias_mag"] ** 2 / np.maximum(g["rmse"] ** 2, 1e-30)
    g["efficiency"] = g["rmse"] / crb
    return g


def _as_map(g, values):
    """Scatter the per-direction values back onto the (phi, theta) grid."""
    nt, np_ = g["theta_axis"].size, g["phi_axis"].size
    out = np.full((nt, np_), np.nan)
    ti = np.searchsorted(g["theta_axis"], g["doa"][:, 0])
    pi = np.searchsorted(g["phi_axis"], g["doa"][:, 1])
    out[ti, pi] = values
    return out


def figure(g):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    th, ph = g["theta_axis"], g["phi_axis"]
    # Cell edges, so pcolormesh puts each measured direction at a cell centre.
    def edges(a):
        d = np.diff(a).mean()
        return np.r_[a - d / 2, a[-1] + d / 2]

    fig, ax = plt.subplots(2, 2, figsize=(11.6, 8.4))

    for k, (key, title, cmap) in enumerate(
            [("bias_mag", r"(a) $|\mathrm{bias}|$ [$^\circ$]", "inferno"),
             ("scatter", r"(b) scatter (std) [$^\circ$]", "viridis")]):
        a = ax[0, k]
        m = a.pcolormesh(edges(ph), edges(th), _as_map(g, g[key]), cmap=cmap,
                         norm=LogNorm(vmin=0.2, vmax=40.0), shading="flat")
        a.set_title(title)
        a.set_xlabel(r"azimuth $\phi$ [$^\circ$]")
        a.set_ylabel(r"zenith $\theta$ [$^\circ$]")
        a.invert_yaxis()
        a.grid(False)
        fig.colorbar(m, ax=a, pad=0.02)
        for b in (90, 180, 270):
            a.axvline(b, color="w", lw=0.8, alpha=0.6)
        a.axhline(60, color="w", lw=0.8, alpha=0.6)

    # (c) bias direction. Arrows are drawn in (phi, theta) axes, so the
    # azimuthal component is converted back from arc to phi degrees; the radial
    # component is already a theta displacement.
    a = ax[1, 0]
    m = a.pcolormesh(edges(ph), edges(th), _as_map(g, g["bias_fraction"]),
                     cmap="RdYlBu_r", vmin=0, vmax=1, shading="flat")
    sin_t = np.sin(np.deg2rad(g["doa"][:, 0]))
    a.quiver(g["doa"][:, 1], g["doa"][:, 0],
             g["bias_p"] / np.maximum(sin_t, 0.1), g["bias_t"],
             angles="xy", scale_units="xy", scale=1.0, width=0.0035,
             color="k", alpha=0.85)
    centres = S.sector_centres(g["_geom"])
    a.scatter(centres[:, 1], centres[:, 0], marker="*", s=150, c="lime",
              edgecolors="k", linewidths=0.6, zorder=5, label="sector centre")
    a.set_title(r"(c) bias vector over bias fraction $|b|^2/\mathrm{MSE}$")
    a.set_xlabel(r"azimuth $\phi$ [$^\circ$]")
    a.set_ylabel(r"zenith $\theta$ [$^\circ$]")
    a.set_xlim(ph[0] - 4, ph[-1] + 4)
    a.set_ylim(th[-1] + 4, th[0] - 4)
    a.grid(False)
    a.legend(loc="upper right", fontsize=8)
    fig.colorbar(m, ax=a, pad=0.02)

    # (d) the actual claim: below-bound points are the bias-dominated ones.
    a = ax[1, 1]
    sc = a.scatter(g["bias_fraction"], g["efficiency"], c=g["crb"], s=14,
                   cmap="cividis", norm=LogNorm(), rasterized=True)
    a.axhline(1.0, color="k", ls="--", lw=1.5)
    a.set_yscale("log")
    a.set_xlabel(r"bias fraction $|b|^2/\mathrm{MSE}$")
    a.set_ylabel(r"RMSE / CRLB")
    a.set_title("(d) sub-bound points are bias-dominated")
    a.text(0.03, 0.06, "below the bound\n(biased, not superefficient)",
           transform=a.transAxes, fontsize=8,
           bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.colorbar(sc, ax=a, pad=0.02, label=r"CRLB [$^\circ$]")

    fig.tight_layout()
    C.save_fig(fig, "phase1_bias_variance", "phase1")
    plt.close(fig)


def main() -> None:
    C.setup_style()
    with C.Tee(C.RESULT_DIR / "PHASE1_BIAS_VARIANCE_V2.txt"):
        print("=" * 70)
        print("SPATIAL BIAS / VARIANCE DECOMPOSITION (1 JAMMER)")
        print("=" * 70)
        sm = S.SceneModel()
        print(f"grid: {THETA_AXIS.size} zenith rings x {PHI_AXIS.size} azimuth "
              f"spokes, {N_TRIALS} noise trials each, JNR {JNR_DB:.0f} dB")
        print("the DoA is held fixed within a direction, so MSE = |bias|^2 + "
              "scatter^2 exactly\n")

        g = S.build_bias_variance_grid(sm, THETA_AXIS, PHI_AXIS,
                                       n_trials=N_TRIALS, jnr_db=JNR_DB)
        g["_geom"] = sm.geom
        g = add_bounds(g, sm)
        n = g["doa"].shape[0]
        print(f"  {n:,} directions x {N_TRIALS} trials = "
              f"{n * N_TRIALS:,} queries")

        print(f"\n  median |bias|   {np.median(g['bias_mag']):.3f} deg")
        print(f"  median scatter  {np.median(g['scatter']):.3f} deg")
        print(f"  median RMSE     {np.median(g['rmse']):.3f} deg")
        print(f"  median bias fraction {np.median(g['bias_fraction']):.3f}"
              "   (1.0 = purely systematic, 0.0 = purely random)")
        print(f"  -> the error is predominantly "
              f"{'SYSTEMATIC' if np.median(g['bias_fraction']) > 0.5 else 'RANDOM'}")

        # Claim 1 and 2: split by the bound, exactly as Phase 1 does.
        med = np.median(g["crb"])
        good = g["crb"] <= med
        print(f"\n  split at the median CRLB ({med:.3f} deg):")
        print(f"  {'half':>18} {'RMSE':>8} {'|bias|':>8} {'scatter':>9} "
              f"{'bias frac':>10} {'RMSE/CRLB':>10}")
        for lab, m in (("well-conditioned", good), ("ill-conditioned", ~good)):
            print(f"  {lab:>18} {np.sqrt(np.mean(g['rmse'][m]**2)):>8.3f} "
                  f"{np.median(g['bias_mag'][m]):>8.3f} "
                  f"{np.median(g['scatter'][m]):>9.3f} "
                  f"{np.median(g['bias_fraction'][m]):>10.3f} "
                  f"{np.sqrt(np.mean(g['rmse'][m]**2)) / np.sqrt(np.mean(g['crb'][m]**2)):>10.2f}")

        below = g["efficiency"] < 1.0
        if below.any():
            print(f"\n  directions with RMSE below their own CRLB: "
                  f"{below.sum()}/{n} ({100*below.mean():.1f}%)")
            print(f"    their median bias fraction {np.median(g['bias_fraction'][below]):.3f}"
                  f"  vs {np.median(g['bias_fraction'][~below]):.3f} for the rest")
            print("    A genuinely unbiased estimator cannot sit below the bound,")
            print("    so this is the regularised estimator trading bias for")
            print("    variance where the geometry carries little information.")

        # Claim 3: does the bias point at the anchor's fallback?
        cw = np.isfinite(g["cos_inward"])
        w = g["bias_mag"][cw]
        mean_cos = float(np.average(g["cos_inward"][cw], weights=w))
        frac_in = float(np.average((g["cos_inward"][cw] > 0).astype(float),
                                   weights=w))
        print(f"\n  bias direction vs the bearing to the sector centre:")
        print(f"    magnitude-weighted mean cos  {mean_cos:+.3f}  "
              f"(+1 = straight at the centre, 0 = unrelated)")
        print(f"    magnitude-weighted P(points inward) {frac_in:.3f}")
        print(f"    unweighted P(points inward) "
              f"{(g['cos_inward'][cw] > 0).mean():.3f}")
        verdict = ("TOWARD the sector centre -> the anchor's fallback is being "
                   "exercised" if mean_cos > 0.2 else
                   "NOT systematically toward the sector centre")
        print(f"    -> the bias points {verdict}")

        print(f"\n  by zenith band:")
        print(f"  {'theta band':>14} {'n':>5} {'|bias|':>8} {'scatter':>9} "
              f"{'CRLB':>8} {'RMSE/CRLB':>10}")
        for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 75), (75, 90)):
            m = (g["doa"][:, 0] >= lo) & (g["doa"][:, 0] < hi)
            if m.sum() < 3:
                continue
            print(f"  {f'{lo}-{hi} deg':>14} {m.sum():>5} "
                  f"{np.median(g['bias_mag'][m]):>8.3f} "
                  f"{np.median(g['scatter'][m]):>9.3f} "
                  f"{np.median(g['crb'][m]):>8.3f} "
                  f"{np.sqrt(np.mean(g['rmse'][m]**2)) / np.sqrt(np.mean(g['crb'][m]**2)):>10.2f}")
        print("    The theta = 60 deg line is a sector boundary, so a jump across")
        print("    it is the partition showing through, not a property of the sky.")

        print("\nfigures")
        figure(g)


if __name__ == "__main__":
    main()
