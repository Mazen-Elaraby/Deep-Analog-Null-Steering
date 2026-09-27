#!/usr/bin/env python3
"""
Phase 1 (v2): single-jammer spatial accuracy against the detector-limited CRLB.

The narrative here inverts relative to v1. Under v1's fixed thermal noise the
bound fell as 1/sqrt(P), so a flat measured error curve was evidence of a
bias-limited estimator sitting orders of magnitude above the limit. Under a
relative-accuracy detector the bound is JNR-independent, so a flat curve is the
*expected* shape and the question becomes how close to the floor the network sits.

Figures:
  phase1_spatial_map    hemisphere map of arc error vs true DoA
  phase1_error_cdf      CDF of azimuth, zenith and arc error
  phase1_jnr_efficiency error and CRLB vs JNR, plus the efficiency ratio
"""

from __future__ import annotations

import numpy as np

import v2_common as C
from crlb_v2 import crlb_v2_from_jnr
import crpa_physics as phys


def compute_bounds(ev, n_max=1500, seed=0):
    """Per-sample CRLB for a random 1J subset; the FIM inversion is the cost."""
    idx = np.flatnonzero(ev.mask(1))
    rng = np.random.default_rng(seed)
    if idx.size > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    manifold = phys.ArrayManifold()
    geom = phys.load_sector_geometry()

    crb, keep = [], []
    for i in idx:
        W = geom.intended_null_matrix(int(ev.sector_ids[i, 0]))
        r = crlb_v2_from_jnr(W, ev.doa_true[i, 0:1], [float(ev.jnr_db[i, 0])],
                             manifold, ev.cfg)
        if np.isfinite(r.arc_deg[0]):
            crb.append(r.arc_deg[0])
            keep.append(i)
    return np.array(keep), np.array(crb)


def fig_spatial_map(ev, sel):
    import matplotlib.pyplot as plt
    theta = ev.doa_true[sel, 0, 0]
    phi = ev.doa_true[sel, 0, 1]
    err = ev.arc_direct[sel, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             subplot_kw={"projection": "polar"})
    for ax, (vals, label, cmap, vmax) in zip(
            axes, [(err, r"Arc error [$^\circ$]", "viridis",
                    np.percentile(err, 95)),
                   (ev.contrast_db[sel, 0], "Probe contrast [dB]", "magma", None)]):
        sc = ax.scatter(np.deg2rad(phi), theta, c=vals, s=4, cmap=cmap,
                        vmin=0, vmax=vmax, rasterized=True)
        ax.set_theta_zero_location("N")
        ax.set_rlim(0, 90)
        ax.set_rlabel_position(135)
        ax.set_xlabel(r"azimuth $\phi$;  radius = zenith $\theta$")
        fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.85, label=label)
    axes[0].set_title("(a) Arc error over the hemisphere")
    axes[1].set_title("(b) Measured probe contrast")
    fig.tight_layout()
    C.save_fig(fig, "phase1_spatial_map", "phase1")
    plt.close(fig)


def fig_error_cdf(ev, sel, crb_median):
    import matplotlib.pyplot as plt
    arc = ev.arc_direct[sel, 0]
    pred, true = ev.doa_pred[sel, 0], ev.doa_true[sel, 0]
    dth = np.abs(pred[:, 0] - true[:, 0])
    dph = C.azimuth_error_deg(pred[:, 1], true[:, 1])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for v, lab, style in ((dth, r"zenith $|\Delta\theta|$", "-"),
                          (dph, r"azimuth $|\Delta\phi|$", "--"),
                          (arc, "great-circle arc", "-")):
        xs = np.sort(v)
        ax.plot(xs, np.linspace(0, 1, xs.size), style, lw=1.8, label=lab)
    ax.axvline(crb_median, color="k", ls=":", lw=1.6,
               label=f"median CRLB ({crb_median:.2f}$^\\circ$)")
    ax.set_xlim(0, np.percentile(arc, 99))
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Error [$^\circ$]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Single-jammer error distribution")
    ax.legend(loc="lower right")
    fig.tight_layout()
    C.save_fig(fig, "phase1_error_cdf", "phase1")
    plt.close(fig)


def fig_jnr_efficiency(ev, keep, crb, well_conditioned):
    """
    Error and bound vs JNR, restricted to well-conditioned geometries.

    The bound spans p10 0.9 deg to p90 15 deg across the hemisphere. Pooling all
    of it would put the quadrature-aggregated bound above the network's RMSE,
    because on ill-conditioned geometries the unbiased bound diverges while a
    regularised network stays near its sector. Restricting to the well-conditioned
    half is the comparison in which the efficiency ratio means what it says. The
    JNR-flatness being demonstrated holds on either subset.
    """
    import matplotlib.pyplot as plt
    keep = keep[well_conditioned]
    crb = crb[well_conditioned]
    jnr = ev.jnr_db[keep, 0]
    err = ev.arc_direct[keep, 0]
    edges = np.arange(20, 52, 2.0)
    mid = 0.5 * (edges[:-1] + edges[1:])

    rmse, p10, p90, bnd = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (jnr >= lo) & (jnr < hi)
        if m.sum() < 5:
            rmse.append(np.nan); p10.append(np.nan)
            p90.append(np.nan); bnd.append(np.nan)
            continue
        rmse.append(np.sqrt(np.mean(err[m] ** 2)))
        p10.append(np.percentile(err[m], 10))
        p90.append(np.percentile(err[m], 90))
        # Aggregate the bound in QUADRATURE, not linearly. For an unbiased
        # estimator var_i >= CRB_i^2 holds per sample, so the comparable
        # quantity is sqrt(mean(CRB^2)); using mean(CRB) understates it (Jensen)
        # and makes the network look like it beats the bound.
        bnd.append(np.sqrt(np.mean(crb[m] ** 2)))
    rmse, p10, p90, bnd = map(np.asarray, (rmse, p10, p90, bnd))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.fill_between(mid, p10, p90, alpha=0.25, label="10th-90th percentile")
    ax.plot(mid, rmse, "o-", lw=1.8, ms=4, label="network RMSE")
    ax.plot(mid, bnd, "k--", lw=1.8, label="CRLB (detector-limited)")
    ax.set_yscale("log")
    ax.set_ylabel(r"Arc error [$^\circ$]")
    ax.set_title("Single-jammer accuracy vs JNR\n"
                 "(well-conditioned geometries)")
    ax.legend()

    ax2.plot(mid, rmse / bnd, "s-", color="crimson", lw=1.8, ms=4)
    ax2.axhline(1.0, color="k", ls=":", lw=1.4)
    ax2.set_ylabel("RMSE / CRLB")
    ax2.set_xlabel("JNR [dB]")
    ax2.set_ylim(0, max(3.0, np.nanmax(rmse / bnd) * 1.15))
    fig.tight_layout()
    C.save_fig(fig, "phase1_jnr_efficiency", "phase1")
    plt.close(fig)

    return mid, rmse, bnd


def main() -> None:
    C.setup_style()
    with C.Tee(C.RESULT_DIR / "PHASE1_V2.txt"):
        ev = C.load_eval()
        sel = np.flatnonzero(ev.mask(1))
        arc = ev.arc_direct[sel, 0]

        print("=" * 70)
        print("PHASE 1 (v2): SINGLE-JAMMER ACCURACY")
        print("=" * 70)
        print(f"samples: {sel.size:,}")
        print(f"  arc error   p50 {np.median(arc):.3f}  "
              f"p90 {np.percentile(arc, 90):.3f}  "
              f"RMSE {np.sqrt(np.mean(arc**2)):.3f} deg")
        pred, true = ev.doa_pred[sel, 0], ev.doa_true[sel, 0]
        dth = np.abs(pred[:, 0] - true[:, 0])
        dph = C.azimuth_error_deg(pred[:, 1], true[:, 1])
        print(f"  zenith      p50 {np.median(dth):.3f}  p90 {np.percentile(dth,90):.3f} deg")
        print(f"  azimuth     p50 {np.median(dph):.3f}  p90 {np.percentile(dph,90):.3f} deg")
        print(f"  theta range of predictions: [{pred[:,0].min():.2f}, "
              f"{pred[:,0].max():.2f}] deg  (no folding needed, rec #5)")

        print("\ncomputing CRLB ...")
        keep, crb = compute_bounds(ev)
        e_keep = ev.arc_direct[keep, 0]
        print(f"  {keep.size:,} bounds, median {np.median(crb):.3f} deg, "
              f"p10 {np.percentile(crb,10):.3f}, p90 {np.percentile(crb,90):.3f}")
        rms_crb = np.sqrt(np.mean(crb ** 2))
        rmse = np.sqrt(np.mean(e_keep ** 2))
        print(f"  RMS of the bound {rms_crb:.3f} deg (quadrature aggregation)")
        print(f"  RMSE / RMS(CRLB) = {rmse / rms_crb:.2f}x")

        # The bound spans p10 0.9 deg to p90 15 deg across the hemisphere, and
        # where it is very loose the geometry is ill-conditioned. The network is a
        # regularised, biased estimator whose output stays near its sector, so on
        # those samples it can sit BELOW an unbiased bound. Restricting to
        # well-conditioned geometries is the informative comparison.
        med = np.median(crb)
        good = crb <= med
        print(f"\n  well-conditioned half (CRLB <= {med:.2f} deg):")
        print(f"    RMSE {np.sqrt(np.mean(e_keep[good]**2)):.3f} deg vs "
              f"RMS(CRLB) {np.sqrt(np.mean(crb[good]**2)):.3f} deg -> "
              f"{np.sqrt(np.mean(e_keep[good]**2)) / np.sqrt(np.mean(crb[good]**2)):.2f}x")
        print(f"  ill-conditioned half (CRLB > {med:.2f} deg):")
        print(f"    RMSE {np.sqrt(np.mean(e_keep[~good]**2)):.3f} deg vs "
              f"RMS(CRLB) {np.sqrt(np.mean(crb[~good]**2)):.3f} deg -> "
              f"{np.sqrt(np.mean(e_keep[~good]**2)) / np.sqrt(np.mean(crb[~good]**2)):.2f}x"
              "   (below 1 means the regulariser is biasing, not violating)")

        # The defining property of the v2 bound: flat in JNR.
        jnr = ev.jnr_db[keep, 0]
        lo, hi = jnr < 30, jnr > 40
        print(f"\n  CRLB at JNR<30 dB : {np.median(crb[lo]):.3f} deg")
        print(f"  CRLB at JNR>40 dB : {np.median(crb[hi]):.3f} deg"
              f"   -> flat, as Phase B predicts")
        e_lo = np.sqrt(np.mean(ev.arc_direct[keep][lo, 0] ** 2))
        e_hi = np.sqrt(np.mean(ev.arc_direct[keep][hi, 0] ** 2))
        print(f"  network RMSE  JNR<30 {e_lo:.3f} deg,  JNR>40 {e_hi:.3f} deg")

        print("\nfigures")
        fig_spatial_map(ev, sel)
        fig_error_cdf(ev, sel, float(np.median(crb)))
        fig_jnr_efficiency(ev, keep, crb, good)


if __name__ == "__main__":
    main()
