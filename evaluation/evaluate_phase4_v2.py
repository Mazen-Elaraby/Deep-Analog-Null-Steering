#!/usr/bin/env python3
"""
Phase 4 (v2): operational jammer suppression.

Arc error is the estimator metric; JSR is what the system actually delivers. The
mapping between them is steep and nonlinear, because suppression depends on how
far the null has been steered relative to the array's null width, not on the
angular error directly. A few degrees of error near zenith costs far less than
the same error near the horizon, where the manifold varies fastest.

Part 1 (1J) establishes the suppression ceiling without multi-target masking.
Part 2 (2J) maps mean JSR over the dJNR x separation plane, matching Phase 3.5's
axes so the estimator and operational views can be read side by side.

As in Phase 3.5, every quoted STATISTIC comes from the held-out eval split
while the 2J MAP is sampled cell by cell. The axes, the cell grid and the
synthesised scene set are imported from Phase 3.5 rather than redefined, so
the error map and the JSR map describe exactly the same scenes and can be laid
side by side without a caveat about differing sample sets.

Figures:
  phase4_jsr_vs_error   1J scatter of JSR against arc error
  phase4_2_jsr_heatmap  2J mean JSR over dJNR x separation
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binned_statistic_2d

import scene_synth as S
import v2_common as C
import crpa_physics as phys
from evaluate_phase3_5_v2 import GRID_CACHE, MIN_CELL, X_EDGES, Y_EDGES


def part1_single(ev, manifold, n_max=4000, seed=0):
    import matplotlib.pyplot as plt
    idx = np.flatnonzero(ev.mask(1))
    rng = np.random.default_rng(seed)
    if idx.size > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))

    jsr, err, degen = [], [], 0
    for i in idx:
        v, d = C.jsr_db(ev.doa_pred[i, 0], ev.doa_true[i, 0], manifold)
        degen += int(d)
        jsr.append(v[0])
        err.append(ev.arc_direct[i, 0])
    jsr, err = np.asarray(jsr), np.asarray(err)

    print(f"  1J samples evaluated: {idx.size:,} "
          f"({degen} near-zenith degenerate)")
    print(f"  JSR  median {np.median(jsr):.2f} dB, "
          f"p10 {np.percentile(jsr,10):.2f}, p90 {np.percentile(jsr,90):.2f}")
    for gate in (10.0, 20.0, 30.0):
        print(f"    P(JSR >= {gate:.0f} dB) = {(jsr >= gate).mean()*100:.1f}%")
    for e in (1.0, 3.0, 5.0):
        m = err <= e
        if m.any():
            print(f"    median JSR when arc error <= {e:.0f} deg: "
                  f"{np.median(jsr[m]):.2f} dB  (n={m.sum():,})")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(err, jsr, s=6, alpha=0.35, c=ev.jnr_db[idx, 0],
                    cmap="viridis", rasterized=True)
    ax.axhline(np.median(jsr), color="crimson", ls="--", lw=1.6,
               label=f"median JSR = {np.median(jsr):.1f} dB")
    ax.set_xscale("log")
    ax.set_xlabel(r"Great-circle arc error [$^\circ$]")
    ax.set_ylabel("Achieved JSR [dB]")
    ax.set_title("Single-jammer suppression vs estimation error")
    ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, pad=0.02, label="JNR [dB]")
    txt = (f"median arc err  {np.median(err):.2f}$^\\circ$\n"
           f"median JSR      {np.median(jsr):.1f} dB\n"
           f"P90 JSR         {np.percentile(jsr, 90):.1f} dB")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, va="bottom", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.tight_layout()
    C.save_fig(fig, "phase4_jsr_vs_error", "phase4")
    plt.close(fig)


def part2_two(ev, manifold, n_max=4000, seed=0):
    import matplotlib.pyplot as plt
    idx = np.flatnonzero(ev.mask(2))
    rng = np.random.default_rng(seed)
    if idx.size > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))

    x, y, z = [], [], []
    for i in idx:
        # Joint nulling: both predictions steer one weight vector.
        v, _ = C.jsr_db(ev.doa_pred[i, :2], ev.doa_true[i, :2], manifold)
        j0, j1 = ev.jnr_db[i, 0], ev.jnr_db[i, 1]
        x += [j0 - j1, j1 - j0]
        y += [ev.separation[i], ev.separation[i]]
        z += [v[0], v[1]]
    x, y, z = map(np.asarray, (x, y, z))

    print(f"\n  2J samples evaluated: {idx.size:,} "
          f"({len(z):,} target evaluations)")
    weak, strong = x <= -10, x >= 10
    print(f"  mean JSR   weak target {z[weak].mean():.2f} dB, "
          f"strong target {z[strong].mean():.2f} dB")
    wide = y >= 60
    print(f"  mean JSR   separation >= 60 deg: {z[wide].mean():.2f} dB, "
          f"< 30 deg: {z[y < 30].mean():.2f} dB")

    # -- the map: same plane, sampled where the figure needs samples --------
    print("\n  heatmap coverage")
    print("    " + S.coverage_report(x, y, X_EDGES, Y_EDGES, MIN_CELL,
                                     label="eval split "))
    sm = S.SceneModel()
    g = S.build_2j_grid(sm, X_EDGES, Y_EDGES, per_cell=30, with_jsr=True,
                        cache=GRID_CACHE)
    gx, gy, gz = g["d_jnr"], g["sep"], g["jsr"]
    ok = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gz)
    gx, gy, gz = gx[ok], gy[ok], gz[ok]
    print("    " + S.coverage_report(gx, gy, X_EDGES, Y_EDGES, MIN_CELL,
                                     label="synthesised"))
    print(f"    {gz.size:,} target evaluations from "
          f"{int(g['n_scenes'][0]):,} synthesised 2J scenes")
    print(f"    mean JSR   weak target {gz[gx <= -10].mean():.2f} dB, "
          f"strong target {gz[gx >= 10].mean():.2f} dB  (grid, uniform over "
          f"the plane)")

    stat, _, _, _ = binned_statistic_2d(gx, gy, gz, "mean",
                                        bins=[X_EDGES, Y_EDGES])
    cnt, _, _, _ = binned_statistic_2d(gx, gy, gz, "count",
                                       bins=[X_EDGES, Y_EDGES])
    stat = np.where(cnt >= MIN_CELL, stat, np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # Unreachable cells read as grey, not as blank paper.
    ax.set_facecolor("0.82")
    mesh = ax.pcolormesh(X_EDGES, Y_EDGES, stat.T, cmap="viridis",
                         vmin=0, vmax=50, shading="flat")
    ax.axvline(0, color="white", ls="--", lw=1.6)
    ax.set_xlabel(r"$\Delta$JNR (target $-$ interferer) [dB]")
    ax.set_ylabel(r"True great-circle separation [$^\circ$]")
    ax.set_title("Two-jammer joint nulling")
    ax.grid(False)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Mean JSR [dB]")
    fig.tight_layout()
    C.save_fig(fig, "phase4_2_jsr_heatmap", "phase4")
    plt.close(fig)
    print(f"  heatmap cells rendered: {int(np.isfinite(stat).sum())}/{stat.size}")


def main() -> None:
    C.setup_style()
    with C.Tee(C.RESULT_DIR / "PHASE4_V2.txt"):
        ev = C.load_eval()
        manifold = phys.ArrayManifold()

        print("=" * 70)
        print("PHASE 4 (v2): JAMMER SUPPRESSION")
        print("=" * 70)
        print("JSR capped at 50 dB (finite hardware dynamic range), floored at 0.\n")
        part1_single(ev, manifold)
        part2_two(ev, manifold)


if __name__ == "__main__":
    main()
