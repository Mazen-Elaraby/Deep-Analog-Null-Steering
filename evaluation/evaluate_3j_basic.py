#!/usr/bin/env python3
"""
Three-jammer case: headline numbers only, with the conditioning caveat.

This is deliberately shallow. The identifiability analysis places J <= 5 for the
real HFSS manifold (16 independent observables, against 9 under the idealised
common-element-pattern assumption that `Algorithmic Bounds.md` Section I uses to
derive J <= 2), so three jammers are algebraically identifiable. But
identifiable is not the same as well conditioned: at J = 3 the system consumes 10
of the 16 observables and the FIM becomes badly scaled, so the achievable
accuracy is poor for reasons that are a property of the array, not the estimator.

Reporting arc error, suppression and resolution rate is therefore appropriate.
Reading spatial structure into the residuals would not be.

Figure:
  phase3j_summary   error CDF by jammer count and JSR by jammer count
"""

from __future__ import annotations

import numpy as np

import v2_common as C
import crpa_physics as phys


def main() -> None:
    C.setup_style()
    with C.Tee(C.RESULT_DIR / "PHASE3J_V2.txt"):
        ev = C.load_eval()
        manifold = phys.ArrayManifold()

        print("=" * 70)
        print("THREE-JAMMER BASIC RESULTS")
        print("=" * 70)
        print("Reported for completeness. J=3 is identifiable on the measured")
        print("manifold but ill-conditioned; see the caveat at the end.\n")

        print(f"  {'case':>12} {'n':>8} {'p50':>8} {'p90':>8} {'RMSE':>8}")
        for j in (1, 2, 3):
            m = ev.mask(j)
            e = ev.arc_direct[m, :j].ravel()
            e = e[np.isfinite(e)]
            print(f"  {f'{j} jammer':>12} {e.size:>8,} {np.median(e):>8.2f} "
                  f"{np.percentile(e, 90):>8.2f} {np.sqrt(np.mean(e**2)):>8.2f}")

        # Resolution: all three within Delta_min/3, where Delta_min is the
        # closest true pair. The gate has to key off the tightest pair, since
        # that is what sets the difficulty.
        three = np.flatnonzero(ev.mask(3))
        doa = ev.doa_true[three, :3]
        pairs = [(0, 1), (0, 2), (1, 2)]
        seps = np.stack([C.arc_deg(doa[:, a], doa[:, b]) for a, b in pairs], 1)
        d_min = seps.min(axis=1)
        worst = ev.arc_matched[three, :3].max(axis=1)
        resolved = worst < d_min / 3.0
        print(f"\n  3J resolution (all three within Delta_min/3): "
              f"{resolved.mean()*100:.1f}%")
        print(f"  median closest-pair separation: {np.median(d_min):.1f} deg")

        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(three, min(2000, three.size), replace=False))
        jsr_all, degen = [], 0
        for i in sub:
            v, d = C.jsr_db(ev.doa_pred[i, :3], ev.doa_true[i, :3], manifold)
            degen += int(d)
            jsr_all.append(v)
        jsr_all = np.concatenate(jsr_all)
        print(f"\n  3J joint nulling over {sub.size:,} samples "
              f"({degen} degenerate)")
        print(f"    mean JSR {jsr_all.mean():.2f} dB, "
              f"median {np.median(jsr_all):.2f} dB")
        print(f"    P(JSR >= 10 dB) = {(jsr_all >= 10).mean()*100:.1f}%")
        print("    Note: 3 nulls from a 4-element array leaves a single degree of")
        print("    freedom, so even exact DoAs would give limited quiescent gain.")

        print("\n  CAVEAT")
        print("  Degradation from J=2 to J=3 is not evidence of an estimator")
        print("  deficiency. Ten of the sixteen available real observables are")
        print("  consumed at J=3 and the FIM is badly scaled, so the accuracy")
        print("  ceiling is set by the array. Treat these as feasibility numbers.")

        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for j, style in zip((1, 2, 3), ("-", "--", ":")):
            m = ev.mask(j)
            e = ev.arc_direct[m, :j].ravel()
            e = np.sort(e[np.isfinite(e)])
            a1.plot(e, np.linspace(0, 1, e.size), style, lw=1.9,
                    label=f"{j} jammer{'s' if j > 1 else ''}")
        a1.set_xlim(0, 60)
        a1.set_ylim(0, 1)
        a1.set_xlabel(r"Arc error [$^\circ$]")
        a1.set_ylabel("Empirical CDF")
        a1.set_title("(a) Error vs jammer count")
        a1.legend(loc="lower right")

        a2.hist(jsr_all, bins=np.arange(0, 52, 2.5), color="tab:green",
                alpha=0.8, edgecolor="k", linewidth=0.5)
        a2.axvline(np.median(jsr_all), color="crimson", ls="--", lw=1.6,
                   label=f"median {np.median(jsr_all):.1f} dB")
        a2.set_xlabel("Achieved JSR [dB]")
        a2.set_ylabel("Count")
        a2.set_title("(b) Three-jammer joint nulling")
        a2.legend()

        fig.tight_layout()
        C.save_fig(fig, "phase3j_summary", "phase3j")
        plt.close(fig)


if __name__ == "__main__":
    main()
