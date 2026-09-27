#!/usr/bin/env python3
"""
Phase 2 (v2): two-jammer spatial resolvability.

Resolution is scored with the dynamic criterion agreed for v1: a sample counts as
resolved when BOTH Hungarian-matched predictions land within Delta/3 of their
assigned truth, where Delta is the true separation. A fixed angular gate is
unusable here because at small Delta it would accept the inter-jammer midpoint as
a success, and at large Delta it would reject genuinely resolved pairs.

The centroid-collapse diagnostic is included directly: an estimator that gives up
and predicts the midpoint has error exactly Delta/2, which is permanently outside
the Delta/3 gate. Plotting the measured error against both Delta/2 and Delta/3
shows at a glance whether failures are collapse or noise.

Figures:
  phase2_resolution   resolution probability and conditional error vs separation
"""

from __future__ import annotations

import numpy as np

import v2_common as C


def main() -> None:
    C.setup_style()
    with C.Tee(C.RESULT_DIR / "PHASE2_V2.txt"):
        ev = C.load_eval()
        two = np.flatnonzero(ev.mask(2))
        sep = ev.separation[two]
        arc = ev.arc_matched[two, :2]
        jnr = ev.jnr_db[two, :2]
        d_jnr = np.abs(jnr[:, 0] - jnr[:, 1])

        worst = arc.max(axis=1)
        resolved = worst < sep / 3.0
        swapped = (ev.match_perm[two, 0] != 0)

        print("=" * 70)
        print("PHASE 2 (v2): TWO-JAMMER RESOLVABILITY")
        print("=" * 70)
        print(f"{two.size:,} two-jammer samples")
        print(f"  overall resolution rate (both within Delta/3): "
              f"{resolved.mean() * 100:.1f}%")
        print(f"  query/truth swaps under Hungarian matching: "
              f"{swapped.mean() * 100:.1f}%")

        # Centroid collapse: is the error tracking Delta/2?
        midpoint = C.arc_deg(
            ev.doa_pred[two, 0], ev.doa_true[two, 0]) / np.maximum(sep, 1e-6)
        print(f"\n  median error / separation = {np.median(midpoint):.3f}")
        print("    0.50 would be exact midpoint collapse, 0.00 perfect recovery")

        print(f"\n  restricted to near-equal power (|dJNR| <= 5 dB): "
              f"{resolved[d_jnr <= 5].mean() * 100:.1f}% resolved "
              f"(n={int((d_jnr <= 5).sum()):,})")
        print(f"  restricted to |dJNR| > 15 dB: "
              f"{resolved[d_jnr > 15].mean() * 100:.1f}% resolved "
              f"(n={int((d_jnr > 15).sum()):,})")

        edges = np.arange(0, 95, 5.0)
        mid = 0.5 * (edges[:-1] + edges[1:])
        prob, cond, n_bin, mean_err = [], [], [], []
        print(f"\n  {'separation':>12} {'n':>7} {'resolved':>10} {'mean err':>10}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sep >= lo) & (sep < hi)
            n_bin.append(m.sum())
            if m.sum() < 10:
                prob.append(np.nan); cond.append(np.nan); mean_err.append(np.nan)
                continue
            prob.append(resolved[m].mean() * 100)
            mean_err.append(arc[m].mean())
            cond.append(arc[m & resolved].mean() if (m & resolved).any() else np.nan)
            print(f"  {f'{lo:.0f}-{hi:.0f}':>12} {m.sum():>7,} "
                  f"{prob[-1]:>9.1f}% {mean_err[-1]:>10.2f}")
        prob, cond, mean_err = map(np.asarray, (prob, cond, mean_err))

        cross = np.nan
        ok = np.isfinite(prob)
        if (prob[ok] >= 50).any():
            i = np.argmax((prob >= 50) & ok)
            cross = mid[i]
            print(f"\n  50% resolution crossing near {cross:.0f} deg separation")
        else:
            print("\n  resolution never reaches 50% at any separation")

        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

        a1.plot(mid, prob, "o-", lw=1.8, ms=4)
        a1.axhline(50, color="k", ls=":", lw=1.4)
        if np.isfinite(cross):
            a1.axvline(cross, color="crimson", ls="--", lw=1.5,
                       label=f"50% at {cross:.0f}$^\\circ$")
            a1.legend()
        a1.set_xlabel(r"True separation $\Delta$ [$^\circ$]")
        a1.set_ylabel("Resolution probability [%]")
        a1.set_title(r"(a) Resolved when both errors $<\Delta/3$")
        a1.set_ylim(0, 100)

        a2.plot(mid, mean_err, "o-", lw=1.8, ms=4, label="mean matched error")
        a2.plot(mid, mid / 2.0, "r--", lw=1.6, label=r"$\Delta/2$ (midpoint collapse)")
        a2.plot(mid, mid / 3.0, "k:", lw=1.6, label=r"$\Delta/3$ (success gate)")
        a2.set_xlabel(r"True separation $\Delta$ [$^\circ$]")
        a2.set_ylabel(r"Arc error [$^\circ$]")
        a2.set_title("(b) Error against the collapse signature")
        a2.legend(fontsize=9)

        fig.tight_layout()
        C.save_fig(fig, "phase2_resolution", "phase2")
        plt.close(fig)


if __name__ == "__main__":
    main()
