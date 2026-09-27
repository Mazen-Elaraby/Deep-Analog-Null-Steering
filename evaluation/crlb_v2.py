#!/usr/bin/env python3
"""
CRLB for the dataset v2 measurement model, plus proofs that it is the right one.

This is NOT a transcription of Section II of `Algorithmic Bounds.md`. Three
things in that section do not survive contact with the v2 generator, and each is
handled explicitly below and measured in `main()`.

1. NOISE MODEL. Section II uses C_n = sigma_det^2 I with sigma_det^2 = alpha
   pbar^2. The generator also carries a finite-snapshot term, so its true
   covariance is

       Var(p_m) = pbar_m^2 / K  +  alpha * pbar^2                     (diagonal)

   The two terms are comparable, not separated: with K = 8192 and
   alpha = 10^-3.5 the snapshot term is 1.22e-4 pbar_m^2 against a detector term
   of 3.16e-4 pbar^2, so their ratio is 0.386 (pbar_m / pbar)^2, which exceeds 1
   for any probe reading more than 1.6x the sector mean. The claim that
   K >> 1/alpha makes the detector term dominant is false at K = 8192.
   `snapshot_vs_detector_ratio()` reports this per probe.

2. PARAMETER-DEPENDENT COVARIANCE. Section II's F = (1/sigma_det^2) J^T J is the
   mean-sensitivity term only. Because C_n depends on eta through pbar, a
   Gaussian model contributes a second term

       F_cov[i,j] = 1/2 tr( C^-1 dC/deta_i C^-1 dC/deta_j )

   Dropping it is conservative -- F_mean <= F_mean + F_cov in the PSD order, and
   the Schur complement is matrix-monotone, so the marginalised angle bound can
   only shrink when it is included. It is available via `include_cov_term` and
   its size is reported rather than assumed small.

3. PROBE CONVENTION. Section II writes |w_m^H a|^2. The validated codebook nulls
   its own subsector centre under w_m^T u, so the document's w_m is this code's
   conj(w_m). Taken literally the document's notation would reinstate exactly the
   conjugation bug that v1 shipped. This module takes W in the same convention as
   the generator (rows applied as w_m^T u) and never conjugates it.

What DOES survive, and is verified numerically in `main()`:

- The 1-jammer bound is JNR-flat. Both noise terms scale as pbar^2 and the
  Jacobian scales as P ~ pbar, so the FIM is independent of JNR. This holds for
  any mix of the two noise terms, so it is robust to the K mis-specification.
- The -1 dB/dB weak-jammer law. In the strong-jammer-dominated regime every
  noise term scales as P_s^2 while the weak jammer's gradient scales as P_w, so
  Section III's conclusion is unaffected by the added snapshot term. Whether the
  *nuisance-marginalised* bound also shows -1 dB/dB is a separate question that
  Section III does not address; it is measured here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "dataset_generation"))

import crpa_physics as phys  # noqa: E402
from crpa_physics import ArrayManifold, N_ANT, NOISE_POWER_W, jnr_to_power_w  # noqa: E402
from generate_dataset_v2 import GenConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Mean model and its Jacobian
# ---------------------------------------------------------------------------

def probe_mean(W: np.ndarray, doas_deg: np.ndarray, powers_w: Sequence[float],
               manifold: ArrayManifold, noise_var: float = NOISE_POWER_W) -> np.ndarray:
    """
    pbar_m = sigma_n^2 ||w_m||^2 + sum_j P_j |w_m^T u_j|^2.

    Incoherent sources, so this is exactly diag(W R W^H) with no cross terms.
    """
    doas = np.atleast_2d(doas_deg)
    pbar = noise_var * np.sum(np.abs(W) ** 2, axis=1)
    for j, (theta, phi) in enumerate(doas):
        pbar = pbar + powers_w[j] * np.abs(W @ manifold.steering(theta, phi)) ** 2
    return pbar


def probe_jacobian(W: np.ndarray, doas_deg: np.ndarray, powers_w: Sequence[float],
                   manifold: ArrayManifold, noise_var: float = NOISE_POWER_W):
    """
    d pbar / d eta with eta = [theta_j, phi_j, ln P_j, ..., ln sigma_n^2].

    Angles in radians. Powers are log-parameterised for the same conditioning
    reason as the v1 module: with P ~ 1e-13 beside angles ~ 1 the FIM columns
    differ by ~1e13 and the inverse is destroyed by units alone. The
    marginalised angle bound is invariant to smooth reparameterisation of the
    nuisances.
    """
    doas = np.atleast_2d(doas_deg)
    n_j = doas.shape[0]
    columns, names = [], []

    for j, (theta, phi) in enumerate(doas):
        u, du_dt, du_dp = manifold.steering_and_derivatives(theta, phi)
        g = W @ u
        # d|g|^2/dx = 2 Re{conj(g) * (W du/dx)}
        columns.append(2.0 * powers_w[j] * np.real(np.conj(g) * (W @ du_dt)))
        names.append(f"theta_{j + 1}")
        columns.append(2.0 * powers_w[j] * np.real(np.conj(g) * (W @ du_dp)))
        names.append(f"phi_{j + 1}")
        # d P/d lnP = P
        columns.append(powers_w[j] * np.abs(g) ** 2)
        names.append(f"lnP_{j + 1}")

    columns.append(noise_var * np.sum(np.abs(W) ** 2, axis=1))
    names.append("ln_sigma_n2")
    return np.stack(columns, axis=1), names


# ---------------------------------------------------------------------------
# Noise covariance of the generator
# ---------------------------------------------------------------------------

def noise_covariance(pbar: np.ndarray, cfg: GenConfig, detector_only: bool = False):
    """
    Diagonal covariance actually realised by the generator.

    Returns (diagonal_variance, d_diag/d_pbar_m as a (M, M) matrix), the second
    being needed for the covariance-sensitivity FIM term.
    """
    M = pbar.size
    mean_p = pbar.mean()
    det_var = cfg.alpha * mean_p ** 2
    snap_var = (np.zeros_like(pbar) if detector_only
                else pbar ** 2 / cfg.n_snapshots)

    var = snap_var + det_var
    # d var_m / d pbar_l = 2 pbar_m/K delta_ml + 2 alpha mean_p / M
    dvar = np.full((M, M), 2.0 * cfg.alpha * mean_p / M)
    if not detector_only:
        dvar[np.diag_indices(M)] += 2.0 * pbar / cfg.n_snapshots
    return var, dvar


def snapshot_vs_detector_ratio(pbar: np.ndarray, cfg: GenConfig) -> np.ndarray:
    """Per-probe ratio of the snapshot variance to the detector variance."""
    return (pbar ** 2 / cfg.n_snapshots) / (cfg.alpha * pbar.mean() ** 2)


# ---------------------------------------------------------------------------
# Bound
# ---------------------------------------------------------------------------

@dataclass
class CRLBv2Result:
    arc_deg: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    arc_deg_known_nuisance: np.ndarray
    fim_cond: float = np.nan
    n_params: int = 0
    singular: bool = False


def crlb_v2(W: np.ndarray, doas_deg: np.ndarray, powers_w: Sequence[float],
            manifold: ArrayManifold, cfg: GenConfig,
            include_cov_term: bool = False, detector_only: bool = False,
            cond_limit: float = 1e12) -> CRLBv2Result:
    """
    Marginalised CRLB for the v2 model.

    `detector_only=True` reproduces Section II of `Algorithmic Bounds.md`
    (C_n = sigma_det^2 I) so the cost of that simplification can be measured.
    """
    doas = np.atleast_2d(np.asarray(doas_deg, dtype=float))
    n_j = doas.shape[0]

    pbar = probe_mean(W, doas, powers_w, manifold, cfg.noise_power_w)
    J_mat, names = probe_jacobian(W, doas, powers_w, manifold, cfg.noise_power_w)
    var, dvar = noise_covariance(pbar, cfg, detector_only)

    # Normalise so the algebra runs on O(1) numbers; FIM = J^T C^-1 J is
    # invariant under p -> p/scale.
    scale = max(pbar.max(), np.finfo(float).tiny)
    J_s = J_mat / scale
    var_s = var / scale ** 2

    fim = J_s.T @ (J_s / var_s[:, None])

    if include_cov_term:
        # F_cov[i,j] = 1/2 tr(C^-1 dC_i C^-1 dC_j); C diagonal, so this reduces
        # to a sum over probes. dC_i has diagonal dvar @ (dpbar/deta_i).
        D = (dvar @ J_mat) / scale ** 2          # (M, n_params) d var_m / d eta_i
        fim = fim + 0.5 * (D / var_s[:, None]).T @ (D / var_s[:, None])

    fim = 0.5 * (fim + fim.T)
    eigs = np.linalg.eigvalsh(fim)
    cond = float(np.abs(eigs).max() / max(np.abs(eigs).min(), np.finfo(float).tiny))
    singular = bool(eigs.min() <= 0 or cond > cond_limit)

    arc = np.full(n_j, np.nan)
    th = np.full(n_j, np.nan)
    ph = np.full(n_j, np.nan)
    arc_known = np.full(n_j, np.nan)

    for j in range(n_j):
        keep = [3 * j, 3 * j + 1]
        rest = [i for i in range(fim.shape[0]) if i not in keep]
        sin2 = np.sin(np.deg2rad(doas[j, 0])) ** 2

        F_aa = fim[np.ix_(keep, keep)]
        if rest:
            F_ab = fim[np.ix_(keep, rest)]
            F_bb = fim[np.ix_(rest, rest)]
            try:
                schur = F_aa - F_ab @ np.linalg.solve(F_bb, F_ab.T)
            except np.linalg.LinAlgError:
                schur = F_aa
        else:
            schur = F_aa

        if np.all(np.isfinite(schur)) and np.linalg.cond(schur) < cond_limit:
            inv = np.linalg.inv(schur)
            if inv[0, 0] > 0 and inv[1, 1] > 0:
                th[j] = np.rad2deg(np.sqrt(inv[0, 0]))
                ph[j] = np.rad2deg(np.sqrt(inv[1, 1]))
                arc[j] = np.rad2deg(np.sqrt(inv[0, 0] + sin2 * inv[1, 1]))

        if np.linalg.cond(F_aa) < cond_limit:
            inv0 = np.linalg.inv(F_aa)
            if inv0[0, 0] > 0 and inv0[1, 1] > 0:
                arc_known[j] = np.rad2deg(np.sqrt(inv0[0, 0] + sin2 * inv0[1, 1]))

    return CRLBv2Result(arc, th, ph, arc_known, cond, fim.shape[0], singular)


def crlb_v2_from_jnr(W, doas_deg, jnr_db, manifold, cfg, **kwargs) -> CRLBv2Result:
    powers = np.atleast_1d(jnr_to_power_w(np.atleast_1d(jnr_db), cfg.noise_power_w))
    return crlb_v2(W, doas_deg, powers, manifold, cfg, **kwargs)


# ---------------------------------------------------------------------------
# Monte Carlo: does the bound describe the generator's own noise?
# ---------------------------------------------------------------------------

def _draw_generator_noise(pbar, cfg, rng, n):
    """Exactly the generator's noise: multiplicative snapshot + additive detector."""
    mean_p = pbar.mean()
    zeta = rng.standard_normal((n, pbar.size))
    eps = rng.standard_normal((n, pbar.size)) * np.sqrt(cfg.alpha) * mean_p
    p = pbar * (1.0 + zeta / np.sqrt(cfg.n_snapshots)) + eps
    return np.maximum(p, 1e-6 * mean_p)


def validate_monte_carlo(W, doa_deg, jnr_db, manifold, cfg, n_trials=600, seed=0):
    """
    Local efficiency check against the generator's noise.

    A weighted nonlinear least-squares fit over [theta, phi, lnP, ln sigma_n^2]
    is started at the truth, which is what a CRLB describes. Whitening uses the
    true covariance, so a ratio near 1 means the FIM is the right one for this
    dataset -- not merely self-consistent.
    """
    from scipy.optimize import least_squares

    doa = np.atleast_2d(doa_deg)
    P = float(jnr_to_power_w(jnr_db, cfg.noise_power_w))
    bound = crlb_v2(W, doa, [P], manifold, cfg)

    pbar_true = probe_mean(W, doa, [P], manifold, cfg.noise_power_w)
    var_true, _ = noise_covariance(pbar_true, cfg)
    sd = np.sqrt(var_true)

    rng = np.random.default_rng(seed)
    obs = _draw_generator_noise(pbar_true, cfg, rng, n_trials)

    def residual(x, p_obs):
        d = np.array([[x[0], x[1]]])
        # Clip the log-parameters: the LM trust region occasionally probes far
        # enough to overflow exp(), which would poison the step.
        pb = probe_mean(W, d, [np.exp(np.clip(x[2], -400, 20))], manifold,
                        np.exp(np.clip(x[3], -400, 20)))
        return (p_obs - pb) / sd

    x0 = np.array([doa[0, 0], doa[0, 1], np.log(P), np.log(cfg.noise_power_w)])
    sin2 = np.sin(np.deg2rad(doa[0, 0])) ** 2
    errs = []
    for p_obs in obs:
        try:
            sol = least_squares(residual, x0, args=(p_obs,), method="lm",
                                xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception:
            continue
        d_t = np.deg2rad(sol.x[0] - doa[0, 0])
        d_p = np.deg2rad(sol.x[1] - doa[0, 1])
        errs.append(np.rad2deg(np.sqrt(d_t ** 2 + sin2 * d_p ** 2)))

    errs = np.array(errs)
    keep = errs < 50 * max(bound.arc_deg[0], 1e-12)
    emp = float(np.sqrt(np.mean(errs[keep] ** 2))) if keep.any() else np.nan
    # An RMSE from n samples carries a relative standard error of 1/sqrt(2n),
    # which is the scale against which any MC/CRB gap has to be judged.
    rel_se = 1.0 / np.sqrt(2 * max(int(keep.sum()), 1))
    return {
        "jnr_db": jnr_db,
        "crb": float(bound.arc_deg[0]),
        "mc": emp,
        "ratio": emp / bound.arc_deg[0] if bound.arc_deg[0] > 0 else np.nan,
        "rel_se": rel_se,
        "n": int(keep.sum()),
    }


# ---------------------------------------------------------------------------
# Proofs
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = GenConfig()
    manifold = ArrayManifold()
    geom = phys.load_sector_geometry()
    W = geom.intended_null_matrix(1)          # v2 convention: rows act as w^T u
    doa = np.array([[30.0, 20.0]])

    print("=" * 78)
    print("CRLB v2: VERIFICATION AGAINST THE GENERATOR")
    print("=" * 78)
    print(f"alpha = {cfg.alpha:.4e}   K = {cfg.n_snapshots}   "
          f"1/K = {1 / cfg.n_snapshots:.4e}")

    # -- 1. Is the document's C_n = sigma_det^2 I justified? -----------------
    print("\n[1] Is the snapshot term negligible, as Section II assumes?")
    print("  " + "-" * 74)
    pbar = probe_mean(W, doa, [jnr_to_power_w(35.0, cfg.noise_power_w)],
                      manifold, cfg.noise_power_w)
    ratio = snapshot_vs_detector_ratio(pbar, cfg)
    print(f"  probe pbar / mean      : "
          + " ".join(f"{v:6.2f}" for v in pbar / pbar.mean()))
    print(f"  snapshot/detector var  : "
          + " ".join(f"{v:6.3f}" for v in ratio))
    k_needed = (pbar.max() / pbar.mean()) ** 2 / (0.05 * cfg.alpha)
    print(f"  -> worst-case ratio {ratio.max():.3f}; K > {k_needed:.2g} is required to")
    print(f"     hold it under 0.05. The condition is NOT K >> 1/alpha: the ratio is")
    print(f"     (1/(K alpha)) (pbar_m/pbar)^2, so the probe dynamic range sets it.")
    verdict = ("negligible, so C_n = sigma_det^2 I is a valid approximation here"
               if ratio.max() < 0.05 else
               "NOT negligible, so C_n = sigma_det^2 I is the wrong covariance")
    print(f"     At K = {cfg.n_snapshots:g} the snapshot term is {verdict}.")

    # -- 2. Cost of using the document's model --------------------------------
    print("\n[2] Cost of the Section II simplification (detector term only)")
    print("  " + "-" * 74)
    print(f"  {'JNR dB':>7} {'exact CRB':>13} {'Sec II CRB':>13} {'ratio':>8}")
    for jnr in (20, 30, 40, 50):
        r_ex = crlb_v2_from_jnr(W, doa, [jnr], manifold, cfg)
        r_s2 = crlb_v2_from_jnr(W, doa, [jnr], manifold, cfg, detector_only=True)
        print(f"  {jnr:7d} {r_ex.arc_deg[0]:13.4e} {r_s2.arc_deg[0]:13.4e} "
              f"{r_s2.arc_deg[0] / r_ex.arc_deg[0]:8.3f}")

    # -- 3. Cost of dropping the covariance-sensitivity term ------------------
    print("\n[3] Cost of dropping F_cov (Section II keeps only J^T C^-1 J)")
    print("  " + "-" * 74)
    print(f"  {'JNR dB':>7} {'mean only':>13} {'+ F_cov':>13} {'ratio':>8}")
    for jnr in (20, 35, 50):
        r_m = crlb_v2_from_jnr(W, doa, [jnr], manifold, cfg)
        r_c = crlb_v2_from_jnr(W, doa, [jnr], manifold, cfg, include_cov_term=True)
        print(f"  {jnr:7d} {r_m.arc_deg[0]:13.4e} {r_c.arc_deg[0]:13.4e} "
              f"{r_c.arc_deg[0] / r_m.arc_deg[0]:8.3f}")
    print("  -> ratio <= 1 confirms dropping F_cov is conservative, as required.")

    # -- 4. Monte Carlo against the generator's own noise ---------------------
    print("\n[4] Monte Carlo: is this the bound for the generator's noise?")
    print("  " + "-" * 74)
    print(f"  {'JNR dB':>7} {'CRB arc':>13} {'MC RMSE':>13} "
          f"{'MC/CRB':>8} {'+-1sd':>7} {'n':>6}")
    for k, jnr in enumerate((20, 35, 50)):
        v = validate_monte_carlo(W, doa, jnr, manifold, cfg, n_trials=800,
                                 seed=100 + k)
        print(f"  {jnr:7d} {v['crb']:13.4e} {v['mc']:13.4e} "
              f"{v['ratio']:8.3f} {v['rel_se']:7.3f} {v['n']:6d}")
    print("  -> ratio near 1 means the FIM matches the dataset's actual noise.")
    print("     A fit started at the truth is mildly biased toward it, so a")
    print("     ratio a few percent under 1 is expected and is not a violation.")

    # -- 5. Is the 1J bound JNR-flat? ----------------------------------------
    print("\n[5] Claim: the 1-jammer bound is JNR-flat")
    print("  " + "-" * 74)
    jnrs = np.arange(20, 51, 5)
    arcs = np.array([crlb_v2_from_jnr(W, doa, [j], manifold, cfg).arc_deg[0]
                     for j in jnrs])
    print("  " + "  ".join(f"{j}dB:{a:.3e}" for j, a in zip(jnrs, arcs)))
    spread = 10 * np.log10(arcs.max() / arcs.min())
    print(f"  -> spread over 30 dB of JNR = {spread:.3f} dB "
          f"({'FLAT' if spread < 0.5 else 'NOT FLAT'})")
    print("     Both noise terms scale as pbar^2 and the Jacobian as P, so this")
    print("     holds regardless of the snapshot/detector mix.")

    # -- 6. The -1 dB/dB law, on the marginalised bound ----------------------
    print("\n[6] Claim: weak-jammer bound degrades at -1 dB/dB (Section III)")
    print("  " + "-" * 74)
    W16 = np.vstack([geom.intended_null_matrix(1), geom.intended_null_matrix(3)])
    doas2 = np.array([[30.0, 20.0], [45.0, 150.0]])
    jnr_s = 45.0
    print(f"  strong jammer fixed at {jnr_s:.0f} dB; sweeping the weak one")
    print(f"  {'delta dB':>9} {'weak CRB':>13} {'slope dB/dB':>13}")
    prev = None
    slopes = []
    for delta in range(0, 26, 5):
        jnr_w = jnr_s - delta
        r = crlb_v2_from_jnr(W16, doas2, [jnr_s, jnr_w], manifold, cfg)
        cur = r.arc_deg[1]
        slope = np.nan
        if prev is not None:
            slope = -(10 * np.log10(cur / prev)) / 5.0
            slopes.append(slope)
        print(f"  {delta:9d} {cur:13.4e} {slope:13.3f}")
        prev = cur
    print(f"  -> mean slope {np.nanmean(slopes):.3f} dB/dB "
          f"(Section III predicts -1.000)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
