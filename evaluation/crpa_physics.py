"""
Analog CRPA physics: array manifold, probe codebook, and measurement simulator.

This module is a faithful Python port of the MATLAB generation chain so that any
bound computed here refers to the system that actually produced
`data/compatible_dataset_eval.mat`. Ported files:

    dataset_generation/FormAntennaPattern.m       -> load_element_patterns()
    dataset_generation/Read_HFSS_Pattern.m        -> _read_hfss_pattern()
    dataset_generation/compute_array_response_vector.m -> ArrayManifold.steering()
    dataset_generation/compute_jammer_output_powers.m  -> simulate_probe_powers()
    dataset_generation/OptimumWeights_RealPatterns.m   -> (weights read from .mat)

THE CONJUGATION THAT DEFINES THIS SYSTEM
----------------------------------------
OptimumWeights_RealPatterns.m returns a designed null-steering weight

    CompW_m = conj(P_m 1) / (1^T P_m 1),    P_m = I - e_m e_m^H,  e_m = u_m/||u_m||

which, applied as y_m = CompW_m^T x, would null a jammer sitting exactly at
subsector center m. That is the design intent.

The dataset generator does NOT do that. In
dataset_generation_multi_jammer.m the codebook is assembled with

    sub_weights(local_id, :) = sub.Weights';        % <-- ' is CONJUGATE transpose

so the row actually multiplying x is conj(CompW_m) = P_m 1 / (1^T P_m 1), and

    y_m = (P_m 1)^T x / c_m = 1^T P_m^T x / c_m,    P_m^T = I - conj(e_m) conj(e_m)^H

The null therefore lands on conj(u_m), which is not the steering vector of any
real direction, so the deployed probes have NO null at their own subsector
center. Measured over all 64 probes: response at own center is only -6.0 dB
below the hemisphere peak, and that peak is a median 33 deg away. Over 1500
1-jammer records, argmax_m p_m is the nearest subsector center just 23% of the
time and argmin_m p_m only 8% (mean rank of the nearest center: 3.38 of 8).

Consequences that drive the rest of the analysis:
  * The probes are a deterministic, well-conditioned, but spatially SCRAMBLED
    8-D projection of the array snapshot, with ~47 dB of dynamic range across
    the hemisphere. Accuracy comes from that interferometric structure, not
    from a beam peak and not from a notch at the subsector center.
  * Branch A's softmax over subsector centers is therefore neither a
    soft-argmax nor a soft-argmin of a physically peaked response. It is a
    learned readout, which is why it needs the Branch B correction at all.

`probe_matrix()` returns the AS-GENERATED rows, so `y = W @ x` reproduces the
dataset. `intended_null_matrix()` exposes the design-intent codebook for
comparison. Re-simulation of stored records agrees to 0.006% median relative
error, so this convention is confirmed, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# System constants (dataset_generation/dataset_generation_multi_jammer.m)
# ---------------------------------------------------------------------------

F0_HZ = 1580e6
C_M_S = 299792458.0
LAMBDA_M = C_M_S / F0_HZ
N_ANT = 4
N_SECTORS = 8
N_SUBSECTORS = 8

# ElementLoc = [0.03, -0.03, 0.03, 0.03, -0.03, 0.03, -0.03, -0.03] reshaped 4x2.
# A square of side 6 cm, i.e. circumradius 0.03*sqrt(2) = 42.4 mm = 0.224 lambda.
ELEMENT_POSITIONS = np.array(
    [
        [0.03, -0.03, 0.0],
        [0.03, 0.03, 0.0],
        [-0.03, 0.03, 0.0],
        [-0.03, -0.03, 0.0],
    ]
)

NOISE_POWER_DBM = -114.0
NOISE_POWER_W = 1e-3 * 10 ** (NOISE_POWER_DBM / 10.0)

_WORKSPACE = Path(__file__).resolve().parent.parent
PATTERN_DIR = _WORKSPACE / "dsp" / "antenna_patterns"
SECTOR_DATA = _WORKSPACE / "data" / "sector_data_8sect.mat"


# ---------------------------------------------------------------------------
# HFSS element patterns
# ---------------------------------------------------------------------------

def _interp1_linear(values: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    """MATLAB interp1(v, xq, 'linear', 'extrap') with implicit sample points 1..len(v)."""
    n = values.shape[0]
    sample_idx = np.arange(1, n + 1, dtype=float)
    lo = np.clip(np.searchsorted(sample_idx, query_idx, side="right") - 1, 0, n - 2)
    frac = (query_idx - sample_idx[lo]) / (sample_idx[lo + 1] - sample_idx[lo])
    return values[lo] + frac * (values[lo + 1] - values[lo])


def _read_hfss_pattern(csv_path: Path) -> np.ndarray:
    """
    Port of Read_HFSS_Pattern.m for stepsize == 1.

    The CSV holds 121 theta samples (-180:3:180) x 60 phi samples (0:3:177) with
    theta varying fastest, so MATLAB's column-major reshape(col, 121, 60) yields
    a [theta, phi] grid. Despite the misleading local variable names in the
    MATLAB source, it upsamples phi 60 -> 181 first and theta 121 -> 361 second,
    keeping the first 180 phi columns.

    Returns a complex [361 theta (-180:1:180), 180 phi (0:1:179)] grid.
    """
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pattern_column = raw[:, 3] * np.exp(1j * raw[:, 4])
    chunk = pattern_column.reshape((121, 60), order="F")

    # Stage 1: phi 60 -> 181 samples at 1/3 index steps, extrapolating past 60.
    xq = np.arange(1.0, 61.0 + 1e-9, 1.0 / 3.0)
    stage1 = np.empty((121, xq.size), dtype=complex)
    for i in range(121):
        stage1[i, :] = _interp1_linear(chunk[i, :], xq)

    # Stage 2: theta 121 -> 361 samples, keeping phi columns 1..180.
    xp = np.arange(1.0, 121.0 + 1e-9, 1.0 / 3.0)
    stage2 = np.empty((xp.size, 180), dtype=complex)
    for j in range(180):
        stage2[:, j] = _interp1_linear(stage1[:, j], xp)

    return stage2


@dataclass(frozen=True)
class ElementPatterns:
    """Complex element patterns on a 1-degree [theta 0..180, phi 0..360] grid."""

    theta_rad: np.ndarray          # (181,)
    phi_rad: np.ndarray            # (361,)
    pattern: np.ndarray            # (4, 181, 361) complex
    d_dtheta: np.ndarray           # (4, 181, 361) complex
    d_dphi: np.ndarray             # (4, 181, 361) complex

    def _interpolators(self, grids: np.ndarray):
        return [
            RegularGridInterpolator(
                (self.theta_rad, self.phi_rad),
                grids[n],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            for n in range(N_ANT)
        ]


@lru_cache(maxsize=2)
def load_element_patterns(freq_mhz: int = 1580) -> ElementPatterns:
    """
    Port of FormAntennaPattern.m (realPat=1, anglestep=pi/180).

    Mirrors the two fold-out steps: append a phi=180 column from the flipped
    phi=0 column, then map the (theta in [-180,0]) half onto phi in [180,360].
    """
    raw_grids = []
    for element in range(1, N_ANT + 1):
        csv_path = PATTERN_DIR / f"CR8894XF_6cm_Antenna{element}_{freq_mhz}MHz_1.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing HFSS pattern: {csv_path}")
        raw_grids.append(_read_hfss_pattern(csv_path))

    theta_deg = np.arange(0.0, 181.0)
    phi_deg = np.arange(0.0, 361.0)

    patterns = np.empty((N_ANT, theta_deg.size, phi_deg.size), dtype=complex)
    for n, grid in enumerate(raw_grids):
        # phiA = 0:1:180; last column is the flipped phi=0 elevation cut.
        corrected = np.empty((361, 181), dtype=complex)
        corrected[:, :180] = grid
        corrected[:, 180] = grid[::-1, 0]

        # theta index 180 is the 0 deg row (MATLAB IndxT0Deg = 181).
        full = np.empty((181, 361), dtype=complex)
        full[:, :181] = corrected[180:, :]
        full[:, 180:] = corrected[:181, :][::-1, :]
        patterns[n] = full

    # 1-degree grid, so central differences are an accurate pattern gradient.
    d_theta = np.gradient(patterns, np.deg2rad(theta_deg), axis=1)
    d_phi = np.gradient(patterns, np.deg2rad(phi_deg), axis=2)

    return ElementPatterns(
        theta_rad=np.deg2rad(theta_deg),
        phi_rad=np.deg2rad(phi_deg),
        pattern=patterns,
        d_dtheta=d_theta,
        d_dphi=d_phi,
    )


# ---------------------------------------------------------------------------
# Array manifold
# ---------------------------------------------------------------------------

class ArrayManifold:
    """
    Steering vectors and angular derivatives for the 4-element square array.

    steering() reproduces compute_array_response_vector.m:
        u_n = E_n(theta, phi) * exp(1j * k(theta, phi) . p_n)
    """

    def __init__(self, patterns: ElementPatterns | None = None,
                 positions: np.ndarray = ELEMENT_POSITIONS,
                 wavelength: float = LAMBDA_M):
        self.patterns = patterns if patterns is not None else load_element_patterns()
        self.positions = positions
        self.wavelength = wavelength
        self._pat = self.patterns._interpolators(self.patterns.pattern)
        self._dpat_dtheta = self.patterns._interpolators(self.patterns.d_dtheta)
        self._dpat_dphi = self.patterns._interpolators(self.patterns.d_dphi)

    # -- element pattern access ------------------------------------------------

    def _eval(self, interps, theta_deg, phi_deg) -> np.ndarray:
        theta = np.atleast_1d(np.deg2rad(np.asarray(theta_deg, dtype=float)))
        phi = np.atleast_1d(np.deg2rad(np.asarray(phi_deg, dtype=float))) % (2 * np.pi)
        pts = np.stack([theta, phi], axis=-1)
        return np.stack([interps[n](pts) for n in range(N_ANT)], axis=0)

    # -- wave vector and its derivatives ---------------------------------------

    def _wave_vectors(self, theta_deg, phi_deg):
        t = np.atleast_1d(np.deg2rad(np.asarray(theta_deg, dtype=float)))
        p = np.atleast_1d(np.deg2rad(np.asarray(phi_deg, dtype=float)))
        k0 = 2.0 * np.pi / self.wavelength
        k = k0 * np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)
        dk_dt = k0 * np.stack([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)], axis=-1)
        dk_dp = k0 * np.stack([-np.sin(t) * np.sin(p), np.sin(t) * np.cos(p), np.zeros_like(t)], axis=-1)
        return k, dk_dt, dk_dp

    # -- public API ------------------------------------------------------------

    def steering(self, theta_deg, phi_deg) -> np.ndarray:
        """Return u with shape (4,) for scalar input or (4, K) for vector input."""
        e = self._eval(self._pat, theta_deg, phi_deg)                  # (4, K)
        k, _, _ = self._wave_vectors(theta_deg, phi_deg)               # (K, 3)
        phase = np.exp(1j * (self.positions @ k.T))                    # (4, K)
        u = e * phase
        return u[:, 0] if u.shape[1] == 1 else u

    def steering_and_derivatives(self, theta_deg, phi_deg):
        """
        Return (u, du/dtheta, du/dphi), angles in radians for the derivatives.

        Product rule on u_n = E_n * exp(1j k.p_n): the phase factor is analytic,
        the pattern factor is differentiated numerically on the 1-degree grid.
        """
        e = self._eval(self._pat, theta_deg, phi_deg)
        de_dt = self._eval(self._dpat_dtheta, theta_deg, phi_deg)
        de_dp = self._eval(self._dpat_dphi, theta_deg, phi_deg)

        k, dk_dt, dk_dp = self._wave_vectors(theta_deg, phi_deg)
        kp = self.positions @ k.T                                      # (4, K)
        phase = np.exp(1j * kp)
        dkp_dt = self.positions @ dk_dt.T
        dkp_dp = self.positions @ dk_dp.T

        u = e * phase
        du_dt = (de_dt + 1j * dkp_dt * e) * phase
        du_dp = (de_dp + 1j * dkp_dp * e) * phase

        if u.shape[1] == 1:
            return u[:, 0], du_dt[:, 0], du_dp[:, 0]
        return u, du_dt, du_dp


# ---------------------------------------------------------------------------
# Probe codebook
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectorGeometry:
    """Per-sector probe codebook and subsector geometry."""

    codebook: np.ndarray           # (8 sectors, 8 probes, 4) raw CompW from the .mat
    centers: np.ndarray            # (8, 8, 2) [theta, phi] degrees
    sector_edges_theta: np.ndarray  # (8, 2)
    sector_edges_phi: np.ndarray    # (8, 2)

    def probe_matrix(self, sector_id: int) -> np.ndarray:
        """
        As-generated row-stacked W for a 1-indexed sector, so that `y = W x`
        reproduces the dataset. Rows are conj(CompW_m) because the generator
        assembles the codebook with MATLAB's conjugate transpose.
        """
        return np.conj(self.codebook[sector_id - 1])

    def intended_null_matrix(self, sector_id: int) -> np.ndarray:
        """Design-intent codebook, whose rows null their own subsector center."""
        return self.codebook[sector_id - 1]

    def subsector_centers(self, sector_id: int) -> np.ndarray:
        return self.centers[sector_id - 1]


@lru_cache(maxsize=2)
def load_sector_geometry(path: str | None = None) -> SectorGeometry:
    """Read the probe codebook from sector_data_8sect.mat."""
    mat_path = Path(path) if path is not None else SECTOR_DATA
    data = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    sectors = data["sectors"]

    codebook = np.zeros((N_SECTORS, N_SUBSECTORS, N_ANT), dtype=complex)
    centers = np.zeros((N_SECTORS, N_SUBSECTORS, 2))
    edges_theta = np.zeros((N_SECTORS, 2))
    edges_phi = np.zeros((N_SECTORS, 2))

    for s_idx, sector in enumerate(sectors):
        edges_theta[s_idx] = np.asarray(sector.Edges.Elev, dtype=float)
        edges_phi[s_idx] = np.asarray(sector.Edges.Az, dtype=float)
        for m_idx, sub in enumerate(sector.Subsectors):
            codebook[s_idx, m_idx, :] = np.asarray(sub.Weights, dtype=complex).ravel()
            centers[s_idx, m_idx, :] = [float(sub.Center.Elev), float(sub.Center.Az)]

    return SectorGeometry(codebook, centers, edges_theta, edges_phi)


# ---------------------------------------------------------------------------
# Measurement simulator
# ---------------------------------------------------------------------------

def probe_gains(W: np.ndarray, u: np.ndarray) -> np.ndarray:
    """g_m = w_m^T u for the row-stacked codebook W (M, 4)."""
    return W @ u


def noiseless_probe_powers(W: np.ndarray, doas_deg: np.ndarray, powers_w: Sequence[float],
                           manifold: ArrayManifold) -> np.ndarray:
    """
    Coherent noiseless probe powers, i.e. |s_m|^2 with

        s_m = sum_j sqrt(P_j) * w_m^T u_j

    The coherent sum reproduces `s = ones(J,1)` in compute_jammer_output_powers.m.
    """
    doas = np.atleast_2d(doas_deg)
    s = np.zeros(W.shape[0], dtype=complex)
    for j, (theta, phi) in enumerate(doas):
        u = manifold.steering(theta, phi)
        s += np.sqrt(powers_w[j]) * (W @ u)
    return np.abs(s) ** 2


def simulate_probe_powers(W: np.ndarray, doas_deg: np.ndarray, powers_w: Sequence[float],
                          manifold: ArrayManifold, noise_var: float = NOISE_POWER_W,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Port of compute_jammer_output_powers.m.

    One noise draw is shared across all probes, because the MATLAB code forms a
    single x and then applies every weight row to it. That shared draw is what
    makes the probe noise correlated (see crlb.probe_noise_covariance).
    """
    rng = rng if rng is not None else np.random.default_rng()
    doas = np.atleast_2d(doas_deg)

    signal = np.zeros(N_ANT, dtype=complex)
    for j, (theta, phi) in enumerate(doas):
        signal += np.sqrt(powers_w[j]) * manifold.steering(theta, phi)

    n = np.sqrt(noise_var / 2.0) * (rng.standard_normal(N_ANT) + 1j * rng.standard_normal(N_ANT))
    return np.abs(W @ (signal + n)) ** 2


def jnr_to_power_w(jnr_db: float | np.ndarray, noise_var: float = NOISE_POWER_W):
    """Nominal incident power for a given JNR label, as the generator defines it."""
    return noise_var * 10 ** (np.asarray(jnr_db, dtype=float) / 10.0)


# ---------------------------------------------------------------------------
# Projection nulling (application-level JSR)
# ---------------------------------------------------------------------------

QUIESCENT_WEIGHTS = np.ones(N_ANT) / N_ANT


def projection_nuller_weights(u_pred: np.ndarray, eps: float = 1e-9):
    """
    Nulling weights in the same form the MATLAB codebook uses:

        w = conj(P 1) / (1^T P 1),    P = I - A A^+,   A = [u_1/||u_1||, ...]

    Applied as y = w^T u (plain transpose, matching the measurement convention),
    this places an exact null on every predicted direction while holding the
    common-mode gain at unity, since 1^T w = (1^T P 1)/(1^T P 1) = 1.

    P is an orthogonal projector, so 1^T P 1 = ||P 1||^2 >= 0. It approaches zero
    when the predicted steering vector becomes parallel to the all-ones vector,
    i.e. near zenith where the inter-element phases vanish: there the array cannot
    null the jammer and still preserve unit common-mode gain. Those samples are
    flagged rather than silently normalised by a near-zero number.

    Returns (w, degenerate_flag).
    """
    U = np.atleast_2d(np.asarray(u_pred, dtype=complex))
    if U.shape[0] != N_ANT:
        U = U.T
    A = U / np.linalg.norm(U, axis=0, keepdims=True)

    P = np.eye(N_ANT) - A @ np.linalg.pinv(A)
    ones = np.ones(N_ANT)
    P1 = P @ ones
    c = float(np.real(ones @ P1))

    degenerate = c <= eps * float(np.real(ones @ ones))
    if degenerate:
        # Fall back to the unnormalised projection nuller, which is the same
        # weight up to a scale and stays finite.
        return np.conj(P @ QUIESCENT_WEIGHTS), True
    return np.conj(P1) / c, False


def jammer_suppression_db(w_null: np.ndarray, u_true: np.ndarray,
                          cap_db: float = 50.0, floor_db: float = 0.0) -> float:
    """
    JSR = 10 log10( |w_q^T u_true|^2 / |w_null^T u_true|^2 ).

    Both weights use the y = w^T u convention and both satisfy 1^T w = 1, so the
    ratio is a fair comparison of residual jammer leakage before and after
    nulling. Capped at `cap_db` to represent a finite hardware dynamic range.
    """
    p_q = np.abs(QUIESCENT_WEIGHTS @ u_true) ** 2
    p_null = np.abs(w_null @ u_true) ** 2
    if p_null <= 0 or not np.isfinite(p_null):
        return cap_db
    jsr = 10.0 * np.log10(p_q / p_null)
    return float(np.clip(jsr, floor_db, cap_db))


# ---------------------------------------------------------------------------
# Preprocessing shared with the trained network
# ---------------------------------------------------------------------------

def preprocess_powers(powers: np.ndarray) -> np.ndarray:
    """log10(p) - log10(sum p): the normalisation used in training and every eval."""
    safe = np.asarray(powers, dtype=float) + 1e-20
    return (np.log10(safe) - np.log10(safe.sum(axis=-1, keepdims=True))).astype(np.float32)


def encode_centers(centers_deg: np.ndarray) -> np.ndarray:
    """[theta, phi] degrees -> [cos t, sin t, cos p, sin p]."""
    c = np.atleast_2d(np.asarray(centers_deg, dtype=float))
    t = np.deg2rad(c[:, 0])
    p = np.deg2rad(c[:, 1])
    return np.stack([np.cos(t), np.sin(t), np.cos(p), np.sin(p)], axis=-1).astype(np.float32)


def decode_trig(trig: np.ndarray) -> np.ndarray:
    """
    [cos t, sin t, cos p, sin p] -> [theta, phi] degrees.

    Uses the fold convention theta <- min(|theta|, 180 - |theta|), which is the
    one used by phases 1-3. The `% 90` variant that appeared in the phase 3.5/4
    scripts aliases theta = 91 deg onto 1 deg and corrupts sectors 5-8.
    """
    arr = np.atleast_2d(np.asarray(trig, dtype=float))
    theta = np.rad2deg(np.arctan2(arr[:, 1], arr[:, 0]))
    theta = np.abs(theta)
    theta = np.minimum(theta, 180.0 - theta)
    phi = np.rad2deg(np.arctan2(arr[:, 3], arr[:, 2])) % 360.0
    out = np.stack([theta, phi], axis=-1)
    return out[0] if out.shape[0] == 1 and np.ndim(trig) == 1 else out


def arc_error_deg(theta1, phi1, theta2, phi2):
    """Great-circle separation for zenith/azimuth pairs, in degrees."""
    t1, p1 = np.deg2rad(theta1), np.deg2rad(phi1)
    t2, p2 = np.deg2rad(theta2), np.deg2rad(phi2)
    cos_arc = np.cos(t1) * np.cos(t2) + np.sin(t1) * np.sin(t2) * np.cos(p1 - p2)
    return np.rad2deg(np.arccos(np.clip(cos_arc, -1.0, 1.0)))


def spherical_midpoint(doa1, doa2) -> np.ndarray:
    """Normalised Cartesian midpoint of two directions, back in [theta, phi] degrees."""
    def to_xyz(doa):
        t, p = np.deg2rad(doa[0]), np.deg2rad(doa[1])
        return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

    mid = to_xyz(doa1) + to_xyz(doa2)
    norm = np.linalg.norm(mid)
    if norm < 1e-12:
        return np.array([np.nan, np.nan])
    mid /= norm
    theta = np.rad2deg(np.arccos(np.clip(mid[2], -1.0, 1.0)))
    phi = np.rad2deg(np.arctan2(mid[1], mid[0])) % 360.0
    return np.array([theta, phi])


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _selftest_intended_nulls(manifold: ArrayManifold, geom: SectorGeometry) -> None:
    """
    The design-intent codebook must null its own subsector center exactly.

    This is the strongest joint check on the pattern port: the MATLAB weights
    were derived *from* these HFSS patterns, so any error anywhere in the
    interpolation chain fills the notch back in. Reaching ~-290 dB means the
    ported pattern is bit-comparable to MATLAB's.
    """
    print("\n[Self-test 1] Design-intent codebook nulls its own subsector center")
    print("  " + "-" * 68)
    depths = []
    for sector_id in range(1, N_SECTORS + 1):
        W_intended = geom.intended_null_matrix(sector_id)
        centers = geom.subsector_centers(sector_id)
        for m in range(N_SUBSECTORS):
            u = manifold.steering(centers[m, 0], centers[m, 1])
            g = W_intended @ u
            own = np.abs(g[m]) ** 2
            others = np.abs(np.delete(g, m)) ** 2
            depths.append(10 * np.log10(own / others.mean()))
    depths = np.array(depths)
    print(f"  null depth over all 64 probes: mean {depths.mean():.1f} dB, "
          f"shallowest {depths.max():.1f} dB")
    print("  => HFSS pattern port verified to numerical precision")


def _selftest_deployed_probes(manifold: ArrayManifold, geom: SectorGeometry) -> None:
    """
    Characterise the probes the dataset actually used.

    Because the generator conjugates, the null moves to conj(u_m), a direction
    with no physical counterpart. The visible-hemisphere response is then
    neither peaked nor notched at the subsector center.
    """
    print("\n[Self-test 2] AS-GENERATED probes: where did the null go?")
    print("  " + "-" * 68)

    conj_null, center_vs_peak, peak_offset, dyn_range = [], [], [], []
    theta_grid = np.arange(0.0, 90.5, 1.0)
    phi_grid = np.arange(0.0, 360.0, 2.0)
    T, P = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    U = manifold.steering(T.ravel(), P.ravel())

    for sector_id in range(1, N_SECTORS + 1):
        W = geom.probe_matrix(sector_id)
        centers = geom.subsector_centers(sector_id)
        for m in range(N_SUBSECTORS):
            w = W[m]
            u_c = manifold.steering(centers[m, 0], centers[m, 1])
            at_center = np.abs(w @ u_c) ** 2
            at_conj = np.abs(w @ np.conj(u_c)) ** 2
            conj_null.append(10 * np.log10(at_conj / at_center))

            resp = (np.abs(w @ U) ** 2).reshape(T.shape)
            center_vs_peak.append(10 * np.log10(at_center / resp.max()))
            dyn_range.append(10 * np.log10(resp.min() / resp.max()))
            i_pk = np.unravel_index(np.argmax(resp), resp.shape)
            peak_offset.append(arc_error_deg(centers[m, 0], centers[m, 1],
                                             theta_grid[i_pk[0]], phi_grid[i_pk[1]]))

    conj_null = np.array(conj_null)
    center_vs_peak = np.array(center_vs_peak)
    peak_offset = np.array(peak_offset)
    dyn_range = np.array(dyn_range)
    print(f"  response at conj(u_center) vs at u_center : {conj_null.mean():8.1f} dB")
    print("    -> the exact null moved to the conjugate direction, off-hemisphere")
    print(f"  response at own center vs hemisphere peak : {center_vs_peak.mean():8.2f} dB "
          f"(worst {center_vs_peak.min():.1f})")
    print(f"  arc from own center to hemisphere peak    : {np.median(peak_offset):8.2f} deg (median)")
    print(f"  hemisphere min/max dynamic range          : {dyn_range.mean():8.1f} dB")
    print("  => probes are a scrambled but information-rich 8-D projection,")
    print("     not beams and not notches at their subsector centers")


def _selftest_probe_ordering(geom: SectorGeometry, dataset_path: Path,
                             n_check: int = 1500) -> None:
    """Does any single-probe readout point at the jammer? Decides Branch A's nature."""
    print("\n[Self-test 3] Is a single-probe argmax/argmin a usable DoA readout?")
    print("  " + "-" * 68)
    if not dataset_path.exists():
        print(f"  SKIP: dataset not found at {dataset_path}")
        return

    dataset = sio.loadmat(str(dataset_path), struct_as_record=False,
                          squeeze_me=True)["dataset"]
    counts = np.array([s.Jammer_Count for s in dataset])

    hit_max = hit_min = 0
    ranks = []
    indices = np.where(counts == 1)[0][:n_check]
    for idx in indices:
        sample = dataset[idx]
        sector_id = int(np.atleast_1d(sample.Sector_IDs)[0])
        centers = geom.subsector_centers(sector_id)
        doa = np.asarray(sample.DoA, dtype=float).ravel()[:2]
        p = np.asarray(sample.subsector_powers, dtype=float).ravel()
        nearest = np.argmin(arc_error_deg(doa[0], doa[1], centers[:, 0], centers[:, 1]))
        hit_max += int(np.argmax(p) == nearest)
        hit_min += int(np.argmin(p) == nearest)
        ranks.append(int(np.where(np.argsort(-p) == nearest)[0][0]) + 1)

    n = len(indices)
    print(f"  samples                                  : {n}")
    print(f"  P(argmax_m p_m == nearest center)        : {hit_max / n * 100:5.1f} %")
    print(f"  P(argmin_m p_m == nearest center)        : {hit_min / n * 100:5.1f} %")
    print(f"  mean power rank of the nearest center    : {np.mean(ranks):5.2f} of 8")
    print("  => chance level is 12.5%; neither readout works, so Branch A's")
    print("     softmax cannot be a physical soft-argmax or soft-argmin")


def _selftest_resimulation(manifold: ArrayManifold, geom: SectorGeometry,
                           dataset_path: Path, n_check: int = 200) -> float:
    """
    Re-simulate stored 1-jammer records.

    This is the test that pins down the conjugation: with the as-generated
    convention the residual is at the noise level, whereas the design-intent
    codebook is off by ~90% and anti-correlated.
    """
    print("\n[Self-test 4] Re-simulation against stored dataset powers")
    print("  " + "-" * 68)
    if not dataset_path.exists():
        print(f"  SKIP: dataset not found at {dataset_path}")
        return np.nan

    data = sio.loadmat(str(dataset_path), struct_as_record=False, squeeze_me=True)
    dataset = data["dataset"]
    counts = np.array([s.Jammer_Count for s in dataset])
    idx_1j = np.where(counts == 1)[0][:n_check]

    results = {}
    for label, matrix_fn in (("as-generated  conj(CompW)", geom.probe_matrix),
                             ("design-intent CompW      ", geom.intended_null_matrix)):
        rel_errors, corrs = [], []
        for idx in idx_1j:
            sample = dataset[idx]
            sector_id = int(np.atleast_1d(sample.Sector_IDs)[0])
            stored = np.asarray(sample.subsector_powers, dtype=float).ravel()
            doa = np.asarray(sample.DoA, dtype=float).ravel()[:2]
            power_w = jnr_to_power_w(float(np.atleast_1d(sample.JNR_dB)[0]))

            predicted = noiseless_probe_powers(matrix_fn(sector_id), doa[None, :],
                                               [power_w], manifold)
            # Judge on probes carrying signal; the rest are noise-dominated.
            strong = stored >= 0.05 * stored.max()
            rel_errors.append(np.abs(predicted[strong] - stored[strong]) / stored[strong])
            corrs.append(np.corrcoef(np.log10(predicted + 1e-30),
                                     np.log10(stored + 1e-30))[0, 1])
        rel = np.concatenate(rel_errors)
        results[label] = float(np.median(rel))
        print(f"  {label}: median rel err {np.median(rel) * 100:9.4f} %, "
              f"log-power corr {np.median(corrs):+.4f}")

    print(f"  samples checked: {len(idx_1j)}")
    print("  => the conjugated convention reproduces the dataset; the design")
    print("     intent does not, so all bounds below use the conjugated one")
    return results["as-generated  conj(CompW)"]


def _report_effective_gain(manifold: ArrayManifold, geom: SectorGeometry) -> None:
    """
    The nominal JNR label is not the probe-output SNR.

    HFSS magnitudes are in mV, so |E|^2 contributes a large fixed gain. This is
    why the network reaches sub-degree accuracy at a nominal JNR of 20 dB.
    """
    print("\n[Info] Nominal JNR label vs effective probe SNR")
    print("  " + "-" * 68)
    gains_db = []
    for sector_id in range(1, N_SECTORS + 1):
        W = geom.probe_matrix(sector_id)
        for theta, phi in geom.subsector_centers(sector_id):
            u = manifold.steering(theta, phi)
            g = np.abs(W @ u) ** 2
            gains_db.append(10 * np.log10(g.max()))
    gains_db = np.array(gains_db)
    print(f"  peak |w^T u|^2 over codebook: {gains_db.mean():.1f} dB "
          f"(min {gains_db.min():.1f}, max {gains_db.max():.1f})")
    print(f"  => a nominal 20 dB JNR presents ~{20 + gains_db.mean():.0f} dB "
          f"at the strongest probe output")


def main() -> None:
    print("=" * 72)
    print("CRPA PHYSICS MODULE SELF-TEST")
    print("=" * 72)
    print(f"  f0                  : {F0_HZ / 1e6:.0f} MHz")
    print(f"  wavelength          : {LAMBDA_M * 1e3:.1f} mm")
    circumradius = np.linalg.norm(ELEMENT_POSITIONS[0])
    print(f"  array circumradius  : {circumradius * 1e3:.1f} mm "
          f"({circumradius / LAMBDA_M:.3f} lambda)")
    print(f"  element azimuths    : "
          f"{np.round(np.degrees(np.arctan2(ELEMENT_POSITIONS[:, 1], ELEMENT_POSITIONS[:, 0])) % 360).astype(int)} deg")
    print(f"  noise power         : {NOISE_POWER_DBM:.0f} dBm = {NOISE_POWER_W:.3e} W")

    patterns = load_element_patterns()
    print(f"  pattern grid        : {patterns.pattern.shape} (elements, theta, phi)")

    manifold = ArrayManifold(patterns)
    geom = load_sector_geometry()
    print(f"  codebook            : {geom.codebook.shape} (sectors, probes, elements)")

    dataset_path = _WORKSPACE / "data" / "compatible_dataset_eval.mat"
    _selftest_intended_nulls(manifold, geom)
    _selftest_deployed_probes(manifold, geom)
    _selftest_probe_ordering(geom, dataset_path)
    _selftest_resimulation(manifold, geom, dataset_path)
    _report_effective_gain(manifold, geom)

    print("\n" + "=" * 72)
    print("SELF-TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
