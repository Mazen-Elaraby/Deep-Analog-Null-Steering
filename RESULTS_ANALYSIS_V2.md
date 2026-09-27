# RESULTS ANALYSIS — v2

Consolidated, artifact-backed reading of Phases 1 through 5 for the v2 pipeline.

**Status:** authoritative results record for this architecture.

**Scope rule.** This document describes **one** architecture — the subject of the
paper — and compares it only against physical bounds (CRLB, identifiability) and
against chance. No earlier iteration of the design is presented, tabulated or
used as a baseline anywhere below.

---

## 0. Scope and provenance

Every v2 number in this document is transcribed from a file in
`evaluation_results/` or from another explicitly named local artifact. Each subsection names its source
artifact. Nothing here is quoted from memory or from the planning document.

Two provenance rules are applied throughout:

1. **All reported numbers** come from on-disk artifacts, named per section.
   Nothing is quoted from memory.
2. **Retracted numbers** are collected in §10 and must never be quoted. Two
   efficiency figures (1.75x and 1.41x) fall in this category.

The planning document's task statuses are stale by the executing agent's own
admission (session log L15329) and are not used as evidence anywhere below.

---

## 1. System under test

**Array.** Four elements at (+/-0.03, +/-0.03) m, i.e. a square of circumradius
0.0424 m = 0.224 lambda at f0 = 1580 MHz. Measured HFSS element patterns (not
isotropic). Noise floor -114 dBm.

**Sectorisation.** The upper hemisphere is partitioned into 8 sectors (4 azimuth
quadrants x 2 equal-solid-angle elevation bands, boundary theta = 60 deg). Each
sector carries 8 subsectors, giving 64 global subsector IDs (0-indexed).

**Measurement.** Per query the receiver reports **8 scalar probe powers** — one
per subsector of the active sector. This is the entire input. There is no phase,
no I/Q, and no per-element data: the recovery problem is DoA estimation from
eight non-negative scalars.

**Probe codebook.** The codebook is built from `geom.intended_null_matrix(s)`,
and the rows are applied in the generator's own convention, `w_m^T u`, without
conjugation. This matters more than it appears: `w_m^T u` and `w_m^H u` place the
null in different directions, so the convention has to be fixed once and carried
identically through the generator, the bound and the model. `data/VALIDATION_v2.txt`
check 3 confirms the deployed codebook nulls its own subsector centre to a median
depth of **90.49 dB**, which is the evidence that the convention in force is the
intended one.

**Dataset generator** (`dataset_generation/generate_dataset_v2.py`). Incoherent
closed-form SCM with

```
p_obs = pbar * (1 + zeta / sqrt(K)) + eps,    zeta ~ N(0,1),
eps ~ N(0, alpha * mean(pbar)^2)
```

with `alpha = 10**-3.5` (L79) and `K = n_snapshots = 1_000_000` (L80). The
shipped dataset metadata in `data/VALIDATION_v2.txt` confirms
`alpha 3.162e-04, K 1000000`, i.e. generator, bound and verification all run at
the same K. K was raised from the originally planned 8192 because, as
`evaluation/crlb_v2.py` records in its docstring (L15-19), *"the claim that
K >> 1/alpha makes the detector term dominant is false at K = 8192."*

**Preprocessing** (`train_v2.expand_queries`):

```
norm_p    = log10(p + 1e-20) - log10(sum(p + 1e-20))
log_scale = log10(total) - log10(noise_var)
sub_ids   = sectors[:, None] * 8 + arange(8)
```

The split into a **shape** vector (`norm_p`) and an **absolute-scale** scalar
(`log_scale`) is what makes recommendation #2 expressible at all.

**Model** (`models/model_builder_v2.py`, `HybridSARRefinerV2`).

- **Branch A, the geometric anchor.** `attention_mlp` =
  LayerNorm(8) -> Linear(8,32) -> ReLU -> Linear(32,8) -> Softmax, whose weights
  form a convex combination of the 8 subsector-centre trig encodings, **plus
  `hull_offset(powers)`** (recommendation #6, zero-initialised).
- **Branch B, the refiner.** `nn.MultiheadAttention(64, num_heads=4,
  batch_first=True)` over a **10-token sequence**: `CLS_INDEX = 0`,
  `SCALE_INDEX = 1`, `PROBE_SLICE = slice(2, 10)`.
- **Output.** `correction_head(cat([cls_out, fg_emb, log_scale]))` emits **3**
  values. `_assemble` applies the theta correction **in the logit of cos theta**
  (recommendation #5) and keeps phi as a unit-circle residual.

**Training.** 50 epochs, approximately 23 min on an RTX 3060 (GPU retrain was
mandated by the user at session log L14583). Final train loss 0.022478, val loss
0.022601 — a 0.5% gap, so no meaningful overfit. Held-out test arc error
p50 8.798 deg / p90 25.654 deg / mean 11.462 deg. Note that this held-out split
mixes 1J/2J/3J; the 1J-only figure in §2 is far better.

---

## 2. Phase 1 — single-jammer accuracy vs the CRLB

Source: `evaluation_results/PHASE1_V2.txt`,
`evaluation/CRLB_V2_VERIFICATION.txt`.

**Accuracy.** Arc error p50 **3.717 deg**, p90 **12.843 deg**, RMSE
**7.072 deg**. Decomposed: zenith p50 3.140 / p90 11.478; azimuth p50 2.163 /
p90 6.462. Zenith is the harder axis, as expected for a 0.224-lambda array whose
elevation sensitivity comes mostly from the element patterns rather than from
inter-element delay.

**Recommendation #5 works structurally.** Predicted theta spans
**[6.62 deg, 85.83 deg]** — strictly inside (0, 90). The logit-of-cos-theta
parameterisation means the hemisphere constraint is satisfied by construction:
**no folding, clipping or wrapping is needed anywhere in the v2 pipeline.**

**Efficiency vs the bound.** 1,500 per-geometry CRLB evaluations give bound arc
median 5.020 deg, p10 0.920 deg, p90 15.130 deg, RMS **9.464 deg**. The correct
aggregate comparison is RMSE against RMS of the bound:

```
7.072 / 9.464 = 0.75x
```

The network is at **0.75x the RMS bound** — i.e. below it in aggregate, which is
only possible because the bound is a *per-geometry* local quantity and the RMS
mixes wildly different geometries. Splitting by conditioning makes the picture
honest:

| Half (by FIM conditioning) | Network RMSE | RMS CRLB | Ratio |
|---|---|---|---|
| Well-conditioned | 3.932 deg | 2.040 deg | **1.93x** (above bound) |
| Ill-conditioned  | 9.234 deg | 13.228 deg | **0.70x** (below bound) |

**This split is the result, not the 0.75x.** Where the geometry is informative
the network is a factor of ~2 off optimal and there is real headroom. Where the
geometry is uninformative the *bound itself* blows up (p90 15.1 deg, and see the
weak-jammer divergence in §4), while the network's error is capped by its
learned prior — a regression-to-the-mean effect that looks like beating the
bound but is not. Any single-number efficiency claim for this system is an
artifact of which half dominates the sample.

### 2.1 The sub-CRLB behaviour is bias, not variance — and it is spatial

Source: `evaluation_results/PHASE1_BIAS_VARIANCE_V2.txt`, figure
`evaluation_results/figures/phase1/phase1_bias_variance.{pdf,png}`.

The claim above — that the below-bound ratio is a prior falling back on the
architecture rather than a genuinely efficient estimator — is an assertion about
**bias**, so it has to be measured as one. The eval split cannot do this: it
visits each direction at most once, so any "bias" binned from it is contaminated
by the spread of true DoAs inside the bin. Instead the DoA is **held fixed** on a
grid of 21 zenith rings x 48 azimuth spokes (1,008 directions) and only the
detector noise is redrawn, 96 trials each (96,768 queries, JNR 35 dB). With the
direction fixed, `MSE = |bias|^2 + scatter^2` is an exact identity rather than an
approximation.

Aggregate: median |bias| **1.994 deg**, median scatter **1.575 deg**, median RMSE
**2.956 deg**, median bias fraction **0.645**. The error is predominantly
**systematic**.

Split at the median CRLB (1.985 deg):

| Half | RMSE | abs(bias) | scatter | bias frac | RMSE/CRLB |
|---|---|---|---|---|---|
| Well-conditioned | 3.812 | 1.217 | 1.540 | 0.415 | **3.19** |
| Ill-conditioned | 7.775 | **5.470** | 1.642 | **0.894** | **0.68** |

**This is the mechanism, measured directly.** Between the two halves the
*scatter barely moves* (1.540 -> 1.642 deg, +7%) while the *bias grows 4.5x*
(1.217 -> 5.470 deg). The ill-conditioned half does not become noisier; it
becomes **systematically displaced**. Its bias fraction of 0.894 means about 89%
of its mean-square error is a fixed offset that no amount of averaging removes.

The decisive statistic: **268 of 1,008 directions (26.6%) have an RMSE below
their own CRLB**, and their median bias fraction is **0.884** against **0.508**
for the rest. A genuinely unbiased estimator *cannot* sit below the Cramer-Rao
bound, because the bound is stated for unbiased estimators. Sitting below it is
therefore positive evidence of bias, and the bias fraction confirms that the
sub-bound directions are precisely the biased ones. **The sub-CRLB result is not
a violation and must never be presented as "beating the bound".**

**The bias points at the sector centre.** Resolving each bias vector against the
bearing to the owning sector's centre gives a magnitude-weighted mean cosine of
**+0.536**, with the bias pointing inward on **83.6%** of directions by magnitude
(71.2% unweighted). This ties the statistical effect to the specific
architectural structure that produces it: Branch A's anchor is a convex
combination of subsector centres, so when the probes carry little information the
anchor relaxes toward the sector centre and the prediction is pulled inward. The
fallback is not an abstract "prior" — it is the sector geometry itself, which is
what makes this figure the link between the array's physical partition and the
bias/variance trade-off.

By zenith band (|bias|, scatter and CRLB in deg):

| theta band | n | abs(bias) | scatter | CRLB | RMSE/CRLB |
|---|---|---|---|---|---|
| 0-20 deg | 192 | 1.767 | 1.539 | 1.011 | 4.74 |
| 20-40 deg | 240 | 1.053 | 1.367 | 0.958 | 2.00 |
| 40-60 deg | 240 | 1.562 | 2.322 | 2.263 | 1.45 |
| 60-75 deg | 192 | **7.549** | 1.312 | 9.296 | 0.70 |
| 75-90 deg | 144 | 6.110 | 1.379 | **11.693** | 0.56 |

The trend is physical: toward the horizon the CRLB rises by more than **11x**
(1.011 -> 11.693 deg), because a 0.224-lambda array loses elevation sensitivity
at grazing incidence; the bias rises with it, the scatter does not, and the ratio
crosses below 1 between the 40-60 and 60-75 deg bands. **Caveat that must travel
with this figure:** theta = 60 deg is itself a sector boundary, so part of the
jump across that row is the partition showing through rather than a property of
the sky. The boundary is marked in the figure for exactly this reason.

**The CRLB is flat in JNR — and so is the network.** This is the most
counter-intuitive Phase 1 finding and it is confirmed twice:

| | JNR < 30 dB | JNR > 40 dB |
|---|---|---|
| CRLB arc | 6.059 deg | 5.831 deg |
| Network RMSE | 7.240 deg | 7.114 deg |

`CRLB_V2_VERIFICATION.txt` states the 1J bound spread over a 30 dB JNR sweep as
**0.000 dB (FLAT)**. The reason is structural: at K = 1e6 the multiplicative
snapshot term is negligible against the additive detector term (worst
snapshot/detector ratio **0.027**; one would need K > 5.4e5 merely to push it
under 0.05), and the additive term scales with `mean(pbar)^2`. Raising the jammer
power scales signal and the dominant noise term together, so the Fisher
information is *invariant to absolute JNR*. **Single-jammer accuracy in this
system is set by geometry and by detector noise, not by jammer strength.** Any
paper text promising a JNR-dependent accuracy curve for 1J is wrong.

**Bound validity checks** (`CRLB_V2_VERIFICATION.txt`): the Section II
simplification changes the bound by a ratio of 0.999; dropping `F_cov` gives
1.000 and is conservative; Monte-Carlo-to-CRB ratios are 0.901 / 0.926 / 0.898 at
20 / 35 / 50 dB, i.e. the MC estimator sits just under the bound at all three
levels, consistent with a correct bound plus finite-sample bias. The bound is
sound.

---

## 3. Phase 2 — two jammers, resolution

Source: `evaluation_results/PHASE2_V2.txt`. 10,000 two-jammer samples.

**Headline:** resolution **62.1%**, Hungarian assignment swaps **0.2%**, median
error/separation ratio **0.155**.

The **0.155** figure settles the collapse question. A model that simply predicted
the centroid of the two jammers for both outputs would score err/sep = 0.50 by
construction. At 0.155 the two outputs are genuinely tracking two distinct
sources, so the centroid-collapse failure mode is ruled out directly by this
statistic and needs no separate diagnostic.

**Resolution vs separation** (percent resolved / mean error in deg):

| Sep (deg) | Resolved | Mean err | Sep (deg) | Resolved | Mean err |
|---|---|---|---|---|---|
| 5-10 | 0.0% | 8.05 | 50-55 | 36.3% | 15.18 |
| 10-15 | 7.5% | 7.40 | 55-60 | 44.6% | 15.42 |
| 15-20 | 9.9% | 9.55 | 60-65 | 44.7% | 15.98 |
| 20-25 | 13.7% | 10.67 | 65-70 | 52.2% | 16.18 |
| 25-30 | 19.5% | 11.08 | 70-75 | 60.5% | 15.46 |
| 30-35 | 17.6% | 12.60 | 75-80 | 65.1% | 15.67 |
| 35-40 | 21.2% | 13.58 | 80-85 | 70.7% | 15.20 |
| 40-45 | 23.8% | 13.88 | 85-90 | 77.4% | 14.90 |
| 45-50 | 28.9% | 14.92 | | | |

The **50% resolution crossing is near 68 deg** — a very coarse resolution limit,
which is the honest consequence of a 0.224-lambda aperture read through 8 scalars.

**The mean-error column is the more informative one.** It does *not* track
Delta/2 (which would rise to 45 deg at Delta = 90 deg); it **flattens at
15-16 deg** from Delta ~ 50 deg onward. So the failure mode at wide separation is
not "outputs merge onto the centroid" — it is a bounded, roughly separation-
independent assignment/localisation error. The network finds two lobes but places
each imprecisely.

**Power disparity is not the limiting factor.** Resolution is 65.1% when
|dJNR| <= 5 dB (n = 3,073) and 57.4% when |dJNR| > 15 dB (n = 2,560). A 7.7-point
spread across a >15 dB disparity is small compared with the 0% -> 77% swing
driven by separation alone. **Two-jammer performance is separation-limited, not
dynamic-range-limited** — which is the opposite of the single-target-in-clutter
case in §4.

---

## 4. Phase 3.5 — dynamic range and the weak-target asymmetry

Source: `evaluation_results/PHASE3_5_V2.txt`. 20,000 target/interferer points.

Error vs dJNR (dJNR = target minus interferer, dB):

| dJNR band | n | p50 (deg) | RMSE (deg) | p90 (deg) |
|---|---|---|---|---|
| -30 to -20 | 1,177 | 21.10 | 26.23 | 40.95 |
| -20 to -10 | 3,290 | 20.85 | 25.41 | 38.49 |
| -10 to -5 | 2,460 | 17.86 | 22.22 | 33.70 |
| -5 to +5 | 6,146 | 12.56 | 17.12 | 26.52 |
| +5 to +10 | 2,460 | 7.45 | 10.43 | 15.89 |
| +10 to +20 | 3,290 | 4.95 | 7.98 | 13.32 |
| +20 to +30 | 1,177 | 4.26 | 7.35 | 12.96 |

Weak-target RMSE **25.62 deg** vs strong-target **7.82 deg**: an asymmetry of
**3.28x**. Relative power, not absolute power, is what governs accuracy — exactly
consistent with the JNR-flatness of §2.

**The honest verdict, and it is a negative result.** Section III of the paper
predicts a weak-target slope of **+1.000 dB/dB** (error degrading one-for-one as
the target is buried). The **CRLB obeys this law**: the measured weak-jammer
slopes in `CRLB_V2_VERIFICATION.txt` are -0.561, -0.798, -0.925, -0.975, -0.992
at Delta = 5, 10, 15, 20, 25 dB, mean **-0.850 dB/dB** against the predicted
-1.000, and the weak-jammer CRB diverges from 4.633 deg at Delta = 0 to
**617.99 deg at Delta = 25 dB**. The **network does not**: its measured
weak-target slope is **+0.051 dB/dB**, i.e. essentially flat.

The information-theoretic law is real and is confirmed in the bound. The network
simply **saturates** long before it: past roughly 10 dB of burial it stops
extracting the (rapidly vanishing) weak-target information and falls back on its
prior, producing a bounded ~21-26 deg error instead of the unbounded degradation
the bound permits. This is the same regression-to-the-prior mechanism that
produced the 0.70x ill-conditioned ratio in §2, seen from a second direction.

The paper must state this as a two-part result: **law holds for the CRLB,
network saturates.** Reporting only "+0.051, flat, therefore robust" would be
dishonest; reporting only the -0.850 slope would misrepresent the estimator.

**Figure coverage — how the error map is sampled.** The dJNR-vs-separation error
map (`figures/phase3.5/phase3_5_error_heatmap.{pdf,png}`) is binned on a fixed
grid of 2 dB x 5 deg cells: 30 dJNR columns x 20 separation rows = **600 cells**,
with a 5-sample minimum per cell. Binning the evaluation split fills only
**468/600 (78.0%)** of them, median cell n = 22, because that split follows the
deployment distribution and the corners of the grid are rare under it:
|dJNR| = 30 dB requires the single JNR pair (50, 20) dB, and separation < 5 deg
requires two jammers straddling a sector boundary. The published map is therefore
rendered from **18,000 scenes synthesised directly on the grid** — 721,819
feasible candidates out of 1,500,000 drawn, thinned to 60 per cell — pushed
through the generator's own measurement path with the three physical constraints
*enforced* rather than sampled around (distinct sectors, both jammers visible at
theta <= 89 deg, both JNRs inside 20-50 dB). Coverage is **600/600 (100%)**,
median cell n = 60. Cells that are physically unreachable are painted grey
(`set_facecolor("0.82")`) rather than left white, so a blank cell now reads as a
constraint of the geometry and not as missing data.

**Every statistic quoted in this section still comes from the evaluation split.**
Grid synthesis changed the heatmap only; the p50/RMSE/p90 table, the 3.28x
asymmetry and both slopes are unchanged.

---

## 5. Phase 3J — three jammers

Source: `evaluation_results/PHASE3J_V2.txt`.

Arc error (p50 / p90 / RMSE, deg):

| Case | n queries | p50 | p90 | RMSE |
|---|---|---|---|---|
| 1J | 10,000 | 3.72 | 12.84 | 7.07 |
| 2J | 20,000 | 11.86 | 29.49 | 18.01 |
| 3J | 30,000 | 15.26 | 34.27 | 21.35 |

Three-jammer resolution **14.0%** at a median closest-pair separation of
48.9 deg. Joint nulling of all three (2,000 samples) gives mean JSR **9.91 dB**,
median 7.93 dB, P(JSR >= 10 dB) = 43.2%.

Two structural facts explain the degradation, and both are array limits rather
than network limits:

1. Nulling three sources with a 4-element array consumes three of four degrees of
   freedom, leaving **a single DoF** for gain toward the satellite. The 9.91 dB
   mean JSR is close to what that DoF budget allows.
2. The artifact's own caveat: *"Ten of the sixteen available real observables are
   consumed at J = 3 and the FIM is badly scaled, so the accuracy ceiling is set
   by the array."* See §7.

**The 1J -> 2J jump (3.72 -> 11.86 deg p50) is much larger than 2J -> 3J
(11.86 -> 15.26).** The cost is paid on admitting the second source; the third is
comparatively cheap because by then the estimator is already prior-dominated.

---

## 6. Phase 4 — nulling and JSR

Source: `evaluation_results/PHASE4_V2.txt`. JSR capped at 50 dB and floored at 0
(so reported means are conservative at the top and truncated at the bottom).

**Single jammer** (4,000 samples, 0 degenerate): median JSR **27.71 dB**, p10
19.77, p90 37.71. P(JSR >= 10 dB) = 98.6%, P(>= 20 dB) = **89.3%**,
P(>= 30 dB) = 37.3%.

JSR tracks DoA accuracy monotonically, as a projection null must:

| DoA arc error | n | Median JSR |
|---|---|---|
| <= 1 deg | 423 | 34.24 dB |
| <= 3 deg | 1,735 | 29.17 dB |
| <= 5 deg | 2,376 | 28.21 dB |

**Two jammers** (4,000 samples / 8,000 target evaluations): mean JSR **6.54 dB**
against the weak jammer, **24.20 dB** against the strong one. The weak jammer is
the one that survives — and since suppression is what matters operationally, the
weak-target accuracy deficit of §4 propagates directly into a ~17.7 dB
suppression deficit.

**The counterintuitive inversion.** Mean JSR at separation >= 60 deg is
**14.04 dB**, but at separation < 30 deg it is **17.76 dB** — *closely spaced
jammers are suppressed better* (session log L15990). This is not a bug. A
projection null has finite angular width; when two jammers fall inside one
null's width, a single well-placed null attenuates both, and the second DoF is
spare. When they are far apart, two independent nulls must each be placed
accurately from a 62.1%-resolution estimate, and each mis-placement costs
suppression directly. **Resolution and suppression are not the same objective**,
and in this system they are locally anti-correlated. That is a genuinely
publishable observation and it should be presented as such, with the mechanism
given, not buried as an anomaly.

**Figure coverage — how the JSR map is sampled.** The JSR map
(`figures/phase4/phase4_2_jsr_heatmap.{pdf,png}`) is binned on the *same* grid
and drawn from the *same* cached scene set as the Phase 3.5 error map
(`data/grid_2j_djnr_sep.npz`, imported directly so the two cannot drift apart).
The two figures therefore describe **identical geometries** and can legitimately
be read against each other cell by cell. On the evaluation split the JSR map
filled **368/600 (61.3%)** of cells at median n = 10; on the synthesised grid it
fills **600/600 (100%)** at median n = 60. Recomputed on that grid the two-jammer
split reads **8.69 dB** (weak) / **25.18 dB** (strong) against **6.54 / 24.20 dB**
on the evaluation split: the same ~16-17 dB asymmetry, confirming it is a
property of the geometry rather than of how the evaluation set happened to be
drawn. As in §4, the numbers quoted above are the evaluation-split numbers; the
grid affects the figure only.

---

## 7. Identifiability and the algebraic ceiling

This section supplies the limit against which §§3, 5 and 6 should be read.

Source: `evaluation_results/CONDITIONING_VS_DOF.txt`, figure
`evaluation_results/figures/phase3j/phase3j_conditioning_vs_dof.{pdf,png}`. The
observable-count *derivation* is analytic (below); the *measured* conditioning
and bound degradation come from that script and corroborate the caveat line in
`PHASE3J_V2.txt`.

**Observable count.** With measured HFSS element patterns the per-sector
measurement manifold has **d = 16** real dimensions. Under the idealized
common-pattern assumption the same count collapses to **d = 9**. The real array's
element patterns differ from one another, and that dissimilarity is itself
information — the measured array is *more* identifiable than the idealized model
of it, not less.

**Jammer capacity.** Each jammer contributes 3 real unknowns (2 angles + 1
power). Counting against d = 16 gives

```
3J + (bookkeeping) <= 16   ->   J <= 5 algebraically
```

So **J <= 5** is the algebraic identifiability ceiling for the real array.

**Correction to the bounds document.** `paper_model_workspace/Algorithmic
Bounds.md` §I states a cap of **J <= 2**. That cap is an **artifact of the
common-pattern assumption** (d = 9), not a property of the physical array. It
should be corrected when the bounds material is folded into the paper; leaving it
would understate the array's capacity by more than a factor of two.

**Why J = 3 nonetheless fails in practice — measured.** Algebraic
identifiability is necessary, not sufficient. Over 300 random scenes per J, each
jammer's own sector codebook deployed and JNRs drawn over 20-50 dB:

| J | params | probes | budget | kappa_corr | CRLB med | network RMSE | RMSE/CRLB |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 8 | 4/16 | 2.450e+03 | 5.793 deg | 7.072 deg | **1.22** |
| 2 | 7 | 16 | 7/16 | 1.705e+04 | 28.918 deg | 18.015 deg | **0.62** |
| 3 | 10 | 24 | 10/16 | 1.089e+05 | 49.969 deg | 21.348 deg | **0.43** |

From 1J to 3J the scaled FIM conditioning degrades **44.4x** and the bound rises
**8.63x** (5.793 -> 49.969 deg), while the network's RMSE rises only **3.02x**
(7.072 -> 21.348 deg). **The RMSE/CRLB ratio therefore falls, 1.22 -> 0.43: the
network moves *closer* to the bound as jammers are added.** The information
available to *any* estimator degrades faster than this estimator's performance
does, which is the argument that the 3J drop is a hardware/observability limit
rather than a network deficiency. Note the consistency with §2.1: the ratio drops
below 1 exactly where bias takes over, so 0.43 at 3J is the prior-dominated
regime and not a claim of efficiency.

**Which condition number, and why it matters.** `kappa_corr` is the condition
number after symmetric diagonal scaling, `D^-1/2 F D^-1/2` with `D = diag(F)`.
The **raw** FIM condition number is unusable here and is carried in the artifact
only to document that it was examined and rejected: it reads ~5e18 at J = 1 and
~1.2e23 at J = 3, but that is an artifact of *units*, not of observability — the
noise-floor nuisance column of the Jacobian carries `sigma_n^2 ||w||^2` while a
jammer-power column carries `P_j |g|^2`, and at these JNRs those differ by
10^2-10^5, which the FIM then squares. Quoting the raw kappa would imply a
degeneracy at J = 1, where four parameters are recovered from eight probes and
nothing is degenerate. Every marginalised CRLB in the table above was finite
(0 failures across the 900 scenes).

Consistently with this, §2's ill-conditioned half (0.70x, prior-dominated) and
§4's saturation apply in the same regime. The measured 14.0% three-jammer
resolution and 9.91 dB joint JSR are therefore **consistent with the array's own
conditioning ceiling**, and are not evidence of a network deficiency.

The clean statement for the paper: *identifiability permits up to five jammers;
conditioning, not identifiability, is what limits the real system to
approximately two.*

---

## 8. Phase 5 — what the network actually does

Source: `evaluation_results/PHASE5_V2.txt`, figures in
`evaluation_results/figures/phase5/`. 60,000 queries (1J 10,000 / 2J 20,000 /
3J 30,000), sequence length 10.

**Self-tests pass.** Manual recomposition of `forward()` matches to
max|diff| = **0.000e+00**. Attention weights are read with
`need_weights=True, average_attn_weights=False` and **all 10 columns** are
retained, including CLS self-attention; rows consequently sum to 1 with max
deviation **2.204e-07**. Dropping the self-attention column is the easy mistake
here and it silently renormalises every published row, so the row-sum check is
reported as a precondition for reading any attention number below.

### 8.1 The anchor performs the physically correct readout

The r(p, w) sign test correlates each query's probe powers with Branch A's
softmax weights:

| Case | Pearson mean | Pearson median | frac < 0 | Spearman median |
|---|---|---|---|---|
| 1J | **-0.603** | **-0.691** | **0.965** | -0.643 |
| 2J | — | -0.488 | — | — |
| 3J | — | -0.404 | — | — |
| all | — | -0.480 | — | — |

The correlation is strongly **negative**, with 96.5% of 1J queries negative. The
artifact's own verdict: *"strongly NEGATIVE -> the anchor reads the notch.
Branch A is behaving as a genuine soft-argmin."*

This is the single most important mechanistic result, and the sign is the whole
point. The probes are *nulling* beams, so the informative probe is the one with
the **least** power — it is the probe whose null is pointed at the jammer.
A softmax anchor is, structurally, a soft-*argmax* over its 8 weights; for it to
localise correctly it must therefore learn to place its weight where the power is
*low*, i.e. to invert the sign of the mapping from powers to weights. It does:
r(p, w) is negative on 96.5% of queries. **Nothing in the loss or the
architecture specifies this** — the network is given 8 unlabelled scalars and
infers from data that they are nulls rather than beams.

The 1J readout diagnostics confirm it against a chance level of 0.125 (1 in 8
subsectors):

| Statistic | Value | Chance |
|---|---|---|
| P(argmax w == argmin p) | **0.511** | 0.125 |
| P(argmax w == nearest centre) | **0.490** | 0.125 |
| P(argmin p == nearest centre) | **0.697** | 0.125 |

The last row involves **no network at all** — it is a property of the
measurement. The deepest probe names the occupied subsector 69.7% of the time
against a 12.5% chance level, which is the direct evidence that the 8 scalars
carry the DoA in the first place. It is corroborated by
`data/VALIDATION_v2.txt` check 3: median on-centre null depth **90.49 dB**. The
first two rows then show the network recovering roughly 0.5 of that available
signal in a single hard readout, with the remainder supplied by Branch B (§8.3).

### 8.2 Branch B attention is degenerate, and the SCALE token is ignored

Per-head CLS attention entropy is **0.000-0.001 nats** against 2.303 nats for
uniform over 10 tokens. Each head is **near one-hot per query**. Probe mass is
approximately 1.000.

**Consequence for reporting:** the *mean* attention row is meaningless. Head 0's
mean row, for example, is
`[0, 0, .125, 0, .249, .127, 0, 0, 0, .499]` — which looks like a smooth
multi-probe combination but is in fact the average of many near-one-hot rows
selecting different probes on different queries. **Mean != typical must be stated
explicitly** beside any published attention map, and the entropy must be
reported alongside it.

**The SCALE token receives 0.000 attention from CLS in all four heads.** Head 1
retains CLS self-attention of 0.085; all other mass is on probe tokens.
Recommendation #2's absolute-scale information therefore reaches the output
**only** through the direct `log_scale` concatenation into `correction_head`, and
**not** through attention. The token is inert; the skip path is doing the work.
This is a precise, measured architectural finding — see the critique document.

### 8.3 Anchor -> final decomposition: Branch B does ~94% of the work

1J arc error at three stages:

| Stage | p50 | p90 | mean |
|---|---|---|---|
| hull only | 10.120 | 20.460 | 11.173 |
| hull + offset (rec #6) | **9.726** | 22.054 | 11.345 |
| final (after Branch B) | **3.717** | 12.843 | 5.404 |

Rec #6 improves the median by **+0.394 deg** and **degrades both p90
(20.460 -> 22.054) and the mean (11.173 -> 11.345)**. Branch B's correction is
worth **+6.009 deg** of median. Of the total 6.403 deg anchor-to-final
improvement, Branch B supplies **~94%**.

Two reframings follow, and both matter for the paper's argument:

**(i) The convex hull is not the accuracy-limiting structure.** The concern with
a hull anchor is that its output is confined to the convex hull of the 8
subsector-centre encodings, which would impose a floor on how well it can
localise. Measured, that floor is not where the error is: the hull alone already
reaches **10.120 deg** p50, and releasing it from the hull via rec #6 buys only
0.394 deg. The remaining 6.009 deg comes from Branch B. So the hull's geometric
restriction costs little; what the anchor lacks is not freedom of position but
resolution, and that is supplied by attention over the probe tokens.

**(ii) Recommendation #6 is approximately accuracy-neutral.** Its value was
removing the *structural* floor (giving the anchor the freedom to leave the hull
at all), not reducing error. Claiming #6 as an accuracy win is not supportable:
median +0.394 deg with p90 and mean both worse is noise-level at best.

---

## 9. Divergences between plan and execution

Recorded here so the paper's methods section matches what was actually run:

1. Recommendation **#7 (stacked layers + FFN) was cancelled by the user**
   (session log L13882): *"for Rec #7 I do not need to do that, I do not need
   full transformer architectures, I need my architecture only."*
2. A separate **dedicated workspace was mandated** with go/no-go gating (L12117),
   so every artifact cited in this document was produced by a single code base.
3. **GPU retrain mandated** (L14583): 50 epochs, ~23 min.
4. **K raised 8192 -> 1e6** (see §1).
5. **Bounds math re-proven** from first principles against the dataset actually
   used here, rather than inherited from any earlier derivation (L13140).
6. Phases A-D executed in ~70 min against a 10.75 h budget.
7. Plan todo statuses are **stale by the agent's own admission** (L15329).
8. The **1.75x / 1.41x efficiency figures were retracted** (L15565-15570).
9. Phases 4 and 4.2 were **merged** into one script, and the shared evaluation
   scaffolding was consolidated into `evaluation/v2_common.py` (261 lines).
10. `diagnose_collapse_mode.py` was **never run**; the centroid question was
    settled instead by Phase 2's err/sep = 0.155 (§3).

Phase E was executed in the order Phase 1 -> Phase 3.5 -> Phase 2 -> Phase 4
(1J+2J merged) -> 3J -> invariances, not in numeric order.

---

## 10. Retracted and do-not-quote numbers

| Item | Status | Reason |
|---|---|---|
| **1.75x** efficiency | **RETRACTED** | Median/mean-bound artifact (log L15565-15570). Never quote. |
| **1.41x** efficiency | **RETRACTED** | Same. Never quote. |
| Any single-number CRLB efficiency | Do not quote alone | Use the conditioning split of §2 (1.93x / 0.70x). |
| `Algorithmic Bounds.md` §I "J <= 2" | **Superseded** | Artifact of the common-pattern d = 9 assumption; the real array gives J <= 5 (§7). |
| Raw FIM condition number (5e18 to 1e23) | Do not quote | Units artifact of the `ln sigma_n^2` Jacobian column; use the diagonally scaled `kappa_corr` (§7). |
| Mean attention rows without entropy | Do not quote alone | Heads are near one-hot; mean != typical (§8.2). |
| CRLB Phase-1 single geometry (0.893 deg) | Benign, do not headline | Sits on the dataset p10 of 0.920 deg; Phase 1's 1,500-bound aggregate is the correct figure. |

---

## 11. Summary of claims the v2 data supports

1. The probe measurement contains the DoA: 90.49 dB median on-centre null depth,
   and argmin-p identifies the nearest sector centre 69.7% of the time against a
   chance rate of 12.5%.
2. Branch A learned the **physically correct soft-argmin** readout: r(p,w) median
   -0.691 at 1J, 96.5% negative.
3. 1J: p50 3.717 deg, RMSE 7.072 deg, median JSR 27.71 dB, P(JSR >= 20 dB) 89.3%.
4. 1J accuracy is **flat in absolute JNR** for both the bound (0.000 dB spread)
   and the network — a structural consequence of the additive detector noise at
   K = 1e6.
5. Efficiency is geometry-dependent: **1.93x** the bound where well-conditioned,
   **0.70x** (prior-dominated) where not. No single efficiency number is valid.
6. 2J is **separation-limited** (50% crossing ~68 deg), not dynamic-range-limited
   (65.1% vs 57.4% across >15 dB disparity).
7. The weak-target law **holds for the CRLB** (mean slope -0.850 dB/dB, CRB
   diverging to 617.99 deg) but the **network saturates** (+0.051 dB/dB),
   trading unbounded degradation for a bounded ~21-26 deg prior-driven error.
8. Suppression **inverts** with separation: 17.76 dB at sep < 30 deg vs 14.04 dB
   at sep >= 60 deg. Resolution and suppression are distinct objectives.
9. **Identifiability permits J <= 5** on the real array (d = 16); conditioning,
   not identifiability, limits the system to about two jammers.
10. **Branch B does ~94%** of the anchor-to-final work; rec #6 is
    accuracy-neutral and rec #2's scale token is bypassed by attention entirely.
11. The sub-bound behaviour is **bias, not variance, and it is spatial**: across
    1,008 fixed directions x 96 noise trials, |bias| grows 4.5x from
    well-conditioned to ill-conditioned geometry (1.217 -> 5.470 deg) while
    scatter barely moves (1.540 -> 1.642 deg), and the 268/1,008 directions that
    sit below their own CRLB carry a median bias fraction of 0.884 vs 0.508
    elsewhere. The bias points **inward, at the sector centre**
    (magnitude-weighted mean cosine +0.536; inward 83.6% of the time by
    magnitude), which is exactly the structure Branch A's convex hull imposes.
12. The three-jammer drop is an **observability limit, not a network
    deficiency**: from J = 1 to J = 3 the parameter budget goes 4/16 -> 10/16 of
    the array's real observable dimension, the scaled FIM condition number
    degrades **44.4x** (2.45e3 -> 1.09e5) and the CRLB itself rises **8.63x**
    (5.79 -> 49.97 deg), while network RMSE rises only **3.02x**
    (7.07 -> 21.35 deg). The estimator closes on the bound as the bound
    collapses: RMSE/CRLB falls 1.22 -> 0.62 -> 0.43.

---

## 12. Artifact index

| Artifact | Contents |
|---|---|
| `evaluation_results/PHASE1_V2.txt` | 1J accuracy, CRLB comparison, conditioning split, JNR flatness |
| `evaluation_results/PHASE2_V2.txt` | 2J resolution, separation table, err/sep, dJNR bands |
| `evaluation_results/PHASE3_5_V2.txt` | Target/interferer dynamic range, weak/strong asymmetry, slope |
| `evaluation_results/PHASE3J_V2.txt` | 1J/2J/3J scaling, 3J resolution and joint JSR, FIM caveat |
| `evaluation_results/PHASE4_V2.txt` | 1J and 2J JSR, JSR-vs-accuracy, separation inversion |
| `evaluation_results/PHASE5_V2.txt` | Attention entropy, r(p,w) sign test, readout diagnostics, decomposition |
| `evaluation_results/INVARIANCES_V2.txt` | Corrected invariance verdicts (see ARCHITECTURE_CRITIQUE_V2.md) |
| `evaluation/CRLB_V2_VERIFICATION.txt` | Bound validity, MC checks, JNR flatness, weak-jammer slopes |
| `data/VALIDATION_v2.txt` | Dataset metadata (alpha, K), null-depth checks |
| `evaluation_results/PHASE1_BIAS_VARIANCE_V2.txt` | Per-direction bias/scatter decomposition, conditioning split, bias-direction test, zenith-band table |
| `evaluation_results/CONDITIONING_VS_DOF.txt` | Parameter budget vs d = 16, scaled and raw FIM conditioning, CRLB and network RMSE at J = 1, 2, 3 |
| `evaluation_results/figures/phase1/` | `phase1_bias_variance` (.pdf/.png): (a) hemispherical \|bias\| map, (b) scatter map, (c) bias vectors over bias fraction with sector centres marked, (d) RMSE/CRLB vs bias fraction |
| `evaluation_results/figures/phase3j/` | `phase3j_conditioning_vs_dof` (.pdf/.png): (a) parameter budget vs observable ceiling, (b) `kappa_corr` vs J, (c) CRLB vs J, (d) RMSE and RMSE/CRLB vs J |
| `evaluation_results/figures/phase3.5/` | `phase3_5_error_heatmap` (.pdf/.png): dJNR x separation error map, grid-sampled, 600/600 cells (§4) |
| `evaluation_results/figures/phase4/` | `phase4_2_jsr_heatmap` (.pdf/.png): JSR map on the same scenes as the Phase 3.5 map, 600/600 cells (§6) |
| `evaluation_results/figures/phase5/` | `phase5_attention_heads`, `phase5_branch_a_readout`, `phase5_decomposition` (.pdf/.png) |
| `data/grid_2j_djnr_sep.npz` | Cached 18,000-scene two-jammer grid shared by the Phase 3.5 and Phase 4 heatmaps |
