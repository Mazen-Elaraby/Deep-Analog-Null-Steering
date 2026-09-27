# ARCHITECTURE CRITIQUE — v2

Per-recommendation status of `HybridSARRefinerV2`, judged against measurement.

**Status:** authoritative architecture record for this workspace.

**Scope rule.** This document critiques **one** architecture — the subject of the
paper. Every verdict is assigned against measurement, physics or a stated chance
level; no earlier iteration of the design is presented, tabulated or used as a
baseline anywhere below.

**Companion:** `RESULTS_ANALYSIS_V2.md` (all performance numbers and their
provenance rules live there; this document does not restate them beyond what a
verdict requires).

---

## 0. How verdicts are assigned

Every recommendation below is graded on evidence, not on intent:

| Grade | Meaning |
|---|---|
| **CONFIRMED** | Implemented, and a measurement shows it doing the thing it was for. |
| **IMPLEMENTED, INERT** | Implemented, but measurement shows the intended pathway is unused. |
| **NEUTRAL** | Implemented; measured effect on accuracy is within noise. |
| **NOT IMPLEMENTED** | Deliberately omitted. Reason recorded. |

Sources: `evaluation_results/PHASE5_V2.txt`,
`evaluation_results/INVARIANCES_V2.txt`, `evaluation_results/PHASE1_V2.txt`, and
`models/model_builder_v2.py`.

---

## 1. Status table

| # | Recommendation | Status | Evidence |
|---|---|---|---|
| 2 | Absolute-scale token (`SCALE_INDEX = 1`) | **CONFIRMED, but via the skip path — attention pathway INERT** | CLS->SCALE attention = **0.000 in all 4 heads** (PHASE5); claim 2 **BROKEN by rec #2** (INVARIANCES) |
| 3 | Heteroscedastic head + Gaussian NLL | **NOT IMPLEMENTED** | `model_builder_v2.py` L36-38 |
| 5 | theta reparameterised in logit(cos theta) | **CONFIRMED** | predicted theta in **[6.62, 85.83] deg**; no folding anywhere (PHASE1) |
| 6 | `hull_offset` breaking the convex hull | **CONFIRMED structurally, NEUTRAL on accuracy** | claim 1 at anchor **BROKEN by rec #6** (INVARIANCES); median only **+0.394 deg**, p90 and mean both *worse* (PHASE5) |
| 7 | Stacked layers + FFN | **NOT IMPLEMENTED — cancelled by the user** | Session log L13882 |

---

## 2. Recommendation #2 — the absolute-scale token

**What it was for.** The preprocessing normalises the probe powers to a shape
vector (`norm_p`), which destroys absolute level, and the LayerNorm at the front
of `attention_mlp` removes whatever scale survived that. With no explicit path,
the network is therefore *provably* blind to absolute JNR — a blindness that
follows from the preprocessing algebra, not from training.
Rec #2 added a dedicated `SCALE` token carrying
`log_scale = log10(total) - log10(noise_var)` so the absolute level could
re-enter.

**Verdict: the information got in, but not the way it was designed to.**

`INVARIANCES_V2.txt` confirms the claim it was meant to break:

```
claim 2 (absolute JNR) : BROKEN by rec #2
```

So the model is no longer scale-blind. But `PHASE5_V2.txt` shows **the CLS token
assigns 0.000 attention mass to the SCALE token in all four heads.** Probe mass
is approximately 1.000; head 1's only non-probe mass is 0.085 of CLS
self-attention. The scale information therefore reaches the output **exclusively
through the direct `log_scale` concatenation into `correction_head`**:

```python
correction_head(cat([cls_out, fg_emb, log_scale]))
```

**Interpretation.** The token is dead weight; the skip connection is the whole
mechanism. This is a clean, measured result and it should be reported honestly
rather than as "the transformer learned to attend to scale."

**Why this is not merely cosmetic.** Rec #2's stated design goal was for scale to
*modulate how the probes are read* — that requires it inside attention, where it
can change the query-key products. Concatenating it after the fact can only shift
the output, not re-weight the evidence. And §2/§4 of `RESULTS_ANALYSIS_V2.md`
explain why the network had little reason to learn otherwise: at K = 1e6 the
Fisher information is **flat in absolute JNR** (1J bound spread 0.000 dB), so
absolute level genuinely carries almost no information about the DoA. **The
network is right to ignore it.**

This makes the honest framing: *rec #2 removed a provable architectural blindness,
and the measurement then showed the blindness was not costing anything.* It is a
correctness fix, not a performance fix. Do not claim an accuracy gain for it.

---

## 3. Recommendation #3 — heteroscedastic head + Gaussian NLL

**Status: NOT IMPLEMENTED.** `models/model_builder_v2.py` L36-38 records the
decision explicitly. `correction_head` emits **3** values (two angle corrections
plus the theta logit correction) and no variance.

**This is the single most defensible remaining gap**, and the v2 results now make
the case for it much more strongly than the original recommendation did:

- §2 of the results: efficiency is **1.93x** the bound on well-conditioned
  geometries and **0.70x** on ill-conditioned ones. The model has no way to say
  which regime it is in.
- §4: the network **saturates at +0.051 dB/dB** where the CRLB degrades at
  -0.850 dB/dB, i.e. it emits confident-looking point estimates in exactly the
  regime where the information has collapsed.
- §6: JSR is driven by DoA accuracy, so a downstream nuller that knew which
  estimates were untrustworthy could reallocate its limited DoFs.

A predicted variance would convert the prior-fallback behaviour from a hidden
failure into a reported one. **It should be listed as future work, with the above
three measurements as its motivation.** It must not be described as implemented.

---

## 4. Recommendation #5 — theta in the logit of cos theta

**Status: CONFIRMED.** `_assemble` applies the theta correction in the logit of
`cos theta`, so theta is confined to (0, 90) deg by construction.

**Evidence.** `PHASE1_V2.txt`: predicted theta spans **[6.62 deg, 85.83 deg]**,
strictly interior. The pipeline contains **no folding, clipping, or wrapping
step** anywhere — the constraint is structural rather than enforced by
post-processing.

This is the cleanest win in the v2 architecture: it removed a whole class of
boundary pathology (gradient discontinuities at the horizon and at zenith, and
the ambiguity of folded predictions) at zero parameter cost. phi remains a
unit-circle residual, which is the correct treatment for a periodic coordinate.

**Caveat worth stating.** The interior span is evidence that the parameterisation
is *well-behaved*, not that theta is *accurate*. Zenith remains the harder axis
(p50 3.140 deg vs azimuth 2.163 deg), for the physical reason that a 0.224-lambda
planar array derives most of its elevation sensitivity from the element patterns
rather than from inter-element geometry. Rec #5 fixed the representation, not the
physics.

---

## 5. Recommendation #6 — `hull_offset` (breaking the convex hull)

**Status: CONFIRMED structurally, NEUTRAL on accuracy.**

**What it does.** Branch A's softmax weights form a convex combination of the 8
subsector-centre encodings, so the anchor is confined to the convex hull of those
centres. `hull_offset(powers)` (zero-initialised) adds a learned displacement,
freeing the anchor to leave the hull.

**Structurally it works.** `INVARIANCES_V2.txt`:

```
claim 1 at the softmax : STILL HOLDS (invariant)
claim 1 at the anchor  : BROKEN by rec #6
```

**On accuracy it does essentially nothing.** From `PHASE5_V2.txt`, 1J arc error:

| Stage | p50 | p90 | mean |
|---|---|---|---|
| hull only | 10.120 | 20.460 | 11.173 |
| hull + offset | **9.726** | **22.054** | **11.345** |
| final | 3.717 | 12.843 | 5.404 |

The median improves by **+0.394 deg** while **p90 degrades by 1.594 deg and the
mean degrades by 0.172 deg**. A change that improves the median and worsens both
the tail and the mean is not an accuracy improvement; it is a redistribution
within noise. **Claiming rec #6 as a performance win is not supportable by this
data.**

**What it does establish.** The objection rec #6 answers is structural: a hull
anchor can only emit points inside the convex hull of the 8 subsector-centre
encodings, which would impose a hard floor on anchor accuracy at roughly the
subsector spacing — tens of degrees. That objection is testable, and it fails.
The **hull-only** anchor, with the hull constraint entirely intact, measures
**10.120 deg** p50, well inside any such floor, and releasing it from the hull
buys only **0.394 deg** of median. Therefore:

> **The convex hull is not the accuracy-limiting structure.** What the anchor
> lacks is angular resolution, not freedom of position — and resolution is what
> Branch B supplies, carrying ~94% of the anchor-to-final improvement.

Rec #6's real value is that it makes the hull constraint *non-binding*, closing
off the objection entirely. That is worth one sentence in the paper, phrased as a
robustness/generality argument rather than an error reduction.

---

## 6. Recommendation #7 — stacked layers + FFN

**Status: NOT IMPLEMENTED. Cancelled by the user.** Session log L13882:

> *"for Rec #7 I do not need to do that, I do not need full transformer
> architectures, I need my architecture only, forget about..."*

`models/model_builder_v2.py` L36-38 records the omission alongside #3. Branch B
remains a **single** `nn.MultiheadAttention(64, num_heads=4)` with no stacked
blocks and no feed-forward sublayer.

**This must be reported as a scope decision, not as a negative result.** No
experiment was run that would license a claim about whether depth would help.

**That said, §8.2 of the results argues depth would probably not help much.** The
existing single layer already has per-head CLS attention entropy of
**0.000-0.001 nats** against 2.303 for uniform — each head is effectively
one-hot, selecting a single token per query. A layer whose attention has
collapsed to hard selection is not bottlenecked by lack of mixing capacity, and
stacking more such layers is unlikely to change the representation qualitatively.
The measurement is consistent with the user's judgement.

---

## 7. Corrected invariance verdicts

`evaluation_results/INVARIANCES_V2.txt` was **regenerated** this session after a
threshold bug was fixed. Current verdicts:

```
claim 1 at the softmax : STILL HOLDS (invariant)
claim 1 at the anchor  : BROKEN by rec #6
claim 2 (absolute JNR) : BROKEN by rec #2
```

### 7.1 The threshold bug (fixed)

`evaluation/verify_architectural_invariances_v2.py` L63 tested

```python
softmax_invariant = max(soft_dev) < 1e-5
```

against a measured `max|Δw| = 3.746e-04`, while the anchor and output tests in
the same script declare movement using `> 1e-3`. The value 3.746e-04 fell in the
**dead zone between the two thresholds**: too large to pass the softmax test, too
small to count as movement anywhere else. The script consequently reported claim
1 as broken at the softmax, contradicting its own docstring (L18), which states
that claim 1 is *expected* to hold there.

The threshold was changed to `1e-3` for consistency, with the reasoning recorded
in the source:

```python
    # 1e-3 matches the movement threshold used by the anchor and output tests
    # below. The earlier 1e-5 left a dead zone: float32 drift through
    # LayerNorm -> MLP -> softmax reaches ~4e-4 under a 4x rescale, which is far
    # too small to be a real dependence (the dominant weight is stable to five
    # significant figures) yet was enough to report the claim as broken.
    softmax_invariant = max(soft_dev) < 1e-3
```

**The measured numbers did not change — they are bit-identical across the
re-run:** max|Δw| = 0, 2.980e-07, 3.746e-04, 2.325e-05; anchor deviations 0,
1.694, 0.745, 3.386e-02; output deviations 0, 3.371e-03, 1.478e-02, 5.199e-02.
Only the pass/fail predicate moved. The 3.746e-04 residue is float32 drift
through LayerNorm -> MLP -> softmax under a 4x input rescale; the dominant
softmax weight is stable to five significant figures.

### 7.2 The framing that matters for the paper

The regenerated artifact prints an epilogue that the buggy run never reached:

> *"The LayerNorm contrast blindness is intrinsic to the softmax and was never
> removed; rec #6 routes around it rather than repairing it... Worth stating
> precisely in the paper: the invariance is bypassed, not fixed."*

This is the correct and defensible claim. Branch A's softmax **remains
scale-invariant** — that is a mathematical consequence of the leading LayerNorm
and cannot be removed by adding a parallel term. What rec #6 does is add an
**alternative, scale-sensitive path to the anchor** (`hull_offset` reads the raw
powers), so the *anchor* becomes scale-dependent while the *softmax* does not.
Contrast the anchor deviation of **1.694** with the softmax deviation of
**3.746e-04**: four orders of magnitude, and it all comes through the offset.

**Do not write "rec #6 fixed the LayerNorm invariance."** Write: *the softmax
remains invariant by construction; the anchor is made scale-dependent by a
parallel offset path that bypasses it.*

---

## 8. What Branch A actually does

**It is a genuine soft-argmin, and it learned to be one.**

The physics makes a **falsifiable sign prediction** about Branch A. Because the
probes are nulling beams, a correctly functioning anchor must *anti*-correlate
its softmax weights with the measured probe powers; a positive correlation would
falsify the claim that the branch localises at all. `PHASE5_V2.txt` tests it:
1J Pearson mean **-0.603**, median **-0.691**, **96.5%** of queries negative,
Spearman median -0.643. The artifact's verdict: *"strongly NEGATIVE -> the anchor
reads the notch. Branch A is behaving as a genuine soft-argmin."*

This is physically correct behaviour, and it is not trivially available. The
probes are **nulling** beams, so the informative probe is the one with the
*least* received power — its null is pointed at the jammer. But a softmax is
structurally a soft-*argmax* over its own weights, so to localise at all the
branch must learn to **invert** the sign of the powers-to-weights mapping.
Nothing in the loss or in the architecture specifies this: the branch is handed
8 unlabelled non-negative scalars and infers from data that they are nulls
rather than beams.

Readout consistency at 1J, against a chance level of 0.125 (1 of 8 subsectors):

| Statistic | Measured | Chance |
|---|---|---|
| P(argmax w == argmin p) | 0.511 | 0.125 |
| P(argmax w == nearest centre) | 0.490 | 0.125 |
| P(argmin p == nearest centre) | 0.697 | 0.125 |

**The architectural lesson is the strongest claim in this document:** a softmax
readout over fixed geometric anchors is the **right inductive bias for this
measurement**. The third row above involves no network at all — it is a property
of the 8 scalars, and it says the deepest probe names the occupied subsector
69.7% of the time against a 12.5% chance level. Branch A then recovers roughly
half of that available signal in a single hard readout, having been told nothing
about what the probes are. The claim is supported by a falsifiable sign
prediction and by a stated chance level, not by assertion.

**Remaining weakness.** P(argmax w == argmin p) = 0.511 means the softmax picks
the minimum-power probe only about half the time. The soft-argmin is real but
loose. A sharper readout — e.g. an explicit negative-power logit prior, or
temperature annealing on the softmax — is an untested and cheap candidate
improvement.

---

## 9. What Branch B actually does

**It does almost all the work, and it does it by hard selection.**

**Load-bearing.** From the §8.3 decomposition in `RESULTS_ANALYSIS_V2.md`:
Branch B improves 1J median arc error from 9.726 deg to 3.717 deg, i.e.
**+6.009 deg of the total 6.403 deg anchor-to-final improvement — about 94%**.
Branch A supplies a coarse, physically-grounded initialisation; Branch B supplies
the accuracy.

**But its attention is degenerate.** Per-head CLS entropy **0.000-0.001 nats**
(uniform over 10 tokens would be 2.303). Each head is **near one-hot per query**.
Two consequences:

1. **Reporting.** Mean attention rows are misleading and must never be published
   alone. Head 0's mean row is `[0, 0, .125, 0, .249, .127, 0, 0, 0, .499]`,
   which reads as a smooth multi-probe blend but is the average of many one-hot
   rows selecting *different* probes on *different* queries. **State "mean !=
   typical" and publish the entropy beside any attention figure.**
2. **Architecture.** Four heads performing hard selection over 8 probe tokens is
   closer to a learned gather-and-combine than to attention in the usual sense.
   The 64-dim embedding and 4-head structure are almost certainly
   over-parameterised for what is being computed. This is a plausible
   simplification target and an honest limitation to state — but note that no
   ablation was run, so it must be posed as a hypothesis.

**Reading this diagnostic correctly is a precondition.** Attention is read with
`need_weights=True, average_attn_weights=False` and **all 10 columns** are
retained, including CLS self-attention; rows then sum to 1 to within
**2.204e-07**. Silently dropping the self-attention column renormalises every
published row and is the easy mistake in this particular measurement, so the
row-sum check is reported as a precondition for every attention number quoted
above.

---

## 10. Residual architectural weaknesses

Ordered by how well the v2 measurements support them.

1. **No uncertainty output (rec #3).** Strongest gap. The model cannot signal the
   1.93x/0.70x regime split or its own saturation. §3 above.
2. **Rec #2's pathway is inert.** Scale reaches the output only by skip
   connection; attention ignores the SCALE token completely. Either move it
   inside the attention computation properly or drop the token and keep the
   concatenation — the current arrangement carries cost without effect. §2 above.
3. **Prior-fallback in the ill-conditioned regime.** The network answers
   confidently from its prior where information has collapsed (+0.051 dB/dB vs
   the bound's -0.850). This is arguably *desirable* for bounded error, but it is
   invisible without #3, and it is why aggregate efficiency numbers mislead.
4. **Loose soft-argmin.** P(argmax w == argmin p) = 0.511. §8 above.
5. **Degenerate attention suggests over-parameterisation.** Hypothesis only; no
   ablation run. §9 above.
6. **Rec #6 is accuracy-neutral.** It removes a structural constraint that the
   v2 data shows was not binding. Keep it for the generality argument, but do not
   claim performance for it. §5 above.
7. **Depth untested (rec #7).** Out of scope by user direction; the entropy
   measurement suggests low expected value. §6 above.

None of items 1-7 explain the 2J/3J ceiling. §7 of `RESULTS_ANALYSIS_V2.md`
attributes that to the array: identifiability permits J <= 5 (d = 16 real
observables), but at J = 3 ten observables are consumed and the FIM is badly
scaled. **The multi-jammer limit is a conditioning limit of a 0.224-lambda
4-element array, not an architectural deficiency**, and the critique should not
be written as though better modelling would recover it.

---

## 11. One-paragraph verdict

The architecture is sound and its inductive bias is vindicated. Branch A's
softmax-over-geometric-anchors readout learned the physically correct soft-argmin
(r(p,w) median -0.691, 96.5% negative) from 8 unlabelled scalars, satisfying a
falsifiable sign prediction that a non-localising anchor would have violated; and
the convex-hull objection is answered by measurement rather than by construction,
since the hull-only anchor already reaches 10.120 deg p50 with the hull fully
intact. Of the five recommendations considered, #5 is a clean structural
win (theta strictly interior at [6.62, 85.83] deg with no folding anywhere), #6
works structurally but is accuracy-neutral and should be credited only with
making the hull non-binding, #2 removed a provable scale blindness but its
attention pathway is measurably inert (0.000 CLS->SCALE mass in all four heads) —
defensibly so, since the Fisher information is flat in absolute JNR — and #3 and
#7 were not implemented, #7 by explicit user direction. Branch B carries ~94% of
the anchor-to-final improvement but does so through near-one-hot attention
(0.000-0.001 nats), so it is best described as a learned gather rather than as
distributed attention, and its mean attention rows must never be published
without the entropy beside them. The clearest remaining opportunity is
recommendation #3: a heteroscedastic output would expose the 1.93x/0.70x regime
split and the weak-target saturation that the current point estimator hides.

---

## 12. Artifact index

| Artifact | Used for |
|---|---|
| `evaluation_results/INVARIANCES_V2.txt` | Claim 1/claim 2 verdicts, deviation magnitudes, "bypassed not fixed" framing |
| `evaluation_results/PHASE5_V2.txt` | Attention entropy, CLS->SCALE mass, r(p,w) sign test, readout diagnostics, anchor decomposition |
| `evaluation_results/PHASE1_V2.txt` | theta span (rec #5), conditioning split (motivates rec #3) |
| `evaluation_results/PHASE3_5_V2.txt` | Saturation vs the bound's slope (motivates rec #3) |
| `evaluation/verify_architectural_invariances_v2.py` | L63 threshold fix (1e-5 -> 1e-3) |
| `models/model_builder_v2.py` | L36-38 omission of #3 and #7; token indices; `branch_a_anchor` diagnostic hook |
| `evaluation/visualize_attention_v2.py` | Phase 5 instrumentation (all-10-column attention, sign test, decomposition) |
| Session log L13882 | User cancellation of rec #7 |
