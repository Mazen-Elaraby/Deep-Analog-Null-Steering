# Manuscript

**Title:** Deep Analog Null Steering: Real-Time DoA Estimation for GNSS
Multi-Jammer Mitigation on SWaP-Constrained Analog CRPAs

## Layout

```
manuscript/
  main.tex        the manuscript body (no section fragments)
  macros.tex      single binding point for every measured number
  references.bib  bibliography database (52 entries)
  figures/        figure PDFs copied in for portability
  README.md       this file
```

`main.tex` holds the whole body in one file; there is no `sections/` directory.
References live in `references.bib` and are formatted by `IEEEtran.bst`
(`\bibliographystyle{IEEEtran}` + `\bibliography{references}`). The inline
`thebibliography` environment that used to sit at the end of `main.tex` is gone.

The one deliberate deviation from strict self-containment is
`\input{macros}`. Every measured number in the manuscript is bound once, as a
macro, in `macros.tex`, with its artifact of origin recorded in a comment
beside it. The point is that a number can only be wrong in one place, and that
re-running an experiment is a one-line edit rather than a hunt through prose.
Before submission, inline `macros.tex` into `main.tex` if the venue requires a
single source file.

## Build

```
cd manuscript
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The `bibtex` pass plus the two following `pdflatex` passes are required so that
`\cite`, `\eqref`, `\ref` and the `hyperref` anchors all resolve. Skipping
`bibtex` leaves every citation as `[?]`. The document class is `IEEEtran` with
the `journal` option.

**Soft constraint:** 15 pages nominal. Figures take priority over the cap and
the author has authorised overrun; prose is trimmed during final polish. Check
the page count after every writing session:

```
pdfinfo main.pdf | findstr Pages
```

## Local conventions in `main.tex`

- `\modelpart{...}` gives the five parts of the System Model subsection their
  own displayed, **numbered** headings (`1)` ... `5)`). IEEEtran renders
  `\subsubsection` as an unnumbered run-in heading, which would hide that
  structure; the custom counter avoids patching IEEEtran internals. The
  numbering is deliberately arabic, not alphabetic: `\subsection{System Model
  and Measurement}` already prints as "A.", so a lettered `\modelpart` produced
  a confusing "A. System Model ..." immediately followed by "A. Array and ...".
  Arabic numerals match IEEEtran's own `\subsubsection` convention anyway.
- `\Tr` is the ordinary transpose and `\Herm` the conjugate transpose. The
  distinction is load-bearing: probes are applied as `w^T u`, never `w^H u`.
- `\graphicspath{{figures/}}`, so `\includegraphics{phase1_bias_variance}`
  resolves without a path prefix.
- Measured numbers are never typed as literals in the body. They are macros
  from `macros.tex` (`\crbMedOneJ`, `\kappaGrowth`, ...). If a number appears
  as a bare literal in `main.tex`, that is a defect.
- `\kappaRawOneJ` is bound but registered **do-not-quote as a result**: the
  raw Fisher condition number is a units artifact of the `ln sigma_n^2`
  column, not a statement about geometry. It appears only where the text is
  explaining why the scaled `kappa_corr` is reported instead.

## Figure provenance

Figures are produced by the evaluation scripts in this repository
and are copied here from `../evaluation_results/figures/`.
The copies in `figures/` are artifacts, not sources -- regenerate upstream and
re-copy rather than editing them.

All 12 figure PDFs used by the manuscript are present.

| File in `figures/` | Upstream subdirectory |
| --- | --- |
| `phase1_error_cdf.pdf` | `phase1/` |
| `phase1_jnr_efficiency.pdf` | `phase1/` |
| `phase1_spatial_map.pdf` | `phase1/` |
| `phase1_bias_variance.pdf` | `phase1/` |
| `phase2_resolution.pdf` | `phase2/` |
| `phase3_5_dynamic_range.pdf` | `phase3.5/` |
| `phase3_5_error_heatmap.pdf` | `phase3.5/` |
| `phase3j_summary.pdf` | `phase3j/` |
| `phase3j_conditioning_vs_dof.pdf` | `phase3j/` |
| `phase4_jsr_vs_error.pdf` | `phase4/` |
| `phase4_2_jsr_heatmap.pdf` | `phase4/` |
| `phase5_decomposition.pdf` | `phase5/` |

All paths are relative to
`evaluation_results/figures/`. Note that the source
directory is literally named `phase3.5` (with a dot) even though the files
inside it use `phase3_5` (with an underscore).

Re-copy after regenerating upstream:

```
powershell -NoProfile -Command "Get-ChildItem -Recurse '..\evaluation_results\figures' -Filter *.pdf | Copy-Item -Destination figures -Force"
```

## Figure placement (final; all kept figures are placed)

Decision of this revision: the six-composite "8-page figure diet" is
**abandoned**. Composites A and B were blocked for want of source art, and the
author's instruction was to use every useful figure rather than let composites
strand four of them. The page cap is now **soft** -- 15 pages nominal, figures
take priority, prose is trimmed later if needed.

Second decision of this revision: the interpretability section
("What the Network Learned") is **cut entirely** for want of room. With it go
`phase5_attention_heads` and `phase5_branch_a_readout`, which have been deleted
from `figures/`. The attention/Branch-A numbers in `PHASE5_V2.txt` are
therefore **not** cited anywhere in the manuscript; their macros remain bound in
`macros.tex` but unused, and should not be reintroduced without restoring the
section. `phase5_decomposition` survives, relocated into Sec. V-E, where it
carries the narrower and still-needed claim that the stages contribute
unequally.

| # | Figure | Float | Home | Label |
|---|--------|-------|------|-------|
| 1 | `phase3j_conditioning_vs_dof` | 1-col | III-F Conditioning and the Multi-Jammer Limit | `fig:cond` |
| 2 | `phase1_error_cdf` | 1-col | V-A Accuracy | `fig:cdf` |
| 3 | `phase1_jnr_efficiency` | 1-col | V-B Efficiency Is a Function of Geometry | `fig:eff` |
| 4 | `phase1_bias_variance` | 2-col `figure*`, full `\textwidth` | V-D Where the Bias Points | `fig:biasvar` |
| 5 | `phase1_spatial_map` | 2-col `figure*`, full `\textwidth` | V-D Where the Bias Points | `fig:spatialmap` |
| 6 | `phase5_decomposition` | 1-col | V-E Each Stage Earns Its Place, but Not Equally | `fig:decomp` |
| 7 | `phase3j_summary` | 1-col | VI-A How the Error Grows with the Jammer Count | `fig:multij` |
| 8 | `phase2_resolution` | 1-col | VI-B Resolving Two Jammers | `fig:res` |
| 9 | `phase3_5_dynamic_range` | 1-col | VII Dynamic Range | `fig:dynrange` |
| 10 | `phase3_5_error_heatmap` | 1-col | VII Dynamic Range | `fig:drmap` |
| 11 | `phase4_jsr_vs_error` | 1-col | VIII-A Single Jammer | `fig:jsrerr` |
| 12 | `phase4_2_jsr_heatmap` | 1-col | VIII-B Two and Three Jammers | `fig:jsrmap` |


`figures/` contains exactly 12 PDFs and `main.tex` contains exactly 12
`\includegraphics` calls. Those two counts must stay equal; if they diverge,
either a figure is orphaned or a float points at a file that no longer exists.

**No composites remain.** The last one (bias/variance beside the spatial map)
was split on author feedback: `phase1_spatial_map.pdf` is a 772 x 349 pt
panorama, so at `0.32\textwidth` it rendered barely an inch tall and was
illegible. Both panels now get their own 2-column `figure*` at full
`\textwidth`, which makes the map about 2.2x wider than before and matches its
native aspect ratio. They are too tall to share a page, so they float
separately; the prose cites them together.

Consequence: `\usepackage{subcaption}` and the `fig:biasvar:decomp` /
`fig:biasvar:map` sub-labels are gone. The package is still loaded but is now
unused -- harmless, and worth keeping only if a composite is reintroduced.

Rule of thumb learned here: check a figure's native aspect ratio with
`pdfinfo figures/<name>.pdf` before choosing a width. Wide panoramas must not
be placed in a narrow subfigure column.

## Sourcing rule for every number in the manuscript

Every quoted figure must be transcribable from an artifact file. Do not
restate a number from memory. The artifacts are stored in this repository:

| Shorthand in `macros.tex` | Path |
| --- | --- |
| `VAL` | `data/VALIDATION_v2.txt` |
| `CRBV` | `evaluation/CRLB_V2_VERIFICATION.txt` |
| `COND` | `evaluation_results/CONDITIONING_VS_DOF.txt` |
| `PH1` | `evaluation_results/PHASE1_V2.txt` |
| `PH1BV` | `evaluation_results/PHASE1_BIAS_VARIANCE_V2.txt` |

Note the split: `CRLB_V2_VERIFICATION.txt` lives in **`evaluation/`**, beside
the code that writes it, *not* in `evaluation_results/` with the other
artifacts. Earlier notes recorded this incorrectly.

Two numbers that look like the same measurement are not. `PH1`'s CRLB median
(`\crbPhaseOneMed`, 1,500 draws) and `COND`'s J=1 row
(`\crbMedOneJ`, 300 scenes) come from different experiments and must never be
conflated or averaged.

The same trap sits between `PH1` and `PH1BV`, and it is the easier one to fall
into because both report a well/ill conditioning split of `RMSE/CRB`:

- `PH1` is the **random-DoA** sweep (10,000 scenes, JNR 20--50 dB, median
  bound 5.020 deg). Its split is **1.93x / 0.70x** (`\effWell`, `\effIll`).
- `PH1BV` is the **fixed-DoA** experiment (1,008 held directions x 96 noise
  trials at JNR 35 dB, median bound 1.985 deg). Its split is
  **3.19x / 0.68x** (`\fdWellEff`, `\fdIllEff`).

These are different experiments with different designs, sample counts and
median bounds. Never quote one split as the other, and never average them.
Sec. V-C states this explicitly in the prose and presents the agreement as
qualitative corroboration only. `macros.tex` carries the same warning as a
comment above the `PH1BV` block.

Two standing prohibitions:

1. **No claim of beating the Cram\'er--Rao bound.** The estimator is biased;
   where its error falls below the unbiased bound this is reported as a bias
   diagnostic, never as an efficiency result.
2. **No comparison to, or mention of, `reports/analog_crpa_paper.tex`.** That
   document is unpublished and internal. It may be consulted for method and
   physics formulations only -- e.g. the projection-nulling weight and the
   incident-referenced JSR definition reused in the System Model -- and none of
   its own results, baselines or numbers may appear here.

## Status

| Item | State |
|------|-------|
| Abstract, keywords, title | written, macro-bound |
| I Introduction + System Model A-E | written, macro-bound |
| II Literature Review A-G | written, macro-bound, fully cited |
| III Identifiability and Performance Bounds A-F | written |
| IV Estimator Architecture A-C | written |
| V Single-Jammer Accuracy A-E | written (E = stage decomposition, new) |
| VI Multiple Jammers: Degradation and Resolvability A-B | written |
| VII Dynamic Range: the Dominant Failure Mode | written |
| VIII From Direction to Suppression A-B | written |
| IX Discussion and Limitations | written |
| X Conclusion | written |
| ~~What the Network Learned (interpretability)~~ | **cut for space; do not reinstate without restoring its two figures** |
| Bibliography | BibTeX; 52 entries in `references.bib`, 51 cited. No placeholder fields remain -- see below |
| Figures | 12 of 12 placed (no composites; all floats single-image) |

**Last build:** 18 pages (about 16 of body plus 2 of references). Zero errors,
zero undefined control sequences, zero undefined references, zero undefined
citations, zero overfull boxes, zero BibTeX warnings. Three benign
`Underfull \vbox ... while \output is active`, which is ordinary float-page
slack and not worth chasing.

### Section II provenance and bibliography metadata policy

Section II reuses and reshapes the literature review of an earlier unpublished
manuscript by the same author (`latest guess/IEEE_Journal_Paper_Template/`).
Reuse is legitimate -- that paper was never published -- but its prose carried
v1 numbers and modelling assumptions that contradict this paper. The following
were deliberately scrubbed and **must not be reintroduced**:

| Old claim | Why it was removed |
|-----------|--------------------|
| "sub-1 degree mode in estimation accuracy" | actual median arc error is `\netArcPfifty` |
| "only 14k parameters" | actual is `\modelParams` |
| "32 total null probes per inference" | this paper uses `\numProbes` per sector query |
| "N isotropic antenna elements" | contradicts Sec. III: the `\dObs`-vs-`\dObsIdeal` result *depends* on full-wave pattern diversity |
| error **increases** with angular separation | contradicts the PH2 artifact, where resolution improves with separation; also a v1 result, which is prohibited |
| "perfect detector", "at most one jammer per coarse sector" | this paper assumes only that the coarse sector index is supplied (Sec. IX) |
| bare `[13]` citation | broken reference in the old draft |
| bib key `Compton1979` | normalised to `compton1979` to match existing citations |

Bibliographic metadata that could not be verified offline is **omitted**, not
guessed and not marked with a placeholder. An earlier revision wrote `XXX` into
such fields, but `XXX` is not a comment -- BibTeX happily typeset it, so the
printed reference list showed literal `pages XXX--XXX`, which looks broken.
Omitting the field instead is safe: `IEEEtran.bst` simply formats the entry
without it. Confirm nothing placeholder-like survives with:

```
powershell -NoProfile -Command "Select-String -Path references.bib,main.bbl -Pattern 'XXX|VERIFY'"
```

Twelve field lines across ten entries were cleared this way. Seven were a plain
`pages` range (`liu2018doa`, `papageorgiou2021`, `ozanich2020`,
`shlezinger2023`, `elbir2019`, `huang2019`, `brieger2022`). The two special
cases were:

- `jaganathan2016` -- the venue block was entirely unverified because the work
  exists both as a book chapter in *Optical Compressive Imaging* and as
  arXiv:1510.07713. It is now cited as the arXiv preprint, which is the form
  that can be stated with confidence.
- `ferre2019` -- *Sensors* uses article numbers rather than page ranges. The
  unverified number (believed 4841) and its `VERIFY` note were both removed;
  the entry now prints as volume/number/year only.

Authors, titles, venues, volumes, numbers and years are filled in for every
entry, but these were populated offline from domain knowledge rather than from
a live database lookup. **A final pass against IEEE Xplore or DOI is still
required before submission**, both to confirm the existing fields and to
restore the page ranges that were deliberately dropped. The highest-risk
entries are `liu2018doa`, `papageorgiou2021`, `shlezinger2023` and `huang2019`.

**Bare-literal audit:** clean. Every measured number in `main.tex` is a macro
from `macros.tex`. The decimals that remain in the source are all structural or
analytically derived rather than measured -- element coordinates (`0.03`),
derived geometry (`42.4` mm, `189.7` mm), the definitional `10^{-3.5}`, the
derived `0.40` s scan time, the `0.05` bias threshold, the analytic `0.5`
midpoint-collapse reference, and `subfigure` width fractions. Re-run the audit
with:

```
powershell -NoProfile -Command "Select-String -Path main.tex -Pattern '\d+\.\d+'"
```

**Remaining work:** verify all bibliography entries against IEEE Xplore/DOI and
restore the omitted page ranges; then trim prose if
the page count has to come back toward 15. Sections II-A (sensor fusion) and
II-B (spectral mitigation) are the intended trim targets -- they are the most
tangential to the contribution and were deliberately written long, so cutting
is a matter of deletion rather than rewriting.

**Standing constraints (unchanged):** never claim to beat the Cramer-Rao bound;
transcribe every number from an artifact rather than from memory.
