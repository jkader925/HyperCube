# Changelog

All notable changes to HyperCube are recorded here. Versions follow
[Semantic Versioning](https://semver.org/) with `0.x` semantics: minor bumps may change
output formats, and every such change is called out under **Output format** below.

---

## [Unreleased]

---

## [v0.5.0] — 2026-09-15

Rest-frame fitting templates and headless batch mode, self-sky subtraction, and a single
definition of which spaxels get fitted. This release also ships everything that
accumulated after the `v0.4.0` tag: a wavelength-dependent instrument LSF, per-component
constraints, multi-extension FITS ingest, an S/N mask that measures the line rather than
the brightness, resolving power read from the cube, and UI scaling.

### Added — self-sky subtraction

`HyperCube_SelfSky.py` (Qt-free), `Cube: → Self-Sky…` in the GUI, `--selfsky` in
`hypercube_batch`, and 26 unit tests in `test_selfsky.py`.

The driver case is a KCWI cube with an under-subtracted airglow line at 6866 Å — 2.1 Å
(≈92 km/s) redward of [N II] 6548 at *z* = 0.048232 and **2.5× brighter than the real
line**, which the fitter picks up in essentially every spaxel.

The residual is **additive**: its excess over the local sidebands is flat at ≈0.008 while
the continuum beneath varies by 50×. A naive regression suggests `excess = 0.57·continuum
+ 0.00725`, but that slope comes entirely from the top 5% bin, which is the galaxy with
its real [N II] 6548 on top. Multiplicative artifacts are out of scope.

- **The statistic is the first line of defence, the mask the second.** Continuum-faint
  does not mean line-free: 12.7% of a faintest-60%-by-white-light pool still has
  Hα+[N II] at S/N > 3 (6.8% at S/N > 5). Bias at Hα as a fraction of the galaxy line
  peak is 0.63% for a mean against 0.37% for a median, and injecting contamination up to
  60% of the pool moves the median by 0.2%. `_STATISTICS` therefore contains **no mean**,
  and callers pass a *name* rather than a callable so one cannot be smuggled in.
- **Measure line contamination with per-spaxel channel-to-channel noise**, not the spatial
  scatter of the continuum map — the latter is ≈2× too lenient and hides exactly the faint
  outflow the mask exists to exclude.
- **Vetoes AND, never OR, and are meant to be stacked.** A veto built on one line cannot
  remove an outflow visible only in another. Four sources: faintest *N*% by white light,
  white light below a value, a channel map off the live **C** window minus locked X/V
  sidebands, and any fitted line or quality map. On the driver cube, adding an Hα+[N II]
  channel-map veto to the faintest-60% continuum veto removed **2270 spaxels the continuum
  veto had kept** (8035 → 5765).
- **A veto window overlapping the artifact is refused, not warned about.** A channel map
  at 6866 Å *is* a map of the sky residual, so vetoing on it biases the pool low and the
  bias is invisible in the output. The guard is opt-in and armed deliberately, because
  prefilling it from the live C window rejected the user's own *C-drag over a line → add
  channel-map veto* workflow.
- **Variance propagation is mandatory**, as var(median) = (π/2)·var(mean). Using the
  mean's σ²/N understates the added noise by 57% and over-weights the corrected region in
  a 1/σ-weighted fit.
- **Subtract only where the spaxel has data.** Off-detector voxels are exact zeros rather
  than NaN and the DRP `MASK` extension is often all-zero, so subtracting everywhere turns
  those zeros into −sky and creates real negative data.
- **An unusable pool blocks.** `SkyPoolTooSmall` rather than a noisy sky; per-column
  blocks below the floor fall back to global and are recorded in `fallback_cols` so a
  fallback never looks like a measurement.

Measured on the driver cube (faintest-60% white-light mask, 8035 spaxels):

| mode | faint-region artifact | galaxy excess | gradient span |
|---|---|---|---|
| before | 0.00772 | 0.01065 | 0.00251 |
| global | **−0.00028** | 0.00265 | 0.00251 |
| per-column | **−0.00001** | 0.00369 | **0.00028** |

The real [N II] 6548 survives at ≈0.003 in both modes; per-column additionally flattens
the ≈30% across-slice gradient by ≈9×, matching the DRP modelling sky per slice.

- The session stores the **mask**, not the cube — ≈13 kB against ≈150 MB — and storing the
  resolved boolean rather than the recipe means a fit-derived mask restores exactly
  without needing the fit back. The pristine cube lives in `SELFSKY_ORIG_*`, so a second
  Apply cannot double-subtract; there is a test for exactly that. A failed restore reverts
  rather than leaving a half-corrected cube.
- Provenance rides in the fit CSV's scale/units block in both writers and in the window
  title, so a fit from a corrected cube is distinguishable from one that is not.
- **The batch rebuilds the mask from a declared recipe, never a shipped array**, so a run
  is reproducible from its arguments and a stale `.npy` applied to the wrong cube cannot
  happen. Batch therefore supports the faintest-*N*% source only. `--selfsky-statistic` is
  restricted by argparse `choices` to the same no-mean list. GUI and batch produce
  **bit-identical** cubes and sky spectra in both modes.

> This is a workaround, not a fix. For KCWI/KCRM the root cause is upstream: `SubtractSky`
> scales the sky master by exposure time only and never fits the airglow amplitude. This
> module is for cubes that will not be re-reduced, and must not become the reason the
> reduction is never fixed.

### Changed — Mask Spaxels is tabbed, with a threshold per line

**Tab 1, *Map criterion*** — map, operator, value, plus a **`C-region (channel map)`**
source built from the live C window minus locked X/V sidebands, through the same Qt-free
`channel_map` the self-sky vetoes use, so the two cannot drift. It needs no fit, so the
dialog no longer refuses to open when nothing has been fitted — the old early return would
have hidden the one source that does not require a fit. The map is rebuilt on every access
rather than cached: the dialog is modeless and the user is expected to move the C window
while it is open.

**Tab 2, *S/N*** — every line in the model, **each with its own threshold**, one operator,
and an **any / all** combiner.

- One universal cut is wrong whenever lines differ in brightness, which is the normal
  case: [O III] 4959 is a third of 5007 by atomic physics, so a cut that keeps 4959 is far
  too lax for 5007. Measured on a graded synthetic pair, `all` at 15/5 keeps 146 spaxels
  against 31 for a universal 15/15 — **115 spaxels differ**, all of them strong in 5007
  where the 3× fainter 4959 simply cannot reach 15.
- **`any` + `>` with one shared threshold reproduces the old SNR buttons exactly** —
  verified spaxel-for-spaxel on a synthetic two-line cube: the old gate
  (`nanmax(per-line) ≥ thr`) and the new `any line, > thr` select the identical 54
  spaxels, 0 disagreements. `all` is a strict subset (6 spaxels) — the new capability, for
  when a line *ratio* must be measurable in the same spaxel.
- S/N is deliberately **not** offered in tab 1. Two routes to the same cut, one of which
  cannot express per-line thresholds, is how the old per-line SNR buttons became
  confusing.
- `compute_snr_map(..., per_line=True)` returns the per-line maps it already builds
  internally instead of their `nanmax` — a flag rather than a second function, so the
  combined and per-line maps can never disagree. Verified `max(per_line) == combined`.
- The dialog is modeless and floats above the app, so channel selection and reading fluxes
  off the map stay live while it is open.

### Changed — one definition of which spaxels get fitted

`FitParamsWindow.fit_gate()` returns an `(ny, nx)` boolean ANDing the S/N gate
(`snr_map >= snr_value`) with **everything hidden by Mask Spaxels**. All four call sites go
through it — the main cube fit, the Rectify re-fit, the stellar fit, and the Rectify count
preview, which must agree with the loop it predicts. No raw `snr_map[j, i] >= snr_value`
comparisons remain.

> ⚠️ **This changes documented behaviour.** The Mask help text used to read "It is a
> display mask — it does not change the fit"; masked spaxels are now excluded from cube
> fits.

Accept keeps the legacy S/N globals in step, so templates' `snr_threshold`, the batch
manifest column and the N-σ contour keep working now that this tab owns the gate. A
template can only express "any line, >", so a stricter criterion writes a permissive
`snr_value = 0` and leaves the display mask to carry the real rule. Unmask clears the S/N
gate too — leaving it set would keep the fit silently gated with nothing on screen to say
so.

- **Removed:** the `SNR` button column in the Spectral Region tabs. The `df['SNR']`
  dataframe column is deliberately **kept** — it is written by the CSV/session/template
  round-trip in four places, so removing it is a file-format change, not a UI change.
- The S/N column in the Spectral Region tab was *not* redundant with the new per-line mask
  and was only removed once the gate moved: it set the pre-fit threshold, computed and
  cached `snr_map`, drew the N-σ contour, and is persisted in templates.

### Changed — NED supplies measurements, never the label

**The user's supplied name is authoritative and is never overwritten by a resolver.**
Every NED path previously wrote NED's canonical identifier back into the name field,
`df_obs['sourcename']` and the Source button — so typing `F01364-1042` could relabel the
target `2MASX J01385289-1027113`, silently breaking every join the user has against their
own catalogues, file names, target directories and manifests.

All five paths (two dialogs × {by name, by coords} plus the Qt-free `resolve_redshift`)
now use `label = user_supplied_name or ned_name`. A non-empty user name always wins; NED's
name is adopted only when the user gave none, which is the resolve-by-coordinates case
where there is no user name to protect. What *is* taken from the resolver is the redshift
and coordinates; what is *shown* is `z = 0.048232  (NED matched: <ned name>)`, so the
resolution can be confirmed without the identity being swapped.
`HyperCube_Templates.resolve_redshift` now returns `(name_as_supplied, ned_name, z)` for
the same reason — a manifest draft records what matched as evidence, beside the name the
user chose.

This matters beyond cosmetics: MUSE cubes carry `OBJECT = 2MASX J07273754-0254540` rather
than the IRAS ID, which is exactly why a manifest exists. A resolver that rewrites names
recreates the problem the manifest was built to solve.


### FIXED — applying a template raised the sigma floor to the instrumental width

**Yes, sigma bounds are converted per cube** — every bound goes through the LSF
quadrature at its own line's observed wavelength. The bug was in what a *zero*
floor meant on the way back.

When a model's sigma lower bound sits below the instrumental width,
`template_from_model` records it as intrinsic **0**, which is correct: as an
intrinsic statement it is "no constraint". But `expand()` then put that 0 back
through `intrinsic_to_observed`, which returns the LSF width — so "no
constraint" came back as *a constraint at the instrument's own width*. On MUSE
that silently raised a 20 km/s floor to about **71 km/s at Hβ** (46 km/s at
[S II]), and any line genuinely narrower than that could not be fitted: the
optimizer sat pinned against the bound.

A zero intrinsic floor now expands to no observed constraint. A **stated**
intrinsic floor is still broadened per line and per cube — an 80 km/s intrinsic
floor becomes 109.4 km/s observed at Hβ and 107.1 at [O III] 5007, correctly
different at each wavelength.

The asymmetry is deliberate: for a lower bound, "unconstrained" and "at least as
wide as the instrument" are different statements, and only the first is what a
missing constraint means.

### FIXED — a template-applied model started with its continuum at zero

`expand()` set `Slope_0 = Intercept_0 = 0` for every region, because the
template format has no columns for a linear continuum's level. So applying a
template to a new galaxy produced a model whose continuum sat at zero while the
data sat at hundreds of flux units — the initial guess visibly below the
spectrum even with every line at the right wavelength, and a poor starting point
for lmfit.

The design error was treating the continuum as part of the model's *shape*. It
is not: **the continuum level is galaxy-specific**, like the amplitude scale, and
the spec's own rule is that galaxy-specific things do not live in a template.
Storing the first galaxy's intercept would have been just as wrong — BUNIT and
the flux scale differ between cubes.

- **The continuum is now measured, not stored.** `seed_linear_continuum` fits a
  robust straight line to the reference spaxel's own spectrum inside each
  region, clipping emission asymmetrically so the lines the model exists to fit
  do not drag the baseline up with them. On NGC 6240 the initial guess moves
  from **516/1021/1465 flux units below the data to 26/218/92** (the middle
  region is the Na I D trough, where a continuum legitimately sits below the
  median).
- **The amplitude scale is measured above that continuum.** It was the peak of
  the raw spectrum, so every amplitude guess carried the continuum level with
  it — which, paired with a zero continuum, was wrong in both directions at once.
- A dry run reads no pixels, so it keeps the zero continuum and says so.

**Measured effect on the fit** (NGC 6240, single-component template): the
calibrated continuum χ² goes from **7.21 to 1.93** — 1.0 is ideal. The core
residual rises, which is the point: a continuum fitted 25 flux units too low let
the Gaussians absorb the deficit and score better in χ² while being physically
wrong. The metric now reports honestly that a single Gaussian does not describe
NGC 6240's line profiles.

### FIXED — `slope` was never divided by the flux scale

`fit_one_spaxel` rescales the spectrum by its 95th percentile and divides the
flux-unit parameters to match — `amp`, `intercept`, `knoty`, `polyc` — but not
`slope`, while the *output* multiplies the fitted slope back by `flux_scale`.
The asymmetry was invisible for as long as every slope started at zero
(`0/scale == 0`). Seeding a real continuum exposed it immediately: the seeded
slope entered as though already scaled, making the model continuum steeper by
`flux_scale` and driving `rchisq_w` into the millions. `slope*` (and the
inter-region `slope_int*`) now scale in and out symmetrically. No existing
workflow changes — a cube fit is byte-identical, because a slope of zero scales
to zero.

### FIXED — Return in the line-name box opened the help window, and submitted twice

Two separate faults on the same keystroke.

- **Return activated a button as well as the text box.** Qt makes every
  `QPushButton` in a dialog an `autoDefault` button, so Return fired the first
  one in tab order — the constraint-syntax `?` — and its help window opened on
  top of the rename. Three buttons in that dialog had never been given the flag
  that `Submit Constraints` carries. Rather than adding it to each and relying
  on the next button to remember, the whole dialog is cleared before it opens;
  nothing in it wants to be the default.
- **`returnPressed` was connected to the same field twice**, once where the box
  is created and again just before the dialog opened, with identical lambdas —
  so one Return ran the rename twice. The second pass tried to remove a
  matplotlib actor the first had already removed. The duplicate is gone.

### Added — a show/hide toggle for the S/N mask contour

- **An `S/N mask` toggle on the image toolbar**, beside Reset. It is checkable
  because it is a *display state*, not a one-shot draw: the red contour was
  previously drawn once by "Calculate S/N map" and silently lost on the next
  redraw, which is the same latent bug the selection box had — `draw_image`
  clears the figure and rebuilds its overlays from state, and the contour was
  not among them.
- **It survives everything that rebuilds the map.** Verified against colormap,
  stretch/scale, rotate, flip horizontal, flip vertical, spatial-mask redraw and
  parameter-map redraw. The contour is drawn through the same rotate/flip as the
  image, so it does not sit at the original orientation over a transformed field.
- **Loading a template computes the mask and shows it** — the template just
  changed which spaxels the gate selects, and a mask you cannot see is one you
  will forget is set.
- The choice is saved in `.hcsession` and restored with the button in the
  matching state.

### FIXED — rotating then flipping the map crashed the viewer

`draw_image` updated `_last_from_fits` unconditionally but `_last_data` only on
a real data load, so after any transform redraw the flag said "2-D" while the
cached array was still the 3-D cube. The next transform handed that cube to
`imshow`: **`TypeError: Invalid shape (nchan, ny, nx) for image data`**. Any two
consecutive viewport transforms did it — rotate then flip, flip then rotate. The
flag now moves with the array it describes. Found while checking that the S/N
contour survives a flip; it turned out the flip did not.

### Added — Save Template / Load Template in the GUI

Two buttons on the **Output:** toolbar row, beside Save/Load Session. This is
the half of the template feature that was missing: the machinery existed and was
tested, but nothing in the GUI could reach it.

- **Save Template** writes the current model as a rest-frame `.hct.csv` —
  rest wavelengths, velocity offsets from systemic, *intrinsic* dispersions,
  bounds, K-groups, constraints, continuum regions, and the S/N threshold. It
  refuses without a redshift, since that is what makes the conversion out of the
  observed frame possible, and it reports how many σ bounds sat below the
  instrumental width (those are "no constraint" as intrinsic values and are
  stored as 0).
- **Load Template** applies one to whatever cube is open: lines placed at that
  galaxy's redshift, dispersions broadened by *that* instrument's LSF at each
  line's own wavelength, amplitudes scaled to the cube's flux units. It reports
  what it dropped for coverage and any K-group it had to re-anchor, rather than
  doing it silently.
- **The template's S/N threshold is now applied on load**, not merely stored —
  the map is recomputed for the new model and the gate set.
- The cube's own source name, redshift and resolving power are **kept**, not
  overwritten: R was derived from the cube, and a template has no business
  carrying another galaxy's identity.
- Loading a template clears the locked per-spaxel schema and any overrides,
  which described the previous model.

Verified end to end across two different real MUSE cubes: a model built on
IRAS 07251-0248 (z=0.0876) saved and applied to NGC 6240 (z=0.0243), all 9 lines
and both regions landing correctly, velocity offsets preserved, S/N gate carried
across.

### Added — `hypercube_batch`, a headless runner (Stage 5)

`python -m hypercube_batch --template … --manifest … --out …` expands a
rest-frame template across a manifest of cubes and fits each one with no
`QApplication`, writing the five-section CSV the GUI's **Load Fit (CSV)** reads.
README's "Pipeline Usage Mode", empty since it was written, now documents it.

- **Verified equal to the GUI.** Same template, same cube, same gate: 2466 rows
  agree **bit-for-bit** on `amp_fit`, `cen_fit`, `vel_fit`, `amp_std`,
  `cen_std`, `rchisq_w` and `BIC`. σ agrees to 4.4e-16 — one ulp, the
  irreducible round-off from σ being stored in Å and exported in km/s. The
  runner is also bit-reproducible run to run.
- **Verified reopenable.** A headless product loads through `load_cube_fit` and
  renders an Hβ velocity map over all 822 fitted spaxels.
- **`--dry-run` reads headers only.** It expands and reports coverage without
  touching pixel data, which is what makes it usable on cubes that live on an
  external drive: all 22 MAUNA MUSE targets in **17 seconds** rather than the
  hours a full read would take.
- Progress goes to stdout at 5% granularity; a failing target is logged and the
  run continues, exiting non-zero at the end. `run_log.csv` records the
  template, its version, the git SHA and per-target timings.
- **FIXED before it shipped — the runner was downcasting cubes to float32.**
  KCWI coadds are `BITPIX -64`, so every spectrum reached the fitter at half the
  GUI's precision; it showed up as ~1e-3 disagreement in `sigma_fit` against the
  same model. The native dtype is now preserved.

### Changed — one parameter builder and one pool driver, not three

Stage 4 of `HyperCube_Templates_SPEC.md`. The lmfit parameter builder was
written out twice — inline in `fit_cube` and again in `_fit_single_spaxel_impl`
— and the pool driver was inline in `_fit_cube_parallel`, all inside Qt methods
and so unreachable from a batch run. Both are now free functions in the Qt-free
kernel `HyperCube_fit.py`.

- **`build_params(df, df_cont, z, …)`** replaces ~200 lines of duplicated inline
  code across the two GUI paths. The two constraint translators are injected
  rather than imported, so the kernel stays free of Qt.
- **`run_pool(…, progress_cb, is_cancelled)`** replaces the inline pool driver.
  The only Qt in it was the progress bar and the cancel flag, which are now a
  callback and a predicate.
- **Verified byte-identical.** A 36-spaxel cube fit produces a `df_fit` that is
  byte-for-byte the same before and after both extractions, and the serial
  fallback agrees with the pool to 0.0.

- **FIXED — a cube fit gave stellar continuum regions a free linear continuum.**
  The single-spaxel path zeroed and froze `slope`/`intercept` for
  `cont_type == 'stellar'` (the stellar baseline is already subtracted before
  the line fit); the cube path never did. The duplicated builders had drifted,
  so **the same spaxel fitted by hand and fitted as part of a cube was not
  fitting the same model.** On the MAUNA MUSE models, where both regions are
  stellar, that is **4 spurious free parameters** — two per region — sitting
  degenerate with the line amplitudes, free to trade flux with the lines
  per spaxel with nothing recording it. Their starting values were already 0
  (`Slope_0`/`Intercept_0` are NaN for stellar rows); it was `vary=True` that
  did the damage. Sharing one builder makes the drift impossible to repeat.
  **This changes cube-fit results for stellar-continuum models** — any such fit
  produced before this build should be re-run.
- Removed on the way: the dead `df_cont_sort`/`df_sort` in `fit_cube`, and the
  line-parameter loop that re-added every line once per continuum region.

### Added — rest-frame fitting templates (Stages 0–3 of `HyperCube_Templates_SPEC.md`)

The MAUNA MUSE sub-sample is 22 galaxies × 11 line groups × {1,2} components =
484 fits, and building each model by hand is the bottleneck. A **template**
describes a model the way physics does — rest wavelengths, velocity offsets,
intrinsic dispersions — so one file applies to every galaxy; a **manifest**
carries the per-galaxy facts; **expansion** combines them.

- **`HyperCube_LSF.py` (new, Qt-free)** — instrument line-spread functions as
  FWHM(λ), for MUSE, KCWI/KCRM, MIRI MRS and NIRSpec, with
  `intrinsic_to_observed` / `observed_to_intrinsic`. R is not a constant: it
  runs **1873 at Hβ to 2978 at [S II]** across one MAUNA model, so a single R
  mis-states the instrumental width differently at each line being compared.
  `resolving_power_from_header` now delegates here — output is byte-identical
  for every cube in the repo, and a MUSE cube derives R ≈ 2831 with named
  provenance instead of the hand-typed 3000 in the existing session.
  - **The MUSE curve is a documented fork, not a silent choice.** The measured
    (Bacon+2017 sky-line) and nominal (ESO endpoint) parameterisations disagree
    by ~9% at 4800 Å, agree in the red, and **cross near 6000 Å** — so they
    differ at Hα too. Both ship, the default is the measured one, and the
    provenance string records which was used.
  - An unresolved line yields `NaN`, never 0: "unresolved" and "resolved and
    narrow" mean different things and must not average together.
- **`HyperCube_Templates.py` (new, Qt-free)** — the template/manifest format
  (sectioned CSV, the existing blank-row convention), `template_from_model`,
  `expand`, and the manifest + NED bootstrap. A file is refused as a template
  unless it declares `wavelength frame: rest` and `sigma convention:
  intrinsic`, so an observed-frame fit CSV cannot be misread as one.
- **Templates state intrinsic σ; expansion adds the LSF in quadrature** at each
  line's own observed wavelength. Fit output is unchanged — `sigma_fit` remains
  the observed width — so `agnpipe` and existing scripts are untouched.
- **Coverage dropping is audited, not silent.** Lines and regions outside a
  cube's range are dropped, `Line_ID` renumbered, dangling constraints removed
  by name, and K-groups re-anchored or cleared — each with a reason in an
  `ExpansionReport`. Running expansion without fitting produces the coverage
  map, i.e. the N/A grid for the tracking sheet.
- **K-group ties are regenerated after repair.** Dropping a line that other
  lines were tied to removes those ties; without regeneration a repaired model
  would keep its K-group labels while quietly fitting every member
  independently. Ties are rebuilt against the surviving anchor with σ ratios
  recomputed from its rest wavelength.
- Round trip verified against the real 07251 MUSE session: centroids, σ,
  amplitudes, bounds, K-groups, constraints and region edges all recover to
  machine precision. The one deliberate change is that σ floors below the
  instrumental width — 20 km/s observed, at every line — normalise to the LSF
  floor, which correctly now **varies per line** (68.0 km/s at Hβ, 42.8 at
  [S II]) instead of being one meaningless number.

### Fixed — the continuum row no longer bunches up on the left

- **A spectral region's continuum buttons now fill the frame's width.** The
  continuum row shares one `QGridLayout` with the line rows below it, which are
  17 columns wide and carry equal stretch on every column — but a linear
  continuum has only 8 cells, so it was laid out across the leftmost 8 of those
  17 and squeezed into **43%** of the frame while the line rows spread over all
  of it. Each cell now spans its share of the spare columns, taking the row to
  99%, with the remainder going to the leftmost cells since those carry the
  longest text. The header labels reuse the button row's spans, so each stays
  over the cell it names.
- The same squeeze applied to every continuum type — spline and polynomial were
  narrower still at 7 cells — so all four are spread by one shared helper rather
  than the reported case alone. The `x` delete button is deliberately left at a
  single column instead of being stretched to match its neighbours.

### Added — choose which line anchors a K-group

- **"Make this line the reference for its K-group"**, in the Line Name window.
  A K-group's reference is the member whose velocity and dispersion stay free
  while the rest are tied to it, and it was always simply the first member in
  model order — so a group spanning Hγ and Hα anchored on Hγ, the *worse*
  measured of the two, with no way to say otherwise short of reordering the
  model. The button moves the anchor to the line whose constraints you have
  open.
- **The ties are rewritten immediately**, with each σ ratio rescaled to the new
  anchor's rest wavelength (promoting Hα in an Hγ/Hβ/Hα group turns
  `sigma == 1.119986 * sigma_[H_gamma]` into
  `sigma == 0.740736 * sigma_[H_alpha]`), and the new anchor's own ties are
  dropped so it is genuinely free. The status line and the button's own enabled
  state update in place.
- **A promotion belongs to the group it was made in.** Moving a line to another
  group — or out of grouping altogether — drops it rather than carrying an
  anchor role along, and re-running Smart Constraints resets every anchor,
  since it rebuilds the grouping from scratch and its new groups never had one
  chosen for them. Re-submitting the *same* group keeps it.
- Stored as a `kgroup_ref` column beside `kgroup`, so it rides sessions with no
  new plumbing; a group with no promotion — including one restored from a
  session predating the column — anchors on the first member exactly as before.

### Changed — brightness and contrast swap drag axes in the map viewer

- **Left/right now sets brightness and up/down sets contrast**, the other way
  round from before. The handler's docstring had claimed this mapping all along
  while the code did the opposite, so the two now agree.

### Changed — the S/N mask measures the line, not the brightness

- **The S/N map now subtracts the local continuum** before dividing by the noise.
  It did not, so the ratio was (continuum + line)/noise — a brightness cut wearing
  a line cut's name. Any bright continuum source passed at any threshold: the
  foreground star in UGC 05101 scored **S/N = 43** on a spectrum with no Hα in it
  at all, and duly landed inside every `S/N_Hα > 5` mask. It now scores 0.95.
- **The continuum flanks are no longer assumed to be line-free**, because they
  routinely are not — at z ≈ 0.04 both [N II] lines sit inside Hα's default
  flanks. The level is taken after two asymmetric clipping passes that drop
  channels more than 3σ *above* the running median, which removes line cores
  while leaving the noise distribution, and any absorption, untouched. A plain
  median would be pulled up by exactly the lines being measured and that bias
  subtracted straight off the line. Clipping also repairs the *noise* estimate
  where the flanks are bright: the UGC 05101 nucleus rises from 354 to 623.
- **This lowers every S/N**, so a threshold carried over from a previous version
  is now a stricter cut than it was — re-check it against the contour. On
  UGC 05101 at Hα, `> 5` goes from 3114 to 2016 spaxels; most of what leaves is
  marginal 3–5σ emission in the faint outskirts, not stars (their median
  continuum is 0.6× that of the spaxels that stay).
- **Maps built under the old definition cannot survive into the new one.** The
  formula version is part of the cache key and is written into sessions; a
  session carrying an older map reports it and falls back to the open gate the
  cube load installs, rather than gating a fit on numbers this build no longer
  produces.

### Added — resolving power is read from the cube

- **R is derived at ingest** instead of being left blank for the user to
  remember. It matters: R convolves the stellar templates before pPXF fits a
  LOSVD, so a wrong R biases σ\*, and the old code fell back to R = 3000 without
  saying so. Three tiers are tried — a header keyword stating R (or a resolution
  element); then the named instrument (KCWI/KCRM grating + slicer, MIRI MRS
  channel + band, NIRSpec disperser, the MUSE LSF polynomial); then nothing.
- **KCWI is modelled by its resolution element, not by R.** A grating's FWHM is
  fixed; R = λ/FWHM slides across the band. This repo's own coadds show it:
  `SPECRES` is 1800 @ 6150 Å for RH1 and 2025 @ 6900 Å for RH2 — one FWHM (3.42
  vs 3.41 Å) at two wavelengths, not two resolving powers. Entries for BH1–3,
  RH1 and RH2 are calibrated against those headers; the rest are nominal and are
  labelled as such wherever they are used. Cross-check: `UGC05101_coadd_RH2L`,
  which carries no `SPECRES`, derives R = 2022 against the 2025 that
  `Mrk273_coadd_RH2L` states for the same configuration.
- **A KCWI header names both gratings whatever arm the cube holds**, so the arm
  is chosen by comparing the cube's own coverage against `BCWAVE`/`RCWAVE`
  rather than by trusting that only one is present.
- **A coadded supercube gets no R, deliberately.** It keeps no instrument
  keywords and mixes gratings of different resolution, so no single R describes
  it. The combine step's `HISTORY` is read instead and the constituent gratings
  and their resolution elements printed (`RH2 3.41 Å, RH1 3.42 Å, BH1 1.15 Å,
  BL 5.00 Å`), leaving R to be set for the region actually being fitted.
- **The R button's tooltip attributes the value**, so a derived number is never
  mistaken for one that was typed; editing R by hand clears the attribution.
- **pPXF's R = 3000 fallback announces itself** (once per cube) instead of
  silently turning an unknown into an unmarked assumption.

### Added — opening a cube tells you it is working

- **A green progress bar reports the load**, in the status-bar slot the filename
  will occupy — the label is hidden while the bar runs, so the row does not shift
  as the two swap. Reading a cube takes long enough to look like a hang (worse
  since the loader holds the GUI thread), and until now the only sign of life was
  `No file loaded` sitting there. The left of the status bar names the stage:
  *Opening file*, *Merging the primary header* (the slow step on MUSE, whose
  primary carries thousands of `HIERARCH ESO` cards), *Collapsing the cube*,
  *Reading the WCS*, *Looking for measurement errors*.
- **The bar is cleared however the load ends.** The stages live in
  `_load_fits_from_path_impl`; `_load_fits_from_path` is now a wrapper that owns
  only the bar, in a `try/finally`, so a cancelled extension picker, a binary
  table (which returns early into its column dialog) or an outright exception all
  restore the filename label rather than leaving a bar stuck at 40%.
- **A second load cannot start while one is running.** The loader is on the GUI
  thread and the bar only advances because we pump the event loop by hand, so the
  pump excludes user input and a `_loading_file` flag refuses re-entry — a
  double-click on Open, or a second drop, would otherwise reach into the module
  globals mid-rewrite.

### Added — drag and drop a FITS file onto the window

- **Dropping a `.fits` anywhere on the viewer loads it**, by the same path as
  `Open FITS` (`.fit`, `.fts` and the `.gz`/`.fz`/`.bz2` forms too). The drag is
  only accepted if it actually carries a local, existing FITS file, so a dropped
  `.txt` is refused by the cursor rather than by an error afterwards; with several
  files in the drag the first FITS one wins. While the drag hovers, the status bar
  reads `Drop to load <name>`.
- **Anywhere means anywhere.** Qt propagates a drag up to the first ancestor that
  accepts drops, so the canvases and toolbars are covered for free — but a
  `QLineEdit` accepts drops itself and would have pasted the path into a scale
  factor. Nothing in this window wants a text drop, so its whole widget tree now
  defers to the window.
- The drop is loaded from a zero-delay timer, letting the source application
  release the drag before the loader takes the GUI thread for the length of a
  cube read.

### Added — colorbars, and maps that keep up

- **The map overlay reads out the value under the cursor**, as a compact `z: 0.892` line —
  the colorbar carries the field's name and units, so repeating them in the overlay only
  cost space. Previously the value appeared only for quality maps and only while hovering;
  a parameter map never reported one at all. The quoted value is the **physical** one, read
  before the transfer function and clipping — a log-stretched or clipped number is not the
  measurement — and it shows as `z: —` before the cursor is over a spaxel so the overlay
  does not change height as the pointer moves.
- **Every map now carries a labelled colorbar** — a map of velocities or χ² was unreadable
  without one. It is built from the same artist as the image, so it always reflects the
  clipping and stretch actually applied, and it is labelled with the quantity and its unit
  (`Hα velocity [km/s]`, `Core / continuum ratio`, `flux [BUNIT]`).
- **FIXED — the selected-spaxel box no longer vanishes on redraw.** `draw_image` clears the
  figure and rebuilds its overlays; the blue init-guess marker was restored from state but
  the red selection box was recreated hidden, so *any* redraw silently lost it. Now
  restored from `current_spaxel`, keeping its locked/unlocked colour. This was always
  latent — making maps refresh automatically is what made it constant.
- **"Fit This Spaxel" and "Clear Spaxel Fit" refresh the displayed map**, so a map cannot
  keep showing a spaxel's old value (or a fit that no longer exists).
- **The Rectify dialog reopens with the settings you last used** — criteria, operators,
  limits, seeds/radius/passes and both checkboxes — remembered on Cancel as well as OK,
  since closing the window is not a reason to discard limits you just dialled in.

### Added — choose whether the rules combine as OR or AND

- **A selector at the top of the Rectify dialog** decides how the ticked rules combine when
  marking a spaxel for repair: **any** (one tripped rule is enough — the previous behaviour
  and the default) or **all** (every ticked rule must trip, narrowing the selection to
  spaxels that are unambiguously broken on every count). The live count and the rule text
  update accordingly (`… > 2 OR … |·| > 3` / `… AND …`).
- **It deliberately does not touch the donor gate.** A spaxel must still pass *every*
  ticked rule to seed a repair or be accepted as an improvement: loosening what gets
  repaired must not loosen what does the repairing. Under **all** this leaves a middle
  band — failing some rules but not all — that is neither repaired nor used as a seed,
  which is the intended reading of "only repair the thoroughly broken ones, but still only
  trust the clean ones as seeds".
- **FIXED — the repair loop was using the wrong goodness test.** It admitted a spaxel to
  the donor pool when its *mean* score fell below 1, while `is_good` requires *every*
  metric below 1. A spaxel failing one metric badly but averaging well could therefore seed
  its neighbours from a fit the criteria reject. Both now use the same AND-gate.

### Changed — the displayed map now tracks the fit

- **A cube fit and every Rectify pass re-render whichever fit map is on screen.** The
  display was a snapshot of whatever `df_fit` held when the map was last clicked, so a
  Rectify run finished with the map still showing its pre-repair state. Both map paths
  (the Quality Map menu and the fitted-parameter maps) now record how to re-render
  themselves, and Rectify redraws after each pass — so a repair front is visible spreading
  across the cube rather than appearing all at once at the end.

### Changed — Rectify's dialog is one table

- **The separate map/operator/threshold row is gone**; the criteria table now carries a
  per-metric operator (`>`, `<`, `|·| >`, `|·| <`) beside its limit, so each number means
  what it says — z-scores default to `|·| >`, ratios to `>`. The spaxels that fail the
  rules are exactly the spaxels re-fit, so flagging and the donor gate cannot disagree.
- The **live count** (`N of M spaxels (rule) will be re-fit`) is back, updating on every
  change to any row, operator, limit or checkbox.
- **"Also re-fit failed (non-finite) spaxels" now does something.** A failed fit trips
  every rule automatically, so the switch works by *exclusion*: untick it to leave those
  spaxels alone rather than spending fits on them. The count says which way it went.
- **Max search radius** is no longer capped at 8.
- **The `?` help opens in a scrollable window** capped at 70% of screen height, instead of
  laying itself out taller than the display and cutting off the bottom. The Smart
  Constraints help, which had the same problem, uses it too.

### Changed — Rectify seeds every refit from quality-gated neighbours

`Rectify Bad Fits` re-fits a bad spaxel from a neighbour's solution. Its spatial prior was
much weaker than it looked, and two of its behaviours were wrong.

- **"Good" is now a quality judgement.** The donor pool was `everything the user's filter
  did not flag` — a neighbour with a terrible core/continuum ratio was an eligible seed as
  long as the flagging criterion happened to miss it, and the quality metric only broke
  ties within that uncurated pool. A donor must now pass the goodness criteria outright.
- **Several metrics, not one.** Donor ranking and acceptance were hardcoded to
  `qa_core_cont_ratio` regardless of what the user flagged on. A criteria table now takes
  any combination of the calibrated quality metrics, each with its own limit. A fit is good
  only if it passes **every** ticked criterion (an AND-gate — a severe failure in one metric
  cannot be outvoted by several healthy ones), while a composite score ranks fits against
  each other. Both come from one normalised quantity: `n = |v − ideal| / |limit − ideal|`,
  so `n ≤ 1` *is* "passes", and the score is the mean of the `n`s. Defaults leave only the
  core/continuum ratio ticked at 2.0, reproducing the previous rule exactly.
- **Every good neighbour is a candidate seed, not just the best one.** The best *K*
  (default 3) are each tried as an independent starting point and the best result kept.
  The search stops at the first seed that lands a good fit, so on a synthetic cube K=3 still
  cost exactly **one fit per spaxel** — raising K only spends time on genuinely hard spaxels.
- **The search grows outward** — radius 1, then 2, then 3 (configurable). Previously a
  spaxel with no good immediate neighbour fell back to the generic base template, so the
  interior of any bad patch larger than ~3×3 received **no spatial information at all**,
  which is exactly where it is most needed. Nearer donors are preferred over better-scoring
  distant ones.
- **FIXED — repairs now propagate.** The good set was snapshotted before the loop, so a
  spaxel repaired at one moment was still "bad" to its neighbour a moment later; repairs
  stopped at the rim of a bad patch and never flooded inward. Rescued spaxels now join the
  good set immediately, spaxels are processed most-good-neighbours-first, and passes repeat
  until one rescues nothing. On a 5×5 bad patch the old frozen radius-1 search could reach
  only its 16 rim spaxels; the new pass repairs all 25.
- **FIXED — Rectify can no longer make a spaxel worse.** README claimed this; the code did
  not do it. The pre-Rectify fit was never a candidate, so a flagged spaxel was *always*
  overwritten with the best of {neighbour-seed, 4 restarts} even when all of them were
  worse than the fit it already had. The incumbent now competes as candidate zero (scored
  from the existing row, costing no re-fit), and only a strictly better result is committed.
  This is also what makes iterating safe: scores are monotone, so passes cannot oscillate.
- **New provenance columns** `rectify_pass`, `rectify_seed` (`neighbour(12,34)@r2`,
  `restart:swapped`, `incumbent`, `base`) and `rectify_score`, written into the same row
  dict the fit kernel returns so they reach CSV/FITS with no new plumbing. A rectified
  spaxel was previously indistinguishable from an ordinary fit, which made the pass
  impossible to audit — and the Context-is-Key design designates Rectify as the baseline
  against which its learned patch mode will be validated.
- The goodness rule lives in a new Qt-free module, **`HyperCube_Quality.py`**, mirroring
  `HyperCube_Noise.py`, so the GUI, the batch kernel and (later) CiK share one definition
  of "good fit".

### Added — blue/red side constraints for broad components

- **Smart Constraints can now hold each broad component on one side of its core.**
  A line fitted as core + two broad components (Hα + Hα_b + Hα_c) gets one constrained
  blueward and one redward, with the pairing read from the initial centroid guesses;
  seeding the two wings apart is what says which is which. A line with only *one* broad
  component has no pair to read a side from, so the dialog asks: it may sit either side
  (the previous behaviour), or it is blueshifted, or redshifted. The **AGN Outflow**
  preset now defaults to blueshifted — the case the preset is named for — and **Shock /
  LINER** splits pairs but leaves a lone component free.
- **Why it matters:** with both wings free, the fit cannot distinguish "blue wing at
  −400 km/s, red at +500" from the same solution with the labels swapped — identical χ².
  Per-spaxel, that means the blue-wing map and the red-wing map trade places wherever the
  minimiser happens to land. Verified on a synthetic Hα: from deliberately swapped seeds
  the unconstrained fit converges with the labels crossed and the sided fit does not, at
  the same reduced χ².
- **Blue and red wings are no longer forced into the same K-group.** They were both
  "secondary" and both landed in K2, which materialises `vel == vel_[ref]` — one shared
  velocity, i.e. exactly the collapse the two components exist to resolve. Wings are now
  keyed by side and ionization across K2–K5. With the new options off, the K-group
  labelling is byte-for-byte what it was.
### Added — per-component-tier bound table in Smart Constraints

- **A compact table sets the bounds for every component of a tier at once** — one row per
  tier present in the model (primary = narrowest of each rest-wavelength group, then
  secondary, tertiary, …), three editable columns:
  - **Velocity ± km/s** — for non-primary rows, how far that component may sit from its
    own primary counterpart; each tier carries its own number, so a tertiary can be given
    more room than a secondary. Works with or without the blue/red split: with it the
    interval is signed (`-W..0` / `0..W`), without it the distance alone is bounded
    (`+- W`). Only ever one velocity relation per line, so the two cannot overwrite
    each other.
  - **σ / σ_primary** and **amp / amp_primary** — allowed ranges, written `lo..hi`
    (`inf` allowed). The defaults `1..inf` and `0..1` are exactly the old one-sided
    "broader and fainter than its core" bounds, so an untouched table reproduces the
    previous output; narrowing a cell states something stronger in one place
    (`2..4` = "every secondary is 2–4× the core's width").
- **The primary row's velocity window is measured from systemic** (rest × 1+z, from the
  Source *z* field) rather than from each line's own initial guess — the guess just
  re-centres the window on whatever was typed, which is not a physical statement. Falls
  back to the old per-line anchor when no redshift is available.
- Seeded per scenario and reset with one button; an unreadable cell is repaired to its
  default on Preview rather than silently dropping the constraint. Note the velocity
  window **replaces** the absolute centroid window on the lines it touches — a constrained
  centroid is an expression, and lmfit does not apply bounds to expressions — which the
  preview states in-line.
- **New constraint syntax: `sigma == LO..HI * sigma_[B]`** (and the same for `amp`), a
  two-sided ratio range. Two inequalities could not express this: only one `.expr` may be
  assigned per parameter, so `sigma >= …` followed by `sigma <= …` silently kept whichever
  came last. The ratio is seeded from the components' own initial guesses, not from the
  constant 0.9 the generic inequality path uses — that constant is what made the old
  σ branch need an additive reparameterisation to avoid collapsing the two components.
  Verified to reproduce the previous path's fit exactly (identical χ² and recovered
  values) while a tightened range binds at its limit.
- **New constraint syntax: `vel == vel_[B] LO..HI`**, an explicit signed Δv interval in
  km/s. The existing one-sided forms (`vel <= vel_[B] + D`) leave the far side open to the
  edge of the fit window, so a bare "stay blueward" would have *loosened* the velocity
  range; the interval carries the scenario's own window instead.
- Fixed, in the same parser: an exact tie reaching `_apply_velocity_constraint` directly
  raised `min == max` from lmfit (the normal path rewrites ties earlier, so this only bit
  callers that skipped `update_constraints_with_velocity`); and a constraint overridden by
  a later one on the same parameter left its helper varying but referenced by nothing — an
  all-zero Jacobian column that makes the covariance, and so every reported uncertainty on
  that fit, untrustworthy. Orphaned `offset_*` / `ratio_*` helpers are now frozen. This
  bites in normal use: a K-group sigma tie is applied *after* Smart Constraints' bounded
  ratio and overwrites it.
- The K-group sigma-tie signature (`sigma == <factor> * sigma_[ref]`) matched `1.5..4` as
  a "factor", so a K-group sync would have stripped Smart Constraints' own bounded-ratio
  constraint. It now matches a single decimal only.

### Fixed — unreadable numbers on the Fit Parameters buttons

- **Values are formatted to fit their buttons.** `Centroid_0_lowlim` and friends fell
  through to a raw `str()`, so a button read `6559.108765432109` — 18 characters in a cell
  sized for about 9, and unreadable without clicking it open. Formatting is now unit-aware
  (`_fmt_param`): wavelengths get 2 decimals, km/s gets 1, and amplitudes keep significant
  figures because a flux may be `1e-18` or `1e4`. `inf` renders as `inf` and NaN as blank.
- `Centroid_0` itself *gained* precision in the process: `_fmt`'s 4 significant figures
  turned 6592.12 Å into `6592`, discarding 0.14 Å ≈ 6 km/s.
- The two places that build these buttons had drifted apart (the full-panel rebuild and
  the single-line append handled the limit columns differently); both now call one shared
  `_line_button_text`.

### Changed — S/N map is computed once and cached

- **Changing the S/N threshold no longer recomputes the S/N map.** The threshold only
  picks the contour level; the map itself depends on the cube, the wavelength grid, the
  line centres and the window widths. It is now cached on exactly those, so the second and
  subsequent thresholds just redraw the contour. The cache is dropped when the cube, its
  wavelength axis or its flux scale changes.
- **The map computation itself is ~27× faster.** Every mask depends only on `wavelengths`
  and the line centre, never on the spaxel, but they were being rebuilt inside an
  nx×ny Python loop; each line is now a handful of whole-array operations. On
  `UGC05101_supercube.fits` (14792 × 146 × 129) with a ten-line model: **19.9 s → 0.74 s**
  for the first calculation, and free for each threshold after it. Output verified
  bit-identical to the previous implementation (`max|diff| = 0`), NaN spaxels included.

### Added — extension picker for multi-extension FITS files

- **Opening a file with more than one loadable extension now asks which one to load**
  instead of silently taking the first that looked like data. The dialog lists every
  extension HyperCube can ingest (cubes, 1D spectra, tables) with its number, `EXTNAME`,
  type, dimensions and `BUNIT`, and pre-selects the most likely science array — a spectral
  cube first, and anything whose name reads as ancillary (`ERR`, `DQ`, `WMAP`, `VAR`, …)
  last. This matters for JWST `s3d` products, where `SCI`/`ERR`/`DQ`/`WMAP` all have the
  same shape. Cancelling the dialog leaves the previously loaded cube untouched.
- **Any extension may now be loaded, not just one carrying its own spectral WCS.** An
  image extension inherits the cards it does not state itself from the primary header —
  which is where multi-extension files usually keep `OBJECT`, `REDSHIFT` and often the
  spectral WCS — and a cube with no `CRVAL3` anywhere falls back to channel indices with a
  warning rather than failing the load.
- **Sessions reopen the extension they were built on.** `.hcsession` already recorded
  `fits_ext`, but restore re-ran the guess and then overwrote the number; it now loads that
  extension directly and never shows the picker.

### Fixed — display scaling

- **The UI now follows the display's scale factor.** Qt's high-DPI support was never
  switched on, so on an OS-scaled monitor (a 4K desktop at Windows 125%/150%) the whole
  interface rendered at 100% — i.e. tiny. `AA_EnableHighDpiScaling` and
  `AA_UseHighDpiPixmaps` are now set before the `QApplication` is created, together with
  the `PassThrough` rounding policy, without which Qt rounds a fractional scale factor
  *down* to the nearest integer and a 150% desktop still renders at 100%.
- **Font sizes come from the platform's UI font instead of hardcoded pixels.** The theme
  pinned `font-size: 9px` on every `QPushButton` — roughly half the platform UI font, and
  HyperCube's interface is almost entirely buttons. Stylesheet font sizes are now rewritten
  at load time into points scaled from the platform font; borders, padding and corner radii
  are deliberately left unscaled so the theme keeps its hairlines.
- **Every hardcoded widget dimension scales with the font** (`setFixedHeight`,
  `setFixedSize`, `setMinimumWidth`, …), so larger text cannot clip its control. Values
  computed from `sizeHint()` are left alone, as they already track the font.
- **Embedded matplotlib panels scale too** — rcParams and the inline annotation sizes — so
  axis labels no longer stay tiny while the surrounding UI grows.
- **New View ▸ UI Scale menu** (Larger / Smaller / Reset, `Ctrl/Cmd +`, `Ctrl/Cmd -`,
  `Ctrl/Cmd 0`) as a manual override on top of the automatic sizing, clamped to 70–300% and
  remembered across sessions.

---

## [v0.4.0] — 2026-08-24

Measurement errors, propagated parameter uncertainties, and explicit output units. This
release also ships the work that accumulated after the `v0.3.0` tag (parallel cube
fitting, Rectify, sequential core→outflow fitting, calibrated quality metrics) and two
modules that were previously missing from the repository.

### Measurement errors & uncertainties

- **Measurement-error cubes are discovered and used automatically.** On cube ingest
  HyperCube looks for per-pixel flux uncertainties: an `ERR` / `VAR` / `IVAR` / `STAT` /
  `FLUXERR` extension of the science file (JWST `s3d`, MUSE, …), then a sidecar file next
  to it (KCWI DRP `*_icubes.fits` + `*_vcubes.fits`, `*_err`, `*_var`, `*_ivar`, …).
  Variance and inverse-variance are converted to 1σ; non-physical entries (negative
  variance, non-positive inverse variance) are dropped rather than trusted. Detection is
  silent and never blocks ingest.
- **Empirical fallback.** When no error cube exists — or the one that exists is unusable,
  e.g. a PSF-subtracted product whose `ERR` extension is all zeros — the noise is measured
  per spaxel from the **line-free continuum inside each fit window** with the DER_SNR
  estimator (Stoehr et al. 2008): a MAD-style statistic on second differences, so a sloped
  or curved continuum cannot inflate it and neither can a minority of line pixels.
- **New `Measurement Errors…` dialog** (Fit Parameters → *Cube:* row) to override the
  automatic choice: pick any extension of any FITS file, declare whether it holds 1σ /
  variance / inverse variance, or force the empirical estimate. The choice is saved in
  `.hcsession` files.
- **The fit is now weighted by 1/σ**, and zero-weighted outside the fit windows.
  `PiecewiseModel` is identically zero outside a continuum region, so those pixels carry
  no information about any parameter; including them in χ² was inflating every reported
  uncertainty via lmfit's `scale_covar` rescaling. Reported `*_std` values are therefore
  **propagated measurement errors** (`scale_covar=False`), not residual-scatter estimates.
- **`vel_std`** — the velocity uncertainty is now reported directly, propagated exactly as
  σ_v = c·σ_λ/λ₀ with λ₀ = λ_rest(1+z). Previously only `cen_std` was written and callers
  had to derive it.
- **Provenance in every fit row**: `noise_source` (which error cube, or the empirical
  estimator), `noise_median` (median σ over the pixels used) and `noise_npix`.
- **`rchisq_w`** — reduced χ² over the weighted pixels only, available as a Quality Map and
  as a Rectify/Mask criterion. lmfit's native `rchisq` divides by the *full* spectrum
  length and is not a usable goodness-of-fit on its own; `rchisq_w` sits near 1 for a good
  fit with a correct noise model.

Validated by Monte Carlo: reported `vel_std` matches the true scatter of refits to 0.3%
for unbounded parameters. It runs ~7% low when a parameter sits against its `min`/`max`,
which comes from lmfit's bounded-parameter covariance transform.

### Output format

- **CSV column names now carry their units** — `cen_fit_A`, `vel_std_kms`, `amp_fit_flux`,
  `cont_region1_slope_fit_fluxperA`, `RA_deg`, `spaxel_x_pix`, … The flux unit itself (the
  cube's `BUNIT`) is written into the CSV's scale/units header row alongside the wavelength
  and velocity units.
- **σ is a velocity dispersion (km/s) everywhere the user sees it** — GUI, CSV, and the
  FITS maps. Ångström remains the internal storage unit and is recoverable from the
  companion centroid column.
- **FITS output**: `SIGMA_<line>` is now km/s (the redundant `SIGMAKMS_<line>` extension is
  retired), every map carries a `BUNIT` header, and new `AMP_STD_`, `CEN_STD_`, `VEL_STD_`
  and `SIGMA_STD_` uncertainty maps are written next to the value maps.
- Loading is backward compatible: pre-v0.4.0 CSVs (σ in Å with derived `*_kms` companions)
  and FITS products (`SIGMA_` in Å with `SIGMAKMS_`) are detected and read correctly.

### Packaging

- **`HyperCube_SmartConstraints.py` and `HyperCube_Noise.py` are now in the repository.**
  Both are imported by `HyperCube.py`; without them a fresh clone cannot start.
- **The stellar template libraries ship with the repository** (`eMILES/`,
  `indo_us_library/`; 8.5 MB). Previously excluded by `.gitignore`, so stellar fitting did
  not work from a clean clone.

### Included from the unreleased work after v0.3.0

- **Parallel cube fitting** — `Fit Cube` runs across a process pool over a shared-memory
  cube, driven by the Qt-free `HyperCube_fit.py` kernel so serial and parallel paths give
  identical results. Core count is configurable.
- **Sequential core→outflow fitting** — a staged narrow-then-broad-then-joint fit that
  structurally breaks the narrow/broad degeneracy, with no per-galaxy tuning.
- **Calibrated fit-quality metrics & Quality Map** — core/continuum residual ratio, signed
  residual z, runs-test z and calibrated continuum χ², all scale-free and comparable
  between bright and faint spaxels.
- **Rectify Bad Fits** — repairs only the spaxels a cube fit got wrong, seeding each from
  its best-scoring good neighbour with targeted multi-start fallbacks.
- **Velocity constraints** — `vel == vel_[B]`, `vel == vel_[B] +- 300`, and one-sided
  forms, realized as a bounded additive centroid offset that is correct for lines at
  different rest wavelengths.
- **Integrated-flux constraints** — `flux == 2.94 * flux_[[N II]_6548]` and ranged forms
  such as `flux == 0.44..1.45 * flux_[[S II]_6731]`, exact even when widths differ.
- **Smart Constraints** — auto-fills kinematic groups, doublet flux ratios and parameter
  bounds from a chosen physical scenario.
- **Multiple stellar regions**, plus per-spaxel model overrides and assorted crash/render
  fixes.

---

## [v0.3.0] — 2026-06-11

### Added
- **Kinematic groups (K1–K5)** — tie the velocity *and* velocity dispersion of multiple
  lines into one kinematic solution from the Line Name window. Dispersion is tied in km/s
  via rest-wavelength ratios; the group's reference line is surfaced in the UI.
- **Velocity dispersion in km/s** displayed and edited throughout the GUI, and exported as
  companion km/s columns (CSV) and a companion km/s map (FITS).
- Constraints dialog: syntax help button, **Auto-suggest constraints**, and clearer
  "constraints saved" feedback.

### Fixed
- Relational constraints referencing bracketed forbidden-line names (`[S II]`, `[N II]`,
  `[O III]`, …) were silently dropped.
- Amplitude constraints could be lost during per-spaxel flux rescaling.
- `update_constraints_with_velocity` no longer crashes on NaN constraint rows.
- Dark-theme checkbox check-marks now render correctly.

---

## [v0.2.0] — 2026-05-21

### Added
- Channel maps.
- Docked fit-parameters panel.
- NED integration for source resolution.
- Cube zoom / pan.

### Fixed
- Flux rescaling.

---

## [v0.1.0] — 2026-05-21

Initial tagged release.

[v0.4.0]: https://github.com/jkader925/HyperCube/releases/tag/v0.4.0
[v0.3.0]: https://github.com/jkader925/HyperCube/releases/tag/v0.3.0
[v0.2.0]: https://github.com/jkader925/HyperCube/releases/tag/v0.2.0
[v0.1.0]: https://github.com/jkader925/HyperCube/releases/tag/v0.1.0
