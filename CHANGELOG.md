# Changelog

All notable changes to HyperCube are recorded here. Versions follow
[Semantic Versioning](https://semver.org/) with `0.x` semantics: minor bumps may change
output formats, and every such change is called out under **Output format** below.

---

## [Unreleased]

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
