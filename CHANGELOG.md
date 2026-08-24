# Changelog

All notable changes to HyperCube are recorded here. Versions follow
[Semantic Versioning](https://semver.org/) with `0.x` semantics: minor bumps may change
output formats, and every such change is called out under **Output format** below.

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
