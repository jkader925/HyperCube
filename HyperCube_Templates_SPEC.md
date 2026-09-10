# SPEC — HyperCube Fitting Templates + Headless Batch Mode

## 0. Purpose and audience

This is a build spec for two coupled additions to **HyperCube**:

1. **Rest-frame fitting templates** — a human-authorable file that describes a
   complete emission-line model (lines, continuum regions, initial guesses,
   bounds, constraints, K-groups, S/N gate, fit options) in a form that is
   **independent of any particular galaxy**, plus a **target manifest** carrying
   the per-galaxy facts, and an **expansion** step that combines the two into the
   concrete observed-frame model HyperCube already knows how to fit.
2. **A headless batch runner** that consumes those artifacts and produces fit
   products the GUI can reopen, with no `QApplication`.

The immediate driver is the MAUNA MUSE sub-sample: `MAUNA_Sample_Fitting_transposed.csv`
tracks **22 MUSE galaxies × 11 line groups × {1,2} components = 484 fits, all
still unfilled**. Building each model by hand in the GUI is the bottleneck, and
it is already producing errors — `~/Downloads/MAUNA_MUSE/IR09022-3615_MUSE.hcsession`
points at the **07251 cube** and carries 07251's redshift under a WISEA name.

This document is written to be handed to **Claude Code**. Build it in the
**stages** defined in §9 — each stage must be independently runnable and
inspectable before the next is started. Do not build the whole thing in one pass.

§2 states requirements, not suggestions. Where a choice encodes a physical
assumption (σ convention, LSF model, coverage margin), the code must **expose it
and record it in the output**, never bury it.

---

## 1. Tech stack and conventions

- **Language:** Python 3.11+, matching the existing tree.
- **Dependency discipline:** the new modules `HyperCube_LSF.py` and
  `HyperCube_Templates.py` are **Qt-free** — `numpy`, `pandas`, `astropy` only —
  for the same reason `HyperCube_fit.py` is: they are imported by the batch
  runner and (indirectly) by the process pool. This is a hard constraint.
- **No new file format machinery.** Templates are sectioned CSVs, blank-row
  separated, exactly like the existing `save_cube_fit` layout (`HyperCube.py:9852`).
  Reuse `to_export_units` / `from_export_units` (`:593`/`:610`) and the
  `UNIT_SUFFIXES` convention (`:517`) rather than inventing a parallel one.
- **Units, and the naming that carries them.** Every template column name ends in
  its unit (`_A`, `_kms`, `_rest_A`), following the repo's output-units
  convention. Internally σ stays in Å; km/s appears only at file boundaries.
- **Style:** module-level free functions, no global state in the new modules,
  type hints, dataclasses for structured records.
- **Testing:** `pytest`. Every conversion (rest↔observed, intrinsic↔observed σ,
  coverage decision, K-group repair) gets a unit test.

---

## 2. Requirements (the "why" — do not deviate without flagging)

### 2.1 Nothing galaxy-specific may live in a template

A template must be applicable to any galaxy observed with the same instrument
class. Exactly three things are galaxy-specific, and each has a defined source:

| Fact | Source | Never in a template |
|---|---|---|
| Redshift | **Manifest** (bootstrapped from NED, then reviewed) | ✓ |
| Cube path + extension | **Manifest** | ✓ |
| Instrument resolution | **The cube itself**, via `resolving_power_from_header` | ✓ |

This is the crux of the design. Redshift is not recoverable from MUSE headers
(no `REDSHIFT` keyword; `OBJECT` is `2MASX J07273754-0254540`, not an IRAS ID),
so cube↔galaxy identity must be stated explicitly — that is what the manifest is
for. Resolution *is* recoverable, so it must be read, not retyped.

### 2.2 σ in a template is intrinsic; σ in a fit is observed

MUSE's resolving power varies **59% across a single one of these models** —
R = 1873 at Hβ to 2978 at [S II] at z = 0.0876. A template that specified
observed σ would therefore mean a different *physical* width at each line, and
would not survive being moved to another galaxy (different observed λ) or another
instrument.

Templates therefore carry **intrinsic** σ (guesses *and* bounds). Expansion adds
the instrument LSF in quadrature **at each line's own observed wavelength**:

```
sigma_obs_kms = sqrt(sigma_int_kms**2 + sigma_lsf_kms(lam_obs)**2)
```

and the same for `lowlim`/`highlim`. Note the lower bound is physically
meaningful under this rule: an intrinsic floor of 20 km/s becomes an observed
floor of ~71 km/s at Hβ, correctly forbidding a fit narrower than the LSF.

**Fit output is unchanged** — `sigma_fit` stays the raw observed Gaussian width
in Å, exported in km/s, exactly as today. `agnpipe` and existing scripts must not
need to change. Deconvolved output columns are an explicit non-goal (§10).

### 2.3 Coverage dropping is load-bearing, not a nicety

The same template applied across the sample will contain lines that a given cube
cannot see. Over MUSE's 4700–9350 Å: H-gamma is **out** at z ≤ 0.05 and **in** at
z ≥ 0.0876; [O II] 3727 and [Ne III] 3869 are out for the entire sample.

Expansion must therefore drop out-of-coverage lines and regions *gracefully and
audibly*: renumber, repair everything that referenced them (§6.4), and emit a
report. A line silently missing from a fit is a scientific error.

A `required` flag per line and per region inverts this where it matters: a
required item that falls out of coverage **fails the expansion loudly** rather
than yielding a quietly different model.

### 2.4 The expansion report is itself a deliverable

Running expansion across the manifest without fitting (`--dry-run`) answers
"which of the 484 cells are even possible?" — i.e. it **generates the N/A map for
`MAUNA_Sample_Fitting_transposed.csv`**. Treat this as a first-class output, not
a debug log.

### 2.5 Headless must not fork the fitting code

The batch runner must call the *same* parameter builder and the *same* pool
driver as the GUI. Any divergence between them is a bug factory: the GUI and the
single-spaxel path have already drifted (§3.3). Extraction, not duplication.

---

## 3. Host contract (existing HyperCube components)

### 3.1 Reused verbatim — do not modify

| Component | Location | Role |
|---|---|---|
| `HyperCube_fit.py` (whole kernel) | — | `build_model`, `fit_one_spaxel`, `staged_fit`, `_worker_init`, `_worker_fit_one` |
| `update_constraints_with_velocity` | `HyperCube.py:732` | `vel == vel_[X]` → centroid ratio |
| `add_dataframe_constraints_to_params` | `:1062` | constraint strings → lmfit `.expr` |
| `to_export_units` / `from_export_units` | `:593` / `:610` | unit suffixing, σ↔km/s |
| `sigma_wl_to_kms` / `sigma_kms_to_wl` | `:485` / `:500` | Å ↔ km/s |
| `HyperCube_SmartConstraints.build_plan` | `:484` | scenario → constraints, K-groups, bounds |
| `HyperCube_Noise.detect` / `load_sigma` | — | σ-cube discovery |
| `compute_snr_map` | `HyperCube.py:10341` | numpy-only; **not** `calculate_snr_map`, which draws |

All are already Qt-free and module-level.

### 3.2 Replaced

`FitParamsWindow.save_file()` (`:8042`) and `open_file()` (`:8088`) are the
orphaned ancestor of this feature: a 4-section CSV writer/reader that nothing can
reach, because the `QAction`s at `:7243-7251` were never `.triggered.connect()`ed
and the Ctrl+O handler at `:8190` is dead (the window is never shown — only its
`centralWidget()` is reparented into a dock at `:7106`). `README.md:200` still
instructs users to press Cmd-O.

They are superseded by the template I/O of Stage 1. Known defects to **not**
carry forward: actors are removed before the cancel check (`:8097`); the parsed
scale/units section is discarded; `knots_*`/`poly_coef_*` are left as strings;
`base_df`/`base_df_cont` are never set, so a loaded model never becomes the
locked schema `fit_cube` uses; the `float_columns` list still names `spaxel_x`.

### 3.3 Repaired in passing (Stage 4)

- `fit_cube` (`:11954`) **does not** zero `slope`/`intercept` for
  `cont_type == 'stellar'` regions, although the single-spaxel path does
  (`:11812-11822`). This directly affects the MUSE models, whose regions are both
  stellar. Fixing it changes cube-fit results for stellar continua — flag it as a
  behaviour change, do not fold it in silently.
- `df_cont_sort` (`:12003`) and `df_sort` (`:12006`) are dead.
- The line-parameter loop is nested inside the region loop, re-adding every line
  once per region (idempotent, wasteful).
- `region_index` is `df_cont[...].index[0] + 1` (`HyperCube_fit.py:414`), so any
  driver must hand the kernel a `df_cont` whose index is exactly `0..N-1`.

### 3.4 Line placement runs backwards today

`_identify_line` (`:1547`) derives `Rest Wavelength` from a user-drawn observed
centroid. There is **no** "place this line at rest·(1+z)" path anywhere in the
tree; expansion introduces it. This is small but is the conceptual inversion at
the heart of the feature.

---

## 4. Template file format (`*.hct.csv`)

Sectioned CSV, blank-row separated, in this order. Sections 1–4 are required.

**§1 — Template metadata** (header row + one data row)

```
template_name,template_version,instrument,scenario,n_components,snr_threshold,sequential,max_nfev,amp_convention
MAUNA-MUSE-core,1,MUSE,outflow,1,5.0,False,512,relative
```

`instrument` is a compatibility assertion, checked against the cube's `INSTRUME`
at expansion (mismatch = loud failure). `scenario` names a
`HyperCube_SmartConstraints` scenario, or `none`.

**§2 — Conventions** (header row + one data row) — the self-describing header
that distinguishes a template from an observed-frame model CSV

```
wavelength frame,wavelength unit,velocity unit,sigma convention,amplitude convention
rest,Angstrom,km/s,intrinsic,relative
```

A reader **must** verify `wavelength frame == rest` and `sigma convention ==
intrinsic` before treating a file as a template.

**§3 — Continuum regions** (rest frame)

```
Continuum Name,x1_rest_A,x2_rest_A,cont_type,region_ID,required,stellar_library,stellar_moments,poly_degree,knots_x_rest_A
Stellar-blue,4780,5060,stellar,0,True,indo_us_library,2,,
Stellar-red,6250,6800,stellar,1,False,indo_us_library,2,,
```

**§4 — Lines** (rest frame)

```
Line_Name,Rest Wavelength_A,region_ID,component,vel_0_kms,vel_lowlim_kms,vel_highlim_kms,sigma_int_0_kms,sigma_int_lowlim_kms,sigma_int_highlim_kms,amp_rel_0,amp_lowlim,amp_highlim,kgroup,kgroup_ref,required
H_beta_4861,4861.333,0,1,0,-500,500,80,20,350,0.30,0,inf,K1,True,False
[O III]_5007,5006.843,0,1,0,-500,500,80,20,350,0.10,0,inf,K1,False,False
```

- `component`: 1 = core, 2 = broad/outflow. Two rows sharing a
  `Rest Wavelength_A` form the pair `HyperCube_fit.component_pairs` looks for.
- `amp_rel_0`: dimensionless, **relative to the brightest line in the template**
  (which is 1.0). BUNIT differs per instrument, so absolute amplitudes are not
  portable; expansion rescales from the cube (§6.3).
- `kgroup_ref`: at most one `True` per group. Honours the promotion machinery
  already in `df` (`kgroup_ref` column).

**§5 — Explicit constraint overrides** (optional)

```
Line_Name,constraint_1,constraint_2,constraint_3,constraint_4,constraint_5
[O III]_5007,flux == 2.98 * flux_[[O III]_4959],,,,
[S II]_6716,flux == 0.44..1.45 * flux_[[S II]_6731],,,,
```

Applied **after** the scenario, replacing the generated list for the named line.
Saving a template from the GUI writes this section in full (the explicit form),
so a round trip is lossless even when `scenario` is set.

---

## 5. Manifest file format (`*.manifest.csv`)

One flat table, one row per target. Optional columns override the template.

```
target_id,cube_path,cube_ext,redshift,z_source,ref_spaxel_x,ref_spaxel_y,snr_threshold,enabled,notes
IRAS07251-0248,/Users/justin/Downloads/MAUNA_MUSE/IRAS07251-0248_MUSEDEEP.fits,1,0.087557,NED,160,163,5.0,True,
```

- `z_source` is provenance, free text (`NED`, `literature`, `stellar fit`).
- `ref_spaxel_x/y`: the spaxel amplitudes are scaled from. Blank ⇒ use the
  continuum-peak spaxel, excluding a border margin.
- `enabled`: lets a target be parked without deleting the row.

Bootstrapping: a helper resolves each `target_id` through the existing NED path
(`HyperCube.py:12988`, `astroquery.ipac.ned` with an HTTP fallback) to fill
`redshift`. **The generated manifest is a draft for human review, never
authoritative** — NED returns a redshift for an identifier, not necessarily the
systemic redshift you want to fit against.

---

## 6. Expansion contract

```python
expand(template, manifest_row, cube_header, wavelengths)
    -> (df_obs, df_cont, df, ExpansionReport)
```

Pure, Qt-free, no file I/O. Ordered steps:

**6.1 Identity and resolution.** `z` from the manifest. Assert
`cube_header['INSTRUME']` matches `template.instrument`. Build the LSF model from
the header (§7). Write `df_obs` = (`sourcename`, `redshift`, `resolvingpower`),
where `resolvingpower` is the LSF evaluated at mid-coverage — retained for pPXF,
which still wants a scalar.

**6.2 Placement and coverage.** For each line,
`lam_obs = rest * (1 + z) * (1 + v_0/c)`. A line is *covered* iff
`lam_obs ± margin` lies inside `wavelengths`, where `margin = coverage_nsigma *
sigma_obs_A` (default `coverage_nsigma = 3`). Regions are covered iff both
`x1,x2` (redshifted) lie inside. Uncovered + `required` ⇒ raise. Uncovered +
optional ⇒ drop and record.

**6.3 Values.** σ through the quadrature rule of §2.2, then `sigma_kms_to_wl`.
Centroid bounds from `vel_lowlim/highlim_kms` about `lam_obs`. Amplitudes:
measure the reference spaxel's peak within each region, then
`Amp_0 = amp_rel_0 * peak`. Amplitude bounds pass through unscaled (`0..inf`).

**6.4 Repair — the part that must not be hand-waved.** After dropping:

1. Renumber `Line_ID` contiguously in model order; renumber `region_ID` and
   **reset `df_cont.index` to `0..N-1`** (§3.3).
2. For each K-group: if its `kgroup_ref` line was dropped, fall back to the first
   surviving member (the existing `_kgroup_reference` fallback semantics). If a
   group falls below two members, clear the group and its ties.
3. For each constraint string, extract the referenced line with the existing
   `_ref_line_name` (`:277`) and drop any constraint naming a dropped line —
   recording each one. Do not attempt to re-point it at a substitute.

**6.5 Constraints.** Run `build_plan` for the scenario, apply its
`constraint_additions` / `kgroup_assignments` / `bound_updates`, then overlay
§5 explicit overrides per line.

**6.6 Report.** `ExpansionReport` carries, per target: lines kept/dropped with
reason and observed λ, regions kept/dropped, constraints dropped, K-groups
re-anchored or cleared, the LSF provenance string, and the amplitude reference
spaxel. Serialisable to both text and a one-row-per-(line,target) CSV — the
latter is the N/A map of §2.4.

---

## 7. Instrument LSF (`HyperCube_LSF.py`, new, Qt-free)

```python
@dataclass
class LSFModel:
    label: str                      # provenance, e.g. "MUSE LSF polynomial"
    def fwhm_A(self, lam_obs): ...
    def sigma_kms(self, lam_obs): ...
    def R(self, lam_obs): ...

lsf_from_header(header, wl) -> (LSFModel, provenance)
intrinsic_to_observed(sigma_int_kms, lam_obs, lsf) -> float
observed_to_intrinsic(sigma_obs_kms, lam_obs, lsf) -> float   # NaN if unresolved
```

Instruments: MUSE (polynomial), KCWI/KCRM (per-grating FWHM — the
`_KCWI_FWHM_LARGE` table and `_kcwi_grating_fwhm` already at `HyperCube.py:1728`
move here), MIRI MRS, NIRSpec.

**`resolving_power_from_header` (`:1789`) is refactored to delegate**: it builds
the `LSFModel`, evaluates at mid-coverage, and returns `(R, provenance)` exactly
as now. Its current behaviour and console output must not change — it is already
covered by the R work in this branch.

**Open item, resolve before building on it:** the MUSE polynomial
`FWHM(λ) = 5.866e-8 λ² − 9.187e-4 λ + 6.040` reproduces published R in the red
(3620 vs 3590 at 9300 Å; 2819 vs ~2800 at 7000 Å) but gives R = 1610 at 4800 Å
against a published 1770 — **~9% low, inside the MAUNA band**. Confirm against
the ESO LSF reference and record the chosen parameterisation in the module
docstring with its source.

---

## 8. Headless batch runner (`hypercube_batch.py`)

```
python -m hypercube_batch --template MAUNA_MUSE_core.hct.csv \
                          --manifest MAUNA_MUSE.manifest.csv \
                          --out runs/ [--targets A,B] [--cores 8] [--dry-run]
```

- `--dry-run`: expand + report only, no fitting. This is how the N/A map is made.
- Per target it writes `<target>_<template>_Fit.csv` in the **full 5-section
  `save_cube_fit` layout** (so the GUI can reopen it), optionally a
  `.hcsession`, plus `<target>_<template>_coverage.csv` and a run log recording
  template name/version, manifest row, LSF provenance, git SHA, and timings.
- Must set the `df_fit['color']` column the GUI's overlay expects
  (`HyperCube.py:12447-12463`).
- A failing target logs and continues; the run exits non-zero if any failed.

Expectation to state plainly in the README: **this does not make per-spaxel
fitting faster** — same kernel, same process pool. It buys unattended queued
runs, reproducibility, and remote execution.

---

## 9. STAGED BUILD PLAN (build and stop at each stage for review)

- **Stage 0 — this spec.** *Checkpoint: reviewed before any code.*
- **Stage 1 — Template I/O.** Reader/writer for §4, built on the maintained
  `load_cube_fit` machinery; wire the menu actions; retire `save_file`/`open_file`.
  *Checkpoint: save a template from the 07251 session, reload, identical model.*
- **Stage 2 — `HyperCube_LSF.py`.** Resolve the §7 open item first.
  *Checkpoint: R(λ) checked at 4800/7000/9300 Å; `resolving_power_from_header`
  output unchanged for every cube in the repo.*
- **Stage 3 — Manifest + expansion (§5, §6).** Includes the NED bootstrap helper.
  *Checkpoint: expand both templates across all 22 targets; open 2–3 in the GUI
  and confirm lines land on real features; review the coverage CSV.*
- **Stage 4 — Qt-free extraction.** `build_params(df, df_cont, z)` out of
  `HyperCube.py:12000-12109`; `run_pool(..., progress_cb)` out of `:11500-11599`.
  Repair §3.3. *Checkpoint: GUI fit of one cube identical to before, except the
  stellar slope/intercept fix, which is reported separately.*
- **Stage 5 — Batch runner (§8).** Fill in README's empty Pipeline section.
  *Checkpoint: headless and GUI agree on one galaxy.*
- **Stage 6 — Run the MUSE sample.** 44 jobs, overnight batches.
  *Checkpoint: coverage CSV folded back into the tracking sheet.*

Add a decisions-log section to `CLAUDE.md` when Stage 0 is accepted, following
the CiK precedent (spec is the authority; CLAUDE.md is orientation).

---

## 10. Explicit non-goals (for now)

- **No deconvolved σ in the output.** Fit products keep today's observed-width
  schema so `agnpipe` is untouched. Revisit once templates are in use.
- **No KCWI expansion in v1.** The `instrument` field and the LSF module are
  designed so the KCWI-blue/red modes drop in later; do not build them now.
- **No new fitting algorithm, noise model, or quality metric.**
- **No per-spaxel speedup.** Out of scope; see §8.
- **No template inheritance/includes.** Two flat files beat a small language.
- **No automatic acceptance of NED redshifts** (§5).

---

## 11. Acceptance criteria (Stages 0–5)

1. **Round trip** — save a template from the 07251 session, reload it: identical
   line count, rest wavelengths, bounds, constraints, K-groups.
2. **Portability** — one template expands onto 07251 and onto a target at a
   different redshift; centroids land on the lines; coverage-dropped lines appear
   in the report, never silently missing.
3. **LSF** — R(λ) matches the confirmed reference at 4800/7000/9300 Å; a 100 km/s
   intrinsic σ yields measurably different observed σ at Hβ and [S II]; and
   intrinsic → observed → intrinsic returns 100 km/s.
4. **Repair** — a synthetic template whose K-group reference and a
   constraint target both fall out of coverage expands to a valid model, with
   both repairs recorded in the report.
5. **No regression** — a GUI cube fit before and after Stage 4 matches, the
   stellar slope/intercept fix excepted and reported.
6. **Headless == GUI** — same template, same cube, both paths, `df_fit` agrees.
7. **Reopen** — a headless-produced CSV loads in the GUI and renders a parameter
   map.
