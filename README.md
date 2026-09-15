<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" width="100%" srcset="https://github.com/user-attachments/assets/223adf80-8792-454e-ad99-94bbef751a5c">
    <img width="100%" alt="auto light/dark mode" src="https://github.com/user-attachments/assets/2e03646a-c6d5-444f-bcf6-bf43dfd4dd5c">
  </picture>
</div>


HyperCube is a python-based spectral fitting tool designed to make integral field spectroscopic (IFS), or hyperspectral data analysis more interactive and intuitive, while preserving automation and repeatability. The tool combines a user-friendly [PyQT5](https://github.com/PyQt5) GUI with the robust and flexible fitting capabilities of [lmfit](https://github.com/lmfit/lmfit-py), and is particularly well-suited for interactive and batch process spectral modeling of 3D spectral data.


---

## What's New in v0.5.0

- **Rest-frame fitting templates.** A template (`*.hct.csv`) states a model the way physics does — rest wavelengths, velocity offsets from systemic, *intrinsic* velocity dispersions, bounds, K-groups, constraints — so **one file applies to every galaxy observed with a given instrument**. Nothing galaxy-specific lives in it: redshift and cube path come from a **manifest**, and the instrument's resolution from the cube itself. **Save Template / Load Template** are on the *Output:* toolbar row. See [Initiating Models with Configuration Files](#initiating-models-with-configuration-files).
- **Headless batch mode.** `python -m hypercube_batch` expands a template across a manifest and fits every target unattended, writing products the GUI reopens plus a `run_log.csv` recording the template, its version and the git SHA. `--dry-run` reports which lines are observable per galaxy before you spend time fitting. See [Pipeline Usage Mode](#pipeline-usage-mode).
- **A wavelength-dependent instrument LSF.** Intrinsic σ is broadened at **each line's own observed wavelength**, which is what makes one template portable — MUSE's resolving power varies 59% across a single model, so a flat `R` understates instrumental σ by 25 km/s at Hβ and 0.3 km/s at [S II]: a *differential* error across the very lines you are comparing.
- **Self-sky subtraction** for cubes with an under-subtracted airglow residual — the driver case is a sky line 2.1 Å from [N II] 6548 and 2.5× brighter than the real line. A sky spectrum is built by combining object-free spaxels (**median, never mean**) and subtracted, with its variance propagated into σ. Masks stack, a veto drawn on the artifact's own window is refused, and Revert restores the cube bit-identically. In the GUI as `Cube: → Self-Sky…`, headless as `--selfsky`. See [Self-sky subtraction](#self-sky-subtraction).
- **Spaxel masking is tabbed, and now gates the fit.** A map criterion (including a channel map off the live **C** window, which needs no fit) and a per-line S/N tab where **each line carries its own threshold** with an any/all combiner — one universal cut is wrong whenever lines differ in brightness, since [O III] 4959 is a third of 5007 by atomic physics. ⚠️ **Behaviour change:** masked spaxels are now **excluded from cube fits**; masking used to be display-only. See [Masking spaxels & the fit gate](#masking-spaxels--the-fit-gate).
- **The S/N mask measures the line, not the brightness.** Continuum is subtracted before the ratio, so a bright continuum source no longer passes at any threshold with no emission at all (a foreground star scored S/N = 43 on a spectrum containing no Hα). ⚠️ The numbers are smaller than before, so **a threshold carried over from v0.4 is a stricter cut** — re-check it against the contour. The map is also computed once and cached.
- **Resolving power is read from the cube** at ingest, with a tooltip saying where the value came from, so a derived number is never mistaken for one you typed. KCWI/KCRM, MIRI MRS, NIRSpec and MUSE are recognised; a coadded supercube deliberately gets no value.
- **Per-component constraints, multi-extension FITS ingest, and UI scaling.** Smart Constraints gains a per-component-tier bound table and blue/red side constraints for broad wings; an extension picker handles multi-extension FITS; and the interface scales to the display and platform font.
- **Names are yours.** NED is asked for the **measurement**, never for a relabelling — resolving `F01364-1042` can no longer rename your target `2MASX J01385289-1027113` and break every join you have against your own catalogues.

Templates, sessions and fit products from v0.4 still load.

The full history is in [CHANGELOG.md](CHANGELOG.md).

---

## Table of Contents
1. [Installation](#installation)
   - [Source Version](#hypercube-source-version)
   - [Standalone Version](#hypercube-standalone-version)
2. [Quick Start Guide](#quick-start-guide)
3. [Interactive Usage Mode](#interactive-usage-mode)
   - [Initiating Models Interactively](#initiating-models-interactively)
   - [Adjusting Model Parameters Interactively](#adjusting-model-parameters-interactively)
   - [Per-Spaxel Fit Correction](#per-spaxel-fit-correction)
   - [Saving and Restoring Sessions](#saving-and-restoring-sessions)
   - [Relational Constraints](#relational-constraints)
4. [Fitting Techniques](#fitting-techniques)
   - [Relational constraints & kinematic groups](#relational-constraints--kinematic-groups)
   - [Sequential core→outflow fitting](#sequential-coreoutflow-fitting)
   - [Measurement errors & parameter uncertainties](#measurement-errors--parameter-uncertainties)
   - [The S/N mask](#the-sn-mask)
   - [Masking spaxels & the fit gate](#masking-spaxels--the-fit-gate)
   - [Self-sky subtraction](#self-sky-subtraction)
   - [Resolving power](#resolving-power)
   - [Calibrated fit-quality metrics & the Quality Map](#calibrated-fit-quality-metrics--the-quality-map)
   - [Rectify Bad Fits](#rectify-bad-fits)
5. [Stellar Kinematics with pPXF](#stellar-kinematics-with-ppxf)
6. [Pipeline Usage Mode](#pipeline-usage-mode)
   - [Initiating Models with Configuration Files](#initiating-models-with-configuration-files)
   - [Batch Processing](#batch-processing)
7. [Troubleshooting](#troubleshooting)
8. [Acknowledging HyperCube](#acknowledging-hypercube)

---



# Installation
Installation and use of this tool has been tested on MacOS and Windows, it has not yet been tested on Linux operating systems. The first step is to clone the repository to a directory on your local machine where you have read/write/execute privileges:
```
git clone https://github.com/jkader925/HyperCube.git
```
This will create a "HyperCube" directory containing the distribution. Alternatively, download source files as a .zip and unpack to desired location.

### HyperCube Source Version
The tool was designed for quick and painless installation using `conda` environment management via the included environment file `hypercube.yml`. In a terminal, from your base conda environment, navigate to the new HyperCube directory and issue the following command:

```
conda env create -f hypercube.yml
```

Conda will install all of the required packages automatically. If not using conda, you can manually install the required packages (listed in hypercube.yml) via `pip`.

> **Stellar kinematics fitting** (the [Stellar Kinematics with pPXF](#stellar-kinematics-with-ppxf) section) additionally requires the [`ppxf`](https://pypi.org/project/ppxf/) package. If it is not already in your environment, install it with `pip install ppxf`. The bundled stellar template libraries live in the `indo_us_library/` and `eMILES/` folders and ship with the distribution.

#### Updating the Source Version
 
New releases may require additional packages. To bring your existing environment up to date, first get the latest code — if you cloned the repository with git:
 
```
git pull origin main
```
 
Or download the latest zip from the [Releases](https://github.com/jkader925/HyperCube/releases) page and replace the contents of your HyperCube folder with the new files. Then update your conda environment from your base conda environment:
 
```
conda env update -f hypercube.yml --prune
```
 
The `--prune` flag removes any packages that are no longer needed. If you run into environment conflicts, a clean rebuild is the most reliable fix:
 
```
conda env remove -n HyperCube
conda env create -f hypercube.yml
```
 
Then launch as normal:
 
```
python hypercube.py
```
 
---

### Hypercube Standalone Version
The repository also comes with a `hypercube.spec` file for use with [pyinstaller](https://github.com/pyinstaller/pyinstaller), in order to package a standalone app version of HyperCube. From a Python console (conda or otherwise), install pyinstaller:

```
pip install pyinstaller
```

Next, navigate to your HyperCube directory and install the HyperCube standalone app:

```
pyinstaller hypercube.spec
```

This will generate a `dist` folder which contains hypercube.app, which can be double-clicked to open the tool. You can create a shortcut to this application from anywhere on your machine.

#### Updating the Standalone Version
 
The app does not update automatically — each new version requires a fresh build. First get the latest code by downloading the latest zip from the [Releases](https://github.com/jkader925/HyperCube/releases) page and replacing the contents of your HyperCube folder, or if you cloned with git:
 
```
git pull origin main
```
 
Then rebuild from your HyperCube directory:
 
```
pyinstaller hypercube.spec
```

This regenerates the `dist` folder with an updated `hypercube.app`. Replace your existing app with the new one from `dist/`. If you have a shortcut or dock icon pointing to the old ap, update it to point to the newly built version.

---
 
# Quick Start Guide
This guide walks you through a basic analysis of a Keck Cosmic Wave Imager (KCWI) data cube observation of the luminous infrared galaxy IRAS F23365+3604. The purpose of this guide is to familiarize you with the basic features and modes available to you when using HyperCube to fit 3D spectral data, it is not intended as a comprehensive introduction to every feature the tool offers.

From your new `hypercube` conda environment, open the tool via the following command:

```
python hypercube.py
```

This should launch the main application window. You can now load the IFS data using the `Open FITS` button on the bottom right of the application window and selecting the file `IRAS_F23365+3604.fits`. You can also **drag the file from Finder and drop it anywhere on the window** — the same load, without the dialog. Either way a green progress bar runs in the status bar, in the slot where the filename will appear, naming the stage it is on; a large cube spends most of it collapsing the white-light image. The main application (visualizer) window will now show to panels: on the left is the image viewer, which initially shows a white light image of the galaxy (from integrating the spectrum in each spectral pixel, or "spaxel"), on the right is a live spectrum viewer that updates as you move the cursor across the white light image. 

### Interacting with the Image Viewer Panel
As the cursor is moved around the image, an orange rectangle indicates the currently focused spaxel. You can lock the spaxel by pressing the `L` key. To unlock, move the cursor back to the image viewer panel and press `L` again.

### Interacting with the the Spectrum Viewer Panel
The spectrum viewer panel shows the spectrum contained in the currently-selected spaxel. You can zoom into a portion of the spectrum by clicking and dragging across the spectrum. As you do, a grey rectangular region will indicate the range that will be zoomed to when you release click. The new horizontal (spectral) range reflects the one you selected, while the new vertical (signal or flux) range is auto-scaled to show the continuum and the peaks of any lines in that spectral window. Right-click anywhere on the spectrum to bring up a `reset zoom` button which can be clicked to set the spectrum viewer window to its original range.

### Draw Continuum and Gaussians to Initialize a Model
Select and lock onto a spaxel containing a spectrum with nice, bright emission lines (in **Fig. 1**, we've locked onto position x=16, y=10), then zoom into the Hα-[N II] line complex (6940--7040 Å). With the cursor hovered over the continuum at ~6950 Å, press the `d` key to start placing your linear continuum model. One end of the line remains locked at the initial position, while the free end follows the cursor. Move to the continuum at ~7030 Å and press the `d` key once more to lock in the continuum model; this will bring up the parameter window which we can ignore for now.

<div align="center">
  <picture>
<img width="985" alt="Screenshot 2025-04-15 at 12 28 36 PM" src="https://github.com/user-attachments/assets/e0862f7a-3ea1-4185-8827-fb76939fa7d2" />
  </picture>
</div>

**Figure 1:** <em>HyperCube visualizer window showing the white light image of IRAS F23365+3604 (left) and spectrum of spaxel (x,y)=(16,10) zoomed into the Hα-[N II] region of the spectrum (right). The solid green and orange dashed lines overlaid on the spectrum represent the currently-defined model and model components, respectively.</em>

Now that a linear continuum model has been placed, we can start to place the Gaussians. With the cursor at the position of the peak of the [N II] 6548 Å line (redshifted to ~6970 Å in this case, press the `g` key to initialize a Gaussian model: horizontal mouse movement affects the Gaussian width, vertical mouse movement affects the amplitude. When you are satisfied with the Gaussian, press the `g` key again to lock it. Repeat this for the other two emission lines Hα and [N II]6584 Å. Congratulations, you have now interactively specified the initial parameter values for a spectral model composed of a continuum line and three Gaussians! 

### Final Preparations and Fitting the Cube
To inspect the parameters of the model, bring the fit parameters window to the foreground (it should already be open but hidden behind the visualizer window). Scroll down until you see the "Spectral Region 1" panel, containing all of the initial parameter guesses for your model. If you were to go back to the visualizer window and place a line+Gaussians to, say, the [S II] line doublet in the same spectrum, you would see a "Spectral Region 2" panel in the fit parameters window. Click the `Line Name` button in the first row, corresponding to the first Gaussian you placed. This will bring up the "Line Name and Parameter Constraints" window for this emission line (**Fig. 2**). Replace "Line 0" with a name of your choice and press enter -- a green checkmark will notify you the name has been accepted -- then close the window (red 'x' or `esc` key). Repeat this for the other two lines. Next, specify the rest wavelength for your emission lines (in the same units as in your spectra) by clicking the `λ_rest` buttons for each line: 6548, 6563, and 6584 Å. **Save this configuration for later use by pressing `Cmd+S` (mac) or `Ctr+S` (win).**

<div align="center">
  <picture>
<img width="785" alt="Screenshot 2025-04-15 at 1 13 48 PM" src="https://github.com/user-attachments/assets/898b9316-a4df-4436-84ad-713949a2be28" />
  </picture>
</div>

**Figure 2:** <em>The Fit Parameters window displays all model parameter information as well as pertinent information about the observation and the selected spaxel. Here, one of the line name buttons has been pressed, bringing up the Line Name and Parameter Constraints windows.</em>

For this simple example, let's leave the parameter limits at their initial values and forego specifying any relational constraints between model parameters. We can input the observation details at the top of the Fit Parameters window, in the "Observation Data" panel. The tool will attempt to scrape the source name from the FITS header, but if that fails, or if you want to change the name, click the Source Name button. For this observation, the Source Redshift is 0.064 and the Resolving Power is 4000. Now we are ready to fit the cube! Press the `Fit Cube` button in the "Spectral Fitting" panel at the top right of the window. For this example, we are using a cropped version of the full cube, containing only around 800 spaxels, so the fit will only take a few seconds to complete. 

### Inspecting the Fit
First, let's visually inspect the model fit to the spectrum by bringing back the Visualizer window. Unlock the current spaxel in the white light image by pressing `L`, then, as the cursor moves around the image the Spectrum Viewer window will show the spectrum and the best fit model (**Fig. 3**). The total model is represented with a solid red line and the individual Gaussians are each assigned a unique color. The reduced chi-square value of the fit is shown to the top-right of the Spectrum Viewer panel. 

<div align="center">
  <picture>
<img width="947" alt="Screenshot 2025-04-15 at 1 24 10 PM" src="https://github.com/user-attachments/assets/2cef7095-efaf-425b-a806-6a7d149482d8" />
  </picture>
</div>

**Figure 3:** <em>After fitting the cube, the Spectrum Viewer panel (right) shows the spectrum + best-fitting model in the spaxel currently highlighted in the Image Viewer panel (left). The total model is shown in red, model components are shown with dashed lines colored-coded according to the Fit Parameters window. The reduced Chi-square value for the fit is shown to the top-right of the Spectrum Viewer panel.</em>

To inspect the fitted values of each parameter for each model component, e.g., the continuum slope or the amplitude (flux density) of the Hα line, bring the Fit Parameters Window forward. Like the Spectrum Viewer panel, the fitted values will update in realtime to reflect the best-fit model in the currently-selected spaxel in the Image Viewer panel. Any of the parameter_fit buttons can be pressed to show the spatially-resolved map of that fitted parameter in the Image Viewer panel (**Fig. 4**). 

<div align="center">
  <picture>
<img width="645" alt="Screenshot 2025-04-15 at 1 39 13 PM" src="https://github.com/user-attachments/assets/8b95f6ed-698c-49f4-b1dc-b8eb1fa4cf06" />
  </picture>
</div>

**Figure 4:** <em>Visualizing the fitted values of your model spatially is as easy as clicking the button corresponding to the parameter.</em>

If you are not satisfied with the fit, you can specify parameter limits or parameter constraints to obtain a better result. You can also reset your initial parameter guesses by pressing the red delete button at the far-right of each row of buttons, and retry drawing the Gaussian on the spectrum. In the current version, it is recommended to restart the program, load the FITS file, open the Fit Parameters window and use `Cmd-O` (mac) or `Ctr-O` (win) to load your original configuration.

### Outputting the Fit ###
If you are satisfied with the cube fit, you can save the result to a csv table and/or a multi-extension FITS file. To do this, bring forward the Fit Parameters window and look at the Spectral Fitting panel. Here, you will find the `Save Cube Fit` and `Save Fit to FITS File` buttons, which output the fit to csv and FITS files, respectively. You can always view your fit result in HyperCube at a later time by opening the tool, loading the original FITS cube, opening the Fit Parameters window, and clicking the `Load Cube Fit` button.

**Units are explicit in both outputs.** Every CSV column that has a physical unit carries it in its name:

| suffix | meaning | examples |
|---|---|---|
| `_A` | Ångström | `cen_fit_A`, `cen_std_A`, `rest_wavelength_A` |
| `_kms` | km/s | `vel_fit_kms`, `vel_std_kms`, `sigma_fit_kms`, `sigma_std_kms` |
| `_flux` | the cube's flux unit | `amp_fit_flux`, `amp_std_flux`, `qa_noise_flux` |
| `_fluxperA` | flux unit per Ångström | `cont_region1_slope_fit_fluxperA` |
| `_deg`, `_pix` | degrees, spaxel index | `RA_deg`, `Dec_deg`, `spaxel_x_pix` |

The flux unit itself is the cube's `BUNIT`, written into the CSV's scale/units header row next to the wavelength and velocity units. **Velocity dispersion is reported in km/s throughout** — in the GUI, in the CSV, and in the FITS maps (Ångström remains the internal storage unit, and is recoverable from the companion centroid column). In the FITS output every map carries a `BUNIT` header, `SIGMA_<line>` is in km/s, and each value map is paired with a `*_STD_<line>` 1σ uncertainty map. Uncertainty columns (`amp_std`, `cen_std`, `vel_std`, `sigma_std`) and their provenance (`noise_source`, `noise_median`, `noise_npix`) are described under [Measurement errors & parameter uncertainties](#measurement-errors--parameter-uncertainties). CSV and FITS products written by earlier versions still load.

# Interactive Usage Mode
One of the two main use cases for HyperCube is intuitive/dynamic spectral fitting (or data exploration), the other being automated/batch spectral fitting (described below in the [Pipeline Usage Mode](#pipeline-usage-mode) section). In interactive mode, spectral fitting more or less follows the steps outlined in the [Quick Start Guide](#quick-start-guide), i.e., we dynamically place continuum+line sets using the cursor and specify parameter values, names, limits, and constraints using the interactive GUI. *This usage mode is ideal for quick exploration of data cubes where visual feedback is critical.*

### Initiating Models Interactively

Each spectral region is built from a **continuum** plus any number of **Gaussian** emission lines drawn on top of it. Three continuum types can be drawn interactively over the spectrum:

- **Linear** (`d`) — press `d` at one end of the continuum and again at the other to set a straight line.
- **Spline** (`s`) — press `s` to drop interpolation knots (connect-the-dots); press `Enter` to finalize, `Backspace` to undo the last knot, `Esc` to cancel.
- **Polynomial** (`p`) — press `p` at the start and end of a wavelength range to fit a Chebyshev polynomial to the data in that range.

Once a continuum is placed, press `g` over a line peak to begin a Gaussian (horizontal motion sets the width, vertical motion the amplitude) and `g` again to lock it. Each continuum + line set becomes a "Spectral Region" panel in the Fit Parameters window.

### Adjusting Model Parameters Interactively

Every value in a Spectral Region panel is an editable button. Click a continuum cell (slope, intercept, knots, polynomial degree) or a line cell (amplitude *f*<sub>λ,0</sub>, observed wavelength λ<sub>obs,0</sub>, width σ<sub>0</sub>, and their min/max limits) to type a new value; the model overlay updates immediately. Line widths (σ) are shown and entered in **km/s** (velocity dispersion). Click `Line Name` to name a line, assign it to a [kinematic group](#relational-constraints), and set relational constraints, and `λ_rest` to set its rest wavelength.

### Per-Spaxel Fit Correction

After fitting the whole cube, individual spaxels can be corrected without re-fitting everything. Lock onto a spaxel (`L`) and use the per-spaxel controls in the **This Spaxel** group at the top of the Fit Parameters window:

- **Fit This Spaxel** — (re)fit the model to just the locked spaxel.
- **Clear Spaxel Fit** — remove this spaxel's fit and enter *edit mode*. The old (poor) model is greyed for reference, and a dialog lets you seed the edit from the spaxel's existing fit or from the base template. You can then re-specify the continuum type and emission-line initial guesses for **this spaxel only** — graphically (`d`/`s`/`p` *replace* the region's continuum in place; `g` snaps to the nearest existing line and updates its guess) or by editing the panel cells. While editing, the model **schema is locked** (the same number of regions/lines and the same line names as every other spaxel) so spatially-resolved line maps never develop NaN holes.
- **Cancel Edit** — discard an in-progress per-spaxel edit and restore the original fit.
- **Toggle Edited** — overlay translucent blue boxes on the image marking every spaxel that has a per-spaxel edit.

Per-spaxel edits are remembered and reloaded whenever you lock back onto that spaxel. To wipe every fit for the whole cube while keeping the model definition (so you can re-fit), use **Clear All Fits** in the cube-level controls.

### Saving and Restoring Sessions

The entire tool state — cube, model, fits, per-spaxel edits, stellar results, and display/background settings — can be saved to a `.hcsession` file with **Save Session** and restored later with **Load Session**, so you can close HyperCube and resume exactly where you left off.

### Relational Constraints

Open the **Line Name and Parameter Constraints** window (click any `Line Name` button) to tie a line's parameters to other lines in the model. Up to five constraints per line can be entered, using the syntax:

```
param  op  param_[line name]
param  op  factor * param_[line name]
```

where `param` is one of `amp` (amplitude), `flux` (integrated line flux), `sigma`, `cen` (centroid), or `vel` (velocity), and `op` is one of `<=`, `<`, `>=`, `>`, `==`. For example:

- `amp <= amp_[Halpha]` — keep a component fainter than another line
- `sigma >= sigma_[Halpha]` — keep a component broader than another
- `amp <= 0.33 * amp_[Halpha]` — fixed *amplitude*-ratio bound
- `flux == 2.94 * flux_[[N II]_6548]` — fixed **integrated-flux** ratio (e.g. the [N II] 6584/6548 doublet)
- `flux == 0.44..1.45 * flux_[[S II]_6731]` — flux ratio confined to a **range** (e.g. the density-sensitive [S II] 6716/6731 doublet); the fit picks the best ratio within the bounds
- `vel == vel_[nii_1]` — tie velocities (shared kinematics)

> **`amp` vs `flux`:** an `amp` constraint fixes the *peak height* ratio, which equals the flux ratio only when the two lines share the same width. A `flux` constraint fixes the *integrated* flux ratio directly (flux = amp·σ·√2π), automatically accounting for the fitted widths — so it gives the exact ratio even if the σ's differ or vary. Use `flux` for fixed atomic doublet ratios.

A **?** help button lists the available parameters, operators, and the lines currently in the model. The **Auto-suggest constraints** button proposes sensible constraints based on the components' initial guesses, which you can review before applying. Constraints reference lines by **name**, so forbidden-line names containing brackets (e.g. `[S II]_6716`) are fully supported.

#### Kinematic Groups (K-groups)

For multi-component fits it is often desirable for several lines to share one kinematic solution. Assign lines to the same **K-group** (K1–K5, via the checkboxes in the Line Name window) to tie their **velocity and velocity dispersion** together during fitting — every member shares the same velocity and the same km/s dispersion, with widths and centroids scaled by each line's rest wavelength. One member is the **reference** that carries the group's free kinematics, and the window indicates which line that is. A group anchors on its first line in model order until you choose otherwise — press **Make this line the reference for its K-group** in the Line Name window of whichever line should hold it. The reference is the line whose velocity and dispersion stay free while the rest of the group is tied to them, so it wants to be the best-measured line in the group rather than whichever happens to come first: if a group spans Hγ and Hα, anchoring on Hα is usually what you want. Promoting a line rewrites the group's ties immediately, rescaling each σ ratio to the new anchor's rest wavelength. The promotion belongs to the group it was made in — moving a line to another group, or out of grouping entirely, drops it, and re-running Smart Constraints (which rebuilds the grouping from scratch) resets every anchor to the default. K-groups are a shortcut that writes the equivalent relational constraints for you, and they coexist non-destructively with any manual sigma constraints — a manual constraint is held inactive while the line is grouped and re-activates if you remove it from the group.

> **Note:** velocity dispersion (σ) is displayed and entered in **km/s** throughout the GUI and is included (alongside the wavelength-space values) in the CSV and FITS output.


# Fitting Techniques

HyperCube's per-spaxel fitting is powered by [`lmfit`](https://github.com/lmfit/lmfit-py): named parameters with bounds, algebraic constraints between parameters (`.expr`), per-parameter uncertainties from the covariance matrix, and χ²/BIC statistics. On top of that core, HyperCube layers several techniques aimed at getting **accurate, consistent fits across a whole cube** without hand-tuning every spaxel, and it feeds lmfit a real per-pixel noise model so those covariance uncertainties mean something (see [Measurement errors & parameter uncertainties](#measurement-errors--parameter-uncertainties)). This section summarizes them; the constraint syntax itself is covered under [Relational Constraints](#relational-constraints).

### Relational constraints & kinematic groups

Tie parameters across lines to encode physics and reduce free parameters:

- **Amplitude / flux / width / centroid relations** — e.g. `amp <= amp_[Halpha]`, `sigma >= sigma_[Halpha_b]`, amplitude bounds `amp <= 0.33 * amp_[Halpha]`, and fixed **integrated-flux** ratios `flux == 2.94 * flux_[[N II]_6548]` (exact even when widths differ, since flux = amp·σ·√2π).
- **Velocity ties, windows, and one-sided bounds** — `vel == vel_[B]` (exact tie), `vel == vel_[B] +- 300` (within ±300 km/s), `vel <= vel_[B] + 300` / `vel >= vel_[B] - 300` (one-sided). These are realized internally as a bounded additive centroid offset (`cen_A = (restA/restB)·cen_B + offset`), so they are physically correct for lines at different rest wavelengths.
- **Kinematic groups (K1–K5)** — a one-click shortcut that ties a set of lines to share one velocity *and* one km/s velocity dispersion (see [Kinematic Groups](#kinematic-groups-k-groups)).

### Sequential core→outflow fitting

Lines with a narrow core **and** a broad/outflow component (an AGN/starburst outflow, a second velocity system) are degenerate for a single joint fit: from one static initial guess the broad component often collapses to its σ-minimum on top of the core and the offset outflow flux is left unmodeled — and because the outflow can sweep from one side of the core to the other across the field (and from sub-dominant to dominant), no single initial guess works everywhere.

The **Sequential** toggle (in the Spectral Fitting toolbar) breaks this degeneracy structurally. For every line that has a broad (`_b`) partner it fits in stages: **(1)** fit the narrow core(s) with the broad amplitudes suppressed; **(2)** freeze the core and continuum and fit each broad component to the *residual*, where it is the only feature and cannot collapse onto the core; **(3)** a joint polish from that solution. It needs no per-galaxy tuning and leans on the (typically tight) bounds you already set on the narrow component. It has no effect on lines without a broad partner, and toggling it off reproduces the standard joint fit exactly. Rectify's refits inherit the staged fit automatically.

### Measurement errors & parameter uncertainties

Every uncertainty HyperCube reports — `amp_std`, `cen_std`, `vel_std`, `sigma_std` — comes from the **covariance matrix of a fit weighted by the per-pixel flux measurement errors**. This section states exactly where those flux errors come from and what the resulting uncertainties do and do not include.

**Where the flux errors come from.** At cube ingest HyperCube looks for a per-pixel 1σ flux uncertainty, in this order, and prints what it found:

1. **An error extension of the science file** — `ERR`, `VAR`, `VARIANCE`, `IVAR`, `STAT`, `FLUXERR`, … (JWST `s3d` products, MUSE, and similar). Variance and inverse-variance are converted to 1σ; entries that cannot be a real error (negative variance, non-positive inverse variance) are discarded rather than trusted, and those pixels drop out of the fit.
2. **A sidecar file next to the cube** — the KCWI DRP convention `*_icubes.fits` → `*_vcubes.fits`, plus generic `*_err`, `*_var`, `*_sigma`, `*_ivar` companions. The shape must match the cube.
3. **An empirical estimate** — used when neither exists, or when what exists turns out to be unusable (for example a PSF-subtracted product whose `ERR` extension was never populated). The noise is measured **per spaxel, from the line-free pixels inside each fit window**, using the DER_SNR estimator (Stoehr et al. 2008): a MAD-style statistic built on second differences, σ ≈ 1.4826/√6 · median(|2f<sub>i</sub> − f<sub>i−2</sub> − f<sub>i+2</sub>|). The second difference removes any smooth continuum, so a sloped or curved baseline cannot inflate it, and the median makes it robust to the minority of pixels carrying emission lines (which are additionally masked out to ±3σ of their initial guesses).

Detection is automatic and silent. To override it, press **`Measurement Errors…`** in the *Cube:* row of the Fit Parameters window: choose any extension of any FITS file, declare whether its values are 1σ / variance / inverse variance, or force the empirical estimate. The choice is stored in `.hcsession` files, and **which source was actually used is written into every row of the fit output** as `noise_source`, together with `noise_median` (the median σ over the pixels used) and `noise_npix`.

**How the errors enter the fit.** The fit is weighted by 1/σ inside the fit windows and **zero-weighted outside them**. This matters: HyperCube's model is identically zero outside a continuum region, so out-of-window pixels contain no information about any parameter — but they used to enter χ² anyway, and lmfit's default `scale_covar` rescaling then inflated every reported uncertainty by that irrelevant residual. With a real noise model in hand the covariance is left unscaled (`scale_covar=False`), so the reported `*_std` are honest 1σ measurement uncertainties.

**What this does and doesn't include.** These are the standard Levenberg–Marquardt covariance errors: symmetric, Gaussian, and propagated through any relational constraints or kinematic groups you have set (tied parameters get their uncertainties propagated through the constraint expressions). They capture photon/detector noise as described by your error cube or the empirical estimate. They do **not** capture model inadequacy — an unmodelled broad wing or a misplaced continuum shifts the best-fit values without necessarily widening the covariance. Two practical caveats:

- **Bounded parameters.** A parameter sitting against its `min`/`max` gets a slightly optimistic error (~7% low in testing) from lmfit's bounded-parameter transform, and a symmetric ±σ is the wrong shape for a truncated distribution anyway. Treat uncertainties on railed parameters with suspicion — the [Quality Map](#calibrated-fit-quality-metrics--the-quality-map) is the better guide there.
- **Integrated flux.** Line flux is √2π·amp·σ<sub>λ</sub>, and amp and σ are strongly anti-correlated. Only the diagonal of the covariance is written to the output, so combining `amp_std` and `sigma_std` as if independent will misestimate the flux error.

Validated by Monte Carlo against the true scatter of repeated refits: the reported `vel_std` reproduces that scatter to 0.3% for unbounded parameters.

**Reading the χ².** With weights in place, `rchisq_w` — the reduced χ² over the weighted pixels — is the meaningful goodness-of-fit number and sits near 1 for a good fit with a correct noise model. It is available as a Quality Map and as a Rectify/Mask criterion. lmfit's native `rchisq` divides by the *full* spectrum length rather than the fitted pixels, so it reads low by roughly the fraction of the spectrum your fit windows cover and should not be compared to 1.

### The S/N mask

The S/N map gates which spaxels get fit. Per spaxel and per line in the model it is

> S/N = (95th-percentile flux in a window around the line centre − **the local continuum**) ÷ (MAD noise in the two continuum flanks)

taking the best line for each spaxel. The continuum subtraction is what makes this a measurement of the *line* rather than of *brightness*: without it the ratio is (continuum + line)/noise, so any bright continuum source passes at any threshold even with no emission whatsoever — a foreground star in UGC 05101 scored S/N = 43 on a spectrum containing no Hα at all.

The flanks are not assumed to be line-free, because they frequently are not — at z ≈ 0.04 both [N II] lines fall inside Hα's default flanks. The continuum level is therefore taken after two asymmetric clipping passes that drop channels more than 3σ *above* the running median, removing line cores while leaving the noise distribution and any absorption intact. Without that clipping the flank median is biased upward by exactly the lines being measured, and that bias is then subtracted straight off the line.

Because the numbers this produces are smaller than the un-subtracted ones it replaced, a threshold carried over from an older version of HyperCube is a **stricter** cut than it used to be; re-check it against the mask contour. Maps cached or stored in a session under the old definition are discarded rather than reused.

### Masking spaxels & the fit gate

**Mask Spaxels** (the `Cube:` row) is two tabs over one shared Visualize / Accept / Unmask.

*Map criterion* keeps spaxels where a chosen map passes a test — any fitted parameter or quality map, or a **channel map built from the live C window** minus any locked X/V sidebands. The channel map needs no fit, so the dialog opens on a cube you have not fitted yet, and it is rebuilt on every access: the dialog is modeless, you are expected to move the C window while it is open, and a cached map would mask on a stale selection.

*S/N* offers every line in the model with **its own threshold per line**, one operator, and an **any / all** combiner. One universal cut is wrong whenever lines differ in brightness — [O III] 4959 is a third of 5007 by atomic physics, so a cut that keeps 4959 is far too lax for 5007. On a graded synthetic pair, `all` at 15/5 keeps 146 spaxels against 31 for a universal 15/15. Use `any` when one detected line is enough; use `all` when a line *ratio* has to be measurable in the same spaxel.

> ⚠️ **This changed in v0.5.0.** Masking used to be display-only. There is now a single definition of which spaxels are fitted — the S/N gate **AND** everything hidden by Mask Spaxels — and the main cube fit, the Rectify re-fit, the stellar fit and Rectify's count preview all use it. **Masked spaxels are no longer fitted.** *Unmask* clears the S/N gate too, so the fit is never left silently gated with nothing on screen to say so.

### Self-sky subtraction

Some reduced cubes keep an under-subtracted airglow line. The case this was built for sits at 6866 Å — 2.1 Å (≈92 km/s) from [N II] 6548 at *z* = 0.048232 and **2.5× brighter than the real line** — and the fitter picks it up in essentially every spaxel. `Cube: → Self-Sky…` builds a sky spectrum from the cube's own object-free spaxels and subtracts it.

The residual is **additive**: its excess over the local sidebands is flat at ≈0.008 while the continuum beneath it varies by 50×. Multiplicative artifacts are out of scope.

- **Object-free is defined by stacked vetoes, ANDed.** Faintest *N*% by white light, white light below a value, a channel map, or any fitted line or quality map — as many as you like, because a veto built on one line cannot remove an outflow visible only in another. The right criterion is "faint in the continuum *and* faint in every line I care about"; on the driver cube adding an Hα+[N II] veto removed 2270 spaxels the continuum veto had kept.
- **The combining statistic is a median, never a mean.** Continuum-faint does not mean line-free: 12.7% of a faintest-60% pool still has Hα+[N II] at S/N > 3. The mean carries that through (0.63% bias at Hα against the median's 0.37%), so no mean is offered.
- **A veto may not be drawn on the artifact's own window.** A channel map at 6866 Å *is* a map of the sky residual, so vetoing on it biases the pool low — invisibly. Point the guard at the artifact with **← set from C window** and an overlapping veto is refused, not warned about.
- **Sky variance propagates into σ**, as var(median) = (π/2)·var(mean). The fit is 1/σ-weighted, so omitting it over-weights the corrected region.
- Subtraction happens only where a spaxel has data — off-detector voxels are exact zeros, not NaN — and an unusably small pool refuses to produce a sky rather than producing a noisy one.

Per-column mode fits the residual's ≈30% across-slice gradient, which matches the DRP modelling sky per slice. On the driver cube the faint-region artifact goes from 0.00772 to **−0.00001** and the gradient flattens ≈9×, while the real [N II] 6548 survives. Provenance rides in the fit CSV's header and the window title, so a fit from a corrected cube is distinguishable from one that is not; **Revert** restores the cube bit-identically.

Headless, `--selfsky 60` does the same with the faintest-*N*% recipe. The batch rebuilds the mask from its arguments rather than loading a saved array, so a run stays reproducible and a stale mask cannot be applied to the wrong cube.

> This is a workaround, not a fix. For KCWI/KCRM the root cause is upstream — `SubtractSky` scales the sky master by exposure time and never fits the airglow amplitude. Use it for cubes that will not be re-reduced.

### Resolving power

`R` is read from the cube at ingest and shown on the **R:** button; its tooltip says where the value came from, so a derived number is never mistaken for one you typed. R sets the width HyperCube convolves the stellar templates to before pPXF fits a LOSVD, so getting it wrong biases σ\*.

Almost no IFU cube states R outright, but every one records the configuration that determines it, so HyperCube works down three tiers: a header keyword giving R (or a resolution element) directly; then the named instrument — KCWI/KCRM grating + slicer, MIRI MRS channel + band, NIRSpec disperser, the MUSE LSF polynomial; then nothing, leaving the field blank.

For KCWI the stored quantity is the **resolution element** (FWHM in Å) per grating, not R, because a grating's FWHM is fixed while R = λ/FWHM slides across the band. The survey's own coadds confirm this: RH1 and RH2 carry `SPECRES` 1800 @ 6150 Å and 2025 @ 6900 Å, which is one FWHM (3.42 vs 3.41 Å) seen at two wavelengths rather than two resolving powers. Entries for BH1–3, RH1 and RH2 are calibrated against those headers; the rest are nominal and say so wherever they are used.

A coadded **supercube** gets no value at all, deliberately: it keeps no instrument keywords, and its constituent gratings have different resolutions, so no single R describes it. HyperCube reads the combine step's `HISTORY` and prints which gratings went in and their resolution elements, leaving you to set R for the region you are fitting. If R is left blank, pPXF assumes 3000 — and now says so rather than assuming it silently.

### Calibrated fit-quality metrics & the Quality Map

Reduced χ² alone — even the weighted `rchisq_w` — averages the line cores together with the far more numerous continuum pixels, so a badly-fit line profile can hide inside a good-looking global number. HyperCube therefore computes, per spaxel, a set of calibrated, scale-free quality statistics that compare the line cores against the off-line continuum directly:

- **Core/continuum ratio** — mean(residual²) in the line cores ÷ in the continuum; ≈1 = good, ≫1 = poorly-fit profile. Noise-independent and the headline metric.
- **Signed-residual z** — direction and significance of leftover flux: large positive = missed flux (an unmodeled component), negative = over-subtracted.
- **Runs z** — a runs test on the residual signs flags systematic *shape* errors even when amplitudes look right.
- **Calibrated continuum χ²** — reduced χ² over off-line pixels (≈1 for a good fit); the calibration anchor.

The **S/N mask** toggle on the image toolbar draws the current threshold as a red
contour over the map, and keeps it there through colormap, stretch, rotation,
flip, spatial-mask and parameter-map changes. Loading a template turns it on, since
the template's own S/N threshold has just been applied.

The **Quality Map ▾** button renders any of these — plus the weighted `rchisq_w` and the native rChi² — as a cube map, so you can *see* which spaxels actually failed. These columns are written to the CSV/FITS output.

### Rectify Bad Fits

**Rectify Bad Fits** *repairs* the spaxels a cube fit got wrong, rather than re-fitting the whole cube from scratch. It operates only on spaxels that already have a fit, leaves the good ones untouched, and replaces only the bad ones in place. Press it after a `Fit Cube` to clean up the failures the [Quality Map](#calibrated-fit-quality-metrics--the-quality-map) reveals.

It works in four steps:

1. **Decide what counts as a bad fit.** One table lists the [calibrated quality metrics](#calibrated-fit-quality-metrics--the-quality-map); tick the ones that matter and give each its own operator (`>`, `<`, `|·| >`, `|·| <`) and limit, so the number you type means what it says — a z-score defaults to `|·| > 3`, a ratio to `> 2`. The selector at the top chooses how they combine: **any** (one tripped rule marks a spaxel for repair) or **all** (every ticked rule must trip). A live count underneath shows how many spaxels will be re-fit, updating as you change anything.

   That choice governs *which spaxels are repaired* only. A spaxel must always pass **every** ticked rule to be trusted as a seed — an AND-gate, so a severe failure in one metric cannot be outvoted by several healthy ones. Loosening what gets repaired must not loosen what does the repairing.
2. **One table drives everything.** The spaxels that fail are the ones re-fit; the spaxels that pass are the only ones allowed to seed a repair or to be accepted as an improvement. Because it is a single rule, the spaxels being repaired and the spaxels trusted to repair them can never disagree. The same criteria also produce a composite score — the mean distance from each metric's ideal, in units of its own limit — which ranks fits against each other. Spaxels below the SNR threshold are skipped entirely; *Also re-fit failed (non-finite) spaxels* controls whether fits that produced no usable metrics at all are repaired or left alone.
3. **Seed from the good fits nearby.** Each bad spaxel is re-fit from initial guesses (amplitude, centroid, width, and the region-1 continuum) copied from its best *good* neighbors, clamped to each parameter's bounds — a spatial-smoothness prior that exploits the spatial coherence of real kinematic fields. Three things make this stronger than a single lookup:
   - **The best *K* neighbors are each tried as an independent starting point** (default 3), and the best result wins. The search stops at the first seed that lands a good fit, so raising K costs nothing on easy spaxels.
   - **The search grows outward** — radius 1 is the 8 immediate neighbors; if none is good it tries radius 2, then 3 (configurable). Without this the middle of any bad patch larger than ~3×3 has no good neighbor at all and falls back to the generic initial guess, which is exactly where the spatial prior is most needed. Nearer neighbors are always preferred over better-scoring distant ones.
   - **Repairs propagate.** A rescued spaxel immediately becomes a valid donor for its neighbors, and spaxels are processed most-good-neighbors-first, so repairs flood inward from the rim of a bad region rather than stopping at it. Passes repeat until one rescues nothing (default cap 5).
4. **Keep only the winner, and never go backwards.** Every candidate is evaluated without committing, and **the spaxel's existing fit competes as one of the candidates**. Only a strictly better-scoring result is written back, so a rectify pass cannot degrade a spaxel — which is also what makes iterating safe. If every neighbor seed still fails, Rectify falls back to a small set of physically-motivated restarts for the known two-component (core + broad) failure modes — **narrow-only**, **broad-only**, **swapped**, and **equal-split**.

Every repaired row records where it came from: `rectify_pass`, `rectify_seed` (e.g. `neighbour(12,34)@r2`, `restart:swapped`, `incumbent`) and `rectify_score`, so a rectified fit is auditable rather than indistinguishable from an ordinary one.

Constraints, kinematic groups, and (if enabled) [sequential core→outflow staging](#sequential-coreoutflow-fitting) all carry through to the rectify refits automatically. The combination of a calibrated metric, a spatial prior, and targeted restarts resolves most degenerate/initialization failures without manual intervention; the handful that survive can be cleaned up with [per-spaxel correction](#per-spaxel-fit-correction).


# Stellar Kinematics with pPXF

HyperCube can model the **stellar continuum** and recover the stellar line-of-sight velocity (V) and velocity dispersion (σ) using the Penalized Pixel-Fitting method ([pPXF](https://pypi.org/project/ppxf/); Cappellari 2017). The stellar fit is integrated as a new **continuum type**: pPXF fits a combination of stellar templates convolved with the line-of-sight velocity distribution (LOSVD), and emission-line Gaussians are added on top exactly as for the linear/spline/polynomial continua.

Two template libraries ship with HyperCube (in the `indo_us_library/` and `eMILES/` folders):
- **Indo-US** — empirical stellar spectra (Valdes et al. 2004); ideal for pure kinematics.
- **eMILES** — single stellar population models with wide wavelength coverage.

### Fitting a stellar continuum to a spaxel
1. Set the **Source Redshift** and **Resolving Power** in the Observation Data panel — pPXF needs both. HyperCube auto-fills them from the FITS header when the relevant keywords are present.
2. Lock onto a spaxel (`L`) and click **Stellar Template…** in the *This Spaxel* group. The Stellar Templates window lets you choose a library (showing its wavelength coverage and resolution against your data's observed and rest-frame ranges), the fit range, the number of LOSVD moments (V, σ or V, σ, h3, h4), additive/multiplicative polynomial degrees, an initial σ guess, and whether to mask emission lines.
3. Click **Fit**. pPXF fits the spaxel, a **Stellar** spectral-region panel appears spanning the fit range with editable **V**, **σ**, and **scale** cells, and the best-fit stellar model is overplotted on the spectrum.

Editing the V, σ, or scale cells re-renders the model instantly (no re-fit); **Refit Stellar** re-runs pPXF for the current spaxel. Add emission lines with `g` on top of the stellar continuum — when you fit, the stellar baseline is held fixed and the lines are fit to the stellar-subtracted residual.

### Stellar maps across the cube
Press **Fit Cube** to fit the stellar continuum *and* the emission lines across every spaxel in one pass (or **Fit Stellar (Cube)** for kinematics only). The **V map** and **σ map** buttons in the Stellar panel then display the spatially-resolved stellar velocity and dispersion maps; the buttons themselves show the current spaxel's best-fit V and σ as you move the cursor. A **Cancel** button in the progress bar stops a long fit, keeping the spaxels already fit.

Stellar results are included when you **Save Fit** (CSV) and **Save Fit to FITS File** (as `stellar_vel`, `stellar_sigma`, … image extensions), and are fully preserved in saved sessions.


# Pipeline Usage Mode

HyperCube can run without a GUI: `hypercube_batch` expands a rest-frame
**template** across a **manifest** of cubes and fits each one, writing products
the GUI reopens. Full design in `HyperCube_Templates_SPEC.md`.

```
python -m hypercube_batch --template MAUNA_MUSE_core.hct.csv \
                          --manifest MAUNA_MUSE.manifest.csv \
                          --out runs/ [--targets A,B] [--cores 8] [--dry-run]
```

### Initiating Models with Configuration Files

A **template** (`*.hct.csv`) describes a model the way physics does — rest
wavelengths, velocity offsets from systemic, *intrinsic* velocity dispersions,
bounds, K-groups, constraints — so one file applies to every galaxy observed
with a given instrument. Nothing galaxy-specific lives in it. The three things
that are come from elsewhere: redshift and cube path from the **manifest**
(`*.manifest.csv`, one row per target), and the instrument's resolution from
**the cube itself**.

Expansion combines them. Rest wavelengths are placed at `rest·(1+z)·(1+v/c)`;
intrinsic σ is broadened by the instrument LSF **at each line's own observed
wavelength** (which is why one template is portable — MUSE's resolving power
varies 59% across a single model); relative amplitudes are scaled to the cube's
own flux units. Lines and continuum regions outside a given cube's coverage are
dropped, `Line_ID` renumbered, constraints naming a dropped line removed, and
K-groups re-anchored — every one of those recorded in a coverage report rather
than done silently. A `required` flag makes a line or region abort the target
instead of being dropped.

**In the GUI**, the **Save Template** and **Load Template** buttons sit on the
*Output:* toolbar row. Save writes the model you have built; Load applies one to
whatever cube is currently open, placing its lines at that galaxy's redshift and
broadening its dispersions by that instrument's LSF. Both need the cube's
redshift set first (the **z:** button) — that is what converts between the rest
frame the template is stored in and the observed frame the fit works in. Load
reports anything it dropped for coverage and any K-group it re-anchored.

You can also write a template in a spreadsheet: it is a sectioned CSV.

### Batch Processing

`--dry-run` expands and reports coverage without fitting — this is how you find
out which line/galaxy combinations are observable at all before spending any
time on them. Each target writes `<target>_<template>_Fit.csv` in the same
five-section layout **Load Fit (CSV)** reads, plus a coverage report; a
`run_log.csv` records the template, its version, the git SHA, and per-target
timings. A failing target is logged and the run continues, exiting non-zero at
the end.

This does **not** make per-spaxel fitting faster — it is the same kernel and the
same process pool the GUI uses (`build_params` and `run_pool` in
`HyperCube_fit.py`, shared by both). What it buys is unattended queued runs,
reproducibility, and running somewhere other than a desktop.

# Troubleshooting

If you get the "UnboundLocalError: cannot access local variable 'piecewise_model' where it is not associated with a value" error, it means you need to add a model to the HyperCube_ModelFunctions.py file because it doesn't yet include a model for your Nregions+Nlines, e.g., it doesn't have one already for 3 continuum regions with one line each (Nregions=3,Nlines=3) -- you'd need to add it manually (following the syntax of the other models in that script).

# Acknowledging HyperCube
If you used HyperCube in your research, please consider acknowledging the use of the tool by including this text in your publications:

_This research has made use of HyperCube, the interactive analysis tool for integral field spectroscopic data, written by Justin A Kader._
