#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:01:22 2025

@author: justin
"""

# Real module attribute (it used to sit inside the docstring, where nothing could
# read it and it went stale). Keep in step with CHANGELOG.md and the git tag.
__version__ = "0.4.0"



import sys
import numpy as np
import pandas as pd
import ast  # For safely evaluating strings as Python literals
import re
from functools import partial
import csv
import gc
import pickle
import psutil
import os
from multiprocessing import shared_memory
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches

from astropy.io import fits
from astropy.wcs import WCS

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLineEdit, QScrollArea, QGridLayout,
    QWidget, QLabel, QFrame, QMenu, QTextEdit, QMainWindow,
    QHBoxLayout, QMenuBar, QProgressBar, QDialog, QSplashScreen,
    QAction, QSplitter, QFileDialog, QApplication, QGroupBox, QMessageBox,
    QDockWidget, QComboBox, QScrollBar, QCheckBox, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QFontMetrics, QPixmap
from PyQt5.QtGui import QKeySequence

from lmfit import Model, Parameters

import HyperCube_ModelFunctions
import HyperCube_fit  # Qt-free per-spaxel fit kernel (shared by serial + parallel)
import HyperCube_Noise  # Qt-free variance/error-cube discovery + empirical noise
import HyperCube_SmartConstraints as hcsc  # Qt-free auto-constraint/K-group logic
try:
    import HyperCube_pPXF as hcppxf
except Exception as _e:
    hcppxf = None
    print(f"pPXF stellar fitting unavailable: {_e}")







# Global variables for application-wide FITS access
global FITS_HEADER, FITS_DATA
FITS_HEADER = None
FITS_DATA = None

# ── Measurement errors ───────────────────────────────────────────────────────
# ERROR_CUBE holds per-pixel 1σ flux uncertainties with the same shape as
# FITS_DATA (converted from whatever the file stores — variance, error or
# inverse variance), or None when no error cube is in use, in which case each
# spaxel's noise is measured empirically from its own line-free continuum.
# ERROR_INFO records where it came from, for the fit output and the dialog.
global ERROR_CUBE, ERROR_INFO
ERROR_CUBE = None
ERROR_INFO = {'source': HyperCube_Noise.SRC_EMPIRICAL, 'mode': 'empirical'}

global viewing_fit
viewing_fit = False

global spectral_count
spectral_count = None

global df_obs, df, df_cont, df_fit

global fit_results
fit_results = []
constraints = []

# ── Per-spaxel model overrides ───────────────────────────────────────────────
# base_df_cont / base_df hold the locked schema template that "Fit Cube" uses.
# spaxel_overrides maps (x, y) -> {'df_cont', 'df'} for spaxels the user has
# manually edited.  The override keeps the SAME schema as the base (same number
# of regions and lines, same region_IDs / Line_IDs / Line_Names) — only values
# (continuum type/shape, line initial guesses & bounds) differ.  This keeps line
# maps hole-free: every spaxel still fits the same set of lines.
global base_df_cont, base_df, spaxel_overrides
base_df_cont = None
base_df = None
spaxel_overrides = {}

# ── Stellar (pPXF) state ─────────────────────────────────────────────────────
# STELLAR_CACHE[(x,y)] holds the cached "optimal template" + grids returned by
# HyperCube_pPXF.fit_stellar, so a stellar region's baseline can be re-rendered
# instantly (on V/sigma/scale edits) and used as a fixed baseline under emission
# lines. _STELLAR_LIBS caches prepared TemplateLibrary objects by name.
global STELLAR_CACHE, _STELLAR_LIBS, df_stellar
STELLAR_CACHE = {}
_STELLAR_LIBS = {}
# Per-spaxel stellar kinematics results (one row per spaxel) for maps/persistence.
df_stellar = pd.DataFrame({})

# Columns a stellar continuum region adds to df_cont (alongside the standard ones).
_STELLAR_COLS = {
    'stellar_library': '', 'stellar_V_0': np.nan, 'stellar_sigma_0': np.nan,
    'stellar_h3_0': np.nan, 'stellar_h4_0': np.nan, 'stellar_scale_0': np.nan,
    'stellar_V_fit': np.nan, 'stellar_sigma_fit': np.nan, 'stellar_h3_fit': np.nan,
    'stellar_h4_fit': np.nan, 'stellar_scale_fit': np.nan, 'stellar_moments': 2,
    'stellar_chi2': np.nan,
}

global spectrum

global snr_map
global snr_value
snr_value = 0

data_observation_init = {'sourcename': [''],
                    'redshift': [''],
                    'resolvingpower': ['']}

df_obs = pd.DataFrame(data_observation_init)


# Sample data for continuum and lines.
# cont_type is 'linear' (Slope_0/Intercept_0 model) or 'spline' (knot-based).
# For spline rows the knots_* columns hold lists of floats and the Slope/
# Intercept columns are NaN; for linear rows the knots_* columns are empty.
# New columns are appended AFTER lineactor so region_ID stays at index 7.
data_cont_init = {'Continuum Name': [],
             'x1': [],
             'x2': [],
             'Slope_0': [],
             'Intercept_0': [],
             'Slope_fit': [],
             'Intercept_fit': [],
             'region_ID': [],
             'lineactor': [],
             'cont_type': [],
             'knots_x': [],
             'knots_y_0': [],
             'knots_y_fit': [],
             # Polynomial (Chebyshev) continuum: degree + coefficient lists
             # (coefficients are over the Chebyshev domain [x1, x2]).
             'poly_degree': [],
             'poly_coef_0': [],
             'poly_coef_fit': []}

df_cont = pd.DataFrame(data_cont_init)

data_lines_init = {'Line_ID': [],
        'Line_Name': [],
        'SNR': [],
        'Rest Wavelength':[],
        'Amp_0': [],
        'Amp_0_lowlim': [],
        'Amp_0_highlim': [],
        'Centroid_0': [],
        'Centroid_0_lowlim': [],
        'Centroid_0_highlim': [],
        'Sigma_0': [],
        'Sigma_0_lowlim': [],
        'Sigma_0_highlim': [],
        'Constraints': np.array(5),
        'kgroup': [],
        'Amp_fit': [],
        'Centroid_fit': [],
        'Sigma_fit': [],
        'region_ID': [],
        'curveactor': []}

df = pd.DataFrame(data_lines_init)


df_fit = pd.DataFrame({})


def _ref_line_name(token):
    """Extract the referenced line name from a 'param_[line name]' token.

    Tolerates line names that themselves contain brackets or spaces, e.g.
    '[S II]_6716' in 'amp_[[S II]_6716]'. The greedy match captures up to the
    final ']', so forbidden-line names parse correctly.
    """
    m = re.search(r'_\[(.*)\]', token)
    return m.group(1).strip() if m else None


# Speed of light (km/s), used to present Gaussian sigma as a velocity dispersion.
C_KMS = 299792.458


def sigma_wl_to_kms(sigma_wl, centroid_wl):
    """Convert a wavelength-space sigma (Å) to velocity dispersion (km/s).

    σ_v = c · σ_λ / λ_obs. Returns NaN if inputs are missing/zero.
    """
    try:
        sigma_wl = float(sigma_wl)
        centroid_wl = float(centroid_wl)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(sigma_wl) or not np.isfinite(centroid_wl) or centroid_wl == 0:
        return np.nan
    return C_KMS * sigma_wl / centroid_wl


def sigma_kms_to_wl(sigma_kms, centroid_wl):
    """Inverse of sigma_wl_to_kms: km/s → wavelength-space sigma (Å)."""
    try:
        sigma_kms = float(sigma_kms)
        centroid_wl = float(centroid_wl)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(sigma_kms) or not np.isfinite(centroid_wl):
        return np.nan
    return sigma_kms * centroid_wl / C_KMS


# ── CSV column units ─────────────────────────────────────────────────────────
# Every exported CSV column that has a physical unit carries that unit in its
# name, and σ is exported as a velocity dispersion (km/s) everywhere (the Å value
# stays recoverable from the companion centroid column). Flux-like quantities
# carry a generic '_flux' token because their true unit is the cube's BUNIT,
# which is written into the CSV's scale/units header block instead.
#
#   _A         Angstrom            _kms   km/s
#   _flux      cube flux unit      _fluxperA  cube flux unit per Angstrom
#   _deg       degrees             _pix   spaxel index (pixels)
#
# `to_export_units()` renames + converts on the way out; `from_export_units()`
# inverts it on the way in and still accepts the pre-units CSVs written by
# earlier versions (σ in Å, with derived *_kms companions).
UNIT_SUFFIXES = ('_fluxperA', '_flux', '_kms', '_deg', '_pix', '_A')

# σ columns stored internally in Å → exported in km/s using this centroid column.
SIGMA_KMS_PAIRS = (
    ('Sigma_0', 'Centroid_0'),
    ('Sigma_0_lowlim', 'Centroid_0'),
    ('Sigma_0_highlim', 'Centroid_0'),
    ('Sigma_fit', 'Centroid_fit'),
    ('sigma_init', 'cen_init'),
    ('sigma_fit', 'cen_fit'),
    ('sigma_std', 'cen_fit'),
)
_SIGMA_COLS = {c for c, _ in SIGMA_KMS_PAIRS}

_COL_UNITS = {
    # df_fit — per-spaxel fit results
    'spaxel_x': 'pix', 'spaxel_y': 'pix', 'RA': 'deg', 'Dec': 'deg',
    'amp_init': 'flux', 'amp_fit': 'flux', 'amp_std': 'flux',
    'cen_init': 'A', 'cen_fit': 'A', 'cen_std': 'A',
    'vel_init': 'kms', 'vel_fit': 'kms', 'vel_std': 'kms',
    'rest_wavelength': 'A',
    'sigma_init': 'kms', 'sigma_fit': 'kms', 'sigma_std': 'kms',
    'noise_median': 'flux', 'qa_noise': 'flux',
    # df — line definitions / initial guesses
    'Rest Wavelength': 'A',
    'Amp_0': 'flux', 'Amp_0_lowlim': 'flux', 'Amp_0_highlim': 'flux', 'Amp_fit': 'flux',
    'Centroid_0': 'A', 'Centroid_0_lowlim': 'A', 'Centroid_0_highlim': 'A',
    'Centroid_fit': 'A',
    'Sigma_0': 'kms', 'Sigma_0_lowlim': 'kms', 'Sigma_0_highlim': 'kms', 'Sigma_fit': 'kms',
    # df_cont — continuum regions
    'x1': 'A', 'x2': 'A', 'knots_x': 'A',
    'Slope_0': 'fluxperA', 'Slope_fit': 'fluxperA',
    'Intercept_0': 'flux', 'Intercept_fit': 'flux',
    'knots_y_0': 'flux', 'knots_y_fit': 'flux',
    'poly_coef_0': 'flux', 'poly_coef_fit': 'flux',
    'stellar_V_0': 'kms', 'stellar_V_fit': 'kms',
    'stellar_sigma_0': 'kms', 'stellar_sigma_fit': 'kms',
    # df_stellar — per-spaxel stellar kinematics (pPXF returns km/s already)
    'stellar_V': 'kms', 'stellar_sigma': 'kms',
}

# Units for the per-region columns of df_fit, keyed on the part after
# 'cont_region{N}_'.
_CONT_COL_UNITS = {
    'x_start': 'A', 'x_end': 'A', 'x_int_start': 'A', 'x_int_end': 'A',
    'slope_init': 'fluxperA', 'slope_fit': 'fluxperA',
    'slope_int_init': 'fluxperA', 'slope_int_fit': 'fluxperA',
    'intercept_init': 'flux', 'intercept_fit': 'flux',
    'poly_coef_init': 'flux', 'poly_coef_fit': 'flux',
    'knots_x': 'A', 'knots_y_init': 'flux', 'knots_y_fit': 'flux',
    'stellar_V': 'kms', 'stellar_sigma': 'kms',
}

_CONT_RE = re.compile(r'^(cont_region\d+_)(.+)$')


def csv_unit_for(col):
    """Unit token for an internal column name, or None if it is dimensionless."""
    col = str(col)
    if col in _COL_UNITS:
        return _COL_UNITS[col]
    m = _CONT_RE.match(col)
    if m:
        return _CONT_COL_UNITS.get(m.group(2))
    return None


def to_export_units(frame):
    """Copy of `frame` with σ converted to km/s and units appended to names."""
    if not isinstance(frame, pd.DataFrame) or frame.empty and not len(frame.columns):
        return frame
    out = frame.copy()
    for sig_col, cen_col in SIGMA_KMS_PAIRS:
        if sig_col in out.columns and cen_col in out.columns:
            out[sig_col] = [sigma_wl_to_kms(s, c)
                            for s, c in zip(out[sig_col], out[cen_col])]
    renames = {}
    for col in out.columns:
        unit = csv_unit_for(col)
        if unit:
            renames[col] = f'{col}_{unit}'
    return out.rename(columns=renames)


def from_export_units(frame):
    """Inverse of `to_export_units`: strip unit suffixes and put σ back in Å.

    Also accepts the older layout (σ in Å plus derived *_kms companions), which
    is detected by a σ column being present both with and without '_kms'.
    """
    if not isinstance(frame, pd.DataFrame):
        return frame
    cols = list(frame.columns)
    out = frame.copy()

    # Old layout: '*_kms' sat alongside the canonical Å column → drop the copy.
    legacy = [c for c in cols
              if str(c).endswith('_kms') and str(c)[:-4] in cols and str(c)[:-4] in _SIGMA_COLS]
    if legacy:
        out = out.drop(columns=legacy)

    renames, to_angstrom = {}, []
    for col in out.columns:
        name = str(col)
        for suf in UNIT_SUFFIXES:
            if not name.endswith(suf):
                continue
            base = name[:-len(suf)]
            if csv_unit_for(base) is None:
                continue                      # not a column we renamed — leave it
            renames[col] = base
            if suf == '_kms' and base in _SIGMA_COLS:
                to_angstrom.append(base)
            break
    out = out.rename(columns=renames)
    for sig_col in to_angstrom:
        cen_col = dict(SIGMA_KMS_PAIRS)[sig_col]
        if cen_col in out.columns:
            out[sig_col] = [sigma_kms_to_wl(s, c)
                            for s, c in zip(out[sig_col], out[cen_col])]
    return out


def sigma_at_spaxel(cx, cy):
    """This spaxel's 1σ measurement errors from ERROR_CUBE, or None if no error
    cube is loaded (the fit then measures the noise empirically)."""
    if ERROR_CUBE is None:
        return None
    try:
        if ERROR_CUBE.ndim == 1:                     # 1D spectrum
            return np.asarray(ERROR_CUBE, dtype=float)
        y = int(np.clip(cy, 0, ERROR_CUBE.shape[1] - 1))
        x = int(np.clip(cx, 0, ERROR_CUBE.shape[2] - 1))
        return np.asarray(ERROR_CUBE[:, y, x], dtype=float)
    except Exception:
        return None


def noise_source_label():
    """Short provenance token for the fit output's `noise_source` column."""
    if ERROR_CUBE is None:
        return None
    return str(ERROR_INFO.get('source') or 'error-cube')


def set_error_cube(sigma, info):
    """Install (or clear, with sigma=None) the measurement-error cube."""
    global ERROR_CUBE, ERROR_INFO
    ERROR_CUBE = sigma
    ERROR_INFO = dict(info or {})
    if sigma is None:
        ERROR_INFO.setdefault('mode', 'empirical')
        ERROR_INFO.setdefault('source', HyperCube_Noise.SRC_EMPIRICAL)


def flux_unit_str(header=None):
    """The cube's flux unit (BUNIT) for the CSV units block; 'unknown' if absent."""
    hdr = header if header is not None else FITS_HEADER
    try:
        bunit = str(hdr.get('BUNIT', '')).strip() if hdr is not None else ''
    except Exception:
        bunit = ''
    return bunit or 'unknown'


def convert_velocity_to_centroid(velocity_constraint, line_name, df, galaxy_redshift):
    """
    Convert velocity-based constraints to centroid-based constraints.
    
    Parameters:
    - velocity_constraint (str): The velocity constraint (e.g., "vel == vel_[sii_1]")
    - line_name (str): The current line for which we're modifying the constraints
    - df (DataFrame): The DataFrame containing line information
    - galaxy_redshift (float): The galaxy's redshift (z)
    
    Returns:
    - str: A new centroid-based constraint.
    """
    # Extract the reference line from an EXACT velocity tie ("vel == vel_[B]").
    # Anchored at the end so windowed/one-sided forms ("vel == vel_[B] +- 300",
    # "vel <= vel_[B] + 300") are left untouched here and handled later as a
    # bounded centroid offset in add_dataframe_constraints_to_params.
    match = re.match(r"\s*vel\s*==\s*vel_\[(.*)\]\s*$", velocity_constraint)
    if not match:
        return velocity_constraint  # Return unchanged if not an exact tie

    ref_line_name = match.group(1).strip()

    # Get rest wavelengths of the current and reference lines
    current_rest_wl = np.float64(df.loc[df['Line_Name'] == line_name, 'Rest Wavelength'].values[0])
    ref_rest_wl = np.float64(df.loc[df['Line_Name'] == ref_line_name, 'Rest Wavelength'].values[0])

    # Ensure the reference line exists
    if not ref_rest_wl:
        raise ValueError(f"Reference line {ref_line_name} not found in DataFrame.")

    # Convert velocity to centroid relation
    ref_cen_param = f"cen_[{ref_line_name}]"
    current_cen_param = "cen"

    # Centroid relation: λ_obs = λ_rest * (1 + z) * (1 + v/c)
    # Since v_ref == v_current, we derive:
    centroid_relation = f"{current_cen_param} == {ref_cen_param} * {current_rest_wl / ref_rest_wl:.6f}"

    return centroid_relation

def update_constraints_with_velocity(df, galaxy_redshift):
    """
    Update DataFrame constraints by converting velocity constraints to centroid constraints.
    
    Parameters:
    - df (DataFrame): Input DataFrame with Gaussian fitting information.
    - galaxy_redshift (float): The galaxy redshift.
    
    Returns:
    - DataFrame: Updated DataFrame with corrected centroid constraints.
    """
    updated_constraints = []

    for idx, row in df.iterrows():
        line_name = row['Line_Name']
        rest_wavelength = np.float64(row['Rest Wavelength'])
        constraints = row['constraints']

        # Normalize to a list of strings: a row may hold NaN (never set), a
        # stringified list (from CSV load), or already a list.
        if isinstance(constraints, str):
            try:
                parsed = ast.literal_eval(constraints)
                constraints = parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                constraints = []
        elif not isinstance(constraints, list):
            constraints = []
        constraints = [str(c) for c in constraints]
        while len(constraints) < 5:
            constraints.append('')
        constraints = constraints[:5]

        wl_emitted_redshift = rest_wavelength * (1 + galaxy_redshift)

        # If 'vel' appears in constraints, process it
        if any('vel' in c for c in constraints):
            new_constraints = [
                convert_velocity_to_centroid(c, line_name, df, galaxy_redshift) if 'vel' in c else c
                for c in constraints
            ]
        else:
            new_constraints = constraints

        print(f'centroid constraint converted from velocity: {new_constraints}')
        updated_constraints.append(new_constraints)

    # Update the DataFrame
    df['constraints'] = updated_constraints
    return df


def _apply_velocity_constraint(found_op, right_side, line_id, df, params):
    """Translate a velocity tie/window/one-sided constraint into a bounded
    additive centroid offset.

    The fit has no `vel` parameter, so a constraint on the velocity *difference*
    between this line (A) and a reference (B) is realised on the centroid as

        cen_A = (restA / restB) * cen_B + offset

    where the helper parameter `offset` carries a [min, max] box bound that
    encodes the allowed Δv interval (Δv = vel_A − vel_B, km/s). Δλ = λ_A · Δv/c.

    Recognised right-hand sides:
        vel_[B]          → exact tie (Δv = 0)
        vel_[B] +- D     → symmetric window |Δv| ≤ D            (use with ==)
        vel_[B] + D      → one-sided threshold +D               (use with <=/>=)
        vel_[B] - D      → one-sided threshold −D               (use with <=/>=)

    A two-sided *minimum* separation (|Δv| ≥ D) is intentionally NOT supported:
    it is a disconnected feasible region (a forbidden band around 0) that
    box-bounded least-squares cannot represent in a single fit. Use a one-sided
    form (pick the sign) instead. Returns True if applied.
    """
    C_KMS = 299792.458
    if '_[' not in right_side or ']' not in right_side:
        return False
    ref_name = _ref_line_name(right_side)
    cen_self = f'cen{line_id + 1}'
    if cen_self not in params:
        return False
    try:
        other = df[df['Line_Name'] == ref_name].iloc[0]
        cen_ref = f"cen{int(other['Line_ID']) + 1}"
    except (IndexError, KeyError, ValueError):
        return False
    if cen_ref not in params:
        return False

    # Rest-wavelength ratio → the "same velocity" centroid mapping.
    try:
        restA = float(df.loc[df['Line_ID'] == line_id, 'Rest Wavelength'].iloc[0])
        restB = float(other['Rest Wavelength'])
        ratio = (restA / restB) if (np.isfinite(restA) and np.isfinite(restB)
                                    and restA > 0 and restB > 0) else 1.0
    except Exception:
        ratio = 1.0

    # Wavelength scale (Å per km/s) at line A's observed position.
    lam = float(params[cen_self].value)
    if not (np.isfinite(lam) and lam > 0):
        lam = float(params[cen_ref].value) * ratio
    if not (np.isfinite(lam) and lam > 0):
        return False

    def dlam(dv):
        return lam * dv / C_KMS

    # Generous wavelength span for the "open" side of a one-sided bound.
    spans, r = [], 1
    while f'x{r}_start' in params and f'x{r}_end' in params:
        spans += [float(params[f'x{r}_start'].value), float(params[f'x{r}_end'].value)]
        r += 1
    edge = (max(spans) - min(spans)) if spans else 1000.0
    if not (np.isfinite(edge) and edge > 0):
        edge = 1000.0

    tail = right_side.split(']', 1)[1].strip()
    try:
        if tail.startswith('+-') or tail.startswith('±'):
            numstr = tail[2:] if tail.startswith('+-') else tail[1:]
            amp = abs(dlam(abs(float(numstr.replace(' ', '')))))
            lo, hi = -amp, amp
        elif tail == '':
            lo = hi = 0.0
        else:
            thr = dlam(float(tail.replace(' ', '')))   # signed km/s
            if found_op in ('<=', '<'):
                lo, hi = -edge, thr
            elif found_op in ('>=', '>'):
                lo, hi = thr, edge
            else:                                       # '==' with a fixed offset
                lo = hi = thr
    except (ValueError, IndexError):
        return False
    if lo > hi:
        lo, hi = hi, lo

    offset_name = f'offset_vel_{cen_self}_{cen_ref}'
    init = min(max(0.0, lo), hi)
    if offset_name not in params:
        params.add(offset_name, value=init, min=lo, max=hi, vary=(lo != hi))
    else:
        params[offset_name].min = lo
        params[offset_name].max = hi
        params[offset_name].value = init
        params[offset_name].vary = (lo != hi)
    params[cen_self].expr = f'{ratio:.8f} * {cen_ref} + {offset_name}'
    print(f"Velocity constraint: {cen_self} = {ratio:.4f}*{cen_ref} + offset "
          f"∈ [{lo:.4f}, {hi:.4f}] Å")
    return True


def _apply_flux_constraint(found_op, right_side, line_id, df, params):
    """Constrain a line's integrated Gaussian FLUX relative to another line's.

    There is no `flux` parameter — flux = amp · σ · √(2π), so a flux relation is
    realised as an amplitude expression carrying the σ ratio:

        flux_A == k · flux_B   →   amp_A = k · amp_B · σ_B / σ_A

    Because it uses the *fitted* σ's, the flux ratio is exact even when the two
    lines have different or freely-varying widths (the √(2π) cancels). Useful for
    fixed atomic ratios, e.g. [N II] 6584/6548 = 2.94.

    Recognised right-hand side:  [factor *] flux_[OtherLine], where `factor` is a
    single number (==, <=/<, >=/>) or a `lo..hi` RANGE for a two-sided bound, e.g.
    the density-sensitive [S II] doublet  flux == 0.44..1.45 * flux_[[S II]_6731].
    Returns True if applied, False otherwise.
    """
    if '_[' not in right_side or ']' not in right_side:
        return False
    ref_name = _ref_line_name(right_side)
    if ref_name is None:
        return False
    amp_self, sig_self = f'amp{line_id + 1}', f'sigma{line_id + 1}'
    if amp_self not in params or sig_self not in params:
        return False
    try:
        other = df[df['Line_Name'] == ref_name].iloc[0]
        oid = int(other['Line_ID'])
    except (IndexError, KeyError, ValueError):
        return False
    amp_ref, sig_ref = f'amp{oid + 1}', f'sigma{oid + 1}'
    if amp_ref not in params or sig_ref not in params:
        return False

    # Factor token: a single number, OR a 'lo..hi' RANGE for a two-sided
    # (windowed) flux-ratio bound — e.g. the density-sensitive [S II] doublet
    # flux_[6716]/flux_[6731] ∈ [0.44, 1.45]. The token is the part of the RHS
    # that is not the flux_[...] reference.
    factor_tok = None
    for tok in (t.strip() for t in right_side.split('*')):
        if tok and '_[' not in tok:
            factor_tok = tok
            break
    try:
        if factor_tok is None:
            lo = hi = 1.0
        elif '..' in factor_tok:
            a, b = factor_tok.split('..')
            lo, hi = float(a), float(b)
        else:
            lo = hi = float(factor_tok)
    except ValueError:
        return False
    if lo > hi:
        lo, hi = hi, lo

    # amp_A = r · amp_B · σ_B/σ_A  ⇒  flux_A/flux_B = r  (the √2π and σ cancel).
    sigratio = f'{amp_ref} * {sig_ref} / {sig_self}'

    if lo != hi:
        # Two-sided bound: a single ratio parameter r ∈ [lo, hi] (op ignored).
        rname = f'fluxr_{amp_self}_{amp_ref}'
        init = min(max(1.0, lo), hi)
        if rname not in params:
            params.add(rname, value=init, min=lo, max=hi, vary=True)
        else:
            params[rname].set(value=init, min=lo, max=hi, vary=True)
        params[amp_self].expr = f'{rname} * {sigratio}'
        print(f"Flux constraint: flux[{amp_self}]/flux[{amp_ref}] ∈ [{lo:g}, {hi:g}]")
        return True

    base = f'{lo:.8g} * {sigratio}'      # fixed ratio (lo == hi)
    if found_op == '==':
        params[amp_self].expr = base
    elif found_op in ('<=', '<'):
        rname = f'fluxr_{amp_self}_{amp_ref}'
        if rname not in params:
            params.add(rname, value=0.9, min=0.0, max=1.0, vary=True)
        params[amp_self].expr = f'{rname} * ({base})'
    elif found_op in ('>=', '>'):
        rname = f'fluxr_{amp_ref}_{amp_self}'
        if rname not in params:
            params.add(rname, value=0.9, min=0.0, max=1.0, vary=True)
        params[amp_self].expr = f'({base}) / {rname}'
    else:
        return False
    print(f"Flux constraint: flux[{amp_self}] {found_op} {lo:g}*flux[{amp_ref}] "
          f"→ {amp_self} = {params[amp_self].expr}")
    return True


def add_dataframe_constraints_to_params(df, params):
    """
    Adds constraints to lmfit Parameters object, using:
    - Additive offsets for centroid inequalities (cen1 >= cen2 → cen1 = cen2 + offset)
    - Multiplicative ratios for all other inequalities (amp1 >= amp2 → amp1 = amp2 * ratio)
    - Direct expressions for equality constraints
    """
    if 'constraints' not in df.columns:
        print("Warning: 'constraints' column not found in DataFrame.")
        return

    for index, row in df.iterrows():
        line_id = int(row['Line_ID'])
        line_name = row['Line_Name']
        constraints_list_str = row['constraints']

        try:
            # Parse constraints string into list
            if isinstance(constraints_list_str, str):
                constraints_list = ast.literal_eval(constraints_list_str)
            elif isinstance(constraints_list_str, list):
                constraints_list = constraints_list_str
            else:
                constraints_list = ['', '', '', '', '']

            if not isinstance(constraints_list, list) or len(constraints_list) != 5:
                print(f"Warning: Invalid constraints format for Line_ID {line_id} ({line_name}): {constraints_list_str}")
                continue

            for constraint in constraints_list:
                constraint = constraint.strip()
                if not constraint:
                    continue

                # Split into left and right parts of the main operator
                operators = ['<=', '<', '>=', '>', '==']
                found_op = None
                for op in operators:
                    if op in constraint:
                        parts = constraint.split(op, 1)
                        found_op = op
                        break

                if not found_op:
                    print(f"Warning: Unsupported constraint format (no valid operator): {constraint}")
                    continue

                left_side = parts[0].strip()
                right_side = parts[1].strip()

                # Parse left side parameter
                param1_base = ''.join(filter(str.isalpha, left_side)).lower()
                param1_num = line_id + 1
                param1_name = f"{param1_base}{param1_num}"

                # VELOCITY constraints (tie / symmetric window / one-sided bound).
                # There is no `vel` parameter — these are realised as a bounded
                # additive offset on the centroid. Handle before the param check.
                if param1_base == 'vel':
                    if not _apply_velocity_constraint(found_op, right_side, line_id, df, params):
                        print(f"Warning: could not parse velocity constraint: {constraint}")
                    continue

                # FLUX constraints (integrated Gaussian flux). There is no `flux`
                # parameter — realised as an amplitude expression carrying the σ
                # ratio (flux = amp·σ·√2π). Handle before the param check.
                if param1_base == 'flux':
                    if not _apply_flux_constraint(found_op, right_side, line_id, df, params):
                        print(f"Warning: could not parse flux constraint: {constraint}")
                    continue

                if param1_name not in params:
                    print(f"Warning: Parameter not found in params: {param1_name}")
                    continue

                # SPECIAL CASE: Centroid inequalities use additive offsets
                if param1_base == 'cen' and found_op in ['>=', '>', '<=', '<']:
                    if '_[' in right_side and ']' in right_side:
                        # Reference to another line's centroid
                        param2_base = ''.join(filter(str.isalpha, right_side.split('_[')[0])).lower()
                        other_line_name = _ref_line_name(right_side)

                        try:
                            other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                            other_line_id = int(other_row['Line_ID'])
                            param2_num = other_line_id + 1
                            param2_name = f"{param2_base}{param2_num}"
                            
                            if param2_name not in params:
                                print(f"Warning: Parameter not found in params: {param2_name}")
                                continue

                            # Get current centroid values
                            cen1_init = params[param1_name].value
                            cen2_init = params[param2_name].value
                            
                            # Calculate initial offset value (ensure it's positive)
                            initial_offset = max(0.1, abs(cen1_init - cen2_init))
                            
                            # Determine max offset based on spectral range constraints
                            if found_op in ['>=', '>']:
                                # cen1 >= cen2 → cen1 = cen2 + offset
                                # Max offset should keep cen1 within x1_end
                                if 'x1_end' in params:
                                    max_offset = max(initial_offset, params['x1_end'].value - cen2_init)
                                else:
                                    max_offset = 100.0  # fallback
                                
                                offset_name = f"offset_{param2_name}_{param1_name}"
                                if offset_name not in params:
                                    params.add(
                                        offset_name,
                                        value=initial_offset,
                                        min=0.0,
                                        max=max_offset,
                                        vary=True
                                    )
                                params[param1_name].expr = f"{param2_name} + {offset_name}"
                                print(f"Centroid constraint: {param1_name} >= {param2_name} "
                                      f"(offset={initial_offset:.1f}, max={max_offset:.1f})")
                            
                            elif found_op in ['<=', '<']:
                                # cen1 <= cen2 → cen1 = cen2 - offset
                                # Max offset should keep cen1 within x1_start
                                if 'x1_start' in params:
                                    max_offset = max(initial_offset, cen2_init - params['x1_start'].value)
                                else:
                                    max_offset = 100.0  # fallback
                                
                                offset_name = f"offset_{param1_name}_{param2_name}"
                                if offset_name not in params:
                                    params.add(
                                        offset_name,
                                        value=initial_offset,
                                        min=0.0,
                                        max=max_offset,
                                        vary=True
                                    )
                                params[param1_name].expr = f"{param2_name} - {offset_name}"
                                print(f"Centroid constraint: {param1_name} <= {param2_name} "
                                      f"(offset={initial_offset:.1f}, max={max_offset:.1f})")
                        
                        except (IndexError, KeyError) as e:
                            print(f"Warning: Could not find line '{other_line_name}' for constraint: {constraint} - {e}")
                    else:
                        print(f"Warning: Centroid constraint must reference another line: {constraint}")
                    continue

                # SPECIAL CASE: Sigma inequalities use additive offsets.
                # The ratio method (default) always initialises at ratio=0.9,
                # making σ_b ≈ σ_core and collapsing the two components.
                # Additive reparameterisation (σ_b = σ_core + δ, δ≥0) seeds
                # δ from the user's initial-guess difference, so the broad
                # component starts at its intended width.
                if param1_base == 'sigma' and found_op in ['>=', '>', '<=', '<']:
                    if '_[' in right_side and ']' in right_side and '*' not in right_side:
                        other_line_name = _ref_line_name(right_side)
                        try:
                            other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                            other_line_id = int(other_row['Line_ID'])
                            param2_name = f"sigma{other_line_id + 1}"

                            if param2_name not in params:
                                print(f"Warning: Parameter not found in params: {param2_name}")
                            else:
                                sig1_init = params[param1_name].value
                                sig2_init = params[param2_name].value

                                if found_op in ['>=', '>']:
                                    # σ1 >= σ2  →  σ1 = σ2 + δ,  δ ≥ 0
                                    delta_init = max(0.0, sig1_init - sig2_init)
                                    sig1_max = params[param1_name].max
                                    delta_max = (float(sig1_max)
                                                 if sig1_max is not None and np.isfinite(float(sig1_max))
                                                 else max(delta_init * 10, sig2_init * 5, 1.0))
                                    offset_name = f"offset_{param2_name}_{param1_name}"
                                    if offset_name not in params:
                                        params.add(offset_name, value=delta_init,
                                                    min=0.0, max=delta_max, vary=True)
                                    params[param1_name].expr = f"{param2_name} + {offset_name}"
                                    print(f"Sigma constraint: {param1_name} >= {param2_name} "
                                          f"(delta_init={delta_init:.4f}, max={delta_max:.4f})")
                                elif found_op in ['<=', '<']:
                                    # σ1 <= σ2  →  σ1 = σ2 - δ,  δ ≥ 0
                                    delta_init = max(0.0, sig2_init - sig1_init)
                                    sig2_max = params[param2_name].max
                                    delta_max = (float(sig2_max)
                                                 if sig2_max is not None and np.isfinite(float(sig2_max))
                                                 else max(delta_init * 10, sig1_init * 5, 1.0))
                                    offset_name = f"offset_{param1_name}_{param2_name}"
                                    if offset_name not in params:
                                        params.add(offset_name, value=delta_init,
                                                    min=0.0, max=delta_max, vary=True)
                                    params[param1_name].expr = f"{param2_name} - {offset_name}"
                                    print(f"Sigma constraint: {param1_name} <= {param2_name} "
                                          f"(delta_init={delta_init:.4f}, max={delta_max:.4f})")
                        except (IndexError, KeyError) as e:
                            print(f"Warning: Could not find line '{other_line_name}' "
                                  f"for sigma constraint: {constraint} - {e}")
                    else:
                        print(f"Warning: Sigma inequality must reference another line "
                              f"without a multiplicative factor: {constraint}")
                    continue

                # DEFAULT CASE: Non-centroid parameters use ratio method
                # Parse right side (could be simple parameter or complex expression)
                if '_[' in right_side and ']' in right_side:
                    # Case 1: Reference to another line parameter (e.g., amp_[Halpha_c1])
                    param2_base = ''.join(filter(str.isalpha, right_side.split('_[')[0])).lower()
                    other_line_name = _ref_line_name(right_side)

                    try:
                        other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                        other_line_id = int(other_row['Line_ID'])
                        param2_num = other_line_id + 1
                        param2_name = f"{param2_base}{param2_num}"
                        
                        if param2_name not in params:
                            print(f"Warning: Parameter not found in params: {param2_name}")
                            continue
                            
                        # Handle multiplicative factors
                        if '*' in right_side:
                            left_part, right_part = [x.strip() for x in right_side.split('*')]
                        
                            # Check which part is numeric and which is the parameter
                            if left_part.replace('.', '', 1).isdigit():
                                factor = float(left_part)
                                expr = f"{factor} * {param2_name}"
                            elif right_part.replace('.', '', 1).isdigit():
                                factor = float(right_part)
                                expr = f"{factor} * {param2_name}"
                            else:
                                print(f"Warning: Could not parse factor in constraint: {constraint}")
                                continue
                        else:
                            expr = param2_name
                            
                        # Apply the constraint
                        if found_op in ['<=', '<']:
                            # For A <= B, we express A = factor * B where factor <= 1
                            factor_name = f"ratio_{param1_name}_{param2_name}"
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{factor_name} * {expr}"
                            print(f"Constraint added: {param1_name} <= {expr} (using factor {factor_name})")
                        elif found_op in ['>=', '>']:
                            # For A >= B, we express A = B / factor where factor <= 1
                            factor_name = f"ratio_{param2_name}_{param1_name}"
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{expr} / {factor_name}"
                            print(f"Constraint added: {param1_name} >= {expr} (using factor {factor_name})")
                        elif found_op == '==':
                            params[param1_name].expr = expr
                            print(f"Constraint added: {param1_name} == {expr}")
                            
                    except (IndexError, KeyError):
                        print(f"Warning: Could not find line '{other_line_name}' for constraint: {constraint}")
                
                elif any(op in right_side for op in ['+', '-', '*', '/']):
                    # Case 2: Mathematical expression (e.g., "0.1 * amp_[Halpha_c1]")
                    try:
                        # Find all parameter references in the expression
                        expr_parts = right_side.split()
                        parsed_expr = []
                        
                        for part in expr_parts:
                            if '_[' in part and ']' in part:
                                # It's a parameter reference
                                param_base = ''.join(filter(str.isalpha, part.split('_[')[0])).lower()
                                other_line_name = _ref_line_name(part)

                                other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                                other_line_id = int(other_row['Line_ID'])
                                param_num = other_line_id + 1
                                param_name = f"{param_base}{param_num}"
                                
                                if param_name not in params:
                                    print(f"Warning: Parameter not found in params: {param_name}")
                                    raise ValueError
                                    
                                parsed_expr.append(param_name)
                            else:
                                parsed_expr.append(part)
                                
                        expr = ' '.join(parsed_expr)
                        
                        # Apply the constraint
                        if found_op in ['<=', '<']:
                            factor_name = f"ratio_{param1_name}_expr_{hash(expr)}"[:30]
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{factor_name} * ({expr})"
                            print(f"Constraint added: {param1_name} <= {expr} (using factor {factor_name})")
                        elif found_op in ['>=', '>']:
                            factor_name = f"ratio_expr_{hash(expr)}_{param1_name}"[:30]
                            if factor_name not in params:
                                params.add(factor_name, value=1.1, min=1.0, max=10.0, vary=True)
                            params[param1_name].expr = f"({expr}) / {factor_name}"
                            print(f"Constraint added: {param1_name} >= {expr} (using factor {factor_name})")
                        elif found_op == '==':
                            params[param1_name].expr = f"({expr})"
                            print(f"Constraint added: {param1_name} == {expr}")
                            
                    except (ValueError, IndexError, KeyError):
                        print(f"Warning: Could not parse complex expression in constraint: {constraint}")
                
                else:
                    print(f"Warning: Unsupported right side format in constraint: {constraint}")

        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse constraints string for Line_ID {line_id} ({line_name}): {constraints_list_str} - {e}")



def generalized_model(x, slope, intercept, *gaussian_params):
    """
    Generalized Gaussian model for N Gaussians.
    :param x: Wavelength data.
    :param slope: Continuum slope.
    :param intercept: Continuum intercept.
    :param gaussian_params: Model parameters for N Gaussians [Amp_0, Centroid_0, Sigma_0, ...]
    """
    model = slope * x + intercept  # Linear model for continuum
    num_gaussians = len(gaussian_params) // 3
    for i in range(num_gaussians):
        amp = gaussian_params[3 * i]
        centroid = gaussian_params[3 * i + 1]
        sigma = gaussian_params[3 * i + 2]
        model += amp * np.exp(-(x - centroid)**2 / (2 * sigma**2))

    return model

# def load_stylesheet(app, filename):
#     with open(filename, 'r') as f:
#         style = f.read()
#         app.setStyleSheet(style)

# Apply PyQt5-like styling to Matplotlib
def _fmt(value, sig=4):
    """Format a numeric value for button display.
    Uses scientific notation when the absolute value is < 0.001 or >= 1e6,
    otherwise uses up to `sig` significant figures.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return "0"
    if abs(v) < 0.001 or abs(v) >= 1e5:
        return f"{v:.{sig-1}e}"
    return f"{v:.{sig}g}"


def _as_float_list(v):
    """Coerce a knots cell (list, numpy array, or a string like '[1.0, 2.0]'
    from a loaded CSV) into a plain list of floats. Returns [] on failure."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return []
    try:
        return [float(z) for z in v]
    except Exception:
        return []


def _safe_float(v):
    """Best-effort float conversion; returns np.nan on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan



# ── Emission line library ──────────────────────────────────────────────────────
_LINE_LIBRARY = None  # loaded lazily

def _load_line_library():
    """Load LineLibrary.csv from the HyperCube directory.

    Expected columns (exact names):
        wavelength_AA  : float, wavelength in Angstroms (range 770.409 – 10938.086)
        ion            : str,   emission line name
        IP             : float, ionization potential in eV
    """
    global _LINE_LIBRARY
    if _LINE_LIBRARY is not None:
        return _LINE_LIBRARY
    try:
        lib_path = resource_path('LineLibrary.csv')
        _LINE_LIBRARY = pd.read_csv(lib_path,
                                    usecols=['wavelength_AA', 'ion', 'IP'],
                                    dtype={'wavelength_AA': float, 'ion': str, 'IP': float})
        print(f"Loaded line library: {len(_LINE_LIBRARY)} lines from {lib_path}")
    except Exception as e:
        print(f"Could not load LineLibrary.csv: {e}")
        _LINE_LIBRARY = pd.DataFrame(columns=['wavelength_AA', 'ion', 'IP'])
    return _LINE_LIBRARY


def _identify_line(obs_wavelength_AA, redshift):
    """Given an observed wavelength and source redshift, identify the nearest line.

    Computes rest_wav = obs_wav / (1 + z), then finds the nearest entry in
    LineLibrary.csv by rest wavelength.
    Returns (ion_name, rest_wav_AA) or ('', nan) if redshift is missing/invalid.
    """
    try:
        z = float(redshift)
    except (TypeError, ValueError):
        return '', np.nan
    if z <= 0 or not np.isfinite(z):
        return '', np.nan

    lib = _load_line_library()
    if len(lib) == 0:
        return '', np.nan

    rest_wav = obs_wavelength_AA / (1.0 + z)
    wavs = lib['wavelength_AA'].values.astype(float)
    idx = np.argmin(np.abs(wavs - rest_wav))
    ion_name   = str(lib.iloc[idx]['ion'])
    nearest_wav = float(wavs[idx])
    return ion_name, nearest_wav


def apply_mpl_qss_style(fig, ax, line):
    """Apply a PyQt5-style QSS theme to the given Matplotlib figure, axes, and line."""
    
    # Set the figure (entire canvas) background color
    fig.patch.set_facecolor('#323232')  # Matches QWidget background
    
    # Axis background color
    ax.set_facecolor('#2E3440')  
    
    # Axis labels and title (matches QWidget text color)
    ax.xaxis.label.set_color('#b1b1b1')
    ax.yaxis.label.set_color('#b1b1b1')
    ax.title.set_color('#b1b1b1')
    
    # Ticks (color and size)
    ax.tick_params(axis='x', colors='#b1b1b1', labelsize=12)
    ax.tick_params(axis='y', colors='#b1b1b1', labelsize=12)
    
    # Axis borders (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b1b1b1')
        spine.set_linewidth(0.75)
    
    # Grid (optional – enable for a clearer view)
    ax.grid(color='#4C566A', linestyle='--', linewidth=0.5)
    
    if line != None:
        # Adjust the spectrum line styling
        line.set_color('whitesmoke')  # Hover color from QPushButton
        line.set_linewidth(0.75)  # Slightly thicker for better visibility
    
    # Legend styling (if needed)
    # legend = ax.legend()
    # if legend:
    #     legend.get_frame().set_facecolor('#2E3440')
    #     legend.get_frame().set_edgecolor('#D8DEE9')
    #     plt.setp(legend.get_texts(), color='#D8DEE9')

    # Leave enough room for axis labels (tight_layout triggers a compat warning
    # and can still clip labels on small canvases).
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.97)

class SpaxelButton(QPushButton):
    def __init__(self, text, col, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.col = col

class FrameButton(QPushButton):
    def __init__(self, text, row, col, frame_id, feature, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.row = row
        self.col = col
        self.frame_id = frame_id  # Store frame information in the button
        self.feature = feature
        # Expand horizontally so buttons fill their grid column as the dock is resized
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )

    def minimumSizeHint(self):
        # Allow buttons to compress below their text width when the dock is narrow
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fit_params_window = None  # Placeholder, initially None
        self.fit_params_dock = None   # QDockWidget wrapper, initially None
        self.fits_header = None
        self.fits_data = None
        self.is_1d_spectrum = False
        self.current_spaxel = None  # Stores the currently highlighted spaxel (x, y)
        # Gaussian component line actors (populated during interactive 'g' draws;
        # must exist up front so update_buttons can .clear() it after a CSV fit load)
        self.gaussian_component_lines = []
        self.locked = False  # Initially unlocked
        self._chanmap_active = False   # True while C is held
        # Brightness/contrast drag state
        self._bc_drag_active = False
        # Background image display settings
        self._bkg_cmap  = 'gray'
        self._bkg_scale = 'Linear'
        self._bkg_brightness = 0.0   # shift: fraction of data range
        self._bkg_contrast  = 1.0   # scale multiplier on span
        self._init_guess_spaxel = None   # (x, y) that has interactive init guesses
        self._blue_rect = None            # blue rectangle marking that spaxel
        self._edit_spaxel = None          # (x, y) in single-spaxel edit mode, else None
        self._edit_orig_fit_rows = None   # this spaxel's fit rows before Clear (for revert)
        self._edit_prev_override = None   # this spaxel's override before Clear (for revert)
        self._edited_overlay_on = False   # show translucent boxes on edited spaxels
        self._edited_overlay_patches = [] # the overlay Rectangle artists
        self._bc_drag_start  = None   # (x, y) in axes data coords at press
        self._bc_vmin0 = None         # vmin at drag start
        self._bc_vmax0 = None         # vmax at drag start
        self._chanmap_start  = None    # wavelength where C was pressed
        self._chanmap_span   = None    # axvspan patch on spectrum_ax
        self._chanmap_locked = False   # True after C released (selection fixed)
        self._chanmap_handle_left  = None   # ◀ annotation on left  edge of span
        self._chanmap_handle_right = None   # ▶ annotation on right edge of span
        self._chanmap_drag_active  = False  # True while user is dragging the span
        self._chanmap_drag_x_start = None   # xdata at drag-press
        self._chanmap_drag_wav_start = None # (wav0, wav1) at drag-press
        # Subtraction windows (X and V keys)
        self._submap = {'x': {'active': False, 'start': None, 'span': None, 'locked': False},
                        'v': {'active': False, 'start': None, 'span': None, 'locked': False}}
        self.spectrum_cursor_pos = None  # Add tracking for spectrum cursor position
        self.spectrum_ax = None
        self.drawing_line = None
        # Spline continuum drawing state ('S' key connect-the-dots)
        self._spline_drawing = False
        self._spline_knots = []          # list of (x, y) placed knots
        self._spline_dot_artist = None   # markers at the knots
        self._spline_poly_artist = None  # connect-the-dots / spline preview curve
        self._spline_rubber_artist = None  # live segment from last knot to cursor
        # Polynomial continuum drawing state ('P' key two-click range)
        self._poly_drawing = False
        self._poly_x1 = None             # range start (first 'P' press)
        self._poly_preview_artist = None  # live fitted-polynomial preview curve
        self._poly_edge_artists = []     # vertical markers at the range edges
        self._poly_default_degree = 3    # default Chebyshev degree
        # On-spectrum instructional hint shown during interactive drawing
        self._spectrum_hint_artist = None
        self.fluxscalefactor = 1 # Default flux scale factor
        self.WLscalefactor = 1 # Default wavelength scale factor
        self.slope = 0  # Default slope
        self.intercept = 0  # Default intercept
        self.gaussian_active = False  # Toggle for Gaussian fitting
        self.current_gaussian = None  # Store the Gaussian plot
        self.initial_x = None  # Stores initial cursor position for Gaussian
        self.current_sigma = 10  # Default sigma
        self.current_amplitude = 1  # Default amplitude
        self.line_region = None
        
        self.zoom_rect = None  # Store the rectangle
        self.zoom_start_x = None  # Start position for zoom
        self.zoom_active = False
        self.zoom_limits = None  # Store zoom limits globally
        self.zoom_pad = 0.2
        self.modelfit = None
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("HyperCube")
        self.setGeometry(100, 100, 1400, 800)

        # ── Shared style helpers ────────────────────────────────────────
        _combo_qss = (
            "QComboBox { color: #eff0f1; background-color: #3c3f41; border: 1px solid #555; padding: 0 4px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView {"
            "    color: #eff0f1; background-color: #3c3f41;"
            "    selection-background-color: #d7801a; selection-color: #000000;"
            "    border: 1px solid #555;}"
        )
        def _vsep():
            s = QFrame(self); s.setFrameShape(QFrame.VLine); s.setFrameShadow(QFrame.Sunken)
            return s
        def _tbtn(label, tip, size=28):
            b = QPushButton(label, self); b.setFixedSize(size, size); b.setToolTip(tip)
            return b

        # ── Primary toolbar (Open FITS · Fit Parameters · λScale · FScale) ──
        primary_bar = QHBoxLayout()
        primary_bar.setContentsMargins(8, 4, 8, 4)
        primary_bar.setSpacing(8)

        self.open_fits_button = QPushButton("  ⬆  Open FITS", self)
        self.open_fits_button.setFixedHeight(32)
        self.open_fits_button.setStyleSheet(
            "QPushButton { background-color: #d7801a; color: white; font-weight: bold;"
            "  border-radius: 5px; padding: 0 14px; }"
            "QPushButton:hover { background-color: #ffa02f; }"
        )
        self.open_fits_button.clicked.connect(self.open_fits_file)
        primary_bar.addWidget(self.open_fits_button)

        primary_bar.addWidget(_vsep())

        self.open_fit_params_button = QPushButton("▼  Fit Parameters", self)
        self.open_fit_params_button.setFixedHeight(32)
        self.open_fit_params_button.setToolTip("Toggle Fit Parameters panel")
        self.open_fit_params_button.clicked.connect(self.toggle_fit_params_dock)
        primary_bar.addWidget(self.open_fit_params_button)

        primary_bar.addWidget(_vsep())

        self.WLscalefactor_button = QPushButton("λ Scale", self)
        self.WLscalefactor_button.setFixedHeight(32)
        self.WLscalefactor_button.setToolTip("Wavelength scale factor")
        self.WLscalefactor_button.setEnabled(False)
        self.WLscalefactor_button.clicked.connect(self.press_scaleWL_button)
        primary_bar.addWidget(self.WLscalefactor_button)

        self.fluxscalefactor_button = QPushButton("F Scale", self)
        self.fluxscalefactor_button.setFixedHeight(32)
        self.fluxscalefactor_button.setToolTip("Flux scale factor")
        self.fluxscalefactor_button.setEnabled(False)
        self.fluxscalefactor_button.clicked.connect(self.press_scaleflux_button)
        primary_bar.addWidget(self.fluxscalefactor_button)

        primary_bar.addStretch()

        primary_bar_widget = QWidget()
        primary_bar_widget.setLayout(primary_bar)
        primary_bar_widget.setFixedHeight(42)

        # ── Main layout ─────────────────────────────────────────────────
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(primary_bar_widget)

        # Horizontal splitter: cube image left, spectrum viewer right
        self.splitter = QSplitter(Qt.Horizontal)

        # ── Left panel: image toolbar + cube canvas ─────────────────────
        self.left_panel = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)

        # Image toolbar — all cube display controls above the canvas
        img_bar = QHBoxLayout()
        img_bar.setContentsMargins(4, 2, 4, 2)
        img_bar.setSpacing(3)

        # Zoom
        self._cube_zoom_level = 1.0
        self._zoom_levels = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]

        self.zoom_out_btn = _tbtn("−", "Zoom out", 24)
        self.zoom_out_btn.clicked.connect(self._cube_zoom_out)
        img_bar.addWidget(self.zoom_out_btn)

        self.zoom_combo = QComboBox(self)
        self.zoom_combo.setFixedHeight(24)
        self.zoom_combo.setMinimumWidth(100)
        self.zoom_combo.setMaximumWidth(130)
        self.zoom_combo.addItems([
            "Fit window", "Fit width", "Fit height",
            "3.125%", "6.25%", "12.5%", "25%", "50%",
            "100%", "200%", "400%", "800%", "1600%", "3200%"
        ])
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.setStyleSheet(_combo_qss)
        self.zoom_combo.currentTextChanged.connect(lambda t: [self._cube_zoom_combo_changed(t), self.setFocus()])
        img_bar.addWidget(self.zoom_combo)

        self.zoom_in_btn = _tbtn("+", "Zoom in", 24)
        self.zoom_in_btn.clicked.connect(self._cube_zoom_in)
        img_bar.addWidget(self.zoom_in_btn)

        img_bar.addWidget(_vsep())

        # Stretch + Scale
        self.stretch_combo = QComboBox(self)
        self.stretch_combo.setFixedHeight(24)
        self.stretch_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.stretch_combo.addItems(["minmax", "99.9%", "99.5%", "99%", "98%", "95%", "zscale", "manual"])
        self.stretch_combo.setCurrentText("99%")
        self.stretch_combo.setToolTip("Image stretch (clip level)")
        self.stretch_combo.setStyleSheet(_combo_qss)
        self.stretch_combo.currentTextChanged.connect(lambda _: [self._cube_redraw_with_current_settings(), self.setFocus()])
        img_bar.addWidget(self.stretch_combo)

        self.scale_combo = QComboBox(self)
        self.scale_combo.setFixedHeight(24)
        self.scale_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.scale_combo.addItems(["Linear", "Log", "Square root", "Squared", "Asinh"])
        self.scale_combo.setCurrentText("Linear")
        self.scale_combo.setToolTip("Image scaling (transfer function)")
        self.scale_combo.setStyleSheet(_combo_qss)
        self.scale_combo.currentTextChanged.connect(lambda _: [self._cube_redraw_with_current_settings(), self.setFocus()])
        img_bar.addWidget(self.scale_combo)

        img_bar.addWidget(_vsep())

        # Colormap + Reset
        self.cube_cmap_combo = QComboBox(self)
        self.cube_cmap_combo.setFixedHeight(24)
        self.cube_cmap_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.cube_cmap_combo.addItems(['gray','gray_r','viridis','plasma','inferno','magma','hot','cool','Blues','Reds','Greens','bwr','seismic'])
        self.cube_cmap_combo.setCurrentText('gray')
        self.cube_cmap_combo.setToolTip("Cube image colormap")
        self.cube_cmap_combo.setStyleSheet(_combo_qss)
        self.cube_cmap_combo.currentTextChanged.connect(
            lambda cmap: [setattr(self, '_last_cmap', cmap), self._cube_redraw_with_current_settings(), self.setFocus()]
        )
        img_bar.addWidget(self.cube_cmap_combo)

        self.reset_view_btn = QPushButton("Reset", self)
        self.reset_view_btn.setFixedHeight(24)
        self.reset_view_btn.setToolTip("Reset to original view: gray colormap, default zoom and scaling")
        self.reset_view_btn.clicked.connect(self._cube_reset_view)
        img_bar.addWidget(self.reset_view_btn)

        img_bar.addWidget(_vsep())

        # Rotate / flip buttons — formerly the floating side panel
        self._cube_rotation = 0
        self._cube_flip_h = False
        self._cube_flip_v = False
        self._cube_coords_mode = "xy"

        _xform_btns = [
            ("⟳", "Rotate 90° clockwise",         lambda: self._cube_rotate(90)),
            ("⟲", "Rotate 90° counter-clockwise",  lambda: self._cube_rotate(-90)),
            ("⇔", "Flip horizontal",               lambda: self._cube_flip('h')),
            ("⇕", "Flip vertical",                 lambda: self._cube_flip('v')),
        ]
        self._xform_buttons = []
        for label, tip, fn in _xform_btns:
            b = _tbtn(label, tip, 28)
            b.clicked.connect(fn)
            b.setEnabled(False)
            img_bar.addWidget(b)
            self._xform_buttons.append(b)

        self.coords_toggle_btn = QPushButton("X/Y", self)
        self.coords_toggle_btn.setFixedSize(40, 24)
        self.coords_toggle_btn.setToolTip("Toggle between X/Y pixel and RA/Dec axis labels")
        self.coords_toggle_btn.clicked.connect(self._cube_toggle_coords)
        img_bar.addWidget(self.coords_toggle_btn)

        img_bar.addWidget(_vsep())

        # Background image + opacity slider
        self.bkg_image_btn = QPushButton("🌌 Bkg", self)
        self.bkg_image_btn.setFixedHeight(24)
        self.bkg_image_btn.setToolTip("Overlay background image (HST/Chandra) — requires resolved source")
        self.bkg_image_btn.setEnabled(False)
        self.bkg_image_btn.clicked.connect(self._show_bkg_image_dialog)
        img_bar.addWidget(self.bkg_image_btn)

        self.cube_opacity_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.cube_opacity_slider.setRange(0, 100)
        self.cube_opacity_slider.setValue(100)
        self.cube_opacity_slider.setFixedWidth(70)
        self.cube_opacity_slider.setFixedHeight(20)
        self.cube_opacity_slider.setToolTip("Cube overlay opacity")
        self.cube_opacity_slider.setVisible(False)
        self.cube_opacity_slider.valueChanged.connect(self._cube_opacity_changed)
        img_bar.addWidget(self.cube_opacity_slider)

        # No trailing stretch: the toolbar hugs its content so it can be
        # left-aligned in the secondary toolbar row above the panels. The
        # toolbar is added to that row (not the cube panel) further below, so
        # resizing the panel never hides it.
        img_bar_widget = QWidget()
        img_bar_widget.setLayout(img_bar)
        img_bar_widget.setFixedHeight(30)
        # The cube panel now holds only the canvas, so it can shrink freely.
        self.left_panel.setMinimumWidth(120)

        # Canvas + scrollbars
        self.canvas = FigureCanvas(plt.Figure())
        self._init_placeholder_canvas()

        canvas_grid = QWidget()
        canvas_grid_layout = QGridLayout(canvas_grid)
        canvas_grid_layout.setContentsMargins(0, 0, 0, 0)
        canvas_grid_layout.setSpacing(0)

        self.cube_hbar = QScrollBar(Qt.Horizontal)
        self.cube_vbar = QScrollBar(Qt.Vertical)
        self.cube_hbar.setRange(0, 1000); self.cube_vbar.setRange(0, 1000)
        self.cube_hbar.setValue(500);     self.cube_vbar.setValue(500)
        self.cube_hbar.setSingleStep(20); self.cube_vbar.setSingleStep(20)
        self.cube_hbar.setPageStep(100);  self.cube_vbar.setPageStep(100)
        _bar_qss = (
            "QScrollBar:horizontal { height: 14px; }"
            "QScrollBar:vertical   { width:  14px; }"
            "QScrollBar::handle:horizontal { min-width:  30px; }"
            "QScrollBar::handle:vertical   { min-height: 30px; }"
        )
        self.cube_hbar.setStyleSheet(_bar_qss); self.cube_vbar.setStyleSheet(_bar_qss)
        self.cube_hbar.hide(); self.cube_vbar.hide()
        self._cube_scrollbar_updating = False
        self.cube_hbar.valueChanged.connect(self._cube_scrollbar_pan)
        self.cube_vbar.valueChanged.connect(self._cube_scrollbar_pan)

        canvas_grid_layout.addWidget(self.canvas,    0, 0)
        canvas_grid_layout.addWidget(self.cube_vbar, 0, 1)
        canvas_grid_layout.addWidget(self.cube_hbar, 1, 0)
        canvas_grid_layout.setColumnStretch(0, 1)
        canvas_grid_layout.setRowStretch(0, 1)
        self.left_layout.addWidget(canvas_grid)
        self.left_panel.setLayout(self.left_layout)
        self.splitter.addWidget(self.left_panel)

        # ── Right panel: spectrum toolbar + canvas ──────────────────────
        self.right_panel = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        spec_bar = QHBoxLayout()
        spec_bar.setContentsMargins(4, 2, 4, 2)
        spec_bar.setSpacing(4)

        _cml = QLabel("Channel map:")
        _cml.setStyleSheet("color: #aaa; font-size: 11px;")
        spec_bar.addWidget(_cml)

        self.chanmap_prev_btn = QPushButton("−", self)
        self.chanmap_prev_btn.setFixedSize(24, 24)
        self.chanmap_prev_btn.setToolTip("Shift channel map selection left by 1 pixel")
        self.chanmap_prev_btn.clicked.connect(lambda: self._chanmap_shift(-1))
        spec_bar.addWidget(self.chanmap_prev_btn)

        self.chanmap_centre_btn = SpaxelButton("Pixel: —", "Channel centre")
        self.chanmap_centre_btn.setFixedHeight(24)
        self.chanmap_centre_btn.setMinimumWidth(90)
        self.chanmap_centre_btn.setToolTip("Centre pixel of channel map selection — click to edit")
        self.chanmap_centre_btn.clicked.connect(self._chanmap_centre_clicked)
        spec_bar.addWidget(self.chanmap_centre_btn)

        self.chanmap_next_btn = QPushButton("+", self)
        self.chanmap_next_btn.setFixedSize(24, 24)
        self.chanmap_next_btn.setToolTip("Shift channel map selection right by 1 pixel")
        self.chanmap_next_btn.clicked.connect(lambda: self._chanmap_shift(+1))
        spec_bar.addWidget(self.chanmap_next_btn)

        # No trailing stretch: hugs content. Placed in the secondary toolbar
        # row above the panels (not inside the spectrum panel), so it stays
        # visible when the panels are resized.
        self.spec_bar_widget = QWidget()
        self.spec_bar_widget.setLayout(spec_bar)
        self.spec_bar_widget.setFixedHeight(30)
        self.spec_bar_widget.hide()

        self.right_panel.setLayout(self.right_layout)
        self.splitter.addWidget(self.right_panel)

        self.splitter.setSizes([600, 800])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setHandleWidth(6)
        self.splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # ── Secondary toolbar row (above the panels, full width) ────────
        # Cube/image controls (left) and channel-map controls (right) live here,
        # OUTSIDE the splitter, so resizing the panels never hides them.
        secondary_bar = QHBoxLayout()
        secondary_bar.setContentsMargins(4, 0, 4, 0)
        secondary_bar.setSpacing(6)
        secondary_bar.addWidget(img_bar_widget)
        secondary_bar.addStretch()
        secondary_bar.addWidget(self.spec_bar_widget)
        secondary_bar_widget = QWidget()
        secondary_bar_widget.setLayout(secondary_bar)
        secondary_bar_widget.setFixedHeight(34)
        main_layout.addWidget(secondary_bar_widget)

        main_layout.addWidget(self.splitter, stretch=1)

        # ── Status bar ─────────────────────────────────────────────────
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(
            "QStatusBar { background: #2b2b2b; color: #aaa; font-size: 11px; border-top: 1px solid #444; }"
            "QStatusBar::item { border: none; }"
        )
        self._sb_file    = QLabel("No file loaded")
        self._sb_spaxel  = QLabel("")
        self._sb_radec   = QLabel("")
        self._sb_snr     = QLabel("")
        for lbl in (self._sb_file, self._sb_spaxel, self._sb_radec, self._sb_snr):
            lbl.setStyleSheet("color: #ccc; padding: 0 8px;")
            self.status_bar.addPermanentWidget(lbl)

        # Set up central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set menu bar
        self.setMenuBar(self.create_menu_bar_VisualizerWindow())

        # Handle and separator styles are defined in QDarkOrange_style.qss

        # Ensure the window comes to the front
        # self.show()
        self.raise_()
        self.activateWindow()

        # Enable mouse tracking for the white-light image
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self._cube_on_scroll)
        self.canvas.mpl_connect('button_press_event', self._cube_pan_start)
        self.canvas.mpl_connect('button_press_event', self._cube_bc_press)
        self.canvas.mpl_connect('motion_notify_event', self._cube_pan_move)
        self.canvas.mpl_connect('motion_notify_event', self._cube_bc_drag)
        self.canvas.mpl_connect('button_release_event', self._cube_pan_end)
        self.canvas.mpl_connect('button_release_event', self._cube_bc_release)
        self._cube_pan_active = False
        self._cube_pan_start_pos = None
        self._cube_pan_start_xlim = None
        self._cube_pan_start_ylim = None

        self.cursor_pos = None

    # ── Cube viewport zoom ────────────────────────────────────────────────────


    def _cube_scrollbar_pan(self):
        """Pan the cube axes from scrollbar position (0-1000 range)."""
        if self._cube_scrollbar_updating:
            return
        if not hasattr(self, 'ax') or self.ax is None:
            return
        ext = self._cube_field_extent()
        if ext is None:
            return

        x0, x1, y0, y1 = ext
        img_w = abs(x1 - x0)
        img_h = abs(y1 - y0)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        view_w = abs(xlim[1] - xlim[0])
        view_h = abs(ylim[1] - ylim[0])

        pan_w = max(img_w - view_w, 0)
        pan_h = max(img_h - view_h, 0)

        frac_h = self.cube_hbar.value() / 1000.0
        frac_v = 1.0 - self.cube_vbar.value() / 1000.0  # invert: 0=top

        cx = x0 + view_w / 2 + frac_h * pan_w
        cy = y0 + view_h / 2 + frac_v * pan_h

        self.ax.set_xlim(cx - view_w / 2, cx + view_w / 2)
        self.ax.set_ylim(cy - view_h / 2, cy + view_h / 2)
        self.canvas.draw_idle()

    def _cube_update_scrollbars(self):
        """Sync scrollbar positions to current axes limits and show/hide them.
        
        Bars are only visible when the view is smaller than the image
        (i.e. the user is zoomed in enough that parts of the image are hidden).
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return
        if not hasattr(self, 'cube_hbar'):
            return
        ext = self._cube_field_extent()
        if ext is None:
            return

        x0, x1, y0, y1 = ext
        img_w = abs(x1 - x0)
        img_h = abs(y1 - y0)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        view_w = abs(xlim[1] - xlim[0])
        view_h = abs(ylim[1] - ylim[0])

        pan_w = img_w - view_w
        pan_h = img_h - view_h

        self._cube_scrollbar_updating = True

        # Horizontal bar
        if pan_w > 1e-6:
            cx = (xlim[0] + xlim[1]) / 2
            frac_h = np.clip((cx - (x0 + view_w / 2)) / pan_w, 0, 1)
            self.cube_hbar.setValue(int(frac_h * 1000))
            self.cube_hbar.show()
        else:
            self.cube_hbar.setValue(500)
            self.cube_hbar.hide()

        # Vertical bar
        if pan_h > 1e-6:
            cy = (ylim[0] + ylim[1]) / 2
            frac_v = np.clip((cy - (y0 + view_h / 2)) / pan_h, 0, 1)
            self.cube_vbar.setValue(int((1 - frac_v) * 1000))
            self.cube_vbar.show()
        else:
            self.cube_vbar.setValue(500)
            self.cube_vbar.hide()

        self._cube_scrollbar_updating = False

    def _cube_bc_press(self, event):
        """Start brightness/contrast drag on left-click (button 1) in the cube axes."""
        if event.button != 1 or event.inaxes != self.ax:
            return
        bkg = getattr(self, '_bkg_artist', None)
        # Get clim from the cube image only (skip the background artist)
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_clim') and callable(c.get_clim) and c is not bkg]
        if not imgs:
            return
        self._bc_vmin0, self._bc_vmax0 = imgs[0].get_clim()
        self._bc_drag_start  = (event.x, event.y)  # pixels, not data coords
        self._bc_drag_active = True

    def _cube_bc_drag(self, event):
        """Adjust brightness (left/right) and contrast (up/down) during drag."""
        if not self._bc_drag_active or event.inaxes != self.ax:
            return
        if event.x is None or event.y is None:
            return

        dx = event.x - self._bc_drag_start[0]   # pixels rightward
        dy = event.y - self._bc_drag_start[1]   # pixels upward

        span = self._bc_vmax0 - self._bc_vmin0
        if span == 0:
            span = 1.0

        # Sensitivity: 300 px of drag = 1 full span shift/scale
        # Up/down = brightness (shift midpoint); left/right = contrast (scale span)
        brightness_shift = (dy / 300.0) * span   # up = brighter
        contrast_scale   = 10 ** (dx / 300.0)    # right = wider range (less contrast)

        mid = (self._bc_vmin0 + self._bc_vmax0) / 2 + brightness_shift
        half = (span / 2) * contrast_scale

        bkg = getattr(self, '_bkg_artist', None)
        # Only adjust clim on the cube image — leave the background untouched
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'set_clim') and callable(c.set_clim) and c is not bkg]
        for img in imgs:
            img.set_clim(mid - half, mid + half)
        self.canvas.draw_idle()

    def _cube_bc_release(self, event):
        """End brightness/contrast drag."""
        if event.button == 1:
            self._bc_drag_active = False

    def _cube_field_extent(self):
        """Return (x0, x1, y0, y1) of the full displayed field.

        Uses the background image's (larger) extent when one is loaded so the
        user can pan and zoom out into the surrounding field; otherwise falls
        back to the cube image's extent.
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return None
        bkg = getattr(self, '_bkg_artist', None)
        if bkg is not None:
            try:
                return bkg.get_extent()
            except Exception:
                pass
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_extent') and callable(c.get_extent)]
        if not imgs:
            return None
        return imgs[0].get_extent()

    def _cube_clamp_limits(self, xlim, ylim):
        """Clamp xlim/ylim so the view never pans outside the image bounds."""
        ext = self._cube_field_extent()
        if ext is None:
            return xlim, ylim
        x0, x1, y0, y1 = ext
        half_w = (xlim[1] - xlim[0]) / 2
        half_h = (ylim[1] - ylim[0]) / 2
        cx = np.clip((xlim[0] + xlim[1]) / 2, x0 + half_w, x1 - half_w)
        cy = np.clip((ylim[0] + ylim[1]) / 2, y0 + half_h, y1 - half_h)
        return (cx - half_w, cx + half_w), (cy - half_h, cy + half_h)

    def _cube_on_scroll(self, event):
        """Pan the cube image on trackpad/mouse scroll.
        
        Vertical scroll (step) pans up/down.
        Horizontal scroll (guiEvent with angleDeltaX) pans left/right.
        Hold Ctrl to pan horizontally on a one-axis scroll device.
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return
        if event.inaxes != self.ax:
            return

        xlim = list(self.ax.get_xlim())
        ylim = list(self.ax.get_ylim())
        pan_frac = 0.1  # 10% of current view per scroll step

        dx = (xlim[1] - xlim[0]) * pan_frac
        dy = (ylim[1] - ylim[0]) * pan_frac

        # Dominant-axis locking: whichever axis has the larger delta wins,
        # preventing diagonal drift on trackpads.
        vert_raw = 0
        horiz_raw = 0
        if hasattr(event, 'guiEvent') and event.guiEvent is not None:
            ge = event.guiEvent
            if hasattr(ge, 'angleDelta'):
                horiz_raw = ge.angleDelta().x()
                vert_raw  = ge.angleDelta().y()

        if abs(horiz_raw) > abs(vert_raw) and horiz_raw != 0:
            # Pure horizontal gesture → pan left/right only
            direction = -1 if horiz_raw > 0 else 1
            xlim[0] += direction * dx
            xlim[1] += direction * dx
        else:
            # Vertical scroll (wheel or dominant vertical gesture) → pan up/down
            if vert_raw != 0:
                direction = 1 if vert_raw > 0 else -1
            else:
                direction = 1 if event.button == 'up' else -1
            ylim[0] += direction * dy
            ylim[1] += direction * dy

        xlim, ylim = self._cube_clamp_limits(xlim, ylim)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()
        self._cube_update_scrollbars()

    def _cube_pan_start(self, event):
        """Begin middle-mouse-button drag pan."""
        if event.button == 2 and event.inaxes == self.ax:
            self._cube_pan_active = True
            self._cube_pan_start_pos = (event.xdata, event.ydata)
            self._cube_pan_start_xlim = list(self.ax.get_xlim())
            self._cube_pan_start_ylim = list(self.ax.get_ylim())

    def _cube_pan_move(self, event):
        """Pan while middle-mouse button is held."""
        if not self._cube_pan_active or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = self._cube_pan_start_pos[0] - event.xdata
        dy = self._cube_pan_start_pos[1] - event.ydata
        xlim = [self._cube_pan_start_xlim[0] + dx,
                self._cube_pan_start_xlim[1] + dx]
        ylim = [self._cube_pan_start_ylim[0] + dy,
                self._cube_pan_start_ylim[1] + dy]
        xlim, ylim = self._cube_clamp_limits(xlim, ylim)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()
        self._cube_update_scrollbars()

    def _cube_pan_end(self, event):
        """End middle-mouse-button drag pan."""
        if event.button == 2:
            self._cube_pan_active = False
            self._cube_pan_start_pos = None

    def _cube_zoom_apply(self, zoom_level=None, mode=None):
        """Apply zoom to the cube axes.
        
        zoom_level: float multiplier (1.0 = fit window, 2.0 = 200%, etc.)
        mode: 'fit_window' | 'fit_width' | 'fit_height' | None (use zoom_level)
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return

        # Natural extent of the image in data coords
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_extent') and callable(c.get_extent)]
        if not imgs:
            return
        x0, x1, y0, y1 = imgs[0].get_extent()  # (left, right, bottom, top)
        img_w = abs(x1 - x0)
        img_h = abs(y1 - y0)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        canvas_w = self.canvas.width()
        canvas_h = self.canvas.height()
        if canvas_w == 0 or canvas_h == 0:
            return

        if mode == 'fit_window':
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)
            self._cube_zoom_level = 1.0
        elif mode == 'fit_width':
            # Scale so image width fills canvas; crop/pad height
            scale = img_w / img_w  # width always fits
            half_h = (img_w / canvas_w * canvas_h) / 2
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(cy - half_h, cy + half_h)
            self._cube_zoom_level = 1.0
        elif mode == 'fit_height':
            half_w = (img_h / canvas_h * canvas_w) / 2
            self.ax.set_xlim(cx - half_w, cx + half_w)
            self.ax.set_ylim(y0, y1)
            self._cube_zoom_level = 1.0
        else:
            # Percentage zoom: zoom_level=1 shows the full image,
            # zoom_level=2 shows half the image (2× magnification), etc.
            half_w = img_w / (2 * zoom_level)
            half_h = img_h / (2 * zoom_level)
            # Keep centre fixed
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            cur_cx = (cur_xlim[0] + cur_xlim[1]) / 2
            cur_cy = (cur_ylim[0] + cur_ylim[1]) / 2
            # Clamp centre against the full field (the larger background extent
            # when one is loaded) so zooming out stays centred and reveals the
            # surrounding field rather than pinning to the cube edge.
            fx0, fx1, fy0, fy1 = self._cube_field_extent() or (x0, x1, y0, y1)
            cur_cx = np.clip(cur_cx, min(fx0 + half_w, fx1 - half_w),
                                     max(fx0 + half_w, fx1 - half_w))
            cur_cy = np.clip(cur_cy, min(fy0 + half_h, fy1 - half_h),
                                     max(fy0 + half_h, fy1 - half_h))
            self.ax.set_xlim(cur_cx - half_w, cur_cx + half_w)
            self.ax.set_ylim(cur_cy - half_h, cur_cy + half_h)
            self._cube_zoom_level = zoom_level

        self.canvas.draw_idle()
        self._cube_update_scrollbars()

    def _cube_zoom_in(self):
        lvls = self._zoom_levels
        idx = min(range(len(lvls)), key=lambda i: abs(lvls[i] - self._cube_zoom_level))
        new_idx = min(idx + 1, len(lvls) - 1)
        self._cube_zoom_level = lvls[new_idx]
        self._sync_zoom_combo()
        self._cube_zoom_apply(zoom_level=self._cube_zoom_level)

    def _cube_zoom_out(self):
        lvls = self._zoom_levels
        idx = min(range(len(lvls)), key=lambda i: abs(lvls[i] - self._cube_zoom_level))
        new_idx = max(idx - 1, 0)
        self._cube_zoom_level = lvls[new_idx]
        self._sync_zoom_combo()
        self._cube_zoom_apply(zoom_level=self._cube_zoom_level)

    def _sync_zoom_combo(self):
        """Update the combo text to reflect current zoom level without triggering signal."""
        pct_map = {
            1/32: "3.125%", 1/16: "6.25%", 1/8: "12.5%", 1/4: "25%",
            1/2: "50%", 1: "100%", 2: "200%", 4: "400%",
            8: "800%", 16: "1600%", 32: "3200%"
        }
        label = pct_map.get(self._cube_zoom_level, f"{int(self._cube_zoom_level*100)}%")
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentText(label)
        self.zoom_combo.blockSignals(False)

    def _cube_zoom_combo_changed(self, text):
        mode_map = {
            "Fit window": ("fit_window", 1.0),
            "Fit width":  ("fit_width",  1.0),
            "Fit height": ("fit_height", 1.0),
        }
        pct_map = {
            "3.125%": 1/32, "6.25%": 1/16, "12.5%": 1/8, "25%": 1/4,
            "50%": 1/2, "100%": 1, "200%": 2, "400%": 4,
            "800%": 8, "1600%": 16, "3200%": 32
        }
        if text in mode_map:
            mode, lvl = mode_map[text]
            self._cube_zoom_level = lvl
            self._cube_zoom_apply(mode=mode)
        elif text in pct_map:
            lvl = pct_map[text]
            self._cube_zoom_level = lvl
            self._cube_zoom_apply(zoom_level=lvl)

    # ── Cube image transform (rotate/flip) ───────────────────────────────────

    def _cube_get_transformed_data(self):
        """Apply current rotation and flip to _last_data and return the result.

        Verified empirically:
          k=1 → 90° CW visual rotation  (with imshow origin=lower)
          k=2 → 180°, k=3 → 270° CW
        _cube_rotation stores k directly (0-3).
        """
        if not hasattr(self, '_last_data') or self._last_data is None:
            return None
        data = self._last_data
        if getattr(self, '_last_from_fits', False):
            img = np.nansum(data, axis=0).astype(np.float64)
        else:
            img = np.array(data, dtype=np.float64)

        self._cube_orig_shape = img.shape  # (ny_orig, nx_orig)

        k = getattr(self, '_cube_rotation', 0) % 4
        if k:
            img = np.rot90(img, k=k)

        if getattr(self, '_cube_flip_h', False):
            img = np.fliplr(img)
        if getattr(self, '_cube_flip_v', False):
            img = np.flipud(img)
        return img

    def _cube_transform_coords(self, xd, yd):
        """Map display coords (after rotate/flip) back to original data coords.

        Inverse formulas verified against np.rot90 + imshow(origin=lower):
          k=0: x_orig=xd,          y_orig=yd
          k=1: x_orig=nx_orig-1-yd, y_orig=xd
          k=2: x_orig=nx_orig-1-xd, y_orig=ny_orig-1-yd
          k=3: x_orig=yd,           y_orig=ny_orig-1-xd
        Flips are applied before rotation in the forward pass, so undo after.
        """
        if not hasattr(self, '_cube_orig_shape'):
            return int(round(xd)), int(round(yd))
        ny_orig, nx_orig = self._cube_orig_shape

        # Undo flips first (they were applied after rotation in the forward pass)
        x, y = xd, yd
        # Need the rotated shape to undo flips correctly
        k = getattr(self, '_cube_rotation', 0) % 4
        if k in (1, 3):
            ny_rot, nx_rot = nx_orig, ny_orig
        else:
            ny_rot, nx_rot = ny_orig, nx_orig
        if getattr(self, '_cube_flip_h', False):
            x = nx_rot - 1 - x
        if getattr(self, '_cube_flip_v', False):
            y = ny_rot - 1 - y

        # Undo rotation
        if k == 0:
            x_orig, y_orig = x, y
        elif k == 1:
            x_orig, y_orig = nx_orig - 1 - y, x
        elif k == 2:
            x_orig, y_orig = nx_orig - 1 - x, ny_orig - 1 - y
        else:  # k == 3
            x_orig, y_orig = y, ny_orig - 1 - x

        return int(round(x_orig)), int(round(y_orig))

    def _cube_redraw_transformed(self):
        """Redraw the cube image with the current transform + toolbar settings."""
        img = self._cube_get_transformed_data()
        if img is None:
            return
        self._applying_transform = True
        try:
            self.draw_image(img, cmap=self._last_cmap, scale=self._last_scale, from_fits=False)
        finally:
            self._applying_transform = False

    def _cube_rotate(self, degrees):
        """Rotate the cube image. degrees=90 → CW, degrees=-90 → CCW."""
        steps = (degrees // 90) % 4
        self._cube_rotation = (getattr(self, '_cube_rotation', 0) + steps) % 4
        self._cube_redraw_transformed()

    def _cube_flip(self, axis):
        """Flip the cube image horizontally ('h') or vertically ('v')."""
        if axis == 'h':
            self._cube_flip_h = not getattr(self, '_cube_flip_h', False)
        else:
            self._cube_flip_v = not getattr(self, '_cube_flip_v', False)
        self._cube_redraw_transformed()

    # ── Cube axis coord label toggle (X/Y ↔ RA/Dec) ──────────────────────────

    def _cube_toggle_coords(self):
        """Toggle cube axis tick labels between pixel X/Y and RA/Dec."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        mode = getattr(self, '_cube_coords_mode', 'xy')
        if mode == 'xy':
            self._cube_coords_mode = 'radec'
            self.coords_toggle_btn.setText('RA/Dec')
            self._cube_apply_radec_labels()
        else:
            self._cube_coords_mode = 'xy'
            self.coords_toggle_btn.setText('X/Y')
            self._cube_apply_xy_labels()
        self.canvas.draw_idle()

    def _cube_apply_xy_labels(self):
        """Set cube axes to plain pixel X/Y ticks."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        self.ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        self.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')
        # Restore auto pixel ticks
        self.ax.xaxis.set_major_locator(plt.AutoLocator())
        self.ax.yaxis.set_major_locator(plt.AutoLocator())

    def _cube_apply_radec_labels(self):
        """Set cube axes to RA/Dec tick labels derived from WCS."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        # Build tick positions and labels from the current pixel grid
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        # Sample ~5 evenly-spaced pixel positions across each axis
        x_pix = np.linspace(max(0, int(xlim[0])), int(xlim[1]), 5).astype(int)
        y_pix = np.linspace(max(0, int(ylim[0])), int(ylim[1]), 5).astype(int)

        try:
            ra_vals,  _ = self.pixel_to_ra_dec(x_pix, np.full_like(x_pix, int((ylim[0]+ylim[1])/2)))
            _,  dec_vals = self.pixel_to_ra_dec(np.full_like(y_pix, int((xlim[0]+xlim[1])/2)), y_pix)

            x_labels = [self.decimal_to_sexagesimal(float(r), is_ra=True)  for r in ra_vals]
            y_labels = [self.decimal_to_sexagesimal(float(d), is_ra=False) for d in dec_vals]

            self.ax.set_xticks(x_pix)
            self.ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
            self.ax.set_yticks(y_pix)
            self.ax.set_yticklabels(y_labels, fontsize=7)
            self.ax.set_xlabel('RA (J2000)')
            self.ax.set_ylabel('Dec (J2000)')
        except Exception as e:
            print(f"RA/Dec label error: {e}")
            self._cube_apply_xy_labels()

    # ── Background HST image overlay ──────────────────────────────────────────

    def _show_bkg_image_dialog(self):
        """Open dialog to select/fetch an HST background image and adjust its display."""
        dialog = QDialog(self)
        dialog.setWindowTitle('Background Image')
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(6)

        # ── Band selection ────────────────────────────────────────────────────
        layout.addWidget(QLabel('Select HST HiPS band:'))
        _hips_i = 'https://alaskybis.cds.unistra.fr/HST-hips/filter_I_hips/'
        _hips_b = 'https://alaskybis.cds.unistra.fr/HST-hips/filter_B_hips/'
        i_btn   = QPushButton('HST  I-band');  i_btn.setFixedHeight(30)
        b_btn   = QPushButton('HST  B-band');  b_btn.setFixedHeight(30)
        # Reference the CXC HiPS by its Aladin HiPS ID; the CDS hips2fits service
        # resolves it to a fast mirror (~0.8 s vs ~4.7 s hitting the Harvard FTP
        # host). 903.9 sq deg coverage, PNG/RGB tiles.
        _hips_cxo = 'cxc.harvard.edu/P/cda/hips/allsky/rgb'
        cxo_btn = QPushButton('Chandra X-ray (broadband RGB)'); cxo_btn.setFixedHeight(30)
        clr_btn = QPushButton('Clear background'); clr_btn.setFixedHeight(26)
        layout.addWidget(i_btn)
        layout.addWidget(b_btn)
        layout.addWidget(cxo_btn)
        layout.addWidget(clr_btn)
        status = QLabel(''); status.setWordWrap(True)
        layout.addWidget(status)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # ── Display settings ──────────────────────────────────────────────────
        layout.addWidget(QLabel('Background display settings:'))

        _combo_qss = (
            "QComboBox { color: #eff0f1; background-color: #3c3f41; border: 1px solid #555; padding: 0 4px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { color: #eff0f1; background-color: #3c3f41;"
            "  selection-background-color: #d7801a; selection-color: #000; border: 1px solid #555; }"
        )

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel('Colormap:'))
        bkg_cmap_combo = QComboBox()
        bkg_cmap_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        bkg_cmap_combo.addItems(['gray','gray_r','viridis','plasma','inferno','magma','hot','cool','Blues','Reds','Greens','bwr','seismic'])
        bkg_cmap_combo.setCurrentText(getattr(self, '_bkg_cmap', 'gray'))
        bkg_cmap_combo.setStyleSheet(_combo_qss)
        cmap_row.addWidget(bkg_cmap_combo)
        layout.addLayout(cmap_row)

        # Scale (transfer function)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel('Scale:'))
        bkg_scale_combo = QComboBox()
        bkg_scale_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        bkg_scale_combo.addItems(['Linear','Log','Square root','Asinh'])
        bkg_scale_combo.setCurrentText(getattr(self, '_bkg_scale', 'Linear'))
        bkg_scale_combo.setStyleSheet(_combo_qss)
        scale_row.addWidget(bkg_scale_combo)
        layout.addLayout(scale_row)

        # Stretch (clip level) — mirrors the cube's stretch menu.
        stretch_row = QHBoxLayout()
        stretch_row.addWidget(QLabel('Stretch:'))
        bkg_stretch_combo = QComboBox()
        bkg_stretch_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        bkg_stretch_combo.addItems(['minmax','99.9%','99.5%','99%','98%','95%','zscale'])
        bkg_stretch_combo.setCurrentText(getattr(self, '_bkg_stretch', '99%'))
        bkg_stretch_combo.setToolTip('Background clip level (zscale = DS9-style)')
        bkg_stretch_combo.setStyleSheet(_combo_qss)
        stretch_row.addWidget(bkg_stretch_combo)
        layout.addLayout(stretch_row)

        # Brightness slider
        bright_row = QHBoxLayout()
        bright_row.addWidget(QLabel('Brightness:'))
        bkg_bright_slider = QtWidgets.QSlider(Qt.Horizontal)
        bkg_bright_slider.setRange(-100, 100)
        bkg_bright_slider.setValue(int(getattr(self, '_bkg_brightness', 0.0) * 100))
        bkg_bright_slider.setFixedHeight(20)
        bright_row.addWidget(bkg_bright_slider)
        layout.addLayout(bright_row)

        # Contrast slider
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel('Contrast:'))
        bkg_contrast_slider = QtWidgets.QSlider(Qt.Horizontal)
        bkg_contrast_slider.setRange(10, 300)
        bkg_contrast_slider.setValue(int(getattr(self, '_bkg_contrast', 1.0) * 100))
        bkg_contrast_slider.setFixedHeight(20)
        contrast_row.addWidget(bkg_contrast_slider)
        layout.addLayout(contrast_row)

        # Manual registration nudge (shifts the overlay by whole cube pixels to
        # correct residual astrometric offsets between the cube WCS and the HST
        # frame; the cube WCS itself is left untouched until it is updated).
        _nudge_spin_qss = (
            "QSpinBox { color:#eff0f1; background-color:#3c3f41; border:1px solid #555;"
            "  padding:1px 4px; min-width:38px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width:0px; border:none; }"
        )
        _nudge_btn_qss = (
            "QPushButton { color:#eff0f1; background-color:#55585a; border:1px solid #666;"
            "  border-radius:3px; font-weight:bold; }"
            "QPushButton:hover { background-color:#d7801a; color:#000; }"
            "QPushButton:pressed { background-color:#ffa02f; }"
        )

        def _make_nudge(val, tip):
            # Spinbox flanked by explicit −/+ buttons. The built-in spinbox
            # arrows don't render reliably under the dark stylesheet on macOS,
            # so we hide them and drive the value with our own buttons.
            sp = QtWidgets.QSpinBox()
            sp.setRange(-500, 500)
            sp.setValue(int(val))
            sp.setSuffix(' px')
            sp.setToolTip(tip)
            sp.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            sp.setAlignment(Qt.AlignCenter)
            sp.setStyleSheet(_nudge_spin_qss)
            sp.setFixedHeight(24)
            minus = QPushButton('−'); plus = QPushButton('+')
            for b in (minus, plus):
                b.setFixedSize(24, 24)
                b.setStyleSheet(_nudge_btn_qss)
            minus.setToolTip(tip); plus.setToolTip(tip)
            minus.clicked.connect(lambda: sp.setValue(sp.value() - 1))
            plus.clicked.connect(lambda: sp.setValue(sp.value() + 1))
            return sp, minus, plus

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel('Align X:'))
        bkg_dx_spin, dx_minus, dx_plus = _make_nudge(
            getattr(self, '_bkg_dx', 0),
            'Shift the background overlay left/right (cube pixels)')
        offset_row.addWidget(dx_minus); offset_row.addWidget(bkg_dx_spin); offset_row.addWidget(dx_plus)
        offset_row.addSpacing(10)
        offset_row.addWidget(QLabel('Y:'))
        bkg_dy_spin, dy_minus, dy_plus = _make_nudge(
            getattr(self, '_bkg_dy', 0),
            'Shift the background overlay up/down (cube pixels)')
        offset_row.addWidget(dy_minus); offset_row.addWidget(bkg_dy_spin); offset_row.addWidget(dy_plus)
        offset_row.addStretch()
        layout.addLayout(offset_row)

        # Write the current alignment into the cube's WCS (re-registers the cube
        # to the background). Backs up the original WCS first so it can be
        # restored. See _lock_in_wcs.
        lock_row = QHBoxLayout()
        lock_btn = QPushButton('💾 Update cube WCS from alignment')
        lock_btn.setFixedHeight(26)
        lock_btn.setToolTip('Apply the Align X/Y shift to the cube FITS WCS so '
                            'the cube is registered to the background '
                            '(saves a backup of the original WCS first)')
        lock_btn.setStyleSheet(
            "QPushButton { background-color:#3c3f41; color:#eff0f1; border:1px solid #666;"
            "  border-radius:3px; padding:2px 8px; }"
            "QPushButton:hover { background-color:#d7801a; color:#000; }"
        )
        lock_row.addWidget(lock_btn)
        restore_btn = QPushButton('↩ Revert cube WCS to original')
        restore_btn.setFixedHeight(26)
        restore_btn.setToolTip('Undo the WCS update — restore the cube WCS from the saved backup')
        restore_btn.setStyleSheet(
            "QPushButton { background-color:#3c3f41; color:#eff0f1; border:1px solid #666;"
            "  border-radius:3px; padding:2px 8px; }"
            "QPushButton:hover { background-color:#55585a; }"
        )
        lock_row.addWidget(restore_btn)
        lock_row.addStretch()
        layout.addLayout(lock_row)

        # ── Live update helpers ───────────────────────────────────────────────
        def _apply_display():
            self._bkg_cmap       = bkg_cmap_combo.currentText()
            self._bkg_scale      = bkg_scale_combo.currentText()
            self._bkg_stretch    = bkg_stretch_combo.currentText()
            self._bkg_brightness = bkg_bright_slider.value() / 100.0
            self._bkg_contrast   = bkg_contrast_slider.value() / 100.0
            self._bkg_dx         = bkg_dx_spin.value()
            self._bkg_dy         = bkg_dy_spin.value()
            if getattr(self, '_bkg_data', None) is not None:
                # Keep the live _bkg_artist reference so _draw_bkg_overlay can
                # remove the previous image instead of leaking one per tick.
                self._draw_bkg_overlay()

        bkg_cmap_combo.currentTextChanged.connect(lambda _: _apply_display())
        bkg_scale_combo.currentTextChanged.connect(lambda _: _apply_display())
        bkg_stretch_combo.currentTextChanged.connect(lambda _: _apply_display())
        bkg_bright_slider.valueChanged.connect(lambda _: _apply_display())
        bkg_contrast_slider.valueChanged.connect(lambda _: _apply_display())
        bkg_dx_spin.valueChanged.connect(lambda _: _apply_display())
        bkg_dy_spin.valueChanged.connect(lambda _: _apply_display())

        lock_btn.clicked.connect(
            lambda: self._lock_in_wcs(bkg_dx_spin, bkg_dy_spin, status))
        restore_btn.clicked.connect(
            lambda: self._restore_original_wcs(bkg_dx_spin, bkg_dy_spin, status))

        def _fetch(hips_url, label):
            status.setText(f'Fetching {label} image…')
            dialog.repaint()
            try:
                self._load_hips_background(hips_url, label, status, dialog)
            except Exception as e:
                status.setText(f'Error: {e}')

        i_btn.clicked.connect(lambda: _fetch(_hips_i, 'HST I-band'))
        cxo_btn.clicked.connect(lambda: _fetch(_hips_cxo, 'Chandra X-ray'))
        b_btn.clicked.connect(lambda: _fetch(_hips_b, 'HST B-band'))
        clr_btn.clicked.connect(lambda: [self._clear_bkg_image(), dialog.accept()])

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec_()

    def _load_hips_background(self, hips_url, label, status_label, dialog):
        """Fetch a HiPS cutout matched to the cube WCS and display it behind the cube."""
        global FITS_HEADER, FITS_DATA

        # Build a minimal WCS from the cube header
        try:
            from astropy.wcs import WCS as AstroWCS
            from astropy.io.fits import Header as FITSHeader
            # Use only the 2D spatial axes from the header
            h = FITSHeader(FITS_HEADER)
            # Make a clean 2-axis WCS header
            wcs_keys = ['NAXIS1','NAXIS2','CRPIX1','CRPIX2','CRVAL1','CRVAL2',
                        'CDELT1','CDELT2','CTYPE1','CTYPE2','CD1_1','CD1_2',
                        'CD2_1','CD2_2','PC1_1','PC1_2','PC2_1','PC2_2']
            wcs_h = FITSHeader()
            wcs_h['NAXIS'] = 2
            wcs_h['WCSAXES'] = 2
            for k in wcs_keys:
                if k in h:
                    wcs_h[k] = h[k]
            if 'NAXIS1' not in wcs_h and FITS_DATA is not None:
                wcs_h['NAXIS1'] = FITS_DATA.shape[-1]
                wcs_h['NAXIS2'] = FITS_DATA.shape[-2]
            wcs2d = AstroWCS(wcs_h)
            # Explicitly set pixel_shape so hips2fits uses the cube's spatial
            # dimensions rather than falling back to its own default size.
            _nx = int(wcs_h['NAXIS1']) if 'NAXIS1' in wcs_h else int(FITS_DATA.shape[-1])
            _ny = int(wcs_h['NAXIS2']) if 'NAXIS2' in wcs_h else int(FITS_DATA.shape[-2])
            wcs2d.pixel_shape = (_nx, _ny)  # FITS order: (NAXIS1, NAXIS2)

            # Expand the footprint so we fetch a larger surrounding field than
            # the cube itself. The viewport is left at the current zoom (see
            # _draw_bkg_overlay); the user can zoom out to explore this wider
            # field. Shift CRPIX so the same sky coords map into the padded
            # grid, then enlarge pixel_shape.
            BKG_FIELD_FACTOR = 3.0  # fetched field ≈ 3× the cube FOV per axis
            pad_x = int(round(_nx * (BKG_FIELD_FACTOR - 1) / 2))
            pad_y = int(round(_ny * (BKG_FIELD_FACTOR - 1) / 2))
            big_nx = _nx + 2 * pad_x
            big_ny = _ny + 2 * pad_y
            try:
                wcs2d.wcs.crpix = [wcs2d.wcs.crpix[0] + pad_x,
                                   wcs2d.wcs.crpix[1] + pad_y]
            except Exception:
                pass
            wcs2d.pixel_shape = (big_nx, big_ny)
        except Exception as e:
            if status_label: status_label.setText(f'WCS error: {e}')
            return

        # Fetch via astroquery.hips2fits.query_with_wcs
        try:
            from astroquery.hips2fits import hips2fits
            print(f"Querying hips2fits for: {hips_url}")
            # PNG-only HiPS (e.g. Chandra CXC RGB) needs format='jpg'; FITS HiPS
            # use 'fits'. Match any Chandra/CXC reference (FTP URL, HiPS ID, or
            # CDS mirror).
            _u = hips_url.lower()
            _fmt = 'jpg' if ('cxc' in _u or 'chandra' in _u) else 'fits'
            result = hips2fits.query_with_wcs(
                hips=hips_url,
                wcs=wcs2d,
                format=_fmt,
                min_cut=0.5,
                max_cut=99.5,
                stretch='linear',
            )
            # FITS → HDUList; JPG/PNG → numpy array
            if _fmt == 'fits':
                bkg_data = result[0].data if hasattr(result, '__getitem__') else result.data
            else:
                # JPG/PNG arrays are top-row-first (display order); FITS arrays
                # are bottom-row-first. The overlay is drawn with origin='lower',
                # so flip the image rows to match the FITS convention (otherwise
                # the Chandra RGB appears vertically mirrored).
                bkg_data = np.flipud(np.array(result))  # (ny, nx, 3) or (3, ny, nx)
            print(f"hips2fits returned data shape: {np.array(bkg_data).shape}")
        except ImportError:
            # Fallback: plain HTTP fetch of the hips2fits endpoint
            import urllib.request, io
            nx = int(wcs2d.pixel_shape[0] if wcs2d.pixel_shape else
                     FITS_DATA.shape[-1] if FITS_DATA is not None else 256)
            ny = int(wcs2d.pixel_shape[1] if wcs2d.pixel_shape else
                     FITS_DATA.shape[-2] if FITS_DATA is not None else 256)
            ra  = float(wcs2d.wcs.crval[0])
            dec = float(wcs2d.wcs.crval[1])
            cdelt = abs(float(wcs2d.wcs.cdelt[0])) if wcs2d.wcs.cdelt[0] != 0 else 1/3600
            fov = cdelt * max(nx, ny)
            # Strip trailing slash for the hips parameter
            _hips_param = hips_url.rstrip('/')
            url = (f'https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits?'
                   f'hips={urllib.request.quote(_hips_param)}'
                   f'&width={nx}&height={ny}&fov={fov:.6f}'
                   f'&ra={ra:.6f}&dec={dec:.6f}&projection=TAN'
                   f'&format=fits')
            req = urllib.request.Request(url, headers={'User-Agent': 'HyperCube/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            from astropy.io import fits as astrofits
            with astrofits.open(io.BytesIO(raw)) as hdul:
                bkg_data = hdul[0].data

        if bkg_data is None:
            if status_label: status_label.setText('No data returned.')
            return

        # If RGB/RGBA (PNG HiPS like Chandra), convert to grayscale luminosity
        bkg_data = np.array(bkg_data)
        if bkg_data.ndim == 3:
            if bkg_data.shape[0] in (3, 4):   # (bands, ny, nx) from astroquery
                rgb = bkg_data[:3].astype(np.float64)
            elif bkg_data.shape[2] in (3, 4): # (ny, nx, bands)
                rgb = bkg_data[:, :, :3].astype(np.float64).transpose(2, 0, 1)
            else:
                rgb = bkg_data.astype(np.float64)
            # ITU-R BT.601 luminosity
            bkg_data = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

        # Resample to the enlarged grid at the cube's pixel scale (1 background
        # pixel = 1 cube pixel) if hips2fits returned a different size — the CDS
        # service ignores NAXIS1/NAXIS2 and uses its own default.
        cube_ny = int(FITS_DATA.shape[-2])
        cube_nx = int(FITS_DATA.shape[-1])
        if bkg_data.shape != (big_ny, big_nx):
            from scipy.ndimage import zoom as _zoom
            zy = big_ny / bkg_data.shape[0]
            zx = big_nx / bkg_data.shape[1]
            # grid_mode=True aligns pixel *edges* (preserving the WCS footprint)
            # rather than first/last pixel centres. The default (grid_mode=False)
            # shrinks the footprint by a sub-pixel fraction and introduces a
            # position-dependent shift, which misaligns the cube from the
            # background whenever the returned size differs from the request.
            bkg_data = _zoom(bkg_data.astype(float), (zy, zx),
                             order=1, grid_mode=True, mode='nearest')
            print(f"Background resampled to field shape: {bkg_data.shape}")

        # Extent in cube-pixel coordinates: padded-grid pixel j maps to cube
        # pixel j - pad, so the field extends `pad` pixels beyond the cube on
        # every side. (imshow origin='lower' places pixel edges at ±0.5.)
        self._bkg_extent = (-pad_x - 0.5, cube_nx + pad_x - 0.5,
                            -pad_y - 0.5, cube_ny + pad_y - 0.5)

        # Store and display
        self._bkg_data = bkg_data.astype(np.float64)
        self._bkg_label = label
        self._draw_bkg_overlay()
        self.cube_opacity_slider.setVisible(True)

        if status_label:
            status_label.setText(f'✓  {label} loaded ({bkg_data.shape[1]}×{bkg_data.shape[0]} px)')

    def _draw_bkg_overlay(self):
        """Render the background image and cube overlay on self.ax."""
        if not hasattr(self, '_bkg_data') or self._bkg_data is None:
            return
        if not hasattr(self, 'ax') or self.ax is None:
            return

        bkg = self._bkg_data.copy()
        # Clip level (stretch). Matches the cube's stretch menu: percentile
        # clips or DS9-style zscale.
        bmin, bmax = self._stretch_limits(bkg, getattr(self, '_bkg_stretch', '99%'))

        # Apply scale (transfer function).
        # Normalise to [0,1] FIRST, then stretch. Applying log/asinh to the raw
        # data is ~linear when its values sit in a small range (e.g. HiPS
        # cutouts), which is why "Log" looked identical to "Linear". The
        # parametrised stretches below (astropy LogStretch a=1000 / AsinhStretch
        # a=0.1 / SqrtStretch) bend consistently regardless of data magnitude.
        scale = getattr(self, '_bkg_scale', 'Linear')
        span = bmax - bmin
        if span > 0:
            bkg = np.clip((bkg - bmin) / span, 0, 1)
        else:
            bkg = np.zeros_like(bkg)

        if scale == 'Log':
            a = 1000.0
            bkg = np.log(a * bkg + 1.0) / np.log(a + 1.0)
        elif scale == 'Square root':
            bkg = np.sqrt(bkg)
        elif scale == 'Asinh':
            a = 0.1
            bkg = np.arcsinh(bkg / a) / np.arcsinh(1.0 / a)
        # Linear: leave as-is

        # Apply brightness and contrast
        brightness = getattr(self, '_bkg_brightness', 0.0)
        contrast   = getattr(self, '_bkg_contrast',   1.0)
        mid = 0.5
        bkg_norm = np.clip((bkg - mid) * contrast + mid + brightness, 0, 1)

        # Preserve the current viewport so loading/redrawing a background never
        # changes the zoom — the user can zoom out afterwards to explore the
        # wider field that now extends beyond the cube.
        prev_xlim = self.ax.get_xlim()
        prev_ylim = self.ax.get_ylim()

        # Remove any existing bkg artist
        if hasattr(self, '_bkg_artist') and self._bkg_artist is not None:
            try: self._bkg_artist.remove()
            except: pass
        # Use the stored field extent (larger than the cube) when available,
        # falling back to the cube's extent for legacy data.
        bkg_extent = getattr(self, '_bkg_extent', None)
        if bkg_extent is None and self.ax.images:
            bkg_extent = self.ax.images[0].get_extent()
        # Apply the manual registration nudge (in cube pixels) on top of the
        # WCS-derived extent.
        dx = getattr(self, '_bkg_dx', 0)
        dy = getattr(self, '_bkg_dy', 0)
        if bkg_extent is not None and (dx or dy):
            x0, x1, y0, y1 = bkg_extent
            bkg_extent = (x0 + dx, x1 + dx, y0 + dy, y1 + dy)
        self._bkg_artist = self.ax.imshow(
            bkg_norm, origin='lower', cmap=getattr(self, '_bkg_cmap', 'gray'),
            extent=bkg_extent,
            zorder=0, alpha=1.0
        )
        # Set cube image alpha from slider
        opacity = self.cube_opacity_slider.value() / 100.0
        for img in self.ax.images:
            if img is not self._bkg_artist:
                img.set_alpha(opacity)
                img.set_zorder(1)
        # Restore the viewport (imshow can trigger an autoscale).
        self.ax.set_xlim(prev_xlim)
        self.ax.set_ylim(prev_ylim)
        self.canvas.draw_idle()

    def _cube_opacity_changed(self, value):
        """Update cube overlay alpha when slider moves."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        alpha = value / 100.0
        bkg_artist = getattr(self, '_bkg_artist', None)
        for img in self.ax.images:
            if img is not bkg_artist:
                img.set_alpha(alpha)
        self.canvas.draw_idle()

    def _clear_bkg_image(self):
        """Remove the background image and reset cube opacity."""
        if hasattr(self, '_bkg_artist') and self._bkg_artist is not None:
            try: self._bkg_artist.remove()
            except: pass
            self._bkg_artist = None
        self._bkg_data = None
        self._bkg_extent = None
        # Restore full opacity on cube
        if hasattr(self, 'ax') and self.ax is not None:
            for img in self.ax.images:
                img.set_alpha(1.0)
            self.canvas.draw_idle()
        self.cube_opacity_slider.setVisible(False)
        self.cube_opacity_slider.setValue(100)

    # ── WCS re-registration ───────────────────────────────────────────────────

    def _wcs_backup_path(self):
        """Path of the sidecar FITS that stores the cube's original WCS."""
        path = getattr(self, 'fits_path', None)
        if not path:
            return None
        base, _ext = os.path.splitext(path)
        return base + '_oldWCS.fits'

    def _lock_in_wcs(self, dx_spin, dy_spin, status_label):
        """Bake the current Align X/Y shift into the cube's FITS WCS.

        The Align nudge moves the background overlay by (dx, dy) cube pixels to
        sit on top of the cube data. Locking that in re-registers the cube to
        the background by shifting the spatial reference pixel: a feature now at
        cube pixel (p + dx, p + dy) takes on the sky coordinate the old WCS
        assigned to pixel p, i.e. CRPIX += (dx, dy).

        The original WCS is saved to a sidecar `*_oldWCS.fits` (once) before the
        first edit so it can always be restored. The on-disk cube header is
        updated in place; the already-fetched overlay stays put because the
        shift is folded into its extent and the nudge is reset to 0.
        """
        from astropy.io import fits as _fits
        path = getattr(self, 'fits_path', None)
        if not path or not os.path.exists(path):
            if status_label: status_label.setText('No cube file on disk to update.')
            return
        dx = int(dx_spin.value()); dy = int(dy_spin.value())
        if dx == 0 and dy == 0:
            if status_label: status_label.setText('Align X/Y are 0 — nothing to lock in.')
            return

        ext = getattr(self, 'fits_ext', 0)
        try:
            with _fits.open(path, mode='update') as hdul:
                # Locate the HDU holding the spatial WCS (prefer the tracked
                # extension; fall back to the first HDU that has CRPIX1).
                if not (ext < len(hdul) and 'CRPIX1' in hdul[ext].header):
                    ext = next((i for i in range(len(hdul))
                                if 'CRPIX1' in hdul[i].header), ext)
                hdr = hdul[ext].header
                if 'CRPIX1' not in hdr or 'CRPIX2' not in hdr:
                    if status_label: status_label.setText('Cube header has no CRPIX1/2 to adjust.')
                    return

                # Save the original WCS once, before the first modification.
                # NAXIS1/NAXIS2 are intentionally excluded: a header-only
                # PrimaryHDU has NAXIS==0 and would fail verification if they
                # were present (and they never change, so restore doesn't need
                # them).
                bkpath = self._wcs_backup_path()
                if bkpath and not os.path.exists(bkpath):
                    wcs_keys = ['WCSAXES','CRPIX1','CRPIX2',
                                'CRVAL1','CRVAL2','CDELT1','CDELT2','CTYPE1','CTYPE2',
                                'CUNIT1','CUNIT2','CD1_1','CD1_2','CD2_1','CD2_2',
                                'PC1_1','PC1_2','PC2_1','PC2_2','CROTA2','RADESYS','EQUINOX']
                    bk = _fits.Header()
                    bk['ORIGFILE'] = (os.path.basename(path), 'Cube this WCS came from')
                    bk['WCSEXT'] = (ext, 'HDU extension of the WCS')
                    for k in wcs_keys:
                        if k in hdr:
                            bk[k] = hdr[k]
                    _fits.PrimaryHDU(header=bk).writeto(
                        bkpath, overwrite=False, output_verify='silentfix')

                # Apply the shift to the reference pixel.
                hdr['CRPIX1'] = float(hdr['CRPIX1']) + dx
                hdr['CRPIX2'] = float(hdr['CRPIX2']) + dy
                hdr['HISTORY'] = (f'WCS re-registered to background: '
                                  f'CRPIX += ({dx}, {dy}) px [HyperCube]')
                hdul.flush()
        except Exception as e:
            if status_label: status_label.setText(f'Lock-in failed: {e}')
            return

        # Update the in-memory header/globals so the running session matches.
        # FITS_HEADER and self.fits_header are often the same object, so dedupe
        # by identity to avoid applying the shift twice.
        global FITS_HEADER
        seen = []
        for h in (getattr(self, 'fits_header', None), FITS_HEADER):
            if h is not None and 'CRPIX1' in h and all(h is not s for s in seen):
                h['CRPIX1'] = float(h['CRPIX1']) + dx
                h['CRPIX2'] = float(h['CRPIX2']) + dy
                seen.append(h)

        # Fold the nudge into the overlay extent so the view is unchanged, then
        # zero the nudge (future fetches use the new WCS and need no offset).
        ext_now = getattr(self, '_bkg_extent', None)
        if ext_now is not None:
            x0, x1, y0, y1 = ext_now
            self._bkg_extent = (x0 + dx, x1 + dx, y0 + dy, y1 + dy)
        self._bkg_dx = 0
        self._bkg_dy = 0
        for sp in (dx_spin, dy_spin):
            sp.blockSignals(True); sp.setValue(0); sp.blockSignals(False)
        if getattr(self, '_bkg_data', None) is not None:
            self._draw_bkg_overlay()

        bkname = os.path.basename(self._wcs_backup_path() or '')
        if status_label:
            status_label.setText(f'✓ WCS locked in (CRPIX += {dx}, {dy}). '
                                 f'Original saved to {bkname}.')

    def _restore_original_wcs(self, dx_spin, dy_spin, status_label):
        """Restore the cube WCS from the `*_oldWCS.fits` backup."""
        from astropy.io import fits as _fits
        path = getattr(self, 'fits_path', None)
        bkpath = self._wcs_backup_path()
        if not path or not bkpath or not os.path.exists(bkpath):
            if status_label: status_label.setText('No saved WCS backup found to restore.')
            return
        try:
            bk = _fits.getheader(bkpath)
            ext = int(bk.get('WCSEXT', getattr(self, 'fits_ext', 0)))
            restore_keys = [k for k in bk.keys()
                            if k not in ('SIMPLE','BITPIX','NAXIS','EXTEND',
                                         'ORIGFILE','WCSEXT','COMMENT','HISTORY')]
            with _fits.open(path, mode='update') as hdul:
                if not (ext < len(hdul) and 'CRPIX1' in hdul[ext].header):
                    ext = next((i for i in range(len(hdul))
                                if 'CRPIX1' in hdul[i].header), ext)
                hdr = hdul[ext].header
                for k in restore_keys:
                    hdr[k] = bk[k]
                hdr['HISTORY'] = 'WCS restored from backup [HyperCube]'
                hdul.flush()
                new_hdr = hdul[ext].header.copy()
        except Exception as e:
            if status_label: status_label.setText(f'Restore failed: {e}')
            return

        # Sync in-memory header/globals.
        global FITS_HEADER
        for h in (getattr(self, 'fits_header', None), FITS_HEADER):
            if h is not None:
                for k in restore_keys:
                    if k in new_hdr:
                        h[k] = new_hdr[k]

        self._bkg_dx = 0
        self._bkg_dy = 0
        for sp in (dx_spin, dy_spin):
            sp.blockSignals(True); sp.setValue(0); sp.blockSignals(False)
        if status_label:
            status_label.setText('✓ Original WCS restored. Re-fetch the '
                                 'background to realign to it.')

    def _cube_reset_view(self):
        """Reset the cube viewport to its state at FITS load time:
        - B/W (gray) colormap
        - Linear scaling, 99% stretch
        - Fit-window zoom (full image visible)
        """
        if not hasattr(self, '_last_data') or self._last_data is None:
            return

        # Reset toolbar combos (blockSignals to avoid double-redraw)
        self.scale_combo.blockSignals(True)
        self.stretch_combo.blockSignals(True)
        self.scale_combo.setCurrentText("Linear")
        self.stretch_combo.setCurrentText("99%")
        self.scale_combo.blockSignals(False)
        self.stretch_combo.blockSignals(False)

        # Redraw with gray cmap, original data
        if FITS_DATA is not None:
            self.draw_image(FITS_DATA, cmap='gray', scale='linear', from_fits=True)
        elif self._last_data is not None:
            self.draw_image(self._last_data, cmap='gray', scale='linear',
                            from_fits=getattr(self, '_last_from_fits', False))

        # Reset zoom to fit window
        self._cube_zoom_apply(mode='fit_window')
        self._sync_zoom_combo()

    def _cube_redraw_with_current_settings(self):
        """Redraw the cube image using current toolbar stretch and scale settings."""
        if not hasattr(self, '_last_data') or self._last_data is None:
            return
        self.draw_image(
            self._last_data,
            cmap=self._last_cmap,
            scale=self._last_scale,
            from_fits=self._last_from_fits
        )

    def _init_placeholder_canvas(self):
        """Draw the HyperCube logo centred on a dark canvas before a FITS file is loaded."""
        fig = self.canvas.figure
        fig.patch.set_facecolor('#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)

        try:
            logo_path = resource_path('hypercube_logo.png')
            logo = plt.imread(logo_path)
            # Place logo in a centred inset axes so it scales with the window
            logo_ax = fig.add_axes([0.25, 0.2, 0.5, 0.5])
            logo_ax.imshow(logo)
            logo_ax.axis('off')
        except Exception:
            # Fallback to text if image can't be loaded
            ax.text(0.5, 0.55, 'HYPERCUBE',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=28, fontweight='bold', color='#d7801a',
                    fontfamily='monospace')

        ax.text(0.5, 0.1, 'Open a FITS file to begin',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='#888888')

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas.draw()

    def set_fit_params_window(self, fit_params_window):
        """Store the reference to FitParamsWindow after it's created."""
        self.fit_params_window = fit_params_window


    def on_mouse_move(self, event):
        global spectrum
        """Update rectangle position and spectrum in right panel."""
        if self.locked:
            return  # Do nothing if locked
        if getattr(self, '_bc_drag_active', False):
            return  # Suppress spaxel update during brightness/contrast drag
        
        if (event.inaxes) and (self.is_1d_spectrum == False):
            # Round to nearest integer spaxel centre in display coords.
            _xd = int(np.floor(event.xdata + 0.5))
            _yd = int(np.floor(event.ydata + 0.5))
            # Map display coords back to original data coords (undo rotate/flip)
            x, y = self._cube_transform_coords(_xd, _yd)
            self.cursor_pos = (x, y)
            
            if self.fits_data is not None:
                self.current_spaxel = (x, y)
                spectrum = self.get_spectrum_at_spaxel(x, y)
                self.update_spectrum_panel(spectrum)
                
                # Update red rectangle position
                self.red_rect.set_xy((x - 0.5, y - 0.5))
                self.red_rect.set_visible(True)
                self.canvas.draw_idle()
                
                # Always update the spaxel info overlay on the image
                self.update_spaxel_overlay(x, y)
                
                # Update fit result buttons in the side panel (only when it exists)
                if self.fit_params_window is not None:
                    self.update_buttons(x, y)




    def _spectrum_hbar_pan(self, value):
        """Pan the spectrum axes horizontally from the scrollbar position (0-1000)."""
        if not hasattr(self, 'spectrum_ax') or self.spectrum_ax is None:
            return
        if not hasattr(self, '_spec_wav_min') or self._spec_wav_min is None:
            return
        xlim = self.spectrum_ax.get_xlim()
        view_w = xlim[1] - xlim[0]
        full_w = self._spec_wav_max - self._spec_wav_min
        pan_range = full_w - view_w
        if pan_range <= 0:
            return
        frac = value / 1000.0
        new_x0 = self._spec_wav_min + frac * pan_range
        self.spectrum_ax.set_xlim(new_x0, new_x0 + view_w)
        self.spectrum_canvas.draw_idle()

    def _spectrum_update_hbar(self):
        """Sync the spectrum scrollbar to the current xlim.
        Shows/hides it based on whether the view is zoomed in."""
        if not hasattr(self, 'spec_hbar') or not hasattr(self, 'spectrum_ax'):
            return
        if self.spectrum_ax is None or not hasattr(self, '_spec_wav_min'):
            return
        xlim = self.spectrum_ax.get_xlim()
        view_w = xlim[1] - xlim[0]
        full_w = self._spec_wav_max - self._spec_wav_min
        pan_range = full_w - view_w
        if pan_range <= 1e-6:
            self.spec_hbar.hide()
            return
        self.spec_hbar.show()
        frac = (xlim[0] - self._spec_wav_min) / pan_range
        self.spec_hbar.blockSignals(True)
        self.spec_hbar.setValue(int(np.clip(frac, 0, 1) * 1000))
        self.spec_hbar.blockSignals(False)

    def on_spectrum_mouse_move(self, event):
        """Handle mouse movement in the spectrum plot."""
        if event.inaxes == self.spectrum_ax:
            # Update the cursor position in the spectrum plot
            self.spectrum_cursor_pos = (event.xdata, event.ydata)

            # Show a horizontal-resize cursor when hovering over a locked span
            if (self._chanmap_locked and self._chanmap_span is not None
                    and event.xdata is not None and not self._chanmap_active):
                wav0, wav1 = self._span_x_extent(self._chanmap_span)
                if wav0 <= event.xdata <= wav1:
                    self.spectrum_canvas.setCursor(Qt.PointingHandCursor)
                else:
                    self.spectrum_canvas.setCursor(Qt.ArrowCursor)
            elif not self._chanmap_drag_active:
                self.spectrum_canvas.setCursor(Qt.ArrowCursor)

            # Update channel map span while C is held
            if self._chanmap_active and self._chanmap_span is not None and event.xdata is not None:
                pass  # handled below
            # Update any active subtraction window spans
            if event.xdata is not None:
                for _key, _sm in self._submap.items():
                    if _sm['active'] and _sm['span'] is not None:
                        _x0 = min(_sm['start'], event.xdata)
                        _x1 = max(_sm['start'], event.xdata)
                        self._span_set_x_extent(_sm['span'], _x0, _x1)
            # Update channel map span while C is held (main)
            if self._chanmap_active and self._chanmap_span is not None and event.xdata is not None:
                wav = event.xdata
                x0 = min(self._chanmap_start, wav)
                x1 = max(self._chanmap_start, wav)
                self._span_set_x_extent(self._chanmap_span, x0, x1)
                self.spectrum_canvas.draw_idle()

            # Update wavelength/flux overlay
            if hasattr(self, 'spectrum_info_text') and event.xdata is not None:
                _pix = int(np.argmin(np.abs(wavelengths - event.xdata))) if wavelengths is not None else "—"
                self.spectrum_info_text.set_text(
                    f"px  {_pix}\n\u03bb  {event.xdata:.4f}\nF  {event.ydata:.4e}"
                )
                self.spectrum_canvas.draw_idle()
    
            # If we are in the middle of drawing a line, update the line's end position
            if self.drawing_line and self.current_line:
                # Get current position and update the line end coordinates
                end_x, end_y = self.spectrum_cursor_pos
                self.current_line.set_xdata([self.line_start[0], end_x])
                self.current_line.set_ydata([self.line_start[1], end_y])

                # Redraw the canvas to reflect the updated line
                self.spectrum_canvas.draw()

            # If we are drawing a spline, update the live rubber-band segment
            if self._spline_drawing and self._spline_knots:
                self._spline_redraw_preview()

            # If we are selecting a polynomial range, update the live preview
            if self._poly_drawing and self._poly_x1 is not None:
                self._poly_redraw_preview()


    def on_mouse_press(self, event):
        """Start zoom selection, or begin dragging a locked channel map span."""
        if event.button == 1 and event.inaxes == self.spectrum_ax and event.xdata is not None:
            # If click lands inside a locked channel map span, enter drag mode
            if (self._chanmap_locked and self._chanmap_span is not None
                    and not self._chanmap_active):
                wav0, wav1 = self._span_x_extent(self._chanmap_span)
                if wav0 <= event.xdata <= wav1:
                    self._chanmap_drag_active      = True
                    self._chanmap_drag_x_start     = event.xdata
                    self._chanmap_drag_wav_start   = (wav0, wav1)
                    self._chanmap_drag_last_centre = (wav0 + wav1) / 2
                    return  # consume event — don't start zoom
            # Normal zoom start
            self.zoom_start_x = event.xdata
            if self.zoom_rect:
                self.zoom_rect.remove()
            self.zoom_rect = self.spectrum_ax.axvspan(self.zoom_start_x, self.zoom_start_x, color='grey', alpha=0.3)
            self.spectrum_canvas.draw_idle()

    def on_mouse_drag(self, event):
        """Update zoom selection rectangle, or slide a dragged channel map span."""
        if event.button == 1 and event.inaxes == self.spectrum_ax and event.xdata is not None:
            # Channel map span drag
            if self._chanmap_drag_active and self._chanmap_drag_wav_start is not None:
                wav0_s, wav1_s = self._chanmap_drag_wav_start
                delta   = event.xdata - self._chanmap_drag_x_start
                new_w0  = wav0_s + delta
                new_w1  = wav1_s + delta
                self._span_set_x_extent(self._chanmap_span, new_w0, new_w1)
                self._chanmap_draw_handles()
                self._chanmap_update_centre_display()
                self.spectrum_canvas.draw_idle()
                # Recompute channel map live, throttled to one update per wavelength channel
                new_centre = (new_w0 + new_w1) / 2
                last_centre = getattr(self, '_chanmap_drag_last_centre', None)
                chan_step = abs(wavelengths[1] - wavelengths[0]) if wavelengths is not None and len(wavelengths) > 1 else 0
                if last_centre is None or abs(new_centre - last_centre) >= chan_step:
                    self._chanmap_drag_last_centre = new_centre
                    self._compute_channel_map(new_w0, new_w1)
                return  # don't update zoom rect
            # Normal zoom drag
            if self.zoom_start_x is not None:
                x_end = event.xdata
                if self.zoom_rect:
                    self.zoom_rect.remove()
                self.zoom_rect = self.spectrum_ax.axvspan(self.zoom_start_x, x_end, color='grey', alpha=0.3)
                self.spectrum_canvas.draw_idle()

    def on_mouse_release(self, event):
        """Apply zoom when mouse is released, or finalise a channel map span drag."""
        if event.button == 1 and event.inaxes == self.spectrum_ax:
            # Finalise channel map drag
            if self._chanmap_drag_active:
                self._chanmap_drag_active   = False
                self._chanmap_drag_x_start  = None
                self._chanmap_drag_wav_start = None
                if self._chanmap_span is not None:
                    wav0, wav1 = self._span_x_extent(self._chanmap_span)
                    self._compute_channel_map(wav0, wav1)
                return  # don't trigger zoom

            # Normal zoom release
            if self.zoom_start_x is not None:
                zoom_end_x = event.xdata
                if zoom_end_x is not None and self.zoom_start_x != zoom_end_x:
                    x_min, x_max = min(self.zoom_start_x, zoom_end_x), max(self.zoom_start_x, zoom_end_x)
                    self.spectrum_ax.set_xlim(x_min, x_max)
                    self._spectrum_update_hbar()
                    self.zoom_limits = (x_min, x_max)
                    self.zoom_active = True

                    mask = (wavelengths >= x_min) & (wavelengths <= x_max)
                    if np.any(mask):
                        y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()[mask]), np.nanmax(self.spectrum_line.get_ydata()[mask])
                    else:
                        y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()), np.nanmax(self.spectrum_line.get_ydata())

                    padding = self.zoom_pad * (y_max - y_min)
                    self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)
                    self.spectrum_ax.figure.canvas.draw_idle()

                if self.zoom_rect:
                    self.zoom_rect.remove()
                    self.zoom_rect = None
                self.zoom_start_x = None

    def show_context_menu(self, event):
        if event.button == 3:  # Right-click
            menu = QMenu(self)
            reset_action = menu.addAction("Reset Zoom")
            action = menu.exec_(self.spectrum_canvas.mapToGlobal(event.guiEvent.pos()))
            if action == reset_action:
                self.reset_zoom()

    def reset_zoom(self):
        """Reset zoom to the full spectrum range for both x and y axes."""
        self.spectrum_ax.set_xlim(wavelengths.min(), wavelengths.max())
    
        # Compute full y-range and apply padding
        y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()), np.nanmax(self.spectrum_line.get_ydata())
        padding = self.zoom_pad * (y_max - y_min)
        self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)
    
        self.zoom_active = False
        self.spectrum_canvas.draw_idle()  # Ensure the reset happens immediately

    

    def press_scaleflux_button(self):
        text_box = QLineEdit()
        text_box.setText(str(self.fluxscalefactor))  # Set the initial value from the dataframe
        text_box.show()
        text_box.returnPressed.connect(lambda: self.scale_flux(text_box))
        
    def press_scaleWL_button(self):
        text_box = QLineEdit()
        text_box.setText(str(self.WLscalefactor))  # Set the initial value from the dataframe
        text_box.show()
        text_box.returnPressed.connect(lambda: self.scale_WL(text_box))

    def scale_flux(self,text_box):
        global FITS_DATA, ERROR_CUBE
        self.fluxscalefactor = text_box.text()
        FITS_DATA = FITS_DATA*np.float64(self.fluxscalefactor)
        # The measurement errors are in the same units as the flux, so they
        # follow the same rescaling.
        if ERROR_CUBE is not None:
            ERROR_CUBE = ERROR_CUBE*np.float64(self.fluxscalefactor)
        text_box.hide()
        
    def scale_WL(self,text_box):
        global wavelengths
        self.WLscalefactor = text_box.text()
        wavelengths = wavelengths*np.float64(self.WLscalefactor)
        text_box.hide()

    def open_fits_file(self):
        """Opens a FITS file and stores header & data, handling 1D spectra, 3D cubes, and bintables"""
        print('Opening file...')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "FITS Files (*.fits);;All Files (*)", options=options)
        if file_path:
            self._load_fits_from_path(file_path)

    def _load_fits_from_path(self, file_path, for_session=False):
        """Load a FITS file from a path (shared by the Open dialog and session
        restore). When for_session=True, the observation-info extraction and
        the source-resolve popup are skipped because the caller restores
        df_obs from the saved session instead."""
        global FITS_HEADER, FITS_DATA, wavelengths, snr_map, spectrum
        if file_path:
            self.fits_path = file_path
            self.fits_ext = 0
            with fits.open(file_path) as hdul:
                self.fits_header = None
                self.fits_data = None
                self.is_1d_spectrum = False
                self.is_bintable = False
                self.is_3d_cube = False
                self.column_names = []
                self.has_explicit_wavelengths = False
    
                # Check extensions for data type
                for ext in range(len(hdul)):
                    header = hdul[ext].header
                    data = hdul[ext].data
    
                    if data is None:
                        continue
                        
                    # Check for binary table
                    if header.get('XTENSION') == 'BINTABLE':
                        print(f"Found binary table in extension {ext}")
                        self.fits_header = header
                        self.fits_data = data
                        self.is_bintable = True
                        spectrum = data
                        self.is_1d_spectrum = True
                        self.column_names = [header.get(f'TTYPE{i}') for i in range(1, header.get('TFIELDS', 0)+1)]
                        if any(col and 'wave' in col.lower() for col in self.column_names):
                            self.has_explicit_wavelengths = True
                        break
                    
                    # Check for 1D spectrum
                    elif data.ndim == 1:
                        print(f"Found 1D spectrum in extension {ext}")
                        self.fits_header = header
                        self.fits_data = data
                        spectrum = data
                        self.is_1d_spectrum = True
                        self.has_explicit_wavelengths = True
                        if any(key in header for key in ['CRVAL1', 'CDELT1', 'CD1_1']):
                            wavelengths = self._construct_wavelengths_from_header()
                        break
                    
                    # Check for 3D/4D cube (like ALMA data)
                    elif data.ndim in [3, 4] and 'CTYPE3' in header and 'CRVAL3' in header:
                        print(f"Found {data.ndim}D cube in extension {ext}")
                        self.fits_header = header
                        self.fits_data = data
                        self.fits_ext = ext
                        self.is_3d_cube = True
                        
                        # For ALMA-style cubes (1, nchan, ny, nx)
                        if data.ndim == 4 and data.shape[0] == 1:  # Stokes axis
                            self.fits_data = data[0]  # Remove stokes dimension
                        
                        # Generate wavelength array from header
                        if 'CTYPE3' in header and header['CTYPE3'].startswith(('FREQ','VELO','VRAD','WAVE')):
                            wavelengths = self._construct_wavelengths_from_header(axis=3)
                            self.has_explicit_wavelengths = True
                        break
    
                # Fallback to primary extension
                if self.fits_header is None:
                    print("No valid extension found, defaulting to primary.")
                    self.fits_header = hdul[0].header
                    self.fits_data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                    if self.fits_data.ndim == 1:
                        self.is_1d_spectrum = True
                        if any(key in self.fits_header for key in ['CRVAL1', 'CDELT1', 'CD1_1']):
                            self.has_explicit_wavelengths = True
    
                FITS_HEADER = self.fits_header
                FITS_DATA = self.fits_data
                
                # Handle different data types
                if self.is_bintable:
                    print("Binary table detected - showing column selection")
                    self.show_column_selection_dialog()
                    return
                    
                elif self.is_1d_spectrum:
                    print("1D spectrum detected")
                    if not self.has_explicit_wavelengths:
                        print("Warning: No wavelength info found - using pixel indices")
                        wavelengths = np.arange(len(FITS_DATA))
                    self.process_1d_spectrum()
                    
                elif self.is_3d_cube:
                    print(f"{FITS_DATA.ndim}D cube detected")
                    if FITS_DATA.ndim == 3:  # (nchan, ny, nx)
                        self.process_3d_cube()
                        # For 3D cubes, create field selection dialog
                    #     self.column_names = ['Flux']  # Default name
                    #     self.show_column_selection_dialog()
                    # else:
                    #     self.process_3d_cube()
    
                # Measurement errors: look for a variance / error / inverse-
                # variance cube for this file. Silent — it prints what it found
                # and falls back to the empirical estimator, never blocks ingest.
                self.detect_measurement_errors()

                # Extract observation info (skipped on session restore — the
                # session supplies df_obs and we don't want the NED popup).
                if for_session:
                    self.WLscalefactor_button.setEnabled(True)
                    self.fluxscalefactor_button.setEnabled(True)
                    if hasattr(self, 'bkg_image_btn'):
                        self.bkg_image_btn.setEnabled(True)
                else:
                    self.extract_observation_info()

    def detect_measurement_errors(self):
        """Find this cube's 1σ measurement errors, silently.

        Looks for a variance / error / inverse-variance extension in the science
        file, then for a sidecar file next to it (KCWI `*_vcubes.fits`, `*_err`,
        …). Anything not found or not usable simply leaves the empirical
        (line-free continuum) estimator in charge — the fit always has a noise
        model. The user can override the choice from "Measurement Errors…".
        """
        set_error_cube(None, {'mode': 'empirical',
                              'source': HyperCube_Noise.SRC_EMPIRICAL})
        if FITS_DATA is None or getattr(FITS_DATA, 'ndim', 0) not in (1, 3):
            return
        path = getattr(self, 'fits_path', None)
        try:
            spec = HyperCube_Noise.detect(path, FITS_DATA.shape,
                                          getattr(self, 'fits_ext', 0))
            if spec is None:
                print('Measurement errors: no variance/error cube found — using '
                      'the empirical line-free-continuum estimate (DER_SNR).')
                return
            self.apply_error_cube_spec(spec, announce=True)
        except Exception as e:
            print(f'Measurement errors: {e}\n  → falling back to the empirical estimate.')

    def apply_error_cube_spec(self, spec, announce=False):
        """Load the σ cube described by `spec` and install it. Raises on failure
        (the caller decides whether to warn or fall back silently)."""
        sigma, info = HyperCube_Noise.load_sigma(spec, FITS_DATA.shape)
        # Keep the errors on the same flux scale as the cube itself.
        try:
            fscale = np.float64(self.fluxscalefactor)
        except (TypeError, ValueError):
            fscale = 1.0
        if np.isfinite(fscale) and fscale != 1.0:
            sigma = sigma * fscale
        set_error_cube(sigma, info)
        if announce:
            print(f"Measurement errors: {info['source']} "
                  f"({100 * info.get('good_fraction', 0):.1f}% usable pixels)")
        return info

    # ── Session save / restore ────────────────────────────────────────────────

    def save_session(self):
        """Pickle the entire tool state to a .hcsession file so the user can
        close HyperCube and later resume exactly where they left off."""
        global df_obs, df, df_cont, df_fit, fit_results, snr_map, snr_value, wavelengths
        global base_df_cont, base_df, spaxel_overrides, df_stellar, STELLAR_CACHE
        if not getattr(self, 'fits_path', None):
            QMessageBox.warning(self, 'Save Session',
                                'Open a FITS cube before saving a session.')
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Session', '',
            'HyperCube Session (*.hcsession);;All Files (*)', options=options)
        if not file_path:
            return
        if not file_path.endswith('.hcsession'):
            file_path += '.hcsession'

        # Cube display state
        cube = {
            'zoom_level': getattr(self, '_cube_zoom_level', 1.0),
            'rotation':   getattr(self, '_cube_rotation', 0),
            'flip_h':     getattr(self, '_cube_flip_h', False),
            'flip_v':     getattr(self, '_cube_flip_v', False),
        }
        for attr, key in (('cube_cmap_combo', 'cmap'),
                          ('scale_combo', 'scale'),
                          ('stretch_combo', 'stretch')):
            combo = getattr(self, attr, None)
            if combo is not None:
                cube[key] = combo.currentText()
        if getattr(self, 'ax', None) is not None:
            cube['xlim'] = list(self.ax.get_xlim())
            cube['ylim'] = list(self.ax.get_ylim())

        # Background overlay state (stores the image array for exact restore)
        bkg = None
        if getattr(self, '_bkg_data', None) is not None:
            bkg = {
                'data':       self._bkg_data,
                'extent':     getattr(self, '_bkg_extent', None),
                'label':      getattr(self, '_bkg_label', ''),
                'cmap':       getattr(self, '_bkg_cmap', 'gray'),
                'scale':      getattr(self, '_bkg_scale', 'Linear'),
                'stretch':    getattr(self, '_bkg_stretch', '99%'),
                'brightness': getattr(self, '_bkg_brightness', 0.0),
                'contrast':   getattr(self, '_bkg_contrast', 1.0),
                'dx':         getattr(self, '_bkg_dx', 0),
                'dy':         getattr(self, '_bkg_dy', 0),
                'opacity':    (self.cube_opacity_slider.value()
                               if hasattr(self, 'cube_opacity_slider') else 100),
            }

        # Matplotlib artist references (lineactor / curveactor columns) cannot
        # be pickled; drop them. The rebuild path recreates the actors on load,
        # exactly as the CSV Load Fit path does.
        def _strip_actors(d):
            if not isinstance(d, pd.DataFrame):
                return d
            d = d.copy()
            for col in d.columns:
                if 'actor' in str(col).lower():
                    d[col] = None
            return d

        session = {
            'version':           1,
            'fits_path':         self.fits_path,
            'fits_ext':          getattr(self, 'fits_ext', 0),
            'df_obs':            _strip_actors(df_obs),
            'df':                _strip_actors(df),
            'df_cont':           _strip_actors(df_cont),
            'df_fit':            _strip_actors(df_fit),
            'fit_results':       fit_results,
            'snr_map':           snr_map,
            'snr_value':         snr_value,
            'wavelengths':       wavelengths,
            'current_spaxel':    getattr(self, 'current_spaxel', None),
            'init_guess_spaxel': getattr(self, '_init_guess_spaxel', None),
            'cube':              cube,
            'bkg':               bkg,
            # Per-spaxel model overrides + locked schema template
            'base_df_cont':      _strip_actors(base_df_cont),
            'base_df':           _strip_actors(base_df),
            'spaxel_overrides':  {k: {'df_cont': _strip_actors(v['df_cont']),
                                      'df': _strip_actors(v['df'])}
                                  for k, v in spaxel_overrides.items()},
            # Stellar (pPXF) results + optimal-template cache
            'df_stellar':        _strip_actors(df_stellar),
            'stellar_cache':     STELLAR_CACHE,
            # Measurement errors: store the *spec*, not the array — the cube is
            # reloaded from disk on restore. None ⇒ the empirical estimator.
            'error_spec':        ({k: ERROR_INFO.get(k)
                                   for k in ('mode', 'path', 'ext', 'kind', 'label')}
                                  if ERROR_CUBE is not None else None),
        }
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            QMessageBox.critical(self, 'Save Session', f'Failed to save session:\n{e}')
            return
        print(f'Session saved: {file_path}')
        QMessageBox.information(self, 'Save Session',
                               f'Session saved to:\n{os.path.basename(file_path)}')

    def load_session(self):
        """Restore a previously saved .hcsession: reopen the cube, restore all
        dataframes and display/background settings, and rebuild the fit panel."""
        global df_obs, df, df_cont, df_fit, fit_results, snr_map, snr_value, wavelengths, viewing_fit
        global base_df_cont, base_df, spaxel_overrides, df_stellar, STELLAR_CACHE
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Session', '',
            'HyperCube Session (*.hcsession);;All Files (*)', options=options)
        if not file_path:
            return
        try:
            with open(file_path, 'rb') as f:
                session = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, 'Load Session', f'Failed to read session:\n{e}')
            return

        # 1) Reopen the cube (by path; the array itself is not stored).
        cube_path = session.get('fits_path')
        if not cube_path or not os.path.exists(cube_path):
            QMessageBox.critical(
                self, 'Load Session',
                f'The cube file referenced by this session was not found:\n{cube_path}')
            return
        try:
            self._load_fits_from_path(cube_path, for_session=True)
        except Exception as e:
            QMessageBox.critical(self, 'Load Session', f'Failed to reopen cube:\n{e}')
            return
        self.fits_ext = session.get('fits_ext', getattr(self, 'fits_ext', 0))

        # Restore the measurement-error choice (ingest already ran detection;
        # an explicit saved choice overrides whatever it found).
        if 'error_spec' in session:
            _espec = session.get('error_spec')
            if _espec is None:
                set_error_cube(None, {'mode': 'empirical',
                                      'source': HyperCube_Noise.SRC_EMPIRICAL})
            else:
                try:
                    self.apply_error_cube_spec(_espec, announce=True)
                except Exception as e:
                    print(f'Session: measurement errors could not be restored ({e}); '
                          'using the empirical estimate.')
                    set_error_cube(None, {'mode': 'empirical',
                                          'source': HyperCube_Noise.SRC_EMPIRICAL})

        # 2) Restore the analysis dataframes / globals.
        df_obs      = session.get('df_obs', df_obs)
        df          = session.get('df', df)
        df_cont     = session.get('df_cont', df_cont)
        df_fit      = session.get('df_fit', df_fit)
        fit_results = session.get('fit_results', fit_results)
        if session.get('snr_map') is not None:
            snr_map = session['snr_map']
        snr_value   = session.get('snr_value', snr_value)
        if session.get('wavelengths') is not None:
            wavelengths = session['wavelengths']
        if session.get('current_spaxel') is not None:
            self.current_spaxel = session['current_spaxel']
        if session.get('init_guess_spaxel') is not None:
            self._init_guess_spaxel = session['init_guess_spaxel']
        # Restore the per-spaxel model overrides + base template (if present).
        base_df_cont = session.get('base_df_cont', None)
        base_df      = session.get('base_df', None)
        spaxel_overrides = session.get('spaxel_overrides', {}) or {}
        df_stellar = session.get('df_stellar', pd.DataFrame({}))
        if df_stellar is None:
            df_stellar = pd.DataFrame({})
        STELLAR_CACHE = session.get('stellar_cache', {}) or {}
        # Back-compat: older sessions keyed the stellar cache by (x, y). The cache
        # is now keyed by (x, y, region_ID). If exactly one stellar region exists,
        # migrate the 2-tuple keys onto it; otherwise drop the ambiguous entries.
        if any(isinstance(k, tuple) and len(k) == 2 for k in STELLAR_CACHE):
            _srids = ([] if not (isinstance(df_cont, pd.DataFrame) and 'cont_type' in df_cont.columns)
                      else [int(np.int64(r)) for r in df_cont.loc[df_cont['cont_type'] == 'stellar', 'region_ID']])
            _lone = _srids[0] if len(_srids) == 1 else None
            STELLAR_CACHE = {
                ((k[0], k[1], _lone) if (isinstance(k, tuple) and len(k) == 2) else k): v
                for k, v in STELLAR_CACHE.items()
                if not (isinstance(k, tuple) and len(k) == 2 and _lone is None)
            }
        self._edit_spaxel = None  # never resume mid-edit

        # 3) Ensure the Fit Parameters dock exists and is visible.
        if self.fit_params_dock is None:
            self.toggle_fit_params_dock()       # creates and leaves it open
        elif not self.fit_params_dock.isVisible():
            self.fit_params_dock.show()
            self.open_fit_params_button.setText("▲  Fit Parameters")

        # 3b) Display a spectrum so spectrum_ax exists before the fit overlays
        #     are drawn (rebuild_plot plots onto spectrum_ax). On a fresh
        #     session load no spaxel has been clicked yet, so create it here.
        disp_xy = None
        if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0:
            try:
                disp_xy = (int(df_fit.iloc[0]['spaxel_x']), int(df_fit.iloc[0]['spaxel_y']))
            except Exception:
                disp_xy = None
        if disp_xy is None and session.get('current_spaxel') is not None:
            disp_xy = tuple(session['current_spaxel'])
        if disp_xy is None and FITS_DATA is not None and getattr(FITS_DATA, 'ndim', 0) == 3:
            disp_xy = (FITS_DATA.shape[2] // 2, FITS_DATA.shape[1] // 2)
        if disp_xy is not None and self.fits_data is not None and not self.is_1d_spectrum:
            try:
                self.current_spaxel = disp_xy
                self.update_spectrum_panel(self.get_spectrum_at_spaxel(disp_xy[0], disp_xy[1]))
                if hasattr(self, 'red_rect') and self.red_rect is not None:
                    self.red_rect.set_xy((disp_xy[0] - 0.5, disp_xy[1] - 0.5))
                    self.red_rect.set_visible(True)
            except Exception as e:
                print(f'Session: spectrum display issue: {e}')

        # 3c) Rebuild the fit-params frames + overlays from the restored data.
        if self.fit_params_window is not None:
            self.fit_params_window.rebuild_fit_panel(show_fit=True)

        # 4) Apply cube display settings.
        cube = session.get('cube', {}) or {}
        self._cube_rotation = cube.get('rotation', 0)
        self._cube_flip_h   = cube.get('flip_h', False)
        self._cube_flip_v   = cube.get('flip_v', False)
        for attr, key in (('scale_combo', 'scale'),
                          ('stretch_combo', 'stretch'),
                          ('cube_cmap_combo', 'cmap')):
            combo = getattr(self, attr, None)
            if combo is not None and key in cube:
                combo.blockSignals(True)
                combo.setCurrentText(cube[key])
                combo.blockSignals(False)
        # Carry the colormap through the redraw (draw_image reads _last_cmap).
        if 'cmap' in cube:
            self._last_cmap = cube['cmap']
        # Redraw with the restored transform + scale/stretch + colormap.
        try:
            if self._cube_rotation or self._cube_flip_h or self._cube_flip_v:
                self._cube_redraw_transformed()
            else:
                self._cube_redraw_with_current_settings()
        except Exception as e:
            print(f'Session: cube redraw issue: {e}')

        # 5) Restore the background overlay.
        bkg = session.get('bkg')
        if bkg is not None and bkg.get('data') is not None:
            self._bkg_data       = bkg['data']
            self._bkg_extent     = bkg.get('extent')
            self._bkg_label      = bkg.get('label', '')
            self._bkg_cmap       = bkg.get('cmap', 'gray')
            self._bkg_scale      = bkg.get('scale', 'Linear')
            self._bkg_stretch    = bkg.get('stretch', '99%')
            self._bkg_brightness = bkg.get('brightness', 0.0)
            self._bkg_contrast   = bkg.get('contrast', 1.0)
            self._bkg_dx         = bkg.get('dx', 0)
            self._bkg_dy         = bkg.get('dy', 0)
            if hasattr(self, 'cube_opacity_slider'):
                self.cube_opacity_slider.setValue(int(bkg.get('opacity', 100)))
                self.cube_opacity_slider.setVisible(True)
            self._draw_bkg_overlay()

        # 6) Restore the exact viewport last (after all redraws).
        if getattr(self, 'ax', None) is not None and 'xlim' in cube and 'ylim' in cube:
            self.ax.set_xlim(cube['xlim'])
            self.ax.set_ylim(cube['ylim'])
            self._cube_zoom_level = cube.get('zoom_level', getattr(self, '_cube_zoom_level', 1.0))
            try:
                self._sync_zoom_combo()
                self._cube_update_scrollbars()
            except Exception:
                pass
            self.canvas.draw_idle()

        print(f'Session loaded: {file_path}')
        QMessageBox.information(self, 'Load Session',
                               f'Session restored from:\n{os.path.basename(file_path)}')

    def show_column_selection_dialog(self):
        """Create a dialog for selecting wavelength and flux columns from bintable"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Table Columns")
        dialog.setMinimumWidth(500)  # Slightly wider for better button spacing
        
        layout = QVBoxLayout()
        info_label = QLabel("This FITS file contains multiple columns. Please select:")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Wavelength selection
        wave_group = QGroupBox("Select Wavelength Vector")
        wave_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        wave_layout = QVBoxLayout()
        wave_layout.setSpacing(10)
        wave_layout.setContentsMargins(10, 25, 10, 15)  # More vertical padding
        
        self.wave_buttons = []
        for i, name in enumerate(self.column_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 5px;
                    border: 1px solid #888;
                    border-radius: 4px;
                    min-width: 80px;
                }
                QPushButton:checked {
                    background-color: lime;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            btn.clicked.connect(lambda _, x=i: self.select_column(x, 'wave'))
            self.wave_buttons.append(btn)
            wave_layout.addWidget(btn)
        wave_group.setLayout(wave_layout)
        layout.addWidget(wave_group)
        
        # Flux selection
        flux_group = QGroupBox("Select Flux Vector")
        flux_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        flux_layout = QVBoxLayout()
        flux_layout.setSpacing(10)
        flux_layout.setContentsMargins(10, 25, 10, 15)  # More vertical padding
        
        self.flux_buttons = []
        for i, name in enumerate(self.column_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 5px;
                    border: 1px solid #888;
                    border-radius: 4px;
                    min-width: 80px;
                }
                QPushButton:checked {
                    background-color: lime;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            btn.clicked.connect(lambda _, x=i: self.select_column(x, 'flux'))
            self.flux_buttons.append(btn)
            flux_layout.addWidget(btn)
        flux_group.setLayout(flux_layout)
        layout.addWidget(flux_group)
        
        # Confirm button
        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        confirm_btn.clicked.connect(lambda: self.finalize_bintable_selection(dialog))
        layout.addWidget(confirm_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def select_column(self, index, col_type):
        """Handle column selection (single selection per type)"""
        buttons = self.wave_buttons if col_type == 'wave' else self.flux_buttons
        for i, btn in enumerate(buttons):
            # Store current style to preserve other properties
            current_style = btn.styleSheet()
            if i == index:
                btn.setChecked(True)
                # Add the checked style while preserving other styles
                btn.setStyleSheet(current_style + """
                    QPushButton {
                        background-color: lime;
                        font-weight: bold;
                    }
                """)
            else:
                btn.setChecked(False)
                # Reset to base style
                btn.setStyleSheet("""
                    QPushButton {
                        padding: 5px;
                        border: 1px solid #888;
                        border-radius: 4px;
                        min-width: 80px;
                    }
                """)
    
    def finalize_bintable_selection(self, dialog):
        """Process selected columns and continue with file loading"""
        global wavelengths, FITS_DATA  # Declare globals
        
        try:
            # Get selected columns
            wave_idx = next((i for i, btn in enumerate(self.wave_buttons) if btn.isChecked()), -1)
            flux_idx = next((i for i, btn in enumerate(self.flux_buttons) if btn.isChecked()), -1)
            
            if wave_idx == -1 or flux_idx == -1:
                QMessageBox.warning(self, "Selection Error", 
                                  "Please select both wavelength and flux columns")
                return
            
            # Extract selected data - proper FITS binary table access
            
            # print(f'wavelength index selected: {wave_idx}')
            # print(f'flux index selected: {flux_idx}')
            wavelengths = self.fits_data[0][wave_idx]  # Direct column access
            flux = self.fits_data[0][flux_idx]
            
            # Update global data references
            FITS_DATA = flux
            self.process_1d_spectrum()
            self.extract_observation_info()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to load selected columns:\n{str(e)}")
        finally:
            dialog.close()
    
    def _construct_wavelengths_from_header(self, axis=1):
        """Construct wavelength array from FITS header for given axis"""
        header = self.fits_header
        naxis = header[f'NAXIS{axis}']
        
        # Try standard WCS keywords
        crval = header.get(f'CRVAL{axis}')
        cdelt = header.get(f'CDELT{axis}')
        crpix = header.get(f'CRPIX{axis}', 1)
        
        if None not in (crval, cdelt):
            return crval + cdelt * (np.arange(naxis) + 1 - crpix)
        
        # Try CD matrix
        cd = header.get(f'CD{axis}_{axis}')
        if cd is not None:
            return crval + cd * (np.arange(naxis) + 1 - crpix)
        
        # Try PC matrix
        pc = header.get(f'PC{axis}_{axis}')
        if pc is not None and cdelt is not None:
            return crval + pc * cdelt * (np.arange(naxis) + 1 - crpix)
        
        # Fallback for frequency axes
        if header.get(f'CTYPE{axis}','').startswith('FREQ'):
            restfreq = header.get('RESTFRQ', header.get('RESTFREQ'))
            if restfreq is not None:
                return (1 - np.arange(naxis) * header.get('CDELT3', 1) / restfreq * 299792.458)
        
        print(f"Warning: No valid WCS found for axis {axis} - using pixel indices")
        return np.arange(naxis)
    
    
    
    def process_1d_spectrum(self):
        """Common processing for all 1D spectra"""
        global wavelengths, snr_map
        
        # Create dummy 2D array for visualization
        dummy_2d = np.tile(FITS_DATA, (10, 1)).T
        
        # Initialize dummy WCS and SNR map
        self.wcs = WCS(naxis=2)
        snr_map = np.zeros((10, 10))
        
        # Show the spectrum
        self.update_spectrum_panel(FITS_DATA)
    
    def process_3d_cube(self):
        """Processing for 3D cubes"""
        global wavelengths, snr_map

        self._spaxel_mask = None  # drop any mask from a previously-loaded cube
        self.draw_image(FITS_DATA, cmap='gray', scale='linear', from_fits=True)
        self.wcs = WCS(self.fits_header)
        
        # Initialize SNR map
        _, nx, ny = FITS_DATA.shape
        snr_map = np.zeros((nx, ny)) + 1E8
        
        # Extract spectral axis info
        spectral_sampling = FITS_HEADER.get('CDELT3', FITS_HEADER.get('CD3_3', 1))
        wavelengths = (FITS_HEADER['CRVAL3'] + 
                     spectral_sampling * 
                     (np.arange(FITS_DATA.shape[0]) - FITS_HEADER.get('CRPIX3', 1) + 1))
    
    def extract_observation_info(self):
        """Extract common observation info"""
        global df_obs
        
        source_name = FITS_HEADER.get('OBJECT', '')

        def _first_hdr(keys):
            """Return the first present, numeric-parseable header value among keys."""
            for k in keys:
                if k in FITS_HEADER:
                    v = _safe_float(FITS_HEADER.get(k))
                    if np.isfinite(v):
                        return v
            return ''

        source_redshift = _first_hdr(['REDSHIFT', 'Z', 'ZSYS', 'REDSHIFT0'])
        # Resolving power may be stored directly (R/RESOLVINGP/SPECRES) or derived
        # from a resolution element (FWHM in Å via RESOLWAV/CDELT) — try direct keys.
        resolving_power = _first_hdr(['RESOLVINGP', 'R', 'SPECRES', 'RESOLUTIO',
                                      'RESOLUTION', 'RESOLVING_POWER'])

        df_obs.loc[0] = [source_name, source_redshift, resolving_power]
        print("FITS file loaded successfully!")
        # Enable data-dependent toolbar buttons now that a file is loaded
        self.WLscalefactor_button.setEnabled(True)
        self.fluxscalefactor_button.setEnabled(True)
        for b in getattr(self, '_xform_buttons', []):
            b.setEnabled(True)
        self._sb_file.setText(os.path.basename(getattr(self, 'fits_path', '') or ''))
        # Prompt user to confirm / resolve the source name from NED
        # Use QTimer to let the main window finish painting before showing the dialog
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(300, self._prompt_source_on_load)



    def _prompt_source_on_load(self):
        """Show the source information dialog after a FITS file is loaded.
        If FitParamsWindow is open, delegate to it; otherwise call standalone version.
        """
        if self.fit_params_window is not None:
            self.fit_params_window._show_source_dialog()
        else:
            # Build a minimal standalone dialog matching FitParamsWindow._show_source_dialog
            global df_obs, FITS_HEADER
            current_name = str(df_obs.loc[0, 'sourcename']) if len(df_obs) > 0 else ''
            dialog = QDialog(self)
            dialog.setWindowTitle('Source Information')
            dialog.setMinimumWidth(420)
            layout = QVBoxLayout(dialog)
            layout.setSpacing(10)
            layout.addWidget(QLabel('Source name:'))
            name_edit = QLineEdit(current_name)
            name_edit.setPlaceholderText('e.g. NGC 1068')
            layout.addWidget(name_edit)
            resolve_name_btn  = QPushButton('Resolve name  (NED)')
            resolve_coord_btn = QPushButton('Resolve coordinates  (NED)')
            layout.addWidget(resolve_name_btn)
            layout.addWidget(resolve_coord_btn)
            status = QLabel(''); status.setWordWrap(True)
            layout.addWidget(status)
            btn_row = QHBoxLayout()
            ok_btn = QPushButton('OK'); ok_btn.setDefault(True)
            cancel_btn = QPushButton('Cancel')
            btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
            layout.addLayout(btn_row)

            def _apply(ned_name, ned_z):
                name_edit.setText(ned_name)
                df_obs.loc[0, 'sourcename'] = ned_name
                df_obs.loc[0, 'redshift']   = ned_z
                # Enable the background image button now that source is resolved
                if hasattr(self, 'bkg_image_btn'):
                    self.bkg_image_btn.setEnabled(True)

            def _by_name():
                name = name_edit.text().strip()
                if not name: status.setText('Enter a name first.'); return
                status.setText(f'Querying NED for "{name}"...'); dialog.repaint()
                try:
                    from astroquery.ipac.ned import Ned
                    r = Ned.query_object(name)
                    ned_name = str(r['Object Name'][0])
                    ned_z    = float(r['Redshift'][0])
                    _apply(ned_name, ned_z)
                    status.setText(f'✓  {ned_name}   z = {ned_z:.6f}')
                except Exception as e:
                    status.setText(f'NED query failed: {e}')

            def _by_coords():
                try:
                    ra  = float(FITS_HEADER.get('CRVAL1', ''))
                    dec = float(FITS_HEADER.get('CRVAL2', ''))
                except: status.setText('Could not read RA/Dec from FITS header.'); return
                status.setText(f'Querying NED at RA={ra:.4f}, Dec={dec:.4f}...'); dialog.repaint()
                try:
                    from astroquery.ipac.ned import Ned
                    import astropy.units as u
                    from astropy.coordinates import SkyCoord
                    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                    r = Ned.query_region(coord, radius=0.5 * u.arcmin)
                    if len(r) == 0: raise ValueError('No objects found')
                    ned_name = str(r['Object Name'][0])
                    ned_z    = float(r['Redshift'][0])
                    _apply(ned_name, ned_z)
                    status.setText(f'✓  {ned_name}   z = {ned_z:.6f}')
                except Exception as e:
                    status.setText(f'NED query failed: {e}')

            def _commit():
                df_obs.loc[0, 'sourcename'] = name_edit.text().strip()
                dialog.accept()

            resolve_name_btn.clicked.connect(_by_name)
            resolve_coord_btn.clicked.connect(_by_coords)
            ok_btn.clicked.connect(_commit)
            name_edit.returnPressed.connect(_commit)
            cancel_btn.clicked.connect(dialog.reject)
            dialog.exec_()

    def pixel_to_ra_dec(self, x, y):
        """Convert pixel coordinates (x, y) to RA, Dec using the WCS information from FITS header.
        Handles both 2D and 4D WCS (like ALMA data with Stokes parameter)."""
        
        # Ensure inputs are numpy arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        # Check WCS dimensionality
        if self.wcs.naxis == 4:
            # For 4D WCS (RA, Dec, Freq, Stokes) - ALMA case
            z = np.zeros_like(x)  # Frequency axis (use 0 or reference pixel)
            stokes = np.zeros_like(x)  # Stokes axis (I=0)
            ra, dec, _, _ = self.wcs.all_pix2world(x, y, z, stokes, 0)
        elif self.wcs.naxis == 3:
            # For 3D WCS (RA, Dec, Freq)
            z = np.zeros_like(x)  # Frequency axis
            ra, dec, _ = self.wcs.all_pix2world(x, y, z, 0)
        else:
            # Standard 2D WCS
            ra, dec = self.wcs.all_pix2world(x, y, 0)
        
        return ra, dec


    # Helper function to convert decimal to sexagesimal
    def decimal_to_sexagesimal(self, ra_dec, is_ra=True):
        """
        Convert decimal RA or Dec to sexagesimal format (RA: h:m:s, Dec: d:m:s).
        :param ra_dec: decimal RA or Dec.
        :param is_ra: True if it's RA (hours), False if it's Dec (degrees).
        :return: sexagesimal string.
        """
        if is_ra:
            # Convert RA from degrees to hours, minutes, seconds
            total_seconds = ra_dec * 3600.0 / 15.0  # RA in hours, 15 degrees per hour
        else:
            # Convert Dec from degrees to degrees, arcminutes, arcseconds
            total_seconds = ra_dec * 3600.0
    
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
    
        if is_ra:
            return f"{hours:02}:{minutes:02}:{seconds:05.2f}"  # RA: h:m:s
        else:
            sign = '+' if ra_dec >= 0 else '-'
            degrees = abs(int(ra_dec))
            minutes = int((abs(ra_dec) * 60) % 60)
            seconds = (abs(ra_dec) * 3600) % 60
            return f"{sign}{degrees:02}:{minutes:02}:{seconds:05.2f}"  # Dec: ±d:m:s


    
    def update_spaxel_overlay(self, x, y):
        """Update the RA/Dec/X/Y/N text overlay on the cube image and the
        status bar. Always called on mouse move."""
        if self.is_1d_spectrum:
            return
        ra, dec = self.pixel_to_ra_dec(x, y)
        ra_sexagesimal  = self.decimal_to_sexagesimal(ra[0],  is_ra=True)
        dec_sexagesimal = self.decimal_to_sexagesimal(dec[0], is_ra=False)
        n = self.get_spaxel_number(x, y)
        if hasattr(self, 'spaxel_info_text'):
            qual_label = getattr(self, '_quality_map_label', None)
            if qual_label is not None:
                try:
                    img = self._last_data
                    val = float(img[y, x]) if (img is not None and
                          hasattr(img, 'ndim') and img.ndim == 2 and
                          0 <= y < img.shape[0] and 0 <= x < img.shape[1]) else None
                    val_str = f"\n{qual_label}: {val:.3g}" if (val is not None and np.isfinite(val)) else ""
                except Exception:
                    val_str = ""
            else:
                val_str = ""
            self.spaxel_info_text.set_text(
                f"RA  {ra_sexagesimal}\n"
                f"Dec {dec_sexagesimal}\n"
                f"X {x}   Y {y}   N {n}"
                f"{val_str}"
            )
            self.canvas.draw_idle()
        # Status bar
        if hasattr(self, '_sb_spaxel'):
            self._sb_spaxel.setText(f"Spaxel ({x}, {y})")
        if hasattr(self, '_sb_radec'):
            self._sb_radec.setText(f"RA {ra_sexagesimal}  Dec {dec_sexagesimal}")
        if hasattr(self, '_sb_snr'):
            try:
                snr_val = float(snr_map[y, x]) if (snr_map is not None and
                          snr_map.shape[0] > y and snr_map.shape[1] > x) else None
                self._sb_snr.setText(f"SNR {snr_val:.1f}" if snr_val is not None else "")
            except Exception:
                pass

    def update_buttons(self, x, y):
        """Refresh fit-result values in the Fit Parameters side panel for spaxel (x, y).
        Only called when fit_params_window exists."""
        # Only clear non-spectrum curves when NOT on the init-guess spaxel,
        # so interactive initial guesses persist while the user stays there.
        on_init_spaxel = (self._init_guess_spaxel is not None and
                          int(x) == self._init_guess_spaxel[0] and
                          int(y) == self._init_guess_spaxel[1])
        if not on_init_spaxel:
            for line in self.spectrum_ax.get_lines():
                if line.get_label() != "_child0":
                    line.remove()
            self.gaussian_component_lines.clear()
            self.spectrum_canvas.draw()

        print(f"update_buttons: spaxel ({x},{y}), df_fit rows={len(df_fit)}, on_init={on_init_spaxel}")
        for region_ID in df_cont['region_ID'].astype(int).unique():
            show_init = on_init_spaxel  # redraw init curves when back on home spaxel
            self.rebuild_plot(region_ID, from_file=False,
                              show_init=show_init, show_fit=(not show_init), x=x, y=y)
        # Keep the stellar V/σ map button faces in sync with the hovered spaxel.
        if self.fit_params_window is not None:
            try:
                self.fit_params_window._refresh_stellar_map_buttons()
            except Exception:
                pass

    def get_spaxel_number(self, x, y):
        """Get the spaxel number for the given pixel coordinates."""
        # Example: convert (x, y) to a spaxel index (could be linear index or 2D)
        return y #* self.image_width + x  # Example of converting to a 1D spaxel number




    def update_spectrum_panel(self, spectrum):
        """Efficiently update the right panel to show the 1D spectrum."""
        if spectrum is not None:
            if not hasattr(self, 'spectrum_fig'):
                # Initialize the spectrum figure and canvas
                self.spectrum_fig = plt.Figure(figsize=(5, 4))
                self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
                self.spectrum_ax = self.spectrum_fig.add_subplot(111)
                
                # Create the spectrum line (step plot)
                self.spectrum_line, = self.spectrum_ax.step([], [], lw=0.5, color='w')
                
                # Apply the updated QSS styling
                apply_mpl_qss_style(self.spectrum_fig, self.spectrum_ax, self.spectrum_line)
                
                # Set axis labels and title
                self.spectrum_ax.set_title("1D Spectrum at Cursor Position")
                self.spectrum_ax.set_xlabel("Wavelength")
                self.spectrum_ax.set_ylabel("Flux")
                
                # Mouse event connections
                self.spectrum_canvas.mpl_connect('button_press_event', self.on_mouse_press)
                self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
                self.spectrum_canvas.mpl_connect('button_release_event', self.on_mouse_release)
                self.spectrum_canvas.mpl_connect("button_press_event", self.show_context_menu)
                self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_spectrum_mouse_move)
                
                # Ensure canvas reflects the QSS style (PyQt5-level styling)
                self.spectrum_canvas.setStyleSheet("background-color: #2E3440; border: none;")
                
                # Wavelength/flux overlay (top-right corner, mirrors the cube RA/Dec overlay)
                self.spectrum_info_text = self.spectrum_ax.text(
                    0.01, 0.99, '',
                    transform=self.spectrum_ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none')
                )

                # Reveal the spectrum toolbar now that we have a spectrum
                self.spec_bar_widget.show()
                # Add the Matplotlib canvas to the layout
                layout = self.right_panel.layout()
                layout.addWidget(self.spectrum_canvas)

                # Horizontal pan scrollbar below the spectrum
                self.spec_hbar = QScrollBar(Qt.Horizontal)
                self.spec_hbar.setRange(0, 1000)
                self.spec_hbar.setValue(500)
                self.spec_hbar.setSingleStep(10)
                self.spec_hbar.setPageStep(100)
                self.spec_hbar.setFixedHeight(14)
                self.spec_hbar.setStyleSheet(
                    "QScrollBar:horizontal { height: 14px; }"
                    "QScrollBar::handle:horizontal { min-width: 30px; }"
                )
                self.spec_hbar.valueChanged.connect(self._spectrum_hbar_pan)
                layout.addWidget(self.spec_hbar)

            # Update the data instead of recreating the plot
            self.spectrum_line.set_xdata(wavelengths)
            self.spectrum_line.set_ydata(spectrum)

            if not self.zoom_active:
                # Default full range
                self.spectrum_ax.set_xlim(wavelengths.min(), wavelengths.max())
                self._spec_wav_min = float(wavelengths.min())
                self._spec_wav_max = float(wavelengths.max())
                self._spectrum_update_hbar()
                y_min, y_max = np.nanmin(spectrum), np.nanmax(spectrum)
            else:
                # Get the zoomed x-axis range
                xlim_min, xlim_max = self.spectrum_ax.get_xlim()
    
                # Select only the part of the spectrum within the zoomed range
                mask = (wavelengths >= xlim_min) & (wavelengths <= xlim_max)
                if np.any(mask):  # Ensure there are valid values in range
                    y_min, y_max = np.nanmin(spectrum[mask]), np.nanmax(spectrum[mask])
                else:  # If no valid data, use default
                    y_min, y_max = np.nanmin(spectrum), np.nanmax(spectrum)
    
            # Apply 10% padding to the y-limits
            padding = self.zoom_pad * (y_max - y_min)
            self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)

            # Refresh the figure
            self.spectrum_canvas.draw()

    @staticmethod
    def _span_x_extent(span):
        """Return (x0, x1) of an axvspan.

        matplotlib >= 3.10 makes ``axvspan`` return a ``Rectangle`` (whose
        ``get_xy`` is a corner 2-tuple), while older versions returned a
        ``Polygon`` (whose ``get_xy`` is an Nx2 vertex array). Handle both.
        """
        if isinstance(span, patches.Rectangle):
            x = span.get_x()
            return x, x + span.get_width()
        xy = span.get_xy()
        return xy[:, 0].min(), xy[:, 0].max()

    @staticmethod
    def _span_set_x_extent(span, x0, x1):
        """Set the x-extent of an axvspan, for both Rectangle and Polygon."""
        if isinstance(span, patches.Rectangle):
            span.set_x(min(x0, x1))
            span.set_width(abs(x1 - x0))
            return
        xy = span.get_xy()
        n = len(xy)
        xy[:, 0] = [x0, x0, x1, x1, x0][:n]
        span.set_xy(xy)

    def _chanmap_draw_handles(self):
        """Draw (or refresh) ◀ / ▶ drag handles at the top edges of the channel map span."""
        for attr in ('_chanmap_handle_left', '_chanmap_handle_right'):
            h = getattr(self, attr, None)
            if h is not None:
                try: h.remove()
                except Exception: pass
            setattr(self, attr, None)

        if not self._chanmap_locked or self._chanmap_span is None:
            return
        if not hasattr(self, 'spectrum_ax') or self.spectrum_ax is None:
            return

        wav0, wav1 = self._span_x_extent(self._chanmap_span)
        xform = self.spectrum_ax.get_xaxis_transform()  # x: data coords, y: axes [0,1]

        kw = dict(transform=xform, va='top', fontsize=9,
                  color='#4fc3f7', fontweight='bold', clip_on=True,
                  zorder=10)
        self._chanmap_handle_left  = self.spectrum_ax.text(
            wav0, 0.97, '◀', ha='right', **kw)
        self._chanmap_handle_right = self.spectrum_ax.text(
            wav1, 0.97, '▶', ha='left',  **kw)

    def _chanmap_update_centre_display(self):
        """Update the centre-pixel button label from the current span limits."""
        global wavelengths
        if self._chanmap_span is None or wavelengths is None:
            self.chanmap_centre_btn.setText("Pixel: —")
            return
        wav0, wav1 = self._span_x_extent(self._chanmap_span)
        wav_centre = (wav0 + wav1) / 2
        pix = np.argmin(np.abs(wavelengths - wav_centre))
        self.chanmap_centre_btn.setText(f"Pixel: {pix}")

    def _chanmap_shift(self, delta_pix):
        """Shift the locked channel map selection by delta_pix pixels."""
        global wavelengths
        if not self._chanmap_locked or self._chanmap_span is None or wavelengths is None:
            return
        wav0, wav1 = self._span_x_extent(self._chanmap_span)
        half_w = (wav1 - wav0) / 2
        wav_centre = (wav0 + wav1) / 2

        pix_centre = np.argmin(np.abs(wavelengths - wav_centre))
        new_pix = int(np.clip(pix_centre + delta_pix, 0, len(wavelengths) - 1))
        new_centre = wavelengths[new_pix]

        new_wav0 = new_centre - half_w
        new_wav1 = new_centre + half_w
        self._span_set_x_extent(self._chanmap_span, new_wav0, new_wav1)
        self._chanmap_draw_handles()
        self.spectrum_canvas.draw_idle()
        self._chanmap_update_centre_display()
        self._compute_channel_map(new_wav0, new_wav1)

    def _chanmap_centre_clicked(self):
        """Open a dialog to enter a new centre pixel — avoids keyboard focus stealing."""
        global wavelengths
        if not self._chanmap_locked or self._chanmap_span is None or wavelengths is None:
            return
        wav0, wav1 = self._span_x_extent(self._chanmap_span)
        current_pix = int(np.argmin(np.abs(wavelengths - (wav0 + wav1) / 2)))

        dialog = QDialog(self)
        dialog.setWindowTitle("Set centre pixel")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Centre pixel (current: {current_pix})"))
        text_box = QLineEdit(str(current_pix))
        text_box.selectAll()
        layout.addWidget(text_box)
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _apply():
            try:
                pix = int(np.clip(int(text_box.text()), 0, len(wavelengths) - 1))
            except ValueError:
                return
            half_w = (wav1 - wav0) / 2
            new_centre = wavelengths[pix]
            new_wav0 = new_centre - half_w
            new_wav1 = new_centre + half_w
            self._span_set_x_extent(self._chanmap_span, new_wav0, new_wav1)
            self._chanmap_draw_handles()
            self.spectrum_canvas.draw_idle()
            self._chanmap_update_centre_display()
            self._compute_channel_map(new_wav0, new_wav1)
            dialog.accept()

        ok_btn.clicked.connect(_apply)
        text_box.returnPressed.connect(_apply)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec_()

    def keyReleaseEvent(self, event):
        """Release C/X/V to lock selections; C=line window, X/V=continuum windows."""
        # Handle subtraction window release (X or V)
        if event.key() in (Qt.Key_X, Qt.Key_V) and not event.isAutoRepeat():
            key_char = 'x' if event.key() == Qt.Key_X else 'v'
            sm = self._submap[key_char]
            if sm['active']:
                sm['active'] = False
                if self.spectrum_cursor_pos and sm['start'] is not None:
                    wav_end = self.spectrum_cursor_pos[0]
                    w0 = min(sm['start'], wav_end)
                    w1 = max(sm['start'], wav_end)
                    if abs(w1 - w0) < 1e-6:
                        # Tap = clear this window
                        if sm['span'] is not None:
                            try: sm['span'].remove()
                            except ValueError: pass
                        sm['span'] = None
                        sm['locked'] = False
                        self.spectrum_canvas.draw_idle()
                    else:
                        sm['locked'] = True
                self._compute_channel_map_with_subtraction()

        if event.key() == Qt.Key_C and self._chanmap_active and not event.isAutoRepeat():
            self._chanmap_active = False
            if self.spectrum_cursor_pos and self._chanmap_start is not None:
                wav_end = self.spectrum_cursor_pos[0]
                wav0 = min(self._chanmap_start, wav_end)
                wav1 = max(self._chanmap_start, wav_end)
                if abs(wav1 - wav0) < 1e-6:
                    # Zero-width tap: clear selection and restore white-light
                    if self._chanmap_span is not None:
                        try: self._chanmap_span.remove()
                        except ValueError: pass
                        self._chanmap_span = None
                    self._chanmap_locked = False
                    self._chanmap_draw_handles()  # removes handles
                    # Also clear all subtraction windows
                    for _sm in self._submap.values():
                        if _sm['span'] is not None:
                            try: _sm['span'].remove()
                            except ValueError: pass
                        _sm.update({'span': None, 'locked': False, 'active': False, 'start': None})
                    self.chanmap_centre_btn.setText("Pixel: —")
                    self.spectrum_canvas.draw_idle()
                    self.draw_image(FITS_DATA, self._last_cmap, self._last_scale, from_fits=True)
                else:
                    self._chanmap_locked = True
                    self._compute_channel_map(wav0, wav1)
                    self._chanmap_update_centre_display()
                    self._chanmap_draw_handles()
                    self.spectrum_canvas.draw_idle()

    def _compute_channel_map(self, wav0, wav1):
        """Store the current line window limits and trigger the full (subtracted) map."""
        self._chanmap_wav0 = wav0
        self._chanmap_wav1 = wav1
        self._compute_channel_map_with_subtraction()

    def _compute_channel_map_with_subtraction(self):
        """Compute the (optionally continuum-subtracted) channel map and display it.

        Line map  = sum of flux within the C window [wav0, wav1] per spaxel.
        Continuum = mean flux per channel, averaged across any locked X/V windows,
                    then scaled to the number of channels in the C window.
        Result    = Line map − Continuum  (if at least one subtraction window exists)
                  = Line map              (if no subtraction windows are locked)
        """
        global wavelengths
        if self.fits_data is None or wavelengths is None:
            return
        if not hasattr(self, '_chanmap_wav0'):
            return

        wav0, wav1 = self._chanmap_wav0, self._chanmap_wav1
        line_mask = (wavelengths >= wav0) & (wavelengths <= wav1)
        if not np.any(line_mask):
            return

        n_line = line_mask.sum()
        line_map = np.nansum(self.fits_data[line_mask, :, :], axis=0)

        # Gather locked subtraction windows
        cont_estimates = []
        for key_char, sm in self._submap.items():
            if sm['locked'] and sm['span'] is not None:
                sw0, sw1 = self._span_x_extent(sm['span'])
                cont_mask = (wavelengths >= sw0) & (wavelengths <= sw1)
                n_cont = cont_mask.sum()
                if n_cont > 0:
                    # Mean flux per channel in this window → scale to line width
                    mean_per_chan = np.nansum(self.fits_data[cont_mask, :, :], axis=0) / n_cont
                    cont_estimates.append(mean_per_chan)

        cmap  = getattr(self, '_last_cmap',  'gray')
        scale = getattr(self, '_last_scale', 'linear')

        if cont_estimates:
            # Average the sideband estimates then scale to line window width
            cont_map = np.mean(cont_estimates, axis=0) * n_line
            result = line_map - cont_map
            print(f"Continuum-subtracted map: {n_line} line ch, "
                  f"{len(cont_estimates)} sideband(s)")
        else:
            result = line_map
            print(f"Channel map: {n_line} channels [{wav0:.2f} – {wav1:.2f}]")

        self.draw_image(result, cmap, scale)

    def keyPressEvent(self, event):
        key = event.text().lower()

        global df_cont, df
        """Handle key press events for locking and drawing a line."""
        # In single-spaxel edit mode (schema locked), the drawing keys
        # d/s/p/g REPLACE an existing region's continuum or RE-SPECIFY an
        # existing line in place — they never append regions/lines. The commit
        # paths (Key_D block, _spline_finalize, _poly_finalize, save_gaussian)
        # branch on self._edit_spaxel to do this.

        if event.key() == Qt.Key_L:
            # Toggle the locking state
            self.locked = not self.locked
            print(f"Lock {'enabled' if self.locked else 'disabled'}")
            # Red when locked, orange when free
            if hasattr(self, 'red_rect'):
                self.red_rect.set_edgecolor('#cc2222' if self.locked else '#d7801a')
                self.canvas.draw_idle()
            # On lock, load this spaxel's per-spaxel model into the panel so the
            # continuum type/values and line guesses reflect THIS spaxel.
            if self.locked and self.current_spaxel is not None:
                self._load_spaxel_model(int(self.current_spaxel[0]),
                                        int(self.current_spaxel[1]))
            elif not self.locked and self._edit_spaxel is None:
                # On unlock, revert the active model + panel to the base template
                # so hovering doesn't keep showing the last-locked spaxel's
                # continuum type (e.g. a stale 'spline').
                self._reset_to_base_model()

        if event.key() == Qt.Key_F:
            print(df)
            # for line in self.spectrum_ax.get_lines():
            #     print(line)
                
            # self.active_line.remove()
            # self.spectrum_canvas.draw_idle()  # Efficient redrawing
            
            
        if event.key() == Qt.Key_D and not event.isAutoRepeat():
            # Start or finish drawing a line in the spectrum plot
            if not self.drawing_line:
                # Start drawing a line: store the first point (cursor position)
                self.line_start = self.spectrum_cursor_pos
                self.drawing_line = True
                self._set_spectrum_hint(
                    'Linear continuum   •   D: set end point    (drag to position)')

                # Create a new line object in the spectrum plot (only x-coordinates for the start)
                start_x, start_y = self.line_start
                self.current_line, = self.spectrum_ax.plot([start_x, start_x], [start_y, start_y], color='lime', lw=1)
                self.spectrum_canvas.draw()  # Refresh the canvas with the new line
            else:
                # Finish the line at the current cursor position
                self.finalize_line()
                self.drawing_line = False
                self._clear_spectrum_hint()
                # print(f"Line finalized from {self.line_start} to {self.spectrum_cursor_pos}")
                self.slope = (self.spectrum_cursor_pos[1]-self.line_start[1])/(self.spectrum_cursor_pos[0]-self.line_start[0])
                self.intercept = self.line_start[1] - self.slope * self.line_start[0]

                if self._edit_spaxel is not None:
                    # Edit mode: REPLACE the start-x region's continuum in place
                    rid = self._region_id_at(self.line_start[0])
                    if rid is None:
                        self._set_spectrum_hint('Draw start was not inside an existing region.')
                    else:
                        self._replace_region_continuum(rid, 'linear',
                                                       slope=self.slope, intercept=self.intercept)
                        self._mark_init_guess_spaxel()
                        self._persist_edit_override()
                        self._redraw_edit_model()
                else:
                    data_cont = {'Continuum Name': ['Continuum'],
                                 'x1': [round(self.line_start[0],2)],
                                 'x2': [round(self.spectrum_cursor_pos[0],2)],
                                 'Slope_0': [self.slope],
                                 'Intercept_0': [self.intercept],
                                 'Slope_fit': [np.nan],
                                 'Intercept_fit': [np.nan],
                                 'region_ID': [len(df_cont)],
                                 'lineactor': [self.current_line],
                                 'cont_type': ['linear'],
                                 'knots_x': [[]],
                                 'knots_y_0': [[]],
                                 'knots_y_fit': [[]],
                                 'poly_degree': [np.nan],
                                 'poly_coef_0': [[]],
                                 'poly_coef_fit': [[]]}
                    df_cont_new = pd.DataFrame(data_cont)
                    if len(df_cont) == 0:
                        df_cont = df_cont_new
                    else:
                        df_cont = pd.concat([df_cont, df_cont_new], ignore_index=True)

                    # Ensure the Fit Parameters dock is open, then add the new spectral region frame
                    self._ensure_fit_params_open_and_add_region()
                    # Mark this spaxel with a blue square immediately on continuum placement
                    if self.current_spaxel is not None:
                        self._init_guess_spaxel = (int(self.current_spaxel[0]), int(self.current_spaxel[1]))
                        if self._blue_rect is not None:
                            self._blue_rect.set_xy((self._init_guess_spaxel[0] - 0.5,
                                                    self._init_guess_spaxel[1] - 0.5))
                            self._blue_rect.set_visible(True)
                            self.canvas.draw_idle()

        if event.key() == Qt.Key_C and not event.isAutoRepeat():
            # Always start a fresh selection immediately on press.
            # If a previous locked box exists, discard it without a separate clear step.
            if not hasattr(self, 'spectrum_cursor_pos') or not self.spectrum_cursor_pos:
                pass  # cursor not over spectrum — do nothing
            else:
                wav = self.spectrum_cursor_pos[0]
                if wav is not None:
                    # Clear any existing span
                    if self._chanmap_span is not None:
                        try: self._chanmap_span.remove()
                        except ValueError: pass
                        self._chanmap_span = None
                    # Start new selection immediately at current cursor wavelength
                    self._chanmap_active = True
                    self._chanmap_locked = False
                    self._chanmap_start  = wav
                    self._chanmap_span = self.spectrum_ax.axvspan(
                        wav, wav,
                        alpha=0.25, color='#4fc3f7', zorder=0
                    )
                    self.spectrum_canvas.draw_idle()

        # X and V keys: subtraction windows (only when channel map is locked)
        if event.key() in (Qt.Key_X, Qt.Key_V) and not event.isAutoRepeat():
            if not self._chanmap_locked:
                pass  # No channel window yet — ignore
            elif hasattr(self, 'spectrum_cursor_pos') and self.spectrum_cursor_pos:
                wav = self.spectrum_cursor_pos[0]
                key_char = 'x' if event.key() == Qt.Key_X else 'v'
                sm = self._submap[key_char]
                # Clear existing span for this key
                if sm['span'] is not None:
                    try: sm['span'].remove()
                    except ValueError: pass
                    sm['span'] = None
                sm['active'] = True
                sm['locked'] = False
                sm['start']  = wav
                sm['span'] = self.spectrum_ax.axvspan(
                    wav, wav, alpha=0.20, color='#ef5350', zorder=0
                )
                self.spectrum_canvas.draw_idle()

        if event.key() == Qt.Key_G and not event.isAutoRepeat():
            if not self.gaussian_active:
                # First press of 'g': Start Gaussian fitting
                x_cursor, y_cursor = self.spectrum_cursor_pos  # Get cursor position
                self.line_region = self.get_line_region(x_cursor)  # Check if cursor is in x1, x2 range
                if self.line_region is not None:
                    # self.gaussian_active = True
                    self.initial_x = x_cursor  # Lock centroid position
                    self.current_sigma = 10
                    self.current_amplitude = y_cursor
                    
                    # Plot initial Gaussian
                    self.start_gaussian_adjustment(self.line_region['region_ID'])
                    if self.gaussian_active:
                        self._set_spectrum_hint(
                            'Gaussian line   •   move cursor to set width/height    '
                            'G: lock')
                    # self.draw_gaussian(line_region, self.initial_x, self.current_sigma, self.current_amplitude)
            else:
                # Second press of 'g': Lock the Gaussian
                self.gaussian_active = False
                self.save_gaussian(self.line_region['region_ID'])
                self._clear_spectrum_hint()

        # ── Spline continuum drawing (connect-the-dots) ──────────────────────
        # S = add a knot at the cursor; Enter = finalize; Backspace = undo last
        # knot; Esc = cancel. A live segment follows the cursor while drawing.
        if event.key() == Qt.Key_S and not event.isAutoRepeat():
            if (getattr(self, 'spectrum_ax', None) is not None
                    and self.spectrum_cursor_pos
                    and self.spectrum_cursor_pos[0] is not None):
                self._spline_add_knot(*self.spectrum_cursor_pos)
        if self._spline_drawing and not event.isAutoRepeat():
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._spline_finalize()
            elif event.key() == Qt.Key_Backspace:
                self._spline_undo_knot()
            elif event.key() == Qt.Key_Escape:
                self._spline_cancel()

        # ── Polynomial continuum drawing (two-click range) ───────────────────
        # 1st P = range start; 2nd P = range end → fit a Chebyshev polynomial to
        # the spectrum in that range. Esc cancels.
        if event.key() == Qt.Key_P and not event.isAutoRepeat():
            if (getattr(self, 'spectrum_ax', None) is not None
                    and self.spectrum_cursor_pos
                    and self.spectrum_cursor_pos[0] is not None):
                self._poly_click(self.spectrum_cursor_pos[0])
        if self._poly_drawing and event.key() == Qt.Key_Escape and not event.isAutoRepeat():
            self._poly_cancel()

    # ── Spline continuum helpers ─────────────────────────────────────────────

    @staticmethod
    def _eval_spline(knots_x, knots_y, x):
        """Evaluate a continuum spline defined by (knots_x, knots_y) at x.

        Uses an interpolating B-spline of order k = min(3, n_knots-1): 2 knots
        → linear, 3 → quadratic, >=4 → cubic. Values outside the knot span are
        clamped to the nearest end knot.
        """
        kx = np.asarray(knots_x, dtype=float)
        ky = np.asarray(knots_y, dtype=float)
        order = np.argsort(kx)
        kx, ky = kx[order], ky[order]
        # Drop duplicate x (spline requires strictly increasing x)
        keep = np.concatenate(([True], np.diff(kx) > 0))
        kx, ky = kx[keep], ky[keep]
        x = np.asarray(x, dtype=float)
        if kx.size < 2:
            return np.full_like(x, ky[0] if ky.size else 0.0)
        k = min(3, kx.size - 1)
        from scipy.interpolate import make_interp_spline
        spl = make_interp_spline(kx, ky, k=k)
        xc = np.clip(x, kx[0], kx[-1])   # clamp outside knot span
        return spl(xc)

    @staticmethod
    def _eval_poly(coefs, x1, x2, x):
        """Evaluate a Chebyshev-basis continuum polynomial over domain [x1, x2].

        coefs are the Chebyshev coefficients (length degree+1). Uses
        numpy.polynomial.Chebyshev so the domain mapping [x1,x2]→[-1,1] (which
        keeps the fit well-conditioned at spectral wavelengths) is handled
        automatically. Mirrors HyperCube_ModelFunctions.eval_poly.
        """
        c = np.asarray(coefs, dtype=float)
        x = np.asarray(x, dtype=float)
        if c.size == 0:
            return np.zeros_like(x)
        if not (np.isfinite(x1) and np.isfinite(x2) and x2 > x1):
            return np.full_like(x, c[0])
        return np.polynomial.Chebyshev(c, domain=[float(x1), float(x2)])(x)

    def _spline_redraw_preview(self):
        """Redraw the knot markers, the connect-the-dots spline preview, and the
        live rubber-band segment to the cursor during drawing."""
        ax = self.spectrum_ax
        if ax is None:
            return
        # Clear old preview artists
        for attr in ('_spline_dot_artist', '_spline_poly_artist', '_spline_rubber_artist'):
            a = getattr(self, attr, None)
            if a is not None:
                try: a.remove()
                except Exception: pass
                setattr(self, attr, None)
        if not self._spline_knots:
            self.spectrum_canvas.draw_idle()
            return
        kx = [p[0] for p in self._spline_knots]
        ky = [p[1] for p in self._spline_knots]
        # Knot markers
        self._spline_dot_artist, = ax.plot(kx, ky, 'o', color='#ffd54f',
                                           ms=6, zorder=6)
        # Spline preview through the knots (or single point)
        if len(self._spline_knots) >= 2:
            xs = np.linspace(min(kx), max(kx), 300)
            ys = self._eval_spline(kx, ky, xs)
            self._spline_poly_artist, = ax.plot(xs, ys, color='lime', lw=1.2, zorder=5)
        # Rubber-band from last knot to current cursor
        if self.spectrum_cursor_pos and self.spectrum_cursor_pos[0] is not None:
            cx, cy = self.spectrum_cursor_pos
            self._spline_rubber_artist, = ax.plot([kx[-1], cx], [ky[-1], cy],
                                                  color='lime', lw=0.8,
                                                  linestyle='--', zorder=5)
        self.spectrum_canvas.draw_idle()

    def _spline_add_knot(self, x, y):
        """Add a knot at (x, y) and refresh the preview."""
        self._spline_drawing = True
        self._spline_knots.append((float(x), float(y)))
        self._set_spectrum_hint(
            'Spline continuum   •   S: add knot    Enter: submit    '
            'Backspace: undo    Esc: cancel')
        self._spline_redraw_preview()

    def _spline_undo_knot(self):
        """Remove the most recently placed knot."""
        if self._spline_knots:
            self._spline_knots.pop()
        if not self._spline_knots:
            self._spline_drawing = False
            self._clear_spectrum_hint()
        self._spline_redraw_preview()

    def _spline_cancel(self):
        """Discard the in-progress spline."""
        self._spline_knots = []
        self._spline_drawing = False
        self._clear_spectrum_hint()
        self._spline_redraw_preview()

    def _spline_clear_preview_artists(self):
        for attr in ('_spline_dot_artist', '_spline_poly_artist', '_spline_rubber_artist'):
            a = getattr(self, attr, None)
            if a is not None:
                try: a.remove()
                except Exception: pass
                setattr(self, attr, None)

    def _set_spectrum_hint(self, text):
        """Show a small instructional banner at the top of the spectrum axes
        (used during interactive spline/polynomial drawing)."""
        ax = getattr(self, 'spectrum_ax', None)
        if ax is None:
            return
        self._clear_spectrum_hint()
        self._spectrum_hint_artist = ax.text(
            0.5, 0.98, text, transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='#ffd54f', zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                      alpha=0.65, edgecolor='#ffd54f'))
        if getattr(self, 'spectrum_canvas', None) is not None:
            self.spectrum_canvas.draw_idle()

    def _clear_spectrum_hint(self):
        a = getattr(self, '_spectrum_hint_artist', None)
        if a is not None:
            try: a.remove()
            except Exception: pass
            self._spectrum_hint_artist = None
        if getattr(self, 'spectrum_canvas', None) is not None:
            self.spectrum_canvas.draw_idle()

    def _redraw_spline_region(self, region_id):
        """Redraw a committed spline region's continuum line from its current
        knots_x / knots_y_0 (used after editing knots in the panel)."""
        global df_cont
        if getattr(self, 'spectrum_ax', None) is None:
            return
        row = df_cont.loc[df_cont['region_ID'] == region_id]
        if len(row) == 0:
            return
        kx = _as_float_list(row['knots_x'].item())
        ky = _as_float_list(row['knots_y_0'].item())
        if len(kx) < 2:
            return
        xs = np.linspace(min(kx), max(kx), 400)
        ys = self._eval_spline(kx, ky, xs)
        la = row['lineactor'].item()
        if isinstance(la, plt.Line2D):
            la.set_data(xs, ys)
        else:
            new_line, = self.spectrum_ax.plot(xs, ys, color='lime', lw=1, zorder=4)
            df_cont.loc[df_cont['region_ID'] == region_id, 'lineactor'] = new_line
        self.spectrum_canvas.draw_idle()

    def _redraw_poly_region(self, region_id):
        """Redraw a polynomial region's continuum line from its current degree /
        poly_coef_0 (used after editing the degree in the panel)."""
        global df_cont
        if getattr(self, 'spectrum_ax', None) is None:
            return
        row = df_cont.loc[df_cont['region_ID'] == region_id]
        if len(row) == 0:
            return
        x1 = float(row['x1'].item()); x2 = float(row['x2'].item())
        coefs = _as_float_list(row['poly_coef_0'].item())
        if len(coefs) < 1 or not (x2 > x1):
            return
        xs = np.linspace(x1, x2, 400)
        ys = self._eval_poly(coefs, x1, x2, xs)
        la = row['lineactor'].item()
        if isinstance(la, plt.Line2D):
            la.set_data(xs, ys)
        else:
            new_line, = self.spectrum_ax.plot(xs, ys, color='lime', lw=1, zorder=4)
            df_cont.loc[df_cont['region_ID'] == region_id, 'lineactor'] = new_line
        self.spectrum_canvas.draw_idle()

    def _spline_finalize(self):
        """Commit the placed knots as a spline continuum region in df_cont."""
        global df_cont
        if len(self._spline_knots) < 2:
            print("Spline needs at least 2 knots to finalize.")
            return
        # Sort knots by x and split into arrays
        knots = sorted(self._spline_knots, key=lambda p: p[0])
        kx = [round(float(p[0]), 4) for p in knots]
        ky = [float(p[1]) for p in knots]

        # Remove the transient preview artists, then draw the committed spline
        # as this region's continuum 'lineactor' so the existing plot/refresh
        # machinery can manage it.
        self._spline_clear_preview_artists()

        if self._edit_spaxel is not None:
            # Edit mode: REPLACE the region containing the knots, in place.
            rid = self._region_id_at(kx[0])
            if rid is None:
                rid = self._region_id_at(0.5 * (kx[0] + kx[-1]))
            self._spline_knots = []
            self._spline_drawing = False
            self._clear_spectrum_hint()
            if rid is None:
                self._set_spectrum_hint('Spline knots were not inside an existing region.')
                self.spectrum_canvas.draw_idle()
                return
            self._replace_region_continuum(rid, 'spline', knots_x=kx, knots_y=ky)
            self._mark_init_guess_spaxel()
            self._persist_edit_override()
            self._redraw_edit_model()
            return

        xs = np.linspace(kx[0], kx[-1], 400)
        ys = self._eval_spline(kx, ky, xs)
        spline_line, = self.spectrum_ax.plot(xs, ys, color='lime', lw=1, zorder=4)

        region_id = len(df_cont)
        data_cont = {'Continuum Name': ['Spline'],
                     'x1': [kx[0]],
                     'x2': [kx[-1]],
                     'Slope_0': [np.nan],
                     'Intercept_0': [np.nan],
                     'Slope_fit': [np.nan],
                     'Intercept_fit': [np.nan],
                     'region_ID': [region_id],
                     'lineactor': [spline_line],
                     'cont_type': ['spline'],
                     'knots_x': [kx],
                     'knots_y_0': [ky],
                     'knots_y_fit': [[]],
                     'poly_degree': [np.nan],
                     'poly_coef_0': [[]],
                     'poly_coef_fit': [[]]}
        df_cont_new = pd.DataFrame(data_cont)
        if len(df_cont) == 0:
            df_cont = df_cont_new
        else:
            df_cont = pd.concat([df_cont, df_cont_new], ignore_index=True)

        # Reset drawing state
        self._spline_knots = []
        self._spline_drawing = False
        self._clear_spectrum_hint()
        self.spectrum_canvas.draw_idle()

        # Open / add the region panel and mark the init-guess spaxel (mirrors D).
        self._ensure_fit_params_open_and_add_region()
        if self.current_spaxel is not None:
            self._init_guess_spaxel = (int(self.current_spaxel[0]), int(self.current_spaxel[1]))
            if self._blue_rect is not None:
                self._blue_rect.set_xy((self._init_guess_spaxel[0] - 0.5,
                                        self._init_guess_spaxel[1] - 0.5))
                self._blue_rect.set_visible(True)
                self.canvas.draw_idle()

    # ── Polynomial continuum helpers ─────────────────────────────────────────

    def _load_spaxel_model(self, x, y):
        """Load spaxel (x,y)'s per-spaxel model into the active df_cont / df and
        rebuild the fit-params panel so the continuum type/values and line
        initial guesses reflect THIS spaxel.  Unedited spaxels fall back to the
        base template (so the panel 'snaps back' to the base continuum type)."""
        global df_cont, df, base_df_cont, base_df, spaxel_overrides
        if base_df_cont is None:
            return  # no template captured yet (model still being built / no fit)
        key = (int(x), int(y))
        if key in spaxel_overrides:
            df_cont = spaxel_overrides[key]['df_cont']
            df = spaxel_overrides[key]['df']
        else:
            df_cont = base_df_cont.copy(deep=True)
            df = base_df.copy(deep=True)
        # Artist references don't survive being shared across panel rebuilds;
        # null them so rebuild_plot creates fresh curves for this spaxel.
        if 'lineactor' in df_cont.columns:
            df_cont['lineactor'] = None
        if 'curveactor' in df.columns:
            df['curveactor'] = None
        if self.fit_params_window is not None:
            self.fit_params_window.rebuild_fit_panel(show_fit=True, x=int(x), y=int(y))

    def _reset_to_base_model(self):
        """Set the active df_cont/df back to the base template and rebuild the
        panel — used on unlock so the panel doesn't keep showing the last-locked
        spaxel's per-spaxel continuum type."""
        global df_cont, df, base_df_cont, base_df
        if base_df_cont is None:
            return
        df_cont = base_df_cont.copy(deep=True)
        df = base_df.copy(deep=True)
        if 'lineactor' in df_cont.columns:
            df_cont['lineactor'] = None
        if 'curveactor' in df.columns:
            df['curveactor'] = None
        if self.fit_params_window is not None:
            cx = cy = None
            if self.current_spaxel is not None:
                cx, cy = int(self.current_spaxel[0]), int(self.current_spaxel[1])
            self.fit_params_window.rebuild_fit_panel(show_fit=True, x=cx, y=cy)

    def _draw_edited_overlay(self):
        """Redraw the translucent blue boxes over edited spaxels (the keys of
        spaxel_overrides) when the overlay is toggled on."""
        global spaxel_overrides
        for p in getattr(self, '_edited_overlay_patches', []):
            try: p.remove()
            except Exception: pass
        self._edited_overlay_patches = []
        if getattr(self, '_edited_overlay_on', False) and getattr(self, 'ax', None) is not None:
            for (x, y) in spaxel_overrides.keys():
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                         linewidth=1.0, edgecolor='#4fc3f7',
                                         facecolor='#4fc3f7', alpha=0.30, zorder=6)
                self.ax.add_patch(rect)
                self._edited_overlay_patches.append(rect)
        if getattr(self, 'canvas', None) is not None:
            self.canvas.draw_idle()

    def toggle_edited_overlay(self):
        """Toggle the translucent-box overlay marking edited spaxels."""
        self._edited_overlay_on = not getattr(self, '_edited_overlay_on', False)
        self._draw_edited_overlay()

    # ── Stage B: graphical in-place editing in single-spaxel edit mode ────────

    def _region_id_at(self, x):
        """region_ID of the region whose [x1,x2] contains x, else None."""
        row = self.get_line_region(x)
        if row is None:
            return None
        return int(np.int64(row['region_ID']))

    def _replace_region_continuum(self, region_id, cont_type, slope=np.nan, intercept=np.nan,
                                  knots_x=None, knots_y=None, poly_coef=None, poly_degree=np.nan):
        """Replace one region's continuum (type + shape) in place, preserving its
        x-range, region_ID and lines. Used by edit-mode d/s/p so the locked
        schema (region/line count) is never changed."""
        global df_cont
        # Ensure all continuum-type columns exist (regions made via '+ Region'
        # may lack the spline/poly columns).
        for col in ('cont_type', 'knots_x', 'knots_y_0', 'knots_y_fit',
                    'poly_degree', 'poly_coef_0', 'poly_coef_fit'):
            if col not in df_cont.columns:
                if col in ('poly_degree',):
                    df_cont[col] = np.nan
                elif col == 'cont_type':
                    df_cont[col] = 'linear'
                else:
                    df_cont[col] = [[] for _ in range(len(df_cont))]
        m = np.int64(df_cont['region_ID']) == np.int64(region_id)
        if not m.any():
            return
        idx = df_cont.index[m][0]
        # Reset type-specific columns, then set the chosen type's params.
        df_cont.at[idx, 'cont_type']    = cont_type
        df_cont.at[idx, 'lineactor']    = None
        df_cont.at[idx, 'Slope_0']      = np.nan
        df_cont.at[idx, 'Intercept_0']  = np.nan
        df_cont.at[idx, 'knots_x']      = []
        df_cont.at[idx, 'knots_y_0']    = []
        df_cont.at[idx, 'knots_y_fit']  = []
        df_cont.at[idx, 'poly_degree']  = np.nan
        df_cont.at[idx, 'poly_coef_0']  = []
        df_cont.at[idx, 'poly_coef_fit'] = []
        if cont_type == 'linear':
            df_cont.at[idx, 'Slope_0']        = slope
            df_cont.at[idx, 'Intercept_0']    = intercept
            df_cont.at[idx, 'Continuum Name'] = 'Continuum'
        elif cont_type == 'spline':
            df_cont.at[idx, 'knots_x']        = list(knots_x or [])
            df_cont.at[idx, 'knots_y_0']      = list(knots_y or [])
            df_cont.at[idx, 'Continuum Name'] = 'Spline'
        elif cont_type == 'poly':
            df_cont.at[idx, 'poly_coef_0']    = list(poly_coef or [])
            df_cont.at[idx, 'poly_degree']    = poly_degree
            df_cont.at[idx, 'Continuum Name'] = 'Polynomial'

    def _persist_edit_override(self):
        """Snapshot the active model into this spaxel's override entry."""
        global spaxel_overrides, df_cont, df
        if self._edit_spaxel is None:
            return
        ov_cont = df_cont.copy(deep=True)
        ov_df   = df.copy(deep=True)
        if 'lineactor' in ov_cont.columns:
            ov_cont['lineactor'] = None
        if 'curveactor' in ov_df.columns:
            ov_df['curveactor'] = None
        spaxel_overrides[self._edit_spaxel] = {'df_cont': ov_cont, 'df': ov_df}

    def _redraw_edit_model(self):
        """Clear model curves, draw the override's init-guess model for every
        region, and rebuild the panel (continuum type may have changed)."""
        global df_cont
        for ln in self.spectrum_ax.lines[:]:
            if ln.get_label() != '_child0':
                try: ln.remove()
                except ValueError: pass
        self.gaussian_component_lines.clear()
        if 'lineactor' in df_cont.columns:
            df_cont['lineactor'] = None
        ex, ey = self._edit_spaxel if self._edit_spaxel else (0, 0)
        for rid in np.unique(np.int64(df_cont['region_ID'])):
            self.rebuild_plot(rid, from_file=True, show_init=True, show_fit=False, x=ex, y=ey)
        self.spectrum_canvas.draw_idle()
        if self.fit_params_window is not None:
            self.fit_params_window.rebuild_fit_panel(show_fit=False, x=ex, y=ey)

    def _mark_init_guess_spaxel(self):
        """Mark the current spaxel as the init-guess spaxel (blue square)."""
        if self.current_spaxel is not None:
            self._init_guess_spaxel = (int(self.current_spaxel[0]), int(self.current_spaxel[1]))
            if self._blue_rect is not None:
                self._blue_rect.set_xy((self._init_guess_spaxel[0] - 0.5,
                                        self._init_guess_spaxel[1] - 0.5))
                self._blue_rect.set_visible(True)
                self.canvas.draw_idle()

    def _poly_clear_preview_artists(self):
        a = getattr(self, '_poly_preview_artist', None)
        if a is not None:
            try: a.remove()
            except Exception: pass
            self._poly_preview_artist = None
        for e in getattr(self, '_poly_edge_artists', []) or []:
            try: e.remove()
            except Exception: pass
        self._poly_edge_artists = []

    def _poly_redraw_preview(self):
        """Live preview during 'P' selection: thin markers at the range edges
        plus the polynomial that would be fit to the data in the range so far."""
        ax = self.spectrum_ax
        if ax is None or self._poly_x1 is None:
            return
        self._poly_clear_preview_artists()
        cur = self.spectrum_cursor_pos[0] if (self.spectrum_cursor_pos and
                                              self.spectrum_cursor_pos[0] is not None) else self._poly_x1
        x1, x2 = sorted([self._poly_x1, cur])
        # Range-edge markers
        for xe in (x1, x2):
            self._poly_edge_artists.append(
                ax.axvline(xe, color='#ffb74d', lw=0.8, ls='--', zorder=5))
        # Fitted-polynomial preview curve
        if getattr(self, 'spectrum_line', None) is not None and x2 > x1:
            xx = np.asarray(self.spectrum_line.get_xdata(), dtype=float)
            yy = np.asarray(self.spectrum_line.get_ydata(), dtype=float)
            m = (xx >= x1) & (xx <= x2) & np.isfinite(yy)
            n = int(np.count_nonzero(m))
            if n >= 2:
                deg = int(max(1, min(self._poly_default_degree, n - 1)))
                try:
                    cheb = np.polynomial.Chebyshev.fit(xx[m], yy[m], deg, domain=[x1, x2])
                    xs = np.linspace(x1, x2, 300)
                    self._poly_preview_artist, = ax.plot(xs, cheb(xs),
                                                         color='lime', lw=1.2, zorder=6)
                except Exception:
                    pass
        self.spectrum_canvas.draw_idle()

    def _poly_click(self, x):
        """Handle a 'P' press: first sets the range start, second finalizes."""
        if not self._poly_drawing:
            self._poly_drawing = True
            self._poly_x1 = float(x)
            self._set_spectrum_hint(
                'Polynomial continuum   •   P: set range end (fit to data)    '
                'Esc: cancel')
            self._poly_redraw_preview()
        else:
            self._poly_finalize(float(x))

    def _poly_cancel(self):
        """Abort an in-progress polynomial range selection."""
        self._poly_clear_preview_artists()
        self._poly_drawing = False
        self._poly_x1 = None
        self._clear_spectrum_hint()
        self.spectrum_canvas.draw_idle()

    def _poly_finalize(self, x2):
        """Commit the [x1, x2] range as a Chebyshev-polynomial continuum region,
        seeded by a least-squares fit to the displayed spectrum in that range."""
        global df_cont
        x1, x2 = sorted([self._poly_x1, float(x2)])
        # Clear the preview curve + edge markers
        self._poly_clear_preview_artists()
        self._poly_drawing = False
        self._poly_x1 = None
        self._clear_spectrum_hint()

        if not np.isfinite(x1) or not np.isfinite(x2) or (x2 - x1) <= 0:
            self.spectrum_canvas.draw_idle()
            return

        # Sample the displayed spectrum inside the range for the initial LSQ fit.
        if getattr(self, 'spectrum_line', None) is None:
            return
        xx = np.asarray(self.spectrum_line.get_xdata(), dtype=float)
        yy = np.asarray(self.spectrum_line.get_ydata(), dtype=float)

        if self._edit_spaxel is not None:
            # Edit mode: REPLACE the region under the drawn range, in place.
            # x-range is locked to base, so fit over the REGION's range with a
            # matching domain (so stored coefs evaluate correctly there).
            rid = self._region_id_at(0.5 * (x1 + x2))
            if rid is None:
                rid = self._region_id_at(x1)
            if rid is None:
                self._set_spectrum_hint('Polynomial range was not inside an existing region.')
                self.spectrum_canvas.draw_idle()
                return
            reg = df_cont.loc[np.int64(df_cont['region_ID']) == np.int64(rid)]
            rx1 = float(reg['x1'].item()); rx2 = float(reg['x2'].item())
            mm = (xx >= rx1) & (xx <= rx2) & np.isfinite(yy)
            if np.count_nonzero(mm) < 2:
                print("Polynomial region too narrow / no data.")
                self.spectrum_canvas.draw_idle()
                return
            deg = int(max(1, min(self._poly_default_degree, np.count_nonzero(mm) - 1)))
            try:
                cheb = np.polynomial.Chebyshev.fit(xx[mm], yy[mm], deg, domain=[rx1, rx2])
                coefs = [float(c) for c in cheb.coef]
            except Exception as e:
                print(f"Polynomial fit failed: {e}")
                return
            self._replace_region_continuum(rid, 'poly', poly_coef=coefs, poly_degree=deg)
            self._mark_init_guess_spaxel()
            self._persist_edit_override()
            self._redraw_edit_model()
            return

        m = (xx >= x1) & (xx <= x2) & np.isfinite(yy)
        if np.count_nonzero(m) < 2:
            print("Polynomial range too narrow / no data.")
            self.spectrum_canvas.draw_idle()
            return
        xr, yr = xx[m], yy[m]

        degree = int(max(1, min(self._poly_default_degree, np.count_nonzero(m) - 1)))
        try:
            cheb = np.polynomial.Chebyshev.fit(xr, yr, degree, domain=[x1, x2])
            coefs = [float(c) for c in cheb.coef]
        except Exception as e:
            print(f"Polynomial seed fit failed: {e}")
            return

        # Draw the seeded continuum and register the region.
        xs = np.linspace(x1, x2, 400)
        ys = self._eval_poly(coefs, x1, x2, xs)
        poly_line, = self.spectrum_ax.plot(xs, ys, color='lime', lw=1, zorder=4)

        region_id = len(df_cont)
        data_cont = {'Continuum Name': ['Polynomial'],
                     'x1': [round(x1, 4)], 'x2': [round(x2, 4)],
                     'Slope_0': [np.nan], 'Intercept_0': [np.nan],
                     'Slope_fit': [np.nan], 'Intercept_fit': [np.nan],
                     'region_ID': [region_id], 'lineactor': [poly_line],
                     'cont_type': ['poly'],
                     'knots_x': [[]], 'knots_y_0': [[]], 'knots_y_fit': [[]],
                     'poly_degree': [degree], 'poly_coef_0': [coefs], 'poly_coef_fit': [[]]}
        df_cont_new = pd.DataFrame(data_cont)
        if len(df_cont) == 0:
            df_cont = df_cont_new
        else:
            df_cont = pd.concat([df_cont, df_cont_new], ignore_index=True)

        self.spectrum_canvas.draw_idle()
        self._ensure_fit_params_open_and_add_region()
        self._mark_init_guess_spaxel()

    def start_gaussian_adjustment(self, region_ID):
        """Initialize Gaussian drawing when 'G' is pressed, linked to mouse motion."""
        if self.gaussian_active:
            # If Gaussian adjustment is already active, finalize it
            self.gaussian_active = False
            return  
        
        self.gaussian_active = True
        
        # Retrieve the line associated with frame_id
        line_obj = df_cont.loc[df_cont["region_ID"] == region_ID, 'lineactor'].item()
    
        if not isinstance(line_obj, plt.Line2D):
            print(f"Error: No valid line object found.")
            self.gaussian_active = False  # don't leave a half-started state for the next 'G'
            return

        # Store the line and initialize Gaussian parameters
        self.active_line = line_obj
        region = df_cont.loc[df_cont["region_ID"] == region_ID]
        self.x1, self.x2 = region["x1"].item(), region["x2"].item()
        self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
        # Keep the region slice so the continuum baseline (linear or spline) can
        # be evaluated while the Gaussian is dragged.
        self._gauss_region = region

        # Capture cursor position for Gaussian center
        if self.spectrum_cursor_pos:
            self.gaussian_x0, _ = self.spectrum_cursor_pos  # x0 is fixed at cursor position
    
        # self.gaussian_sigma = 5  # Initial sigma
        # self.gaussian_amplitude = 1  # Initial amplitude
        self.gaussian_sigma = 20 * abs(wavelengths[1]-wavelengths[0])
        self.gaussian_amplitude = 0.1 * (np.diff(self.spectrum_ax.get_ylim())[0])
    
        # Store Gaussian component lines for individual display
        self.gaussian_component_lines = []
    
        # Set flag to indicate active adjustment
        self.gaussian_active = True
    
        # Connect the mouse move event
        self.spectrum_canvas.mpl_connect("motion_notify_event", self.update_gaussian_dynamic)
    
    
    def update_gaussian_dynamic(self, event):
        """Update the Gaussian shape dynamically based on cursor movement."""
        if not self.gaussian_active or event.inaxes != self.spectrum_ax:
            return
    
        # Adjust sigma (horizontal motion) and amplitude (vertical motion)
        dx = event.xdata - self.gaussian_x0
        dy = event.ydata
    
        self.gaussian_sigma = max(0.0001, abs(dx))  # Sigma changes with horizontal movement
    
        # Compute the continuum level at the Gaussian's center position
        # (linear or spline baseline for this region).
        _reg = getattr(self, '_gauss_region', None)
        y_continuum_x0 = float(self._region_baseline(_reg, np.array([self.gaussian_x0]))[0])

        # Adjust amplitude so that peak of the Gaussian reaches dy
        self.gaussian_amplitude = dy - y_continuum_x0#max(1E-20, dy - y_continuum_x0)

        # Generate updated Gaussian + continuum baseline
        x_vals = np.linspace(self.x1, self.x2, 1000)
        y_line = self._region_baseline(_reg, x_vals)  # baseline (without Gaussians)
    
        # Clear existing individual component lines
        while self.gaussian_component_lines:
            line = self.gaussian_component_lines.pop()
            line.remove()

        # Existing Gaussians are shown GREY for reference only while a new one is
        # being placed — they are NOT summed into the live model until the user
        # presses 'g' a second time to lock the new line.
        for _, row in df.iterrows():
            y_gaussian = row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
            component_line, = self.spectrum_ax.plot(
                x_vals, y_gaussian + y_line, color='#888888', alpha=0.45,
                linestyle='--', linewidth=1)
            self.gaussian_component_lines.append(component_line)

        # The new Gaussian being placed: prominent orange dashed.
        y_gaussian_new = self.gaussian_amplitude * np.exp(-((x_vals - self.gaussian_x0) ** 2) / (2 * self.gaussian_sigma ** 2))
        new_component_line, = self.spectrum_ax.plot(x_vals, y_gaussian_new + y_line, color='#d7801a', linestyle='--', linewidth=1)
        self.gaussian_component_lines.append(new_component_line)

        # Live model = continuum + the new Gaussian only.
        y_new = y_line + y_gaussian_new

        # Update the existing line object for the summed model
        self.active_line.set_data(x_vals, y_new)
    
        # Redraw dynamically
        self.spectrum_ax.figure.canvas.draw_idle()





    def save_gaussian(self, region_ID):
        global df
        """Save final Gaussian parameters to df_cont"""

        if self._edit_spaxel is not None:
            # Edit mode: snap to the NEAREST existing line in this region (by
            # centroid) and re-specify its initial guess — never add a line.
            # Use positional indexing: df can carry duplicate index labels
            # (built via pd.concat without ignore_index), so .loc[label] would
            # match multiple rows and collapse all lines to one parameter set.
            reg = df[np.int64(df['region_ID']) == np.int64(region_ID)]
            if len(reg) == 0:
                return
            cens = reg['Centroid_0'].astype(float).to_numpy()
            pos = int(np.argmin(np.abs(cens - float(self.gaussian_x0))))
            nearest_lid = float(reg['Line_ID'].to_numpy()[pos])
            mask = ((np.int64(df['region_ID']) == np.int64(region_ID)) &
                    (df['Line_ID'].astype(float) == nearest_lid))
            df.loc[mask, 'Amp_0']      = self.gaussian_amplitude
            df.loc[mask, 'Centroid_0'] = self.gaussian_x0
            df.loc[mask, 'Sigma_0']    = self.gaussian_sigma
            self._mark_init_guess_spaxel()
            self._persist_edit_override()
            self._redraw_edit_model()
            return

        if len(df) == 0:
            line_id = 0
        else:
            line_id = np.max(np.int64(df['Line_ID']))+1
        
        # Auto-identify the line using the line library and source redshift
        _z        = df_obs.loc[0, 'redshift'] if len(df_obs) > 0 else ''
        _ion, _rest_wav = _identify_line(self.gaussian_x0, _z)
        # Build base name: ion + rest wavelength subscript
        if _ion and np.isfinite(_rest_wav):
            _base_name = f'{_ion}_{int(round(_rest_wav))}'
        elif _ion:
            _base_name = _ion
        else:
            _base_name = f'Line {len(df.loc[df["region_ID"]==region_ID])}'
        # If the base name already exists in df, append _b, _c, _d … to disambiguate
        existing_names = set(df['Line_Name'].astype(str).tolist())
        _line_name = _base_name
        if _line_name in existing_names:
            for _letter in 'bcdefghijklmnopqrstuvwxyz':
                _candidate = f'{_base_name}_{_letter}'
                if _candidate not in existing_names:
                    _line_name = _candidate
                    break
        _rest_wav_store = _rest_wav if np.isfinite(_rest_wav) else np.nan

        df_new = pd.DataFrame({'Line_ID': [line_id],
                'Line_Name': [_line_name],
                'SNR': [0],
                'Rest Wavelength': [_rest_wav_store],
                'Amp_0': [self.gaussian_amplitude],
                'Amp_0_lowlim': [0],
                'Amp_0_highlim': [np.inf],
                'Centroid_0': [self.gaussian_x0],
                'Centroid_0_lowlim': [-np.inf],
                'Centroid_0_highlim': [np.inf],
                'Sigma_0': [self.gaussian_sigma],
                'Sigma_0_lowlim': [0],
                'Sigma_0_highlim': [np.inf],
                'Amp_fit': [np.nan],
                'Centroid_fit': [np.nan],
                'Sigma_fit': [np.nan],
                'region_ID': [region_ID],
                'curveactor': [self.active_line]})
        
        
        
        
        df = pd.concat([df,df_new])

        # Mark this spaxel with a blue square on the cube viewport
        if self.current_spaxel is not None:
            self._init_guess_spaxel = (int(self.current_spaxel[0]), int(self.current_spaxel[1]))
            if self._blue_rect is not None:
                self._blue_rect.set_xy((self._init_guess_spaxel[0] - 0.5,
                                        self._init_guess_spaxel[1] - 0.5))
                self._blue_rect.set_visible(True)
                self.canvas.draw_idle()

        self.fit_params_window.on_addline_button_click(df,frame_id=region_ID)
        
        
        
    def _region_baseline(self, region, x_vals, use_fit=False, xy=None):
        """Continuum baseline y over x_vals for a single-row df_cont `region`
        slice. Branches on cont_type: linear → slope·x+intercept; spline →
        interpolating spline through the knots; stellar → pPXF optimal template
        broadened by the LOSVD. use_fit selects the fitted params (Slope_fit /
        knots_y_fit) when available, else the initial guess. `xy` selects which
        spaxel's cached optimal template to use for a stellar region.
        """
        x_vals = np.asarray(x_vals, dtype=float)
        if region is None or len(region) == 0:
            return np.zeros_like(x_vals)
        ctype = 'linear'
        if 'cont_type' in region.columns:
            try:
                ctype = str(region['cont_type'].item())
            except Exception:
                ctype = 'linear'
        if ctype == 'stellar':
            return self._stellar_baseline(region, x_vals, use_fit=use_fit, xy=xy)
        if ctype == 'spline':
            kx = _as_float_list(region['knots_x'].item())
            ky0 = _as_float_list(region['knots_y_0'].item())
            ky_fit = _as_float_list(region['knots_y_fit'].item()) if 'knots_y_fit' in region.columns else []
            ky = ky_fit if (use_fit and len(ky_fit) == len(kx) and len(ky_fit) > 0) else ky0
            return self._eval_spline(kx, ky, x_vals)
        if ctype == 'poly':
            x1 = float(region['x1'].item()); x2 = float(region['x2'].item())
            c0 = _as_float_list(region['poly_coef_0'].item())
            cfit = _as_float_list(region['poly_coef_fit'].item()) if 'poly_coef_fit' in region.columns else []
            coefs = cfit if (use_fit and len(cfit) > 0) else c0
            return self._eval_poly(coefs, x1, x2, x_vals)
        # Linear
        if use_fit:
            m = np.float64(region['Slope_fit'].item()); b = np.float64(region['Intercept_fit'].item())
        else:
            m = np.float64(region['Slope_0'].item()); b = np.float64(region['Intercept_0'].item())
        return m * x_vals + b

    def _ensure_stellar_cache(self, x, y, rid):
        """Lazily compute & cache the pPXF optimal template for spaxel (x,y),
        region `rid`, if it is absent. The parallel cube fit stores kinematics
        only (df_stellar drives the maps), so a spaxel's optimal template is
        recomputed on demand here the first time its stellar baseline is drawn.
        No-op if already cached or pPXF/data unavailable."""
        key = (int(x), int(y), int(rid))
        if (STELLAR_CACHE.get(key) is not None or hcppxf is None
                or FITS_DATA is None or 'cont_type' not in df_cont.columns):
            return
        reg = df_cont[(np.int64(df_cont['region_ID']) == np.int64(rid)) &
                      (df_cont['cont_type'] == 'stellar')]
        if len(reg) == 0:
            return
        r = reg.iloc[0]
        try:
            library = str(r['stellar_library'])
            fit_range = (float(r['x1']), float(r['x2']))
            moments = int(r['stellar_moments']) if pd.notna(r.get('stellar_moments')) else 2
            z = _safe_float(df_obs.loc[0, 'redshift']) if len(df_obs) else 0.0
            z = 0.0 if not np.isfinite(z) else z
            R = _safe_float(df_obs.loc[0, 'resolvingpower']) if len(df_obs) else np.nan
            R = 3000.0 if not np.isfinite(R) or R <= 0 else R
            spectrum = np.nan_to_num(self.get_spectrum_at_spaxel(int(x), int(y)))
            velscale, lam_rest = hcppxf.galaxy_velscale(wavelengths, z, fit_range)
            lib = _STELLAR_LIBS.get(library) or hcppxf.TemplateLibrary(library).load()
            _STELLAR_LIBS[library] = lib
            lib.prepare(velscale, R, lam_rest)
            mask = (df['Centroid_0'].astype(float).to_numpy()
                    if len(df) > 0 and 'Centroid_0' in df.columns else ())
            res = hcppxf.fit_stellar(spectrum, wavelengths, z, R, lib,
                                     fit_range=fit_range, mask_centroids=mask,
                                     moments=moments, degree=-1, mdegree=10,
                                     sigma_guess=150.0, velscale=velscale)
            STELLAR_CACHE[key] = res['cache']
        except Exception as e:
            print(f"lazy stellar cache failed for ({x},{y},{rid}): {e}")

    def _stellar_baseline(self, region, x_vals, use_fit=False, xy=None):
        """Stellar continuum baseline for a 'stellar' region: the cached pPXF
        optimal template broadened by the region's LOSVD (V, σ, h3, h4) and scaled.
        Falls back to zeros if the optimal-template cache for this spaxel is absent."""
        if hcppxf is None:
            return np.zeros_like(x_vals)
        # Cache is keyed per (spaxel, region) so several stellar regions in the
        # same spaxel each keep their own optimal template.
        try:
            rid = int(np.int64(region['region_ID'].iloc[0]))
        except Exception:
            rid = None
        xy_key = None
        if xy is not None:
            xy_key = (int(xy[0]), int(xy[1]))
        elif self.current_spaxel is not None:
            xy_key = (int(self.current_spaxel[0]), int(self.current_spaxel[1]))
        cache = None
        if xy_key is not None and rid is not None:
            # Lazily (re)compute this spaxel/region's optimal template if a
            # parallel fit stored kinematics only.
            self._ensure_stellar_cache(xy_key[0], xy_key[1], rid)
            cache = STELLAR_CACHE.get((xy_key[0], xy_key[1], rid))
        if cache is None:
            return np.zeros_like(x_vals)
        # Each spaxel's cache carries its OWN kinematics (set by pPXF, updated on
        # manual cell edits) — use them so every spaxel renders correctly during
        # cube fits and navigation. Fall back to the region's df_cont cells.
        r = region.iloc[0] if hasattr(region, 'iloc') else region
        sfx = 'fit' if use_fit and np.isfinite(_safe_float(r.get('stellar_V_fit'))) else '0'
        V = _safe_float(cache.get('V', r.get(f'stellar_V_{sfx}')))
        sig = _safe_float(cache.get('sigma', r.get(f'stellar_sigma_{sfx}')))
        h3 = _safe_float(cache.get('h3', r.get(f'stellar_h3_{sfx}'))) or 0.0
        h4 = _safe_float(cache.get('h4', r.get(f'stellar_h4_{sfx}'))) or 0.0
        scale = _safe_float(cache.get('scale', r.get(f'stellar_scale_{sfx}')))
        if not (np.isfinite(V) and np.isfinite(sig) and np.isfinite(scale)):
            return np.zeros_like(x_vals)
        try:
            y = hcppxf.eval_stellar_baseline(cache, V, sig, h3, h4, scale, x_vals)
        except Exception as e:
            print(f"stellar baseline eval failed: {e}")
            return np.zeros_like(x_vals)
        return np.nan_to_num(y)

    def rebuild_plot(self,region_ID, from_file, show_init, show_fit, x, y):
        """used after deleting emission line, adding new line, or editing line"""
        global df, df_cont
        
        region = df_cont.loc[np.int64(df_cont["region_ID"]) == np.int64(region_ID)]
        # '''
        
        if show_init == True:
            self.x1 = np.float64(region["x1"].item())
            self.x2 = np.float64(region["x2"].item())
            # Linear m/b kept for backward compat (unused for spline rows).
            self.m = np.float64(region["Slope_0"].item())
            self.b = np.float64(region["Intercept_0"].item())

            # Generate updated Gaussian + continuum baseline (linear/spline/stellar)
            x_vals = np.linspace(self.x1, self.x2, 1000)
            y_line = self._region_baseline(region, x_vals, use_fit=False, xy=(x, y))
            
            # '''
            gaussians = df.loc[np.int64(df["region_ID"]) == region_ID]

            # Recompute all Gaussians from scratch (prevents runaway summing)
            y_gaussian_total = np.zeros_like(x_vals)

            # Sum over all existing Gaussians in df
            i = 0
            for _, row in gaussians.iterrows():
                row['Amp_0'] = np.float64(row['Amp_0'])
                row['Centroid_0'] = np.float64(row['Centroid_0'])
                row['Sigma_0'] = np.float64(row['Sigma_0'])
                gaussian = row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
                new_curve, = self.spectrum_ax.plot(x_vals, gaussian+y_line, color='#d7801a', lw=1,linestyle='--')
                df.loc[(np.int64(df['region_ID']==region_ID)) & (df['Line_ID'] == i), 'curveactor'] = new_curve
                y_gaussian_total += gaussian
                i = i+1


            # Final y-values
            y_new = y_line + y_gaussian_total

            # Update the existing line object
            if from_file == False:
                line_obj = df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'].item()
                if line_obj is not None:  # None in edit mode; fresh curve drawn below
                    line_obj.set_data(x_vals, y_new)


            new_line, = self.spectrum_ax.plot(x_vals, y_new, color='lime', lw=1)
            df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'] = new_line  # Update reference


        if show_fit == True:
            # Model definition for this region (continuum type lives in df_cont).
            _model_reg = df_cont.loc[np.int64(df_cont['region_ID']) == int(region_ID)]
            _is_stellar = (len(_model_reg) > 0 and 'cont_type' in df_cont.columns
                           and str(_model_reg.iloc[0]['cont_type']) == 'stellar')
            # Fitted emission-line rows for this spaxel/region (may be empty).
            if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0 and 'spaxel_x' in df_fit.columns:
                region = df_fit.loc[(df_fit['spaxel_x'].astype(int) == int(x)) &
                                    (df_fit['spaxel_y'].astype(int) == int(y)) &
                                    (df_fit['region_ID'].astype(int) == int(region_ID))]
            else:
                region = df_fit.iloc[0:0] if isinstance(df_fit, pd.DataFrame) else pd.DataFrame()

            if _is_stellar:
                # Stellar baseline from the cached optimal template; draw it even
                # if no emission lines were fit for this spaxel.
                has_cache = STELLAR_CACHE.get((int(x), int(y), int(region_ID))) is not None
                if not has_cache and len(region) == 0:
                    return
                self.x1 = float(_model_reg.iloc[0]['x1'])
                self.x2 = float(_model_reg.iloc[0]['x2'])
                x_vals = np.linspace(self.x1, self.x2, 1000)
                self.m, self.b = np.nan, np.nan
                y_line = np.nan_to_num(self._stellar_baseline(
                    _model_reg, x_vals, use_fit=True, xy=(x, y)))
            else:
                if len(region) == 0:
                    return  # no fit data for this spaxel/region (e.g. below SNR cut)
                _pfx = "cont_region" + str(region_ID + 1) + "_"
                self.x1 = np.float64(region.iloc[0][_pfx + "x_start"].item())
                self.x2 = np.float64(region.iloc[0][_pfx + "x_end"].item())
                x_vals = np.linspace(self.x1, self.x2, 1000)
                _ctype_col = _pfx + "cont_type"
                _ctype = str(region.iloc[0][_ctype_col]) if _ctype_col in region.columns else 'linear'
                if _ctype == 'poly':
                    coefs = _as_float_list(region.iloc[0][_pfx + "poly_coef_fit"])
                    self.m, self.b = np.nan, np.nan
                    y_line = self._eval_poly(coefs, self.x1, self.x2, x_vals)
                elif _ctype == 'spline':
                    kx = _as_float_list(region.iloc[0][_pfx + "knots_x"])
                    ky = _as_float_list(region.iloc[0][_pfx + "knots_y_fit"])
                    self.m, self.b = np.nan, np.nan
                    y_line = self._eval_spline(kx, ky, x_vals)
                else:
                    self.m = np.float64(region.iloc[0][_pfx + "slope_fit"].item())
                    self.b = np.float64(region.iloc[0][_pfx + "intercept_fit"].item())
                    y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
            
            # '''
            # gaussians = df.loc[np.int64(df["region_ID"]) == region_ID]
    
            # Recompute all Gaussians from scratch (prevents runaway summing)
            y_gaussian_total = np.zeros_like(x_vals)

            # Sum over all existing Gaussians in df
            for _, row in region.iterrows():
                row['amp_fit'] = np.float64(row['amp_fit'])
                row['cen_fit'] = np.float64(row['cen_fit'])
                row['sigma_fit'] = np.float64(row['sigma_fit'])
                plotcolor = row['color']
                y_gaussian = row["amp_fit"] * np.exp(-((x_vals - row["cen_fit"]) ** 2) / (2 * row["sigma_fit"] ** 2))
                y_gaussian_total += y_gaussian
                fit_component, = self.spectrum_ax.plot(x_vals,y_line+y_gaussian,lw=1,linestyle='--',color=plotcolor)


            # Final y-values
            y_new = y_line + y_gaussian_total

            # Update the existing line object
            if from_file == False:
                line_obj = df_cont.loc[np.int64(df_cont["region_ID"]) == np.int64(region_ID), 'lineactor'].item()
                if line_obj is not None:  # None in edit mode; fresh curve drawn below
                    line_obj.set_data(x_vals, y_new)


            new_line, = self.spectrum_ax.plot(x_vals, y_new, color='red', lw=0.5)
            df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'] = new_line  # Update reference

            # ── Push fitted values into the parameter buttons ─────────────
            if self.fit_params_window is not None:
                fp = self.fit_params_window
                for _, frow in region.iterrows():
                    lid = np.int64(frow['LineID'])
                    prefix = str(lid)
                    _map = {
                        f'{prefix}~Amp_fit':      frow.get('amp_fit', np.nan),
                        f'{prefix}~Centroid_fit': frow.get('cen_fit', np.nan),
                        f'{prefix}~v_fit':        frow.get('vel_fit', np.nan),
                        # σ_fit shown as velocity dispersion (km/s) via fitted centroid
                        f'{prefix}~Sigma_fit':    sigma_wl_to_kms(frow.get('sigma_fit', np.nan),
                                                                  frow.get('cen_fit', np.nan)),
                    }
                    for btn_name, val in _map.items():
                        key = (np.int64(region_ID), btn_name)
                        if key in fp.buttons_dict:
                            try:
                                fval = float(val)
                                txt = f"{fval:.4g}" if np.isfinite(fval) else "—"
                            except (TypeError, ValueError):
                                txt = "—"
                            fp.buttons_dict[key].setText(txt)

        # Redraw dynamically
        self.spectrum_ax.figure.canvas.draw_idle()
        
        
        
    def get_line_region(self, x_cursor):
        """Check if cursor is within x1, x2 of any existing lines"""
        for index, row in df_cont.iterrows():
            if row["x1"] <= x_cursor <= row["x2"]:
                return row  # Return the matching line's data
        return None

                
    # Percentile clip levels shared by the cube and background stretch menus.
    _STRETCH_PCT = {
        'minmax': (0, 100),
        '99.9%':  (0.05, 99.95),
        '99.5%':  (0.25, 99.75),
        '99%':    (0.5,  99.5),
        '98%':    (1,    99),
        '95%':    (2.5,  97.5),
    }

    def _stretch_limits(self, data, name):
        """Return (vmin, vmax) for a named stretch.

        Supports the percentile clips in _STRETCH_PCT plus 'zscale' (the DS9 /
        IRAF z-scale algorithm via astropy's ZScaleInterval). Falls back to the
        99% clip for unknown names or if zscale fails.
        """
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        if name == 'zscale':
            try:
                from astropy.visualization import ZScaleInterval
                vmin, vmax = ZScaleInterval().get_limits(finite)
                if vmax > vmin:
                    return float(vmin), float(vmax)
            except Exception:
                pass
            name = '99%'
        lo_pct, hi_pct = self._STRETCH_PCT.get(name, (0.5, 99.5))
        return float(np.percentile(finite, lo_pct)), float(np.percentile(finite, hi_pct))

    def draw_image(self, data, cmap, scale, from_fits=False):
        """Generates and displays the white-light image in the left panel."""
        self._last_cmap  = cmap
        self._last_scale = scale
        self._last_from_fits = from_fits
        if from_fits:
            self._quality_map_label = None
        # Only update _last_data when called from a real data load (not from
        # _cube_redraw_transformed, which passes already-transformed 2D data).
        # _cube_redraw_transformed sets _applying_transform=True before calling.
        if not getattr(self, '_applying_transform', False):
            self._last_data = data
            # Cache the original 2D shape for coord inverse transform
            if from_fits and hasattr(data, 'ndim') and data.ndim == 3:
                self._cube_orig_shape = (data.shape[1], data.shape[2])  # (ny, nx)
            elif not from_fits and hasattr(data, 'ndim') and data.ndim == 2:
                self._cube_orig_shape = data.shape
        global npix_x, npix_y
        if from_fits:
            image = np.nansum(data, axis=0).astype(np.float64)
            npix_x = np.shape(image)[0]
            npix_y = np.shape(image)[1]
        else:
            image = np.array(data, dtype=np.float64)

        # ── Apply spatial mask, if one is active ──────────────────────
        # _spaxel_mask is a boolean (ny, nx) array: True = hidden. It is
        # applied to every displayed field (white-light or any parameter /
        # quality map) so masked spaxels read as NaN (transparent).
        _smask = getattr(self, '_spaxel_mask', None)
        if _smask is not None and hasattr(image, 'shape') \
                and image.ndim == 2 and _smask.shape == image.shape:
            image = np.where(_smask, np.nan, image)

        # ── Apply transfer function (scale) ───────────────────────────
        # Prefer toolbar setting; fall back to passed-in scale arg
        tf = getattr(self, 'scale_combo', None)
        tf_name = tf.currentText() if tf else scale

        # Shift to positive before non-linear transforms
        min_val = np.nanmin(image)
        if min_val < 0:
            image = image - min_val

        if tf_name == 'Log':
            image = np.log10(np.where(image > 0, image, np.nan))
        elif tf_name == 'Square root':
            image = np.sqrt(np.where(image >= 0, image, np.nan))
        elif tf_name == 'Squared':
            image = image ** 2
        elif tf_name == 'Asinh':
            scale_factor = np.nanpercentile(image, 99) or 1.0
            image = np.arcsinh(image / scale_factor)
        # Linear: no transform

        # ── Apply stretch (clip level) ────────────────────────────────
        st = getattr(self, 'stretch_combo', None)
        st_name = st.currentText() if st else 'minmax'

        if st_name == 'manual':
            vmin = getattr(self, '_manual_vmin', np.nanmin(image))
            vmax = getattr(self, '_manual_vmax', np.nanmax(image))
        else:
            vmin, vmax = self._stretch_limits(image, st_name)

        # Preserve the current viewport across a re-render of the same field so
        # that changing scale/stretch/colormap (or reloading the background)
        # doesn't snap the zoom back to 100%. Only restore when the new image
        # has the same dimensions — a fresh load or a 90° rotation changes the
        # shape and should reset to the default fit.
        prev_xlim = prev_ylim = prev_shape = None
        if getattr(self, 'ax', None) is not None and self.ax.images:
            _bkg = getattr(self, '_bkg_artist', None)
            _cube_imgs = [im for im in self.ax.images if im is not _bkg]
            if _cube_imgs:
                prev_xlim = self.ax.get_xlim()
                prev_ylim = self.ax.get_ylim()
                prev_shape = _cube_imgs[0].get_array().shape[:2]

        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        apply_mpl_qss_style(self.canvas.figure, self.ax, None)
        self.ax.grid(False)  # No grid on the cube image

        # Restore the saved viewport before any overlays are redrawn so the
        # background-overlay restore picks up the correct (zoomed) limits.
        if prev_xlim is not None and prev_shape == image.shape[:2]:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)

        # Initialize red rectangle but keep it hidden initially
        self.red_rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='#d7801a', facecolor='none', visible=False)
        self.ax.add_patch(self.red_rect)
        self._blue_rect = patches.Rectangle((0, 0), 1, 1, linewidth=2.0, edgecolor='#4fc3f7', facecolor='none', visible=False)
        self.ax.add_patch(self._blue_rect)
        # Restore blue rect position if an init-guess spaxel was previously set
        if getattr(self, '_init_guess_spaxel', None) is not None:
            _ix, _iy = self._init_guess_spaxel
            self._blue_rect.set_xy((_ix - 0.5, _iy - 0.5))
            self._blue_rect.set_visible(True)

        # Overlay text for spaxel info (top-left corner of the image axes)
        self.spaxel_info_text = self.ax.text(
            0.01, 0.99, '',
            transform=self.ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none')
        )
        # Restore background image overlay if one was loaded
        if getattr(self, '_bkg_data', None) is not None:
            self._bkg_artist = None  # force redraw since axes were cleared
            self._draw_bkg_overlay()

        self.canvas.draw()





    def redraw_current_image(self):
        """Re-render the currently-displayed field (white-light or a map) using
        the last draw_image arguments. Used after the spatial mask changes."""
        if getattr(self, '_last_data', None) is None:
            return
        self.draw_image(self._last_data,
                        cmap=getattr(self, '_last_cmap', 'gray'),
                        scale=getattr(self, '_last_scale', 'linear'),
                        from_fits=getattr(self, '_last_from_fits', False))

    def show_mask_preview(self, keep_field):
        """Outline the kept region as a contour on the cube image (mirrors the
        SNR-mask contour). keep_field is a 2D array, 1 where kept, 0 elsewhere."""
        self.clear_mask_preview()
        if keep_field is None or getattr(self, 'ax', None) is None:
            return
        try:
            self._mask_preview_cs = self.ax.contour(
                keep_field, levels=[0.5], colors='#4fc3f7', linewidths=1.5)
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Mask preview contour failed: {type(e).__name__}: {e}")

    def clear_mask_preview(self):
        """Remove any mask-preview contour drawn by show_mask_preview."""
        cs = getattr(self, '_mask_preview_cs', None)
        if cs is not None:
            try:
                cs.remove()  # matplotlib ≥3.8: ContourSet is itself an Artist
            except Exception:
                try:
                    for coll in cs.collections:  # older matplotlib
                        coll.remove()
                except Exception:
                    pass
            self._mask_preview_cs = None
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

    def get_spectrum_at_spaxel(self, x, y):
        """Fetch the 1D spectrum corresponding to the given spaxel (x, y)."""
        x = int(np.clip(x, 0, FITS_DATA.shape[2] - 1))
        y = int(np.clip(y, 0, FITS_DATA.shape[1] - 1))
        spectrum = FITS_DATA[:, y, x]
        return spectrum


    def finalize_line(self):
        """Finalize the drawn line when the 'd' key is pressed again."""
        if self.drawing_line and self.line_start is not None:
            # Get the final position of the cursor in the spectrum plot
            end_x, end_y = self.spectrum_cursor_pos
            self.current_line.set_xdata([self.line_start[0], end_x])
            self.current_line.set_ydata([self.line_start[1], end_y])
            
            # Redraw the canvas to reflect the final line
            self.spectrum_canvas.draw()
    
    def update_line(self, x1, x2):
        """Update the line in the plot while keeping y1 and y2 fixed."""
        if hasattr(self, 'current_line') and hasattr(self, 'line_start') and hasattr(self, 'spectrum_cursor_pos'):
            y1 = self.line_start[1]  # Keep the original y1
            y2 = self.spectrum_cursor_pos[1]  # Keep the original y2
            
            # Recalculate the slope and intercept to maintain (y1, y2)
            if x1 != x2:  # Avoid division by zero
                self.slope = (y2 - y1) / (x2 - x1)
                self.intercept = y1 - self.slope * x1
            else:
                print("Warning: x1 and x2 are the same. Vertical line assumed.")
                self.slope = float('inf')  # Indicates a vertical line
                self.intercept = None  # Not defined for vertical lines
    
            self.current_line.set_data([x1, x2], [y1, y2])  # Update with new x1, x2
            self.spectrum_canvas.draw()  # Refresh the plot
            # print(f"Updated line: x1={x1}, y1={y1}, x2={x2}, y2={y2}, slope={self.slope}, intercept={self.intercept}")

    
    def update_line_with_slope_intercept(self, x1, x2, slope, intercept):
        """Update the line in the plot based on the updated slope and intercept."""
        if hasattr(self, 'current_line'):
            y1 = slope * x1 + intercept
            y2 = slope * x2 + intercept
            print(y1,y2)
            self.current_line.set_data([x1, x2], [y1, y2])
            self.spectrum_canvas.draw()  # Refresh the plot
            print(f"Redrawing line with updated slope={slope} and intercept={intercept}")
        
        # Store the slope and intercept for future updates
        self.slope = slope
        self.intercept = intercept






    def create_menu_bar_VisualizerWindow(self):
        """Creates and returns the menu bar for ViewerWindow"""
        menu_bar = QMenuBar(self)

        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Import (Cmd+I)", self)
        open_action.setShortcut(QKeySequence("Cmd+O" if sys.platform == "darwin" else "Ctrl+O"))
        save_action = QAction("Save (Cmd+S)", self)
        save_action.setShortcut(QKeySequence("Cmd+S" if sys.platform == "darwin" else "Ctrl+S"))
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        save_session_action = QAction("Save Session…", self)
        save_session_action.triggered.connect(self.save_session)
        load_session_action = QAction("Load Session…", self)
        load_session_action.triggered.connect(self.load_session)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(save_session_action)
        file_menu.addAction(load_session_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(QAction("Zoom In", self))
        view_menu.addAction(QAction("Zoom Out", self))

        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("About", self))

        return menu_bar

    def showEvent(self, event):
        """Ensure ViewerWindow always has its own menu bar when focused"""
        self.setMenuBar(self.create_menu_bar_VisualizerWindow())
        super().showEvent(event)


    def _ensure_fit_params_open_and_add_region(self):
        """Called after each interactive continuum draw (D-key).
        
        Guarantees:
        1. The Fit Parameters dock is created and visible (never accidentally closes it).
        2. The newly-added spectral region frame is always appended to the panel.
        """
        new_id = len(df_cont) - 1  # index of the region just added

        if self.fit_params_dock is None:
            # First ever open: create the dock. FitParamsWindow.__init__ will
            # call add_spectral_frame for region 0 via its own init logic.
            self.toggle_fit_params_dock()
            # __init__ only adds region 0; if somehow more regions exist, add them
            for extra_id in range(1, len(df_cont)):
                self.fit_params_window.add_spectral_frame(
                    f"Spectral Region {extra_id + 1}", df_cont, df, ID=extra_id, addframe=False
                )
        else:
            # Dock already exists — make sure it is visible (never hide it here)
            if not self.fit_params_dock.isVisible():
                self.fit_params_dock.show()
                self.resizeDocks([self.fit_params_dock], [self.width() // 2], Qt.Horizontal)
                self.open_fit_params_button.setText("Fit Params \u25c0")
            # Always add the new region frame explicitly
            self.fit_params_window.add_spectral_frame(
                f"Spectral Region {new_id + 1}", df_cont, df, ID=new_id, addframe=False
            )

    def toggle_fit_params_dock(self):
        """Toggle the Fit Parameters panel between docked and hidden states.
        
        First call: creates FitParamsWindow and docks it on the right side of
        ViewerWindow as a QDockWidget.  Subsequent calls show/hide the dock so
        the viewer layout is never obscured by a floating window.
        """
        # --- First-time creation ---
        if self.fit_params_dock is None:
            # Create the FitParamsWindow (still a QMainWindow; we just embed it)
            self.fit_params_window = FitParamsWindow(
                viewer_window=self, df=df, df_cont=df_cont, current_line=''
            )

            # Wrap it in a QDockWidget
            self.fit_params_dock = QDockWidget("Fit Parameters", self)
            self.fit_params_dock.setObjectName("FitParamsDock")
            self.fit_params_dock.setAllowedAreas(
                Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
            )
            # Embed the central widget of FitParamsWindow (the scroll area etc.)
            self.fit_params_dock.setWidget(self.fit_params_window.centralWidget())
            self.fit_params_dock.setFeatures(
                QDockWidget.DockWidgetMovable |
                QDockWidget.DockWidgetFloatable |
                QDockWidget.DockWidgetClosable
            )
            # Keep the button label in sync when the user closes the dock via its X
            self.fit_params_dock.visibilityChanged.connect(self._on_dock_visibility_changed)

            self.addDockWidget(Qt.BottomDockWidgetArea, self.fit_params_dock)
            # Size the dock to ~40% of the window height
            self.resizeDocks([self.fit_params_dock], [self.height() * 2 // 5], Qt.Vertical)
            self.open_fit_params_button.setText("▲  Fit Parameters")
            return

        # --- Toggle visibility on subsequent calls ---
        if self.fit_params_dock.isVisible():
            self.fit_params_dock.hide()
            self.open_fit_params_button.setText("▼  Fit Parameters")
        else:
            self.fit_params_dock.show()
            # Re-apply height whenever the dock is shown again
            self.resizeDocks([self.fit_params_dock], [self.height() * 2 // 5], Qt.Vertical)
            self.open_fit_params_button.setText("▲  Fit Parameters")

    def _on_dock_visibility_changed(self, visible):
        """Keep the toggle button label in sync when dock is closed via its own X button."""
        if visible:
            self.open_fit_params_button.setText("▲  Fit Parameters")
        else:
            self.open_fit_params_button.setText("▼  Fit Parameters")

    def open_fit_params_window(self):
        """Opens the FitParamsWindow when the button is clicked.
        
        Kept for backwards-compatibility (e.g. calls from other code paths).
        Delegates to the dock toggle so behaviour is consistent.
        """
        self.toggle_fit_params_dock()

    def closeEvent(self, event):
        """Override closeEvent to ensure the console is still active after closing the window"""
        app = QApplication.instance()
        if app:
            app.quit()  # Quit the QApplication when the window is closed
        event.accept()  # Accept the event and close the window
        

class FitParamsWindow(QtWidgets.QMainWindow):
    buttons_dict = {}

    def __init__(self, viewer_window, df_cont, df, current_line):
        super().__init__()
        self.viewer_window = viewer_window
        df_cont = df_cont
        df = df
        self.fitloaded = False
        self._sequential_fit = False   # sequential core→outflow fit mode (off by default)
        self.current_line = current_line
        self.setWindowTitle("Fit Parameters")
        
        # self.create_menu_bar_FitParamsWindow()  # Add menu bar
        # Assign the menu bar explicitly to this window
        self.setMenuBar(self.create_menu_bar_FitParamsWindow())

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Scroll Area
        self.scroll_area = QScrollArea(self.main_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area)

        # ── Embedded fit-progress bar ──────────────────────────────────────
        # Shown in-panel (at the bottom of the Fit Parameters dock) while a
        # Fit Cube / Rectify run is in progress, instead of a separate popup.
        self.fit_progress_frame = QFrame()
        self.fit_progress_frame.setObjectName('fit_progress_frame')
        self.fit_progress_frame.setFrameShape(QFrame.StyledPanel)
        _pf = QHBoxLayout(self.fit_progress_frame)
        _pf.setContentsMargins(8, 3, 8, 3)
        _pf.setSpacing(8)
        self.fit_progress_label = QLabel('')
        self.fit_progress_label.setStyleSheet('font-weight: bold;')
        self.fit_progress_bar = QProgressBar()
        self.fit_progress_bar.setTextVisible(True)
        self.fit_progress_bar.setFixedHeight(18)
        self.fit_cancel_button = QPushButton('Cancel')
        self.fit_cancel_button.setFixedHeight(20)
        self.fit_cancel_button.setToolTip('Stop the in-progress cube fit')
        self.fit_cancel_button.clicked.connect(self._cancel_running_fit)
        _pf.addWidget(self.fit_progress_label)
        _pf.addWidget(self.fit_progress_bar, 1)
        _pf.addWidget(self.fit_cancel_button)
        self.fit_progress_frame.setVisible(False)
        self._fit_cancelled = False
        layout.addWidget(self.fit_progress_frame)

        # Container for frames inside the scroll area
        self.scroll_container = QWidget()
        # Vertical layout: obs toolbar at top, then one spectral region card per row
        self.outer_scroll_layout = QVBoxLayout(self.scroll_container)
        self.outer_scroll_layout.setSpacing(8)
        self.outer_scroll_layout.setContentsMargins(4, 4, 4, 4)
        # scroll_layout alias points to same layout for compatibility
        self.scroll_layout = self.outer_scroll_layout
        self.scroll_container.setLayout(self.outer_scroll_layout)
        self.scroll_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.scroll_area.setWidget(self.scroll_container)

        self._build_obs_and_fit_toolbar(df_obs)
        
        if len(df_cont) == 0:
            self.spectral_count = 0
        else:
            self.spectral_count = 1
            self.add_spectral_frame("Spectral Region 1", df_cont, df, ID=0, addframe=False)

        self.setGeometry(100, 100, 1200, 400)
        # Note: visibility is now managed by the QDockWidget in ViewerWindow.

    def create_menu_bar_FitParamsWindow(self):
        """Creates the menu bar for this window"""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)  # Assign it to the window

        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open (Cmd+O)", self)
        open_action.setShortcut(QKeySequence("Cmd+O" if sys.platform == "darwin" else "Ctrl+O"))
        save_action = QAction("Save (Cmd+S)", self)
        save_action.setShortcut(QKeySequence("Cmd+S" if sys.platform == "darwin" else "Ctrl+S"))
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Edit Menu
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(QAction("Undo", self))
        edit_menu.addAction(QAction("Redo", self))

        # View Menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(QAction("Zoom In", self))
        view_menu.addAction(QAction("Zoom Out", self))

        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("About", self))

        return menu_bar  # Return the menu bar

    def _add_spline_continuum_buttons(self, regionID, grid_layout):
        """Compact panel display for a spline continuum region: name, type,
        x-range, knot count, an Edit-knots button, and the delete button."""
        reg = df_cont.loc[df_cont['region_ID'] == regionID]
        if len(reg) == 0:
            return
        r = reg.iloc[0]
        kx = _as_float_list(r['knots_x'])
        name = str(r['Continuum Name'])

        labels = ['Continuum Name', 'Type', 'x1', 'x2', '# Knots', '']
        for c, lbl in enumerate(labels):
            grid_layout.addWidget(QLabel(lbl), 0, c)

        # Editable name (same behaviour as linear regions)
        name_btn = FrameButton(name, 0, 0, regionID, 'continuum~Continuum Name')
        name_btn.clicked.connect(partial(self.on_button_click, data_frame=df_cont,
                                         frame_id=regionID, button_name='continuum~Continuum Name'))
        grid_layout.addWidget(name_btn, 1, 0)
        self.buttons_dict[(regionID, 'continuum~Continuum Name')] = name_btn

        # Read-only display cells
        type_btn = FrameButton('spline', 0, 1, regionID, 'continuum~cont_type')
        type_btn.setDisabled(True)
        grid_layout.addWidget(type_btn, 1, 1)

        x1_btn = FrameButton(_fmt(min(kx)) if kx else '—', 0, 2, regionID, 'continuum~x1')
        x1_btn.setDisabled(True)
        grid_layout.addWidget(x1_btn, 1, 2)
        self.buttons_dict[(regionID, 'continuum~x1')] = x1_btn

        x2_btn = FrameButton(_fmt(max(kx)) if kx else '—', 0, 3, regionID, 'continuum~x2')
        x2_btn.setDisabled(True)
        grid_layout.addWidget(x2_btn, 1, 3)
        self.buttons_dict[(regionID, 'continuum~x2')] = x2_btn

        nk_btn = FrameButton(str(len(kx)), 0, 4, regionID, 'continuum~nknots')
        nk_btn.setDisabled(True)
        grid_layout.addWidget(nk_btn, 1, 4)
        self.buttons_dict[(regionID, 'continuum~nknots')] = nk_btn

        edit_btn = QPushButton('Edit knots…')
        edit_btn.clicked.connect(partial(self._edit_spline_knots, regionID))
        grid_layout.addWidget(edit_btn, 1, 5)

        del_btn = FrameButton('x', 0, 6, regionID, 'continuum_delete')
        del_btn.setStyleSheet("background-color: lightcoral; color: black;")
        del_btn.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=regionID))
        grid_layout.addWidget(del_btn, 1, 6)
        self.buttons_dict[(regionID, 'continuum_delete')] = del_btn

    def _add_poly_continuum_buttons(self, regionID, grid_layout):
        """Compact panel display for a polynomial (Chebyshev) continuum region:
        name, type, x-range, degree, an Edit-degree button, and delete."""
        reg = df_cont.loc[df_cont['region_ID'] == regionID]
        if len(reg) == 0:
            return
        r = reg.iloc[0]
        deg = r['poly_degree']
        deg_txt = str(int(deg)) if pd.notna(deg) else '—'

        labels = ['Continuum Name', 'Type', 'x1', 'x2', 'Degree', '']
        for c, lbl in enumerate(labels):
            grid_layout.addWidget(QLabel(lbl), 0, c)

        name_btn = FrameButton(str(r['Continuum Name']), 0, 0, regionID, 'continuum~Continuum Name')
        name_btn.clicked.connect(partial(self.on_button_click, data_frame=df_cont,
                                         frame_id=regionID, button_name='continuum~Continuum Name'))
        grid_layout.addWidget(name_btn, 1, 0)
        self.buttons_dict[(regionID, 'continuum~Continuum Name')] = name_btn

        for c, (txt, key) in enumerate([
                ('poly', 'continuum~cont_type'),
                (_fmt(r['x1']), 'continuum~x1'),
                (_fmt(r['x2']), 'continuum~x2'),
                (deg_txt, 'continuum~poly_degree')], start=1):
            b = FrameButton(txt, 0, c, regionID, key); b.setDisabled(True)
            grid_layout.addWidget(b, 1, c)
            self.buttons_dict[(regionID, key)] = b

        edit_btn = QPushButton('Edit degree…')
        edit_btn.clicked.connect(partial(self._edit_poly_degree, regionID))
        grid_layout.addWidget(edit_btn, 1, 5)

        del_btn = FrameButton('x', 0, 6, regionID, 'continuum_delete')
        del_btn.setStyleSheet("background-color: lightcoral; color: black;")
        del_btn.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=regionID))
        grid_layout.addWidget(del_btn, 1, 6)
        self.buttons_dict[(regionID, 'continuum_delete')] = del_btn

    def _add_stellar_continuum_buttons(self, regionID, grid_layout):
        """Compact panel for a stellar (pPXF) continuum region: name, type,
        library, editable V / σ / scale (+ h3/h4), a Refit Stellar button, delete."""
        reg = df_cont.loc[df_cont['region_ID'] == regionID]
        if len(reg) == 0:
            return
        r = reg.iloc[0]
        moments = int(r['stellar_moments']) if pd.notna(r.get('stellar_moments')) else 2
        lib = str(r.get('stellar_library', '') or '')

        cells = [('Name', 'continuum~Continuum Name', str(r['Continuum Name']), True),
                 ('Type', 'continuum~cont_type', 'stellar', False),
                 ('Library', 'continuum~stellar_library', lib, False),
                 ('V (km/s)', 'stellar~stellar_V_0', _fmt(r.get('stellar_V_0')), True),
                 ('σ (km/s)', 'stellar~stellar_sigma_0', _fmt(r.get('stellar_sigma_0')), True),
                 ('scale', 'stellar~stellar_scale_0', _fmt(r.get('stellar_scale_0')), True)]
        if moments >= 4:
            cells += [('h3', 'stellar~stellar_h3_0', _fmt(r.get('stellar_h3_0')), True),
                      ('h4', 'stellar~stellar_h4_0', _fmt(r.get('stellar_h4_0')), True)]

        for c, (lbl, key, txt, editable) in enumerate(cells):
            grid_layout.addWidget(QLabel(lbl), 0, c)
            b = FrameButton(txt, 0, c, regionID, key)
            if editable:
                b.clicked.connect(partial(self.on_button_click, data_frame=df_cont,
                                          frame_id=regionID, button_name=key))
            else:
                b.setDisabled(True)
            grid_layout.addWidget(b, 1, c)
            self.buttons_dict[(regionID, key)] = b

        ncol = len(cells)
        refit = QPushButton('Refit Stellar')
        refit.setToolTip('Re-run pPXF for the current spaxel')
        refit.clicked.connect(partial(self._refit_stellar_region, regionID))
        grid_layout.addWidget(refit, 1, ncol)

        fitcube = QPushButton('Fit Stellar (Cube)')
        fitcube.setToolTip('Run pPXF on every SNR-gated spaxel → stellar V/σ maps')
        fitcube.clicked.connect(self.fit_stellar_cube)
        grid_layout.addWidget(fitcube, 1, ncol + 1)

        # V/σ map buttons: header label above, current-spaxel best-fit value on
        # the button face; clicking displays the corresponding cube map.
        cur_V, cur_sig = self._current_stellar_kin(regionID)
        grid_layout.addWidget(QLabel('V map'), 0, ncol + 2)
        vmap = QPushButton(_fmt(cur_V))
        vmap.setToolTip('Show this region\'s stellar velocity map (current spaxel value shown)')
        vmap.clicked.connect(partial(self._stellar_map, 'stellar_V', regionID))
        grid_layout.addWidget(vmap, 1, ncol + 2)
        self.buttons_dict[(regionID, 'stellar~vmap')] = vmap
        grid_layout.addWidget(QLabel('σ map'), 0, ncol + 3)
        smap = QPushButton(_fmt(cur_sig))
        smap.setToolTip('Show this region\'s stellar dispersion map (current spaxel value shown)')
        smap.clicked.connect(partial(self._stellar_map, 'stellar_sigma', regionID))
        grid_layout.addWidget(smap, 1, ncol + 3)
        self.buttons_dict[(regionID, 'stellar~smap')] = smap

        del_btn = FrameButton('x', 0, ncol + 4, regionID, 'continuum_delete')
        del_btn.setStyleSheet("background-color: lightcoral; color: black;")
        if self._edit_mode():
            del_btn.setEnabled(False)
            del_btn.setToolTip('Disabled while editing a single spaxel (schema is locked)')
        else:
            del_btn.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=regionID))
        grid_layout.addWidget(del_btn, 1, ncol + 4)
        self.buttons_dict[(regionID, 'continuum_delete')] = del_btn

    def _edit_poly_degree(self, regionID):
        """Dialog to change a polynomial region's degree; re-seeds the Chebyshev
        coefficients by re-fitting the displayed spectrum over the region range."""
        global df_cont
        reg = df_cont.loc[df_cont['region_ID'] == regionID]
        if len(reg) == 0:
            return
        idx = reg.index[0]
        x1 = float(reg.iloc[0]['x1']); x2 = float(reg.iloc[0]['x2'])
        cur_deg = int(reg.iloc[0]['poly_degree']) if pd.notna(reg.iloc[0]['poly_degree']) else 3

        sl = getattr(self.viewer_window, 'spectrum_line', None)
        npts = 0
        if sl is not None:
            xx = np.asarray(sl.get_xdata(), float); yy = np.asarray(sl.get_ydata(), float)
            npts = int(np.count_nonzero((xx >= x1) & (xx <= x2) & np.isfinite(yy)))
        max_deg = max(1, min(7, npts - 1)) if npts >= 2 else 7

        dlg = QDialog(self)
        dlg.setWindowTitle(f'Polynomial degree — region {regionID + 1}')
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f'Range {x1:.2f} – {x2:.2f}   ({npts} pixels)'))
        rowl = QHBoxLayout(); rowl.addWidget(QLabel('Chebyshev degree:'))
        spin = QtWidgets.QSpinBox(); spin.setRange(1, max_deg); spin.setValue(min(cur_deg, max_deg))
        rowl.addWidget(spin); rowl.addStretch(); v.addLayout(rowl)
        okrow = QHBoxLayout(); apply_btn = QPushButton('Apply'); cancel_btn = QPushButton('Cancel')
        okrow.addStretch(); okrow.addWidget(cancel_btn); okrow.addWidget(apply_btn); v.addLayout(okrow)
        cancel_btn.clicked.connect(dlg.reject)

        def _apply():
            deg = int(spin.value())
            if sl is None:
                dlg.reject(); return
            xx = np.asarray(sl.get_xdata(), float); yy = np.asarray(sl.get_ydata(), float)
            mm = (xx >= x1) & (xx <= x2) & np.isfinite(yy)
            if np.count_nonzero(mm) < deg + 1:
                QMessageBox.warning(dlg, 'Degree', 'Not enough points in range for this degree.')
                return
            try:
                cheb = np.polynomial.Chebyshev.fit(xx[mm], yy[mm], deg, domain=[x1, x2])
                coefs = [float(c) for c in cheb.coef]
            except Exception as e:
                QMessageBox.warning(dlg, 'Degree', f'Fit failed: {e}')
                return
            df_cont.at[idx, 'poly_degree'] = deg
            df_cont.at[idx, 'poly_coef_0'] = coefs
            df_cont.at[idx, 'poly_coef_fit'] = []   # stale until next fit
            self.viewer_window._redraw_poly_region(regionID)
            if (regionID, 'continuum~poly_degree') in self.buttons_dict:
                self.buttons_dict[(regionID, 'continuum~poly_degree')].setText(str(deg))
            dlg.accept()
        apply_btn.clicked.connect(_apply)
        dlg.exec_()

    def _edit_spline_knots(self, regionID):
        """Dialog to view/edit the (x, y) knots of a spline continuum region."""
        global df_cont
        reg = df_cont.loc[df_cont['region_ID'] == regionID]
        if len(reg) == 0:
            return
        idx = reg.index[0]
        kx = _as_float_list(reg.iloc[0]['knots_x'])
        ky = _as_float_list(reg.iloc[0]['knots_y_0'])

        dlg = QDialog(self)
        dlg.setWindowTitle(f'Edit spline knots — region {regionID + 1}')
        dlg.setMinimumWidth(320)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel('Knot positions (wavelength x, flux y). '
                           'Initial guess for the fit.'))

        table = QtWidgets.QTableWidget(len(kx), 2)
        table.setHorizontalHeaderLabels(['x (wavelength)', 'y (flux)'])
        table.horizontalHeader().setStretchLastSection(True)
        for i, (xv, yv) in enumerate(zip(kx, ky)):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(repr(round(xv, 4))))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(repr(yv)))
        v.addWidget(table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton('+ Knot'); rm_btn = QPushButton('− Selected')
        btn_row.addWidget(add_btn); btn_row.addWidget(rm_btn); btn_row.addStretch()
        v.addLayout(btn_row)

        def _add_row():
            r = table.rowCount(); table.insertRow(r)
            table.setItem(r, 0, QtWidgets.QTableWidgetItem('0'))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem('0'))
        def _rm_row():
            r = table.currentRow()
            if r >= 0:
                table.removeRow(r)
        add_btn.clicked.connect(_add_row)
        rm_btn.clicked.connect(_rm_row)

        ok_row = QHBoxLayout()
        apply_btn = QPushButton('Apply'); cancel_btn = QPushButton('Cancel')
        ok_row.addStretch(); ok_row.addWidget(cancel_btn); ok_row.addWidget(apply_btn)
        v.addLayout(ok_row)
        cancel_btn.clicked.connect(dlg.reject)

        def _apply():
            nkx, nky = [], []
            for r in range(table.rowCount()):
                try:
                    xv = float(table.item(r, 0).text())
                    yv = float(table.item(r, 1).text())
                except (TypeError, ValueError):
                    continue
                nkx.append(xv); nky.append(yv)
            if len(nkx) < 2:
                QMessageBox.warning(dlg, 'Edit knots', 'A spline needs at least 2 valid knots.')
                return
            order = np.argsort(nkx)
            nkx = [round(float(nkx[i]), 4) for i in order]
            nky = [float(nky[i]) for i in order]
            df_cont.at[idx, 'knots_x'] = nkx
            df_cont.at[idx, 'knots_y_0'] = nky
            df_cont.at[idx, 'x1'] = nkx[0]
            df_cont.at[idx, 'x2'] = nkx[-1]
            # Refresh the on-plot spline and the panel display cells
            self.viewer_window._redraw_spline_region(regionID)
            b = self.buttons_dict
            if (regionID, 'continuum~x1') in b: b[(regionID, 'continuum~x1')].setText(_fmt(nkx[0]))
            if (regionID, 'continuum~x2') in b: b[(regionID, 'continuum~x2')].setText(_fmt(nkx[-1]))
            if (regionID, 'continuum~nknots') in b: b[(regionID, 'continuum~nknots')].setText(str(len(nkx)))
            dlg.accept()
        apply_btn.clicked.connect(_apply)
        dlg.exec_()

    def add_continuum_buttons(self, regionID, grid_layout):
        # Spline regions get a dedicated, compact display (type, x-range, knot
        # count, Edit knots…). Linear regions use the original column grid.
        _reg = df_cont.loc[df_cont['region_ID'] == regionID]
        _rtype = str(_reg.iloc[0]['cont_type']) if (len(_reg) > 0 and 'cont_type' in df_cont.columns) else 'linear'
        if _rtype == 'spline':
            self._add_spline_continuum_buttons(regionID, grid_layout)
            return
        if _rtype == 'poly':
            self._add_poly_continuum_buttons(regionID, grid_layout)
            return
        if _rtype == 'stellar':
            self._add_stellar_continuum_buttons(regionID, grid_layout)
            return

        # Columns not rendered as buttons here (region_ID/lineactor and the
        # spline/poly-only columns, which never appear for linear regions).
        _hidden_cont_cols = {'region_ID', 'lineactor',
                             'cont_type', 'knots_x', 'knots_y_0', 'knots_y_fit',
                             'poly_degree', 'poly_coef_0', 'poly_coef_fit'}
        # Add Labels (continuum button labels)
        for col, col_name in enumerate(df_cont.columns):
            if col_name not in _hidden_cont_cols:
                label = QLabel(col_name)
                # label.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 12pt;
                #     border: none;
                #     padding-right: 10px;
                #     padding-left: 10px;
                #     padding-top: 5px;
                #     padding-bottom: 5px;
                #     background-color: snow;
                #     color: black;
                #     font: bold;
                #     width: 64px;
                # """)
                grid_layout.addWidget(label, 0, col)
            
        # Add Data (continuum buttons)
        _rid_col = df_cont.columns.get_loc('region_ID')  # look up by name, not a fixed index
        for row in range(df_cont.shape[0]):
           for col, col_name in enumerate(df_cont.columns):
               if col_name not in _hidden_cont_cols:
                   if np.int64(df_cont.iloc[row, _rid_col]) == np.int64(regionID):
                       button_name = 'continuum~'+col_name
                       # button_text = str(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       if (col_name == 'Continuum Name') or (type(df_cont.iloc[row, col]) == str):
                             button_text = str(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       else:
                             button_text = _fmt(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       button = FrameButton(button_text, row, col, regionID, button_name)
                       # button.setStyleSheet("""
                       #     font-family: "Segoe UI";
                       #     font-size: 10pt;
                       #     border: 2px solid;
                       #     border-color: steelblue;
                       #     border-radius: 5px;
                       #     padding-right: 10px;
                       #     padding-left: 10px;
                       #     padding-top: 5px;
                       #     padding-bottom: 5px;
                       #     background-color: lightskyblue;
                       #     highlight-color; cyan;
                       #     color: k;
                       #     font: bold;
                       #     width: 64px;
                       # """)
                       
                       if col >= 5:  # Check the 'fit' column in df_cont
                           # button.setStyleSheet("""
                           #     font-family: "Segoe UI";
                           #     font-size: 12pt;
                           #     border: 2px solid;
                           #     border-color: black;
                           #     border-radius: 5px;
                           #     padding-right: 10px;
                           #     padding-left: 10px;
                           #     padding-top: 5px;
                           #     padding-bottom: 5px;
                           #     background-color: darkgrey;
                           #     color: white;
                           #     font: bold;
                           #     width: 64px;
                           # """)
                           button.setDisabled(True)  # Disable the button
                       else:
                           button.clicked.connect(partial(self.on_button_click, data_frame=df_cont, frame_id=regionID, button_name=button_name))
                       
                       grid_layout.addWidget(button, 1, col)
       
                       # Store the button reference in buttons_dict by row/col
                       self.buttons_dict[(regionID, button_name)] = button
                       
                       if col == 6:
                            button_name = 'continuum_delete'
                            delete_region_button = FrameButton('x', row, col, regionID, button_name)
                            delete_region_button.setStyleSheet("""
                                background-color: lightcoral; color: black;
                                color: k;
                            """)
                            if self._edit_mode():
                                delete_region_button.setEnabled(False)
                                delete_region_button.setToolTip('Disabled while editing a single spaxel (schema is locked)')
                            else:
                                delete_region_button.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=regionID))

                            grid_layout.addWidget(delete_region_button, 1, col+1)
                            # Store the button reference in buttons_dict by row/col
                            self.buttons_dict[(regionID, button_name)] = delete_region_button
                       
                       
    def add_spectral_lines_button_header(self, grid_layout):
        button_columns = [
            'Line Name', 
            'SNR',
            '<i>&lambda;</i><sub>rest</sub>', 
            '<i>f</i><sub>&lambda;,0</sub>', 
            '<i>f</i><sub>&lambda;,0,min</sub>', 
            '<i>f</i><sub>&lambda;,0,max</sub>', 
            '<i>&lambda;</i><sub>obs,0</sub>', 
            '<i>&lambda;</i><sub>obs,0,min</sub>',
            '<i>&lambda;</i><sub>obs,0,max</sub>',
            '&sigma;<sub>0</sub> [km/s]',
            '&sigma;<sub>0,min</sub> [km/s]',
            '&sigma;<sub>0,max</sub> [km/s]',
            '<i>f</i><sub>&lambda;,fit</sub>',
            '<i>&lambda;</i><sub>obs,fit</sub>',
            'v<sub>fit</sub>',
            '&sigma;<sub>fit</sub> [km/s]'
        ]
    
        for col, col_name in enumerate(button_columns):
            label = QLabel(col_name)
            label.setTextFormat(Qt.RichText)  # Enable HTML formatting
            if col == 0:
                label.setStyleSheet("""
                    font-size: 12pt;
                """)
            else:
                label.setStyleSheet("""
                    font-size: 12pt;
                    padding-left: 10px;
                """)
            # label.setStyleSheet("""
            #     font-family: "Segoe UI";
            #     font-size: 12pt;
            #     border: none;
            #     padding-right: 10px;
            #     padding-left: 10px;
            #     padding-top: 5px;
            #     padding-bottom: 5px;
            #     background-color: snow;
            #     color: black;
            #     font: bold;
            #     width: 64px;
            # """)
            grid_layout.addWidget(label, 2, col)



    def add_spectral_lines(self, regionID, grid_layout):
        global df_cont, df
        # Add Data (spectral line buttons)
        df_region = df.loc[np.int64(df['region_ID']) == np.int64(regionID)]
        
        for row in range(df_region.shape[0]):
            button_columns = ['Line_Name', 'SNR','Rest Wavelength', 
                              'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                              'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                              'Sigma_0','Sigma_0_lowlim','Sigma_0_highlim',
                              'Amp_fit', 'Centroid_fit', 'v_fit', 'Sigma_fit']
            
            main_row_index = 3 + row * 2  # Reserve extra row space
            limits_row_index = main_row_index + 1  # Place limit buttons in a separate row
            
            button_height = 24  # Standardized button height
            
            for col, col_name in enumerate(button_columns):
                # print(type(df_region.iloc[row][col_name]),df_region.iloc[row][col_name])
                button_name = str(np.int64(df_region.iloc[row]['Line_ID']))+'~'+str(col_name)
                if col_name in ('Sigma_0', 'Sigma_0_lowlim', 'Sigma_0_highlim'):
                    # σ stored in Å; displayed as velocity dispersion (km/s)
                    button_text = _fmt(sigma_wl_to_kms(df_region.iloc[row][col_name],
                                                       df_region.iloc[row]['Centroid_0']))
                elif col_name in ['Rest Wavelength', 'Amp_0', 'Centroid_0']:
                    button_text = _fmt(df_region.iloc[row][col_name])
                else:

                    if ('fit' in col_name.lower()) | ('SNR' in col_name) | ('Rest' in col_name):
                        button_text = ''
                    else:
                        button_text = str(df_region.iloc[row][col_name])
                        
                # button_text = str(df_region.iloc[row][col_name]) if 'fit' not in col_name.lower() else ''
                button = FrameButton(button_text, row, col, regionID, button_name)
                # button.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 8pt;
                #     border: 2px solid;
                #     border-color: steelblue;
                #     border-radius: 5px;
                #     padding: 5px 10px;
                #     background-color: lightskyblue;
                #     color: black;
                #     font: bold;
                # """)
                if 'fit' in col_name:  # Check the 'fit' column
                    # button.setStyleSheet("""
                    #     font-family: "Segoe UI";
                    #     font-size: 12pt;
                    #     border: 2px solid;
                    #     border-color: black;
                    #     border-radius: 5px;
                    #     padding: 5px 10px;
                    #     background-color: darkgrey;
                    #     color: white;
                    #     font: bold;
                    # """)
                    # button.setDisabled(True)  # Disable the button
                    button.clicked.connect(partial(self.on_fit_button_click, frame_id=regionID, button_name=button_name))
                else:
                    button.clicked.connect(partial(self.on_button_click, data_frame=df, frame_id=regionID, button_name=button_name))

                # Colored border matching this line's component overlay color
                if col_name == 'Line_Name':
                    _clr = self._line_overlay_color(df_region.iloc[row]['Line_ID'])
                    if _clr:
                        button.setStyleSheet(
                            f"border: 2px solid {_clr}; border-radius: 4px;")

                grid_layout.addWidget(button, main_row_index, col)
                
                # Store button reference
                self.buttons_dict[(regionID, button_name)] = button
                
                line_id = np.float64(df_region.iloc[row]['Line_ID'])
                line_id = np.int64(line_id)

            # Delete button at the rightmost position of each line row.
            # Disabled in edit mode to preserve the locked schema (same lines
            # everywhere) so line maps stay hole-free.
            del_line_btn = FrameButton('x', row, len(button_columns), regionID, f'del_line_{line_id}')
            del_line_btn.setFixedHeight(24)
            del_line_btn.setStyleSheet("background-color: lightcoral; color: black;")
            if self._edit_mode():
                del_line_btn.setEnabled(False)
                del_line_btn.setToolTip('Disabled while editing a single spaxel (schema is locked)')
            else:
                del_line_btn.clicked.connect(partial(self.on_deleteline_button_click, frame_id=regionID, line_id=line_id))
            grid_layout.addWidget(del_line_btn, main_row_index, len(button_columns))
            self.buttons_dict[(regionID, f'del_line_{line_id}')] = del_line_btn
                
    def on_fit_button_click(self, frame_id, button_name):
        print(f'frame ID: {frame_id}, button name: {button_name}')

        if len(df_fit) > 0:
            print(df_fit.iloc[0])
            gaussian_number = np.float64(button_name.split('~')[0])
            print(f'gaussian number = {gaussian_number}')
            # Filter df_fit for the specified line_name and region_ID
            filtered_df = df_fit[(df_fit['LineID'] == np.float64(gaussian_number)) & (df_fit['region_ID'] == frame_id)]

            # Convert spaxel_x and spaxel_y to integers for indexing
            filtered_df['spaxel_x'] = filtered_df['spaxel_x'].astype(int)
            filtered_df['spaxel_y'] = filtered_df['spaxel_y'].astype(int)
            
            # Always use the full cube spatial footprint so the map aligns with
            # the background image regardless of which spaxels passed the SNR cut.
            cube_ny = int(FITS_DATA.shape[-2])
            cube_nx = int(FITS_DATA.shape[-1])
            image_array = np.full((cube_ny, cube_nx), np.nan)
            
            
            
            
            # Mapping of button key parts to the column names in df_fit
            key_to_column_mapping = {
                'continuum~Slope_fit': 'slope_fit',
                'continuum~Intercept_fit': 'intercept_fit',
                'Line_Name': 'LineName',
                'Amp_fit': 'amp_fit',
                'Centroid_fit': 'cen_fit',
                'Sigma_fit': 'sigma_fit',
                'v_fit': 'vel_fit'
            }
            
            df_param = key_to_column_mapping[button_name.split('~')[1]]
            
            
            print(f'df_param: {df_param}')
            # Populate the image array with the selected parameter's values.
            # σ maps are shown as velocity dispersion (km/s) via each spaxel's
            # fitted centroid, matching the km/s sigma display elsewhere.
            for _, row in filtered_df.iterrows():
                if df_param == 'sigma_fit':
                    val = sigma_wl_to_kms(row['sigma_fit'], row['cen_fit'])
                else:
                    val = row[df_param]
                image_array[row['spaxel_y'], row['spaxel_x']] = val
                

            # print('shape of img array'+str(np.shape(image_array)))

            if df_param == 'amp_fit':
                cmap = 'viridis'
                # Use linear scale: log10 of ~1e-18 values collapses to NaN after clipping
                scale = 'linear'
            if df_param == 'cen_fit': 
                cmap='bwr'
                scale='linear'
            if df_param == 'vel_fit': 
                cmap='bwr'
                scale='linear'
            if df_param == 'sigma_fit': 
                cmap='plasma'
                scale='linear'
            
            # fig,ax=plt.subplots()
            # ax.imshow(image_array,origin='lower',aspect='auto',cmap=cmap)
            self.viewer_window.draw_image(image_array, cmap=cmap, scale=scale,from_fits=False)

    # Quality-map menu: (label, df_fit column, colormap, center-on-zero?)
    QUALITY_MAPS = [
        ('Core / continuum ratio', 'qa_core_cont_ratio', 'plasma', False),
        ('Signed residual (z)',    'qa_signed_resid_z',  'bwr',    True),
        ('Runs test (z)',          'qa_runs_z',          'bwr',    True),
        ('Reduced χ² (continuum)', 'qa_chisq_cont',      'plasma', False),
        # Weighted by the measurement errors over the fitted pixels only — the
        # one χ² that should sit near 1 for a good fit with a correct noise model.
        ('Reduced χ² (weighted)',  'rchisq_w',           'plasma', False),
        ('Reduced χ² (native)',    'rchisq',             'plasma', False),
    ]

    def show_quality_menu(self):
        """Pop a menu of fit-quality maps next to the Quality Map button."""
        menu = QMenu(self)
        for label, col, cmap, center in self.QUALITY_MAPS:
            act = menu.addAction(label)
            act.triggered.connect(
                lambda _checked=False, c=col, m=cmap, z=center, lb=label:
                self.show_quality_map(c, cmap=m, center_zero=z, label=lb))
        btn = getattr(self, 'rchisq_map_button', None)
        if btn is not None:
            menu.exec_(btn.mapToGlobal(btn.rect().bottomLeft()))
        else:
            menu.exec_()

    def show_quality_map(self, column, cmap='plasma', center_zero=False, label=''):
        """Render a per-spaxel goodness-of-fit map from a df_fit column."""
        from PyQt5.QtWidgets import QMessageBox
        if len(df_fit) == 0 or column not in df_fit.columns:
            QMessageBox.information(
                self, 'Fit Quality Map',
                f'No "{label or column}" values available.\n'
                'Fit the cube (or load a fit produced by this version) first.')
            return

        cube_ny = int(FITS_DATA.shape[-2])
        cube_nx = int(FITS_DATA.shape[-1])
        image_array = np.full((cube_ny, cube_nx), np.nan)

        # One value per spaxel (repeated across that spaxel's line rows).
        per_spaxel = df_fit.drop_duplicates(subset=['spaxel_x', 'spaxel_y'])
        for _, row in per_spaxel.iterrows():
            try:
                x, y = int(row['spaxel_x']), int(row['spaxel_y'])
            except (TypeError, ValueError):
                continue
            if 0 <= y < cube_ny and 0 <= x < cube_nx:
                image_array[y, x] = _safe_float(row.get(column))

        finite = image_array[np.isfinite(image_array)]
        if finite.size:
            if center_zero:
                # Symmetric scale around 0 for signed/z maps (clip at 98th pct |·|).
                vmax = np.nanpercentile(np.abs(finite), 98)
                if np.isfinite(vmax) and vmax > 0:
                    image_array = np.clip(image_array, -vmax, vmax)
            else:
                vmax = np.nanpercentile(finite, 98)
                if np.isfinite(vmax) and vmax > 0:
                    image_array = np.clip(image_array, None, vmax)

        self.viewer_window.draw_image(image_array, cmap=cmap, scale='linear',
                                      from_fits=False)
        self.viewer_window._quality_map_label = label or column

    def save_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
    
                # Convert region_ID to int64
                df_cont['region_ID'] = np.int64(df_cont['region_ID'])
                df['region_ID'] = np.int64(df['region_ID'])
    
                # Write df_obs (Observation parameters)
                writer.writerow(df_obs.columns)  # Write header
                for row in df_obs.itertuples(index=False, name=None):
                    writer.writerow(row)  # Flatten and write each row
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                writer.writerow(['wavelength scale factor', 'flux scale factor',
                                 'flux unit', 'wavelength unit', 'velocity unit'])

                writer.writerow([self.viewer_window.WLscalefactor,
                                 self.viewer_window.fluxscalefactor,
                                 flux_unit_str(), 'Angstrom', 'km/s'])
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                # Write df_cont (Continuum parameters)
                _cont_out = to_export_units(df_cont)
                writer.writerow(_cont_out.columns)  # Header for df_cont
                writer.writerows(_cont_out.values)  # Data for df_cont
    
                # Insert an empty row to separate the sections
                writer.writerow([])
    
                # Write df header (Emission line parameters); σ in km/s
                _df_out = to_export_units(df.sort_values(by='region_ID', ascending=True))
                writer.writerow(_df_out.columns)  # Header for df
                writer.writerows(_df_out.values)  # Data for df
    
            print(f"File saved: {file_path}")

    
    def open_file(self):
        global df, df_cont, df_obs
        """Open a CSV file and load it into the dataframes"""
        print('Opening file...')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        # Clear existing plot actors
        for frame_id in np.unique(df_cont['region_ID']):
            df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
    
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
    
                # Identify section indices (empty rows)
                empty_indices = [i for i, row in enumerate(rows) if not any(row)]
    
                if len(empty_indices) < 2:
                    print("Error: Invalid file format.")
                    return
    
                # Split the CSV data into three sections
                obs_data = rows[:empty_indices[0]]
                scale_data = rows[empty_indices[0]+1:empty_indices[1]]
                cont_data = rows[empty_indices[1]+1:empty_indices[2]]
                line_data = rows[empty_indices[2]+1:]
    
                # Load Observation Data (df_obs)
                df_obs_columns = obs_data[0]
                df_obs_values = obs_data[1:]
                df_obs = pd.DataFrame(df_obs_values, columns=df_obs_columns)
    
                # Ensure numeric columns are properly cast
                for col in ['redshift', 'resolvingpower']:
                    if col in df_obs.columns:
                        df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce')
    
                # Load Continuum Data (df_cont)
                df_cont_columns = cont_data[0]
                df_cont_values = cont_data[1:]
                df_cont = pd.DataFrame(df_cont_values, columns=df_cont_columns)
                df_cont = from_export_units(df_cont)
    
                # Convert data types for df_cont
                df_cont['x1'] = np.float64(df_cont['x1'])
                df_cont['x2'] = np.float64(df_cont['x2'])
                df_cont['Slope_0'] = np.float64(df_cont['Slope_0'])
                df_cont['Intercept_0'] = np.float64(df_cont['Intercept_0'])
                df_cont['Slope_fit'] = pd.to_numeric(df_cont['Slope_fit'], errors='coerce')
                df_cont['Intercept_fit'] = pd.to_numeric(df_cont['Intercept_fit'], errors='coerce')
                df_cont['region_ID'] = np.int64(df_cont['region_ID'])
    
                # Load Emission Line Data (df)
                df_columns = line_data[0]
                df_values = line_data[1:]
                df = pd.DataFrame(df_values, columns=df_columns)
                df = from_export_units(df)          # σ km/s → Å, strip unit suffixes
    
                # Convert numeric columns in df
                float_columns = [
                    'spaxel_x', 'spaxel_y', 'Line_ID', 'Rest Wavelength',
                    'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                    'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                    'Sigma_0', 'Sigma_0_lowlim', 'Sigma_0_highlim', 'region_ID'
                ]
                for col in float_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if col in ['Line_ID', 'region_ID']:
                            df[col] = np.int64(df[col])
    
                # Clear the existing layout
                for i in reversed(range(self.scroll_layout.count())):
                    widget = self.scroll_layout.itemAt(i).widget()
                    if isinstance(widget, QFrame):
                        self.scroll_layout.removeWidget(widget)
                        widget.deleteLater()
    
                # Populate the UI with the loaded data
                self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
                for ID in np.unique(df_cont['region_ID']):
                    ID = np.int64(ID)
                    self.add_spectral_frame(f'Spectral Region {ID+1}', df_cont, df, ID, addframe=False)
                    self.viewer_window.rebuild_plot(region_ID=ID, from_file=True, show_init=True, show_fit=False, x=0, y=0)
    
                print(f"File loaded: {file_path}")
    
            except Exception as e:
                print(f"Error opening file: {e}")


            
    def toggle_fullscreen(self):
        """Toggles full screen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_S and (event.modifiers() & Qt.ControlModifier):  # Check for Cmd/Ctrl + S
            self.save_file()  # Trigger save file dialog
        if event.key() == Qt.Key_O and (event.modifiers() & Qt.ControlModifier):  # Check for Cmd/Ctrl + S
            self.open_file()  # Trigger save file dialog
        # if event.key() == Qt.Key_D:  # If 'D' key is pressed
        #     self.spectral_count += 1
        #     ID = self.spectral_count - 1
        #     IDName = ID +1
        #     self.add_spectral_frame(f"Spectral Region {IDName}", df_cont, df, ID, addframe=True)
        #debugging
        if event.key() == Qt.Key_F:
            print(df)
            # print(self.buttons_dict)
            # print(df.iloc[0])
            # print(df_fit.iloc[0])
            
            # for col, val in df_cont.iloc[0].items():  # Assuming a single row
            #     print(f"{col}: {val}")

    def _build_obs_and_fit_toolbar(self, df_obs):
        """Build a compact single-row toolbar at the top of the scroll area.
        
        Left side: editable observation fields (Source, z, R).
        Right side: fit action buttons in a horizontal row.
        A '+ Spectral Region' button sits at the far right.
        Everything lives in a slim QFrame added to scroll_layout.
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName('obs_toolbar_frame')
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # Stack category rows vertically so buttons keep their natural width and
        # legible text instead of being squeezed into a single horizontal row.
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(8, 4, 8, 4)
        vbox.setSpacing(4)

        def _new_row():
            r = QHBoxLayout()
            r.setContentsMargins(0, 0, 0, 0)
            r.setSpacing(6)
            return r

        def _cat_label(text):
            lab = QLabel(text)
            lab.setStyleSheet('color: gray; font-size: 11px;')
            return lab

        def _vsep():
            s = QFrame(); s.setFrameShape(QFrame.VLine); s.setFrameShadow(QFrame.Sunken)
            return s

        # ── Row 1: Observation fields + per-spaxel actions ──────────────────
        row1 = _new_row()

        _name = df_obs.loc[0, 'sourcename']    if len(df_obs) > 0 else ''
        _z    = df_obs.loc[0, 'redshift']      if len(df_obs) > 0 else ''
        _rp   = df_obs.loc[0, 'resolvingpower'] if len(df_obs) > 0 else ''

        self.source_name_button    = SpaxelButton(f'Source: {_name}',  'Source Name')
        self.source_redshift_button= SpaxelButton(f'z: {_z}',          'Source Redshift')
        self.resolving_power_button= SpaxelButton(f'R: {_rp}',         'Resolving Power')

        row1.addWidget(_cat_label('Observation:'))
        for btn in [self.source_name_button,
                    self.source_redshift_button,
                    self.resolving_power_button]:
            btn.setFixedHeight(28)
            row1.addWidget(btn)

        self.source_name_button.clicked.connect(
            partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='sourcename'))
        self.source_redshift_button.clicked.connect(
            partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='redshift'))
        self.resolving_power_button.clicked.connect(
            partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='resolvingpower'))

        row1.addWidget(_vsep())

        # ── Per-spaxel fit buttons ──────────────────────────────────────────
        row1.addWidget(_cat_label('This Spaxel:'))

        self.fit_spaxel_button        = QPushButton('Fit This Spaxel')
        self.clear_spaxel_fit_button  = QPushButton('Clear Spaxel Fit')
        self.cancel_edit_button       = QPushButton('Cancel Edit')

        self.fit_spaxel_button.setToolTip('Fit the emission-line model to the currently selected spaxel')
        self.clear_spaxel_fit_button.setToolTip(
            'Remove this spaxel\'s fit and enter edit mode so you can re-specify the\n'
            'continuum type and line guesses, then refit with "Fit This Spaxel"'
        )
        self.cancel_edit_button.setToolTip(
            'Discard the in-progress edit and restore this spaxel\'s original fit')

        self.toggle_edited_button     = QPushButton('Toggle Edited')
        self.toggle_edited_button.setToolTip(
            'Show/hide translucent blue boxes marking spaxels that have a per-spaxel edit')
        self.stellar_template_button  = QPushButton('Stellar Template…')
        self.stellar_template_button.setToolTip(
            'Fit a stellar continuum (pPXF) to this spaxel and add it as a region')

        self.fit_spaxel_button.clicked.connect(self.fit_single_spaxel)
        self.clear_spaxel_fit_button.clicked.connect(self.clear_spaxel_fit)
        self.cancel_edit_button.clicked.connect(self.cancel_spaxel_edit)
        self.toggle_edited_button.clicked.connect(lambda: self.viewer_window.toggle_edited_overlay())
        self.stellar_template_button.clicked.connect(self.open_stellar_templates)
        # Cancel Edit is only meaningful while editing a single spaxel.
        self.cancel_edit_button.setEnabled(self._edit_mode())

        for btn in [self.fit_spaxel_button, self.clear_spaxel_fit_button,
                    self.cancel_edit_button, self.toggle_edited_button,
                    self.stellar_template_button]:
            btn.setFixedHeight(28)
            row1.addWidget(btn)
        row1.addStretch()
        vbox.addLayout(row1)

        # ── Row 2: Cube-level fit / map / mask actions ──────────────────────
        row2 = _new_row()
        row2.addWidget(_cat_label('Cube:'))

        self.smart_constraints_button    = QPushButton('Smart Constraints…')
        self.meas_errors_button          = QPushButton('Measurement Errors…')
        self.fit_cube_button             = QPushButton('Fit Cube')
        self.fix_fit_button              = QPushButton('Rectify Bad Fits')
        self.rchisq_map_button           = QPushButton('Quality Map ▾')
        self.mask_button                 = QPushButton('Mask…')
        self.clear_all_fits_button       = QPushButton('Clear All Fits')

        self.smart_constraints_button.clicked.connect(self.open_smart_constraints_dialog)
        if self._edit_mode():
            self.smart_constraints_button.setEnabled(False)
            self.smart_constraints_button.setToolTip(
                'Disabled while editing a single spaxel (schema is locked)')
        else:
            self.smart_constraints_button.setToolTip(
                'Auto-assign kinematic groups, doublet flux ratios, and parameter\n'
                'bounds for every line in the model, from a chosen physical scenario.')
        self.meas_errors_button.clicked.connect(self.open_measurement_errors_dialog)
        self.meas_errors_button.setToolTip(
            'Choose where per-spaxel 1σ flux uncertainties come from: a variance /\n'
            'error extension of the cube (or a sidecar file), or an empirical\n'
            'estimate from the line-free continuum inside each fit window.\n'
            'These weight the fit and set every reported parameter uncertainty.')
        self.fit_cube_button.clicked.connect(partial(self.fit_cube))
        self.fix_fit_button.clicked.connect(partial(self.fix_fits))
        self.rchisq_map_button.clicked.connect(self.show_quality_menu)
        self.rchisq_map_button.setToolTip(
            'Show a goodness-of-fit map: core/continuum ratio (≈1 good, ≫1 bad),\n'
            'signed-residual z (missed/over-subtracted flux), runs z (shape errors),\n'
            'calibrated continuum χ², or the native reduced χ²')
        self.mask_button.clicked.connect(partial(self.mask_spaxels))
        self.mask_button.setToolTip(
            'Mask the displayed maps by any quality / line / stellar map criterion\n'
            '(keep spaxels that satisfy it, hide the rest). Preview as a contour,\n'
            'then accept; Unmask clears it.')
        self.clear_all_fits_button.clicked.connect(self.clear_all_fits)
        self.clear_all_fits_button.setToolTip(
            'Remove ALL fit results for the whole cube (and per-spaxel edits).\n'
            'The model definition is kept so you can re-fit.')

        for btn in [self.smart_constraints_button, self.meas_errors_button,
                    self.fit_cube_button, self.fix_fit_button,
                    self.rchisq_map_button, self.mask_button, self.clear_all_fits_button]:
            btn.setFixedHeight(28)
            row2.addWidget(btn)

        # Sequential core→outflow fit toggle (no effect unless narrow/broad pairs exist)
        self.sequential_fit_checkbox = QCheckBox('Sequential')
        self.sequential_fit_checkbox.setChecked(getattr(self, '_sequential_fit', False))
        self.sequential_fit_checkbox.setToolTip(
            'Sequential core→outflow fitting: fit the narrow core first, then fit\n'
            'the broad component to the residual, then a joint polish. Breaks the\n'
            'narrow/broad degeneracy — recommended for outflow / multi-component\n'
            'lines. Only affects lines that have a broad (_b) partner.')
        self.sequential_fit_checkbox.toggled.connect(
            lambda checked: setattr(self, '_sequential_fit', bool(checked)))
        # Reserve room for the full label so it isn't clipped to "Sequentia".
        _seq_w = self.sequential_fit_checkbox.sizeHint().width()
        self.sequential_fit_checkbox.setMinimumWidth(_seq_w + 24)
        row2.addWidget(self.sequential_fit_checkbox)

        row2.addSpacing(12)
        row2.addWidget(_vsep())

        # Parallel cube-fit worker count (1 = serial). Default CPU-1.
        row2.addWidget(QLabel('Cores:'))
        self._cores_spin = QtWidgets.QSpinBox()
        _ncpu = os.cpu_count() or 2
        self._cores_spin.setRange(1, max(1, _ncpu))
        self._cores_spin.setValue(max(1, _ncpu - 1))
        self._cores_spin.setFixedHeight(28)
        self._cores_spin.setToolTip(
            'Worker processes for "Fit Cube" (1 = serial). Each spaxel\'s full\n'
            'model fit (continuum + lines across all regions) runs on its own\n'
            f'core. This machine reports {_ncpu} cores.')
        row2.addWidget(self._cores_spin)

        row2.addWidget(_vsep())

        # ── + Spectral Region ───────────────────────────────────────────────
        self.add_region_button = QPushButton('+ Region')
        self.add_region_button.setFixedHeight(28)
        if self._edit_mode():
            self.add_region_button.setEnabled(False)
            self.add_region_button.setToolTip('Disabled while editing a single spaxel (schema is locked)')
        else:
            self.add_region_button.setToolTip(
                'Add a new blank spectral region.\n'
                'You can also draw a continuum with the D key.'
            )
            self.add_region_button.clicked.connect(self._on_add_region_button_click)
        row2.addWidget(self.add_region_button)
        row2.addStretch()
        vbox.addLayout(row2)

        # ── Row 3: Save / load (fits & sessions) ────────────────────────────
        row3 = _new_row()
        row3.addWidget(_cat_label('Output:'))

        self.save_cube_fit_button        = QPushButton('Save Fit (CSV)')
        self.save_cube_fit_fitsfile_button = QPushButton('Save Fit (FITS)')
        self.load_cube_fit_button        = QPushButton('Load Fit')
        self.save_session_button         = QPushButton('Save Session')
        self.load_session_button         = QPushButton('Load Session')

        self.save_cube_fit_button.clicked.connect(partial(self.save_cube_fit))
        self.save_cube_fit_fitsfile_button.clicked.connect(partial(self.save_fit_result_fitsfile))
        self.load_cube_fit_button.clicked.connect(partial(self.load_cube_fit))
        self.save_session_button.clicked.connect(lambda: self.viewer_window.save_session())
        self.load_session_button.clicked.connect(lambda: self.viewer_window.load_session())
        self.save_session_button.setToolTip('Save the entire tool state (cube, fits, display, background) to a .hcsession file')
        self.load_session_button.setToolTip('Restore a previously saved .hcsession and resume where you left off')

        for btn in [self.save_cube_fit_button, self.save_cube_fit_fitsfile_button,
                    self.load_cube_fit_button, self.save_session_button,
                    self.load_session_button]:
            btn.setFixedHeight(28)
            row3.addWidget(btn)
        row3.addStretch()
        vbox.addLayout(row3)

        self.outer_scroll_layout.insertWidget(0, frame)  # toolbar always at top

    def add_spaxel_info_frame(self, title_text, df_cont, df, df_obs):
        """Legacy wrapper — calls the new compact toolbar builder."""
        self._build_obs_and_fit_toolbar(df_obs)

    def _on_add_region_button_click(self):
        """Manually add a new blank spectral region panel without drawing on the spectrum.
        
        Creates a placeholder continuum entry in df_cont with NaN geometry so the
        spectral region frame appears in the dock immediately.  The user can then
        populate x1/x2/slope/intercept by clicking the parameter buttons, or draw
        a real continuum with the 'D' key which will override these values.
        """
        global df_cont, df
        new_id = len(df_cont)  # Next region ID
        
        # Add a placeholder continuum row so add_spectral_frame has data to display
        placeholder_cont = pd.DataFrame({
            'Continuum Name': ['Continuum'],
            'x1': [np.nan],
            'x2': [np.nan],
            'Slope_0': [0.0],
            'Intercept_0': [0.0],
            'Slope_fit': [np.nan],
            'Intercept_fit': [np.nan],
            'region_ID': [new_id],
            'lineactor': [None],
            'cont_type': ['linear'],
            'knots_x': [[]],
            'knots_y_0': [[]],
            'knots_y_fit': [[]],
            'poly_degree': [np.nan],
            'poly_coef_0': [[]],
            'poly_coef_fit': [[]],
        })
        
        if len(df_cont) == 0:
            df_cont = placeholder_cont
        else:
            df_cont = pd.concat([df_cont, placeholder_cont], ignore_index=True)
        
        # Add the spectral region frame (addframe=False because we already updated df_cont above)
        self.add_spectral_frame(
            f"Spectral Region {new_id + 1}", df_cont, df, ID=new_id, addframe=False
        )

    # Default bad-fit threshold on the scale-free core/continuum residual ratio
    # Default thresholds suggested in the Rectify dialog per quality metric.
    _RECTIFY_DEFAULTS = {
        'qa_core_cont_ratio': 2.0,
        'qa_signed_resid_z':  3.0,
        'qa_runs_z':          3.0,
        'qa_chisq_cont':      5.0,
        # Weighted χ²_red is normalised (≈1 for a good fit with a correct noise
        # model), so it flags much closer to 1 than the unnormalised natives.
        'rchisq_w':           3.0,
        'rchisq':             5.0,
    }

    def _rectify_available_maps(self):
        """Enumerate every per-spaxel map the user can currently flag spaxels on.

        Returns a list of descriptors, each a dict:
            label        : human-readable name shown in the combo
            values       : {(x, y): float}  per-spaxel values
            signed       : bool — value can be ±  (→ default to a |·| operator)
            default_op   : suggested operator token ('>','<','abs>','abs<')
            default_thr  : suggested (positive) threshold magnitude

        Sources: the calibrated quality metrics, every fitted emission-line
        parameter (amplitude, centroid, velocity, σ in km/s), and — if a
        stellar cube fit exists — the stellar kinematics (V, σ).
        """
        maps = []

        def _per_spaxel_col(frame, col, conv=None):
            out = {}
            for _, rr in frame.iterrows():
                try:
                    x, y = int(rr['spaxel_x']), int(rr['spaxel_y'])
                except (TypeError, ValueError, KeyError):
                    continue
                v = _safe_float(rr.get(col))
                if conv is not None and np.isfinite(v):
                    v = conv(rr, v)
                out[(x, y)] = v
            return out

        # ── Quality metrics (per-spaxel; one value repeated across line rows) ──
        if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0:
            per = df_fit.drop_duplicates(subset=['spaxel_x', 'spaxel_y'])
            for lbl, col, _cmap, signed in self.QUALITY_MAPS:
                if col in df_fit.columns:
                    maps.append({
                        'label': f'Quality: {lbl}',
                        'values': _per_spaxel_col(per, col),
                        'signed': signed,
                        'default_op': 'abs>' if signed else '>',
                        'default_thr': self._RECTIFY_DEFAULTS.get(col, 2.0),
                    })

            # ── Emission-line fitted-parameter maps (one per line) ──
            # (column, suffix label, signed, default op, default thr, km/s conv)
            line_specs = [
                ('amp_fit',   'amplitude',      False, '<',    None),
                ('cen_fit',   'centroid (Å)',   False, '>',    None),
                ('vel_fit',   'velocity (km/s)', True, 'abs>', 500.0),
                ('sigma_fit', 'σ (km/s)',       False, '>',    300.0),
            ]
            if 'LineID' in df_fit.columns:
                seen = []
                for _, lr in df_fit.drop_duplicates(subset=['LineID', 'region_ID']).iterrows():
                    lid = lr.get('LineID')
                    rid = lr.get('region_ID')
                    name = str(lr.get('LineName', f'Line {lid}'))
                    sub = df_fit[(df_fit['LineID'] == lid) &
                                 (df_fit['region_ID'] == rid)]
                    for col, suffix, signed, dop, dthr in line_specs:
                        if col not in df_fit.columns:
                            continue
                        if col == 'sigma_fit':
                            conv = lambda rr, v: sigma_wl_to_kms(
                                _safe_float(rr.get('sigma_fit')),
                                _safe_float(rr.get('cen_fit')))
                        else:
                            conv = None
                        vals = _per_spaxel_col(sub, col, conv=conv)
                        finite = [v for v in vals.values() if np.isfinite(v)]
                        if dthr is not None:
                            thr = dthr
                        elif finite:
                            thr = float(np.round(np.nanmedian(np.abs(finite)), 4))
                        else:
                            thr = 0.0
                        maps.append({
                            'label': f'{name} — {suffix}',
                            'values': vals,
                            'signed': signed,
                            'default_op': dop,
                            'default_thr': thr,
                        })

        # ── Stellar kinematics maps (per-spaxel, from df_stellar) ──
        if isinstance(df_stellar, pd.DataFrame) and len(df_stellar) > 0 \
                and 'spaxel_x' in df_stellar.columns:
            stellar_specs = [
                ('stellar_V',     'Stellar — V (km/s)',  True,  'abs>', 500.0),
                ('stellar_sigma', 'Stellar — σ (km/s)',  False, '>',    300.0),
            ]
            for col, lbl, signed, dop, dthr in stellar_specs:
                if col in df_stellar.columns:
                    maps.append({
                        'label': lbl,
                        'values': _per_spaxel_col(df_stellar, col),
                        'signed': signed,
                        'default_op': dop,
                        'default_thr': dthr,
                    })

        return maps

    @staticmethod
    def _rectify_flagged(values, op, thresh):
        """Return the set of (x, y) keys in `values` selected by op/thresh.

        Non-finite values are never selected here (the dialog offers a separate
        opt-in for failed fits). Operators:
            '>'    value >  thresh
            '<'    value <  thresh
            'abs>' |value| >  thresh
            'abs<' |value| <  thresh
        """
        out = set()
        for xy, v in values.items():
            fv = _safe_float(v)
            if not np.isfinite(fv):
                continue
            if op == '>':
                hit = fv > thresh
            elif op == '<':
                hit = fv < thresh
            elif op == 'abs>':
                hit = abs(fv) > thresh
            elif op == 'abs<':
                hit = abs(fv) < thresh
            else:
                hit = False
            if hit:
                out.add(xy)
        return out

    def fix_fits(self):
        global df_fit
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QComboBox, QDoubleSpinBox, QPushButton,
                                     QDialogButtonBox, QFrame, QMessageBox)

        if len(df_fit) == 0:
            QMessageBox.information(self, 'Rectify Bad Fits',
                                    'No fitted spaxels found. Run Fit Cube first.')
            return

        from PyQt5.QtWidgets import QCheckBox

        # Every per-spaxel map the user can flag on (quality, line params, stellar).
        maps = self._rectify_available_maps()
        if not maps:
            QMessageBox.information(self, 'Rectify Bad Fits',
                                    'No maps found in the current fit.\n'
                                    'Re-run Fit Cube to generate them.')
            return

        # Operator options offered for every map. Tokens are consumed by
        # _rectify_flagged. The threshold itself is a signed number, so single-
        # sided '<'/'>' can target negative values directly (e.g. vel < −200).
        OPS = [('>',     '>'),
               ('<',     '<'),
               ('|·| >', 'abs>'),
               ('|·| <', 'abs<')]

        dlg = QDialog(self)
        dlg.setWindowTitle('Rectify Bad Fits')
        dlg.setMinimumWidth(470)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # ── Row: "Flag spaxels where: [map combo]" ────────────────────────
        map_row = QHBoxLayout()
        map_row.addWidget(QLabel('Flag spaxels where:'))
        map_combo = QComboBox()
        for m in maps:
            map_combo.addItem(m['label'])
        map_row.addWidget(map_combo, 1)
        layout.addLayout(map_row)

        # ── Row: "[op combo]  [threshold spin]" ───────────────────────────
        thresh_row = QHBoxLayout()
        op_combo = QComboBox()
        op_combo.setFixedWidth(80)
        for disp, _tok in OPS:
            op_combo.addItem(disp)
        thresh_row.addWidget(op_combo)
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(3)
        spin.setSingleStep(0.5)
        thresh_row.addWidget(spin)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        # ── Opt-in: also re-fit failed (non-finite) spaxels ───────────────
        nonfinite_chk = QCheckBox('Also re-fit failed (non-finite) spaxels')
        nonfinite_chk.setChecked(True)
        layout.addWidget(nonfinite_chk)

        # ── Live count preview ────────────────────────────────────────────
        preview = QLabel()
        preview.setWordWrap(True)
        layout.addWidget(preview)

        def _current():
            midx = map_combo.currentIndex()
            oidx = op_combo.currentIndex()
            if not (0 <= midx < len(maps)) or not (0 <= oidx < len(OPS)):
                return None
            return maps[midx], OPS[oidx][1]

        def _refresh(*_):
            cur = _current()
            if cur is None:
                return
            m, op = cur
            vals = m['values']
            thresh = spin.value()
            flagged = self._rectify_flagged(vals, op, thresh)
            n_nonfinite = sum(1 for v in vals.values() if not np.isfinite(_safe_float(v)))
            extra = n_nonfinite if nonfinite_chk.isChecked() else 0
            disp = {'>': f'> {thresh:g}', '<': f'< {thresh:g}',
                    'abs>': f'|·| > {thresh:g}', 'abs<': f'|·| < {thresh:g}'}[op]
            tail = (f" + {extra} failed" if extra else "")
            preview.setText(
                f"<b>{len(flagged) + extra}</b> of <b>{len(vals)}</b> spaxels "
                f"(value {disp}{tail}) will be re-fit."
            )

        def _on_map_changed(midx):
            if 0 <= midx < len(maps):
                m = maps[midx]
                # Set the suggested operator and threshold for this map.
                tok_to_idx = {tok: i for i, (_d, tok) in enumerate(OPS)}
                op_combo.blockSignals(True)
                op_combo.setCurrentIndex(tok_to_idx.get(m['default_op'], 0))
                op_combo.blockSignals(False)
                spin.blockSignals(True)
                spin.setValue(float(m['default_thr']))
                spin.blockSignals(False)
            _refresh()

        map_combo.currentIndexChanged.connect(_on_map_changed)
        op_combo.currentIndexChanged.connect(_refresh)
        spin.valueChanged.connect(_refresh)
        nonfinite_chk.toggled.connect(_refresh)
        _on_map_changed(0)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # ── Help text ─────────────────────────────────────────────────────
        HELP = (
            "<b>Rectify Bad Fits</b> re-fits the spaxels you flag here, seeding "
            "each from its best-fitting 8-neighbour (and falling back to a small "
            "set of targeted restarts). Only the flagged spaxels change.<br><br>"
            "<b>Which map?</b> Flag spaxels on any map currently derivable from "
            "the fit:<br>"
            "• <b>Quality metrics</b> — measure goodness-of-fit. "
            "<i>Core / continuum ratio</i> (≈1 good, ≫1 = missed line profile; "
            "try &gt; 2) is the recommended default; <i>signed residual (z)</i> "
            "and <i>runs (z)</i> are signed; <i>reduced χ²</i> (continuum / "
            "native) catch gross failures (try &gt; 5).<br>"
            "• <b>Emission-line parameters</b> — each line's fitted amplitude, "
            "centroid, velocity (km/s), and σ (km/s). Useful to catch "
            "non-physical outliers, e.g. a velocity map with runaway spaxels.<br>"
            "• <b>Stellar kinematics</b> — V and σ (km/s), if a stellar cube fit "
            "exists.<br><br>"
            "<b>Operator &amp; threshold.</b> The threshold is a signed number; "
            "the operator sets the direction:<br>"
            "&nbsp;&nbsp;<b>&gt; T</b> — value above T<br>"
            "&nbsp;&nbsp;<b>&lt; T</b> — value below T (T may be negative, "
            "e.g. <i>vel &lt; −300</i>)<br>"
            "&nbsp;&nbsp;<b>|·| &gt; T</b> — magnitude above T "
            "(both wings, e.g. <i>|vel| &gt; 500</i>)<br>"
            "&nbsp;&nbsp;<b>|·| &lt; T</b> — magnitude below T (near zero)<br><br>"
            "For a velocity map you can therefore flag one wing (&gt; / &lt;) or "
            "both (|·| &gt;). The live count updates as you adjust.<br><br>"
            "<b>Failed fits.</b> Spaxels whose value is non-finite (NaN / ±Inf) "
            "are not selected by the numeric test; tick the checkbox to re-fit "
            "those too (recommended when flagging on a quality metric)."
        )

        # ── Button row ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        help_btn = QPushButton('?')
        help_btn.setFixedWidth(28)
        help_btn.setToolTip('Explain the maps, operators and thresholds')
        help_btn.clicked.connect(
            lambda: QMessageBox.information(dlg, 'Rectify — Help', HELP))
        btn_row.addWidget(help_btn)
        btn_row.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('Rectify')
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        if dlg.exec_() != QDialog.Accepted:
            return

        cur = _current()
        if cur is None:
            return
        m, op = cur
        thresh = spin.value()
        flagged = self._rectify_flagged(m['values'], op, thresh)
        if nonfinite_chk.isChecked():
            flagged |= {xy for xy, v in m['values'].items()
                        if not np.isfinite(_safe_float(v))}
        if not flagged:
            QMessageBox.information(self, 'Rectify Bad Fits',
                                    'No spaxels matched the criterion.')
            return
        self.fit_cube(refit=True, bad_spaxels_override=flagged)

    def _mask_keep_field(self, m, op, thresh):
        """Build (keep_field, mask_array) for the current map criterion.

        keep_field : float (ny, nx) — 1.0 where the spaxel SATISFIES the
                     criterion (kept), 0.0 in the criterion's domain otherwise,
                     NaN outside the domain. Used for the preview contour.
        mask_array : bool  (ny, nx) — True where the spaxel is masked OUT
                     (in the domain but does NOT satisfy the criterion).
        """
        ny, nx = int(FITS_DATA.shape[-2]), int(FITS_DATA.shape[-1])
        keep_field = np.full((ny, nx), np.nan)
        mask_array = np.zeros((ny, nx), dtype=bool)
        vals = m['values']
        keep = self._rectify_flagged(vals, op, thresh)  # spaxels satisfying
        for (x, y), v in vals.items():
            if not (0 <= y < ny and 0 <= x < nx):
                continue
            if (x, y) in keep:
                keep_field[y, x] = 1.0
            else:
                keep_field[y, x] = 0.0
                mask_array[y, x] = True  # in domain but fails → mask out
        return keep_field, mask_array

    def mask_spaxels(self):
        """Mask the displayed maps by an arbitrary map criterion: keep spaxels
        that satisfy it, hide the rest. Preview as a contour (like the SNR mask),
        then Accept; Unmask clears any active mask."""
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QComboBox, QDoubleSpinBox, QPushButton,
                                     QFrame, QMessageBox)

        vw = self.viewer_window

        maps = self._rectify_available_maps()
        if not maps:
            QMessageBox.information(self, 'Mask Spaxels',
                                    'No maps found. Fit the cube first so there '
                                    'are quality / line / stellar maps to mask on.')
            return

        OPS = [('>',     '>'),
               ('<',     '<'),
               ('|·| >', 'abs>'),
               ('|·| <', 'abs<')]

        dlg = QDialog(self)
        dlg.setWindowTitle('Mask Spaxels')
        dlg.setMinimumWidth(470)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # ── Row: "Keep spaxels where: [map combo]" ────────────────────────
        map_row = QHBoxLayout()
        map_row.addWidget(QLabel('Keep spaxels where:'))
        map_combo = QComboBox()
        for m in maps:
            map_combo.addItem(m['label'])
        map_row.addWidget(map_combo, 1)
        layout.addLayout(map_row)

        # ── Row: "[op combo]  [threshold spin]" ───────────────────────────
        thresh_row = QHBoxLayout()
        op_combo = QComboBox()
        op_combo.setFixedWidth(80)
        for disp, _tok in OPS:
            op_combo.addItem(disp)
        thresh_row.addWidget(op_combo)
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(3)
        spin.setSingleStep(0.5)
        thresh_row.addWidget(spin)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        # ── Live count preview ────────────────────────────────────────────
        preview = QLabel()
        preview.setWordWrap(True)
        layout.addWidget(preview)

        def _current():
            midx = map_combo.currentIndex()
            oidx = op_combo.currentIndex()
            if not (0 <= midx < len(maps)) or not (0 <= oidx < len(OPS)):
                return None
            return maps[midx], OPS[oidx][1]

        def _refresh(*_):
            cur = _current()
            if cur is None:
                return
            m, op = cur
            vals = m['values']
            thresh = spin.value()
            keep = self._rectify_flagged(vals, op, thresh)
            n_keep = len(keep)
            n_mask = len(vals) - n_keep
            disp = {'>': f'> {thresh:g}', '<': f'< {thresh:g}',
                    'abs>': f'|·| > {thresh:g}', 'abs<': f'|·| < {thresh:g}'}[op]
            preview.setText(
                f"Keep <b>{n_keep}</b> (value {disp}); mask out <b>{n_mask}</b> "
                f"of <b>{len(vals)}</b> spaxels in this map."
            )

        def _on_map_changed(midx):
            if 0 <= midx < len(maps):
                m = maps[midx]
                tok_to_idx = {tok: i for i, (_d, tok) in enumerate(OPS)}
                op_combo.blockSignals(True)
                op_combo.setCurrentIndex(tok_to_idx.get(m['default_op'], 0))
                op_combo.blockSignals(False)
                spin.blockSignals(True)
                spin.setValue(float(m['default_thr']))
                spin.blockSignals(False)
            _refresh()

        map_combo.currentIndexChanged.connect(_on_map_changed)
        op_combo.currentIndexChanged.connect(_refresh)
        spin.valueChanged.connect(_refresh)
        _on_map_changed(0)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        HELP = (
            "<b>Mask Spaxels</b> hides spaxels on every displayed map so you can "
            "focus on the region you care about (e.g. only well-fit spaxels, or "
            "only a velocity range). It is a display mask — it does not change "
            "the fit.<br><br>"
            "<b>Keep spaxels where …</b> defines the criterion the kept spaxels "
            "<i>satisfy</i>; every other spaxel in that map is masked out. Pick "
            "any quality metric, emission-line parameter (amplitude, centroid, "
            "velocity, σ), or stellar map, then an operator and threshold — "
            "exactly as in Rectify.<br><br>"
            "<b>Operators.</b> The threshold is signed:<br>"
            "&nbsp;&nbsp;<b>&gt; T</b> / <b>&lt; T</b> — keep values above / below T "
            "(T may be negative, e.g. keep <i>vel &gt; −300</i>)<br>"
            "&nbsp;&nbsp;<b>|·| &gt; T</b> / <b>|·| &lt; T</b> — keep by magnitude "
            "(e.g. keep <i>|vel| &lt; 300</i> to mask high-velocity outliers)<br><br>"
            "<b>Buttons.</b> <i>Visualize</i> outlines the kept region as a cyan "
            "contour (like the SNR mask) without applying it. <i>Accept mask</i> "
            "applies it to the display. <i>Unmask</i> clears any active mask. "
            "<i>Cancel</i> closes and removes the preview, leaving any existing "
            "mask untouched.<br><br>"
            "Spaxels with no value in the chosen map (e.g. unfit) keep their "
            "current visibility."
        )

        # ── Button row: ? | Visualize | Accept | Unmask | Cancel ──────────
        btn_row = QHBoxLayout()
        help_btn = QPushButton('?')
        help_btn.setFixedWidth(28)
        help_btn.setToolTip('Explain masking, maps and operators')
        help_btn.clicked.connect(
            lambda: QMessageBox.information(dlg, 'Mask — Help', HELP))
        btn_row.addWidget(help_btn)
        btn_row.addStretch()

        visualize_btn = QPushButton('Visualize')
        accept_btn    = QPushButton('Accept mask')
        unmask_btn    = QPushButton('Unmask')
        cancel_btn    = QPushButton('Cancel')
        visualize_btn.setToolTip('Preview the kept region as a contour (does not apply)')
        accept_btn.setToolTip('Apply the mask to the displayed maps')
        unmask_btn.setToolTip('Remove any active mask')
        for b in (visualize_btn, accept_btn, unmask_btn, cancel_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        def _on_visualize():
            cur = _current()
            if cur is None:
                return
            m, op = cur
            keep_field, _mask = self._mask_keep_field(m, op, spin.value())
            vw.show_mask_preview(keep_field)

        def _on_accept():
            cur = _current()
            if cur is None:
                return
            m, op = cur
            _keep_field, mask_array = self._mask_keep_field(m, op, spin.value())
            vw.clear_mask_preview()
            vw._spaxel_mask = mask_array
            vw.redraw_current_image()
            dlg.accept()

        def _on_unmask():
            vw.clear_mask_preview()
            vw._spaxel_mask = None
            vw.redraw_current_image()
            dlg.accept()

        def _on_cancel():
            # Leave any already-applied mask intact; just drop the preview.
            vw.clear_mask_preview()
            dlg.reject()

        visualize_btn.clicked.connect(_on_visualize)
        accept_btn.clicked.connect(_on_accept)
        unmask_btn.clicked.connect(_on_unmask)
        cancel_btn.clicked.connect(_on_cancel)
        dlg.rejected.connect(vw.clear_mask_preview)  # safety: X / Esc

        dlg.exec_()

    def _params_from_fit_row(self, base_params, rows):
        """Seed a copy of base_params from a neighbour spaxel's fitted values.

        rows: the df_fit rows for one spaxel (one per line). Sets each line's
        amp/cen/sigma initial guess (and region-1 continuum) to the neighbour's
        best-fit values, clamped to bounds. Derived (expr-constrained) params
        are skipped — they are non-varying and recomputed from their reference.
        """
        p = base_params.copy()
        try:
            rows = rows.sort_values('LineID')
        except Exception:
            pass
        for idx, (_, r) in enumerate(rows.iterrows(), start=1):
            for key, col in ((f'amp{idx}', 'amp_fit'),
                             (f'cen{idx}', 'cen_fit'),
                             (f'sigma{idx}', 'sigma_fit')):
                if key in p and p[key].vary:
                    v = _safe_float(r.get(col))
                    if np.isfinite(v):
                        lo, hi = p[key].min, p[key].max
                        if lo is not None and np.isfinite(lo):
                            v = max(v, lo)
                        if hi is not None and np.isfinite(hi):
                            v = min(v, hi)
                        p[key].value = v
        # Region-1 linear continuum, if the neighbour recorded it and it varies.
        first = rows.iloc[0] if len(rows) else None
        if first is not None:
            for key, col in (('slope1', 'cont_region1_slope_fit'),
                             ('intercept1', 'cont_region1_intercept_fit')):
                if key in p and p[key].vary:
                    v = _safe_float(first.get(col))
                    if np.isfinite(v):
                        p[key].value = v
        return p

    # ── Targeted multi-start (Phase C: fallback for Rectify survivors) ───────
    def _component_pairs(self):
        """Model-order (1-based) index pairs of two-component lines that share a
        rest wavelength, e.g. (narrow, broad). Ordered narrow-first by σ guess.
        """
        if 'Rest Wavelength' not in df.columns:
            return []
        groups = {}
        for idx, (_, r) in enumerate(df.iterrows(), start=1):
            rw = _safe_float(r.get('Rest Wavelength'))
            sig = _safe_float(r.get('Sigma_0'))
            if np.isfinite(rw):
                groups.setdefault(round(rw, 4), []).append((idx, sig))
        pairs = []
        for members in groups.values():
            if len(members) == 2:
                members.sort(key=lambda t: (t[1] if np.isfinite(t[1]) else np.inf))
                pairs.append((members[0][0], members[1][0]))  # (narrow, broad)
        return pairs

    @staticmethod
    def _seed_param(p, key, value):
        """Set a free param's value, clamped to its bounds. Skips derived
        (expr-constrained / non-varying) params so constraints stay intact."""
        if key not in p or not p[key].vary or value is None or not np.isfinite(value):
            return
        lo, hi = p[key].min, p[key].max
        if lo is not None and np.isfinite(lo):
            value = max(value, lo)
        if hi is not None and np.isfinite(hi):
            value = min(value, hi)
        p[key].value = value

    def _targeted_restarts(self, base_params):
        """Physically-motivated restart parameter sets for the known 2-component
        failure modes (narrow-only, broad-only, swapped, equal-split). Best-effort:
        only free params are edited; returns [(label, Parameters), ...]."""
        pairs = self._component_pairs()
        if not pairs:
            return []

        def _amp_floor(p, key):
            lo = p[key].min if key in p else None
            return lo if (lo is not None and np.isfinite(lo)) else 0.0

        variants = []
        # narrow-only / broad-only: drive the other component's amplitude to ~0.
        for label, kill_broad in (('narrow-only', True), ('broad-only', False)):
            p = base_params.copy()
            for (i_n, i_b) in pairs:
                tgt = f'amp{i_b}' if kill_broad else f'amp{i_n}'
                self._seed_param(p, tgt, _amp_floor(p, tgt))
            variants.append((label, p))
        # swapped: exchange amp & σ between the two components.
        p = base_params.copy()
        for (i_n, i_b) in pairs:
            for stem in ('amp', 'sigma'):
                a, b = f'{stem}{i_n}', f'{stem}{i_b}'
                if a in p and b in p:
                    va, vb = p[a].value, p[b].value
                    self._seed_param(p, a, vb)
                    self._seed_param(p, b, va)
        variants.append(('swapped', p))
        # equal-split: average amp & σ across the pair.
        p = base_params.copy()
        for (i_n, i_b) in pairs:
            for stem in ('amp', 'sigma'):
                a, b = f'{stem}{i_n}', f'{stem}{i_b}'
                if a in p and b in p:
                    avg = 0.5 * (p[a].value + p[b].value)
                    self._seed_param(p, a, avg)
                    self._seed_param(p, b, avg)
        variants.append(('equal-split', p))
        return variants

    def _fit_candidate(self, z, candidate_params):
        """Fit the current spaxel with candidate_params and return
        (qa_core_cont_ratio, rows) WITHOUT committing: the rows fit_spaxel
        appended are popped back off fit_results so the caller can keep only
        the best candidate. A failed/degenerate candidate returns +inf."""
        global fit_results
        snap = len(fit_results)
        self.fit_spaxel(z, max_nfev=512, params_to_use=candidate_params)
        rows = fit_results[snap:]
        del fit_results[snap:]
        ratio = np.inf
        if rows:
            r = _safe_float(rows[0].get('qa_core_cont_ratio'))
            ratio = r if np.isfinite(r) else np.inf
        return ratio, rows

    def _staged_fit(self, model, y, params, wavelengths, max_nfev):
        """Sequential core→outflow fit (breaks the narrow/broad degeneracy).

        Stage 1 fits the narrow core(s) with the broad amplitudes suppressed;
        Stage 2 freezes the core + continuum and fits each broad component to
        the residual (where it is the only feature and cannot collapse onto the
        core); Stage 3 is a joint polish from that solution. Falls back to a
        single joint fit if there are no narrow/broad pairs or on any error.
        Operates in the already-flux-rescaled parameter space.
        """
        try:
            pairs = self._component_pairs()
        except Exception:
            pairs = []
        if not pairs:
            return model.fit(y, params, x=wavelengths, max_nfev=max_nfev)

        try:
            broad_amps = {f'amp{i_b}' for (_i_n, i_b) in pairs}
            cont_prefixes = ('slope', 'intercept', 'polyc', 'knoty')  # slope→slope_int too
            orig_vary = {name: p.vary for name, p in params.items()}

            # Stage 1 — core only: suppress free broad amplitudes.
            p1 = params.copy()
            for k in broad_amps:
                if k in p1 and not p1[k].expr:
                    p1[k].set(value=0.0, vary=False)
            r1 = model.fit(y, p1, x=wavelengths, max_nfev=max_nfev)

            # Stage 2 — broad on the residual: freeze core + continuum, free broad.
            p2 = r1.params.copy()
            for name, p in p2.items():
                if name in broad_amps:
                    if not p.expr:
                        p.set(value=params[name].value, vary=True)  # reset to broad init
                        if params[name].min is not None:
                            p.min = params[name].min
                        if params[name].max is not None:
                            p.max = params[name].max
                elif name.startswith('offset_vel') or name.startswith('ratio'):
                    continue  # broad's constraint helpers stay free
                elif (name.startswith(('amp', 'cen', 'sigma')) or
                      name.startswith(cont_prefixes)):
                    if p.expr is None:
                        p.vary = False  # freeze narrow core + continuum
            r2 = model.fit(y, p2, x=wavelengths, max_nfev=max_nfev)

            # Stage 3 — joint polish from the staged solution.
            p3 = r2.params.copy()
            for name, p in p3.items():
                if p.expr is None and name in orig_vary:
                    p.vary = orig_vary[name]
            return model.fit(y, p3, x=wavelengths, max_nfev=max(64, max_nfev // 2))
        except Exception as e:
            print(f"Staged fit fell back to joint fit: {type(e).__name__}: {e}")
            return model.fit(y, params, x=wavelengths, max_nfev=max_nfev)



    def save_fit_result_fitsfile(self):
        """Save df_fit results as a multi-extension FITS file with WCS-based RA/Dec mapping."""
        wcs = self.viewer_window.wcs
        print('Saving cube fit to FITS file')
    
        # Prompt user for file save location
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "FITS Files (*.fits);;All Files (*)", options=options)
        
        if not file_path:
            print("File save canceled.")
            return
    
        # Replace "nan" and "NaN" strings with actual NaNs
        df_fit.replace(["nan", "NaN"], np.nan, inplace=True)
    
        # Convert coordinate columns to int
        df_fit[["spaxel_x", "spaxel_y"]] = df_fit[["spaxel_x", "spaxel_y"]].astype(int)
    
        #   Convert spaxel coordinates to RA, Dec using the 3D WCS
        spaxel_x = df_fit["spaxel_x"].values
        spaxel_y = df_fit["spaxel_y"].values
        spaxel_z = np.zeros_like(spaxel_x)  # Placeholder for spectral axis
        
        ra_vals, dec_vals, _ = wcs.all_pix2world(spaxel_x, spaxel_y, spaxel_z, 0)

        
        # Identify unique spectral lines
        unique_lines = np.unique(df_fit['LineName'])
        extensions = []
    
        # Create a Primary HDU
        primary_hdu = fits.PrimaryHDU()
        extensions.append(primary_hdu)
    
        # Generate image maps for each line and each fit parameter.
        # Every map is paired with its 1σ uncertainty map, and σ is written as a
        # velocity dispersion (km/s) — the same unit the GUI and the CSV use.
        _flux_bunit = flux_unit_str()
        map_specs = [
            # (HDU stem, value column, error column, converter, BUNIT)
            ('amp',   'amp_fit',   'amp_std',   None,             _flux_bunit),
            ('cen',   'cen_fit',   'cen_std',   None,             'Angstrom'),
            ('vel',   'vel_fit',   'vel_std',   None,             'km/s'),
            ('sigma', 'sigma_fit', 'sigma_std', 'kms',            'km/s'),
        ]
        ny_map, nx_map = int(FITS_DATA.shape[-2]), int(FITS_DATA.shape[-1])

        for line in unique_lines:
            line_df = df_fit[df_fit['LineName'] == line]
            for stem, val_col, err_col, conv, bunit in map_specs:
                for col, suffix, comment in ((val_col, '', 'best-fit value'),
                                             (err_col, '_std', '1-sigma uncertainty')):
                    if col not in line_df.columns:
                        continue
                    image_map = np.full((ny_map, nx_map), np.nan)
                    for _, row in line_df.iterrows():
                        x, y = int(row["spaxel_x"]), int(row["spaxel_y"])
                        # σ (and its error) live in Å internally → km/s on output,
                        # both scaled by the *fitted* centroid of the same spaxel.
                        image_map[y, x] = (sigma_wl_to_kms(row[col], row['cen_fit'])
                                           if conv == 'kms' else _safe_float(row[col]))
                    hdu = fits.ImageHDU(data=image_map, name=f"{stem}{suffix}_{line}")
                    hdu.header.update(wcs.to_header())  # Add WCS info
                    hdu.header['BUNIT'] = (bunit, f'{stem} {comment}')
                    extensions.append(hdu)

        # Stellar kinematics maps (one HDU per quantity), if a stellar cube fit
        # has been run.
        if isinstance(df_stellar, pd.DataFrame) and len(df_stellar) > 0 \
                and 'spaxel_x' in df_stellar.columns:
            ny, nx = int(FITS_DATA.shape[-2]), int(FITS_DATA.shape[-1])
            # One map per (quantity, stellar region). With >1 region the HDU name
            # is suffixed with the region id; a single region keeps the bare names.
            if 'region_ID' in df_stellar.columns:
                _srids = sorted(int(r) for r in df_stellar['region_ID'].dropna().unique())
            else:
                _srids = [None]
            _multi = len(_srids) > 1
            for srid in _srids:
                _rows = (df_stellar if srid is None
                         else df_stellar[df_stellar['region_ID'] == srid])
                for scol, sname in [('stellar_V', 'stellar_vel'),
                                    ('stellar_sigma', 'stellar_sigma'),
                                    ('stellar_h3', 'stellar_h3'),
                                    ('stellar_h4', 'stellar_h4'),
                                    ('stellar_chi2', 'stellar_chi2')]:
                    if scol not in df_stellar.columns:
                        continue
                    smap = np.full((ny, nx), np.nan)
                    for _, row in _rows.iterrows():
                        smap[int(row['spaxel_y']), int(row['spaxel_x'])] = _safe_float(row.get(scol))
                    hduname = f'{sname}_r{srid}' if (_multi and srid is not None) else sname
                    shdu = fits.ImageHDU(data=smap, name=hduname)
                    try:
                        shdu.header.update(wcs.to_header())
                    except Exception:
                        pass
                    # pPXF reports V and σ in km/s; h3/h4/χ² are dimensionless.
                    shdu.header['BUNIT'] = ('km/s' if scol in ('stellar_V', 'stellar_sigma')
                                            else '', f'{sname}')
                    extensions.append(shdu)

        # Write all extensions to FITS file
        hdulist = fits.HDUList(extensions)
        hdulist.writeto(file_path, overwrite=True)
    
        print(f"FITS file saved to {file_path}")
        

    def open_measurement_errors_dialog(self):
        """Choose where per-pixel 1σ flux uncertainties come from.

        Either an error/variance/inverse-variance cube (an extension of the
        science file or a sidecar file), or the empirical estimate measured from
        the line-free continuum inside each fit window. Whatever is chosen here
        weights the fit and is recorded per spaxel in the fit output.
        """
        vw = self.viewer_window
        if FITS_DATA is None:
            QMessageBox.information(self, 'Measurement Errors',
                                    'Open a cube first.')
            return

        dlg = QDialog(self)
        dlg.setWindowTitle('Measurement Errors')
        lay = QVBoxLayout(dlg)

        blurb = QLabel(
            'Per-pixel 1σ flux uncertainties. The fit is weighted by 1/σ inside '
            'each fit window,\nso this choice sets every reported parameter '
            'uncertainty (amp_std, cen_std, vel_std, σ_std).')
        blurb.setStyleSheet('color: gray;')
        lay.addWidget(blurb)

        current = QLabel(f"Currently: {ERROR_INFO.get('source', HyperCube_Noise.SRC_EMPIRICAL)}")
        current.setStyleSheet('font-weight: bold;')
        lay.addWidget(current)

        group = QButtonGroup(dlg)
        rb_cube = QRadioButton('Error cube')
        rb_emp = QRadioButton(
            'Empirical — robust noise from the line-free continuum inside each '
            'fit window (DER_SNR)')
        group.addButton(rb_cube); group.addButton(rb_emp)
        lay.addWidget(rb_cube)

        # File + extension + flavour, indented under the "Error cube" radio.
        form = QGridLayout()
        form.setContentsMargins(24, 0, 0, 8)
        path_edit = QLineEdit(str(ERROR_INFO.get('path')
                                  or getattr(vw, 'fits_path', '') or ''))
        browse = QPushButton('Browse…')
        ext_combo = QComboBox()
        kind_combo = QComboBox()
        for kind in (HyperCube_Noise.KIND_VAR, HyperCube_Noise.KIND_SIGMA,
                     HyperCube_Noise.KIND_IVAR):
            kind_combo.addItem(HyperCube_Noise.KIND_LABELS[kind], kind)
        form.addWidget(QLabel('File:'), 0, 0)
        form.addWidget(path_edit, 0, 1)
        form.addWidget(browse, 0, 2)
        form.addWidget(QLabel('Extension:'), 1, 0)
        form.addWidget(ext_combo, 1, 1, 1, 2)
        form.addWidget(QLabel('Values are:'), 2, 0)
        form.addWidget(kind_combo, 2, 1, 1, 2)
        lay.addLayout(form)
        lay.addWidget(rb_emp)

        note = QLabel(
            'Spaxels already fitted keep the uncertainties they were fitted with; '
            're-fit to apply a change.')
        note.setStyleSheet('color: gray; font-size: 11px;')
        lay.addWidget(note)

        def repopulate():
            """List the candidate extensions of the file in the path box."""
            ext_combo.clear()
            cands = HyperCube_Noise.candidate_extensions(
                path_edit.text().strip(), FITS_DATA.shape, getattr(vw, 'fits_ext', 0))
            for c in cands:
                if c['is_sci']:
                    continue
                tag = '' if c['match'] else '  — shape mismatch'
                kind = f"  [{HyperCube_Noise.KIND_LABELS[c['kind']]}]" if c['kind'] else ''
                ext_combo.addItem(f"{c['ext']}: {c['name']}  {tuple(c['shape'])}{kind}{tag}",
                                  (c['ext'], c['kind'], c['match']))
            if ext_combo.count() == 0:
                ext_combo.addItem('(no image extensions found)', None)
            # Prefer the current selection, else the first shape-matching guess.
            want_ext = ERROR_INFO.get('ext')
            for i in range(ext_combo.count()):
                data = ext_combo.itemData(i)
                if data and want_ext is not None and data[0] == want_ext:
                    ext_combo.setCurrentIndex(i)
                    return
            for i in range(ext_combo.count()):
                data = ext_combo.itemData(i)
                if data and data[2] and data[1]:
                    ext_combo.setCurrentIndex(i)
                    return

        def sync_kind():
            data = ext_combo.currentData()
            if data and data[1]:
                idx = kind_combo.findData(data[1])
                if idx >= 0:
                    kind_combo.setCurrentIndex(idx)

        def pick_file():
            fn, _ = QFileDialog.getOpenFileName(
                dlg, 'Select an error / variance cube', os.path.dirname(path_edit.text() or ''),
                'FITS Files (*.fits *.fits.gz *.fit);;All Files (*)')
            if fn:
                path_edit.setText(fn)
                repopulate()
                sync_kind()

        browse.clicked.connect(pick_file)
        ext_combo.currentIndexChanged.connect(sync_kind)
        path_edit.editingFinished.connect(repopulate)

        def sync_enabled():
            for w in (path_edit, browse, ext_combo, kind_combo):
                w.setEnabled(rb_cube.isChecked())
        rb_cube.toggled.connect(sync_enabled)

        repopulate()
        if ERROR_CUBE is not None:
            rb_cube.setChecked(True)
            idx = kind_combo.findData(ERROR_INFO.get('kind'))
            if idx >= 0:
                kind_combo.setCurrentIndex(idx)
        else:
            rb_emp.setChecked(True)
            sync_kind()
        sync_enabled()

        btns = QHBoxLayout()
        btns.addStretch()
        cancel_btn = QPushButton('Cancel')
        apply_btn = QPushButton('Apply')
        apply_btn.setDefault(True)
        btns.addWidget(cancel_btn); btns.addWidget(apply_btn)
        lay.addLayout(btns)
        cancel_btn.clicked.connect(dlg.reject)

        def apply_choice():
            if rb_emp.isChecked():
                set_error_cube(None, {'mode': 'empirical',
                                      'source': HyperCube_Noise.SRC_EMPIRICAL})
                print('Measurement errors: empirical (line-free continuum, DER_SNR)')
                dlg.accept()
                return
            data = ext_combo.currentData()
            if not data:
                QMessageBox.warning(dlg, 'Measurement Errors',
                                    'No usable extension is selected.')
                return
            spec = {'mode': 'ext' if path_edit.text().strip() == str(getattr(vw, 'fits_path', ''))
                    else 'file',
                    'path': path_edit.text().strip(), 'ext': int(data[0]),
                    'kind': kind_combo.currentData(),
                    'label': os.path.basename(path_edit.text().strip())}
            try:
                info = vw.apply_error_cube_spec(spec, announce=True)
            except Exception as e:
                QMessageBox.critical(dlg, 'Measurement Errors', str(e))
                return
            QMessageBox.information(
                dlg, 'Measurement Errors',
                f"Using {info['source']}\n"
                f"{100 * info.get('good_fraction', 0):.1f}% of pixels have a usable σ.")
            dlg.accept()

        apply_btn.clicked.connect(apply_choice)
        dlg.exec_()

    def save_cube_fit(self):
        global df, df_cont, df_obs, df_fit, df_stellar
        """Save csv storing galaxy, observation, and cube fit data"""
        print('Saving cube fit to csv file')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
    
                # Write df_obs (Observation information)
                writer.writerow(df_obs.columns)  # Header for df_obs
                writer.writerows(df_obs.values)  # Data for df_obs
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                writer.writerow(['wavelength scale factor', 'flux scale factor',
                                 'flux unit', 'wavelength unit', 'velocity unit'])

                writer.writerow([self.viewer_window.WLscalefactor,
                                 self.viewer_window.fluxscalefactor,
                                 flux_unit_str(), 'Angstrom', 'km/s'])
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                # Write df_cont (continuum information)
                _cont_out = to_export_units(df_cont)
                writer.writerow(_cont_out.columns)  # Header for df_cont
                writer.writerows(_cont_out.values)  # Data for df_cont
    
                # Insert an empty row for separation
                writer.writerow([])
    
                # Write df (line information); σ in km/s, units in the names
                _df_out = to_export_units(df)
                writer.writerow(_df_out.columns)  # Header for df
                writer.writerows(_df_out.values)  # Data for df

                # Insert an empty row for separation
                writer.writerow([])

                # Write df_fit (Fitting results); σ in km/s, units in the names
                _fit_out = to_export_units(df_fit)
                writer.writerow(_fit_out.columns)  # Header for df_fit
                writer.writerows(_fit_out.values)  # Data for df_fit

                # Write df_stellar (per-spaxel stellar kinematics), if any. This
                # extra section is optional; older loaders simply ignore it.
                if isinstance(df_stellar, pd.DataFrame) and len(df_stellar) > 0:
                    _stel_out = to_export_units(df_stellar)
                    writer.writerow([])
                    writer.writerow(_stel_out.columns)
                    writer.writerows(_stel_out.values)

            print(f"CSV File saved: {file_path}")
            
            
        
    def load_cube_fit(self):
        global df, df_cont, df_fit, df_obs, viewing_fit, df_stellar
        """Open a CSV file and load it into the dataframes"""
        print('opening file')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        if file_path:
            with open(file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                rows = [row for row in reader]  # Don't remove empty rows yet
    
            # Find all the empty rows that separate the dataframes
            split_indices = [i for i, row in enumerate(rows) if not any(row)]
            
            if len(split_indices) < 3:
                raise ValueError("CSV file doesn't have enough sections separated by blank rows")
            
            # Read df_obs (first section)
            df_obs = pd.DataFrame(rows[:split_indices[0]][1:], columns=rows[0])  # Exclude header row from data
            df_obs = df_obs.apply(pd.to_numeric, errors='ignore')  # Convert numeric columns if possible
    
            self.source_name_button.setText('Source: '+str(df_obs['sourcename'].item()))
            self.source_redshift_button.setText('z: '+str(df_obs['redshift'].item()))
            self.resolving_power_button.setText('R: '+str(df_obs['resolvingpower'].item()))
            
            df_scale_header_index = split_indices[0] + 1
            while df_scale_header_index < len(rows) and not any(rows[df_scale_header_index]):
                df_scale_header_index += 1
            
            df_scale_data_start = df_scale_header_index + 1
            df_scale_data_end = split_indices[1]
            
            df_scale = pd.DataFrame(rows[df_scale_data_start:df_scale_data_end], 
                                  columns=rows[df_scale_header_index])
            df_scale = df_scale.apply(pd.to_numeric, errors='ignore')
            
            
            
            # Read df_cont (second section, after first empty row)
            df_cont_header_index = split_indices[1] + 1
            while df_cont_header_index < len(rows) and not any(rows[df_cont_header_index]):
                df_cont_header_index += 1
                
            df_cont_data_start = df_cont_header_index + 1
            df_cont_data_end = split_indices[2]
            
            df_cont = pd.DataFrame(rows[df_cont_data_start:df_cont_data_end],
                                  columns=rows[df_cont_header_index])
            df_cont = df_cont.apply(pd.to_numeric, errors='ignore')
            df_cont = from_export_units(df_cont)

            # Normalise the spline columns. Older CSVs predate them → default to
            # linear with empty knots; for spline rows parse the "[...]" strings
            # back into float lists.
            if 'cont_type' not in df_cont.columns:
                df_cont['cont_type'] = 'linear'
            df_cont['cont_type'] = df_cont['cont_type'].fillna('linear').replace(
                {'': 'linear', 'nan': 'linear'})
            for _kcol in ('knots_x', 'knots_y_0', 'knots_y_fit',
                          'poly_coef_0', 'poly_coef_fit'):
                if _kcol not in df_cont.columns:
                    df_cont[_kcol] = [[] for _ in range(len(df_cont))]
                else:
                    df_cont[_kcol] = df_cont[_kcol].apply(_as_float_list)
            if 'poly_degree' not in df_cont.columns:
                df_cont['poly_degree'] = np.nan

            # Read df (third section, after second empty row)
            df_header_index = split_indices[2] + 1
            while df_header_index < len(rows) and not any(rows[df_header_index]):
                df_header_index += 1
                
            df_data_start = df_header_index + 1
            df_data_end = split_indices[3]
            
            df = pd.DataFrame(rows[df_data_start:df_data_end],
                             columns=rows[df_header_index])
            df = df.apply(pd.to_numeric, errors='ignore')
            # Back to internal units/names (σ → Å); accepts older CSVs too.
            df = from_export_units(df)
            
            # Read df_fit (fourth section, after third empty row)
            df_fit_header_index = split_indices[3] + 1
            while df_fit_header_index < len(rows) and not any(rows[df_fit_header_index]):
                df_fit_header_index += 1
                
            df_fit_data_start = df_fit_header_index + 1
            # Bound df_fit by the next blank row if an extra (df_stellar) section
            # follows; otherwise read to the end (old format).
            df_fit_data_end = split_indices[4] if len(split_indices) >= 5 else len(rows)

            df_fit = pd.DataFrame(rows[df_fit_data_start:df_fit_data_end],
                                 columns=rows[df_fit_header_index])
            df_fit = df_fit.apply(pd.to_numeric, errors='ignore')
            # Back to internal units/names (σ → Å); accepts older CSVs too.
            df_fit = from_export_units(df_fit)

            # Optional df_stellar section (per-spaxel stellar kinematics → maps).
            df_stellar = pd.DataFrame({})
            if len(split_indices) >= 5:
                _sh = split_indices[4] + 1
                while _sh < len(rows) and not any(rows[_sh]):
                    _sh += 1
                if _sh < len(rows):
                    df_stellar = pd.DataFrame(rows[_sh + 1:], columns=rows[_sh])
                    df_stellar = df_stellar.apply(pd.to_numeric, errors='ignore')
                    df_stellar = from_export_units(df_stellar)
            
            if 'RA' not in df_fit.columns:
                df_fit['RA'] = self.viewer_window.pixel_to_ra_dec(df_fit['spaxel_x'],df_fit['spaxel_y'])[0]
                df_fit['Dec'] = self.viewer_window.pixel_to_ra_dec(df_fit['spaxel_x'],df_fit['spaxel_y'])[1]
            
            # Spline/poly continuum columns hold lists / type strings, not
            # scalars — keep them out of the numeric coercion below (which
            # strips '[]'). poly_degree stays numeric (handled normally).
            _spline_cols = [c for c in df_fit.columns
                            if ('knots' in c) or ('poly_coef' in c) or c.endswith('_cont_type')]

            # Define column groups for conversion
            float_columns = [
                col for col in df_fit.columns
                if col not in (["color", "success", "LineName", "LineID",
                                "spaxel_x", "spaxel_y", "region_ID"] + _spline_cols)
            ]

            int_columns = ["LineID", "spaxel_x", "spaxel_y", "region_ID"]

            # Replace "nan" and "NaN" strings with actual NaNs (skip the spline
            # list/type columns so their "[...]" / 'spline' values survive).
            _nonspline = [c for c in df_fit.columns if c not in _spline_cols]
            df_fit[_nonspline] = df_fit[_nonspline].replace(["nan", "NaN"], np.nan)

            # Convert columns to float; strip numpy-array brackets (e.g. '[1.23]')
            # that can appear when a column was saved with array-valued cells.
            def _to_float(s):
                return pd.to_numeric(
                    s.astype(str).str.strip().str.strip('[]'), errors='coerce'
                )
            df_fit[float_columns] = df_fit[float_columns].apply(_to_float)
            # Parse the spline knot / poly coefficient columns back into lists.
            for _kc in _spline_cols:
                if ('knots' in _kc) or ('poly_coef' in _kc):
                    df_fit[_kc] = df_fit[_kc].apply(_as_float_list)
            
            # Convert integer columns (use Int64 dtype to allow NaNs)
            df_fit[int_columns] = df_fit[int_columns].astype("Int64")
            
            # Remove any current actors from the spectrum plot
            # for frame_id in np.unique(df_cont['region_ID']):
            #     df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
            # self.viewer_window.spectrum_canvas.draw()
            
            # Rebuild the fit-params frames + spectrum overlays from the
            # dataframes just loaded.
            self.rebuild_fit_panel(show_fit=True)
            print(f"File loaded: {file_path}")

    def rebuild_fit_panel(self, show_fit=True, x=None, y=None):
        """Rebuild the fit-params frames (and spectrum fit overlays) from the
        current global df_cont / df / df_fit. Shared by Load Fit and Load
        Session so both repopulate the panel identically.

        x, y optionally select which spaxel's fit to draw in the spectrum
        (defaults to the first fitted spaxel for backward compatibility).
        """
        global df, df_cont, df_fit, df_obs, viewing_fit
        # Clear existing frames
        for i in reversed(range(self.scroll_layout.count())):  # reverse to avoid index shift
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()

        # Rebuild the toolbar + region cards
        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        have_fit = (show_fit and isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0)
        # Which spaxel's fit to overlay on the spectrum
        if have_fit:
            _fx = x if x is not None else df_fit.iloc[0]['spaxel_x']
            _fy = y if y is not None else df_fit.iloc[0]['spaxel_y']
        if len(df_cont) > 0:
            for ID in np.unique(df_cont['region_ID']):
                ID = np.int64(ID)
                self.add_spectral_frame('Spectral Region '+str(ID+1), df_cont, df, ID, addframe=False)
                if have_fit:
                    self.viewer_window.rebuild_plot(
                        region_ID=ID, from_file=True, show_init=False, show_fit=True,
                        x=_fx, y=_fy)
        if have_fit:
            self.fitloaded = True
            viewing_fit = True



    def line(self, x, m, b):
        return m*x+b
    
    # Define a Gaussian function
    def gaussian_function(self, x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def generalized_model(self, x, m, b, *gaussian_params):
        """
        Model combining a linear continuum and multiple Gaussian components.
    
        Parameters:
        - x: ndarray, input x values
        - m: float, slope of the linear continuum
        - b: float, intercept of the linear continuum
        - *gaussian_params: list, flattened parameters for each Gaussian in the form (A1, mu1, sigma1, A2, mu2, sigma2, ...)
    
        Returns:
        - y: ndarray, model values
        """
        # Linear component
        y = self.line(x, m, b)
        
        # Add each Gaussian
        num_gaussians = len(gaussian_params) // 3
        for i in range(num_gaussians):
            A = gaussian_params[3 * i]
            mu = gaussian_params[3 * i + 1]
            sigma = gaussian_params[3 * i + 2]
            y += self.gaussian_function(x, A, mu, sigma)
        
        return y
    


    def _stellar_baseline_total(self, spaxel):
        """Sum of all stellar regions' baselines over `wavelengths` for `spaxel`
        (zero outside each stellar region's range, and zero if no stellar region)."""
        total = np.zeros(np.asarray(wavelengths, float).shape)
        if not isinstance(df_cont, pd.DataFrame) or 'cont_type' not in df_cont.columns:
            return total
        xy = (int(spaxel[0]), int(spaxel[1])) if spaxel is not None else None
        lam = np.asarray(wavelengths, float)
        for rid in df_cont.loc[df_cont['cont_type'] == 'stellar', 'region_ID']:
            region = df_cont.loc[np.int64(df_cont['region_ID']) == np.int64(rid)]
            x1 = float(region['x1'].item()); x2 = float(region['x2'].item())
            y = np.nan_to_num(self.viewer_window._region_baseline(
                region, lam, use_fit=False, xy=xy))
            mask = (lam >= x1) & (lam <= x2)
            total[mask] += y[mask]
        return total

    def _fit_quality_metrics(self, spectrum, wavelengths, result, flux_scale):
        """Calibrated, scale-free goodness-of-fit statistics for one spaxel.

        The native rchisq is an unweighted, flux-rescaled sum-of-squares with no
        noise model, so it is ~constant offset from a true reduced chi-square and
        not comparable across spaxels of different brightness. These metrics fix
        that without touching the fit:

        - qa_chisq_cont : reduced chi-square over off-line (continuum) pixels,
          normalised by a robust per-spaxel noise estimate. ~1 for a good fit.
        - qa_core_cont_ratio : mean(resid^2) in the line cores / mean(resid^2)
          in the continuum. Noise-independent (scale-free); ~1 good, >>1 = the
          line profile is poorly fit (e.g. a missed peak).
        - qa_signed_resid_z : signed summed residual in the cores / (noise*sqrt N);
          large positive = model under-predicts (missed flux), negative = over.
        - qa_runs_z : standardised Wald-Wolfowitz runs statistic on the sign of
          the core residuals; large |z| = systematic shape error.
        """
        nan = float('nan')
        out = {'qa_noise': nan, 'qa_chisq_cont': nan, 'qa_chisq_core': nan,
               'qa_core_cont_ratio': nan, 'qa_signed_resid_z': nan, 'qa_runs_z': nan}
        try:
            model = np.asarray(result.best_fit, dtype=float) * flux_scale
            lam = np.asarray(wavelengths, dtype=float)
            data = np.asarray(spectrum, dtype=float)
            if model.shape != data.shape or model.shape != lam.shape:
                return out
            resid = data - model
            finite = np.isfinite(resid) & np.isfinite(lam)

            # In-fit mask = union of the continuum regions' wavelength spans.
            in_fit = np.zeros_like(lam, dtype=bool)
            nreg = len(df_cont)
            for r in range(1, nreg + 1):
                ks, ke = f'x{r}_start', f'x{r}_end'
                if ks in result.params and ke in result.params:
                    x0 = float(result.params[ks].value)
                    x1 = float(result.params[ke].value)
                    in_fit |= (lam >= min(x0, x1)) & (lam <= max(x0, x1))
            if not in_fit.any():
                in_fit = finite.copy()
            in_fit &= finite

            # Core mask = (a) ±2.5σ around the fitted centroid UNION
            # (b) the full allowed centroid range ±2.5·σ_pad.
            # Including (b) catches missed peaks that sit inside the
            # parameter bounds but outside the fitted window — without
            # it a fit that completely misses the line scores as "good".
            core = np.zeros_like(lam, dtype=bool)
            nlines = len(np.unique(df['Line_ID'])) if 'Line_ID' in df.columns else 0
            for i in range(1, nlines + 1):
                ck, sk = f'cen{i}', f'sigma{i}'
                if ck in result.params and sk in result.params:
                    cen = float(result.params[ck].value)
                    sig = abs(float(result.params[sk].value))
                    if np.isfinite(cen) and np.isfinite(sig) and sig > 0:
                        core |= np.abs(lam - cen) <= 2.5 * sig
                        # Expand to allowed centroid window (part b).
                        p_cen = result.params[ck]
                        cen_lo = (float(p_cen.min)
                                  if p_cen.min is not None and np.isfinite(float(p_cen.min))
                                  else cen)
                        cen_hi = (float(p_cen.max)
                                  if p_cen.max is not None and np.isfinite(float(p_cen.max))
                                  else cen)
                        p_sig = result.params[sk]
                        sig_lo = (float(p_sig.min)
                                  if p_sig.min is not None and np.isfinite(float(p_sig.min))
                                  else sig)
                        sig_pad = max(sig, sig_lo) * 2.5
                        core |= (lam >= cen_lo - sig_pad) & (lam <= cen_hi + sig_pad)
            core &= in_fit
            cont = in_fit & ~core

            r_core, r_cont = resid[core], resid[cont]
            n_core, n_cont = r_core.size, r_cont.size
            if n_cont >= 5:
                noise = 1.4826 * np.median(np.abs(r_cont - np.median(r_cont)))
                out['qa_noise'] = float(noise)
                if noise > 0:
                    out['qa_chisq_cont'] = float(np.mean(r_cont**2) / noise**2)
                    if n_core >= 1:
                        out['qa_chisq_core'] = float(np.mean(r_core**2) / noise**2)
                        out['qa_signed_resid_z'] = float(
                            np.sum(r_core) / (noise * np.sqrt(n_core)))
                mc = np.mean(r_cont**2)
                if n_core >= 1 and mc > 0:
                    out['qa_core_cont_ratio'] = float(np.mean(r_core**2) / mc)

            # Wald-Wolfowitz runs test on the sign of the core residuals.
            if n_core >= 10:
                signs = np.sign(r_core)
                signs = signs[signs != 0]
                npos = int((signs > 0).sum()); nneg = int((signs < 0).sum())
                ntot = npos + nneg
                if npos > 0 and nneg > 0 and ntot > 1:
                    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
                    mu = 2.0 * npos * nneg / ntot + 1.0
                    var = (2.0 * npos * nneg * (2.0 * npos * nneg - ntot)
                           / (ntot**2 * (ntot - 1)))
                    if var > 0:
                        out['qa_runs_z'] = float((runs - mu) / np.sqrt(var))
        except Exception as e:
            print(f"QA-metrics warning: {type(e).__name__}: {e}")
        return out

    def fit_spaxel(self, z, max_nfev=256, params_to_use=None):
        """Serial single-spaxel fit. Thin wrapper around the shared, Qt-free
        kernel HyperCube_fit.fit_one_spaxel so the serial and parallel ("Fit
        Cube") paths produce identical results. Appends result rows to the
        global fit_results (used by single-spaxel fits and Rectify)."""
        global fit_results
        vw = self.viewer_window
        # A 1D spectrum has no spaxel grid; treat it as the single spaxel (0, 0).
        if vw.current_spaxel is None and vw.is_1d_spectrum:
            vw.current_spaxel = (0, 0)
        cx, cy = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])
        if vw.is_1d_spectrum is False:
            spectrum = vw.get_spectrum_at_spaxel(cx, cy)
        else:
            spectrum = FITS_DATA

        # Stellar baseline for this spaxel (zeros if no stellar region). This
        # also lazily recomputes/repopulates STELLAR_CACHE for the spaxel.
        stellar_baseline = self._stellar_baseline_total((cx, cy))
        try:
            _ra, _dec = vw.pixel_to_ra_dec(cx, cy)
        except Exception:
            _ra, _dec = np.nan, np.nan

        # Per-region stellar kinematics for the result rows (from the cache the
        # baseline step just ensured).
        stellar_kin = {}
        if isinstance(df_cont, pd.DataFrame) and 'cont_type' in df_cont.columns:
            for rid in df_cont.loc[df_cont['cont_type'] == 'stellar', 'region_ID']:
                rid = int(np.int64(rid))
                sc = STELLAR_CACHE.get((cx, cy, rid))
                if sc is not None:
                    stellar_kin[rid] = sc

        rows = HyperCube_fit.fit_one_spaxel(
            spectrum, stellar_baseline, wavelengths, params_to_use,
            piecewise_model, df, df_cont, z, max_nfev,
            getattr(self, '_sequential_fit', False), (cx, cy), _ra, _dec,
            stellar_kin, sigma_at_spaxel(cx, cy), noise_source_label())
        fit_results.extend(rows)



    def calculate_snr_map(self, linewl, Nsigma, search_window_width=None, continuum_offset=None, continuum_width=None):
        """
        Calculate an SNR map using wavelength units (same as 'wavelengths' array).
        
        Parameters:
            linewl (float or array-like): Central wavelength(s) of the emission line(s) 
            Nsigma (float): SNR threshold for contour drawing
            search_window_width (float): Width of search window around line center (in wavelength units)
            continuum_offset (float): Offset from line center to start continuum regions (in wavelength units)
            continuum_width (float): Width of continuum regions (in wavelength units)
        
        Returns:
            snr_map (numpy.ndarray): 2D array of max SNR values across emission lines for each spaxel
        """
        global snr_map
        num_wavelengths, nx, ny = np.shape(FITS_DATA)
        snr_map = np.zeros((nx, ny))
        
        # Set default window sizes if not specified
        if search_window_width is None:
            search_window_width = 50 * np.median(np.diff(wavelengths))  # 10x wavelength step
        
        if continuum_offset is None:
            continuum_offset = 60 * np.median(np.diff(wavelengths))  # 20x wavelength step
        
        if continuum_width is None:
            continuum_width = 70 * np.median(np.diff(wavelengths))  # 30x wavelength step
    
        print(f'Calculating S/N map for line_wl = {linewl}, masking at {Nsigma}-sigma!')
    
        for i in range(nx):
            for j in range(ny):
                spectrum = FITS_DATA[:, i, j]
                snr_values = []
                
                for index, row in df.iterrows():
                    line_center = row['Centroid_0']  # Line center in wavelength units
                    
                    # Define line region
                    line_mask = (wavelengths >= line_center - search_window_width/2) & \
                               (wavelengths <= line_center + search_window_width/2)
                    line_region = spectrum[line_mask]
                    
                    # Use robust maximum for peak flux
                    peak_flux = np.percentile(line_region, 95) if len(line_region) > 0 else 0
                    
                    # Define continuum regions (both sides of line)
                    left_cont_mask = (wavelengths >= line_center - continuum_offset - continuum_width) & \
                                    (wavelengths <= line_center - continuum_offset)
                    right_cont_mask = (wavelengths <= line_center + continuum_offset + continuum_width) & \
                                     (wavelengths >= line_center + continuum_offset)
                    
                    # Combine continuum regions
                    continuum_flux = np.concatenate([
                        spectrum[left_cont_mask],
                        spectrum[right_cont_mask]
                    ])
                    
                    # Compute noise as MAD scaled to STD (robust against outliers)
                    if len(continuum_flux) > 1:
                        noise_level = 1.4826 * np.median(np.abs(continuum_flux - np.median(continuum_flux)))
                    else:
                        noise_level = np.nan
                    
                    # Compute SNR if valid
                    if noise_level > 0 and not np.isnan(noise_level):
                        snr = peak_flux / noise_level
                        snr_values.append(snr)
                    else:
                        snr_values.append(0)
                
                # Store maximum SNR for this spaxel
                snr_map[i, j] = np.nanmax(snr_values) if snr_values else 0
    
        # Draw the contour on the viewer window at the specified Nsigma level
        contour = self.viewer_window.ax.contour(snr_map, levels=[Nsigma], colors='red', linewidths=1.5)
    
        # Update the canvas
        self.viewer_window.canvas.draw()
        self.viewer_window.spectrum_canvas.draw_idle()
        
        return snr_map




    def clear_spaxel_fit(self):
        try:
            self._clear_spaxel_fit_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Error', f'Could not clear spaxel fit:\n\n{type(e).__name__}: {e}')

    def _edit_mode(self):
        """True when the panel is editing a single spaxel's per-spaxel model."""
        vw = getattr(self, 'viewer_window', None)
        return vw is not None and getattr(vw, '_edit_spaxel', None) is not None

    def _ask_edit_seed(self, have_fit):
        """Ask how to seed the editable values when entering edit mode.
        Returns 'fit', 'base', or None (cancelled)."""
        from PyQt5.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle('Edit Spaxel Model')
        box.setText('Seed the editable initial guesses from:')
        fit_btn  = box.addButton("This spaxel's fit", QMessageBox.AcceptRole)
        base_btn = box.addButton('Base template',     QMessageBox.AcceptRole)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        if not have_fit:
            fit_btn.setEnabled(False)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is cancel_btn:
            return None
        return 'fit' if clicked is fit_btn else 'base'

    def _seed_override_from_fit(self, ov_cont, ov_df, fit_rows):
        """Copy this spaxel's fitted values into the override's initial-guess
        columns (line amp/cen/sigma and per-region continuum params)."""
        # Lines
        for _, fr in fit_rows.iterrows():
            lid = _safe_float(fr.get('LineID'))
            if not np.isfinite(lid):
                continue
            mask = ov_df['Line_ID'].astype(float) == lid
            if np.isfinite(_safe_float(fr.get('amp_fit'))):
                ov_df.loc[mask, 'Amp_0'] = float(fr['amp_fit'])
            if np.isfinite(_safe_float(fr.get('cen_fit'))):
                ov_df.loc[mask, 'Centroid_0'] = float(fr['cen_fit'])
            if np.isfinite(_safe_float(fr.get('sigma_fit'))):
                ov_df.loc[mask, 'Sigma_0'] = float(fr['sigma_fit'])
        # Continuum (region index N is 1-based, matching fit_spaxel's cont_region{N}_)
        for p in range(len(ov_cont)):
            idx = ov_cont.index[p]
            rid = ov_cont.iloc[p]['region_ID']
            N = p + 1
            if 'region_ID' in fit_rows.columns:
                rrows = fit_rows[fit_rows['region_ID'].astype(float) == _safe_float(rid)]
            else:
                rrows = fit_rows
            if len(rrows) == 0:
                continue
            rr = rrows.iloc[0]
            pfx = f'cont_region{N}_'
            ctype = str(rr.get(pfx + 'cont_type', 'linear'))
            if ctype == 'spline':
                ov_cont.at[idx, 'cont_type'] = 'spline'
                if (pfx + 'knots_x') in rr:
                    ov_cont.at[idx, 'knots_x'] = _as_float_list(rr[pfx + 'knots_x'])
                if (pfx + 'knots_y_fit') in rr:
                    ov_cont.at[idx, 'knots_y_0'] = _as_float_list(rr[pfx + 'knots_y_fit'])
            elif ctype == 'poly':
                ov_cont.at[idx, 'cont_type'] = 'poly'
                if (pfx + 'poly_coef_fit') in rr:
                    ov_cont.at[idx, 'poly_coef_0'] = _as_float_list(rr[pfx + 'poly_coef_fit'])
                if (pfx + 'poly_degree') in rr:
                    ov_cont.at[idx, 'poly_degree'] = _safe_float(rr[pfx + 'poly_degree'])
            else:
                ov_cont.at[idx, 'cont_type'] = 'linear'
                if (pfx + 'slope_fit') in rr:
                    ov_cont.at[idx, 'Slope_0'] = _safe_float(rr[pfx + 'slope_fit'])
                if (pfx + 'intercept_fit') in rr:
                    ov_cont.at[idx, 'Intercept_0'] = _safe_float(rr[pfx + 'intercept_fit'])

    def _clear_spaxel_fit_impl(self):
        global df_fit, df_cont, df, viewing_fit, base_df_cont, base_df, spaxel_overrides

        vw = self.viewer_window
        if vw.current_spaxel is None:
            return
        cx, cy = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])

        # Capture the base template lazily (e.g. a fit/session predating this).
        if base_df_cont is None:
            base_df_cont = df_cont.copy(deep=True)
            base_df = df.copy(deep=True)
            if 'lineactor' in base_df_cont.columns:
                base_df_cont['lineactor'] = None
            if 'curveactor' in base_df.columns:
                base_df['curveactor'] = None

        # Snapshot this spaxel's fit rows before dropping them (for 'seed from fit').
        fit_rows = None
        if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0 and 'spaxel_x' in df_fit.columns:
            fit_rows = df_fit[(df_fit['spaxel_x'] == cx) & (df_fit['spaxel_y'] == cy)].copy()

        seed = self._ask_edit_seed(have_fit=(fit_rows is not None and len(fit_rows) > 0))
        if seed is None:
            return  # cancelled

        # Stash originals so "Cancel Edit" can restore the pre-Clear state.
        vw._edit_orig_fit_rows = fit_rows.copy() if fit_rows is not None else None
        vw._edit_prev_override = spaxel_overrides.get((cx, cy))

        # Build the override: a base copy whose values may be re-seeded from the fit.
        ov_cont = base_df_cont.copy(deep=True)
        ov_df   = base_df.copy(deep=True)
        if 'lineactor' in ov_cont.columns:
            ov_cont['lineactor'] = None
        if 'curveactor' in ov_df.columns:
            ov_df['curveactor'] = None
        if seed == 'fit' and fit_rows is not None and len(fit_rows) > 0:
            self._seed_override_from_fit(ov_cont, ov_df, fit_rows)

        spaxel_overrides[(cx, cy)] = {'df_cont': ov_cont, 'df': ov_df}

        # Drop this spaxel's rows from df_fit
        if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0 and 'spaxel_x' in df_fit.columns:
            df_fit = df_fit[~((df_fit['spaxel_x'] == cx) & (df_fit['spaxel_y'] == cy))].reset_index(drop=True)

        # Activate the override and enter edit mode (lock so hover can't change spaxel).
        df_cont = ov_cont
        df = ov_df
        vw._edit_spaxel = (cx, cy)
        vw.locked = True
        if hasattr(vw, 'red_rect'):
            vw.red_rect.set_edgecolor('#cc2222')
            vw.canvas.draw_idle()

        # Grey out the existing (bad) fit curves in place — don't remove them
        for _line in vw.spectrum_ax.lines[:]:
            if _line.get_label() != '_child0':
                _line.set_color('#888888')
                _line.set_alpha(0.45)
        vw.spectrum_canvas.draw_idle()

        # Blue box marks the spaxel under edit
        vw._init_guess_spaxel = (cx, cy)
        if vw._blue_rect is not None:
            vw._blue_rect.set_xy((cx - 0.5, cy - 0.5))
            vw._blue_rect.set_visible(True)
            vw.canvas.draw_idle()

        if isinstance(df_fit, pd.DataFrame) and len(df_fit) == 0:
            viewing_fit = False

        # Rebuild the panel from the override (edit mode disables +Line/+Region/delete).
        self.rebuild_fit_panel(show_fit=False, x=cx, y=cy)
        # Draw the editable init-guess model OVER the greyed reference so the
        # continuum 'lineactor' exists (needed by graphical 'g' editing) and the
        # user sees the model they're about to adjust.
        for rid in np.unique(np.int64(df_cont['region_ID'])):
            vw.rebuild_plot(rid, from_file=True, show_init=True, show_fit=False, x=cx, y=cy)
        vw.spectrum_canvas.draw_idle()
        print(f"Edit mode: spaxel ({cx},{cy}). Adjust guesses, then 'Fit This Spaxel'.")

    def cancel_spaxel_edit(self):
        try:
            self._cancel_spaxel_edit_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Error', f'Could not cancel edit:\n\n{type(e).__name__}: {e}')

    def _cancel_spaxel_edit_impl(self):
        """Revert: discard the in-progress edit and restore this spaxel's
        pre-Clear fit and override."""
        global df_fit, df_cont, df, spaxel_overrides, viewing_fit
        vw = self.viewer_window
        if vw._edit_spaxel is None:
            return
        cx, cy = vw._edit_spaxel

        # Restore the override entry to its pre-Clear state.
        if vw._edit_prev_override is not None:
            spaxel_overrides[(cx, cy)] = vw._edit_prev_override
        else:
            spaxel_overrides.pop((cx, cy), None)

        # Restore the original fit rows into df_fit.
        if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0 and 'spaxel_x' in df_fit.columns:
            df_fit = df_fit[~((df_fit['spaxel_x'] == cx) & (df_fit['spaxel_y'] == cy))].reset_index(drop=True)
        if vw._edit_orig_fit_rows is not None and len(vw._edit_orig_fit_rows) > 0:
            if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0:
                df_fit = pd.concat([df_fit, vw._edit_orig_fit_rows], ignore_index=True)
            else:
                df_fit = vw._edit_orig_fit_rows.copy()
        if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0:
            viewing_fit = True

        # Exit edit mode.
        vw._edit_spaxel = None
        vw._edit_orig_fit_rows = None
        vw._edit_prev_override = None
        vw._init_guess_spaxel = None
        if getattr(vw, '_blue_rect', None) is not None:
            vw._blue_rect.set_visible(False)
            vw.canvas.draw_idle()

        # Clear edit-mode curves, then reload the spaxel's model + restored fit.
        for ln in vw.spectrum_ax.lines[:]:
            if ln.get_label() != '_child0':
                try: ln.remove()
                except ValueError: pass
        vw.gaussian_component_lines.clear()
        vw._load_spaxel_model(cx, cy)
        if getattr(vw, '_edited_overlay_on', False):
            vw._draw_edited_overlay()
        print(f"Reverted edit for spaxel ({cx},{cy}).")

    def clear_all_fits(self):
        try:
            self._clear_all_fits_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Error', f'Could not clear fits:\n\n{type(e).__name__}: {e}')

    def _clear_all_fits_impl(self):
        """Remove ALL fit results for the cube (df_fit, per-spaxel overrides),
        keeping the model definition so the user can re-fit."""
        global df_fit, fit_results, spaxel_overrides, viewing_fit, base_df_cont, base_df
        from PyQt5.QtWidgets import QMessageBox

        has_fits = isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0
        if not has_fits and not spaxel_overrides:
            QMessageBox.information(self, 'Clear All Fits', 'There are no fits to clear.')
            return
        resp = QMessageBox.question(
            self, 'Clear All Fits',
            'Remove ALL fit results for the entire cube (and per-spaxel edits)?\n'
            'The model definition is kept so you can re-fit.',
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if resp != QMessageBox.Yes:
            return

        df_fit = pd.DataFrame({})
        fit_results = []
        spaxel_overrides = {}
        viewing_fit = False
        # Stand down the per-spaxel base/override machinery: with no fits to
        # navigate, df_cont/df become the single source of truth again, so model
        # edits (e.g. deleting a region) persist instead of being reloaded from
        # base on lock/unlock/move. Fit Cube recaptures the base template.
        base_df_cont = None
        base_df = None

        vw = self.viewer_window
        # Exit edit mode if active
        vw._edit_spaxel = None
        vw._edit_orig_fit_rows = None
        vw._edit_prev_override = None

        # Remove fit/model curves from the spectrum (keep the raw step-line)
        for ln in vw.spectrum_ax.lines[:]:
            if ln.get_label() != '_child0':
                try: ln.remove()
                except ValueError: pass
        vw.gaussian_component_lines.clear()
        if 'lineactor' in df_cont.columns:
            df_cont['lineactor'] = None
        vw.spectrum_canvas.draw_idle()

        # Hide the edited-spaxel overlay and the init-guess marker
        vw._edited_overlay_on = False
        vw._draw_edited_overlay()
        vw._init_guess_spaxel = None
        if getattr(vw, '_blue_rect', None) is not None:
            vw._blue_rect.set_visible(False)
            vw.canvas.draw_idle()

        # Rebuild the panel with no fit overlay
        self.rebuild_fit_panel(show_fit=False)
        print('Cleared all fits for the cube.')

    # ── Stellar (pPXF) continuum ─────────────────────────────────────────────

    def open_smart_constraints_dialog(self):
        """Open Smart Constraints: auto-assign K-groups, doublet flux ratios,
        and parameter bounds for every line in the model, from a chosen
        physical scenario. See HyperCube_SmartConstraints.build_plan."""
        try:
            self._open_smart_constraints_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(None, 'Smart Constraints', f'{type(e).__name__}: {e}')

    def _open_smart_constraints_impl(self):
        global df
        if len(df) == 0:
            QMessageBox.information(self, 'Smart Constraints',
                                     'No lines in the model yet — add lines first.')
            return

        dlg = QDialog(self)
        dlg.setWindowTitle('Smart Constraints')
        dlg.setMinimumWidth(500)
        v = QVBoxLayout(dlg)

        # ── Scenario ─────────────────────────────────────────────────────
        v.addWidget(QLabel('<b>Physical scenario:</b>'))
        scenario_group = QtWidgets.QButtonGroup(dlg)
        scenario_buttons = {}
        SCENARIO_LABELS = [
            ('quiescent', 'Quiescent / Star-forming — single kinematic component'),
            ('outflow', 'AGN Outflow — core + blueshifted broad component(s)'),
            ('shock', 'Shock / LINER — broad, wide velocity range'),
            ('custom', 'Custom — set toggles manually, no preset'),
        ]
        for key, label in SCENARIO_LABELS:
            rb = QtWidgets.QRadioButton(label, dlg)
            scenario_group.addButton(rb)
            scenario_buttons[key] = rb
            v.addWidget(rb)
        scenario_buttons['quiescent'].setChecked(True)

        # ── Toggles ──────────────────────────────────────────────────────
        v.addWidget(QLabel('<b>Options:</b>'))
        kgroup_cb = QCheckBox('Assign K-groups (core → K1, secondary → K2/K3)', dlg)
        v.addWidget(kgroup_cb)

        ion_row = QHBoxLayout()
        ion_cb = QCheckBox('Split secondary K-group by ionization potential, threshold (eV):', dlg)
        ion_edit = QLineEdit(str(hcsc.DEFAULT_IP_THRESHOLD_EV), dlg)
        ion_edit.setFixedWidth(60)
        ion_row.addWidget(ion_cb); ion_row.addWidget(ion_edit); ion_row.addStretch()
        v.addLayout(ion_row)

        rel_cb = QCheckBox('Add relational sigma/amp bounds for secondary components '
                            '(broader + fainter than core)', dlg)
        v.addWidget(rel_cb)

        bounds_cb = QCheckBox('Set absolute sigma / centroid bounds per scenario', dlg)
        v.addWidget(bounds_cb)

        dens_row = QHBoxLayout()
        dens_row.addWidget(QLabel('Density-sensitive doublets ([S II], [O II]):'))
        dens_group = QtWidgets.QButtonGroup(dlg)
        dens_fix = QtWidgets.QRadioButton('Fix to low-density limit', dlg)
        dens_float = QtWidgets.QRadioButton('Float (bounded)', dlg)
        dens_group.addButton(dens_fix); dens_group.addButton(dens_float)
        dens_row.addWidget(dens_fix); dens_row.addWidget(dens_float); dens_row.addStretch()
        v.addLayout(dens_row)

        balmer_row = QHBoxLayout()
        balmer_row.addWidget(QLabel('Balmer decrement (Hα/Hβ):'))
        balmer_group = QtWidgets.QButtonGroup(dlg)
        balmer_float = QtWidgets.QRadioButton('Float (measure reddening)', dlg)
        balmer_fix = QtWidgets.QRadioButton('Fix to case B (2.86)', dlg)
        balmer_group.addButton(balmer_float); balmer_group.addButton(balmer_fix)
        balmer_row.addWidget(balmer_float); balmer_row.addWidget(balmer_fix); balmer_row.addStretch()
        v.addLayout(balmer_row)

        note = QLabel(
            '[N II] 6548/6584 and [O III] 4959/5007 doublet ratios are always fixed '
            'automatically when both lines are present (set by atomic transition '
            'probabilities, not gas conditions) — not affected by these toggles.', dlg)
        note.setWordWrap(True)
        note.setStyleSheet('color: gray; font-size: 11px;')
        v.addWidget(note)

        def _apply_scenario_defaults():
            key = next(k for k, rb in scenario_buttons.items() if rb.isChecked())
            if key == 'quiescent':
                kgroup_cb.setChecked(True); ion_cb.setChecked(False)
                rel_cb.setChecked(True); bounds_cb.setChecked(True)
                dens_fix.setChecked(True); balmer_float.setChecked(True)
            elif key in ('outflow', 'shock'):
                kgroup_cb.setChecked(True); ion_cb.setChecked(True)
                rel_cb.setChecked(True); bounds_cb.setChecked(True)
                dens_float.setChecked(True); balmer_float.setChecked(True)
            else:  # custom
                kgroup_cb.setChecked(False); ion_cb.setChecked(False)
                rel_cb.setChecked(False); bounds_cb.setChecked(False)
                dens_fix.setChecked(True); balmer_float.setChecked(True)

        dens_fix.setChecked(True)
        balmer_float.setChecked(True)
        for rb in scenario_buttons.values():
            rb.toggled.connect(lambda checked: _apply_scenario_defaults() if checked else None)
        _apply_scenario_defaults()

        # ── Preview / Apply ──────────────────────────────────────────────
        v.addWidget(QLabel('<b>Preview:</b>'))
        preview_box = QTextEdit(dlg)
        preview_box.setReadOnly(True)
        preview_box.setMinimumHeight(160)
        v.addWidget(preview_box)

        state = {'plan': None}

        def _current_settings():
            try:
                ip_threshold = float(ion_edit.text())
            except ValueError:
                ip_threshold = hcsc.DEFAULT_IP_THRESHOLD_EV
            scenario = next(k for k, rb in scenario_buttons.items() if rb.isChecked())
            return dict(
                scenario=scenario,
                split_ionization=ion_cb.isChecked(),
                density_mode='float' if dens_float.isChecked() else 'fix',
                balmer_mode='fix' if balmer_fix.isChecked() else 'float',
                assign_kgroups=kgroup_cb.isChecked(),
                add_relational_bounds=rel_cb.isChecked(),
                set_absolute_bounds=bounds_cb.isChecked(),
                ip_threshold=ip_threshold)

        def _preview():
            settings = _current_settings()
            line_lib = _load_line_library() if settings['split_ionization'] else None
            plan = hcsc.build_plan(
                df, settings['scenario'],
                split_ionization=settings['split_ionization'],
                density_mode=settings['density_mode'],
                balmer_mode=settings['balmer_mode'],
                assign_kgroups=settings['assign_kgroups'],
                add_relational_bounds=settings['add_relational_bounds'],
                set_absolute_bounds=settings['set_absolute_bounds'],
                ip_threshold=settings['ip_threshold'],
                line_library=line_lib)
            state['plan'] = plan
            preview_box.setPlainText('\n'.join(plan.summary_lines))
            apply_btn.setEnabled(True)

        def _invalidate_preview():
            state['plan'] = None
            apply_btn.setEnabled(False)
            preview_box.setPlainText('(settings changed — click Preview to refresh)')

        for w in (kgroup_cb, ion_cb, rel_cb, bounds_cb, dens_fix, dens_float,
                  balmer_float, balmer_fix):
            w.toggled.connect(_invalidate_preview)
        for rb in scenario_buttons.values():
            rb.toggled.connect(lambda checked: _invalidate_preview() if checked else None)
        ion_edit.textChanged.connect(_invalidate_preview)

        brow = QHBoxLayout()
        help_btn = QPushButton('?', dlg)
        help_btn.setFixedSize(22, 22)
        help_btn.setToolTip('What Smart Constraints does')
        help_btn.clicked.connect(lambda: self._show_smart_constraints_help(dlg))
        preview_btn = QPushButton('Preview', dlg)
        preview_btn.clicked.connect(_preview)
        apply_btn = QPushButton('Apply', dlg)
        apply_btn.setEnabled(False)
        cancel_btn = QPushButton('Cancel', dlg)
        cancel_btn.clicked.connect(dlg.reject)
        brow.addWidget(help_btn); brow.addStretch()
        brow.addWidget(preview_btn); brow.addWidget(apply_btn); brow.addWidget(cancel_btn)
        v.addLayout(brow)

        def _apply():
            if state['plan'] is None:
                return
            self._apply_smart_constraint_plan(state['plan'])
            dlg.accept()
        apply_btn.clicked.connect(_apply)

        _invalidate_preview()
        dlg.exec_()

    def _apply_smart_constraint_plan(self, plan):
        """Write a HyperCube_SmartConstraints.SmartConstraintPlan into the
        global df (bounds, doublet/relational constraints, then K-groups
        last so kinematic ties take precedence in the 5-constraint-slot
        list), then refresh the Fit Parameters panel."""
        global df
        for lid, updates in plan.bound_updates.items():
            for col, value in updates.items():
                df.loc[df['Line_ID'] == lid, col] = value
        for line_name, additions in plan.constraint_additions.items():
            self._rewrite_smart_constraints(line_name, add=additions)
        if plan.kgroup_assignments:
            for lid, group in plan.kgroup_assignments.items():
                df.loc[df['Line_ID'] == lid, 'kgroup'] = group
            self._sync_kgroup_constraints()

        x, y = None, None
        vw = self.viewer_window
        if vw is not None and getattr(vw, 'current_spaxel', None) is not None:
            x, y = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])
        self.rebuild_fit_panel(show_fit=False, x=x, y=y)

    def _show_smart_constraints_help(self, parent):
        text = (
            "<b>Smart Constraints</b><br><br>"
            "Auto-fills the same K-group / constraint / bound fields you'd "
            "otherwise set line-by-line in the Edit Line dialog, for every "
            "line in the model at once:<br><br>"
            "&nbsp;&nbsp;• <b>K-groups</b> — core (narrowest of each "
            "rest-wavelength group) → K1; broader partners → K2 (or K2/K3 "
            "split by ionization potential).<br>"
            "&nbsp;&nbsp;• <b>Atomic-fixed doublets</b> — [N II] 6548/6584 "
            "and [O III] 4959/5007 flux ratios, always fixed when both "
            "lines are present.<br>"
            "&nbsp;&nbsp;• <b>Density-sensitive doublets</b> — [S II] "
            "6716/6731, [O II] 3726/3729: fixed to the low-density limit, "
            "or floated within the physical range.<br>"
            "&nbsp;&nbsp;• <b>Balmer decrement</b> — Hα/Hβ, floated to "
            "measure reddening, or fixed to case B (2.86).<br>"
            "&nbsp;&nbsp;• <b>Relational bounds</b> — secondary components "
            "get sigma ≥ core's and amp ≤ core's (broader + fainter).<br>"
            "&nbsp;&nbsp;• <b>Absolute bounds</b> — per-scenario sigma / "
            "centroid windows (km/s).<br><br>"
            "Re-running Smart Constraints replaces its own previous output "
            "(picking a new scenario, toggling options) without touching "
            "constraints you typed by hand in the Edit Line dialog."
        )
        d = QDialog(parent)
        d.setWindowTitle('Smart Constraints — Help')
        d.setMinimumWidth(380)
        lay = QVBoxLayout(d)
        lbl = QLabel(text, d)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        lay.addWidget(lbl)
        ok = QPushButton('OK', d)
        ok.setDefault(True)
        ok.clicked.connect(d.accept)
        lay.addWidget(ok, alignment=Qt.AlignRight)
        d.exec_()

    def open_stellar_templates(self):
        """Open the Stellar Templates window for the current spaxel; on Accept,
        run pPXF and create/replace a stellar continuum region."""
        try:
            self._open_stellar_templates_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Stellar Fit', f'{type(e).__name__}: {e}')

    def _open_stellar_templates_impl(self):
        from PyQt5.QtWidgets import QMessageBox
        vw = self.viewer_window
        if hcppxf is None:
            QMessageBox.warning(self, 'Stellar Fit',
                                'pPXF is not available (pip install ppxf).')
            return
        if FITS_DATA is None or vw.current_spaxel is None:
            QMessageBox.warning(self, 'Stellar Fit',
                                'Open a cube and select a spaxel first.')
            return
        libs = hcppxf.list_libraries()
        if not libs:
            QMessageBox.warning(self, 'Stellar Fit', 'No template libraries found.')
            return

        z = _safe_float(df_obs.loc[0, 'redshift']) if len(df_obs) else np.nan
        z = 0.0 if not np.isfinite(z) else z
        lam_lo, lam_hi = float(np.nanmin(wavelengths)), float(np.nanmax(wavelengths))

        dlg = QDialog(self)
        dlg.setWindowTitle('Stellar Templates (pPXF)')
        v = QVBoxLayout(dlg)

        v.addWidget(QLabel('Template library:'))
        combo = QtWidgets.QComboBox()
        names = list(libs.keys())
        for nm in names:
            combo.addItem(f"{libs[nm]['label']}  ({libs[nm]['n_files']} spectra)", nm)
        v.addWidget(combo)
        info = QLabel(''); info.setStyleSheet('color: gray; font-size: 11px;')
        info.setWordWrap(True)
        v.addWidget(info)

        def _update_info():
            m = libs[names[combo.currentIndex()]]
            cov = (f"{m.get('lam_min',0):.0f}–{m.get('lam_max',0):.0f} Å"
                   if m.get('lam_min') else '—')
            rlo, rhi = lam_lo / (1 + z), lam_hi / (1 + z)
            covered = (m.get('lam_min') is not None
                       and m['lam_min'] <= rlo and m['lam_max'] >= rhi)
            flag = '' if covered else '   ⚠ data extends beyond template coverage'
            info.setText(
                f"{m['note']}\nTemplate coverage: {cov}   FWHM ≈ {m['fwhm']} Å\n"
                f"Data observed: {lam_lo:.0f}–{lam_hi:.0f} Å   (z = {z:g})\n"
                f"Data rest-frame: {rlo:.0f}–{rhi:.0f} Å{flag}")
        combo.currentIndexChanged.connect(_update_info); _update_info()

        # Fit range (observed Å)
        rrow = QHBoxLayout(); rrow.addWidget(QLabel('Fit range (Å):'))
        e1 = QLineEdit(f"{lam_lo:.1f}"); e2 = QLineEdit(f"{lam_hi:.1f}")
        rrow.addWidget(e1); rrow.addWidget(QLabel('to')); rrow.addWidget(e2)
        v.addLayout(rrow)

        # Moments / polynomials / sigma
        orow = QHBoxLayout()
        orow.addWidget(QLabel('Moments:'))
        mom = QtWidgets.QComboBox(); mom.addItems(['2  (V, σ)', '4  (V, σ, h3, h4)'])
        orow.addWidget(mom)
        orow.addWidget(QLabel('add. degree:'))
        adeg = QtWidgets.QSpinBox(); adeg.setRange(-1, 20); adeg.setValue(-1)
        orow.addWidget(adeg)
        orow.addWidget(QLabel('mult. degree:'))
        mdeg = QtWidgets.QSpinBox(); mdeg.setRange(0, 20); mdeg.setValue(10)
        orow.addWidget(mdeg)
        v.addLayout(orow)

        srow = QHBoxLayout()
        srow.addWidget(QLabel('σ guess (km/s):'))
        sig = QtWidgets.QDoubleSpinBox(); sig.setRange(10, 1000); sig.setValue(150)
        srow.addWidget(sig)
        maskcb = QtWidgets.QCheckBox('Mask emission lines'); maskcb.setChecked(True)
        srow.addWidget(maskcb); srow.addStretch()
        v.addLayout(srow)

        brow = QHBoxLayout(); ok = QPushButton('Fit'); ok.setDefault(True)
        cancel = QPushButton('Cancel')
        brow.addStretch(); brow.addWidget(cancel); brow.addWidget(ok); v.addLayout(brow)
        cancel.clicked.connect(dlg.reject)

        def _accept():
            try:
                r0, r1 = float(e1.text()), float(e2.text())
            except ValueError:
                QMessageBox.warning(dlg, 'Stellar Fit', 'Invalid fit range.'); return
            dlg._settings = dict(
                library=names[combo.currentIndex()], fit_range=(min(r0, r1), max(r0, r1)),
                moments=(4 if mom.currentIndex() == 1 else 2),
                degree=int(adeg.value()), mdegree=int(mdeg.value()),
                sigma_guess=float(sig.value()), mask_lines=maskcb.isChecked())
            dlg.accept()
        ok.clicked.connect(_accept)

        if dlg.exec_() != QDialog.Accepted:
            return
        self._fit_stellar_for_spaxel(dlg._settings)

    def _fit_stellar_for_spaxel(self, settings, target_rid=None):
        """Run pPXF on the current spaxel and write a stellar continuum region.
        target_rid=None appends a new region; otherwise the given region is refit."""
        global df_cont, STELLAR_CACHE, _STELLAR_LIBS
        from PyQt5.QtWidgets import QMessageBox, QApplication
        vw = self.viewer_window
        cx, cy = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])
        spectrum = np.nan_to_num(vw.get_spectrum_at_spaxel(cx, cy))
        z = _safe_float(df_obs.loc[0, 'redshift']) if len(df_obs) else 0.0
        z = 0.0 if not np.isfinite(z) else z
        R = _safe_float(df_obs.loc[0, 'resolvingpower']) if len(df_obs) else np.nan
        R = 3000.0 if not np.isfinite(R) or R <= 0 else R
        fit_range = settings['fit_range']

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            velscale, lam_rest = hcppxf.galaxy_velscale(wavelengths, z, fit_range)
            lib = _STELLAR_LIBS.get(settings['library'])
            if lib is None:
                lib = hcppxf.TemplateLibrary(settings['library']).load()
                _STELLAR_LIBS[settings['library']] = lib
            lib.prepare(velscale, R, lam_rest)
            mask = ()
            if settings['mask_lines'] and len(df) > 0 and 'Centroid_0' in df.columns:
                mask = df['Centroid_0'].astype(float).to_numpy()
            res = hcppxf.fit_stellar(
                spectrum, wavelengths, z, R, lib, fit_range=fit_range,
                mask_centroids=mask, moments=settings['moments'],
                degree=settings['degree'], mdegree=settings['mdegree'],
                sigma_guess=settings['sigma_guess'], velscale=velscale)
        finally:
            QApplication.restoreOverrideCursor()

        rid = self._write_stellar_region(settings, res, fit_range, target_rid=target_rid)
        STELLAR_CACHE[(cx, cy, int(rid))] = res['cache']
        # Clear any previous model curves so a re-fit doesn't leave stale ones.
        for ln in vw.spectrum_ax.lines[:]:
            if ln.get_label() != '_child0':
                try: ln.remove()
                except ValueError: pass
        vw.gaussian_component_lines.clear()
        if 'lineactor' in df_cont.columns:
            df_cont['lineactor'] = None
        self.rebuild_fit_panel(show_fit=False, x=cx, y=cy)
        # Redraw EVERY region's overlay (all stellar baselines + any line regions),
        # not just the new one — the clear above dropped all lineactors, so redrawing
        # only `rid` would make previously-drawn stellar templates disappear and
        # leave their lineactor None (which then breaks Gaussian placement on them).
        for _rid in df_cont['region_ID'].astype(int).unique():
            vw.rebuild_plot(int(_rid), from_file=True, show_init=True, show_fit=False, x=cx, y=cy)
        vw.spectrum_canvas.draw_idle()
        vw._mark_init_guess_spaxel()
        print(f"Stellar fit ({settings['library']}) spaxel ({cx},{cy}): "
              f"V={res['V']:.1f}  σ={res['sigma']:.1f}  χ²/dof={res['chi2']:.2f}")

    def _write_stellar_region(self, settings, res, fit_range, target_rid=None):
        """Write a stellar continuum region to df_cont and return its region_ID.

        target_rid is None  → append a NEW stellar region (so several stellar
                              templates can coexist on their own ranges, like
                              linear/poly/spline regions).
        target_rid given    → update that existing region in place (Refit Stellar).
        """
        global df_cont
        for col, default in _STELLAR_COLS.items():
            if col not in df_cont.columns:
                df_cont[col] = default
        vals = {
            'Continuum Name': 'Stellar', 'x1': round(fit_range[0], 2),
            'x2': round(fit_range[1], 2), 'Slope_0': np.nan, 'Intercept_0': np.nan,
            'Slope_fit': np.nan, 'Intercept_fit': np.nan, 'cont_type': 'stellar',
            'knots_x': [], 'knots_y_0': [], 'knots_y_fit': [], 'poly_degree': np.nan,
            'poly_coef_0': [], 'poly_coef_fit': [], 'lineactor': None,
            'stellar_library': settings['library'], 'stellar_V_0': res['V'],
            'stellar_sigma_0': res['sigma'], 'stellar_h3_0': res['h3'],
            'stellar_h4_0': res['h4'], 'stellar_scale_0': res['scale'],
            'stellar_V_fit': res['V'], 'stellar_sigma_fit': res['sigma'],
            'stellar_h3_fit': res['h3'], 'stellar_h4_fit': res['h4'],
            'stellar_scale_fit': res['scale'], 'stellar_moments': settings['moments'],
            'stellar_chi2': res['chi2'],
        }
        # Refit: update the targeted region in place.
        if target_rid is not None:
            match = df_cont.index[np.int64(df_cont['region_ID']) == np.int64(target_rid)]
            if len(match) > 0:
                idx = match[0]
                for k, val in vals.items():
                    df_cont.at[idx, k] = val
                return int(np.int64(target_rid))
        # Otherwise append a brand-new stellar region. Use max+1 (not len) so the
        # id stays unique even after regions have been deleted (ids aren't renumbered).
        rid = (int(np.int64(df_cont['region_ID']).max()) + 1
               if len(df_cont) and 'region_ID' in df_cont.columns else 0)
        vals['region_ID'] = rid
        # Column-dict construction wraps list/None values as single object cells.
        df_new = pd.DataFrame({k: [v] for k, v in vals.items()})
        # Preserve the canonical column order (region_ID stays at its usual index)
        # even when df_cont started empty — add_continuum_buttons relies on it.
        df_new = df_new.reindex(columns=df_cont.columns)
        df_cont = df_new if len(df_cont) == 0 else pd.concat([df_cont, df_new], ignore_index=True)
        return int(rid)

    def _refit_stellar_region(self, regionID):
        """Re-run pPXF for the current spaxel using this region's stored settings."""
        global df_cont
        reg = df_cont.loc[np.int64(df_cont['region_ID']) == np.int64(regionID)]
        if len(reg) == 0:
            return
        r = reg.iloc[0]
        settings = dict(
            library=str(r.get('stellar_library', '') or
                        next(iter(hcppxf.list_libraries()), '')),
            fit_range=(float(r['x1']), float(r['x2'])),
            moments=int(r['stellar_moments']) if pd.notna(r.get('stellar_moments')) else 2,
            degree=-1, mdegree=10, sigma_guess=150.0, mask_lines=True)
        self._fit_stellar_for_spaxel(settings, target_rid=int(np.int64(regionID)))

    def fit_stellar_cube(self):
        try:
            self._fit_stellar_cube_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Stellar Cube Fit', f'{type(e).__name__}: {e}')

    # ── Shared cube-fit progress + cancel ────────────────────────────────────
    def _cancel_running_fit(self):
        self._fit_cancelled = True
        self.fit_progress_label.setText('Cancelling…')

    def _begin_progress(self, total, label):
        from PyQt5.QtWidgets import QApplication
        self._fit_cancelled = False
        self.fit_progress_bar.setRange(0, max(int(total), 1))
        self.fit_progress_bar.setValue(0)
        self.fit_progress_label.setText(f"{label} 0 / {total}")
        self.fit_progress_frame.setVisible(True)
        QApplication.processEvents()

    def _update_progress(self, done, text):
        from PyQt5.QtWidgets import QApplication
        self.fit_progress_bar.setValue(int(done))
        self.fit_progress_label.setText(text)
        QApplication.processEvents()

    def _end_progress(self):
        self.fit_progress_frame.setVisible(False)

    # ── Cube-fit parallelism ────────────────────────────────────────────────
    def _fit_worker_count(self):
        """Number of parallel fit-worker processes (1 = serial). Reads the
        'Cores' spinbox if present, else defaults to CPU-1."""
        default = max(1, (os.cpu_count() or 2) - 1)
        spin = getattr(self, '_cores_spin', None)
        try:
            return int(spin.value()) if spin is not None else default
        except Exception:
            return default

    def _fit_cube_serial(self, gated, params, z, progress_bar, status_label, total):
        """Serial per-spaxel fit over `gated` spaxels (the fallback path and the
        correctness oracle). Returns the list of stellar-kinematics rows; line
        rows are appended to the global fit_results by fit_spaxel."""
        from PyQt5.QtWidgets import QApplication
        _stellar_preps = self._stellar_cube_prep()
        _stellar_rows, done = [], 0
        for (i, j) in gated:
            if self._fit_cancelled:
                break
            if self.fit_progress_frame.isVisible() and psutil.virtual_memory().percent > 80:
                QApplication.processEvents()
            self.viewer_window.current_spaxel = (i, j)
            for _prep in _stellar_preps:
                _srow = self._stellar_fit_one(i, j, _prep)
                if _srow is not None:
                    _stellar_rows.append(_srow)
            self.fit_spaxel(z, max_nfev=512, params_to_use=params)
            done += 1
            progress_bar.setValue(done)
            status_label.setText(f"Fitting spaxel {done} / {total}")
            QApplication.processEvents()
            if done % 500 == 0 and psutil.virtual_memory().percent > 80:
                gc.collect()
        return _stellar_rows

    def _fit_cube_parallel(self, gated, params, z, n_regions, n_lines, n_workers,
                           progress_bar, status_label, total):
        """Fit `gated` spaxels across `n_workers` processes. Line rows are
        appended to the global fit_results; returns the stellar-kinematics rows.

        The cube is shared read-only via shared memory; the constant context
        (params, df/df_cont, model spec, stellar specs) is pickled once per
        worker. Workers compute each spaxel's stellar fit + baseline and the
        full line fit, returning only small row dicts (kinematics-only — no
        per-spaxel optimal templates; those recompute lazily on view)."""
        global fit_results
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from PyQt5.QtWidgets import QApplication

        cube = np.ascontiguousarray(FITS_DATA)
        shm = shared_memory.SharedMemory(create=True, size=int(cube.nbytes))
        err_shm = None
        try:
            shm_arr = np.ndarray(cube.shape, dtype=cube.dtype, buffer=shm.buf)
            shm_arr[:] = cube[:]

            # Measurement errors, shared read-only the same way (float32 keeps
            # the extra footprint to half the cube's).
            err_ctx = {}
            if ERROR_CUBE is not None and ERROR_CUBE.shape == cube.shape:
                err = np.ascontiguousarray(ERROR_CUBE, dtype=np.float32)
                err_shm = shared_memory.SharedMemory(create=True, size=int(err.nbytes))
                err_arr = np.ndarray(err.shape, dtype=err.dtype, buffer=err_shm.buf)
                err_arr[:] = err[:]
                err_ctx = dict(err_shm_name=err_shm.name, err_shape=tuple(err.shape),
                               err_dtype=str(err.dtype),
                               sigma_label=noise_source_label())

            # Lightweight stellar region specs (workers load/prepare the libs).
            stellar_specs, stellar_mask = [], ()
            if 'cont_type' in df_cont.columns:
                for _, r in df_cont[df_cont['cont_type'] == 'stellar'].iterrows():
                    stellar_specs.append(dict(
                        rid=int(np.int64(r['region_ID'])),
                        library=str(r['stellar_library']),
                        fit_range=(float(r['x1']), float(r['x2'])),
                        moments=int(r['stellar_moments']) if pd.notna(r.get('stellar_moments')) else 2))
                if stellar_specs and len(df) > 0 and 'Centroid_0' in df.columns:
                    stellar_mask = df['Centroid_0'].astype(float).to_numpy()

            R = _safe_float(df_obs.loc[0, 'resolvingpower']) if len(df_obs) else np.nan
            R = 3000.0 if not np.isfinite(R) or R <= 0 else R

            # Strip unpicklable matplotlib actor columns before shipping df/df_cont.
            df_w = df.drop(columns=['curveactor'], errors='ignore').copy()
            df_cont_w = df_cont.drop(columns=['lineactor'], errors='ignore').copy()

            ctx = dict(
                shm_name=shm.name, shape=tuple(cube.shape), dtype=str(cube.dtype),
                wavelengths=np.asarray(wavelengths, float),
                params_dumps=params.dumps(), n_regions=int(n_regions),
                n_lines=int(n_lines), df=df_w, df_cont=df_cont_w, z=z, R=R,
                sequential=bool(getattr(self, '_sequential_fit', False)),
                max_nfev=512, stellar_specs=stellar_specs, stellar_mask=stellar_mask,
                **err_ctx)

            # Precompute RA/Dec for all gated spaxels (keeps WCS out of workers).
            xs = np.array([ij[0] for ij in gated])
            ys = np.array([ij[1] for ij in gated])
            try:
                ras, decs = self.viewer_window.pixel_to_ra_dec(xs, ys)
                ras = np.atleast_1d(ras); decs = np.atleast_1d(decs)
            except Exception:
                ras = np.full(len(gated), np.nan); decs = np.full(len(gated), np.nan)
            tasks = [(int(xs[k]), int(ys[k]), float(ras[k]), float(decs[k]))
                     for k in range(len(gated))]

            print(f"Parallel Fit Cube: {len(tasks)} spaxels across {n_workers} workers")
            mpctx = mp.get_context('spawn')
            _stellar_rows, done = [], 0
            # Pin BLAS to 1 thread per worker so N workers don't oversubscribe the
            # cores. Spawned children inherit the env (workers spawn lazily on the
            # first submit), so keep it pinned for the whole pool lifetime and
            # restore afterwards — setting it inside the worker is too late because
            # numpy/OpenBLAS read it at import.
            _thr_vars = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
            _thr_saved = {k: os.environ.get(k) for k in _thr_vars}
            for k in _thr_vars:
                os.environ[k] = '1'
            try:
                ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=mpctx,
                                         initializer=HyperCube_fit._worker_init,
                                         initargs=(ctx,))
                try:
                    futures = [ex.submit(HyperCube_fit._worker_fit_one, t) for t in tasks]
                    for fut in as_completed(futures):
                        if self._fit_cancelled:
                            break
                        try:
                            line_rows, srows = fut.result()
                        except Exception as e:
                            print(f"worker task error: {type(e).__name__}: {e}")
                            line_rows, srows = [], []
                        fit_results.extend(line_rows)
                        _stellar_rows.extend(srows)
                        done += 1
                        if done % 16 == 0 or done == len(tasks):
                            progress_bar.setValue(done)
                            status_label.setText(f"Fitting spaxel {done} / {total}")
                            QApplication.processEvents()
                finally:
                    ex.shutdown(wait=not self._fit_cancelled, cancel_futures=True)
            finally:
                for k, v in _thr_saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            return _stellar_rows
        finally:
            for _shm in (shm, err_shm):
                if _shm is None:
                    continue
                _shm.close()
                try:
                    _shm.unlink()
                except FileNotFoundError:
                    pass

    def _stellar_cube_prep(self):
        """Prepare a per-spaxel stellar fit across the cube. Returns a LIST of
        prep dicts (one per stellar region, each carrying its own region_ID/lib/
        velscale/fit_range/moments), or [] if there is no stellar region /
        pPXF is unavailable."""
        if hcppxf is None or FITS_DATA is None:
            return []
        sreg = (df_cont[df_cont['cont_type'] == 'stellar']
                if isinstance(df_cont, pd.DataFrame) and 'cont_type' in df_cont.columns
                else None)
        if sreg is None or len(sreg) == 0:
            return []
        z = _safe_float(df_obs.loc[0, 'redshift']) if len(df_obs) else 0.0
        z = 0.0 if not np.isfinite(z) else z
        R = _safe_float(df_obs.loc[0, 'resolvingpower']) if len(df_obs) else np.nan
        R = 3000.0 if not np.isfinite(R) or R <= 0 else R
        mask = (df['Centroid_0'].astype(float).to_numpy()
                if len(df) > 0 and 'Centroid_0' in df.columns else ())
        preps = []
        for _, r in sreg.iterrows():
            library = str(r['stellar_library'])
            fit_range = (float(r['x1']), float(r['x2']))
            moments = int(r['stellar_moments']) if pd.notna(r.get('stellar_moments')) else 2
            velscale, lam_rest = hcppxf.galaxy_velscale(wavelengths, z, fit_range)
            lib = _STELLAR_LIBS.get(library) or hcppxf.TemplateLibrary(library).load()
            _STELLAR_LIBS[library] = lib
            lib.prepare(velscale, R, lam_rest)
            preps.append(dict(rid=int(np.int64(r['region_ID'])), lib=lib,
                              velscale=velscale, z=z, R=R, mask=mask,
                              moments=moments, fit_range=fit_range, library=library))
        return preps

    def _stellar_fit_one(self, i, j, prep):
        """pPXF for one spaxel; caches the optimal template and returns a
        df_stellar row dict (or None on failure)."""
        global STELLAR_CACHE
        vw = self.viewer_window
        flux = np.nan_to_num(vw.get_spectrum_at_spaxel(i, j))
        try:
            res = hcppxf.fit_stellar(
                flux, wavelengths, prep['z'], prep['R'], prep['lib'],
                fit_range=prep['fit_range'], mask_centroids=prep['mask'],
                moments=prep['moments'], degree=-1, mdegree=10,
                sigma_guess=150.0, velscale=prep['velscale'])
        except Exception:
            return None
        STELLAR_CACHE[(i, j, int(prep['rid']))] = res['cache']
        try:
            ra, dec = vw.pixel_to_ra_dec(i, j)
        except Exception:
            ra, dec = np.nan, np.nan
        return {'spaxel_x': i, 'spaxel_y': j, 'region_ID': int(prep['rid']),
                'RA': float(ra), 'Dec': float(dec),
                'stellar_V': res['V'], 'stellar_sigma': res['sigma'],
                'stellar_h3': res['h3'], 'stellar_h4': res['h4'],
                'stellar_scale': res['scale'], 'stellar_chi2': res['chi2'],
                'success': True}

    def _fit_stellar_cube_impl(self):
        """Run pPXF on every SNR-gated spaxel and fill df_stellar + STELLAR_CACHE
        (stellar kinematics only — no emission lines)."""
        global df_stellar
        from PyQt5.QtWidgets import QMessageBox, QApplication
        preps = self._stellar_cube_prep()
        if not preps:
            QMessageBox.warning(self, 'Stellar Cube Fit',
                                'Add a stellar template to a spaxel first.')
            return
        nx, ny = FITS_DATA.shape[2], FITS_DATA.shape[1]
        smap = globals().get('snr_map', None)
        nspax = int(np.sum(smap >= snr_value)) if smap is not None else nx * ny
        total = nspax * len(preps)
        self._begin_progress(total, 'Stellar fit')

        rows, done = [], 0
        for i in range(nx):
            for j in range(ny):
                if self._fit_cancelled:
                    break
                if smap is not None and smap[j, i] < snr_value:
                    continue
                for prep in preps:
                    row = self._stellar_fit_one(i, j, prep)
                    done += 1
                    if row is not None:
                        rows.append(row)
                    if done % 10 == 0:
                        self._update_progress(done, f"Stellar fit {done} / {total}")
            if self._fit_cancelled:
                break

        df_stellar = pd.DataFrame(rows)
        self._end_progress()
        print(f"Stellar cube fit: {len(df_stellar)} spaxels"
              f"{' (cancelled)' if self._fit_cancelled else ''}.")
        if len(df_stellar) == 0:
            QMessageBox.information(self, 'Stellar Cube Fit', 'No spaxels fit.')

    def _current_stellar_kin(self, rid=None):
        """(V, σ) for the current spaxel's stellar region `rid`, from its cache,
        df_stellar, else NaN."""
        vw = self.viewer_window
        if vw is None or vw.current_spaxel is None:
            return np.nan, np.nan
        x, y = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])
        if rid is not None:
            cache = STELLAR_CACHE.get((x, y, int(rid)))
            if cache is not None and 'V' in cache:
                return _safe_float(cache.get('V')), _safe_float(cache.get('sigma'))
        if isinstance(df_stellar, pd.DataFrame) and len(df_stellar) and 'spaxel_x' in df_stellar.columns:
            sr = df_stellar[(df_stellar['spaxel_x'] == x) & (df_stellar['spaxel_y'] == y)]
            if rid is not None and 'region_ID' in df_stellar.columns:
                sr = sr[sr['region_ID'] == int(rid)]
            if len(sr):
                return _safe_float(sr.iloc[0].get('stellar_V')), _safe_float(sr.iloc[0].get('stellar_sigma'))
        return np.nan, np.nan

    def _refresh_stellar_map_buttons(self):
        """Update the V/σ map button faces to the current spaxel's best-fit
        values (called on cursor move so they track the hovered spaxel)."""
        if 'cont_type' not in df_cont.columns:
            return
        for rid in df_cont.loc[df_cont['cont_type'] == 'stellar', 'region_ID']:
            rid = int(np.int64(rid))
            cur_V, cur_sig = self._current_stellar_kin(rid)
            b = self.buttons_dict.get((rid, 'stellar~vmap'))
            if b is not None:
                b.setText(_fmt(cur_V))
            b = self.buttons_dict.get((rid, 'stellar~smap'))
            if b is not None:
                b.setText(_fmt(cur_sig))

    def _stellar_map(self, col, rid=None):
        """Display a per-spaxel stellar map (e.g. stellar_V / stellar_sigma) for
        stellar region `rid` (None = all rows, for back-compat)."""
        from PyQt5.QtWidgets import QMessageBox
        if not isinstance(df_stellar, pd.DataFrame) or len(df_stellar) == 0:
            QMessageBox.information(self, 'Stellar Map',
                                    'Run "Fit Stellar (Cube)" first.')
            return
        rows = df_stellar
        if rid is not None and 'region_ID' in df_stellar.columns:
            rows = df_stellar[df_stellar['region_ID'] == int(rid)]
        if len(rows) == 0:
            QMessageBox.information(self, 'Stellar Map',
                                    'No stellar fit for this region yet.')
            return
        ny, nx = FITS_DATA.shape[1], FITS_DATA.shape[2]
        arr = np.full((ny, nx), np.nan)
        for _, row in rows.iterrows():
            arr[int(row['spaxel_y']), int(row['spaxel_x'])] = _safe_float(row.get(col))
        cmap = 'bwr' if col == 'stellar_V' else 'plasma'
        self.viewer_window.draw_image(arr, cmap=cmap, scale='linear', from_fits=False)


    def fit_single_spaxel(self):
        try:
            self._fit_single_spaxel_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Fit Error',
                f'Fit failed for spaxel {self.viewer_window.current_spaxel}:\n\n'
                f'{type(e).__name__}: {e}')

    def _fit_single_spaxel_impl(self):
        global df, df_cont, df_fit, df_obs, fit_results, piecewise_model, line, viewing_fit
        global base_df_cont, base_df, spaxel_overrides

        # A 1D spectrum has no spaxel grid; treat it as the single spaxel (0, 0)
        # so the single-spaxel fit path works without a prior image click.
        vw = self.viewer_window
        if vw.current_spaxel is None and vw.is_1d_spectrum:
            vw.current_spaxel = (0, 0)

        # Capture the base template lazily so lock-to-load works afterwards.
        if base_df_cont is None:
            base_df_cont = df_cont.copy(deep=True)
            base_df = df.copy(deep=True)
            if 'lineactor' in base_df_cont.columns:
                base_df_cont['lineactor'] = None
            if 'curveactor' in base_df.columns:
                base_df['curveactor'] = None

        if df_obs['redshift'].item() == '':
            z = 0
        else:
            z = df_obs['redshift'].item()

        params = Parameters()
        for i, row in df_cont.iterrows():
            region_id = row['region_ID']
            params.add(f'x{i + 1}_start', value=row['x1'], vary=False)
            params.add(f'x{i + 1}_end', value=row['x2'], vary=False)

            _ctype = str(row['cont_type']) if ('cont_type' in df_cont.columns
                                               and pd.notna(row['cont_type'])) else 'linear'
            _kx = _as_float_list(row['knots_x']) if 'knots_x' in df_cont.columns else []
            _ky = _as_float_list(row['knots_y_0']) if 'knots_y_0' in df_cont.columns else []
            _pc = _as_float_list(row['poly_coef_0']) if 'poly_coef_0' in df_cont.columns else []
            is_poly = (_ctype == 'poly' and len(_pc) >= 1)
            is_spline = (_ctype == 'spline' and len(_kx) >= 2 and len(_kx) == len(_ky))
            is_stellar = (_ctype == 'stellar')

            slope = row['Slope_0'] if np.isfinite(row['Slope_0']) else 0
            intercept = row['Intercept_0'] if np.isfinite(row['Intercept_0']) else 0
            # Stellar continuum is a fixed baseline subtracted from the spectrum
            # before the line fit, so its model continuum is held at zero.
            if is_stellar:
                slope = intercept = 0

            params.add(f'slope{i + 1}', value=slope, vary=not (is_spline or is_poly or is_stellar))
            params.add(f'intercept{i + 1}', value=intercept, vary=not (is_spline or is_poly or is_stellar))

            if is_poly:
                params.add(f'NP{i + 1}', value=len(_pc), vary=False)
                for j in range(len(_pc)):
                    params.add(f'polyc{i + 1}_{j}', value=float(_pc[j]), vary=True)
            else:
                params.add(f'NP{i + 1}', value=0, vary=False)

            if is_spline:
                ptp = (max(_ky) - min(_ky)) if len(_ky) > 1 else 0.0
                _delta = 0.5 * ptp if ptp > 0 else max(abs(np.mean(_ky)), 1.0)
                params.add(f'NK{i + 1}', value=len(_kx), vary=False)
                for k in range(len(_kx)):
                    params.add(f'knotx{i + 1}_{k}', value=float(_kx[k]), vary=False)
                    params.add(f'knoty{i + 1}_{k}', value=float(_ky[k]), vary=True,
                               min=float(_ky[k]) - _delta, max=float(_ky[k]) + _delta)
            else:
                params.add(f'NK{i + 1}', value=0, vary=False)

            if i < len(df_cont) - 1:
                next_row = df_cont.iloc[i + 1]
                params.add(f'x_int_{i + 1}_start', value=row['x2'], vary=False)
                params.add(f'x_int_{i + 1}_end', value=next_row['x1'], vary=False)
                params.add(f'slope_int_{i + 1}', value=slope, vary=True)
                params.add(f'intercept_int_{i + 1}', value=intercept, vary=True)

            region_lines = df[df['region_ID'] == region_id]
            for j, line in enumerate(df.itertuples(), start=1):
                _amp   = np.float64(line.Amp_0)
                _cen   = np.float64(line.Centroid_0)
                _sigma = np.float64(line.Sigma_0)
                _amp_lo  = np.float64(line.Amp_0_lowlim)   if np.isfinite(np.float64(line.Amp_0_lowlim))  else None
                _amp_hi  = np.float64(line.Amp_0_highlim)  if np.isfinite(np.float64(line.Amp_0_highlim)) else None
                _cen_lo  = np.float64(line.Centroid_0_lowlim)  if np.isfinite(np.float64(line.Centroid_0_lowlim))  else None
                _cen_hi  = np.float64(line.Centroid_0_highlim) if np.isfinite(np.float64(line.Centroid_0_highlim)) else None
                _sig_lo  = np.float64(line.Sigma_0_lowlim)  if np.isfinite(np.float64(line.Sigma_0_lowlim))  else None
                _sig_hi  = np.float64(line.Sigma_0_highlim) if np.isfinite(np.float64(line.Sigma_0_highlim)) else None
                if _amp == 0:
                    _amp = 1e-30
                params.add(f'amp{j}',   value=_amp,   vary=True, min=_amp_lo,  max=_amp_hi)
                params.add(f'cen{j}',   value=_cen,   vary=True, min=_cen_lo,  max=_cen_hi)
                params.add(f'sigma{j}', value=_sigma, vary=True, min=_sig_lo,  max=_sig_hi)

                if df.iloc[j-1]['Rest Wavelength']:
                    if type(df.iloc[j-1]['Rest Wavelength']) == str:
                        rest_wavelength = ast.literal_eval(df.iloc[j-1]['Rest Wavelength'])
                    else:
                        rest_wavelength = df.iloc[j-1]['Rest Wavelength']
                else:
                    rest_wavelength = np.nan
                if np.isfinite(rest_wavelength) and 'constraints' in df.columns:
                    df = update_constraints_with_velocity(df, z)

            params.add(f'NR{i + 1}', value=len(region_lines), vary=False)

        Nregions = len(df_cont)
        Nlines = len(np.unique(df['Line_ID']))
        add_dataframe_constraints_to_params(df, params)
        print(f'Nregions={Nregions}, Nlines={Nlines}')
        model_maker = HyperCube_ModelFunctions.PiecewiseModel(n_regions=Nregions, n_gaussians=Nlines)
        piecewise_model = Model(model_maker.model_function)

        fit_results = []
        self.fit_spaxel(z, params_to_use=params)

        new_df = pd.DataFrame(fit_results)
        if len(new_df) == 0 or 'LineID' not in new_df.columns:
            print("Fit failed for current spaxel.")
            return

        new_df['LineID'] = new_df['LineID'].astype(float)
        svg_colors = [
            'dodgerblue', 'mediumseagreen', 'darkorange', 'mediumpurple',
            'deepskyblue', 'gold', 'steelblue', 'mediumaquamarine',
            'peru', 'cornflowerblue'
        ]
        unique_lines = new_df['LineID'].unique()
        lineid_to_color = {ln: svg_colors[i % len(svg_colors)] for i, ln in enumerate(unique_lines)}
        new_df['color'] = new_df['LineID'].map(lineid_to_color)

        vw = self.viewer_window
        cx, cy = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])

        if isinstance(df_fit, pd.DataFrame) and len(df_fit) > 0 and 'spaxel_x' in df_fit.columns:
            df_fit = df_fit[~((df_fit['spaxel_x'] == cx) & (df_fit['spaxel_y'] == cy))]
            df_fit = pd.concat([df_fit, new_df], ignore_index=True)
        else:
            df_fit = new_df

        viewing_fit = True

        for _line in vw.spectrum_ax.lines[:]:
            if _line.get_label() != '_child0':
                try: _line.remove()
                except ValueError: pass
        vw.gaussian_component_lines.clear()
        df_cont['lineactor'] = None

        for region_ID in np.unique(np.int64(df_cont['region_ID'])):
            vw.rebuild_plot(region_ID, from_file=True, show_init=False, show_fit=True, x=cx, y=cy)
        vw.spectrum_canvas.draw_idle()

        # Persist this spaxel's edited model as its override (df may have been
        # reassigned by constraint expansion, so snapshot the current active model).
        ov_cont = df_cont.copy(deep=True)
        ov_df = df.copy(deep=True)
        if 'lineactor' in ov_cont.columns:
            ov_cont['lineactor'] = None
        if 'curveactor' in ov_df.columns:
            ov_df['curveactor'] = None
        spaxel_overrides[(cx, cy)] = {'df_cont': ov_cont, 'df': ov_df}
        if getattr(vw, '_edited_overlay_on', False):
            vw._draw_edited_overlay()

        # Exit edit mode (re-enables structure controls); the override persists.
        if getattr(vw, '_edit_spaxel', None) is not None:
            vw._edit_spaxel = None
            self.rebuild_fit_panel(show_fit=True, x=cx, y=cy)

        vw._init_guess_spaxel = None
        if getattr(vw, '_blue_rect', None) is not None:
            vw._blue_rect.set_visible(False)
            vw.canvas.draw_idle()

        print(f"Fit complete for spaxel ({cx}, {cy}).")


    def fit_cube(self, refit=False, rchisq_thresh=None, qual_col=None, qual_op='>',
                 bad_spaxels_override=None):
        global df, df_fit, fit_results, snr_mask, piecewise_model, line, new_results#,params
        global base_df_cont, base_df, spaxel_overrides, df_stellar

        # A fresh cube fit (re)defines the locked schema template that all
        # per-spaxel overrides must conform to, and invalidates old overrides.
        if not refit:
            base_df_cont = df_cont.copy(deep=True)
            base_df = df.copy(deep=True)
            if 'lineactor' in base_df_cont.columns:
                base_df_cont['lineactor'] = None
            if 'curveactor' in base_df.columns:
                base_df['curveactor'] = None
            spaxel_overrides = {}

        if self.viewer_window.is_1d_spectrum == False:
            nx, ny = np.shape(FITS_DATA)[2], np.shape(FITS_DATA)[1]
        else:
            nx = 1
            ny = 1
            
        total_spaxels = nx * ny
    
        app = QApplication.instance() or QApplication([])

        # Progress UI — embedded in the Fit Parameters panel (not a popup).
        progress_bar = self.fit_progress_bar
        status_label = self.fit_progress_label
        progress_bar.setRange(0, total_spaxels)
        progress_bar.setValue(0)
        status_label.setText(f"Fitting spaxel 0 / {total_spaxels}")
        self.fit_progress_frame.setVisible(True)
        QApplication.processEvents()
    
        # Constants for velocity calculation
        c = 299792.458  # km/s
        
        # Get galaxy redshift
        if df_obs['redshift'].item() == '':
            z = 0
        else:
            z = df_obs['redshift'].item()
    
        # Model parameters
        params = Parameters()
        
        # Sort continuum regions by x1 (start wavelength)
        df_cont_sort = df_cont.sort_values(by='x1').reset_index(drop=True)
        
        # Ensure lines are sorted by Centroid_0 within each region
        df_sort = df.sort_values(by=['region_ID', 'Centroid_0']).reset_index(drop=True)
    
        # Loop through each continuum region and add parameters
        for i, row in df_cont.iterrows():
            region_id = row['region_ID']
    
            # Set x-limits for each region (fixed)
            params.add(f'x{i + 1}_start', value=row['x1'], vary=False)
            params.add(f'x{i + 1}_end', value=row['x2'], vary=False)

            # Continuum: poly (Chebyshev coeffs free), spline (knot-Y free), or
            # linear (slope/intercept free).
            _ctype = str(row['cont_type']) if ('cont_type' in df_cont.columns
                                               and pd.notna(row['cont_type'])) else 'linear'
            _kx = _as_float_list(row['knots_x']) if 'knots_x' in df_cont.columns else []
            _ky = _as_float_list(row['knots_y_0']) if 'knots_y_0' in df_cont.columns else []
            _pc = _as_float_list(row['poly_coef_0']) if 'poly_coef_0' in df_cont.columns else []
            is_poly = (_ctype == 'poly' and len(_pc) >= 1)
            is_spline = (_ctype == 'spline' and len(_kx) >= 2 and len(_kx) == len(_ky))

            # Linear continuum parameters (also added for spline/poly regions so
            # the existing result-extraction code finds them; held fixed & unused
            # by the model when this region is a spline or polynomial).
            slope = row['Slope_0'] if np.isfinite(row['Slope_0']) else 0
            intercept = row['Intercept_0'] if np.isfinite(row['Intercept_0']) else 0

            params.add(f'slope{i + 1}', value=slope, vary=not (is_spline or is_poly))
            params.add(f'intercept{i + 1}', value=intercept, vary=not (is_spline or is_poly))

            if is_poly:
                # NP signals the model to use a Chebyshev polynomial for this
                # region; the coefficients are free, seeded from the initial fit.
                params.add(f'NP{i + 1}', value=len(_pc), vary=False)
                for j in range(len(_pc)):
                    params.add(f'polyc{i + 1}_{j}', value=float(_pc[j]), vary=True)
            else:
                params.add(f'NP{i + 1}', value=0, vary=False)

            if is_spline:
                # Knot count (signals the model to use a spline for this region),
                # fixed knot-x, and free knot-y seeded from the user's clicks.
                # Bound each knot-y to init ± 0.5·(peak-to-peak of the knots) to
                # curb the spline from absorbing emission lines.
                ptp = (max(_ky) - min(_ky)) if len(_ky) > 1 else 0.0
                _delta = 0.5 * ptp if ptp > 0 else max(abs(np.mean(_ky)), 1.0)
                params.add(f'NK{i + 1}', value=len(_kx), vary=False)
                for k in range(len(_kx)):
                    params.add(f'knotx{i + 1}_{k}', value=float(_kx[k]), vary=False)
                    params.add(f'knoty{i + 1}_{k}', value=float(_ky[k]), vary=True,
                               min=float(_ky[k]) - _delta, max=float(_ky[k]) + _delta)
            else:
                params.add(f'NK{i + 1}', value=0, vary=False)

            # Add parameters for the linear region between this and the next region
            if i < len(df_cont) - 1:
                next_row = df_cont.iloc[i + 1]
                params.add(f'x_int_{i + 1}_start', value=row['x2'], vary=False)
                params.add(f'x_int_{i + 1}_end', value=next_row['x1'], vary=False)
    
                params.add(f'slope_int_{i + 1}', value=slope, vary=True)
                params.add(f'intercept_int_{i + 1}', value=intercept, vary=True)
    
            # Extract Gaussians associated with this region
            region_lines = df[df['region_ID'] == region_id]
    
            # Add Gaussian parameters for this region
            for j, line in enumerate(df.itertuples(), start=1):
                # Cast all parameter values to float64 explicitly.
                # After pd.concat operations, columns can be object dtype;
                # lmfit silently misbehaves when given non-float initial values.
                _amp   = np.float64(line.Amp_0)
                _cen   = np.float64(line.Centroid_0)
                _sigma = np.float64(line.Sigma_0)
                _amp_lo  = np.float64(line.Amp_0_lowlim)   if np.isfinite(np.float64(line.Amp_0_lowlim))  else None
                _amp_hi  = np.float64(line.Amp_0_highlim)  if np.isfinite(np.float64(line.Amp_0_highlim)) else None
                _cen_lo  = np.float64(line.Centroid_0_lowlim)  if np.isfinite(np.float64(line.Centroid_0_lowlim))  else None
                _cen_hi  = np.float64(line.Centroid_0_highlim) if np.isfinite(np.float64(line.Centroid_0_highlim)) else None
                _sig_lo  = np.float64(line.Sigma_0_lowlim)  if np.isfinite(np.float64(line.Sigma_0_lowlim))  else None
                _sig_hi  = np.float64(line.Sigma_0_highlim) if np.isfinite(np.float64(line.Sigma_0_highlim)) else None
                # Guard: if initial value is exactly 0, use a small non-zero starting point
                # so the optimizer has something to scale against
                if _amp == 0:
                    _amp = 1e-30
                params.add(f'amp{j}',   value=_amp,   vary=True, min=_amp_lo,  max=_amp_hi)
                params.add(f'cen{j}',   value=_cen,   vary=True, min=_cen_lo,  max=_cen_hi)
                params.add(f'sigma{j}', value=_sigma, vary=True, min=_sig_lo,  max=_sig_hi)
                
                
                # Add velocity parameter for this line
                if df.iloc[j-1]['Rest Wavelength']:
                    if type(df.iloc[j-1]['Rest Wavelength']) == str:
                        rest_wavelength = ast.literal_eval(df.iloc[j-1]['Rest Wavelength'])
                    else:
                        rest_wavelength = df.iloc[j-1]['Rest Wavelength']
                    
                else:
                    rest_wavelength = np.nan
                
                if (np.isfinite(rest_wavelength)) & ('constraints' in df.columns):
                    # Update constraints in df
                    df = update_constraints_with_velocity(df, z)
    
            # Add a "fake" parameter to store the number of Gaussians in this region
            params.add(f'NR{i + 1}', value=len(region_lines), vary=False)
    
        # choose the model based on parameters
        Nregions = len(df_cont)
        Nlines = len(np.unique(df['Line_ID']))
        
        # parameter constraints 
        add_dataframe_constraints_to_params(df, params)
    
        # print("\nParameters with Constraints:")
        # params.pretty_print()
        print(f'Nregions={Nregions}, Nlines={Nlines}')
        model_maker = HyperCube_ModelFunctions.PiecewiseModel(n_regions=Nregions, n_gaussians=Nlines)
        piecewise_model = Model(model_maker.model_function)
        # piecewise_model = Model(HyperCube_ModelFunctions.model_chooser(Nregions, Nlines))
        
        # Save the spaxel the user has locked before the loop mutates current_spaxel
        _locked_spaxel = self.viewer_window.current_spaxel

        if len(fit_results) > 0 and not refit:
            fit_results = []
            
        # Handle refit case: either an explicit set of flagged spaxels
        # (bad_spaxels_override, from the Rectify dialog's arbitrary-map
        # selection) or the legacy single-column threshold.
        _have_quality = ('qa_core_cont_ratio' in df_fit.columns
                         or 'rchisq' in df_fit.columns) if (
                            'df_fit' in globals() and len(df_fit) > 0) else False
        if refit and _have_quality and (bad_spaxels_override is not None
                                        or rchisq_thresh is not None):
            fit_results = []

            # Goodness-of-fit column used for (a) ranking neighbours when
            # seeding and (b) deciding whether a re-fit candidate is acceptable.
            # Always a genuine quality metric, independent of how spaxels were
            # flagged for re-fitting.
            _accept_col = ('qa_core_cont_ratio' if 'qa_core_cont_ratio' in df_fit.columns
                           else 'rchisq')
            _accept_thr = self._RECTIFY_DEFAULTS.get(_accept_col, 2.0)
            _per = df_fit.drop_duplicates(subset=['spaxel_x', 'spaxel_y'])
            qual = {}
            for _, rr in _per.iterrows():
                try:
                    sx, sy = int(rr['spaxel_x']), int(rr['spaxel_y'])
                except (TypeError, ValueError):
                    continue
                qual[(sx, sy)] = _safe_float(rr.get(_accept_col))

            def _is_bad(v):
                # Candidate-acceptance quality check (higher = worse).
                return (not np.isfinite(v)) or (v > _accept_thr)

            if bad_spaxels_override is not None:
                # Explicit flagged set from the dialog. Good = fitted & not flagged.
                bad_set = {(int(x), int(y)) for (x, y) in bad_spaxels_override}
                good_set = {xy for xy in qual if xy not in bad_set}
                bad_spaxels = pd.DataFrame(sorted(bad_set),
                                           columns=['spaxel_x', 'spaxel_y'])
                print(f"Rectify: {len(bad_spaxels)} flagged / {len(qual)} fitted "
                      f"spaxels (custom selection)")
            else:
                # Legacy: flag on a single column / operator / threshold.
                _flag_col = (qual_col if (qual_col is not None and qual_col in df_fit.columns)
                             else _accept_col)
                flag = {}
                for _, rr in _per.iterrows():
                    try:
                        sx, sy = int(rr['spaxel_x']), int(rr['spaxel_y'])
                    except (TypeError, ValueError):
                        continue
                    flag[(sx, sy)] = _safe_float(rr.get(_flag_col))

                def _flag_bad(v):
                    if not np.isfinite(v):
                        return True
                    if qual_op == '<-':
                        return v < -rchisq_thresh
                    if qual_op == 'abs>':
                        return abs(v) > rchisq_thresh
                    return v > rchisq_thresh  # default: '>'

                good_set = {xy for xy, v in flag.items() if not _flag_bad(v)}
                bad_spaxels = pd.DataFrame(
                    sorted(xy for xy, v in flag.items() if _flag_bad(v)),
                    columns=['spaxel_x', 'spaxel_y'])
                _op_str = {'<-': f'< -{rchisq_thresh}',
                           'abs>': f'|·|> {rchisq_thresh}'}.get(
                    qual_op, f'> {rchisq_thresh}')
                print(f"Rectify: {len(bad_spaxels)} bad / {len(flag)} fitted spaxels "
                      f"({_flag_col} {_op_str})")

            def _best_neighbour(i, j):
                """Best (lowest-quality-value) good spaxel among the 8 neighbours."""
                best, best_q = None, np.inf
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        nb = (i + di, j + dj)
                        if nb in good_set and qual[nb] < best_q:
                            best, best_q = nb, qual[nb]
                return best

            # Refit each bad spaxel: seed from its best good neighbour (spatial
            # smoothness prior); if that still fails, fall back to targeted
            # multi-start (Phase C). Only the best-scoring candidate is kept.
            n_rescued = 0
            for i, j in bad_spaxels.itertuples(index=False):
                if snr_map[j, i] >= snr_value:
                    self.viewer_window.current_spaxel = (i, j)
                    nb = _best_neighbour(i, j)
                    if nb is not None:
                        nb_rows = df_fit[(df_fit['spaxel_x'] == nb[0]) &
                                         (df_fit['spaxel_y'] == nb[1])]
                        seed = self._params_from_fit_row(params, nb_rows)
                    else:
                        seed = params.copy()  # no good neighbour → base init

                    # Incumbent: neighbour-seeded (or base) fit.
                    best_ratio, best_rows = self._fit_candidate(z, seed)

                    # Targeted multi-start only if the incumbent is still bad.
                    if _is_bad(best_ratio):
                        for _label, cand in self._targeted_restarts(seed):
                            c_ratio, c_rows = self._fit_candidate(z, cand)
                            if c_ratio < best_ratio:
                                best_ratio, best_rows = c_ratio, c_rows

                    fit_results.extend(best_rows)
                    if not _is_bad(best_ratio):
                        n_rescued += 1

                    current_spaxel = i * ny + j + 1
                    progress_bar.setValue(current_spaxel)
                    status_label.setText(f"Rectifying spaxel ({i}, {j})")
                    QApplication.processEvents()

                    if current_spaxel % 500 == 0 and psutil.virtual_memory().percent > 80:
                        gc.collect()

            print(f"Rectify: {n_rescued}/{len(bad_spaxels)} spaxels now below threshold")

            # Merge results (keeping LineID matching)
            new_results = pd.DataFrame(fit_results)
            
            # First, drop the rows from df_fit that match the rows in df_new (on the three columns)
            df_fit_filtered = df_fit.merge(
                new_results[['spaxel_x', 'spaxel_y', 'LineID']],
                on=['spaxel_x', 'spaxel_y', 'LineID'],
                how='left',
                indicator=True
            ).query('_merge == "left_only"').drop(columns='_merge')
            
            # Then, append the new rows
            df_fit = pd.concat([df_fit_filtered, new_results], ignore_index=True)
            # new_results = pd.DataFrame(fit_results[-len(bad_spaxels):])
            # composite_keys = (df_fit['spaxel_x'].astype(str) + '_' + 
            #                  df_fit['spaxel_y'].astype(str) + '_' + 
            #                  df_fit['LineID'].astype(str))
            # new_keys = (new_results['spaxel_x'].astype(str) + '_' + 
            #             new_results['spaxel_y'].astype(str) + '_' + 
            #             new_results['LineID'].astype(str))
            
            # df_fit = df_fit[~composite_keys.isin(new_keys)]
            # df_fit = pd.concat([df_fit, new_results], ignore_index=True)
            
        else:
            # Normal fit over all SNR-gated spaxels. Each spaxel is a full-model
            # fit (continuum of any type + lines across N regions). Distributed
            # across worker processes when Cores > 1; otherwise serial. Any stellar
            # region is fit per spaxel in the same pass (kinematics → V/σ maps;
            # baseline subtracted before the line fit).
            self._fit_cancelled = False
            _has_stellar = ('cont_type' in df_cont.columns and
                            bool((df_cont['cont_type'] == 'stellar').any()))
            gated = [(i, j) for i in range(nx) for j in range(ny)
                     if snr_map[j, i] >= snr_value]
            total = max(len(gated), 1)
            progress_bar.setRange(0, total)
            progress_bar.setValue(0)
            n_workers = self._fit_worker_count()

            if (n_workers > 1 and len(gated) > 1
                    and not self.viewer_window.is_1d_spectrum):
                try:
                    _stellar_rows = self._fit_cube_parallel(
                        gated, params, z, Nregions, Nlines, n_workers,
                        progress_bar, status_label, total)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Parallel fit failed ({type(e).__name__}: {e}); "
                          f"falling back to serial.")
                    fit_results.clear()
                    _stellar_rows = self._fit_cube_serial(
                        gated, params, z, progress_bar, status_label, total)
            else:
                _stellar_rows = self._fit_cube_serial(
                    gated, params, z, progress_bar, status_label, total)

            if _has_stellar:
                df_stellar = pd.DataFrame(_stellar_rows)
            # After fitting, create the results dataframe
            df_fit = pd.DataFrame(fit_results)
        
        # Clean up the dataframe
        if 'fit_success' in df_fit.columns:
            # Handle failed fits
            df_fit['success'] = df_fit.get('success', True) & ~df_fit['fit_success'].isna()
            df_fit.drop(columns=['fit_success'], inplace=True, errors='ignore')
        
        # Assign colors to the lines for plotting:
        df_fit['LineID'] = df_fit['LineID'].astype(float)
        unique_lines = df_fit['LineID'].unique()
        
        svg_colors = [
            'dodgerblue', 'mediumseagreen', 'darkorange', 'mediumpurple',
            'deepskyblue', 'gold', 'steelblue', 'mediumaquamarine',
            'peru', 'cornflowerblue'
        ]
        
        lineid_to_color = {line: svg_colors[i % len(svg_colors)] for i, line in enumerate(unique_lines)}
        df_fit['color'] = df_fit['LineID'].map(lineid_to_color)

        self.fit_progress_frame.setVisible(False)

        # Refresh the fitted model on the CURRENTLY DISPLAYED spaxel
        vw = self.viewer_window
        # Restore the spaxel the user had locked — fit_cube overwrites current_spaxel
        vw.current_spaxel = _locked_spaxel
        cx, cy = int(_locked_spaxel[0]), int(_locked_spaxel[1])

        # ── Wipe every curve from the spectrum (both dashed and solid init-guess) ──
        # Keep only the raw spectrum step-line (_child0 label)
        for _line in vw.spectrum_ax.lines[:]:
            if _line.get_label() != '_child0':
                try: _line.remove()
                except ValueError: pass
        vw.gaussian_component_lines.clear()
        # Also remove the lineactor refs so rebuild_plot plots fresh lines
        df_cont['lineactor'] = None

        # ── Guard: only rebuild if this spaxel was actually fitted ──────────────
        fitted_spaxels = set(
            zip(df_fit['spaxel_x'].astype(int), df_fit['spaxel_y'].astype(int))
        ) if len(df_fit) > 0 else set()

        if (cx, cy) in fitted_spaxels:
            for region_ID in np.unique(np.int64(df_cont['region_ID'])):
                vw.rebuild_plot(region_ID, from_file=True, show_init=False, show_fit=True, x=cx, y=cy)
        vw.spectrum_canvas.draw_idle()
        # Remove the blue init-guess spaxel marker now that the fit is done
        vw._init_guess_spaxel = None
        if getattr(vw, '_blue_rect', None) is not None:
            vw._blue_rect.set_visible(False)
            vw.canvas.draw_idle()

        print("Fitting complete for entire cube.")

        

    def add_spectral_frame(self, title_text, df_cont, df, ID, addframe):
        """Creates a spectral region frame and appends it vertically to the scroll area.

        Each frame has:
          - a compact title bar (region name + delete button)
          - a QGridLayout with continuum params in a header row, then one row per line
          - a + Line button at the bottom
        The frame expands horizontally to fill the dock width (scroll is vertical only).
        """
        # df_cont and df arrive as parameters; use local aliases so we can
        # assign back to the module-level globals when addframe is True.
        _df_cont = df_cont
        _df = df

        # ── Init placeholder data if addframe requested ─────────────────────
        data_cont_initreg = {
            'Continuum Name': ['Continuum'],
            'x1': [5000], 'x2': [5010],
            'Slope_0': [0], 'Intercept_0': [0.002],
            'Slope_fit': [np.nan], 'Intercept_fit': [np.nan],
            'region_ID': [ID], 'lineactor': [self.current_line]
        }
        data_lines_initreg = {
            'Line_ID': [0, 1], 'Line_Name': ['line 1', 'line 2'],
            'SNR': [np.nan, np.nan],
            'Amp_0': [0.2348, 0.343], 'Centroid_0': [5504, 6533], 'Sigma_0': [0.345, 0.45],
            'Amp_fit': [np.nan, np.nan], 'Centroid_fit': [np.nan, np.nan],
            'Sigma_fit': [np.nan, np.nan], 'region_ID': [ID, ID]
        }
        if addframe:
            # Write back to module-level globals (can't use 'global' because
            # df_cont and df are also parameter names in this function).
            import sys as _sys
            _g = vars(_sys.modules[__name__])
            _g['df_cont'] = pd.concat([_df_cont, pd.DataFrame(data_cont_initreg)], ignore_index=True)
            _g['df']      = pd.concat([_df,      pd.DataFrame(data_lines_initreg)], ignore_index=True)
            _df_cont = _g['df_cont']
            _df      = _g['df']

        # ── Outer frame ─────────────────────────────────────────────────────
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName(f"frame_{ID}")
        frame.setMinimumWidth(0)
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(4)
        frame_layout.setContentsMargins(6, 6, 6, 6)

        # ── Title bar ───────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_lbl = QLabel(title_text)
        title_lbl.setStyleSheet("font-size: 12pt; font-weight: bold;")
        title_lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        title_row.addWidget(title_lbl)
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(22, 22)
        del_btn.setStyleSheet("background-color: lightcoral; color: black;")
        if self._edit_mode():
            del_btn.setEnabled(False)
            del_btn.setToolTip("Disabled while editing a single spaxel (schema is locked)")
        else:
            del_btn.setToolTip("Delete this spectral region")
            del_btn.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=ID))
        title_row.addWidget(del_btn)
        frame_layout.addLayout(title_row)

        # ── Single grid: continuum header + continuum row + line header + line rows
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(6)
        grid_layout.setVerticalSpacing(3)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        frame_layout.addLayout(grid_layout)

        self.add_continuum_buttons(ID, grid_layout)
        self.add_spectral_lines_button_header(grid_layout)
        self.add_spectral_lines(ID, grid_layout)

        # Equal stretch on every column so content fills the frame width
        n_cols = 17
        for col in range(n_cols):
            grid_layout.setColumnStretch(col, 1)
            grid_layout.setColumnMinimumWidth(col, 0)

        # ── + Line button ───────────────────────────────────────────────────
        add_line_btn = QPushButton('+ Line')
        add_line_btn.setFixedHeight(26)
        if self._edit_mode():
            add_line_btn.setEnabled(False)
            add_line_btn.setToolTip('Disabled while editing a single spaxel (schema is locked)')
        else:
            add_line_btn.clicked.connect(partial(self.on_addline_widget, data_frame=df, frame_id=ID))
        # Row = header(1) + continuum_header(1) + continuum_row(1) + line_header(1) + line_rows
        n_lines = len(df.loc[df['region_ID'] == ID]) if 'region_ID' in df.columns else 0
        grid_layout.addWidget(add_line_btn, 3 + n_lines * 2, 0)
        self.buttons_dict[(ID, '__add_line__')] = add_line_btn

        # ── Append to vertical scroll layout ────────────────────────────────
        # Insert before the trailing stretch if one exists, otherwise just append
        count = self.scroll_layout.count()
        last = self.scroll_layout.itemAt(count - 1) if count > 0 else None
        if last and last.spacerItem():
            self.scroll_layout.insertWidget(count - 1, frame)
        else:
            self.scroll_layout.addWidget(frame)

    def update_button_value(self, frame_id, button_name, new_value):
        """Update the button text based on row and column"""
        if any(x in button_name for x in ['sourcename','redshift','resolvingpower']):
            self.source_name_button.setText(f"Source: {df_obs.loc[0, 'sourcename']}")
            self.source_redshift_button.setText(f"z: {df_obs.loc[0, 'redshift']}")
            self.resolving_power_button.setText(f"R: {df_obs.loc[0, 'resolvingpower']}")
        else:
        
            if (frame_id, button_name) in self.buttons_dict:
                button = self.buttons_dict[(frame_id, button_name)]
                button.setText(str(new_value))
    
    def on_deleteregion_button_click(self, frame_id):
        global df, df_cont, base_df_cont, base_df, spaxel_overrides
        """Deletes an entire spectral region: removes all its curves from the plot,
        drops it from df and df_cont, and regenerates the UI."""
        print(f"Deleting region {frame_id}")

        # Keep the locked schema consistent: drop this region from the base
        # template and every per-spaxel override too, so it doesn't reappear
        # when the model is reloaded on lock/unlock/move.
        def _drop_region(d):
            if isinstance(d, pd.DataFrame) and 'region_ID' in d.columns and len(d) > 0:
                return d.drop(d.index[np.int64(d['region_ID']) == frame_id]).reset_index(drop=True)
            return d
        base_df_cont = _drop_region(base_df_cont)
        base_df      = _drop_region(base_df)
        for _k, _ov in spaxel_overrides.items():
            _ov['df_cont'] = _drop_region(_ov['df_cont'])
            _ov['df']      = _drop_region(_ov['df'])

        vw = self.viewer_window
        ax = vw.spectrum_ax

        # Remove all dashed component curves belonging to this region
        for line in ax.lines[:]:
            if line.get_linestyle() in ('--', 'dashed'):
                try:
                    line.remove()
                except ValueError:
                    pass
        vw.gaussian_component_lines.clear()

        # Remove the solid total-model lineactor for this region
        line_actor = df_cont.loc[np.int64(df_cont["region_ID"]) == frame_id, 'lineactor'].item()
        if line_actor is not None and line_actor in ax.lines:
            try:
                line_actor.remove()
            except ValueError:
                pass

        # Drop region from dataframes
        df = df.reset_index(drop=True)
        df = df.drop(df.index[np.int64(df["region_ID"]) == frame_id]).reset_index(drop=True)
        df_cont = df_cont.drop(df_cont.index[np.int64(df_cont["region_ID"]) == frame_id]).reset_index(drop=True)

        # Rebuild UI frames
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()
        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        for ID in np.unique(df_cont['region_ID']):
            ID = np.int64(ID)
            self.add_spectral_frame('Spectral Region ' + str(ID + 1), df_cont, df, ID, addframe=False)

        vw.spectrum_canvas.draw_idle()
        # Remove the blue init-guess marker — region is gone
        if len(df_cont) == 0:
            vw._init_guess_spaxel = None
            if getattr(vw, '_blue_rect', None) is not None:
                vw._blue_rect.set_visible(False)
                vw.canvas.draw_idle()

    def on_deleteline_button_click(self, frame_id, line_id):
        global df, df_cont, base_df, spaxel_overrides
        """Deletes a spectral line from df and redraws surviving components."""
        print(f"Deleting line {line_id} from frame {frame_id}")

        # Keep the locked schema consistent: drop this line from the base
        # template and every per-spaxel override too, so it doesn't reappear
        # when the model is reloaded on lock/unlock/move.
        def _drop_line(d):
            if isinstance(d, pd.DataFrame) and {'region_ID', 'Line_ID'}.issubset(d.columns) and len(d) > 0:
                return d.drop(d[(d['region_ID'].astype(int) == frame_id) &
                                (d['Line_ID'].astype(int) == line_id)].index).reset_index(drop=True)
            return d
        base_df = _drop_line(base_df)
        for _k, _ov in spaxel_overrides.items():
            _ov['df'] = _drop_line(_ov['df'])

        vw = self.viewer_window
        ax = vw.spectrum_ax

        # Step 1: wipe every dashed line from the spectrum axes.
        # curveactor refs fall out of sync after reindexing so we nuke all
        # dashed lines and let rebuild_plot redraw survivors cleanly.
        for line in ax.lines[:]:
            if line.get_linestyle() in ('--', 'dashed'):
                try:
                    line.remove()
                except ValueError:
                    pass
        vw.gaussian_component_lines.clear()

        # Step 2: remove the solid total-model lineactor for this region
        for actor in df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor']:
            if actor and actor in ax.lines:
                try:
                    actor.remove()
                except ValueError:
                    pass

        # Step 3: drop the line from df
        df = df.reset_index(drop=True)
        df = df.drop(df[(df["region_ID"].astype(int) == frame_id) &
                        (df["Line_ID"].astype(int) == line_id)].index)
        df = df.reset_index(drop=True)

        # Step 4: rebuild UI frames
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()
        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        for ID in np.unique(df_cont['region_ID']):
            ID = np.int64(ID)
            self.add_spectral_frame(f'Spectral Region {ID + 1}', df_cont, df, ID, addframe=False)

        # Step 5: redraw survivors (use the current spaxel so a stellar region's
        # baseline cache resolves; (0,0) would miss and flatten the model to 0).
        _cx, _cy = 0, 0
        if vw.current_spaxel is not None:
            _cx, _cy = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])
        vw.spectrum_canvas.draw_idle()
        vw.rebuild_plot(region_ID=frame_id, from_file=False, show_init=True, show_fit=False, x=_cx, y=_cy)

    def on_addline_widget(self, data_frame, frame_id):
        global df, df_cont
        # Retrieve the line associated with frame_id
        line_obj = df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor'].item()
    
        if not isinstance(line_obj, plt.Line2D):
            print(f"Error: No valid line object found.")
            return
    
        # Store the line and initialize Gaussian parameters
        self.active_line = line_obj
        region = df_cont.loc[df_cont["region_ID"] == frame_id]
        self.x1, self.x2 = region["x1"].item(), region["x2"].item()
        self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
        # Generate updated Gaussian + continuum baseline (linear or spline)
        x_vals = np.linspace(self.x1, self.x2, 1000)
        y_line = self.viewer_window._region_baseline(region, x_vals)  # baseline (without Gaussians)
        
        
        # Recompute all Gaussians from scratch (prevents runaway summing)
        y_gaussian_total = np.zeros_like(x_vals)
    
        # Sum over all existing Gaussians in df
        for _, row in df.iterrows():
            y_gaussian_total += row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
    
        self.gaussian_amplitude = 0.1
        self.gaussian_x0 = np.mean(x_vals)
        self.gaussian_sigma = 1
        # If we're actively adjusting a new Gaussian, include it
        y_gaussian_new = self.gaussian_amplitude * np.exp(-((x_vals - self.gaussian_x0) ** 2) / (2 * self.gaussian_sigma ** 2))
        y_gaussian_total += y_gaussian_new
    
        # Final y-values
        y_new = y_line + y_gaussian_total
    
        # Update the existing line object
        self.active_line.set_data(x_vals, y_new)
    
        # Redraw dynamically
        self.viewer_window.spectrum_ax.figure.canvas.draw_idle()
        
        if len(df) == 0: 
            line_id = 0
        else:
            line_id = np.max(np.int64(df['Line_ID']))+1
        
        df_new = pd.DataFrame({'Line_ID': [line_id],
                'Line_Name': [f'Line '+str(len(df.loc[df["region_ID"]==frame_id]))],
                'SNR':[np.nan],
                'Rest Wavelength': [np.nan],
                'Amp_0': [self.gaussian_amplitude],
                'Amp_0_lowlim': [0],
                'Amp_0_highlim': [np.inf],
                'Centroid_0': [self.gaussian_x0],
                'Centroid_0_lowlim': [-np.inf],
                'Centroid_0_highlim': [np.inf],
                'Sigma_0': [self.gaussian_sigma],
                'Sigma_0_lowlim': [0],
                'Sigma_0_highlim': [np.inf],
                'Amp_fit': [np.nan],
                'Centroid_fit': [np.nan],
                'Sigma_fit': [np.nan],
                'region_ID': [frame_id],
                'curveactor': [self.active_line]})
        df = pd.concat([df,df_new])
            
        self.on_addline_button_click(data_frame=df, frame_id=frame_id)
        
        
        
    def print_active_frames(self):
        print("Active frames in the scroll window:")
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                print(f"Frame ID: {widget.objectName()}")
            else:
                print("Non-frame widget found in scroll layout.")
        
    
    def on_addline_button_click(self, data_frame, frame_id):
        global df

        # Find the corresponding frame in the UI
        frame = None
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame) and str(frame_id) in widget.objectName():
                frame = widget
                break
    
        # self.print_active_frames()
    
        if not frame:
            print("Frame not found!")
            return
    
    
        # Get the grid layout inside the frame (second item in the card VBoxLayout)
        grid_layout = frame.layout().itemAt(1).layout()

        # Retrieve the stored + Line button and remove it from its current position
        add_line_button = self.buttons_dict.get((frame_id, '__add_line__'))
        if add_line_button:
            grid_layout.removeWidget(add_line_button)

        # Only add the NEW (last) line row — re-drawing all rows creates duplicates
        df_region = df.loc[np.int64(df['region_ID']) == np.int64(frame_id)]
        n_lines = len(df_region)
        last_row_idx = n_lines - 1  # 0-based index of the new line in df_region

        button_columns = ['Line_Name', 'SNR', 'Rest Wavelength',
                          'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                          'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                          'Sigma_0', 'Sigma_0_lowlim', 'Sigma_0_highlim',
                          'Amp_fit', 'Centroid_fit', 'v_fit', 'Sigma_fit']

        # grid row for the new line:
        # row 0 = cont header, 1 = cont row, 2 = line header, 3..3+n-1 = line rows
        grid_row = 3 + last_row_idx * 2

        for col, col_name in enumerate(button_columns):
            button_name = str(np.int64(df_region.iloc[last_row_idx]['Line_ID'])) + '~' + col_name
            if col_name in ('Sigma_0', 'Sigma_0_lowlim', 'Sigma_0_highlim'):
                # σ stored in Å; displayed as velocity dispersion (km/s)
                button_text = _fmt(sigma_wl_to_kms(df_region.iloc[last_row_idx][col_name],
                                                   df_region.iloc[last_row_idx]['Centroid_0']))
            elif col_name in ['Rest Wavelength', 'Amp_0', 'Centroid_0']:
                button_text = _fmt(df_region.iloc[last_row_idx][col_name])
            elif 'fit' in col_name.lower() or col_name in ('SNR', 'Rest Wavelength'):
                button_text = ''
            else:
                button_text = str(df_region.iloc[last_row_idx][col_name])
            btn = FrameButton(button_text, last_row_idx, col, frame_id, button_name)
            btn.setFixedHeight(24)
            if 'fit' in col_name:
                btn.clicked.connect(partial(self.on_fit_button_click, frame_id=frame_id, button_name=button_name))
            else:
                btn.clicked.connect(partial(self.on_button_click, data_frame=df, frame_id=frame_id, button_name=button_name))
            # Colored border matching this line's component overlay color
            if col_name == 'Line_Name':
                _clr = self._line_overlay_color(df_region.iloc[last_row_idx]['Line_ID'])
                if _clr:
                    btn.setStyleSheet(f"border: 2px solid {_clr}; border-radius: 4px;")
            grid_layout.addWidget(btn, grid_row, col)
            self.buttons_dict[(frame_id, button_name)] = btn

        # Delete button at rightmost position of the new line row
        new_line_id = np.int64(df_region.iloc[last_row_idx]['Line_ID'])
        del_btn = FrameButton('x', last_row_idx, len(button_columns), frame_id, f'del_line_{new_line_id}')
        del_btn.setFixedHeight(24)
        del_btn.setStyleSheet("background-color: lightcoral; color: black;")
        del_btn.clicked.connect(partial(self.on_deleteline_button_click, frame_id=frame_id, line_id=new_line_id))
        grid_layout.addWidget(del_btn, grid_row, len(button_columns))
        self.buttons_dict[(frame_id, f'del_line_{new_line_id}')] = del_btn

        # Re-place the + Line button immediately below the new last row
        if add_line_button:
            grid_layout.addWidget(add_line_button, grid_row + 1, 0)

        # """
    
    def clear_snr_visualizer(self, frame_id, button_name):
        global df
        self.viewer_window.draw_image(FITS_DATA,cmap='gray',scale='linear',from_fits=True)
        self.update_button_value(frame_id, button_name, np.nan)
        df.loc[(df['region_ID'].astype(int)==frame_id) &
                        (df['Line_ID'].astype(float)==float(button_name.split('~')[0])),'SNR'] = np.nan


    def _show_source_dialog(self):
        """Source name dialog with NED name-resolve and coordinate-resolve options."""
        global df_obs, FITS_HEADER

        current_name = str(df_obs.loc[0, 'sourcename']) if len(df_obs) > 0 else ''
        current_z    = str(df_obs.loc[0, 'redshift'])   if len(df_obs) > 0 else ''

        dialog = QDialog(self)
        dialog.setWindowTitle('Source Information')
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        # ── Name row ─────────────────────────────────────────────────────────
        layout.addWidget(QLabel('Source name:'))
        name_edit = QLineEdit(current_name)
        name_edit.setPlaceholderText('e.g. NGC 1068')
        layout.addWidget(name_edit)

        # ── Resolve buttons ───────────────────────────────────────────────────
        resolve_name_btn  = QPushButton('Resolve name  (NED)')
        resolve_coord_btn = QPushButton('Resolve coordinates  (NED)')
        layout.addWidget(resolve_name_btn)
        layout.addWidget(resolve_coord_btn)

        # ── Status label ──────────────────────────────────────────────────────
        status = QLabel('')
        status.setWordWrap(True)
        layout.addWidget(status)

        # ── OK / Cancel ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        ok_btn     = QPushButton('OK');     ok_btn.setDefault(True)
        cancel_btn = QPushButton('Cancel')
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # ── Helpers ───────────────────────────────────────────────────────────
        def _apply_result(ned_name, ned_z):
            """Write NED name and redshift into HyperCube state."""
            name_edit.setText(ned_name)
            df_obs.loc[0, 'sourcename'] = ned_name
            df_obs.loc[0, 'redshift']   = ned_z
            self.source_name_button.setText(f'Source: {ned_name}')
            self.source_redshift_button.setText(f'z: {ned_z}')

        def _query_ned_by_name():
            name = name_edit.text().strip()
            if not name:
                status.setText('Please enter a source name first.')
                return
            status.setText(f'Querying NED for "{name}"…')
            dialog.repaint()
            try:
                try:
                    from astroquery.ipac.ned import Ned
                    result = Ned.query_object(name)
                    ned_name = str(result['Object Name'][0])
                    ned_z = float(result['Redshift'][0])
                except ImportError:
                    import urllib.request, urllib.parse
                    enc = urllib.parse.quote(name)
                    url = (f'https://ned.ipac.caltech.edu/cgi-bin/nph-objsearch'
                           f'?objname={enc}&extend=no&of=ascii_tab&list_limit=1'
                           f'&img_stamp=false&zv_breaker=30000'
                           f'&out_csys=Equatorial&out_equinox=J2000.0')
                    req = urllib.request.Request(url, headers={'User-Agent': 'HyperCube/1.0'})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        text = r.read().decode('utf-8', errors='replace')
                    lines = [l for l in text.splitlines() if l and not l.startswith('#')]
                    if not lines:
                        raise ValueError('No results returned by NED')
                    header = lines[0].split('\t')
                    row    = lines[1].split('\t')
                    d = dict(zip(header, row))
                    ned_name = d.get('Object Name', name).strip()
                    z_str    = d.get('Redshift', '').strip()
                    ned_z    = float(z_str) if z_str else None

                if ned_z is None:
                    status.setText(f'Found: {ned_name} — no redshift in NED.')
                    name_edit.setText(ned_name)
                    df_obs.loc[0, 'sourcename'] = ned_name
                    self.source_name_button.setText(f'Source: {ned_name}')
                else:
                    _apply_result(ned_name, ned_z)
                    status.setText(f'✓  {ned_name}   z = {ned_z:.6f}')
            except Exception as e:
                status.setText(f'NED query failed: {e}')

        def _query_ned_by_coords():
            try:
                ra  = float(FITS_HEADER.get('CRVAL1', FITS_HEADER.get('RA', '')))
                dec = float(FITS_HEADER.get('CRVAL2', FITS_HEADER.get('DEC', '')))
            except (TypeError, ValueError):
                status.setText('Could not read RA/Dec from FITS header.')
                return
            status.setText(f'Querying NED at RA={ra:.5f}, Dec={dec:.5f}...')
            dialog.repaint()
            try:
                try:
                    from astroquery.ipac.ned import Ned
                    import astropy.units as u
                    from astropy.coordinates import SkyCoord
                    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                    result = Ned.query_region(coord, radius=0.5 * u.arcmin)
                    if len(result) == 0:
                        raise ValueError('No objects found')
                    ned_name = str(result['Object Name'][0])
                    ned_z    = float(result['Redshift'][0])
                except ImportError:
                    import urllib.request
                    url = (f'https://ned.ipac.caltech.edu/cgi-bin/nph-objsearch'
                           f'?search_type=Near+Position+Search&ra={ra}&dec={dec}'
                           f'&radius=0.5&of=ascii_tab&list_limit=1&img_stamp=false'
                           f'&zv_breaker=30000&out_csys=Equatorial&out_equinox=J2000.0')
                    req = urllib.request.Request(url, headers={'User-Agent': 'HyperCube/1.0'})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        text = r.read().decode('utf-8', errors='replace')
                    lines_t = [l for l in text.splitlines() if l and not l.startswith('#')]
                    if len(lines_t) < 2:
                        raise ValueError('No NED object within 0.5 arcmin of coordinates.')
                    header = lines_t[0].split('\t')
                    row    = lines_t[1].split('\t')
                    d = dict(zip(header, row))
                    ned_name = d.get('Object Name', '').strip()
                    z_str    = d.get('Redshift', '').strip()
                    ned_z    = float(z_str) if z_str else None

                if ned_z is None:
                    status.setText(f'Found: {ned_name} — no redshift in NED.')
                    name_edit.setText(ned_name)
                    df_obs.loc[0, 'sourcename'] = ned_name
                    self.source_name_button.setText(f'Source: {ned_name}')
                else:
                    _apply_result(ned_name, ned_z)
                    status.setText(f'Found: {ned_name}   z = {ned_z:.6f}')
            except Exception as e:
                status.setText(f'NED query failed: {e}')

        def _commit():
            # Save whatever name is in the box (user may have typed without resolving)
            df_obs.loc[0, 'sourcename'] = name_edit.text().strip()
            self.source_name_button.setText(f'Source: {name_edit.text().strip()}')
            dialog.accept()

        resolve_name_btn.clicked.connect(_query_ned_by_name)
        resolve_coord_btn.clicked.connect(_query_ned_by_coords)
        ok_btn.clicked.connect(_commit)
        name_edit.returnPressed.connect(_commit)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec_()

    def on_button_click(self, data_frame, frame_id, button_name):
        """Handle button click, show a QLineEdit for editing the value"""
        global df_cont, df
        print(str(frame_id)+', '+button_name)
        if 'SNR' in button_name:
            # Create a dialog window with two input fields
            snr_window = QDialog(self)
            snr_window.setWindowTitle(f'Line Name and Parameter Constraints')
            
            # Create layout
            layout = QVBoxLayout()
            
            # Input box with placeholder text
            text_box = QLineEdit()
            # print(df)
            # print(f'frame_id = {frame_id}')
            # print(f'button_name = {button_name}')
            snr_value = df.loc[
                (df['region_ID'].astype(np.int64) == frame_id) & 
                (df['Line_ID'].astype(np.float64) == np.float64(button_name.split('~')[0])),
                'SNR'].item()
            # Convert to float safely
            try:
                snr_value = np.float64(snr_value)  # Converts valid numbers and "nan" strings to float
            except ValueError:
                snr_value = np.nan  # If conversion fails, assume NaN
                
            # Check for NaN properly
            if not np.isnan(snr_value):
                text_box.setText(str(snr_value))
            else:
                text_box.setPlaceholderText("Enter SNR threshold")
            
            layout.addWidget(text_box)    
            
            # Button layout (side-by-side buttons)
            button_layout = QHBoxLayout()
            
            # "Accept" button — keep the current SNR threshold and close.
            # Disabled until an SNR map has actually been calculated.
            accept_button = QPushButton("Accept")
            accept_button.setEnabled(False)
            accept_button.clicked.connect(snr_window.close)

            # "Calculate" button — compute and display the SNR map, then allow Accept
            visualize_button = QPushButton("Calculate")
            visualize_button.clicked.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='SNR'))
            visualize_button.clicked.connect(lambda: accept_button.setEnabled(True))
            button_layout.addWidget(visualize_button)

            button_layout.addWidget(accept_button)

            # "Cancel" button — reset to white-light image and close
            clear_button = QPushButton("Cancel")
            clear_button.clicked.connect(lambda: self.clear_snr_visualizer(frame_id, button_name))
            clear_button.clicked.connect(snr_window.close)
            button_layout.addWidget(clear_button)
            
            # Add button layout to main layout
            layout.addLayout(button_layout)
            
            # Set layout and show the window
            snr_window.setLayout(layout)
            snr_window.show()
            
        if ('_highlim' in button_name) or ('_lowlim' in button_name):
            _param = button_name.split('~')[1]
            _rowmask = ((df['region_ID'].astype(int)==frame_id) &
                        (df['Line_ID'].astype(float)==float(button_name.split('~')[0])))
            _raw = df.loc[_rowmask, _param].item()
            if _param.startswith('Sigma_0'):
                # σ bounds are stored in Å but edited in km/s
                _cen0 = df.loc[_rowmask, 'Centroid_0'].item()
                string = _fmt(sigma_wl_to_kms(_raw, _cen0))
            else:
                string = str(_raw)
            text_box = QLineEdit()
            text_box.setText(string)  # Set the initial value from the dataframe
            text_box.setGeometry(200, 200, 100, 30)
            text_box.show()
            # # # # Connect the returnPressed signal to handle text submission
            text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='param_limit'))
        
        else:
        
            if any(x in button_name for x in ['sourcename','redshift','resolvingpower']):
                if 'sourcename' in button_name:
                    self._show_source_dialog()
                else:
                    string = df_obs[button_name].item()
                    text_box = QLineEdit()
                    text_box.setText(string)
                    text_box.setGeometry(200, 200, 100, 30)
                    text_box.show()
                    text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='obs_button'))
            else:
            
            
                if button_name.split('~')[0] == 'continuum':
                    string = str(data_frame.loc[np.int64(data_frame['region_ID']) == frame_id][button_name.split('~')[1]].item())
                    button_type = 'continuum'
                elif button_name.split('~')[0] == 'stellar':
                    # Stellar continuum cells are handled by the dedicated block
                    # below; skip the per-line lookup (would fail on df_cont).
                    button_type = 'stellar'
                # elif button_name.split('~')[1] == 'Line_Name':
                #     string = str(data_frame.loc[(data_frame['region_ID'] == np.float64(frame_id)) & (np.float64(data_frame['Line_ID'])==np.float64(button_name.split(',')[0]))][button_name.split('~')[1]].item())
                else:
                    string = str(data_frame.loc[(data_frame['region_ID'].astype(float) == float(frame_id)) & (data_frame['Line_ID'].astype(float)==float(button_name.split('~')[0]))][button_name.split('~')[1]].item())
                    button_type = 'line'
                
            if button_name.split('~')[0] == 'continuum':
                cparam = button_name.split('~')[1]
                # x-ranges are locked to the base template while editing one spaxel
                if self._edit_mode() and cparam in ('x1', 'x2'):
                    return
                cur = data_frame.loc[np.int64(data_frame['region_ID']) == frame_id, cparam]
                current_val = str(cur.item()) if len(cur) else ''
                dialog = QDialog(self)
                dialog.setWindowTitle(f'Edit {cparam}')
                layout = QVBoxLayout(dialog)
                layout.addWidget(QLabel(f'{cparam}  (current: {current_val})'))
                tb = QLineEdit(); tb.setText(current_val); tb.selectAll()
                layout.addWidget(tb)
                btn_row = QHBoxLayout()
                ok_btn = QPushButton('OK'); ok_btn.setDefault(True)
                ok_btn.clicked.connect(lambda: [self.on_submit(tb, data_frame, frame_id, button_name, button_type='continuum_param'), dialog.accept()])
                tb.returnPressed.connect(lambda: [self.on_submit(tb, data_frame, frame_id, button_name, button_type='continuum_param'), dialog.accept()])
                cancel_btn = QPushButton('Cancel'); cancel_btn.clicked.connect(dialog.reject)
                btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
                layout.addLayout(btn_row)
                dialog.exec_()

            if button_name.split('~')[0] == 'stellar':
                sparam = button_name.split('~')[1]   # e.g. stellar_V_0
                cur = df_cont.loc[np.int64(df_cont['region_ID']) == frame_id, sparam]
                current_val = _fmt(cur.item()) if len(cur) else ''
                dialog = QDialog(self)
                dialog.setWindowTitle(f'Edit {sparam}')
                layout = QVBoxLayout(dialog)
                layout.addWidget(QLabel(f'{sparam}  (current: {current_val})'))
                tb = QLineEdit(); tb.setText(current_val); tb.selectAll()
                layout.addWidget(tb)
                btn_row = QHBoxLayout()
                ok_btn = QPushButton('OK'); ok_btn.setDefault(True)
                ok_btn.clicked.connect(lambda: [self.on_submit(tb, df_cont, frame_id, button_name, button_type='stellar_param'), dialog.accept()])
                tb.returnPressed.connect(lambda: [self.on_submit(tb, df_cont, frame_id, button_name, button_type='stellar_param'), dialog.accept()])
                cancel_btn = QPushButton('Cancel'); cancel_btn.clicked.connect(dialog.reject)
                btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
                layout.addLayout(btn_row)
                dialog.exec_()

            if any(p in button_name for p in ['Centroid_0', 'Amp_0', 'Sigma_0']):
                param = button_name.split('~')[1]
                line_id = np.int64(button_name.split('~')[0])
                _rowmask = ((np.int64(df['region_ID']) == frame_id) &
                            (np.int64(df['Line_ID']) == line_id))
                current = df.loc[_rowmask, param]
                _is_sigma = param.startswith('Sigma_0')
                if _is_sigma and len(current):
                    _cen0 = df.loc[_rowmask, 'Centroid_0'].item()
                    current_val = _fmt(sigma_wl_to_kms(current.item(), _cen0))
                else:
                    current_val = _fmt(current.item()) if len(current) else ''
                dialog = QDialog(self)
                dialog.setWindowTitle(f'Edit {param}')
                layout = QVBoxLayout(dialog)
                _unit = ' [km/s]' if _is_sigma else ''
                layout.addWidget(QLabel(f'{param}{_unit}  (current: {current_val})'))
                text_box_p = QLineEdit()
                text_box_p.setText(current_val)
                text_box_p.selectAll()
                layout.addWidget(text_box_p)
                btn_row = QHBoxLayout()
                ok_btn = QPushButton('OK')
                ok_btn.setDefault(True)
                ok_btn.clicked.connect(lambda: [self.on_submit(text_box_p, data_frame, frame_id, button_name, button_type='line_param'), dialog.accept()])
                text_box_p.returnPressed.connect(lambda: [self.on_submit(text_box_p, data_frame, frame_id, button_name, button_type='line_param'), dialog.accept()])
                cancel_btn = QPushButton('Cancel')
                cancel_btn.clicked.connect(dialog.reject)
                btn_row.addWidget(ok_btn)
                btn_row.addWidget(cancel_btn)
                layout.addLayout(btn_row)
                dialog.exec_()

            if 'Rest Wavelength' in button_name:
                text_box = QLineEdit()
                # text_box.setText('')  # Set the initial value from the dataframe
                text_box.show()
                text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='RestWavelength'))
    
            if 'Line_Name' in button_name:
                
                lineid = np.int64(button_name.split('~')[0])
                linename = df.loc[(df['region_ID'] == frame_id) & (np.float64(df['Line_ID'])==np.int64(button_name.split('~')[0]))]['Line_Name'].item()
                parname = button_name.split("~")[1].split("_")[0]            
    
                # Create a dialog window with two input fields
                dialog = QDialog(self)
                dialog.setWindowTitle(f'Line Name and Parameter Constraints')
    
                # Layout for the dialog
                layout = QVBoxLayout(dialog)
    
                # Create a horizontal layout for the label and checkmark
                label_layout = QHBoxLayout()
    
                # Label for Initial Guess
                label_line_name = QLabel('Line name:', dialog)
                label_layout.addWidget(label_line_name)
    
                # Create the confirmation label (green checkmark) and hide it initially
                confirm_label = QLabel('✔️', dialog)
                confirm_label.setStyleSheet("color: limegreen; font-size: 16px;")
                confirm_label.hide()
    
                # Add the checkmark and ensure it's right next to the label
                label_layout.addWidget(confirm_label)
    
                # Ensure no additional stretching pushes them apart - add a stretch to push to the right
                label_layout.addStretch(1)
    
                # Add the horizontal layout to the main vertical layout
                layout.addLayout(label_layout)
    
                # Function to show and hide the checkmark after 2 seconds
                def on_input_confirmed():
                    confirm_label.show()
                    QTimer.singleShot(2000, confirm_label.hide)
    
                # Text box for line name
                text_box = QLineEdit(dialog)
                text_box.setText(linename)  # Set the initial value from the dataframe
                text_box.setPlaceholderText('Enter new line name')
                text_box.returnPressed.connect(lambda: [self.on_submit(text_box, data_frame, frame_id, button_name, button_type='line_name'), on_input_confirmed()]) # Connect return pressed signal
    
                # Add the text box to the main vertical layout
                layout.addWidget(text_box)
    
                # Collect line names for help dialog and auto-suggest
                current_line_names = df.loc[df['region_ID'] == frame_id, 'Line_Name'].tolist()

                # Label for constraints box with "?" help button
                constraints_header = QHBoxLayout()
                label_constraints = QLabel('Parameter constraints:', dialog)
                constraints_header.addWidget(label_constraints)
                constraints_header.addStretch()
                help_btn = QPushButton('?', dialog)
                help_btn.setFixedSize(22, 22)
                help_btn.setToolTip('Constraint syntax help')
                help_btn.clicked.connect(lambda: self._show_constraints_help(dialog, current_line_names))
                constraints_header.addWidget(help_btn)
                layout.addLayout(constraints_header)

                # Auto-suggest button (wired after text boxes are created below)
                autosuggest_btn = QPushButton('Auto-suggest constraints', dialog)
                autosuggest_btn.setToolTip('Suggest constraints based on initial parameter values')
                layout.addWidget(autosuggest_btn)

                # Helper function to create a numbered QTextEdit with dynamic height
                def create_numbered_textbox(dialog, number, placeholder_text, initial_text=''):
                    box_layout = QHBoxLayout()
    
                    # Add the enumerator (e.g., 1., 2., etc.)
                    label_number = QLabel(f'{number}.', dialog)
                    box_layout.addWidget(label_number)
    
                    # Create the QTextEdit box
                    text_box_constraints = QTextEdit(dialog)
                    text_box_constraints.setPlainText(initial_text)  # Set initial text
                    text_box_constraints.setPlaceholderText(placeholder_text)
    
                    # Set dynamic height based on text metrics with padding
                    font_metrics = QFontMetrics(text_box_constraints.font())
                    text_height = font_metrics.height()
                    padding = 12  # Small padding for better appearance
                    text_box_constraints.setFixedHeight(text_height + padding)
    
                    box_layout.addWidget(text_box_constraints)
                    layout.addLayout(box_layout)
                    return text_box_constraints
    
                # Get existing constraints if available
                constraints_for_line = []
                if 'constraints' in df.columns:
                    try:
                        constraints_value = df.loc[(df['region_ID'] == frame_id) & (df['Line_Name'] == linename), 'constraints'].item()
                        if isinstance(constraints_value, str):
                            # Try to safely evaluate the string as a Python literal (list)
                            try:
                                constraints_list = ast.literal_eval(constraints_value)
                                if isinstance(constraints_list, list) and len(constraints_list) == 5:
                                    constraints_for_line = constraints_list
                                    print(f'constraints (from string): {constraints_for_line}')
                            except (ValueError, SyntaxError):
                                print(f"Warning: Could not parse constraints string: {constraints_value}")
                                constraints_for_line = ['', '', '', '', '']
                        elif isinstance(constraints_value, list) and len(constraints_value) == 5:
                            constraints_for_line = constraints_value
                            print(f'constraints (from list): {constraints_for_line}')
                        else:
                            constraints_for_line = ['', '', '', '', '']
                    except (KeyError, ValueError):
                        constraints_for_line = ['', '', '', '', '']
                else:
                    constraints_for_line = ['', '', '', '', '']
    
                # Create the constraints text boxes with enumerators and initial values
                text_box_constraints_01 = create_numbered_textbox(dialog, 1, f'e.g., amp <= amp_[line name]', constraints_for_line[0] if len(constraints_for_line) > 0 else '')
                text_box_constraints_02 = create_numbered_textbox(dialog, 2, '', constraints_for_line[1] if len(constraints_for_line) > 1 else '')
                text_box_constraints_03 = create_numbered_textbox(dialog, 3, '', constraints_for_line[2] if len(constraints_for_line) > 2 else '')
                text_box_constraints_04 = create_numbered_textbox(dialog, 4, '', constraints_for_line[3] if len(constraints_for_line) > 3 else '')
                text_box_constraints_05 = create_numbered_textbox(dialog, 5, '', constraints_for_line[4] if len(constraints_for_line) > 4 else '')
    
                # Wire auto-suggest now that all text boxes exist
                _tbs = [text_box_constraints_01, text_box_constraints_02,
                        text_box_constraints_03, text_box_constraints_04,
                        text_box_constraints_05]
                autosuggest_btn.clicked.connect(
                    lambda: self._apply_autosuggest(linename, frame_id, _tbs, dialog))

                # ── Kinematic group (K1–K5) ────────────────────────────────
                # Assigning a line to a K-group ties its velocity AND velocity
                # dispersion to the other members during fitting. At most one
                # group per line; clicking the active box again clears it.
                kgroup_header = QHBoxLayout()
                kgroup_label = QLabel('Kinematic group:', dialog)
                kgroup_header.addWidget(kgroup_label)
                kgroup_header.addStretch()
                kgroup_help = QPushButton('?', dialog)
                kgroup_help.setFixedSize(22, 22)
                kgroup_help.setToolTip('What are kinematic groups?')
                kgroup_help.clicked.connect(lambda: QMessageBox.information(
                    dialog, 'Kinematic Groups',
                    'Assign lines to the same K-group to tie their velocity AND '
                    'velocity dispersion together during fitting — they share '
                    'one kinematic solution (equal velocity and equal km/s '
                    'dispersion). The first line in the group (model order) is '
                    'the reference; the others are tied to it.\n\n'
                    'Select at most one group per line. Click the active box '
                    'again to remove this line from its group.'))
                kgroup_header.addWidget(kgroup_help)
                layout.addLayout(kgroup_header)

                kgroup_row = QHBoxLayout()
                kgroup_checkboxes = {}
                _current_kgroup = self._kgroup_of(linename)
                for _kg in self.KGROUP_LABELS:
                    cb = QCheckBox(_kg, dialog)
                    cb.setChecked(_kg == _current_kgroup)
                    kgroup_checkboxes[_kg] = cb
                    kgroup_row.addWidget(cb)
                kgroup_row.addStretch()
                layout.addLayout(kgroup_row)

                # Status line that surfaces the group's reference (anchor) line
                kgroup_ref_label = QLabel('', dialog)
                kgroup_ref_label.setWordWrap(True)
                kgroup_ref_label.setStyleSheet('color: gray; font-size: 11px;')
                layout.addWidget(kgroup_ref_label)

                def _refresh_kgroup_label():
                    sel = next((lbl for lbl, box in kgroup_checkboxes.items()
                                if box.isChecked()), '')
                    if not sel:
                        kgroup_ref_label.setText('')
                        return
                    # Would-be members of `sel` in model order, treating this
                    # line's pending selection as authoritative.
                    members = []
                    for n in self._unique_line_names():
                        g = sel if n == linename else self._kgroup_of(n)
                        if g == sel:
                            members.append(n)
                    if len(members) < 2:
                        kgroup_ref_label.setText(
                            f'{sel}: add another line to this group to form a tie.')
                    elif members[0] == linename:
                        kgroup_ref_label.setText(
                            f'This line is the {sel} reference — it holds the '
                            f"group's free velocity & dispersion.")
                    else:
                        kgroup_ref_label.setText(
                            f'Tied to {sel} reference: {members[0]}.')

                # Enforce mutual exclusivity (radio-like, but each is clearable)
                def _make_kgroup_handler(active_label):
                    def _handler(state):
                        if state:  # checking this one unchecks the others
                            for lbl, box in kgroup_checkboxes.items():
                                if lbl != active_label and box.isChecked():
                                    box.blockSignals(True)
                                    box.setChecked(False)
                                    box.blockSignals(False)
                        _refresh_kgroup_label()
                    return _handler
                for _kg, cb in kgroup_checkboxes.items():
                    cb.stateChanged.connect(_make_kgroup_handler(_kg))
                _refresh_kgroup_label()

                # Submit button to send constraints, with inline "saved" feedback
                submit_row = QHBoxLayout()
                submit_button = QPushButton('Submit Constraints', dialog, default=False, autoDefault=False)
                submit_confirm = QLabel('✔️ Constraints saved', dialog)
                submit_confirm.setStyleSheet("color: limegreen; font-weight: bold;")
                submit_confirm.hide()

                def _on_submit_constraints():
                    self.submit_constraints(_tbs, frame_id, lineid)
                    selected_group = next(
                        (lbl for lbl, box in kgroup_checkboxes.items() if box.isChecked()), '')
                    self._save_line_kgroup(linename, selected_group)
                    submit_confirm.show()
                    QTimer.singleShot(2500, submit_confirm.hide)

                submit_button.clicked.connect(_on_submit_constraints)
                submit_row.addWidget(submit_button)
                submit_row.addWidget(submit_confirm)
                submit_row.addStretch()
                layout.addLayout(submit_row)
    
    
                # Increase the width by 10%
                current_width = dialog.width()
                current_height = dialog.height()
                new_width = int(current_width * 2)  # 10% wider
                dialog.resize(350, 100)
    
                # Connect returnPressed for text_box (normal behavior)
                text_box.returnPressed.connect(lambda: [self.on_submit(text_box, data_frame, frame_id, button_name, button_type='line_name'), on_input_confirmed()])
    
                # Show the dialog
                dialog.exec_()

    def submit_constraints(self, text_box_constraints_list, frame_id, lineid):
        global df  # Assuming 'df' is a global DataFrame
        """Handles the submission of the constraints from all text boxes and adds them to the DataFrame."""
        all_constraints_text = [text_box.toPlainText() for text_box in text_box_constraints_list]
        # Ensure all_constraints_text has exactly 5 elements
        while len(all_constraints_text) < 5:
            all_constraints_text.append('')
        all_constraints_text = all_constraints_text[:5]
    
        # Find the row to update
        df = df.reset_index(drop=True)
        linename = df.loc[ (df['region_ID'] == frame_id) & (df['Line_ID'] == lineid)]['Line_Name']
        mask = (df['region_ID'] == frame_id) & (df['Line_ID'] == lineid)
    
        if 'constraints' not in df.columns:
            df['constraints'] = [[] for _ in range(len(df))]  # Initialize with empty lists
        for index, row in df.iterrows():
            if mask[index]:
                df.at[index, 'constraints'] = all_constraints_text
            elif not df.at[index, 'constraints'] or not isinstance(df.at[index, 'constraints'], list) or len(df.at[index, 'constraints']) != 5:
                # Only initialize with blank strings if the 'constraints' column
                # doesn't exist or is not a valid 5-element list
                df.at[index, 'constraints'] = ['', '', '', '', '']
    
        print(f"Constraints submitted for region_ID: {frame_id}, Line_Name: {linename}")
        print("Constraints:", all_constraints_text)

    def _show_constraints_help(self, parent, line_names):
        lines_str = ', '.join(line_names) if line_names else '(none defined)'
        text = (
            "<b>Constraint Syntax Reference</b><br><br>"
            "<b>Parameters:</b><br>"
            "&nbsp;&nbsp;<tt>amp</tt> — amplitude (peak height)<br>"
            "&nbsp;&nbsp;<tt>flux</tt> — integrated line flux (amp·&sigma;·&radic;2&pi;)<br>"
            "&nbsp;&nbsp;<tt>sigma</tt> — line width (&sigma;)<br>"
            "&nbsp;&nbsp;<tt>cen</tt> — centroid (wavelength)<br>"
            "&nbsp;&nbsp;<tt>vel</tt> — velocity (converted to centroid internally)<br><br>"
            "<b>Operators:</b>&nbsp;&nbsp;"
            "<tt>&lt;=&nbsp;&nbsp;&lt;&nbsp;&nbsp;&gt;=&nbsp;&nbsp;&gt;&nbsp;&nbsp;==</tt><br><br>"
            "<b>Format:</b><br>"
            "&nbsp;&nbsp;<tt>param op param_[line_name]</tt><br>"
            "&nbsp;&nbsp;<tt>param op 0.5 * param_[line_name]</tt><br><br>"
            "<b>Examples:</b><br>"
            "&nbsp;&nbsp;<tt>amp &lt;= amp_[Halpha]</tt><br>"
            "&nbsp;&nbsp;<tt>sigma &gt;= sigma_[Halpha_b]</tt><br>"
            "&nbsp;&nbsp;<tt>vel == vel_[nii_1]</tt><br>"
            "&nbsp;&nbsp;<tt>amp &lt;= 0.33 * amp_[Halpha]</tt><br>"
            "&nbsp;&nbsp;<tt>flux == 2.94 * flux_[[N II]_6548]</tt> — fixed flux ratio<br>"
            "&nbsp;&nbsp;<tt>flux == 0.44..1.45 * flux_[[S II]_6731]</tt> — flux ratio within a range<br><br>"
            "<b>Velocity ties & windows</b> (Δv in km/s, relative to the "
            "reference line):<br>"
            "&nbsp;&nbsp;<tt>vel == vel_[nii_1]</tt> — same velocity (Δv = 0)<br>"
            "&nbsp;&nbsp;<tt>vel == vel_[nii_1] +- 300</tt> — within ±300 km/s<br>"
            "&nbsp;&nbsp;<tt>vel &lt;= vel_[nii_1] + 300</tt> — at most 300 km/s redward<br>"
            "&nbsp;&nbsp;<tt>vel &gt;= vel_[nii_1] - 300</tt> — at least 300 km/s blueward<br>"
            "<i>(a two-sided minimum |Δv| ≥ X is not supported — pick a side.)</i><br><br>"
            "<b>Tip:</b> to tie velocities across several lines at once, use the "
            "<b>Kinematic group</b> (K1–K5) checkboxes below.<br><br>"
            f"<b>Lines in this region:</b><br>&nbsp;&nbsp;<tt>{lines_str}</tt>"
        )
        dlg = QDialog(parent)
        dlg.setWindowTitle('Constraint Syntax Help')
        dlg.setMinimumWidth(380)
        lay = QVBoxLayout(dlg)
        lbl = QLabel(text, dlg)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        lay.addWidget(lbl)
        ok = QPushButton('OK', dlg)
        ok.setDefault(True)
        ok.clicked.connect(dlg.accept)
        lay.addWidget(ok, alignment=Qt.AlignRight)
        dlg.exec_()

    def _suggest_constraints(self, line_name, frame_id):
        """
        Infer constraints by comparing Sigma_0 against partner lines sharing
        the same base name (strip trailing _[a-z] suffix) in the same region.
        Returns list of dicts: {'text', 'amp_direction_choice', 'base'}.
        """
        import re
        base = re.sub(r'_[a-z]$', '', line_name)
        region_df = df[df['region_ID'] == frame_id]
        partners = region_df[
            (region_df['Line_Name'] != line_name) &
            region_df['Line_Name'].str.lower().str.startswith(base.lower())
        ]
        if partners.empty:
            return []

        cur_sigma = float(region_df.loc[region_df['Line_Name'] == line_name, 'Sigma_0'].iloc[0])
        partner = partners.iloc[
            (partners['Sigma_0'].astype(float) - cur_sigma).abs().values.argmax()
        ]
        partner_name = partner['Line_Name']
        partner_sigma = float(partner['Sigma_0'])

        suggestions = []
        rel_diff = abs(cur_sigma - partner_sigma) / max(cur_sigma, partner_sigma, 1e-10)
        if rel_diff > 0.05:
            op = '>=' if cur_sigma > partner_sigma else '<='
            suggestions.append({'text': f'sigma {op} sigma_[{partner_name}]',
                                 'amp_direction_choice': False, 'base': partner_name})
        suggestions.append({'text': f'amp ?? amp_[{partner_name}]',
                            'amp_direction_choice': True, 'base': partner_name})
        return suggestions[:5]

    def _ask_amp_direction(self, parent, line_name, base_name):
        """Prompt user to choose amplitude constraint direction. Returns '<=', '>=' or 'skip'."""
        dlg = QDialog(parent)
        dlg.setWindowTitle('Amplitude direction')
        lay = QVBoxLayout(dlg)
        lbl = QLabel(f'Constrain <b>{line_name}</b> amplitude relative to <b>{base_name}</b>?', dlg)
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        result = ['<=']
        btn_fainter  = QPushButton(f'amp <= amp_[{base_name}]   (this component is fainter)', dlg)
        btn_brighter = QPushButton(f'amp >= amp_[{base_name}]   (this component is brighter)', dlg)
        btn_skip     = QPushButton('Skip amplitude constraint', dlg)
        btn_fainter.clicked.connect( lambda: [result.__setitem__(0, '<='),   dlg.accept()])
        btn_brighter.clicked.connect(lambda: [result.__setitem__(0, '>='),   dlg.accept()])
        btn_skip.clicked.connect(    lambda: [result.__setitem__(0, 'skip'), dlg.accept()])
        for b in (btn_fainter, btn_brighter, btn_skip):
            lay.addWidget(b)
        dlg.exec_()
        return result[0]

    def _apply_autosuggest(self, line_name, frame_id, text_boxes, parent):
        raw = self._suggest_constraints(line_name, frame_id)
        if not raw:
            QMessageBox.information(
                parent, 'Auto-suggest',
                'No suggestions available — no other components share this '
                "line's base name in this region.")
            return
        suggestions = []
        for item in raw:
            if item['amp_direction_choice']:
                direction = self._ask_amp_direction(parent, line_name, item['base'])
                if direction == 'skip':
                    continue
                suggestions.append(item['text'].replace('??', direction))
            else:
                suggestions.append(item['text'])
        if not suggestions:
            return
        msg = 'Suggested constraints:\n\n'
        msg += '\n'.join(f'  {i+1}. {s}' for i, s in enumerate(suggestions))
        msg += '\n\nApply to constraint boxes? (replaces current content)'
        reply = QMessageBox.question(parent, 'Auto-suggest Constraints', msg,
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            for i, tb in enumerate(text_boxes):
                tb.setPlainText(suggestions[i] if i < len(suggestions) else '')

    # ── Kinematic groups (velocity ties) ────────────────────────────────────
    def _normalize_constraint_list(self, val):
        """Coerce a stored constraints value into a 5-element list of strings."""
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                val = parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                val = []
        if not isinstance(val, list):
            val = []
        val = [str(c) for c in val]
        while len(val) < 5:
            val.append('')
        return val[:5]

    def _unique_line_names(self):
        """Ordered unique emission-line names in the model."""
        if 'Line_Name' not in df.columns:
            return []
        seen = []
        for n in df['Line_Name'].tolist():
            n = str(n)
            if n and n != 'nan' and n not in seen:
                seen.append(n)
        return seen

    def _get_velocity_tie(self, line_name):
        """Return the reference line this line's velocity is tied to, or None."""
        if 'constraints' not in df.columns:
            return None
        for val in df.loc[df['Line_Name'] == line_name, 'constraints']:
            for c in self._normalize_constraint_list(val):
                m = re.search(r'vel\s*==\s*vel_\[(.+?)\]', c)
                if m:
                    return m.group(1)
        return None

    # Signature identifying a K-group-managed dispersion tie: equality WITH a
    # numeric factor (e.g. 'sigma == 1.002255 * sigma_[ref]'). This lets us
    # strip K-group ties without touching a user's 'sigma <=/>= sigma_[X]'.
    _KG_SIGMA_RE = re.compile(r'sigma\s*==\s*[\d.]+\s*\*\s*sigma_\[')
    # K-group-managed velocity tie is the *exact* form 'vel == vel_[ref]'
    # (anchored, no trailing window). This must NOT match a user's manual
    # velocity window/one-sided form ('vel == vel_[B] +- 300', 'vel <= ...').
    _KG_VEL_RE = re.compile(r'^\s*vel\s*==\s*vel_\[.*\]\s*$')

    # Signatures identifying constraints authored by Smart Constraints, so
    # re-running it (e.g. switching scenario) replaces its own prior output
    # instead of stacking duplicates, while leaving a user's own manually
    # typed constraints untouched. A manual constraint that happens to match
    # one of these forms is treated as smart-managed too — same accepted
    # trade-off as the K-group tie signatures above.
    _SMART_FLUX_RE = re.compile(r'^\s*flux\s*==\s*[\d.]+(?:\.\.[\d.]+)?\s*\*\s*flux_\[')
    _SMART_SIGMA_REL_RE = re.compile(r'^\s*sigma\s*>=\s*sigma_\[')
    _SMART_AMP_REL_RE = re.compile(r'^\s*amp\s*<=\s*amp_\[')

    def _rewrite_smart_constraints(self, line_name, add=None):
        """Rewrite a line's constraints, replacing only Smart-Constraints-
        managed entries (flux ratio / relational sigma / relational amp).
        Mirrors `_rewrite_kgroup_ties`.

        Uses positional indexing (df can carry duplicate index labels, built
        via pd.concat without ignore_index — see save_gaussian's comment —
        so `.at[label, ...]` degrades to `.loc` and raises a length-mismatch
        the moment two lines share a label)."""
        global df
        if 'constraints' not in df.columns:
            df['constraints'] = [['', '', '', '', ''] for _ in range(len(df))]
        col = df.columns.get_loc('constraints')
        for pos in np.where((df['Line_Name'] == line_name).to_numpy())[0]:
            clist = self._normalize_constraint_list(df.iloc[pos, col])
            kept = [c for c in clist if c.strip()
                    and not self._SMART_FLUX_RE.match(c)
                    and not self._SMART_SIGMA_REL_RE.match(c)
                    and not self._SMART_AMP_REL_RE.match(c)]
            if add:
                kept = kept + list(add)
            kept = kept[-5:]
            while len(kept) < 5:
                kept.append('')
            df.iat[pos, col] = kept

    def _rest_wl(self, line_name):
        """Rest wavelength (Å) for a line, or None if missing/invalid."""
        try:
            v = float(df.loc[df['Line_Name'] == line_name, 'Rest Wavelength'].iloc[0])
            return v if (np.isfinite(v) and v > 0) else None
        except (IndexError, KeyError, ValueError, TypeError):
            return None

    def _rewrite_kgroup_ties(self, line_name, add=None):
        """Rewrite a line's constraints, replacing only K-group-managed ties.

        Drops any velocity tie and the K-group sigma-tie signature, then (if
        `add`) appends the new ties LAST so they take precedence over any manual
        sigma constraint in add_dataframe_constraints_to_params (non-destructive:
        the manual constraint stays in the list and re-activates once the ties
        are removed).

        Uses positional indexing (df can carry duplicate index labels, built
        via pd.concat without ignore_index — see save_gaussian's comment —
        so `.at[label, ...]` degrades to `.loc` and raises a length-mismatch
        the moment two lines share a label).
        """
        global df
        if 'constraints' not in df.columns:
            df['constraints'] = [['', '', '', '', ''] for _ in range(len(df))]
        col = df.columns.get_loc('constraints')
        for pos in np.where((df['Line_Name'] == line_name).to_numpy())[0]:
            clist = self._normalize_constraint_list(df.iloc[pos, col])
            kept = [c for c in clist if c.strip()
                    and not self._KG_VEL_RE.match(c)
                    and not self._KG_SIGMA_RE.search(c)]
            if add:
                kept = kept + list(add)      # ties last → override manual sigma
            kept = kept[-5:]                  # keep ties + most-recent manual (5 slots)
            while len(kept) < 5:
                kept.append('')
            df.iat[pos, col] = kept

    def _set_kgroup_ties(self, line_name, reference):
        """Tie this line's velocity AND velocity dispersion to the reference.

        Dispersion is tied in km/s via the rest-wavelength ratio so members
        share one velocity dispersion: sigma = (rest_self/rest_ref) * sigma_ref.
        """
        r_self, r_ref = self._rest_wl(line_name), self._rest_wl(reference)
        ratio = (r_self / r_ref) if (r_self and r_ref) else 1.0
        ties = [f'vel == vel_[{reference}]',
                f'sigma == {ratio:.6f} * sigma_[{reference}]']
        self._rewrite_kgroup_ties(line_name, add=ties)

    def _clear_kgroup_ties(self, line_name):
        self._rewrite_kgroup_ties(line_name, add=None)

    # ── K-group membership (per-line, set from the Line Name dialog) ─────────
    KGROUP_LABELS = ['K1', 'K2', 'K3', 'K4', 'K5']

    def _kgroup_of(self, line_name):
        """Return the K-group label ('K1'..'K5') assigned to a line, or ''."""
        if 'kgroup' not in df.columns:
            return ''
        vals = df.loc[df['Line_Name'] == line_name, 'kgroup']
        for v in vals:
            v = str(v)
            if v in self.KGROUP_LABELS:
                return v
        return ''

    def _kgroup_reference(self, group_label):
        """The reference (anchor) line of a K-group: first member in model order.

        Returns None for a group with fewer than two members (no tie yet).
        """
        members = [n for n in self._unique_line_names()
                   if self._kgroup_of(n) == group_label]
        return members[0] if len(members) >= 2 else None

    def _save_line_kgroup(self, line_name, group):
        """Persist a line's K-group ('' clears it) and refresh kinematic ties."""
        global df
        if 'kgroup' not in df.columns:
            df['kgroup'] = ['' for _ in range(len(df))]
        df.loc[df['Line_Name'] == line_name, 'kgroup'] = group
        if not group:
            # A line just removed from its group is no longer rebuilt by the
            # sync below (which only touches grouped lines), so clear its ties now.
            self._clear_kgroup_ties(line_name)
        self._sync_kgroup_constraints()

    def _sync_kgroup_constraints(self):
        """Materialize velocity + dispersion ties from K-group membership.

        Within each group the first line (model order) is the reference anchor;
        every other member gets ``vel == vel_[reference]`` and a matching
        velocity-dispersion tie. Lines with no K-group are left untouched so
        manual constraints are preserved.
        """
        if 'kgroup' not in df.columns:
            return
        grouped = [n for n in self._unique_line_names() if self._kgroup_of(n)]
        groups = {}
        for n in grouped:
            groups.setdefault(self._kgroup_of(n), []).append(n)
        # Clear ties on all currently grouped lines, then rebuild.
        for n in grouped:
            self._clear_kgroup_ties(n)
        for members in groups.values():
            if len(members) < 2:
                continue
            ref = members[0]
            for m in members[1:]:
                self._set_kgroup_ties(m, ref)

    def _line_overlay_color(self, line_id):
        """SVG color used for this line's component curve in the spectrum view.

        Mirrors the lineid_to_color mapping built at fit time so the Line Name
        button border matches the dashed component overlay.
        """
        svg_colors = [
            'dodgerblue', 'mediumseagreen', 'darkorange', 'mediumpurple',
            'deepskyblue', 'gold', 'steelblue', 'mediumaquamarine',
            'peru', 'cornflowerblue'
        ]
        try:
            unique_lines = list(pd.unique(df['Line_ID'].astype(float)))
            idx = unique_lines.index(float(line_id))
        except (ValueError, KeyError, TypeError):
            return None
        return svg_colors[idx % len(svg_colors)]

    def on_submit(self, text_box, data_frame, frame_id, button_name, button_type):
        global df_cont, df, snrmap, snr_value
        """Submit the value from the text box, update the dataframe, and adjust related parameters."""
        
        if button_type == 'SNR':
            snr_value = np.float64(text_box.text())
            linewl = df.loc[(df['region_ID'].astype(int)==frame_id) &
                            (df['Line_ID'].astype(float)==float(button_name.split('~')[0]))]['Centroid_0']
            self.viewer_window.ax.cla()
            self.viewer_window.draw_image(FITS_DATA,cmap='gray',scale='linear',from_fits=True)
            # snrmap = self.calculate_snr_map(linewl, snr_value, search_window_factor=5, continuum_offset_factor=20, continuum_width_factor=30)
            snrmap = self.calculate_snr_map(linewl, snr_value, search_window_width=None, continuum_offset=None, continuum_width=None)
            self.update_button_value(frame_id, button_name, snr_value)
        if button_type == 'obs_button':
            new_value = np.float64(text_box.text()) if 'name' not in button_name else str(text_box.text())
            data_frame[button_name] = new_value
            self.update_button_value(frame_id, button_name, new_value)
            
        elif button_type == 'param_limit':
            param = button_name.split('~')[1]
            line_id = np.int64(button_name.split('~')[0])
            rowmask = (np.int64(df['region_ID']) == frame_id) & (np.int64(df['Line_ID']) == line_id)
            entered = np.float64(text_box.text())
            if param.startswith('Sigma_0'):
                # σ bounds are entered in km/s; store the Å equivalent
                cen0 = df.loc[rowmask, 'Centroid_0'].iloc[0]
                new_value = sigma_kms_to_wl(entered, cen0)
                display = _fmt(entered)
            else:
                new_value = entered
                display = new_value
            df.loc[rowmask, param] = new_value
            self.update_button_value(frame_id, button_name, display)
            
        elif button_type == 'line_param':
            try:
                entered = np.float64(text_box.text())
            except ValueError:
                print(f"Invalid value: {text_box.text()}")
                return
            param = button_name.split('~')[1]
            line_id = np.int64(button_name.split('~')[0])
            rowmask = (np.int64(df['region_ID']) == frame_id) & (np.int64(df['Line_ID']) == line_id)
            if param.startswith('Sigma_0'):
                # σ is entered in km/s; store the Å equivalent
                cen0 = df.loc[rowmask, 'Centroid_0'].iloc[0]
                new_value = sigma_kms_to_wl(entered, cen0)
                display = _fmt(entered)
            else:
                new_value = entered
                display = _fmt(new_value)
            df.loc[rowmask, param] = new_value
            self.update_button_value(frame_id, button_name, display)
            # Redraw the component curves to reflect the new initial guess
            vw = self.viewer_window
            for line in vw.spectrum_ax.lines[:]:
                if line.get_linestyle() in ('--', 'dashed'):
                    try: line.remove()
                    except ValueError: pass
            vw.gaussian_component_lines.clear()
            for actor in df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor']:
                if actor and actor in vw.spectrum_ax.lines:
                    try: actor.remove()
                    except ValueError: pass
            vw.rebuild_plot(region_ID=frame_id, from_file=False, show_init=True, show_fit=False, x=0, y=0)

        elif button_type == 'continuum_param':
            param = button_name.split('~')[1]
            raw = text_box.text()
            if param == 'Continuum Name':
                new_value = str(raw)
            else:
                try:
                    new_value = np.float64(raw)
                except ValueError:
                    print(f"Invalid value: {raw}")
                    return
            df_cont.loc[np.int64(df_cont['region_ID']) == frame_id, param] = new_value
            self.update_button_value(
                frame_id, button_name,
                new_value if param == 'Continuum Name' else _fmt(new_value))
            # Redraw this region's continuum + init curves to reflect the change
            vw = self.viewer_window
            for line in vw.spectrum_ax.lines[:]:
                if line.get_linestyle() in ('--', 'dashed'):
                    try: line.remove()
                    except ValueError: pass
            vw.gaussian_component_lines.clear()
            for actor in df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor']:
                if actor is not None and actor in vw.spectrum_ax.lines:
                    try: actor.remove()
                    except ValueError: pass
            vw.rebuild_plot(region_ID=frame_id, from_file=False, show_init=True, show_fit=False, x=0, y=0)

        elif button_type == 'stellar_param':
            param = button_name.split('~')[1]   # e.g. stellar_V_0
            try:
                new_value = np.float64(text_box.text())
            except ValueError:
                print(f"Invalid value: {text_box.text()}")
                return
            df_cont.loc[np.int64(df_cont['region_ID']) == frame_id, param] = new_value
            self.update_button_value(frame_id, button_name, _fmt(new_value))
            # Re-render the stellar baseline (no pPXF) for the current spaxel.
            vw = self.viewer_window
            cx = cy = 0
            if vw.current_spaxel is not None:
                cx, cy = int(vw.current_spaxel[0]), int(vw.current_spaxel[1])
            # Push the edit into this spaxel/region's cache so the baseline reflects it.
            # Ensure the cache exists first (parallel fits store kinematics only).
            vw._ensure_stellar_cache(cx, cy, int(np.int64(frame_id)))
            _cache = STELLAR_CACHE.get((cx, cy, int(np.int64(frame_id))))
            if _cache is not None:
                _ckey = {'stellar_V_0': 'V', 'stellar_sigma_0': 'sigma',
                         'stellar_scale_0': 'scale', 'stellar_h3_0': 'h3',
                         'stellar_h4_0': 'h4'}.get(param)
                if _ckey:
                    _cache[_ckey] = float(new_value)
            for line in vw.spectrum_ax.lines[:]:
                if line.get_linestyle() in ('--', 'dashed'):
                    try: line.remove()
                    except ValueError: pass
            vw.gaussian_component_lines.clear()
            for actor in df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor']:
                if actor is not None and actor in vw.spectrum_ax.lines:
                    try: actor.remove()
                    except ValueError: pass
            vw.rebuild_plot(region_ID=frame_id, from_file=False, show_init=True, show_fit=False, x=cx, y=cy)

        elif button_type == 'RestWavelength':
            new_value = ast.literal_eval(text_box.text())
            df.loc[(df['region_ID']==frame_id) &
                            (np.int64(df['Line_ID']==np.int64(button_name.split('~')[0]))),'Rest Wavelength'] = new_value
            self.update_button_value(frame_id, button_name, text_box.text())
        
        else:
            
            try:
                
                # Extract new value from the text box
                new_value = np.float64(text_box.text()) if 'Name' not in button_name else str(text_box.text())
        
                # Identify which parameter is being updated
                param = button_name.split('~')[1]
                
            except ValueError:
                print(f"Invalid input: {text_box.text()} is not valid for this column.")
        
                if button_type == 'continuum':
                    
                    # Directly update the dataframe with the new value
                    data_frame.loc[data_frame['region_ID'] == frame_id, param] = new_value
                    
                    # **Important**: Fetch updated values *after* setting the new value
                    x1 = data_frame.loc[data_frame['region_ID'] == frame_id, 'x1'].values[0]
                    x2 = data_frame.loc[data_frame['region_ID'] == frame_id, 'x2'].values[0]
                    slope = data_frame.loc[data_frame['region_ID'] == frame_id, 'Slope_0'].values[0]
                    intercept = data_frame.loc[data_frame['region_ID'] == frame_id, 'Intercept_0'].values[0]
    
        
                    # Adjust other values based on what changed
                    if param == 'x1' or param == 'x2':
                        # Ensure y1 and y2 remain fixed, update slope and intercept accordingly
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
                        slope = (y2 - y1) / (x2 - x1) if x1 != x2 else 0
                        intercept = y1 - slope * x1
            
                        # **Re-store slope & intercept in dataframe** after updating them
                        data_frame.loc[data_frame['region_ID'] == frame_id, 'Slope_0'] = slope
                        data_frame.loc[data_frame['region_ID'] == frame_id, 'Intercept_0'] = intercept
            
                    elif param == 'Slope_0':
                        # Recalculate y-values based on the new slope
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
            
                    elif param == 'Intercept_0':
                        # Recalculate y-values based on the new intercept
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
            
                    # **Ensure UI updates instantly**: Force refresh of button values
                    self.update_button_value(frame_id, 'continuum~x1', x1)
                    self.update_button_value(frame_id, 'continuum~x2', x2)
                    self.update_button_value(frame_id, 'continuum~Slope_0', slope)
                    self.update_button_value(frame_id, 'continuum~Intercept_0', intercept)
            
                    # **Redraw the updated line with correct values**
                    self.viewer_window.update_line_with_slope_intercept(x1, x2, slope, intercept)
        
            if button_type == 'line':
                df.loc[(df['region_ID'] == np.float64(frame_id)) & (df['Line_ID'] == np.float64(button_name.split('~')[0])), param] = new_value
                self.update_button_value(frame_id, button_name, new_value)
                df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
                self.viewer_window.spectrum_canvas.draw_idle()  # Efficient redrawing
                self.viewer_window.rebuild_plot(frame_id, from_file=False,show_init=True,show_fit=False,x=0,y=0)
            
            if button_type == 'line_name':
                df.loc[(df['region_ID'] == np.float64(frame_id)) & (df['Line_ID'] == np.float64(button_name.split('~')[0])), param] = new_value
                print(f'frame id: {frame_id}, button name: {button_name}, new name: {new_value}')
                self.update_button_value(frame_id, button_name, new_value)
                df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
                self.viewer_window.spectrum_canvas.draw_idle()  # Efficient redrawing
                self.viewer_window.rebuild_plot(frame_id, from_file=False,show_init=True,show_fit=False,x=0,y=0)
    
        # Hide the text box after submission
        if (button_type != 'line_name') & (button_type != 'SNR'):
            text_box.hide()

def load_pixmap(image_path):
    """Load QPixmap with proper error handling"""
    pixmap = QPixmap()
    if not pixmap.load(image_path):
        # Try alternative loading method
        with open(image_path, 'rb') as f:
            pixmap.loadFromData(f.read())
    if pixmap.isNull():
        raise ValueError(f"Failed to load image: {image_path}\n"
                       f"Supported formats: {', '.join(QPixmap.supportedImageFormats())}")
    return pixmap

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent

    # Try multiple possible locations
    search_paths = [
        base_path / relative_path,
        base_path / 'Resources' / relative_path,
        Path.cwd() / relative_path
    ]
    
    for path in search_paths:
        if path.exists():
            print(f"Found resource at: {path}")
            return str(path)
    
    raise FileNotFoundError(f"Resource not found in any of: {search_paths}")
    
    
def load_stylesheet(app,filename):
    try:
        stylesheet_path = resource_path(filename)
        print(f"Loading stylesheet from: {stylesheet_path}")
        
        with open(stylesheet_path, "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
        # Fallback to basic dark theme
        app.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)    
    
def main():
    """Entry point for Hypercube spectral analysis tool."""
    try:
        app = QApplication(sys.argv)
        load_stylesheet(app, 'QDarkOrange_style.qss')
        
        
        print("Application starting...")
        
        # Load splash image
        try:
            image_path = resource_path("hypercube_logo.png")
            print(f"Loading image from: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at: {image_path}")
            
            pixmap = load_pixmap(image_path)
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            print("Image loaded successfully")
            
        except Exception as e:
            print(f"Image loading failed: {str(e)}")
            # Fallback to blank pixmap
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.white)
        
        
        # Show splash screen
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        print("Splash screen shown")
        
        splash.raise_()
        splash.repaint()
        app.processEvents()
        print("Events processed")
        
        # Create main window
        print("Creating main window...")
        viewer = ViewerWindow()
        
        # Set up transition
        def show_main_window():
            print("Closing splash and showing main window")
            splash.finish(viewer)
            viewer.show()
        
        QTimer.singleShot(100, show_main_window)
        print("Timer set")
        
        print("Entering main loop")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()