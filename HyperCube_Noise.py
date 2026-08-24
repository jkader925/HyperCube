"""Measurement-error (variance / error cube) discovery + empirical noise estimation.

Qt-free by design: imports only numpy + astropy so it can be used by the GUI, by
the batch fit kernel (`HyperCube_fit`) and inside multiprocessing workers.

Two ways to get a per-pixel 1σ flux uncertainty for a spaxel:

  1. **From the data** — a variance / error / inverse-variance cube, either as an
     extension of the science file (JWST s3d `ERR`, MUSE `STAT`, …) or as a
     sidecar file next to it (KCWI DRP `*_icubes.fits` + `*_vcubes.fits`).
     `detect()` finds one silently at cube-ingest time; the user can override the
     choice from the "Measurement Errors…" dialog.

  2. **Empirically** — `empirical_sigma()` measures the noise directly from the
     line-free pixels *inside each fit window* of the spectrum being fitted, with
     the DER_SNR estimator (Stoehr et al. 2008): a MAD-style statistic built on
     second differences, so a sloped or curved continuum does not inflate it and
     a minority of line pixels cannot either.

Whichever is used is recorded per spaxel in the fit output (`noise_source`), so a
fit table always says how its uncertainties were derived.
"""

import os
import re

import numpy as np

try:
    from astropy.io import fits
except Exception:                                    # pragma: no cover
    fits = None


# ── error-cube flavours ──────────────────────────────────────────────────────
KIND_SIGMA = 'sigma'        # the extension already holds 1σ
KIND_VAR = 'variance'       # holds σ²
KIND_IVAR = 'ivar'          # holds 1/σ²

KIND_LABELS = {
    KIND_SIGMA: '1σ error',
    KIND_VAR: 'variance (σ²)',
    KIND_IVAR: 'inverse variance (1/σ²)',
}

# Short tokens written to the fit table's `noise_source` column.
SRC_EMPIRICAL = 'empirical-DER_SNR'
SRC_NONE = 'none'

# Substring → kind, most specific first ('IVAR' must beat 'VAR', 'VARIANCE'
# must beat 'VAR', 'FLUXERR' must beat 'ERR').
_KIND_PATTERNS = (
    ('INVERSEVARIANCE', KIND_IVAR),
    ('INVVAR', KIND_IVAR),
    ('IVAR', KIND_IVAR),
    ('FLUXVAR', KIND_VAR),
    ('VARIANCE', KIND_VAR),
    ('VCUBE', KIND_VAR),
    ('STAT', KIND_VAR),          # MUSE / ESO convention
    ('VAR', KIND_VAR),
    ('FLUXERR', KIND_SIGMA),
    ('ERRDATA', KIND_SIGMA),
    ('ERROR', KIND_SIGMA),
    ('ERR', KIND_SIGMA),
    ('SIGMA', KIND_SIGMA),
    ('STDDEV', KIND_SIGMA),
    ('STDEV', KIND_SIGMA),
    ('NOISE', KIND_SIGMA),
    ('UNCERT', KIND_SIGMA),
)


def guess_kind(name):
    """Guess the error flavour from an EXTNAME / filename token. None if no match."""
    key = re.sub(r'[^A-Z0-9]', '', str(name or '').upper())
    if not key:
        return None
    for pat, kind in _KIND_PATTERNS:
        if pat in key:
            return kind
    return None


def to_sigma(data, kind):
    """Convert a variance / ivar / sigma array to 1σ (float32).

    Non-physical entries (negative variance, non-positive ivar) become NaN so the
    fit can drop those pixels instead of trusting them.
    """
    arr = np.asarray(data, dtype=np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        if kind == KIND_VAR:
            out = np.sqrt(np.where(arr > 0, arr, np.nan))
        elif kind == KIND_IVAR:
            out = 1.0 / np.sqrt(np.where(arr > 0, arr, np.nan))
        else:
            out = np.abs(arr)
            out = np.where(out > 0, out, np.nan)
    return np.asarray(out, dtype=np.float32)


# ── discovery ────────────────────────────────────────────────────────────────
def _squeeze_shape(shape):
    """Drop leading length-1 axes (ALMA-style Stokes) for shape comparison."""
    s = tuple(int(v) for v in shape)
    while len(s) > 3 and s[0] == 1:
        s = s[1:]
    return s


def candidate_extensions(path, sci_shape=None, sci_ext=None):
    """List the extensions of `path` that could hold measurement errors.

    Returns a list of dicts: {'ext', 'name', 'shape', 'kind', 'match'} where
    `kind` is the name-based guess (None if the name says nothing) and `match` is
    True when the shape agrees with `sci_shape`. Every data extension is listed —
    the dialog shows them all so a user can pick one whose name we don't know.
    """
    out = []
    if fits is None or not path or not os.path.exists(path):
        return out
    want = _squeeze_shape(sci_shape) if sci_shape is not None else None
    try:
        with fits.open(path, memmap=True) as hdul:
            for i, hdu in enumerate(hdul):
                shape = getattr(hdu, 'shape', None)
                if not shape or len(shape) < 1:
                    continue
                if getattr(hdu.header, 'get', lambda *_: None)('XTENSION') == 'BINTABLE':
                    continue
                name = hdu.name or f'EXT{i}'
                sq = _squeeze_shape(shape)
                out.append({
                    'ext': i,
                    'name': str(name),
                    'shape': tuple(int(v) for v in shape),
                    'kind': guess_kind(name),
                    'match': (want is None or sq == want),
                    'is_sci': (sci_ext is not None and i == int(sci_ext)),
                })
    except Exception as e:
        print(f'Measurement errors: could not scan {os.path.basename(str(path))}: {e}')
    return out


# Sidecar filename rules: (substring to replace, replacement, kind).
# A `None` search term means "append the suffix to the stem".
_SIDECAR_RULES = (
    ('_icubes', '_vcubes', KIND_VAR),        # KCWI DRP
    ('_icube', '_vcube', KIND_VAR),
    ('icubes', 'vcubes', KIND_VAR),
    ('icube', 'vcube', KIND_VAR),
    ('_sci', '_err', KIND_SIGMA),
    ('_sci', '_var', KIND_VAR),
    ('_flux', '_err', KIND_SIGMA),
    ('_flux', '_var', KIND_VAR),
    (None, '_var', KIND_VAR),
    (None, '_variance', KIND_VAR),
    (None, '_vcube', KIND_VAR),
    (None, '_err', KIND_SIGMA),
    (None, '_error', KIND_SIGMA),
    (None, '_sigma', KIND_SIGMA),
    (None, '_noise', KIND_SIGMA),
    (None, '_ivar', KIND_IVAR),
)


def sidecar_candidates(path):
    """Candidate companion error files next to `path`. [(path, kind), …]."""
    out = []
    if not path:
        return out
    directory = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    if ext.lower() in ('.gz', '.fz'):
        stem, ext2 = os.path.splitext(stem)
        ext = ext2 + ext
    for find, repl, kind in _SIDECAR_RULES:
        if find is None:
            cand = stem + repl + ext
        elif find in stem:
            cand = stem.replace(find, repl) + ext
        else:
            continue
        full = os.path.join(directory, cand)
        if full != os.path.abspath(path) and os.path.exists(full):
            out.append((full, kind))
    # De-duplicate, keeping order.
    seen, uniq = set(), []
    for p, k in out:
        if p not in seen:
            seen.add(p)
            uniq.append((p, k))
    return uniq


def _first_matching_ext(path, sci_shape):
    """First data extension of `path` whose shape matches `sci_shape`."""
    for c in candidate_extensions(path, sci_shape):
        if c['match']:
            return c['ext'], c['kind']
    return None, None


def detect(path, sci_shape, sci_ext=0):
    """Find measurement errors for the cube at `path` WITHOUT loading them.

    Returns a spec dict {'mode','path','ext','kind','label'} or None if nothing
    was found. `mode` is 'ext' (same file) or 'file' (sidecar). Never raises.
    """
    if fits is None or not path:
        return None
    try:
        # 1) an extension of the science file itself
        for c in candidate_extensions(path, sci_shape, sci_ext):
            if c['is_sci'] or not c['match'] or c['kind'] is None:
                continue
            return {'mode': 'ext', 'path': path, 'ext': c['ext'], 'kind': c['kind'],
                    'label': f"{c['name']} ext of {os.path.basename(path)}"}

        # 2) a sidecar file next to it
        for cand_path, kind in sidecar_candidates(path):
            ext, name_kind = _first_matching_ext(cand_path, sci_shape)
            if ext is None:
                continue
            return {'mode': 'file', 'path': cand_path, 'ext': ext,
                    'kind': name_kind or kind,
                    'label': os.path.basename(cand_path)}
    except Exception as e:
        print(f'Measurement-error detection failed: {type(e).__name__}: {e}')
    return None


def load_sigma(spec, sci_shape=None):
    """Load the σ cube described by `spec` (from `detect()` or the dialog).

    Returns (sigma_array_float32, info_dict). Raises ValueError with a readable
    message on shape mismatch or unreadable data — callers show it to the user
    and fall back to the empirical estimator.
    """
    if fits is None:
        raise ValueError('astropy is unavailable; cannot read an error cube.')
    path, ext = spec.get('path'), int(spec.get('ext', 0))
    kind = spec.get('kind') or KIND_SIGMA
    if not path or not os.path.exists(path):
        raise ValueError(f'Error-cube file not found:\n{path}')
    with fits.open(path, memmap=False) as hdul:
        if ext >= len(hdul):
            raise ValueError(f'Extension {ext} does not exist in {os.path.basename(path)}.')
        data = hdul[ext].data
        name = hdul[ext].name or f'EXT{ext}'
    if data is None:
        raise ValueError(f'Extension {name} of {os.path.basename(path)} holds no data.')
    data = np.asarray(data)
    while data.ndim > 3 and data.shape[0] == 1:       # drop Stokes axis
        data = data[0]
    if sci_shape is not None and _squeeze_shape(data.shape) != _squeeze_shape(sci_shape):
        raise ValueError(
            f'Shape mismatch: {name} is {tuple(data.shape)} but the cube is '
            f'{tuple(int(v) for v in sci_shape)}.')
    sigma = to_sigma(data, kind)
    info = dict(spec)
    info['source'] = f'{name}[{KIND_LABELS.get(kind, kind)}] ← {os.path.basename(path)}'
    info['label'] = info.get('label') or info['source']
    finite = np.isfinite(sigma)
    info['good_fraction'] = float(finite.mean()) if sigma.size else 0.0
    if not finite.any():
        raise ValueError(f'{name} contains no usable (positive, finite) values.')
    return sigma, info


# ── empirical estimator ──────────────────────────────────────────────────────
# DER_SNR (Stoehr et al. 2008, ST-ECF Newsletter 45): σ ≈ 1.482602/√6 ·
# median(|2f_i − f_{i−2} − f_{i+2}|). The second difference removes any smooth
# continuum, and the median makes it robust to the minority of pixels that carry
# emission lines; the 1.482602 is the usual MAD→σ conversion for Gaussian noise.
_DER_SNR_K = 1.482602 / np.sqrt(6.0)


def der_snr_sigma(flux, valid=None, min_pix=8):
    """Robust 1σ noise of a 1D flux array. `valid` masks pixels to *use* in the
    median (line pixels excluded) while the differences still run over the full
    contiguous array. Returns NaN if there is too little to measure."""
    f = np.asarray(flux, dtype=float)
    n = f.size
    if n < max(min_pix, 5):
        return np.nan
    d = np.abs(2.0 * f[2:n - 2] - f[0:n - 4] - f[4:n])
    keep = np.isfinite(d)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        keep &= v[2:n - 2]
    if keep.sum() < min_pix:                      # not enough line-free pixels
        keep = np.isfinite(d)
        if keep.sum() < 5:
            return np.nan
    sigma = _DER_SNR_K * float(np.median(d[keep]))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def empirical_sigma(flux, wavelengths, windows, line_centers=(), line_halfwidths=(),
                    min_pix=8):
    """Per-pixel 1σ measured from the line-free continuum inside each fit window.

    windows          : [(x_start, x_end), …] — the continuum regions being fitted.
    line_centers /   : line centroids and the half-widths to exclude around them
    line_halfwidths    (typically 3σ of the initial guess).

    Returns (sigma, ok): `sigma` is piecewise-constant within each window and NaN
    outside every window (those pixels take no part in a weighted fit anyway);
    `ok` is True if at least one window yielded a usable estimate.
    """
    lam = np.asarray(wavelengths, dtype=float)
    f = np.asarray(flux, dtype=float)
    sigma = np.full(lam.shape, np.nan, dtype=float)

    line_free = np.ones(lam.shape, dtype=bool)
    for cen, half in zip(np.atleast_1d(line_centers), np.atleast_1d(line_halfwidths)):
        if np.isfinite(cen) and np.isfinite(half) and half > 0:
            line_free &= np.abs(lam - float(cen)) > float(half)

    ok = False
    for (x0, x1) in windows:
        if not (np.isfinite(x0) and np.isfinite(x1)):
            continue
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        in_win = (lam >= lo) & (lam <= hi)
        if in_win.sum() < max(min_pix, 5):
            continue
        idx = np.where(in_win)[0]
        sl = slice(idx[0], idx[-1] + 1)            # contiguous run for the differences
        s = der_snr_sigma(f[sl], valid=line_free[sl], min_pix=min_pix)
        if np.isfinite(s):
            sigma[in_win] = s
            ok = True
    return sigma, ok
