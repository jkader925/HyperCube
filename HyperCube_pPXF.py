#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperCube_pPXF.py
=================
Stellar LOSVD fitting for HyperCube using pPXF (Cappellari 2017) and the bundled
stellar template libraries (Indo-US stars, eMILES SSP models).

This module is pure (no Qt). HyperCube.py treats a stellar fit as a new continuum
type (cont_type='stellar'): pPXF determines the stellar V, sigma (and h3/h4), and a
best-fit "optimal template" (the weighted sum of templates). The optimal template is
cached so the baseline can be re-rendered instantly when the user edits V/sigma/scale
(no pPXF re-run), and so emission-line Gaussians can be fit on top of the fixed
stellar baseline.

Public API
----------
list_libraries()                       -> dict of library metadata for the UI
TemplateLibrary(name).load()           -> read all template spectra
   .prepare(velscale, R, lam_range)    -> log-rebin + resolution-match templates
galaxy_velscale(wavelengths, z, range) -> velscale for a cube/fit-range
fit_stellar(...)                       -> {V, sigma, h3, h4, scale, chi2, cache, bestfit}
eval_stellar_baseline(cache, V, sigma, h3, h4, scale, x_vals) -> baseline on x_vals
"""

import os
import glob
import numpy as np
from astropy.io import fits

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

C_KMS = 299792.458  # speed of light, km/s

# ── Bundled libraries ────────────────────────────────────────────────────────
# fwhm = template instrumental FWHM in Angstrom (constant approximation).
#   Indo-US (Valdes et al. 2004): ~1.35 A.
#   eMILES optical (MILES range): ~2.51 A.
_HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARIES = {
    'indo_us_library': {
        'folder': 'indo_us_library',
        'fwhm': 1.35,
        'label': 'Indo-US Coudé Feed stars',
        'note': 'Empirical stellar spectra — ideal for pure kinematics.',
    },
    'eMILES': {
        'folder': 'eMILES',
        'fwhm': 2.51,
        'label': 'eMILES SSP models',
        'note': 'Single stellar population models — wide λ coverage.',
    },
}


def list_libraries():
    """Return {name: {label, folder, fwhm, note, lam_min, lam_max, n_files}} for
    every bundled library that exists on disk (for the Stellar Templates window)."""
    out = {}
    for name, meta in LIBRARIES.items():
        folder = os.path.join(_HERE, meta['folder'])
        files = sorted(glob.glob(os.path.join(folder, '*.fits')))
        if not files:
            continue
        lam_min = lam_max = None
        try:
            lam, _ = _read_spectrum(files[0])
            lam_min, lam_max = float(lam[0]), float(lam[-1])
        except Exception:
            pass
        out[name] = dict(meta, lam_min=lam_min, lam_max=lam_max, n_files=len(files))
    return out


def _read_spectrum(path):
    """Read a 1D FITS spectrum and build its linear wavelength axis from
    CRVAL1/CDELT1(or CD1_1)/CRPIX1."""
    with fits.open(path, memmap=False) as h:
        flux = np.asarray(h[0].data, dtype=float).ravel()
        hdr = h[0].header
        n = flux.size
        crval = float(hdr.get('CRVAL1', 0.0))
        crpix = float(hdr.get('CRPIX1', 1.0))
        cdelt = float(hdr.get('CDELT1', hdr.get('CD1_1', 1.0)))
        lam = crval + (np.arange(n) - (crpix - 1.0)) * cdelt
    return lam, flux


def _mad_noise(y):
    """Robust noise estimate (MAD) for a 1D array; never zero."""
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 1.0
    nz = 1.4826 * np.median(np.abs(y - np.median(y)))
    return float(nz) if nz > 0 else float(np.std(y) or 1.0)


class TemplateLibrary:
    """Loads a bundled template folder and prepares it for pPXF (log-rebinned to a
    given velscale and convolved to the galaxy's instrumental resolution)."""

    def __init__(self, name):
        if name not in LIBRARIES:
            raise ValueError(f"Unknown template library: {name}")
        self.name = name
        self.meta = LIBRARIES[name]
        self.fwhm = float(self.meta['fwhm'])
        self.folder = os.path.join(_HERE, self.meta['folder'])
        self.files = sorted(glob.glob(os.path.join(self.folder, '*.fits')))
        self._raw = None           # list of (lam, flux)
        self.templates = None      # [n_pix, n_tem] prepared
        self.ln_lam_temp = None    # ln(lambda) grid of prepared templates
        self._prep_key = None      # (velscale, R, lam_range) the prep was built for

    def load(self):
        if not self.files:
            raise FileNotFoundError(f"No FITS templates in {self.folder}")
        if self._raw is None:
            self._raw = [_read_spectrum(p) for p in self.files]
        return self

    def coverage(self):
        self.load()
        lam0 = self._raw[0][0]
        return float(lam0[0]), float(lam0[-1])

    def prepare(self, velscale, R, lam_range):
        """Log-rebin every template to `velscale` and convolve to the galaxy
        resolution FWHM_gal(λ)=λ/R. lam_range = (rest-frame λ1, λ2) used only to
        trim templates to a sensible span (with velocity slack)."""
        self.load()
        key = (round(float(velscale), 4), round(float(R), 2),
               round(float(lam_range[0]), 2), round(float(lam_range[1]), 2))
        if self._prep_key == key and self.templates is not None:
            return self
        # Trim templates to the galaxy range padded by ±3% in wavelength
        # (~9000 km/s) so they comfortably cover pPXF's velocity bounds.
        vfrac = 0.03
        lo, hi = lam_range[0] * (1 - vfrac), lam_range[1] * (1 + vfrac)
        cols, ln_lam_temp = [], None
        for lam, flux in self._raw:
            sel = (lam >= lo) & (lam <= hi)
            if sel.sum() < 50:
                sel = np.ones(lam.size, bool)  # template doesn't overlap; keep full
            lam_s, flux_s = lam[sel], np.nan_to_num(flux[sel])
            dlam = lam_s[1] - lam_s[0]
            # Convolve template up to the (coarser) galaxy resolution.
            fwhm_gal = lam_s / float(R)                     # Å, per pixel
            fwhm_dif = np.sqrt(np.clip(fwhm_gal**2 - self.fwhm**2, 0.0, None))
            sig_pix = fwhm_dif / 2.355 / dlam
            if np.any(sig_pix > 0):
                flux_s = util.gaussian_filter1d(flux_s, sig_pix)
            tnew, ln_lam_temp, _ = util.log_rebin(
                [lam_s[0], lam_s[-1]], flux_s, velscale=velscale)
            med = np.median(tnew[tnew > 0]) if np.any(tnew > 0) else 1.0
            cols.append(tnew / med)
        n = min(c.size for c in cols)
        self.templates = np.column_stack([c[:n] for c in cols])
        self.ln_lam_temp = ln_lam_temp[:n]
        self._prep_key = key
        return self


def galaxy_velscale(wavelengths, z, fit_range):
    """velscale (km/s/pixel) for the de-redshifted cube wavelengths over fit_range
    (fit_range given in OBSERVED wavelength units, like the cube)."""
    lam = np.asarray(wavelengths, float) / (1.0 + z)
    r0, r1 = fit_range[0] / (1.0 + z), fit_range[1] / (1.0 + z)
    sel = (lam >= r0) & (lam <= r1)
    lam_f = lam[sel]
    _, _, velscale = util.log_rebin([lam_f[0], lam_f[-1]], np.ones(lam_f.size))
    return float(velscale), (float(r0), float(r1))


def _losvd_convolve(spec, velscale, V, sigma, h3=0.0, h4=0.0):
    """Convolve a spectrum (log-sampled at `velscale` km/s/pix) by a Gauss-Hermite
    LOSVD with velocity V (km/s, +ve → redshift) and dispersion sigma (km/s)."""
    sigma = max(float(sigma), velscale / 5.0)
    sig_pix = sigma / velscale
    v_pix = V / velscale
    half = int(np.ceil(5 * sig_pix + abs(v_pix))) + 1
    x = np.arange(-half, half + 1)
    y = (x - v_pix) / sig_pix
    k = np.exp(-0.5 * y**2)
    if h3 or h4:
        H3 = (2 * y**3 - 3 * y) / np.sqrt(3)
        H4 = (4 * y**4 - 12 * y**2 + 3) / np.sqrt(24)
        k = k * (1.0 + h3 * H3 + h4 * H4)
    s = k.sum()
    if s != 0:
        k = k / s
    return np.convolve(spec, k, mode='same')


def eval_stellar_baseline(cache, V, sigma, h3, h4, scale, x_vals):
    """Stellar baseline (observed-frame) on x_vals from a cached optimal template.

    cache holds: temp_opt (rest-frame log-sampled weighted template), ln_lam_temp,
    mpoly_temp (multiplicative continuum on the template grid), velscale, z.
    """
    temp_opt = cache['temp_opt']
    ln_lam_temp = cache['ln_lam_temp']
    mpoly_temp = cache.get('mpoly_temp')
    velscale = cache['velscale']
    z = cache['z']
    broad = _losvd_convolve(temp_opt, velscale, V, sigma, h3, h4)
    if mpoly_temp is not None:
        broad = broad * mpoly_temp
    lam_obs = np.exp(ln_lam_temp) * (1.0 + z)
    y = np.interp(np.asarray(x_vals, float), lam_obs, broad,
                  left=np.nan, right=np.nan)
    return y * float(scale)


def _goodpixels(ln_lam_gal, ln_lam_temp, mask_centroids, z, mask_dv=500.0):
    """Indices to fit: pPXF's standard line mask within template coverage, minus
    velocity windows (±mask_dv km/s) around the user's emission-line centroids
    (given in OBSERVED wavelength)."""
    lam_range_temp = np.exp([ln_lam_temp[0], ln_lam_temp[-1]])
    good = util.determine_goodpixels(ln_lam_gal, lam_range_temp, redshift=0)
    keep = np.zeros(ln_lam_gal.size, bool)
    keep[good] = True
    lam_gal = np.exp(ln_lam_gal)
    for cen_obs in (mask_centroids if mask_centroids is not None else []):
        try:
            cen_rest = float(cen_obs) / (1.0 + z)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(cen_rest) or cen_rest <= 0:
            continue
        dv = C_KMS * (lam_gal - cen_rest) / cen_rest
        keep &= ~(np.abs(dv) < mask_dv)
    return np.where(keep)[0]


def fit_stellar(flux, wavelengths, z, R, library, *, fit_range,
                mask_centroids=(), moments=2, degree=-1, mdegree=10,
                sigma_guess=150.0, velscale=None, mask_dv=500.0):
    """Run pPXF on one spaxel.

    flux         : 1D galaxy spectrum (cube units), same length as wavelengths
    wavelengths  : observed-frame wavelength array (Å)
    z, R         : galaxy redshift and resolving power
    library      : a prepared TemplateLibrary (call .prepare(velscale,...) first)
    fit_range    : (λ1, λ2) in OBSERVED wavelength
    mask_centroids : observed-frame emission-line centroids to mask
    mask_dv      : half-width (km/s) of the window masked around each centroid
    Returns a dict: V, sigma, h3, h4, scale, chi2, success, cache, bestfit.
    """
    flux = np.nan_to_num(np.asarray(flux, float))
    lam = np.asarray(wavelengths, float) / (1.0 + z)        # rest frame
    r0, r1 = fit_range[0] / (1.0 + z), fit_range[1] / (1.0 + z)
    sel = (lam >= r0) & (lam <= r1) & np.isfinite(flux)
    if sel.sum() < 50:
        raise ValueError("Fit range overlaps too few pixels.")
    lam_f, flux_f = lam[sel], flux[sel]

    galaxy, ln_lam_gal, velscale_g = util.log_rebin(
        [lam_f[0], lam_f[-1]], flux_f, velscale=velscale)
    velscale = float(velscale_g if velscale is None else velscale)
    norm = np.median(galaxy[galaxy > 0]) if np.any(galaxy > 0) else 1.0
    galaxy = galaxy / norm
    noise = np.full_like(galaxy, _mad_noise(galaxy))

    templates = library.templates
    ln_lam_temp = library.ln_lam_temp
    good = _goodpixels(ln_lam_gal, ln_lam_temp, mask_centroids, z, mask_dv=mask_dv)

    pp = ppxf(templates, galaxy, noise, velscale, [0.0, float(sigma_guess)],
              goodpixels=good, moments=moments, degree=degree, mdegree=mdegree,
              lam=np.exp(ln_lam_gal), lam_temp=np.exp(ln_lam_temp), quiet=True)

    sol = list(np.atleast_1d(pp.sol)) + [0.0, 0.0, 0.0, 0.0]
    V, sigma = float(sol[0]), float(sol[1])
    h3 = float(sol[2]) if moments >= 3 else 0.0
    h4 = float(sol[3]) if moments >= 4 else 0.0

    temp_opt = templates @ pp.weights
    mpoly = pp.mpoly if (mdegree > 0 and getattr(pp, 'mpoly', None) is not None) \
        else np.ones_like(galaxy)
    mpoly_temp = np.interp(ln_lam_temp, ln_lam_gal, mpoly,
                           left=mpoly[0], right=mpoly[-1])
    # Store this spaxel's own kinematics in the cache so the baseline can be
    # reconstructed per-spaxel (during a cube fit, for navigation, and for maps)
    # without consulting a single shared df_cont row.
    cache = dict(temp_opt=temp_opt, ln_lam_temp=ln_lam_temp, mpoly_temp=mpoly_temp,
                 velscale=velscale, z=float(z),
                 V=V, sigma=sigma, h3=h3, h4=h4, scale=float(norm))

    bestfit = eval_stellar_baseline(cache, V, sigma, h3, h4, norm, wavelengths)
    return dict(V=V, sigma=sigma, h3=h3, h4=h4, scale=float(norm),
                chi2=float(pp.chi2), success=True, cache=cache, bestfit=bestfit,
                velscale=velscale)
