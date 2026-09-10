"""Instrument line-spread functions for HyperCube.

**Qt-free by contract** — numpy only, like `HyperCube_fit.py` and
`HyperCube_Noise.py`. This module is imported by the batch runner and by the
template layer, both of which must work without a `QApplication`.

Why this exists
---------------
A spectrograph's resolving power is not a constant. R = lambda/FWHM, and it is
the *FWHM* that a grating holds roughly fixed, so R slides across the band —
for MUSE it very nearly doubles, running 1873 at H-beta to 2978 at [S II]
across a single MAUNA model at z = 0.0876. A model that assumes one R therefore
mis-states the instrumental width differently at each line, which is exactly
the comparison an emission-line study is trying to make.

Two things need that curve:

* **Templates** (`HyperCube_Templates.py`) store *intrinsic* velocity
  dispersions so one file is portable across galaxies and instruments. Turning
  those into the observed widths a fitter wants means adding the LSF in
  quadrature at each line's own observed wavelength -- see
  `intrinsic_to_observed`.
* **pPXF** still wants a single scalar R to convolve stellar templates with.
  `resolving_power_from_header` in HyperCube.py delegates here and evaluates the
  curve at mid-coverage, which is what it has always effectively assumed.

Nothing here deconvolves fit output: `sigma_fit` remains the raw observed
Gaussian width, as it always has been.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

C_KMS = 299792.458
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))     # 2.3548


# ── The model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LSFModel:
    """An instrument's line-spread function as FWHM(lambda), in Angstrom.

    `fwhm_of` takes the observed wavelength in Angstrom and returns the LSF
    FWHM in Angstrom. Everything else is derived from it, so a new instrument
    only has to supply that one function.

    `label` is provenance and is written into template/report output, so it
    should name the parameterisation, not just the instrument.
    """

    label: str
    fwhm_of: Callable[[np.ndarray], np.ndarray] = field(repr=False)

    def fwhm_A(self, lam_obs):
        """LSF FWHM in Angstrom at the given observed wavelength(s)."""
        lam = np.asarray(lam_obs, dtype=float)
        out = np.asarray(self.fwhm_of(lam), dtype=float)
        out = np.where(np.isfinite(out) & (out > 0), out, np.nan)
        return out if out.ndim else float(out)

    def sigma_A(self, lam_obs):
        """LSF sigma in Angstrom."""
        return self.fwhm_A(lam_obs) / FWHM_PER_SIGMA

    def sigma_kms(self, lam_obs):
        """LSF sigma as a velocity dispersion, km/s."""
        lam = np.asarray(lam_obs, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = C_KMS * self.sigma_A(lam) / lam
        return out if np.ndim(out) else float(out)

    def R(self, lam_obs):
        """Resolving power lambda/FWHM at the given wavelength(s)."""
        lam = np.asarray(lam_obs, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = lam / self.fwhm_A(lam)
        return out if np.ndim(out) else float(out)


def constant_fwhm(fwhm_A_value, label):
    """An LSF of fixed FWHM — the grating-like case (KCWI, KCRM)."""
    value = float(fwhm_A_value)
    return LSFModel(label, lambda lam: np.full(np.shape(lam), value))


def constant_R(R_value, label):
    """An LSF of fixed resolving power — FWHM grows with wavelength."""
    R = float(R_value)
    return LSFModel(label, lambda lam: np.asarray(lam, dtype=float) / R)


def polynomial_fwhm(coeffs, label):
    """FWHM(lambda) as a polynomial, `coeffs` highest-order first (np.polyval)."""
    c = tuple(float(x) for x in coeffs)
    return LSFModel(label, lambda lam: np.polyval(c, np.asarray(lam, dtype=float)))


def tabulated_R(lams, Rs, label):
    """R sampled at wavelengths, linearly interpolated (and held at the ends)."""
    lam_t = np.asarray(lams, dtype=float)
    R_t = np.asarray(Rs, dtype=float)
    order = np.argsort(lam_t)
    lam_t, R_t = lam_t[order], R_t[order]

    def _fwhm(lam):
        lam = np.asarray(lam, dtype=float)
        return lam / np.interp(lam, lam_t, R_t)

    return LSFModel(label, _fwhm)


# ── Intrinsic <-> observed ───────────────────────────────────────────────────

def intrinsic_to_observed(sigma_int_kms, lam_obs, lsf):
    """Broaden an intrinsic velocity dispersion by the LSF at `lam_obs`.

    sigma_obs = sqrt(sigma_int^2 + sigma_lsf^2), in km/s.

    This is the direction templates use: a template states the physical width
    it means, and expansion turns it into the width the spectrograph would
    record at whatever wavelength the line lands for this galaxy. Applied to a
    lower bound it does the right thing on its own -- a 20 km/s intrinsic floor
    becomes ~71 km/s observed at H-beta in MUSE, correctly forbidding a fit
    narrower than the instrument can resolve.
    """
    s_int = np.asarray(sigma_int_kms, dtype=float)
    s_lsf = np.asarray(lsf.sigma_kms(lam_obs), dtype=float)
    out = np.sqrt(np.clip(s_int, 0.0, None) ** 2 + s_lsf ** 2)
    # An infinite intrinsic bound stays infinite rather than becoming NaN.
    out = np.where(np.isinf(s_int), s_int, out)
    return out if np.ndim(out) else float(out)


def observed_to_intrinsic(sigma_obs_kms, lam_obs, lsf):
    """Remove the LSF from an observed width. NaN where the line is unresolved.

    The inverse of `intrinsic_to_observed`. A line at or below the instrumental
    width carries no measurable intrinsic dispersion, and this returns NaN
    rather than 0 to keep "unresolved" distinguishable from "resolved and
    narrow" -- the two mean different things and must not be averaged together.
    """
    s_obs = np.asarray(sigma_obs_kms, dtype=float)
    s_lsf = np.asarray(lsf.sigma_kms(lam_obs), dtype=float)
    diff = s_obs ** 2 - s_lsf ** 2
    with np.errstate(invalid='ignore'):
        out = np.where(diff > 0, np.sqrt(np.abs(diff)), np.nan)
    out = np.where(np.isinf(s_obs), s_obs, out)
    return out if np.ndim(out) else float(out)


# ── VLT/MUSE ─────────────────────────────────────────────────────────────────
#
# Two parameterisations, and they disagree at the blue end by ~9%. This is not
# a detail to paper over: the MAUNA models sit right there (H-beta lands at
# 4959-5590 A across the sample's redshift range).
#
#   'bacon2017'   FWHM(lam) = 5.866e-8 lam^2 - 9.187e-4 lam + 6.040   [Angstrom]
#                 The polynomial fitted to sky lines in the MUSE deep fields.
#                 An empirical curve, but sky lines are sparse blueward of
#                 ~5000 A, so it is least constrained exactly where MAUNA needs
#                 it. Gives R = 1610 at 4800 A, 2819 at 7000 A, 3620 at 9300 A.
#
#   'eso_nominal' R interpolated linearly in lambda between the two endpoints
#                 ESO quotes for MUSE, R = 1770 at 4800 A and R = 3590 at
#                 9300 A. A design specification, not a measurement, and the
#                 linear interpolation between the endpoints is this module's
#                 choice, not ESO's -- do not read intermediate values as
#                 authoritative.
#
# They agree to ~1% in the red and differ by ~9% at 4800 A. The default is
# 'bacon2017' because it is measured rather than specified, but the choice is
# exposed (`muse_model=`) and recorded in the provenance label so a run can say
# which curve it used. Confirm against the ESO LSF reference before relying on
# blue-end widths.

MUSE_LSF_MODELS = {
    'bacon2017': lambda: polynomial_fwhm(
        (5.866e-8, -9.187e-4, 6.040),
        'MUSE LSF polynomial (Bacon+2017 sky-line fit)'),
    'eso_nominal': lambda: tabulated_R(
        (4800.0, 9300.0), (1770.0, 3590.0),
        'MUSE nominal R (ESO endpoints, linearly interpolated)'),
}
MUSE_LSF_DEFAULT = 'bacon2017'


def muse_lsf(model=MUSE_LSF_DEFAULT):
    """The MUSE LSF under the named parameterisation (see MUSE_LSF_MODELS)."""
    key = str(model).strip().lower()
    if key not in MUSE_LSF_MODELS:
        raise ValueError(f'unknown MUSE LSF model {model!r}; '
                         f'expected one of {sorted(MUSE_LSF_MODELS)}')
    return MUSE_LSF_MODELS[key]()


# ── Keck/KCWI + KCRM ─────────────────────────────────────────────────────────
#
# The resolution *element* (FWHM in Angstrom) per grating at the Large slicer,
# rather than R itself. A grating's FWHM is fixed while R is not, and this is
# what the MAUNA coadds show: RH1 and RH2 carry SPECRES 1800 @ 6150 A and
# 2025 @ 6900 A, which is one FWHM (3.42 vs 3.41 A) seen at two wavelengths,
# not two different resolving powers.
#
# ANCHORED marks entries calibrated against SPECRES in this survey's own cubes
# (Mrk 273, III Zw 035); the rest are nominal published performance and are
# labelled as unconfirmed wherever they are used.

KCWI_FWHM_LARGE = {
    'BH1': 1.15, 'BH2': 1.15, 'BH3': 1.15,   # ANCHORED: R=4500 @ 5215 Å
    'RH1': 3.42,                             # ANCHORED: R=1800 @ 6150 Å
    'RH2': 3.41,                             # ANCHORED: R=2025 @ 6900 Å
    'BL': 5.00, 'BM': 2.25,                  # nominal
    'RL': 7.00, 'RM1': 6.80, 'RM2': 6.80,    # nominal
    'RH3': 3.42, 'RH4': 3.42,                # nominal (same RH family)
}
KCWI_ANCHORED = {'BH1', 'BH2', 'BH3', 'RH1', 'RH2'}
# The slice is the entrance aperture, so a narrower slicer resolves
# proportionally better: Large 1.35", Medium 0.69", Small 0.35".
KCWI_SLICER_GAIN = {'LARGE': 1.0, 'MEDIUM': 2.0, 'SMALL': 4.0}
KCWI_SLICER_LETTER = {'L': 'LARGE', 'M': 'MEDIUM', 'S': 'SMALL'}


def kcwi_grating_fwhm(token):
    """Resolution element for a KCWI grating token, or None.

    Accepts the bare grating ('RH2') and the grating+slicer spelling used in
    coadd provenance ('BLL' = BL grating, Large slicer). Returns
    `(canonical_name, fwhm_large, slicer_letter_or_None)`.
    """
    name = str(token).strip().upper()
    if not name:
        return None
    if name in KCWI_FWHM_LARGE:
        return name, KCWI_FWHM_LARGE[name], None
    if name[-1] in 'LMS':
        stem, slicer = name[:-1], name[-1]
        if stem in KCWI_FWHM_LARGE:
            return stem, KCWI_FWHM_LARGE[stem], slicer
        # 'RM' stands for the RM1/RM2 pair, which share a resolution element.
        family = [g for g in KCWI_FWHM_LARGE if g.startswith(stem)]
        if family:
            return stem, KCWI_FWHM_LARGE[sorted(family)[0]], slicer
    return None


def kcwi_lsf(grating, slicer='LARGE'):
    """The KCWI/KCRM LSF for a grating + slicer, or None if unrecognised."""
    hit = kcwi_grating_fwhm(grating)
    if hit is None:
        return None
    name, fwhm_large, _ = hit
    gain = KCWI_SLICER_GAIN.get(str(slicer).strip().upper(), 1.0)
    label = f'KCWI {name} grating, {str(slicer).title()} slicer'
    if name not in KCWI_ANCHORED:
        label += ' (nominal, unconfirmed for this grating)'
    return constant_fwhm(fwhm_large / gain, label)


# ── JWST ─────────────────────────────────────────────────────────────────────

# MIRI MRS, band-averaged (Jones et al. 2023). R varies within each band;
# these are the mid-band values.
MIRI_MRS_R = {
    ('1', 'SHORT'): 3515, ('1', 'MEDIUM'): 3470, ('1', 'LONG'): 3355,
    ('2', 'SHORT'): 3050, ('2', 'MEDIUM'): 2960, ('2', 'LONG'): 3080,
    ('3', 'SHORT'): 2705, ('3', 'MEDIUM'): 2215, ('3', 'LONG'): 2385,
    ('4', 'SHORT'): 1695, ('4', 'MEDIUM'): 1725, ('4', 'LONG'): 1480,
}

# NIRSpec IFU, by disperser class.
NIRSPEC_R = {'PRISM': 100, 'G140M': 1000, 'G235M': 1000, 'G395M': 1000,
             'G140H': 2700, 'G235H': 2700, 'G395H': 2700}
