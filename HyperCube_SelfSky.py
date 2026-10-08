"""Self-sky subtraction: build a sky spectrum from a cube's own object-free
spaxels and subtract it.

Why this exists
---------------
Reductions leave additive sky residuals. On the driver cube for this module,
``F01364-1042_2023-09-17_red_RH2@6950_coadd.fits``, an under-subtracted sky line
sits at 6866.0 A -- 2.1 A (~92 km/s) redward of [N II] 6548 at z=0.048232
(6863.88 A) -- and is **2.5x brighter than the real [N II] 6548** it blends
with. A fitter with a free centroid locks onto the residual instead of the line,
in essentially every spaxel.

**This is a workaround, not the fix.** The root cause for KCWI/KCRM is upstream:
``SubtractSky`` scales the sky master by exposure time only and never fits the
airglow amplitude, so the sky is wrong whenever it changed between the science
and sky exposures. The KCWI project has ``measure_sky_scale.py --curve``, which
measures per-wavelength k(lambda) and corrects it properly at reduction time.
Use this module for cubes that will not be re-reduced; do not let it become the
reason the reduction is never fixed.

Scope: **additive residuals only.** Measured on the driver cube, the excess over
local sidebands at 6866 A is flat at ~0.008 while the continuum beneath it
varies by a factor of 50 (p0-20: 0.00755 at continuum -0.00003; p80-95: 0.00886
at continuum 0.00142). That is an additive pedestal. A multiplicative artifact
-- scattered light, a flat-field error -- tracks the continuum instead and must
not be attacked with this tool.

Two design decisions that must survive refactoring
--------------------------------------------------
1. **The statistic is a median, never a mean.** Measured contamination of the
   sky spectrum at H-alpha when the pool includes line-emitting spaxels:
   mean 0.63% of the galaxy line peak, median 0.37%, 40th percentile 0.29%. More
   tellingly, injecting contamination up to 60% of the pool moves the median by
   0.2%. The median is what makes an imperfect mask harmless; a mean has no such
   protection. ``_STATISTICS`` therefore contains no mean, and callers pass a
   name rather than a function so one cannot be smuggled in.

2. **The masks still matter.** Continuum-faint does not mean line-free: 12.7% of
   the "faintest 60% by white light" pool on the driver cube has H-alpha+[N II]
   at SNR > 3 (6.8% at SNR > 5). The median absorbs that here, but it will not
   when an outflow fills much of the field, when the FoV is small, or in
   per-column mode where each block's pool is ~1/6 the size and a local outflow
   can be the majority of it. Hence ``combine_masks``: the physically correct
   criterion is "faint in continuum AND faint in every line I care about", and a
   veto built on one line does not remove an outflow visible only in another.

This module is Qt-free (numpy only) so ``hypercube_batch.py`` and the process
pool can import it. All dialogs live in HyperCube.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "SkyResult",
    "VetoWindowOverlap",
    "channel_map",
    "check_veto_window",
    "build_sky",
    "apply_sky",
    "propagate_sky_variance",
    "faintest_fraction_mask",
    "threshold_mask",
    "combine_masks",
    "white_light",
    "STATISTIC_NAMES",
]

# Name -> percentile. A mean is deliberately absent; see the module docstring.
# Callers pass a name, not a callable, so no mean can be introduced by a caller.
_STATISTICS: dict[str, float] = {
    "median": 50.0,
    "p40": 40.0,
    "p25": 25.0,
}
STATISTIC_NAMES: tuple[str, ...] = tuple(_STATISTICS)

DEFAULT_MIN_SPAXELS = 50
DEFAULT_MIN_PER_COLUMN = 20
DEFAULT_COL_BLOCKS = 6


class VetoWindowOverlap(ValueError):
    """A veto window overlaps the wavelength region being corrected.

    Channel-map and fit-map vetoes select on the same data the sky is being
    measured from. For galaxy and outflow lines that is exactly right. For the
    residual itself it is circular: a channel map at 6866 A *is* a map of the
    sky residual, so "reject spaxels that are bright at 6866 A" rejects the
    spaxels carrying the sky, and the surviving pool is biased low -- the
    correction then under-subtracts, and nothing in the output says so.

    This is refused rather than warned about, because the failure is invisible
    in the result.
    """


class SkyPoolTooSmall(ValueError):
    """The sky region has too few spaxels to estimate a sky spectrum.

    Raised rather than returning a noisy sky: a sky built from a handful of
    spaxels is mostly noise, and subtracting it injects that noise into every
    spaxel of the cube -- a correction that is worse than no correction, and
    invisible once applied.
    """


@dataclass(frozen=True)
class SkyResult:
    """A sky estimate and everything needed to audit or undo it."""

    sky: np.ndarray                     # (nw,) global, or (nw, nblocks)
    n_used: np.ndarray                  # contributing spaxels, per block
    sky_var: np.ndarray                 # variance OF THE ESTIMATE, same shape as sky
    mode: str                           # 'global' | 'per-column'
    statistic: str
    mask_sources: tuple[str, ...] = ()
    n_mask: int = 0
    fallback_cols: tuple[int, ...] = ()
    col_edges: tuple[int, ...] = ()     # block boundaries in x, len == nblocks+1
    wav_guard: tuple[float, float] | None = None

    def sky_cube_view(self, shape: tuple[int, int, int]) -> np.ndarray:
        """Broadcast the sky to (nw, ny, nx) for subtraction."""
        nw, ny, nx = shape
        if self.mode == "global":
            return self.sky[:, None, None]
        out = np.zeros((nw, 1, nx), dtype=float)
        for b in range(len(self.col_edges) - 1):
            lo, hi = self.col_edges[b], self.col_edges[b + 1]
            out[:, 0, lo:hi] = self.sky[:, b][:, None]
        return out

    def describe(self) -> str:
        bits = [f"{self.statistic} sky, {self.mode}, {self.n_mask} spaxels"]
        if self.mask_sources:
            bits.append("mask: " + " AND ".join(self.mask_sources))
        if self.fallback_cols:
            bits.append(f"{len(self.fallback_cols)} column(s) fell back to global")
        return "; ".join(bits)


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def white_light(cube: np.ndarray, step: int = 1) -> np.ndarray:
    """Median-collapsed continuum image, (ny, nx).

    A median over wavelength, not a mean, so that the emission lines and the
    very sky residuals this module targets do not define the continuum level
    they are being compared against.
    """
    data = np.nan_to_num(np.asarray(cube)[::max(int(step), 1)])
    return np.median(data, axis=0)


def _live(cube: np.ndarray) -> np.ndarray:
    """Spaxels with any data at all.

    Off-detector voxels in these coadds are EXACT zeros rather than NaN, so a
    finite-only test keeps them and they would then dominate any "faintest N%"
    selection -- the sky would be built from empty sky.
    """
    data = np.nan_to_num(np.asarray(cube))
    return np.any(data != 0.0, axis=0)


def faintest_fraction_mask(wl_image: np.ndarray, frac: float,
                           live: np.ndarray | None = None) -> np.ndarray:
    """Keep the faintest ``frac`` of live spaxels by continuum brightness."""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    wl = np.asarray(wl_image, dtype=float)
    dom = np.isfinite(wl) if live is None else (live & np.isfinite(wl))
    if not dom.any():
        return np.zeros(wl.shape, dtype=bool)
    cut = np.percentile(wl[dom], 100.0 * frac)
    return dom & (wl <= cut)


def threshold_mask(field: np.ndarray, op: str, value: float,
                   live: np.ndarray | None = None) -> np.ndarray:
    """Keep spaxels satisfying ``field <op> value``.

    Operator tokens match the existing Mask Spaxels dialog ('>', '<', 'abs>',
    'abs<') so the two features cannot drift apart in the user's mental model.
    """
    arr = np.asarray(field, dtype=float)
    dom = np.isfinite(arr) if live is None else (live & np.isfinite(arr))
    if op == ">":
        keep = arr > value
    elif op == "<":
        keep = arr < value
    elif op == "abs>":
        keep = np.abs(arr) > value
    elif op == "abs<":
        keep = np.abs(arr) < value
    else:
        raise ValueError(f"unknown operator {op!r}; expected >, <, abs>, abs<")
    return dom & keep


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Logical AND of every veto.

    AND, not OR, and with no source-specific special cases: the criterion is
    "object-free by EVERY test the user supplied". A veto built on one line
    cannot remove an outflow that is only visible in another, which is why
    several are meant to be stacked.
    """
    supplied = [np.asarray(m, dtype=bool) for m in masks if m is not None]
    if not supplied:
        raise ValueError("combine_masks needs at least one mask")
    shapes = {m.shape for m in supplied}
    if len(shapes) != 1:
        raise ValueError(f"masks have differing shapes: {sorted(shapes)}")
    out = supplied[0].copy()
    for m in supplied[1:]:
        out &= m
    return out


def channel_map(cube: np.ndarray,
                wavelengths: np.ndarray,
                w0: float,
                w1: float,
                sidebands: Sequence[tuple[float, float]] = ()) -> np.ndarray:
    """Continuum-subtracted line map over [w0, w1], as (ny, nx).

    Mirrors the GUI's C-key channel map with locked X/V sidebands
    (``_compute_channel_map_with_subtraction``): the line window is summed, each
    sideband contributes a mean-per-channel estimate, and their average is
    scaled to the line window's width before subtraction. Kept numerically
    identical to that method so the map a user vetoes on is the map they were
    just looking at.

    With no sidebands this is a plain band sum, which carries the continuum --
    fine for a bright line, wrong for a faint outflow wing on a bright galaxy.
    """
    data = np.nan_to_num(np.asarray(cube, dtype=float))
    wav = np.asarray(wavelengths, dtype=float)
    if data.shape[0] != wav.size:
        raise ValueError(f"cube has {data.shape[0]} planes but "
                         f"{wav.size} wavelengths")
    lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
    line_sel = (wav >= lo) & (wav <= hi)
    n_line = int(line_sel.sum())
    if n_line == 0:
        raise ValueError(f"no channels in [{lo:.4g}, {hi:.4g}]")
    line_map = data[line_sel].sum(axis=0)

    estimates = []
    for s0, s1 in sidebands:
        a, b = (s0, s1) if s0 <= s1 else (s1, s0)
        sel = (wav >= a) & (wav <= b)
        if sel.any():
            estimates.append(data[sel].sum(axis=0) / int(sel.sum()))
    if estimates:
        line_map = line_map - np.mean(estimates, axis=0) * n_line
    return line_map


def check_veto_window(window: tuple[float, float],
                      protected: tuple[float, float] | None) -> None:
    """Raise VetoWindowOverlap if a veto window touches the protected region.

    ``protected`` is the wavelength range carrying the artifact being removed.
    None disables the check, which is correct only when the user has genuinely
    no artifact region in mind (e.g. vetoing on a fit map from a different
    part of the spectrum).
    """
    if protected is None:
        return
    v0, v1 = sorted(window)
    p0, p1 = sorted(protected)
    if v0 <= p1 and p0 <= v1:
        raise VetoWindowOverlap(
            f"veto window [{v0:.4g}, {v1:.4g}] A overlaps the protected region "
            f"[{p0:.4g}, {p1:.4g}] A. Vetoing on the artifact's own wavelengths "
            f"removes the spaxels that carry the sky, biasing the estimate low. "
            f"Build the veto from a galaxy or outflow line instead."
        )


# ---------------------------------------------------------------------------
# Sky construction
# ---------------------------------------------------------------------------

def _stat_and_var(block: np.ndarray, pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentile along axis 1 and the variance of that estimate.

    The variance of a median is pi/2 times that of a mean for Gaussian noise --
    the price of the robustness in the module docstring. Using the mean's
    sigma^2/N here would understate the added noise by 57% and quietly
    over-weight the corrected region in a 1/sigma-weighted fit.
    """
    n = block.shape[1]
    est = np.percentile(block, pct, axis=1)
    if n < 2:
        return est, np.zeros_like(est)
    var_of_mean = np.var(block, axis=1, ddof=1) / n
    return est, var_of_mean * (np.pi / 2.0)


def build_sky(cube: np.ndarray,
              mask: np.ndarray,
              *,
              statistic: str = "median",
              mode: str = "global",
              n_col_blocks: int = DEFAULT_COL_BLOCKS,
              min_spaxels: int = DEFAULT_MIN_SPAXELS,
              min_per_column: int = DEFAULT_MIN_PER_COLUMN,
              mask_sources: Sequence[str] = (),
              wav_guard: tuple[float, float] | None = None) -> SkyResult:
    """Estimate the sky spectrum from the spaxels selected by ``mask``.

    ``mask`` is a boolean (ny, nx): True where a spaxel may contribute. This
    function does not know how the mask was built -- that separation is what
    lets the GUI use a dialog and the batch runner use a stored spec.

    Raises SkyPoolTooSmall if fewer than ``min_spaxels`` contribute.
    """
    if statistic not in _STATISTICS:
        raise ValueError(
            f"unknown statistic {statistic!r}; expected one of "
            f"{STATISTIC_NAMES}. A mean is deliberately unavailable -- see the "
            f"module docstring."
        )
    if mode not in ("global", "per-column"):
        raise ValueError(f"unknown mode {mode!r}")

    data = np.nan_to_num(np.asarray(cube, dtype=float))
    if data.ndim != 3:
        raise ValueError(f"cube must be 3-D, got shape {data.shape}")
    nw, ny, nx = data.shape

    sel = np.asarray(mask, dtype=bool) & _live(data)
    if sel.shape != (ny, nx):
        raise ValueError(f"mask shape {sel.shape} != cube spatial shape {(ny, nx)}")
    n_mask = int(sel.sum())
    if n_mask < min_spaxels:
        raise SkyPoolTooSmall(
            f"sky region has {n_mask} spaxels, below the floor of {min_spaxels}. "
            f"A sky built from this few is mostly noise, and subtracting it "
            f"would inject that noise into every spaxel. Loosen the mask."
        )

    pct = _STATISTICS[statistic]

    if mode == "global":
        sky, var = _stat_and_var(data[:, sel], pct)
        return SkyResult(sky=sky,
                         n_used=np.array([n_mask]),
                         sky_var=var,
                         mode="global",
                         statistic=statistic,
                         mask_sources=tuple(mask_sources),
                         n_mask=n_mask,
                         col_edges=(0, nx),
                         wav_guard=wav_guard)

    # Per-column: the residual has real across-slice structure. On the driver
    # cube the amplitude runs 0.00912 at x=0-18 down to 0.00693 at x=90-108, a
    # monotonic ~30% gradient, which matches the DRP modelling sky per slice.
    nb = max(int(n_col_blocks), 1)
    edges = [int(round(i * nx / nb)) for i in range(nb + 1)]
    g_sky, g_var = _stat_and_var(data[:, sel], pct)

    sky = np.zeros((nw, nb), dtype=float)
    var = np.zeros((nw, nb), dtype=float)
    used = np.zeros(nb, dtype=int)
    fallback: list[int] = []
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        col_sel = np.zeros_like(sel)
        col_sel[:, lo:hi] = sel[:, lo:hi]
        n_col = int(col_sel.sum())
        used[b] = n_col
        if n_col < min_per_column:
            # Fall back rather than build a noisy per-column sky, and record it
            # -- a silent fallback would look like a measured column.
            sky[:, b], var[:, b] = g_sky, g_var
            fallback.append(b)
        else:
            sky[:, b], var[:, b] = _stat_and_var(data[:, col_sel], pct)

    return SkyResult(sky=sky,
                     n_used=used,
                     sky_var=var,
                     mode="per-column",
                     statistic=statistic,
                     mask_sources=tuple(mask_sources),
                     n_mask=n_mask,
                     fallback_cols=tuple(fallback),
                     col_edges=tuple(edges),
                     wav_guard=wav_guard)


def apply_sky(cube: np.ndarray, result: SkyResult) -> np.ndarray:
    """Return a new cube with the sky subtracted.

    Off-detector voxels are EXACT zeros in these coadds, not NaN, and the DRP's
    MASK extension is often all-zero, so subtracting everywhere would turn those
    zeros into -sky and hand the coadd/fit real negative data outside the
    detector footprint. The subtraction is therefore applied only where the
    spaxel has data.
    """
    data = np.asarray(cube, dtype=float)
    live = _live(data)
    sky_full = np.broadcast_to(result.sky_cube_view(data.shape), data.shape)
    return np.where(live[None, :, :], data - sky_full, data)


def propagate_sky_variance(sigma_cube: np.ndarray | None,
                           result: SkyResult,
                           shape: tuple[int, int, int] | None = None
                           ) -> np.ndarray | None:
    """Add the sky estimate's variance to the measurement-error cube.

    Required, not bookkeeping: HyperCube weights the fit by 1/sigma inside the
    fit windows, so omitting this biases the weights in exactly the wavelength
    region the correction touched, and the reported ``*_std`` stops being a
    propagated measurement error.

    Returns a new sigma cube, or None if there was none to begin with (the
    empirical DER_SNR estimator then measures the corrected data directly and
    already sees the added noise).
    """
    if sigma_cube is None:
        return None
    sig = np.asarray(sigma_cube, dtype=float)
    tgt = shape if shape is not None else sig.shape
    var_full = np.broadcast_to(
        SkyResult(sky=result.sky_var, n_used=result.n_used, sky_var=result.sky_var,
                  mode=result.mode, statistic=result.statistic,
                  col_edges=result.col_edges).sky_cube_view(tgt), sig.shape)
    return np.sqrt(np.square(sig) + var_full)


# ---------------------------------------------------------------------------
# Writing a correction into the FITS file, and undoing it
# ---------------------------------------------------------------------------

SKY_FLAG = "HCSKYSUB"          # science header: True while the file holds a self-sky subtraction
SKY_BACKUP_KEY = "HCSKYBAK"    # science header: backup file name
SKY_KEYS = (SKY_FLAG, SKY_BACKUP_KEY, "HCSKYSTA", "HCSKYMOD", "HCSKYNM", "HCSKYDAT")
ERR_FLAG = "HCSKYVAR"          # error header: sky variance folded in


class SkyFileError(RuntimeError):
    """Refusal to write or restore; the message says why and what to do."""


def sky_backup_path(cube_path: str) -> str:
    import os
    return os.path.splitext(cube_path)[0] + "_preSelfSky.fits"


def file_sky_state(cube_path: str, sci_ext: int) -> dict | None:
    """The self-sky record in a cube's science header, or None if the file is uncorrected."""
    from astropy.io import fits
    header = fits.getheader(cube_path, sci_ext)
    if not header.get(SKY_FLAG, False):
        return None
    return {k: header.get(k) for k in SKY_KEYS if k in header}


def _sigma_to_kind(sigma: np.ndarray, kind: str) -> np.ndarray:
    """Inverse of HyperCube_Noise.to_sigma for writing an error extension back."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "variance":
            return np.square(sigma)
        if kind == "ivar":
            return np.where(sigma > 0, 1.0 / np.square(sigma), 0.0)
        return sigma


def _changed(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """Voxels whose value differs; NaN == NaN counts as unchanged."""
    return ~((new == old) | (np.isnan(new) & np.isnan(old)))


def write_sky_to_fits(cube_path: str, sci_ext: int,
                      corrected: np.ndarray, pristine: np.ndarray,
                      flux_scale: float = 1.0,
                      error: dict | None = None,
                      result: "SkyResult | None" = None,
                      provenance: dict | None = None) -> dict:
    """Write an applied self-sky correction into the cube file itself.

    `corrected` / `pristine` are the in-memory cubes (after / before), on the
    session's flux scale; `error` optionally describes the error cube to update:
    {'path', 'ext', 'kind', 'corrected', 'pristine'} (1-sigma, same flux scale).

    The original science (and error) arrays are first copied, as stored on
    disk, into ``<cube>_preSelfSky.fits`` -- read from the FILE, not from
    memory, because the in-memory pristine cube may be a memory map of that
    very file. Only voxels the correction changed are then rewritten, so every
    other voxel stays bit-identical. The science header gets HCSKYSUB = T and
    the recipe, so the file cannot later be mistaken for an uncorrected one or
    have the sky subtracted a second time.

    Refuses (SkyFileError) if the file already carries a subtraction or a
    backup already exists. Returns a summary dict.
    """
    import os
    from datetime import datetime, timezone
    from astropy.io import fits

    backup = sky_backup_path(cube_path)
    if file_sky_state(cube_path, sci_ext) is not None:
        raise SkyFileError("This FITS file already has a self-sky subtraction written into it. "
                           "Restore the original first.")
    if os.path.exists(backup):
        raise SkyFileError(f"A backup already exists ({os.path.basename(backup)}). "
                           "Restore from it, or move it aside, before writing again.")
    scale = float(flux_scale) if flux_scale else 1.0

    with fits.open(cube_path, memmap=True) as hdul:
        hdu = hdul[sci_ext]
        for key in ("BSCALE", "BZERO"):
            if float(hdu.header.get(key, 1.0 if key == "BSCALE" else 0.0)) != (1.0 if key == "BSCALE" else 0.0):
                raise SkyFileError(f"The science extension uses {key}; refusing to rewrite scaled data.")
        orig_sci = np.array(hdu.data)                   # a real copy, off the file
    if orig_sci.shape != np.shape(corrected):
        raise SkyFileError(f"Cube shape {np.shape(corrected)} does not match the file {orig_sci.shape}.")

    orig_err = None
    if error is not None:
        with fits.open(error["path"], memmap=True) as hdul:
            orig_err = np.array(hdul[int(error["ext"])].data)
        if orig_err.shape != orig_sci.shape:
            raise SkyFileError("The error extension's shape does not match the cube.")

    # Backup first; nothing is modified until it is on disk and readable.
    primary = fits.PrimaryHDU()
    ph = primary.header
    ph["ORIGFILE"] = os.path.basename(cube_path)
    ph["SCIEXT"] = (int(sci_ext), "science extension index")
    if error is not None:
        ph["ERRFILE"] = os.path.basename(error["path"])
        ph["ERRSAME"] = (os.path.abspath(error["path"]) == os.path.abspath(cube_path), "error ext in the cube file")
        ph["ERREXT"] = (int(error["ext"]), "error extension index")
        ph["ERRKIND"] = (str(error.get("kind", "sigma")), "variance | ivar | sigma")
    ph["CREATED"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ph["HISTORY"] = "HyperCube self-sky backup: original arrays before the subtraction was written"
    hdus = [primary, fits.ImageHDU(orig_sci, name="ORIG_SCI")]
    if orig_err is not None:
        hdus.append(fits.ImageHDU(orig_err, name="ORIG_ERR"))
    if result is not None:
        hdus.append(fits.ImageHDU(np.asarray(result.sky, dtype=np.float64), name="SKY"))
        hdus.append(fits.ImageHDU(np.asarray(result.sky_var, dtype=np.float64), name="SKY_VAR"))
    fits.HDUList(hdus).writeto(backup, overwrite=False)
    with fits.open(backup, memmap=True) as check:
        if check["ORIG_SCI"].data.shape != orig_sci.shape:
            raise SkyFileError("Backup verification failed; the cube was NOT modified.")

    corrected = np.asarray(corrected, dtype=np.float64)
    pristine = np.asarray(pristine, dtype=np.float64)
    changed = _changed(corrected, pristine)
    new_sci = orig_sci.copy()
    new_sci[changed] = (corrected[changed] / scale).astype(orig_sci.dtype)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prov = dict(provenance or {})
    with fits.open(cube_path, mode="update", memmap=True) as hdul:
        hdul[sci_ext].data[...] = new_sci
        h = hdul[sci_ext].header
        h[SKY_FLAG] = (True, "HyperCube self-sky subtracted in file")
        h[SKY_BACKUP_KEY] = os.path.basename(backup)
        h["HCSKYSTA"] = (str(prov.get("selfsky_statistic", getattr(result, "statistic", ""))), "sky statistic")
        h["HCSKYMOD"] = (str(prov.get("selfsky_mode", getattr(result, "mode", ""))), "global | per-column")
        h["HCSKYNM"] = (int(prov.get("selfsky_n_mask", getattr(result, "n_mask", 0)) or 0), "sky-pool spaxels")
        h["HCSKYDAT"] = (stamp, "UTC written")
        h["HISTORY"] = f"HyperCube self-sky subtracted into file; backup {os.path.basename(backup)}"
        n_err = 0
        if error is not None and os.path.abspath(error["path"]) == os.path.abspath(cube_path):
            n_err = _write_error(hdul[int(error["ext"])], orig_err, error, scale)
        hdul.flush()
    if error is not None and os.path.abspath(error["path"]) != os.path.abspath(cube_path):
        with fits.open(error["path"], mode="update", memmap=True) as ehdul:
            n_err = _write_error(ehdul[int(error["ext"])], orig_err, error, scale)
            ehdul.flush()
    return dict(backup=backup, sci_voxels_changed=int(changed.sum()), err_voxels_changed=n_err,
                written_utc=stamp)


def _write_error(hdu, orig_err, error, scale):
    sig_new = np.asarray(error["corrected"], dtype=np.float64)
    sig_old = np.asarray(error["pristine"], dtype=np.float64)
    changed = _changed(sig_new, sig_old)
    out = orig_err.copy()
    out[changed] = _sigma_to_kind(sig_new[changed] / scale, error.get("kind", "sigma")).astype(orig_err.dtype)
    hdu.data[...] = out
    hdu.header[ERR_FLAG] = (True, "self-sky variance added (HyperCube)")
    hdu.header["HISTORY"] = "HyperCube self-sky: sky-estimate variance folded into this extension"
    return int(changed.sum())


def restore_sky_from_backup(cube_path: str, delete_backup: bool = True) -> dict:
    """Put back the original arrays saved by write_sky_to_fits, verify, and
    (by default) delete the backup. Returns the restored science array and a
    summary. Refuses (SkyFileError) if the backup is missing or does not belong
    to this file."""
    import os
    from astropy.io import fits

    backup = sky_backup_path(cube_path)
    if not os.path.exists(backup):
        raise SkyFileError(f"No backup found ({os.path.basename(backup)}).")
    with fits.open(backup, memmap=False) as bk:
        ph = bk[0].header
        if ph.get("ORIGFILE") != os.path.basename(cube_path):
            raise SkyFileError(f"{os.path.basename(backup)} was made from {ph.get('ORIGFILE')}, "
                               f"not {os.path.basename(cube_path)}.")
        sci_ext = int(ph["SCIEXT"])
        orig_sci = np.array(bk["ORIG_SCI"].data)
        orig_err = np.array(bk["ORIG_ERR"].data) if "ORIG_ERR" in bk else None
        err_file = ph.get("ERRFILE")
        err_ext = int(ph["ERREXT"]) if "ERREXT" in ph else None
        err_same = bool(ph.get("ERRSAME", False))
    err_path = cube_path if err_same else (os.path.join(os.path.dirname(cube_path), err_file)
                                           if err_file else None)

    with fits.open(cube_path, mode="update", memmap=True) as hdul:
        if hdul[sci_ext].data.shape != orig_sci.shape:
            raise SkyFileError("The backup's shape does not match the cube; nothing restored.")
        hdul[sci_ext].data[...] = orig_sci
        h = hdul[sci_ext].header
        for key in SKY_KEYS:
            if key in h:
                del h[key]
        h["HISTORY"] = "HyperCube self-sky subtraction reverted from backup"
        if orig_err is not None and err_same:
            hdul[err_ext].data[...] = orig_err
            if ERR_FLAG in hdul[err_ext].header:
                del hdul[err_ext].header[ERR_FLAG]
        hdul.flush()
    if orig_err is not None and not err_same and err_path:
        with fits.open(err_path, mode="update", memmap=True) as ehdul:
            ehdul[err_ext].data[...] = orig_err
            if ERR_FLAG in ehdul[err_ext].header:
                del ehdul[err_ext].header[ERR_FLAG]
            ehdul.flush()

    with fits.open(cube_path, memmap=True) as hdul:
        ok = np.array_equal(np.asarray(hdul[sci_ext].data), orig_sci, equal_nan=True)
        if orig_err is not None and err_same:
            ok &= np.array_equal(np.asarray(hdul[err_ext].data), orig_err, equal_nan=True)
    if orig_err is not None and not err_same and err_path:
        with fits.open(err_path, memmap=True) as ehdul:
            ok &= np.array_equal(np.asarray(ehdul[err_ext].data), orig_err, equal_nan=True)
    if not ok:
        raise SkyFileError("Restore could not be verified; the backup was kept.")
    if delete_backup:
        os.remove(backup)
    return dict(original=orig_sci, sci_ext=sci_ext, verified=True,
                backup_deleted=bool(delete_backup), error_restored=orig_err is not None)
