"""Per-spaxel velocity seeds for cube fits — Qt-free.

Why this exists
---------------
A cube fit starts every spaxel from the same initial guesses. On the MAUNA MUSE
sample the gas rotates by several hundred km/s across the field, so far from
where the model was built the lines sit one or more line widths away from their
seeds. A local optimizer started there wanders off: in the baseline
`SingleFits_NoConstraints` run, H-alpha and [N II] 6583 disagree by >200 km/s in
15-18% of spaxels on F14348-1447 and F09111-1007, and H-alpha lands >300 km/s
from its seed in 17-18%.

The fix here is the cheap one, deliberately — not the coarse-to-fine cascade:

1. fit the **mean spectrum** of the gated spaxels once, with the model as given;
2. **cross-correlate** every spaxel against that fit's lines (a matched filter
   over a velocity grid, all lines at once) to get a velocity and its S/N;
3. **median-filter** the velocity map (3x3 over trustworthy measurements, with
   progressively larger windows for spaxels that have none nearby);
4. shift each spaxel's line centroids by its seed before fitting
   (`HyperCube_fit.shift_line_centroids`).

Only the centroids move. Amplitudes, widths and the number of components still
start from the model, so the seed cannot hand a faint spaxel a component it
does not support — the pitfall that makes cascaded seeding dangerous.

Matching all lines at once is what keeps the correlation off its aliases: a
single-line template locks H-alpha onto [N II] 6583 at +943 km/s, but the whole
pattern only lines up at the true shift. Keep `vmax` below the closest doublet
spacing in the model (641 km/s for [S II], 675 km/s for [N II] 6548-H-alpha).

This module is Qt-free (numpy only, plus the Qt-free fit kernel) so the batch
driver can import it.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import HyperCube_fit as HF

C_KMS = HF.C_KMS

__all__ = [
    "SeedResult",
    "build_seeds",
    "mean_spectrum",
    "velocity_map",
    "smooth_seeds",
]


@dataclass
class SeedResult:
    """Everything a seeded run needs, plus what it takes to audit one."""
    seed: np.ndarray          # (ny, nx) km/s shift applied to each spaxel's centroids
    v_raw: np.ndarray         # (ny, nx) matched-filter velocity, before filtering
    snr: np.ndarray           # (ny, nx) matched-filter S/N at that velocity
    frac: np.ndarray          # (ny, nx) share of line-region signal the template explains
    edge: np.ndarray          # (ny, nx) peak fell on the ±vmax grid edge
    window: np.ndarray        # (ny, nx) median window used; 0 = unseeded, -1 = not gated
    lines: np.ndarray         # (n, 3) amp, cen [A], sigma [A] of the template
    template_source: str
    vmax: float
    dv: float
    smin: float
    fmin: float
    seconds: float

    @property
    def trusted(self) -> np.ndarray:
        """The measurements the seeds were built from."""
        with np.errstate(invalid='ignore'):
            return (np.isfinite(self.v_raw) & (self.snr >= self.smin)
                    & (self.frac >= self.fmin) & ~self.edge)

    def describe(self) -> str:
        gated = self.window >= 0
        n = int(gated.sum())
        measured = int((gated & self.trusted).sum())
        unseeded = int((gated & (self.window == 0)).sum())
        wide = int((gated & (self.window > 3)).sum())
        s = self.seed[gated & (self.window > 0)]
        span = (f'; seed p5/p95 {np.percentile(s, 5):+.0f}/{np.percentile(s, 95):+.0f} km/s'
                if s.size else '')
        return (f'velocity seeds from {self.template_source}: {measured}/{n} trusted '
                f'(S/N >= {self.smin:g}, explained >= {self.fmin:g}, |v| < {self.vmax:g} '
                f'km/s), {wide} from a window wider than 3x3, {unseeded} unseeded{span}')


# ── spectra ──────────────────────────────────────────────────────────────────

def _chunks(gated, chunk):
    xs = np.array([g[0] for g in gated], dtype=int)
    ys = np.array([g[1] for g in gated], dtype=int)
    for k in range(0, len(xs), chunk):
        yield ys[k:k + chunk], xs[k:k + chunk]


def mean_spectrum(cube, gated, err_cube=None, chunk=2048):
    """Mean spectrum of the `gated` (x, y) spaxels and its 1σ (or None).

    The mean, not the sum: the model's amplitude and continuum seeds are on a
    single-spaxel scale, so a sum would start the fit ~N times off.
    """
    nl = cube.shape[0]
    flux = np.zeros(nl)
    var = np.zeros(nl) if err_cube is not None else None
    n = 0
    for ys, xs in _chunks(gated, chunk):
        f = np.asarray(cube[:, ys, xs], dtype=np.float64)
        flux += np.nansum(f, axis=1)
        if var is not None:
            e = np.asarray(err_cube[:, ys, xs], dtype=np.float64)
            var += np.nansum(e * e, axis=1)
        n += len(xs)
    n = max(n, 1)
    return flux / n, (np.sqrt(var) / n if var is not None else None)


def _lines_from_rows(rows, vmax):
    """(amp, cen, sigma) of each line the mean-spectrum fit recovered sanely.

    Amplitudes and widths come from the fit; the centroid is the MODEL's
    (`cen_init`). The measured velocity is applied as a shift to the model's
    centroids, so it has to be measured against them too. The fit's own
    centroids are not the same thing: on a rotating field the mean spectrum is
    double-horned, and a single Gaussian per line lands tens of km/s off
    wherever lines blend — an offset every seed would then inherit.
    """
    out = []
    for r in rows:
        amp, cen, sig = (r.get('amp_fit'), r.get('cen_fit'), r.get('sigma_fit'))
        cen0 = r.get('cen_init')
        vals = np.array([amp, cen, sig, cen0], dtype=float)
        if not np.all(np.isfinite(vals)) or amp <= 0 or sig <= 0:
            continue
        # A line that ran further than the search range from its own seed on
        # the highest-S/N spectrum in the cube has swapped or drifted; leaving
        # it in would put a false feature in the template.
        if abs(cen / cen0 - 1.0) * C_KMS > vmax:
            continue
        out.append((amp, cen0, sig))
    return np.array(out, dtype=float).reshape(-1, 3)


def _lines_from_model(df):
    cols = ('Amp_0', 'Centroid_0', 'Sigma_0')
    arr = np.array([[float(r[c]) for c in cols] for _, r in df.iterrows()], dtype=float)
    ok = np.all(np.isfinite(arr), axis=1) & (arr[:, 0] > 0) & (arr[:, 2] > 0)
    return arr[ok].reshape(-1, 3)


def template_lines(cube, wavelengths, gated, params, df, df_cont, z,
                   err_cube=None, sigma_label=None, max_nfev=512,
                   sequential=False, vmax=500.0):
    """Lines-only template: a fit to the mean spectrum, else the model guesses.

    Stellar-continuum models fall straight back to the guesses — the mean
    spectrum has no stellar baseline to subtract, and a line fit on top of an
    unmodelled continuum would not be a clean template.
    """
    n_lines = len(df)
    if 'cont_type' in df_cont.columns and (df_cont['cont_type'] == 'stellar').any():
        return _lines_from_model(df), 'model guesses (stellar continuum)'
    flux, sigma = mean_spectrum(cube, gated, err_cube)
    try:
        model = HF.build_model(len(df_cont), len(np.unique(df['Line_ID'])))
        rows = HF.fit_one_spaxel(
            flux, np.zeros_like(flux), np.asarray(wavelengths, float), params, model,
            df, df_cont, z, max_nfev, sequential, (-1, -1), np.nan, np.nan, {},
            sigma, sigma_label)
    except Exception as e:
        print(f'   velocity seed: mean-spectrum fit raised {type(e).__name__}: {e}')
        rows = []
    if rows and 'error' not in rows[0]:
        lines = _lines_from_rows(rows, vmax)
        if len(lines):
            return lines, f'mean-spectrum fit ({len(lines)}/{n_lines} lines)'
    return _lines_from_model(df), 'model guesses (mean-spectrum fit failed)'


# ── the velocity map ─────────────────────────────────────────────────────────

def _window_slices(wavelengths, windows):
    lam = np.asarray(wavelengths, dtype=float)
    out = []
    for lo, hi in windows:
        idx = np.nonzero((lam >= lo) & (lam <= hi))[0]
        if idx.size >= 8:
            out.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return out


def velocity_map(cube, wavelengths, lines, windows, gated, err_cube=None,
                 vmax=500.0, dv=10.0, chunk=2048):
    """Matched-filter velocity of each gated spaxel against `lines`.

    Per window, a linear continuum is fitted to the pixels no line can reach
    within ±vmax (±3σ) and removed. The residual r is then matched against the
    template t_v shifted over a velocity grid: with weights w = 1/σ²,

        S/N(v) = Σ w r t_v / sqrt(Σ w t_v²)

    which is sqrt(Δχ²) between "no lines" and "template at v with its best
    amplitude". The peak is refined with a parabola.

    S/N alone cannot tell a match from an alias. A spaxel whose true shift lies
    beyond ±vmax does not peak on the grid edge — it peaks inside, where one
    template line overlaps a *different* data line (H-alpha on [N II] 6548 at
    v_true − 675 km/s), and a bright spaxel does that at high S/N. So `frac`
    is the share of the line-region signal the template explains,
    snr² / (Σ w r² − n_pix) over pixels a line can reach. Measured: ~1 for a
    synthetic match, still 0.66-0.70 when [N II]/H-alpha is off by 4x, and at
    most 0.22 (median 0.13) for synthetic aliases. On the real MUSE cubes the
    single-Gaussian template leaves most bright spaxels at 0.2-0.5, and those
    agree with their neighbours' velocities to <75 km/s at every fraction down
    to 0.1 — so the cut must sit just above the alias ceiling, not at 0.5
    (which rejected ~85% of F09111-1007's good measurements).

    Returns (v, snr, frac, edge) as (ny, nx) maps, NaN off the gated set;
    `edge` marks a peak on the grid edge.
    """
    lam = np.asarray(wavelengths, dtype=float)
    ny, nx = cube.shape[1], cube.shape[2]
    v_out = np.full((ny, nx), np.nan)
    s_out = np.full((ny, nx), np.nan)
    f_out = np.full((ny, nx), np.nan)
    e_out = np.zeros((ny, nx), dtype=bool)

    lines = np.asarray(lines, dtype=float).reshape(-1, 3)
    slices = _window_slices(lam, windows)
    if not len(lines) or not slices or not len(gated):
        return v_out, s_out, f_out, e_out

    pix = np.concatenate([lam[s] for s in slices])
    amp, cen, sig = lines[:, 0], lines[:, 1], lines[:, 2]

    # Continuum pixels: out of reach of every line at any velocity searched.
    reach = cen * vmax / C_KMS + 3.0 * sig
    is_cont = np.all(np.abs(pix[:, None] - cen[None, :]) > reach[None, :], axis=1)
    near = ~is_cont

    vgrid = np.arange(-vmax, vmax + 0.5 * dv, dv)
    f = 1.0 + vgrid[:, None] / C_KMS                      # (nv, 1)
    T = np.zeros((vgrid.size, pix.size))
    for a, c, s in lines:
        cc, ss = c * f, s * f
        T += a * np.exp(-0.5 * ((pix[None, :] - cc) / ss) ** 2)
    T /= max(float(np.max(T)), 1e-300)
    T2 = T * T

    # Per-window normalised abscissae for the linear continuum.
    xs_norm, win_of = [], []
    for w, s in enumerate(slices):
        x = lam[s]
        xs_norm.append((x - x.mean()) / max(np.ptp(x) / 2.0, 1e-9))
        win_of.append(np.full(x.size, w))
    xs_norm = np.concatenate(xs_norm)
    win_of = np.concatenate(win_of)

    for ys, xs in _chunks(gated, chunk):
        F = np.concatenate([np.asarray(cube[s, ys, xs], dtype=np.float64) for s in slices])
        good = np.isfinite(F) & (F != 0.0)       # exact zeros are off-detector voxels
        F = np.where(good, F, 0.0)
        if err_cube is not None:
            E = np.concatenate([np.asarray(err_cube[s, ys, xs], dtype=np.float64)
                                for s in slices])
            good &= np.isfinite(E) & (E > 0)
        else:
            E = None

        # Linear continuum per window and spaxel, closed-form least squares over
        # the good continuum pixels.
        R = np.zeros_like(F)
        for w in range(len(slices)):
            sel = win_of == w
            m = (good[sel] & is_cont[sel, None]).astype(float)
            x = xs_norm[sel, None]
            y = F[sel]
            S0, S1, S2 = m.sum(0), (m * x).sum(0), (m * x * x).sum(0)
            Sy, Sxy = (m * y).sum(0), (m * x * y).sum(0)
            det = S0 * S2 - S1 * S1
            with np.errstate(invalid='ignore', divide='ignore'):
                a0 = np.where(det > 0, (S2 * Sy - S1 * Sxy) / det,
                              np.where(S0 > 0, Sy / np.maximum(S0, 1), 0.0))
                a1 = np.where(det > 0, (S0 * Sxy - S1 * Sy) / det, 0.0)
            R[sel] = y - (a0[None, :] + a1[None, :] * x)

        if E is not None:
            W = np.where(good, 1.0 / np.where(good, E, 1.0) ** 2, 0.0)
        else:
            # No error cube: one σ per spaxel from the continuum residual's MAD.
            cm = good & is_cont[:, None]
            Rc = np.where(cm, R, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = np.nanmedian(Rc, axis=0)
                mad = 1.4826 * np.nanmedian(np.abs(Rc - med[None, :]), axis=0)
            wsp = np.where(np.isfinite(mad) & (mad > 0), 1.0 / np.maximum(mad, 1e-300) ** 2, 0.0)
            W = good * wsp[None, :]

        num = T @ (W * R)                          # (nv, N)
        den = T2 @ W
        with np.errstate(invalid='ignore', divide='ignore'):
            snr = np.where(den > 0, num / np.sqrt(den), np.nan)
        snr_f = np.where(np.isfinite(snr), snr, -np.inf)
        k = np.argmax(snr_f, axis=0)
        cols = np.arange(snr.shape[1])
        s0 = snr_f[k, cols]
        edge = (k == 0) | (k == vgrid.size - 1)
        km, kp = np.clip(k - 1, 0, None), np.clip(k + 1, None, vgrid.size - 1)
        sm, sp = snr_f[km, cols], snr_f[kp, cols]
        with np.errstate(invalid='ignore', divide='ignore'):
            curv = sm - 2.0 * s0 + sp
            delta = np.where(~edge & np.isfinite(curv) & (curv < 0),
                             0.5 * (sm - sp) / curv, 0.0)
        delta = np.clip(np.nan_to_num(delta), -0.5, 0.5)
        v = vgrid[k] + delta * dv
        ok = np.isfinite(s0) & (s0 > 0)

        chi0 = ((W * R * R)[near]).sum(0)
        npix = (W[near] > 0).sum(0)
        s0sq = np.where(ok, s0, 0.0) ** 2
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.clip(s0sq / np.maximum(chi0 - npix, s0sq), 0.0, 1.0)

        v_out[ys, xs] = np.where(ok, v, np.nan)
        s_out[ys, xs] = np.where(ok, s0, np.nan)
        f_out[ys, xs] = np.where(ok, frac, np.nan)
        e_out[ys, xs] = edge
    return v_out, s_out, f_out, e_out


def smooth_seeds(v_raw, snr, edge, gated_mask, frac=None, smin=5.0, fmin=0.25,
                 sizes=(3, 5, 9, 17, 33)):
    """Median-filtered seed map over the trustworthy measurements.

    Trustworthy = S/N >= `smin`, explained fraction >= `fmin` (see
    `velocity_map`), and not on the grid edge. A spaxel's seed is the median of
    the trustworthy velocities in the 3x3 box around it (itself included), so a
    single aliased or noise-driven peak cannot seed its own fit. Spaxels with
    none in that box take the median of the smallest larger box that has one;
    beyond the largest, the seed is 0 and the fit starts from the model exactly
    as before. Returns (seed, window).
    """
    valid = gated_mask & np.isfinite(v_raw) & np.isfinite(snr) & (snr >= smin) & ~edge
    if frac is not None:
        valid &= np.isfinite(frac) & (frac >= fmin)
    vv = np.where(valid, v_raw, np.nan)
    ny, nx = vv.shape
    seed = np.full((ny, nx), np.nan)
    window = np.full((ny, nx), -1, dtype=np.int16)
    window[gated_mask] = 0

    h = sizes[0] // 2
    boxes = sliding_window_view(np.pad(vv, h, constant_values=np.nan),
                                (sizes[0], sizes[0])).reshape(ny, nx, -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(boxes, axis=-1)
    fill = gated_mask & np.isfinite(med)
    seed[fill] = med[fill]
    window[fill] = sizes[0]

    for size in sizes[1:]:
        h = size // 2
        for y, x in zip(*np.nonzero(gated_mask & (window == 0))):
            block = vv[max(0, y - h):y + h + 1, max(0, x - h):x + h + 1]
            if np.isfinite(block).any():
                seed[y, x] = np.nanmedian(block)
                window[y, x] = size
    rest = gated_mask & (window == 0)
    seed[rest] = 0.0
    return seed, window


def build_seeds(cube, wavelengths, gated, params, df, df_cont, z, err_cube=None,
                sigma_label=None, max_nfev=512, sequential=False,
                vmax=500.0, dv=10.0, smin=5.0, fmin=0.25):
    """Template → velocity map → filtered seeds, for the `gated` (x, y) spaxels."""
    t0 = time.perf_counter()
    lines, source = template_lines(cube, wavelengths, gated, params, df, df_cont, z,
                                   err_cube, sigma_label, max_nfev, sequential, vmax)
    windows = HF.fit_windows(params, len(df_cont))
    v_raw, snr, frac, edge = velocity_map(cube, wavelengths, lines, windows, gated,
                                          err_cube, vmax=vmax, dv=dv)
    gmask = np.zeros(cube.shape[1:], dtype=bool)
    for x, y in gated:
        gmask[y, x] = True
    seed, window = smooth_seeds(v_raw, snr, edge, gmask, frac=frac,
                                smin=smin, fmin=fmin)
    return SeedResult(seed=seed, v_raw=v_raw, snr=snr, frac=frac, edge=edge,
                      window=window, lines=lines, template_source=source,
                      vmax=float(vmax), dv=float(dv), smin=float(smin),
                      fmin=float(fmin), seconds=time.perf_counter() - t0)
