"""Rest-frame fitting templates for HyperCube.

**Qt-free by contract** — numpy/pandas/astropy only. See
`HyperCube_Templates_SPEC.md` for the format and the reasoning; this module is
the implementation of §4, §5 and §6.

A *template* describes a model the way physics does — rest wavelengths,
velocity offsets, intrinsic velocity dispersions — so one file applies to every
galaxy observed with a given instrument. A *manifest* carries the handful of
facts that really are per-galaxy (cube path, extension, redshift). *Expansion*
combines the two, with the instrument's LSF read from the cube itself, into the
observed-frame `df_obs` / `df_cont` / `df` that HyperCube already fits.

The two directions here are inverses:

    template_from_model(...)   observed model  ->  rest-frame template
    expand(...)                rest-frame template + galaxy  ->  observed model

Known limitation, deliberately not addressed here
-------------------------------------------------
A K-group tie is written as ``sigma == (rest_self/rest_ref) * sigma_[ref]``,
which equalises the *observed* velocity dispersion of the members. Once the LSF
is wavelength-dependent, equal observed dispersion means slightly *unequal*
intrinsic dispersion — for MUSE, tying H-beta to [S II] this way imposes a
~68 vs ~43 km/s instrumental difference on lines the model asserts are
kinematically identical. Fixing it means folding the quadrature into the
constraint expression, which changes fit results, so it is out of scope for the
template layer and is flagged in the spec instead.
"""

from dataclasses import dataclass, field
from typing import Optional

import ast
import csv
import re
import numpy as np
import pandas as pd

import HyperCube_LSF as hlsf

C_KMS = 299792.458

TEMPLATE_FORMAT_VERSION = 1

#: §2 conventions a file must declare to be treated as a template at all.
REQUIRED_CONVENTIONS = {
    'wavelength frame': 'rest',
    'sigma convention': 'intrinsic',
}

_META_COLS = ['template_name', 'template_version', 'instrument', 'scenario',
              'n_components', 'snr_threshold', 'sequential', 'max_nfev',
              'amp_convention']
_CONVENTION_COLS = ['wavelength frame', 'wavelength unit', 'velocity unit',
                    'sigma convention', 'amplitude convention']
_REGION_COLS = ['Continuum Name', 'x1_rest_A', 'x2_rest_A', 'cont_type',
                'region_ID', 'required', 'stellar_library', 'stellar_moments',
                'poly_degree', 'knots_x_rest_A']
_LINE_COLS = ['Line_Name', 'Rest Wavelength_A', 'region_ID', 'component',
              'vel_0_kms', 'vel_lowlim_kms', 'vel_highlim_kms',
              'sigma_int_0_kms', 'sigma_int_lowlim_kms', 'sigma_int_highlim_kms',
              'amp_rel_0', 'amp_lowlim', 'amp_highlim',
              'kgroup', 'kgroup_ref', 'required']
_OVERRIDE_COLS = ['Line_Name'] + [f'constraint_{i}' for i in range(1, 6)]


# ── Records ──────────────────────────────────────────────────────────────────

@dataclass
class Template:
    meta: dict
    conventions: dict
    regions: pd.DataFrame
    lines: pd.DataFrame
    overrides: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_OVERRIDE_COLS))

    @property
    def name(self):
        return str(self.meta.get('template_name', 'template'))

    @property
    def instrument(self):
        return str(self.meta.get('instrument', '')).strip().upper()


@dataclass
class ExpansionReport:
    """What expansion did, and why. §6.6 — a deliverable, not a debug log."""
    target: str = ''
    redshift: float = float('nan')
    lsf_label: str = ''
    coverage: tuple = ()
    kept_lines: list = field(default_factory=list)
    dropped_lines: list = field(default_factory=list)       # (name, reason, lam_obs)
    kept_regions: list = field(default_factory=list)
    dropped_regions: list = field(default_factory=list)     # (name, reason)
    dropped_constraints: list = field(default_factory=list)  # (line, constraint, reason)
    kgroup_actions: list = field(default_factory=list)      # (group, action, detail)
    unresolved_bounds: list = field(default_factory=list)   # (line, which, obs_kms)
    amp_scale: float = float('nan')

    def to_text(self):
        L = [f'Target      : {self.target}',
             f'Redshift    : {self.redshift:.6f}',
             f'LSF         : {self.lsf_label}',
             f'Coverage    : {self.coverage[0]:.1f} - {self.coverage[1]:.1f} A'
             if self.coverage else 'Coverage    : (unknown)',
             f'Amp scale   : {self.amp_scale:.6g}',
             f'Lines kept  : {len(self.kept_lines)}  dropped: {len(self.dropped_lines)}']
        for nm, reason, lam in self.dropped_lines:
            L.append(f'   DROPPED  {nm:<18} {reason} (lambda_obs = {lam:.1f} A)')
        for nm, reason in self.dropped_regions:
            L.append(f'   DROPPED REGION {nm}: {reason}')
        for line, con, reason in self.dropped_constraints:
            L.append(f'   CONSTRAINT dropped on {line}: {con!r} — {reason}')
        for grp, action, detail in self.kgroup_actions:
            L.append(f'   KGROUP {grp}: {action} ({detail})')
        for line, which, val in self.unresolved_bounds:
            L.append(f'   NOTE {line}: {which} of {val:.1f} km/s is below the LSF '
                     f'— recorded as intrinsic 0')
        return '\n'.join(L)

    def to_frame(self):
        """One row per line, kept or dropped — the N/A map of §2.4."""
        rows = [{'target': self.target, 'Line_Name': nm, 'status': 'kept',
                 'lam_obs_A': lam, 'reason': ''} for nm, lam in self.kept_lines]
        rows += [{'target': self.target, 'Line_Name': nm, 'status': 'dropped',
                  'lam_obs_A': lam, 'reason': reason}
                 for nm, reason, lam in self.dropped_lines]
        return pd.DataFrame(rows)


# ── Small helpers ────────────────────────────────────────────────────────────

def _as_float(v, default=np.nan):
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


def _as_bool(v, default=False):
    s = str(v).strip().lower()
    if s in ('true', 'yes', 'y', '1'):
        return True
    if s in ('false', 'no', 'n', '0'):
        return False
    return default


def _as_list(v):
    """A stored list column ('[1.0, 2.0]' or a real list) as a list of floats."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    s = str(v).strip()
    if not s or s in ('[]', 'nan', 'None'):
        return []
    try:
        parsed = ast.literal_eval(s)
        return [float(x) for x in parsed]
    except (ValueError, SyntaxError, TypeError):
        return []


def _norm_constraints(v):
    """A constraints cell as exactly five strings."""
    if isinstance(v, (list, tuple, np.ndarray)):
        items = [('' if x is None else str(x)) for x in v]
    else:
        s = str(v).strip()
        if not s or s in ('nan', 'None'):
            items = []
        else:
            try:
                parsed = ast.literal_eval(s)
                items = [('' if x is None else str(x)) for x in parsed]
            except (ValueError, SyntaxError, TypeError):
                items = [s]
    items = [x for x in items][:5]
    return items + [''] * (5 - len(items))


def sigma_wl_to_kms(sigma_A, lam_A):
    """Å → km/s at the given wavelength (mirrors HyperCube.sigma_wl_to_kms)."""
    lam = np.asarray(lam_A, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = C_KMS * np.asarray(sigma_A, dtype=float) / lam
    return out if np.ndim(out) else float(out)


def sigma_kms_to_wl(sigma_kms, lam_A):
    """km/s → Å at the given wavelength."""
    out = np.asarray(sigma_kms, dtype=float) * np.asarray(lam_A, dtype=float) / C_KMS
    return out if np.ndim(out) else float(out)


def _ref_line_name(token):
    """The line named inside a constraint token, e.g. flux_[[N II]_6548]."""
    import re
    m = re.search(r'_\[(.*)\]', str(token))
    return m.group(1) if m else None


# ── File I/O (§4) ────────────────────────────────────────────────────────────

def _split_sections(rows):
    """Blank-row-separated sections, matching HyperCube's CSV convention."""
    sections, current = [], []
    for row in rows:
        if not any(str(c).strip() for c in row):
            if current:
                sections.append(current)
                current = []
        else:
            current.append(row)
    if current:
        sections.append(current)
    return sections


def _section_frame(section, columns=None):
    if not section:
        return pd.DataFrame(columns=columns or [])
    header, data = section[0], section[1:]
    width = len(header)
    fixed = [(r + [''] * width)[:width] for r in data]
    return pd.DataFrame(fixed, columns=header)


def read_template(path):
    """Parse a `*.hct.csv` template. Raises ValueError if it is not one."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    sections = _split_sections(rows)
    if len(sections) < 4:
        raise ValueError(f'{path}: expected at least 4 sections '
                         f'(metadata, conventions, regions, lines), found {len(sections)}')

    meta = _section_frame(sections[0]).iloc[0].to_dict()
    conventions = _section_frame(sections[1]).iloc[0].to_dict()

    # Refuse to read an observed-frame model CSV as if it were a template: the
    # numbers would be silently misinterpreted as rest-frame and intrinsic.
    for key, want in REQUIRED_CONVENTIONS.items():
        got = str(conventions.get(key, '')).strip().lower()
        if got != want:
            raise ValueError(
                f'{path}: this is not a template — its "{key}" is {got!r}, '
                f'expected {want!r}. An observed-frame fit CSV cannot be loaded '
                f'as a template; expand a template instead.')

    regions = _section_frame(sections[2], _REGION_COLS)
    lines = _section_frame(sections[3], _LINE_COLS)
    overrides = (_section_frame(sections[4], _OVERRIDE_COLS)
                 if len(sections) > 4 else pd.DataFrame(columns=_OVERRIDE_COLS))
    return Template(meta, conventions, regions, lines, overrides)


def write_template(path, template):
    """Write a template to `path` in the §4 sectioned-CSV layout."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        meta = {k: template.meta.get(k, '') for k in _META_COLS}
        w.writerow(list(meta.keys()))
        w.writerow(list(meta.values()))
        w.writerow([])
        conv = {k: template.conventions.get(k, '') for k in _CONVENTION_COLS}
        w.writerow(list(conv.keys()))
        w.writerow(list(conv.values()))
        w.writerow([])
        for frame, cols in ((template.regions, _REGION_COLS),
                            (template.lines, _LINE_COLS)):
            out = frame.reindex(columns=cols)
            w.writerow(cols)
            w.writerows(out.values.tolist())
            w.writerow([])
        if len(template.overrides):
            out = template.overrides.reindex(columns=_OVERRIDE_COLS)
            w.writerow(_OVERRIDE_COLS)
            w.writerows(out.values.tolist())
    return path


# ── Observed model -> template ───────────────────────────────────────────────

def template_from_model(df_obs, df_cont, df, lsf, meta=None):
    """Turn a live HyperCube model into a rest-frame template.

    `df_obs`/`df_cont`/`df` are the GUI's globals; `lsf` is the instrument LSF
    the cube was fitted with. Redshift comes from `df_obs`.

    Observed values are inverted: centroids become velocity offsets from
    systemic, observed sigmas become intrinsic ones, amplitudes become
    fractions of the brightest line, and region edges are de-redshifted.
    """
    z = _as_float(df_obs.loc[0, 'redshift'], 0.0)
    if not np.isfinite(z):
        z = 0.0
    one_z = 1.0 + z

    meta = dict(meta or {})
    meta.setdefault('template_name', str(df_obs.loc[0, 'sourcename']) or 'template')
    meta.setdefault('template_version', TEMPLATE_FORMAT_VERSION)
    meta.setdefault('scenario', 'none')
    meta.setdefault('n_components', '')
    meta.setdefault('snr_threshold', '')
    meta.setdefault('sequential', False)
    meta.setdefault('max_nfev', 512)
    meta.setdefault('amp_convention', 'relative')
    meta.setdefault('instrument', '')

    conventions = {'wavelength frame': 'rest', 'wavelength unit': 'Angstrom',
                   'velocity unit': 'km/s', 'sigma convention': 'intrinsic',
                   'amplitude convention': 'relative'}

    # ── regions ──
    rrows = []
    for _, r in df_cont.iterrows():
        knots = _as_list(r.get('knots_x'))
        rrows.append({
            'Continuum Name': r.get('Continuum Name', ''),
            'x1_rest_A': _as_float(r.get('x1')) / one_z,
            'x2_rest_A': _as_float(r.get('x2')) / one_z,
            'cont_type': r.get('cont_type', 'linear'),
            'region_ID': int(_as_float(r.get('region_ID'), 0)),
            # Optional by default, like lines: a region outside a given cube's
            # coverage has nothing to fit, and aborting the whole target over it
            # would stop a batch run from fitting the half it can see. Set this
            # True by hand for a region the model is meaningless without.
            'required': False,
            'stellar_library': r.get('stellar_library', ''),
            'stellar_moments': r.get('stellar_moments', ''),
            'poly_degree': r.get('poly_degree', ''),
            'knots_x_rest_A': [k / one_z for k in knots] if knots else [],
        })
    regions = pd.DataFrame(rrows, columns=_REGION_COLS)

    # ── lines ──
    amps = pd.to_numeric(df.get('Amp_0'), errors='coerce')
    amp_scale = float(np.nanmax(amps)) if len(amps) and np.isfinite(np.nanmax(amps)) else 1.0
    if amp_scale <= 0:
        amp_scale = 1.0

    # Components are the 2-member same-rest-wavelength groups the kernel pairs
    # up; narrower member first, matching HyperCube_fit.component_pairs.
    rest_round = pd.to_numeric(df['Rest Wavelength'], errors='coerce').round(4)
    order_in_group = {}
    for rest_val, grp in df.groupby(rest_round, sort=False):
        sig = pd.to_numeric(grp['Sigma_0'], errors='coerce').fillna(np.inf)
        for rank, idx in enumerate(sig.sort_values().index, start=1):
            order_in_group[idx] = rank

    lrows, notes = [], []
    for idx, r in df.iterrows():
        rest = _as_float(r['Rest Wavelength'])
        cen = _as_float(r['Centroid_0'])
        lam_sys = rest * one_z
        vel0 = C_KMS * (cen / lam_sys - 1.0) if np.isfinite(cen) and lam_sys else 0.0
        lo = _as_float(r.get('Centroid_0_lowlim'))
        hi = _as_float(r.get('Centroid_0_highlim'))

        def _v(x):
            return C_KMS * (x / lam_sys - 1.0) if np.isfinite(x) and lam_sys else np.nan

        def _to_int(sig_A, which):
            """Observed sigma (Å at `cen`) -> intrinsic km/s, 0 if unresolved."""
            if not np.isfinite(sig_A):
                return np.nan
            obs_kms = sigma_wl_to_kms(sig_A, cen)
            if np.isinf(obs_kms):
                return np.inf
            val = hlsf.observed_to_intrinsic(obs_kms, cen, lsf)
            if not np.isfinite(val):
                # Below the instrumental width: as an intrinsic statement this
                # is "no constraint", which is 0 — not NaN, and not the
                # meaningless observed number it came from.
                notes.append((str(r['Line_Name']), which, float(obs_kms)))
                return 0.0
            return float(val)

        lrows.append({
            'Line_Name': r['Line_Name'],
            'Rest Wavelength_A': rest,
            'region_ID': int(_as_float(r.get('region_ID'), 0)),
            'component': order_in_group.get(idx, 1),
            'vel_0_kms': vel0,
            'vel_lowlim_kms': _v(lo),
            'vel_highlim_kms': _v(hi),
            'sigma_int_0_kms': _to_int(_as_float(r.get('Sigma_0')), 'sigma guess'),
            'sigma_int_lowlim_kms': _to_int(_as_float(r.get('Sigma_0_lowlim')), 'sigma lower bound'),
            'sigma_int_highlim_kms': _to_int(_as_float(r.get('Sigma_0_highlim')), 'sigma upper bound'),
            'amp_rel_0': _as_float(r.get('Amp_0')) / amp_scale,
            'amp_lowlim': _as_float(r.get('Amp_0_lowlim'), 0.0),
            'amp_highlim': _as_float(r.get('Amp_0_highlim'), np.inf),
            'kgroup': r.get('kgroup', '') or '',
            'kgroup_ref': _as_bool(r.get('kgroup_ref')),
            'required': False,
        })
    lines = pd.DataFrame(lrows, columns=_LINE_COLS)

    # ── explicit constraints ──
    orows = []
    for _, r in df.iterrows():
        cons = _norm_constraints(r.get('constraints'))
        if any(c.strip() for c in cons):
            orows.append({'Line_Name': r['Line_Name'],
                          **{f'constraint_{i+1}': cons[i] for i in range(5)}})
    overrides = pd.DataFrame(orows, columns=_OVERRIDE_COLS)

    t = Template(meta, conventions, regions, lines, overrides)
    t.conversion_notes = notes            # attribute, not part of the file
    return t


# ── Template -> observed model (§6) ──────────────────────────────────────────

def expand(template, redshift, wavelengths, lsf, target_name='',
           amp_scale=1.0, coverage_nsigma=3.0, instrument=None,
           ref_spectrum=None):
    """Build observed-frame `(df_obs, df_cont, df, ExpansionReport)`.

    `ref_spectrum` is the reference spaxel's spectrum. Given one, each region's
    linear continuum is seeded from it (`seed_linear_continuum`) and the
    amplitude scale is measured *above* that continuum, so the initial guess
    sits on the data rather than at zero. Without one — a dry run, which reads
    no pixels — the continuum starts at zero and `amp_scale` is used as passed.

    `coverage_nsigma` sets how far inside the cube a line must sit to count as
    covered, in units of its own observed sigma.
    """
    z = float(redshift)
    one_z = 1.0 + z
    wl = np.asarray(wavelengths, dtype=float)
    wl = wl[np.isfinite(wl)]
    lo_cov, hi_cov = (float(wl.min()), float(wl.max())) if wl.size else (np.nan, np.nan)

    rep = ExpansionReport(target=target_name or template.name, redshift=z,
                          lsf_label=getattr(lsf, 'label', ''),
                          coverage=(lo_cov, hi_cov), amp_scale=float(amp_scale))

    if instrument and template.instrument and \
            template.instrument not in str(instrument).strip().upper():
        raise ValueError(
            f'template {template.name!r} declares instrument '
            f'{template.instrument!r} but the cube reports {instrument!r}')

    # ── regions ──
    keep_regions, rrows = [], []
    for _, r in template.regions.iterrows():
        x1 = _as_float(r['x1_rest_A']) * one_z
        x2 = _as_float(r['x2_rest_A']) * one_z
        name = str(r['Continuum Name'])
        covered = np.isfinite(x1) and np.isfinite(x2) and x1 >= lo_cov and x2 <= hi_cov
        if not covered:
            reason = f'region {x1:.1f}-{x2:.1f} A outside cube coverage'
            if _as_bool(r.get('required'), False):
                raise ValueError(f'required region {name!r}: {reason}')
            rep.dropped_regions.append((name, reason))
            continue
        keep_regions.append(int(_as_float(r['region_ID'], 0)))
        knots = _as_list(r.get('knots_x_rest_A'))
        # Seed the continuum from this galaxy's own data where we have it.
        slope0 = intercept0 = 0.0
        if ref_spectrum is not None and str(r.get('cont_type') or 'linear') == 'linear':
            slope0, intercept0 = seed_linear_continuum(ref_spectrum, wl, x1, x2)
        rrows.append({
            'Continuum Name': name, 'x1': x1, 'x2': x2,
            'Slope_0': slope0, 'Intercept_0': intercept0,
            'Slope_fit': np.nan, 'Intercept_fit': np.nan,
            'region_ID': int(_as_float(r['region_ID'], 0)),
            'lineactor': None,
            'cont_type': str(r.get('cont_type') or 'linear'),
            'knots_x': [k * one_z for k in knots], 'knots_y_0': [], 'knots_y_fit': [],
            'poly_degree': _as_float(r.get('poly_degree')),
            'poly_coef_0': [], 'poly_coef_fit': [],
            'stellar_library': r.get('stellar_library', ''),
            'stellar_moments': _as_float(r.get('stellar_moments'), 2),
        })
    rep.kept_regions = [row['Continuum Name'] for row in rrows]

    # Amplitudes are relative to the brightest line, so the scale has to be the
    # peak *above* the continuum. Measured from the raw spectrum it carries the
    # continuum level with it and every amplitude guess is inflated by it —
    # which, paired with a zero continuum, is how the old initial guess managed
    # to be wrong in both directions at once.
    if ref_spectrum is not None and rrows:
        peaks = []
        for row in rrows:
            m = (wl >= row['x1']) & (wl <= row['x2'])
            if not m.any():
                continue
            resid = (np.asarray(ref_spectrum, dtype=float)[m]
                     - (row['Slope_0'] * wl[m] + row['Intercept_0']))
            if np.isfinite(resid).any():
                peaks.append(np.nanmax(resid))
        if peaks and np.isfinite(max(peaks)) and max(peaks) > 0:
            amp_scale = float(max(peaks))
            rep.amp_scale = amp_scale

    # ── lines ──
    lrows, dropped_names = [], set()
    for _, r in template.lines.iterrows():
        name = str(r['Line_Name'])
        rest = _as_float(r['Rest Wavelength_A'])
        rid = int(_as_float(r['region_ID'], 0))
        lam_sys = rest * one_z
        v0 = _as_float(r['vel_0_kms'], 0.0)
        lam_obs = lam_sys * (1.0 + v0 / C_KMS)

        if rid not in keep_regions:
            rep.dropped_lines.append((name, 'its continuum region was dropped', lam_obs))
            dropped_names.add(name)
            continue

        sig_int = _as_float(r['sigma_int_0_kms'], 0.0)
        sig_obs_kms = hlsf.intrinsic_to_observed(sig_int, lam_obs, lsf)
        sig_obs_A = sigma_kms_to_wl(sig_obs_kms, lam_obs)
        margin = coverage_nsigma * (sig_obs_A if np.isfinite(sig_obs_A) else 0.0)

        if not (np.isfinite(lam_obs) and lo_cov <= lam_obs - margin
                and lam_obs + margin <= hi_cov):
            reason = f'lambda_obs {lam_obs:.1f} A outside {lo_cov:.0f}-{hi_cov:.0f} A'
            if _as_bool(r.get('required'), False):
                raise ValueError(f'required line {name!r}: {reason}')
            rep.dropped_lines.append((name, reason, lam_obs))
            dropped_names.add(name)
            continue

        def _bound_A(which, default, is_floor=False):
            val = _as_float(r[which], default)
            if np.isinf(val):
                return val
            if is_floor and (not np.isfinite(val) or val <= 0):
                # An intrinsic floor of 0 means "no constraint", and it has to
                # expand back to no constraint. Putting it through the
                # quadrature instead would return the LSF width and impose a
                # bound that was never in the model — on MUSE that silently
                # raises a 20 km/s floor to ~71 km/s at H-beta, and a line
                # genuinely narrower than that cannot then be fitted.
                return 0.0
            obs = hlsf.intrinsic_to_observed(val, lam_obs, lsf)
            return sigma_kms_to_wl(obs, lam_obs)

        vlo = _as_float(r['vel_lowlim_kms'])
        vhi = _as_float(r['vel_highlim_kms'])
        lrows.append({
            'Line_ID': 0,                      # renumbered below
            'Line_Name': name,
            'SNR': np.nan,
            'Rest Wavelength': rest,
            'Amp_0': _as_float(r['amp_rel_0'], 0.0) * float(amp_scale),
            'Amp_0_lowlim': _as_float(r['amp_lowlim'], 0.0),
            'Amp_0_highlim': _as_float(r['amp_highlim'], np.inf),
            'Centroid_0': lam_obs,
            'Centroid_0_lowlim': lam_sys * (1.0 + vlo / C_KMS) if np.isfinite(vlo) else np.nan,
            'Centroid_0_highlim': lam_sys * (1.0 + vhi / C_KMS) if np.isfinite(vhi) else np.nan,
            'Sigma_0': sig_obs_A,
            'Sigma_0_lowlim': _bound_A('sigma_int_lowlim_kms', 0.0, is_floor=True),
            'Sigma_0_highlim': _bound_A('sigma_int_highlim_kms', np.inf),
            'kgroup': str(r.get('kgroup') or ''),
            'kgroup_ref': _as_bool(r.get('kgroup_ref')),
            'Amp_fit': np.nan, 'Centroid_fit': np.nan, 'Sigma_fit': np.nan,
            'region_ID': rid,
            'curveactor': None,
            'constraints': ['', '', '', '', ''],
        })
        rep.kept_lines.append((name, lam_obs))

    df = pd.DataFrame(lrows)
    df_cont = pd.DataFrame(rrows)

    # ── §6.4 repair ──
    if len(df):
        df = df.reset_index(drop=True)
        df['Line_ID'] = np.arange(len(df), dtype=float)
    if len(df_cont):
        df_cont = df_cont.reset_index(drop=True)   # HyperCube_fit.py:414 needs 0..N-1

    df = _apply_constraints(df, template, dropped_names, rep)
    df = _repair_kgroups(df, rep, dropped_names, template)
    df = _sync_kgroup_ties(df)

    df_obs = pd.DataFrame({'sourcename': [target_name or template.name],
                           'redshift': [z],
                           'resolvingpower': [float(lsf.R(np.median(wl))) if wl.size else np.nan]})
    return df_obs, df_cont, df, rep


def _apply_constraints(df, template, dropped_names, rep):
    """Overlay the template's explicit constraints, dropping dead references."""
    if not len(df):
        return df
    by_line = {str(r['Line_Name']): r for _, r in template.overrides.iterrows()}
    out = []
    for _, row in df.iterrows():
        name = str(row['Line_Name'])
        cons = ['', '', '', '', '']
        src = by_line.get(name)
        if src is not None:
            raw = [str(src.get(f'constraint_{i+1}', '') or '') for i in range(5)]
            kept = []
            for c in raw:
                if not c.strip():
                    continue
                ref = _ref_line_name(c)
                if ref is not None and ref in dropped_names:
                    rep.dropped_constraints.append(
                        (name, c, f'references dropped line {ref!r}'))
                    continue
                kept.append(c)
            cons = (kept + [''] * 5)[:5]
        row = row.copy()
        row['constraints'] = cons
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


# The exact forms HyperCube treats as K-group-managed (HyperCube.py:13720,
# :13724). Anchored so a user's velocity *window* ('vel == vel_[B] +- 300') is
# left alone — only the bare tie is regenerated.
_KG_SIGMA_RE = re.compile(r'sigma\s*==\s*\d+(?:\.\d+)?\s*\*\s*sigma_\[')
_KG_VEL_RE = re.compile(r'^\s*vel\s*==\s*vel_\[.*\]\s*$')


def _sync_kgroup_ties(df):
    """Rebuild every K-group's velocity + dispersion ties from its membership.

    Mirrors `FitParamsWindow._sync_kgroup_constraints`. Necessary after a
    repair: the ties that were dropped for naming a vanished line *were* the
    group's ties, so without this a repaired model keeps its K-group labels
    while silently fitting every member independently. Ties are appended last,
    as the GUI does, so they take precedence over a manual sigma constraint.
    """
    if not len(df) or 'kgroup' not in df.columns:
        return df
    df = df.copy()
    rest = pd.to_numeric(df['Rest Wavelength'], errors='coerce')
    for group, members in df.groupby(df['kgroup'].astype(str)):
        if not group or group == 'nan':
            continue
        idx = list(members.index)
        flagged = [j for j in idx if bool(df.loc[j, 'kgroup_ref'])]
        ref_i = flagged[0] if flagged else idx[0]
        ref_name = str(df.loc[ref_i, 'Line_Name'])
        r_ref = float(rest.loc[ref_i])
        for j in idx:
            cons = [c for c in _norm_constraints(df.loc[j, 'constraints'])
                    if c.strip() and not _KG_VEL_RE.match(c)
                    and not _KG_SIGMA_RE.search(c)]
            if j != ref_i and len(idx) >= 2:
                r_self = float(rest.loc[j])
                ratio = (r_self / r_ref) if (r_self and r_ref) else 1.0
                cons = cons + [f'vel == vel_[{ref_name}]',
                               f'sigma == {ratio:.6f} * sigma_[{ref_name}]']
            df.at[j, 'constraints'] = (cons[-5:] + [''] * 5)[:5]
    return df


def _repair_kgroups(df, rep, dropped_names=(), template=None):
    """Re-anchor or clear K-groups whose membership changed (§6.4.2).

    Distinguishes two cases that look identical in the data but are not: a
    group whose chosen anchor fell out of coverage (a genuine repair, worth
    reporting) and a group that never named one (the default — first member in
    model order — and not an event at all).
    """
    if not len(df) or 'kgroup' not in df.columns:
        return df
    df = df.copy()
    lost_anchor = set()
    if template is not None and len(template.lines):
        for _, r in template.lines.iterrows():
            if str(r['Line_Name']) in set(dropped_names) and _as_bool(r.get('kgroup_ref')):
                lost_anchor.add(str(r.get('kgroup') or ''))
    for group, members in df.groupby(df['kgroup'].astype(str)):
        if not group or group == 'nan':
            continue
        idx = list(members.index)
        if len(idx) < 2:
            df.loc[idx, 'kgroup'] = ''
            df.loc[idx, 'kgroup_ref'] = False
            rep.kgroup_actions.append(
                (group, 'cleared', 'fewer than two surviving members'))
            continue
        flags = df.loc[idx, 'kgroup_ref'].astype(bool)
        if not flags.any():
            df.loc[idx[0], 'kgroup_ref'] = True
            if group in lost_anchor:
                rep.kgroup_actions.append(
                    (group, 're-anchored',
                     f"chosen reference was dropped; now {df.loc[idx[0], 'Line_Name']}"))
            # else: the group never named one, so the first member in model
            # order is simply the default. Not an event.
        elif int(flags.sum()) > 1:
            # Keep the first flagged member, clear the others — clearing every
            # member after idx[0] would leave the group anchorless when the
            # flag sat further down.
            keep = next(j for j in idx if bool(df.loc[j, 'kgroup_ref']))
            for j in idx:
                if j != keep:
                    df.loc[j, 'kgroup_ref'] = False
            rep.kgroup_actions.append(
                (group, 'de-duplicated',
                 f"more than one reference; kept {df.loc[keep, 'Line_Name']}"))
    return df


# ── Manifest (§5) ────────────────────────────────────────────────────────────

MANIFEST_COLS = ['target_id', 'cube_path', 'cube_ext', 'redshift', 'z_source',
                 'ref_spaxel_x', 'ref_spaxel_y', 'snr_threshold', 'enabled', 'notes']


def read_manifest(path):
    """Read a target manifest, keeping only enabled rows in file order."""
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ('target_id', 'cube_path', 'redshift') if c not in m.columns]
    if missing:
        raise ValueError(f'{path}: manifest is missing required column(s) {missing}')
    if 'enabled' in m.columns:
        m = m[[_as_bool(v, True) for v in m['enabled']]]
    return m.reset_index(drop=True)


def write_manifest(path, frame):
    frame.reindex(columns=MANIFEST_COLS).to_csv(path, index=False)
    return path


def resolve_redshift(name, timeout=10):
    """NED redshift for an object name. -> (name_as_supplied, ned_name, z) or None.

    Qt-free; network-dependent. Mirrors the resolver behind the GUI's "Resolve
    name (NED)" button. Used only to *draft* a manifest — NED returns a redshift
    for an identifier, not necessarily the systemic redshift you want to fit
    against, so the result is always for review (§5).

    **The caller's name is returned unchanged as the first element.** NED is
    asked for the MEASUREMENT, never for a relabelling: the canonical identifier
    for "F01364-1042" may come back as "2MASX J01385289-1027113", and writing
    that into a manifest breaks every join the user has against their own
    catalogues, file names and target directories. NED's own name is returned
    alongside, as evidence of what matched, so a reviewer can confirm the
    resolution was correct without the identity being silently swapped.
    """
    try:
        from astroquery.ipac.ned import Ned
        result = Ned.query_object(name)
        return name, str(result['Object Name'][0]), float(result['Redshift'][0])
    except ImportError:
        pass
    except Exception:
        return None
    import urllib.request, urllib.parse
    url = ('https://ned.ipac.caltech.edu/cgi-bin/nph-objsearch'
           f'?objname={urllib.parse.quote(name)}&extend=no&of=ascii_tab&list_limit=1'
           '&img_stamp=false&zv_breaker=30000&out_csys=Equatorial&out_equinox=J2000.0')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'HyperCube/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode('utf-8', errors='replace')
        rows = [l for l in text.splitlines() if l and not l.startswith('#')]
        if len(rows) < 2:
            return None
        d = dict(zip(rows[0].split('\t'), rows[1].split('\t')))
        z = d.get('Redshift', '').strip()
        return name, d.get('Object Name', name).strip(), (float(z) if z else None)
    except Exception:
        return None


# ── Cube-side helpers (astropy only) ─────────────────────────────────────────

def cube_axes(header):
    """The spectral axis (Å) implied by a cube header, or None."""
    n = header.get('NAXIS3')
    crval = header.get('CRVAL3')
    cdelt = header.get('CD3_3', header.get('CDELT3'))
    if not n or crval is None or not cdelt:
        return None
    return crval + cdelt * (np.arange(int(n)) - header.get('CRPIX3', 1) + 1)


def seed_linear_continuum(spec, wavelengths, x1, x2, clip=2.5, iters=3):
    """Robust slope/intercept for one region, from the reference spectrum.

    The continuum *level* is a property of the galaxy and the instrument's flux
    units, not of the model, so a template cannot carry it — it has to be
    measured from the cube the template is being applied to. Without this the
    expanded model starts with its continuum pinned at zero, which puts the
    initial guess hundreds of flux units below the data and hands lmfit a
    starting point that can fail outright on faint or crowded lines.

    Emission is clipped asymmetrically (points far *above* the running fit are
    dropped, points below are kept) so the lines the model is there to fit do
    not drag the baseline up with them.
    """
    wl = np.asarray(wavelengths, dtype=float)
    y = np.asarray(spec, dtype=float)
    m = (wl >= x1) & (wl <= x2) & np.isfinite(y)
    if m.sum() < 4:
        return 0.0, 0.0
    xs, ys = wl[m], y[m]
    keep = np.ones(xs.shape, dtype=bool)
    slope = intercept = 0.0
    for _ in range(max(1, iters)):
        if keep.sum() < 4:
            break
        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
        resid = ys - (slope * xs + intercept)
        sigma = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        keep = resid <= clip * sigma          # asymmetric: emission only
    return float(slope), float(intercept)


def reference_spectrum(data, wavelengths, regions_obs, ref_xy=None, border=3):
    """The reference spaxel's spectrum and its (x, y).

    Uses the given spaxel, else the brightest by median flux inside the model's
    own regions, away from the border so a hot edge column cannot set the scale.
    """
    wl = np.asarray(wavelengths, dtype=float)
    mask = np.zeros(wl.shape, dtype=bool)
    for x1, x2 in regions_obs:
        mask |= (wl >= x1) & (wl <= x2)
    if not mask.any():
        mask[:] = True
    if ref_xy is not None:
        # The spaxel is already known, so read that one column rather than
        # pulling every spaxel of every masked plane off disk just to find
        # it — the difference between one spectrum and a multi-GB slab.
        x, y = int(ref_xy[0]), int(ref_xy[1])
        return np.asarray(data[:, y, x], dtype=np.float64), (x, y)
    sub = np.asarray(data[mask], dtype=np.float64)
    if True:
        import warnings as _w
        with np.errstate(invalid='ignore'), _w.catch_warnings():
            _w.simplefilter('ignore', RuntimeWarning)
            img = np.nanmedian(sub, axis=0)
        if border > 0 and img.shape[0] > 2 * border and img.shape[1] > 2 * border:
            inner = img[border:-border, border:-border]
            y, x = np.unravel_index(np.nanargmax(inner), inner.shape)
            y, x = y + border, x + border
        else:
            y, x = np.unravel_index(np.nanargmax(img), img.shape)
    return np.asarray(data[:, y, x], dtype=np.float64), (int(x), int(y))


def reference_amplitude(data, wavelengths, regions_obs, ref_xy=None, border=3):
    """Peak flux to scale a template's relative amplitudes by (§6.3).

    Uses the given spaxel, else the brightest spaxel (by median flux inside the
    model's own regions) away from the border, so a hot edge column cannot set
    the scale for the whole cube.
    """
    wl = np.asarray(wavelengths, dtype=float)
    mask = np.zeros(wl.shape, dtype=bool)
    for x1, x2 in regions_obs:
        mask |= (wl >= x1) & (wl <= x2)
    if not mask.any():
        mask[:] = True
    sub = np.asarray(data[mask], dtype=np.float64)
    if ref_xy is not None:
        x, y = int(ref_xy[0]), int(ref_xy[1])
    else:
        import warnings as _w
        with np.errstate(invalid='ignore'), _w.catch_warnings():
            # Cube edges are commonly all-NaN; that is what nanmedian is for,
            # and its warning about it says nothing the caller can act on.
            _w.simplefilter('ignore', RuntimeWarning)
            img = np.nanmedian(sub, axis=0)
        if border > 0 and img.shape[0] > 2 * border and img.shape[1] > 2 * border:
            inner = img[border:-border, border:-border]
            y, x = np.unravel_index(np.nanargmax(inner), inner.shape)
            y, x = y + border, x + border
        else:
            y, x = np.unravel_index(np.nanargmax(img), img.shape)
    spec = np.asarray(sub[:, y, x], dtype=np.float64)
    peak = np.nanmax(spec) if np.isfinite(spec).any() else np.nan
    return (float(peak) if np.isfinite(peak) and peak > 0 else 1.0), (int(x), int(y))


def lsf_for_cube(header, wavelengths, muse_model=hlsf.MUSE_LSF_DEFAULT):
    """The instrument LSF for a cube, from its own header."""
    instrument = str(header.get('INSTRUME', '')).strip().upper()
    if 'MUSE' in instrument:
        return hlsf.muse_lsf(muse_model)
    if 'KCWI' in instrument or 'KCRM' in instrument:
        wl = np.asarray(wavelengths, dtype=float)
        lam_mid = float(np.median(wl)) if wl.size else None
        blue = _as_float(header.get('BCWAVE'))
        red = _as_float(header.get('RCWAVE'))
        arm = 'B'
        if lam_mid and np.isfinite(blue) and np.isfinite(red):
            arm = 'B' if abs(lam_mid - blue) <= abs(lam_mid - red) else 'R'
        elif np.isfinite(red):
            arm = 'R'
        grating = str(header.get('BGRATNAM' if arm == 'B' else 'RGRATNAM', '')).strip()
        model = hlsf.kcwi_lsf(grating, header.get('IFUNAM', 'LARGE'))
        if model is not None:
            return model
    raise ValueError(f'no LSF known for instrument {instrument!r}; '
                     f'add it to HyperCube_LSF before expanding onto this cube')
