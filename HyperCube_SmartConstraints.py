"""Smart Constraints: auto-populate HyperCube's kinematic groups, doublet flux
ratios, and parameter bounds from a chosen physical scenario, instead of
requiring the user to type each relation by hand in the Edit Line dialog.

Pure, Qt-free logic operating on a copy of the emission-line DataFrame (the
same schema as HyperCube.py's module-level `df`). Nothing here mutates its
input or imports PyQt5/HyperCube.py, so it can be unit-tested standalone and
reused later by an eventual "lazy initiate" auto-model-builder.

All generated constraint strings use exactly the syntax HyperCube's own
`add_dataframe_constraints_to_params` parser already understands (see the
in-app "Constraint Syntax Help" dialog): e.g. ``flux == 2.94 * flux_[line]``,
``sigma >= sigma_[line]``, ``amp <= amp_[line]``.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Speed of light (km/s) -- mirrors HyperCube.py's C_KMS.
C_KMS = 299792.458

SCENARIOS = ('quiescent', 'outflow', 'shock', 'custom')

# ── Doublet / decrement reference tables (rest wavelengths, Angstrom) ───────
#
# Atomic-fixed doublets: ratio set by transition probabilities, not gas
# conditions -- always applied when both lines are present, in every
# scenario. Tuple = (rest_faint, rest_bright, bright/faint flux ratio).
ATOMIC_FIXED_DOUBLETS = [
    (6548.05, 6583.45, 2.94),   # [N II] 6548,6584
    (4958.91, 5006.84, 2.98),   # [O III] 4959,5007
    (6363.78, 6300.30, 3.00),   # [O I] 6364,6300
]

# Density-sensitive doublets: ratio genuinely depends on electron density.
# Tuple = (rest_blue, rest_red, ratio_lo, ratio_hi) bounding
# flux_blue / flux_red. Constrained line is the blue member, reference is
# the red member (matches the app's own worked example for [S II]).
DENSITY_SENSITIVE_DOUBLETS = [
    (6716.44, 6730.82, 0.44, 1.45),   # [S II] 6716,6731
    (3726.03, 3728.82, 0.35, 1.50),   # [O II] 3726,3729 (approximate)
]

# Balmer decrement: (rest_Hbeta, rest_Halpha, case-B ratio at Te=1e4K, ne=1e2).
BALMER_PAIR = (4861.35, 6562.79, 2.86)

# Rest-wavelength match tolerance (Angstrom) for identifying a df line
# against the tables above.
_REST_TOL = 1.0

# Per-scenario absolute bound windows, in km/s. 'core' / 'secondary' each
# give (sigma_lo, sigma_hi, centroid_window) -- the centroid window is a
# symmetric +/- velocity range around the line's current Centroid_0 guess.
# Secondary (outflow/shock) components get wider windows than core: they are
# expected to be broader and can be substantially blueshifted.
_BOUND_WINDOWS = {
    'quiescent': {'core':      (20.0, 350.0, 300.0),
                  'secondary': (20.0, 350.0, 300.0)},
    'outflow':   {'core':      (20.0, 350.0, 300.0),
                  'secondary': (150.0, 1200.0, 1500.0)},
    'shock':     {'core':      (20.0, 500.0, 400.0),
                  'secondary': (150.0, 2000.0, 2000.0)},
    'custom':    {'core':      (20.0, 350.0, 300.0),
                  'secondary': (150.0, 1200.0, 1500.0)},
}

# Default ionization-potential split threshold (eV): separates high-
# ionization lines (e.g. [O III], [Ne III], [Ne V]) from low-ionization /
# recombination lines (Balmer series, [N II], [S II], [O II]).
DEFAULT_IP_THRESHOLD_EV = 35.0


@dataclass
class SmartConstraintPlan:
    """Plain-data result of build_plan(): everything Apply needs to write,
    plus a human-readable summary for the Preview box."""
    kgroup_assignments: dict = field(default_factory=dict)      # Line_ID -> 'K1'/'K2'/'K3'/''
    constraint_additions: dict = field(default_factory=dict)    # Line_Name -> [constraint_str, ...]
    bound_updates: dict = field(default_factory=dict)           # Line_ID -> {col: value}
    summary_lines: list = field(default_factory=list)


def _sigma_kms_to_wl(sigma_kms, centroid_wl):
    """km/s -> wavelength-space sigma (Angstrom). Returns NaN on bad input."""
    try:
        sigma_kms, centroid_wl = float(sigma_kms), float(centroid_wl)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(sigma_kms) or not np.isfinite(centroid_wl):
        return np.nan
    return sigma_kms * centroid_wl / C_KMS


def classify_components(df):
    """Classify every line as 'core' (narrowest of its rest-wavelength group,
    or a singleton) or 'secondary' (any broader group-mate).

    Groups lines by rounded 'Rest Wavelength' -- the same signal HyperCube's
    own `_component_pairs` uses to find narrow/broad partners -- so this
    works regardless of what the user named the line (no reliance on a
    '_b' suffix convention).

    Returns {Line_ID: 'core'|'secondary'}.
    """
    result = {}
    if 'Rest Wavelength' not in df.columns or 'Line_ID' not in df.columns:
        return result
    groups = {}
    for _, row in df.iterrows():
        rw = row.get('Rest Wavelength')
        try:
            rw = float(rw)
        except (TypeError, ValueError):
            rw = np.nan
        if not np.isfinite(rw):
            continue
        lid = int(row['Line_ID'])
        sigma = row.get('Sigma_0')
        try:
            sigma = float(sigma)
        except (TypeError, ValueError):
            sigma = np.inf
        groups.setdefault(round(rw, 4), []).append((lid, sigma))
    for members in groups.values():
        members.sort(key=lambda t: (t[1] if np.isfinite(t[1]) else np.inf))
        result[members[0][0]] = 'core'
        for lid, _sigma in members[1:]:
            result[lid] = 'secondary'
    return result


def _pair_core_secondary(df):
    """Pairs (core_line_id, secondary_line_id) for every secondary component,
    matched to the narrowest member of its own rest-wavelength group."""
    pairs = []
    if 'Rest Wavelength' not in df.columns or 'Line_ID' not in df.columns:
        return pairs
    groups = {}
    for _, row in df.iterrows():
        rw = row.get('Rest Wavelength')
        try:
            rw = float(rw)
        except (TypeError, ValueError):
            rw = np.nan
        if not np.isfinite(rw):
            continue
        lid = int(row['Line_ID'])
        sigma = row.get('Sigma_0')
        try:
            sigma = float(sigma)
        except (TypeError, ValueError):
            sigma = np.inf
        groups.setdefault(round(rw, 4), []).append((lid, sigma))
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda t: (t[1] if np.isfinite(t[1]) else np.inf))
        core_id = members[0][0]
        for lid, _sigma in members[1:]:
            pairs.append((core_id, lid))
    return pairs


def ionization_group(df, line_library, threshold_eV=DEFAULT_IP_THRESHOLD_EV):
    """Classify every line as 'high' or 'low' ionization by nearest-rest-
    wavelength lookup into `line_library` (a DataFrame with 'wavelength_AA'
    and 'IP' columns, e.g. HyperCube._load_line_library()).

    Returns {Line_ID: 'high'|'low'}. Lines with no rest wavelength, or when
    the library is empty, default to 'low'.
    """
    result = {}
    if 'Rest Wavelength' not in df.columns or 'Line_ID' not in df.columns:
        return result
    have_lib = line_library is not None and len(line_library) > 0
    if have_lib:
        lib_wavs = line_library['wavelength_AA'].to_numpy(dtype=float)
        lib_ip = line_library['IP'].to_numpy(dtype=float)
    for _, row in df.iterrows():
        lid = int(row['Line_ID'])
        rw = row.get('Rest Wavelength')
        try:
            rw = float(rw)
        except (TypeError, ValueError):
            rw = np.nan
        if not have_lib or not np.isfinite(rw):
            result[lid] = 'low'
            continue
        idx = int(np.argmin(np.abs(lib_wavs - rw)))
        ip = lib_ip[idx]
        result[lid] = 'high' if (np.isfinite(ip) and ip >= threshold_eV) else 'low'
    return result


def _find_core_line_by_rest(df, core_ids, rest_wl, tol=_REST_TOL):
    """Nearest core line (by Line_ID in `core_ids`) to `rest_wl`, within
    `tol` Angstrom. Returns the df row (Series) or None."""
    if 'Rest Wavelength' not in df.columns:
        return None
    candidates = df[df['Line_ID'].isin(core_ids)]
    if candidates.empty:
        return None
    rws = candidates['Rest Wavelength'].astype(float).to_numpy()
    diffs = np.abs(rws - rest_wl)
    idx = int(np.argmin(diffs))
    if diffs[idx] > tol:
        return None
    return candidates.iloc[idx]


def _append_constraint(plan, line_name, text):
    plan.constraint_additions.setdefault(line_name, []).append(text)


def build_plan(df, scenario, *, split_ionization=False, density_mode='fix',
                balmer_mode='float', assign_kgroups=True,
                add_relational_bounds=True, set_absolute_bounds=True,
                ip_threshold=DEFAULT_IP_THRESHOLD_EV, line_library=None):
    """Build a SmartConstraintPlan for the given scenario and toggles.

    Parameters
    ----------
    df : DataFrame
        HyperCube's emission-line table (Line_ID, Line_Name, Rest Wavelength,
        Amp_0, Sigma_0, Centroid_0, region_ID, ...).
    scenario : {'quiescent', 'outflow', 'shock', 'custom'}
        Selects the absolute sigma/centroid bound windows (see
        `_BOUND_WINDOWS`). Grouping/constraint behavior itself is driven by
        the toggle arguments below, not by the scenario name, so 'custom'
        with `assign_kgroups=True` groups exactly like 'outflow'.
    density_mode : {'fix', 'float'}
        Density-sensitive doublets ([S II], [O II]): 'fix' pins the flux
        ratio to its low-density-limit value (`ratio_hi`); 'float' adds a
        bounded ratio parameter (`ratio_lo..ratio_hi`).
    balmer_mode : {'fix', 'float'}
        Balmer decrement (Halpha/Hbeta): 'float' leaves both amplitudes free
        (needed to measure reddening); 'fix' pins Halpha/Hbeta = 2.86
        (case B), useful when Hbeta is too low-S/N to fit freely.
    assign_kgroups : bool
        Core lines -> K1; secondary (broader, same rest wavelength) lines ->
        K2, or K2/K3 split by ionization potential when `split_ionization`.
    add_relational_bounds : bool
        For each core/secondary pair, add `sigma >= sigma_[core]` and
        `amp <= amp_[core]` on the secondary line (broader + fainter than
        its core, the standard outflow-wing assumption).
    set_absolute_bounds : bool
        Write Sigma_0_lowlim/highlim and Centroid_0_lowlim/highlim per the
        scenario's `_BOUND_WINDOWS`.
    ip_threshold : float
        Ionization potential (eV) splitting 'high' vs 'low' ionization
        secondary components when `split_ionization` is True.
    line_library : DataFrame or None
        Needed only when `split_ionization` is True; pass
        HyperCube._load_line_library().

    Returns
    -------
    SmartConstraintPlan
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    plan = SmartConstraintPlan()
    if df is None or len(df) == 0 or 'Line_ID' not in df.columns:
        plan.summary_lines.append('(no lines in the model)')
        return plan

    roles = classify_components(df)              # Line_ID -> 'core'/'secondary'
    core_ids = {lid for lid, role in roles.items() if role == 'core'}
    name_of = dict(zip(df['Line_ID'].astype(int), df['Line_Name'].astype(str)))

    # ── K-groups ─────────────────────────────────────────────────────────
    if assign_kgroups:
        ion_of = {}
        if split_ionization:
            ion_of = ionization_group(df, line_library, ip_threshold)
        for lid, role in roles.items():
            if role == 'core':
                group = 'K1'
            elif split_ionization:
                group = 'K2' if ion_of.get(lid, 'low') == 'high' else 'K3'
            else:
                group = 'K2'
            plan.kgroup_assignments[lid] = group
        # Summaries, grouped for readability.
        for group in ('K1', 'K2', 'K3'):
            members = sorted(name_of[lid] for lid, g in plan.kgroup_assignments.items()
                              if g == group)
            if members:
                plan.summary_lines.append(f"{group}: {', '.join(members)}")

    # ── Relational sigma/amp bounds for secondary components ───────────────
    if add_relational_bounds:
        for core_id, sec_id in _pair_core_secondary(df):
            core_name, sec_name = name_of.get(core_id), name_of.get(sec_id)
            if not core_name or not sec_name:
                continue
            _append_constraint(plan, sec_name, f'sigma >= sigma_[{core_name}]')
            _append_constraint(plan, sec_name, f'amp <= amp_[{core_name}]')
            plan.summary_lines.append(
                f"{sec_name}: sigma >= sigma_[{core_name}], amp <= amp_[{core_name}]")

    # ── Atomic-fixed doublets (always, every scenario) ─────────────────────
    for rest_faint, rest_bright, ratio in ATOMIC_FIXED_DOUBLETS:
        faint = _find_core_line_by_rest(df, core_ids, rest_faint)
        bright = _find_core_line_by_rest(df, core_ids, rest_bright)
        if faint is None or bright is None:
            continue
        faint_name, bright_name = str(faint['Line_Name']), str(bright['Line_Name'])
        text = f'flux == {ratio:.4g} * flux_[{faint_name}]'
        _append_constraint(plan, bright_name, text)
        plan.summary_lines.append(f'{bright_name}: {text}  (fixed atomic ratio)')

    # ── Density-sensitive doublets ───────────────────────────────────────
    for rest_blue, rest_red, ratio_lo, ratio_hi in DENSITY_SENSITIVE_DOUBLETS:
        blue = _find_core_line_by_rest(df, core_ids, rest_blue)
        red = _find_core_line_by_rest(df, core_ids, rest_red)
        if blue is None or red is None:
            continue
        blue_name, red_name = str(blue['Line_Name']), str(red['Line_Name'])
        if density_mode == 'float':
            text = f'flux == {ratio_lo:.4g}..{ratio_hi:.4g} * flux_[{red_name}]'
            note = 'density-sensitive, floated'
        else:
            text = f'flux == {ratio_hi:.4g} * flux_[{red_name}]'
            note = 'density-sensitive, fixed to low-density limit'
        _append_constraint(plan, blue_name, text)
        plan.summary_lines.append(f'{blue_name}: {text}  ({note})')

    # ── Balmer decrement ─────────────────────────────────────────────────
    rest_hbeta, rest_halpha, case_b_ratio = BALMER_PAIR
    hbeta = _find_core_line_by_rest(df, core_ids, rest_hbeta)
    halpha = _find_core_line_by_rest(df, core_ids, rest_halpha)
    if hbeta is not None and halpha is not None:
        hbeta_name, halpha_name = str(hbeta['Line_Name']), str(halpha['Line_Name'])
        if balmer_mode == 'fix':
            text = f'flux == {case_b_ratio:.4g} * flux_[{hbeta_name}]'
            _append_constraint(plan, halpha_name, text)
            plan.summary_lines.append(f'{halpha_name}: {text}  (Balmer decrement fixed, case B)')
        else:
            plan.summary_lines.append(
                f'{halpha_name}/{hbeta_name}: Balmer decrement left free (reddening measurement)')

    # ── Absolute sigma / centroid bounds ────────────────────────────────
    if set_absolute_bounds:
        windows = _BOUND_WINDOWS[scenario]
        for lid, role in roles.items():
            row = df.loc[df['Line_ID'] == lid]
            if row.empty:
                continue
            row = row.iloc[0]
            centroid = row.get('Centroid_0')
            try:
                centroid = float(centroid)
            except (TypeError, ValueError):
                centroid = np.nan
            if not np.isfinite(centroid):
                continue
            sig_lo_kms, sig_hi_kms, cen_win_kms = windows[role]
            sig_lo = _sigma_kms_to_wl(sig_lo_kms, centroid)
            sig_hi = _sigma_kms_to_wl(sig_hi_kms, centroid)
            cen_win = _sigma_kms_to_wl(cen_win_kms, centroid)
            if not (np.isfinite(sig_lo) and np.isfinite(sig_hi) and np.isfinite(cen_win)):
                continue
            plan.bound_updates[lid] = {
                'Sigma_0_lowlim': sig_lo,
                'Sigma_0_highlim': sig_hi,
                'Centroid_0_lowlim': centroid - cen_win,
                'Centroid_0_highlim': centroid + cen_win,
            }
            plan.summary_lines.append(
                f"{name_of.get(lid, lid)}: sigma bounds [{sig_lo_kms:g}, {sig_hi_kms:g}] km/s, "
                f"centroid window +/-{cen_win_kms:g} km/s")

    if not plan.summary_lines:
        plan.summary_lines.append('(no changes -- no recognized lines/toggles matched)')
    return plan
