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

# How broad components are placed relative to their core:
#   'free'  -- no side constraint (a broad component may sit either side)
#   'split' -- each broad component is pinned to the blue or the red side,
#              identified from its own initial centroid guess.
WING_SIDE_MODES = ('free', 'split')

# What to do with a line that has only ONE broad component, where "blue vs red"
# cannot be read off a pair. 'free' leaves it unconstrained.
LONE_WING_SIDES = ('free', 'blue', 'red')

# Velocity separation (km/s) below which a broad component's initial guess is
# treated as sitting *on* the core rather than to one side of it. Guesses this
# close carry no information about which side the user meant.
WING_SIDE_TOL_KMS = 5.0

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

# Per-tier bounds, the defaults the dialog's table starts from.
#   'dv_kms'      -- tier 1: |v - v_systemic| limit for the primary component.
#                    tier >= 2: |delta v| limit from its own primary counterpart.
#   'sigma_ratio' -- (lo, hi) on sigma / sigma_primary. Non-primary only.
#   'amp_ratio'   -- (lo, hi) on amp / amp_primary. Non-primary only.
# (1.0, inf) and (0.0, 1.0) are the historical relational bounds written long-
# hand -- "broader than its core" and "fainter than its core" -- now expressed
# as ranges the user can tighten. `None` for a bound means unbounded on that side.
DEFAULT_SIGMA_RATIO = (1.0, np.inf)
DEFAULT_AMP_RATIO = (0.0, 1.0)


def default_tier_bounds(scenario, tiers=(1, 2, 3)):
    """The per-tier bound table a scenario starts from: velocity windows from
    `_BOUND_WINDOWS`, sigma/amp ratios from the historical relational bounds."""
    windows = _BOUND_WINDOWS[scenario]
    out = {}
    for tier in tiers:
        role = 'core' if tier == 1 else 'secondary'
        out[int(tier)] = {
            'dv_kms': windows[role][2],
            'sigma_ratio': DEFAULT_SIGMA_RATIO,
            'amp_ratio': DEFAULT_AMP_RATIO,
        }
    return out


def _ratio_text(lo, hi):
    """'lo..hi' as the constraint parser reads it. None (or an unreadable
    value) means open on that side: 0 below, inf above."""
    def _num(v, fallback):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return fallback
        return v if not np.isnan(v) else fallback
    return f'{_num(lo, 0.0):g}..{_num(hi, np.inf):g}'


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


def tier_label(tier):
    """'Primary' / 'Secondary' / 'Tertiary' / '4th component' / ... for tier >= 1."""
    return {1: 'Primary', 2: 'Secondary', 3: 'Tertiary'}.get(int(tier),
                                                             f'{int(tier)}th component')


def classify_tiers(df):
    """Rank every line within its rest-wavelength group by width: 1 = primary
    (narrowest, the "core"), 2 = secondary, 3 = tertiary, ...

    The finer-grained form of `classify_components`, which collapses everything
    above 1 into 'secondary'. A singleton line is tier 1.

    Returns {Line_ID: tier}.
    """
    tiers = {}
    for members in _rest_wavelength_groups(df).values():
        ordered = sorted(members, key=lambda t: (t[1] if np.isfinite(t[1]) else np.inf))
        for rank, (lid, _sigma, _cen) in enumerate(ordered, start=1):
            tiers[lid] = rank
    return tiers


def tiers_present(df):
    """Sorted list of the component tiers this model actually uses, e.g.
    [1, 2, 3] for a model where some line is fitted with three components."""
    return sorted(set(classify_tiers(df).values()))


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


def _rest_wavelength_groups(df):
    """{rounded rest wavelength: [(Line_ID, Sigma_0, Centroid_0), ...]} in df
    (model) order. The shared grouping primitive behind `classify_components`,
    `_pair_core_secondary` and `classify_wing_sides`."""
    groups = {}
    if 'Rest Wavelength' not in df.columns or 'Line_ID' not in df.columns:
        return groups
    for _, row in df.iterrows():
        try:
            rw = float(row.get('Rest Wavelength'))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rw):
            continue
        try:
            sigma = float(row.get('Sigma_0'))
        except (TypeError, ValueError):
            sigma = np.inf
        try:
            centroid = float(row.get('Centroid_0'))
        except (TypeError, ValueError):
            centroid = np.nan
        groups.setdefault(round(rw, 4), []).append(
            (int(row['Line_ID']), sigma, centroid))
    return groups


def classify_wing_sides(df, lone_wing_side='free'):
    """Decide which side of its core each broad component sits on.

    Within every rest-wavelength group the narrowest member is the core and the
    rest are broad components ("wings"). A wing's side is read off its *initial
    centroid guess* relative to the core's: bluer than the core by more than
    `WING_SIDE_TOL_KMS` is 'blue', redder is 'red'. The guess is authoritative --
    two wings the user seeded on the same side stay on that side rather than
    being forced into an artificial blue/red pair.

    Ambiguity is resolved as follows, and every resolution is reported:
      * one wing, guess uninformative -> `lone_wing_side` ('free' = no side);
      * two wings, both guesses uninformative -> first in model order is
        assigned 'blue' and the second 'red'. This is a label, not a
        measurement, but it is what stops two identically-seeded wings from
        chasing each other into the same solution;
      * two wings, one informative -> the other takes the opposite side;
      * three or more wings with uninformative guesses -> left free.

    Returns (sides, notes): sides maps Line_ID -> 'blue'|'red' (wings only,
    and only those that got a side); notes is a list of human-readable strings
    for the plan summary.
    """
    sides, notes = {}, []
    name_of = {}
    if 'Line_Name' in df.columns and 'Line_ID' in df.columns:
        name_of = dict(zip(df['Line_ID'].astype(int), df['Line_Name'].astype(str)))

    def label(lid):
        return name_of.get(lid, f'Line {lid}')

    for members in _rest_wavelength_groups(df).values():
        if len(members) < 2:
            continue
        # Narrowest = core; the wings keep model order, which is the tie-break.
        ordered = sorted(members, key=lambda t: (t[1] if np.isfinite(t[1]) else np.inf))
        core_id, _core_sigma, core_cen = ordered[0]
        wings = [m for m in members if m[0] != core_id]

        # Δv of each wing's initial guess relative to the core's, km/s.
        deltas = {}
        for lid, _sigma, cen in wings:
            if np.isfinite(cen) and np.isfinite(core_cen) and core_cen > 0:
                deltas[lid] = C_KMS * (cen - core_cen) / core_cen
            else:
                deltas[lid] = np.nan

        def side_of(lid):
            dv = deltas[lid]
            if not np.isfinite(dv) or abs(dv) <= WING_SIDE_TOL_KMS:
                return None
            return 'blue' if dv < 0 else 'red'

        guessed = {lid: side_of(lid) for lid, _s, _c in wings}
        known = [lid for lid, s in guessed.items() if s]
        unknown = [lid for lid, s in guessed.items() if not s]

        for lid in known:
            sides[lid] = guessed[lid]
            notes.append(f"{label(lid)}: {guessed[lid]} wing "
                         f"({deltas[lid]:+.0f} km/s from {label(core_id)} in the initial guess)")
        if len(known) == len(wings) >= 2 and len(set(guessed.values())) == 1:
            notes.append(f"  note: both broad components of {label(core_id)} were seeded on the "
                         f"{guessed[known[0]]} side -- kept there rather than split blue/red")

        if not unknown:
            continue
        if len(wings) == 1:
            lid = unknown[0]
            if lone_wing_side in ('blue', 'red'):
                sides[lid] = lone_wing_side
                notes.append(f"{label(lid)}: {lone_wing_side} wing (single broad component, "
                             f"side set by the scenario)")
            continue
        if len(wings) == 2 and len(known) == 1:
            lid, taken = unknown[0], guessed[known[0]]
            sides[lid] = 'red' if taken == 'blue' else 'blue'
            notes.append(f"{label(lid)}: {sides[lid]} wing (opposite the {taken} component; "
                         f"its own guess sits on the core)")
            continue
        if len(wings) == 2 and not known:
            first, second = unknown[0], unknown[1]
            sides[first], sides[second] = 'blue', 'red'
            notes.append(f"{label(first)}/{label(second)}: seeded at the same velocity, so "
                         f"blue/red assigned by model order -- check this is the intended pairing")
            continue
        notes.append(f"{label(core_id)}: {len(unknown)} broad components with no distinguishing "
                     f"initial velocity -- left free (seed their centroids apart to pin sides)")
    return sides, notes


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


def _systemic_wavelength(row, redshift):
    """Where this line sits at the systemic redshift: rest * (1 + z), in A.
    None when either the rest wavelength or the redshift is unusable, which is
    the caller's cue to fall back to the line's own initial guess."""
    try:
        rest = float(row.get('Rest Wavelength'))
        z = float(redshift)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(rest) and rest > 0 and np.isfinite(z) and z > -1):
        return None
    return rest * (1.0 + z)


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


def velocity_window_kms(scenario, velocity_offset_kms=None):
    """How far (km/s) a broad component may sit from its core.

    `velocity_offset_kms` is the user's own number; anything missing or
    non-positive falls back to the scenario's secondary centroid window, which
    is also what the dialog shows as the default for each scenario.
    """
    try:
        value = float(velocity_offset_kms)
    except (TypeError, ValueError):
        value = np.nan
    if np.isfinite(value) and value > 0:
        return value
    return _BOUND_WINDOWS[scenario]['secondary'][2]


def _kgroup_labels(roles, sides, ion_of, split_ionization):
    """Map every Line_ID to a K-group label ('K1'..'K5').

    Lines that must share a velocity go in the same group; lines that must not
    go in different ones. Cores are always K1. Wings are keyed by (side,
    ionization) and the distinct keys are handed out in a fixed order, so
    turning an option off reproduces the previous labelling exactly:

        sides off, ionization off -> K1 core, K2 wings
        sides off, ionization on  -> K1 core, K2 high-ion wings, K3 low-ion
        sides on,  ionization off -> K1 core, K2 blue, K3 red, K4 unsided
        sides on,  ionization on  -> K1 core, K2..K5 blue/red x high/low

    Returns (assignments, overflow_keys). More than five distinct keys cannot be
    represented (HyperCube has K1..K5), so the tail shares K5 -- reported, never
    silent, because sharing a group means sharing a velocity.
    """
    keys = {}
    for lid, role in roles.items():
        if role == 'core':
            keys[lid] = ('0core', '', '')
        else:
            side = sides.get(lid, '')
            ion = ion_of.get(lid, 'low') if split_ionization else ''
            keys[lid] = ('1wing',
                         {'blue': '0blue', 'red': '1red', '': '2free'}[side],
                         {'high': '0high', 'low': '1low', '': ''}[ion])
    order = sorted(set(keys.values()))
    labels = {key: f'K{i + 1}' for i, key in enumerate(order[:5])}
    overflow = order[5:]
    for key in overflow:
        labels[key] = 'K5'
    return {lid: labels[key] for lid, key in keys.items()}, overflow


def build_plan(df, scenario, *, split_ionization=False, density_mode='fix',
                balmer_mode='float', assign_kgroups=True,
                add_relational_bounds=True, set_absolute_bounds=True,
                wing_sides='free', lone_wing_side='free',
                limit_velocity_offset=False, velocity_offset_kms=None,
                tier_bounds=None, redshift=None,
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
    wing_sides : {'free', 'split'}
        'split' keeps every broad component on the blue or the red side of its
        core (`vel == vel_[core] -W..0` / `0..W`), with the side read from the
        initial guesses by `classify_wing_sides`. Blue and red wings are also
        placed in separate K-groups, since a shared K-group would tie them to
        one velocity.
    lone_wing_side : {'free', 'blue', 'red'}
        Side for a line carrying only ONE broad component, where no blue/red
        pair exists to read a side from. Only consulted when
        `wing_sides='split'`.
    limit_velocity_offset : bool
        Bound every broad component's velocity offset from its core even when
        the side is not being pinned (`vel == vel_[core] +- W`). Redundant while
        `wing_sides='split'` supplies a side for that line -- the sided form
        already carries the same W -- so only one constraint is ever written.
    velocity_offset_kms : float or None
        W, the largest allowed |delta v| between a broad component and its core,
        for any tier `tier_bounds` does not name. None (or a non-positive value)
        means the scenario's own window: quiescent 300, outflow 1500, shock 2000.
    tier_bounds : dict or None
        Per-tier overrides, {tier: {'dv_kms', 'sigma_ratio', 'amp_ratio'}} as
        built by `default_tier_bounds` -- tier 1 = primary, 2 = secondary, 3 =
        tertiary, by width within each rest-wavelength group (`classify_tiers`).
        One entry sets every line of that tier at once, which is the point:
        "all secondary components, this wide, this bright, this far out".
        For tier 1, 'dv_kms' is measured from SYSTEMIC (needs `redshift`); for
        the rest it is measured from that line's own primary counterpart.
        None keeps the scenario defaults for every tier.
    redshift : float or None
        Systemic redshift, used only to anchor the primary tier's velocity
        window on rest*(1+z). Without it the primary window falls back to the
        historical anchor, each line's own Centroid_0 initial guess.
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
    if wing_sides not in WING_SIDE_MODES:
        raise ValueError(f"Unknown wing_sides {wing_sides!r}; expected one of {WING_SIDE_MODES}")
    if lone_wing_side not in LONE_WING_SIDES:
        raise ValueError(f"Unknown lone_wing_side {lone_wing_side!r}; "
                         f"expected one of {LONE_WING_SIDES}")
    plan = SmartConstraintPlan()
    if df is None or len(df) == 0 or 'Line_ID' not in df.columns:
        plan.summary_lines.append('(no lines in the model)')
        return plan

    roles = classify_components(df)              # Line_ID -> 'core'/'secondary'
    core_ids = {lid for lid, role in roles.items() if role == 'core'}
    name_of = dict(zip(df['Line_ID'].astype(int), df['Line_Name'].astype(str)))
    tiers = classify_tiers(df)                   # Line_ID -> 1 (primary), 2, 3, ...

    def bounds_for(tier):
        """This tier's entry from the user's table, falling back to the
        scenario defaults for anything it does not specify."""
        entry = (tier_bounds or {}).get(int(tier), {})
        fallback = {'dv_kms': velocity_window_kms(scenario, velocity_offset_kms)
                              if tier != 1 else _BOUND_WINDOWS[scenario]['core'][2],
                    'sigma_ratio': DEFAULT_SIGMA_RATIO,
                    'amp_ratio': DEFAULT_AMP_RATIO}
        fallback.update({k: v for k, v in entry.items() if v is not None})
        return fallback

    # ── Blue / red side of each broad component ──────────────────────────
    sides, side_notes = {}, []
    if wing_sides == 'split':
        sides, side_notes = classify_wing_sides(df, lone_wing_side=lone_wing_side)

    # ── K-groups ─────────────────────────────────────────────────────────
    if assign_kgroups:
        ion_of = ionization_group(df, line_library, ip_threshold) if split_ionization else {}
        plan.kgroup_assignments, overflow = _kgroup_labels(
            roles, sides, ion_of, split_ionization)
        # Summaries, grouped for readability.
        for group in ('K1', 'K2', 'K3', 'K4', 'K5'):
            members = sorted(name_of[lid] for lid, g in plan.kgroup_assignments.items()
                              if g == group)
            if members:
                plan.summary_lines.append(f"{group}: {', '.join(members)}")
        if overflow:
            plan.summary_lines.append(
                f"! {len(overflow) + 1} kinematic populations share K5 -- HyperCube has only "
                f"K1-K5, so these are tied to one velocity. Turn off the ionization split "
                f"or the blue/red split to separate them.")

    # ── Velocity offset of each broad component from its core ────────────
    # One constraint carries both ideas: how FAR a wing may sit from the core
    # (the window) and, when the side is known, WHICH WAY (the sign). Emitting
    # only one keeps them from overwriting each other -- two velocity relations
    # on the same line would leave the first one's helper parameter orphaned.
    if wing_sides == 'split' or limit_velocity_offset:
        sided_any = False
        for core_id, sec_id in _pair_core_secondary(df):
            side = sides.get(sec_id) if wing_sides == 'split' else None
            core_name, sec_name = name_of.get(core_id), name_of.get(sec_id)
            if not core_name or not sec_name:
                continue
            # Each component is bounded by ITS OWN tier's number, so a tertiary
            # can be allowed further out than a secondary (or reined in).
            tier = tiers.get(sec_id, 2)
            window_kms = float(bounds_for(tier)['dv_kms'])
            if side == 'blue':
                interval, how = f'-{window_kms:g}..0', f'blueward of {core_name}'
            elif side == 'red':
                interval, how = f'0..{window_kms:g}', f'redward of {core_name}'
            elif limit_velocity_offset:
                interval, how = f'+- {window_kms:g}', f'within {window_kms:g} km/s of {core_name}'
            else:
                continue                      # sides-only mode, side unknown
            sided_any = True
            _append_constraint(plan, sec_name, f'vel == vel_[{core_name}] {interval}')
            plan.summary_lines.append(
                f"{sec_name} [{tier_label(tier).lower()}]: {how} "
                f"(delta v limit {window_kms:g} km/s)")
        plan.summary_lines.extend(side_notes)
        if sided_any and set_absolute_bounds:
            # Worth saying plainly: a velocity relation replaces the centroid
            # with an expression, and lmfit ignores min/max on an expression, so
            # the absolute centroid window written below is inert for these lines.
            plan.summary_lines.append(
                '  note: for the lines above, the velocity limit REPLACES the absolute '
                'centroid window (a constrained centroid is an expression, and bounds '
                'do not apply to expressions).')
        if wing_sides == 'split' and not sides:
            plan.summary_lines.append(
                'Blue/red split requested, but no broad component could be sided '
                '(single-component lines only, or no initial velocity separation).')

    # ── Relational sigma/amp bounds for non-primary components ─────────────
    # Ranges, not one-sided inequalities: the historical `sigma >= sigma_[core]`
    # and `amp <= amp_[core]` are exactly the (1, inf) and (0, 1) defaults, so
    # leaving the table alone reproduces them, while narrowing a cell says
    # "every secondary is 2-4x the core's width" in one place.
    if add_relational_bounds:
        for core_id, sec_id in _pair_core_secondary(df):
            core_name, sec_name = name_of.get(core_id), name_of.get(sec_id)
            if not core_name or not sec_name:
                continue
            tier = tiers.get(sec_id, 2)
            b = bounds_for(tier)
            sig = _ratio_text(*b['sigma_ratio'])
            amp = _ratio_text(*b['amp_ratio'])
            _append_constraint(plan, sec_name, f'sigma == {sig} * sigma_[{core_name}]')
            _append_constraint(plan, sec_name, f'amp == {amp} * amp_[{core_name}]')
            plan.summary_lines.append(
                f"{sec_name} [{tier_label(tier).lower()}]: sigma = {sig} x {core_name}'s, "
                f"amp = {amp} x {core_name}'s")

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
            tier = tiers.get(lid, 1)
            sig_lo_kms, sig_hi_kms, _default_win = windows[role]
            cen_win_kms = float(bounds_for(tier)['dv_kms'])
            sig_lo = _sigma_kms_to_wl(sig_lo_kms, centroid)
            sig_hi = _sigma_kms_to_wl(sig_hi_kms, centroid)
            cen_win = _sigma_kms_to_wl(cen_win_kms, centroid)
            if not (np.isfinite(sig_lo) and np.isfinite(sig_hi) and np.isfinite(cen_win)):
                continue
            # The primary's velocity window is the physically meaningful one to
            # anchor on SYSTEMIC: "this line's core is within N km/s of the
            # galaxy's redshift". Every other anchor (its own initial guess)
            # just re-centres the window on whatever the user happened to type.
            anchor, anchor_note = centroid, 'its initial guess'
            if tier == 1:
                systemic = _systemic_wavelength(row, redshift)
                if systemic is not None:
                    anchor, anchor_note = systemic, 'systemic'
                    cen_win = _sigma_kms_to_wl(cen_win_kms, systemic)
            plan.bound_updates[lid] = {
                'Sigma_0_lowlim': sig_lo,
                'Sigma_0_highlim': sig_hi,
                'Centroid_0_lowlim': anchor - cen_win,
                'Centroid_0_highlim': anchor + cen_win,
            }
            plan.summary_lines.append(
                f"{name_of.get(lid, lid)} [{tier_label(tier).lower()}]: sigma bounds "
                f"[{sig_lo_kms:g}, {sig_hi_kms:g}] km/s, velocity window "
                f"+/-{cen_win_kms:g} km/s about {anchor_note}")

    if not plan.summary_lines:
        plan.summary_lines.append('(no changes -- no recognized lines/toggles matched)')
    return plan
