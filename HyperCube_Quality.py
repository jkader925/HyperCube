"""Fit-quality criteria: is this spaxel's fit good, and how good is it?

Qt-free by design: imports only numpy + pandas, so the GUI, the batch fit kernel
(`HyperCube_fit`) and multiprocessing workers can all share one definition of
"good fit". Nothing here touches a cube, a model or lmfit — it reads the quality
columns a completed fit already wrote.

Two questions need answering, and Rectify used to answer both with a single
hardcoded column (`qa_core_cont_ratio`) at a single threshold:

  1. **Is this fit good enough?** — a yes/no, used to decide which spaxels need
     repair and, more importantly, which neighbouring spaxels are trustworthy
     enough to seed a repair from.
  2. **Is this fit better than that one?** — an ordering, used to rank candidate
     donors and to decide whether a re-fit actually improved anything.

Both come from one quantity here. Every criterion normalises its metric to a
dimensionless *badness* `n`, scaled so that `n = 1` sits exactly on the
threshold:

    signed metric (ideal 0, two-sided)   n = |v| / thr
    plain metric  (ideal 1 or 0, upper)  n = max(0, v - ideal) / (thr - ideal)
    non-finite v (a failed/degenerate fit)   n = +inf

and then, over the enabled criteria,

    is_good(row) == max(n_i) <= 1      # AND-gate: every metric must pass
    score(row)   == mean(n_i)          # ranking; lower is better, 0 is perfect

The AND-gate is deliberate: a severe failure in one metric must not be outvoted
by several healthy ones, because the whole point of the gate is to keep a
visibly-broken fit from being used as a seed. The mean is only ever used to
order fits that have already passed (or to compare candidates for the same
spaxel), where the extra sensitivity of an average is what you want.

`ideal` matters for discrimination, not for the pass/fail boundary: a
core/continuum ratio of 1.0 is a *perfect* fit while 0.5 is merely an
over-estimated continuum, so anchoring at 1 and clipping below keeps both at
n = 0 rather than rewarding one over the other.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


# Value each metric takes for a perfect fit. Ratios and reduced-χ² statistics
# are built to sit at 1; z-scores at 0. Anything not listed (the fitted-parameter
# maps — amplitude, velocity, σ — that the Rectify dialog can also flag on) is
# treated as "0 is ideal, larger is worse", which is the right reading for the
# |·| operators those maps default to.
QA_IDEAL = {
    'qa_core_cont_ratio': 1.0,
    'qa_chisq_cont':      1.0,
    'qa_chisq_core':      1.0,
    'rchisq_w':           1.0,
    # lmfit's native reduced χ² divides by the FULL spectrum length rather than
    # the fitted pixels, so it reads low by roughly the fraction of the spectrum
    # the fit windows cover and does NOT sit at 1 for a good fit. Anchor it at 0
    # and let its (much larger) threshold do the work — it stays usable as an
    # ordering, but must not be read as "distance from a perfect fit".
    'rchisq':             0.0,
    'qa_signed_resid_z':  0.0,
    'qa_runs_z':          0.0,
}

# Operators, in the legacy sense: each says when a fit is BAD.
#   '>'    bad when v > limit          '<'    bad when v < limit
#   'abs>' bad when |v| > limit        'abs<' bad when |v| < limit
OPS = ('>', '<', 'abs>', 'abs<')
OP_LABELS = {'>': '>', '<': '<', 'abs>': '|·| >', 'abs<': '|·| <'}

# Metrics offered as goodness criteria, in the order the dialog lists them:
# (column, label, signed, default operator, default limit, enabled by default).
#
# Only the core/continuum ratio is enabled by default, so an untouched dialog
# reproduces the historical single-metric behaviour exactly; the limits match
# HyperCube.FitParamsWindow._RECTIFY_DEFAULTS.
DEFAULT_CRITERIA_SPEC = (
    ('qa_core_cont_ratio', 'Core / continuum ratio', False, '>',    2.0, True),
    ('qa_signed_resid_z',  'Signed residual (z)',    True,  'abs>', 3.0, False),
    ('qa_runs_z',          'Runs test (z)',          True,  'abs>', 3.0, False),
    ('qa_chisq_cont',      'Reduced χ² (continuum)', False, '>',    5.0, False),
    ('rchisq_w',           'Reduced χ² (weighted)',  False, '>',    3.0, False),
    ('rchisq',             'Reduced χ² (native)',    False, '>',    5.0, False),
)


@dataclass
class Criterion:
    """One metric's contribution to the goodness rule.

    column    : df_fit column holding the metric
    label     : human-readable name (dialog + summaries)
    signed    : whether the metric is naturally two-sided (a z-score). Only sets
                the default operator; `op` is what actually applies.
    threshold : the pass/fail boundary, a magnitude for the |·| operators and a
                signed number for '<' / '>'
    enabled   : whether this criterion takes part at all
    op        : when the fit is BAD — see OPS
    """
    column: str
    label: str
    signed: bool
    threshold: float
    enabled: bool = False
    op: str = ''          # blank -> derived from `signed` (see __post_init__)

    def __post_init__(self):
        # A two-sided metric with no operator stated means both wings; without
        # this, constructing Criterion(..., signed=True) and forgetting `op`
        # would silently give a one-sided rule that ignores the negative wing.
        if not self.op:
            self.op = 'abs>' if self.signed else '>'

    @property
    def ideal(self):
        return QA_IDEAL.get(self.column, 0.0)

    def describe_rule(self):
        return f'{self.label} {OP_LABELS.get(self.op, self.op)} {self.threshold:g}'


def default_criteria():
    """A fresh criteria list at the built-in defaults."""
    return [Criterion(col, lbl, sgn, thr, en, op)
            for col, lbl, sgn, op, thr, en in DEFAULT_CRITERIA_SPEC]


def enabled_criteria(criteria):
    """Only the criteria that are switched on and usably configured."""
    out = []
    for c in (criteria or ()):
        try:
            thr = float(c.threshold)
        except (TypeError, ValueError):
            continue
        if not (c.enabled and np.isfinite(thr)):
            continue
        # A zero limit divides by zero in every form, so it says nothing.
        # The |·| forms additionally need a positive magnitude; the single-sided
        # forms accept a negative limit, which is the point of them
        # (e.g. "bad when vel < -300").
        op = getattr(c, 'op', '>') or '>'
        if thr == 0 or (op in ('abs>', 'abs<') and thr < 0):
            continue
        out.append(replace(c, threshold=thr))
    return out


def normalised(value, criterion):
    """One metric's badness `n`, scaled so n = 1 is exactly on the limit.

    n <= 1 passes, n > 1 fails, +inf for a value that is not a finite number
    (a failed or degenerate fit, which must never be treated as a good donor).

    The '>' and '|·| >' forms are anchored on the metric's ideal, so a perfect
    fit scores 0 and the score discriminates usefully among good fits. The
    reversed forms ('<', '|·| <') have no meaningful ideal — they are veto
    rules, e.g. "amplitude below this is a dead fit" — so they are measured as a
    fractional shortfall against the limit itself. All four cross n = 1 exactly
    at the limit, continuously, so the gate and the ranking stay consistent.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.inf
    if not np.isfinite(v):
        return np.inf
    thr = float(criterion.threshold)
    op = getattr(criterion, 'op', '>') or '>'
    eps = 1e-30

    if op == 'abs>':
        return abs(v) / abs(thr) if thr else np.inf
    if op == 'abs<':
        return max(0.0, 1.0 + (abs(thr) - abs(v)) / max(abs(thr), eps))
    if op == '<':
        return max(0.0, 1.0 + (thr - v) / max(abs(thr), eps))

    # '>' — one-sided upper, anchored on the ideal where one is defined.
    ideal = criterion.ideal
    span = thr - ideal
    if span <= 0:
        # Limit at or below the ideal (e.g. a negative limit): fall back to a
        # plain fractional measure rather than declaring everything bad.
        return max(0.0, v) / max(abs(thr), eps)
    return max(0.0, v - ideal) / span


def badness(row, criteria):
    """Every enabled criterion's `n` for one spaxel's fit row.

    `row` is anything supporting .get(column) — a dict or a pandas Series.
    Returns [] when nothing is enabled, which callers must read as "no opinion"
    rather than as "good" or "bad".
    """
    return [normalised(row.get(c.column), c) for c in enabled_criteria(criteria)]


def is_good(row, criteria):
    """True when EVERY enabled criterion passes (n <= 1).

    This is the **donor gate**, and it is deliberately always the strict AND:
    a spaxel is only trusted to seed a repair, or accepted as an improvement,
    when nothing about it is objectionable. It is NOT affected by the `mode`
    used for flagging (see `is_bad`) — loosening which spaxels get repaired
    must not loosen which spaxels are allowed to do the repairing.

    With no criteria enabled nothing can be judged, so nothing is rejected:
    the caller falls back to whatever it does when it has no quality opinion.
    """
    ns = badness(row, criteria)
    if not ns:
        return True
    return max(ns) <= 1.0


# How the ticked rules combine when deciding a spaxel needs repair.
FLAG_ANY = 'any'    # bad if ANY rule trips  (the complement of is_good)
FLAG_ALL = 'all'    # bad only if EVERY rule trips


def is_bad(row, criteria, mode=FLAG_ANY):
    """Whether this spaxel should be re-fit.

    `mode=FLAG_ANY` is the complement of `is_good`: one tripped rule is enough.
    `mode=FLAG_ALL` requires every ticked rule to trip, which narrows the
    selection to spaxels that are unambiguously broken on all counts.

    Note that FLAG_ALL leaves a middle band — spaxels failing some rules but not
    all — which are neither repaired nor usable as donors (`is_good` still
    demands they pass everything). That is intentional: "only repair the
    thoroughly broken ones, but still only trust the clean ones as seeds".
    """
    ns = badness(row, criteria)
    if not ns:
        return False                     # no opinion -> nothing to repair
    if mode == FLAG_ALL:
        return all(n > 1.0 for n in ns)
    return max(ns) > 1.0


def score(row, criteria):
    """Mean badness over the enabled criteria; lower is better, 0 is perfect.

    +inf when any metric is non-finite, so a failed fit sorts last and can never
    beat a real one. 0.0 when nothing is enabled (no opinion, all fits tie).
    """
    ns = badness(row, criteria)
    if not ns:
        return 0.0
    if any(not np.isfinite(n) for n in ns):
        return np.inf
    return float(np.mean(ns))


def describe(row, criteria):
    """'core/cont 1.42 (n=0.42), runs z 4.10 (n=1.37 FAIL)' — for logs and
    tooltips, so a rejected donor can be explained rather than just refused."""
    parts = []
    for c in enabled_criteria(criteria):
        v = row.get(c.column)
        n = normalised(v, c)
        try:
            vtxt = f'{float(v):.3g}'
        except (TypeError, ValueError):
            vtxt = str(v)
        parts.append(f'{c.label} {vtxt} (n={n:.2f}{"" if n <= 1 else " FAIL"})')
    return ', '.join(parts) if parts else '(no criteria enabled)'


def score_map(df_fit, criteria, xcol='spaxel_x', ycol='spaxel_y'):
    """{(x, y): (good, score)} for every spaxel in a fit table.

    df_fit carries one row per (spaxel, line) with the per-spaxel quality columns
    repeated across a spaxel's line rows, so the frame is de-duplicated first —
    the same `drop_duplicates(subset=['spaxel_x', 'spaxel_y'])` the Rectify loop
    and `_rectify_available_maps` already use.
    """
    out = {}
    if not isinstance(df_fit, pd.DataFrame) or len(df_fit) == 0:
        return out
    if xcol not in df_fit.columns or ycol not in df_fit.columns:
        return out
    crits = enabled_criteria(criteria)
    per = df_fit.drop_duplicates(subset=[xcol, ycol])
    for _, row in per.iterrows():
        try:
            key = (int(row[xcol]), int(row[ycol]))
        except (TypeError, ValueError):
            continue
        out[key] = (is_good(row, crits), score(row, crits))
    return out


def summarise(criteria, mode=FLAG_ANY):
    """'bad when core/cont > 2 OR |runs z| > 3' — the active rule in one line."""
    crits = enabled_criteria(criteria)
    if not crits:
        return '(no criteria enabled)'
    join = ' AND ' if mode == FLAG_ALL else ' OR '
    return 'bad when ' + join.join(c.describe_rule() for c in crits)
