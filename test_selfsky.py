"""Unit tests for HyperCube_SelfSky (Stage 1 + 2 + 3).

Every test uses a synthetic cube with a KNOWN injected sky, so the assertions
are against truth rather than against the module's own output.
"""

import numpy as np
import pytest

import HyperCube_SelfSky as ss


def make_cube(nw=200, ny=40, nx=60, sky_amp=0.008, gal_amp=0.02, seed=0,
              gradient=0.0, noise=1e-4):
    """Cube with: flat continuum, a known sky line everywhere, a galaxy blob.

    `gradient` imposes a fractional left-to-right ramp on the sky amplitude, to
    exercise per-column mode.
    """
    rng = np.random.default_rng(seed)
    w = np.arange(nw, dtype=float)
    sky_prof = np.exp(-0.5 * ((w - 100.0) / 3.0) ** 2)
    gal_prof = np.exp(-0.5 * ((w - 60.0) / 4.0) ** 2)

    ramp = 1.0 + gradient * (np.arange(nx) / max(nx - 1, 1) - 0.5)
    cube = rng.normal(0.0, noise, size=(nw, ny, nx))
    cube += sky_amp * sky_prof[:, None, None] * ramp[None, None, :]

    yy, xx = np.mgrid[0:ny, 0:nx]
    blob = np.exp(-(((yy - ny / 2) / 4.0) ** 2 + ((xx - nx / 2) / 5.0) ** 2))
    cube += gal_amp * gal_prof[:, None, None] * blob[None, :, :]
    cube += 0.05 * blob[None, :, :]                       # galaxy continuum
    return cube, sky_amp, sky_prof, blob


# ---------------------------------------------------------------- masks

def test_faintest_fraction_selects_the_faint_end():
    cube, *_ = make_cube()
    wl = ss.white_light(cube)
    m = ss.faintest_fraction_mask(wl, 0.5)
    assert 0.4 < m.mean() < 0.6
    assert wl[m].max() <= wl[~m].min() + 1e-12


def test_off_detector_zeros_are_excluded():
    """EXACT-zero voxels must not be selected as the faintest spaxels."""
    cube, *_ = make_cube()
    cube[:, :, :10] = 0.0
    wl = ss.white_light(cube)
    live = ss._live(cube)
    m = ss.faintest_fraction_mask(wl, 0.5, live=live)
    assert not m[:, :10].any(), "dead columns leaked into the sky pool"


def test_threshold_mask_operators():
    f = np.array([[-3.0, -1.0], [1.0, 3.0]])
    assert ss.threshold_mask(f, "<", 0).tolist() == [[True, True], [False, False]]
    assert ss.threshold_mask(f, ">", 0).tolist() == [[False, False], [True, True]]
    assert ss.threshold_mask(f, "abs<", 2).tolist() == [[False, True], [True, False]]
    assert ss.threshold_mask(f, "abs>", 2).tolist() == [[True, False], [False, True]]
    with pytest.raises(ValueError):
        ss.threshold_mask(f, "~=", 0)


def test_combine_masks_is_a_plain_and():
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [True, False]])
    assert ss.combine_masks(a, b).tolist() == [[True, False], [False, False]]
    with pytest.raises(ValueError):
        ss.combine_masks()
    with pytest.raises(ValueError):
        ss.combine_masks(a, np.zeros((3, 3), bool))


# ---------------------------------------------------------------- statistic

def test_mean_is_unreachable():
    """Requirement 2.2: a mean must not be offered through any code path."""
    assert "mean" not in ss.STATISTIC_NAMES
    cube, *_ = make_cube()
    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)
    with pytest.raises(ValueError, match="deliberately unavailable"):
        ss.build_sky(cube, m, statistic="mean")


def test_median_recovers_the_injected_sky():
    cube, amp, prof, _ = make_cube()
    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)
    r = ss.build_sky(cube, m)
    peak = r.sky[100]
    assert abs(peak - amp) / amp < 0.05, f"recovered {peak:.5f} vs injected {amp}"


def test_median_is_robust_to_a_contaminated_pool():
    """The protection that lets an imperfect mask be harmless (req 2.2)."""
    cube, amp, prof, blob = make_cube()
    wl = ss.white_light(cube)
    clean = ss.faintest_fraction_mask(wl, 0.5)
    # deliberately admit the galaxy: ~30% of the pool now has line emission
    dirty = clean | (blob > 0.5)
    r_clean = ss.build_sky(cube, clean)
    r_dirty = ss.build_sky(cube, dirty)
    bias = abs(r_dirty.sky[60] - r_clean.sky[60])       # at the GALAXY line
    assert bias < 0.1 * 0.02, f"median moved by {bias:.5f} on a contaminated pool"


# ---------------------------------------------------------------- subtraction

def test_apply_sky_removes_the_line_and_keeps_the_galaxy():
    cube, amp, prof, blob = make_cube()
    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)
    out = ss.apply_sky(cube, ss.build_sky(cube, m))
    faint = ~(blob > 0.2)
    before = np.median(cube[100][faint]) - np.median(cube[150][faint])
    after = np.median(out[100][faint]) - np.median(out[150][faint])
    assert abs(after) < 0.1 * abs(before), f"sky residual {after:.5f} vs {before:.5f}"
    gal = blob > 0.8
    assert np.median(out[60][gal]) > 0.5 * 0.02, "galaxy line was eaten"


def test_apply_sky_leaves_dead_voxels_at_zero():
    """Subtracting everywhere would turn exact zeros into -sky (real data)."""
    cube, *_ = make_cube()
    cube[:, :, :10] = 0.0
    wl = ss.white_light(cube)
    m = ss.faintest_fraction_mask(wl, 0.6, live=ss._live(cube))
    out = ss.apply_sky(cube, ss.build_sky(cube, m))
    assert np.all(out[:, :, :10] == 0.0), "dead voxels became negative data"


# ---------------------------------------------------------------- floors

def test_too_small_a_pool_raises_rather_than_degrading():
    cube, *_ = make_cube()
    tiny = np.zeros(cube.shape[1:], dtype=bool)
    tiny[0, :5] = True
    with pytest.raises(ss.SkyPoolTooSmall):
        ss.build_sky(cube, tiny)


# ---------------------------------------------------------------- per-column

def test_per_column_tracks_a_spatial_gradient():
    """Global mode leaves the gradient behind; per-column removes it."""
    cube, amp, prof, blob = make_cube(gradient=0.4)
    wl = ss.white_light(cube)
    m = ss.faintest_fraction_mask(wl, 0.6)
    faint = m & ~(blob > 0.2)

    g = ss.apply_sky(cube, ss.build_sky(cube, m, mode="global"))
    p = ss.apply_sky(cube, ss.build_sky(cube, m, mode="per-column"))

    def lr_gap(c):
        nx = c.shape[2]
        left = np.median(c[100][:, :nx // 4][faint[:, :nx // 4]])
        right = np.median(c[100][:, -nx // 4:][faint[:, -nx // 4:]])
        return abs(left - right)

    assert lr_gap(p) < 0.5 * lr_gap(g), "per-column did not flatten the gradient"


def test_thin_columns_fall_back_and_say_so():
    cube, *_ = make_cube(nx=60)
    m = np.zeros(cube.shape[1:], dtype=bool)
    m[:, :30] = True                      # right-hand blocks are empty
    r = ss.build_sky(cube, m, mode="per-column", n_col_blocks=6)
    assert r.fallback_cols, "empty blocks were not reported as fallbacks"
    assert all(r.n_used[b] < ss.DEFAULT_MIN_PER_COLUMN for b in r.fallback_cols)


# ---------------------------------------------------------------- variance

def test_sky_variance_is_the_median_penalty_not_the_mean():
    """var(median) = (pi/2) var(mean) for Gaussian noise (req 2.6)."""
    rng = np.random.default_rng(1)
    nw, n = 50, 400
    block = rng.normal(0.0, 1.0, size=(nw, n))
    _, var = ss._stat_and_var(block, 50.0)
    assert abs(np.median(var) / (np.pi / 2 / n) - 1.0) < 0.25


def test_propagate_sky_variance_increases_sigma():
    cube, *_ = make_cube()
    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)
    r = ss.build_sky(cube, m)
    sig = np.full(cube.shape, 1e-4)
    out = ss.propagate_sky_variance(sig, r, cube.shape)
    assert out is not None
    assert np.all(out >= sig - 1e-15)
    assert out[100].mean() > sig[100].mean(), "no penalty added at the sky line"
    assert ss.propagate_sky_variance(None, r, cube.shape) is None


def test_result_describes_itself():
    cube, *_ = make_cube()
    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)
    r = ss.build_sky(cube, m, mask_sources=("faintest 60%", "Ha channel map"))
    d = r.describe()
    assert "median" in d and "AND" in d


# ---------------------------------------------------------------- Stage 4
# The GUI is not importable headlessly (it builds a QApplication), so these
# exercise the module-level apply/revert/provenance contract that the dialog
# drives, which is where the correctness actually lives.

def _fresh_state(mod, cube, sigma=None):
    mod.FITS_DATA = cube
    mod.ERROR_CUBE = sigma
    mod.SELFSKY_ORIG_CUBE = None
    mod.SELFSKY_ORIG_SIGMA = None
    mod.SELFSKY_RESULT = None
    mod.SELFSKY_MASK = None
    mod.SELFSKY_INFO = {}


@pytest.fixture
def hc(monkeypatch):
    """The self-sky globals + apply/revert, lifted out of HyperCube.py.

    Importing HyperCube.py needs Qt; the state machine under test does not, so
    it is exec'd standalone from the source. If this ever drifts from the real
    file the extraction fails loudly rather than testing a copy.
    """
    import types, re, io as _io
    src = _io.open("HyperCube.py", encoding="utf-8").read()
    start = src.index("# ── Self-sky subtraction ──")
    end = src.index("# ── Measurement errors ──")
    block = src[start:end]
    assert "def apply_selfsky" in block and "def revert_selfsky" in block
    mod = types.ModuleType("hc_selfsky_state")
    mod.__dict__["np"] = np
    mod.__dict__["FITS_DATA"] = None
    mod.__dict__["ERROR_CUBE"] = None
    exec(compile(block, "HyperCube.py:selfsky", "exec"), mod.__dict__)
    return mod


def test_apply_then_revert_is_bit_identical(hc):
    """SPEC Stage 4 checkpoint."""
    cube, *_ = make_cube()
    original = cube.copy()
    sigma = np.full(cube.shape, 1e-4)
    _fresh_state(hc, cube, sigma)

    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)
    r = ss.build_sky(cube, m, mask_sources=("faintest 60%",))
    hc.apply_selfsky(ss.apply_sky(cube, r),
                     ss.propagate_sky_variance(sigma, r, cube.shape),
                     r, m, {"selfsky_applied": True})

    assert hc.selfsky_active()
    assert not np.array_equal(hc.FITS_DATA, original), "apply did nothing"

    assert hc.revert_selfsky() is True
    assert np.array_equal(hc.FITS_DATA, original), "revert was not bit-identical"
    assert hc.ERROR_CUBE is sigma
    assert not hc.selfsky_active()
    assert hc.revert_selfsky() is False      # idempotent


def test_reapply_uses_the_pristine_cube_not_the_corrected_one(hc):
    """Applying twice must not subtract the sky twice."""
    cube, *_ = make_cube()
    original = cube.copy()
    _fresh_state(hc, cube)
    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)

    for _ in range(2):
        base = hc.SELFSKY_ORIG_CUBE if hc.SELFSKY_ORIG_CUBE is not None else hc.FITS_DATA
        r = ss.build_sky(base, m)
        hc.apply_selfsky(ss.apply_sky(base, r), None, r, m, {"selfsky_applied": True})

    once = ss.apply_sky(original, ss.build_sky(original, m))
    assert np.allclose(hc.FITS_DATA, once), "second apply double-subtracted"
    hc.revert_selfsky()
    assert np.array_equal(hc.FITS_DATA, original)


def test_provenance_is_false_when_inactive_and_populated_when_active(hc):
    cube, *_ = make_cube()
    _fresh_state(hc, cube)
    assert hc.selfsky_provenance() == {"selfsky_applied": False}

    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.6)
    r = ss.build_sky(cube, m, mask_sources=("faintest 60%",))
    info = {"selfsky_applied": True, "selfsky_statistic": r.statistic,
            "selfsky_mode": r.mode, "selfsky_n_mask": r.n_mask,
            "selfsky_sources": " AND ".join(r.mask_sources)}
    hc.apply_selfsky(ss.apply_sky(cube, r), None, r, m, info)
    p = hc.selfsky_provenance()
    assert p["selfsky_applied"] is True and p["selfsky_n_mask"] == r.n_mask
    assert p is not hc.SELFSKY_INFO, "provenance must be a copy, not the live dict"


def test_session_spec_round_trips_through_the_mask(hc):
    """Open item 1: the session stores the mask, and rebuilds exactly."""
    cube, *_ = make_cube()
    _fresh_state(hc, cube)
    m = ss.faintest_fraction_mask(ss.white_light(cube), 0.55)
    r = ss.build_sky(cube, m, mode="per-column", n_col_blocks=4,
                     mask_sources=("faintest 55%",))
    corrected = ss.apply_sky(cube, r)
    hc.apply_selfsky(corrected, None, r, m, {"selfsky_applied": True})

    spec = {"mask": hc.SELFSKY_MASK, "statistic": r.statistic, "mode": r.mode,
            "n_col_blocks": len(r.col_edges) - 1,
            "sources": list(r.mask_sources)}
    assert spec["mask"].nbytes < cube.nbytes / 1000, "session stored something cube-sized"

    r2 = ss.build_sky(cube, spec["mask"], statistic=spec["statistic"],
                      mode=spec["mode"], n_col_blocks=spec["n_col_blocks"],
                      mask_sources=tuple(spec["sources"]))
    assert np.array_equal(ss.apply_sky(cube, r2), corrected)


# ---------------------------------------------------------------- Stage 5

def test_channel_map_matches_the_gui_formula():
    """Numerically identical to _compute_channel_map_with_subtraction."""
    cube, *_ = make_cube()
    wav = np.arange(cube.shape[0], dtype=float)
    w0, w1 = 55.0, 65.0
    sb = [(20.0, 30.0), (150.0, 160.0)]

    got = ss.channel_map(cube, wav, w0, w1, sb)

    line_sel = (wav >= w0) & (wav <= w1)
    n_line = int(line_sel.sum())
    expect = np.nansum(cube[line_sel], axis=0)
    ests = []
    for a, b in sb:
        m = (wav >= a) & (wav <= b)
        ests.append(np.nansum(cube[m], axis=0) / int(m.sum()))
    expect = expect - np.mean(ests, axis=0) * n_line
    assert np.allclose(got, expect)


def test_channel_map_without_sidebands_is_a_band_sum():
    cube, *_ = make_cube()
    wav = np.arange(cube.shape[0], dtype=float)
    sel = (wav >= 55) & (wav <= 65)
    assert np.allclose(ss.channel_map(cube, wav, 55, 65), cube[sel].sum(axis=0))


def test_channel_map_rejects_an_empty_window_and_a_length_mismatch():
    cube, *_ = make_cube()
    wav = np.arange(cube.shape[0], dtype=float)
    with pytest.raises(ValueError, match="no channels"):
        ss.channel_map(cube, wav, 1e6, 1e6 + 1)
    with pytest.raises(ValueError, match="wavelengths"):
        ss.channel_map(cube, wav[:-1], 55, 65)


def test_channel_map_finds_the_galaxy_line():
    """A veto built on this must actually select the galaxy."""
    cube, amp, prof, blob = make_cube()
    wav = np.arange(cube.shape[0], dtype=float)
    cm = ss.channel_map(cube, wav, 52, 68, [(20, 30), (150, 160)])
    assert np.median(cm[blob > 0.8]) > 10 * np.median(cm[blob < 0.05])


def test_veto_overlapping_the_protected_region_is_refused():
    """SPEC 2.5 — the circularity guard."""
    with pytest.raises(ss.VetoWindowOverlap, match="overlaps the protected region"):
        ss.check_veto_window((6865.0, 6867.0), (6865.2, 6866.9))
    # touching at an endpoint still overlaps
    with pytest.raises(ss.VetoWindowOverlap):
        ss.check_veto_window((6860.0, 6865.2), (6865.2, 6866.9))
    # fully containing it
    with pytest.raises(ss.VetoWindowOverlap):
        ss.check_veto_window((6800.0, 6900.0), (6865.2, 6866.9))


def test_veto_clear_of_the_protected_region_is_allowed():
    ss.check_veto_window((6875.0, 6885.0), (6865.2, 6866.9))   # Ha, redward
    ss.check_veto_window((6850.0, 6860.0), (6865.2, 6866.9))   # blueward
    ss.check_veto_window((6865.0, 6867.0), None)               # check disabled
    # argument order must not matter
    ss.check_veto_window((6885.0, 6875.0), (6866.9, 6865.2))


def test_two_vetoes_and_correctly_and_the_count_is_exact():
    """Stage 5 checkpoint: stacked vetoes, hand-checkable survivor count."""
    cube, amp, prof, blob = make_cube()
    wav = np.arange(cube.shape[0], dtype=float)
    wl = ss.white_light(cube)
    live = ss._live(cube)

    v_cont = ss.faintest_fraction_mask(wl, 0.60, live=live)
    cm = ss.channel_map(cube, wav, 52, 68, [(20, 30), (150, 160)])
    v_line = ss.threshold_mask(cm, "<", float(np.percentile(cm[live], 70)), live=live)

    both = ss.combine_masks(v_cont, v_line)
    assert np.array_equal(both, v_cont & v_line)
    assert int(both.sum()) == int((v_cont & v_line).sum())
    assert both.sum() <= min(v_cont.sum(), v_line.sum())
    # the line veto must actually remove something the continuum veto kept
    assert (v_cont & ~v_line).sum() > 0, "line veto removed nothing the continuum kept"

    r = ss.build_sky(cube, both, mask_sources=("faintest 60%", "line channel map"))
    assert r.n_mask == int(both.sum())
    assert "AND" in r.describe()
