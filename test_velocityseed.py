"""Unit tests for HyperCube_VelocitySeed and the seed plumbing in HyperCube_fit.

Every test uses a synthetic rotating cube with KNOWN per-spaxel velocities, so
the assertions are against truth rather than against the module's own output.
"""

import numpy as np
import pandas as pd
import pytest

import HyperCube_fit as HF
import HyperCube_VelocitySeed as vs

C = HF.C_KMS
Z = 0.03
REST = {'[N II]_6548': 6548.05, 'H_alpha_6563': 6562.819, '[N II]_6583': 6583.46,
        '[S II]_6716': 6716.44, '[S II]_6731': 6730.81}
RATIO = {'[N II]_6548': 0.13, 'H_alpha_6563': 1.0, '[N II]_6583': 0.4,
         '[S II]_6716': 0.25, '[S II]_6731': 0.2}
SIG_KMS = 110.0
WIN = (6630.0, 7060.0)


def make_cube(ny=16, nx=24, vamp=300.0, peak=1.0, noise=0.02, seed=0, ratio=None):
    """Rotating H-alpha/[N II]/[S II] field on a sloped continuum.

    v(x) = vamp * tanh((x - cx) / 3) and the line peak falls off from the
    centre, so the edges are both the most shifted and the faintest spaxels —
    the regime the seeds are for.
    """
    ratio = ratio or RATIO
    rng = np.random.default_rng(seed)
    wl = np.arange(6600.0, 7100.0, 1.25)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    vel = vamp * np.tanh((xx - cx) / 3.0)
    bright = peak * np.exp(-(((xx - cx) / 10.0) ** 2 + ((yy - cy) / 8.0) ** 2))
    cube = rng.normal(0.0, noise, size=(wl.size, ny, nx))
    cube += (0.5 + 1e-4 * (wl - 6800.0))[:, None, None]
    for name, rest in REST.items():
        cen = rest * (1 + Z) * (1 + vel / C)
        sig = cen * SIG_KMS / C
        cube += (ratio[name] * bright)[None] * np.exp(
            -0.5 * ((wl[:, None, None] - cen[None]) / sig[None]) ** 2)
    err = np.full_like(cube, noise)
    return cube, err, wl, vel, bright


def model_frames(peak=1.0):
    names = list(REST)
    df = pd.DataFrame({
        'Line_ID': np.arange(len(names), dtype=float),
        'Line_Name': names,
        'Rest Wavelength': [REST[n] for n in names],
        'region_ID': 0,
        'Amp_0': [RATIO[n] * peak for n in names],
        'Centroid_0': [REST[n] * (1 + Z) for n in names],
        'Sigma_0': [REST[n] * (1 + Z) * SIG_KMS / C for n in names],
        'Amp_0_lowlim': 0.0, 'Amp_0_highlim': np.inf,
        'Centroid_0_lowlim': np.nan, 'Centroid_0_highlim': np.nan,
        'Sigma_0_lowlim': 0.0, 'Sigma_0_highlim': np.nan,
    })
    df_cont = pd.DataFrame({
        'Continuum Name': ['Continuum'], 'region_ID': [0],
        'x1': [WIN[0]], 'x2': [WIN[1]], 'cont_type': ['linear'],
        'knots_x': [[]], 'knots_y_0': [[]], 'poly_coef_0': [[]],
        'Slope_0': [0.0], 'Intercept_0': [0.5],
    })
    return df, df_cont


def true_lines(ratio=None):
    ratio = ratio or RATIO
    return np.array([[ratio[n], REST[n] * (1 + Z), REST[n] * (1 + Z) * SIG_KMS / C]
                     for n in REST])


def all_spaxels(ny, nx):
    return [(i, j) for i in range(nx) for j in range(ny)]


# ---------------------------------------------------------------- velocity map

def test_velocity_map_recovers_truth():
    cube, err, wl, vel, bright = make_cube()
    ny, nx = vel.shape
    v, snr, frac, edge = vs.velocity_map(cube, wl, true_lines(), [WIN],
                                         all_spaxels(ny, nx), err, vmax=500, dv=10)
    good = snr >= 10
    assert good.sum() > 0.5 * good.size
    assert np.median(np.abs(v[good] - vel[good])) < 5.0
    assert np.max(np.abs(v[good] - vel[good])) < 40.0
    assert not edge[good].any()
    assert np.median(frac[good]) > 0.9


def test_velocity_map_without_error_cube_matches():
    cube, err, wl, vel, _ = make_cube()
    ny, nx = vel.shape
    g = all_spaxels(ny, nx)
    v1, s1, _, _ = vs.velocity_map(cube, wl, true_lines(), [WIN], g, err)
    v2, s2, _, _ = vs.velocity_map(cube, wl, true_lines(), [WIN], g, None)
    good = s1 >= 10
    assert np.median(np.abs(v1[good] - v2[good])) < 3.0


AGN = {'[N II]_6548': 0.5, 'H_alpha_6563': 1.0, '[N II]_6583': 1.5,
       '[S II]_6716': 0.5, '[S II]_6731': 0.45}


def test_mismatched_line_ratios_still_match_and_are_trusted():
    """Template with star-forming ratios, data with [N II] 1.5x H-alpha: the
    velocity must still be right, and not rejected as an alias."""
    cube, err, wl, vel, _ = make_cube(vamp=300.0, ratio=AGN)
    ny, nx = vel.shape
    v, snr, frac, edge = vs.velocity_map(cube, wl, true_lines(), [WIN],
                                         all_spaxels(ny, nx), err, vmax=500)
    good = snr >= 10
    assert good.sum() > 0
    assert np.max(np.abs(v[good] - vel[good])) < 40.0
    assert np.median(frac[good]) > 0.5


def test_shift_beyond_vmax_is_not_trusted():
    """True shift beyond the grid: the peak lands on an interior alias, which
    must fail the explained-fraction check rather than seed the fit."""
    cube, err, wl, vel, _ = make_cube(vamp=900.0)
    ny, nx = vel.shape
    v, snr, frac, edge = vs.velocity_map(cube, wl, true_lines(), [WIN],
                                         all_spaxels(ny, nx), err, vmax=400)
    beyond = (np.abs(vel) > 600) & (snr >= 10)
    assert beyond.sum() > 0
    assert (edge[beyond] | (frac[beyond] < 0.25)).all()
    inside = (np.abs(vel) < 300) & (snr >= 10)
    assert np.median(frac[inside]) > 0.9


def test_off_detector_zeros_are_ignored():
    cube, err, wl, vel, _ = make_cube()
    ny, nx = vel.shape
    cube[:, 3, 5] = 0.0
    v, snr, _, _ = vs.velocity_map(cube, wl, true_lines(), [WIN],
                                   all_spaxels(ny, nx), err)
    assert not np.isfinite(v[3, 5]) or not np.isfinite(snr[3, 5])


# ---------------------------------------------------------------- smoothing

def test_single_outlier_does_not_seed_itself():
    ny, nx = 9, 9
    v = np.full((ny, nx), 120.0)
    v[4, 4] = -450.0
    snr = np.full((ny, nx), 50.0)
    seed, window = vs.smooth_seeds(v, snr, np.zeros((ny, nx), bool),
                                   np.ones((ny, nx), bool))
    assert seed[4, 4] == pytest.approx(120.0)
    assert (window == 3).all()


def test_unmeasured_spaxels_fill_from_wider_windows_then_zero():
    ny, nx = 40, 40
    v = np.full((ny, nx), np.nan)
    snr = np.full((ny, nx), np.nan)
    v[0, 0], snr[0, 0] = 200.0, 30.0
    gated = np.ones((ny, nx), bool)
    seed, window = vs.smooth_seeds(v, snr, np.zeros((ny, nx), bool), gated,
                                   sizes=(3, 5, 9))
    assert seed[1, 1] == 200.0 and window[1, 1] == 3
    assert seed[3, 3] == 200.0 and window[3, 3] == 9
    assert seed[30, 30] == 0.0 and window[30, 30] == 0


def test_low_snr_and_edge_measurements_are_not_used():
    ny, nx = 5, 5
    v = np.full((ny, nx), 100.0)
    snr = np.full((ny, nx), 50.0)
    edge = np.zeros((ny, nx), bool)
    v[:, :2], snr[:, :2] = -300.0, 2.0          # noise
    v[:, 3:], edge[:, 3:] = 500.0, True        # beyond the grid
    seed, _ = vs.smooth_seeds(v, snr, edge, np.ones((ny, nx), bool), smin=5.0)
    assert np.allclose(seed, 100.0)


def test_aliases_are_not_used():
    ny, nx = 5, 5
    v = np.full((ny, nx), 100.0)
    snr = np.full((ny, nx), 50.0)
    frac = np.full((ny, nx), 0.95)
    v[:, :2], frac[:, :2] = -575.0, 0.02       # bright, but explains nothing
    seed, _ = vs.smooth_seeds(v, snr, np.zeros((ny, nx), bool),
                              np.ones((ny, nx), bool), frac=frac)
    assert np.allclose(seed, 100.0)


# ---------------------------------------------------------------- shifting params

def test_shift_line_centroids_moves_free_centroids_and_bounds_only():
    from lmfit import Parameters
    p = Parameters()
    p.add('cen1', value=6000.0, min=5990.0, max=6010.0)
    p.add('cen2', expr='1.01*cen1')
    p.add('cen3', value=6500.0, vary=False)
    p.add('cen4', value=6600.0)
    p.add('sigma1', value=2.0)
    q = HF.shift_line_centroids(p, 300.0)
    f = 1 + 300.0 / C
    assert q['cen1'].value == pytest.approx(6000.0 * f)
    assert q['cen1'].init_value == pytest.approx(6000.0 * f)
    assert (q['cen1'].min, q['cen1'].max) == (pytest.approx(5990 * f), pytest.approx(6010 * f))
    assert q['cen2'].expr == '1.01*cen1'
    assert q['cen3'].value == 6500.0
    assert q['cen4'].value == pytest.approx(6600.0 * f) and not np.isfinite(q['cen4'].max)
    assert q['sigma1'].value == 2.0
    assert p['cen1'].value == 6000.0              # the input is not mutated


def test_large_negative_shift_does_not_trip_bounds():
    from lmfit import Parameters
    p = Parameters()
    p.add('cen1', value=6000.0, min=5999.0, max=6001.0)
    q = HF.shift_line_centroids(p, -900.0)
    f = 1 - 900.0 / C
    assert q['cen1'].value == pytest.approx(6000.0 * f)
    assert q['cen1'].min < q['cen1'].value < q['cen1'].max


# ---------------------------------------------------------------- end to end

def _params(df, df_cont):
    params, _, _, df = HF.build_params(df, df_cont, Z)
    return params, df


def test_seeded_fit_is_at_least_as_good_and_recovers_the_velocity():
    cube, err, wl, vel, _ = make_cube(vamp=350.0, noise=0.01)
    df, df_cont = model_frames()
    params, df = _params(df, df_cont)
    model = HF.build_model(len(df_cont), len(df))
    ny, nx = vel.shape
    j = ny // 2
    far = int(np.argmax(vel[j]))             # the most redshifted column
    spec, sig = cube[:, j, far], err[:, j, far]

    base = HF.fit_one_spaxel(spec, np.zeros_like(spec), wl, params, model, df,
                             df_cont, Z, 512, False, (far, j), 0, 0, {}, sig, 'test')
    seeded = HF.fit_one_spaxel(spec, np.zeros_like(spec), wl,
                               HF.shift_line_centroids(params, vel[j, far]), model,
                               df, df_cont, Z, 512, False, (far, j), 0, 0, {}, sig, 'test')
    ha = lambda rows: next(r for r in rows if r['LineName'] == 'H_alpha_6563')
    assert ha(seeded)['rchisq_w'] <= ha(base)['rchisq_w'] * 1.001
    assert abs(ha(seeded)['vel_fit'] - vel[j, far]) < 20.0
    assert ha(seeded)['vel_init'] == pytest.approx(vel[j, far], abs=1e-6)


def test_build_seeds_end_to_end_uses_the_mean_spectrum_fit():
    cube, err, wl, vel, bright = make_cube()
    df, df_cont = model_frames()
    params, df = _params(df, df_cont)
    ny, nx = vel.shape
    gated = all_spaxels(ny, nx)
    res = vs.build_seeds(cube, wl, gated, params, df, df_cont, Z, err, 'test')
    assert res.template_source.startswith('mean-spectrum fit')
    ok = res.window > 0
    assert ok.mean() > 0.9
    assert np.median(np.abs(res.seed[ok] - vel[ok])) < 15.0
    assert 'trusted' in res.describe()


def test_run_pool_seeds_reach_the_workers():
    cube, err, wl, vel, _ = make_cube(ny=4, nx=6, vamp=250.0, noise=0.01)
    df, df_cont = model_frames()
    params, df = _params(df, df_cont)
    gated = [(0, 1), (5, 1)]
    seeds = [vel[1, 0], vel[1, 5]]
    rows, _ = HF.run_pool(cube, wl, params, df, df_cont, Z, 3000.0, gated,
                          (np.zeros(2), np.zeros(2)), n_workers=2, err_cube=err,
                          sigma_label='test', seed_vel=seeds)
    got = {(r['spaxel_x'], r['spaxel_y']): r['vel_init']
           for r in rows if r['LineName'] == 'H_alpha_6563'}
    assert got[(0, 1)] == pytest.approx(seeds[0], abs=1e-6)
    assert got[(5, 1)] == pytest.approx(seeds[1], abs=1e-6)
    rows0, _ = HF.run_pool(cube, wl, params, df, df_cont, Z, 3000.0, gated,
                           (np.zeros(2), np.zeros(2)), n_workers=2, err_cube=err,
                           sigma_label='test')
    assert all(abs(r['vel_init']) < 1e-6 for r in rows0 if r['LineName'] == 'H_alpha_6563')
