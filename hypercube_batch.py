"""Headless batch fitting for HyperCube — Stage 5 of `HyperCube_Templates_SPEC.md`.

Expands a rest-frame template across a target manifest and fits each cube with
no GUI, writing products the GUI can reopen.

    python -m hypercube_batch --template MAUNA_MUSE_core.hct.csv \\
                              --manifest MAUNA_MUSE.manifest.csv \\
                              --out runs/ [--targets A,B] [--cores 8] [--dry-run]

`--dry-run` expands and reports without fitting; that is how the coverage /
"which cells are even possible" map is produced.


Example Usage:

  python -m hypercube_batch \
    --template "/Volumes/Seagate Bac/MUSE_MAUNA/MAUNA_MUSE_Template_SingleFits_NoConstraints.hct.csv" \
    --manifest MAUNA_MUSE.manifest.csv \
    --out "/Volumes/Seagate Bac/MUSE_MAUNA/fits" \
    --targets IRAS07251-0248

  Test I/O without fitting:

  python -m hypercube_batch \
    --template "/Volumes/Seagate Bac/MUSE_MAUNA/MAUNA_MUSE_Template_SingleFits_NoConstraints.hct.csv" \
    --manifest MAUNA_MUSE.manifest.csv \
    --out "/Volumes/Seagate Bac/MUSE_MAUNA/fits" \
    --targets IRAS07251-0248 \
    --dry-run

output are .csv files, placed in ./runs. Viewable in HyperCube GUI.



This does **not** make per-spaxel fitting faster — it is the same kernel and the
same process pool the GUI uses. What it buys is unattended queued runs,
reproducibility, and running somewhere other than a desktop.

Note on imports: the fitting itself goes through `HyperCube_fit`, which is
Qt-free, but this script imports `HyperCube` for the unit/export helpers
(`to_export_units`, `flux_unit_str`, the constraint translators). That import
constructs no widgets and needs no display, but PyQt5 does have to be
installed.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

import HyperCube_LSF as hlsf
import HyperCube_Templates as HT
import HyperCube_fit as HF
import HyperCube_Noise
import HyperCube as HC          # unit helpers + constraint translators

# Line colours the GUI's overlay expects (HyperCube.py, post-fit assembly).
_SVG_COLORS = ['dodgerblue', 'mediumseagreen', 'darkorange', 'mediumpurple',
               'deepskyblue', 'gold', 'steelblue', 'mediumaquamarine',
               'peru', 'cornflowerblue']


# ── cube I/O ─────────────────────────────────────────────────────────────────

def load_cube(path, ext=None, with_data=True):
    """Open a cube headlessly. Returns (data, header, wavelengths, wcs, ext).

    `with_data=False` reads only the header — a dry run needs the wavelength
    axis and the instrument, not several GB of pixels, and these cubes live on
    an external drive.
    """
    with fits.open(path, memmap=True) as hdul:
        if ext is None:
            ext = next(i for i, h in enumerate(hdul) if h.header.get('NAXIS') == 3)
        ext = int(ext)
        header = hdul[ext].header.copy()
        # An image extension inherits the metadata it does not state itself
        # from the primary header, which is where MEF files keep it.
        if ext != 0:
            for key in hdul[0].header:
                if key in ('COMMENT', 'HISTORY', '') or key in header:
                    continue
                try:
                    header[key] = hdul[0].header[key]
                except Exception:
                    pass
        # Native dtype, deliberately: casting to float32 would silently halve
        # the precision of every spectrum relative to what the GUI fits, and
        # showed up as ~1e-3 disagreement in sigma_fit against the same model.
        data = np.asarray(hdul[ext].data) if with_data else None
    wl = HT.cube_axes(header)
    if wl is None:
        raise ValueError(f'{path}[{ext}]: no spectral WCS (CRVAL3/CDELT3)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            wcs = WCS(header).celestial
        except Exception:
            wcs = None
    return data, header, np.asarray(wl, dtype=float), wcs, ext


def sky_coords(wcs, xs, ys):
    """RA/Dec for pixel lists, or NaNs — kept out of the workers deliberately."""
    if wcs is None:
        return np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sky = wcs.pixel_to_world(np.asarray(xs), np.asarray(ys))
        return np.atleast_1d(sky.ra.deg), np.atleast_1d(sky.dec.deg)
    except Exception:
        return np.full(len(xs), np.nan), np.full(len(xs), np.nan)


def error_cube_for(path, shape, ext):
    """Discover a sigma cube the same way the GUI does at ingest."""
    try:
        spec = HyperCube_Noise.detect(path, shape, ext)
        if spec is None:
            return None, None
        sigma, info = HyperCube_Noise.load_sigma(spec, shape)
        return sigma, (info.get('label') if isinstance(info, dict) else str(info))
    except Exception as e:
        print(f'   measurement errors: {e} — using the empirical estimate')
        return None, None


# ── output ───────────────────────────────────────────────────────────────────

def write_fit_csv(path, df_obs, df_cont, df, df_fit, df_stellar, flux_unit):
    """The 5-(or 6-)section layout `FitParamsWindow.load_cube_fit` reads."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(df_obs.columns); w.writerows(df_obs.values); w.writerow([])
        w.writerow(['wavelength scale factor', 'flux scale factor',
                    'flux unit', 'wavelength unit', 'velocity unit'])
        w.writerow([1, 1, flux_unit, 'Angstrom', 'km/s']); w.writerow([])
        for frame in (df_cont, df, df_fit):
            out = HC.to_export_units(frame)
            w.writerow(out.columns); w.writerows(out.values); w.writerow([])
        if df_stellar is not None and len(df_stellar):
            out = HC.to_export_units(df_stellar)
            w.writerow(out.columns); w.writerows(out.values)
    return path


def git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=5
                              ).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


# ── one target ───────────────────────────────────────────────────────────────

def run_target(template, row, args, out_dir):
    """Expand and (unless --dry-run) fit one manifest row. Returns a summary."""
    tid = str(row['target_id'])
    t0 = time.perf_counter()
    z = float(row['redshift'])
    stem = f"{tid}_{template.name}"

    data, header, wl, wcs, ext = load_cube(row['cube_path'],
                                           row.get('cube_ext') or None,
                                           with_data=not args.dry_run)
    lsf = HT.lsf_for_cube(header, wl, muse_model=args.muse_lsf)

    regions_obs = [(HT._as_float(r['x1_rest_A']) * (1 + z),
                    HT._as_float(r['x2_rest_A']) * (1 + z))
                   for _, r in template.regions.iterrows()]
    if args.dry_run:
        # Amplitudes and the continuum are the things that need pixels; a dry
        # run reports coverage, which does not depend on either.
        ref_spec, amp_scale, used_xy = None, 1.0, None
    else:
        ref_xy = None
        if str(row.get('ref_spaxel_x', '')).strip():
            ref_xy = (int(row['ref_spaxel_x']), int(row['ref_spaxel_y']))
        # This galaxy's own spectrum seeds the continuum and the amplitude
        # scale; a template cannot carry either, they are in the cube's units.
        ref_spec, used_xy = HT.reference_spectrum(data, wl, regions_obs,
                                                  ref_xy=ref_xy)
        amp_scale = 1.0

    df_obs, df_cont, df, rep = HT.expand(
        template, z, wl, lsf, target_name=tid, amp_scale=amp_scale,
        coverage_nsigma=args.coverage_nsigma,
        instrument=header.get('INSTRUME'), ref_spectrum=ref_spec)

    cov_path = os.path.join(out_dir, f'{stem}_coverage.csv')
    rep.to_frame().to_csv(cov_path, index=False)
    with open(os.path.join(out_dir, f'{stem}_coverage.txt'), 'w') as f:
        f.write(rep.to_text() + '\n')

    summary = dict(target=tid, z=z, lines=len(df), regions=len(df_cont),
                   dropped=len(rep.dropped_lines), lsf=rep.lsf_label,
                   amp_scale=amp_scale, ref_spaxel=used_xy, fitted=0,
                   seconds=0.0, csv='', status='expanded')
    if args.dry_run:
        summary['seconds'] = time.perf_counter() - t0
        return summary
    if not len(df):
        summary['status'] = 'no lines in coverage — nothing to fit'
        return summary

    # ── S/N gate ──
    snr_thresh = float(row.get('snr_threshold') or args.snr or 0.0)
    d_lambda = float(np.median(np.diff(wl)))
    snr = HC.compute_snr_map(data, wl, df['Centroid_0'],
                             50 * d_lambda, 60 * d_lambda, 70 * d_lambda)
    nx, ny = data.shape[2], data.shape[1]
    gated = [(i, j) for i in range(nx) for j in range(ny)
             if np.isfinite(snr[j, i]) and snr[j, i] >= snr_thresh]
    if not gated:
        summary['status'] = f'no spaxel passed S/N >= {snr_thresh}'
        return summary

    params, n_regions, n_lines, df = HF.build_params(
        df, df_cont, z,
        velocity_updater=HC.update_constraints_with_velocity,
        constraint_applier=HC.add_dataframe_constraints_to_params)

    stellar_specs, stellar_mask = [], None
    for _, r in df_cont[df_cont['cont_type'] == 'stellar'].iterrows():
        stellar_specs.append(dict(rid=int(r['region_ID']),
                                  library=str(r.get('stellar_library', '')),
                                  fit_range=(float(r['x1']), float(r['x2'])),
                                  moments=int(r.get('stellar_moments', 2) or 2)))
    if stellar_specs:
        stellar_mask = df['Centroid_0'].to_numpy()

    err, sigma_label = error_cube_for(row['cube_path'], data.shape, ext)
    ras, decs = sky_coords(wcs, [g[0] for g in gated], [g[1] for g in gated])

    last = [0]

    def progress(done, total):
        pct = 100.0 * done / max(total, 1)
        if done == total or pct - last[0] >= 5:
            last[0] = pct
            print(f'   {tid}: {done}/{total} spaxels ({pct:.0f}%)', flush=True)

    line_rows, stellar_rows = HF.run_pool(
        cube=data, wavelengths=wl, params=params,
        # Ship exactly what the GUI ships: actor columns dropped, so the two
        # paths hand the workers the same frames.
        df=df.drop(columns=['curveactor'], errors='ignore').copy(),
        df_cont=df_cont.drop(columns=['lineactor'], errors='ignore').copy(),
        z=z, R=float(lsf.R(np.median(wl))), gated=gated, radec=(ras, decs),
        n_workers=args.cores, err_cube=err, sigma_label=sigma_label,
        sequential=args.sequential, max_nfev=args.max_nfev,
        stellar_specs=stellar_specs, stellar_mask=stellar_mask,
        progress_cb=progress)

    df_fit = pd.DataFrame(line_rows)
    df_stellar = pd.DataFrame(stellar_rows)
    if len(df_fit):
        if 'fit_success' in df_fit.columns:
            df_fit['success'] = df_fit.get('success', True) & df_fit['fit_success']
        if 'LineID' in df_fit.columns:
            df_fit['LineID'] = pd.to_numeric(df_fit['LineID'], errors='coerce')
            # The GUI's overlay colours by LineID; reproduce them here so a
            # headless product opens looking like a GUI one.
            df_fit['color'] = [_SVG_COLORS[int(v) % len(_SVG_COLORS)]
                               if np.isfinite(v) else _SVG_COLORS[0]
                               for v in df_fit['LineID']]

    csv_path = os.path.join(out_dir, f'{stem}_Fit.csv')
    write_fit_csv(csv_path, df_obs, df_cont, df, df_fit, df_stellar,
                  str(header.get('BUNIT', 'unknown')))
    summary.update(fitted=len(gated), csv=csv_path, status='fitted',
                   seconds=time.perf_counter() - t0)
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='hypercube_batch',
        description='Fit a rest-frame template across a manifest of cubes, headless.')
    ap.add_argument('--template', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--targets', default='',
                    help='comma-separated target_ids; default is every enabled row')
    ap.add_argument('--cores', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--dry-run', action='store_true',
                    help='expand and report coverage only; do not fit')
    ap.add_argument('--snr', type=float, default=None,
                    help='S/N gate, overriding the manifest column')
    ap.add_argument('--max-nfev', type=int, default=512)
    ap.add_argument('--sequential', action='store_true',
                    help='staged core->outflow fit')
    ap.add_argument('--coverage-nsigma', type=float, default=3.0)
    ap.add_argument('--muse-lsf', default=hlsf.MUSE_LSF_DEFAULT,
                    choices=sorted(hlsf.MUSE_LSF_MODELS))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    template = HT.read_template(args.template)
    man = HT.read_manifest(args.manifest)
    if args.targets:
        want = {t.strip() for t in args.targets.split(',') if t.strip()}
        man = man[man['target_id'].isin(want)].reset_index(drop=True)
    if not len(man):
        print('No targets selected.'); return 2

    started = datetime.now(timezone.utc)
    print(f'hypercube_batch  template={template.name!r}  targets={len(man)}  '
          f'cores={args.cores}  dry_run={args.dry_run}  git={git_sha()}')

    rows, failures = [], []
    for _, row in man.iterrows():
        tid = str(row['target_id'])
        print(f'\n=== {tid} ===', flush=True)
        try:
            s = run_target(template, row, args, args.out)
            rows.append(s)
            print(f"   {s['status']}: {s['lines']} lines, {s['regions']} regions, "
                  f"{s['dropped']} dropped, {s['fitted']} spaxels, {s['seconds']:.1f}s")
        except Exception as e:
            failures.append((tid, f'{type(e).__name__}: {e}'))
            rows.append(dict(target=tid, status=f'FAILED {type(e).__name__}: {e}'))
            print(f'   FAILED: {type(e).__name__}: {e}', flush=True)

    log = pd.DataFrame(rows)
    log.insert(0, 'template', template.name)
    log.insert(1, 'template_version', template.meta.get('template_version', ''))
    log.insert(2, 'git', git_sha())
    log.insert(3, 'started_utc', started.isoformat(timespec='seconds'))
    log_path = os.path.join(args.out, 'run_log.csv')
    log.to_csv(log_path, index=False)

    cov = [os.path.join(args.out, f) for f in os.listdir(args.out)
           if f.endswith('_coverage.csv')]
    if cov:
        allcov = pd.concat([pd.read_csv(c, keep_default_na=False) for c in cov],
                           ignore_index=True)
        allcov.to_csv(os.path.join(args.out, 'coverage_all.csv'), index=False)

    print(f'\n{len(rows) - len(failures)}/{len(rows)} targets ok; log -> {log_path}')
    if failures:
        print('failures:')
        for tid, msg in failures:
            print(f'   {tid}: {msg}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
