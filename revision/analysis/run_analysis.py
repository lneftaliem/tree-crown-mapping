#!/usr/bin/env python3
"""
run_analysis.py -- single, restartable entry point that regenerates every tree-derived manuscript analysis, figure and
table from ONE explicitly named tree-crown dataset.

  python run_analysis.py --dataset <dir with <city>_ripley_points_with_income.geojson (+ _polygons_)> --label v1 \
                         --out <run dir> [--tile-footprints tiles.csv] [--build-logs 'glob'] [--legacy-correction] \
                         [--allow PATTERN ...] [--only step,step] [--force]

Every step runs as a separate process with:
  TREE_MAPPING_DATASET_DIR = --dataset   (the ONLY source of tree data; repro_data.py serves nothing else)
  TREE_MAPPING_BASE_DIR    = --out       (staging + outputs; nothing is written anywhere else)
  the contamination guard (guard/sitecustomize.py): every file read is logged to <out>/provenance/files_opened.log and
  any historical tree dataset / cached tree table / old output raises; --allow exempts patterns (only used to re-run
  the ORIGINAL dataset for the reproduction check, together with --legacy-correction).
A step is skipped when its outputs exist (restartable); --force or --only re-runs. After the run, provenance/
inputs_used.csv lists every input with its SHA-256, and provenance/run_manifest.json records the steps and outputs.
"""
import os, sys, json, glob, time, argparse, subprocess, hashlib, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'repo_adapted')
PY = sys.executable
INVENTORY_CSV = os.environ.get('INVENTORY_CSV', '/path/to/Urban_data/sherlock_venv/notebooks/urbantreedata_2023/new_data.csv')

# (name, command (relative to HERE / REPO), cwd = run dir, outputs that mark the step done)
STEPS = [
    ('units', [PY, f'{HERE}/build_unit_table.py'], ['analysis_output/equity_results/all_cities_urban_covariates_corrected.csv']),
    ('fig2_4', [PY, f'{REPO}/generate_plots_corrected.py'], ['analysis_output/equity_results/combined_income_zscore_density_scatter.pdf']),
    ('ed3_4_models', [PY, f'{HERE}/adjusted_models.py'], ['analysis_output/adjusted_models/ols_hc3.csv']),
    ('ed3_4_plots', [PY, f'{HERE}/plot_adjusted.py'], ['analysis_output/extended_data/ED_Fig4_adjusted_income_coefficients.pdf']),
    ('fig1', [PY, f'{REPO}/generate_density_map_from_geojson.py'], ['analysis_output/identical_tree_density_map.pdf']),
    ('ed_table1', [PY, f'{HERE}/ed_table1.py'], ['analysis_output/extended_data/ED_Table1.csv']),
    ('ed_fig1', [PY, f'{HERE}/ed_fig1.py'], ['analysis_output/extended_data/ED_Fig1_dedup_sensitivity.pdf']),
    ('ed_fig2', [PY, f'{REPO}/plot_crown_area_histogram.py'], ['analysis_output/crown_area_histogram.png']),
    ('figS5', [PY, f'{REPO}/inventory_correlation_analysis.py'], ['analysis_output/inventory_correlation/inventory_correlation_stats.csv']),
    ('figS6', [PY, f'{HERE}/ripley_l.py'], ['analysis_output/ripley/ripley_auc_income_spearman.csv']),
    ('figS6_plot', [PY, f'{HERE}/plot_ripley_s6.py'], ['analysis_output/ripley/FigS6_ripley_L_by_income_quartile.svg']),
    ('figS9_11', [PY, f'{REPO}/generate_redlining_figures.py'], ['analysis_output/redlining_figures/austin_holc_grade_metrics.csv']),
    ('figS12_stats', [PY, f'{REPO}/run_surrey_reviewer2_sensitivity.py'], ['analysis_output/surrey_reviewer2_analysis/surrey_sensitivity_scenarios_summary.csv']),
    ('figS12_plot', [PY, f'{REPO}/plot_surrey_s12_three_panel.py'], ['analysis_output/surrey_reviewer2_analysis/surrey_reviewer2_sensitivity_figure.pdf']),
    ('sensitivity', [PY, f'{HERE}/sensitivity.py'], ['analysis_output/sensitivity/canopy_denominator_spearman.csv']),
    ('stats', [PY, f'{HERE}/manuscript_stats.py'], ['analysis_output/stats/spearman_unadjusted.csv']),
]
TREE_FILE_HINTS = ('ripley_points', 'ripley_polygons', 'urban_covariates', '_split.parquet', 'tree_points')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tile-footprints', help='CSV filename,left,bottom,right,top of the dataset tiles (Figure 1 coverage mask)')
    ap.add_argument('--build-logs', help="glob of the dataset's build_log JSON (Extended Data Fig. 1)")
    ap.add_argument('--legacy-correction', action='store_true', help='original dataset only: published latitude correction')
    ap.add_argument('--allow', nargs='*', default=[], help='guard exemptions (original-dataset reproduction only)')
    ap.add_argument('--only', help='comma-separated step names')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    ds, out = os.path.realpath(a.dataset), os.path.realpath(a.out)
    os.makedirs(f'{out}/provenance', exist_ok=True)
    for c in ('austin', 'bloomington', 'cupertino', 'surrey'):
        assert os.path.exists(f'{ds}/{c}_ripley_points_with_income.geojson'), f'missing points for {c} in {ds}'
    if a.tile_footprints:
        shutil.copyfile(a.tile_footprints, f'{out}/tile_metadata.csv')  # Figure 1 reads ./tile_metadata.csv
    os.makedirs(f'{out}/urbantreedata_2023', exist_ok=True)
    inv = f'{out}/urbantreedata_2023/new_data.csv'
    if not os.path.lexists(inv):
        os.symlink(INVENTORY_CSV, inv)  # municipal inventories (non-model input, unchanged)
    env = dict(os.environ, TREE_MAPPING_DATASET_DIR=ds, TREE_MAPPING_BASE_DIR=out, GUARD_ACTIVE='1',
               GUARD_LOG=f'{out}/provenance/files_opened.log', PYTHONPATH=os.path.join(HERE, 'guard'),
               GUARD_ALLOW=os.pathsep.join(a.allow), MPLBACKEND='Agg', PYTHONWARNINGS='ignore')
    env.pop('TREE_MAPPING_TREES_DIR', None)
    if a.legacy_correction:
        env['TREE_MAPPING_LEGACY_CORRECTION'] = '1'
    else:
        env.pop('TREE_MAPPING_LEGACY_CORRECTION', None)
    if a.build_logs:
        env['DATASET_BUILD_LOGS'] = a.build_logs
    only = set(a.only.split(',')) if a.only else None
    man_path = f'{out}/provenance/run_manifest.json'
    man = json.load(open(man_path)) if os.path.exists(man_path) else {'steps': {}}
    man.update(label=a.label, dataset=ds, out=out, legacy_correction=a.legacy_correction, guard_allow=a.allow,
               python=PY, started=man.get('started', time.strftime('%Y-%m-%dT%H:%M:%S')))
    subprocess.run([PY, '-c', 'import repro_data; repro_data.stage_all(); repro_data.stage_redlining()'], cwd=REPO, env=env, check=True)
    for name, cmd, outs in STEPS:
        if only and name not in only:
            continue
        if name == 'ed_fig1' and not a.build_logs:
            man['steps'][name] = {'status': 'skipped (no build logs for this dataset)'}
            continue
        done = all(os.path.exists(f'{out}/{o}') for o in outs)
        if done and not a.force and not only:
            print(f'[skip] {name} (outputs exist)'); continue
        t0 = time.time()
        log = f'{out}/provenance/log_{name}.txt'
        print(f'[run ] {name} ...', flush=True)
        senv = dict(env)
        if name == 'units':  # the one step entitled to the committed census tables, for their NON-tree columns only
            senv['GUARD_ALLOW'] = os.pathsep.join(a.allow + ['/covariates_nontree/'])
        r = subprocess.run(cmd, cwd=out, env=senv, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
        ok = r.returncode == 0 and all(os.path.exists(f'{out}/{o}') for o in outs)
        man['steps'][name] = {'cmd': cmd, 'returncode': r.returncode, 'seconds': round(time.time() - t0), 'ok': ok, 'log': log}
        json.dump(man, open(man_path, 'w'), indent=1)
        if not ok:
            print(open(log).read()[-3000:])
            raise SystemExit(f'step {name} failed (see {log})')
    # contamination audit: every tree-like file that was opened must be inside the dataset directory
    opened = [l.rstrip('\n').split('\t')[1] for l in open(env['GUARD_LOG'])] if os.path.exists(env['GUARD_LOG']) else []
    tree_files = sorted({p for p in opened if any(h in os.path.basename(p) for h in TREE_FILE_HINTS)})
    # the committed census tables are read by the 'units' step for their NON-tree columns only (tree columns are
    # recomputed and checked there); they are recorded separately and are not tree inputs
    nontree = [p for p in tree_files if '/covariates_nontree/' in p]
    foreign = [p for p in tree_files if not (p.startswith(ds) or p.startswith(out)) and p not in nontree]
    man['nontree_census_tables_read_by_units_step'] = nontree
    man['tree_files_opened'] = [p for p in tree_files if p not in nontree]
    man['tree_files_outside_dataset'] = foreign
    man['dataset_files'] = {os.path.basename(f): sha256(f) for f in sorted(glob.glob(f'{ds}/*_with_income.geojson'))}
    man['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    json.dump(man, open(man_path, 'w'), indent=1)
    if foreign:
        raise SystemExit(f'CONTAMINATION: tree files outside the dataset were opened: {foreign}')
    print(f'done; tree files opened: {len(tree_files)}, all inside the dataset/run dir; manifest {man_path}')


if __name__ == '__main__':
    main()
