"""Aggregate analysis of the under-fitting checks (see underfit_plan.md).

Usage: python analyze_underfit.py RUNS_DIR --out NEW_DIR
RUNS_DIR holds the folders written by underfit_check.py (d2, d3, <SETTING>_seed<S>). Per-record prediction
files are read but only aggregate tables are written.
"""
from pathlib import Path
import argparse, csv, json
import numpy as np

SETTINGS = ['PRIMARY', 'LR10', 'EP120', 'RES448', 'LR10_RES448']
SETS = ['fit', 'select', 'proxy_test', 'reference_test']
CULTIVARS = ['AR', 'HJ', 'HR', 'SG']


def read_preds(path):
    out = {s: {} for s in SETS}
    for r in csv.DictReader(path.open()):
        out[r['set']][r['objkey']] = (r['kind'], r['tree_key'], float(r['nir']), float(r['sugar']) if r['sugar'] not in ('', 'None') else np.nan, float(r['pred']))
    return out


def mae(p, y): return float(np.abs(p - y).mean())
def corr(p, y): return float(np.corrcoef(p, y)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs'); ap.add_argument('--out', required=True); a = ap.parse_args()
    R = Path(a.runs); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    seeds = {}
    for st in SETTINGS:
        seeds[st] = [read_preds(R / 'd2' / f'preds_seed{s}.csv') if st == 'PRIMARY' else read_preds(R / f'{st}_seed{s}' / 'preds.csv') for s in (0, 1, 2)]
    keys = {s: sorted(seeds['PRIMARY'][0][s]) for s in SETS}
    base = seeds['PRIMARY'][0]
    meta = {s: dict(kind=np.array([base[s][k][0] for k in keys[s]]), tree=np.array([base[s][k][1] for k in keys[s]]),
                    nir=np.array([base[s][k][2] for k in keys[s]]), sugar=np.array([base[s][k][3] for k in keys[s]])) for s in SETS}
    assert [len(keys[s]) for s in SETS] == [8919, 1643, 3197, 1191]
    cm = {c: float(meta['fit']['nir'][meta['fit']['kind'] == c].mean()) for c in CULTIVARS}          # cultivar NIR mean of proxy fitting records
    cmean = {s: np.array([cm[c] for c in meta[s]['kind']]) for s in SETS}
    rows_seed, rows_ens, rows_curve = [], [], []
    ref = meta['reference_test']; trees = np.unique(ref['tree']); tree_idx = {t: np.where(ref['tree'] == t)[0] for t in trees}
    base_err = np.abs(cmean['reference_test'] - ref['sugar'])
    for st in SETTINGS:
        P = {s: np.array([[sd[s][k][4] for k in keys[s]] for sd in seeds[st]]) for s in SETS}
        info = [json.loads((R / f'{st}_seed{s}' / 'completed.json').read_text()) for s in (0, 1, 2)] if st != 'PRIMARY' else None
        for i in range(3):
            row = dict(setting=st, seed=i, selected_epoch=(info[i]['selected']['epoch'] + 1) if info else '')
            for s in SETS: row[f'{s}_nir_mae'] = mae(P[s][i], meta[s]['nir']); row[f'{s}_nir_r'] = corr(P[s][i], meta[s]['nir'])
            row['reference_mae'] = mae(P['reference_test'][i], ref['sugar']); row['reference_r'] = corr(P['reference_test'][i], ref['sugar']); rows_seed.append(row)
        E = {s: P[s].mean(0) for s in SETS}; row = dict(setting=st)
        for s in SETS: row[f'{s}_nir_mae'] = mae(E[s], meta[s]['nir']); row[f'{s}_nir_r'] = corr(E[s], meta[s]['nir']); row[f'{s}_cultivar_mean_nir_mae'] = mae(cmean[s], meta[s]['nir'])
        err = np.abs(E['reference_test'] - ref['sugar']); d = err - base_err
        rng = np.random.default_rng(20261115); boots = []
        for _ in range(2000):
            idx = np.concatenate([tree_idx[t] for t in rng.choice(trees, len(trees), replace=True)]); boots.append(d[idx].mean())
        row.update(reference_n=len(err), reference_trees=len(trees), reference_mae=float(err.mean()), reference_r=corr(E['reference_test'], ref['sugar']),
                   cultivar_mean_reference_mae=float(base_err.mean()), cultivar_mean_reference_r=corr(cmean['reference_test'], ref['sugar']),
                   paired_mae_difference=float(d.mean()), difference_ci_low=float(np.percentile(boots, 2.5)), difference_ci_high=float(np.percentile(boots, 97.5)))
        for c in CULTIVARS:
            m = ref['kind'] == c; row[f'within_{c}_n'] = int(m.sum()); row[f'within_{c}_reference_r'] = corr(E['reference_test'][m], ref['sugar'][m])
        rows_ens.append(row)
        if st != 'PRIMARY':
            for i in range(3):
                for h in json.loads((R / f'{st}_seed{i}' / 'history.json').read_text()):
                    rows_curve.append(dict(setting=st, seed=i, epoch=h['epoch'] + 1, loss=h['loss'], selection_nir_mae=h['selection_nir_mae']))
    for name, rows in (('underfit_by_seed.csv', rows_seed), ('underfit_three_seed_mean.csv', rows_ens), ('underfit_learning_curves.csv', rows_curve)):
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d2 = json.loads((R / 'd2' / 'completed.json').read_text()); d3 = json.loads((R / 'd3' / 'completed.json').read_text())
    summary = dict(plan_sha256=d2['plan_sha256'], manifest_sha256=d2['manifest_sha256'], torch=d2['torch'], timm=d2['timm'], cultivar_nir_means=cm,
                   primary_checkpoint_agreement={k: v['max_abs_difference_from_saved_primary_predictions'] for k, v in d2['results'].items()},
                   capacity_test=d3['results'], three_seed_mean=rows_ens)
    (out / 'underfit_summary.json').write_text(json.dumps(summary, indent=2))
    for r in rows_ens:
        print(f"{r['setting']:12s} fit {r['fit_nir_mae']:.3f} sel {r['select_nir_mae']:.3f} ptest {r['proxy_test_nir_mae']:.3f} (r {r['proxy_test_nir_r']:.3f}) | ref MAE {r['reference_mae']:.3f} r {r['reference_r']:.3f} diff {r['paired_mae_difference']:+.3f} [{r['difference_ci_low']:+.3f}, {r['difference_ci_high']:+.3f}] | within r " + ' '.join(f"{c} {r[f'within_{c}_reference_r']:+.3f}" for c in CULTIVARS))
    r = rows_ens[0]; print('cultivar-mean: fit', round(r['fit_cultivar_mean_nir_mae'], 3), 'sel', round(r['select_cultivar_mean_nir_mae'], 3), 'ptest', round(r['proxy_test_cultivar_mean_nir_mae'], 3), 'ref', round(r['cultivar_mean_reference_mae'], 3), 'r', round(r['cultivar_mean_reference_r'], 3))
    for r in rows_seed: print(r['setting'], r['seed'], 'epoch', r['selected_epoch'], 'fit', round(r['fit_nir_mae'], 3), 'sel', round(r['select_nir_mae'], 3), 'ref', round(r['reference_mae'], 3), round(r['reference_r'], 3))


if __name__ == '__main__':
    main()
