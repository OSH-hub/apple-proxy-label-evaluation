"""Recompute published pooled MAE and paired intervals from anonymous cluster sums.

This does not require, release, or reconstruct individual source labels. It cannot
reproduce ranks, calibration, model fitting or biological measurement pairing.
Run with --root pointing to the directory containing tables/.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
root=args.root
cluster=pd.read_csv(root/'tables/DataS09_cluster_error_sums.csv')
acc=pd.read_csv(root/'tables/DataS02_accuracy.csv')
paired=pd.read_csv(root/'tables/DataS05_paired_accuracy.csv')
keys=['scheme','cohort','cluster_unit','inference_role'];checks=[]
def select(frame,values):
    for k,v in zip(keys,values):frame=frame[frame[k]==v]
    return frame
for values,g in cluster.groupby(keys,sort=True):
    assert len(g)==g.cluster_index.nunique()
    g=g.sort_values('cluster_index');n=g.n.to_numpy();rng=np.random.default_rng(20261026)
    ix=rng.integers(len(g),size=(5000,len(g)));den=n[ix].sum(axis=1)
    for predictor in ['ensemble','nir','global_mean','cultivar_mean']:
        sums=g[predictor+'_absolute_error_sum'].to_numpy()
        q=select(acc,values);q=q[(q.cultivar=='ALL')&(q.predictor==predictor)];assert len(q)==1
        observed=np.array([sums.sum()/n.sum(),*np.quantile(sums[ix].sum(axis=1)/den,[.025,.975])])
        expected=q[['MAE','MAE_low','MAE_high']].iloc[0].to_numpy()
        assert np.max(np.abs(observed-expected))<1e-12,(values,predictor,observed,expected)
        checks.append({'group':list(values),'predictor':predictor,'max_abs_difference':float(np.max(np.abs(observed-expected)))})
    for baseline in ['nir','global_mean','cultivar_mean']:
        sums=(g.ensemble_absolute_error_sum-g[baseline+'_absolute_error_sum']).to_numpy()
        q=select(paired,values);q=q[q.comparison=='ensemble_minus_'+baseline];assert len(q)==1
        observed=np.array([sums.sum()/n.sum(),*np.quantile(sums[ix].sum(axis=1)/den,[.025,.975])])
        expected=q[['delta_MAE','low','high']].iloc[0].to_numpy()
        assert np.max(np.abs(observed-expected))<1e-12,(values,baseline,observed,expected)
        checks.append({'group':list(values),'paired_baseline':baseline,'max_abs_difference':float(np.max(np.abs(observed-expected)))})
print(json.dumps({'passed':True,'aggregate_comparisons':len(checks),'max_absolute_difference':max(x['max_abs_difference'] for x in checks),'limits':'Only aggregate accuracy and paired conditional cluster intervals; no individual-label reconstruction.'},indent=2))
