"""Aggregate regenerated crops/models without publishing row-level source data."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,hashlib,time,platform
import numpy as np,pandas as pd
from scipy.stats import spearmanr,norm
O=Path(__file__).resolve().parents[1];START=time.time();SOURCES=[]
assert all((O/f'runs/{s}_seed{r}/completed.json').exists() for s in ['tree','zone'] for r in range(3))
for folder in ['tables','results','private']:(O/folder).mkdir(exist_ok=True)
def read(p,**kw):
    SOURCES.append(p);return pd.read_csv(p,**kw)
def save(p,s):
    if p.exists():raise FileExistsError(p)
    p.write_text(s,encoding='utf-8')
def table(rows,name):save(O/'tables'/name,pd.DataFrame(rows).to_csv(index=False))
def js(path,d):save(path,json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item()))
def ci_mean(x,groups,B=5000):
    q=pd.DataFrame({'x':np.asarray(x),'g':np.asarray(groups)}).groupby('g').x.agg(['sum','count'])
    if len(q)<2:return np.nan,np.nan
    ix=np.random.default_rng(20261026).integers(len(q),size=(B,len(q)))
    v=q['sum'].to_numpy()[ix].sum(1)/q['count'].to_numpy()[ix].sum(1)
    return tuple(np.quantile(v,[.025,.975]))
def topweights(x,f):
    n=int(np.floor(len(x)*f));out=np.zeros(len(x))
    if n==0:return out
    cut=np.sort(x)[-n];out[x>cut]=1;eq=x==cut;out[eq]=(n-out.sum())/eq.sum()
    assert np.isclose(out.sum(),n)
    return out
def rho_ci(x,y,g):
    rho=float(spearmanr(x,y).statistic);ids=[np.flatnonzero(g==v) for v in np.unique(g)];rng=np.random.default_rng(20261026);rs=[]
    for _ in range(1000):
        ix=np.concatenate([ids[j] for j in rng.integers(len(ids),size=len(ids))]);r=spearmanr(x[ix],y[ix]).statistic
        if np.isfinite(r):rs.append(r)
    return (rho,*np.quantile(rs,[.025,.975])) if rs else (rho,np.nan,np.nan)
def ensemble(scheme,part,source,manifest):
    fs=[read(O/f'runs/{scheme}_seed{s}/preds_{part}_{source}.csv',dtype={'tree':str,'pos':str}) for s in range(3)]
    assert all(fs[0].objkey.tolist()==f.objkey.tolist() for f in fs)
    f=fs[0].copy();pred=np.stack([x.pred for x in fs]);mc=np.stack([x.mc_std for x in fs]);alea=np.stack([x.sigma_alea for x in fs])
    f['ensemble']=pred.mean(0);f['aleatoric']=np.sqrt((alea**2).mean(0));f['mc_dropout']=np.sqrt((mc**2).mean(0));f['between_seed']=pred.std(0);f['total']=np.sqrt(f.aleatoric**2+f.between_seed**2)
    f['unique_NIR_count']=f.objkey.map(manifest.set_index('objkey').unique_NIR_count)
    assert (f.mc_dropout>0).all() and f.unique_NIR_count.notna().all() and (f.total>0).all()
    return f
inventory=[];accuracy=[];paired=[];uncertainty=[];curves=[];coverage=[];seeds=[];labels=[];clusters=[];within=[];prediction_sets=[];overlap=[]
for scheme in ['tree','zone']:
    manifest=read(O/f'splits/{scheme}_manifest.csv',dtype={'tree':str,'pos':str})
    assert not set(manifest.loc[manifest.partition=='test','tree_key'])&set(manifest.loc[manifest.partition!='test','tree_key'])
    if scheme=='zone':assert not set(manifest.loc[manifest.partition=='test','zone_key'])&set(manifest.loc[manifest.partition!='test','zone_key'])
    for (part,source),g in manifest.groupby(['partition','source']):inventory.append({'scheme':scheme,'partition':part,'source':source,'objects':len(g),'trees':g.tree_key.nunique(),'cultivar_farm_combinations':g.zone_key.nunique(),'farms':g.pos.nunique()})
    testfarms=set(manifest.loc[manifest.partition=='test','pos']);devfarms=set(manifest.loc[manifest.partition!='test','pos'])
    overlap.append({'scheme':scheme,'test_farms':len(testfarms),'development_farms':len(devfarms),'shared_farms':len(testfarms&devfarms),'farm_disjoint':not bool(testfarms&devfarms)})
    fit=manifest[(manifest.partition=='fit')&(manifest.source=='train')];gm=fit.nir.mean();cm=fit.groupby('kind').nir.mean()
    sets={(p,s):ensemble(scheme,p,s,manifest) for p in ['cal','test'] for s in ['train','val']}
    for cohort in ['same_row_complete','strict_unique_NIR']:
        f=sets['test','val'].copy();cal=sets['cal','val'].copy()
        if cohort=='strict_unique_NIR':f=f[f.unique_NIR_count==1].copy();cal=cal[cal.unique_NIR_count==1].copy()
        assert len(f)>0 and len(cal)>0
        f['global_mean']=gm;f['cultivar_mean']=f.kind.map(cm)
        cluster_units=['tree_key','pos'] if scheme=='tree' else ['pos','zone_key']
        for cluster in cluster_units:
            base={'scheme':scheme,'cohort':cohort,'cluster_unit':cluster,'inference_role':('farm_cluster_sensitivity' if scheme=='tree' and cluster=='pos' else ('combination_cluster_sensitivity' if cluster=='zone_key' else 'primary'))}
            groups=f[cluster].to_numpy()
            for predictor in ['ensemble','nir','global_mean','cultivar_mean']:
                for cultivar in ['ALL']+sorted(f.kind.unique()):
                    q=f if cultivar=='ALL' else f[f.kind==cultivar];err=q[predictor].to_numpy()-q.sugar.to_numpy();lo,hi=ci_mean(np.abs(err),q[cluster])
                    accuracy.append({**base,'cultivar':cultivar,'predictor':predictor,'n':len(q),'clusters':q[cluster].nunique(),'MAE':np.abs(err).mean(),'MAE_low':lo,'MAE_high':hi,'RMSE':np.sqrt((err**2).mean()),'bias':err.mean(),'pearson_r':np.corrcoef(q[predictor],q.sugar)[0,1] if np.std(q[predictor])>1e-10 else np.nan,'NIR_MAE':np.abs(q[predictor]-q.nir).mean()})
            for baseline in ['nir','global_mean','cultivar_mean']:
                delta=np.abs(f.ensemble-f.sugar)-np.abs(f[baseline]-f.sugar);lo,hi=ci_mean(delta,groups)
                paired.append({**base,'comparison':'ensemble_minus_'+baseline,'n':len(f),'clusters':len(np.unique(groups)),'delta_MAE':delta.mean(),'low':lo,'high':hi})
            for j,(_,q) in enumerate(f.groupby(cluster)):
                row={**base,'cluster_index':j,'n':len(q)}
                for pred in ['ensemble','nir','global_mean','cultivar_mean']:row[pred+'_absolute_error_sum']=float(np.abs(q[pred]-q.sugar).sum())
                clusters.append(row)
            for nominal in [float(2*norm.cdf(1)-1),.9,.95]:
                for method in ['Gaussian','proxy_calibrated','reference_calibrated']:
                    if method=='Gaussian':mult=norm.ppf((1+nominal)/2);ncal=0
                    else:
                        c=sets['cal','train'] if method=='proxy_calibrated' else cal;y=c.nir if method=='proxy_calibrated' else c.sugar
                        ratio=(np.abs(y-c.ensemble)/c.total).to_numpy();k=min(len(ratio),int(np.ceil((len(ratio)+1)*nominal)));mult=np.sort(ratio)[k-1];ncal=len(c)
                    for target in ['nir','sugar']:
                        covered=(np.abs(f[target]-f.ensemble)<=mult*f.total).to_numpy();lo,hi=ci_mean(covered.astype(float),groups)
                        coverage.append({**base,'method':method,'nominal':nominal,'target':target,'n':len(f),'clusters':len(np.unique(groups)),'calibration_n':ncal,'multiplier':mult,'coverage':covered.mean(),'coverage_low':lo,'coverage_high':hi,'mean_width':2*mult*f.total.mean()})
            if base['inference_role']!='primary':continue
            scores={k:f[k].to_numpy() for k in ['aleatoric','mc_dropout','between_seed','total']}
            scores.update(model_NIR_disagreement=np.abs(f.ensemble-f.nir).to_numpy(),global_NIR_deviation=np.abs(f.nir-gm).to_numpy(),cultivar_NIR_deviation=np.abs(f.nir-f.kind.map(cm)).to_numpy())
            errors={'model_proxy':np.abs(f.ensemble-f.nir).to_numpy(),'model_reference':np.abs(f.ensemble-f.sugar).to_numpy(),'proxy_reference':np.abs(f.nir-f.sugar).to_numpy()}
            for target,error in errors.items():
                large=error>2
                for name,score in scores.items():
                    rho,lo,hi=rho_ci(score,error,groups)
                    for fraction in [.05,.1,.2]:
                        weights=topweights(score,fraction)
                        uncertainty.append({**base,'error_target':target,'score':name,'tautological':target=='model_proxy' and name=='model_NIR_disagreement','fraction':fraction,'n':len(f),'clusters':len(np.unique(groups)),'rho':rho,'rho_low':lo,'rho_high':hi,'lift':np.average(error,weights=weights)/error.mean(),'large_errors':int(large.sum()),'capture_large_error':float((weights*large).sum()/large.sum()) if large.sum() else np.nan,'retained_MAE':np.average(error,weights=1-weights)})
                    for fraction in np.arange(.05,1.001,.05):
                        weights=topweights(score,float(fraction));curves.append({**base,'error_target':target,'score':name,'fraction':fraction,'capture':(weights*large).sum()/large.sum() if large.sum() else np.nan})
            print('FINISHED',scheme,cohort,cluster,flush=True)
        for seed in range(3):
            q=read(O/f'runs/{scheme}_seed{seed}/preds_test_val.csv');q=q[q.objkey.isin(f.objkey)]
            seeds.append({'scheme':scheme,'cohort':cohort,'seed':seed,'n':len(q),'reference_MAE':np.abs(q.pred-q.sugar).mean(),'NIR_MAE':np.abs(q.pred-q.nir).mean(),'mean_MC_sd':q.mc_std.mean(),'min_MC_sd':q.mc_std.min(),'mean_aleatoric_sd':q.sigma_alea.mean()})
        for cultivar,q in f.groupby('kind'):
            scores={k:q[k] for k in ['aleatoric','mc_dropout','between_seed','total']}
            scores.update(model_NIR_disagreement=np.abs(q.ensemble-q.nir),global_NIR_deviation=np.abs(q.nir-gm),cultivar_NIR_deviation=np.abs(q.nir-q.cultivar_mean))
            errors={'model_proxy':np.abs(q.ensemble-q.nir),'model_reference':np.abs(q.ensemble-q.sugar),'proxy_reference':np.abs(q.nir-q.sugar)}
            for target,error in errors.items():
                for name,score in scores.items():within.append({'scheme':scheme,'cohort':cohort,'cultivar':cultivar,'n':len(q),'trees':q.tree_key.nunique(),'farms':q.pos.nunique(),'score':name,'error_target':target,'rho':float(spearmanr(score,error).statistic),'status':'descriptive_within_cultivar_no_superiority_claim'})
        prediction_sets.append(f.assign(scheme=scheme,cohort=cohort))
ref=read(O/'private/crops_v4_oriented/val_meta.csv',dtype={'tree':str,'pos':str})
for cohort in ['same_row_complete','strict_unique_NIR']:
    f=ref if cohort=='same_row_complete' else ref[ref.unique_NIR_count==1]
    for cultivar in ['ALL']+sorted(f.kind.unique()):
        q=f if cultivar=='ALL' else f[f.kind==cultivar];err=q.nir-q.sugar
        labels.append({'cohort':cohort,'cultivar':cultivar,'n':len(q),'trees':q.tree_key.nunique(),'NIR_bias':err.mean(),'NIR_MAE':np.abs(err).mean(),'pearson_r':np.corrcoef(q.nir,q.sugar)[0,1],'large_errors':int((np.abs(err)>2).sum())})
for rows,name in [(inventory,'DataS01_split_inventory.csv'),(accuracy,'DataS02_accuracy.csv'),(paired,'DataS05_paired_accuracy.csv'),(uncertainty,'DataS03_uncertainty_three_targets.csv'),(coverage,'DataS04_interval_coverage.csv'),(seeds,'DataS06_seed_metrics.csv'),(labels,'DataS07_full_reference_label_audit.csv'),(curves,'DataS08_capture_curves.csv'),(clusters,'DataS09_cluster_error_sums.csv'),(within,'DataS10_within_cultivar_scores.csv'),(overlap,'DataS11_farm_overlap.csv')]:table(rows,name)
save(O/'private/test_predictions_internal.csv',pd.concat(prediction_sets).to_csv(index=False))
# Independently reconstruct aggregate MAE and paired bootstrap from anonymous sums.
checks=[];cs=pd.DataFrame(clusters);acc=pd.DataFrame(accuracy);pa=pd.DataFrame(paired)
for (scheme,cohort,unit),g in cs.groupby(['scheme','cohort','cluster_unit']):
    a=acc[(acc.scheme==scheme)&(acc.cohort==cohort)&(acc.cluster_unit==unit)&(acc.cultivar=='ALL')]
    for pred in ['ensemble','nir','global_mean','cultivar_mean']:
        expected=float(a[a.predictor==pred].MAE.iloc[0]);actual=g[pred+'_absolute_error_sum'].sum()/g.n.sum();assert abs(expected-actual)<1e-12
    for b in ['nir','global_mean','cultivar_mean']:
        r=pa[(pa.scheme==scheme)&(pa.cohort==cohort)&(pa.cluster_unit==unit)&(pa.comparison=='ensemble_minus_'+b)].iloc[0]
        sums=g.ensemble_absolute_error_sum.to_numpy()-g[b+'_absolute_error_sum'].to_numpy();counts=g.n.to_numpy();ix=np.random.default_rng(20261026).integers(len(g),size=(5000,len(g)));dist=sums[ix].sum(1)/counts[ix].sum(1);lo,hi=np.quantile(dist,[.025,.975]);assert abs(sums.sum()/counts.sum()-r.delta_MAE)<1e-12 and abs(lo-r.low)<1e-12 and abs(hi-r.high)<1e-12
    checks.append({'scheme':scheme,'cohort':cohort,'cluster_unit':unit,'anonymous_MAE_and_paired_CI_reconstruction':'pass','tolerance':1e-12})
js(O/'results/aggregate_verification.json',checks)
js(O/'results/analysis_metadata.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'accuracy_bootstrap':5000,'rho_bootstrap':1000,'random_seed':20261026,'elapsed_seconds':time.time()-START,'completed_schemes':['tree','zone'],'historical_LOCO_included':False,'historical_LOCO_exclusion_reason':'Not rerun on corrected source/crop pipeline','zone_primary_cluster':'farm code; five test farms already shared with development','tree_additional_sensitivity_cluster':'farm code, complete accuracy/paired/coverage outputs','analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
js(O/'reproducibility/analysis_source_checksums_private.json',[{'path':str(p.relative_to(O)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(SOURCES))])
print(pd.DataFrame(accuracy).query("cultivar=='ALL' and inference_role=='primary'").to_string(index=False),flush=True)
print('ANALYSIS_COMPLETE',time.time()-START,flush=True)
