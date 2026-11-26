"""Publication figures from aggregate CSVs only; no raw records or manuscript writes."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
P=Path(__file__).resolve().parents[1];(P/'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(name):
    plt.tight_layout()
    for ext in ['png','pdf']:
        f=P/'figures'/f'{name}.{ext}'
        if f.exists():f.read_bytes()
        plt.savefig(f,dpi=600,bbox_inches='tight')
        if ext=='png':
            with Image.open(f) as image:rgb=image.convert('RGB')
            rgb.save(f,dpi=(600,600))
    plt.close()
acc=pd.read_csv(P/'tables/accuracy.csv');a=acc[(acc.cultivar=='ALL')&(acc.cohort=='same_row_complete')&(acc.inference_role=='primary')]
names={'ensemble':'RGB ensemble','nir':'NIR proxy','global_mean':'Fit NIR global mean','cultivar_mean':'Fit NIR cultivar mean'}
fig,axs=plt.subplots(1,2,figsize=(7.5,3.6),sharey=True)
for ax,scheme,title in zip(axs,['tree','zone'],['(a) Held-out trees','(b) Held-out cultivar–farm\ncombinations (5 shared farms)']):
    q=a[a.scheme==scheme].set_index('predictor').reindex(names)
    x=np.arange(len(q));ax.errorbar(q.MAE,x,xerr=np.stack([q.MAE-q.MAE_low,q.MAE_high-q.MAE]),fmt='o',capsize=3,color='#277e8e');ax.set(yticks=x,yticklabels=list(names.values()),xlabel='Reference MAE (°Brix)',title=title);ax.grid(axis='x',alpha=.2)
save('figure1_independent_accuracy')
uq=pd.read_csv(P/'tables/uncertainty_three_targets.csv');uq=uq[(uq.cohort=='same_row_complete')&(uq.inference_role=='primary')];scores=['aleatoric','mc_dropout','between_seed','total','model_NIR_disagreement','global_NIR_deviation','cultivar_NIR_deviation'];labels=['Aleatoric SD','MC-dropout SD','Between-seed SD','Total SD','Model–NIR difference','Global NIR deviation','Cultivar NIR deviation']
fig,axs=plt.subplots(1,3,figsize=(8,4.5),sharey=True)
for ax,target,title in zip(axs,['model_proxy','model_reference','proxy_reference'],['(a) |RGB − NIR|','(b) |RGB − reference|','(c) |NIR − reference|']):
    q=uq[(uq.scheme=='tree')&(uq.error_target==target)&(uq.fraction==.1)].set_index('score').reindex(scores)
    for j,r in enumerate(q.itertuples()):
        if r.tautological:ax.text(.05,j,'identity',va='center',fontsize=8,color='gray');continue
        ax.errorbar(r.rho,j,xerr=[[r.rho-r.rho_low],[r.rho_high-r.rho]],fmt='o',capsize=2,color='#277e8e')
    ax.axvline(0,color='gray',ls='--',lw=.8);ax.set(yticks=np.arange(len(scores)),yticklabels=labels,xlim=(-1,1),xlabel='Spearman correlation',title=title)
    ax.set_xticks([-1,-.5,0,.5,1])
save('figure2_three_error_targets')
cov=pd.read_csv(P/'tables/interval_coverage.csv');cov=cov[(cov.cohort=='same_row_complete')&(cov.inference_role=='primary')];fig,axs=plt.subplots(1,3,figsize=(8,3.5),sharey=True)
for ax,method,title in zip(axs,['Gaussian','proxy_calibrated','reference_calibrated'],['(a) Uncalibrated\nGaussian','(b) Proxy-residual\ncalibrated','(c) Reference-residual\ncalibrated']):
    for target,color,label in [('nir','#277e8e','NIR proxy'),('sugar','#c77832','Reference')]:
        q=cov[(cov.scheme=='tree')&(cov.method==method)&(cov.target==target)].sort_values('nominal');ax.plot(q.nominal,q.coverage,'o-',color=color,label=label)
    ax.plot([.65,1],[.65,1],'--',color='gray',lw=.8);ax.set(xlabel='Nominal probability',title=title,xlim=(.65,1),ylim=(0,1.02));ax.set_xticks([.6827,.9,.95]);ax.set_xticklabels(['0.683','0.90','0.95'],fontsize=8)
axs[0].set_ylabel('Observed coverage');axs[0].legend(frameon=False,fontsize=8);save('figure4_independent_calibration')
curve=pd.read_csv(P/'tables/capture_curves.csv');curve=curve[(curve.cohort=='same_row_complete')&(curve.inference_role=='primary')];fig,ax=plt.subplots(figsize=(7,4))
for score,label in [('total','Total SD'),('mc_dropout','MC-dropout SD'),('model_NIR_disagreement','Model–NIR difference'),('cultivar_NIR_deviation','Fit cultivar NIR deviation')]:
    q=curve[(curve.scheme=='tree')&(curve.error_target=='proxy_reference')&(curve.score==score)];ax.plot(q.fraction,q.capture,label=label)
ax.plot([0,1],[0,1],'--',color='gray',label='Random expectation');ax.set(xlabel='Fraction selected',ylabel='Fraction of proxy errors >2 °Brix captured',xlim=(0,1),ylim=(0,1));ax.legend(frameon=False,fontsize=8);save('figure3_proxy_error_capture')
