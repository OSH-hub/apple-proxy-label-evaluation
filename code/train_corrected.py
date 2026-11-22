"""Model fitting on regenerated crops; preserved group assignments and hyperparameters."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import argparse,csv,json,hashlib,random,time,platform,subprocess
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
import timm

BASE=Path('/path/to/project_root')
OUT=BASE/'submission_20261026_v4/P1'
torch.set_num_threads(2);torch.set_num_interop_threads(1)
def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.read_bytes()
    path.write_text(text,encoding='utf-8')
def js(path,obj):write(path,json.dumps(obj,ensure_ascii=False,indent=2))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stable(k):return hashlib.sha256(('20261026:'+k).encode()).hexdigest()
def load():
    rows=[]
    for source in ['train','val']:
        p=OUT/f'private/crops_v4_oriented/{source}_meta.csv'
        for r in csv.DictReader(p.open()):
            assert r['status'] in ['ok','skip'] and Path(r['file']).is_file()
            r['source']=source;r['tree_key']='|'.join(r[k] for k in ['kind','pos','tree']);r['zone_key']='|'.join(r[k] for k in ['kind','pos'])
            rows.append(r)
    assert len({r['objkey'] for r in rows})==len(rows)
    return rows
def assign(rows,scheme):
    assignment={}
    for kind in sorted({r['kind'] for r in rows}):
        kr=[r for r in rows if r['kind']==kind]
        if scheme=='tree':
            keys=sorted({r['tree_key'] for r in kr},key=stable)
            nt=max(1,round(len(keys)*.2));ns=max(1,round(len(keys)*.1));nc=max(1,round(len(keys)*.1))
            for i,k in enumerate(keys):assignment[k]='test' if i<nt else 'select' if i<nt+ns else 'cal' if i<nt+ns+nc else 'fit'
        else:
            zones=sorted({r['zone_key'] for r in kr},key=stable);test=set(zones[:max(1,round(len(zones)*.2))])
            for r in kr:
                if r['zone_key'] in test:assignment[r['tree_key']]='test'
            keys=sorted({r['tree_key'] for r in kr if r['zone_key'] not in test},key=stable)
            ns=max(1,round(len(keys)*.1));nc=max(1,round(len(keys)*.1))
            for i,k in enumerate(keys):assignment[k]='select' if i<ns else 'cal' if i<ns+nc else 'fit'
    result=[dict(r,partition=assignment[r['tree_key']]) for r in rows]
    for kind in sorted({r['kind'] for r in result}):
        for p in ['fit','select','cal','test']:assert any(r['kind']==kind and r['partition']==p for r in result),(kind,p)
    for group in ['tree_key']+(['zone_key'] if scheme=='zone' else []):
        train={r[group] for r in result if r['partition']!='test'};test={r[group] for r in result if r['partition']=='test'}
        assert not train&test
    return result
class CropDS(Dataset):
    def __init__(self,rows,train=False):
        self.rows=rows
        self.tf=T.Compose(([T.RandomResizedCrop(224,scale=(.6,1)),T.RandomHorizontalFlip(),T.ColorJitter(.2,.2,.2)] if train else [T.Resize(256),T.CenterCrop(224)])+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['file']) as im:x=self.tf(im.convert('RGB'))
        return x,torch.tensor(float(self.rows[i]['nir']),dtype=torch.float32),i
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=timm.create_model('resnet50',pretrained=True,num_classes=0,drop_rate=.2);self.head=nn.Linear(2048,2)
    def forward(self,x):
        o=self.head(self.backbone(x));return o[:,0],o[:,1].clamp(-6,3)
def loader(rows,train=False):
    return DataLoader(CropDS(rows,train),batch_size=128,shuffle=train,num_workers=4,pin_memory=True,drop_last=train,persistent_workers=True)
@torch.no_grad()
def predict(model,rows,mean,scale,passes=0):
    dl=loader(rows);model.eval();mu=np.empty(len(rows));lv=np.empty(len(rows));mc=np.empty((passes,len(rows)),dtype=np.float32)
    for x,_,idx in dl:
        with torch.autocast('cuda'):a,b=model(x.cuda(non_blocking=True))
        mu[idx.numpy()]=a.float().cpu().numpy();lv[idx.numpy()]=b.float().cpu().numpy()
    if passes:
        model.train()
        for m in model.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        before={n:b.detach().clone() for n,b in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
        assert not any(m.training for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
        for j in range(passes):
            for x,_,idx in dl:
                with torch.autocast('cuda'):a,_=model(x.cuda(non_blocking=True))
                mc[j,idx.numpy()]=a.float().cpu().numpy()*scale+mean
        assert all(torch.equal(before[n],b) for n,b in model.named_buffers() if n in before),'BN changed during MC'
        assert (mc.std(axis=0)>0).any(),'MC dropout remains degenerate'
    model.eval()
    return mu*scale+mean,np.exp(.5*lv)*scale,mc
@torch.no_grad()
def integrity(model,rows):
    x=next(iter(loader(rows[:4])))[0].cuda();model.eval()
    a=model(x)[0];b=model(x)[0];det=float((a-b).abs().max())
    model.train()
    for m in model.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
    before={n:b.clone() for n,b in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
    samples=torch.stack([model(x)[0] for _ in range(5)])
    sd=samples.std(0,unbiased=False)
    unchanged=all(torch.equal(before[n],b) for n,b in model.named_buffers() if n in before)
    model.backbone.drop_rate=0.;p0=model(x)[0];p0b=model(x)[0];p0diff=float((p0-p0b).abs().max());p0eval=float((p0-a).abs().max())
    model.backbone.drop_rate=.2
    model.eval();assert det<1e-6 and bool((sd>0).all()) and unchanged and p0diff<1e-6 and p0eval<1e-6
    return {'deterministic_max_abs_difference':det,'mc_sd_min':float(sd.min()),'mc_sd_max':float(sd.max()),'batchnorm_buffers_unchanged':unchanged,'dropout_zero_repeat_max_difference':p0diff,'dropout_zero_vs_eval_max_difference':p0eval}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scheme',choices=['tree','zone'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert gpu in ['0','1'], 'Only expressly allocated GPUs 0 and 1 are allowed'
    assert (a.scheme=='tree' and gpu=='0') or (a.scheme=='zone' and gpu=='1')
    assert (OUT/'private/READY.json').exists(), 'Complete source/crop audit required'
    ready=json.loads((OUT/'private/READY.json').read_text())
    assert ready['crop_train_meta_sha256']==sha(OUT/'private/crops_v4_oriented/train_meta.csv')
    assert ready['crop_val_meta_sha256']==sha(OUT/'private/crops_v4_oriented/val_meta.csv')
    assert (OUT/'results/data_integrity_verification.json').exists(), 'Transferred data must pass independent validation'
    out=OUT/'runs'/f'{a.scheme}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():raise RuntimeError('Completed run already exists; will not overwrite')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    rows=list(csv.DictReader((OUT/f'splits/{a.scheme}_manifest.csv').open()))
    assert len(rows)==len({r['objkey'] for r in rows})
    assert all(Path(r['file']).is_file() and r['status']=='ok' for r in rows)
    import io
    groups={(p,s):[r for r in rows if r['partition']==p and r['source']==s] for p in ['fit','select','cal','test'] for s in ['train','val']}
    tr=groups['fit','train'];ho=groups['select','train'];ys=np.array([float(r['nir']) for r in tr]);mean=float(ys.mean());scale=float(ys.std())
    info={'scheme':a.scheme,'seed':a.seed,'ymean':mean,'ystd':scale,'n_proxy_fit':len(tr),'n_proxy_select':len(ho),'epochs':40,'gpu_name':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'torch':torch.__version__,'timm':timm.__version__,'python':platform.python_version(),'numpy':np.__version__,'source_sha256':{str(p):sha(p) for p in [OUT/'private/crops_v4_oriented/train_meta.csv',OUT/'private/crops_v4_oriented/val_meta.csv',OUT/f'splits/{a.scheme}_manifest.csv',OUT/'reproducibility/protocol_correction_v4.md',OUT/'reproducibility/protocol_source_orientation_amendment.md',OUT/'reproducibility/protocol_cluster_sensitivity_amendment.md',OUT/'reproducibility/rebuild_source_crops_local.py',Path(__file__)]},'reference_used_to_select_weights':False}
    js(out/'run_config.json',info);print(json.dumps(info),flush=True)
    model=Net().cuda();opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40);scaler=torch.amp.GradScaler();dl=loader(tr,True);best=float('inf');history=[];start=time.time()
    for ep in range(40):
        t=time.time();model.train();total=0.;n=0
        for x,y,_ in dl:
            x=x.cuda(non_blocking=True);y=(y.cuda(non_blocking=True)-mean)/scale;opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                mu,lv=model(x);loss=nn.functional.huber_loss(mu,y) if ep<5 else .5*(lv+(y-mu).square()/lv.exp()).mean()
            assert torch.isfinite(loss),f'Nonfinite loss at epoch{ep}'
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=float(loss)*len(y);n+=len(y)
        sched.step();pred,_,_=predict(model,ho,mean,scale);mae=float(np.abs(pred-np.array([float(r['nir']) for r in ho])).mean())
        rec={'epoch':ep,'loss':total/n,'selection_nir_mae':mae,'seconds':time.time()-t};history.append(rec);print(json.dumps(rec),flush=True);js(out/'history.json',history)
        if ep>=5 and mae<best:
            best=mae
            if (out/'checkpoint.pt').exists():(out/'checkpoint.pt').read_bytes()
            torch.save(model.state_dict(),out/'checkpoint.pt');js(out/'selected_checkpoint.json',{'epoch':ep,'selection_nir_mae':mae})
    model.load_state_dict(torch.load(out/'checkpoint.pt',weights_only=True));js(out/'mc_integrity.json',integrity(model,groups['test','val']))
    for p,s in [('cal','train'),('cal','val'),('test','train'),('test','val')]:
        q=groups[p,s];mu,sigma,mc=predict(model,q,mean,scale,20)
        buf=io.StringIO();fields=['objkey','kind','pos','tree','date','nir','sugar','tree_key','zone_key','pred','sigma_alea','mc_mean','mc_std'];w=csv.DictWriter(buf,fieldnames=fields);w.writeheader()
        for i,r in enumerate(q):w.writerow({**{k:r[k] for k in fields[:9]},'pred':mu[i],'sigma_alea':sigma[i],'mc_mean':float(mc[:,i].mean()),'mc_std':float(mc[:,i].std())})
        write(out/f'preds_{p}_{s}.csv',buf.getvalue());dest=out/f'mc_{p}_{s}.npz'
        if dest.exists():dest.read_bytes()
        np.savez_compressed(dest,predictions=mc,objkeys=np.array([r['objkey'] for r in q]))
        print('PREDICTION',p,s,len(q),'MC_NONZERO',int((mc.std(0)>0).sum()),flush=True)
    info.update(total_seconds=time.time()-start,checkpoint_sha256=sha(out/'checkpoint.pt'),best_selection_mae=best)
    js(out/'completed.json',info);print('COMPLETED',a.scheme,a.seed,info['total_seconds'],flush=True)
if __name__=='__main__':main()
