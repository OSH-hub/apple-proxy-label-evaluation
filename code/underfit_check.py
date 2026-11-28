import os
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import argparse, csv, json, hashlib, random, time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

SETTINGS = {'LR10': dict(lr=(1e-4, 1e-3), epochs=40, res=224), 'EP120': dict(lr=(1e-5, 1e-4), epochs=120, res=224),
            'RES448': dict(lr=(1e-5, 1e-4), epochs=40, res=448), 'LR10_RES448': dict(lr=(1e-4, 1e-3), epochs=40, res=448)}
torch.set_num_threads(2); torch.set_num_interop_threads(1)


class CropDS(Dataset):
    def __init__(self, rows, res, train=False):
        self.rows = rows
        aug = [T.RandomResizedCrop(res, scale=(.6, 1)), T.RandomHorizontalFlip(), T.ColorJitter(.2, .2, .2)] if train else [T.Resize(res * 8 // 7), T.CenterCrop(res)]
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        with Image.open(self.rows[i]['file']) as im: x = self.tf(im.convert('RGB'))
        return x, torch.tensor(float(self.rows[i]['nir']), dtype=torch.float32), i


class Net(nn.Module):
    def __init__(self, drop=.2):
        super().__init__(); self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0, drop_rate=drop); self.head = nn.Linear(2048, 2)
    def forward(self, x):
        o = self.head(self.backbone(x)); return o[:, 0], o[:, 1].clamp(-6, 3)


def loader(rows, res, train=False, aug=None):
    return DataLoader(CropDS(rows, res, train if aug is None else aug), batch_size=128, shuffle=train, num_workers=4, pin_memory=True, drop_last=train and len(rows) >= 128, persistent_workers=train)


@torch.no_grad()
def predict(model, rows, res, mean, scale):
    model.eval(); mu = np.empty(len(rows))
    for x, _, idx in loader(rows, res):
        with torch.autocast('cuda'): a, _ = model(x.cuda(non_blocking=True))
        mu[idx.numpy()] = a.float().cpu().numpy()
    return mu * scale + mean


def mae(p, y): return float(np.abs(p - y).mean())
def corr(p, y): return float(np.corrcoef(p, y)[0, 1])
def nir(rows): return np.array([float(r['nir']) for r in rows])


def groups_of(p1):
    rows = list(csv.DictReader((p1 / 'splits/tree_manifest.csv').open()))
    assert len(rows) == len({r['objkey'] for r in rows}) and all(r['status'] == 'ok' for r in rows)
    g = {(p, s): [r for r in rows if r['partition'] == p and r['source'] == s] for p in ['fit', 'select', 'cal', 'test'] for s in ['train', 'val']}
    sets = {'fit': g['fit', 'train'], 'select': g['select', 'train'], 'proxy_test': g['test', 'train'], 'reference_test': [r for r in g['test', 'val'] if r['sugar'] not in ('', None)]}
    assert (len(sets['fit']), len(sets['select']), len(sets['proxy_test']), len(sets['reference_test'])) == (8919, 1643, 3197, 1191)
    return sets


def save_preds(path, sets, preds):
    with path.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['set', 'objkey', 'kind', 'tree_key', 'nir', 'sugar', 'pred'])
        for name, rows in sets.items():
            if name in preds:
                for r, p in zip(rows, preds[name]): w.writerow([name, r['objkey'], r['kind'], r['tree_key'], r['nir'], r['sugar'], f'{p:.6f}'])


def metrics(sets, preds):
    out = {}
    for name, p in preds.items():
        y = nir(sets[name]); out[name] = {'n': len(y), 'nir_mae': mae(p, y), 'nir_r': corr(p, y)}
        if name == 'reference_test':
            s = np.array([float(r['sugar']) for r in sets[name]]); out[name].update(reference_mae=mae(p, s), reference_r=corr(p, s))
    return out


def train(sets, cfg, seed, out, subset=None, drop=.2, aug=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    tr = sets['fit'] if subset is None else sets['fit'][:subset]; ho = sets['select']; res = cfg['res']; E = cfg['epochs']
    ys = nir(sets['fit']); mean, scale = float(ys.mean()), float(ys.std())
    model = Net(drop).cuda()
    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': cfg['lr'][0]}, {'params': model.head.parameters(), 'lr': cfg['lr'][1]}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E); scaler = torch.amp.GradScaler(); dl = loader(tr, res, True, aug); best = float('inf'); history = []; sel = None
    for ep in range(E):
        t = time.time(); model.train(); total = 0.; n = 0
        for x, y, _ in dl:
            x = x.cuda(non_blocking=True); y = (y.cuda(non_blocking=True) - mean) / scale; opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                mu, lv = model(x); loss = nn.functional.huber_loss(mu, y) if ep < 5 else .5 * (lv + (y - mu).square() / lv.exp()).mean()
            assert torch.isfinite(loss), f'nonfinite loss at epoch {ep}'
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total += float(loss) * len(y); n += len(y)
        sched.step(); rec = {'epoch': ep, 'loss': total / n, 'seconds': time.time() - t}
        if subset is None:
            m = mae(predict(model, ho, res, mean, scale), nir(ho)); rec['selection_nir_mae'] = m
            if ep >= 5 and m < best: best = m; sel = {'epoch': ep, 'selection_nir_mae': m}; torch.save(model.state_dict(), out / 'checkpoint.pt')
        elif ep % 20 == 19 or ep == E - 1: rec['train_subset_nir_mae'] = mae(predict(model, tr, res, mean, scale), nir(tr))
        history.append(rec); print(json.dumps(rec), flush=True); (out / 'history.json').write_text(json.dumps(history, indent=1))
    if subset is None: model.load_state_dict(torch.load(out / 'checkpoint.pt', weights_only=True))
    return model, mean, scale, sel, history


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--p1', required=True); ap.add_argument('--out', required=True); ap.add_argument('--task', required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--plan'); a = ap.parse_args()
    p1 = Path(a.p1); sets = groups_of(p1); tag = a.task if a.task in ('d2', 'd3') else f'{a.task}_seed{a.seed}'
    out = Path(a.out) / tag; out.mkdir(parents=True, exist_ok=False); t0 = time.time()
    info = {'task': a.task, 'seed': a.seed, 'plan_sha256': hashlib.sha256(Path(a.plan).read_bytes()).hexdigest() if a.plan else None, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256((p1 / 'splits/tree_manifest.csv').read_bytes()).hexdigest(), 'torch': torch.__version__, 'timm': timm.__version__, 'gpu': torch.cuda.get_device_name(0)}
    if a.task == 'd2':
        ys = nir(sets['fit']); mean, scale = float(ys.mean()), float(ys.std()); res = {}
        for s in (0, 1, 2):
            cfgp = json.loads((p1 / f'runs/tree_seed{s}/run_config.json').read_text()); assert abs(cfgp['ymean'] - mean) < 1e-9 and abs(cfgp['ystd'] - scale) < 1e-9
            model = Net().cuda(); model.load_state_dict(torch.load(p1 / f'runs/tree_seed{s}/checkpoint.pt', weights_only=True))
            preds = {k: predict(model, v, 224, mean, scale) for k, v in sets.items()}; save_preds(out / f'preds_seed{s}.csv', sets, preds); res[f'seed{s}'] = metrics(sets, preds)
            old = {r['objkey']: float(r['pred']) for r in csv.DictReader((p1 / f'runs/tree_seed{s}/preds_test_val.csv').open())}
            diff = max(abs(old[r['objkey']] - p) for r, p in zip(sets['reference_test'], preds['reference_test'])); res[f'seed{s}']['max_abs_difference_from_saved_primary_predictions'] = diff
        info['results'] = res
    elif a.task == 'd3':
        cfg = dict(lr=(1e-4, 1e-3), epochs=200, res=224); model, mean, scale, _, hist = train(sets, cfg, 0, out, subset=512, drop=0., aug=False)
        p = predict(model, sets['fit'][:512], 224, mean, scale); y = nir(sets['fit'][:512])
        info['results'] = {'n': 512, 'final_train_subset_nir_mae': mae(p, y), 'final_train_subset_nir_r': corr(p, y), 'subset_nir_sd': float(y.std()), 'constant_mean_mae': mae(np.full_like(y, y.mean()), y)}
    else:
        cfg = SETTINGS[a.task]; model, mean, scale, sel, hist = train(sets, cfg, a.seed, out)
        preds = {k: predict(model, v, cfg['res'], mean, scale) for k, v in sets.items()}; save_preds(out / 'preds.csv', sets, preds)
        info.update(setting=cfg, selected=sel, results=metrics(sets, preds)); (out / 'checkpoint.pt').unlink()
    info['total_seconds'] = time.time() - t0; (out / 'completed.json').write_text(json.dumps(info, indent=2)); print('COMPLETED', tag, json.dumps(info.get('results'))[:600], flush=True)


if __name__ == '__main__':
    main()
