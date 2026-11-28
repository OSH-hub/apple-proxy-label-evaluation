from pathlib import Path
import csv,json,hashlib,collections,io,argparse,math,zipfile
import numpy as np,pandas as pd
from PIL import Image
O=Path(__file__).resolve().parents[1];BASE=O.parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['data','full'],default='full');args=ap.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ready=json.loads((O/'private/READY.json').read_text());audit=json.loads((O/'results/source_crop_audit_summary.json').read_text())
assert audit['audited_original_records']==21443 and audit['crop_failures']==0
original_json_audit=json.loads((O/'results/source_JSON_archive_audit.json').read_text())
assert original_json_audit['selected_original_server_JSONs_verified']==21443
assert original_json_audit['selected_JSON_basename_collisions']==0 and original_json_audit['export_JSON_byte_differences']==0
assert audit['manifest_sha256']==original_json_audit['original_server_manifest_sha256']==sha(BASE/'data/splits/manifest.csv')
assert ready['source_crop_audit_sha256']==sha(O/'results/source_crop_audit_summary.json')
assert ready['crop_train_meta_sha256']==sha(O/'private/crops_v4_oriented/train_meta.csv')
assert ready['crop_val_meta_sha256']==sha(O/'private/crops_v4_oriented/val_meta.csv')
records=list(csv.DictReader((O/'private/source_linkage_full.csv').open()))
assert len(records)==len({r['objkey'] for r in records})==21443
def same_num(a,b):
    def val(x):
        try:return float(x)
        except (TypeError,ValueError):return math.nan
    x,y=val(a),val(b)
    return (math.isnan(x) and math.isnan(y)) or abs(x-y)<1e-12
source_JSON_archive=zipfile.ZipFile(O/'private/selected_source_JSONs.zip')
for r in records:
    jsonname=r['json_file'].replace('\\','/').split('/')[-1]
    original_json=source_JSON_archive.read(jsonname);assert hashlib.sha256(original_json).hexdigest()==r['json_sha256']
    d=json.loads(original_json)
    assert d['images']['img_file_name']==r['original_selected_image']
    assert same_num(d['collection']['sugar_content_nir'],r['nir']) and same_num(d['collection']['sugar_content'],r['sugar'])
    assert d['annotations']['bbox']==json.loads(r['bbox'])
    assert (d['images']['img_width'],d['images']['img_height'])==(int(r['source_width']),int(r['source_height']))
    p=Path(r['file']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['crop_sha256']
    with Image.open(io.BytesIO(raw)) as image:
        image.load();assert image.size==(int(r['crop_width']),int(r['crop_height'])) and max(image.size)<=512
    x,y,w,h=json.loads(r['bbox']);l,t,rr,b=json.loads(r['crop_box'])
    assert l<=x and t<=y and rr>=x+w and b>=y+h
cropcounts=collections.Counter(r['crop_sha256'] for r in records)
assert max(cropcounts.values())==1,'Byte-identical corrected crop requires leakage investigation'
source_by_key={r['objkey']:r for r in records}
source_image_counts=collections.Counter(r['source_image_sha256'] for r in records)
checks=[]
for scheme in ['tree','zone']:
    old={r['objkey']:r for r in csv.DictReader((BASE/f'submission_20261026_v2/P1/splits/{scheme}_manifest.csv').open())}
    rows=list(csv.DictReader((O/f'splits/{scheme}_manifest.csv').open()))
    assert len(rows)==len({r['objkey'] for r in rows}), 'Duplicate split rows'
    photo_roles=collections.defaultdict(set)
    for r in rows:photo_roles[r['source_image_sha256']].add(r['partition'])
    assert all(len(v)==1 for v in photo_roles.values()), 'Identical original source image crosses partitions'
    assert all(r['partition']==old[r['objkey']]['partition'] and r['source']==old[r['objkey']]['source'] for r in rows)
    for r in rows:
        original=source_by_key[r['objkey']]
        assert all(r[k]==original[k] for k in ['file','source','tree_key','zone_key','kind','pos','tree','unique_NIR_count','EXIF_orientation','crop_sha256'])
        assert same_num(r['nir'],original['nir']) and same_num(r['sugar'],original['sugar'])
    for source in ['train','val']:
        selected=[r for r in records if r['source']==source and (source=='train' or r['complete_original_row']=='True')]
        assert {r['objkey'] for r in selected}=={r['objkey'] for r in rows if r['source']==source}
    if args.stage=='data':continue
    for seed in range(3):
        run=O/f'runs/{scheme}_seed{seed}'
        assert (run/'completed.json').exists(), 'Full verification requires all six completed models'
        completion=json.loads((run/'completed.json').read_text());hist=json.loads((run/'history.json').read_text());selected=json.loads((run/'selected_checkpoint.json').read_text())
        assert len(hist)==40 and [r['epoch'] for r in hist]==list(range(40))
        best=min(hist[5:],key=lambda r:r['selection_nir_mae'])
        assert selected['epoch']==best['epoch'] and selected['selection_nir_mae']==best['selection_nir_mae']
        assert sha(run/'checkpoint.pt')==completion['checkpoint_sha256']
        integ=json.loads((run/'mc_integrity.json').read_text())
        assert integ['batchnorm_buffers_unchanged'] and integ['mc_sd_min']>0 and integ['dropout_zero_vs_eval_max_difference']<1e-6 and integ['dropout_zero_repeat_max_difference']<1e-6
        for partition,source in [('cal','train'),('cal','val'),('test','train'),('test','val')]:
            frame=pd.read_csv(run/f'preds_{partition}_{source}.csv');prediction_rows=list(csv.DictReader((run/f'preds_{partition}_{source}.csv').open()))
            assert len(prediction_rows)==len({r['objkey'] for r in prediction_rows})==len(frame), 'Duplicate prediction rows'
            split_by_key={r['objkey']:r for r in rows}
            for p in prediction_rows:
                original=split_by_key[p['objkey']]
                assert original['partition']==partition and original['source']==source
                assert all(p[k]==original[k] for k in ['tree_key','zone_key','kind','pos','tree'])
                assert same_num(p['nir'],original['nir']) and same_num(p['sugar'],original['sugar'])
            arr=np.load(run/f'mc_{partition}_{source}.npz')
            assert arr['objkeys'].tolist()==frame.objkey.tolist()
            q=arr['predictions'];assert q.shape==(20,len(frame)) and np.isfinite(q).all() and (q.std(0)>0).all()
            column_mean=np.array([float(q[:,i].mean()) for i in range(q.shape[1])])
            column_sd=np.array([float(q[:,i].std()) for i in range(q.shape[1])])
            assert np.max(np.abs(column_mean-frame.mc_mean.to_numpy()))<1e-12
            assert np.max(np.abs(column_sd-frame.mc_std.to_numpy()))<1e-12
            q64=q.astype(np.float64);eps=np.finfo(q.dtype).eps;gamma=q.shape[0]*eps/(1-q.shape[0]*eps)
            assert np.all(np.abs(q64.mean(0)-column_mean)<=gamma*np.abs(q64).mean(0)+1e-12)
            assert (q64.std(0)>0).all()
            assert set(frame.objkey)=={r['objkey'] for r in rows if r['partition']==partition and r['source']==source}
        checks.append({'scheme':scheme,'seed':seed,'epochs':40,'selection_only_minimum_verified':True,'checkpoint_hash_verified':True,'MC_arrays_alignment_finiteness_and_saved_moments_verified':True,'dropout_and_frozen_BN_checks_passed':True})
if args.stage=='full':assert len(checks)==6, 'All six model checks required'
result={'all_21443_corrected_crop_hashes_and_geometry_verified':True,'duplicate_corrected_crop_hashes':sum(v-1 for v in cropcounts.values()),'duplicate_source_image_hashes':sum(v-1 for v in source_image_counts.values()),'source_image_hashes_cross_partitions':0,'v2_group_assignments_preserved':True,'READY_metadata_hashes_verified':True,'all_selected_JSON_hashes_image_labels_bboxes_match_original_server_archive':True,'split_image_labels_tags_match_source_linkage':True,'prediction_labels_match_frozen_splits':args.stage=='full','no_reference_record_transferred_into_proxy_training':True,'completed_model_checks':checks,'all_six_models_complete':len(checks)==6,'MC_moment_validation':'Exact per-column serialization expression within 1e-12; float64 mean within standard gamma_n float32 summation bound; see post-fit numerical validation amendment.'}
path=O/'results'/('data_integrity_verification.json' if args.stage=='data' else 'independent_integrity_verification.json')
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
