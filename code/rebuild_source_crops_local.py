from pathlib import Path
import csv, json, hashlib, io, time, threading, collections, zipfile, math, traceback, os, struct, zlib, binascii
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, __version__ as pillow_version, features

BASE=Path(__file__).resolve().parents[3]
OUT=Path(os.environ['P1_WORKER_OUT'])
REMOTE_OUT='/path/to/project_root/submission_20261026_v4/P1'
DEST=OUT/'private/crops_v4_oriented'
START=time.time()
def sha(b): return hashlib.sha256(b).hexdigest()
def write(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(f'Will not overwrite existing artifact: {path.name}')
    path.write_text(text,encoding='utf-8')
def js(path,obj): write(path,json.dumps(obj,ensure_ascii=False,indent=2))
def number(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError): return None
def equal(a,b): return (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-9)

assert not (OUT/'private/READY.json').exists(), 'Already completed'
scope={}
oldcode=(BASE/'pipeline/extract_crops.py').read_text()
exec(compile(oldcode,'historical_extract_crops.py','exec'),scope)
scope['MANIFEST']=str(BASE/'data/splits/manifest.csv')
scope['SRCDIR']=r'/path/to/AIHub_dataset_490/Training/source_data'
val,train=scope['select_items'](5000)
assert len(val)==6176 and len(train)==15267
items=[dict(x,source='train') for x in train]+[dict(x,source='val') for x in val]
keys={r['objkey'] for r in items};selected={};nirs=collections.defaultdict(set)
manifest_path=BASE/'data/splits/manifest.csv'
for rownum,r in enumerate(csv.DictReader(manifest_path.open()),2):
    k='|'.join(r[c] for c in ['date','kind','pos','tree','obj'])
    if k not in keys:continue
    n=number(r['nir'])
    if n is not None:nirs[k].add(n)
    score=0 if r['ang']=='0' else int(r['ang'] or 999)
    if k not in selected or score<selected[k]['score']:
        selected[k]=dict(r,score=score,source_row_number=rownum)
labels=collections.defaultdict(list)
for p in (OUT/'private/selected_labels').rglob('*.json'):labels[p.name].append(p)
gidx={};zipinfos={}
for kind,zips in scope['VARIETY_ZIPS'].items():
    idx=collections.defaultdict(list)
    for zn in zips:
        with zipfile.ZipFile(Path(scope['SRCDIR'])/zn) as z:
            for n in z.namelist():
                if n.lower().endswith(('.jpg','.jpeg','.png')):
                    idx[Path(n).name].append((zn,n));zipinfos[zn,n]=z.getinfo(n)
    gidx[kind]=idx
    print(json.dumps({'stage':'indexed','cultivar':kind,'source_images':len(idx),'elapsed':time.time()-START}),flush=True)
tls=threading.local()
def work(item):
    r=selected[item['objkey']]
    assert r['img']==item['img']
    bn=Path(item['img']).name
    stem=Path(bn).stem
    jsonname=stem if stem.endswith('.json') else stem+'.json'
    candidates=labels[jsonname]
    assert len(candidates)==1, 'Missing or ambiguous normalized JSON'
    jp=candidates[0];jbytes=jp.read_bytes();d=json.loads(jbytes)
    assert Path(d['images']['img_file_name']).name==bn, 'JSON image mismatch'
    nir=number(d['collection']['sugar_content_nir']);sugar=number(d['collection']['sugar_content'])
    assert equal(nir,number(r['nir'])) and equal(sugar,number(r['sugar'])), 'JSON/selected-row label mismatch'
    assert nir is not None,'Selected original row missing NIR'
    hits=gidx[item['kind']].get(bn,[]) or gidx[item['kind']].get(bn.replace('.json',''),[])
    assert len(hits)==1, 'Missing or ambiguous normalized source image'
    zn,entry=hits[0]
    if not hasattr(tls,'archives'):tls.archives={}
    if zn not in tls.archives:tls.archives[zn]=open(Path(scope['SRCDIR'])/zn,'rb',buffering=0)
    info=zipinfos[zn,entry]
    tls.archives[zn].seek(info.header_offset)
    compressed=tls.archives[zn].read(info.compress_size+131100)
    assert compressed[:4]==b'PK\x03\x04' and not info.flag_bits&1, 'Invalid/encrypted ZIP entry'
    fnlen,exlen=struct.unpack_from('<HH',compressed,26);offset=30+fnlen+exlen
    payload=compressed[offset:offset+info.compress_size]
    assert len(payload)==info.compress_size and info.compress_type in [0,8], 'Incomplete/unsupported ZIP payload'
    raw=zlib.decompress(payload,-15) if info.compress_type==8 else payload
    assert len(raw)==info.file_size and binascii.crc32(raw)&0xffffffff==info.CRC, 'ZIP CRC/size mismatch'
    with Image.open(io.BytesIO(raw)) as source:
        source.load();raw_dimensions=source.size;orientation=int(source.getexif().get(274,1));im=ImageOps.exif_transpose(source).convert('RGB')
    W,H=im.size
    assert (W,H)==(d['images']['img_width'],d['images']['img_height']),f'EXIF-oriented dimensions {(W,H)} do not match JSON {(d["images"]["img_width"],d["images"]["img_height"])}; orientation={orientation}'
    bb=d.get('annotations',{}).get('bbox')
    assert isinstance(bb,list) and len(bb)==4,'Missing bbox'
    assert all(number(v) is not None for v in bb),'Nonfinite bbox'
    x,y,w,h=[int(v) for v in bb]
    assert w>0 and h>0 and x>=0 and y>=0 and x+w<=W and y+h<=H,f'Invalid or out-of-bounds bbox: bbox={bb}, actual_dimensions={(W,H)}, JSON_dimensions={(d["images"].get("img_width"),d["images"].get("img_height"))}'
    pad=int(.05*max(w,h));left=max(0,x-pad);top=max(0,y-pad);right=min(W,x+w+pad);bottom=min(H,y+h+pad)
    assert left<=x and top<=y and right>=x+w and bottom>=y+h
    im=im.crop((left,top,right,bottom))
    before_resize=im.size
    if max(im.size)>512:
        factor=512/max(im.size);im=im.resize((max(1,int(im.width*factor)),max(1,int(im.height*factor))),Image.Resampling.LANCZOS)
    memory=io.BytesIO();im.save(memory,'JPEG',quality=90);data=memory.getvalue()
    with Image.open(io.BytesIO(data)) as check:
        check.load();assert check.size==im.size and check.mode=='RGB'
    path=DEST/item['source']/item['kind']/(item['objkey'].replace('|','_')+'.jpg')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==data, 'Existing v4 attempt crop differs from exact regeneration'
    else:
        with path.open('xb') as handle:handle.write(data)
    corrected=dict(item)
    corrected.update(file=REMOTE_OUT+'/'+path.relative_to(OUT).as_posix(),nir=nir,sugar=sugar,status='ok',source_row_number=r['source_row_number'],original_selected_image=bn,json_file=str(jp),json_sha256=sha(jbytes),source_zip=zn,source_entry=entry,source_image_sha256=sha(raw),crop_sha256=sha(data),source_width=W,source_height=H,raw_width=raw_dimensions[0],raw_height=raw_dimensions[1],EXIF_orientation=orientation,EXIF_transformed=orientation in [2,3,4,5,6,7,8],bbox=json.dumps(bb),crop_box=json.dumps([left,top,right,bottom]),pre_resize_width=before_resize[0],pre_resize_height=before_resize[1],crop_width=im.width,crop_height=im.height,json_embedded_version=d.get('info',{}).get('version'),unique_NIR_count=len(nirs[item['objkey']]),complete_original_row=(sugar is not None),historical_NIR_changed=not equal(nir,number(item['nir'])),historical_sugar_changed=not equal(sugar,number(item['sugar'])),historical_full_frame_fallback=Path(bn).stem.endswith('.json'),tree_key='|'.join(item[k] for k in ['kind','pos','tree']),zone_key='|'.join(item[k] for k in ['kind','pos']))
    return corrected

records=[];errors=[];lock=threading.Lock();tasks=collections.defaultdict(list)
order={r['objkey']:i for i,r in enumerate(items)}
for item in items:
    bn=Path(item['img']).name
    hits=gidx[item['kind']].get(bn,[]) or gidx[item['kind']].get(bn.replace('.json',''),[])
    assert len(hits)==1,'Missing/ambiguous image in pre-scheduling check'
    zn,entry=hits[0];tasks[zn].append((zipinfos[zn,entry].header_offset,item))
def archive_queue(pairs):
    for _,item in sorted(pairs,key=lambda x:x[0]):
        try:
            r=work(item)
            with lock:records.append(r)
        except Exception as exc:
            with lock:
                errors.append({'objkey':item['objkey'],'type':type(exc).__name__,'reason':str(exc)})
                with (OUT/'private/extraction_errors_orientation.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(errors[-1])+'\n')
                print(json.dumps({'stage':'error','type':type(exc).__name__,'reason':str(exc),'count':len(errors)}),flush=True)
        with lock:
            i=len(records)+len(errors)
            if i%500==0:print(json.dumps({'stage':'crop_audit','processed':i,'success':len(records),'errors':len(errors),'elapsed_seconds':round(time.time()-START,1)}),flush=True)
with ThreadPoolExecutor(max_workers=9) as pool:
    list(pool.map(archive_queue,tasks.values()))
records.sort(key=lambda r:order[r['objkey']])
if errors:
    js(OUT/'private/extraction_errors.json',errors)
    js(OUT/'results/extraction_failure_summary.json',{'audited':len(items),'passed':len(records),'failure_count':len(errors),'types':dict(collections.Counter(e['reason'] for e in errors))})
    raise RuntimeError('Source/crop failures require inspection; training not allowed')
assert len(records)==21443 and len({r['objkey'] for r in records})==21443
def csvwrite(path,rows):
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);write(path,buf.getvalue())
csvwrite(OUT/'private/source_linkage_full.csv',records)
for source in ['train','val']:
    subset=[r for r in records if r['source']==source and (source=='train' or r['complete_original_row'])]
    if source=='train':assert len(subset)==15267 and all(r['sugar'] is None for r in subset)
    csvwrite(DEST/f'{source}_meta.csv',subset)
for scheme in ['tree','zone']:
    p=BASE/f'submission_20261026_v2/P1/splits/{scheme}_manifest.csv'
    original={r['objkey']:r for r in csv.DictReader(p.open())}
    assert set(original)==keys
    subset=[dict(r,partition=original[r['objkey']]['partition']) for r in records if r['source']=='train' or r['complete_original_row']]
    assert all(r['source']==original[r['objkey']]['source'] for r in subset)
    csvwrite(OUT/f'splits/{scheme}_manifest.csv',subset)
summary={'audited_original_records':len(records),'source_JSON_image_label_matches':len(records),'source_images_decoded':len(records),'EXIF_orientation_corrected':sum(r['EXIF_transformed'] for r in records),'oriented_dimensions_exactly_match_JSON':len(records),'valid_contained_target_bboxes':len(records),'corrected_crops_written_and_decoded':len(records),'crop_failures':0,'historical_fullframe_fallbacks_corrected':sum(r['historical_full_frame_fallback'] for r in records),'proxy_only':sum(r['source']=='train' for r in records),'reference_original_pool':sum(r['source']=='val' for r in records),'reference_same_row_complete':sum(r['source']=='val' and r['complete_original_row'] for r in records),'reference_excluded_missing_selected_row_sugar':sum(r['source']=='val' and not r['complete_original_row'] for r in records),'reference_strict_unique_NIR':sum(r['source']=='val' and r['complete_original_row'] and r['unique_NIR_count']==1 for r in records),'corrected_NIR_labels':sum(r['historical_NIR_changed'] for r in records),'corrected_sugar_labels_including_missing':sum(r['historical_sugar_changed'] for r in records),'pillow':pillow_version,'JPEG':features.version('jpg'),'provider_release_verified':False,'HA_physical_identity_verified':False,'audit_scope':'Complete algorithmic linkage/geometric/decoding validation; physical fruit identity and annotation semantics are not independently established','manifest_sha256':sha(manifest_path.read_bytes()),'code_sha256':sha(Path(__file__).read_bytes()),'protocol_sha256':sha((OUT/'reproducibility/protocol_correction_v4.md').read_bytes()),'elapsed_seconds':time.time()-START}
js(OUT/'results/source_crop_audit_summary.json',summary)
js(OUT/'private/READY.json',{'complete':True,'source_crop_audit_sha256':sha((OUT/'results/source_crop_audit_summary.json').read_bytes()),'crop_train_meta_sha256':sha((DEST/'train_meta.csv').read_bytes()),'crop_val_meta_sha256':sha((DEST/'val_meta.csv').read_bytes())})
print(json.dumps({'stage':'COMPLETE','summary':summary},ensure_ascii=False),flush=True)
