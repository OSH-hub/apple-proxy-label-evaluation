"""Historical representative selection dependency only; no crop implementation.
Set MANIFEST to the authorized original manifest before calling select_items.
The historical pool classification is preserved; its labels are NOT final targets.
The corrected worker replaces labels from the chosen source row.
"""
import csv, collections
MANIFEST="SET_AUTHORIZED_ORIGINAL_MANIFEST_PATH"

VARIETY_ZIPS = {
    "AR": ["TS2_아리수.zip"],
    "SG": ["TS1_시나노골드.zip"],
    "HR": ["TS3_홍로1.zip", "TS4_홍로2.zip", "TS5_홍로3.zip"],
    "HJ": ["TS6_후지1.zip", "TS7_후지2.zip", "TS8_후지3.zip", "TS9_후지4.zip"],
}

def fnum(x):
    try:
        v = float(x)
        return v
    except Exception:
        return None

def select_items(train_per_kind):
    per_obj = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = f'{row["date"]}|{row["kind"]}|{row["pos"]}|{row["tree"]}|{row["obj"]}'
            sugar = fnum(row["sugar"]); nir = fnum(row["nir"])
            score = 0 if row["ang"] == "0" else int(row["ang"] or 999)
            cur = per_obj.get(k)
            if cur is None:
                per_obj[k] = dict(objkey=k, img=row["img"], vdir=row["variety_dir"],
                                  kind=row["kind"], pos=row["pos"], tree=row["tree"],
                                  date=row["date"], obj=row["obj"], nir=nir,
                                  sugar=sugar, grade=row["grade"], score=score)
            else:
                if sugar is not None and cur["sugar"] is None:
                    cur["sugar"] = sugar
                if nir is not None and cur["nir"] is None:
                    cur["nir"] = nir
                if score < cur["score"]:
                    cur.update(img=row["img"], vdir=row["variety_dir"], score=score)
    items = sorted(per_obj.values(), key=lambda r: r["objkey"])
    val = [r for r in items if r["sugar"] is not None]
    pool = [r for r in items if r["sugar"] is None and r["nir"] is not None]
    train = []
    bykind = collections.defaultdict(list)
    for r in pool:
        bykind[r["kind"]].append(r)
    for kind, rs in sorted(bykind.items()):
        if train_per_kind and len(rs) > train_per_kind:
            step = len(rs) / train_per_kind
            rs = [rs[int(i * step)] for i in range(train_per_kind)]
        train += rs
    return val, train
