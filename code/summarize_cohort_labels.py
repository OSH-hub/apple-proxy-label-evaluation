"""Reproduce tables/cohort_label_date_summary.csv from an authorized tree-scheme prediction CSV.

No individual identifiers are written to the output. Dates are retained-key
fields, not independently verified measurement timestamps. Standard library only.
"""
import argparse,csv,statistics
from datetime import datetime
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--predictions',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
with a.predictions.open(encoding='utf-8-sig',newline='') as f:
 rows=list(csv.DictReader(f))
assert len({(r['set'],r['objkey']) for r in rows})==len(rows)
expected={'fit':8919,'select':1643,'proxy_test':3197,'reference_test':1191}
assert {s:sum(r['set']==s for r in rows) for s in expected}==expected
summary=[]
for subset in expected:
 for cultivar in ('ALL','AR','HJ','HR','SG'):
  selected=[r for r in rows if r['set']==subset and (cultivar=='ALL' or r['kind']==cultivar)]
  for field in ('nir','sugar'):
   eligible=[r for r in selected if r[field]!='']
   if not eligible: continue
   values=[float(r[field]) for r in eligible]
   dates=[datetime.strptime(r['objkey'].split('|')[0],'%Y%m%d').date() for r in eligible]
   summary.append({'cohort':subset,'cultivar':cultivar,'variable':'NIR' if field=='nir' else 'reference','n':len(values),'trees':len({r['tree_key'] for r in eligible}),'mean':statistics.mean(values),'sd_sample':statistics.stdev(values),'min':min(values),'max':max(values),'date_min_from_objkey':min(dates).isoformat(),'date_max_from_objkey':max(dates).isoformat(),'zero_values':sum(v==0 for v in values)})
if a.output.exists():
 raise FileExistsError('Choose a new output file to preserve existing results.')
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print(f'Wrote {len(summary)} aggregate rows.')
