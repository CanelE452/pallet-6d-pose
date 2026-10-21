"""Recompute public scalar metrics and recorded gates; no training or raw inputs."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read_csv(path):
 with path.open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))
def quantile(v,p):
 v=sorted(v);z=(len(v)-1)*p;i=math.floor(z);j=math.ceil(z)
 return v[i]+(v[j]-v[i])*(z-i)
def summary(rows):
 out=dict(denominator=len(rows),successes=sum(r['success']=='True'for r in rows),no_pose=sum(r['pose_available']!='True'for r in rows))
 for tag,key in [('T','translation_cm'),('R','rotation_deg')]:
  vals=[float(r[key])for r in rows if r['pose_available']=='True'];out[tag+'_n']=len(vals)
  for name,fun in [('mean',statistics.fmean),('median',statistics.median),('P90',lambda v:quantile(v,.9)),('P95',lambda v:quantile(v,.95)),('max',max)]:out[tag+'_'+name]=fun(vals)if vals else None
 for name,key,limit in [('T_gt50','translation_cm',50),('T_gt100','translation_cm',100),('R_gt45','rotation_deg',45)]:out[name]=sum(r['pose_available']=='True'and float(r[key])>limit for r in rows)
 return out
def gate(a,b):
 why=[]
 if a['successes']<b['successes']:why.append('joint_success_decreased')
 for k in ['T_mean','R_mean','T_P90','T_P95','R_P90','R_P95','T_max','R_max','no_pose','T_gt50','T_gt100','R_gt45']:
  if a[k]>b[k]+1e-7:why.append(k+'_increased')
 if not(a['T_mean']<b['T_mean']-1e-7 or a['R_mean']<b['R_mean']-1e-7):why.append('no_mean_improvement')
 return why
def key(r):return tuple(str(r[k])for k in ['version','population','seed','method'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();root=args.root
 manifest=json.loads((root/'MANIFEST.json').read_text());checks=0
 for f in manifest['files']:
  path=root/f['path'];assert path.stat().st_size==f['bytes']and hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256'],f['path'];checks+=1
 groups=defaultdict(list)
 for r in read_csv(root/'results/per_frame_metrics.csv'):
  if r['pose_available']=='True':assert (float(r['translation_cm'])<=5 and float(r['rotation_deg'])<=5)==(r['success']=='True')
  else:assert not r['translation_cm']and not r['rotation_deg']and r['success']=='False'
  groups[key(r)].append(r)
 got={k:summary(v)for k,v in groups.items()};reported=read_csv(root/'summary.csv')
 for r in reported:
  actual=got[key(r)]
  for k,v in actual.items():
   if v is None:assert r[k]==''
   elif isinstance(v,int):assert int(r[k])==v,(key(r),k)
   else:assert math.isclose(float(r[k]),v,rel_tol=1e-12,abs_tol=1e-7),(key(r),k,float(r[k]),v)
 gates=json.loads((root/'results/gates.json').read_text())
 for g in gates['comparisons']:
  actual=gate(got[tuple(g['method_key'])],got[tuple(g['comparator_key'])]);assert actual==g['recorded_reasons'],(g,actual)
 for p in read_csv(root/'results/paired_comparisons.csv'):
  a={r['frame_id']:r for r in groups[tuple(json.loads(p['before_key']))]};b={r['frame_id']:r for r in groups[tuple(json.loads(p['after_key']))]};assert a.keys()==b.keys()
  assert int(p['recover'])==sum(a[i]['success']=='False'and b[i]['success']=='True'for i in a)
  assert int(p['damage'])==sum(a[i]['success']=='True'and b[i]['success']=='False'for i in a)
  common=[i for i in a if a[i]['pose_available']=='True'and b[i]['pose_available']=='True'];assert len(common)==int(p['common_available_n'])
  for side,rs in [('before',a),('after',b)]:
   for tag,col in [('T','translation_cm'),('R','rotation_deg')]:assert math.isclose(float(p[side+'_common_'+tag+'_mean']),statistics.fmean(float(rs[i][col])for i in common),rel_tol=1e-12,abs_tol=1e-7)
 print(json.dumps(dict(status='PASS',verified_files=checks,summary_groups=len(got),recorded_gate_comparisons=len(gates['comparisons']),metric_rows=sum(map(len,groups.values())),raw_images_checkpoints_or_training_used=False),indent=2))
if __name__=='__main__':main()
