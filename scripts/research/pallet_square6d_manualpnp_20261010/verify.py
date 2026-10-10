"""Independent public numeric verification; no model, PnP or producer imports."""
import argparse, ast, gzip, hashlib, json, math, os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DOC=Path(os.environ.get('PALLET_SQUARE_OUTPUT',ROOT/'_docs/experiments/pallet_square6d_manualpnp_20261010'))
BB=('yolo','dope','resnet18');METHODS=('BASE','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX')
FIELDS={'T_cm':'translation_cm','R_deg':'rotation_deg','ADDsym_m':'ADDsym_m','IoU3D':'IoU3D'}
CONTRASTS={'N3_MINUS_BASE':('N3_DIM_SYM','BASE'),'SEQUENCE_MINUS_BASE':('N3_THEN_SUBPIX','BASE'),'SEQUENCE_MINUS_SUBPIX':('N3_THEN_SUBPIX','SUBPIX')}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def quantile(x,p):
 x=sorted(x);a=(len(x)-1)*p;l,h=math.floor(a),math.ceil(a);return x[l]+(x[h]-x[l])*(a-l)
def statistics(x):
 x=[float(v) for v in x if v is not None];n=len(x)
 if not n:return dict(n=0,mean=None,variance=None,std=None,median=None,P90=None,max=None)
 mean=math.fsum(x)/n;v=math.fsum((z-mean)**2 for z in x)/(n-1) if n>1 else None
 return dict(n=n,mean=mean,variance=v,std=math.sqrt(v) if v is not None else None,median=quantile(x,.5),P90=quantile(x,.9),max=max(x))
def vector(rr):
 v={k:[r['pose'][f] if r['pose']['available'] else None for r in rr] for k,f in FIELDS.items()}
 v['available']=[r['pose']['available'] for r in rr]
 v['success_rate']=[float(r['pose']['available'] and r['pose']['translation_cm']<5 and r['pose']['rotation_deg']<5) for r in rr]
 return v
def average(vv):
 return {k:[all(x) if k=='available' else math.fsum(x)/3 if all(v is not None for v in x) else None for x in zip(*(v[k] for v in vv))] for k in vv[0]}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--source-root',type=Path);parser.add_argument('--baseline-root',type=Path);a=parser.parse_args()
 out=a.output or DOC/'VERIFICATION.json';assert not out.exists()
 rr=rows(DOC/'PREDICTIONS.jsonl.gz');sealed=rows(DOC/'COORDINATES_SEALED.jsonl.gz');seal=read(DOC/'COORDINATES_SEAL.json')
 assert sha(DOC/'COORDINATES_SEALED.jsonl.gz')==seal['sha256']
 assert sha(DOC/'METHOD_LOCK.json')==seal['method_lock_sha256']
 for b in seal['source']:assert sha(ROOT/b['path'])==b['sha256']
 assert len(rr)==len(sealed)==4248
 index={(r['backbone'],r['seed'],r['method'],r['id']):r for r in rr}
 ids=[r['id'] for r in rr if r['backbone']=='yolo' and r['seed']==1 and r['method']=='BASE']
 assert len(ids)==len(set(ids))==118 and '029844' not in ids
 assert len(index)==4248 and set(index)=={(b,s,m,i) for b in BB for s in (1,2,3) for m in METHODS for i in ids}
 refs=read(DOC/'SQUARE_REFERENCE_POSES.json');ref={r['id']:r for r in refs['frames'] if r['eligible']}
 assert set(ref)==set(ids) and len(ref)==118 and refs['reference_prediction_inputs'] is False
 tree=ast.parse((ROOT/'scripts/research/pallet_square6d_manualpnp_20261010/references.py').read_text())
 fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference_fit')
 assert [x.arg for x in fit.args.args]==['manual_xy','K','corner_indices']
 assert sha(ROOT/'scripts/research/pallet_square6d_manualpnp_20261010/references.py')==refs['reference_source_sha256']
 assert read(DOC/'TWO_D_PARITY.json')['max_absolute_difference']==0.
 comparisons=0;maximum=0.
 def close(x,y):
  nonlocal comparisons,maximum
  comparisons+=1
  if y is None:assert x is None;return
  if isinstance(y,(list,tuple)):
   assert len(x)==len(y)
   for i,j in zip(x,y):close(i,j)
   return
  diff=abs(float(x)-float(y));maximum=max(maximum,diff);assert diff<=1e-7+1e-12*abs(y),(x,y,diff)
 X=np.array([[-.55,-.075,-.55],[.55,-.075,-.55],[.55,.075,-.55],[-.55,.075,-.55],[-.55,-.075,.55],[.55,-.075,.55],[.55,.075,.55],[-.55,.075,.55]])
 Qs=[np.array([[math.cos(i*math.pi/2),0,math.sin(i*math.pi/2)],[0,1,0],[-math.sin(i*math.pi/2),0,math.cos(i*math.pi/2)]]) for i in range(4)]
 reference_residuals=[];loo_total=0
 for i,r in ref.items():
  assert r['available'] and r['prediction_inputs'] is False and np.asarray(r['t'])[2]>0
  R=np.asarray(r['R']);t=np.asarray(r['t']);camera=np.asarray(r['K']);idx=r['manual_corner_indices'];manual=np.asarray(r['manual_xy'])
  v=(R@X.T).T+t;screen=(camera@v.T).T;screen=screen[:,:2]/screen[:,2:]
  errs=[math.hypot(*(p-z)) for p,z in zip(screen[idx],manual)]
  close(r['residual_mean_px'],math.fsum(errs)/len(errs));close(r['residual_max_px'],max(errs));close(r['projected_corners8'],screen.tolist())
  reference_residuals.append(math.fsum(errs)/len(errs));loo=r['LOO'];loo_total+=len(loo)
  assert r['LOO_used_as_filter'] is False and len(loo)==(len(idx) if len(idx)>=5 else 0)
  close(r['LOO_max_rotation_change_c4_deg'],max([x['rotation_change_c4_deg'] for x in loo if x['available']],default=None))
  close(r['LOO_max_translation_change_cm'],max([x['translation_change_cm'] for x in loo if x['available']],default=None))
 close(refs['median_mean_residual_px'],quantile(reference_residuals,.5));close(refs['fraction_mean_residual_over5_px'],sum(x>5 for x in reference_residuals)/118)
 assert refs['median_mean_residual_px']<=3 and refs['fraction_mean_residual_over5_px']<=.1 and refs['reference_solver_calls']==118+loo_total
 for r,z in zip(rr,sealed):
  for k in z:assert r[k]==z[k],k
  q0,qN,q=np.asarray(r['q0'],float),np.asarray(r['qN'],float),np.asarray(r['qFinal'],float);support=np.asarray(r['prediction_support'],bool)
  assert np.array_equal(q[8],q0[8],equal_nan=True) and np.array_equal(qN[8],q0[8],equal_nan=True)
  if r['method']=='BASE':assert np.array_equal(q,q0,equal_nan=True)
  elif r['method']=='N3_DIM_SYM':assert np.array_equal(q,qN,equal_nan=True)
  else:
   assert r['correction']['GT_inputs'] is False
   usable=support[:8]&np.isfinite(q0[:8]).all(1)&~(q0[:8]==-1).all(1)
   assert np.max(np.linalg.norm(q[:8][usable]-q0[:8][usable],axis=1),initial=0.)<=8.+1e-10
  if not r['pose']['available']:continue
  p=r['actual_pose'];g=ref[r['id']];R=np.asarray(p['R_physical']);G=np.asarray(g['R']);t=np.asarray(p['centroid']);gt=np.asarray(g['t'])
  close(r['pose']['translation_cm'],math.sqrt(math.fsum(float(v)**2 for v in t-gt))*100)
  angles=[];adds=[]
  for Q in Qs:
   target=G@Q;rel=target.T@R;trace=math.fsum(float(rel[k,k]) for k in range(3))
   angles.append(math.degrees(math.acos(min(1.,max(-1.,(trace-1)/2)))))
   delta=((R@X.T).T+t)-((target@X.T).T+gt)
   adds.append(math.fsum(math.sqrt(math.fsum(float(v)**2 for v in row)) for row in delta)/8)
  close(r['pose']['rotation_deg'],min(angles));close(r['pose']['ADDsym_m'],min(adds));close(r['pose']['ADDsym_normalized'],min(adds)/math.sqrt(1.1**2+.15**2+1.1**2))
 metrics=read(DOC/'METRICS.json');paired=read(DOC/'PAIRED.json');failures=read(DOC/'FAILURES.json');scopes=metrics['scopes_frame_ids']
 assert scopes==dict(ALL118=ids,reference_residual_mean_le5=[i for i in ids if ref[i]['residual_mean_px']<=5],manual4=[i for i in ids if ref[i]['manual_count']==4],manual_ge5=[i for i in ids if ref[i]['manual_count']>=5])
 rng=np.random.default_rng(20260917);draw=np.vstack([rng.multinomial(118,np.ones(118)/118,size=100) for _ in range(100)])
 assert hashlib.sha256(draw.astype('<u2').tobytes()).hexdigest()==metrics['bootstrap']['draws_sha256_uint16_le']
 positions={i:j for j,i in enumerate(ids)}
 for b in BB:
  for scope,selected in scopes.items():
   w=draw[:,[positions[i] for i in selected]].astype(float);cache={}
   def interval(vals):
    key=tuple(vals)
    if key in cache:return cache[key]
    ok=np.array([x is not None for x in vals]);x=np.array([float(v) if k else 0 for v,k in zip(vals,ok)])
    if not ok.any():return None
    d=w@ok.astype(float);samples=((w@x)[d>0]/d[d>0]).tolist();result=[quantile(samples,.025),quantile(samples,.975)];cache[key]=result;return result
   def verify_summary(p,v):
    assert p['frames']==len(selected) and p['available']==sum(v['available']) and p['unavailable']==len(selected)-sum(v['available'])
    close(p['coverage'],sum(v['available'])/len(selected))
    for k in FIELDS:
     for name,value in statistics(v[k]).items():close(p['metrics'][k][name],value)
     close(p['metrics'][k]['CI95'],interval(v[k]));assert p['metrics'][k]['variance_ddof']==1
    rate=p['success_rate'];close(rate['rate'],math.fsum(v['success_rate'])/len(selected));close(rate['positive_count_sum'],math.fsum(v['success_rate']));close(rate['CI95'],interval(v['success_rate']))
    assert rate['denominator']==len(selected) and rate['binary_before_seed_mean'] is True
   vv={(s,m):vector([index[(b,s,m,i)] for i in selected]) for s in (1,2,3) for m in METHODS};packet=metrics['backbones'][b][scope]
   for s in (1,2,3):
    for m in METHODS:verify_summary(packet['by_seed'][str(s)][m],vv[(s,m)])
   for m in METHODS:verify_summary(packet['seed_mean'][m],average([vv[(s,m)] for s in (1,2,3)]))
   for name,(l,r) in CONTRASTS.items():
    pp=paired['backbones'][b][scope][name]
    for k in (*FIELDS,'success_rate'):
     ds=[[a-z if a is not None and z is not None else None for a,z in zip(vv[(s,l)][k],vv[(s,r)][k])] for s in (1,2,3)]
     dm=[math.fsum(x)/3 if all(v is not None for v in x) else None for x in zip(*ds)]
     close(pp['seed_mean'][k]['delta'],statistics(dm)['mean']);close(pp['seed_mean'][k]['CI95'],interval(dm));close(pp['seed_mean'][k]['per_seed_delta'],[statistics(d)['mean'] for d in ds])
     for s,d in zip((1,2,3),ds):close(pp['by_seed'][str(s)][k]['delta'],statistics(d)['mean']);close(pp['by_seed'][str(s)][k]['CI95'],interval(d))
    for s in (1,2,3):
     damage=[i for i,a,z in zip(selected,vv[(s,l)]['success_rate'],vv[(s,r)]['success_rate']) if z and not a];recovery=[i for i,a,z in zip(selected,vv[(s,l)]['success_rate'],vv[(s,r)]['success_rate']) if a and not z]
     f=failures['backbones'][b][scope][name][str(s)];assert f==pp['damage_by_seed'][str(s)]
     assert f['success_to_failure_ids']==damage and f['failure_to_success_ids']==recovery
     assert f['success_to_failure']==len(damage) and f['failure_to_success']==len(recovery)
 private_checked=0
 if a.source_root is not None:
  audit=read(DOC/'INPUT_AUDIT.json')
  for x in [*audit['bound_files'],audit['snapshot'],audit['symmetry'],*[v for r in audit['inputs'] for v in (r['image'],r['annotation'])]]:
   p=Path(x['path']);options=[p] if p.is_absolute() else [a.source_root/p,(a.baseline_root or ROOT)/p,ROOT/p]
   assert any(p.is_file() and sha(p)==x['sha256'] for p in options),x['path'];private_checked+=1
 for p in [*DOC.rglob('*'),*(ROOT/'scripts/research/pallet_square6d_manualpnp_20261010').rglob('*')]:
  if not p.is_file() or p.suffix=='.png':continue
  assert p.suffix not in ('.pt','.pth','.jpg','.jpeg','.npy','.npz','.pyc')
  if p.suffix=='.gz':
   with gzip.open(p,'rt') as f:text=f.read()
  else:text=p.read_text()
  assert '/'+'home'+'/' not in text,p.name
 result=dict(status='PASS',independent_numeric_comparisons=comparisons,max_absolute_difference=maximum,frames=118,rows=4248,
  independent_reference_reprojection=True,reference_residual_gate='PASS',manual_LOO_records=loo_total,
  independent_pose_T_R_ADD=True,IoU3D_statistics_verified=True,IoU3D_geometry_independently_recomputed=False,
  independent_moments='math.fsum; ddof1; sorted linear quantiles',frame_draws_sha256=metrics['bootstrap']['draws_sha256_uint16_le'],
  scopes={k:len(v) for k,v in scopes.items()},sealed_coordinates_exact=True,existing_private_input_hashes_checked=private_checked,
  independent_F_calls=0,independent_model_calls=0,new_training_updates=0,reference_fit_predictor_parameters=False,
  privacy='No raw RGB, checkpoint or personal absolute path in public namespace; PNGs are numeric plots',source_sha256=sha(__file__))
 with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print('SQUARE_INDEPENDENT_VERIFICATION_PASS',comparisons,maximum,flush=True)
if __name__=='__main__':main()
