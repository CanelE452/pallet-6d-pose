"""Exact source forward-input collisions, without model/weight/current reads."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
V78=P/'query_visibility_dino_query_support_20261023_v78'
V77=P/'query_visibility_controlled_pairs_20261023_v77'
OUT=R/'input_collisions'
OLD=('p3_roi','rgb96','crop_valid96','target_rgb128','target_valid128','context42')
NEW=('local_features','target_features','context42','validity_fractions')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for v in iter(lambda:f.read(1<<20),b''):h.update(v)
 return h.hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def ck(x):assert sha(x['path'])==x['sha256'],x['path']
def rd(p):return json.loads(Path(p).read_text())
def put(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def comp(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256()
 h.update(a.dtype.str.encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
 return h.digest()
def combine(items):
 h=hashlib.sha256()
 for key,value in items:
  h.update(key.encode());h.update(b'\0');h.update(value)
 return h.hexdigest().encode()
def freeze():
 assert not(R/'INPUT_COLLISION_PROTOCOL.json').exists()
 gtold=rd(V77/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json')['bindings']['teacher_GT']
 gtafter=rd(V77/'COMPLETED_SOURCE_INPUT_BINDINGS.json')['bindings']['after_teacher_GT']
 bindings=dict(entry=b(R/'AUDIT_ENTRY.json'),feature_manifest=b(V78/'feature_pack/PACK_MANIFEST.json'),
  feature_seal=b(V78/'feature_pack/FINAL_FEATURE_SEAL.json'),old_GT=gtold,after_GT=gtafter,
  selected_scores=b(V78/'runs/SOURCE_DINO_V1/SELECTED_SOURCE_SCORES.npz'))
 for v in bindings.values():ck(v)
 put(R/'INPUT_COLLISION_PROTOCOL.json',dict(frozen=True,code=b(Path(__file__)),bindings=bindings,
  variants=dict(V77='All6 original model tensors: P3/localRGB/localcanvas/targetRGB/targetcanvas/context42',
   V78='All4 compact pooled DINO model tensors: local_features/target_features/context42/fractions',
   RAW_RGB_VIEW='DINO image/readout source views: localRGB/localcanvas/targetRGB/targetcanvas/context42'),
  hash_recipe='SHA256 of ordered named component digests. Each component digest includes dtype,shape,contiguousbytes. Actual target tensor repeated by gather frame_index=query_index//8. No id/source/split/GTref/label or additional absolute q enters hash.',
  scope='All80016 queries includinginvalid/U. Retain allprimary denominators. Exact duplicate inputs with conflictingknownsupport V/C labels are reported, never filtered or relabelled.',
  read_whitelist=dict(old=OLD,new=NEW,scores=['id','source','split','domain','primary_loss_mask'],GT=['id','source','split','label8']),
  interpretation='Exact collisions certify indistinguishable tensor input, but absence of collisions does not prove observability or learnable target identity. RawRGB matches may have tiny batch arithmetic differences in stored latent features.',
  no_new_model_weight_forward_fit_current_PnP=True))
 print(b(R/'INPUT_COLLISION_PROTOCOL.json'))
def summarize(hashes,y,primary,source,split,domain,ids):
 u,inv,n=np.unique(hashes,return_inverse=True,return_counts=True)
 groups=[];bad=[];cross=[]
 for g in np.flatnonzero(n>1):
  ix=np.flatnonzero(inv==g);known=ix[np.isin(y[ix],[0,1])];eligible=ix[primary[ix]]
  rec=dict(hash=u[g].decode(),N=len(ix),query_indices=ix.tolist(),known_counts={str(c):int(np.sum(y[ix]==c))for c in [-1,0,1]},primary_known_counts={str(c):int(np.sum(y[eligible]==c))for c in [0,1]},
   mixed_known_V_C=bool(np.any(y[known]==0)and np.any(y[known]==1)),mixed_primary_V_C=bool(np.any(y[eligible]==0)and np.any(y[eligible]==1)),train_VAL_overlap=bool(np.any(split[ix]=='train')and np.any(split[ix]=='val')))
  groups.append(rec)
  if rec['mixed_known_V_C']:bad.append(rec)
  if rec['train_VAL_overlap']:cross.append(rec)
 return dict(total_queries=len(hashes),unique_input_digests=len(u),duplicate_groups=len(groups),duplicate_queries=int(n[n>1].sum()),conflicting_known_support_groups=len(bad),conflicting_primary_support_groups=sum(q['mixed_primary_V_C']for q in bad),train_VAL_exact_input_groups=len(cross),groups=groups)
def run():
 p=rd(R/'INPUT_COLLISION_PROTOCOL.json');assert p['frozen'];ck(p['code']);[ck(v)for v in p['bindings'].values()]
 assert not OUT.exists();OUT.mkdir();start=time.monotonic()
 man=rd(p['bindings']['feature_manifest']['path']);assert man['completed']and man['frame_count']==10002
 with np.load(p['bindings']['selected_scores']['path'],allow_pickle=False)as z:meta={k:z[k]for k in p['read_whitelist']['scores']}
 gt=[]
 for name in ['old_GT','after_GT']:
  with np.load(p['bindings'][name]['path'],allow_pickle=False)as z:gt.append({k:z[k]for k in p['read_whitelist']['GT']})
 g={k:np.concatenate([a[k]for a in gt])for k in gt[0]}
 for k in ['id','source','split']:assert np.array_equal(meta[k],g[k])
 assert meta['primary_loss_mask'].shape==(10002,8)
 hashes={key:[]for key in ['V77','V78','RAW_RGB_VIEW']};at=0
 for chunk in man['chunks']:
  ck(chunk['file']);ck(chunk['input_chunk']);assert chunk['start']==at
  with np.load(chunk['file']['path'],allow_pickle=False)as z:d={k:z[k]for k in NEW}
  with np.load(chunk['input_chunk']['path'],allow_pickle=False)as z:o={k:z[k]for k in OLD}
  F=chunk['end']-at
  assert o['p3_roi'].shape==(F,8,65,9,9)and d['local_features'].shape==(F,8,2,384)
  assert np.array_equal(o['context42'],d['context42'])
  for f in range(F):
   ot={key:comp(o[key][f])for key in ['target_rgb128','target_valid128']}
   nt={'target_features':comp(d['target_features'][f])}
   for k in range(8):
    oc={key:comp(o[key][f,k])for key in ['p3_roi','rgb96','crop_valid96','context42']};oc.update(ot)
    nc={key:comp(d[key][f,k])for key in ['local_features','context42','validity_fractions']};nc.update(nt)
    hashes['V77'].append(combine((key,oc[key])for key in OLD))
    hashes['RAW_RGB_VIEW'].append(combine((key,oc[key])for key in OLD if key!='p3_roi'))
    hashes['V78'].append(combine((key,nc[key])for key in NEW))
  at=chunk['end']
  if at%1000<128:print(json.dumps(dict(frames=at,total=10002,seconds=time.monotonic()-start)),flush=True)
 assert at==10002
 hashes={k:np.asarray(v,dtype='S64')for k,v in hashes.items()}
 y=g['label8'].reshape(-1);primary=meta['primary_loss_mask'].reshape(-1)
 src=np.repeat(g['source'],8);split=np.repeat(g['split'],8);dom=np.repeat(meta['domain'],8)
 arr=OUT/'ROW_INPUT_HASHES.npz';np.savez_compressed(arr,**hashes,id=g['id'],source=g['source'],split=g['split'],domain=meta['domain'],label8=g['label8'],primary_loss_mask=meta['primary_loss_mask'])
 report=dict(completed=True,protocol=b(R/'INPUT_COLLISION_PROTOCOL.json'),hashes=b(arr),variants={key:summarize(h,y,primary,src,split,dom,g['id'])for key,h in hashes.items()},
  all_queries_retained=True,new_model_weight_forward_fit_current_PnP=0,dense_GT_fingerprint_join='Pendingwholecoverageoutput; knownsupportconflictsreportednow. No input-only corpus filtering.',seconds=time.monotonic()-start)
 put(OUT/'INPUT_COLLISION_AUDIT.json',report);put(OUT/'SEAL.json',dict(completed=True,bindings=[b(p)for p in sorted(OUT.iterdir())if p.is_file()]))
 print(b(OUT/'INPUT_COLLISION_AUDIT.json'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','run']);x=a.parse_args();dict(freeze=freeze,run=run)[x.stage]()
