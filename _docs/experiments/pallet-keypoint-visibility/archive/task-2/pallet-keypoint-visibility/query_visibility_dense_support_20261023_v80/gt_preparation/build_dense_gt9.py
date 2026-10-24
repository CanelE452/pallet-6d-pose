"""Materialize previously audited nominal native9 mask targets; no RGB/model access."""
from pathlib import Path
from collections import OrderedDict
import hashlib, importlib.util, json, sys, time
import numpy as np
from PIL import Image

R=Path(__file__).resolve().parents[1]; P=R.parent
A=P/'query_visibility_natural_occlusion_audit_20261023_v79'
V76=P/'query_visibility_reference_local_support_20261023_v76'
V77=P/'query_visibility_controlled_pairs_20261023_v77'
OUT=R/'dense_gt'; PROTOCOL=R/'gt_preparation/GT_BUILD_PROTOCOL.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(1<<20),b''): h.update(a)
 return h.hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p): return json.loads(Path(p).read_text())
def write(p,j):
 with open(p,'x') as f: f.write(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(b): assert sha(b['path'])==b['sha256'],b['path']
def math_module():
 f=A/'coverage/coverage_math.py'; sp=importlib.util.spec_from_file_location('frozen_v79_gtmath',f)
 m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def generated():
 m=math_module(); q=np.tile([4.,4.],(8,1));q[1]=[-1.,-1.];q[2]=[-.25,4.]
 g=m.coordinates(q,9.,9); valid=m.cached_formula(g,(9,9),9,q)
 z=np.zeros((9,9),bool);f=z.copy();f[:,2:7]=True;v=f.copy();v[3:6,3:6]=False
 counts,grid,pix,native=m.count_grid(f,v,q,9.,9,valid); use=native&valid
 assert not use[1].any(); assert counts['class_counts'].shape==(8,3,3)
 assert not(v&~f).any(); assert np.array_equal(counts['class_counts'][:,2].sum(1),use.sum((1,2)))
 write(R/'gt_preparation/GENERATED_GT9_TESTS.json',dict(PASS=True,only_generated_inputs=True,code=bind(__file__),math=bind(A/'coverage/coverage_math.py'),source_mask_RGB_model_current_reads=0))
def freeze():
 generated=R/'gt_preparation/GENERATED_GT9_TESTS.json'; assert load(generated)['PASS'] and load(generated)['code']==bind(__file__)
 paths=[Path(__file__),A/'coverage/coverage_math.py',generated,A/'coverage/results/FINAL_COVERAGE_SEAL.json',A/'coverage/results/COVERAGE_PER_QUERY.npz',A/'independent_audits/COVERAGE_AGGREGATE_AND_SCALAR_AUDIT.json',V76/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl',V77/'pair_generation/PAIR_LINEAGE_GT_ONLY.jsonl',V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz']
 write(PROTOCOL,dict(frozen=True,shape=[10002,8,9,9],bindings=[bind(x)for x in paths],outputs=['valid9','expected_support_M0','visible_target_M4'],grid='Unchanged nominal9 literal q+(((j+.5)/9)-.5)*L; NN floor+.5; no q/r recenter.',valid='Native continuous AND NN bounds AND cached9 intersection; missing/invalid ignored, never negative.',mask_threshold='Original sealed per-row threshold; afterSYNTH>127/PUBLIC>0; M4subsetM0.',proof='All80016 nominal9 fingerprints and V/C/O counts must equal sealed V79 intersection before targets admitted.',scope='GT only for auxiliary loss/diagnostics; no inference inputs, RGB, current223, model, fit or PnP.',never_overwrite=True))
def run():
 p=load(PROTOCOL);assert p['frozen'] and not OUT.exists()
 for x in p['bindings']:check(x)
 OUT.mkdir();start=time.monotonic();m=math_module()
 with np.load(A/'coverage/results/COVERAGE_PER_QUERY.npz',allow_pickle=False)as z:
  c={k:z[k]for k in ['id','source','split','domain','query8_xy','L','raw_hw','class_counts','gt_map_intersection_sha256']}
 original=[json.loads(x)for x in (V76/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl').read_text().splitlines()]
 after=[json.loads(x)for x in (V77/'pair_generation/PAIR_LINEAGE_GT_ONLY.jsonl').read_text().splitlines()]
 assert len(original)==5203 and len(after)==4799
 targets={k:np.zeros((10002,8,9,9),bool)for k in p['outputs']};cache=OrderedDict();decoded=0
 def mask(binding,threshold):
  nonlocal decoded
  key=(binding['path'],binding['sha256'],threshold)
  if key not in cache:
   check(binding);a=np.asarray(Image.open(binding['path']));assert a.ndim==2;cache[key]=a>threshold;decoded+=1
   if len(cache)>40:cache.popitem(last=False)
  else:cache.move_to_end(key)
  return cache[key]
 for i in range(10002):
  if i<5203:
   pair=original[i]['mask_pair'];b0,b4,t=pair['M0'],pair['M4'],pair['threshold_strict_gt']
  else:
   pair=after[i-5203];b0,b4=pair['M0_unchanged'],pair['M4_after'];t=127 if c['source'][i]=='SYNTHETIC'else 0
  f,v=mask(b0,t),mask(b4,t);assert f.shape==v.shape==tuple(c['raw_hw'][i]);assert not(v&~f).any()
  q,L=c['query8_xy'][i],float(c['L'][i]);grid=m.coordinates(q,L,9);cached=m.cached_formula(grid,f.shape,9,q)
  result,_,pix,native=m.count_grid(f,v,q,L,9,cached);valid=native&cached
  ff=np.zeros((8,9,9),bool);vv=ff.copy();ff[valid]=f[pix[...,1][valid],pix[...,0][valid]];vv[valid]=v[pix[...,1][valid],pix[...,0][valid]]
  assert np.array_equal(result['gt_map_intersection_sha256'],c['gt_map_intersection_sha256'][i,:,0]),i
  assert np.array_equal(result['class_counts'][:,2],c['class_counts'][i,:,0,2]),i
  targets['valid9'][i]=valid;targets['expected_support_M0'][i]=ff;targets['visible_target_M4'][i]=vv
  if(i+1)%1000==0:print(json.dumps(dict(rows=i+1,seconds=time.monotonic()-start)),flush=True)
 targets.update({k:c[k]for k in ['id','source','split','domain']})
 np.savez_compressed(OUT/'DENSE_GT9.npz',**targets)
 write(OUT/'GT9_BUILD_RESULTS.json',dict(completed=True,PASS=True,rows=10002,queries=80016,all_intersection_fingerprints_and_class_counts_exact=True,decoded_masks=decoded,grid_positions=int(targets['valid9'].sum()),visible=int(targets['visible_target_M4'].sum()),covered=int((targets['expected_support_M0']&~targets['visible_target_M4']).sum()),outside=int((targets['valid9']&~targets['expected_support_M0']).sum()),seconds=time.monotonic()-start,current_RGB_model_forward_fit_PnP=0))
 write(OUT/'SEAL.json',dict(completed=True,protocol=bind(PROTOCOL),targets=bind(OUT/'DENSE_GT9.npz'),results=bind(OUT/'GT9_BUILD_RESULTS.json')))
 print(json.dumps(bind(OUT/'SEAL.json')),flush=True)
if __name__=='__main__':dict(generated=generated,freeze=freeze,run=run)[sys.argv[1]]()
