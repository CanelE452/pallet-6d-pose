"""Independent ID-based replay of all actual joined v66 source caches.

No merge helper is imported; no model forward, fitting, GPU or private read.
"""
from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'source_audit/JOINED_CACHE_AUDIT.json'
QUERY_FEATURES=('x','rgb_native','sam','context','box_grid','sample_id','group_id','image_sha256','global_index64','global_index8','global_query_valid')
GLOBAL_FEATURES=('global_rgb','global_sam','global_prompt','global_id','global_image_sha256','global_box')
TRUTH_SUFFIX=('allowed','valid','exact_state','radius2_state')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
 if sha(b['path'])!=b['sha256']:raise AssertionError('Changed binding: '+b['path'])
def same(a,b,name):
 # Concatenation widens fixed-width Unicode metadata, preserving its values.
 dtype_equal=a.dtype==b.dtype or(a.dtype.kind in 'US'and b.dtype.kind in 'US')
 if a.shape!=b.shape or not dtype_equal or not np.array_equal(a,b):raise AssertionError('Array mismatch: '+name)
def array_sha(a):
 h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()

def audit_source(d,source,paths):
 rows=np.flatnonzero(d['source']==source);targets=np.flatnonzero(d['global_source']==source)
 counters=dict(source=source,queries=len(rows),targets=len(targets),query_feature_fields_exact=0,global_feature_fields_exact=0,truth_fields_exact=0)
 with np.load(paths['feature'],allow_pickle=False)as f:
  fs=f['sample_id'];fg=f['global_id'];query_index={str(k):i for i,k in enumerate(fs)};global_index={str(k):i for i,k in enumerate(fg)}
  if set(d['sample_id'][rows])!=set(fs)or set(d['global_id'][targets])!=set(fg):raise AssertionError('Source identity sets '+source)
  qi=np.asarray([query_index[str(k)]for k in d['sample_id'][rows]],np.int64);ti=np.asarray([global_index[str(k)]for k in d['global_id'][targets]],np.int64)
  if len(query_index)!=len(rows)or len(global_index)!=len(targets):raise AssertionError('Duplicate source IDs')
  for k in QUERY_FEATURES:
   same(d[k][rows],f[k][qi],source+'/'+k);counters['query_feature_fields_exact']+=1
  for k in GLOBAL_FEATURES:
   same(d[k][targets],f[k][ti],source+'/'+k);counters['global_feature_fields_exact']+=1
  owner=f['global_id'][f['target_index'][qi]]
  same(d['global_id'][d['target_index'][rows]],owner,source+'/target owner identity')
  lookup={str(k):i for i,k in enumerate(d['global_id'])};expected=np.asarray([lookup[str(k)]for k in owner],np.int64)
  same(d['target_index'][rows],expected,source+'/target offset')
  if 'label'in paths:
   with np.load(paths['label'],allow_pickle=False)as g:
    gqi={str(k):i for i,k in enumerate(g['sample_id'])};gti={str(k):i for i,k in enumerate(g['global_id'])}
    if set(gqi)!=set(query_index)or set(gti)!=set(global_index):raise AssertionError('Separate source truth ID set mismatch')
    qidx=np.asarray([gqi[str(k)]for k in d['sample_id'][rows]],np.int64);tidx=np.asarray([gti[str(k)]for k in d['global_id'][targets]],np.int64)
    for prefix,ix,take in (('dense',qidx,rows),('global',tidx,targets)):
     for suffix in TRUTH_SUFFIX:
      k=prefix+'_'+suffix;same(d[k][take],g[k][ix],source+'/'+k);counters['truth_fields_exact']+=1
    for k in ('image_sha256','global_image_sha256','global_box'):
     ix=qidx if k=='image_sha256'else tidx;take=rows if k=='image_sha256'else targets
     same(d[k][take],g[k][ix],source+'/truth binding '+k)
    if 'query'in f and 'roi_side'in f:
     same(g['query'][qidx],f['query'][qi],source+'/query geometry')
     same(g['side'][qidx],f['roi_side'][qi],source+'/side geometry')
    else:
     meta={r['id']:r for r in map(json.loads,Path(paths['metadata']).open())}
     expected_q=[];expected_side=[]
     for sid,oid in zip(d['sample_id'][rows],owner):
      r=meta[str(oid)];corner=int(str(sid).rsplit(':',1)[1]);expected_q.append(r['query'][corner]);expected_side.append(r['side'])
     if not np.array_equal(g['query'][qidx],expected_q)or not np.array_equal(g['side'][qidx],expected_side):raise AssertionError('Original source geometry label join')
  else:
   if source!='SYNTH':raise AssertionError('Only SYNTH direct truth allowed')
   pairs=np.asarray(((0,0),(1,0),(1,1),(2,0),(2,1),(2,2)),np.int8)
   for prefix,ix,take in (('dense',qi,rows),('global',ti,targets)):
    valid=f[prefix+'_valid'][ix].astype(bool);joint=f['joint'if prefix=='dense'else'global_joint'][ix]
    codes=pairs[joint.clip(0)];allowed=np.where(valid,np.left_shift(1,joint.clip(0)),63).astype(np.uint8)
    same(d[prefix+'_allowed'][take],allowed,source+'/'+prefix+' allowed')
    same(d[prefix+'_valid'][take],valid,source+'/'+prefix+' valid')
    for j,suffix in enumerate(('exact_state','radius2_state')):
     same(d[prefix+'_'+suffix][take],np.where(valid,codes[...,j],-1).astype(np.int8),source+'/'+prefix+' state')
    counters['truth_fields_exact']+=4
 counters['center_informative']=int((d['dense_valid'][rows,4,4]&(d['dense_allowed'][rows,4,4]!=63)).sum())
 counters['center_exact_E']=int((d['dense_exact_state'][rows,4,4]==1).sum())
 counters['center_exact_U']=int((d['dense_exact_state'][rows,4,4]==-1).sum())
 counters['truth_path']=paths.get('label','SYNTH unchanged six-joint source renderer labels')
 return counters

def main():
 if OUT.exists():raise FileExistsError('Preserve earlier joined-cache audit')
 start=time.monotonic();sealpath=ROOT/'joined_source_cache/SEAL.json'
 if not sealpath.exists():raise FileNotFoundError('Wait for complete merger seal; do not audit partial files')
 sealbinding=bind(sealpath);seal=json.loads(sealpath.read_text())
 if seal.get('complete')is not True or seal.get('current223_used')is not False:raise PermissionError('Complete source-only joined seal required')
 check(seal['input_lock']);lock=json.loads(Path(seal['input_lock']['path']).read_text())
 for b in lock['bindings']:check(b)
 for b in seal['outputs']:check(b)
 merge=ROOT/'model/merge_partial_caches.py'
 if sha(merge)!=lock['merge_code_sha256']or sha(merge)!=seal['merge_code_sha256']:raise AssertionError('Join implementation binding')
 trainerpath=ROOT/'model/fit_partial_head.py';modelpath=ROOT/'model/query_global_amodal_head.py';trainerbinding=bind(trainerpath);modelbinding=bind(modelpath)
 # Source-checked import only. No CLI entry, smoke, fit, model constructor or GPU.
 sys.path.insert(0,str(trainerpath.parent));spec=importlib.util.spec_from_file_location('v66_actual_cache_load_only',trainerpath);trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
 import torch
 torch.set_num_threads(1)
 deny=ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json';denied=set(json.loads(deny.read_text())['image_sha256'])
 reports=[];valhash=None;commonval=True;identities={}
 for b in seal['outputs']:
  d=trainer.load_cache(b['path']);n=len(d['x']);t=len(d['global_id']);candidate=b['candidate'];split=b['split']
  if len(set(d['sample_id']))!=n or len(set(d['global_id']))!=t:raise AssertionError('Joined duplicate IDs')
  expected=lock['validation_sources']if split=='VAL'else lock['candidates'][candidate]
  if set(d['source'])!=set(expected)or set(d['global_source'])!=set(expected):raise AssertionError('Candidate source set')
  if set(d['image_sha256'])&denied or set(d['global_image_sha256'])&denied:raise AssertionError('Private RGB SHA overlap')
  if not np.array_equal(d['global_id'][d['target_index']],np.asarray([str(s).rsplit(':',1)[0]for s in d['sample_id']])):raise AssertionError('Sample ID/target owner identity')
  sources=[audit_source(d,s,lock['sources'][split][s])for s in expected]
  for prefix in ('dense','global'):
   if not(d[prefix+'_allowed'][~d[prefix+'_valid']]==63).all():raise AssertionError('Invalid label mapping')
  hashes={k:array_sha(v)for k,v in d.items()}
  if split=='VAL':
   if valhash is None:valhash=hashes
   elif hashes!=valhash:raise AssertionError('Candidate common VAL arrays differ')
  identities[(candidate,split)]={k:set(d[k])for k in ('group_id','sample_id','image_sha256','global_id','global_image_sha256')}
  reports.append(dict(candidate=candidate,split=split,path=b['path'],sha256=b['sha256'],trainer_load_cache_PASS=True,queries=n,targets=t,fields=len(d),field_shapes={k:list(v.shape)for k,v in d.items()},array_sha256=hashes,sources=sources))
  print(json.dumps(dict(candidate=candidate,split=split,queries=n,targets=t,source_ID_feature_truth_replay='PASS',seconds=time.monotonic()-start)),flush=True)
  del d;gc.collect()
 intersections={}
 for candidate in lock['candidates']:
  intersections[candidate]={}
  for k in ('group_id','sample_id','image_sha256','global_id','global_image_sha256'):
   overlap=identities[(candidate,'TRAIN')][k]&identities[(candidate,'VAL')][k];intersections[candidate][k]=len(overlap)
   if overlap:raise AssertionError('TRAIN/VAL overlap '+candidate+'/'+k)
 check(sealbinding);check(trainerbinding);check(modelbinding)
 result=dict(schema='V66_INDEPENDENT_ACTUAL_JOINED_SOURCE_CACHE_AUDIT',PASS=True,joined_seal=sealbinding,input_lock=seal['input_lock'],trainer=trainerbinding,model=modelbinding,audit_code=bind(__file__),
  runtime=dict(executable=sys.executable,torch=str(torch.__version__)),outputs=reports,TRAIN_VAL_intersections=intersections,current_evaluation_SHA_overlap=0,eval_denylist=bind(deny),
  all_input_fields_preserved_byte_exact=True,all_truth_replayed_by_ID_from_named_source_caches=True,all_global_target_offsets_exact=True,common_both_VAL_all_arrays_exact=True,
  truth_distinction='SYNTH explicit renderer joint; ORIGINAL_YCB original-mask partial uncertainty; CONTROLLED_YCB certified-paste partial truth; PALLOX paired conditional teacher, not gold. No original-full-only-to-hardE conversion.',
  new_model_forward_calls=0,optimizer_updates=0,GPU_calls=0,private_RGB_GT_score_reads=0,PnP_calls=0,sealed_files_modified=False,seconds=time.monotonic()-start)
 with OUT.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(PASS=True,audit=bind(OUT),seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
