"""Join sealed input-only features and label-only partial truth by stable IDs."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np

QUERY_INPUTS=['x','rgb_native','sam','context','box_grid','sample_id','group_id','image_sha256','global_index64','global_index8','global_query_valid']
GLOBAL_INPUTS=['global_rgb','global_sam','global_prompt','global_id','global_image_sha256','global_box']
QUERY_TRUTH=['dense_allowed','dense_valid','dense_exact_state','dense_radius2_state']
GLOBAL_TRUTH=['global_allowed','global_valid','global_exact_state','global_radius2_state']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def align(source,feature_path,label_path=None,metadata_path=None):
 f=read(feature_path);n=len(f['x']);t=len(f['global_rgb'])
 d={k:f[k]for k in QUERY_INPUTS+GLOBAL_INPUTS};d['target_index']=f['target_index'].copy()
 d['source']=np.full(n,source);d['global_source']=np.full(t,source)
 if label_path is None:
  if source!='SYNTH':raise ValueError('Only unchanged synthetic rendered hard labels may bypass new partialGT')
  for prefix in ['dense','global']:
   valid=f[prefix+'_valid'].astype(bool);joint=f['joint'if prefix=='dense'else'global_joint']
   if not np.isin(joint[valid],np.arange(6)).all():raise ValueError('synthetic valid labels must be explicit six-state codes')
   d[prefix+'_allowed']=np.where(valid,1<<joint.clip(0),63).astype(np.uint8);d[prefix+'_valid']=valid
   pairs=np.asarray([(0,0),(1,0),(1,1),(2,0),(2,1),(2,2)],np.int8)
   states=pairs[joint.clip(0)]
   d[prefix+'_exact_state']=np.where(valid,states[...,0],-1).astype(np.int8)
   d[prefix+'_radius2_state']=np.where(valid,states[...,1],-1).astype(np.int8)
 else:
  g=read(label_path)
  si={str(s):i for i,s in enumerate(g['sample_id'])};gi={str(s):i for i,s in enumerate(g['global_id'])}
  if len(si)!=n or len(gi)!=t or set(si)!=set(f['sample_id'])or set(gi)!=set(f['global_id']):raise ValueError('Feature/GT complete identity sets differ')
  qi=np.asarray([si[str(s)]for s in f['sample_id']]);ti=np.asarray([gi[str(s)]for s in f['global_id']])
  for k in QUERY_TRUTH:d[k]=g[k][qi]
  for k in GLOBAL_TRUTH:d[k]=g[k][ti]
  if not np.array_equal(g['image_sha256'][qi],f['image_sha256']):raise ValueError('query RGB SHA mismatch')
  if not np.array_equal(g['global_image_sha256'][ti],f['global_image_sha256']):raise ValueError('target RGB SHA mismatch')
  if not np.array_equal(g['global_box'][ti],f['global_box']):raise ValueError('input prompt box mismatch')
  if not np.array_equal(g['global_id'][g['target_index'][qi]],f['global_id'][f['target_index']]):raise ValueError('query-target identity mapping differs')
  if 'query'in f and 'roi_side'in f:
   if not np.array_equal(g['query'][qi],f['query'])or not np.array_equal(g['side'][qi],f['roi_side']):raise ValueError('input query/side mismatch')
  else:
   if not metadata_path:raise ValueError('feature cache without query metadata requires frozen original source manifest')
   metadata={r['id']:r for r in map(json.loads,Path(metadata_path).open())}
   for i,sample in enumerate(f['sample_id']):
    target=str(f['global_id'][f['target_index'][i]]);m=metadata[target];k=int(str(sample).rsplit(':',1)[1])
    if m['image_sha256']!=str(f['image_sha256'][i])or not np.array_equal(m['box'],f['global_box'][f['target_index'][i]]):raise ValueError('frozen source input identity/box mismatch')
    if not np.array_equal(m['query'][k],g['query'][qi[i]])or m['side']!=g['side'][qi[i]]:raise ValueError('frozen original source query/side mismatch')
  # Native truth can be more restrictive than the continuous P3 mask at a
  # rounded image edge, but cannot certify a point outside the input canvas.
  if(d['dense_valid']&~f['x'][:,64].astype(bool)).any():raise ValueError('label outside feature native canvas')
 for prefix in ['dense','global']:
  if(d[prefix+'_allowed'][~d[prefix+'_valid']]!=63).any():raise ValueError('invalid partial labels')
 return d
def concatenate(parts):
 result={};offset=0;indices=[]
 for d in parts:
  indices.append(d['target_index']+offset);offset+=len(d['global_rgb'])
 for k in QUERY_INPUTS+GLOBAL_INPUTS+QUERY_TRUTH+GLOBAL_TRUTH+['source','global_source']:
  result[k]=np.concatenate([d[k]for d in parts],axis=0)
 result['target_index']=np.concatenate(indices).astype(np.int64)
 if len(set(result['sample_id']))!=len(result['sample_id'])or len(set(result['global_id']))!=len(result['global_id']):raise ValueError('duplicate joined IDs')
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--lock',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 lock=json.loads(Path(a.lock).read_text());out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
 for b in lock['bindings']:
  if sha(b['path'])!=b['sha256']:raise ValueError('input binding changed '+b['path'])
 if lock['merge_code_sha256']!=sha(__file__):raise ValueError('merge code changed')
 seals=[];start=time.monotonic()
 for split in ['TRAIN','VAL']:
  parts={}
  for src,paths in lock['sources'][split].items():
   parts[src]=align(src,paths['feature'],paths.get('label'),paths.get('metadata'))
  for candidate,sources in lock['candidates'].items():
   if split=='VAL':sources=lock['validation_sources']
   joined=concatenate([parts[s]for s in sources]);path=out/(candidate+'_'+split+'.npz')
   np.savez_compressed(path,**joined)
   counts=[]
   for s in sources:
    d=parts[s];mask=d['dense_allowed'][:,4,4];valid=d['dense_valid'][:,4,4]
    counts.append(dict(source=s,queries=len(mask),targets=len(d['global_id']),center_native_valid=int(valid.sum()),center_informative=int((valid&(mask!=63)).sum()),center_unique_joint=int((valid&((mask&(mask-1))==0)).sum()),global_informative_cells=int((d['global_valid']&(d['global_allowed']!=63)).sum())))
   seals.append(dict(path=str(path),sha256=sha(path),candidate=candidate,split=split,source_counts=counts,bytes=path.stat().st_size))
   print(candidate,split,counts,flush=True)
 result=dict(schema='V66_JOINED_PARTIAL_SOURCE_CACHE_SEAL',complete=True,input_lock=dict(path=a.lock,sha256=sha(a.lock)),merge_code_sha256=sha(__file__),outputs=seals,current223_used=False,new_training_updates=0,new_PnP=0,seconds=time.monotonic()-start)
 (out/'SEAL.json').write_text(json.dumps(result,indent=2)+'\n');print('MERGE_COMPLETE',result['seconds'],flush=True)
if __name__=='__main__':main()
