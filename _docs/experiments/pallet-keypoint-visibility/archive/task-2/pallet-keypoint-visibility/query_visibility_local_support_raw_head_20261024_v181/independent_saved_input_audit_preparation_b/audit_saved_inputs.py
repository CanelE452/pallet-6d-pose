"""Prospective saved SAM/raw-input audit. No model or weight loader/import.

Actual saved bodies require a root entry. This script never reads full scene
RGB, source GT/depth, features, private data, or the official checkpoint.
"""
from pathlib import Path
import argparse,hashlib,json,math,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
OUT=V/'independent_actual_saved_input_review_b'

def req(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def sha(p):
 p=Path(p).resolve();req(p.is_relative_to(P) and not any(s in str(p).lower() for s in ('_v112','_v169','_v177','_v178','current223','/private_evaluation/')),'Forbidden path')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):req(bind(b['path'])==b,'FileSHA '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)

def sample(logits,q,L,on):
 """Independent direct four-tap exact96 coordinates, FP64, zero-pad, >0."""
 mask=np.zeros((96,96),bool);valid=mask.copy()
 if not on:return mask,valid
 q=qarray(q);req(q.shape==(2,) and np.isfinite(q).all() and math.isfinite(L) and L>0,'Available qL')
 h,w=logits.shape;axis=(np.arange(96,dtype=np.float64)+.5)/96-.5
 for row in range(96):
  y=float(q[1]+axis[row]*L);x=q[0]+axis*L;xx=np.floor(x).astype(np.int64);yy=math.floor(y);fx=x-xx;fy=y-yy
  z=np.zeros(96,np.float64)
  for ay in [0,1]:
   for ax in [0,1]:
    cx=xx+ax;cy=yy+ay;inside=(cx>=0)&(cx<w)&(cy>=0)&(cy<h)
    weight=(fx if ax else 1-fx)*(fy if ay else 1-fy)
    z[inside]+=logits[cy,cx[inside]].astype(np.float64)*weight[inside]
  valid[row]=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5);mask[row]=(z>0)&valid[row]
 return mask,valid

def sam_packet(group,used):
 if group['outputs'] is None:
  req(not group['runtime_available'] and not group['prompt_available'],'Only explicit missing prompt yields no output');return None
 req(group['runtime_available'] and group['prompt_available'],'Saved SAM is actual available')
 path=check(group['outputs']);used[str(path)]=group['outputs']
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 mm=group['original_input_metadata8'];hw=tuple(group['raw_hw'] if 'raw_hw' in group else mm[0]['raw_hw'])
 schema={'image_logits':(np.float32,hw),'lowres_logits':(np.float32,(256,256)),'predicted_quality':(np.float32,(1,)),
  'mask_logits_gt0':(np.bool_,hw),'box_xyxy':(np.float64,(4,)),'original_query_xy':(np.float64,(8,2)),
  'original_L':(np.float64,(8,)),'native_slot':(np.int8,(8,)),'original_global_index':(np.int64,(8,)),
  'original_query_available':(np.bool_,(8,)),'prediction_available':(np.bool_,(8,))}
 req(set(a)==set(schema)|{'original_occurrence_id'},'SAM exact12 fields')
 for k,(dtype,shape) in schema.items():req(a[k].dtype==dtype and a[k].shape==shape,'SAM schema '+k)
 req(a['original_occurrence_id'].dtype.kind=='U' and a['original_occurrence_id'].shape==(8,),'SAM original strings')
 req(all(np.isfinite(a[k]).all() for k in ['image_logits','lowres_logits','predicted_quality']),'Finite SAM outputs')
 req(same(a['mask_logits_gt0'],a['image_logits']>0),'SAM threshold fixed0')
 exp={'original_query_xy':np.asarray([qarray(r['original_query_xy']) for r in mm],np.float64),
  'original_L':np.asarray([np.nan if r['original_L'] is None else r['original_L'] for r in mm],np.float64),
  'native_slot':np.asarray([r['native_slot'] for r in mm],np.int8),'original_global_index':np.asarray([r['original_global_index'] for r in mm],np.int64),
  'original_occurrence_id':np.asarray([r['original_occurrence_id'] for r in mm]),
  'original_query_available':np.asarray([r.get('original_query_available',r.get('original_input_flags',{}).get('query_available')) for r in mm],bool)}
 for k,v in exp.items():req(same(a[k],v),'SAM original input metadata '+k)
 req(same(a['prediction_available'],exp['original_query_available']) and same(a['box_xyxy'],group['box_xyxy']),'SAM original U and exact q8box')
 return a

def run(entry):
 e=read(entry);req(e.get('approved_saved_source_input_audit') is True and e.get('source_only') is True,'Separate root saved-input entry required')
 req(e['code']==bind(__file__) and e['scope']==bind(SCOPE),'Frozen audit entry')
 sam=read(check(e['public_CF_SAM_complete']));source=read(check(e['synthetic_SAM_complete']));cache=read(check(e['raw_cache_complete']))
 req(all(x['completed'] and x['PASS'] for x in [sam,source,cache]),'Completed source dependencies')
 req(sam['model_state_byte_unchanged'] and source['model_state_byte_unchanged'],'Saved frozen model states unchanged')
 req(sam['checkpoint']==source['checkpoint'] and sam['source_commit']==source['source_commit'],'One frozen official model')
 frozen=lines(check(sam['groups_manifest']));req(len(frozen)==len(sam['groups'])==662,'All new SAM groups')
 for old,g in zip(frozen,sam['groups']):
  for k,v in old.items():req(g[k]==v,'Frozen groupmetadata preserved '+k)
 repeat=sam['first_group_repeat'];req(repeat['attempted'] and repeat['PASS'],'Actual newSAM first repeat sealed')
 with np.load(check(repeat['outputs']),allow_pickle=False) as z:ra={k:z[k].copy() for k in z.files}
 with np.load(check(sam['groups'][0]['outputs']),allow_pickle=False) as z:
  req(set(ra)=={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0'},'Independent repeat image fields')
  for k,a in ra.items():req(z[k].dtype==a.dtype and z[k].shape==a.shape and z[k].tobytes()==a.tobytes(),'Actual firstrepeat savedbytes '+k)
 groups={'SYNTH_SCENE_%06d'%g['scene_index']:g for g in source['scenes']}
 groups.update({g['group_id']:g for g in sam['groups']});req(len(groups)==1174,'All512+662 SAM groups')
 req(not OUT.exists(),'Preserve prior audit attempt');OUT.mkdir();start=time.monotonic();used={};counts={};pixels=0
 try:
  req(set(cache['domains'])=={'SYNTH_TRAIN4096','PUBLIC_TRAIN1480','CF3816'},'All source domains')
  for domain,record in cache['domains'].items():
   rows=lines(check(record['metadata']))
   with np.load(check(record['inputs']),allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
   n=len(rows);req(n==record['rows'] and same(x['row_id'],[r['row_id'] for r in rows]),'All original row order')
   expected=4096 if domain=='SYNTH_TRAIN4096' else 1480 if domain=='PUBLIC_TRAIN1480' else 3816;req(n==expected,'Exact source counts')
   current_sam=None;sp=None;raw_path=None;raw=None
   for i,r in enumerate(rows):
    req(r['split']==('TRAIN' if domain=='SYNTH_TRAIN4096' else 'SOURCE_DEVELOPMENT'),'No public held inputs')
    req(same(x['query_xy'][i],qarray(r['query_xy'])) and (x['L'][i]==r['L'] or r['L'] is None and np.isnan(x['L'][i])),'Original qL')
    req(int(x['native_slot'][i])==r['native_slot'] and bool(x['original_query_available'][i])==r['original_query_available'],'Original native and U')
    if r['sam_group_id']!=current_sam:
     current_sam=r['sam_group_id'];sp=sam_packet(groups[current_sam],used)
    mm=groups[current_sam]['original_input_metadata8'][r['native_slot']]
    req(mm['original_global_index']==r['original_global_index'] and mm['native_slot']==r['native_slot'] and same(qarray(mm['original_query_xy']),qarray(r['query_xy'])) and mm['original_L']==r['L'],'Raw row to original SAM identity/qL')
    req(mm['original_occurrence_id']==r['original'].get('original_query_id',r['row_id']),'Raw row to original SAM query string')
    runtime=bool(groups[current_sam]['prompt_available'] and groups[current_sam]['runtime_available']);on=runtime and r['original_query_available']
    mask,av=sample(None if sp is None else sp['image_logits'],r['query_xy'],r['L'],on)
    req(same(mask,x['sam_mask96'][i]) and same(av,x['sam_available96'][i]),'Independent full96 SAM sampling')
    if on:req(same(av,x['valid96'][i]),'Original same-grid domain')
    if r['raw_packet']['path']!=raw_path:
     raw_path=str(check(r['raw_packet']));used[raw_path]=r['raw_packet']
     with np.load(raw_path,allow_pickle=False) as z:raw={k:z[k].copy() for k in ['rgb96','valid96']}
    k=r['raw_row'];req(raw['rgb96'].dtype==np.uint8 and raw['valid96'].dtype==np.bool_,'Original raw dtypes')
    req(x['rgb96'][i].tobytes()==raw['rgb96'][k].tobytes() and x['valid96'][i].tobytes()==raw['valid96'][k].tobytes(),'Original raw pixel/valid bytes')
    req(ash(x['rgb96'][i])==r['actual_raw_RGB96_sha256'] and ash(x['valid96'][i])==r['actual_raw_valid96_sha256'],'Raw row digest')
    pixels+=96*96
   counts[domain]=dict(rows=n,original_inputU=int((~x['original_query_available']).sum()),all_raw_pixels_and_SAM96_exact=True)
   req(counts[domain]['original_inputU']=={'SYNTH_TRAIN4096':80,'PUBLIC_TRAIN1480':194,'CF3816':360}[domain],'Original U count by domain')
  for b in used.values():check(b)
  write(OUT/'REVIEW.json',dict(completed=True,PASS=True,source_only=True,code=bind(__file__),entry=bind(entry),public_CF_SAM_complete=e['public_CF_SAM_complete'],synthetic_SAM_complete=e['synthetic_SAM_complete'],raw_cache_complete=e['raw_cache_complete'],domains=counts,all96_sample_positions_checked=pixels,used_saved_SAM_or_original_raw_files=len(used),original9392_and_U634_preserved=True,source_code_or_model_not_rerun=True,no_fullscene_RGB_GT_depth_feature_or_weight_array_read=True,physical_native_or_model_success_certified=False,selected_model=None,fully_solved=False,seconds=time.monotonic()-start))
 except BaseException:
  write(OUT/'FAILED_PRESERVED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

def prepare():
 # Pure invented four-tap boundary/zero/U oracle, no actual source bodies.
 z=np.arange(25,dtype=np.float32).reshape(5,5)-12
 m,a=sample(z,[2,2],2.,True);req(m.shape==a.shape==(96,96) and a.all(),'Invented inside')
 m0,a0=sample(np.zeros_like(z),[2,2],2.,True);req(not m0.any() and a0.all(),'Strict0')
 m,a=sample(None,None,None,False);req(not m.any() and not a.any(),'OriginalU no grid')
 q=[-.5,2];m,a=sample(z,q,2.,True);req(not a[:,:48].any() and a[:,48:].any(),'Pixelcenter support domain')
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),invented_checks4=True,actual_saved_body_reads=0,no_model_import_or_forward=True,root_saved_audit_entry_required=True))

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
