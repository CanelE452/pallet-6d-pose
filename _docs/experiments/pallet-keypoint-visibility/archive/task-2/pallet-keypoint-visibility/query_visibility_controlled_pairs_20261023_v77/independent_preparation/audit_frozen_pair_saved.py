"""Independent V77 source data/saved-result audit, no torch/model/RGB/P3 decode.

Actual execution is explicit after data or model seals. Reads only GT,
metadata, coordinate/context/canvas-validity input arrays and saved logits.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from pair_audit_primitives import (assert_teacher_order,audit_donor_split,
 audit_natural_selector,binary_metric,paired_saved_metrics,stable_bce,audit_batch96)
R=Path(__file__).resolve().parent;V77=R.parent;P=V77.parent
V76=P/'query_visibility_reference_local_support_20261023_v76'
SOURCES=('SYNTHETIC','YCB_PUBLIC_REAL')
FACTS=('query8_xy','L','query_center_valid','ROI_anyvalid','context42','query_box_xyxy','native_index')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(x):assert sha(x['path'])==x['sha256'],x['path']
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()]
def save(p,d):
 with Path(p).open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def read_npz(p,keys):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in keys}
def same(a,b):
 a=np.asarray(a);b=np.asarray(b);return np.array_equal(a,b,equal_nan=True)if a.dtype.kind in 'fc'and b.dtype.kind in 'fc'else np.array_equal(a,b)
def facts_from_manifests(manifests):
 d={k:[]for k in FACTS};d.update(local_valid=[],target_valid=[]);n=0
 for path in manifests:
  man=load(path);assert man['completed']
  for c in man['chunks']:
   check(c['file'])
   with np.load(c['file']['path'],allow_pickle=False)as z:
    for k in FACTS:d[k].append(z[k])
    d['local_valid'].append(z['crop_valid96']);d['target_valid'].append(z['target_valid128'])
    n+=len(z['L'])
 return {k:np.concatenate(v)for k,v in d.items()}
def admission(g,f):
 off=(g['native_GTref8']-f['query8_xy'])/f['L'][:,None,None]
 sq=np.isfinite(off).all(-1)&(off>=-.5).all(-1)&(off<.5).all(-1)
 n=len(f['L']);available=f['ROI_anyvalid']&f['local_valid'].reshape(n,8,-1).any(-1)&f['target_valid'].reshape(n,-1).any(-1)[:,None]
 primary=np.isin(g['label8'],[0,1])&f['query_center_valid']&available&sq
 return primary,sq,available

def data_audit():
 protocol=load(V77/'pair_generation/PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json')
 for b in protocol['bindings'].values():check(b)
 seal=load(V77/'pair_generation/SEAL.json');assert seal['completed']
 for b in seal['outputs']:check(b)
 selected=lines(V77/'pair_generation/PAIR_SELECTION_GT_ONLY.jsonl')
 eligibility=lines(V77/'pair_generation/ORIGINAL_5203_ELIGIBILITY_GT_ONLY.jsonl')
 original_meta=lines(V76/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl')
 manifest=lines(V77/'pair_generation/INPUTS_RGB_ONLY.jsonl')
 lineage=lines(V77/'pair_generation/INPUT_LINEAGE.jsonl')
 keys=['id','source','split','label8','native_GTref8','convex_self_proxy8_separate','near_boundary8','rule8']
 before=read_npz(protocol['bindings']['original_teacher_corpus']['path'],keys)
 after=read_npz(V77/'pair_generation/AFTER_GT_ONLY.npz',keys+['original_row_index','after_row_index','anchor_k','pair_anchor_success','anchor_training_target','original_primary8','query_valid8','reference_in_local_square8'])
 old_bind=load(V76/'mixed_fit/SOURCE_INPUT_BINDINGS.json')
 old_man=[]
 for pack in old_bind['packs']:
  for k in ('manifest','seal','lineage'):check(pack[k])
  old_man.append(pack['manifest']['path'])
 f0=facts_from_manifests(old_man);f1=facts_from_manifests([V77/'after_input_pack/PACK_MANIFEST.json'])
 n=len(selected);orig=after['original_row_index'];ak=after['anchor_k'].astype(int);rr=np.arange(n)
 assert len(eligibility)==5203 and len(before['id'])==5203 and len(manifest)==len(lineage)==len(orig)==n
 assert np.array_equal(after['after_row_index'],np.arange(n)) and len(np.unique(orig))==n
 assert np.all((orig>=0)&(orig<5203))and np.all((ak>=0)&(ak<8))
 for key in FACTS+('local_valid','target_valid'):assert same(f1[key],f0[key][orig]),('Changed pair fact',key)
 for key in ['source','split','native_GTref8','convex_self_proxy8_separate']:assert same(after[key],before[key][orig]),key
 assert all(after['id'][i]==before['id'][j]+'::V77_OPAQUE_AFTER' for i,j in enumerate(orig))
 p0,sq0,av0=admission(before,f0);p1,sq1,av1=admission(after,f1)
 expected=np.flatnonzero((p0&(before['label8']==1)).any(-1));assert np.array_equal(orig,expected)
 salt=protocol['salt']
 def digest(kind,ident):return hashlib.sha256((salt+'\n'+kind+'\n'+ident).encode()).hexdigest()
 donors=[]
 for i,j in enumerate(orig):
  row=selected[i];m=manifest[i];l=lineage[i];em=eligibility[j]
  eligible=np.flatnonzero(p0[j]&(before['label8'][j]==1));fixed=min(eligible,key=lambda k:(digest('anchor',str(before['id'][j])+'\n'+str(k)),int(k)))
  assert int(fixed)==ak[i]==row['anchor_k'] and em['eligible']and em['anchor_k']==ak[i]
  assert row['original_row_index']==j and m['frame_index']==i and l['original_row_index']==j
  assert m['id']==l['after_id']==after['id'][i] and m['source']==after['source'][i]and m['split']==after['split'][i]
  assert np.array_equal(np.asarray(m['initial_points']),f1['query8_xy'][i])and m['roi_side']==f1['L'][i]
  assert set(m)==set(protocol['input_manifest_whitelist'])and set(l)==set(protocol['input_lineage_whitelist'])
  d=row['donor'];donors.append(dict(source=after['source'][i],split=after['split'][i],image_sha256=row['image_sha256'],donor_image_sha256=d['image_sha256'],donor_split=d['split'],donor_source='YCB_PUBLIC_REAL',object_id=original_meta[j].get('object_id'),donor_object_id=d['object_id']))
 assert_donor=audit_donor_split(donors)
 assert_teacher_order(before['label8'][orig],after['label8'])
 assert np.array_equal(after['original_primary8'],p0[orig])and np.array_equal(after['query_valid8'],f1['query_center_valid'])and np.array_equal(after['reference_in_local_square8'],sq1)
 assert np.all(p0[orig,ak]&(before['label8'][orig,ak]==1))
 good=after['pair_anchor_success']&p1[rr,ak]&(after['label8'][rr,ak]==0)
 assert np.array_equal(after['anchor_training_target'],np.where(after['pair_anchor_success'],0,-1))
 counts=[]
 for source in SOURCES:
  for split in ['train','val']:
   for domain,g,p in [('NATURAL_ORIGINAL',before,p0),('CONTROLLED_AFTER',after,p1)]:
    base=np.broadcast_to(((g['source']==source)&(g['split']==split))[:,None],g['label8'].shape)
    counts.append(dict(domain=domain,source=source,split=split,all_slots=int(base.sum()),U=int((base&(g['label8']==-1)).sum()),C=int((base&(g['label8']==0)).sum()),V=int((base&(g['label8']==1)).sum()),primary_C=int((base&p&(g['label8']==0)).sum()),primary_V=int((base&p&(g['label8']==1)).sum())))
 pools={};nonanchor=np.ones((n,8),bool);nonanchor[rr,ak]=False
 for s in SOURCES:
  for c in [0,1]:
   pools[f'natural|{s}|{c}']=int((p0&(before['source']==s)[:,None]&(before['split']=='train')[:,None]&(before['label8']==c)).sum())
   pools[f'after_nonanchor|{s}|{c}']=int((p1&nonanchor&(after['source']==s)[:,None]&(after['split']=='train')[:,None]&(after['label8']==c)).sum())
  pools[f'matched_pairs|{s}']=int((good&(after['source']==s)&(after['split']=='train')).sum())
 assert min(pools.values())>0,'Frozen recipe requires all pools nonempty; no fallback'
 report=dict(PASS=True,scope='SOURCE GT + input coordinate/context/canvas validity only; RGB/P3 not decoded.',original_rows=5203,after_rows=n,original_slots=41624,after_slots=n*8,all_eligible_retained=True,deterministic_primaryV_anchor_exact=True,all_fixed_geometry_context_canvasvalid_exact=True,donor_split_audit=assert_donor,teacher_monotonicity=True,attempted_anchor_success=int(after['pair_anchor_success'].sum()),attempted_anchor_failure=int((~after['pair_anchor_success']).sum()),known_admitted_V_to_C_pairs=int(good.sum()),counts=counts,train_pools=pools,RGB_P3_model_current_forward_fit_PnP=0)
 return report,before,after,f0,f1,p0,p1,sq0,sq1,av0,av1,good

def saved_audit(data,run):
 dr,g0,g1,f0,f1,p0,p1,sq0,sq1,av0,av1,good=data
 freeze=load(run/'MODEL_FREEZE.json');complete=load(run/'FIT_COMPLETE.json');history=load(run/'HISTORY.json');pub=load(run/'SOURCE_RESULTS.json')
 assert freeze['completed']and complete['completed']and freeze['optimizer_updates']==complete['optimizer_updates']==2000
 assert freeze['threshold']==.5 and freeze['parameter_count']==41237 and freeze['current_used_for_fit_or_checkpoint_or_threshold_selection']is False and freeze['PnP']==0
 for k in ['checkpoint','model_code','decisions','trainer','entry','source_inputs']:check(freeze[k])
 for k in ['model_freeze','history','scores','source_results']:check(complete[k])
 selector=audit_natural_selector(history,metric_key='naturalVAL4strata_macroBCE',strata_key='stratumBCE')
 assert selector['best_update']==freeze['best_update']==complete['best_update']==pub['best_update']and selector['natural4strata_macroBCE']==freeze['best_naturalVAL4strata_macroBCE']==pub['selection_naturalVALmacroBCE']
 y=np.concatenate([g0['label8'],g1['label8']]);primary=np.concatenate([p0,p1]);sq=np.concatenate([sq0,sq1]);av=np.concatenate([av0,av1]);N=len(y)
 with np.load(run/'SELECTED_SOURCE_SCORES.npz',allow_pickle=False)as z:
  for k in ['id','source','split']:assert np.array_equal(z[k],np.concatenate([g0[k],g1[k]]))
  assert np.array_equal(z['primary_loss_mask'],primary)and np.array_equal(z['reference_in_local_square_GT_only'],sq)and np.array_equal(z['input_available'],av)
  assert np.array_equal(z['domain'],np.r_[np.zeros(5203,int),np.ones(len(g1['id']),int)])
  assert np.array_equal(z['after_original_row_index'],g1['original_row_index'])and np.array_equal(z['after_anchor_k'],g1['anchor_k'])and np.array_equal(z['pair_good'],good)
  logits=z['support_logit'];nll=z['NLL_FP32'];assert logits.shape==nll.shape==y.shape
 assert np.isfinite(logits).all()and np.isfinite(nll).all()
 diff=abs(stable_bce(logits[primary].astype(float),y[primary])-nll[primary]);assert diff.max()<1e-5
 metrics=[];strata=[]
 for domain,name,start,g,pm in [(0,'NATURAL_ORIGINAL',0,g0,p0),(1,'CONTROLLED_AFTER',5203,g1,p1)]:
  z=logits[start:start+len(g['id'])];nl=nll[start:start+len(g['id'])]
  for s in SOURCES:
   for split in ['train','val']:
    take=pm&(g['source']==s)[:,None]&(g['split']==split)[:,None];m=binary_metric(z,g['label8'],take);published=pub['results']['natural_and_after'][f'{name}|{s}|{split}']['metrics']
    assert m['N']==published['N']and m['CM_rows_CV_cols_CV']==published['confusion_rows_GT_cols_pred_COVERED_VISIBLE']
    for key,pkey in [('V_precision','visible_precision'),('V_recall','visible_recall'),('C_precision','covered_precision'),('C_recall','covered_recall')]:assert m[key]==published[pkey]
    assert float(nl[take].astype(float).mean())==published['mean_BCE_FP32']
    metrics.append(dict(domain=name,source=s,split=split,**m))
    if domain==0 and split=='val':
     for c in [0,1]:
      mask=take&(g['label8']==c);strata.append(dict(source=s,label=c,N=int(mask.sum()),BCE_FP32=float(nl[mask].astype(float).mean())))
 orig=g1['original_row_index'];paired=paired_saved_metrics(logits[orig],logits[5203:],g0['label8'][orig],g1['label8'],p0[orig],p1,g1['anchor_k'],g1['source'],g1['split'],g1['pair_anchor_success'])
 for row in paired:
  pp=pub['results']['paired_anchors'][row['source']+'|'+row['split']]
  assert row['attempted_pairs']==pp['attempted_rows'] and row['failed_or_unknown_or_invalid_anchor_rows']==pp['failed_or_unknown_or_invalid_anchor_rows']
  assert row['known_admitted_V_to_C_pairs']==pp['N']and row['both_correct_known_pair']==pp['both_correct']and row['before_V_correct_known_pair']==pp['before_VISIBLE_correct']and row['after_C_correct_known_pair']==pp['after_COVERED_correct']
 return dict(PASS=True,checkpoint=freeze['checkpoint'],selection=selector,saved_natural4strata_macroBCE=float(np.mean([r['BCE_FP32']for r in strata])),saved_primary_NLL64_vs_FP32_maxdiff=float(diff.max()),source_metrics=metrics,natural_VAL_strata=strata,paired_and_same_after_control_diagnostics=paired,known_U_NLL_placeholder_excluded=True,model_forward_fit_current_PnP=0)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--saved',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 data=data_audit();save(a.output/'INDEPENDENT_PAIR_DATA_AUDIT.json',dict(**data[0],code=bind(__file__)))
 if a.saved:save(a.output/'INDEPENDENT_SAVED_RESULT_AUDIT.json',dict(**saved_audit(data,V77/'runs/PAIRED_SUPPORT_V1'),code=bind(__file__)))
 save(a.output/'AUDIT_SEAL.json',dict(bindings=[bind(p)for p in sorted(a.output.iterdir())if p.is_file()],original_files_modified=False))
 print(json.dumps(dict(PASS=True,output=str(a.output.resolve()))))
if __name__=='__main__':main()
