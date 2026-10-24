"""Independent saved-only C6/C7 math. No model/trainer import or feature read."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent
ARMS=('C6_EXTERNAL_ONLY_RAW_PRIOR','C7_EXTERNAL_ONLY_CLIPPED_PRIOR2')
DESIGN=V/'ROOT_C6_C7_DESIGN_BEFORE_SOURCE_FORWARD.json';DESIGN_SHA='7782c8e994a1c370d293ad483e446dbed4f315d3e883aba386052ad374bd51cc'
NOLD=31680;N=33984
NAMES=('old_LOCO_BEFORE_D','old_PALLOX_BEFORE_D','old_SHAM_D','old_BEFORE_E','old_COVER_E','new_BEFORE_D','new_SHAM_D','new_COVER_E')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def arrays(p):
 with np.load(p,allow_pickle=False)as f:return {k:np.array(f[k],copy=True)for k in f.files}
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def close(a,b,tol=1e-12):
 assert a is not None and b is not None and abs(float(a)-float(b))<=tol,(a,b)
def canonical(source,raw):
 m={('LOCO','TRAIN'):'TRAIN',('LOCO','VALIDATION'):'VALIDATION',('PALLOX','TRAIN_LAB'):'TRAIN',('PALLOX','VAL_HALL'):'VALIDATION'}
 assert(source,raw)in m;return m[source,raw]
def cm(pred,y,mask):return [[int((mask&(y==c)&(pred==j)).sum())for j in(0,1,-1)]for c in(0,1)]
def pair_result(pred,pp,split,splits):
 q=pp[splits[pp[:,0]]==split];good=(pred[q]==[0,1,0]).all(1)
 return dict(N=len(q),correct=int(good.sum()),rate=float(good.mean())if len(q)else None)
def selector(z,a,y,masks):
 safe=np.where(a,z,0).astype(np.float64);loss=np.logaddexp(0,safe)-np.where(y>=0,y,0)*safe
 vals={k:dict(N=int(q.sum()),BCE=float(loss[q].mean()),unavailable=int((q&~a).sum()))for k,q in masks.items()}
 return float(np.mean([x['BCE']for x in vals.values()])),vals,loss

def source_meta(bindings):
 om=read(check(bindings['old_manifest']));nm=read(check(bindings['new_manifest']));ol=rows(check(om['input_lineage']));nl=rows(check(nm['input_lineage']));line=ol+nl
 og=rows(check(bindings['old_GT']));ng=rows(check(bindings['new_GT']));q=rows(check(bindings['new_queries']));cfg=rows(check(bindings['new_config']))
 assert len(ol)==len(og)==NOLD and len(nl)==2304 and len(ng)==len(q)==len(cfg)==96
 raw=[];oldpairs=[]
 for i,(r,g)in enumerate(zip(ol,og)):
  assert r['occurrence_index']==g['occurrence_index']==i and r['occurrence_id']==g['occurrence_id']
  assert r['slot_index']==g['base_row_index']==g['slot_index']==i//33 and r['variant']==g['variant']and r['donor_rank']==g['donor_rank']
  label=g['candidate_aux_hidden_label_GT_only'];assert label in(-1,0,1);raw.append({0:'D',1:'E',-1:'U'}[label])
  if i%33 and i%33%2 and g['pair_ready']:
   assert label==1 and og[i+1]['pair_ready']and og[i+1]['candidate_aux_hidden_label_GT_only']==0
   oldpairs.append([i//33*33,i,i+1])
 newpairs=[]
 for bi,g in enumerate(ng):
  assert g['base_index']==q[bi]['base_index']==cfg[bi]['base_index']==bi and g['base_id']==q[bi]['base_id']==cfg[bi]['base_id']
  for si,stage in enumerate(('BEFORE','COVER','SHAM')):
   for k,x in enumerate(g['stages'][stage]):
    j=bi*24+si*8+k;r=nl[j];assert r['occurrence_index']==j and r['variant']==stage and r['aux_corner_index']==k and r['base_index']==bi
    assert r['query_xy']==q[bi]['query8_xy'][k]and r['L']==q[bi]['L'][k]and r['split']==g['split']==cfg[bi]['split']
    assert x['state']in('D','E','S','U');raw.append(x['state'])
  k=g['selected_aux_index'];pp=np.array([bi*24+k,bi*24+8+k,bi*24+16+k])+NOLD;newpairs.append(pp)
 raw=np.array(raw);y=np.where(raw=='D',0,np.where(raw=='E',1,-1)).astype(np.int8)
 assert [int((raw==x).sum())for x in('D','E','S','U')]==[1215,303,396,32070]
 split=np.array([canonical(r['source'],r['split'])for r in ol]+[r['split']for r in nl]);source=np.array([r['source']for r in ol]+['SYNTH_EXPECTED_CUE']*2304);variant=np.array([r['variant']for r in line]);domain=np.array(['V96_REAL']*NOLD+['V99_SYNTH']*2304);a=np.array([r['input_facts']['operator_query_available']for r in line],bool)
 pair_old=np.array(oldpairs,np.int64);pair_new=np.array(newpairs,np.int64);assert pair_old.shape==(194,3)and pair_new.shape==(96,3)
 assert(y[pair_old]==[0,1,0]).all()and(y[pair_new]==[0,1,0]).all()
 meta=dict(line=line,raw=raw,y=y,split=split,source=source,variant=variant,domain=domain,a=a,old_pairs=pair_old,new_pairs=pair_new)
 return meta

def cohort_masks(m,split):
 val=m['split']==split;old=m['domain']=='V96_REAL';new=~old;y=m['y'];v=m['variant'];src=m['source']
 return dict(old_LOCO_BEFORE_D=val&old&(src=='LOCO')&(v=='BEFORE')&(y==0),old_PALLOX_BEFORE_D=val&old&(src=='PALLOX')&(v=='BEFORE')&(y==0),old_SHAM_D=val&old&(v=='CUE_PRESERVING_SHAM')&(y==0),old_BEFORE_E=val&old&(v=='BEFORE')&(y==1),old_COVER_E=val&old&(v=='OPAQUE_COVER')&(y==1),new_BEFORE_D=val&new&(v=='BEFORE')&(y==0),new_SHAM_D=val&new&(v=='SHAM')&(y==0),new_COVER_E=val&new&(v=='COVER')&(y==1))
def pools(m):
 t=cohort_masks(m,'TRAIN');p={'old_triplets':m['old_pairs'][m['split'][m['old_pairs'][:,0]]=='TRAIN'],'new_triplets':m['new_pairs'][m['split'][m['new_pairs'][:,0]]=='TRAIN']}
 for k in('old_LOCO_BEFORE_D','old_PALLOX_BEFORE_D','old_BEFORE_E'):p[k]=np.flatnonzero(t[k])
 for state in('D','E'):p['new_ALL_'+state]=np.flatnonzero((m['domain']=='V99_SYNTH')&(m['split']=='TRAIN')&(m['raw']==state))
 assert all(len(x)>0 for x in p.values());return p

def check_sampling(records,m,p,expected_hash):
 assert len(records)==2000;g=torch.Generator(device='cpu').manual_seed(205);digest=hashlib.sha256();ixs=[]
 for update,r in enumerate(records,1):
  assert r['update']==update
  oi=torch.randint(len(p['old_triplets']),(4,),generator=g).numpy();ni=torch.randint(len(p['new_triplets']),(4,),generator=g).numpy();triples=np.concatenate([p['old_triplets'][oi],p['new_triplets'][ni]])
  pieces=[triples.reshape(-1)]
  for name,n in(('old_LOCO_BEFORE_D',4),('old_PALLOX_BEFORE_D',4),('old_BEFORE_E',8),('new_ALL_D',8),('new_ALL_E',16)):pieces.append(p[name][torch.randint(len(p[name]),(n,),generator=g).numpy()])
  un=np.concatenate(pieces);perm=torch.randperm(64,generator=g).numpy();inv=np.argsort(perm);ix=un[perm];pos=inv[np.arange(24).reshape(8,3)]
  assert np.array_equal(ix,r['occurrence_indices'])and np.array_equal(pos,r['matched_triplet_batch_positions'])
  assert r['draw']['old_pair_pool_indices']==oi.tolist()and r['draw']['new_pair_pool_indices']==ni.tolist()and r['draw']['shuffle_permutation']==perm.tolist()and r['draw']['unshuffled_occurrence_indices']==un.tolist()
  assert np.all(m['split'][ix]=='TRAIN')and np.all(m['y'][ix]>=0)and np.sum(m['y'][ix]==0)==np.sum(m['y'][ix]==1)==32
  digest.update(ix.astype('<i8').tobytes());digest.update(pos.astype('<i8').tobytes());ixs.append(ix)
 assert digest.hexdigest()==expected_hash
 return np.array(ixs),digest.hexdigest()

def gates(pred,m,design):
 y=m['y'];val=m['split']=='VALIDATION';old=m['domain']=='V96_REAL';co=cohort_masks(m,'VALIDATION');G={}
 spec=[('old_LOCO_before_D_recall_min','old_LOCO_BEFORE_D',0),('old_PALLOX_before_D_recall_min','old_PALLOX_BEFORE_D',0),('old_natural_before_E_recall_min','old_BEFORE_E',1),('old_sham_D_recall_min','old_SHAM_D',0),('old_cover_E_recall_min','old_COVER_E',1),('new_BEFORE_D_recall_min','new_BEFORE_D',0),('new_SHAM_D_recall_min','new_SHAM_D',0),('new_COVER_E_recall_min','new_COVER_E',1)]
 for name,c,target in spec:G[name]=float((pred[co[c]]==target).mean())
 G['old_accepted_VAL_pair_all3_min']=pair_result(pred,m['old_pairs'],'VALIDATION',m['split'])['rate'];G['new_selected_VAL_pair_all3_min']=pair_result(pred,m['new_pairs'],'VALIDATION',m['split'])['rate']
 for group,mask in[('old',old),('new',~old)]:G[group+'_known_VAL_U_rate_max']=float((pred[val&mask&(y>=0)]<0).mean())
 assert len(G)==12
 return {k:dict(value=v,limit=design['source_gates'][k],PASS=v<=design['source_gates'][k]if k.endswith('_max')else v>=design['source_gates'][k])for k,v in G.items()}

def audit_arm(arm,complete_sha):
 run=V/'runs'/arm;complete=run/'FIT_COMPLETE.json';assert sha(complete)==complete_sha
 fc=read(complete);assert fc['completed']and fc['PASS_execution']and fc['optimizer_updates']==2000 and fc['arm']==arm
 freeze=read(check(fc['MODEL_FREEZE']));protocol=read(check(freeze['protocol']));design=read(DESIGN);assert sha(DESIGN)==DESIGN_SHA and protocol['design']==bind(DESIGN)
 entry=read(check(freeze['entry']));assert entry['arm']==arm and entry['approved_actual_source_fit']and not entry['current_allowed']and not entry['PnP_allowed']
 m=source_meta(protocol['bindings']);s=arrays(check(fc['scores']));assert len(s['query_logit'])==N
 assert np.array_equal(s['global_occurrence_index'],np.arange(N))and np.array_equal(s['raw_teacher_state'],m['raw'])and np.array_equal(s['external_loss_candidate'],m['y']>=0)and np.array_equal(s['external_target_D0_E1_no_target_minus1'],m['y'])
 assert np.array_equal(s['known_geometric_S_out_of_task'],m['raw']=='S')and np.array_equal(s['unresolved_teacher_U'],m['raw']=='U')and np.array_equal(s['query_available'],m['a'])and np.array_equal(s['split'],m['split'])
 for i,r in enumerate(m['line']):
  assert s['occurrence_id'][i]==r.get('occurrence_id',r.get('image_id','')+'::AUX'+str(r.get('aux_corner_index',-1)))and s['original_split'][i]==r['split']and s['variant'][i]==r['variant']
  q=np.array([np.nan,np.nan]if r['query_xy']is None else r['query_xy']);L=np.nan if r['L']is None else r['L'];assert np.array_equal(s['query_xy_raw'][i],q,equal_nan=True)and(s['L_raw'][i]==L or(np.isnan(s['L_raw'][i])and np.isnan(L)))
  assert s['input_unit_index'][i]==(r['input_unit_index']if i<NOLD else1348+r['input_unit_index'])
 ref=arrays(check(protocol['bindings']['C5_scores_reference_base_only']));oldref=arrays(check(protocol['bindings']['old_C1_scores']))
 assert np.array_equal(s['reference_base_logit'],ref['base_logit'])and np.array_equal(s['reference_base_logit'][:NOLD],oldref['query_logit'])
 expected=s['reference_base_logit']if arm==ARMS[0]else np.clip(s['reference_base_logit'],-2,2)
 assert np.array_equal(s['base_logit'].view(np.uint32),expected.view(np.uint32))and np.array_equal(s['query_logit'],s['base_logit']+s['delta_logit'])
 pred=np.where(m['a'],(s['query_logit']>=0).astype(np.int8),-1);assert np.array_equal(pred,s['prediction'])
 history=rows(check(fc['history']));assert len(history)==21 and[r['update']for r in history]==list(range(0,2001,100));best=float('inf');step=None
 for h in history:
  parts=h['VAL_cohorts'];assert tuple(parts)==NAMES
  own=float(np.mean([parts[k]['mean_BCE']for k in NAMES]));close(own,h['criterion_eight_VAL_cohorts_BCE'])
  assert all(parts[k]['count']==int(cohort_masks(m,'VALIDATION')[k].sum())for k in NAMES)
  if own<best:best=own;step=h['update']
 assert step==freeze['selected_update']==fc['selected_update'];value,parts,loss=selector(s['query_logit'],m['a'],m['y'],cohort_masks(m,'VALIDATION'));close(value,best)
 for k,v in parts.items():close(v['BCE'],history[step//100]['VAL_cohorts'][k]['mean_BCE'])
 assert np.all(np.isnan(s['NLL_external_known_FP64'][m['y']<0]));assert np.allclose(s['NLL_external_known_FP64'][m['y']>=0],loss[m['y']>=0],rtol=0,atol=1e-12)
 gg=gates(pred,m,design);savedg=read(check(fc['source_gates']));assert savedg['all_gates_pass']==all(x['PASS']for x in gg.values())
 for k,v in gg.items():close(v['value'],savedg['gates'][k]['value']);assert v['PASS']==savedg['gates'][k]['PASS']and v['limit']==savedg['gates'][k]['limit']
 draws,digest=check_sampling(rows(check(fc['train_updates'])),m,pools(m),protocol['frozen_2000_draws_sha256'])
 initial_receipt=read(check(fc['update0_parity']));assert initial_receipt['PASS']and initial_receipt['all_delta_zero']and initial_receipt['anchor_clamp_or_raw_exact']
 initial_file=run/'INITIAL_ALL33984_LOGIT_COMPONENTS.npz';initial_saved=False
 if initial_file.exists():
  ini=arrays(initial_file);assert np.array_equal(ini['reference_base_logit'],s['reference_base_logit'])and np.array_equal(ini['base_logit'].view(np.uint32),expected.view(np.uint32))and np.array_equal(ini['query_logit'],expected)and np.count_nonzero(ini['delta_logit'])==0 and np.array_equal(ini['query_available'],m['a']);initial_saved=True
 metrics={}
 for dom in('V96_REAL','V99_SYNTH'):
  metrics[dom]={}
  for split in('TRAIN','VALIDATION'):
   mask=(m['domain']==dom)&(m['split']==split);metrics[dom][split]=dict(matrix=cm(pred,m['y'],mask),S_out_of_task=int((mask&(m['raw']=='S')).sum()),teacher_U=int((mask&(m['raw']=='U')).sum()),pairs=pair_result(pred,m['old_pairs']if dom=='V96_REAL'else m['new_pairs'],split,m['split']))
 return dict(arm=arm,FIT_COMPLETE=bind(complete),MODEL_FREEZE=fc['MODEL_FREEZE'],scores=fc['scores'],selected_update=step,selected_source_VAL_BCE=value,gates=gg,all12gates_pass=all(x['PASS']for x in gg.values()),metrics=metrics,source_counts={k:int((m['raw']==k).sum())for k in('D','E','S','U')},same33984_qL_folds_rawlabels_factualU_verified=True,originalC1_reference_exact=True,selected_anchor_raw_or_clip_bitexact=True,initial_components_independently_saved_replayed=initial_saved,initial_residual_state_sha256=initial_receipt['initial_residual_state_sha256'],initial_base_state_sha256=initial_receipt['base_state_sha256'],sample2000_digest=digest,sampled_rows=128000,checkpoint_bytes_loaded_or_hashed=False),s,draws

def main(c6sha,c7sha):
 policy=read(H/'AUDIT_SCOPE_BEFORE_OUTPUTS.json');assert policy['code']==bind(__file__)and sha(DESIGN)==DESIGN_SHA
 out=[];data=[];draw=[]
 for arm,sh in zip(ARMS,[c6sha,c7sha]):
  r,s,ix=audit_arm(arm,sh);out.append(r);data.append(s);draw.append(ix)
 for k in('raw_teacher_state','query_xy_raw','L_raw','split','original_split','domain','source','variant','query_available','global_occurrence_index','domain_occurrence_index','occurrence_id','input_unit_index','reference_base_logit'):
  if data[0][k].dtype.kind=='f':assert np.array_equal(data[0][k],data[1][k],equal_nan=True),k
  else:assert np.array_equal(data[0][k],data[1][k]),k
 assert np.array_equal(draw[0],draw[1])and out[0]['initial_residual_state_sha256']==out[1]['initial_residual_state_sha256']and out[0]['initial_base_state_sha256']==out[1]['initial_base_state_sha256']
 eligible=[r for r in out if r['all12gates_pass']];winner=min(eligible,key=lambda r:(r['selected_source_VAL_BCE'],ARMS.index(r['arm'])))['arm']if eligible else None
 result=dict(completed=True,PASS=True,scope=bind(H/'AUDIT_SCOPE_BEFORE_OUTPUTS.json'),code=bind(__file__),arms=out,all33984_botharms_same_teacher_q_L_folds_availability=True,S396_distinct_from_U32070_and_DE1518=True,same128000sampled_occurrences_and_initial_branch_fingerprint=True,between_arm_prospective_choice=dict(eligible_arms=[r['arm']for r in eligible],source_selected_arm=winner,rule='Only12gatePASSeligible; lowestselected8cohortVALBCE; exacttieC6'),current_inference_authorized_by_this_review=False,limits=['Saved hashes/fingerprints are producer certificates; actual checkpoint weights are not loaded or hashed.','History21 earliest-min and8componentmeans replayed; only selected fullscores are available for independent score-to-BCE verification, not all intermediate model predictions.','Source-development proxy labels/nativeUNKNOWN/humanfalse, and queryemulation/clusteredoldcontrols remain unchanged.','No model/head/backbone forward, optimizer/fit, private/current/PnP.'])
 write(H/'REVIEW.json',result);write(H/'SEAL.json',dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),scope=bind(H/'AUDIT_SCOPE_BEFORE_OUTPUTS.json'),code=bind(__file__),new_forward_fit_private_current_PnP=0));print(json.dumps(dict(review=bind(H/'REVIEW.json'),winner=winner,arms=[dict(arm=r['arm'],selected=r['selected_update'],BCE=r['selected_source_VAL_BCE'],gates=sum(x['PASS']for x in r['gates'].values()))for r in out])))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--c6-complete-sha',required=True);ap.add_argument('--c7-complete-sha',required=True);a=ap.parse_args();main(a.c6_complete_sha,a.c7_complete_sha)
