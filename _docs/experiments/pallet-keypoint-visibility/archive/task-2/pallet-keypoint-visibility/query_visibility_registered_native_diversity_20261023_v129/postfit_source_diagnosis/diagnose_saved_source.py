"""Independent saved-only C20/C21 arithmetic, sampling and feature diagnosis.
No model/checkpoint construction, learned forward, optimizer, image or private I/O.
"""
from pathlib import Path
import hashlib,json,collections,sys,time
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
RUN=V/'runs/C20_C21_FIXED1000_SOURCE_CONTROL'
ARMS=('C20_PUBLIC_REPLAY_CONTROL','C21_PUBLIC_PLUS_NATIVE')

def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def checked(b):assert bind(b['path'])==b;return Path(b['path'])
def pack(p):
 with np.load(p,allow_pickle=False)as z:return{k:np.array(z[k],copy=True)for k in z.files}
def clean(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return{k:clean(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return[clean(v)for v in x]
 return x
def save(p,o):
 with Path(p).open('x')as f:json.dump(clean(o),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cm(y,p,m):return[[int((m&(y==a)&(p==b)).sum())for b in(0,1,-1)]for a in(0,1)]
def bce(z,y):return np.logaddexp(0.,z.astype(np.float64))-y*z.astype(np.float64)
def recall(c,label):return c[label][label]/sum(c[label])if sum(c[label])else None
def metric(y,pr,state,m):
 c=cm(y,pr,m);div=lambda a,b:a/b if b else None
 return dict(all_rows=int(m.sum()),raw_states={s:int((m&(state==s)).sum())for s in'DESU'},CM_D_E_rows_D_E_U_columns=c,
 D_recall=recall(c,0),E_recall=recall(c,1),D_precision=div(c[0][0],c[0][0]+c[1][0]),E_precision=div(c[1][1],c[0][1]+c[1][1]),
 known_input_U=c[0][2]+c[1][2],teacher_S_hidden_response=int((m&(state=='S')&(pr==1)).sum()),teacher_S_available=int((m&(state=='S')&(pr>=0)).sum()),teacher_U_available=int((m&(state=='U')&(pr>=0)).sum()))
def loss_stats(trace,key):
 a=np.array([t[key]for t in trace],np.float64)
 return dict(first=float(a[0]),last=float(a[-1]),first100_mean=float(a[:100].mean()),last100_mean=float(a[-100:].mean()),last20_mean=float(a[-20:].mean()),
 rolling100_means=[float(a[i:i+100].mean())for i in range(0,1000,100)],min=float(a.min()),max=float(a.max()))
def group_stats(s,m,baseline,exposure=None):
 y=s['external_target'];z=s['query_logit'];a=s['query_available'];pr=s['prediction'];known=m&(y>=0);avail=known&a
 c=cm(y,pr,m);out=dict(rows=int(m.sum()),known=int(known.sum()),available_known=int(avail.sum()),CM=c,D_recall=recall(c,0),E_recall=recall(c,1),
 BCE_mean=float(bce(z[known],y[known]).mean())if known.any()else None,initial_C16_BCE=float(bce(baseline[known],y[known]).mean())if known.any()else None)
 if exposure is not None:
  out['draw_exposure']=dict(total=int(exposure[avail].sum()),seen=int((avail&(exposure>0)).sum()),unseen=int((avail&(exposure==0)).sum()),
   mean=float(exposure[avail].mean())if avail.any()else None,min=int(exposure[avail].min())if avail.any()else None,max=int(exposure[avail].max())if avail.any()else None)
 return out

def main():
 started=time.monotonic();control=read(RUN/'CONTROL_COMPLETE.json');assert control['completed']and control['PASS']
 entry=read(checked(control['entry']));protocol=read(checked(entry['protocol']));initial_baseline_old=pack(checked(protocol['old_baseline_scores']));initial_baseline_new=pack(checked(entry['new_baseline_scores']))
 baseline=np.r_[initial_baseline_old['query_logit'],initial_baseline_new['query_logit']]
 a0=np.r_[initial_baseline_old['query_available'],initial_baseline_new['query_available']]
 plan=pack(RUN/'SAMPLER_PLAN_BEFORE_HEAD.npz');pubplan=plan['public_occurrence_indices'];natplan=plan['native_occurrence_indices']
 assert pubplan.shape==natplan.shape==(1000,64)
 score={};traces={};bindings={};CMs={};gates_replayed={};group_details={};rolling={};exposure=np.bincount(pubplan.ravel(),minlength=61110)
 native_exposure=np.bincount(natplan.ravel(),minlength=61110)
 for arm,fb in zip(ARMS,control['outputs']):
  done=read(checked(fb));assert done['completed']and done['PASS']and done['optimizer_updates']==1000 and done['fixed_final_not_VAL_selection']
  freeze=read(checked(done['model_freeze']));assert freeze['selected_update']==1000 and freeze['fixed_final_not_VAL_selection']and freeze['threshold_logit']==0 and not freeze['current_used_for_fit_or_selection']
  parity=read(RUN/arm/'INITIAL_C16_PARITY.json');assert all(parity[k]for k in('completed','old4428_units_bitexact','new56214_raw_rows_bitexact','all61110_safe_logits_bitexact'))
  s=pack(checked(done['scores']));t=lines(RUN/arm/'TRAIN_UPDATES.jsonl');assert len(t)==1000 and[t0['update']for t0 in t]==list(range(1,1001))
  assert np.array_equal(s['global_index'],np.arange(61110))and np.array_equal(s['query_available'],s['factual_query_available']&s['context_available'])and np.array_equal(s['query_available'],a0)
  assert np.array_equal(s['external_target'],np.where(s['raw_state']=='D',0,np.where(s['raw_state']=='E',1,-1)))
  expected=np.where(s['query_available'],(s['query_logit']>=0).astype(np.int8),-1)
  assert np.array_equal(s['prediction'],expected)and not s['query_logit'][~s['query_available']].any()
  assert np.array_equal(s['query_logit'][s['query_available']],s['raw_head_logit_before_U'][s['query_available']])
  for i,row in enumerate(t):
   assert np.array_equal(row['public_occurrence_indices'],pubplan[i])and row['known_available_TRAIN_only']
   assert np.array_equal(row['native_occurrence_indices'],natplan[i]if arm==ARMS[1]else np.array([],np.int64))
   assert np.isfinite([row[k]for k in('public_loss','native_loss','total_loss','gradient_norm_before_clip')]).all()
   assert np.float32(row['public_loss'])+np.float32(row['native_loss'])==np.float32(row['total_loss'])and row['old_proxy_loss']==0
  score[arm]=s;traces[arm]=t;bindings[arm]=dict(fit_complete=fb,scores=done['scores'],freeze=done['model_freeze'],trace=bind(RUN/arm/'TRAIN_UPDATES.jsonl'),parity=bind(RUN/arm/'INITIAL_C16_PARITY.json'))
  rolling[arm]={k:loss_stats(t,k)for k in('public_loss','native_loss','total_loss')}
  gn=np.array([q['gradient_norm_before_clip']for q in t]);rolling[arm]['gradient_norm']=dict(first=float(gn[0]),last=float(gn[-1]),median=float(np.median(gn)),steps_over_clip5=int((gn>5).sum()),last100_over_clip5=int((gn[-100:]>5).sum()))
 assert read(RUN/ARMS[0]/'INITIAL_C16_PARITY.json')['initial_state_sha256']==read(RUN/ARMS[1]/'INITIAL_C16_PARITY.json')['initial_state_sha256']
 s=score[ARMS[0]];y=s['external_target'];state=s['raw_state'];split=s['split'];corpus=s['corpus'];avail=s['query_available'];N=len(y)
 assert N==61110 and np.array_equal(corpus[:4896],np.full(4896,'OLD_PUBLIC'))
 for arm in ARMS[1:]:
  for k in('external_target','raw_state','split','corpus','source_occurrence_index','global_index','occurrence_id','query_xy_raw','L_raw','feature_unit_index'):assert np.array_equal(s[k],score[arm][k]),k
 # Compare labels against the original bound GT packets, independent of scorepack labels.
 oldgt=lines(checked(protocol['old_GT']));nativegt=read(checked(entry['native_GT']))['rows'];publicgt=read(checked(entry['public_GT']))['rows']
 by={'NATIVE':nativegt,'PUBLIC_AUX':publicgt};assert len(oldgt)==4896 and len(nativegt)==55296 and len(publicgt)==918
 gt=oldgt+[by[str(tag)][int(i)]for tag,i in zip(corpus[4896:],s['source_occurrence_index'][4896:])]
 assert all((g.get('id')or g.get('occurrence_id'))==s['occurrence_id'][i]for i,g in enumerate(gt))
 yy=np.array([g['candidate_aux_hidden_label_GT_only']if i<4896 else g['external_target_D0_E1_else_minus1']for i,g in enumerate(gt)])
 assert np.array_equal(y,yy)and all(g['raw_state']==state[i]for i,g in enumerate(gt)if i>=4896)
 assert np.isin(s['source_occurrence_index'][corpus=='NATIVE'],np.arange(55296)).all()
 public=(split=='TRAIN')&avail&(y>=0)&np.isin(corpus,['OLD_PUBLIC','PUBLIC_AUX']);nd=(split=='TRAIN')&avail&(corpus=='NATIVE')&(y==0);ne=(split=='TRAIN')&avail&(corpus=='NATIVE')&(y==1)
 assert public[pubplan].all()and(nd|ne)[natplan].all()and all(np.bincount(y[row],minlength=2).tolist()==[32,32]for row in natplan)
 assert int(public.sum())==1407 and int(nd.sum())==17154 and int(ne.sum())==1206
 # CPU generators only: no model, CUDA tensor, parameter load or optimizer.
 import torch
 pg=torch.Generator().manual_seed(1301);ng=torch.Generator().manual_seed(1302);pp=np.flatnonzero(public);dp=np.flatnonzero(nd);ep=np.flatnonzero(ne)
 for i in range(1000):
  assert np.array_equal(pp[torch.randint(len(pp),(64,),generator=pg).numpy()],pubplan[i])
  expected=np.r_[dp[torch.randint(len(dp),(32,),generator=ng).numpy()],ep[torch.randint(len(ep),(32,),generator=ng).numpy()]]
  assert np.array_equal(expected,natplan[i])
 coverage={}
 for tag,m,counts in [('old_public',public&(corpus=='OLD_PUBLIC'),exposure),('new_public',public&(corpus=='PUBLIC_AUX'),exposure),('native_D',nd,native_exposure),('native_E',ne,native_exposure)]:
  xx=counts[m];seen=m&(counts>0);unseen=m&(counts==0);pr21=score[ARMS[1]]['prediction']
  coverage[tag]=dict(pool=int(m.sum()),draws=int(xx.sum()),mean=float(xx.mean()),unique_seen=int((xx>0).sum()),never_seen=int((xx==0).sum()),quantiles_0_25_50_75_90_95_100=np.quantile(xx,[0,.25,.5,.75,.9,.95,1]).tolist(),C21_seen_CM=cm(y,pr21,seen),C21_unseen_CM=cm(y,pr21,unseen),C16_seen_CM=cm(y,np.where(avail,(baseline>=0).astype(np.int8),-1),seen),C16_unseen_CM=cm(y,np.where(avail,(baseline>=0).astype(np.int8),-1),unseen))
 save(H/'DRAW_COVERAGE.json',dict(completed=True,PASS=True,sampler=bind(RUN/'SAMPLER_PLAN_BEFORE_HEAD.npz'),coverage=coverage,actual_all1000plans_matchbothtraces=True,independent_CPU_RNG_exact=True))
 print(json.dumps(dict(sampler_and_labels_replay=True,rows=N,coverage=coverage)),flush=True)
 # Independent CM/precision/recall/U replay; saved group metrics include S/U separately.
 variants=np.array([g.get('variant',('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')[(int(s['source_occurrence_index'][i])//72)%4] if corpus[i]=='NATIVE' else 'UNRESOLVED')for i,g in enumerate(gt)]);oldalpha=np.array([-1 if g.get('nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only')is None else int(g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for g in gt]);ready=np.array([g.get('geometry_pair_ready',False)for g in gt])
 for arm in ARMS:
  ss=score[arm];pr=ss['prediction'];saved=read(RUN/arm/'SOURCE_RESULTS.json');primary={};details={}
  for c in('OLD_PUBLIC','PUBLIC_AUX','NATIVE'):
   for f in('TRAIN','VALIDATION'):
    mask=(corpus==c)&(split==f);key=c+'::'+f;root=saved['groups'][key]
    assert metric(y,pr,state,mask)==root['all_planned']and metric(y,pr,state,mask&avail)==root['input_available']
    for variant in np.unique(variants[mask]):assert metric(y,pr,state,mask&(variants==variant))==root['by_variant'][variant]
    primary[key]=dict(CM=cm(y,pr,mask),D_recall=recall(cm(y,pr,mask),0),E_recall=recall(cm(y,pr,mask),1))
    ex=exposure+(native_exposure if arm==ARMS[1]else 0);details[key]=group_stats(ss,mask,baseline,ex if f=='TRAIN'else None)
  CMs[arm]=primary;group_details[arm]=details
  savedgate=read(RUN/arm/'SOURCE_GATE_RESULTS.json');gg={};rec=[];old=(corpus=='OLD_PUBLIC')&(split=='VALIDATION');basepred=np.where(avail,(baseline>=0).astype(np.int8),-1)
  for label,tag in((0,'D'),(1,'E')):
   m=old&(y==label);val=float((pr[m]==label).mean());ref=float((basepred[m]==label).mean());gg['old_public_'+tag]=dict(value=val,baseline=ref,threshold=ref-.01,denominator=int(m.sum()),PASS=val>=ref-.01);rec.append(val)
  for name,label,m in [('old_Eout_plus1',1,old&(y==1)&(oldalpha==0)&(variants=='OPAQUE_COVER')&ready),('old_SHAMin_plus1',0,old&(y==0)&(oldalpha==1)&(variants=='CUE_PRESERVING_SHAM')&ready)]:
   correct=int((pr[m]==label).sum());ref=int((basepred[m]==label).sum());gg[name]=dict(correct=correct,baseline_correct=ref,minimum_correct=ref+1,denominator=int(m.sum()),PASS=correct>=ref+1)
  for c in('PUBLIC_AUX','NATIVE'):
   for label,tag in((0,'D'),(1,'E')):
    m=(corpus==c)&(split=='VALIDATION')&(y==label);val=float((pr[m]==label).mean());gg[c+'_VAL_'+tag]=dict(value=val,threshold=.8,denominator=int(m.sum()),PASS=val>=.8);rec.append(val)
  assert gg==savedgate['gates']and sum(g['PASS']for g in gg.values())==savedgate['passed_gates']and min(rec)==savedgate['worst_six_class_recall']
  gates_replayed[arm]=dict(gates=gg,passed=sum(g['PASS']for g in gg.values()),all_pass=all(g['PASS']for g in gg.values()))
 assert not any(g['all_pass']for g in gates_replayed.values())and control['eligible_source_arm']is None
 # Loss on the complete TRAIN objective is more comparable than two random mini-batches.
 objective={};public_cases={};native_pixels={};sample_exposure={}
 for name,zz in [('FROZEN_C16',baseline)]+[(a,score[a]['query_logit'])for a in ARMS]:
  objective[name]=dict(public1407_BCE=float(bce(zz[public],y[public]).mean()),native_D_BCE=float(bce(zz[nd],y[nd]).mean()),native_E_BCE=float(bce(zz[ne],y[ne]).mean()))
  objective[name]['native_class_balanced_BCE']=(objective[name]['native_D_BCE']+objective[name]['native_E_BCE'])/2
  objective[name]['C21_sum_two_domain_means']=objective[name]['public1407_BCE']+objective[name]['native_class_balanced_BCE']
 for arm in ARMS:
  ss=score[arm];public_cases[arm]={};native_pixels[arm]={}
  for case in(4,10,12,16,17,23):
   m=(corpus=='PUBLIC_AUX')&np.array([g.get('selection_index')==case for g in gt]);public_cases[arm][str(case)]=group_stats(ss,m,baseline,exposure)
  pix=np.array([g.get('q_nearest_pixel_owner_diagnostic',{}).get('category','NONE')for g in gt])
  for lab in(0,1):
   for category in np.unique(pix[(corpus=='NATIVE')&(split=='TRAIN')&(y==lab)]):
    m=(corpus=='NATIVE')&(split=='TRAIN')&(y==lab)&(pix==category);native_pixels[arm][('D'if lab==0 else'E')+'::'+category]=group_stats(ss,m,baseline)
  sample_exposure[arm]={}
  ex=exposure+(native_exposure if arm==ARMS[1]else 0)
  for category,m in [('public',public),('native_D',nd),('native_E',ne)]:
   table={}
   for low,high,label in [(0,0,'unseen'),(1,1,'once'),(2,4,'2to4'),(5,9,'5to9'),(10,100000,'10plus')]:
    mm=m&(ex>=low)&(ex<=high);table[label]=group_stats(ss,mm,baseline)
   sample_exposure[arm][category]=table
 save(H/'ARITHMETIC_AND_OPTIMIZATION.json',dict(completed=True,PASS=True,code=bind(__file__),control=bind(RUN/'CONTROL_COMPLETE.json'),bindings=bindings,
 sampler=bind(RUN/'SAMPLER_PLAN_BEFORE_HEAD.npz'),checks=dict(actual_1000_each=True,same_public_draws_all1000=True,CPU_RNG1301_1302_exact=True,TRAIN_known_available_only=True,no_S_U_VAL_unavailable_loss=True,source_GT_score_joins_exact=True,primary_CM_precision_recall_U_variants_exact=True,eight_gates_exact=True,neither_arm_admitted=True,initial_C16_parity_same_state=True),
 primary_CM=CMs,gates=gates_replayed,loss_and_gradient_trace=rolling,full_TRAIN_objective=objective,group_details=group_details,
 new_public_by_case=public_cases,native_TRAIN_by_qpixel_category=native_pixels,TRAIN_draw_exposure_vs_scores=sample_exposure,sampling_coverage=coverage,
 allowed_saved_only_reads=True,model_weight_image_private_V112_reads_or_calls=0,seconds=time.monotonic()-started))
 print(json.dumps(dict(complete_arithmetic=bind(H/'ARITHMETIC_AND_OPTIMIZATION.json'),objective=objective,loss_trace=rolling,primary_CM=CMs)),flush=True)


if __name__=='__main__':main()
