"""Invented functional fixtures only; no SOURCE/checkpoint/image reads."""
import hashlib,json,copy
from pathlib import Path
import numpy as np
import torch
import audit_saved_pair as A
RESULT=[]
def ok(name):RESULT.append(name)
def rejects(f):
 try:f()
 except (ValueError,AssertionError):return
 raise AssertionError('expected reject')
def meta(n=4):return [dict(row_id=str(i),component_role='TRAIN',component_id='a'*64,nominal_fit_eligible=True,lineage_bound=True,original_query_available=True)for i in range(n)]
def logs_fixture(steps=3):
 rr={d:meta()for d in A.DOMAINS};point=[0,0,1,1];weak=[dict(target=t,fit=True)for t in point];logs=[]
 pp=[np.array([0,1]),np.array([2,3])]
 for arm in range(2):
  pr=np.random.default_rng(1811);wr=np.random.default_rng(1812);hh=hashlib.sha256();ww=hashlib.sha256();ll=[]
  for update in range(1,steps+1):
   draw={k:[0]*n for k,n in zip(A.TASKS,A.SIZES)};pi=np.concatenate([pr.choice(p,16,replace=True)for p in pp]);wi=np.concatenate([wr.choice(p,8,replace=True)for p in pp])if arm else None
   for k,v in draw.items():hh.update(k.encode());hh.update(np.array(v,dtype='<i8').tobytes())
   hh.update(pi.astype('<i8').tobytes())
   if wi is not None:ww.update(wi.astype('<i8').tobytes())
   weakloss=.3 if arm else 0.;ll.append(dict(update=update,old192=draw,strong32=pi.tolist(),weak16=None if wi is None else wi.tolist(),old5=[1.]*5,strong=.2,weak=weakloss,total=1.2+.25*weakloss,gradient_norm=.8,old_point_sha=hh.hexdigest(),weak_sha=ww.hexdigest(),all_fit_masks_valid=True))
  logs.append(ll)
 return logs,rr,point,weak
x={'path':'/does-not-exist/source.json','sha256':'a'*64,'size_bytes':123};y=dict(path=x['path'],sha256=x['sha256'],kind='ANNOTATION');assert A.same(x,y);rejects(lambda:A.checked(dict(x,sha256='b'*64),[y],('.json',)));ok('binding_identity_ignores_extra_annotations_but_wrongSHA_rejected_before_FS')
rejects(lambda:A.lexical('/tmp/never/V112/metadata.json'));ok('lexical_V112_before_FS')
q8=[[0.,0.],[100.,-100.]]+[[0.,0.]]*6;z=A.context([0.,0.],1.,0,True,q8,[1.]*8,[True,True]+[False]*6);assert z[8]==0 and z[9]==1 and z[18:20].tolist()==[8.,-8.];u=A.context(None,None,7,False,[None]*8,[None]*8,[False]*8);assert u[7]==1 and not u[8:].any();ok('F64_clip_original_native_selfmask_and_inputU')
r=dict(component_role='TRAIN',mapping_status='EXACT_ORIGINAL_IMAGE_OWN_SCENE',family='SELF_SCENE15',state_index=0,state_name='NOMINAL',original_query_available=True,query_xy=[0.,0.],L=96.,saved_projected_reference_xy=[0.,0.],point0_target_alone_state='D',point0_combined_state='D',reference_locality_for_actual_input=dict(raw96_reference_index_xy=[47.5,47.5],inside_crop_canvas_extent=True,inside_original_frame_canvas_extent=True))
assert A.point_target(r)==0 and A.point_target(dict(r,point0_combined_state='E'))==1 and A.point_target(dict(r,point0_combined_state='S'))==-1 and A.point_target(dict(r,original_query_available=False))==-1;ok('saved_point_D_E_S_and_originalU_distinct')
v=A.rle_valid([[0,1],[96,2]]);assert v.sum()==3;rejects(lambda:A.rle_valid([[0,2],[1,2]]));rejects(lambda:A.rle_valid([[9215,2]]));ok('valid_RLE_exact_nonoverlap_bounds')
logs,rr,pt,weak=logs_fixture();out=A.log_audit(logs,rr,pt,weak,set(),3,.25);assert out['log_rows']==6;ok('paired_all_draw_hashes_point1811_weak1812_objective_and_masks')
bad=copy.deepcopy(logs);bad[1][1]['strong32'][0]=1-bad[1][1]['strong32'][0];rejects(lambda:A.log_audit(bad,rr,pt,weak,set(),3,.25));badrr=copy.deepcopy(rr);badrr[A.DOMAINS[1]][0]['component_role']='HOLDOUT';rejects(lambda:A.log_audit(logs,badrr,pt,weak,set(),3,.25));ok('wrong_point_draw_or_old_HOLDOUT_loss_rejected')
p=torch.zeros(2);os={'param_groups':[{'params':[0]}],'state':{0:dict(step=torch.tensor(4000.),exp_avg=torch.zeros(2),exp_avg_sq=torch.zeros(2))}};assert A.optimizer_audit(os,[p],4000);os['state'][0]['step']=torch.tensor(3999.);rejects(lambda:A.optimizer_audit(os,[p],4000));ok('typed_finite_AdamW_all4000_steps_not3999')
assert A.score_agreement([-1.,0.,1.],[-1.+1e-5,0.,1.])<1e-4;rejects(lambda:A.score_agreement([-1e-8],[1e-8]));rejects(lambda:A.score_agreement([1.],[1.001]));ok('replay_fixed1e_4_and_ge0_sign_exact')
h=Path(__file__).parent
with (h/'INVENTED_CHECKS.json').open('x')as f:json.dump(dict(completed=True,PASS=True,checks=RESULT,count=len(RESULT),audit_code_sha256=hashlib.sha256((h/'audit_saved_pair.py').read_bytes()).hexdigest(),SOURCE_or_checkpoint_or_image_reads=0,operational_model_instantiations=0,model_forward_fit_calls=0),f,indent=2);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,count=len(RESULT),checks=RESULT)))
