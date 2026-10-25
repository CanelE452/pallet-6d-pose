"""Pure invented tests; no saved SOURCE file, PNG or trained checkpoint access."""
from pathlib import Path
import hashlib,json,copy,importlib.util,ast
import numpy as np
import torch
H=Path(__file__).parent
spec=importlib.util.spec_from_file_location('qa_invented',H/'audit_saved_split_pair.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
passed=[]
def check(name,f):f();passed.append(name)
def rejects(f):
 try:f()
 except ValueError:return
 raise AssertionError('expected strict rejection')

def test_optimizer():
 named=[('direct_end_head.'+str(i),torch.nn.Parameter(torch.zeros(1)))for i in range(2)]+[('spatial_residual.'+str(i),torch.nn.Parameter(torch.zeros(1)))for i in range(4)]+[('other.'+str(i),torch.nn.Parameter(torch.zeros(1)))for i in range(36)]
 def state(sep):return dict(param_groups=[dict(params=list(range(42)))],state={i:dict(step=torch.tensor(1100.),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros(1))for i in range(42)if sep or i>=6})
 assert a.optimizer_audit(state(False),named,1100,False)['state_parameters']==36
 assert a.optimizer_audit(state(True),named,1100,True)['state_parameters']==42
 rejects(lambda:a.optimizer_audit(state(True),named,1100,False))
 z=state(True);z['state'][0]['step']=torch.tensor(1099.);rejects(lambda:a.optimizer_audit(z,named,1100,True))
check('actual_optimizer_helper_CONTROL36_TREAT42_and_wrongstate_step_reject',test_optimizer)

def test_logs():
 dev={'d'*64};metadata={d:[dict(component_role='TRAIN',nominal_fit_eligible=True,lineage_bound=True,component_id='a'*64,original_query_available=True)for _ in range(2)]for d in a.DOMAINS};point=np.array([0,1]);weak=[dict(fit=True,target=k)for k in(0,1)];parents=['P0','P1','P2'];alllogs=[]
 for arm in range(2):
  p=np.random.default_rng(1811);w=np.random.default_rng(1812);c=np.random.default_rng(1813);hh=hashlib.sha256();ww=hashlib.sha256();ch=hashlib.sha256();logs=[]
  for u in range(1,4):
   draws={t:[i%2 for i in range(n)]for t,n in zip(a.TASKS,a.SIZES)}
   for t,ii in draws.items():hh.update(t.encode());hh.update(np.asarray(ii,dtype='<i8').tobytes())
   pi=np.concatenate([p.choice([k],16,replace=True)for k in(0,1)]);wi=np.concatenate([w.choice([k],8,replace=True)for k in(0,1)]);ci=c.choice(len(parents),4,replace=True);hh.update(pi.astype('<i8').tobytes());ww.update(wi.astype('<i8').tobytes());ch.update(ci.astype('<i8').tobytes())
   logs.append(dict(update=u,old192=draws,strong32=pi.tolist(),naturalweak16=wi.tolist(),CF4parents=ci.tolist(),CF_parent_row_ids=[parents[i]for i in ci],old_point_sha=hh.hexdigest(),naturalweak_sha=ww.hexdigest(),CF_sha=ch.hexdigest(),old5=[1.]*5,strong=2.,naturalweak=3.,CF_weighted_parent_BCE=4.,total=4.75,gradient_norm=2.,all_fit_masks_valid=True))
  alllogs.append(logs)
 assert a.log_check(alllogs,metadata,point,weak,parents,dev,3)['all_log_rows']==6
 bad=copy.deepcopy(alllogs);bad[1][1]['CF_parent_row_ids'][0]='DEV';rejects(lambda:a.log_check(bad,metadata,point,weak,parents,dev,3))
 bad=copy.deepcopy(alllogs);bad[0][0]['total']=8.;rejects(lambda:a.log_check(bad,metadata,point,weak,parents,dev,3))
check('actual_log_helper_both_CF_fourRNG_commonloss_and_parent_loss_negatives',test_logs)

def test_fingerprint():
 rgb=np.zeros((96,96,3),np.uint8);valid=np.ones((96,96),bool);c=np.zeros(32,np.float32);c[3]=1;c[8:16]=1;c[11]=0;c[16:32]=np.arange(16,dtype=np.float32)/8
 want=hashlib.sha256(b'V223_RGB96_VALID_NATIVE_MASK_XY_V1\0'+rgb.tobytes()+valid.tobytes()+np.asarray(3,dtype='<i8').tobytes()+c[8:16].astype('<f4').tobytes()+c[16:].astype('<f4').tobytes()).hexdigest();assert a.input_fingerprint(rgb,valid,c)==want
 rgb[0,0,0]=1;assert a.input_fingerprint(rgb,valid,c)!=want
 bad=c.copy();bad[:8]=0;rejects(lambda:a.input_fingerprint(rgb,valid,bad))
 r=[dict(input_sha256='a'*64,target=t,row_id=str(i),family='X',source_role='SOURCE_FIT')for i,t in enumerate((0,1))];z=a.collision_summary(r);assert z['opposite_label_groups']==1 and z['opposite_label_rows']==2
check('effective_input_byte_fingerprint_and_opposite_label_collision',test_fingerprint)

def test_decoder():
 assert a.score_agreement([1.,-1.],[1.00005,-1.00005])<1e-4
 rejects(lambda:a.score_agreement([-.000001],[.000001]));rejects(lambda:a.score_agreement([1.],[1.001]))
 z=a.metrics([dict(target=0,prediction=0),dict(target=1,prediction=1),dict(target=-1,prediction=-1)]);assert z['CM_truthDE_predDEU']==[[1,0,0],[0,1,0]]and z['UNKNOWN']==1
check('fixed_replay1e4_sign_exact_and_UNKNOWN_separate',test_decoder)

def test_path():
 rejects(lambda:a.lexical('/does/not/exist/v112/no.pt'))
 rejects(lambda:a.checked(dict(path='/does/not/exist/other.pt',sha256='a'*64),[],('.pt',)))
check('forbidden_and_undeclared_reject_before_FS',test_path)

# Metadata/decoder helper source carried exactly from prior saved audit.
old=H.parent/'V222_ACTUAL_MODEL_QA_A/audit_saved_continued_pair.py'
def funs(z):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(z).body if isinstance(n,ast.FunctionDef)}
f,g=funs(old.read_text()),funs((H/'audit_saved_split_pair.py').read_text());names=['need','ident','same','lexical','raw','checked','js','binding','write','meta_equal','context','fit_mask','point_target','weak_rows','indices','state_sha','score_agreement','rle_valid','module','cf_metadata','metrics','reduce_output'];assert all(f[n]==g[n]for n in names);passed.append('22_inherited_metadata_security_decoder_helpers_AST_exact')
with(H/'INVENTED_CHECKS.json').open('x')as out:json.dump(dict(completed=True,PASS=True,checks=passed,actual_SOURCE_payload_or_trained_weight_reads=0,model_training_calls=0),out,ensure_ascii=False,indent=2);out.write('\n')
print(json.dumps(dict(completed=True,PASS=True,checks=passed),ensure_ascii=False))
