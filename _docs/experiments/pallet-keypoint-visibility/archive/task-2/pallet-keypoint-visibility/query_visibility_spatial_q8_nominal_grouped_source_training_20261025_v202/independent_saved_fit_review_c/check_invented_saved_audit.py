"""Small NumPy/scalar invented tests only. No model imports or source bodies."""
import sys,types,json,hashlib,tempfile,math
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent;p=H/'audit_saved_spatial_source_fit.py';m=types.ModuleType('independent_audit_fixture');m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
checks=[]
def ok(name,condition):assert condition,name;checks.append({'name':name,'PASS':True})
def rejected(name,fn):
 try:fn();passed=False
 except (AssertionError,ValueError):passed=True
 ok(name,passed)
shapes=m.parameter_shapes();ok('36tensors118359_scalar_shapes',len(shapes)==36 and sum(math.prod(s)for s in shapes.values())==118359 and shapes['spatial_residual.0.weight']==(32,96))
for arm,flag in zip(m.ARMS,(False,True)):
 envelope=dict(state_dict={},optimizer_state={},update=4000,seed=1810,arm=arm,use_sam=False,spatial_q8=flag,protocol={'p':1},source_split={'s':1});m.checkpoint_envelope(envelope,{'p':1},arm,{'s':1});ok('checkpoint_envelope_'+str(flag),True)
 bad=dict(envelope,spatial_q8=not flag);rejected('wrong_arm_flag_'+str(flag),lambda:m.checkpoint_envelope(bad,{'p':1},arm,{'s':1}))
rejected('input_NPZ_no_whitelist_before_header',lambda:m.arrays({'path':'/no/file/input.npz','sha256':'0'*64},m.META,{}))
rejected('generic_raw_binary_before_filesystem',lambda:m.raw('/no/file/input.npz'))
rejected('forbidden_name_before_filesystem',lambda:m.safe('/no/file/V112/body.json'))
# Old five pools with original full owner class denominators; all invented in RAM.
N=6;S={'x':{'original_query_available':np.ones(N,bool)},'y':{'support_target':np.array([[1,1],[0,0],[0,0],[1,1],[1,0],[0,1]],np.int8),'owner_soft':np.array([[.5,.2,0],[.1,.2,.4],[0,.2,.3],[.1,.1,.1],[.2,0,.1],[0,.3,.1]],np.float64)}}
P={'x':{'original_query_available':np.ones(253,bool)},'y':{'support_target':np.full((253,2),-1,np.int8),'owner_target':np.repeat(np.arange(3,dtype=np.int8),[53,52,148])}}
F={'x':{'original_query_available':np.ones(314,bool)},'y':{'support_target':np.full((314,2),-1,np.int8),'owner_target':np.repeat(np.arange(2,dtype=np.int8),[157,157])}}
D={'x':{'original_query_available':np.array([True,True,True,True,True,False])},'y':{'direct_end_external_target':np.array([0,0,1,1,-1,-1],np.int8)}}
dom=dict(SYNTH_AUG65536=S,PUBLIC_TRAIN1480=P,CF3816=F,DIRECT_CONTROL=D);masks=dict(SYNTH_AUG65536=np.array([True,False,True,False,True,True]),PUBLIC_TRAIN1480=np.ones(253,bool),CF3816=np.ones(314,bool),DIRECT_CONTROL=np.array([True,False,True,False,True,False]))
originalmask={k:v.copy()for k,v in masks.items()};pool,dp=m.filtered_pools(dom,masks)
ok('all_pool_masks_no_role_mutation',all(np.array_equal(v,originalmask[k])for k,v in masks.items())and all(masks['SYNTH_AUG65536'][q].all()for g in pool['support']for q in g)and dp[0].tolist()==[0]and dp[1].tolist()==[2])
for j in range(3):
 full=S['y']['owner_soft'][:,j];expected=full.copy();expected[~masks['SYNTH_AUG65536']]=0;expected/=sum(expected);ok('rawmass_classconditional_'+str(j),np.allclose(pool['conditional'][j],expected,rtol=0,atol=2e-16))
# Independent explicit RNG log construction; files live only in external tempdir.
with tempfile.TemporaryDirectory(prefix='v202_c_saved_audit_')as td:
 path=Path(td)/'invented4000.jsonl';rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);dh=hashlib.sha256();ddh=hashlib.sha256()
 with path.open('x')as f:
  for step in range(1,4001):
   draw={'visible':np.r_[rng.choice(pool['support'][0][0],16),rng.choice(pool['support'][0][1],16)],'foreign':np.r_[rng.choice(pool['support'][1][0],16),rng.choice(pool['support'][1][1],16)],'synthetic_owner':np.concatenate([rng.choice(6,16,p=prob)for prob in pool['conditional']]),'public_owner':np.concatenate([rng.choice(q,16)for q in pool['public']]),'CF_owner':np.concatenate([rng.choice(q,16)for q in pool['CF']])};direct=np.r_[drng.choice(dp[0],16),drng.choice(dp[1],16)]
   for k,q in draw.items():dh.update(k.encode());dh.update(q.astype('<i8').tobytes())
   ddh.update(direct.astype('<i8').tobytes());r=dict(update=step,old_five_losses=[1.,2.,3.,4.,5.],old_five_mean=3.,direct_loss=.25,total_loss=3.25,grad_norm=1.,all_drawn_rows_TRAIN=True,original192_draws={k:q.tolist()for k,q in draw.items()},direct32_draws=direct.tolist(),old_draw_sha=dh.hexdigest(),direct_draw_sha=ddh.hexdigest());f.write(json.dumps(r)+'\n')
 binding={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};oldopen=m.opened
 def onlytemp(p):assert str(p)==str(path);return path.open('rb')
 m.opened=onlytemp
 result=m.audit_log(binding,pool,dp,dh.hexdigest(),ddh.hexdigest(),masks);m.opened=oldopen
 ok('all4000_rng_old192_direct32_and_hashes',result['updates']==4000 and result['all192_plus32_RNG_draws_exact'])
 bad=dict(r,total_loss=99.);rejected('bad_additive_loss',lambda:m.loss_record(bad,4000));bad=dict(r,grad_norm=float('nan'));rejected('nonfinite_logged_gradient',lambda:m.loss_record(bad,4000))
# Partial UNKNOWN group retained, no fabrication of missing COVER/SHAM.
rows=[];roles=[];target=[]
for attempt,native,variants,truth,role in [('A',0,['BEFORE','COVER','SHAM'],[0,1,0],'TRAIN'),('A',1,['BEFORE','COVER','SHAM'],[-1,-1,-1],'TRAIN'),('B',2,['BEFORE'],[-1],'HOLDOUT')]:
 for variant,y in zip(variants,truth):rows.append(dict(domain='SYNTH_FINITE_END',attempt_id=attempt,native_slot=native,variant=variant));roles.append(dict(role=role,component_id=attempt));target.append(y)
target=np.array(target,np.int8);preds={m.ARMS[0]:np.array([0,1,0,1,0,1,-1],np.int8),m.ARMS[1]:np.array([0,0,0,0,1,0,-1],np.int8)}
summary,records=m.independent_triplets(rows,roles,target,preds)
ok('partial_UNKNOWN_not_imputed_or_scored',summary['FULL']['physical_native_attempt_groups']==3 and summary['FULL']['known_D_E_D_triplets']==1 and summary['FULL']['incomplete_UNKNOWN_groups']==1 and records[-1]['targets']==[-1]and records[-1]['missing_variants']==['COVER','SHAM'])
ok('only_knownDED_accuracy_denominator',summary['FULL']['models'][m.ARMS[0]]['all3_correct']==1 and summary['FULL']['models'][m.ARMS[1]]['all3_correct']==0 and summary['FULL']['paired']['damaged_by_XY']==1)
av=np.array([True]*6+[False]);cell=m.exact_metric(target,preds[m.ARMS[0]],av,list(range(7)))
ok('CM_unknown_and_inputU_separate',cell['known']==3 and cell['D']==2 and cell['E']==1 and cell['target_UNKNOWN']==4 and cell['original_inputU']==1 and cell['CM_D_E_by_D_E_U']==[[2,0,0],[0,1,0]]and cell['UNKNOWN_target_prediction_counts']=={'-1':1,'0':1,'1':2})
# Context/role join test uses only injected fabricated row dicts, not loader/body access.
base=[]
for i,(family,state,name)in enumerate([('SELF_SCENE15',0,'NOMINAL'),('SELF_SCENE15',1,'JITTER'),('BEFORE_ANCHOR4',0,'NOMINAL')]):base.append(dict(row_id='r'+str(i),native_slot=i,original_query_available=True,family=family,state_index=state,state_name=name))
mask=np.array([True,True,False]);mapping=[dict(cache_index=i,row_id=r['row_id'],native_slot=i,original_query_available=True,fit_nominal_eligible=i!=1,immutable_TRAIN_role=bool(mask[i]),context_FP32_native_mask_xy_sha256='0'*64)for i,r in enumerate(base)]
census=dict(completed=True,PASS=True,binary_header_pixel_weight_model_target_array_or_score_body_reads=0,row_mappings={'SYNTH_AUG65536':{'path':'invented'}});oldjl=m.jl;m.jl=lambda binding:mapping
fit,eligible=m.nominal_and_context({'SYNTH_AUG65536':{'rows':base}},{'SYNTH_AUG65536':mask},census);m.jl=oldjl
ok('nominal_only_fit_originalTRAIN_unchanged',fit['SYNTH_AUG65536'].tolist()==[True,False,False]and mask.tolist()==[True,True,False]and eligible['SYNTH_AUG65536'].tolist()==[True,False,True])
report={'completed':True,'PASS':True,'invented_only':True,'code':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'checks':checks,'count':len(checks),'source_array_header_pixel_target_checkpoint_or_score_reads':0,'model_objects_forwards_optimizer_or_CUDA':0,'toy_NPZ_or_checkpoint_files_written':0,'external_temporary_JSON_log_removed':True}
z=(json.dumps(report,indent=2)+'\n').encode()
with(H/'GENERATED_INVENTED_CHECKS.json').open('xb')as f:f.write(z)
print(json.dumps({'PASS':True,'checks':len(checks),'sha256':hashlib.sha256(z).hexdigest()}))
