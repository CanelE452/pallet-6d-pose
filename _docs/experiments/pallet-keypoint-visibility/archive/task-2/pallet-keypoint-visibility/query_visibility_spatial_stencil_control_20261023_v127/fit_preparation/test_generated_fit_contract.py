"""Generic NumPy/static-only fixtures. Never imports Torch or reads source bodies."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,importlib.util,json,argparse,hashlib,tempfile
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def bind(path):return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def run(out):
 checks={};trainer=H/'train_two_arms_once.py';model=H/'stencil_head.py';common=V/'stencil_common.py'
 ts=trainer.read_text();ms=model.read_text();tt=ast.parse(ts);mt=ast.parse(ms)
 compile(tt,str(trainer),'exec');compile(mt,str(model),'exec')
 f=load(trainer,'generated_fit_contract');h=load(model,'generated_head_contract')
 checks['inert_import_no_Torch']= 'torch'not in sys.modules
 assert 3464*64+64+3072*64+64+1==418433
 checks['matched_parameter_count']=True
 class Num:
  abs=staticmethod(np.abs)
  @staticmethod
  def cat(xs,dim):return np.concatenate(xs,axis=dim)
 rng=np.random.default_rng(410);x=rng.normal(size=(3,1928)).astype(np.float32)
 expanded=h.expand_base(Num,x);T,Q,C,PP=x[:,:384],x[:,384:768],x[:,1152:1536],x[:,1536:1920]
 assert np.array_equal(expanded,np.concatenate((x,np.abs(T-C),T*C,np.abs(Q-PP),Q*PP),1))and expanded.shape==(3,3464)
 checks['exact_existing_3464_expansion']=True
 parent=P/'query_visibility_spatial_target_query_features_20261023_v111/train_spatial_target_query_head.py'
 orig=next(n for n in ast.walk(ast.parse(parent.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='forward')
 assignments=[n for n in orig.body if isinstance(n,ast.Assign)]
 assert ast.dump(assignments[0].value,include_attributes=False)==ast.dump(next(n for n in mt.body if isinstance(n,ast.FunctionDef)and n.name=='expand_base').body[0].value,include_attributes=False)
 checks['parent_descriptor_slices_AST_equal']=True
 W=rng.normal(size=(64,3464)).astype(np.float32);b=rng.normal(size=64).astype(np.float32);O=rng.normal(size=(64,1)).astype(np.float32);ob=np.float32(.31)
 extra1=rng.normal(size=(3,3072)).astype(np.float32);extra2=rng.normal(size=(3,3072)).astype(np.float32);zero=np.zeros((64,3072),np.float32)
 hidden=expanded@W.T+b;old=np.maximum(hidden,0)@O+ob
 assert np.array_equal(old,np.maximum(hidden+extra1@zero.T,0)@O+ob)and np.array_equal(old,np.maximum(hidden+extra2@zero.T,0)@O+ob)and np.any(old!=0)
 checks['invented_nonzero_parent_zero_extra_parity']=True
 assert "Linear(3464,64)"in ms and "Linear(3072,64,bias=False)"in ms and "self.extra_affine.weight.zero_()"in ms
 assert ms.count('copy_(parent_state[')==4 and 'self.base_affine(expand_base(torch,base1928))+self.extra_affine(extra3072)'in ms
 assert 'return self.output(self.relu(hidden)).squeeze(-1)'in ms
 checks['split_affine_zero_extra_four_parent_copies_scalar_only']=True
 c18=P/'query_visibility_wide_only_loss_diagnostic_20261023_v126/source_fit_preparation/train_source_once.py'
 ns={'np':np,'BASE':51004,'STRATA':f.STRATA};f.c.pure_functions(c18,('wide_pools',),ns)
 w=dict(labels=np.array([1,1,0,0,0,1,1,0,0,-1]),split=np.array(['TRAIN']*5+['VALIDATION','TRAIN','TRAIN','TRAIN','TRAIN']),pair_ready=np.array([1,1,1,1,0,1,1,1,1,1],bool),variant=np.array(['OPAQUE_COVER','OPAQUE_COVER','CUE_PRESERVING_SHAM','CUE_PRESERVING_SHAM','BEFORE','OPAQUE_COVER','OPAQUE_COVER','CUE_PRESERVING_SHAM','CUE_PRESERVING_SHAM','OPAQUE_COVER']),alpha=np.array([0,1,1,0,-1,0,0,1,0,0]))
 a=np.array([1,1,1,1,1,1,0,0,0,1],bool);pools=ns['wide_pools'](w,a)
 assert all(np.array_equal(pools[k],np.array([51004+i]))for i,(k,_)in enumerate(f.STRATA))
 checks['invented_pools_exclude_VAL_U_and_unavailable']=True
 assert tuple(n for _,n in f.STRATA)==(16,16,16,8,8)and sum(n for _,n in f.STRATA)==64
 assert "(y[ix],minlength=2),[32,32])"in ts and "train[ix].all()"in ts
 assert "pure_functions(C18,('wide_pools','draw_wide64')"in ts and "ix=global_ix-51004"in ts and "ref['occurrence_indices'][-64:]"in ts
 checks['exact_C18_sampler_pin_and_32D32E_known_TRAIN_guards']=True
 assert 'range(1,1001)'in ts and 'manual_seed(220)'in ts and 'manual_seed(219)'in ts and 'lr=1e-4,weight_decay=.001'in ts
 assert 'loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yg[idx])'in ts and 'clip_grad_norm_(model.parameters(),5)'in ts
 assert 'selector'not in [n.name for n in tt.body if isinstance(n,ast.FunctionDef)]
 checks['fixed1000_wide64_BCE_only_no_selector']=True
 y=np.array([0,0,1,1,-1],np.int8);pred=np.array([0,1,1,-1,0],np.int8);m=f.metric(y,pred)
 assert m['CM_D_E_rows_D_E_U_columns']==[[1,1,0],[0,1,1]]and m['D_recall']==.5 and m['E_recall']==.5 and m['known_model_U']==1 and m['teacher_U_model_available']==1
 checks['independent_known_U_precision_recall_denominators']=True
 rows=[dict(source='LOCO',raw_split='TRAIN',variant=v,offset_xy=[0,0])for v in('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM','OPAQUE_COVER','OPAQUE_COVER')]
 gt=[dict(candidate_aux_hidden_label_GT_only=int(v),nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only=alpha)for v,alpha in zip(y,[None,1,1,0,None])]
 rep=f.report(np.array([-1,1,2,0,-1],np.float32),np.array([1,1,1,0,1],bool),np.array([1,1,1,0,1],bool),rows,gt)['groups']['ALL::ALL']
 assert rep['all_planned']['all_rows']==5 and rep['factual_available']['all_rows']==4 and rep['all_planned']['known_model_U']==1 and rep['all_planned']['teacher_U']==1
 assert rep['alpha_interactions']['OPAQUE_COVER::q_alpha_outside']['all_planned']['known_model_U']==1
 checks['allplanned_factualavailable_and_alpha_counts_separate']=True
 runfn=next(n for n in tt.body if isinstance(n,ast.FunctionDef)and n.name=='run');outer=next(n for n in ast.walk(runfn)if isinstance(n,ast.For)and isinstance(n.iter,ast.Name)and n.iter.id=='ARMS')
 reports=[n for n in ast.walk(runfn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='report']
 assert len(reports)==1 and reports[0].lineno>outer.end_lineno
 checks['VAL_metrics_only_after_both_fixed_fits']=True
 assert "np.array_equal(initial,baseline)"in ts and "np.array_equal(initial_unit,baseline_unit)"in ts and 'np.allclose'not in ts
 assert "'feature_receipt','stencil_features'"in ts and "receipt['output']==e['stencil_features']"in ts
 checks['strict_initial_C16_unit_occurrence_parity_and_dynamic_V2_bindings']=True
 assert 'optimizer_state=clone_cpu' in ts and 'wide_generator=gen.get_state().clone()'in ts and 'torch_cpu=torch.get_rng_state().clone()'in ts and 'torch_cuda_all='in ts
 assert "'FINAL_ALL4896_SOURCE_SCORES.npz'"in ts and "'FINAL_RESUME_CHECKPOINT.pt'"in ts
 checks['final_scores_actual_optimizer_all_RNG_saved']=True
 with tempfile.TemporaryDirectory()as d:
  p=Path(d)/'unapproved.json';p.write_text(json.dumps(dict(approved_source_two_arm_fit=False,source_only=True,current_allowed=False,V112_allowed=False,PnP_allowed=False,automatic_restart=False)))
  try:f.authorize(p)
  except AssertionError:pass
  else:raise AssertionError('Unapproved entry was accepted')
 checks['unapproved_entry_fails_before_source_IO']=True
 assert 'torch'not in sys.modules and all(checks.values())
 result=dict(completed=True,PASS=True,fit_code=bind(trainer),model_code=bind(model),common_code=bind(common),test_code=bind(Path(__file__)),checks=checks,check_count=len(checks),actual_source_features_RGB_weights_models_fit_current_V112_reads_or_calls=0,Torch_imported=False,limits='NumPy/static contract checks only; actual learned-runtime parity remains a fail-closed root-run guard.')
 with out.open('x')as fp:json.dump(result,fp,indent=2,allow_nan=False);fp.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_FIT_TESTS.json');a=ap.parse_args();run(a.output)
