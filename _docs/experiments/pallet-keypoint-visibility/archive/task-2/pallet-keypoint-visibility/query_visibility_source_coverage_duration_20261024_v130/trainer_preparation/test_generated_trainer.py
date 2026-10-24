"""Invented CPU sampler/static duration checks; no model, weights or source arrays."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys,tempfile
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def node(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def run(output):
 code=H/'train_C22_C23_once.py';text=code.read_text();tree=ast.parse(text);compile(tree,str(code),'exec');sp=importlib.util.spec_from_file_location('v130_generated',code);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);assert 'torch'not in sys.modules
 parenttree=ast.parse(m.PARENT_TRAINER.read_text());unchanged=('metric','report','gate_results','choose','eligibility','ordered_new_rows','load_metadata','clone')
 assert all(dump(node(tree,k))==dump(node(parenttree,k))for k in unchanged)
 checks={'inert_import_no_Torch_weights_or_arrays':True,'unchanged_parent_metadata_report_gate_armchoice_helper_ASTs':True}
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'entry.json';p.write_text(json.dumps({'approved_source_C22_C23_duration_fit':False}))
  try:m.authorize(p)
  except AssertionError:pass
  else:raise AssertionError('Unapproved fit accepted')
 checks['unapproved_entry_fails_before_data_parameters']=True
 y=np.array([0,1,0,1,-1,-1,0,1,0,1]);available=np.array([1,1,1,1,1,1,0,1,1,1],bool);split=np.array(['TRAIN']*8+['VALIDATION']*2);corpus=np.array(['OLD_PUBLIC','PUBLIC_AUX','NATIVE','NATIVE','NATIVE','NATIVE','NATIVE','PUBLIC_AUX','PUBLIC_AUX','NATIVE']);pub,d,e=m.eligibility(y,available,split,corpus)
 assert np.flatnonzero(pub).tolist()==[0,1,7]and np.flatnonzero(d).tolist()==[2]and np.flatnonzero(e).tolist()==[3]
 checks['known_available_TRAIN_only_no_S_U_VAL']=True
 # CPU generator calls only. No nn.Module, optimizer or CUDA/model calls.
 import torch
 assert torch.__version__=='2.1.1+cu118'
 pools=[np.arange(1407,dtype=np.int64),np.arange(2000,19154,dtype=np.int64),np.arange(20000,21206,dtype=np.int64)]
 a,b,pg,ng=m.make_plans(torch,pools,8000);aa,bb,_,_=m.make_plans(torch,pools,1000)
 assert a.shape==b.shape==(8000,64)and np.array_equal(a[:1000],aa)and np.array_equal(b[:1000],bb)
 g=torch.Generator().manual_seed(1301);g2=torch.Generator().manual_seed(1302)
 original_public=np.stack([pools[0][torch.randint(len(pools[0]),(64,),generator=g).numpy()]for _ in range(1000)])
 original_native=np.stack([np.r_[pools[1][torch.randint(len(pools[1]),(32,),generator=g2).numpy()],pools[2][torch.randint(len(pools[2]),(32,),generator=g2).numpy()]]for _ in range(1000)])
 assert np.array_equal(a[:1000],original_public)and np.array_equal(b[:1000],original_native)
 assert np.isin(a,pools[0]).all()and np.isin(b[:,:32],pools[1]).all()and np.isin(b[:,32:],pools[2]).all()
 # A consumed generator continues rather than restarting at1000.
 continued=np.stack([pools[0][torch.randint(len(pools[0]),(64,),generator=g).numpy()]for _ in range(7000)]);assert np.array_equal(a[1000:],continued)
 assert pg.get_state().dtype==ng.get_state().dtype==torch.uint8
 checks.update({'8000_sampler_first1000_exact_original_seeds_and_calls':True,'native64_has32D32E_all8000':True,'public_and_native_generator_continuation_not_reset':True,'both_arms_use_same_precomputed_public_plan':True})
 fit=node(tree,'run');armloop=next(n for n in ast.walk(fit)if isinstance(n,ast.For)and isinstance(n.iter,ast.Name)and n.iter.id=='ARMS');step=next(n for n in ast.walk(armloop)if isinstance(n,ast.For)and isinstance(n.iter,ast.Call)and isinstance(n.iter.func,ast.Name)and n.iter.func.id=='range'and n.iter.args and isinstance(n.iter.args[0],ast.Name)and n.iter.args[0].id=='TOTAL')
 optcalls=[n for n in ast.walk(fit)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='AdamW'];assert len(optcalls)==1 and not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='AdamW'for n in ast.walk(step))
 assert 'lr=1e-4,weight_decay=.001'in text and 'clip_grad_norm_(model.parameters(),5)'in text and "if arm==ARMS[1]:nl="in text and 'loss=pl+nl'in text
 checks['one_optimizer_per_arm_no_reset_at1000_same_loss_lr_clip']=True
 assert "record==reference_traces[arm][step]"in text and "actual==prior['model_state_sha256']"in text and "if step+1==PREFIX:"in text and "assert parity['PASS']"in text
 checks['first1000_exact_loss_gradient_trace_and_state_SHA_before1001']=True
 reports=[n for n in ast.walk(fit)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in('report','gate_results')];assert min(n.lineno for n in reports)>armloop.end_lineno
 assert not any(isinstance(n,ast.FunctionDef)and'selector'in n.name for n in tree.body)
 checks['both_final8000_optimizations_before_VAL_no_checkpoint_selector']=True
 assert 'optimizer_state=clone('in text and 'public_generator_end=pg.get_state()'in text and 'native_generator_end=ng.get_state()'in text and 'update1000_parity=bind('in text and 'FINAL_ALL61110_SOURCE_SCORES.npz'in text and 'current_evaluation_authorized=False'in text
 checks['final_scores_resume_optimizer_RNG_and_prefix_certificate_saved']=True
 assert 'No state-hash tolerance or optimizer reset allowed.'in text and 'No sampler-prefix tolerance or reseeding allowed.'in text and 'source_validation_development_exposed=True'in text
 checks['strict_prefix_failclosed_and_development_exposure_explicit']=True
 result={'completed':True,'PASS':True,'code':bind(code),'test_code':bind(Path(__file__)),'runtime':{'python':sys.executable,'torch':torch.__version__,'CUDA_model_calls':0},'checks':checks,'check_count':len(checks),'sampler_first1000_invented_SHA256':hashlib.sha256(a[:1000].tobytes()+b[:1000].tobytes()).hexdigest(),'unchanged_parent_helper_ASTs':{k:hashlib.sha256(dump(node(tree,k)).encode()).hexdigest()for k in unchanged},'actual_source_arrays_weights_model_fit_current_V112_reads_or_calls':0}
 with output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'completed':True,'PASS':True,'code':result['code'],'checks':len(checks),'output':bind(output)}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_TRAINER_TESTS.json');args=ap.parse_args();run(args.output)
