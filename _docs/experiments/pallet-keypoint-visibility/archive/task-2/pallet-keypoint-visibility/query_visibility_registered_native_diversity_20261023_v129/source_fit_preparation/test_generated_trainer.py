"""Bounded invented NumPy/static fit checks; never loads model/data/weights."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,tempfile
import numpy as np
H=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def run(out):
 code=H/'train_C20_C21_once.py';text=code.read_text();tree=ast.parse(text);compile(tree,str(code),'exec');spec=importlib.util.spec_from_file_location('invented_C20_C21',code);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);checks={}
 assert 'torch'not in sys.modules;checks['inert_no_Torch_arrays_or_weights']=True
 by={'NATIVE':([{'id':'n0'},{'id':'n1'}],[{'id':'n0'},{'id':'n1'}]),'PUBLIC_AUX':([{'id':'p0'}],[{'id':'p0'}])};names=np.array(['PUBLIC_AUX','NATIVE','NATIVE']);indices=np.array([0,1,0]);r,g=m.ordered_new_rows(names,indices,by);assert [x['id']for x in r]==['p0','n1','n0']
 try:m.ordered_new_rows(names,np.array([0,0,0]),by)
 except AssertionError:pass
 else:raise AssertionError('Duplicate/missing source mapping accepted')
 checks['explicit_new_corpus_mapping_order_and_unique_retention']=True
 y=np.array([0,1,0,1,-1,-1,0,1,0,1]);a=np.array([1,1,1,1,1,1,0,1,1,1],bool);split=np.array(['TRAIN']*8+['VALIDATION']*2);corpus=np.array(['OLD_PUBLIC','PUBLIC_AUX','NATIVE','NATIVE','NATIVE','NATIVE','NATIVE','PUBLIC_AUX','PUBLIC_AUX','NATIVE']);pub,d,e=m.eligibility(y,a,split,corpus)
 assert np.flatnonzero(pub).tolist()==[0,1,7]and np.flatnonzero(d).tolist()==[2]and np.flatnonzero(e).tolist()==[3];checks['TRAIN_known_available_only_S_U_VAL_excluded']=True
 zp=np.array([-2.,1.]);yp=np.array([0,1]);zn=np.array([.2,-.4]);yn=np.array([1,0]);pl=float((np.logaddexp(0,zp)-yp*zp).mean());nl=float((np.logaddexp(0,zn)-yn*zn).mean());assert pl+nl==2*np.mean([pl,nl])and not np.isclose(pl+nl,np.mean([pl,nl]));checks['C20_public_mean_C21_sum_domain_means_not_pooled128']=True
 assert 'public_plan=np.stack' in text and 'native_plan=np.stack'in text and text.index('public_plan=np.stack')<text.index('parent=payload')<text.index('for arm in ARMS')
 assert "manual_seed(1301)"in text and "manual_seed(1302)"in text and "manual_seed(1300)"in text and "pi=public_plan[step]"in text and "if arm=='C21_PUBLIC_PLUS_NATIVE':nl="in text and "old_proxy_loss=0"in text
 checks['shared_prehead_public_sequence_C21_only_native_seed_contract']=True
 metric=m.metric(np.array([0,0,1,1,-1,-1]),np.array([0,1,1,-1,1,0]),np.array(['D','D','E','E','S','U']));assert metric['CM_D_E_rows_D_E_U_columns']==[[1,1,0],[0,1,1]]and metric['known_input_U']==1 and metric['teacher_S_hidden_response']==1 and metric['teacher_U_available']==1;json.dumps(metric,allow_nan=False);checks['known_U_denominators_S_separate_JSON_scalars']=True
 # Synthetic full-sized oldVAL stress masks retain the frozen483/394 denominators.
 size=3582+20;y=np.r_[np.zeros(1908,np.int8),np.ones(1674,np.int8),np.tile([0,1],10)];state=np.where(y==0,'D','E');split=np.full(size,'VALIDATION');corpus=np.r_[np.full(3582,'OLD_PUBLIC'),np.full(10,'PUBLIC_AUX'),np.full(10,'NATIVE')];aa=np.ones(size,bool);gt=[dict(variant='BEFORE',geometry_pair_ready=False,nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only=None)for _ in range(size)]
 for i in range(394):gt[i]=dict(variant='CUE_PRESERVING_SHAM',geometry_pair_ready=True,nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only=True)
 for i in range(1908,1908+483):gt[i]=dict(variant='OPAQUE_COVER',geometry_pair_ready=True,nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only=False)
 base=np.where(y==1,1.,-1.).astype(np.float32);base[100:394]=1;base[2008:2391]=-1;z=base.copy();z[100]=-1;z[2008]=1;good=m.gate_results(z,aa,base,aa,y,state,split,corpus,gt);assert len(good['gates'])==8 and good['all_gates_pass'];bad=m.gate_results(base,aa,base,aa,y,state,split,corpus,gt);assert not bad['all_gates_pass']and not bad['gates']['old_Eout_plus1']['PASS']and not bad['gates']['old_SHAMin_plus1']['PASS'];checks['eight_gates_joint_plus1_no_tradeoff']=True
 assert m.choose({arm:good for arm in m.ARMS})=='C20_PUBLIC_REPLAY_CONTROL'and m.choose({arm:bad for arm in m.ARMS})is None
 stronger=dict(good,worst_six_class_recall=good['worst_six_class_recall']+.001);assert m.choose({m.ARMS[0]:good,m.ARMS[1]:stronger})==m.ARMS[1];checks['source_only_worstclass_choice_tieC20_none_if_fail']=True
 assert "raw_old=model.eval()(xo)"in text and "range(0,NEW,512)"in text and "np.array_equal(ro,ob['unit_raw_logit'])"in text and "np.array_equal(rn,nb['raw_head_logit_before_U'])"in text and 'all61110_safe_logits_bitexact'in text;checks['warmstart_old4428_new512_chunk_strict_parity']=True
 runfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');fitloop=next(n for n in ast.walk(runfn)if isinstance(n,ast.For)and isinstance(n.iter,ast.Name)and n.iter.id=='ARMS');reports=[n for n in ast.walk(runfn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in('report','gate_results')];assert reports and min(n.lineno for n in reports)>fitloop.end_lineno
 assert 'range(1000)'in text and 'lr=1e-4,weight_decay=.001'in text and 'clip_grad_norm_(model.parameters(),5)'in text and 'selector'not in [n.name for n in tree.body if isinstance(n,ast.FunctionDef)]
 checks['both_fixed1000_fits_before_VAL_no_checkpoint_selector']=True
 assert 'optimizer_state=clone('in text and 'public_generator_end=pg.get_state()'in text and 'native_generator_end=ng.get_state()'in text and 'FINAL_ALL61110_SOURCE_SCORES.npz'in text and 'current_evaluation_authorized=False'in text;checks['all61110_scores_optimizer_RNG_no_private_autorun']=True
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'entry.json';p.write_text(json.dumps(dict(approved_source_C20_C21_fit=False,current223=False,V112=False,PnP=False,encoder_forward=False,automatic_retry=False)))
  try:m.authorize(p)
  except AssertionError:pass
  else:raise AssertionError('Unapproved source fit accepted')
 checks['unapproved_fit_fails_before_arrays_parameters']=True
 assert all(checks.values())and 'torch'not in sys.modules
 result=dict(completed=True,PASS=True,code=bind(code),test_code=bind(Path(__file__)),checks=checks,check_count=len(checks),actual_source_array_weight_model_fit_current_V112_reads_or_calls=0)
 with out.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,code=result['code'],checks=len(checks),output=bind(out))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_TRAINER_TESTS.json');a=ap.parse_args();run(a.output)
