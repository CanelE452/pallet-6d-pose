"""Prospective C13 head-only current operator; import/generated paths are inert."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import argparse,ast,csv,hashlib,importlib.util,json,math,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OPERATOR=V/'ROOT_CURRENT_SAVED_FEATURE_OPERATOR_BEFORE_C13_FIT.json';OPERATOR_SHA='8824f676e1fdea306aec8b78f1103b99b5c61c17206b2b82a26ae6d034beb854'
C11=P/'query_visibility_spatial_target_query_features_20261023_v111'
HEAD=C11/'train_spatial_target_query_head.py';HEAD_SHA='f34ce3b44cc491593bbe6b63fa498c0417a5ba70f54318787094ac0802efa5de'
V94=P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation'
DECODER=V94/'infer_same_query_once.py';DECODER_SHA='53fe3d7031aa9a610638465d12d8c17c0778ba0ed289a6a03e4f3540c0d372bd'
CORE=V94/'evaluate_saved_only.py';CORE_SHA='0045abb266c4e3341aa2bf4b02f387f24ec5f04b5c19b74b200000e5e55113b5'
REPORT=C11/'current_evaluation_preparation/evaluate_current_saved.py';REPORT_SHA='83c7cf31ab29154b447e98db7614fec6604307b15c9522508b8b8719740f8478'
METRIC=P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py';METRIC_SHA='d079650b65aef777675af552da61335a28b73f978580413d74d143433019bef6'
OUT=V/'private_evaluation/C13_FROZEN_CURRENT223'
WHITELIST=('features1928','query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside','query_xy','side_L','original_query_xy','original_side_L','id','native_corner')
EVAL_KEYS={'prior_input_rows','evaluation_manifest','manual_provenance','baseline_v91_scores','baseline_v91_inference_seal','historical_v91_metrics','original_baseline_metrics','baseline_C11_scores','historical_C11_metrics'}
NEW='V118_C13_RAW';NEW_GEO='V118_C13_PLUS_OLD_GEOMETRY';OLD='V91_RAW';OLD_GEO='V91_PLUS_OLD_GEOMETRY';BASE='V111_C11_RAW';BASE_GEO='V111_C11_PLUS_OLD_GEOMETRY';MODELS=('v61','legacy',OLD,OLD_GEO,BASE,BASE_GEO,NEW,NEW_GEO)
def require(x,why):
 if not x:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def same(a,b):return isinstance(a,dict)and isinstance(b,dict)and a.get('path')==b.get('path')and a.get('sha256')==b.get('sha256')
def contains(x,b):return same(x,b)or(isinstance(x,dict)and any(contains(v,b)for v in x.values()))or(isinstance(x,list)and any(contains(v,b)for v in x))
def read(p):return json.loads(Path(p).read_text())
def check(d):require(sha(d['path'])==d['sha256'],'Changed binding: '+d['path']);return Path(d['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def pure_core():
 for p,digest in((CORE,CORE_SHA),(DECODER,DECODER_SHA),(REPORT,REPORT_SHA)):require(sha(p)==digest,'Pinned evaluator code changed')
 ns=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(require=require),NEW=NEW,NEW_GEO=NEW_GEO,OLD=OLD,OLD_GEO=OLD_GEO,MODELS=MODELS,Path=Path,csv=csv)
 names={'readcsv','writecsv','key','boolean','numeric_same','score_index','response','join_all','calculate_all','historical_replay'}
 ff=[n for n in ast.parse(CORE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 ff+=[n for n in ast.parse(DECODER.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='decode_rows'];require(len(ff)==11,'Pinned core coverage')
 exec(compile(ast.Module(body=ff,type_ignores=[]),str(CORE)+'::pure_only','exec'),ns)
 extra=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(NEW=NEW,NEW_GEO=NEW_GEO,OLD=OLD,OLD_GEO=OLD_GEO,MODELS=MODELS))
 ff=[n for n in ast.parse(REPORT.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('error_bins','matrices')]
 exec(compile(ast.Module(body=ff,type_ignores=[]),str(REPORT)+'::report_math_only','exec'),extra)
 return SimpleNamespace(**ns),SimpleNamespace(**extra)

def source_guard(e,freeze,fit,gates,review,admission):
 require(freeze.get('completed')is True and freeze['threshold_logit']==0 and freeze['selected_update']in range(0,2001,100),'Frozen source selection')
 require(same(freeze['checkpoint'],e['checkpoint'])and same(freeze['trainer'],e['source_trainer'])and same(freeze['head_class_code'],bind(HEAD)),'Wrong source checkpoint/head')
 require(freeze.get('current_used_for_fit_or_selection')is False and freeze.get('PnP')==0,'Source scope changed')
 require(fit.get('completed')is True and fit['optimizer_updates']==2000 and same(fit['model_freeze'],e['source_freeze']),'Incomplete source fit')
 require(gates.get('completed')is True and gates.get('all_gates_pass')is True and len(gates['gates'])==15 and all(v['PASS']is True for v in gates['gates'].values()),'All15source gates required')
 require(review.get('completed')is True and review.get('PASS')is True and not review.get('blockers',[])and(contains(review,e['source_freeze'])or contains(review,e['source_fit_complete'])),'Independent savedfit review required')
 require(admission.get('completed')is True and admission.get('frozen_model_admitted_for_once_current_comparison')is True and admission.get('source_all15_gates_pass')is True and same(admission['source_model_freeze'],e['source_freeze'])and same(admission['source_saved_review'],e['source_review']),'Root source admission required')

def authorize(path,stage):
 e=read(path);require(e.get('approved')is True and e['stage']==stage,'Explicit root stage entry required')
 require(all(e[k]is False for k in('training_allowed','V112_allowed','PnP_allowed','current_used_for_selection')),'Forbidden scope')
 for k in('code','operator','generated','source_trainer','source_freeze','source_fit_complete','source_gates','source_review','source_admission'):check(e[k])
 require(same(e['code'],bind(__file__))and same(e['operator'],bind(OPERATOR))and sha(OPERATOR)==OPERATOR_SHA and sha(HEAD)==HEAD_SHA,'Frozen code/operator changed')
 proof=read(e['generated']['path']);require(proof['PASS']and same(proof['code'],e['code']),'Stale generated proof')
 source_guard(e,read(e['source_freeze']['path']),read(e['source_fit_complete']['path']),read(e['source_gates']['path']),read(e['source_review']['path']),read(e['source_admission']['path']))
 check(e['checkpoint'])
 # No current descriptor/file hash/body has been read before this point.
 check(e['C11_inference_seal']);seal=read(e['C11_inference_seal']['path'])
 require(seal['completed']and seal['model_unchanged']and seal['slots']==1784 and seal['valid']==1739 and seal['U']==45 and seal['additional_U']==0,'Original C11 input certificate')
 require(contains(seal['outputs'],e['C11_current_packet']),'Original saved features not sealed')
 return e,seal

def load_saved_inputs(path):
 with np.load(path,allow_pickle=False)as f:a={k:np.array(f[k],copy=True)for k in WHITELIST}
 n=1784;x=a['features1928'];require(x.shape==(n,1928)and x.dtype==np.float32 and np.isfinite(x).all(),'Frozen feature schema')
 for k in('query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside'):require(a[k].shape==(n,)and a[k].dtype==bool,'Availability schema')
 require(np.array_equal(a['valid'],a['query_available'])and np.array_equal(a['valid'],a['prior_valid']&a['context_available']&a['local_central4_available']),'Availability changed')
 require((~a['prior_valid']).sum()==45 and (~a['valid']).sum()==45 and not(a['prior_valid']&~a['valid']).any(),'Original45U/noextra required')
 require(a['original_query_xy'].shape==(n,2)and a['original_side_L'].shape==(n,)and np.array_equal(a['query_xy'],a['original_query_xy'].astype(np.float32),equal_nan=True)and np.array_equal(a['side_L'],a['original_side_L'].astype(np.float32),equal_nan=True),'Original q/L metadata changed')
 require(a['id'].shape==a['native_corner'].shape==(n,)and len(set(zip(a['id'].tolist(),a['native_corner'].tolist())))==n and len(set(a['id'].tolist()))==223,'Original native identities')
 require(np.array_equal(x[:,1927],a['context_Q_inside'].astype(np.float32)),'Qflag changed')
 return a

def state_sha(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def predict_generated_or_actual(model,features,available,torch,device):
 with torch.inference_mode():
  x=torch.as_tensor(features,device=device);a=model(x);b=model(x);require(torch.equal(a,b),'Frozen head repeat differs')
  raw=a.float().cpu().numpy();require(raw.shape==(1784,)and np.isfinite(raw).all(),'Finite scalar schema')
  z=np.where(available,raw,0).astype(np.float32);pv=torch.sigmoid(-torch.as_tensor(z)).numpy()
 return raw,z,pv

def infer(entry):
 e,c11seal=authorize(entry,'INFER');out=Path(e['output_directory']);require(out.resolve()==OUT.resolve()and not out.exists(),'One fixed output; no restart')
 check(e['C11_current_packet']);a=load_saved_inputs(e['C11_current_packet']['path'])
 out.mkdir(parents=True);write(out/'STARTED.json',dict(entry=bind(entry),head_only=True,GT_or_prior_geometry_or_baseline_body_read=False,no_automatic_restart=True))
 try:
  import torch
  require(torch.__version__=='2.1.1+cu118'and torch.cuda.is_available(),'Exact learned runtime')
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  check(e['checkpoint']);hm=module(HEAD,'c13_current_exact_C11_head');payload=torch.load(e['checkpoint']['path'],map_location='cpu',weights_only=True);freeze=read(e['source_freeze']['path'])
  require(payload['update']==freeze['selected_update']and same(payload['trainer'],e['source_trainer'])and same(payload['head_class_code'],bind(HEAD)),'Wrong selected source head')
  model=hm.SpatialTargetQueryHead().cuda().eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True);require(sum(p.numel()for p in model.parameters())==221825,'Head architecture changed');del payload
  before=state_sha(model);started=time.monotonic();raw,z,pv=predict_generated_or_actual(model,a['features1928'],a['valid'],torch,'cuda');require(before==state_sha(model),'Frozen head state changed')
  core,_=pure_core();rs=[]
  for i in range(1784):rs.append(dict(id=str(a['id'][i]),native_corner=int(a['native_corner'][i]),query_x=float(a['original_query_xy'][i,0]),query_y=float(a['original_query_xy'][i,1]),roi_side=float(a['original_side_L'][i]),valid=bool(a['valid'][i]),raw_query_aux_hidden_logit=float(z[i]),raw_p_visible_FP32=float(pv[i]),authoritative_decision=('D'if z[i]<0 else'E')if a['valid'][i]else'UNKNOWN'))
  core.writecsv(out/'HEAD_ONLY_SCORES.csv',rs);metadata={k:v for k,v in a.items()if k!='features1928'}
  np.savez_compressed(out/'RAW_HEAD_OUTPUTS.npz',query_logit=z,raw_head_logit_before_U=raw,p_visible_FP32=pv,**metadata)
  write(out/'INFERENCE_SEAL.json',dict(completed=True,PASS=True,entry=bind(entry),code=e['code'],operator=e['operator'],source_model_freeze=e['source_freeze'],checkpoint=e['checkpoint'],C11_inference_seal=e['C11_inference_seal'],C11_current_packet=e['C11_current_packet'],input_fields_read=list(WHITELIST),model_input_fields=['features1928'],feature_values_SHA256=hashlib.sha256(a['features1928'].tobytes()).hexdigest(),feature_shape=[1784,1928],original_q_L_ID_order_and45U_exact=True,frames=223,slots=1784,valid=1739,U=45,additional_U=0,D_rule='z<0',raw_E_threshold_logit=0,probability_is_display_only=True,head_calls=2,head_repeat_exact=True,head_state_before=before,head_state_after=state_sha(model),model_unchanged=True,encoder_RGB_initializer_calls=0,current_GT_or_baseline_scores_read=False,current_used_for_selection=False,optimizer_updates=0,PnP=0,seconds=time.monotonic()-started,outputs=[bind(out/'HEAD_ONLY_SCORES.csv'),bind(out/'RAW_HEAD_OUTPUTS.npz')]))
  print(json.dumps(dict(completed=True,inference=bind(out/'INFERENCE_SEAL.json'))))
 except BaseException:
  write(out/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),restart_automatically=False));raise

def calculate_tables(core,extra,base,truth,prior,newrows,oldrows,C11rows,manual):
 c11=core.score_index(C11rows,'raw_query_aux_hidden_logit');v91=core.score_index(oldrows,'raw_physical_external_logit');core.join_all(truth,prior,c11,v91)
 original=base.call
 def call(row,model,task,probabilities,threshold,rules=None):
  if model in(BASE,BASE_GEO):return core.response(c11[core.key(row)],model==BASE_GEO)
  return original(row,model,task,probabilities,threshold,rules)
 base.call=call
 try:
  tables=core.calculate_all(base,truth,prior,newrows,oldrows,manual)
  tables['coordinate_error_bins']=extra.error_bins(core,base,truth,manual,newrows,oldrows);tables['D_vs_external_E_2x3']=extra.matrices(tables['metrics'])
 finally:base.call=original
 return tables

def evaluate(entry):
 e,c11seal=authorize(entry,'EVALUATE_SAVED');check(e['inference_seal']);seal=read(e['inference_seal']['path'])
 require(seal['completed']and seal['PASS']and seal['model_unchanged']and seal['slots']==1784 and seal['U']==45 and seal['additional_U']==0 and same(seal['source_model_freeze'],e['source_freeze'])and same(seal['checkpoint'],e['checkpoint'])and seal['D_rule']=='z<0'and not seal['current_GT_or_baseline_scores_read'],'Complete frozen new predictions required')
 for d in seal['outputs']:check(d)
 # Human GT, prior geometry and baseline score bodies/hashes begin only after the new prediction seal.
 b=e['bindings'];require(set(b)==EVAL_KEYS,'Exact saved-eval bindings')
 for d in b.values():check(d)
 require(contains(c11seal['outputs'],b['baseline_C11_scores']),'C11 baseline scores unsealed')
 core,extra=pure_core();base=module(check(dict(path=str(METRIC),sha256=METRIC_SHA)),'c13_frozen_metric');truth=core.readcsv(b['evaluation_manifest']['path']);manual=core.readcsv(b['manual_provenance']['path']);prior=[json.loads(x)for x in Path(b['prior_input_rows']['path']).read_text().splitlines()if x]
 old=core.readcsv(b['baseline_v91_scores']['path']);c11rows=core.readcsv(b['baseline_C11_scores']['path']);oldseal=read(b['baseline_v91_inference_seal']['path']);require(oldseal['completed']and contains(oldseal['outputs'],b['baseline_v91_scores']),'V91 baseline not sealed')
 scorepath=next(d['path']for d in seal['outputs']if Path(d['path']).name=='HEAD_ONLY_SCORES.csv');rawpath=next(d['path']for d in seal['outputs']if Path(d['path']).name=='RAW_HEAD_OUTPUTS.npz');minimal=core.readcsv(scorepath)
 with np.load(rawpath,allow_pickle=False)as f:a={k:f[k]for k in f.files}
 require(len(minimal)==len(prior)==1784 and a['query_logit'].shape==(1784,)and a['query_logit'].dtype==np.float32,'Saved rows/schema')
 for i,(r,p)in enumerate(zip(minimal,prior)):
  require(core.key(r)==core.key(p)==(str(a['id'][i]),str(int(a['native_corner'][i]))),'Original order changed')
  require(core.boolean(r['valid'])==bool(a['valid'][i])and float(r['raw_query_aux_hidden_logit'])==float(a['query_logit'][i]),'CSV/NPZ mismatch')
  for axis,k in enumerate(('query_x','query_y')):require(core.numeric_same(r[k],p[k])and core.numeric_same(a['original_query_xy'][i,axis],p[k]),'Original q changed')
  require(core.numeric_same(r['roi_side'],p['roi_side'])and core.numeric_same(a['original_side_L'][i],p['roi_side']),'Original L changed')
  p['prior_reason']=p['reason']
 enriched=core.decode_rows(prior,a['query_logit'],a['p_visible_FP32'],a['query_available'])
 newrows=[{k:(''if v is None else str(v))for k,v in r.items()}for r in enriched]
 tables=calculate_tables(core,extra,base,truth,prior,newrows,old,c11rows,manual)
 count=core.historical_replay(tables['metrics'],core.readcsv(b['original_baseline_metrics']['path']),{'v61','legacy'})
 count+=core.historical_replay(tables['metrics'],core.readcsv(b['historical_v91_metrics']['path']),{OLD,OLD_GEO})
 c11count=core.historical_replay(tables['metrics'],core.readcsv(b['historical_C11_metrics']['path']),{BASE,BASE_GEO})
 out=Path(e['output_directory']);require(out.resolve()==(OUT/'evaluation').resolve()and not out.exists(),'Fixed new evaluation output');out.mkdir()
 for name in('metrics','confusion','human_self_binary_response','manual_near_D','availability_by_group','coordinate_error_bins','D_vs_external_E_2x3'):core.writecsv(out/(name.upper()+'.csv'),tables[name])
 core.writecsv(out/'SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv',enriched)
 write(out/'CURRENT_PRIMARY_RESULTS.json',dict(completed=True,tables=tables,historical_v61_legacy_V91_rows_exact=count,historical_C11_rows_exact=c11count,primary_model=NEW,geometry_routes_separate=[NEW_GEO,BASE_GEO,OLD_GEO],original1784_retained=True,rawsign='D iff z<0, else binaryhidden; factualU first',self='Existing initialgeometry route and humanS binary-response diagnostics separate; no learnedS or cause claim.',coordinate='Original query/reference error diagnostics only; coordinate correctness not established.',current_historically_development_exposed=True,current_used_for_choice=False,new_forward_fit_threshold_selection_PnP=0,bindings=e))
 write(out/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,entry=bind(entry),inference=e['inference_seal'],source_freeze=e['source_freeze'],files=[bind(p)for p in sorted(out.iterdir())if p.is_file()],new_model_forward_fit_current_selection_PnP=0));print(json.dumps(dict(completed=True,evaluation=bind(out/'EVALUATION_SEAL.json'))))

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--infer-entry',type=Path);g.add_argument('--eval-entry',type=Path);a=p.parse_args()
 if a.infer_entry:infer(a.infer_entry)
 else:evaluate(a.eval_entry)
