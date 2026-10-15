"""V31 postfit independent replay from saved CAL errors only; no reference/fit imports."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import gzip,hashlib,json,math
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/research').is_dir())
OUT=Path(__file__).resolve().parent
MODELS=OUT.parent/'CAL_geometry_guided_edge_noise_v4_31'
PLAN=OUT/'SAVED_ERROR_AUDIT_PLAN_V1.json'
OLD_SCRIPT=OUT.parent/'CAL_geometry_guided_edge_noise_v4_29_checks/replay_saved_CAL_models_v1.py'
OLD_RECEIPT=OLD_SCRIPT.parent/'SAVED_CAL_MODEL_REPLAY_V1.json'
FAMILY='OBSERVED_Q_ANCHORED_FIXED_NATIVE_RAY_ACTUAL_EDGES_V31'
POPULATION='V31_Q_ANCHORED_SIGNED_RAY_RUNTIME_ACTIVE_ONE_LINE_PER_PHYSICAL_EDGE'
PROFILE='OBSERVED_BI_Q_QUERY_ANCHOR_AND_FIXED_SIGNED_NATIVE_RAY_PLUS_NORMAL_REACH_V31'
RAW='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
CENTRAL=OUT.parent/'observed_anchor_native_ray_method_plan_v4_31/PREDECLARATION.json'
CHECKS=[];MAXIMUM=0.
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ck(name,value):CHECKS.append(dict(name=name,passed=bool(value)))
def near(name,a,b,tol=1e-9):
 global MAXIMUM
 d=float(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float))));MAXIMUM=max(MAXIMUM,d)
 ck(name,np.isfinite(d)and d<=tol*max(1.,float(np.max(np.abs(np.asarray(b,float))))))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def balanced(records,kind,held):
 key='groups'if kind=='line'else'tangents';train=[r for r in records if r['compatible']and r['session']!=held and r[key]]
 sessions=sorted({r['session']for r in train});counts=Counter(r['session']for r in train);items=[];weights=[]
 for row in train:
  items.extend(row[key]);weights.extend([1./len(sessions)/counts[row['session']]/len(row[key])]*len(row[key]))
 return train,sessions,items,np.asarray(weights)
def evaluate(model,items,w,kind,parameters):
 delta=3.;value=0.;gradient=np.zeros(2 if kind=='line'else 1)
 for item,weight in zip(items,w):
  if kind=='line':
   if len(item['components'])!=1:raise ValueError('Graphselected component1 required')
   c=item['components'][0];error=np.array([c['normal_error_px'],c['angular_sine_error']]);floor=np.asarray(c['conditional_variance_floor'])
   fn=max(1.,c['normal_floor_px']);span=c['actual_inlier_span_px'];expected=np.array([fn*fn,2*fn*fn/(span*span+2*fn*fn)])
   near('floor',floor,expected,1e-13);variance=np.asarray(parameters)+floor;axis=np.ones(2)
   ck('prior1',c['mixture_component_prior']==1.)
  else:
   error=np.array([item['signed_tangent_error_px']]);axis=np.array([item['original_variance_axis_px2']]);variance=1.+float(parameters)*axis
   near('sensitivity4a',item['exact_saved_old_tangent_variance_px2'],4.*axis[0])
  z=float(np.sum(error*error/variance));den=math.sqrt(1.+z/(delta*delta))
  value+=weight*(float(np.log(variance).sum())+2.*delta*delta*(den-1.))
  gradient+=weight*axis*(1./variance-error*error/(variance*variance*den))
 return float(value),gradient

def model_check(policy,kind,entry,records,held):
 model=entry.get('model',entry);prefix=policy+'/'+kind+'/'+str(held)
 train,sessions,items,w=balanced(records,kind,held)
 if not items:
  ck(prefix+'/explicitunavailable',entry.get('status','').startswith('MODEL_UNAVAILABLE')and entry.get('model')is None);return
 ck(prefix+'/newgeneration',model['observation_generation']==FAMILY and model['raw_observation_generation']==RAW and model['source_policy']==policy and model['noise_population']==POPULATION and model['selection_profile']==PROFILE)
 ck(prefix+'/schema',model['schema']==('CAL_OBSERVED_Q_ANCHORED_SIGNED_NATIVE_RAY_EDGE_NOISE_V31'if kind=='line'else'CAL_OBSERVED_Q_ANCHORED_SIGNED_NATIVE_RAY_TANGENT_SCALE_V31'))
 ck(prefix+'/newlyfitted',model['newly_fitted_selection_population']is True and model['prior_V29_fitted_model_relabelled']is False and model['prior_V30_fitted_model_relabelled']is False)
 ck(prefix+'/digest',digest({k:v for k,v in model.items()if k!='model_digest_sha256'})==model['model_digest_sha256'])
 ck(prefix+'/trainids',model['fit_ids']==[r['id']for r in train]and model['fit_sessions']==sessions)
 ck(prefix+'/heldexcluded',model['excluded_heldout_ids']==[r['id']for r in records if r['session']==held])
 ck(prefix+'/CAL46fixed44',model['inventory_frame_count']==46 and model['compatible_inventory_count']==44 and model['fit_frame_count']==len(train))
 ck(prefix+'/singleSession',model['single_session_scale_caveat']==(len(sessions)<2))
 near(prefix+'/equalweights',w.sum(),1.)
 values=[]
 for i,search in enumerate(model['optimizer_searches']):
  p=search['variance_components']if kind=='line'else search['gamma'];value,gradient=evaluate(model,items,w,kind,p)
  near(prefix+'/NLL'+str(i),value,search['training_mixture_nll']if kind=='line'else search['training_NLL'])
  near(prefix+'/gradient'+str(i),gradient,search['gradient_wrt_variance']if kind=='line'else[search['gradient']],1e-7)
  values.append(value)
 winner=min(range(len(values)),key=lambda i:((model['optimizer_searches'][i]['training_mixture_nll']if kind=='line'else model['optimizer_searches'][i]['training_NLL']),i))
 ck(prefix+'/selected',winner==model['selected_optimizer_start_index'])
 if kind=='line':
  near(prefix+'/sigmaN',model['sigma_normal_px']**2,model['normal_variance_component_px2']);near(prefix+'/sigmaS',model['sigma_sine']**2,model['angular_variance_component_sine2'])
  ck(prefix+'/componentflooronce',model['normal_variance_component_px2']>=0 and model['angular_variance_component_sine2']>=0)
 else:ck(prefix+'/gammafloor',model['gamma']>=0 and model['pixel_floor_variance_px2']==1.)

def _central(expected):
 if not isinstance(expected,str)or len(expected)!=64 or any(c not in '0123456789abcdef'for c in expected)or sha(CENTRAL)!=expected:raise ValueError('Explicit same central31 SHA required before decode')
 central=read(CENTRAL)
 if central['observation_family']!=FAMILY or central['population']!=POPULATION or central['selection_profile']!=PROFILE:raise ValueError('New31 typed central required')
 merged={}
 for key in ('source_sha256','input_sha256','protected_user_sha256'):
  for relative,h in central[key].items():
   if relative in merged and merged[relative]!=h:raise ValueError('Central cross-class conflict')
   if sha(ROOT/relative)!=h:raise ValueError('Central source/input/protected changed: '+relative)
   merged[relative]=h
 return central

def main(*,root_saved_error_audit_GO=False,expected_model_seal_sha256=None,expected_central_plan_sha256=None):
 if root_saved_error_audit_GO is not True:raise PermissionError('Separate root postfit saved-error audit GO before IO')
 _central(expected_central_plan_sha256)
 declaration=read(PLAN)
 if declaration['central_plan_sha256']!=expected_central_plan_sha256:raise ValueError('Same predeclared central31 required')
 if sha(PLAN)!=read(OUT/'SAVED_ERROR_AUDIT_PLAN_SEAL_V1.json')['plan_sha256']:raise ValueError('Frozen audit plan required')
 for mapping in (declaration['source_sha256'],declaration['input_sha256']):
  for relative,h in mapping.items():
   if sha(ROOT/relative)!=h:raise ValueError('Frozen audit source/input changed: '+relative)
 if not isinstance(expected_model_seal_sha256,str)or len(expected_model_seal_sha256)!=64 or sha(MODELS/'MODEL_SEAL.json')!=expected_model_seal_sha256:raise ValueError('Explicit root-approved fitted V31 model seal required')
 dest=OUT/'SAVED_CAL_MODEL_REPLAY_V1.json'
 if dest.exists():raise FileExistsError('Preserve completed owner checks')
 seal=read(MODELS/'MODEL_SEAL.json')
 ck('samecentral',seal['input_sha256'].get(str(CENTRAL.relative_to(ROOT)))==expected_central_plan_sha256)
 ck('newsealpopulation',seal['observation_generation']==FAMILY and seal['raw_observation_generation']==RAW and seal['noise_population']==POPULATION and seal['selection_profile']==PROFILE)
 ck('prepseal',sha(MODELS/'PREPARATION_SEAL.json')==seal['preparation_seal_sha256'])
 for rel,h in seal['assets'].items():ck('asset/'+rel,sha(MODELS/rel)==h)
 for key in ('source_sha256','input_sha256'):
  for rel,h in seal[key].items():ck(key+'/'+rel,sha(ROOT/rel)==h)
 logo=read(MODELS/'CAL_LOGO_REPORT.json')
 for policy in ('ALL_BI_REPLACE','RGB_NONFALLBACK_PRESERVE'):
  with gzip.open(MODELS/policy/'CAL_ERRORS.jsonl.gz','rt')as stream:records=[json.loads(x)for x in stream if x.strip()]
  ck(policy+'/newrowpopulation',all(r['observation_generation']==FAMILY and r['raw_observation_generation']==RAW and r['noise_population']==POPULATION and r['selection_profile']==PROFILE and r['source_policy']==policy for r in records))
  ck(policy+'/inventory',len(records)==46 and len({r['id']for r in records})==46 and all(r['partition']=='CAL'for r in records)and sum(r['compatible']for r in records)==44)
  for kind,name in (('line','MODEL.json'),('tangent','TANGENT_MODEL.json')):model_check(policy,kind,read(MODELS/policy/name),records,None)
  folds=logo[policy]['LOGO_folds'];ck(policy+'/4LOGO',len(folds)==4 and {f['heldout_session']for f in folds}=={r['session']for r in records})
  for fold in folds:
   for kind in ('line','tangent'):model_check(policy,kind,fold[kind],records,fold['heldout_session'])
 failures=[x for x in CHECKS if not x['passed']]
 result=dict(schema='V31_INDEPENDENT_SEALED_CAL_ERROR_MODEL_NUMERIC_REPLAY_V1',UTC=datetime.now(timezone.utc).isoformat(),passed=not failures,checks=len(CHECKS),failure_count=len(failures),failures=failures,maximum_numeric_difference=MAXIMUM,source_sha256=sha(__file__),model_seal_sha256=sha(MODELS/'MODEL_SEAL.json'),audit_plan_sha256=sha(PLAN),raw_observation_generation=RAW,observation_family=FAMILY,noise_population=POPULATION,selection_profile=PROFILE,new_reference_decode=0,new_fit_or_image_NN_pose_calls=0,only_CAL_error_journals_decoded=True)
 dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=not failures,checks=len(CHECKS),failures=failures,maximum_numeric_difference=MAXIMUM,receipt_sha256=sha(dest))))
 if failures:raise SystemExit(1)
def plan(*,expected_central_plan_sha256=None):
 _central(expected_central_plan_sha256)
 if PLAN.exists():raise FileExistsError('Preserve every saved-error audit declaration')
 paths=[OLD_SCRIPT,OLD_RECEIPT,OUT.parent/'observed_anchor_native_ray_method_plan_v4_31/PREDECLARATION.json',OUT.parent/'CAL_geometry_guided_edge_noise_plan_v4_31/PLAN.json',OUT.parent/'CAL_geometry_guided_edge_noise_plan_v4_31/PLAN_SEAL.json']
 declaration=dict(central_plan_sha256=expected_central_plan_sha256,schema='V31_SAVED_CAL_ERROR_NUMERIC_AUDIT_PREDECLARATION_V1',UTC=datetime.now(timezone.utc).isoformat(),source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),str(OLD_SCRIPT.relative_to(ROOT)):sha(OLD_SCRIPT),str((OUT/'test_replay_saved_CAL_models_v1.py').relative_to(ROOT)):sha(OUT/'test_replay_saved_CAL_models_v1.py')},input_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},future_model_folder=str(MODELS.relative_to(ROOT)),raw_observation_generation=RAW,observation_family=FAMILY,noise_population=POPULATION,selection_profile=PROFILE,numeric_functions='balanced/evaluate/near/digest retain original V29 audit exact arithmetic; only path/version gates+rootGO+plan bindings added',scope='all full+4LOGO, 2source policies x line/tangent, every saved optimizer start NLL/gradient/equalweights/digest/floors/currentbindings',actual_V31_models_or_error_rows_decoded=0,new_labels_fit_RGB_NMS_or_native_pose_calls=0,explicit_future_root_GO_and_model_seal_required=True)
 PLAN.write_text(json.dumps(declaration,ensure_ascii=False,indent=2)+'\n')
 (OUT/'SAVED_ERROR_AUDIT_PLAN_SEAL_V1.json').write_text(json.dumps(dict(plan_sha256=sha(PLAN),source_sha256=declaration['source_sha256']),indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(PLAN),source_sha256=sha(__file__),actual_model_or_error_reads=0)))

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('plan','run'));parser.add_argument('--root-saved-error-audit-GO',action='store_true');parser.add_argument('--expected-model-seal-sha256');parser.add_argument('--central-plan-sha256',required=True);args=parser.parse_args()
 plan(expected_central_plan_sha256=args.central_plan_sha256)if args.stage=='plan'else main(root_saved_error_audit_GO=args.root_saved_error_audit_GO,expected_model_seal_sha256=args.expected_model_seal_sha256,expected_central_plan_sha256=args.central_plan_sha256)
