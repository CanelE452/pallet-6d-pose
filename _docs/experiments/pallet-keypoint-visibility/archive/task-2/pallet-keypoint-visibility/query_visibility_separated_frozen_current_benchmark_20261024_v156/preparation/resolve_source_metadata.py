"""Resolve fixed source JSON descriptors; never hash/open model or current bodies."""
from pathlib import Path
import copy,hashlib,json
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
F=P/'query_visibility_independent_auxiliary_source_control_20261024_v155/future_current_preparation'
DRAFT=F/'FUTURE_PROTOCOL_DRAFT_NO_CURRENT_AUTHORIZATION.json'
CODE=F/'infer_future_four.py';CODE_SHA='8bc1caff2992f514f84c268c146aeca84b39242e2b657b4923ba516cfe9ad13d'
POLICY=V/'ROOT_DIAGNOSTIC_POLICY_BEFORE_CURRENT_BODIES.json'
RESOLVED=V/'RESOLVED_FOUR_HEAD_PROTOCOL.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check_json(b):
 assert Path(b['path']).suffix=='.json'and'_v112'not in b['path'].lower()
 assert bind(b['path'])==b;return read(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def resolve():
 assert sha(CODE)==CODE_SHA
 d=read(DRAFT);fixed=copy.deepcopy(d['models_fixed']);controls=[];audits=[]
 for cb,ab in zip(d['source_controls'],d['required_source_audits']):
  actual=bind(cb['path']);c=check_json(actual);audit=bind(ab['path']);a=check_json(audit)
  assert c['completed']and c['PASS']and c['selected_model']is None
  assert a['completed']and a['PASS']and a['control_complete']==actual and a['selected_model']is None
  controls.append(actual);audits.append(audit)
 for i,name in enumerate(d['models']):
  f=fixed[name];control=check_json(controls[int(i>=2)])
  fitbinding=control['arms'][f['arm']]['fit'];fit=check_json(fitbinding)
  freeze=check_json(fit['model_freeze']);res=check_json(fit['source_results'])
  assert fit['optimizer_updates']==f['fixed_update']==freeze['update']==res['update']
  assert freeze['parameters']==f['parameters']and freeze['threshold_logit']==0 and freeze['all_geometry_input_zero']
  assert freeze['selected_model']is None and not freeze['current_allowed']and not res['adopted']and not res['current_allowed']
  assert freeze['checkpoint']==res['checkpoint']and freeze['source_checks_all_pass']==res['prospective_source_checks_all_pass']
  for k,value in dict(source_fit=fitbinding,source_freeze=fit['model_freeze'],source_results=fit['source_results'],
    source_checkpoint=freeze['checkpoint'],source_state_sha=res['final_state'],source_checks_all_pass=freeze['source_checks_all_pass']).items():
   if i<2:assert f[k]==value
   f[k]=value
  assert res['arm']==f['arm']and res['auxiliary_weight']==f['auxiliary_weight']
  if i>=2:assert res['visibility_initialization']==f['visibility_initialization']
 return d,fixed,controls,audits
def main():
 d,fixed,controls,audits=resolve()
 declaration=V/'SOURCE_FOUR_FIXED_BINDINGS_BEFORE_PRIVATE_BODIES.json'
 if not declaration.exists():write(declaration,dict(completed=True,models=d['models'],models_fixed_before_current_body_reads=fixed,
  source_controls=controls,source_audits=audits,code=bind(CODE),draft_origin=bind(DRAFT),checkpoint_current_payload_body_or_hash_reads=0))
 else:assert read(declaration)['models_fixed_before_current_body_reads']==fixed
 if not POLICY.exists():
  print(json.dumps(dict(source_metadata_ready=True,declaration=bind(declaration),awaiting_root_policy=str(POLICY),actual_private_payload_hash_or_body_reads=0)));return
 policy=read(POLICY)
 assert policy['phase']=='V156'and policy['approved_current_frozen_diagnostic']and policy['models_fixed_before_current_body_reads']==fixed
 assert policy['benchmark_only_not_admitted']and policy['source_admission_unchanged']and policy['selected_model']is None
 p=dict(ready_for_actual_execution=True,prepared_only=True,code=bind(CODE),draft_origin=bind(DRAFT),future_phase_policy=bind(POLICY),
   models=d['models'],models_fixed=fixed,original=d['original'],source_controls=controls,source_audits=audits,
   mandatory_order=d['mandatory_order'],future_primary_cohort=d['future_primary_cohort'],future_eval_models=d['future_eval_models'],
   future_eval_contract=d['future_eval_contract'],future_quality_eval=d['future_quality_eval'],no_compaction_or_chunking=True,
   private_payload_body_or_hash_reads_during_preparation=0,model_instantiations_or_calls_during_preparation=0,selected_model=None)
 write(RESOLVED,p)
 template=read(F/'FUTURE_ROOT_ENTRY_TEMPLATE_NOT_AUTHORIZED.json')
 template.update(future_phase_policy=bind(POLICY),resolved_protocol=bind(RESOLVED),output_directory=str(V/'frozen_four_head_current_benchmark_v1'))
 write(V/'ROOT_FROZEN_FOUR_HEAD_ENTRY_TEMPLATE.json',template)
 receipt=dict(completed=True,prepared_only=True,code=bind(CODE),resolver=bind(__file__),draft_origin=bind(DRAFT),source_declarations=bind(declaration),
  policy=bind(POLICY),protocol=bind(RESOLVED),template=bind(V/'ROOT_FROZEN_FOUR_HEAD_ENTRY_TEMPLATE.json'),models=d['models'],
  all_four_source_freezes_preserved_without_admission_or_choice=True,checkpoint_or_private_payload_body_hash_reads=0,actual_models_instantiated_or_forwarded=0,
  command=d['original']['runtime']['executable']+' '+str(CODE)+' --entry '+str(V/'ROOT_APPROVED_FROZEN_FOUR_HEAD_ENTRY.json'))
 write(V/'PREPARATION_READY.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
