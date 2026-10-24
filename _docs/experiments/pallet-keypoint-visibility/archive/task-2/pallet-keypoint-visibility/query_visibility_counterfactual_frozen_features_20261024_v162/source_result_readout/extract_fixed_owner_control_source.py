"""Two fixed V162 arms; exact pinned V160 pure readout helpers, no model code.

Prepare opens code/protocol metadata only. Extract requires root both-frozen
completion message and final CONTROL_COMPLETE. No feature/weight inputs.
"""
from pathlib import Path
import argparse, ast, hashlib, json
from datetime import datetime, timezone
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
HELPER=P/'query_visibility_native_slot_owner_factorial_20261024_v160/factorial_source_readout/extract_fixed_four_arm_source.py'
HELPER_SHA='b4bc6df33ca8130ca5a0b6ccf9b1ace35b55d0528c1dc22d460c7607fed77465'
CONTRACT=P/'query_visibility_native_slot_owner_factorial_20261024_v160/input_contract_audit/INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json'
OLD_PROTOCOL=P/'query_visibility_native_slot_owner_factorial_20261024_v160/source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json'
RUN=V/'runs/C43_C44_OWNER_CF_FIXED8000'
ARMS=('C43_REPLAY_C41_OWNER253_NATIVE0','C44_OWNER567_WITH_CF314_NATIVE0')
FAMILIES=('OLD6144','NEW20480','PALLOX512','MR1840');COUNTS=(6144,20480,512,1840)
AUDIT_READS=[]
NAMES=('safe','sha','bind','read','check','rows','write','ratio','metric','paired_metric',
       'normalized_existing','same_metric','metadata','masks','output_prediction')
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
tree=ast.parse(HELPER.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in NAMES]
assert {n.name for n in nodes}==set(NAMES)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HELPER)+'::pinned_pure_helpers','exec'),globals())


def difference(flat):
 assert set(flat[ARMS[0]])==set(flat[ARMS[1]])
 out={}
 for name in sorted(flat[ARMS[0]]):
  before,after=[flat[a][name]for a in ARMS]
  for key in ('all_original_rows','known_D','known_E','unknown_truth','factual_input_U_all'):
   assert before[key]==after[key],(name,key)
  out[name]=dict(CM_signed_count_change=(np.asarray(after['CM_D_E_rows_D_E_U_columns'])-
    np.asarray(before['CM_D_E_rows_D_E_U_columns'])).tolist(),
    rate_percentage_point_change={k:None if before[k]is None or after[k]is None else 100*(after[k]-before[k])
     for k in ('D_precision','D_recall','E_precision','E_recall','prediction_U_rate_known')})
 return out


def prepare():
 original=read(OLD_PROTOCOL);contract=read(CONTRACT)
 assert original['updates_each']==8000 and contract['source_rows_total']==28976
 plan=dict(completed=True,prepared_only=True,code=bind(__file__),pinned_V160_pure_helpers=bind(HELPER),
  exact_AST_functions=list(NAMES),arms=list(ARMS),families=dict(zip(FAMILIES,COUNTS)),
  source_rows_each=28976,factual_input_U_each=535,threshold_logit=0,updates_each=8000,
  no_generated_result_bodies_read=True,completion_barrier='Root both-frozen/all2 completion message plus final CONTROL_COMPLETE before any new result read.',
  outputs=['ARM_DATASET_SUMMARY.json','FULL_NATIVE_MATERIAL_ASPECT_TABLES.json','C44_MINUS_C43_DIAGNOSTIC.json',
   'ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json','READOUT_COMPLETE.json'],
  schema='Same V160 arms[arm][family][fold].all_original_rows/input_valid_diagnostic plus gates; all source and native/material/aspect/cross groups.',
  denominators='Full original known D/E includes factual U. Unknown truth separate. MR held154=D102/E52 with9D factualU is headline, input-valid145=D93/E52 separate.',
  separate_channels='Native35 and original Pallox/MR full gates, entity/quality, owner253 and CF314 weakTRAIN diagnostics; CF no primary/native/quality truth.',
  changes='C44 minus C43 fixed signed counts and rate percentage points only; no factorial/selection/threshold tuning.',
  selected_model=None,fully_solved=False,training_admitted_by_readout=False)
 write('OUTPUT_PLAN_BEFORE_RESULT_BODIES.json',plan)
 write('PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),
  plan=bind(H/'OUTPUT_PLAN_BEFORE_RESULT_BODIES.json'),pinned_helpers=bind(HELPER),
  no_new_results_features_weights_or_source_NPZ_read=True,model_or_fit_calls=0))
 print(json.dumps(dict(prepared=True,plan=bind(H/'OUTPUT_PLAN_BEFORE_RESULT_BODIES.json'))))


def extract(note):
 assert note and len(note.strip())>=20,'Root both-frozen completion message required'
 prep=read(H/'PREPARATION_COMPLETE.json');check(prep['code']);check(prep['pinned_helpers'])
 control=read(RUN/'CONTROL_COMPLETE.json')
 assert control['completed']and control['PASS']and control['updates_each']==8000
 assert set(control['arms'])==set(ARMS)and control['selected_model']is None
 assert control['source_only']and control['all_original_native26624_public512_MR1840_retained']
 assert control['all_CF3816_preserved']and control['CF_inputU']==360
 frozen=read(check(control['all2_frozen_before_source_comparison']))
 assert frozen['completed']and frozen['arms']==list(ARMS)and frozen['source_validation_reports_emitted']is False
 replay=read(check(control['baseline_C41_exact_replay']));assert replay['completed']and replay['PASS']
 protocol=read(check(control['protocol']));contract=read(CONTRACT);meta=metadata(protocol,contract)
 summary={};tables={};aux={};flat={};bindings=[]
 for arm in ARMS:
  fit=read(check(control['arms'][arm]['fit']));assert fit['completed']and fit['optimizer_updates']==8000
  report=read(check(fit['source_results']));freeze=read(check(fit['model_freeze']))
  assert report['completed']and report['arm']==arm and report['update']==freeze['update']==8000
  assert freeze['threshold_logit']==0 and not report['native_slot_hint']and report['owner_auxiliary_weight']==1
  assert report['selected_model']is None and not report['current_allowed']
  summary[arm]={};tables[arm]={};flat[arm]={};joined={k:[]for k in ('y','state','av','pred')}
  for family in FAMILIES:
   m=meta[family];output=RUN/arm/f'FINAL_{family}_OUTPUTS.npz';pred=output_prediction(output,m);bindings.append(bind(output))
   table={name:paired_metric(m['y'],pred,m['av'],mask,m['state'])for name,mask in masks(m).items()}
   tables[arm][family]=table
   summary[arm][family]={fold:table[fold]for fold in ['ALL',*sorted(set(m['split']))]}
   for name,pair in table.items():
    for conditioning,values in pair.items():flat[arm][f'{family}::{name}::{conditioning}']=values
   for k in ('y','state','av'):joined[k].append(m[k])
   joined['pred'].append(pred)
   for fold,pair in summary[arm][family].items():
    if family in FAMILIES[:2]:same_metric(pair['all_original_rows'],report['native']['old_native'if family==FAMILIES[0]else'new_native'][fold])
    elif family=='PALLOX512':
     same_metric(pair['all_original_rows'],report['Pallox']['groups'][fold])
     same_metric(pair['input_valid_diagnostic'],report['Pallox']['groups'][fold+'::INPUT_VALID'])
    else:
     same_metric(pair['all_original_rows'],report['MR'][fold]['visibility'],binary=True)
     same_metric(pair['input_valid_diagnostic'],report['MR'][fold]['visibility_inputvalid'],binary=True)
  j={k:np.concatenate(v)for k,v in joined.items()}
  allsource=paired_metric(j['y'],j['pred'],j['av'],np.ones(28976,bool),j['state'])
  assert allsource['all_original_rows']['factual_input_U_all']==535
  summary[arm]['ALL_SOURCE28976']=dict(ALL=allsource)
  for conditioning,values in allsource.items():flat[arm][f'ALL_SOURCE28976::ALL::{conditioning}']=values
  held=summary[arm]['MR1840']['SOURCE_VALIDATION']
  assert [held['all_original_rows'][k]for k in ('known_D','known_E','factual_input_U_known')]==[102,52,9]
  assert [held['input_valid_diagnostic'][k]for k in ('known_D','known_E')]==[93,52]
  native=report['native'];assert len(native['gates'])==35
  assert sum(g['PASS']for g in native['gates'].values())==native['native35_passed']
  vis=report['primary_visibility_component']['MR_held_visibility']
  component=bool(native['native35_all_pass']and report['Pallox']['sanity']['PASS']and
   vis['negative_recall']is not None and vis['negative_recall']>=.90 and vis['positive_recall']is not None and vis['positive_recall']>=.80)
  gates=dict(native35_passed=native['native35_passed'],native35_total=35,native35_all_pass=native['native35_all_pass'],
   primary_visibility_component=report['primary_visibility_component'],primary_visibility_component_diagnostic_PASS=component,
   MR_full_visibility_and_entity_weak_sanity_PASS=report['MR_weak_sanity_PASS'],
   original_full_prospective_source_checks_all_pass=report['prospective_source_checks_all_pass'],
   original_AUX29_not_retested=True,original64_qualification_not_claimed=True,fully_solved=False,adopted=False,current_allowed=False,selected_model=None)
  summary[arm]['gates']=gates
  owner=report['owner_diagnostic'];cf=report['CF_owner_diagnostic']
  cmkey='CM_rows_truth_T_F_BG_columns_prediction_T_F_BG'
  cmo=np.asarray(owner[cmkey]);cmcf=np.asarray(cf[cmkey])
  assert cmo.sum()==253 and cmcf.sum()==314
  assert cmo.sum(1).tolist()==[53,52,148]and cmcf.sum(1).tolist()==[157,157,0]
  aux[arm]=dict(original_native35_gates=native['gates'],gate_summary=gates,
   Pallox_original_sanity=report['Pallox']['sanity'],MR_entity_and_secondary_gate_diagnostics=report['MR'],
   synthetic_point_quality_diagnostic=report['synthetic_point_quality'],
   owner_weak_TRAIN253_diagnostic=owner,CF_owner_weak_TRAIN314_diagnostic=cf,
   combined567_weak_TRAIN_diagnostic=dict(CM_rows_truth_T_F_BG_columns_prediction_T_F_BG=(cmo+cmcf).tolist(),
    accuracy=float(np.trace(cmo+cmcf)/567),not_held_or_primary_success=True),
   owner_rows_used=report['owner_total_known'],CF_owner_rows_used=report['CF_owner_rows_used'],
   CF3816_all_new_native_visibility_and_quality_UNKNOWN=True,CF_not_primary_or_quality_loss=True)
 header=dict(schema='V162_FIXED_TWO_CONDITION_SOURCE_READOUT',source_only=True,selected_model=None,fully_solved=False,
  adopted=False,current_allowed=False,threshold_logit=0,CM_order=dict(rows=['D','E'],columns=['D','E','U']),
  truth='Existing unchanged CF_PRIMARY_DECLARED synthetic native role and public weak role labels; owner253/CF314 separate weakTRAIN diagnostic.',
  factual_U='Original input fact; known factualU stays in CM/recall; unknown truth separate. CF inputU360 is separate from original535.',
  rates='Fractions; precision among known truth predictions of that class. Changes in percentage points; no arm selection.')
 write('ARM_DATASET_SUMMARY.json',dict(**header,arms=summary))
 write('FULL_NATIVE_MATERIAL_ASPECT_TABLES.json',dict(**header,arms=tables))
 write('C44_MINUS_C43_DIAGNOSTIC.json',dict(**header,diagnostic_only=True,formula='C44 minus C43',groups=difference(flat)))
 write('ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json',dict(**header,arms=aux,baseline_C41_exact_replay=replay))
 names=('ARM_DATASET_SUMMARY.json','FULL_NATIVE_MATERIAL_ASPECT_TABLES.json','C44_MINUS_C43_DIAGNOSTIC.json','ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json')
 write('READOUT_COMPLETE.json',dict(completed=True,PASS=True,UTC=datetime.now(timezone.utc).isoformat(),root_completion_message=note,
  code=bind(__file__),pinned_helpers=bind(HELPER),preparation=bind(H/'PREPARATION_COMPLETE.json'),control=bind(RUN/'CONTROL_COMPLETE.json'),
  final_protocol=control['protocol'],all2_frozen=control['all2_frozen_before_source_comparison'],
  baseline_C41_exact_replay=control['baseline_C41_exact_replay'],outputs=[bind(H/n)for n in names],
  saved_final_output_bindings=bindings,exact_reads=AUDIT_READS,
  original28976_U535_MRheld154_including9U_and145_inputvalid_preserved=True,
  source_features_checkpoint_weights_RGB_or_encoder_model_fit_calls=0,selected_model=None,fully_solved=False))
 print(json.dumps(dict(completed=True,summary=bind(H/'ARM_DATASET_SUMMARY.json'),receipt=bind(H/'READOUT_COMPLETE.json'))))


if __name__=='__main__':
 ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True)
 mode.add_argument('--prepare',action='store_true');mode.add_argument('--extract',action='store_true');ap.add_argument('--root-both-completion-note');a=ap.parse_args()
 if a.prepare:
  assert a.root_both_completion_note is None;prepare()
 else:extract(a.root_both_completion_note)
