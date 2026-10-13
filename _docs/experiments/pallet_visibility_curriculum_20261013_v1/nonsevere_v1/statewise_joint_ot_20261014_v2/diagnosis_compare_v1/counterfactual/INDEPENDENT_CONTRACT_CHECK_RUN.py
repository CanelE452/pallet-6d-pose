"""Independent result audit of the sealed posthoc counterfactual contracts."""
from pathlib import Path
from collections import Counter
import json,gzip,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'scripts/research/pallet_visibility_curriculum_20261013_v1/scoring.py').is_file())
N=HERE.parent.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s)for s in gzip.open(p,'rt')]
def same(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'))==json.dumps(b,sort_keys=True,separators=(',',':'))

protocol=json.loads((HERE/'PROTOCOL.json').read_text());seal=json.loads((HERE/'PREDICTION_SEAL.json').read_text())
assert sha(HERE/'PROTOCOL.json')==seal['protocol_sha256']and sha(HERE/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
for p,e in protocol['source_sha256'].items():assert sha(ROOT/p)==e
for p,e in protocol['evidence_sha256'].items():assert sha(ROOT/p)==e
for p,e in protocol['protected_user_sha256'].items():assert sha(ROOT/p)==e
actual=rows(HERE/'PREDICTIONS.jsonl.gz');cached={r['id']:r for r in rows(N/'evaluation_v3/PREDICTIONS.jsonl.gz')if r['cohort']=='PRIMARY_NONSEVERE'}
groups={a:{r['id']:r for r in actual if r['arm']==a}for a in set(r['arm']for r in actual)}
assert len(groups)==16 and all(len(v)==95 and set(v)==set(cached)for v in groups.values())
checks=Counter();fallbacks=Counter();constraints=Counter();ideal=Counter()
for i,row in cached.items():
 rt=row['runtime'];raw=rt['original_raw_points'];frozen=rt['frozen_visible_measurements'];domains=rt['ot_domains']['domains']
 for a,pose in [('ORIGINAL_PNP',row['initial_pose']),('CURRENT_CORRECTED_BI_NO_OT',row['BI_only']['actual_pose']),('CURRENT_CORRECTED_BI_OT',rt['actual_pose'])]:
  assert same(groups[a][i]['actual_pose'],pose);checks['cached_real_pose_exact']+=1
 for arm in ('RAW_BI_OT','RAW_BI_NO_OT'):
  value=groups[arm][i]['solver_input'];expect=json.loads(json.dumps(frozen))
  for m in expect:m['xy']=raw[m['native_corner']]
  assert same(value['visible'],expect);checks['raw_xy_only_same_covariance']+=1
  if arm=='RAW_BI_OT':assert same(value['states'],rt['states'])and same(value['domains'],domains)
  else:assert value['domains']==[]and value['states']==['U'if s in('O','T')else s for s in rt['states']]
 for arm in ('CURRENT_RAW_STANDARD_COV','ORACLE_BINARY_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV_BUDGET6'):
  value=groups[arm][i]['solver_input']
  for m in value['visible']:
   variance=4. if m['state']=='B'else 9.;assert np.array_equal(m['covariance_px2'],[[variance,0.],[0.,variance]])and m['xy']==raw[m['native_corner']]
  checks['matched_raw_standard_covariance']+=1
 full=groups['FULL_HUMAN_RAW_STANDARD_COV'][i]['solver_input'];interaction=groups['FULL_HUMAN_RAW_STANDARD_COV_BUDGET6'][i]['solver_input'];assert same(full['visible'],interaction['visible'])and same(full['states'],interaction['states'])and same(full['domains'],interaction['domains'])
 if groups['FULL_HUMAN_RAW_STANDARD_COV'][i]['compatible']:assert interaction['config']['maximum_visible_mahalanobis']==6.
 else:assert same(full,interaction)
 checks['state_budget_interaction_preserves_matched_inputs']+=1
 drop=groups['CURRENT_DROP_KNOWN_FALSE_OT'][i]['solver_input'];assert same(drop['visible'],frozen)
 for d in drop['domains']:assert any(same(d,old)for old in domains)
 for arm in ('CURRENT_BI_BUDGET6','CURRENT_REFERENCE_INIT'):
  value=groups[arm][i]['solver_input'];assert same(value['visible'],frozen)and same(value['domains'],domains)and same(value['states'],rt['states']);checks['budget_or_starts_changes_only_declared_quantity']+=1
 center=groups['CURRENT_REFERENCE_MODEL_CENTER_O'][i]['solver_input'];assert same(center['visible'],frozen)and same(center['states'],rt['states'])
 for new,old in zip(center['domains'],domains):
  if old['state']=='T':assert same(new,old)
  else:assert new['half_axes_px']==old['half_axes_px']and new['shape']==old['shape']and new['halfplanes']==old['halfplanes']
 checks['reference_center_preserves_width_and_T']+=1
 for arm in [x['name']for x in protocol['plan_exact']['diagnostic_arms']]:
  r=groups[arm][i];s=r['solution'];constraints['diagnostic_rows']+=1
  if s['accepted']:
   assert all(d['feasible']for d in s['joint_constraint_audit'])and all(m['visible_support_guards_pass']for m in s['visible_protection_audit']);constraints['accepted_all_guards_pass']+=1
  else:
   seed=s.get('corrected_BI_seed')or{}
   if r['actual_pose']['source']=='EXPLICIT_VALID_BI_SEED_FALLBACK':assert seed.get('visible_support_guards_pass')and not seed.get('contains_underconstrained_pose_family');fallbacks['valid_identified_BI_seed']+=1
   else:assert np.array_equal(r['actual_pose']['R_cf'],row['initial_pose']['R_cf'])and np.array_equal(r['actual_pose']['centroid'],row['initial_pose']['centroid']);fallbacks['explicit_original_coarse']+=1
  if len(r['solver_input']['visible'])==2 and s['accepted']:assert r['material_ambiguity'];constraints['BI2_forced_family']+=1
  if arm=='MODEL_IDEAL_PROJECTIONS':
   assert r['ideal_model_consistency']['construction_reference_constraints_feasible'];ideal['construction_feasible']+=1
   ideal['accepted']+=s['accepted'];ideal['ambiguous']+=r['material_ambiguity'];ideal['construction_found']+=r['ideal_model_consistency']['construction_reference_retained_in_found_samples']
scoring=json.loads((HERE/'POSTSEAL_SCORING.json').read_text());scored=rows(HERE/'SCORED_ROWS.jsonl.gz')
for arm in groups:
 for part,n in [('CAL',46),('TEST',49)]:
  rr=[r for r in scored if r['arm']==arm and r['partition']==part];assert len(rr)==n
  for metric in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
   value=sum(float(r['metrics'][metric])for r in rr)/len(rr);assert abs(value-scoring['summary'][arm][part]['metrics'][metric]['mean'])<1e-12
  checks['independent_aggregate_means']+=1
result=dict(schema='INDEPENDENT_COUNTERFACTUAL_CONTRACT_AUDIT_V1',rows=len(actual),arms=len(groups),checks=dict(checks),accepted_constraints=dict(constraints),fallbacks=dict(fallbacks),ideal_model=dict(ideal),current19_and_new_source_bindings_unchanged=True,primary95_inputs_and_predictions_unchanged=True,protected_six_user_files_unchanged=True,oracle_results_not_production_benchmark=True,script_sha256=sha(__file__))
p=HERE/'INDEPENDENT_CONTRACT_CHECK.json'
assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
