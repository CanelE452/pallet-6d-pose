"""Postseal descriptive session/outlier/factorial analysis, no new fit."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 assert not p.exists();p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')

seal=json.loads((HERE/'PREDICTION_SEAL.json').read_text());assert sha(HERE/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
rs=[json.loads(s)for s in gzip.open(HERE/'SCORED_ROWS.jsonl.gz','rt')];metrics=('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg');groups=defaultdict(dict)
for r in rs:groups[r['arm']][r['id']]=r
CURRENT='CURRENT_CORRECTED_BI_OT';current=groups[CURRENT];output={};reference={}
for arm,g in groups.items():
 output[arm]={};reference[arm]={}
 for part in ('CAL','TEST'):
  selected=[r for r in g.values()if r['partition']==part];d=[]
  for r in selected:d.append(dict(id=r['id'],session=r['session'],compatible=r['compatible'],accepted=r['accepted'],pose_source=r['actual_pose'].get('source'),delta_arm_minus_current={k:float(r['metrics'][k])-float(current[r['id']]['metrics'][k])for k in metrics},metrics=r['metrics']))
  sessions=[]
  for s in sorted({r['session']for r in selected}):
   ss=[r for r in d if r['session']==s];sessions.append(dict(session=s,n=len(ss),delta_mean={k:float(np.mean([r['delta_arm_minus_current'][k]for r in ss]))for k in metrics},translation_sum_delta_cm=sum(r['delta_arm_minus_current']['translation_cm']for r in ss)))
  output[arm][part]=dict(session_contributions=sessions,largest_translation_wins=sorted(d,key=lambda r:r['delta_arm_minus_current']['translation_cm'])[:10],largest_translation_losses=sorted(d,key=lambda r:r['delta_arm_minus_current']['translation_cm'],reverse=True)[:10],status_counts=dict(Counter(r.get('solution',{}).get('status','CACHED_SELECTOR_OR_ORIGINAL')if r.get('solution')else'CACHED_SELECTOR_OR_ORIGINAL'for r in selected)),accepted_joint_counts=sum(r['accepted']for r in selected),material_ambiguity_counts=sum(r['material_ambiguity']for r in selected))
  audited=[r for r in selected if 'reference_constraint_audit'in r]
  if audited:
   reference[arm][part]=dict(all_physical_symmetry_reference_feasible=sum(r['reference_constraint_audit']['any_representative_feasible']for r in audited),compatible_fixed_native_reference_feasible=sum(bool(r['reference_constraint_audit']['fixed_native_reference_feasible'])for r in audited),compatible_n=sum(r['compatible']for r in audited),reference_exclusion_by_fixed_native_phase={family:sum(bool(next(a for a in r['reference_constraint_audit']['allowed_physical_representatives']if a['matches_fixed_native_reference_phase'])[family])for r in audited if r['compatible'])for family in ('failed_BI_mahalanobis_ids','failed_BI_geometry_ids','failed_BI_canvas_ids','failed_OT_ids')},ideal_consistency=dict(Counter({k:sum(r.get('ideal_model_consistency',{}).get(k,False)for r in audited)for k in ('construction_reference_constraints_feasible','construction_reference_retained_in_found_samples','accepted_ambiguous_is_possible')}))if arm=='MODEL_IDEAL_PROJECTIONS'else None)
# OT x correction interaction is a difference of paired deltas, with same held Sigma.
interaction={}
for part in ('CAL','TEST'):
 selected=[i for i,r in current.items()if r['partition']==part];interaction[part]={}
 for k in metrics:
  corrected_OT=np.array([float(current[i]['metrics'][k])-float(groups['CURRENT_CORRECTED_BI_NO_OT'][i]['metrics'][k])for i in selected]);raw_OT=np.array([float(groups['RAW_BI_OT'][i]['metrics'][k])-float(groups['RAW_BI_NO_OT'][i]['metrics'][k])for i in selected]);diff=corrected_OT-raw_OT
  interaction[part][k]=dict(corrected_OT_minus_noOT_mean=float(corrected_OT.mean()),raw_OT_minus_noOT_mean=float(raw_OT.mean()),correction_by_OT_interaction_mean=float(diff.mean()),per_frame=[dict(id=i,value=float(v))for i,v in zip(selected,diff)])
result=dict(schema='POSTSEAL_DIAGNOSTIC_SESSION_OUTLIER_ANALYSIS_V1',production_or_generalization_claim=False,all95_preserved=True,per_arm=output,reference_feasibility=reference,held_covariance_correction_OT_factorial=interaction,scored_rows_sha256=sha(HERE/'SCORED_ROWS.jsonl.gz'),prediction_seal_sha256=sha(HERE/'PREDICTION_SEAL.json'),script_sha256=sha(__file__))
write(HERE/'POSTSEAL_DIAGNOSTIC_ANALYSIS.json',result)
summary=json.loads((HERE/'POSTSEAL_SCORING.json').read_text())['summary'];table=['| 진단 조건, TEST49 | 위치 cm | 회전 ° | 성공 /49 | 공동 채택 | 모호성 |','|---|---:|---:|---:|---:|---:|']
for arm,value in summary.items():
 s=value['TEST'];table.append(f"| {arm} | {s['metrics']['translation_cm']['mean']:.4f} | {s['metrics']['rotation_deg']['mean']:.4f} | {s['success_count']} | {s['accepted']} | {s['material_ambiguity']} |")
text='''# 봉인 입력의 통제된 사후 원인 진단\n\n이 결과는 이미 본 CAL46/TEST49에서의 posthoc intervention이며 새 production 방법이나 일반화 성능 비교가 아니다. 기존 실제3조건은 그대로 보존했다. 새11조건과 cached2selector에 대해 source/input/output을 바인딩·봉인하고, 그 뒤 동일 physical pose metric을 집계했다. Oracle에는 human state 또는 역사적 derived reference가 입력된다는 점을 명시한다. 원RGB·마스크·head를 다시 추론하지 않았다.\n\n'''+ '\n'.join(table)+'''\n\nRAW_BI_OT/NO_OT는 q만 바꾸고 기존 corrected covariance를 그대로 둔 인위적 xy 통제다. CURRENT_RAW_STANDARD_COV는 별도 B2px/I3px control이며 binary/full-human oracle은 같은 covariance를 쓴다. FULL_HUMAN_BUDGET6는 compatible86에서만 budget을6으로 바꾸고 비호환9는 정확히 standard3 control을 유지한다. 가중 objective/Sigma와 hard budget은 다른 조작이다.\n\nNative reference 대응이 호환되는 CAL44/TEST42=86을 pixel oracle 해석 범위로 제한했다. 비호환9에서 state/center/drop oracle은 control을 유지한다. MODEL_IDEAL9는 physical phase0에서 현재 native 모델을 자체 투영한 virtual positive control이며 human E를 적용하지 않는다. 이 모델 투영을 실제 영상 관측이나 정확한 physical visibility로 표현하지 않는다.\n\nReference-init은 objective/feasible set을 바꾸지 않지만 탐색과 BI seed fallback을 바꿀 수 있다. 각 phase의 BI Maha/geometry/canvas/depth와 개별O/T 제약, fixed native reference의 가능성과 any allowed physical-symmetry reference의 가능성을 분리했다. MODEL_IDEAL의 construction pose가 가능했는지·found samples에 남았는지·최종 선택이 모호한지는 별도 항목이다. 모호한 경우 모든 pose error가0이어야 한다고 요구하지 않는다.\n\nSELECT_CLOSEST_COARSE_FIXED는 기존 feasible sample의 all8 모델 투영과 초기 모델 투영 사이 RMS/초기bboxdiag를 최소화한다. S raw나GT를 쓰지 않는다. ORACLE_BEST_FOUND는 기존 pool에서 translation error 최소 후보 하나를 선택하고 그 pose의 모든 metric을 보고한다. 각metric의 attainable floor는 별도로 제공하며 서로 다른 후보의 최소값을 가짜 한 pose로 합치지 않는다. 새 전역 search의 완전성이나 production selector 우수성은 증명하지 않는다.\n\nPOSTSEAL_SCORING.json에는 전체95 및 compatible 부분집합·짝비교·성공flip·session-cluster bootstrap가 있다. POSTSEAL_DIAGNOSTIC_ANALYSIS.json에는 세션별 기여, 큰 손실/개선, 상태/fallback/모호성, reference exclusion과 q보정×OT 상호작용이 있다. PREDICTIONS에는 모든 선언 입력, pose, 원인별 제약 감사와 provenance를 보존했고 source/real predictions/보호 사용자파일은 유지했다.\n'''
p=HERE/'README_KO.md';assert not p.exists();p.write_text(text)
