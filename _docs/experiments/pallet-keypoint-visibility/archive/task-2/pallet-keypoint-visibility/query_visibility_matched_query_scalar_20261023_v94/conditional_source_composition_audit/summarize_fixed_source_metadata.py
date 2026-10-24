import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'query_visibility_matched_query_scalar_20261023_v94';O=R/'conditional_source_composition_audit'
L=R/'conditional_source_fit_contract_preparation/CLEAN_ALL_SAME_ALL800_PROPOSAL_GT_ONLY.jsonl'
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
 assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def quantile(a,p):
 x=sorted(a);i=(len(x)-1)*p;lo=math.floor(i);hi=math.ceil(i)
 return x[lo]+(x[hi]-x[lo])*(i-lo)
def stats(a):return dict(N=len(a),min=min(a),median=quantile(a,.5),p90=quantile(a,.9),max=max(a)) if a else dict(N=0)

def main():
 write(O/'SCOPE_BEFORE_FIT_RESULTS.json',dict(created_UTC=datetime.now(timezone.utc).isoformat(),scope='Fixed source metadata only before any fit/VAL metric result read. No source tensors/pixels/model/current/results.',GT_ledger=bind(L),root_definition=bind(R/'ROOT_CANDIDATE1_DEFINITION_BEFORE_FIT.json'),root_admission=bind(R/'ROOT_C1_LABEL_ADMISSION_BEFORE_FIT.json'),new_gate_or_label_rule=False))
 allrows=[json.loads(x) for x in L.read_text().splitlines() if x.strip()];known=[r for r in allrows if r['candidate_known']]
 assert len(allrows)==800 and len(known)==157
 table=[]
 for s in ['TRAIN_LAB','VAL_HALL']:
  for view in ['before','after']:
   rr=[r for r in known if r['split']==s and r['view']==view]
   table.append(dict(split=s,view=view,D=sum(r['scalar_target_D0_E1_Uminus1']==0 for r in rr),E=sum(r['scalar_target_D0_E1_Uminus1']==1 for r in rr),query_N=len(rr),unique_frames=len({r['frame_input_id'] for r in rr}),unique_pairs=len({r['pair_id'] for r in rr})))
 slot_table=[]
 for k in range(8):
  for s in ['TRAIN_LAB','VAL_HALL']:
   rr=[r for r in known if r['native_slot']==k and r['split']==s]
   slot_table.append(dict(native_slot_metadata_only=k,split=s,D=sum(r['scalar_target_D0_E1_Uminus1']==0 for r in rr),E=sum(r['scalar_target_D0_E1_Uminus1']==1 for r in rr),N=len(rr)))
 incidence=Counter();unique_role_states=defaultdict(set);dist=[];norm=[];linf=[];q_in_unc=0;fullsetN=Counter();nullN=Counter()
 for r in known:
  q=r['original_q'];side=r['original_L'];fullsetN[r['known_FULL_certified_group_count']]+=1;nullN[r['null_unlocated_count']]+=1
  roles=set()
  for c in r['retained_FULL_GT_cues']:
   role=c['cue_id'].split(':')[-1];roles.add(role)
   unique_role_states[(r['split'],r['view'],r['scalar_target_D0_E1_Uminus1'],role)].add(c['cue_id'])
   ref=c['cue_reference_xy_unchanged'];bb=c['cue_uncertainty_bbox_unchanged'];dx=ref[0]-q[0];dy=ref[1]-q[1]
   dist.append(math.hypot(dx,dy));norm.append(math.hypot(dx,dy)/side);linf.append(max(abs(dx),abs(dy))/side)
   q_in_unc+=bb[0]<=q[0]<bb[2] and bb[1]<=q[1]<bb[3]
  for role in roles:incidence[(r['split'],r['view'],r['scalar_target_D0_E1_Uminus1'],role)]+=1
 pairs=defaultdict(dict)
 for r in allrows:pairs[r['pair_id']][(r['view'],r['native_slot'])]=(r['original_q'],r['original_L'])
 valid_same=[p for p,d in pairs.items() if all(d[('before',k)][0] is not None and d[('before',k)]==d[('after',k)] for k in range(8))]
 assert valid_same==[]
 after=[r for r in known if r['view']=='after'];after_byframe=defaultdict(set)
 for r in after:after_byframe[r['frame_input_id']].add(r['scalar_target_D0_E1_Uminus1'])
 report=dict(schema='V94_FIXED157_CONDITIONAL_SOURCE_SUPERVISION_COMPOSITION_KO',completed=True,PASS_metadata_counts=True,frozen_rule='CLEAN_ALL_SAME_CERTIFIED_GROUP_KNOWNSET',known157=157,all_original800_retained=True,
  split_view_D_E=table,unique_physical_objects=1,unique_known_pairs=len({r['pair_id'] for r in known}),unique_known_frames=len({r['frame_input_id'] for r in known}),all_source_pairs=50,all_source_frames=100,
  unique_known_pairs_by_split={s:len({r['pair_id'] for r in known if r['split']==s}) for s in ['TRAIN_LAB','VAL_HALL']},unique_known_frames_by_split={s:len({r['frame_input_id'] for r in known if r['split']==s}) for s in ['TRAIN_LAB','VAL_HALL']},
  native_slot_counts_metadata_NOT_model_input=slot_table,
  knownset_role_query_incidence=[dict(split=k[0],view=k[1],scalar_D0_E1=k[2],source_role=k[3],query_incidence=n,unique_cue_IDs=len(unique_role_states[k])) for k,n in sorted(incidence.items())],
  known_FULL_group_count_hist=dict(fullsetN),null_unlocated_count_hist_among_known=dict(nullN),known_rows_with_null=sum(r['null_unlocated_count']>0 for r in known),
  cue_reference_geometry_diag=dict(unit='Every original retained FULL cue-to-query incidence, including certified seed original IDs; not independent examples',reference_query_distance_native_px=stats(dist),reference_query_distance_over_L=stats(norm),reference_query_Linf_over_L=stats(linf),query_point_inside_cue_uncertainty_incidence_count=q_in_unc,geometry_does_not_define_native_binding=True),
  after_D_present=dict(D=sum(r['scalar_target_D0_E1_Uminus1']==0 for r in after),E=sum(r['scalar_target_D0_E1_Uminus1']==1 for r in after),frames_with_D=sum(0 in x for x in after_byframe.values()),frames_with_E=sum(1 in x for x in after_byframe.values()),same_after_frame_has_both_D_E=sum(x=={0,1} for x in after_byframe.values())),
  valid_natural_pairs_same_initial_q_L=0,valid_natural_pairs_same_initial_q_L_IDs=valid_same,controlledV3_included=False,
  practical_limits_KO=[
   'After에도 D 후보가 존재한다. 따라서 before/after 또는 cardboard 존재 자체를 E로 읽는 규칙과 cue-region visibility는 다르다. 그래도 query들이 소수 frame/pair와 같은 physical bin에 상관되어 있어 shortcut이 제거됐다는 증명은 아니다.',
   '유효한 natural before/after pair 중 모든 원래 q8/L이 같은 쌍은0이다. 같은 고정cue의 자연 상태를 관찰했지만 SAME-q controlled causal contrast가 아니다. 별도 controlledV3는 이번 C1 fit에 포함되지 않는다.',
   'Cue 역할은 view-relative wall/runner 등 기존 source cue ID이다. Native slot별 label 표는 metadata 진단이며 모델 forward는 P3/RGB/valid 세 tensor뿐이다. Native expectedpart UNKNOWN/humanfalse가 유지된다.',
   '다중 FULL known cues가 같은 상태인 conditioned known-set proxy다. Null/unlocated cue가 남으므로 complete crop truth나 native expectedcorner association을 인증하지 않는다. r은 query와 떨어질 수 있으며 q를 보정하거나 GTregion을 모델 readout으로 쓰지 않는다.',
   'One tall wooden pallet-bin, LAB/HALL environment holdout이다. Flat wooden/square pallet와 물체/시점/재질 분포가 다르며157query는157독립 object가 아니다. 새 gate나 label 변경을 추가하지 않았다.',
  ],bindings=dict(ledger=bind(L),scope=bind(O/'SCOPE_BEFORE_FIT_RESULTS.json')),source_metrics_or_fit_results_read=0,new_pixels_arrays_models_weights_current_forward_fit_PnP=0,new_gates_or_labels=0)
 write(O/'REVIEW_KO.json',report)
 write(O/'SEAL.json',dict(completed=True,PASS_metadata_counts=True,scope=bind(O/'SCOPE_BEFORE_FIT_RESULTS.json'),code=bind(Path(__file__)),review=bind(O/'REVIEW_KO.json'),original_modified=0,new_pixels_arrays_models_fit_current_PnP=0))
 print(json.dumps(dict(report=bind(O/'REVIEW_KO.json'),seal=bind(O/'SEAL.json'),split_view=table,after_D=report['after_D_present'],known_pairs=report['unique_known_pairs'],known_frames=report['unique_known_frames']),ensure_ascii=False))

if __name__=='__main__':main()
