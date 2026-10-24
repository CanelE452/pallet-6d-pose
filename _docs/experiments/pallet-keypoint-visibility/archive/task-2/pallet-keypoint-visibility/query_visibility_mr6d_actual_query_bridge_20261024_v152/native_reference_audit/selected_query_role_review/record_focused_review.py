"""Actual public frame6/8 same-order structural review; no new inference or source changes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections
H=Path(__file__).resolve().parent;V=H.parents[1]
def rd(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
root=V/'root_saved_query_role_diagnosis';result=rd(root/'RESULTS.json');assert result['completed']and result['full_query_inventory']==80
seal=rd(Path(result['prediction_seal']['path']));assert seal['completed']
rows=[json.loads(l)for l in(root/'ALL80_QUERY_IDENTITY_ROLE_AND_LOCAL_EVIDENCE.jsonl').read_text().splitlines()];rows=[x for x in rows if x['frame_index']in[6,8]];assert len(rows)==16
refs=[json.loads(l)for l in(V/'native_reference_audit/provided_reference_v1/ALL104_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl').read_text().splitlines()]
refmap={(x['frame_index'],x['gt_instance_index'],x['native_slot']):x for x in refs}
# Independent actual fullscene/query-board observations, not an automatic distance rule.
supported={6:{0,1,2,3,5},8:{0,1,2,3,4,5}}
why={
(6,0):'원 선택은 하나의 평판형 target pallet이며, 왼쪽 앞 deck 끝과 아래 지지 끝을 잇는 front 수직변이 보인다. native0은 그 위 끝 역할로 원 순서와 일치한다.',
(6,1):'오른쪽 앞 deck 끝의 위치 관계가 native0과 같은 앞면의 반대 위 끝이다. 명목 offset이 크더라도 노출된 front 구조와 고정 채널 역할은 지지되며 다른 slot으로 바꾸지 않는다.',
(6,2):'오른쪽 앞 위 끝 native1 아래에 같은 target의 끝 block/runner가 보이며 native2 하단 역할을 지지한다. EPAL block 픽셀만으로 역할을 정하지 않고 전체 앞면의 순서를 함께 확인했다.',
(6,3):'왼쪽 앞 아래 역할은 native0과 동일 끝의 하부 연장이고 원 native3 채널의 순서가 일관된다. 실제 q와 명목 참조 모두 하단 밖이므로 대응은 지지하되 입력 유효성·OOF는 그대로 둔다.',
(6,4):'rear top-left의 채널 이름은 알려져 있지만 실제 q는 배경·목재 경계 쪽이고 blue KLT와 pallet 뒤쪽 끝의 관계가 분명하지 않다. 다른 점으로 당기거나 U를 D/E로 바꾸지 않는다.',
(6,5):'target의 앞 오른쪽 위 끝에서 뒤쪽 오른쪽 위 deck 역할로 이어지는 원 순서가 유지된다. 회색 적재물이 그 rear top 영역을 덮는 구조가 source wholeRGB에서 지지된다. q가 회색 물체 위라는 사실만으로 E를 만든 것이 아니며, q의 위치 정확도는 미인증이다.',
(6,6):'native6는 형식상 rear bottom-right 채널이지만 crop에는 회색 적재물과 target deck가 섞인다. 자체 하부 구조와 외부 적재물 중 무엇이 기대 지지를 가리는지, 실제 q의 하부 역할 대응을 확정하지 못한다.',
(6,7):'native7의 crop은 target deck/blue cargo 주변이나, 뒤쪽 아래 runner 끝과 같은 실제 역할을 가리키는 증거가 부족하다. 목재 픽셀이 보인다는 이유로 D를 붙이지 않는다.',
(8,0):'선택 box는 흰 적재물을 받치는 단일 pallet을 포함한다. front-left deck 끝이 보이고 원 native0은 앞면 위/왼쪽 역할과 일치한다.',
(8,1):'앞 오른쪽 위 deck 끝은 적재물 바깥으로 드러나며 native0과 같은 앞면의 위 오른쪽 끝이다. 원 채널과 순서를 그대로 지지한다.',
(8,2):'앞 오른쪽 끝의 위 역할 native1 바로 아래에 끝 block/runner가 노출되어 native2를 지지한다.',
(8,3):'앞 왼쪽 하부 끝 구조는 실제 영상 하단에서 보인다. 원 q는 inframe이나 crop 절반이 padding이므로 위치/입력 경계 진단과 의미D를 분리한다.',
(8,4):'앞 왼쪽 위 native0에서 depth edge로 이어지는 왼쪽 rear deck 끝이 흰 적재물 왼쪽 바깥으로 노출된다. q의 중심이 배경 쪽이라는 이유로 다른 slot이나 E로 바꾸지 않는다. 실제 wholeRGB 구조는 같은 native4 의미D를 지지한다.',
(8,5):'회색 KLT 아래에 놓인 target의 오른쪽 rear deck 역할이다. front-right top1↔rear top5의 원 대응을 유지하며 wholeRGB에서 KLT가 기대 deck 끝을 덮는 외부 blocker로 확인된다. KLT 픽셀 소속만으로 E를 정의하지 않는다.',
(8,6):'q6는 KLT/흰 적재물 주변에 있으나 rear bottom-right 지지가 자체 deck에 숨는지 외부 적재물이 가리는지 분리되지 않는다. formal slot은 유지하고 실제 역할/상태는 U.',
(8,7):'q7 주변의 목재와 흰 적재물만으로 rear bottom-left runner 끝의 직접 보임을 확정할 수 없다. source nominal projection·convex proxy를 semanticD/S로 자동 복사하지 않는다.'}
local={
(6,0):'보이는 왼쪽 앞 deck/끝 목재와 주변 바닥', (6,1):'보이는 오른쪽 앞 deck 끝과 바닥', (6,2):'오른쪽 앞 block/runner 목재와 바닥',
(6,3):'원 q는 OOB; 저장 crop에는 상단 목재 일부와 아래 invalid padding', (6,4):'바닥·케이블/배경·목재 edge·blue cargo 일부',
(6,5):'회색 적재물 표면과 target deck 일부', (6,6):'회색 적재물 앞 표면과 deck 일부', (6,7):'목재 deck와 바닥·blue cargo 주변',
(8,0):'왼쪽 앞 deck/EPAL block 끝 목재', (8,1):'오른쪽 앞 deck 끝 목재', (8,2):'오른쪽 앞 EPAL block/runner 끝 목재',
(8,3):'하단 왼쪽 끝 목재와 crop invalid padding 약절반', (8,4):'중심은 배경 쪽이고 crop 하단/오른쪽에 실제 rear 목재 edge와 흰 적재물 경계',
(8,5):'회색 KLT 표면', (8,6):'회색 KLT/흰 적재물 경계', (8,7):'목재 edge·바닥·흰 적재물 경계'}
out=[]
for r in rows:
 fi=r['frame_index'];k=r['native_slot'];assert r['associated_entity_id_or_U']==0
 ref=refmap[(fi,0,k)];comparison=r['source_GT_native_comparisons'][0];assert comparison['reference_xy']==ref['reference_xy']
 status='SUPPORTED'if k in supported[fi]else'UNRESOLVED'
 role=r['associated_role_semantic_state']
 # Native convention is known in every row. SUPPORTED does not mean q is a pixel-correct corner.
 out.append({'global_index':r['global_index'],'frame_index':fi,'frame_id':r['frame_id'],'source_id':r['source_id'],'associated_GT_instance_index':0,'selected_index_unchanged':r['selected_index'],
 'native_slot':k,'native_name':ref['native_name'],'original_native_channel_contract_known':True,'same_native_index_comparison_only':True,
 'structural_role_correspondence':status,'structural_correspondence_scope':'Same selected target and original ordered role; not exact query-coordinate or model-point accuracy certification.',
 'correspondence_reason_KO':why[(fi,k)],'role_semantic_proposal':role,'role_label_unchanged_from_source_only_proposal':True,'human_gold':False,'proposal_tier':'SOURCE_ASSISTANT_WEAK_PARTIAL',
 'query_xy_unchanged':r['original_query'],'L_unchanged':r['original_L'],'original_continuous_valid':r['continuous_valid'],'original_nearest_valid':r['nearest_valid'],'crop_constructed':r['crop_constructed'],'crop_valid_fraction':r['crop_valid_fraction'],
 'query_local_visual_evidence_KO':local[(fi,k)],'literal_query_pixel_or_mask_is_not_semantic_label':True,
 'provided_nominal_reference_xy':ref['reference_xy'],'nominal_offset_px_diagnostic_only':comparison['nominal_error_px'],'physical_pixel_accuracy_certified':False,
 'nominal_pose_and_dimensions_uncertainty':'cm-scale, source supplied; neither <=3px gold nor a q-error filter.',
 'convex_self_visible_proxy_only':ref['convex_self_visible_proxy_only'],'semantic_S_not_autofilled':True,
 'visibility_supervision_candidate_if_root_adjudicated':status=='SUPPORTED'and role in['D','E'],'coordinate_accuracy_supervision_ready':False,'no_GT_nearest_permutation_shift_or_correction':True,'train_ready':False})
assert len(out)==16 and sum(x['structural_role_correspondence']=='SUPPORTED'for x in out)==11
assert sum(x['visibility_supervision_candidate_if_root_adjudicated']for x in out)==10
p=H/'FRAME6_8_ORDERED_ROLE_CORRESPONDENCE_AND_LOCAL_EVIDENCE.jsonl'
with p.open('x')as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
boards=[V/'original_query_evidence'/f'FRAME_{fi:02d}_ORIGINAL_Q_AND_CROPS.png'for fi in[6,8]]
overlays=[V/'native_reference_audit/provided_reference_v1'/n for n in['06_000002_000009_ALL_INSTANCES_NATIVE.png','08_000006_000007_ALL_INSTANCES_NATIVE.png']]
summary={'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),'review_code':bind(Path(__file__).resolve()),'prediction_seal':result['prediction_seal'],'root_saved_join':bind(root/'RESULTS.json'),'root_saved_inventory':bind(root/'ALL80_QUERY_IDENTITY_ROLE_AND_LOCAL_EVIDENCE.jsonl'),'focused_rows':bind(p),
 'actually_viewed_original_query_boards':[bind(x)for x in boards],'provided_reference_overlays_viewed_and_per_slot_refs_inspected':[bind(x)for x in overlays],
 'case_selection':'Root fixed request frame6 and8 after completed saved source outputs. No new model or additional sample selection.',
 'entity_correspondence':'Both selected firstargmax0 targets are the single physical pallet GTinstance0; no target reselection.',
 'formal_native_slot_contract_known_for_all16':True,
 'structurally_SUPPORTED':{'frame6':[0,1,2,3,5],'frame8':[0,1,2,3,4,5],'count':11},
 'structurally_UNRESOLVED':{'frame6':[4,6,7],'frame8':[6,7],'count':5},
 'source_role_proposals_unchanged':dict(collections.Counter(x['role_semantic_proposal']for x in out)),
 'partial_visibility_candidates_given_this_review':{'D':8,'E':2,'total':10,'input_valid_DE':10,'supported_OOF_but_original_input_U':1},
 'important_findings_KO':['frame6·8의 앞0..3은 같은 순서의 target 구조에 대응한다. frame6native3의 OOB는 대응을 없애거나 E로 바꾸는 이유가 아니다.','frame8 rear4는 실제 rear 목재 끝이 보이는 D 역할을 지지한다. 원 crop 중심이 배경을 보더라도 의미D와 q-local 품질은 다르다.','두frame rear5는 기존front1과의 top/depth 순서를 유지하며 외부 회색 적재물이 기대 역할을 덮는 E 후보다. frame6의 큰 명목 offset도 role을 다른slot으로 바꾸는 근거로 사용하지 않았다.','rear6/7은 두frame 모두 U를 유지한다. 배경/화물/목재 픽셀과 convex proxy만으로 D/E/S를 채우지 않았다.','11개 구조대응 지원은 전체16 q의 위치정확성이나 전체80의 학습 준비 완료를 뜻하지 않는다.'],
 'all80_and_all104_parent_inventories_unchanged':True,'label_changes':0,'human_gold':False,'train_ready':False,'no_model_fit_private_download_or_coordinate_correction':True}
write(H/'REVIEW_COMPLETE.json',summary);print(json.dumps({'review':bind(H/'REVIEW_COMPLETE.json'),'rows':bind(p),'supported':11,'unresolved':5,'DE_candidates':10}))
