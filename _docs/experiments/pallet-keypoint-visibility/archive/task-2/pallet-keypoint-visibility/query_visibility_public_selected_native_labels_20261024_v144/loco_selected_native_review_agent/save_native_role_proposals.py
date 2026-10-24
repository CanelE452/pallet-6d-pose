"""Independent assistant visual proposals on all fixed actual LOCO native slots."""
from pathlib import Path
from collections import Counter
import hashlib,json
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
facts=json.loads((H/'fixed_source96_contact_v1/CROP_FACTS96.json').read_text())
assert facts['completed']and len(facts['rows'])==96 and facts['all_official_instances']==424
roles=['FRONT_TOP_LEFT','FRONT_TOP_RIGHT','FRONT_BOTTOM_RIGHT','FRONT_BOTTOM_LEFT','REAR_TOP_LEFT','REAR_TOP_RIGHT','REAR_BOTTOM_RIGHT','REAR_BOTTOM_LEFT']
frames={
0:dict(target_annotation_id=None,identity='UNIDENTIFIED_PALLET_OR_RACK_REGION',related=[],
 why='첫 argmax는 흐릿한 파란 철제 경계/선반 하부의 화면 아래 영역에 있습니다. 팔레트 하나의 외곽과 front/rear 역할을 확인할 수 없습니다. 배경 픽셀이나 q3의 OOB를 외부 가림 E로 바꾸지 않습니다.'),
1:dict(target_annotation_id=None,identity='NO_SELECTED_TARGET',related=[],
 why='원래 추론이 무검출이므로 선택된 대상과 native 역할이 없습니다. 화면의 다른 공식 팔레트를 대신 골라8슬롯을 만들지 않습니다.'),
2:dict(target_annotation_id=335065,identity='VISUALLY_ASSOCIATED_LOADED_FLAT_PALLET',related=[335065],
 why='화물을 올린 단일 낮은 팔레트가 선택 대상입니다. 앞쪽 긴 전면과 양 끝 하부 받침이 보이며 원래 front0..3 역할을 구조적으로 읽을 수 있습니다. 투명 포장/끈이 있고 q는 일부 빗나가지만, 이것만으로 역할의 보임을 가림으로 바꾸지 않습니다.'),
3:dict(target_annotation_id=329865,identity='PARTIAL_VISUAL_ASSOCIATION_BACKGROUND_CARTON_PALLET',related=[329847,329865],
 why='통로 왼쪽 뒤, 상자 아래 작은 팔레트의 전면 오른쪽 끝이 선택 영역에 있습니다. 원래 front1/2의 위·아래 오른쪽 역할은 목재 외곽으로 보입니다. 왼쪽 q0/3은 인접 파란 구조와 겹쳐 같은 팔레트의 왼쪽 역할 대응이 불확실합니다. 다른 개체로 재선택하거나 파란 픽셀만으로 E를 만들지 않습니다.'),
4:dict(target_annotation_id=None,identity='PARTIAL_TINY_RIGHT_BORDER_ENTITY_UNCERTAIN',related=[284764],
 why='첫 선택은 전경 사람 뒤 팔레트가 아니라 화면 맨 오른쪽의 작은 노란 물체/받침 부근입니다. 부분적으로 잘린 대상의 팔레트 구조와 native 역할이 불명확합니다. 사람에 의한 전경 가림을 이 대상의 E로 복사하지 않습니다.'),
5:dict(target_annotation_id=None,identity='PARTIAL_TINY_RIGHT_BORDER_ENTITY_UNCERTAIN',related=[286380],
 why='오른쪽 화면 끝의 작은 받침/기둥 영역이며 원래 q다수가 OOB입니다. 하나의 팔레트와 그 앞뒤 모서리 정의가 확인되지 않습니다. 입력 OOB와 teacher 의미 불확실성을 별도 보존합니다.'),
6:dict(target_annotation_id=None,identity='MULTIPLE_PALLETS_CROSSED_BY_ONE_PREDICTED_CUBOID',related=[429343,429345,429347,429349,429353],
 why='선택 박스와 native8은 왼쪽 전경 포장 화물, 녹색 선반 기둥, 오른쪽 뒤 포대/통 아래 서로 다른 팔레트들을 가로지릅니다. q0/3의 포장 픽셀을 곧바로 외부 가림으로 볼 수 없고 어느 단일 footprint의 native 역할인지 먼저 불확실합니다.'),
7:dict(target_annotation_id=None,identity='RACK_END_MULTIPLE_SMALL_BASES_ROLE_UNCERTAIN',related=[430281,430283,430300],
 why='선반 끝의 작은 밝은 통/받침과 녹색 기둥 부근입니다. 복수 공식 팔레트와 선반 부재가 가깝고 선택된 한 팔레트의 전체 전면/뒤쪽 역할을 확정하기 어렵습니다. 녹색 기둥 픽셀이나 검은 틈만으로 E/S를 만들지 않습니다.'),
8:dict(target_annotation_id=431549,identity='PARTIAL_VISUAL_ASSOCIATION_BACKGROUND_WRAPPED_LOAD_PALLET',related=[431549,431552],
 why='사람 뒤 배경의 포장 화물 아래 팔레트 오른쪽 전면 상·하단 외곽이 보입니다. 원래 front1/2의 구조적 역할은 D 후보입니다. 왼쪽은 이웃 팔레트와 겹쳐 실제 선택의 front0/3 역할 연결이 더 불확실합니다. 전경 사람은 선택 대상의 직접 가림 근거가 아닙니다.'),
9:dict(target_annotation_id=None,identity='RACK_PROTECTIVE_BEAM_OR_FALSE_POSITIVE_UNRESOLVED',related=[],
 why='선택 박스가 회색 선반 보호판과 노란 기둥의 하부에 놓이며, 이 구역에서 하나의 팔레트 native 외곽을 확인할 수 없습니다. 공식 박스와 겹치지 않는다는 사실만으로 부재 E를 만들지 않습니다.'),
10:dict(target_annotation_id=371963,identity='PARTIAL_VISUAL_ASSOCIATION_BACKGROUND_SMALL_CARTON_PALLET',related=[371963],
 why='첫 선택은 앞쪽 큰 종이상자 아래 ann371955가 아니라 뒤쪽 작은 상자 아래 ann371963입니다. 이 작은 팔레트의 전면 오른쪽 상·하단은 보입니다. 왼쪽은 파란 인접 물체와 역할 경계가 모호합니다. 기존 앞쪽 대상의 R1/R2:D를 복사한 것이 아니라 실제 선택 대상을 새로 검토한 제안입니다.'),
11:dict(target_annotation_id=None,identity='WIRE_CAGE_OR_RACK_BASE_TARGET_DEFINITION_UNCERTAIN',related=[],
 why='첫 선택은 바닥의 철망 케이지/선반 하부와 노란 기둥 주변입니다. 위 선반의 공식 팔레트와 다르며, 선택된 구조가 별도 팔레트인지 케이지/선반 일체형 받침인지 불확실합니다. 보이는 금속 모서리를 목재 팔레트 native 라벨로 간주하지 않습니다.')}
known={2:{0:'D',1:'D',2:'D',3:'D'},3:{1:'D',2:'D'},8:{1:'D',2:'D'},10:{1:'D',2:'D'}}
rows=[]
for original in facts['rows']:
    r=dict(original);i,k=r['frame_index'],r['native_slot'];f=frames[i];state=known.get(i,{}).get(k,'U')
    if state=='D':
        why=('선택된 단일 팔레트의 전면 왼쪽 위 구조적 외곽이 보입니다.'if k==0 else'선택된 팔레트의 전면 오른쪽 위 모서리 역할이 목재 외곽으로 보입니다.'if k==1 else'선택된 팔레트의 전면 오른쪽 아래 받침/외곽 역할이 바닥 경계에서 보입니다.'if k==2 else'선택된 단일 팔레트의 전면 왼쪽 아래 받침/외곽 역할이 보입니다.')
        why+=' q의 정확한 픽셀 소속 또는 좌표 오차 크기를 보임 라벨의 조건으로 사용하지 않았습니다.'
    elif f['identity']in('VISUALLY_ASSOCIATED_LOADED_FLAT_PALLET','PARTIAL_VISUAL_ASSOCIATION_BACKGROUND_CARTON_PALLET','PARTIAL_VISUAL_ASSOCIATION_BACKGROUND_WRAPPED_LOAD_PALLET','PARTIAL_VISUAL_ASSOCIATION_BACKGROUND_SMALL_CARTON_PALLET'):
        why='뒤쪽 역할의 가시성/자기 가림/화물에 의한 외부 가림을 분리할 근거가 충분하지 않습니다.'if k>=4 else'이웃 물체/팔레트와 원래 왼쪽 native 역할의 대상 대응이 불확실합니다. q가 배경/다른 물체에 있다는 이유만으로 E를 만들지 않습니다.'
    else:why=f['why']
    r.update(native_role=roles[k],selected_target_annotation_id=f['target_annotation_id'],target_identity_status=f['identity'],
        related_official_annotation_ids=f['related'],semantic_state=state,semantic_reason_KO=why,
        target_and_role_evidence_KO=f['why'],reference_xy=None,reference_status='STRUCTURAL_ROLE_REVIEW_NO_NEW_COORDINATE_REFERENCE',
        assistant_visual_review=True,human_gold=False,training_now=False,trainready=False,
        visibility_label_independent_of_input_validity=True,no_coordinate_error_filter=True,
        original_firstargmax_unchanged=True,no_GT_nearest_or_permutation=True,no_unrelated_label_transfer=True)
    rows.append(r)
assert len(rows)==96 and [r['global_index']for r in rows]==list(range(96))
counts=Counter(r['semantic_state']for r in rows);assert counts==dict(D=10,U=86)
path=H/'LOCO_SELECTED_NATIVE96_PROPOSALS.jsonl';assert not path.exists()
with path.open('x')as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
summary=dict(completed=True,source_only=True,code=bind(__file__),crop_facts=bind(H/'fixed_source96_contact_v1/CROP_FACTS96.json'),
    proposals=bind(path),all12_frames_personally_viewed=True,all96_factual_crops_personally_viewed=True,all424_official_instance_contexts_preserved=True,
    original_frames=12,original_slots=96,semantic_state_counts=dict(counts),input_status_separate=dict(noquery=sum(r['no_detection']for r in rows),finite_OOB=sum(not r['original_input_valid']and not r['no_detection']for r in rows),valid=sum(r['original_input_valid']for r in rows)),
    per_frame=[dict(frame_index=i,**frames[i],semantic_counts=dict(Counter(r['semantic_state']for r in rows if r['frame_index']==i)),sheet=facts['sheets'][i]['sheet'])for i in range(12)],
    labels_are_assistant_proposals_not_human_gold=True,training_now=False,trainready=False,
    interpretation_KO='실제 선택된 대상과 native 역할에 대한 약한 source 제안입니다. 보임10개를 제안했고 자연 외부 가림/자기 가림을 확정할 근거가 부족한86개는U로 유지했습니다. U는 negative가 아닙니다. 작은 대상·blur·wrongentity·part/whole 역할 불확실성이 많으며 q픽셀/오차로 semantic 라벨을 바꾸지 않았습니다.',
    model_encoder_initializer_fit_current223_V112_PnP_calls_or_reads=0,original_q_L_order_and_labels_not_modified=True)
sp=H/'LOCO_REVIEW_SUMMARY.json';assert not sp.exists();sp.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(completed=True,proposals=bind(path),summary=bind(sp),semantic_counts=dict(counts))))
