"""Freeze reviewer A's actual raw-RGB observations; no additional RGB/model IO."""
from pathlib import Path
import csv,hashlib,json
R=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def write(p,o):
    with p.open('x') as f:json.dump(o,f,indent=2,ensure_ascii=False);f.write('\n')
policy=read(R/'REVIEW_POLICY_BEFORE_RGB.json')
inventory=read(R/'SAVED_INITIALIZER_QUERY_INVENTORY.json')
display=read(R/'DISPLAY_RECEIPT.json')
notes=[dict(
 id='P0',reference_xy_unchanged=[409.,450.],
 local_cue_name_KO='왼쪽 큰 carton 아래 낮은 목재 pallet의 오른쪽 deck 끝/수직 support edge가 꺾이는 부분',
 actual_RGB_evidence_KO='밝은 가로 목재 상판 띠가 끝나고 세로 목재 면으로 내려가는 꺾임이 원본과96-native 확대본에서 보입니다. 위 carton 끝과 구분되는 낮은 목재 구조입니다.',
 target_identity_KO='왼쪽 carton을 받치는 낮은 flat-pallet base로 해석하는 것이 합리적입니다. 오른쪽 큰 수송 crate와 다른 물체이며 전체 팔레트는 왼쪽 화면 경계에 잘립니다.',
 alternate_cue_KO='carton/wood contact seam은 가까운 다른 선입니다. 개별 deck board 끝과 전체 pallet cuboid 꼭짓점이 같다고 확정하지 않습니다.',
 uncertainty_assessment_KO='기존 r 주변±4px는 흐린 board-end 꺾임을 포함하는 그럴듯한 범위입니다. 검교정된 좌표 오차나 모든 픽셀의 D 보증은 아닙니다. r/radius를 수정하지 않았습니다.',
 occluder_KO='화면 중앙 사람은 이 국소 cue를 덮지 않습니다. carton이 다른 부분을 가리지만 제안된 노출 목재 꺾임은 관찰 가능합니다. semantic E/S는 부여하지 않습니다.',
 saved_q_association_KO='선택된 top1 box는 왼쪽 하단 목재 base 부근이나 고정 r는 box 밖입니다. 동일 instance 대응의 공식 증거와 r↔native slot dictionary가 없습니다. box밖이라는 이유만으로 cropMISS를 선언하지 않습니다.'),
 dict(
 id='P8',reference_xy_unchanged=[945.,615.],
 local_cue_name_KO='오른쪽 큰 carton 아래 가까운 목재 support block의 낮은 끝/밑판과 바닥 경계가 돌아가는 부분',
 actual_RGB_evidence_KO='밝은 목재 블록 면과 아래 얇은 목재 띠가 바닥 가까이에서 방향을 바꾸는 경계가 보입니다. carton/AprilTag는 위쪽이며 국소 목재 부분과 재질/높이가 구분됩니다.',
 target_identity_KO='오른쪽 foreground carton을 지탱하는 낮은 목재 flat pallet의 구성부로 해석할 수 있습니다. 전체 pallet 오른쪽은 화면 밖으로 이어집니다.',
 alternate_cue_KO='support block 외곽, 얇은 하부 board 끝, floor-contact shadow가 인접합니다. 어느 재료 모서리를 한 pixel reference로 삼는지는 불확실하며 전형적인 cuboid8 corner라고 확정하지 않습니다.',
 uncertainty_assessment_KO='기존±4px는 국소 경계 주변을 포함하나 둥근/겹친 밑판과 그림자로 단일 정밀 교점은 확정되지 않습니다. 보이는 부분선 cue의 약한 증거이며 centerpixel GT가 아닙니다.',
 occluder_KO='carton/marker는 상부 deck 일부를 덮지만 낮은 목재 경계는 직접 보입니다. 왼쪽 robot은 cue 위치를 덮지 않습니다.',
 saved_q_association_KO='보존된 top1은 화면 뒤쪽 작은 영역(box x342..462,y251..330)이며 오른쪽 foreground pallet와 다릅니다. 다른 후보를 GT-nearest/GT-best로 바꾸지 않았습니다. 이 cue에 대한 배포 입력의 instance/native-slot 대응은 UNKNOWN입니다.'),
 dict(
 id='P15',reference_xy_unchanged=[945.,470.],
 local_cue_name_KO='오른쪽 purple crate stack 아래 낮은 목재 pallet의 왼쪽 top-deck/end-board 끝의 방향 전환 부분',
 actual_RGB_evidence_KO='낮게 놓인 밝은 목재의 상판/끝판 띠와 아래 세로 목재 면이 r 주변에서 만납니다. 위 보라색 plastic crate ribs와 다른 목재 구조입니다.',
 target_identity_KO='보라색 load stack 아래 flat wooden pallet라는 국소 해석은 합리적입니다. 오른쪽/뒤쪽 전체 외곽은 화면 밖이라 전체 구조 식별은 부분적입니다.',
 alternate_cue_KO='개별 slat 끝, top-deck 외곽, end-board 윗선이 가깝습니다. 배경의 가는 검은 선/바닥 경계를 pallet 선으로 바꾸어 쓰지 않습니다.',
 uncertainty_assessment_KO='코너/부분선은 보이나 blur와 둥근 재료 끝 때문에 단일 pixel 위치와±4px 충분성을 정량 인증하지 못합니다. 범위는 보존하며 native8 정체성은 UNKNOWN입니다.',
 occluder_KO='load는 deck 뒤쪽을 덮지만 제안된 왼쪽 노출 목재 끝은 보입니다. 숨은 다른 reference의 E/S를 RGB만으로 만들지 않습니다.',
 saved_q_association_KO='기존 frozen initializer는 이 frame에서 no_detection이므로 선택 q9/q8가 없습니다. GT r를 q로 쓰거나 누락 q를 생성하지 않았고 crop availability도 판정하지 않았습니다.')]
for note,fixed,inv,fig in zip(notes,policy['fixed_records'],inventory['records'],display['records']):
    assert note['id']==fixed['id']==inv['cue_id']==fig['id']
    note.update(assessment='APPARENT_VISIBLE_LOCAL_PALLET_CUE_WEAK_AI_EVIDENCE',original_uncertainty_radius_px=fixed['original_uncertainty_radius_px'],source_RGB=dict(path=fixed['image'],sha256=fixed['image_sha256']),reference_relocalized=False,uncertainty_changed=False,official_or_human_semantic_state=None,native8_semantic_id=None,native_mapping_status='UNKNOWN',can_certify_centerpixel=False,human_verified=False,official_verified=False,training_admitted=False,figure=fig['panel'],unmarked_native96=fig['unmarked_native96'],initializer_input_id=inv['input_id'],initializer_selected_index=inv['selected_index'])
review=dict(schema='V83_REVIEWER_A_PUBLIC_CUE_EVIDENCE_REVIEW_V1',completed=True,reviewer='AI_REVIEWER_A',
 status='THREE_LOCAL_CUE_WEAK_EVIDENCE_CANDIDATES; NATIVE_ASSOCIATION_AND_TRAINING_HOLD',
 findings_KO='세 기존 제안 주변에 낮은 목재 pallet의 국소 corner/부분선 단서를 RGB로 관찰할 근거가 있습니다. 구체적인 약한 cue-reference 후보가 가능하다는 뜻이며 native8 semantic identity, centerpixel GT, 사람 D/E/S gold를 제공하지 않습니다. 기존 제안/GT admission은 변경하지 않았습니다.',
 records=notes,counts=dict(existing_candidates_reviewed=3,apparent_local_visible_cue_weak_evidence=3,human_confirmed=0,official_confirmed=0,native8_associations_certified=0,training_rows_admitted=0,independent_capture_families=1,same_bag_frames=3),
 initializer_availability=inventory['fixed3_summary'],
 future_feasibility_KO='고정 r/원래 불확실 범위/국소 cue 설명을 GT-only로 묶은 source conditional cue pilot은 구체화할 수 있습니다. 실제 q와 target/reference association 및 crop FULL/PARTIAL/MISS를 독립적으로 고정하고 원래 보이는 cue만 조건부 evidence로 다루는 후속 프로토콜이 필요합니다. 현재 top1 기준 P8은 다른 대상, P15는 q 없음이며 P0도 native8 대응 미증명입니다. GT로 q를 recentre하거나 GT-best detection을 골라 간극을 숨길 수 없습니다.',
 identifiability_KO='V69 숨은 reference identity 충돌은 단순 q-jitter나 동일 RGB의 여러 r 지정으로 없어지지 않습니다. 배포 target/native-slot/cue association 규칙을 별도로 정해야 합니다. 현재8 logical query ordinal은 physical semantic8 dictionary가 아닙니다.',
 negative_scope_KO='자연 E/S gold, 실제 before/after, 불투명 가림 certificate, depth teacher는 이번에 만들지 않았습니다. RGB의 mask0/안 보임/구멍을 E/S로 바꾸지 않았습니다.',
 bias_limit_KO=policy['prior_exposure'],capture_group_KO=policy['split'],
 bindings=dict(policy=bind(R/'REVIEW_POLICY_BEFORE_RGB.json'),raw_verification=bind(R/'RAW_FILE_VERIFICATION.json'),saved_initializer_inventory=bind(R/'SAVED_INITIALIZER_QUERY_INVENTORY.json'),display=bind(R/'DISPLAY_RECEIPT.json'),old_proposals=policy['bindings']['old_candidate_proposals'],V69_draft=policy['bindings']['V69_draft'],code=bind(Path(__file__))),
 actual_raw_full3_unmarked_then_native_crops_viewed=True,actual_three_contact_directly_viewed=True,new_model_forward_train_composite_depthteacher_current_PnP=0,old_files_modified=False)
write(R/'CUE_EVIDENCE_REVIEW.json',review)
with (R/'CUE_SUMMARY.csv').open('x') as f:
    writer=csv.DictWriter(f,fieldnames=['id','r_x','r_y','old_uncertainty_radius_px','assessment','native8_status','selected_initializer_index','training_admitted']);writer.writeheader()
    for n in notes:writer.writerow(dict(id=n['id'],r_x=n['reference_xy_unchanged'][0],r_y=n['reference_xy_unchanged'][1],old_uncertainty_radius_px=n['original_uncertainty_radius_px'],assessment=n['assessment'],native8_status='UNKNOWN',selected_initializer_index=n['initializer_selected_index'],training_admitted=False))
files=sorted(p for p in R.iterdir() if p.is_file())
write(R/'REVIEW_SEAL.json',dict(completed=True,PASS=True,status=review['status'],files=[bind(p) for p in files],read_and_display_scope='Exactly3 existing raw publicsource frames; V69/V67 source docs and saved V72 public initializer q metadata. No current/GT masks/depth/model/weights.',review_is_AI_weak_source_evidence_not_human_gold=True,native8_mapping_remains_UNKNOWN=True,source_family_count=1,no_controlled_data_or_training_admission=True,new_forward_fit_composites_current_PnP=0,old_proposals_preserved_exact=True))
print(json.dumps(dict(review=bind(R/'CUE_EVIDENCE_REVIEW.json'),seal=bind(R/'REVIEW_SEAL.json'),contact=bind(R/'THREE_FIXED_PUBLIC_CUES_CONTACT.png'))))
