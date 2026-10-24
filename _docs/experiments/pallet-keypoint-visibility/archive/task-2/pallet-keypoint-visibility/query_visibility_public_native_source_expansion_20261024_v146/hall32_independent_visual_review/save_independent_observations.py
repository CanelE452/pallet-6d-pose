"""Save already independently viewed public HALL observations, not an auto-labeler.
Labels/evidence below are manual assistant proposals from all32 fullscene/crop
sheets plus4 overviews. No other agent proposals or model outputs were read.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
V=Path(__file__).resolve().parents[1];H=Path(__file__).resolve().parent;D=V/'saved_source64_contact_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,a):
    with Path(p).open('x')as f:json.dump(a,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
CONTACT=D/'CONTACT_RENDER_COMPLETE.json';assert sha(CONTACT)=='ee5884b2c5beec495ef0de2d0c11b1dad7f9dd37d987964ee3192ba863c8d961'
contact=read(CONTACT);assert contact['completed']and contact['PASS']and contact['all512_semantic_U_UNREVIEWED']
facts=[json.loads(s)for s in Path(contact['all512_crop_facts']['path']).read_text().splitlines()];assert len(facts)==512 and bind(contact['all512_crop_facts']['path'])==contact['all512_crop_facts']
frames={
32:('SAME','U','D','큰 목재통이 선택되었다. 왼쪽 하부 받침과 앞 바닥판은 드러난다. 오른쪽 실제 받침 일부는 보이지만 원 whole-AABB lower2 참조는 영상 밖이라 제한된 virtual 역할의 관측 범위를 확정하지 않는다.'),
33:('SAME','D','E','목재통 전면의 오른쪽 하부 받침은 드러나고, 앞에 선 별도 종이상자가 왼쪽 하부 받침/모서리 방향을 가로막는다.'),
34:('SAME','U','D','목재통 왼쪽 하부는 보이고 오른쪽 끝은 화면 오른쪽과 아래에서 잘려 lower2를 판단할 수 없다.'),
35:('SAME','D','D','선택된 목재통의 양쪽 전면 하부 받침과 바닥판 끝이 모두 노출되어 있다. 별도 가림물은 전면을 가리지 않는다.'),
36:('SAME','D','D','중앙 오른쪽 목재통의 양쪽 앞 하부가 보인다. 왼쪽 종이상자는 떨어져 있어 전면 하부 역할을 덮지 않는다.'),
37:('SAME','D','D','목재통 양쪽 앞 하부가 드러난다. 화면 맨 왼쪽 상자는 목표와 떨어져 있다.'),
38:('SAME','U','D','목재통 왼쪽 앞 하부는 노출되지만 오른쪽 하부 끝은 화면 밖으로 이어진다.'),
39:('SAME','U','D','목재통 왼쪽 앞 하부는 보인다. 오른쪽 전면 끝은 잘려 lower2 역할을 확정하지 않는다.'),
40:('SAME','D','E','원 firstargmax는 뒤의 목재통이다. 오른쪽 하부 받침은 드러나고 앞의 별도 종이상자가 왼쪽 하부를 덮는다.'),
41:('SAME','D','D','중앙 목재통의 양쪽 전면 하부 받침과 바닥판 끝이 노출되어 있다.'),
42:('SAME','U','E','목재통은 선택되어 있으나 오른쪽 하부는 화면 밖이다. 왼쪽 하부 방향은 전경 종이상자에 의해 가려져 있다.'),
43:('SAME','U','D','목재통이 선택되었고 왼쪽 하부는 보인다. 오른쪽 실제 받침 일부는 경계 가까이에 있지만 원 lower2 참조가 영상 아래라 보수적으로 U로 둔다.'),
44:('DIFFERENT','U','U','원 firstargmax의 작은 박스/키포인트는 목재통이 아니라 중앙 바닥의 흰색 기둥/받침 근처에 모여 있다. 목재통 lower 역할을 이 후보에 연결하지 않는다.'),
45:('DIFFERENT','U','U','원 firstargmax는 전경 종이상자의 Lidl 글자 부근을 선택한다. 뒤 목재통의 역할/가림 라벨을 전이하지 않는다.'),
46:('SAME','D','E','선택된 목재통 오른쪽 하부는 드러나고, 전경 종이상자가 왼쪽 하부를 덮는다.'),
47:('DIFFERENT','U','U','원 firstargmax는 전경 종이상자 글자 부분의 좁은 영역이다. 오른쪽 목재통에 재선택하지 않고 U를 유지한다.'),
48:('SAME','D','E','선택된 목재통 오른쪽 하부는 보인다. 앞에 놓인 별도 종이상자가 왼쪽 하부 받침/모서리 역할을 가로막는다.'),
49:('SAME','D','D','중앙 목재통 양쪽 전면 하부가 노출되어 있다. 왼쪽 종이상자는 목재통 앞 하부와 겹치지 않는다.'),
50:('SAME','D','D','오른쪽 목재통이 선택되었다. 영상 오른쪽 경계 근처의 앞 오른쪽 하부 받침/바닥판 끝과 왼쪽 하부가 실제로 보인다. query의 OOB 여부와 별개로 구조 근거에 따른 D 제안이다.'),
51:('SAME','D','D','목재통 양쪽 앞 하부가 명확히 보인다. 왼쪽 상자와 빨간 운반기는 목표 앞 하부를 가리지 않는다.'),
52:('DIFFERENT','U','U','원 firstargmax는 빨간 팔레트 운반기 포크를 선택한다. 오른쪽 가장자리 목재통의 의미 라벨을 연결하지 않는다.'),
53:('SAME','D','D','목재통이 선택되었고 양쪽 앞 하부 받침/바닥판이 드러난다. 상자와 운반기는 앞 하부에서 떨어져 있다.'),
54:('SAME','D','D','가까운 목재통의 양쪽 전면 하부 받침/바닥판 끝이 드러난다. 오른쪽 query가 영상 밖이어도 원 장면에서 해당 하부 구조가 보인다는 약한 의미 제안은 D다.'),
55:('SAME','D','D','중앙 목재통의 양쪽 앞 하부가 노출되어 있고 외부 상자는 왼쪽 멀리 떨어져 있다.'),
56:('SAME','D','D','큰 목재통 양쪽 앞 하부와 받침이 보인다. 통 안의 내용물은 이 하부 역할을 가리지 않는다.'),
57:('UNRESOLVED','U','U','원 박스가 종이상자 오른쪽과 빈 바닥, 오른쪽 목재통 일부를 함께 걸친다. 하나의 원 firstargmax 개체가 제공 pose의 목재통이라고 안정적으로 확인되지 않아 재연결 없이 U를 둔다.'),
58:('DIFFERENT','U','U','원 firstargmax는 빨간 운반기의 포크를 선택한다. 뒤의 목재통에 새로 대응시키지 않는다.'),
59:('SAME','U','D','목재통 왼쪽 앞 하부는 노출되지만 오른쪽 끝은 화면에 잘려 lower2를 판단할 수 없다.'),
60:('DIFFERENT','U','U','원 firstargmax는 앞의 종이상자 위쪽을 선택한다. 오른쪽 목재통의 하부 역할/가림을 이 후보의 라벨로 쓰지 않는다.'),
61:('DIFFERENT','U','U','원 firstargmax의 작은 후보는 먼 흰색 기둥 받침 부근이다. 실제 목재통과 다른 개체이므로 U를 둔다.'),
62:('SAME','D','D','중앙 목재통의 양쪽 전면 하부가 드러난다. 상자와 운반기는 왼쪽에 있고 하부를 가리지 않는다.'),
63:('SAME','D','E','목재통의 오른쪽 앞 하부는 드러난다. 목재통 앞 왼쪽의 종이상자가 왼쪽 하부 받침/모서리 역할을 가로막는다.')}
assert set(frames)==set(range(32,64))
viewed=[]
for b in contact['overviews'][4:8]:assert bind(b['path'])==b;viewed.append(dict(binding=b,view_type='ACTUALLY_VIEWED_HALL_OVERVIEW'))
for s in contact['sheets'][32:64]:assert bind(s['sheet']['path'])==s['sheet'];viewed.append(dict(binding=s['sheet'],view_type='ACTUALLY_VIEWED_WHOLE_RAW_ANNOTATED_AND_ALL8_FACTUAL96',frame_index=s['frame_index']))
observations=[];rows=[]
for frame in range(32,64):
    identity,l2,l3,evidence=frames[frame];s=contact['sheets'][frame]
    observations.append(dict(frame_index=frame,id=s['id'],selected_target_identity=identity,lower2_proposal=l2,lower3_proposal=l3,visual_evidence_KO=evidence,viewed_sheet=s['sheet'],other_agent_proposals_read=False))
    for k in range(8):
        original=facts[frame*8+k];assert original['frame_index']==frame and original['native_slot']==k and original['split']=='VAL_HALL'and original['semantic_state']=='U'
        state={2:l2,3:l3}.get(k,'U')
        if identity!='SAME':reason=evidence;unknown='WRONG_SELECTED_ENTITY'if identity=='DIFFERENT'else'SELECTED_ENTITY_UNRESOLVED'
        elif k not in(2,3):reason='하부2/3 limited whole-assembly 역할 연결만 검토했다. upper/rear의 원 native 의미 대응과 가림은 이 자료에서 확정하지 않는다.';unknown='UPPER_OR_REAR_ROLE_NOT_CERTIFIED'
        elif state=='U':reason=evidence;unknown='LOWER_ROLE_OUTSIDE_OR_BOUNDARY_REFERENCE_UNRESOLVED'
        else:reason=evidence;unknown=None
        if state in'DE':assert identity=='SAME'and k in(2,3)and original['lower_role_continuity_status']=='PASS'
        rows.append(dict(original,semantic_state=state,semantic_reason_KO=reason,assistant_reviewed=True,assistant_independent_proposal=True,
            selected_target_identity=identity,unknown_reason=unknown,evidence_quality='ASSISTANT_STRUCTURAL_WEAK'if state in'DE'else'EXPLICIT_UNRESOLVED',
            source_scope='V144/V146 original whole-assembly virtual footprint + bottom plane LOWER2/3 only; original UP=-OBJY and scanAABB precision caveat unchanged',
            structural_observation_border_note='Physical lower-right support is partly visible, but unchanged whole-AABB virtual reference is beyond image; proposal remains U.'if frame in(32,43)and k==2 else None,
            label_uses_query_error_or_query_pixel_color=False,teacher_and_original_input_availability_separate=True,reference_not_modified=True,
            human_gold=False,precise_physical_vertex_gold=False,trainready=False,training_now=False,viewed_sheet=s['sheet']))
assert len(rows)==256 and {r['global_index']for r in rows}==set(range(256,512))and all(r['human_gold']is False for r in rows)
assert all(r['semantic_state']=='U'for r in rows if r['native_slot']not in(2,3))
counts=dict(Counter(r['semantic_state']for r in rows));availability={s:dict(Counter('INPUT_VALID'if r['original_input_valid']else'INPUT_U'for r in rows if r['semantic_state']==s))for s in'DESU'}
output=dict(completed=True,independent_visual_review_complete=True,frames32_all8slots256_preserved=True,contact_complete=bind(CONTACT),source_crop_facts=contact['all512_crop_facts'],review_script=bind(__file__),
    reviewer='Independent assistant /root/query_failure_mechanism',viewed_assets=viewed,other_agent_proposals_read=False,all32_whole_and_original_all8_crops_actually_viewed=True,
    lower_scope_unchanged=True,original_firstargmax_q_L_source_refs_unchanged=True,all_upper_rear_U=True,human_gold=False,automatic_geometry_mask_depth_teacher=False,
    semantic_counts=counts,semantic_by_original_input_availability=availability,selected_identity_counts=dict(Counter(r['selected_target_identity']for r in observations)),
    expected_foreign_blocker='separate foreground cardboard box',E_native3_frames=[i for i in frames if frames[i][2]=='E'],
    boundary_cases_for_root_adjudication=[32,43,50,54,57],frames=observations,rows=rows,
    trainready=False,training_now=False,current_V112_model_fit_PnP_calls_or_reads=0)
target=H/'HALL32_INDEPENDENT_WEAK_PROPOSALS256.json';write(target,output)
write(H/'REVIEW_COMPLETE.json',dict(completed=True,PASS=True,proposals=bind(target),contact_complete=bind(CONTACT),review_script=bind(__file__),frames=32,rows=256,
    semantic_counts=counts,semantic_by_original_input_availability=availability,viewed32_whole_all8_crops_and4overviews=True,other_agent_proposals_read=False,
    assistant_weak_only_not_human_gold=True,training_or_model_calls=0,current_or_V112_access=False,selected_target_identity_counts=output['selected_identity_counts']))
print(json.dumps(dict(completed=True,receipt=bind(H/'REVIEW_COMPLETE.json'),counts=counts,availability=availability),ensure_ascii=False))
