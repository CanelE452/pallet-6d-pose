"""Save reader C's tool-viewed nominal-family observations; never decode source images."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, io, json, re

D=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_intended_part_query_targets_20261025_v218')
H=D/'actual_gt_query_part_observations_c'; O=D/'public_source_gt_query_jitter_v1'
STATES={'D':'D_VISIBLE_EXPECTED_PART_WEAK','E':'E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK','S':'SELF_OR_OWN_STRUCTURE_UNOBSERVABLE','W':'WRONG_PART_OR_GEOMETRY_UNRESOLVED','U':'UNKNOWN','O':'OUT_OF_FRAME'}

# Manual observations from all15 contact PNGs, fixed display order; no A/root notes consumed.
# Each in-frame row is written explicitly. OOF follows the stored canvas flag, never visibility E.
NOTES={
 (0,0):{
  4:('D',[36,46,73,65],'같은 pallet의 먼 왼쪽 DECK 보드 끝의 밝은 윗면과 대각 바깥 경계가 nominal 주변에서 보입니다.',''),
  5:('D',[35,44,56,65],'같은 pallet의 오른쪽 DECK 보드 끝에서 윗면·끝 경계가 만나는 유한 영역이 보입니다.',''),
  6:('W',None,'crop의 목재는 상판 보드의 아래 경계와 틈입니다. 선언 BOTTOM_RUNNER 끝으로 식별되지 않습니다.','DECK 아래변을 BOTTOM_RUNNER로 복사하지 않습니다.'),
  7:('W',None,'상판 사이 틈과 상판 보드 면이 보이며 선언 BOTTOM_RUNNER 끝의 대응은 확인되지 않습니다.','DECK-vs-BOTTOM_RUNNER 불확실성입니다.')},
 (0,1):{
  5:('W',None,'nominal 주변은 같은 상판 보드의 길이 방향 중간 면/경계입니다. 외곽 DECK 끝 대응을 확인하지 못했습니다.','목재 자체의 가시성을 의도 끝 부위 D로 복사하지 않습니다.'),
  6:('W',None,'보이는 것은 상판 보드와 틈이며 BOTTOM_RUNNER 끝의 영상 근거가 없습니다.','다른 GT candidate를 골라 대체하지 않습니다.')},
 (1,0):{},(1,1):{},
 (2,0):{
  0:('D',[49,40,77,68],'fullcontext와 이어지는 가까운 왼쪽 외곽 DECK 보드 끝의 윗면/끝 경계가 crop에 보입니다.',''),
  3:('W',None,'가까운 왼쪽 지지 블록의 세로 면이 중심 주변에 보입니다. BOTTOM_RUNNER 끝과의 대응은 확인되지 않습니다.','블록 면과 하부 runner를 구분합니다.'),
  4:('D',[35,45,66,63],'pallet의 먼 왼쪽 외곽 DECK 보드 끝과 짧은 끝 경계가 보입니다.',''),
  5:('D',[25,35,49,62],'pallet의 먼 오른쪽 DECK 보드 끝의 윗면과 바깥 둥근/비스듬한 끝 경계가 보입니다.','nominal 중심 옆의 유한 부위 근거이며 정확 점 오차를 판정하지 않습니다.'),
  6:('W',None,'오른쪽 상판의 옆/끝 경계가 보이나 선언 BOTTOM_RUNNER 부위로 식별되지 않습니다.','상판 옆면은 runner 가시성 증거가 아닙니다.'),
  7:('W',None,'상판 보드의 면과 사이 틈이 보입니다. BOTTOM_RUNNER 끝 대응은 미해결입니다.','')},
 (2,1):{
  0:('D',[36,40,69,66],'candidate 1의 nominal 가까운 왼쪽 DECK 끝 주변에 실제 pallet 외곽 보드 코너가 보입니다.','candidate 0과 관찰 영역이 겹치지만 best candidate 선택이나 독립 frame 증가가 아닙니다.'),
  3:('W',None,'nominal 주변은 세로 지지 블록 면입니다. 하부 runner 끝의 대응을 확정하지 못했습니다.',''),
  4:('W',None,'nominal 중심은 내부 상판 보드 면에 놓입니다. 선언 외곽 DECK 끝의 국소 대응은 불확실합니다.','임의 상판 표면을 끝 부위로 채택하지 않습니다.'),
  5:('D',[14,38,41,61],'nominal 근방 왼쪽에 같은 pallet 외곽 DECK 보드 끝의 둥근 경계와 윗면이 보입니다.','중심은 목재 경계 밖에 있으나 유한 끝 부위는 crop 안에 있습니다. 정확 q 포함을 요구하지 않습니다.'),
  6:('W',None,'nominal 주변에 상판 아래 블록/옆면과 바닥이 섞입니다. BOTTOM_RUNNER 끝을 독립 식별하지 못했습니다.',''),
  7:('W',None,'상판 보드와 틈이 보일 뿐 BOTTOM_RUNNER 끝의 근거가 아닙니다.','')},
 (3,0):{
  0:('D',[49,47,71,65],'같은 pallet의 가까운 왼쪽 DECK 보드 끝 모서리가 검은 배경 옆에 보입니다.',''),
  3:('D',[48,42,78,58],'같은 pallet 전면을 따라 이어지는 바닥 쪽 runner의 왼쪽 끝과 아래 경계가 보입니다.','지지 블록 면이 아닌 낮은 가로 plank의 끝 영역을 선택했습니다.'),
  4:('W',None,'상판의 길이 방향 경계와 뒤 물체가 보입니다. nominal 외곽 끝 부위와 실제 긴 보드 중간 경계의 대응이 불확실합니다.',''),
  5:('E',[18,18,79,48],'fullcontext에서 같은 pallet DECK가 회색 트레이 아래로 이어지고 선언 먼 DECK 끝 주변을 트레이 몸체가 가립니다.','GRAY_PLASTIC_TRAY'),
  6:('W',None,'nominal 주변은 내부 블록 면/틈으로 보입니다. BOTTOM_RUNNER의 실제 끝인지 확인하지 못했습니다.','자체 구조 차폐인지 기하 대응 불일치인지 판별하지 않습니다.'),
  7:('W',None,'세로 블록 면과 내부 가로 구조가 보이며 선언 runner 끝의 대응이 불확실합니다.','')},
 (4,0):{
  0:('D',[47,42,69,58],'fullcontext의 가까운 왼쪽 DECK 외곽 끝과 crop의 윗면/짧은 끝 경계가 이어집니다.',''),
  3:('D',[47,39,70,56],'pallet 가장 아래 가로 runner의 가까운 왼쪽 끝과 바닥 경계가 보입니다.',''),
  4:('W',None,'nominal 주변에 상판 긴 경계·파란 상자·배경이 섞이며 선언 먼 DECK 끝의 정체가 불확실합니다.',''),
  5:('E',[20,20,78,67],'동일 pallet 위의 회색 트레이가 DECK의 먼 끝 주변을 불투명하게 가립니다. 주변 상판 경계가 트레이 아래로 이어집니다.','GRAY_PLASTIC_TRAY'),
  6:('W',None,'상판 아래 어두운 틈/블록과 하부 보드가 섞입니다. nominal BOTTOM_RUNNER 끝 대응이 불확실합니다.',''),
  7:('W',None,'nominal 주변은 내부 구조 사이 틈이며 의도 runner 끝을 확인하지 못했습니다.','틈 자체를 E로 바꾸지 않습니다.')},
 (5,0):{
  1:('D',[20,46,42,64],'같은 pallet의 가까운 오른쪽 DECK 외곽 끝의 윗면과 둥근 끝 경계가 금속통 옆에서 보입니다.','nominal 중심과 정확히 일치하는 점을 주장하지 않습니다.'),
  2:('W',None,'nominal 주변에는 세로 지지 블록 옆면과 바닥이 보입니다. BOTTOM_RUNNER 끝은 식별되지 않습니다.',''),
  4:('D',[48,43,74,66],'동일 pallet의 먼 왼쪽 DECK 보드 끝과 비스듬한 바깥 경계가 보입니다.',''),
  5:('E',[12,12,82,80],'같은 pallet의 먼 DECK 쪽으로 이어지는 상판 위에 별도의 파란 플라스틱 상자가 놓여 해당 nominal 끝 주변을 가립니다.','BLUE_PLASTIC_CRATE'),
  6:('W',None,'nominal BOTTOM_RUNNER 위치는 파란 상자 면으로 보이나 실제 runner의 높이/부위 대응을 확인하지 못했습니다.','외부 물체가 보인다는 사실만으로 runner E를 만들지 않습니다.'),
  7:('W',None,'crop은 상판 보드의 면/옆 경계입니다. 하부 runner 끝의 대응은 확인되지 않습니다.','')},
 (6,3):{
  0:('D',[48,43,77,65],'세워 둔 동일 pallet의 큰 DECK 면 상단 왼쪽 외곽 보드 끝과 경계가 보입니다.','카메라 위쪽을 해부학적 +Z로 정의하지 않으며 보이는 deck 면의 유한 끝 부위만 기록합니다.'),
  1:('D',[32,42,51,68],'동일 서 있는 pallet의 DECK 면 상단 오른쪽 외곽 보드 끝이 배경과 구분되어 보입니다.','정확한 cuboid 3D 코너 인증은 아닙니다.'),
  2:('W',None,'상단 오른쪽에 보이는 것은 앞쪽 DECK 보드 끝입니다. 선언 BOTTOM_RUNNER 끝과 동일하다는 근거는 없습니다.',''),
  3:('W',None,'상단 왼쪽 DECK 보드 면/끝이 보이지만 BOTTOM_RUNNER 대응은 미해결입니다.',''),
  4:('W',None,'서 있는 pallet의 아래 끝과 전경 작은 목재 상자의 경계가 겹쳐 어느 끝이 해당 DECK 부위인지 불확실합니다.','전경 상자 코너를 pallet 끝 D로 복사하지 않습니다.'),
  5:('W',None,'crop 중심은 별도 전경 목재 상자의 안쪽 보드입니다. 서 있는 pallet의 선언 DECK 끝과의 대응이 불확실합니다.','원 nominal pose의 실제 물체 대응 불확실성입니다.'),
  6:('W',None,'전경 목재 상자/틈이 보이며 대상 pallet BOTTOM_RUNNER 끝의 정체가 확인되지 않습니다.',''),
  7:('W',None,'서 있는 pallet 아래 보드와 전경 작은 상자가 겹칩니다. BOTTOM_RUNNER 끝인지 확인되지 않습니다.','')},
 (7,3):{
  0:('D',[33,35,62,61],'동일 서 있는 pallet DECK 면의 상단 오른쪽 외곽 보드 끝과 옆 경계가 보입니다.',''),
  1:('D',[25,20,48,42],'동일 서 있는 pallet의 아래 오른쪽 외곽 DECK 보드 끝이 바닥 옆에 보입니다.','nominal 중심은 끝 밖의 바닥 쪽이지만 가까운 유한 끝 영역은 보입니다.'),
  2:('W',None,'crop에 있는 하단 외곽 목재가 DECK 면/지지부 중 어느 것인지 보이며 BOTTOM_RUNNER 대응을 확인하지 못했습니다.',''),
  3:('W',None,'상단 DECK 앞면 보드 끝이 보입니다. BOTTOM_RUNNER 끝으로 전환하지 않습니다.',''),
  4:('D',[47,44,78,69],'서 있는 pallet의 상단 왼쪽 DECK 외곽 보드 끝이 배경과 구분되어 보입니다.',''),
  5:('E',[18,38,78,69],'동일 서 있는 pallet의 아래 왼쪽 DECK 방향 외곽선이 전경 작은 별도 목재 상자 뒤로 이어지며 상자 윗판이 해당 끝 주변을 가립니다.','SEPARATE_SMALL_WOODEN_CRATE'),
  6:('U',None,'전경 목재 상자와 대상 pallet의 하단 구조가 겹쳐 의도 BOTTOM_RUNNER의 외부 차폐/자체 구조/기하 오차를 구분하지 못했습니다.','긍정적인 runner 정체 근거가 부족합니다.'),
  7:('W',None,'상단 왼쪽의 보이는 DECK 보드를 BOTTOM_RUNNER로 식별할 근거가 없습니다.','')},
 (8,3):{
  1:('D',[24,24,46,47],'동일 서 있는 pallet의 아래 오른쪽 외곽 DECK 보드 끝이 바닥 왼쪽에서 보입니다.','중심 바깥의 유한 끝 근거이며 점 위치 gold가 아닙니다.'),
  2:('W',None,'보이는 하단 외곽 목재는 앞면 보드/지지부와 섞여 BOTTOM_RUNNER 끝의 대응이 불확실합니다.',''),
  4:('D',[47,43,72,66],'동일 서 있는 pallet 상단 왼쪽 DECK 외곽 보드 끝과 배경 경계가 보입니다.',''),
  5:('E',[19,28,78,72],'대상 standing pallet 아래 왼쪽 DECK 방향을 별도 전경 목재 상자가 가리며, 상자의 테두리/판재가 nominal 주변에 불투명하게 보입니다.','SEPARATE_WOODEN_CRATE'),
  6:('U',None,'상자 판재와 pallet 구조가 겹치며 BOTTOM_RUNNER의 정체 및 차폐 원인을 분리할 수 없습니다.','외부 목재만 보고 E로 채우지 않습니다.'),
  7:('W',None,'상단 왼쪽의 DECK 보드 끝이 보이지만 선언 BOTTOM_RUNNER 부위와는 정체가 다르거나 미해결입니다.','')},
 (9,0):{
  4:('E',[15,17,80,76],'같은 pallet의 상판 외곽은 회색 운반 상자 밑으로 이어지고 선언 먼 DECK 끝 주변은 상자의 불투명 내벽/바닥으로 가려집니다.','GRAY_PLASTIC_CONTAINER'),
  5:('W',None,'nominal 중심은 바닥이고 목재는 crop 가장자리의 얇은 조각뿐입니다. 의도 DECK 끝 영역의 대응을 확인하지 못했습니다.',''),
  6:('W',None,'바닥과 가장자리 목재만 보이며 BOTTOM_RUNNER 끝의 정체를 확인하지 못했습니다.',''),
  7:('W',None,'회색 운반 상자 내부가 보이지만 하부 runner 위치와의 실제 부위 대응 근거는 부족합니다.','상자 자체를 runner E 근거로 자동 복사하지 않습니다.')},
 (10,0):{
  1:('W',None,'nominal 중심은 바닥이며 DECK 목재는 crop 구석 일부만 보입니다. 의도 끝 대응이 미해결입니다.',''),
  2:('W',None,'crop 왼쪽 목재 경계와 바닥에서 BOTTOM_RUNNER 끝을 식별하지 못했습니다.',''),
  5:('E',[15,15,80,78],'동일 pallet의 먼 DECK 외곽이 회색 운반 상자 아래에 놓이고 선언 끝 주변이 상자의 불투명 내벽/바닥으로 가려집니다.','GRAY_PLASTIC_CONTAINER'),
  6:('W',None,'운반 상자 내부 면이 보이나 선언 BOTTOM_RUNNER의 실제 위치/정체와 대응을 확인하지 못했습니다.','')},
 (11,0):{
  4:('E',[22,20,80,76],'같은 pallet의 먼 DECK 끝 방향이 회색 상자 밑으로 이어지며 그 주변은 회색 상자와 내부 금속통의 불투명 전경으로 가려집니다.','GRAY_PLASTIC_CONTAINER_AND_METAL_CAN'),
  5:('D',[12,42,34,66],'같은 pallet의 먼 오른쪽 외곽 DECK 보드의 둥근 끝 경계와 윗면이 crop 왼쪽에 보입니다.','nominal 중심은 바닥 쪽이지만 인접 유한 끝은 보입니다. 정확 위치 gold를 주장하지 않습니다.'),
  6:('W',None,'목재 옆 경계와 바닥은 보이나 하부 runner 끝을 식별할 수 없습니다.',''),
  7:('W',None,'crop은 회색 상자 외벽입니다. BOTTOM_RUNNER 끝의 직접적인 부위 대응은 확인되지 않습니다.','')}
}

def bind(p):
    p=Path(p); assert p.is_absolute() and not re.search('v112',str(p),re.I) and not any(x.is_symlink() for x in [p,*p.parents])
    b=p.read_bytes(); return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def save(name,obj):
    p=H/name; raw=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x') as f:f.write(raw)
    return bind(p)

def main():
    proto=json.loads((H/'PROTOCOL_BEFORE_ALL15_CONTACT_OBSERVATIONS.json').read_text()); complete=json.loads((O/'COMPLETE.json').read_text())
    assert bind(O/'COMPLETE.json')['sha256']=='c0b65a6ec259255f08742ea6078937b2d73fdabd7a0f9ca05daff733dba356c8'
    assert bind(complete['all360']['path'])==complete['all360'] and bind(complete['images']['path'])==complete['images']
    allrows=[json.loads(x) for x in Path(complete['all360']['path']).read_text().splitlines()]; original=[x for x in allrows if x['normalized_offset']==[0,0]]
    images=json.loads(Path(complete['images']['path']).read_text())['nominal_contacts15']; assert images==proto['exact_contact15']
    viewlist=[dict(binding=b,position=i,tool='tools.view_image via functions.exec',detail='original',actually_viewed=True,
        scope='Full contact including fullcontext and all8 enlarged nominal crops',batch_index=i//3) for i,b in enumerate(images)]
    views=save('ACTUAL_IMAGE_VIEW_RECEIPT.json',dict(completed=True,PASS=True,reader_id='C',views=viewlist,own_views_only=True,actual_distinct_contact_views=15,
        separate_training_RGB360_or_jitter_views=0,view_calls_were_before_saved_annotations=True,preexposed_context_and_renderer_author=True,blind_or_human_gold=False))
    bycontact={Path(b['path']).name.replace('_S0_CONTACT.png',''):b for b in images}; rows=[]
    for r in original:
        f,c,n=r['frame_position'],r['candidate_gt_list_index'],r['native_slot']; assert (f,c) in NOTES
        if not r['nominal_target_in_frame']:
            assert n not in NOTES[(f,c)]; code,roi,evidence,extra='O',None,'저장 nominal reference가 선언 영상 canvas 밖입니다. contact의 해당 crop은 padding 또는 일부 가장자리만 보입니다.','OOF는 외부 차폐 E나 원 initializer U와 다릅니다.'
        else:
            assert n in NOTES[(f,c)],(f,c,n); code,roi,evidence,extra=NOTES[(f,c)][n]
        E=code=='E'; occluder=extra if E else None
        row=dict(row_id=r['row_id'],family=r['family'],frame_position=f,candidate_gt_list_index=c,native_slot=n,canonical_index=r['canonical_index'],anatomical_family=r['anatomical_family'],
            SOURCE_role=r['source_role'],component_id=r['component_id'],generated_query_xy=r['generated_query_xy'],L=r['L'],generated_query_available=r['generated_query_available'],
            nominal_target_in_frame=r['nominal_target_in_frame'],original_input=r['original_input'],state=STATES[code],part_identity_evidence=evidence,
            foreign_evidence=evidence if E else None,occluder=occluder,optionalroi96=roi,
            ambiguity_reason=('선언 GT candidate/nominal 해부학에 근거한 유한 부위 약관찰입니다. 정확한 물리 native point, 보이지 않는 재료 형상, 원 detector-instance 대응을 인증하지 않습니다.' if code in ['D','E'] else extra),
            local_note=extra if not E else 'ROI는 전경을 포함한 weak bag이며 segmentation gold가 아닙니다.',
            actualview_image_binding=bycontact[f'F{f:02d}_C{c:06d}'],selected_or_best_GT_candidate=False,
            original_initializer_instance_to_GT_correspondence='UNKNOWN',source_GT_candidate_by_construction=True,
            transferred_to_jitter=False,training_admission=False,new_primary_native_gold=False)
        if roi is not None: assert len(roi)==4 and all(type(x) is int and 0<=x<=95 for x in roi) and roi[0]<=roi[2] and roi[1]<=roi[3]
        if code in ['D','E']: assert roi is not None and evidence
        if E: assert occluder and row['foreign_evidence'] and r['anatomical_family']=='DECK'
        rows.append(row)
    assert len(rows)==120 and len({r['row_id'] for r in rows})==120
    assert [r['row_id'] for r in rows]==[r['row_id'] for r in original]
    assert sum(r['state']==STATES['O'] for r in rows)==50 and sum(not r['original_input']['original_query_available'] for r in rows)==17
    assert Counter(r['SOURCE_role'] for r in rows)=={'SOURCE_FIT':80,'SOURCE_DEV':40}
    for x,y in zip(rows,original):
        assert x['original_input']==y['original_input'] and x['generated_query_xy']==y['generated_query_xy'] and x['L']==y['L'] and x['native_slot']==y['native_slot']
    out=save('ALL120_INTENDED_PART_OBSERVATIONS.json',dict(completed=True,PASS=True,reader_id='C',scope=proto['scope'],family_complete=bind(O/'COMPLETE.json'),protocol=bind(H/'PROTOCOL_BEFORE_ALL15_CONTACT_OBSERVATIONS.json'),view_receipt=views,rows120=rows,
        independence_other_reader_and_root_new_observation_reads=0,renderer_author_and_preexposure_disclosed=True,training_admission=False))
    columns=['row_id','frame_position','candidate_gt_list_index','native_slot','canonical_index','anatomical_family','SOURCE_role','state','optionalroi96','part_identity_evidence','occluder','foreign_evidence','ambiguity_reason']
    buf=io.StringIO(); writer=csv.DictWriter(buf,fieldnames=columns);writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(row[k],ensure_ascii=False) if k=='optionalroi96' else row[k] for k in columns})
    csv_b=save('ALL120_INTENDED_PART_OBSERVATIONS.csv',buf.getvalue())
    counts=dict(Counter(r['state'] for r in rows)); roles={role:dict(Counter(r['state'] for r in rows if r['SOURCE_role']==role)) for role in ['SOURCE_FIT','SOURCE_DEV']}
    qa=save('SAVED_OBSERVATION_SCHEMA_QA.json',dict(completed=True,PASS=True,rows120_exact=True,exact_all15_nominal_contacts_viewed=True,identity_q_L_native_original_U_roles_exact=True,
        candidate_original_U17_retained=True,OOF50_separate=True,ROI_inclusive96_valid=True,D_E_evidence_required=True,all_training_admission_false=True,
        all_jitter240_unassigned=True,counts=counts,by_role=roles,new_primary_native_gold=0,source_GT_or_model_protected_payload_reads=0))
    independence=save('INDEPENDENCE_READ0_RECEIPT.json',dict(completed=True,PASS=True,reader_id='C',own_observations_binding=out,A_new_observation_body_reads=0,root_new_observation_body_reads=0,
        own_actual_contact_views=15,preexposed_previous_context_and_renderer_code=True,blind_or_human_annotator_claim=False,GT_pose_nominal_origin_disclosed=True))
    note='C는 새 nominal contact 15개를 tools.view_image(original)로 모두 보았습니다. 120행을 원 row ID·FIT80/DEV40·candidate 반복 U17과 함께 보존했습니다. 원 unique96/U11은 별도이며 수정하지 않았습니다.\n\n'+json.dumps(counts,ensure_ascii=False,indent=2)+'\n\nD/E는 선언 candidate와 기대 DECK/BOTTOM_RUNNER의 유한 부위에 대한 assistant SOURCE 약관찰입니다. GT-derived q8과 nominal 형상 불확실성을 공개하며 정확한 물리 native gold나 원 detector→GT instance 대응 인증이 아닙니다. 다른 candidate를 best로 선택하지 않았습니다. 원 crop 중심 일치나 완벽 mesh를 요구하지 않았고, 단순 목재 면·블록·바닥·상자 존재만으로 의도 부위 D/E를 채우지 않았습니다.\n\n추가 jitter 240행으로 복사하지 않았고 모든 training admission은 false입니다. DEV는 향후 모든 fit/normalizer에서 제외해야 합니다. A/root 새 관찰은 봉인 전 읽지 않았습니다. C는 renderer 작성자이며 이전 맥락을 본 적이 있으므로 blind/human gold가 아닙니다. 별도 training PNG360/jitter240, 원 GT/NPZ/모델/보호 평가를 새로 열지 않았습니다.\n'
    ko=save('KO.md',note)
    result=save('COMPLETE.json',dict(completed=True,PASS=True,reader_id='C',observations=out,CSV=csv_b,view_receipt=views,schema_QA=qa,independence=independence,recording_code=bind(Path(__file__)),protocol=bind(H/'PROTOCOL_BEFORE_ALL15_CONTACT_OBSERVATIONS.json'),
        family_complete=bind(O/'COMPLETE.json'),counts=counts,by_role=roles,KO=ko,training_admission=False,new_primary_native_gold=False))
    print(json.dumps(dict(COMPLETE=result,counts=counts,by_role=roles),ensure_ascii=False))

if __name__=='__main__':main()
