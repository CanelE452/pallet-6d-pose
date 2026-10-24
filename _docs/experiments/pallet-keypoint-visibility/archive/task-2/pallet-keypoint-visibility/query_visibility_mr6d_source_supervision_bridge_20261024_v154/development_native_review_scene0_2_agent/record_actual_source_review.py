"""Record source-only structural weak review of MR6D scenes000000/000002.

No q, predictions, features, weights, masks or depth are loaded. The manual
physical-part families below summarize the agent's actual full-scene review of
all121 contact observations and29 additional full native overlays. Original
V74 supplied-pose correspondence is used only to name a source role consistently
across camera views; it does not select a query, an object, or a new permutation.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib, json

O = Path(__file__).resolve().parent
V = O.parent
P = V.parent
SRC = P / 'query_visibility_mr6d_full_public_source_20261024_v153'
R = SRC / 'native_reference_preparation/provided_reference_v1'
C = SRC / 'native_reference_preparation/source_reference_contact_v1'
POLICY_SHA = '3dafcb17115eed8d8ee232e7544132bbe0b0b0cbc601d81866f107329f245d72'
SCENES = {'000000', '000002'}
FULL_SC0 = [0,6,12,17,25,33,40,48,55,59,60,62,69,72]
FULL_SC2 = [0,7,9,15,19,20,23,29,30,34,37,41,42,44,47]
RAW_CASES = [('000000',6),('000000',17),('000000',33),
             ('000002',9),('000002',20),('000002',37),('000002',47)]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(p),'sha256':sha(p)}
def read(p): return json.loads(Path(p).read_text())
def write(p,obj):
    with Path(p).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def scene0_review(r):
    """Manual physical-part review of the two static, visibly distinct pallets.

    Fixed source part indices retain their physical identity as the original
    normal-facing native names change. This mapping is not a generic rule for
    another scene/object and never uses a visible-mask/pixel/proxy threshold.
    """
    k=int(r['frame_key']);ix=r['gt_instance_index'];c=r['fixed_canonical_index']
    if k>=61:
        return ('U','UNRESOLVED','PALLET_FRAGMENT_NOT_WHOLE_ENTITY',
          '후반61~72번은 팔레트가 대부분 또는 전부 화면 밖이다. 남는 극히 작은 목재 조각만으로 위/아래 팔레트의 정확한 하부 역할을 확정하지 않았다. 화면의 큰 회색 KLT는 별도 벤치 위 다른 물체이며 E로 복사하지 않는다.')
    if ix==0:
        if c==7 or (c==3 and k>=31):
            return ('U','UNRESOLVED','TARGET_OWN_DECK_SIDE_OR_GAP_UNCERTAIN',
              '윗팔레트의 먼 하부 역할은 자기 deck, 블록 측면과 틈이 겹친다. 목재 픽셀이나 명목 convex-self proxy로 보임/자기가림을 결정하지 않고 U를 유지했다.')
        if c in (2,3,6):
            return ('D','MEDIUM_STRUCTURAL_SIDE_END','NONE_SUPPORTED',
              '개별 윗팔레트의 외곽 상부 끝과 이어지는 세로 블록/runner 끝면을 실제 장면에서 식별했다. 다른 아래팔레트의 목재를 해당 역할로 복사한 것이 아니며 명목 투영점의 정확한 픽셀 일치는 주장하지 않는다.')
        return ('D','MEDIUM_TO_HIGH_STRUCTURAL','NONE_SUPPORTED',
          '적재물이 없는 윗팔레트 자체의 노출된 외곽 deck 끝과 연결된 판재 구조를 확인했다. 배경의 벤치 위 KLT는 해당 팔레트를 가리지 않는다. source 역할 D이지 정확한3px 좌표 gold가 아니다.')
    assert ix==1
    if k==60:
        return ('U','UNRESOLVED','STACK_FRAGMENT_ENTITY_HEIGHT_UNCERTAIN',
          '60번에는 화면 아래 작은 겹친 목재 조각만 남아 아래팔레트의 이 상부/하부 역할을 윗팔레트와 명확히 분리하기 어렵다. 다른 목재 픽셀을 D/E 근거로 삼지 않았다.')
    # The lower pallet protrudes at its near pair of corners, while the opposite
    # upper endpoints are visibly covered by the separate upper pallet.
    if c in (0,4):
        return ('E','MEDIUM_STRUCTURAL_FOREIGN_PALLET','OTHER_WOOD_PALLET_GT0',
          '아래팔레트 GT1의 먼 상부 외곽 끝은 별도 개체인 윗팔레트 GT0의 deck/body 아래로 들어간다. 두 팔레트의 돌출 폭과 높이 연결을 구분한 외부 가림 제안이며 같은 목재 색, mask0, 투영점의 재질만으로 정한 E가 아니다.')
    if c in (1,2,5,6):
        return ('D','MEDIUM_STRUCTURAL_DISTINCT_LOWER_PALLET','NONE_SUPPORTED',
          '윗팔레트보다 바깥으로 돌출된 아래팔레트 GT1 자체의 deck 끝 또는 전면 블록/runner 끝을 식별했다. 위/아래 팔레트의 겹침을 분리한 부분 역할 D이며 개체 전체8점이 모두 보인다는 뜻은 아니다.')
    return ('U','UNRESOLVED','UPPER_FOREIGN_PALLET_AND_OWN_GAPS_MIXED',
      '아래팔레트의 먼 하부 역할은 윗팔레트의 외부 가림과 아래팔레트 자체 deck/틈의 숨음이 섞인다. 정확한 물리점 또는 개체별 하부 구조를 분리할 근거가 부족해 U로 보존했다.')

def scene2_review(r):
    k=int(r['frame_key']);n=r['native_slot']
    if n<4:
        return ('D','MEDIUM_TO_HIGH_STRUCTURAL','NONE_SUPPORTED',
          '단일 적재 Euro 팔레트의 카메라 쪽 외곽 deck/전면 블록·runner 연결 끝이 노출돼 있다. 파란 KLT·회색 통은 deck 안쪽의 별도 물체이며 보이는 전면 끝 역할과 구분했다. 정확한 투영 오차나 query 품질 라벨은 아니다.')
    if n==4:
        if k<=8:
            return ('U','UNRESOLVED','LOW_ANGLE_KLT_BASE_ROLE_BOUNDARY',
              '낮은 시점에서 뒤쪽 상부 역할이 KLT 밑 경계와 겹친다. cm급 명목 pose로 실제 끝점이 노출됐는지 KLT 뒤에 있는지 결정하지 않았다. 파란 픽셀이라는 이유로 E를 만들지 않았다.')
        if 9<=k<=19 or 30<=k<=34 or k>=42:
            return ('D','MEDIUM_STRUCTURAL_WITH_NOMINAL_OFFSET','NONE_SUPPORTED',
              'KLT 바깥으로 나온 같은 팔레트의 외곽 목재 deck 끝과 이어지는 측면이 보인다. 원래 normal-facing 역할을 유지하며 명목 marker가 경계/배경 쪽에 남는 오차를 고치지 않고 구조적 노출만 제안했다.')
        blocker='GRAY_MOULDED_TUB' if 20<=k<=29 else 'BLUE_PLASTIC_KLT'
        return ('E','MEDIUM_STRUCTURAL',blocker,
          '전체 팔레트의 고정 외곽 역할과 적재물의 깊이 관계를 따라 보면 뒤쪽 상부 끝이 별도 통/KLT 뒤로 들어간다.20~29번은 원래 V74 앞면 선택이 바뀌어 같은 native4가 회색 통 쪽 역할이다. 순열을 추론점에 맞춘 것이 아니다.')
    if n==5:
        if k<=6 or k==41:
            return ('U','UNRESOLVED','TUB_OR_METAL_BIN_BOUNDARY_NOMINAL_UNCERTAINTY',
              '낮은 통 밑 경계 또는 금속 통 근처의 바깥 끝이 일부 보이나, 제공 pose의 cm급 오차 안에서 정확히 같은 rear 상부 역할의 노출을 확정하기 어렵다. 물체 픽셀과 역할 E를 동일시하지 않았다.')
        if 20<=k<=29:
            return ('D','MEDIUM_STRUCTURAL_WITH_NOMINAL_OFFSET','NONE_SUPPORTED',
              '누운 통 바깥쪽의 팔레트 외곽 상부 목재 끝이 노출돼 있다. 원래 깊이 edge 연결을 유지해 적재물의 실루엣과 팔레트 구조를 구분했다. 명목 ref의 수px~cm 오차를 좌표 보정하거나 학습 gold로 사용하지 않는다.')
        blocker='BLUE_PLASTIC_KLT' if k>=42 else 'GRAY_MOULDED_TUB'
        return ('E','MEDIUM_STRUCTURAL',blocker,
          '원래 역할 연결을 유지하면 먼 상부 끝이 누운 회색 통 또는 KLT 뒤에 가려진다.42번 이후의 앞면 전환으로 native5는 KLT 쪽의 먼 상부 끝이 된다. mask나 query 픽셀의 색으로 정한 E가 아니다.')
    return ('U','UNRESOLVED','OWN_DECK_AND_EXTERNAL_CARGO_OR_GAPS_MIXED',
      '뒤쪽 하부 역할은 팔레트 자신의 deck/블록·틈과 KLT/통 가림을 명확히 분리하기 어렵다. 다른 면의 목재나 적재물 픽셀, self proxy를 자동 라벨로 쓰지 않고 U를 유지했다.')

def main():
    policy=V/'ROOT_POLICY_BEFORE_SOURCE_BRIDGE_REVIEW.json';assert sha(policy)==POLICY_SHA
    rp=R/'REFERENCE_PROJECTION_COMPLETE.json';cp=C/'CONTACT_RENDER_COMPLETE.json'
    assert read(rp)['completed'] and read(rp)['PASS'] and read(cp)['completed'] and read(cp)['PASS']
    reference_path=R/'ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl'
    refs=[json.loads(x) for x in reference_path.read_text().splitlines()]
    index=read(C/'ALL230_SOURCE_REFERENCE_CONTACT_INDEX.json');frame_by={x['frame_index']:x for x in index['frames']}
    selected=[(i,r) for i,r in enumerate(refs) if r['scene'] in SCENES]
    assert len(selected)==1552
    rows=[];manual_instances=defaultdict(list);per_frame=defaultdict(Counter)
    full_global=set(FULL_SC0+[118+k for k in FULL_SC2])
    for gi,r in selected:
        rr=dict(r);fi=r['frame_index'];n=r['native_slot'];k=int(r['frame_key']);ix=r['gt_instance_index']
        assert r['split']=='SOURCE_DEVELOPMENT'
        if not r['continuous_inframe']:
            state,confidence,blocker,reason=('OOF','NOMINAL_SOURCE_BOUNDARY','OUTSIDE_IMAGE',
              '提供 pose/CAD의 해당 명목 역할이 원본 장면 경계 밖이다. 실제 접촉시트의 clipping 맥락도 확인했다. OOF는 외부가림 E나 binary 음성이 아니며 cm급 reference 불확실성을 남긴다.')
        else:
            state,confidence,blocker,reason=(scene0_review(r) if r['scene']=='000000' else scene2_review(r))
        entity=('UPPER_PHYSICAL_PALLET_GT0' if ix==0 else 'LOWER_PHYSICAL_PALLET_GT1') if r['scene']=='000000' else 'SINGLE_LOADED_PHYSICAL_PALLET_GT0'
        rr.update(reference_global_index=gi,proposal_state=state,proposal_confidence=confidence,
          review_reason_KO=reason,occluder_or_uncertainty=blocker,target_material='WOOD_EURO_PALLET',
          semantic_state_original_preserved=r['semantic_state'],source_entity_review=entity,
          actual_visual_review=True,actual_view_scope='CONTACT_ALL_FRAMES_AND_SELECTED_FULL_IMAGES',
          this_frame_full_native_overlay_also_viewed=fi in full_global,
          evidence_full_native_overlay=frame_by[fi]['all_instance_overlay'],raw_RGB=frame_by[fi]['raw_RGB'],
          actual_query_entity_correspondence='NOT_READ_NOT_ASSUMED',actual_query_availability='NOT_READ',
          human_gold=False,train_ready=False,assistant_weak_proposal=True,root_adjudication_pending=True,
          exact_reference_pixel_gold=False,coordinate_supervision_ready=False,
          native_order_or_provided_pose_or_reference_changed=False,
          source_native_visibility_candidate_only=state in ('D','E'),
          input_or_head_score_used_for_label=False,mask_used_to_autofill_label=False,
          convex_self_proxy_used_to_autofill_label=False)
        rows.append(rr);manual_instances[(fi,ix)].append(state);per_frame[fi][state]+=1
    assert len(rows)==1552 and len(manual_instances)==194 and len(per_frame)==121
    counts=Counter(r['proposal_state'] for r in rows)
    assert counts==Counter({'D':536,'E':153,'OOF':561,'U':302})
    assert all(rr[k]==r[k] for rr,(_,r) in zip(rows,selected) for k in r)
    assert [r['reference_global_index'] for r in rows]==sorted(r['reference_global_index'] for r in rows)
    # Record explicit per-instance native0..7 manual decisions for reviewer access.
    annotations=[]
    for (fi,ix),states in manual_instances.items():
        assert len(states)==8
        annotations.append({'frame_index':fi,'id':frame_by[fi]['id'],'source_id':frame_by[fi]['source_id'],
          'scene':frame_by[fi]['scene'],'frame_key':frame_by[fi]['frame_key'],'gt_instance_index':ix,
          'native0_to7_proposals':states,'all8_retained':True,'human_gold':False,'train_ready':False})
    write_rows(O/'ALL1552_DEVELOPMENT_SCENE0_2_NATIVE_ROLE_PROPOSALS_AGENT.jsonl',rows)
    write(O/'PER_INSTANCE194_EXPLICIT_NATIVE_STATES.json',{'completed':True,'rows':annotations})
    frame_counts=[{'frame_index':fi,'source_id':frame_by[fi]['source_id'],'scene':frame_by[fi]['scene'],
      'gt_instance_indices':frame_by[fi]['obj1_gt_instance_indices'],'roles':sum(c.values()),
      'state_counts':dict(c)} for fi,c in per_frame.items()]
    write(O/'PER_FRAME121_COUNTS.json',{'completed':True,'frames':frame_counts})
    pages=[x for x in index['pages'] if x['split']=='SOURCE_DEVELOPMENT' and any(i in per_frame for i in x['frame_indices'])]
    assert len(pages)==16
    views={'completed':True,'all_assigned_frames_seen_in_contacts':121,
      'contacts_actually_viewed':[x['contact'] for x in pages],
      'contact_frame_indices':[x['frame_indices'] for x in pages],
      'assigned_frames_only':[fi for fi in per_frame],
      'extra_unassigned_scene5_in_last_mixed_page_not_annotated':True,
      'full_native_overlays_actually_viewed':[frame_by[fi]['all_instance_overlay'] for fi in sorted(full_global)],
      'raw_original_JPEGs_actually_viewed':[frame_by[k if sc=='000000' else 118+k]['raw_RGB'] for sc,k in RAW_CASES],
      'full_native_count':29,'raw_count':7,
      'did_not_claim_all121_full_images_individually_viewed':True,
      'model_query_feature_score_mask_depth_body_read':False,
      'reviewer':'source assistant actual view_image; not human annotator'}
    write(O/'ACTUAL_VISUAL_VIEW_RECORD.json',views)
    by_scene={s:dict(Counter(r['proposal_state'] for r in rows if r['scene']==s)) for s in sorted(SCENES)}
    by_target={f"{s}:GT{ix}":dict(Counter(r['proposal_state'] for r in rows if r['scene']==s and r['gt_instance_index']==ix))
      for s,ix in [('000000',0),('000000',1),('000002',0)]}
    result={'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),
      'code':bind(Path(__file__)),'root_policy':bind(policy),
      'source_reference_seal':bind(rp),'source_reference_rows':bind(reference_path),'source_contact_seal':bind(cp),
      'roles_packet':bind(O/'ALL1552_DEVELOPMENT_SCENE0_2_NATIVE_ROLE_PROPOSALS_AGENT.jsonl'),
      'explicit_instance_states':bind(O/'PER_INSTANCE194_EXPLICIT_NATIVE_STATES.json'),
      'per_frame_counts':bind(O/'PER_FRAME121_COUNTS.json'),'actual_view_record':bind(O/'ACTUAL_VISUAL_VIEW_RECORD.json'),
      'counts':{'frames':121,'GT_instance_observations_not_unique_physical_objects':194,'roles':1552,
                'state_counts':dict(counts),'by_scene':by_scene,'by_GT_target':by_target,
                'source_role_D_E_proposals_before_actual_query_association':counts['D']+counts['E']},
      'all_original_native_reference_fields_and_order_preserved':True,
      'GT_role_and_actual_R0_bridge_are_separate':True,
      'actual_query_selected_entity_role_or_crops_read':False,'visibility_head_scores_read':False,
      'feature_model_fit_PnP_private_access':0,'human_gold':False,'train_ready':False,
      'root_independent_adjudication_pending':True,
      'limits_KO':[
       '이689개 D/E는 제공 GT 개체의 역할 weak제안이며 actual R0 query/crop689개와 대응됐다는 뜻이 아니다. 다른 개체·stack-composite query로 복사 금지다.',
       '모든121장 contact를 실제봤지만 fullnative 개별추가view는29장, rawJPEG 추가view는7장이다. 보지 않은 개별 fullimage를 봤다고 주장하지 않는다.',
       '위/아래 팔레트는 두 개체로 분리했다. 아래팔레트의 숨은 하부점은 외부/자기/틈 구분이 불명확해 U이며 S를 자동생성하지 않았다.',
       '의미 역할·물리 개체·명목 좌표 정확도·입력 availability는 별도다. source nominal pose/CAD는 정밀3px 실측교사가 아니다.',
       '동일 연구실·Euro목재 형상의 반복 관측이다. 관측194개를 독립 physical pallet194개라고 주장하지 않는다.',
       'sourceOOF는 제공명목투영과 실제frame clipping을 함께기록한 항목이며 binary E/negative에 포함하지 않는다.',
       '이전 V151/V152/V153의 immutable source reference와 제안은 수정하지 않았다. 새 전체개발범위의 독립 weak검토는 root판정이 필요하다.']}
    write(O/'SOURCE_ROLE_REVIEW_COMPLETE.json',result)
    print(json.dumps({'seal':bind(O/'SOURCE_ROLE_REVIEW_COMPLETE.json'),'counts':result['counts']}))

if __name__=='__main__':main()
