"""Record actual visual source-annotation proposals; no automatic visibility model.

The 45 explicit native-state rows below are the assistant's full-scene review,
not predictions or rules obtained from masks/proxy/coordinates. Source geometry
is retained as separate nominal evidence. Root adjudication remains separate.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json

H = Path(__file__).resolve().parent
V = H.parent
R = H / 'provided_reference_v1'
C = H / 'source_reference_contact_v1'
O = H / 'source_validation_visual_review_agent'

# Explicit manual frame_key -> native0..7 decisions after all45 full PNGs viewed.
# O is nominal source role outside the scene, never an external negative.
MANUAL = {
 0:'OOOOEEUU', 1:'DDOOEEUU', 2:'DDOOEEUU', 3:'DDOOEEUU', 4:'DDOOEEUU',
 5:'DDDDEEUU', 6:'DDDDEEUU', 7:'DDDOEEUU', 8:'DDOOEEUU', 9:'ODOOEEUU',
 10:'ODOOEEUU', 11:'ODOOEEUU', 12:'DDDOEEUU', 13:'DDDOEEUU', 14:'DDDOEEUU',
 15:'DDDOEEUU', 16:'DDDOUEUU', 17:'DOODUEUU', 18:'DDDDDEUS', 19:'DDDODEUS',
 20:'OODODEUS', 21:'ODOODEUS', 22:'ODOODEUS', 23:'OOOODEUS', 24:'OOOODEUS',
 25:'OOOODEUS', 26:'OOOODEUS', 27:'DOODEUUU', 28:'DOOOEUUU', 29:'DOODEUUU',
 30:'DOODEUUU', 31:'DOODEUUU', 32:'DOODEUUU', 33:'DOODEUUU', 34:'DOODEDUU',
 35:'DDODEDUU', 36:'DOODEDDU', 37:'DDODEDDU', 38:'DOODEDDU', 39:'ODDODEUD',
 40:'ODDODEUD', 41:'ODOODEUD', 42:'ODDODEUD', 43:'ODDODEUD', 44:'ODDODEUD'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(p), 'sha256':sha(p)}
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def decision(k,n,state):
    if state=='OOF':
        return ('HIGH_CONTEXT_NOMINAL_BOUNDARY_NOT_PIXEL_GOLD','OUTSIDE_IMAGE',
          '제공 pose의 해당 역할이 장면 경계 밖으로 나가며 원본에서도 해당 끝점 전체를 확인할 수 없다. OOF는 외부가림 E나 음성 학습 라벨이 아니다. 경계 근처의 cm급 불확실성은 남긴다.')
    if n<4:
        conf='MEDIUM' if k in [9,10,12,13,40] else 'HIGH_STRUCTURAL'
        return (conf,'NONE_SUPPORTED',
          '단일 Euro 팔레트의 카메라 쪽 외곽 deck/하부 block·runner 연결 끝점이 노출돼 있다. 명목 marker의 정확한 픽셀 일치가 아니라 같은 순서의 구조적 역할이 보임을 제안한다.')
    if n==4:
        if state=='E':
            return ('MEDIUM_STRUCTURAL','WICKER_BASKET',
              '전체 팔레트 폭·깊이의 연결을 따라 본 뒤쪽 상부 끝점이 팔레트 위의 별도 바구니 뒤에 가려진다. 바구니 픽셀이나 mask0만으로 정한 E가 아니다.')
        if state=='D':
            return ('MEDIUM_STRUCTURAL','NONE_SUPPORTED',
              '바구니 바깥으로 나온 같은 팔레트의 외곽 상부 deck 끝/이어지는 측면을 확인했다. 명목 투영 오차를 보정하지 않고 노출된 역할 D로 제안한다.')
        return ('UNRESOLVED','BASKET_BOUNDARY_VS_EXPOSED_WOOD',
          '바구니 실루엣과 작은 목재 돌출 끝이 맞닿아, 제공 pose의 cm급 오차 안에서 같은 rear 상부 끝점의 노출/외부가림을 확정하기 어렵다.')
    if n==5:
        if state=='E':
            blocker='WICKER_BASKET' if k>=39 else 'SLATTED_WOOD_CRATE'
            return ('MEDIUM_STRUCTURAL',blocker,
              '원래 고정 역할·연결 edge를 유지하면 먼 상부 끝점이 별도 적재물 뒤로 들어간다. 39번 이후 시점에서는 native 앞면 선택이 바뀌므로 이 역할의 가림 물체는 바구니다. 순열을 새로 맞춘 것이 아니다.')
        if state=='D':
            return ('MEDIUM_WITH_NOMINAL_OFFSET','NONE_SUPPORTED_METAL_BIN_INSIDE_DECK',
              '목재 상자/금속 통의 바깥쪽으로 같은 팔레트의 먼 외곽 목재 끝이 노출돼 있다. 통은 deck 안쪽에 있고 실제 외곽 끝을 모두 덮지 않는다. marker가 배경 쪽에 남는 오차와 역할 D를 분리한다.')
        return ('UNRESOLVED','WOOD_CRATE_METAL_BIN_ROLE_BOUNDARY',
          '목재 상자 및 금속 통 근처의 외곽 footprint 끝이 일부 보이나, 어느 면의 정확한 rear 상부 끝까지 노출됐는지 cm급 명목 투영만으로 결정하지 않았다.')
    if n==6:
        if state=='D':
            return ('LOW_TO_MEDIUM_STRUCTURAL','NONE_SUPPORTED_METAL_BIN_INSIDE_DECK',
              '36~38번의 측면 관측에서 통 바깥쪽 팔레트의 외곽 하부 측면/끝 연결이 노출된다. 자기 표면 위의 목재 픽셀을 D로 삼은 것이 아니며, 이 세 제안은 root 독립 확인이 특히 필요하다.')
        return ('UNRESOLVED','OWN_DECK_AND_EXTERNAL_CARGO_MIXED',
          '먼 하부 끝은 상부 적재물·팔레트 자신의 deck/블록 및 틈의 관계가 겹친다. 색/목재 존재나 convex proxy만으로 D/E/S를 나누지 않고 U를 유지했다.')
    if n==7:
        if state=='S':
            return ('MEDIUM_STRUCTURAL_SELF_ONLY','TARGET_OWN_SOLID_DECK',
              '18~26번은 같은 rear 상부 끝(native4)의 노출을 보며, 그 아래 하부 끝은 팔레트 자신의 연속된 상부 deck 뒤에 숨는다. 외부 바구니 색이나 self proxy의 자동 라벨이 아닌 구조적 자기 가림 제안이다.')
        if state=='D':
            return ('MEDIUM_STRUCTURAL','NONE_SUPPORTED',
              '앞면 선택이 바뀐 후 같은 팔레트의 왼쪽 외곽 측면 하부 끝이 상부 끝(native4)과 함께 노출된다. 원래 V74 dynamic-to-fixed 대응을 그대로 유지했다.')
        return ('UNRESOLVED','BASKET_OR_SELF_OR_GAPS',
          '바구니의 가림과 팔레트 자체 상부 deck/뒤쪽 하부 구조의 숨음이 섞여 있다. 투영점의 재질이나 mask 값으로 실제 하부 역할의 상태를 결정하지 않았다.')
    raise AssertionError(n)

def main():
    assert set(MANUAL)==set(range(45)) and all(len(x)==8 for x in MANUAL.values())
    projection=read(R/'REFERENCE_PROJECTION_COMPLETE.json');contact=read(C/'CONTACT_RENDER_COMPLETE.json')
    assert projection['completed'] and projection['PASS'] and contact['completed'] and contact['PASS']
    index=read(C/'ALL230_SOURCE_REFERENCE_CONTACT_INDEX.json')
    frame_by={x['frame_index']:x for x in index['frames']}
    refs=[json.loads(x) for x in (R/'ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl').read_text().splitlines()]
    O.mkdir(exist_ok=False)
    rows=[];all_rows=[];frames=[]
    for ref in refs:
        rr=dict(ref)
        if ref['split']!='SOURCE_VALIDATION':
            rr.update(proposal_state='U_UNREVIEWED',actual_visual_review=False,
                      review_reason_KO='개발용 185장은 이 검증용 검토 단계에서 아직 개별 주석하지 않았다.',
                      human_gold=False,train_ready=False)
            all_rows.append(rr);continue
        k=int(ref['frame_key']);n=ref['native_slot'];fi=ref['frame_index'];assert fi==73+k and ref['gt_instance_index']==0
        state={'O':'OOF'}.get(MANUAL[k][n],MANUAL[k][n]);conf,blocker,reason=decision(k,n,state)
        if state=='OOF':assert not ref['continuous_inframe']
        else:assert ref['continuous_inframe']
        rr.update(proposal_state=state,semantic_state_original_preserved=ref['semantic_state'],
          actual_visual_review=True,proposal_confidence=conf,occluder_or_uncertainty=blocker,
          review_reason_KO=reason,source_entity_review='SINGLE_PHYSICAL_FLAT_EURO_PALLET_GT0_WITH_BASKET_AND_WOOD_CRATE',
          actual_query_entity_correspondence='NOT_READ_NOT_ASSUMED',actual_query_availability='NOT_READ',
          human_gold=False,train_ready=False,assistant_weak_proposal=True,
          exact_reference_pixel_gold=False,coordinate_supervision_ready=False,
          native_order_or_provided_pose_or_reference_changed=False,
          source_native_visibility_candidate_only=state in ['D','E'],
          evidence_full_native_overlay=frame_by[fi]['all_instance_overlay'],
          raw_RGB=frame_by[fi]['raw_RGB'],
          input_or_head_score_used_for_label=False,mask_used_to_autofill_label=False,
          convex_self_proxy_used_to_autofill_label=False)
        rows.append(rr);all_rows.append(rr)
    assert len(rows)==360 and len(all_rows)==2424
    counts=Counter(x['proposal_state'] for x in rows)
    assert counts==Counter({'D':116,'E':61,'S':9,'OOF':93,'U':81})
    for k in range(45):
        fi=73+k;f=frame_by[fi]
        frames.append({'frame_index':fi,'frame_key':str(k),'source_id':f['source_id'],'gt_instance_index':0,
                       'native0_to7_proposals':[{'O':'OOF'}.get(x,x) for x in MANUAL[k]],
                       'full_native_overlay_actually_viewed':f['all_instance_overlay']})
    viewed_pages=[x for x in index['pages'] if x['split']=='SOURCE_VALIDATION']
    extra_raw_keys=[24,31,34,36,38,39]
    actual_view_record={'completed':True,'all45_full_native_images_actually_viewed':True,
       'all45_frames_covered_by_six_contact_pages_actually_viewed':True,
       'full_native_images':[frame_by[73+k]['all_instance_overlay'] for k in range(45)],
       'contact_pages':[x['contact'] for x in viewed_pages],
       'extra_raw_RGB_actually_viewed':[frame_by[73+k]['raw_RGB'] for k in extra_raw_keys],
       'method':'Actual view_image observations in this agent; no head scores, masks, q, model calls or private images',
       'reviewer':'source assistant; not human annotator'}
    write_rows(O/'SOURCE_VALIDATION360_NATIVE_ROLE_PROPOSALS_AGENT.jsonl',rows)
    write_rows(O/'ALL2424_NATIVE_ROLE_PROPOSALS_PARTIAL_AGENT.jsonl',all_rows)
    write(O/'VALIDATION45_PER_FRAME_REVIEW.json',{'completed':True,'frames':frames})
    write(O/'ACTUAL_VISUAL_VIEW_RECORD.json',actual_view_record)
    correction={'completed':True,'original_completed_contact_seal':bind(C/'CONTACT_RENDER_COMPLETE.json'),
       'counts_from_actual_pages':{'SOURCE_VALIDATION':6,'SOURCE_DEVELOPMENT':24,'total':30},
       'correction':'presentation_storage prose said29 while actual pages[] and generated files correctly contain30; split boundary is not mixed. No pixel/output/seal overwritten.',
       'native_full_overlays':230,'redundant_raw_paired_or_solo_PNGs':0}
    write(O/'CONTACT_PAGE_COUNT_PROSE_CORRECTION.json',correction)
    summary={'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),
      'code':bind(Path(__file__)),'root_policy':bind(V/'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json'),
      'source_projection':bind(R/'REFERENCE_PROJECTION_COMPLETE.json'),
      'source_contact':bind(C/'CONTACT_RENDER_COMPLETE.json'),
      'source_intake':bind(V/'full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json'),
      'validation_roles':bind(O/'SOURCE_VALIDATION360_NATIVE_ROLE_PROPOSALS_AGENT.jsonl'),
      'full2424_partial_inventory':bind(O/'ALL2424_NATIVE_ROLE_PROPOSALS_PARTIAL_AGENT.jsonl'),
      'per_frame':bind(O/'VALIDATION45_PER_FRAME_REVIEW.json'),'actual_views':bind(O/'ACTUAL_VISUAL_VIEW_RECORD.json'),
      'contact_count_correction':bind(O/'CONTACT_PAGE_COUNT_PROSE_CORRECTION.json'),
      'counts':dict(counts),'validation_frames':45,'validation_roles':360,
      'development_roles_preserved_unreviewed':2064,
      'D_E_weak_candidate_roles_before_actual_query_association':177,
      'all_states_and_reference_rows_retained':True,'visibility_model_scores_read':False,
      'query_outputs_read':False,'new_model_fit_PnP_private_access':0,
      'human_gold':False,'train_ready':False,'root_adjudication_pending':True,
      'scientific_limits_KO':[
       '공개 단일 Euro 형상과 같은 연구실 적재물 범위이며 정사각형·야외·사람 가림 다양성을 대신하지 않는다.',
       'source held validation의 역할 라벨 제안은 새 headfit 전 봉인한다. 실제 R0가 다른 물체를 선택하면 이 역할 라벨을 복사하면 안 된다.',
       '제공 pose/CAD는 cm급 명목 reference이고 <=3px 실측 정답이나 정확한 좌표 학습 교사가 아니다.',
       'OOF/S/U는 외부가림 binary D/E 밖에 보존한다. S9는 직접 구조 검토 제안이며 convex proxy와 별도다.',
       '34~38 rear5 및36~38 rear6 노출 제안은 통/목재상자 경계와 명목 오차를 분리한 약한 역할 판정이며 root 독립검토가 필요하다.',
       '전체230/303/2424 분모는 그대로이며 개발185의 주석은 후속 단계다.']}
    write(O/'VALIDATION_STRUCTURAL_REVIEW_COMPLETE.json',summary)
    print(json.dumps({'seal':bind(O/'VALIDATION_STRUCTURAL_REVIEW_COMPLETE.json'),'counts':dict(counts),'roles':360}))

if __name__=='__main__':main()
