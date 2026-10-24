"""Serialize the AI visual observations made on all8 locked source frames."""
from pathlib import Path
import collections,csv,hashlib,json
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent

# These observations were entered only after viewing the locked full-frame,
# marked crop and untouched-raster native96 crop sheets. No mask was opened.
OBS=[
 [
  ('D_candidate','nearby_physical_corner','medium','왼쪽 상부 wood deck 끝의 윤곽 꺾임이 보인다. 팔레트가 작고 어두워 exact pixel 일치까지 확인하지 못했다.'),
  ('D_candidate','nearby_physical_corner','low','앞쪽 deck/지지부 끝 부근에서 팔레트 모서리 단서가 보인다. 적재물과 낮은 해상도 때문에 native corner 대응은 조건부이다.'),
  ('ambiguous','support_floor_boundary_unresolved','low','아래쪽 projected r이 지지부 하단/바닥 경계 부근이다. 이 해상도에서 독립된 물리 vertex 대응을 분리하기 어렵다.'),
  ('virtual-no-cue','background_near_empty_envelope_corner','medium','왼쪽 아래 cuboid 끝은 얇은 deck 아래 바닥 쪽에 놓인다. 해당 r에서 wood 지지부의 물리적 모서리가 보이지 않는다. 빈 cuboid 외곽점 후보이며 absence GT는 아니다.'),
  ('ambiguous','load_deck_junction_unresolved','low','뒤쪽 위 r은 carton 하단과 deck 근처에 중첩된다. 뚜렷한 독립 pallet corner가 보이지 않아 D/E를 관찰만으로 확정하지 않는다.'),
  ('D_candidate','nearby_physical_corner','medium','오른쪽 wood deck 끝의 상부 외곽 꺾임이 보인다. 작은 투영 크기로 subpixel 대응은 미확정이다.'),
  ('ambiguous','far_lower_endpoint_unresolved','low','오른쪽 하단 pallet 끝 부근이지만 지지부 형상과 바닥의 정확한 corner 대응이 해상도상 불명확하다.'),
  ('S_proxy_only','other_target_surface_at_projected_hidden_corner','low','기존 convex self proxy는 hidden이다. r 투영에는 앞쪽 fork opening/wood 지지부가 보이며, 이것을 뒤쪽 아래 reference의 직접 가시성으로 바꿀 수 없다.'),
 ],
 [
  ('virtual-no-cue','ground_shadow_no_target_corner','medium','r0는 검은 bin 왼쪽 바닥/그림자 경계에 있다. 인접 wood pallet는 보이지만 이 위치에 고유한 wood corner cue가 없다. ray-proxy D와 물리 corner 관찰은 다르다.'),
  ('D_candidate','nearby_physical_corner','medium','오른쪽 wood deck 전면 끝의 모서리/세로 지지면이 보인다. q는 r에서 약6px 떨어져 있어 둘의 exact occupancy를 동일시하지 않는다.'),
  ('D_candidate','nearby_physical_corner','medium','오른쪽 아래 wood 지지부와 바닥의 끝 모서리가 보인다. round/bevel와 낮은 광량 때문에 exact pixel ownership은 별도 미확정이다.'),
  ('virtual-no-cue','ground_shadow_no_target_corner','medium','r3는 bin 옆 바닥/그림자에 놓인다. wood pallet lower envelope corner에 해당하는 물리 vertex가 관찰되지 않는다.'),
  ('E_candidate','distinct_external_occluder','medium','기준 위치는 검은 별도 bin 표면 안에 있다. 주변 wood pallet와 구분되는 외부 물체가 뒤쪽 pallet reference 영역을 덮는 후보이다. 해당 reference 형상의 실재는 RGB만으로 증명하지 못한다.'),
  ('ambiguous','out_of_frame','high','r5와 q5 모두 이미지 오른쪽 밖이다. padding을 물체 부재나 가림 라벨로 사용하지 않는다.'),
  ('ambiguous','out_of_frame','high','r6와 q6 모두 이미지 오른쪽 밖이다. semantic D/E/S 관찰 불가.'),
  ('S_proxy_only','external_occluder_also_visible','low','convex self proxy가 hidden이고 투영 위치에는 검은 bin도 보인다. self와 external 원인을 RGB 한 장으로 분리하지 못한다.'),
 ],
 [
  ('ambiguous','out_of_frame','high','r0/q0가 왼쪽 이미지 밖이다. 목재가 crop 일부에 보여도 reference 자체의 가시성을 정하지 않는다.'),
  ('D_candidate','nearby_physical_corner','medium','앞쪽 wood deck 오른쪽 끝에서 상판과 세로 지지면의 물리적 꺾임이 보인다.'),
  ('D_candidate','nearby_physical_corner','low','앞쪽 오른쪽 wood 지지부의 하단 끝이 보인다. r 근처 cue는 있으나 최하단 bbox point와 실제 목재 vertex의 exact 일치는 확정하지 못했다.'),
  ('ambiguous','out_of_frame','high','r3/q3가 왼쪽 이미지 밖이다. 시야 밖 상태를 유지한다.'),
  ('E_candidate','distinct_external_load','medium','뒤쪽 reference는 carton 표면 안에 투영된다. 목재 대신 적재 상자가 보이므로 외부 덮임 후보로 기록하되 teacher 후보만이다.'),
  ('virtual-no-cue','ground_beyond_slatted_mesh_extremum','medium','r5 중심은 wood slat 끝 주변 바닥에 놓이고, r에서 고유한 물리적 pallet corner가 보이지 않는다. bbox envelope와 실제 slat 형상 사이 빈 영역 후보이다.'),
  ('virtual-no-cue','ground_beyond_slatted_mesh_extremum','medium','r6 아래쪽 extremum 역시 wood 끝 옆 바닥에 놓인다. 보이는 인접 slat를 이 lower reference 자체로 바꾸지 않는다.'),
  ('S_proxy_only','external_load_also_visible','low','뒤쪽 lower self proxy이며 carton도 투영 위치를 덮는다. hidden reference의 직접 cue가 없고 S/E 복합 원인 미해결이다.'),
 ],
 [
  ('D_candidate','nearby_physical_corner','medium','왼쪽 앞 상부 plastic deck 외곽의 꺾임이 보인다. r은 rounded/mesh 외곽의 근처여서 exact material pixel을 뜻하지 않는다.'),
  ('D_candidate','nearby_physical_corner','medium','가까운 전면 우상부 deck/body 꺾임이 뚜렷하다. q는 다른 쪽으로 크게 이동해 있다.'),
  ('D_candidate','nearby_physical_corner','medium','가까운 전면 우하부 body 끝/바닥 경계가 보인다. geometry envelope와 bevel 대응은 조건부이다.'),
  ('D_candidate','nearby_physical_corner','medium','왼쪽 lower plastic body 끝이 보인다. 작은 bevel 때문에 raster 한 점의 material 여부와 corner 의미를 분리한다.'),
  ('ambiguous','far_upper_envelope_near_bin_edge','low','뒤쪽 위 r은 bin 옆 바닥/상판 외곽 근처이다. 고유한 physical vertex가 분리되지 않아 D나 external cover를 확정하지 않는다.'),
  ('E_candidate','distinct_external_occluder','high','검은 독립 bin이 projected reference 위치를 덮는다. 주변 팔레트 상판과 별도 물체의 경계가 분명하다.'),
  ('E_candidate','distinct_external_occluder','high','뒤쪽 아래 reference 위치도 검은 bin 표면에 놓여 있다. 단순 검은 pallet 구멍으로 해석할 수 없다.'),
  ('S_proxy_only','visible_top_grid_is_other_surface','low','self-hidden lower reference가 상판 grid 쪽에 투영된다. 그 위치의 상판 픽셀은 lower corner가 보인다는 증거가 아니다.'),
 ],
 [
  ('D_candidate','nearby_rounded_physical_corner','high','가까운 왼쪽 상부 plastic body/상판의 rounded 외곽 모서리가 명확하다. 수학 cuboid vertex는 bevel 바깥일 수 있다.'),
  ('D_candidate','nearby_rounded_physical_corner','high','오른쪽 상판/body 외곽 모서리가 보이고 occluder가 없다. exact pixel 인증은 하지 않는다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','오른쪽 아래 body 끝의 rounded corner가 보인다. reference는 부드러운 외곽 근처이다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','가까운 왼쪽 아래 body 끝이 보인다. raster의 r 자체는 bevel 바깥/바닥일 수 있어 semantic cue와 pixel label을 구분한다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','뒤쪽 왼쪽 top 외곽의 rounded 꺾임이 보인다. r은 material 외곽과 완전히 일치하는 vertex라기보다 envelope 근처이다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','먼 쪽 상판 외곽 모서리 cue가 보인다. top grid/rounded outline로 위치 의미가 관찰 가능하다.'),
  ('S_proxy_only','visible_top_grid_is_other_surface','medium','convex self-hidden lower far reference는 상판 내부에 투영된다. 눈에 보이는 grid를 해당 lower corner의 직접 가시성으로 사용하지 않는다.'),
  ('virtual-no-cue','background_near_rounded_missing_lower_corner','medium','뒤쪽 왼쪽 lower envelope r은 외곽 옆 바닥으로 빠진다. 가까운 측면은 보이나 해당 full cuboid lower vertex의 물리 cue는 관찰되지 않는다.'),
 ],
 [
  ('ambiguous','out_of_frame','high','r0/q0가 이미지 왼쪽 밖이다. 바깥 부분의 visibility를 추론 라벨로 채우지 않는다.'),
  ('D_candidate','nearby_physical_corner','high','전면 오른쪽 upper plastic body/deck 모서리가 보인다. q도 근처이나 r과 q를 동일 좌표로 덮어쓰지 않았다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','전면 오른쪽 body 하단 끝이 바닥과 만나는 cue가 보인다. bbox와 둥근 외곽 exact 일치는 미인증이다.'),
  ('ambiguous','out_of_frame','high','r3/q3가 이미지 왼쪽 밖이다. 시야 밖 ambiguity를 유지한다.'),
  ('E_candidate','load_deck_boundary','low','carton의 하단이 r 부근 deck 위를 덮는다. r가 carton/deck 경계와 가까워 raster 정합·가림 원인 확정은 보류하는 조건부 E 후보이다.'),
  ('D_candidate','nearby_physical_corner','medium','오른쪽 먼 deck/body 끝의 상부 꺾임이 보인다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','오른쪽 먼 body 외곽 끝이 보이지만 둥근 형상이라 수학 lower vertex와 exact 일치는 미확정이다.'),
  ('S_proxy_only','fork_opening_other_surface_projection','low','self-hidden back lower reference 투영은 front fork opening 안이다. 구멍이나 주변 front face가 보이는 것으로 뒤쪽 reference visibility를 확정할 수 없다.'),
 ],
 [
  ('D_candidate','nearby_physical_corner','high','가까운 green plastic top/body 모서리의 꺾임이 뚜렷하다.'),
  ('D_candidate','nearby_physical_corner','high','오른쪽 top/body 외곽 끝이 보인다. carton과 구분된다.'),
  ('D_candidate','nearby_physical_corner','medium','오른쪽 lower support/body 끝의 cue가 보인다. rounded outline 때문에 exact r material 여부는 별도이다.'),
  ('D_candidate','nearby_physical_corner','medium','가까운 lower support의 바닥 끝이 보인다. 구멍을 full solid cuboid로 간주하지 않는다.'),
  ('D_candidate','nearby_physical_corner','high','왼쪽 먼 top 외곽의 green pallet corner가 명확하다.'),
  ('E_candidate','distinct_external_load','medium','뒤쪽 top reference는 carton 안에 투영된다. green pallet가 아니라 적재 상자가 덮는 후보이다.'),
  ('S_proxy_only','external_load_also_visible','low','self-hidden lower far reference이며 projected 위치는 carton 표면이다. self/external cause 분리는 proxy만으로 인증하지 않는다.'),
  ('virtual-no-cue','ground_beside_open_or_rounded_lower_extent','medium','왼쪽 먼 lower cuboid point는 green pallet 옆 바닥에 놓인다. 실제 lower support corner가 그 r에 관찰되지 않아 virtual/no-cue 후보로 기록한다.'),
 ],
 [
  ('D_candidate','nearby_physical_corner','high','왼쪽 앞 상판/전면 외곽의 물리적 꺾임이 보인다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','오른쪽 앞 상판/body corner가 선반 옆에 보인다. 선반 자체의 corner와 구분한다.'),
  ('D_candidate','nearby_rounded_physical_corner','medium','오른쪽 앞 body 하단의 둥근 끝이 보인다. 그림자/바닥과 exact bbox vertex는 미인증이다.'),
  ('D_candidate','nearby_physical_corner','medium','왼쪽 앞 lower body 끝이 보인다. 검은 front opening을 해당 lower corner 부재로 취급하지 않는다.'),
  ('E_candidate','distinct_external_load','medium','뒤쪽 reference가 carton 표면에 투영된다. carton corner는 target pallet corner로 라벨링하지 않는다.'),
  ('E_candidate','distinct_external_shelf','high','우측 뒤 top reference 위치에는 나무 선반이 있다. 팔레트와 다른 외부 물체가 덮는다.'),
  ('E_candidate','distinct_external_shelf','high','우측 뒤 lower reference도 선반 표면 뒤에 있다. 선반 모서리의 가시성을 target D로 바꾸지 않는다.'),
  ('S_proxy_only','external_load_also_visible','low','self-hidden lower back reference가 carton 안에 투영된다. carton의 가시성은 pallet reference의 D 증거가 아니며 S/E는 분리 미인증이다.'),
 ],
]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=OUT/'EIGHT_FRAME_REFERENCE_VISUAL_REVIEW.json'
    assert not target.exists()
    lock=json.loads((OUT/'SELECTION_AND_REVIEW_PROTOCOL.json').read_text())
    renders=json.loads((OUT/'RENDER_OUTPUTS.json').read_text())
    raws=json.loads((OUT/'RAW_CROP_RENDER_OUTPUTS.json').read_text())
    frames=[]; csvrows=[]
    for j,row in enumerate(lock['frames']):
        assert len(OBS[j])==8
        points=[]
        for a,obs in zip(row['anchors8'],OBS[j]):
            k=int(a['native_corner']);tag,appearance,certainty,note=obs
            r=[float(a['reference_x']),float(a['reference_y'])];q=[float(a['initial_query_x']),float(a['initial_query_y'])]
            point={'native_corner':k,'original_writer_corner':row['perm_v4_dynamic_to_original'][k],
                'reference_xy':r,'saved_initial_query_xy':q,'q_minus_r_xy':[q[0]-r[0],q[1]-r[1]],
                'q_reference_displacement_px':float(a['source_coordinate_displacement_px']),
                'reference_inframe':a['reference_continuous_inframe']=='True','query_nearestpixel_valid':a['initial_query_nearestpixel_valid']=='True',
                'AI_observation_tag':tag,'observed_appearance_at_or_near_r':appearance,
                'AI_observation_confidence_only':certainty,'observation_ko':note,
                'legacy_proxy_kept_separate':{'convex_self_visible_proxy':a['convex_self_visible_proxy']=='True','external_9ray_fraction':float(a['reference_external_9ray_fraction']),'conditional_proxy_state':a['diagnostic_conditional_proxy_state']},
                'evidence_tier':'AI_OBSERVATIONAL_CONDITIONAL_TEACHER_CANDIDATE_ONLY',
                'human_verified':False,'certified_semantic_D_E_S':None,'admitted_for_training':False,
                'exact_r_pixel_ownership_certified':False,'coordinate_adjustment_performed':False}
            points.append(point)
            csvrows.append({'frame_id':row['id'],'asset':row['source_asset'],'native_corner':k,'original_writer_corner':point['original_writer_corner'],'reference_x':r[0],'reference_y':r[1],'initial_query_x':q[0],'initial_query_y':q[1],'displacement_px':point['q_reference_displacement_px'],'reference_inframe':point['reference_inframe'],'tag':tag,'appearance':appearance,'AI_observation_confidence_only':certainty,'legacy_proxy_state':a['diagnostic_conditional_proxy_state'],'human_verified':False,'admitted_for_training':False,'observation_ko':note})
        frames.append({'frame_id':row['id'],'split':'train','source_asset':row['source_asset'],'image':row['image'],'image_sha256':row['image_sha256'],'annotation':row['annotation'],'annotation_sha256':row['annotation_sha256'],'raw_hw':row['raw_hw'],'keypoint_convention':row['keypoint_convention'],'perm_dynamic_to_original':row['perm_v4_dynamic_to_original'],'marked_sheet':renders['frame_outputs'][j],'raw_crop_sheet':raws['frames'][j],'observations8':points})
    counts=dict(collections.Counter(r['tag'] for r in csvrows))
    proxy_cross={}
    for r in csvrows:proxy_cross.setdefault(r['legacy_proxy_state'],collections.Counter())[r['tag']]+=1
    all_disp=[r['displacement_px'] for r in csvrows]
    result={
        'schema':'SYNTH_NATIVE8_OPERATIONAL_REFERENCE_VISUAL_REVIEW_V1','status':'COMPLETED_AI_REVIEW_ONLY_NO_NEW_GT_ADMISSION',
        'selection_lock':{'path':str(OUT/'SELECTION_AND_REVIEW_PROTOCOL.json'),'sha256':sha(OUT/'SELECTION_AND_REVIEW_PROTOCOL.json')},
        'review_method':'Actually viewed full marked overview,8 per-frame r/q overlays and all64 native96 RGB crops, including untouched center RGB crop sheets. No RGB-based sample replacement. No masks/depth/model score used.',
        'frame_count':8,'point_count':64,'asset_count':3,'selected_asset_counts':lock['selected_asset_counts'],
        'TRAIN_only_asset_limitation':'The4th manifest asset has no TRAIN records;3/3/2 samples from3 TRAIN assets. No VAL image was opened. This is a deterministic pilot, not a representative prevalence estimate.',
        'AI_tag_counts':counts,'reference_inframe':sum(r['reference_inframe'] for r in csvrows),'reference_out_of_frame':sum(not r['reference_inframe'] for r in csvrows),
        'legacy_proxy_vs_AI_observation':{k:dict(v) for k,v in proxy_cross.items()},
        'q_reference_displacement_px':{'min':min(all_disp),'max':max(all_disp),'over3':sum(x>3 for x in all_disp),'meaning':'Existing source q-r coordinate relationship, not model performance. Saved q is not a freshly applied fixed ±3 jitter.'},
        'main_findings_ko':[
            '같은 native8 projected cuboid라도 실제 외곽 corner cue가 보이는 점, 외부 물체가 덮는 점, lower/back virtual envelope가 구멍·다른 target면·바닥으로 투영되는 점이 섞인다.',
            'AI D_candidate는 r 근처 physical corner cue가 있다는 조건부 관찰이다. 1pixel material ownership이나 human semantic D를 인증하지 않으며 r좌표를 cue로 이동하지 않았다.',
            '특히 기존 proxy-D41점 중7점에서 고유한 physical corner cue가 보이지 않았고4점은 분해가 불명확했다. 이8프레임 밖 빈도나 학습 가능 정답 수로 일반화하지 않는다.',
            '보이는 top/front corner와 hidden lower/back corner는 같은 target 픽셀에 투영될 수 있다. visible target mask가1이어도 해당 reference 자체의 visibility를 증명하지 않는다.',
            '외부 bin/선반/carton이 관찰되는 E_candidate는9점이다. self proxy8점 중 일부에도 별도 occluder가 함께 보여 S/E 배타 원인 확정은 불가능하다.',
            '6개 out-of-frame reference는 ambiguous로 유지했다. saved q의 raster를 그대로 보여주었고, q와 r를 대체하거나 bbox/좌표를 최적화하지 않았다.'
        ],
        'native8_mapping_notes':{
            'dynamic':'0 frontTL,1 frontTR,2 frontBR,3 frontBL;4..7 corresponding back. This is existing camera_dynamic_0123_v4 convention, not a newly human-verified physical landmark dictionary.',
            'permutation':'Each annotation and inventory is already dynamic. original_writer_corner=perm_v4_dynamic_to_original[k] is recorded for identity traceability; no second permutation applied.',
            'viewpoint':'Dynamic index may refer to different original mesh corners across frames. New q-jitter must hold each reference identity/state fixed per source frame; a cross-frame physical corner ID needs the recorded original-index mapping.',
            'mesh_envelope':'RGB shows slatted wood and hollow/rounded plastic. A mathematical cuboid extremum can be outside actual material or inside an opening. This does not make it an invalid semantic reference, but its operational visibility policy must be explicit instead of inherited from mask occupancy.'},
        'concrete_contract_recommendation':{
            'version_separation':'Propose REFERENCE_VISIBILITY_WITH_INITIAL_QUERY_JITTER_V1 separately; do not reinterpret old M4/radius2 truth or reports.',
            'source_reference_policy':'First declare whether each asset/index means a physical visible corner cue or a virtual envelope landmark. Preserve r and identity, label its semantic visibility with an admitted evidence tier, then perturb input q only. Native q occupancy remains a separate audit target.',
            'physical_corner_subset':'These30 D_candidate and9 E_candidate are reviewable AI proposals only; use human confirmation or stronger supplied annotation evidence before gold/admission. Ambiguous/virtual-no-cue/S_proxy_only are not silently converted to D/E/S.',
            'virtual_reference_policy':'If operational goal deliberately includes virtual bbox corners, retain them and explicitly annotate whether the reference is visually supported/externally covered/self-hidden. Do not demand physical mesh DIRECT or PnP as a mandatory full-goal prerequisite.',
            'scope':'This pilot settles the need for asset/index correspondence review; it does not complete semantic labels for all SYNTH or train a solution.'},
        'read_scope':{'unique_TRAIN_RGB_decoded':8,'source_annotation_content_read_in_this_visual_task':0,'existing_source_proxy_metadata_read':True,'source_mask_depth_feature_checkpoint_arrays_read':0,'VAL_RGB_read':0,'current_RGB_GT_metrics_features_read':0,'new_render_forward_training_PnP':False,'existing_snapshots_modified':False},
        'human_verified_anchor_count':0,'certified_semantic_GT_count':0,'new_training_admitted_anchor_count':0,
        'contact_sheet':renders['contact_sheet'],'frames':frames,
    }
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'SIXTY_FOUR_POINT_VISUAL_REVIEW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    # Additive small tag card, original marked/raw raster artifacts remain intact.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    card=Image.new('RGB',(1200,450),(24,26,29));d=ImageDraw.Draw(card)
    d.text((12,8),'Fixed8 TRAIN / native8 reference visual review: AI candidates, not human GT',font=font,fill='white')
    for j,(frame,obs) in enumerate(zip(frames,OBS)):
        short={'D_candidate':'D?','E_candidate':'E?','S_proxy_only':'Sproxy','ambiguous':'U','virtual-no-cue':'virtual'}
        d.text((12,45+45*j),frame['frame_id']+'  '+frame['source_asset'],font=font,fill='white')
        d.text((520,45+45*j),'  '.join(f'{k}:{short[o[0]]}' for k,o in enumerate(obs)),font=font,fill=(180,230,230))
    d.text((12,415),'No new GT/fit. For evidence use the full frame sheets and untouched native96 crops.',font=font,fill=(235,200,150))
    card.save(OUT/'FIXED8_AI_TAG_OVERVIEW.png')
    bindings=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name!='VISUAL_REVIEW_SEAL.json':bindings.append({'path':str(path),'sha256':sha(path),'size':path.stat().st_size})
    seal={'schema':'SOURCE_REFERENCE_VISUAL_REVIEW_SEAL_V1','status':'COMPLETED_AI_CONDITIONAL_REVIEW_ONLY','review_json_sha256':sha(target),'human_verified_count':0,'training_admission':'HOLD_NO_NEW_GT_ADMITTED','all64_reviewed':True,'source_TRAIN_only':True,'current_or_VAL_read':False,'model_train_render_PnP':False,'bindings':bindings}
    (OUT/'VISUAL_REVIEW_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps({'review_sha256':sha(target),'seal_sha256':sha(OUT/'VISUAL_REVIEW_SEAL.json'),'counts':counts,'proxy_cross':result['legacy_proxy_vs_AI_observation'],'displacements':result['q_reference_displacement_px']},ensure_ascii=False))

if __name__=='__main__':main()
