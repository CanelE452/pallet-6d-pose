from pathlib import Path
import json,hashlib,datetime,collections
R=Path(__file__).resolve().parent; V=R.parent
B=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
rows=[json.loads(x) for x in (V/'rgb96_review/ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
figs=json.loads((V/'rgb96_review/RGB_PROVENANCE_AND_FIGURES.json').read_text())['frames']
D='VISIBLE_DISTINCT_CORNER_EDGE_FOOT'; M='TARGET_MATERIAL_ONLY'; F='FOREIGN_SURFACE_OR_BACKGROUND'; A='AMBIGUOUS'; O='OOB'; N='NO_QUERY'
notes={
10:('NO_DETECTION','Warehouse aisle with racks and cartons, but no selected prediction or query set. Black crop panels are explicit zero placeholders, not observed occlusion or darkness.',[(N,'No detected queryset; this panel is a zero placeholder and contains no actual query crop.')]*8),
11:('MULTIPLE_OR_AMBIGUOUS','Selected span crosses a long left rack row containing several wooden pallet sections, tubs/cartons and red/yellow rack posts; one intended pallet instance is not independently clear.',[
(O,'Original q x=-100 is outside the native image; black panel is factual invalid input.'),
(A,'A wooden pallet rail/opening and a red upright with yellow protection overlap in this crop. A unique intended wooden part at q is unclear.'),
(F,'The yellow rack guard and its floor/base area dominate near q; nearby pallet wood does not establish the intended reference.'),
(O,'Original q x=-100 is outside the native image; black panel is factual invalid input.'),
(F,'Gray storage tub/bin wall and a dark horizontal rim dominate; this is not a verified pallet part.'),
(D,'A wooden rail/opening/end junction is visible beside the red rack upright; a local physical pallet-like cue exists, but its intended instance/native slot is not established.'),
(F,'Floor occupies the query neighbourhood beside the rack/stack; pallet rails are offset from q.'),
(A,'A dark bin/carton-like oblique surface and a wooden rail share the crop. Ownership and a unique structural corner are unclear.')]),
12:('MULTIPLE_OR_AMBIGUOUS','Full context contains stacks of multiple pallets/containers. The saved query set is entirely left of the image, so none of the eight crops supplies scene evidence.',[(O,'Original native x is negative; this black panel is the unchanged invalid-query placeholder, not a semantic absence label.')]*8),
13:('CLEAR_SINGLE_TARGET','A loaded wooden pallet beneath the white crates at image right can be visually distinguished in full context. This is weak visual association, not a GT-selected bbox or verified native reference; the neighboring yellow rack protector intrudes.',[
(A,'White crate lower edge and wooden support/rail appear together, but blur makes the precise junction/ownership at q uncertain.'),
(A,'The right loaded-base support/end is blurred; visible crate and wood do not isolate the expected native part.'),
(D,'A wooden support-foot outer edge meets the floor; an exposed coarse physical end/foot cue is visible.'),
(D,'The lower wooden block/foot meets the floor beside the yellow protector, giving a visible physical support-end cue.'),
(F,'Yellow/black rack protector surface occupies the crop and query area.'),
(F,'White crate face/edge dominates above a narrow wooden rail; material at q is cargo, not an independently identified pallet reference.'),
(D,'A dark fork/opening boundary and wooden block/rail with floor are visible as a local structural junction; the opening itself is not semantic absence.'),
(F,'Yellow guard and gray metal foot/base dominate; no intended pallet corner is independently identified.')]),
14:('MULTIPLE_OR_AMBIGUOUS','The prediction addresses a small distant base near guards and stacked goods. Low native crop scale and blur prevent a unique pallet instance/part association; no semantic label is inferred.',[
(F,'Yellow guard/post fills the local patch.'),
(A,'Blurred brown/red horizontal edge above a dark region; cannot certify wood pallet versus load/rack part.'),
(A,'Blurred dark support-like region above gray floor; no defensible distinct part at q.'),
(F,'Yellow guard and dark base/floor dominate.'),
(F,'Yellow guard surface dominates.'),
(A,'Blurred stacked brown/red band and dark underside; target identity and exact structural cue remain unclear.'),
(A,'Small blurred base/leg-like structure against floor; insufficient detail for a unique corner/foot claim.'),
(F,'Yellow guard corner and dark lower base are visible, not a verified pallet reference.')]),
15:('MULTIPLE_OR_AMBIGUOUS','Loaded pallet feet/deck are visible around rack uprights, but the query span also includes the yellow guard and adjacent load/rack parts. Distinct local wooden cues do not establish a single intended target for all slots.',[
(F,'Yellow guard/upright and gray metal base occupy the query region.'),
(A,'Dark rack upright/beam, timber slats and neighboring load overlap; ownership of the local end is unclear.'),
(D,'Exposed wooden slat/support end and floor boundary are visible below the dark beam; local cue is coarse and instance association is unresolved.'),
(F,'Yellow guard and gray bolted base meet the floor at q.'),
(D,'A wooden block/support endpoint with a deck rail and floor contact is visible.'),
(F,'Bottles/cargo and a dark horizontal rack-like beam dominate the crop; no intended native pallet point is proven.'),
(D,'A wooden support block/rail and adjacent dark opening beneath the load form a visible structural junction.'),
(D,'Wooden foot/end and lower deck edge meet the floor; this is local physical evidence only.')]),
16:('NO_CLEAR_TARGET','Full scene identifies a red wheeled transport cart at the selected region. The saved points lie on its lower red/gray structure, white panel or wheel/floor vicinity; no clear pallet target is established there.',[
(F,'White cart/door panel dominates the crop.'),
(F,'Red cart panel and gray horizontal metal trim occupy the crop.'),
(F,'Red cart lower panel and dark undercarriage dominate.'),
(F,'White panel surface occupies the local patch.'),
(F,'Red panel/gray frame corner of the wheeled cart is visible, not an independently identified pallet cue.'),
(F,'Cart red panel and gray metal rail/trim dominate.'),
(F,'Dark cart undercarriage/wheel-side structure and red panel occupy the patch.'),
(F,'White panel and cart wheel/undercarriage/floor vicinity are visible; no pallet ownership is established.')]),
17:('MULTIPLE_OR_AMBIGUOUS','Several adjacent gray plastic pallets under red crates form a continuous row; predicted query span crosses multiple supports and a yellow guard. A visible pallet-like physical edge is not an instance/native correspondence proof.',[
(F,'Red cargo crate/label dominates above the gray pallet rail; cargo is not the intended pallet reference by itself.'),
(M,'Gray molded pallet deck/side material is visible; q is on a repeated rail area and no unique endpoint/native part is established.'),
(F,'Gray floor below the support occupies the query neighbourhood; feet visible above do not relabel the floor.'),
(D,'A gray molded support foot and dark fork-opening side meet the floor, forming a visible structural part junction.'),
(D,'Adjacent crate/gray support endpoint and opening junction are visible; multiple neighboring pallet instances remain possible.'),
(F,'Yellow rack guard crosses the crop with cargo/pallets behind; q does not uniquely identify a pallet part.'),
(F,'Floor dominates between the yellow guard and receding pallet feet.'),
(D,'A molded plastic foot/opening/floor junction is visible; repeated similar parts prevent native-slot identification.')]),
18:('MULTIPLE_OR_AMBIGUOUS','The selected span includes a gray pallet-like side/deck, red crates and a yellow guard. Nearby separate gray structures and the guard obscure a unique intended object association.',[
(D,'A gray ribbed deck-to-side/support junction is exposed beside the yellow post. Its canonical/native identity is unknown.'),
(F,'Red cargo crate and clear plastic wrapping dominate the local patch.'),
(F,'Red cargo crate wall/label occupies the crop.'),
(F,'Floor and a dark post/guard edge occupy q; the yellow guard is visible at the side.'),
(M,'Gray molded/ribbed pallet-like top/side material is visible, without a distinct localized end or foot at q.'),
(F,'Yellow rack protection surface fills the patch.'),
(F,'Yellow protection surface with a red crate edge dominates.'),
(D,'Gray support/ribbed side, dark vertical boundary and floor form a visible local structural edge/foot junction.')]),
19:('MULTIPLE_OR_AMBIGUOUS','Predicted span crosses left-side wrapped/carton loads and multiple wooden pallet supports along the aisle. Some local pallet feet are visible but no single intended pallet is proved across the eight points.',[
(F,'Brown carton face dominates; black pixels at the image edge are crop padding, not a covered reference.'),
(D,'Wooden deck/support foot and dark fork-opening boundary below wrapping are visible as a local physical junction.'),
(F,'Floor below the wooden support dominates q, while the foot is offset above.'),
(F,'Brown carton lower face/edge and dark underside dominate with image-edge padding; a specific pallet reference is unproved.'),
(F,'Wrapped goods surface/lower lip occupies q, with some wood below; packaging is not independently a pallet reference.'),
(F,'Floor dominates, with distant pallet/load and yellow guard near the patch edge.'),
(F,'Mostly floor with receding pallet feet at the edge; no unique local reference at q.'),
(D,'Wooden deck/foot and dark opening below the wrapped load form a visible coarse structural junction.')])
}
report_frames=[];allslots=[]
for fi in range(10,20):
 fr=figs[fi]; assert fr['frame_index']==fi
 figure=Path(fr['figure']['path']); assert B(figure)['sha256']==fr['figure']['sha256']
 rr=[x for x in rows if x['frame_index']==fi];assert len(rr)==8 and [x['native_slot'] for x in rr]==list(range(8))
 target,full,obs=notes[fi];assert len(obs)==8
 slots=[]
 for r,(category,detail) in zip(rr,obs):
  if not r['query_exists']:assert category==N
  elif not r['query_center_valid']:assert category==O
  else:assert category not in [N,O]
  z={'frame_index':fi,'input_id':r['input_id'],'native_slot':r['native_slot'],'query_xy':r['query_xy'],'L':r['L'],'query_exists':r['query_exists'],'query_center_valid':r['query_center_valid'],'crop_valid_fraction':r['crop_valid_fraction'],'input_status':r['input_status'],'query_local_evidence':category,'observation':detail,'expected_native_part_identity':'UNKNOWN','semantic_D_E_S':'UNKNOWN','usable_training_gold':False,'new_training_label':False,'coordinate_or_input_changed':False}
  slots.append(z);allslots.append(z)
 report_frames.append({'frame_index':fi,'input_id':rr[0]['input_id'],'split':rr[0]['split'],'image_sha256':rr[0]['image_sha256'],'figure':B(figure),'actually_viewed_fullframe':True,'actually_viewed_all8_crop_panels':True,'actual_scene_crop_count':sum(x['query_center_valid'] for x in rr),'invalid_or_missing_placeholder_panels':sum(not x['query_center_valid'] for x in rr),'target_instance_visual_association':target,'target_association_is_human_or_GT_proof':False,'fullframe_observation':full,'slots':slots})
assert len(allslots)==80
report={'schema':'V92_REVIEWER_B_FIXED20_LAST10_PUBLIC_INITIAL_QUERY_ACTUAL_VISUAL_OBSERVATIONS','completed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policy':B(R/'POLICY_BEFORE_ASSIGNED_IMAGES.json'),'root_contract':B(V/'SOURCE_QUERY_VISUAL_OBSERVATION_CONTRACT_BEFORE_REVIEW.json'),'public_input_only_rows':B(V/'rgb96_review/ROWS_INPUT_ONLY.jsonl'),'render_manifest':B(V/'rgb96_review/RGB_PROVENANCE_AND_FIGURES.json'),'actual_view_receipt':{'tool':'functions tools.view_image','files_opened_count':10,'assigned_frame_indices':list(range(10,20)),'fullframe_panels_viewed':10,'all_slot_panels_viewed':80,'nonplaceholder_actual_RGB96_crops_viewed':sum(r['query_center_valid'] for r in allslots),'no_query_or_invalid_zero_panels_viewed':sum(not r['query_center_valid'] for r in allslots),'raw_source_RGB_or_NPZ_not_opened_in_this_review':True,'source_pixel_evidence_was_completed_figures':True},'counts':{'target_instance_visual_association':dict(collections.Counter(f['target_instance_visual_association'] for f in report_frames)),'query_local_evidence':dict(collections.Counter(r['query_local_evidence'] for r in allslots)),'native_part_identity_UNKNOWN':80,'semantic_D_E_S_UNKNOWN':80,'training_gold':0},'frames':report_frames,'cross_case_findings_ko':['미검출8점과 OOB10점을 실제 scene의 가림으로 세지 않았습니다. 할당80점 중 원래 factual valid는62점이며 모두 원좌표와 입력상태를 유지했습니다.','각 query 부근에 선반 보호대, 빨간 적재 crate, 바닥, 운반카트가 반복해서 나타납니다. 유효 crop에 물체 모양이 있다는 사실이 원하는 팔레트 instance나 native8의 기대 part를 증명하지는 않습니다.','일부 실제 나무·플라스틱 support/foot/rail junction은 보이므로 물리 단서의 관찰 근거는 있습니다. 그러나 반복된 발/포크 구멍과 여러 인접 팔레트 때문에 native slot 대응과 D/E/S는 전부UNKNOWN입니다.','frame16은 full scene상 바퀴 달린 빨간 카트 하부에 예측이 놓였다는 구체적인 대상 대응 한계를 보입니다. 이는 source palette 조사이지 현재223의 실패율이나 모델 인과 원인 추정이 아닙니다.','단순 material/빈공간/padding/foreignsurface를 D/E/S로 변환하지 않았습니다. 가상 cuboid 참조를 exact mesh 표면 점으로 제한하거나 input q를 물리 코너로 옮기지 않았습니다.'],'limitations':['AI weak visual observation, not human gold.','Observed categories describe these assigned panels only; no unviewed dataset-wide prevalence or semantic performance estimate.','Figure full context helps distinguish objects but does not prove crop-only identifiability or predicted target ownership.','Blur and small crop scales remain ambiguous; no label repair, recrop, coordinate adjustment or nearest-cue native remapping.'],'actions':{'figure_views':10,'input_metadata_reads':True,'other_reviewer_judgments_read':False,'GT_bbox_mask_depth_pose_reads':0,'raw_source_RGB_NPZ_reads':0,'models_weights_features_reads':0,'new_forward_fit_PnP':0,'current223_reads':0,'new_training_labels':0,'existing_artifact_changes':0}}
p=R/'FIXED10_FULLFRAME_AND80_QUERY_CROP_OBSERVATIONS.json';assert not p.exists();p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
s={'completed':True,'report':B(p),'policy':report['policy'],'root_contract':report['root_contract'],'writer_code':B(Path(__file__)),'actually_viewed_figure_bindings':[f['figure'] for f in report_frames],'counts':report['counts'],'no_semantic_training_label_admission':True,'actions':report['actions']};q=R/'VISUAL_OBSERVATION_SEAL.json';assert not q.exists();q.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'report':B(p),'seal':B(q),'counts':report['counts']},ensure_ascii=False))
