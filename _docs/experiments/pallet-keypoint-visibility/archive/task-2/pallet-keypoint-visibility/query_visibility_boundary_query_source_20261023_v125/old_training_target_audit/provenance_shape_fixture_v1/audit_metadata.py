"""Read source label/availability metadata only; never access logits/features/weights."""
from pathlib import Path
import hashlib, json, math
from collections import Counter
import numpy as np

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OUT=Path(__file__).resolve().parent
V={n:next(P.glob('*_v'+str(n)))for n in (96,97,99,101,102,109,110,113,114,116,118,119,122,123,124,125)}
def bind(p):
    p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def read(p):return json.loads(Path(p).read_text())
def raw(y):return np.where(y==0,'D',np.where(y==1,'E','U'))

scores=V[124]/'runs/C16_EXACT_C15_DURATION_CONTROL_8000/SELECTED_ALL51004_SOURCE_SCORES.npz'
# Explicit key allowlist: no query_logit, prediction, or tensor feature array read.
keys=('external_target','raw_teacher_state','split','source','variant','domain','query_available')
with np.load(scores,allow_pickle=False)as z:
    m={k:z[k]for k in keys}
    pairs={k:z[k]for k in ('old_accepted_triplets','new_selected_triplets','samewood_accepted_cover_sham_pairs')}
N=len(m['split']);assert N==51004
train=m['split']=='TRAIN';known=np.isin(m['external_target'],[0,1])
assert int((train&known).sum())==4079 and m['query_available'][train&known].all()
assert np.array_equal(m['external_target'],np.where(m['raw_teacher_state']=='D',0,np.where(m['raw_teacher_state']=='E',1,-1)))

# Replay every typed-label join from the original GT-only source, no trainer imports.
g96=V[96]/'actual_query_anchored_image_intake/OCCURRENCES31680_GT_ONLY.jsonl'
g99=V[99]/'geometry_query_preparation/EXPECTED_CUE_TARGETS96_GT_ONLY.jsonl'
g109=V[109]/'corpus_build_v1/RAW_SURFACE_TEACHER8000_GT_ONLY.npz'
g113=V[113]/'corpus_build_v1/RAW_SURFACE_TEACHER3200_GT_ONLY.npz'
g116=V[116]/'corpus_build_v1/OCCURRENCES924_GT_ONLY.jsonl'
g122=V[122]/'actual_corpus/ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY.jsonl'
rr96=rows(g96);rr99=rows(g99);rr116=rows(g116);rr122=rows(g122)
parts=[raw(np.array([r['candidate_aux_hidden_label_GT_only']for r in rr96])),
       np.array([x['state']for r in rr99 for s in ('BEFORE','COVER','SHAM')for x in r['stages'][s]])]
for p in (g109,g113):
    with np.load(p,allow_pickle=False)as z:parts.append(z['raw_surface_teacher_state'])
parts.extend(raw(np.array([r['candidate_aux_hidden_label_GT_only']for r in rr]))for rr in(rr116,rr122))
assert np.array_equal(np.concatenate(parts),m['raw_teacher_state'])

definitions={
'V96_REAL':('CONDITIONAL_RECORDED_STRUCTURAL_CUE','Original query central positive-bilinear stencil must be wholly in one consistent known FULL cue group, with competing/unknown/unlocated groups excluded. Natural BEFORE labels are weak recorded cue D/E. Controlled COVER hides the matched complete cue group plus central stencil; SHAM preserves both. Not a native expected-part certificate.'),
'V99_SYNTH':('GEOMETRIC_EXPECTED_JUNCTION_WITNESS','D requires one complete front-facing endpoint+face-patch+two-edge-arm witness DIRECT; E requires all originally direct witnesses FOREIGN covered. q is projected virtual-reference plus fixed jitter and only guards crop/frame support. This is semantic expected-junction geometry, not q mask membership; native pallet correspondence UNKNOWN.'),
'V109_YCB_SURFACE':('CLEAR_TARGET_GRANULARITY_DIFFERENCE_SURFACE_PROXY','Visibility of a fixed original mesh vertex, not a certified structural junction. Target-alone stable rays separate self geometry; depth/owner evidence is sampled at nearest projected reference r and its 3x3 neighbourhood. q=r+disk4 jitter only guards input support. D/E are conservative surface-visibility proxies, not generic pixel membership at q.'),
'V113_MATERIAL_SURFACE':('CLEAR_TARGET_GRANULARITY_DIFFERENCE_SURFACE_PROXY','Same unchanged V109 ray/depth/complete-owner rule at fixed original woodblock/foam mesh reference r, with independent disk4 q emulation. Material-relevant surface reference is not certified as a structural cue.'),
'V116_SAME_WOOD_CONTROLLED':('CONDITIONAL_RECORDED_STRUCTURAL_CUE','Original V96 matched-cue group plus central stencil is protected. Exact wood foreground transform covers the full protected union for E and avoids it for D sham. Original q/L remain fixed. BEFORE aliases are reported but this added sampler draws only accepted cover/sham pairs.'),
'V122_PUBLIC_FLAT_PALLET':('CONDITIONAL_RECORDED_STRUCTURAL_CUE','AI-weak visible 2D structural cue/uncertainty region; native-instance/corner truth not claimed. Same physical cover/sham is shared by nine q=r+{-3,0,3}^2 input queries; D/E describes the protected cue component. Failed construction is teacher U while factual availability remains separate.')}
pop=[]
for dom in sorted(set(m['domain'])):
    mask=train&(m['domain']==dom)
    counts=Counter(m['raw_teacher_state'][mask].tolist())
    detail=[]
    for source in sorted(set(m['source'][mask])):
        for variant in sorted(set(m['variant'][mask])):
            mm=mask&(m['source']==source)&(m['variant']==variant)&known
            if mm.any():detail.append(dict(source=source,variant=variant,D=int((mm&(m['external_target']==0)).sum()),E=int((mm&(m['external_target']==1)).sum())))
    category,definition=definitions[str(dom)]
    pop.append(dict(domain=str(dom),TRAIN_rows=int(mask.sum()),raw_states=dict(counts),known_D=int((mask&(m['external_target']==0)).sum()),known_E=int((mask&(m['external_target']==1)).sum()),known_unavailable=int((mask&known&~m['query_available']).sum()),definition_category=category,definition=definition,by_source_variant=detail))

# Recorded stencil+full-union certificates imply native nearest-q alpha membership.
# No protected-mask/alpha/RGB arrays are read in this audit.
inventory_path=V[96]/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl'
inventory=rows(inventory_path)
feas96=V[96]/'actual_query_anchored_image_intake/FEASIBILITY15360_GT_ONLY.jsonl'
feas116=V[116]/'corpus_build_v1/FEASIBILITY448_GT_ONLY.jsonl'
cert_counts={}
for name,path,basekey,only_train in [('V96',feas96,'base_row_index',True),('V116',feas116,'original_base_row_index',False)]:
    accepted=[r for r in rows(path)if r['pair_ready']and(not only_train or r['raw_split']=='TRAIN_LAB')]
    for r in accepted:
        base=inventory[r[basekey]];q=base['original_q'];nearest=[math.floor(q[0]+.5),math.floor(q[1]+.5)]
        assert nearest in base['central4_positive_bilinear_native_cells']
        assert r['protected_cover_all1']is True and r['protected_sham_all0']is True
    cert_counts[name]=dict(accepted_TRAIN_pairs=len(accepted),COVER_E_q_inside=len(accepted),COVER_E_q_outside=0,SHAM_D_q_inside=0,SHAM_D_q_outside=len(accepted),basis='Certificate-derived: nearest q belongs to original central stencil; protected union includes that stencil; saved all-cover/no-sham certificates. Not fresh alpha-pixel replay.')
alpha_audit_path=V[124]/'ROOT_PUBLIC_QUERY_BOUNDARY_COVERAGE_AUDIT.json'
alpha_audit=read(alpha_audit_path);a=alpha_audit['by_source_rawsplit']['LOCO::TRAIN']
assert a['known_COVER_E']['q_inside_foreground_alpha']==396 and a['known_SHAM_D']['q_outside_foreground_alpha']==396
cert_counts['V122']=dict(accepted_TRAIN_query_triplets=396,COVER_E_q_inside=396,COVER_E_q_outside=0,SHAM_D_q_inside=0,SHAM_D_q_outside=396,basis='Existing exact nearest floor(q+.5) alpha replay, reused as saved metadata. Nine jitters share physical composition; not 396 independent scenes.')

q99=rows(V[99]/'geometry_query_preparation/QUERY_INPUTS96_INPUT_ONLY.jsonl')
probe=Counter();dist=[]
for q,g in zip(q99,rr99):
    if q['split']!='TRAIN':continue
    for stage in ('BEFORE','COVER','SHAM'):
        for k,r in enumerate(g['stages'][stage]):
            if r['state']not in ('D','E'):continue
            probe[(r['state'],r['combined_probe_states'][0]['state'])]+=1
            dist.append(float(np.linalg.norm(np.array(q['query8_xy'][k])-r['projected_endpoint'])))
assert dict(probe)=={('D','DIRECT'):751,('E','FOREIGN'):71}

sampler=[
dict(domain='V96_REAL',pool='75 accepted TRAIN triplets + LOCO BEFORE D3 + Pallox BEFORE D25 + natural BEFORE E2',draw_rows=20,D=12,E=8),
dict(domain='V99_SYNTH',pool='64 selected TRAIN triplets + all TRAIN D751/E71',draw_rows=24,D=12,E=12),
dict(domain='V109_YCB_SURFACE',pool='D1181/E340',draw_rows=20,D=8,E=12),
dict(domain='V113_MATERIAL_SURFACE',pool='D356/E202',draw_rows=32,D=16,E=16),
dict(domain='V116_SAME_WOOD_CONTROLLED',pool='62 accepted TRAIN cover/sham pairs; 28 BEFORE aliases not directly drawn in this added pool',draw_rows=32,D=16,E=16),
dict(domain='V122_PUBLIC_FLAT_PALLET',pool='6 LOCO TRAIN cues: BEFORE54/COVER396/SHAM396; no TOMIE TRAIN',draw_rows=64,D=32,E=32)]
assert sum(x['draw_rows']for x in sampler)==192 and sum(x['D']for x in sampler)==sum(x['E']for x in sampler)==96
for key,n in [('old_accepted_triplets',75),('new_selected_triplets',64),('samewood_accepted_cover_sham_pairs',62)]:
    pp=pairs[key];assert int((m['split'][pp[:,0]]=='TRAIN').sum())==n

paths=[scores,g96,g99,g109,g113,g116,g122,inventory_path,feas96,feas116,alpha_audit_path,
V[102]/'source_fit_preparation/SOURCE_BINDINGS.json',V[102]/'source_fit_preparation/external_data_adapter.py',
V[101]/'source_fit_preparation/data_adapter.py',V[96]/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json',
V[97]/'cue_contract_preparation/endpoint_cue_teacher.py',V[99]/'geometry_query_preparation/build_geometry96.py',
V[109]/'producer/build_surface_source.py',V[109]/'ROOT_SOURCE_POLICY_V2_CORRECTED_INVENTORY_BEFORE_BULK.json',
V[113]/'ROOT_EXACT3200_MATERIAL_TEACHER_POLICY_BEFORE_LABELS.json',V[116]/'ROOT_SOURCE_POLICY_BEFORE_DONORS_AND_PIXELS.json',
V[110]/'train_natural_surface_rgb_head.py',V[114]/'source_fit_preparation/train_source_once.py',
V[118]/'source_fit_preparation/train_source_once.py',V[119]/'source_fit_preparation/train_source_once.py',
V[123]/'source_fit_preparation/train_source_once.py',V[124]/'ROOT_C16_DESIGN_BEFORE_SOURCE_FIT.json',
V[125]/'ROOT_C17_DESIGN_BEFORE_SOURCE_FIT.json',Path(__file__)]
report=dict(completed=True,PASS=True,scope='SOURCE-only code and GT/availability metadata audit of the unchanged C16/C17 old192 loss; no quality judgment on every old label.',
    main_conclusion_KO='old192는 하나의 동일한 semantic cue teacher가 아니다. 조건부 기록 구조 cue, 합성 expected junction witness, 실제 mesh 표면 reference의 depth/owner 가시성이 함께 학습된다. 표면 proxy와 구조 cue는 명백히 세분성이 다르지만, old 라벨 전체가 틀렸거나 동일 입력에 상충 라벨이 있다는 결론은 아니다.',
    provenance_chain='C16/C17 old192 -> pinned C15 draw192/loss192 -> C14/C13 old128 -> C12 draw96 -> C10 draw64 + V113 material32 + V116 pair32; C15 adds V122 flat64. External adapter preserves S but excludes it from learned D/E.',
    all_51004_typed_labels_replayed_from_original_GT=True,metadata_npz_keys_read=list(keys)+list(pairs),TRAIN_rows=int(train.sum()),TRAIN_raw_states=dict(Counter(m['raw_teacher_state'][train].tolist())),TRAIN_known_D=int((train&(m['external_target']==0)).sum()),TRAIN_known_E=int((train&(m['external_target']==1)).sum()),TRAIN_known_total=4079,TRAIN_known_unavailable=0,
    populations=pop,sampler192=sampler,sampleable_unique_TRAIN_occurrences=4051,reported_but_not_directly_sampled_V116_BEFORE_aliases=28,
    loss_weighting=dict(actual='mean BCE(old128) + 0.25*mean hinge24(cover-sham margin2) + mean BCE(V122 new64). C17 adds separate mean BCE(wide64); it is not flat mean BCE of 256 rows.',old192_rows=dict(conditional_recorded_cue=116,synthetic_expected_junction=24,mesh_surface_proxy=52),old192_BCE_coefficient_share=dict(conditional_recorded_cue=.703125,synthetic_expected_junction=.09375,mesh_surface_proxy=.203125),warning='Counts and fixed sampler mass are not independent physical scenes; shared bases, ranks, and jitters repeat.'),
    q_alpha_TRAIN_controlled=cert_counts,controlled_total_pairs_or_query_triplets=533,controlled_semantic_E_qout=0,controlled_SHAM_D_qin=0,
    unmeasured_q_alpha=dict(V96_natural_BEFORE_E='2 TRAIN natural recorded cue-E rows; no controlled foreground alpha. Expected cue binding remains conditional weak evidence.',V99='No q-alpha raster replay in this audit. D/E are junction-witness ray states at endpoint/patch, with q jitter and crop guard, not q-pixel labels.',V109_V113='No pasted alpha. Complete owner/depth evidence is at r-neighbourhood, not q; q-pixel target/foreign ownership is not established by these teacher packets.'),
    V99_reference_evidence=dict(TRAIN_D_endpoint_DIRECT=751,TRAIN_E_endpoint_FOREIGN=71,query_endpoint_distance_native_px=dict(N=len(dist),minimum=min(dist),median=float(np.median(dist)),maximum=max(dist))),
    distinctions=dict(clear_target_difference='V109/V113 surface-point visibility vs recorded/synthetic structural-cue semantics.',conditional_cue='V96/V116/V122 weak 2D recorded cue groups and controlled component cover; V99 explicit geometric junction witness. All retain native mapping UNKNOWN/human-gold false.',unverified='No identical-three-input contradictory-label census or feature-capacity/grounding test here; metadata cannot establish gradient interference or a feature bottleneck.',historical_V95='Earlier V95 had semantic-E/q-alpha separation; it is a different predecessor corpus. The 533 old controlled TRAIN relations above do not generalize to all earlier sources or all natural labels.'),
    interpretation_KO='이 old controlled TRAIN 범위에서는 E와 q-alpha 내부, SHAM D와 외부가 완전히 함께 변했다. 새 wide 조건은 원 weak cue/uncertainty를 유지한 채 그 상관을 분리하므로 기존 학습 분포 밖 조합을 추가한다. 이는 shortcut 가능성과 목표 granularity 혼합의 근거이며 실제 loss 충돌의 증명은 아니다. S/U를 hidden negative로 학습한 혼합은 확인되지 않았고, surface teacher도 q-mask membership으로 잘못 요약하면 안 된다. old/new 학습 간섭과 고정 feature의 표현 한계는 이 metadata 감사만으로 분리되지 않는다.',
    actual_RGB_depth_masks_alpha_features_logits_predictions_weights_model_fit_current_V112_PnP_reads_or_calls=0,original_labels_queries_folds_or_files_modified=False,new_gate_or_training_decision=False,bindings=[bind(p)for p in paths])
with (OUT/'REVIEW.json').open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(OUT/'REVIEW.json'),TRAIN_known_total=4079,D=2931,E=1148,sampleable_unique=4051,controlled_TRAIN_pairs_or_query_triplets=533),ensure_ascii=False))
