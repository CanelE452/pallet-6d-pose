"""TRAIN-only saved source contract audit and outcome-independent smoke metadata."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import numpy as np

O=Path(__file__).resolve().parent
P=O.parents[1]
V=P/'query_visibility_full_source_query_causal_census_20261024_v173'
reads={};checks=Counter();errors=[]
def sha(p):
    p=Path(p)
    assert not any(x in str(p).lower() for x in ('_v112','_v177','_v178','_v169','/private_evaluation/'))
    return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):
    p=Path(p);reads[str(p.resolve())]=sha(p);return json.loads(p.read_text())
def lines(p):
    p=Path(p);reads[str(p.resolve())]=sha(p);return [json.loads(x) for x in p.read_text().splitlines() if x]
def check(v,label):
    checks[label]+=1
    if not bool(v):errors.append(label)
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sumweights(x):return [math.fsum(float(v) for v in x[:,j]) for j in range(x.shape[1])]

scope=read(V/'ROOT_SCOPE_BEFORE_FULL_SOURCE_MASK_BODIES.json')
complete=read(V/'literal_raster_census_v1/COMPLETE.json')
protocol=read(V/'census_preparation/PROTOCOL_BEFORE_FULL_SOURCE_MASK_BODIES.json')
metadata=lines(V/'census_preparation/ALL6144_ORIGINAL_QUERY_METADATA_BEFORE_MASKS.jsonl')
rows=[r for r in metadata if r['split']=='TRAIN']
check(len(rows)==4096,'TRAIN4096 metadata retained')
train_path=Path(complete['sidecars']['TRAIN']['path'])
reads[str(train_path)]=sha(train_path)
check(reads[str(train_path)]==complete['sidecars']['TRAIN']['sha256'],'TRAIN sidecar sealed SHA')
with np.load(train_path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
check(all(x=='TRAIN' for x in a['split']),'NPZ TRAIN only')
flags=tuple(rows[0]['original_input_flags'])
for i,r in enumerate(rows):
    for k,m in [('original_global_index','original_global_index'),('original_occurrence_id','original_occurrence_id'),('scene_index','scene_index'),('base_index','base_index'),('native_slot','native_slot'),('variant','variant'),('original_semantic_state','original_semantic_state'),('declared_canonical_index','declared_canonical_index'),('correspondence_boundary','correspondence_boundary')]:
        check(a[k][i]==r[m],'TRAIN identity semantic mapping exact')
    missing=r['query_xy'] is None
    check(bool(a['query_xy_missing'][i])==missing and (np.isnan(a['query_xy'][i]).all() if missing else np.array_equal(a['query_xy'][i],np.asarray(r['query_xy'],np.float64))),'TRAIN q and null exact')
    check(bool(a['L_missing'][i])==(r['L'] is None) and (math.isnan(a['L'][i]) if r['L'] is None else a['L'][i]==r['L']),'TRAIN L and null exact')
    check(bool(a['selected_index_missing'][i])==(r['selected_index'] is None) and int(a['selected_index'][i])==(-1 if r['selected_index'] is None else r['selected_index']),'TRAIN selected candidate/null exact')
    for k in flags:check(bool(a[k][i])==r['original_input_flags'][k],'TRAIN all9 original flags exact')
    for k in ('material','aspect','geometry_family','foreign_family','signed_quarter'):check(a[k][i]==r['factors'][k],'TRAIN factors exact')
    check(np.array_equal(a['nominal_reference_xy'][i],r['nominal_reference_xy']),'TRAIN nominal source reference exact')
    check(r['incident_part_ids']==[r['primary_part_id']],'TRAIN incident set is designated primary singleton')
check(a['training_admission'].item() is False and a['initializer_physical_native_identity_certified'].item() is False,'No admission/physical initializer certificate')

softkeys=[k for k in a if k.startswith(('scene_owner_','BEFORE_owner_','joint_')) and a[k].ndim==2]
check(len(softkeys)==9,'Exactly nine measured distributions')
summaries={}
for k in softkeys:
    x=a[k];good=np.isfinite(x).all(1);bad=~good
    check(x.dtype==np.float64 and x.shape==(4096,7 if k.startswith('joint_') else 6),'Soft dtype dimensions')
    check(np.isnan(x[bad]).all() and np.all(x[good]>=0) and np.allclose(x[good].sum(1),1,rtol=0,atol=2e-13),'Soft mass or explicit missing')
    check(np.array_equal(bad,a['query_xy_missing']),'Measured missing equals original missing q')
    summaries[k]=dict(measured_N=int(good.sum()),missing_N=int(bad.sum()),sum_weights=sumweights(x[good]),max_unit_mass_error=float(np.max(np.abs(x[good].sum(1)-1))))

def group_summary(mask):
    oo={}
    for k in softkeys:
        good=mask&np.isfinite(a[k]).all(1)
        oo[k]=dict(measured_N=int(good.sum()),sum_weights=sumweights(a[k][good]))
    return dict(N=int(mask.sum()),input_U=int((mask&~a['query_available']).sum()),semantic_counts={s:int((mask&(a['original_semantic_state']==s)).sum()) for s in 'DESU'},soft=oo)

semantic={s:group_summary(a['original_semantic_state']==s) for s in 'DESU'}
for st in 'DESU':
    x=a['scene_owner_point'];good=(a['original_semantic_state']==st)&np.isfinite(x).all(1)
    semantic[st]['point_nonzero_mass_rows']={name:int((good&(x[:,j]>0)).sum()) for j,name in enumerate(a['owner_class_order'].tolist())}
    semantic[st]['point_exact_unit_class_rows']={name:int((good&(x[:,j]==1)).sum()) for j,name in enumerate(a['owner_class_order'].tolist())}
groups={}
for material,aspect in sorted({(r['factors']['material'],r['factors']['aspect']) for r in rows}):
    groups[material+'::'+aspect]=group_summary((a['material']==material)&(a['aspect']==aspect))
native={str(slot):group_summary(a['native_slot']==slot) for slot in range(8)}
valid=a['query_available'];point=a['scene_owner_point'];known3=np.column_stack((point[:,0]+point[:,1],point[:,2],point[:,3]))
check(int(valid.sum())==4016 and int((~valid).sum())==80,'TRAIN available4016/U80')
check(np.isfinite(point[valid]).all() and np.all(point[valid,4:]==0),'Available TRAIN contains no ambiguous/invalid/missing mass')
check(int(a['query_xy_missing'].sum())==48 and int((~valid&~a['query_xy_missing']).sum())==32,'U80 missing48 finiteinvalid32')
check(int(a['correspondence_boundary'].sum())==0,'TRAIN no CF/V4 correspondence disagreement')
numbers=dict(completed=True,PASS=not errors,TRAIN_only=True,rows=4096,bases=len(set(a['base_index'].tolist())),scenes=len(set(a['scene_index'].tolist())),
    semantic_counts={s:int((a['original_semantic_state']==s).sum()) for s in 'DESU'},input_valid4016=int(valid.sum()),inputU80=int((~valid).sum()),missing_q_L_selected48=int(a['query_xy_missing'].sum()),finite_invalid32=int((~valid&~a['query_xy_missing']).sum()),
    correspondence_boundary_rows=int(a['correspondence_boundary'].sum()),physical_initializer_identity_certified=False,
    native_rows={str(k):int((a['native_slot']==k).sum()) for k in range(8)},material_aspect_rows={k:v['N'] for k,v in groups.items()},geometry_rows=dict(Counter(str(x) for x in a['geometry_family'])),variant_rows=dict(Counter(str(x) for x in a['variant'])),
    q_reference_position_band_counts=dict(Counter(r['position_band'] for r in rows)),owner_class_order=a['owner_class_order'].tolist(),joint_class_order=a['joint_class_order'].tolist(),
    nine_soft_distribution_summaries=summaries,semantic_by_soft_distribution=semantic,material_aspect_tables=groups,native8_tables=native,
    available4016_literal_point_threeclass_mass=dict(zip(('ALL_TARGET_incident_plus_other','FOREIGN','BG'),sumweights(known3[valid]))),
    class_mass_is_fractional_sample_equivalent_not_integer_label_counts=True,
    soft_nonzero_and_unit_counts_are_diagnostics_not_hard_GT_thresholds=True,
    full96_crop_any_or_whole_end_presence_array_exists=False,source_keys=[dict(name=k,dtype=str(v.dtype),shape=list(v.shape)) for k,v in a.items()])
write(O/'TRAIN4096_AUDITED_NUMBERS.json',numbers)

# Resolve only original TRAIN input metadata; stat payload paths but do not open/hash them.
scene_inputs=read(protocol['original_scene_input']['path'])['rows'];facts=read(protocol['original_crop_facts']['path'])['rows']
train_scene={int(r['scene_index']):r for r in scene_inputs if r['split']=='TRAIN'}
for k in ('original_scene_input','original_crop_facts'):
    check(reads[str(Path(protocol[k]['path']).resolve())]==protocol[k]['sha256'],'Original input metadata bound SHA')
selection=[];smoke=[];variants=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
for material,aspect in sorted({(r['factors']['material'],r['factors']['aspect']) for r in rows}):
    b=min(r['base_index'] for r in rows if (r['factors']['material'],r['factors']['aspect'])==(material,aspect))
    rr=[r for r in rows if r['base_index']==b]
    check(len(rr)==32 and set(r['native_slot'] for r in rr)==set(range(8)) and set(r['variant'] for r in rr)==set(variants),'Smoke four variants native8 all retained')
    selection.append(dict(material=material,aspect=aspect,base_index=b,geometry_family=rr[0]['factors']['geometry_family'],scene_indices=sorted({r['scene_index'] for r in rr}),original_global_indices=[r['original_global_index'] for r in rr]))
    for r in sorted(rr,key=lambda r:r['original_global_index']):
        f=facts[r['original_global_index']];sc=train_scene[r['scene_index']]
        check(f['occurrence_id']==r['original_occurrence_id'] and f['native_slot']==r['native_slot'] and f['query_xy']==r['query_xy'] and f['L']==r['L'],'Smoke original q/L identity exact')
        check(f['raw_RGB_PNG']==sc['raw_RGB_PNG'] and f['decoded_RGB_sha256']==sc['decoded_RGB_sha256'],'Smoke source RGB alias metadata exact')
        check(all(Path(binding['path']).is_file() for binding in (sc['full_RGB'],sc['raw_RGB_PNG'],f['rgb96_chunk'])),'Smoke payload path readiness stat only')
        smoke.append(dict(original_global_index=r['original_global_index'],original_occurrence_id=r['original_occurrence_id'],scene_index=r['scene_index'],base_index=b,native_slot=r['native_slot'],split='TRAIN',variant=r['variant'],factors=r['factors'],
            original_query_xy=r['query_xy'],original_L=r['L'],original_input_flags=r['original_input_flags'],original_initial_points8=f['initial_points'],original_q8_context_box_xyxy=f['context_box_xyxy'],
            full_RGB_packet=sc['full_RGB'],full_RGB_key=sc['full_RGB_key'],full_RGB_format=sc['full_RGB_format'],raw_hw=sc['raw_hw'],raw_RGB_PNG=sc['raw_RGB_PNG'],decoded_RGB_sha256_inherited=sc['decoded_RGB_sha256'],
            saved_crop_chunk=f['rgb96_chunk'],saved_crop_row=f['rgb96_row'],saved_crop_keys=dict(rgb96='rgb96',valid96='valid96',global_index='global_index',crop_constructed='crop_constructed'),saved_rgb96_sha256_inherited=f['rgb96_sha256'],saved_valid96_sha256_inherited=f['valid96_sha256'],
            declared_CF_to_canonical=r['declared_canonical_index'],declared_incident_part_ids=r['incident_part_ids'],declared_primary_part_id=r['primary_part_id'],
            prompt_authority='Original q8/context only if used; declared GT part/canonical fields are audit metadata, not automatically admitted SAM prompt or training target.',
            future_probe_only=True,training_admission=False))
smoke_path=O/'SAM_SMOKE_INPUTS128_METADATA_ONLY.jsonl'
with smoke_path.open('x') as f:
    for r in smoke:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
smoke_receipt=dict(completed=True,PASS=not errors,selection_rule='For each of the four existing TRAIN (material,aspect) groups choose the smallest original base_index; retain its original 4 variants in source order and all native slots0..7. No model output/teacher success/search/replacement criterion.',
    selected_bases=selection,rows=len(smoke),scenes=len({r['scene_index'] for r in smoke}),native_slots=list(range(8)),payload_paths_exist=True,payload_body_or_header_or_hash_reads=0,
    all128_original_input_available=all(r['original_input_flags']['query_available'] for r in smoke),input_manifest=bind(smoke_path),
    actual_SAM_model_or_runtime_probe_runs=0,readiness_is_metadata_only=True,
    coverage_limit='Runtime smoke only: this minimum selection happens to contain BOARD_BLOCK_RUNNER alone and one TRAIN asset/camera family. RIB_LEG_RUNNER, remaining source assets/views/rotations, full native visibility/self/external/coordinate qualification and real transfer remain full-goal obligations.',
    original_per_variant_q_preserved=True,BEFORE_q_not_copied_to_other_variants=True,GT_part_metadata_not_automatically_model_inputs=True,full_WOOD_SQUARE_native8_goal_preserved=True)
write(O/'SAM_SMOKE_READINESS.json',smoke_receipt)

code_paths={
    'V173_census':V/'census_preparation/build_full_source_census.py',
    'V172_owner_sampling':P/'query_visibility_source_mask_semantics_audit_20261024_v172/measurement_preparation/measure_saved_source_semantics.py',
    'V127_constructive_mesh_ray_teacher':P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py',
    'V129_scaled_geometry_and_mapping':P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/corpus_math.py',
    'V129_renderer':P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/render_corpus_once.py',
    'V132_original_crop_cache':P/'query_visibility_actual_initializer_source_20261024_v132/frozen_visibility_preparation/run_frozen_visibility.py',
    'V175_TRAIN_supervision_contract':P/'query_visibility_direct_query_owner_control_20261024_v175/ROOT_SCOPE_BEFORE_DIRECT_OWNER_SOURCE_FIT.json',
    'V172_source_definition_authority':P/'query_visibility_source_mask_semantics_audit_20261024_v172/source_definition_review_a/REVIEW.json'}
bindings={k:bind(p) for k,p in code_paths.items()}
for path,before in reads.items():check(sha(path)==before,'Audit inputs unchanged')
review=dict(schema='V179_SOURCE_TRAIN_GOAL_ALIGNMENT_A_v1',completed=True,PASS=not errors,errors=errors,scope='V173 TRAIN4096 soft sidecar plus original input/code metadata and public SOURCE TRAIN contract only. No validation-soft, held/private body or metric, model output/feature/weight, RGB/mask/depth body read.',
    exact_inputs=[dict(path=k,sha256=v) for k,v in reads.items()],code_bindings=bindings,code=bind(__file__),numbers=bind(O/'TRAIN4096_AUDITED_NUMBERS.json'),smoke_readiness=bind(O/'SAM_SMOKE_READINESS.json'),smoke_rows=bind(smoke_path),checks=dict(checks),number_checks=sum(checks.values()),
    supervision_contracts=[
      dict(question='Original native D/E/S/U',authority='V127 region_teacher called by V129 at the declared source anchor/endpoint-edge-face witnesses; not original initializer q.',keys='original_semantic_state; original metadata incident_part_ids/primary_part_id/active_witness_indices; upstream records_CF',
           definition='Source anchors are exact vertices of designated deck/runner boxes. The37 finite registered witnesses contain endpoint0,9 edge samples,27 face-grid samples. D requires target-alone at least2 front axes and all active rays D, then all active combined rays D. E requires target-alone D and all active combined rays intercepted by owner999. S is all37 target-alone rays self-hidden. Mixture, ties, misses, insufficient front-axis condition or any projected witness out of frame gives U. This is the native region-teacher question, not full dense part visibility.'),
      dict(question='Literal owner at original q',authority='Saved original scene raster part/ambiguity and declared per-native incident set, bilinear onehot distribution at q.',keys='scene_owner_point[4096,6]; BEFORE_owner_point[4096,6]',definition='T=incident plus OTHER_TARGET if collapsed for owner3. FOREIGN=owner999. Background and ambiguity/invalid stay separate. This does not certify that q represents the intended physical native role.'),
      dict(question='Center 2x2 and32x32 on saved96 grid',authority='Original q/L exact96 sample grid; averages of source categorical bilinear distributions.',keys='scene_owner_central4/central32; BEFORE_owner_central4/central32; joint_central4/central32',definition='Central4 means four sample positions indices47:49, central32 means1024 sample positions indices32:64. They are area-average owner mass, not an ANY-pixel visibility or whole-end presence boolean. OTHER_TARGET is visible selected-pallet structure but can be a different deck/runner, facet or end.'),
      dict(question='Crop-neighbor intended target-end visible',authority='Not directly stored as a V173 target.',keys='NO full96 owner array / NO crop-any-target / NO intended-end-presence array. Original crop_constructed and context_available are input facts, not labels.',definition='Nonzero central32 incident mass only establishes sampled visibility of that entire designated part ID in the local region; part ID spans a board/runner and does not isolate the intended material end. Crop presence, intended-end region visibility and exact corner identity require separate fixed spatial definition.'),
      dict(question='Same-pixel causal occlusion',authority='Per-source-pixel BEFORE/SCENE conjunction before bilinear averaging.',keys='joint_point/central4/central32[4096,7]; nearest_joint_class_index[4096]',definition='EXTERNAL_OVER_BEFORE_TARGET uses any BEFORE target part and scene owner999 strictly in front. FOREIGN_WITHOUT_BEFORE_TARGET is foreign over BEFORE background. Neither class is the original37-ray native E rule or intended-end-only occlusion.'),
      dict(question='Public SOURCE TRAIN owner253',authority='Frozen V175 public TRAIN contract, assistant-weak observations of all4 saved96 center2x2 samples with clearly identified selected instance.',keys='Contract counts T53/F52/BG148; public per-class mean CE distinct domain.',definition='Center2x2 weak supported label is not exact original-q point gold, crop-presence label, native E/S label or coordinate-quality gold. No held owner labels or metrics were read.')],
    TRAIN_evidence_KO=[
      '4096행 D1906/E134/S512/U1544, input가능4016/U80를 그대로 보존했다. q/L/선택instance missing48과 유한한 입력무효32를 BG나native U와 합치지 않는다.',
      '원native D1906 중 point측정1884의 BG 가중질량은1179.4892556602135, foreign은11.134514324367046이다. 이는 원q 소속과 native 부위-ray 가시성이 다르다는 사실이며 원native D의 오기 증명이 아니다.',
      '원native E134 중 point측정132의 foreign질량105.09368359996006 중55.43141998536885는 BEFORE target위 외부물체,49.66226361459121는 BEFORE 배경위 외부물체이다. q중심foreign과 의도한native부위 external을 동일시할 수 없다.',
      '원native S512 중 point측정506에서 OTHER_TARGET 질량494.2769234161824이다. 숨은native와 전면 같은팔레트 구조가 동시에 존재할 수 있다.',
      'TRAIN의 CF/V4 correspondence_boundary는0/4096이다. 그러나 initializer_physical_native_identity_certified=false는 유지된다. 학습native 역할·실제모서리대응이 이 숫자만으로 인증되지는 않는다.'],
    wrong_virtual_ray_teacher_assessment='No evidence here of a free-floating virtual-box anchor: V127 asserts each chosen anchor is a literal designated-part box vertex, and V129 scales anchors/vertices/patches together. The strict finite-ray/all-active/two-front-axis rule is a prospective source definition, which may differ from user desired visible-end/nearby-content semantics. This audit does not independently rerun rays or certify every saved teacher; q/raster disagreement alone is not a wrong-teacher proof.',
    source_geometry_safe_preparation=dict(feasible=True,definition='Use frozen constructive primary-part ID plus the small incident face/edge end patch native_patch_quads/native_witness_points registered before q. Project that fixed patch with existing source camera; derive separate scene-visible end support and target-alone/end-occlusion support from original part/depth/triangle provenance after a separately frozen source-only operator. Do not substitute whole-board presence or q-centered owner.',
        benefits='Source geometry gives stable material-part/end identity, known same-material foreign owner999 and BEFORE counterfactual, avoiding pixel-color pseudo-GT and result-based nearest-vertex/permutation selection.',
        limits='This would create a new explicit end-region supervision sidecar, not reinterpret native D/E/S or original q. Dense surface/end visibility, all8 identity and full real-data qualification remain separate. SAM masks/runtime cannot by themselves certify native identity, external/self causality or physical corner coordinates.'),
    concrete_next_action='Use the frozen128 TRAIN-only smoke manifest (bases0/1/16/17;16 original scenes;4variants×8 each) for a source SAM runtime probe under root source-only entry. Preserve original per-variant q/L/native/flags and inherited raw/crop hashes. Treat success only as runtime/input feasibility; any future part-end target must retain all source groups/native8 and be defined before results.',
    code_evidence=[dict(code='V127_constructive_mesh_ray_teacher',lines='50-80',claim='Constructive parts, exact material vertices, fixed37 witness patches; not q-derived or nearest-vertex labels.'),dict(code='V127_constructive_mesh_ray_teacher',lines='89-135',claim='Ray/region D/E/S/U, primarypart, active set and target-alone precedence.'),dict(code='V129_scaled_geometry_and_mapping',lines='42-54,72-76',claim='Common positive scale, prospective selected patch/foreign placement, declared CF mapping once; sham does not guarantee every native unoccluded.'),dict(code='V129_renderer',lines='56-72,78-80',claim='Same sourcegeometry ray/raster, owner999 independent of material, immutable target surfaces and original native records.'),dict(code='V172_owner_sampling',lines='28-64,99-109',claim='Incident-vs-othertarget, float64 originalq bilinear, exact saved96 central4/32, no GT-centered crop.'),dict(code='V173_census',lines='39-91,119-152,207-252',claim='Joint7 per-pixel conditions, no crop-any target, nine soft arrays and immutable metadata.'),dict(code='V132_original_crop_cache',lines='282-291',claim='Original rgb96/valid96/crop_constructed/global_index payload keys and row references.')],
    no_VAL_soft_held_private_or_metric_reads=True,new_fit_model_encoder_SAM_PnP_q_correction_or_label_rewrite=0,source_binary_reads_only=[str(train_path)],rawRGB_mask_depth_mesh_or_crop_payload_body_header_hash_reads=0,public_TRAIN_contract_only_no_public_score_or_label_body_reads=True,
    full_WOOD_SQUARE_native8_goal_preserved=True,training_admission=False,selected_model=None,fully_solved=False)
write(O/'REVIEW.json',review)
print(json.dumps(dict(PASS=review['PASS'],checks=review['number_checks'],errors=errors,rows=4096,smoke_rows=len(smoke),review=bind(O/'REVIEW.json'),numbers=bind(O/'TRAIN4096_AUDITED_NUMBERS.json'),smoke=bind(O/'SAM_SMOKE_READINESS.json')),ensure_ascii=False))
