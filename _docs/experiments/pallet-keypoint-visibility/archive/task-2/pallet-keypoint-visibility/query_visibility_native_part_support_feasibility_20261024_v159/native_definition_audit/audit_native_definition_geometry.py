"""Public-source definition and CAD facts only. No RGB, model, fit or labels."""
import ast, hashlib, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent; V159=OUT.parent; P=V159.parent
V151=P/'query_visibility_mr6d_public_native_intake_20261024_v151'
V153=P/'query_visibility_mr6d_full_public_source_20261024_v153'
V158=P/'query_visibility_train_query_owner_witness_20261024_v158'
V74=P/'query_visibility_loco_native_context_20261023_v74/source_contract'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bind=lambda p:{'path':str(p),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
scope=V159/'ROOT_SCOPE_BEFORE_SOURCE_NATIVE_SUPPORT_BODIES.json'
assert sha(scope)=='49ca7c44ac80611c8be988b9c30f020528d4090ee40a31f8825a405690c25be3'
contract=read(V74/'NATIVE_ORDER_AND_INPUT_CONTRACT.json')
projection_seal=read(V153/'native_reference_preparation/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json')
assert sha(V74/'native_context_contract_v2.py')==projection_seal['exact_original_V74_AST']['original_code']['sha256']
mesh=V151/'factual_sample_v1/original/models/obj_000001.ply'
seal=read(V151/'factual_sample_v1/TINY_SOURCE_DOWNLOAD_COMPLETE.json')
mesh_lineage=next(r for r in seal['files'] if r['relative_path']=='models/obj_000001.ply')
payload=mesh.read_bytes();assert sha(mesh)==mesh_lineage['sha256']
git_sha1=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
assert git_sha1==mesh_lineage['expected_source_content_ID']
lines=payload.decode().splitlines(); start=lines.index('end_header')+1
nv=int(next(x for x in lines[:start] if x.startswith('element vertex ')).split()[-1])
nf=int(next(x for x in lines[:start] if x.startswith('element face ')).split()[-1])
vertices=np.array([[float(x) for x in s.split()[:3]] for s in lines[start:start+nv]],dtype=np.float64)
face_rows=[[int(x) for x in s.split()] for s in lines[start+nv:]]
assert len(face_rows)==nf and all(r[0]==3 for r in face_rows)
faces=np.array([r[1:] for r in face_rows]);tri=vertices[faces]
assert vertices.shape==(424,3) and faces.shape==(240,3)
mi=read(V151/'provided_metadata/models/models_info.json')['1']
dims=np.array([mi[f'size_{c}'] for c in 'xyz']);lo=np.array([mi[f'min_{c}'] for c in 'xyz']);center=lo+dims/2
assert np.array_equal(vertices.min(0),lo) and np.array_equal(vertices.max(0),lo+dims)
assert np.array_equal(dims,[800,1200,144]) and np.array_equal(center,[0,0,0])
signs=np.array(contract['writer_fixed_Zup_signs']);corners=center+signs*dims/2
area_vectors=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);areas=np.linalg.norm(area_vectors,axis=1)/2
assert np.all(areas>0)
normal=np.round(area_vectors/np.linalg.norm(area_vectors,axis=1)[:,None],12)
fixed=[]
for k,point in enumerate(corners):
 match=np.flatnonzero(np.all(vertices==point,axis=1))
 incident=np.flatnonzero(np.isin(faces,match).any(axis=1))
 assert len(match)>0 and len(incident)>0
 nearest=float(np.linalg.norm(vertices-point,axis=1).min());assert nearest==0.
 fixed.append({'fixed_canonical_index':k,'reference_model_XYZ_mm':point.tolist(),
               'nominal_aabb_corner':True,'exact_matched_vertex_indices':match.tolist(),
               'incident_triangle_indices':incident.tolist(),'incident_triangle_unit_normals':np.unique(normal[incident],axis=0).tolist(),
               'nearest_vertex_distance_mm':nearest,'nearest_triangle_surface_distance_mm':0.,
               'zero_surface_distance_proof':'Exact matching vertex occurs in nondegenerate incident triangles.',
               'nominal_structural_part':'outer top deck-board end corner' if point[2]>0 else 'outer bottom-runner end corner',
               'real_image_material_corner_certified':False})
layers=[]
for z in np.unique(vertices[:,2]):
 m=(np.ptp(tri[:,:,2],axis=1)==0)&(tri[:,0,2]==z)
 bounds=np.unique(np.c_[tri[m,:,:2].min(1),tri[m,:,:2].max(1)],axis=0)
 layers.append({'object_Z_mm':float(z),'horizontal_triangles':int(m.sum()),'triangle_area_sum_mm2':float(areas[m].sum()),'XY_triangle_bounds_mm':bounds.tolist()})
geometry={'schema':'V159_INDEPENDENT_NOMINAL_CAD_NATIVE8_GEOMETRY_FACTS','completed':True,
          'mesh':bind(mesh),'pinned_mesh_url':mesh_lineage['pinned_url'],'git_blob_sha1_recomputed':git_sha1,
          'vertices':nv,'triangles':nf,'bounds_min_mm':vertices.min(0).tolist(),'bounds_max_mm':vertices.max(0).tolist(),
          'fixed_canonical_corners':fixed,'horizontal_layers':layers,
          'physical_UP_evidence':'Nominal +Z72 plane has five continuous deck boards; nominal -Z72 plane has three bottom runners. This proves asset anatomy, not real-image calibration or initializer-channel alignment.',
          'all8_nominal_CAD_material_corners':True,'all8_real_visible_corners_certified':False,
          'no_virtual_hole_at_exact_nominal_eight_in_this_mesh':True,
          'limitation':'These are task AABB reference corners that happen to be material vertices in MR obj1 CAD. This does not change the general virtual-reference native8 definition or certify real-instance point coordinates.'}
write(OUT/'EXACT_NATIVE8_NOMINAL_CAD_SUPPORT.json',geometry)

tops=[r for r in fixed if r['reference_model_XYZ_mm'][2]>0];bottoms=[r for r in fixed if r['reference_model_XYZ_mm'][2]<0]
names=[contract['deployed_native_slot_order'][str(k)] for k in range(8)]
requirements=[
 {'id':'instance_identity','required':'Bind selected target to (official split, scene, frame, scene_gt list index, obj_id) and matching IMID_GTID masks; another obj1 pallet remains foreign.','missing_proof':'Class identity or literal q-owner alone cannot establish target-instance correspondence.'},
 {'id':'role_identity','required':'Use stored task channel k and immutable GT-only dynamic-to-fixed mapping from provided geometry, with source +Z deck anatomy and face/LR margins recorded.','missing_proof':'Original historical writer code/tie behavior is not recovered; no nearest-q or fitted permutation is permitted.'},
 {'id':'part_support','required':'Review the expected deck-board or bottom-runner neighborhood around the nominal reference, distinguishing self boundary, hole/gap, same-material other instance, and external object.','missing_proof':'Exact CAD membership is neither real material correspondence nor a universal prerequisite for virtual-native part review.'},
 {'id':'reference_uncertainty','required':'Keep nominal CAD/pose projection and actual expected part separate; retain UNKNOWN when dimensional variation or projection alignment makes correspondence ambiguous.','missing_proof':'Release supplies no per-frame native coordinate covariance or hard <=3px bound.'},
 {'id':'native_visibility','required':'Assess expected native part in context and preserve self/external/unknown distinctions independently of literal q-center owner and pose/mask proxies.','missing_proof':'Target surface at a projected ray does not identify a rear native reference; foreign at q is not proof of native external E.'},
 {'id':'mask_depth_evidence','required':'Keep full-mask, visible-mask, measured valid depth, registration and overlap flags separately with original mask-generation uncertainty.','missing_proof':'Per-instance masks are nonexclusive; MR-specific visibility mode/delta and a unique overlap cause are not certified.'},
]
native_rows=[]
for k,name in enumerate(names):
 is_top=k in [0,1,4,5];candidates=tops if is_top else bottoms
 native_rows.append({'native_slot':k,'native_name':name,
  'definition_class':'CAMERA_DYNAMIC_CUBOID_ROLE',
  'fixed_physical_identity':False,'front_or_rear':'front' if k<4 else 'rear',
  'nominal_expected_part':'outer deck-board end corner' if is_top else 'outer bottom-runner end corner',
  'possible_fixed_canonical_indices':[x['fixed_canonical_index'] for x in candidates],
  'possible_reference_XYZ_mm':[x['reference_model_XYZ_mm'] for x in candidates],
  'nominal_CAD_material_support_all_possible_fixed_indices':True,
  'real_instance_material_corner_status':'UNCERTIFIED',
  'native_visibility_or_coordinate_gold_status':'UNKNOWN_NOT_CREATED',
  'rear_pair_policy':None if k<4 else f'Continues front slot {k-4} across opposite depth edge; never independently image-X sorted.',
  'required_proof_ids':[r['id'] for r in requirements],
  'not_inferred_from_native_name':'front does not guarantee visible; rear does not automatically mean self-hidden.'})
(OUT/'NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in native_rows))

source_rows_path=V158/'joined_owner_witness_v1/ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl'
source_rows=[json.loads(x) for x in source_rows_path.read_text().splitlines()]
assert len(source_rows)==390 and all(r['split']=='SOURCE_DEVELOPMENT' for r in source_rows)
source_summary={'fixed_rows':390,'distinct_frames':len({r['frame_index'] for r in source_rows}),
 'source_scene_counts':dict(Counter(r['source_id'].split(':')[0] for r in source_rows)),
 'native_slot_counts':{str(k):sum(r['native_slot']==k for r in source_rows) for k in range(8)},
 'original_weak_role_counts':dict(Counter(r['original_role_state'] for r in source_rows)),
 'target_material_scope':'MR6D obj1 wooden Euro pallet, nominal 800x1200x144mm; no selected SQUARE or other pallet material/design.',
 'all_native_slots_retained':True,'native6_known_weak_role_rows':2,'same_native_geometry390_join_created_here':False}
docs=read(OUT/'PRIMARY_DOCUMENT_SOURCE_BINDINGS.json')
inputs=[scope,V74/'NATIVE_ORDER_AND_INPUT_CONTRACT.json',V74/'NATIVE_ORDER_SOURCE_AUDIT.json',V74/'native_context_contract_v2.py',
 V151/'provided_metadata/models_names.json',V151/'provided_metadata/models/models_info.json',V151/'provided_metadata/README.md',
 V151/'metadata/PINNED_REMOTE_SOURCE.json',V151/'factual_sample_v1/TINY_SOURCE_DOWNLOAD_COMPLETE.json',
 V151/'native_readiness_review/actual_tiny_audit/RESULTS.json',V151/'native_readiness_review/actual_tiny_audit/BOP_OFFICIAL_VISIBILITY_SOURCE.py.txt',
 V153/'native_reference_preparation/project_all_source_native_references.py',V153/'native_reference_preparation/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json',
 V153/'native_reference_preparation/SOURCE_MANIFEST_IDENTITY_JOIN.json',V153/'full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json',source_rows_path,
 Path('/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py')]
result={
 'schema':'V159_NATIVE_DEFINITION_AND_FEASIBILITY_SOURCE_ONLY','completed':True,'UTC':datetime.now(timezone.utc).isoformat(),
 'status':'DEFINITION_AND_NOMINAL_CAD_SUPPORT_ESTABLISHED_REAL_NATIVE_SUPPORT_UNCERTIFIED',
 'root_scope':bind(scope),'audit_code':bind(Path(__file__)),'exact_local_input_bindings':[bind(p) for p in inputs],
 'primary_remote_documents':docs,
 'authority_separation':{
  'MR6D_primary_release':'Provides nominal object mesh, BOP-style instance poses/camera/depth/masks. No official MR6D camera-dynamic native8 semantic labels identified in reviewed release metadata/docs.',
  'task_native8':'Eight camera-dynamic cuboid roles are the existing task/trained-channel contract. MR projections are derived reference preparation using that contract, not native8 gold supplied by MR authors.',
  'historical_writer_provenance':'Local original C#/Blender historical writer was not recovered in the existing audit. V74 reports exact empirical normal-facing reconstruction for 60000 stored synthetic records; this run verifies bound code and nominal CAD, not those 60000 bodies.',
  'no_order_optimization':'No new permutation optimized against queries, old roles, predictions, or real pixels. Stored dynamic arrays remain in place.'},
 'exact_coordinate_and_order_contract':{
  'unit':'millimeter','handedness':'right-handed','origin_model_mm':[0,0,0],
  'model_axes_bounds_mm':{'X':[-400,400],'Y':[-600,600],'Z':[-72,72]},
  'axis_scope':'X=800 and Y=1200 are fixed nominal model spans, not invariant camera-dynamic front width/depth. Source +Z is anatomical deck side; camera coordinates use OpenCV +Z forward.',
  'projection':'X_camera = R_m2c X_model + t_m2c; homogeneous image = K X_camera, then divide by camera Z. Row-vector code implements points @ R.T + t.',
  'camera_object':'solve(R, -t), used in V153; exact linear inverse avoids silently treating rounded R as perfectly orthonormal.',
  'fixed_canonical_XYZ_mm':corners.tolist(),
  'native_names':names,'front_candidates_fixed_indices':[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2]],
  'front_choice':'Maximum outward-normal cosine to camera from cuboid center, among four vertical faces only; top/bottom excluded.',
  'top_bottom':'V74 uses source world Z; V153 MR adaptation passes centered model coordinates, retaining model +Z deck anatomy for upright source objects. This is an explicit adaptation, not a recovered official MR native8 rule.',
  'front_left_right':'Front top endpoints sorted by projected image X; slots0/1 are left/right top and slots3/2 their vertical bottom partners.',
  'rear':'Slots4..7 continue front0..3 through opposite depth edges; rear is not independently image-X sorted. A rear point can lie on a visible adjacent side/top; rear name does not force S.',
  'permutation_direction':'dynamic[k] = fixed[perm[k]]; never reapply to already dynamic stored arrays.',
  'ties':'np.argmax/face list order in replay helper. Exact historical writer tie policy uncertified. Face-score and top-LR margins are evidence flags, not automatic semantic thresholds.',
  'centroid':'Separate projection of model center (0,0,0); index8 is outside native8 and is not the mean of eight 2D projections.',
  'training_visibility_flag_caveat':'prepare_yolo_pose.py writes v=2 for known in-padded-canvas coordinates; this transport flag is not native semantic D.'},
 'CAD_geometry_recalculation':bind(OUT/'EXACT_NATIVE8_NOMINAL_CAD_SUPPORT.json'),
 'native_expected_part_table':bind(OUT/'NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl'),
 'per_native_conclusion':'All native0..7 are dynamic roles over nominal AABB corners. In this Euro CAD all possible corresponding fixed corners are material vertices. Real-image physical corner/part/visibility identity remains uncertified for every role until target-instance, expected-part and alignment proof exists.',
 'source_fixed_row_metadata':source_summary,
 'pose_and_dimension_uncertainty':{
  'primary_source':'https://arxiv.org/html/2508.13775v1#S3.SS2',
  'scope':'Static validation annotations use tracked camera and manually refined initial object poses. The manually designed Euro mesh is nominal; real pallets vary dimensionally.',
  'quantitative_language_provided':{'pallet_dimension_variation':'few centimeters','expected_static_annotation_accuracy':'low-centimeter range'},
  'unprovided':['numeric hard upper bound','per-frame translation/rotation covariance','per-native pixel covariance','per-instance as-built pallet dimensions','<=3px coordinate certificate'],
  'consequence':'Do not invent +/-20mm or a pixel bound, fit a correction, or certify exact q support from nominal projection alone. CAD membership and valid rotation matrices do not remove this uncertainty.',
  'source_calibration_quality_vs_bound':'The paper describes a process and expected accuracy; this is not a measured guarantee for each selected point/frame.'},
 'masks_and_depth':{
  'BOP_identity':'IMID_GTID uses scene_gt list index, not object class ID.',
  'BOP_mask_scope':'full mask is object silhouette; visible mask is per-instance visible-surface silhouette proxy.',
  'nonexclusive':'Existing independently checked source frame5/frame9 visible masks overlap19371/16152 pixels; most overlap has positive measured depth. No unique cause inferred.',
  'BOP19_formula':'(d_model - d_test <= delta OR d_test == 0) AND d_model > 0; this is not an exclusive instance-owner rule.',
  'MR_generation_uncertainty':'Actual release mask visibility mode/delta not certified. Do not import the toolkit generic delta as MR ground truth.',
  'negative_mask':'Can reflect virtual/mesh gap, nominal pose/dimension drift, border, OOF or missing evidence; not automatically native external E.',
  'positive_mask':'Some target surface at the ray does not prove the expected rear/native point is visible.'},
 'required_native_part_proof':requirements,
 'feasible_now':['Use exact nominal mesh anatomical neighborhoods to prepare source-only structural support review sheets, keeping immutable q, nominal reference and uncertainty separate.',
  'Quantify same-selected-instance mask/reference/depth support as diagnostics with UNKNOWN and nonexclusive flags; another agent prepares all390 facts.',
  'Review coarse expected part support where context is clear; preserve assistant-weak status and never infer native D/E or point-quality labels from q-center owner.'],
 'missing_coverage':['Real SQUARE pallet native8 support and material/part definitions are not supplied by MR Euro obj1.',
  'Diverse WOOD geometries/material designs and real-instance precise native coordinates remain unfilled.',
  'Native6 has only2 currently known-weak-role rows; all8 full final objective remains required.'],
 'concrete_preparation_completed':{'action':'Recomputed pinned CAD vertex/triangle support and published immutable eight-role expected-part/proof requirement dictionary.',
  'outputs':[bind(OUT/'EXACT_NATIVE8_NOMINAL_CAD_SUPPORT.json'),bind(OUT/'NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl')],
  'no_native_label_creation':True},
 'next_safe_preparation_action':'Attach each fixed source query to selectedGT and its immutable nominal expected deck/runner neighborhood in model/reference view; root fixed12 panels can test whether structural support is visually reviewable. Record target identity, self/external ambiguity, pose/mask-edge uncertainty separately and keep unsupported native certification UNKNOWN. Add source-specific SQUARE geometry/real support later without changing final scope.',
 'accesses':{'public_mesh_geometry_read':True,'public_source_row_metadata_read':True,'public_primary_documents_browsed':True,
  'RGB_or_depth_or_mask_images_read_here':0,'held45_RGB_or_labels_reviewed':0,'private_current_or_V112_access':0,
  'model_weights_forward_fit_PnP_or_qL_change':0,'new_native_D_E_S_labels':0,'training_admission':False},
 'final_scope':'Full WOOD/SQUARE/native8 objective retained; source-only Euro wood audit is partial coverage.',
 'physical_pixel_gold':False,'goal_solved':False,
}
write(OUT/'DEFINITION_AND_FEASIBILITY.json',result)
print(json.dumps({'definition':bind(OUT/'DEFINITION_AND_FEASIBILITY.json'),'geometry':bind(OUT/'EXACT_NATIVE8_NOMINAL_CAD_SUPPORT.json'),'native_table':bind(OUT/'NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl'),'source_metadata':source_summary},ensure_ascii=False,indent=2))
