"""Source-only nominal camera-facing references; original queries never move.

--prepare reads code/protocol/spec text and runs invented fixtures only.
--entry additionally reads the root-bound sealed C metadata outputs, never assets.
"""
from pathlib import Path
from collections import Counter
import argparse, ast, hashlib, json, math, os, stat, traceback
import numpy as np

H = Path(__file__).absolute().parent
V, P = H.parent, H.parent.parent
C = V / 'selected_instance_preparation_c'
CP = C / 'PROTOCOL_BEFORE_TRAIN_INSTANCE_METADATA_BODIES_V2_NESTED_R.json'
CC = V / 'selected_instance_metadata_diagnostic_v2_nested_R/COMPLETE.json'
SPEC = H / 'FORMULA_AND_OUTPUT_SPEC_BEFORE_ACTUAL_BODIES.json'
SCOPE = V / 'ROOT_SCOPE_BEFORE_SOURCE_CORRESPONDENCE_BODIES.json'
ORDER = P / 'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
ORDER_SHA = '4ddd134aa26da1d676628bd6987a161ab36f09328dfdaa89bedd213e397755ab'
CROP = P / 'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_SHA = '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
PROTOCOL = H / 'PROTOCOL_BEFORE_SEALED_SOURCE_GT_REFERENCE_METADATA.json'
CHECKS = H / 'GENERATED_INVENTED_CHECKS.json'
OUT = V / 'nominal_source_native_projection_v1'
SCENES = {'000000', '000002', '000005', '000006'}
NAMES = ('front_top_left', 'front_top_right', 'front_bottom_right',
         'front_bottom_left', 'rear_top_left', 'rear_top_right',
         'rear_bottom_right', 'rear_bottom_left')

def require(v, message):
    if not bool(v):
        raise AssertionError(message)

def permitted(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'Explicit absolute path')
    require(all('v112' not in s.casefold() for s in p.parts), 'Forbidden name before access')
    require(P in p.parents, 'Source project path only')
    return p

def safe_open(path):
    p = permitted(path)
    parts = p.relative_to(P).parts
    fd = os.open(P, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        final = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode), 'Ordinary file only')
    return os.fdopen(final, 'rb')

def sha(path):
    h = hashlib.sha256()
    with safe_open(path) as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def bind(path):
    return {'path': str(permitted(path)), 'sha256': sha(path)}

def checked(b):
    require(isinstance(b.get('sha256'), str) and len(b['sha256']) == 64, 'Exact SHA required')
    require(sha(b['path']) == b['sha256'], 'Binding drift: ' + b['path'])
    return permitted(b['path'])

def read(path):
    with safe_open(path) as f:
        return json.load(f)

def lines(path):
    with safe_open(path) as f:
        return [json.loads(s) for s in f if s.strip()]

def write(path, value):
    with permitted(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def writelines(path, rows):
    with permitted(path).open('x') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')

def load_order():
    require(sha(ORDER) == ORDER_SHA, 'Pinned V74 pure order')
    with safe_open(ORDER) as f:
        tree = ast.parse(f.read().decode())
    constants = {'FIXED_ZUP_SIGNS', 'VERTICAL_FIXED_FACES'}
    functions = {'outward_normals', 'normal_facing_permutation'}
    nodes = [n for n in tree.body if
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
             or isinstance(n, ast.FunctionDef) and n.name in functions]
    require(len(nodes) == 4, 'Only two exact constants and two exact pure functions')
    ns = {'np': np}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ORDER), 'exec'), ns)
    proof = {'source': bind(ORDER), 'exact_AST_nodes': sorted(constants | functions),
             'AST_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()}
    return ns, proof

def footprint(reference_xy, q, L, hw):
    """No RGB sampling or query-validity modification; formula frozen separately."""
    unknown = {'valid_nominal_discrepancy': False, 'reason': 'MISSING_NONFINITE_SENTINEL_Q_OR_INVALID_L',
               'delta_xy_px': None, 'nominal_distance_px': None, 'delta_xy_over_L': None,
               'nominal_distance_over_L': None, 'reference_crop_index_xy': None,
               'inside_outer_crop_extent': None, 'inside_sample_center_rectangle': None,
               'reference_in_original_sampler_valid_domain': None}
    try:
        q, ref, L = np.asarray(q, np.float64), np.asarray(reference_xy, np.float64), float(L)
    except (TypeError, ValueError):
        return unknown
    if q.shape != (2,) or not np.isfinite(q).all() or np.array_equal(q, [-1., -1.]) or not math.isfinite(L) or L <= 0:
        return unknown
    if ref.shape != (2,) or not np.isfinite(ref).all():
        return dict(unknown, reason='NONFINITE_OR_UNAVAILABLE_NOMINAL_REFERENCE')
    d = ref - q
    t = 96. * d / L + 47.5
    h, w = hw
    return {'valid_nominal_discrepancy': True, 'reason': 'NOMINAL_DIAGNOSTIC_NOT_PHYSICAL_COORDINATE_GOLD',
            'delta_xy_px': d.tolist(), 'nominal_distance_px': float(np.hypot(*d)),
            'delta_xy_over_L': (d / L).tolist(), 'nominal_distance_over_L': float(np.hypot(*d) / L),
            'reference_crop_index_xy': t.tolist(),
            'inside_outer_crop_extent': bool(np.all((t >= -.5) & (t < 95.5))),
            'inside_sample_center_rectangle': bool(np.all((t >= 0.) & (t <= 95.))),
            'reference_in_original_sampler_valid_domain': bool(-.5 <= ref[0] < w-.5 and -.5 <= ref[1] < h-.5)}

def nominal_instance(inst, hw, op):
    common = {k: inst[k] for k in ('frame_index', 'scene', 'frame_key', 'gt_instance_index', 'obj_id')}
    original = {k: inst[k] for k in ('provided_cam_K', 'provided_cam_R_m2c', 'provided_cam_t_m2c_mm')}
    result = dict(common, provided_values=original, raw_hw=list(hw), status='UNKNOWN_INVALID_PROVIDED_GEOMETRY',
                  dynamic_to_fixed_permutation=None, normal_face_scores=None,
                  exact_tie_original_writer_policy_uncertified=None, references=[])
    fixed = op['FIXED_ZUP_SIGNS'].astype(np.float64) * np.array([400., 600., 72.])
    try:
        K = np.asarray(original['provided_cam_K'], np.float64).reshape(3, 3)
        R = np.asarray(original['provided_cam_R_m2c'], np.float64).reshape(3, 3)
        tr = np.asarray(original['provided_cam_t_m2c_mm'], np.float64).reshape(3)
        require(np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(tr).all(), 'Finite K/R/t')
        require(np.array_equal(K[2], [0., 0., 1.]), 'Original pinhole K')
        cam = fixed @ R.T + tr
        image = cam @ K.T
        require(np.isfinite(image).all() and np.all(image[:, 2] != 0), 'Nonzero finite projection denominator')
        xy = image[:, :2] / image[:, 2:]
        camera = np.linalg.solve(R, -tr)
        perm, scores = op['normal_facing_permutation'](fixed, xy, camera)
        require(sorted(perm.tolist()) == list(range(8)), 'Dynamic-to-fixed bijection')
        margin = float(np.sort(scores)[-1] - np.sort(scores)[-2])
        separation = float(xy[perm[1], 0] - xy[perm[0], 0])
        uncertain = margin <= 1e-10 or separation <= 0
        result.update(status='NOMINAL_COMPUTED_ROLE_UNCERTAIN' if uncertain else 'NOMINAL_COMPUTED',
                      camera_in_model_exact_linear_inverse=camera.tolist(), dynamic_to_fixed_permutation=perm.tolist(),
                      normal_face_scores=scores.tolist(), front_face_index=int(np.argmax(scores)),
                      front_runner_margin=margin, front_top_imageX_separation_px=separation,
                      exact_tie_original_writer_policy_uncertified=bool(margin <= 1e-10),
                      R_orthogonality_max_abs=float(np.max(np.abs(R.T @ R-np.eye(3)))),
                      R_determinant=float(np.linalg.det(R)), geometry_error=None)
        for k, c in enumerate(perm):
            z = float(cam[c, 2]); x, y = map(float, xy[c]); h, w = hw
            result['references'].append(dict(common, native_slot=k, native_name=NAMES[k],
                canonical_fixed_index=int(c), nominal_model_XYZ_mm=fixed[c].tolist(),
                nominal_camera_XYZ_mm=cam[c].tolist(), nominal_reference_xy=[x, y],
                positive_camera_Z=bool(z > 0), continuous_reference_inframe=bool(z > 0 and 0 <= x < w and 0 <= y < h),
                nominal_role_status=result['status'], semantic_state='UNKNOWN', native_visibility_target=-1,
                point_quality_target=-1, training_admission=False, physical_coordinate_gold=False))
    except (ValueError, AssertionError, np.linalg.LinAlgError) as e:
        result['geometry_error'] = type(e).__name__ + ': ' + str(e)
        result['references'] = [dict(common, native_slot=k, native_name=NAMES[k], canonical_fixed_index=None,
            nominal_model_XYZ_mm=None, nominal_camera_XYZ_mm=None, nominal_reference_xy=None,
            positive_camera_Z=None, continuous_reference_inframe=None, nominal_role_status=result['status'],
            semantic_state='UNKNOWN', native_visibility_target=-1, point_quality_target=-1,
            training_admission=False, physical_coordinate_gold=False) for k in range(8)]
    return result

def query_join(row, assoc, nominal, hw):
    status = 'UNKNOWN_NO_UNIQUE_OBJ1_WEAK_ASSOCIATION'
    ref = None
    if assoc['status'] == 'UNIQUE_POSITIVE_MAX_BBOX_IOU_WEAK_IDENTITY' and assoc['selected_obj_id'] == 1:
        require(nominal is not None and nominal['gt_instance_index'] == assoc['selected_gt_instance_index'], 'Exact selected GT-list lookup, no fallback')
        ref = nominal['references'][row['native_slot']]
        status = ('WEAK_INSTANCE_NOMINAL_REFERENCE' if nominal['status'] == 'NOMINAL_COMPUTED'
                  else 'UNKNOWN_NOMINAL_ROLE_OR_GEOMETRY')
        if ref['positive_camera_Z'] is not True:
            status = 'UNKNOWN_NONPOSITIVE_OR_INVALID_REFERENCE_Z'
    diagnostic = footprint(None if ref is None else ref['nominal_reference_xy'], row['original_query_xy'], row['original_L'], hw)
    return dict(row, nominal_correspondence={
        'status': status, 'weak_association_status': assoc['status'],
        'selected_gt_instance_index': assoc['selected_gt_instance_index'], 'selected_obj_id': assoc['selected_obj_id'],
        'selected_instance_identity_certified': False, 'nominal_reference': ref,
        'nominal_discrepancy_and_crop_diagnostic': diagnostic,
        'original_input_U_preserved': not row['original_query_available'],
        'semantic_state': 'UNKNOWN', 'native_visibility_target': -1, 'point_quality_target': -1,
        'training_admission': False, 'physical_coordinate_gold': False})

def invented(op):
    checks = []
    def test(name, v): checks.append({'name': name, 'PASS': bool(v)})
    def inst(t=(0., 2000., 2000.), R=None):
        return {'frame_index':7, 'scene':'000000', 'frame_key':'9', 'gt_instance_index':3, 'obj_id':1,
                'provided_cam_K':[800.,0.,640.,0.,800.,360.,0.,0.,1.],
                'provided_cam_R_m2c':np.eye(3).ravel().tolist() if R is None else R.ravel().tolist(),
                'provided_cam_t_m2c_mm':list(t)}
    a = nominal_instance(inst(), [720,1280], op)
    test('all8_nominal_preserved',len(a['references']) == 8 and sorted(a['dynamic_to_fixed_permutation']) == list(range(8)))
    test('independent_pinhole_oracle',all(np.allclose(r['nominal_reference_xy'], [800*r['nominal_camera_XYZ_mm'][0]/r['nominal_camera_XYZ_mm'][2]+640,800*r['nominal_camera_XYZ_mm'][1]/r['nominal_camera_XYZ_mm'][2]+360],rtol=0,atol=1e-12) for r in a['references']))
    test('known_negativeY_front_order',a['dynamic_to_fixed_permutation'] == [0,1,2,3,4,5,6,7])
    b = nominal_instance(inst((2000.,0.,2000.)),[720,1280],op)
    test('known_negativeX_front_rear_not_independent_Xsort',b['dynamic_to_fixed_permutation'] == [0,4,7,3,1,5,6,2])
    tie = nominal_instance(inst((2000.,3000.,2000.)),[720,1280],op)
    test('normal_cosine_not_scaled_box_area',tie['exact_tie_original_writer_policy_uncertified'] is False)
    exact = nominal_instance(inst((2000.,2000.,2000.)),[720,1280],op)
    test('inherited_exact_side_tie_uncertainty',exact['exact_tie_original_writer_policy_uncertified'] is True and exact['status']=='NOMINAL_COMPUTED_ROLE_UNCERTAIN')
    R=np.eye(3);R[0,1]=1e-7
    pert=nominal_instance(inst(R=R),[720,1280],op)
    test('supplied_nonorthogonal_R_not_repaired',pert['provided_values']['provided_cam_R_m2c']==R.ravel().tolist() and pert['R_orthogonality_max_abs']>0 and np.array_equal(np.asarray(pert['camera_in_model_exact_linear_inverse']),np.linalg.solve(R,-np.array([0.,2000.,2000.]))))
    bad=nominal_instance(inst(R=np.zeros((3,3))),[720,1280],op)
    test('invalid_geometry_all8_UNKNOWN_not_dropped',len(bad['references'])==8 and all(r['nominal_reference_xy'] is None for r in bad['references']))
    behind=nominal_instance(inst((0.,2000.,-2000.)),[720,1280],op)
    test('nonpositive_Z_is_not_visibility',all(r['positive_camera_Z'] is False and r['semantic_state']=='UNKNOWN'for r in behind['references']))
    d=footprint([12.,20.],[12.,20.],96.,[720,1280])
    test('q_maps_to_central4_between_centers',d['reference_crop_index_xy']==[47.5,47.5] and d['nominal_distance_px']==0.)
    test('outer_extent_lower_in_upper_out',footprint([-48.,0.],[0.,0.],96.,[720,1280])['inside_outer_crop_extent'] is True and footprint([48.,0.],[0.,0.],96.,[720,1280])['inside_outer_crop_extent'] is False)
    test('sample_center_rectangle_inclusive',footprint([-47.5,47.5],[0.,0.],96.,[720,1280])['inside_sample_center_rectangle'] is True and footprint([-48.,0.],[0.,0.],96.,[720,1280])['inside_sample_center_rectangle'] is False)
    test('sampler_halfpixel_domain_not_continuous_input_validity',footprint([-.25,2.],[0.,2.],96.,[720,1280])['reference_in_original_sampler_valid_domain'] is True and footprint([1279.75,2.],[0.,2.],96.,[720,1280])['reference_in_original_sampler_valid_domain'] is False)
    with safe_open(CROP) as f: crop_tree=ast.parse(f.read().decode())
    crop_nodes=[n for n in crop_tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop']
    require(len(crop_nodes)==1 and sha(CROP)==CROP_SHA,'Exact original pure crop fixture')
    crop_ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=crop_nodes,type_ignores=[])),str(CROP),'exec'),crop_ns)
    yy,xx=np.mgrid[:256,:256];rgb=np.stack((xx,yy,np.zeros_like(xx)),axis=-1).astype(np.uint8)
    q=np.array([128.25,128.75]);side=97.3;raw,valid=crop_ns['crop'](rgb,q,side,96)
    oracle=True
    for i,j in [(0,0),(47,47),(48,48),(95,95)]:
        xy=q+(np.array([j,i],np.float64)+.5-48.)*side/96.
        f=footprint(xy,q,side,[256,256])
        oracle=oracle and np.allclose(f['reference_crop_index_xy'],[j,i],rtol=0,atol=2e-14) and np.array_equal(raw[:2,i,j],np.floor(xy+.5).astype(np.uint8)) and bool(valid[0,i,j])
    test('original_V90_crop_invented_coordinate_ramp_and_inverse_grid',oracle)
    test('missing_q_L_null_safe',all(not footprint([1.,1.],q,L,[720,1280])['valid_nominal_discrepancy'] for q,L in [(None,1.),([None,None],2.),([-1.,-1.],2.),([0.,0.],None),([0.,0.],0.)]))
    row={'native_slot':0,'original_query_xy':[10.,20.],'original_L':96.,'original_query_available':False,'original_global_index':56,'original_input_flags':{'continuous_valid':False,'nearest_valid':True}}
    assoc={'status':'UNIQUE_POSITIVE_MAX_BBOX_IOU_WEAK_IDENTITY','selected_obj_id':1,'selected_gt_instance_index':3}
    joined=query_join(row,assoc,a,[720,1280])
    test('U_qL_flags_original_keys_unchanged',all(joined[k]==v for k,v in row.items()) and joined['nominal_correspondence']['original_input_U_preserved'])
    other=query_join(row,dict(assoc,selected_obj_id=7),None,[720,1280])
    test('other_obj_no_obj1_fallback',other['nominal_correspondence']['nominal_reference'] is None and other['nominal_correspondence']['status'].startswith('UNKNOWN'))
    amb=query_join(row,dict(assoc,status='UNKNOWN_EXACT_MAX_IOU_TIE',selected_obj_id=None,selected_gt_instance_index=None),None,[720,1280])
    test('association_tie_remains_UNKNOWN',amb['nominal_correspondence']['nominal_reference'] is None)
    shifted=query_join(dict(row,original_query_xy=[1000.,500.]),assoc,a,[720,1280])
    test('query_shift_never_changes_permutation',shifted['nominal_correspondence']['nominal_reference']==joined['nominal_correspondence']['nominal_reference'])
    test('no_new_gold_or_labels',all(r['training_admission'] is False and r['native_visibility_target']==r['point_quality_target']==-1 and r['physical_coordinate_gold'] is False for r in a['references']))
    try: permitted(P/'forbidden_v112'/'never_access'); ok=False
    except AssertionError: ok=True
    test('name_rejected_before_file_access',ok)
    return {'completed':True,'PASS':all(r['PASS']for r in checks),'invented_only':True,'checks':checks,'source_GT_or_results_body_reads':0,'model_calls':0}

def prepare():
    c = read(CP)
    require(c['expected']['inputU']==194 and c['expected']['obj1_GT_instance_observations']==258,'Frozen source denominators')
    require(sha(CROP)==CROP_SHA,'Exact original crop math source')
    op, proof=load_order(); tests=invented(op);tests['code']=bind(__file__);tests['order_AST']=proof
    write(CHECKS,tests);require(tests['PASS'],'Invented fixtures')
    protocol={'completed':True,'prepared_only':True,'code':bind(__file__),'spec':bind(SPEC),'scope':bind(SCOPE),'checks':bind(CHECKS),
              'order_AST':proof,'original_crop_math':bind(CROP),'C_preparation_protocol':bind(CP),'C_code':c['code'],
              'original_frame_plan':c['frame_plan'],'original_row_plan':c['row_plan'],
              'expected_C_complete_path':str(CC),'expected_C_output_names':['ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl','ALL1480_QUERY_INSTANCE_DIAGNOSTICS.jsonl','ALL_GT_OBSERVATIONS_IN185_SOURCE_FRAMES.jsonl'],
              'expected':{'frames':185,'original_rows':1480,'original_inputU':194,'obj1_instances':258,'nominal_reference_rows':2064},
              'output':str(OUT),'actual_allowed_reads':'Only bound sealed C COMPLETE and its three metadata JSONL outputs plus bound original frame/row plans; no upstream GT assets, RGB, depth, masks, mesh, NPZ headers/body, model or evaluation bodies.',
              'authority':'Every nominal reference is UNKNOWN semantic state, no physical coordinate gold. C instance association stays weak; no fallback/GT-best permutation/q correction.',
              'automatic_retry':False,'new_labels_or_training_admission':False}
    write(PROTOCOL,protocol)
    template={'approved_actual_source_nominal_native_projection':False,'source_TRAIN_only':True,'code':protocol['code'],'protocol':bind(PROTOCOL),'spec':protocol['spec'],'checks':protocol['checks'],'scope':protocol['scope'],'C_complete':{'path':str(CC),'sha256':None},'output':str(OUT),'source_asset_or_model_or_evaluation_reads':False,'new_labels_or_training_admission':False,'automatic_retry':False}
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    ready={'completed':True,'prepared_only':True,'code':protocol['code'],'protocol':bind(PROTOCOL),'spec':protocol['spec'],'checks':protocol['checks'],'template':bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),'expected':protocol['expected'],'actual_source_results_GT_geometry_or_pixels_read':0,'model_calls':0,'runtime':{'numpy':np.__version__},'command':'/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).absolute())+' --entry <root-entry>'}
    write(H/'PREPARATION_COMPLETE.json',ready);print(json.dumps(ready),flush=True)

def run(entry):
    e=read(entry)
    require(e['approved_actual_source_nominal_native_projection'] is True and e['source_TRAIN_only'] is True,'Separate root entry required')
    for k in ('source_asset_or_model_or_evaluation_reads','new_labels_or_training_admission','automatic_retry'):
        require(e[k] is False,'Forbidden extra action '+k)
    p=read(checked(e['protocol']))
    require(e['protocol']==bind(PROTOCOL) and e['code']==p['code']==bind(__file__),'Frozen code/protocol')
    for k in ('spec','checks','scope'):
        require(e[k]==p[k],'Frozen '+k);checked(p[k])
    require(e['C_complete']['path']==p['expected_C_complete_path']==str(CC),'Exact source C result path')
    complete=read(checked(e['C_complete']))
    require(complete['completed'] is True and complete['PASS'] is True,'Sealed C success')
    require(complete['code']==p['C_code'] and complete['protocol']==p['C_preparation_protocol'],'Frozen C producer authority')
    require(complete['counts']['frames']==185 and complete['counts']['rows']==1480 and complete['counts']['inputU']==194 and complete['counts']['obj1_GT_instance_observations']==258,'Original C counts')
    outputs=complete['outputs'];require(set(outputs)==set(p['expected_C_output_names']),'Exact C metadata outputs')
    for n,b in outputs.items():require(b['path']==str(CC.parent/n),'No replacement C output path')
    bindings=[e['C_complete'],p['original_frame_plan'],p['original_row_plan']]+list(outputs.values())
    before=[dict(b)for b in bindings]
    require(e['output']==p['output']==str(OUT),'Fresh fixed output')
    OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',{'completed':False,'entry':bind(entry),'automatic_retry':False})
    try:
        frames=lines(checked(p['original_frame_plan']));original=lines(checked(p['original_row_plan']))
        cf=lines(checked(outputs['ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl']));cr=lines(checked(outputs['ALL1480_QUERY_INSTANCE_DIAGNOSTICS.jsonl']));gt=lines(checked(outputs['ALL_GT_OBSERVATIONS_IN185_SOURCE_FRAMES.jsonl']))
        require(len(frames)==len(cf)==185 and len(original)==len(cr)==1480,'All original rows')
        byfi={r['frame_index']:r for r in frames};associations={r['frame_index']:r for r in cf}
        require(len(byfi)==len(associations)==185 and set(byfi)==set(associations),'Exact original frame set')
        for fi,f in byfi.items():
            require(f['split']=='SOURCE_DEVELOPMENT' and f['scene']in SCENES,'Source TRAIN only')
            require(all(associations[fi][k]==v for k,v in f.items()),'Original frame identity/qL/flags unchanged')
        require(len({(g['frame_index'],g['gt_instance_index'])for g in gt})==len(gt),'GT observation identity unique')
        for g in gt:
            f=byfi[g['frame_index']];require(g['scene']==f['scene'] and g['frame_key']==f['frame_key'],'GT exact frame provenance')
        op,proof=load_order();require(proof==p['order_AST'],'Pinned AST unchanged')
        instances=[nominal_instance(g,byfi[g['frame_index']]['raw_hw'],op)for g in gt if g['obj_id']==1]
        require(len(instances)==258,'All258 obj1, no candidate filtering')
        refs=[r for g in instances for r in g['references']];require(len(refs)==2064,'All2064 native references')
        index={(g['frame_index'],g['gt_instance_index']):g for g in instances};result=[]
        for r,crow in zip(original,cr):
            require(all(crow[k]==v for k,v in r.items()),'Original1480 metadata unchanged by C')
            require(r['split']=='SOURCE_DEVELOPMENT' and 0<=r['native_slot']<8 and r['original_global_index']==r['original_frame_index']*8+r['native_slot'],'Original row/native identity')
            f=byfi[r['original_frame_index']];a=associations[r['original_frame_index']]['diagnostic_association']
            require(crow['selected_instance_diagnostic']==a,'Same C association for every unchanged slot')
            nominal=index.get((r['original_frame_index'],a['selected_gt_instance_index'])) if a['selected_obj_id']==1 else None
            new=query_join(r,a,nominal,f['raw_hw']);require(all(new[k]==v for k,v in r.items()),'Original row immutable')
            result.append(new)
        require(sum(not r['original_query_available']for r in result)==194,'Original U194 unchanged')
        require(Counter(r['native_slot']for r in result)==Counter({k:185 for k in range(8)}),'All native8 each185')
        for b in before:checked(b)
        files={'instances':'ALL258_OBJ1_NOMINAL_INSTANCES.jsonl','references':'ALL2064_OBJ1_NOMINAL_NATIVE_REFERENCES.jsonl','queries':'ALL1480_ORIGINAL_QUERY_NOMINAL_CORRESPONDENCE.jsonl'}
        for key,rows in [('instances',instances),('references',refs),('queries',result)]:writelines(OUT/files[key],rows)
        done={'completed':True,'PASS':True,'entry':bind(entry),'code':bind(__file__),'protocol':bind(PROTOCOL),'spec':bind(SPEC),'C_complete':e['C_complete'],'all_source_metadata_bindings_unchanged':before,'outputs':{k:bind(OUT/n)for k,n in files.items()},
              'counts':{'frames':185,'original_rows':1480,'original_inputU':194,'obj1_instances':258,'nominal_reference_rows':2064,'nominal_instance_status':dict(Counter(r['status']for r in instances)),'query_lookup_status':dict(Counter(r['nominal_correspondence']['status']for r in result))},
              'new_labels_or_training_admission':False,'physical_coordinate_gold':False,'source_RGB_depth_mask_mesh_NPZ_or_model_reads':0,'qL_or_pose_changes':0,'automatic_retry':False}
        write(OUT/'COMPLETE.json',done);print(json.dumps({'completed':True,'PASS':True,'counts':done['counts'],'complete':bind(OUT/'COMPLETE.json')}),flush=True)
    except Exception:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',{'completed':False,'automatic_retry':False,'entry':bind(entry),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare:prepare()
    elif args.entry:run(args.entry)
    else:ap.error('--prepare or --entry required')
