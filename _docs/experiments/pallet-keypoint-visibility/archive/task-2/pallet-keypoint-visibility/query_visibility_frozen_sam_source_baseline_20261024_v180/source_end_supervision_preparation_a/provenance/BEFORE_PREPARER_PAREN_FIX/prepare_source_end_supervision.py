"""Preparation only: source text/metadata and invented arrays, never source pixels."""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import source_end_support as op

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V173=P/'query_visibility_full_source_query_causal_census_20261024_v173'
SCOPE=V/'ROOT_SCOPE_BEFORE_SAM_CODE_WEIGHTS_OR_SOURCE_RGB.json'
CODE=H/'source_end_support.py'
TEACHER=P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py'
CONFIG=P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/BASE_CONFIGS192_BEFORE_RGB.json'
INVENTORY=V173/'census_preparation/ALL6144_ORIGINAL_QUERY_METADATA_BEFORE_MASKS.jsonl'
SCENES=V173/'census_preparation/ALL768_SCENE_PATHS_BEFORE_MASKS.jsonl'
GEOMETRY=H/'TRAIN1024_FIXED_END_PATCH_GEOMETRY_METADATA.jsonl'
PLAN=H/'TRAIN4096_END_SUPERVISION_INPUT_ONLY_PLAN.jsonl'
PROTOCOL=H/'PROTOCOL_BEFORE_NEW_SOURCE_END_PIXEL_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_lines(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def generated():
    checks={}
    def record(name,value):
        checks[name]=bool(value)
        assert value,name
    camera=dict(K=[[4.,0,0],[0,4.,0],[0,0,1]],R=np.eye(3).tolist(),t=[0.,0,0.],near=.001)
    quad=np.array([[[1.,1,2],[3,1,2],[3,3,2],[1,3,2]]])
    patch=op.expected_patch_depth(quad,camera,[8,8])
    def packet(pid=0,depth=2.,tri=7,amb=False):
        return dict(part_id=np.full((8,8),pid,np.int32),camera_depth=np.full((8,8),depth,np.float64),global_triangle_id=np.full((8,8),tri,np.int32),ambiguity=np.full((8,8),amb,bool))
    b=packet();points=op.original_crop_positions([4.,4.],6.)
    def apply(bb,ss,pp=patch,qq=points,available=True):
        c=op.classify_end_pixels(bb,ss,pp,0)
        return op.summarize(op.sample_classes(c,qq),pp,available),c
    r,c=apply(b,packet())
    record('Perspective_quad_plane_depth_region_and_full_visible',[r['visible_support_target'],r['foreign_front_support_target']]==[1,0] and np.isfinite(patch['expected_depth']).sum()==25 and np.all(patch['expected_depth'][patch['region_mask']]==2.))
    record('Whole_primary_board_outside_small_end_never_counted',np.all(c[~patch['region_mask']]==0) and r['sampled_end_region_mass']<1)
    r,_=apply(b,packet(999,1.))
    record('Target_alone_support_then_foreign_front',[r['visible_support_target'],r['foreign_front_support_target']]==[0,1])
    mix=packet();mix['part_id'][:,:4]=999;mix['camera_depth'][:,:4]=1.
    r,_=apply(b,mix)
    record('Partial_end_both_visible_and_foreign_nonexclusive',[r['visible_support_target'],r['foreign_front_support_target']]==[1,1] and r['visible_support_mass']>0 and r['foreign_front_support_mass']>0)
    r,_=apply(packet(-1,np.nan),packet(999,1.))
    record('Foreign_over_BEFORE_background_not_end_external',[r['visible_support_target'],r['foreign_front_support_target']]==[-1,-1])
    r,c=apply(packet(1,1.),packet(1,1.))
    record('Before_target_self_hiding_expected_patch_UNKNOWN',np.all(c[patch['region_mask']]==3) and r['visible_support_target']==-1 and r['foreign_front_support_target']==-1)
    r,c=apply(packet(amb=True),packet())
    record('BEFORE_ambiguity_UNKNOWN',np.all(c[patch['region_mask']]==4) and r['visible_support_target']==-1)
    r,c=apply(b,packet(amb=True))
    record('SCENE_ambiguity_UNKNOWN',np.all(c[patch['region_mask']]==4) and r['foreign_front_support_target']==-1)
    r,_=apply(b,packet(tri=8))
    record('Changed_target_triangle_not_certified_visible',r['visible_support_target']==-1)
    r,_=apply(b,packet(depth=2.1))
    record('Changed_target_depth_UNKNOWN',r['visible_support_target']==-1)
    r,_=apply(b,packet(999,2.5))
    record('Foreign_not_strictly_front_UNKNOWN',r['foreign_front_support_target']==-1)
    r,_=apply(packet(1,2.),packet(1,2.))
    record('Same_depth_wrong_part_is_registration_UNKNOWN',r['visible_support_target']==-1)
    off=quad.copy();off[:,:,0]+=2.
    offpatch=op.expected_patch_depth(off,camera,[8,8]);r,_=apply(b,packet(),offpatch)
    record('Projected_patch_out_of_frame_explicit_UNKNOWN',offpatch['projected_region_out_of_frame'] and r['visible_support_target']==-1 and r['foreign_front_support_target']==-1)
    invalid=quad.copy();invalid[:,:,2]=-.1
    invalidpatch=op.expected_patch_depth(invalid,camera,[8,8]);r,_=apply(b,packet(),invalidpatch)
    record('Patch_near_plane_or_behind_camera_UNKNOWN',not invalidpatch['geometry_valid'] and r['visible_support_target']==-1)
    r,_=apply(b,packet(),available=False)
    record('Original_input_U_not_replaced_or_converted_to_BG',r['visible_support_target']==-1 and r['foreign_front_support_target']==-1 and 'ORIGINAL_INPUT_U_UNCHANGED' in r['reasons'])
    r,_=apply(b,packet(),qq=op.original_crop_positions([0.,0.],1.))
    record('No_end_in_fixed_crop_visibleNO_externalUNKNOWN',[r['visible_support_target'],r['foreign_front_support_target']]==[0,-1] and r['sampled_end_region_mass']==0)
    grid=op.original_crop_positions([64.,64.],96.).reshape(96,96,2)
    record('Original96_grid_central4_is_exact47_49',grid[47:49,47:49].reshape(-1,2).tolist()==[[63.5,63.5],[64.5,63.5],[63.5,64.5],[64.5,64.5]])
    toy=np.array([[0,1],[2,3]],np.int8);ww=op.sample_classes(toy,np.array([[.5,.5],[-.75,0.]]))
    record('Independent_bilinear_four_classes_and_invalid_preserved',ww[0].tolist()==[.25,.25,.25,.25,0,0,0] and ww[1].tolist()==[0,0,0,0,0,0,1.])
    shift=np.array([4.,-3.,2.]);cam2=dict(camera,t=(-shift).tolist());p2=op.expected_patch_depth(quad+shift,cam2,[8,8])
    record('World_camera_joint_translation_same_projection',np.array_equal(p2['expected_depth'],patch['expected_depth']))
    before_bytes={k:v.tobytes() for k,v in b.items()};apply(b,packet())
    record('Before_arrays_not_modified',all(b[k].tobytes()==v for k,v in before_bytes.items()))
    mass=np.zeros((1,7));mass[0,[1,3]]=[.5,.5];r=op.summarize(mass,patch)
    record('Positive_visible_witness_with_self_mass_keeps_externalUNKNOWN',r['visible_support_target']==1 and r['foreign_front_support_target']==-1 and r['unknown_end_mass']==.5)
    mass=np.zeros((1,7));mass[0,[2,4]]=[.5,.5];r=op.summarize(mass,patch)
    record('Positive_foreign_witness_with_ambiguity_visibleUNKNOWN',r['visible_support_target']==-1 and r['foreign_front_support_target']==1)
    # This path tests the actual row function, including metadata camera checks and missing q.
    cam_fields={'camera_'+k:np.asarray(camera[k],np.float64) for k in ('K','R','t')}
    nominal=np.zeros((8,2),np.float64);perm=np.arange(8,dtype=np.int32)
    for pp in (b,):pp.update(cam_fields,native8_CF_xy=nominal,permCF_to_canonical=perm)
    scene=packet();scene.update(cam_fields,native8_CF_xy=nominal,permCF_to_canonical=perm)
    g=dict(geometry_key='INVENTED',base_index=0,native_slot=0,canonical_index=0,primary_part_id=0,quads_world=np.repeat(quad,3,axis=0).tolist(),camera=camera,raw_hw=[8,8],native8_CF_xy=nominal.tolist(),permCF_to_canonical=perm.tolist())
    row=dict(original_global_index=0,original_occurrence_id='INVENTED',scene_index=0,base_index=0,native_slot=0,declared_canonical_index=0,primary_part_id=0,split='TRAIN',variant='BEFORE',query_xy=None,L=None,original_input_flags=dict(query_available=False),original_semantic_state='D')
    rr=op.process_one(row,g,b,scene)
    record('Actual_process_one_missing_q_L_retained_UNKNOWN',rr['visible_support_target']==-1 and rr['full96_average_mass'] is None and rr['original_input_flags']==row['original_input_flags'])
    row.update(query_xy=[4.,4.],L=6.,original_input_flags=dict(query_available=True));rr=op.process_one(row,g,b,scene)
    record('Actual_process_one_fixed_crop_and_no_native_quality_target',rr['visible_support_target']==1 and rr['native_visibility_target']==rr['point_quality_target']==-1 and not rr['training_admission'] and rr['query_xy']==row['query_xy'] and rr['L']==row['L'])
    bad=dict(scene,camera_K=np.eye(3))
    try:op.process_one(row,g,b,bad)
    except AssertionError:rejected=True
    else:rejected=False
    record('Actual_process_one_camera_metadata_mismatch_rejected',rejected)
    return dict(completed=True,PASS=True,invented_only=True,checks=checks,number_checks=len(checks),runtime=dict(executable=sys.executable,numpy=np.__version__),source_pixel_RGB_mask_depth_mesh_NPZ_reads=0)


def reconstruct_templates():
    # Execute only the two exact deterministic constructive functions and their
    # literal array constants. The original module/teacher/renderer is not imported.
    tree=ast.parse(TEACHER.read_text());nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in ('add_box','make_mesh'):nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('DIM','SIGNS') for x in n.targets):nodes.append(n)
    assert len(nodes)==4
    ns=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(TEACHER)+'::CONSTRUCTIVE_GEOMETRY_ONLY','exec'),ns)
    anchors=ns['SIGNS']*ns['DIM']/2
    return [ns['make_mesh'](name,anchors)[0] for name in ('WOOD_BOARD_BLOCK','POLYMER_RIB_LEG')]


def prepare():
    assert not PROTOCOL.exists(),'Preparation is sealed; preserve before any revision.'
    scope=read(SCOPE);assert 'TRAIN only' in scope['development_data']
    rows=[r for r in lines(INVENTORY) if r['split']=='TRAIN']
    scenes={r['scene_index']:r for r in lines(SCENES) if r['split']=='TRAIN'}
    configs={r['base_index']:r for r in read(CONFIG)['rows'] if r['split']=='TRAIN'}
    assert len(rows)==4096 and len(scenes)==512 and len(configs)==128
    templates=reconstruct_templates();geometries=[];plan=[];max_projection_delta=0.
    for base,cfg in configs.items():
        template=templates[cfg['template_index']];scale=np.asarray(cfg['positive_scale_xyz'],np.float64)
        anchors=template['native_anchor3D']*scale
        assert np.array_equal(anchors,np.asarray(cfg['native8_canonical_xyz'],np.float64))
        for slot,canonical in enumerate(cfg['permCF_to_canonical']):
            patch=template['native_patch_quads'][canonical]*scale
            pid=int(template['native_primary_part_id'][canonical])
            assert np.array_equal(patch[:,0],np.broadcast_to(anchors[canonical],(3,3)))
            cc=cfg['camera'];p=anchors[canonical]@np.asarray(cc['R']).T+np.asarray(cc['t']);K=np.asarray(cc['K']);xy=p[:2]/p[2]*K.diagonal()[:2]+K[:2,2]
            delta=float(np.max(np.abs(xy-np.asarray(cfg['native8_CF_xy'][slot]))));max_projection_delta=max(max_projection_delta,delta)
            assert delta<=1e-12
            geometries.append(dict(geometry_key=f'BASE{base:03d}_NATIVE{slot}',base_index=base,native_slot=slot,canonical_index=canonical,primary_part_id=pid,quads_world=patch.tolist(),anchor_world=anchors[canonical].tolist(),camera=cfg['camera'],raw_hw=cfg['raw_hw'],native8_CF_xy=cfg['native8_CF_xy'],permCF_to_canonical=cfg['permCF_to_canonical'],geometry_family=cfg['factors']['geometry_family'],geometry_reconstructed_from_exact_source_code=True,GT_supervision_only=True))
    gm={(g['base_index'],g['native_slot']):g for g in geometries}
    for r in rows:
        g=gm[r['base_index'],r['native_slot']];scene=scenes[r['scene_index']]
        assert g['canonical_index']==r['declared_canonical_index'] and g['primary_part_id']==r['primary_part_id'] and r['incident_part_ids']==[g['primary_part_id']]
        assert scene['camera']==g['camera'] and scene['permCF_to_canonical']==g['permCF_to_canonical'] and scene['native8_CF_xy']==g['native8_CF_xy']
        plan.append(dict(**r,geometry_key=g['geometry_key'],scene_buffer=scene['scene_buffer'],BEFORE_buffer=scene['BEFORE_buffer'],raw_hw=scene['raw_hw'],new_supervision_not_generated=True))
    assert len(geometries)==1024 and len(plan)==4096
    write_lines(GEOMETRY,geometries);write_lines(PLAN,plan);write(CHECKS,generated())
    protocol=dict(schema='V180_FIXED_END_SUPPORT_SOURCE_TRAIN_PREPARATION_A_v1',completed=True,prepared_only=True,code=bind(CODE),preparer=bind(__file__),scope=bind(SCOPE),source_teacher_code=bind(TEACHER),source_configs=bind(CONFIG),original_inventory=bind(INVENTORY),original_scene_inventory=bind(SCENES),geometry_metadata=bind(GEOMETRY),all4096_input_plan=bind(PLAN),checks=bind(CHECKS),
        counts=dict(TRAIN_rows=4096,bases=128,scenes=512,fixed_native_geometries=1024,all_native_slots=list(range(8)),inputU=sum(not r['original_input_flags']['query_available'] for r in rows),material_aspect={a+'::'+b:sum(r['factors']['material']==a and r['factors']['aspect']==b for r in rows) for a,b in sorted({(r['factors']['material'],r['factors']['aspect']) for r in rows})},
        registration=dict(exact_constructive_source_AST_functions=['add_box','make_mesh'],original_teacher_or_renderer_module_imported=False,source_mesh_NPZ_not_opened=True,scaled_patch_anchor_exact=True,max_native_projection_numeric_difference=max_projection_delta,all4096_camera_CF_canonical_primarypart_consistent=True),
        questions='Inside the unchanged original q/L full96 crop, is any fixed intended-end patch support visible, and is any support externally covered by owner999 in front of its target-alone depth? These are two non-exclusive new support variables, not native D/E/S or generic crop-any-pallet presence.',
        fixed_geometry='Use existing native_patch_quads: 3 small incident local face patches; they already contain the registered endpoint and edge borders. Reconstruct exact declared source constructive geometry and positive scale, map CF slot to canonical once, use existing scene camera. Never select nearest vertex/part/q-to-GT mapping or change patch dimensions from outcomes.',
        dense_vs_sparse37='Original teacher traces endpoint0+9edge+27face samples with front-axis/all-active rules. This operator intersects original raw integer-centre camera rays with all3 fixed face quads and chooses the nearest projected patch surface at each pixel; it then samples their class support on the exact original96 grid. It does not require two front faces, all-active D, or a fully visible end. It is finite raster support for the user crop, not dense continuous physical/corner certification.',
        source_pixel_class_order=list(op.CLASSES),
        pixel_rules='Outside projected end is OUTSIDE_END even when another part or another end of the same long board is visible. Either-buffer ambiguity gives UNKNOWN. BEFORE must have primarypart at expected patch depth within64 float64ULPs (source arithmetic only). A closer target BEFORE surface gives UNKNOWN_SELF; foreign/BG/wrong-part/depth at BEFORE gives UNKNOWN registration. Supported BEFORE plus SCENE unchanged part/triangle/exact stored depth gives END_VISIBLE; scene999 finitepositive strictly nearer gives END_FOREIGN_FRONT; remaining changes UNKNOWN.',
        crop_sampling='Original q+((index+.5)/96-.5)*L, all96x96 positions, bilinear onehot source classes before averaging. Preserve invalid source mass and existing halfpixel crop-domain mask. No GT-centred crop, recentering or q/L/input-U change.',
        targets='visible_support_target and foreign_front_support_target each YES1/NO0/UNKNOWN-1, non-exclusive (partially covered region can be bothYES). Positive observed support witnesses existence even when a different patch subset is UNKNOWN; no support plus UNKNOWN region =>UNKNOWN. If no sampled end region in crop: visibleNO, foreignUNKNOWN. Projected patch OOF/nearplane or original inputU =>bothUNKNOWN. Self/ambiguous/inconsistent mass is explicit and never converted to foreign/BG or renormalized away. All mass values retained; no outcome-tuned area or confidence threshold.',
        meaning_limits='These new source support targets do not replace original native D/E/S, certify physical corner/pose, identify full-part visibility or establish real transfer. Source GT geometry/part/depth is supervision only, never SAM input/prompt. New native_visibility_target and point_quality_target stay-1; training_admission=false.',
        runtime_contract='Preparation command only. Pure process_one is concrete and checked on invented arrays. A later root-reviewed source runner must verify inherited TRAIN buffer SHA, ID, shape/dtypes, camera/native/CF metadata before calling it and preserve all4096 rows, including inputU/missing; new supervision generation has not run.',
        source_RGB_part_depth_or_binary_header_hash_reads=0,new_supervision_generation_runs=0,new_fit_model_SAM_encoder_PnP_q_correction_or_label_rewrite=0,private_held_VALsoft_or_V112_reads=0,full_WOOD_SQUARE_native8_goal_preserved=True,selected_model=None,fully_solved=False)
    write(PROTOCOL,protocol)
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(CODE),preparer=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),geometry_metadata=bind(GEOMETRY),all4096_input_plan=bind(PLAN),counts=protocol['counts'],actual_source_supervision_generation=0,source_pixel_binary_reads=0,models_or_fit=0,selected_model=None,fully_solved=False))
    print(json.dumps(dict(PASS=True,code=bind(CODE),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(H/'PREPARATION_COMPLETE.json')),ensure_ascii=False))


if __name__=='__main__':prepare()
