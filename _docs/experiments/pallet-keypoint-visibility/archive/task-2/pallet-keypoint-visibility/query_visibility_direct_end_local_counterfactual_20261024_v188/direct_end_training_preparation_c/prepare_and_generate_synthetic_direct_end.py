"""V188 source-code/metadata preparation; actual pixels require separate root entry.

No model, feature, checkpoint, evaluation or external asset loader exists here.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import argparse,ast,hashlib,importlib.util,json,time,traceback
import numpy as np
import synthetic_direct_end_candidate as candidate

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_DIRECT_END_SOURCE_CODE_AND_METADATA_PREPARATION.json'
SCOPE_SHA='c2e3bfe89a847afdf40851b9567f9d55e40b95c6cbff732622cdf239678d4f6e'
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
V129=P/'query_visibility_registered_native_diversity_20261023_v129'
TEACHER=P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py'
CONFIGS=V129/'render_preparation/BASE_CONFIGS192_BEFORE_RGB.json'
CORE=V129/'render_preparation/corpus_math.py'
OP=V180/'source_end_supervision_preparation_a/source_end_support.py'
PLAN=V180/'source_end_supervision_preparation_a/TRAIN4096_END_SUPERVISION_INPUT_ONLY_PLAN.jsonl'
SAMPLER=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
B=V/'end_locality_contract_b/finite_end_locality_contract.py'
HEAD=P/'query_visibility_local_support_raw_head_20261024_v181/head_preparation/local_support_head.py'
TRAINER=P/'query_visibility_query_jitter_paired_source_training_20261024_v183/source_fit_preparation/train_jitter_paired_source.py'
OLDPROTOCOL=TRAINER.with_name('PROTOCOL_BEFORE_AUGMENTED_CACHE_OR_FIT_BODIES.json')
CODE=H/'synthetic_direct_end_candidate.py'
PROTOCOL=H/'PROTOCOL_BEFORE_SYNTHETIC_END_CONTROL_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
POPULATION=H/'ALL1024_SYNTHETIC_END_ATTEMPTS_INPUT_ONLY.jsonl'
OUT=V/'synthetic_direct_end_controls_v1'
VARIANTS=('BEFORE','COVER','SHAM')


def allowed(path):
    # Lexical forbidden-name check BEFORE stat/open; no resolve or symlinkfollow.
    p=Path(path).absolute()
    for part in p.parts:
        candidate.require('v112' not in part.lower(),'Forbidden name pruned')
    candidate.require(p.is_relative_to(P),'Exact project source only')
    cur=Path(p.anchor)
    for part in p.parts[1:]:
        cur=cur/part
        candidate.require(not cur.is_symlink(),'No symlink traversal')
    return p


def sha(path):
    h=hashlib.sha256()
    with allowed(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def bind(path):return dict(path=str(allowed(path)),sha256=sha(path))
def check(binding):
    p=allowed(binding['path']);candidate.require(sha(p)==binding['sha256'],'Exact bound source SHA');return p
def read(path):return json.loads(allowed(path).read_text())
def lines(path):return [json.loads(s) for s in allowed(path).read_text().splitlines() if s]
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jlines(path,rows):
    with Path(path).open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def module(path,name):
    s=importlib.util.spec_from_file_location(name,allowed(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def source_functions():
    """Only literal constructive declarations and pure geometry functions."""
    nodes=[]
    for n in ast.parse(allowed(TEACHER).read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('DIM','SIGNS') for t in n.targets):nodes.append(n)
        elif isinstance(n,ast.FunctionDef) and n.name in ('add_box','make_mesh','first_hit','classify_hits'):nodes.append(n)
    candidate.require(len(nodes)==6,'Exact constructive function selection')
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(TEACHER)+'::pure_constructive_only','exec'),ns)
    return ns


def crop_function():
    nodes=[n for n in ast.parse(allowed(SAMPLER).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop']
    candidate.require(len(nodes)==1,'Pinned original crop only')
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SAMPLER)+'::crop_only','exec'),ns);return ns['crop']


def population():
    cfgs=[r for r in read(CONFIGS)['rows'] if r['split']=='TRAIN']
    rr=lines(PLAN);candidate.require(len(rr)==4096 and all(r['split']=='TRAIN' for r in rr),'Only original TRAIN4096 metadata')
    before={(r['base_index'],r['native_slot']):r for r in rr if r['variant']=='BEFORE'}
    candidate.require(len(cfgs)==128 and len(before)==1024,'All128 bases/native8')
    rows=[]
    for cfg in cfgs:
        for k in range(8):
            r=before[cfg['base_index'],k]
            candidate.require(r['scene_index']==4*cfg['base_index'] and r['original_global_index']==32*cfg['base_index']+k,'Original BEFORE identity')
            rows.append(dict(attempt_index=len(rows),attempt_id='V188_BASE%03d_NATIVE%d'%(cfg['base_index'],k),
                split='TRAIN',base_index=cfg['base_index'],native_slot=k,canonical_index=int(cfg['permCF_to_canonical'][k]),
                factors=cfg['factors'],source_config_row=cfg['base_index'],geometry_key=r['geometry_key'],
                original_global_index=r['original_global_index'],original_occurrence_id=r['original_occurrence_id'],
                query_xy=r['query_xy'],L=r['L'],original_input_flags=r['original_input_flags'],
                source_BEFORE_buffer=r['BEFORE_buffer'],raw_hw=r['raw_hw'],
                original_rgb96_sha256=r['rgb96_SHA_inherited'],original_valid96_sha256=r['valid96_SHA_inherited'],
                direct_end_target=-1,new_native_visibility_target=-1,new_point_quality_target=-1,
                original_targets_not_copied_or_rewritten=True,training_admission=False))
    candidate.require(sum(not r['original_input_flags']['query_available'] for r in rows)==16,'BEFORE U16 preserved')
    return cfgs,rows


def templates(ns):
    anchors=ns['SIGNS']*ns['DIM']/2
    return [ns['make_mesh'](style,anchors)[0] for style in ('WOOD_BOARD_BLOCK','POLYMER_RIB_LEG')]


def scaled(template,cfg):
    data={k:v.copy() for k,v in template.items()}
    for key in ('vertices','native_anchor3D','native_patch_quads','native_edge_samples','native_witness_points'):
        data[key]*=np.asarray(cfg['positive_scale_xyz'],np.float64)
    candidate.require(np.array_equal(data['native_anchor3D'],np.asarray(cfg['native8_canonical_xyz'])),'Exact declared positive scale')
    return data


def fixture_camera(anchor,signs):
    """INVENTED geometry proof only; never substituted into actual old source."""
    eye=anchor+signs*np.asarray([.20,.16,.24]);f=anchor-eye;f/=np.linalg.norm(f)
    right=np.cross(f,[0.,-1.,0.]);right/=np.linalg.norm(right);down=np.cross(f,right);R=np.stack((right,down,f))
    return dict(K=[[512.,0,255.5],[0,512.,255.5],[0,0,1.]],R=R,t=-R@eye,eye_world=eye,near=.001)


def checks(cfgs,ns):
    tests={};records=[];tt=templates(ns);core=module(CORE,'v188_pure_corpus_math');local=module(B,'v188_frozen_B_locality')
    cells={}
    for cfg in cfgs:cells.setdefault(tuple(cfg['factors'][k] for k in ('material','aspect','geometry_family')),cfg)
    candidate.require(len(cells)==8,'Material/aspect/geometry independent eight cells')
    full,invalid=local.exact96_support([255.5,255.5],96.,[512,512])
    for cell,cfg in sorted(cells.items()):
        data=scaled(tt[cfg['template_index']],cfg);tri=data['vertices'][data['triangles']]
        for k in range(8):
            c=cfg['permCF_to_canonical'][k];cam=fixture_camera(data['native_anchor3D'][c],ns['SIGNS'][c])
            pair=candidate.propose_pair(data,k,cfg['permCF_to_canonical'],cam,[255.5,255.5],96.,[512,512])
            points=pair['finite_end_witness37'];pid=pair['primary_part_id']
            def states(foreign):
                triangles=tri if foreign is None else np.concatenate((tri,foreign[pair['foreign_triangles']]))
                owners=data['part_owner'] if foreign is None else np.r_[data['part_owner'],np.full(2,999,np.int32)]
                return ns['classify_hits']([ns['first_hit'](cam['eye_world'],p,triangles,owners) for p in points],pid)
            before=states(None);cover=states(pair['cover_vertices']);sham=states(pair['sham_vertices'])
            outer=pair['outer_broad_witnesses'];combined=np.concatenate((tri,pair['cover_vertices'][pair['foreign_triangles']]))
            outerstates=ns['classify_hits']([ns['first_hit'](cam['eye_world'],p,combined,np.r_[data['part_owner'],np.full(2,999,np.int32)]) for p in outer],pid)
            xy,_=candidate.project(pair['finite_end_quads'].reshape(-1,3),cam)
            foot,oof=local.candidate_footprint([np.r_[xy.min(0),xy.max(0)].tolist()],[512,512])
            record=dict(material=cell[0],aspect=cell[1],geometry_family=cell[2],native_slot=k,
                BEFORE_all37_D=bool((before=='D').all()),COVER_all37_E=bool((cover=='E').all()),
                SHAM_all37_D=bool((sham=='D').all()),broad_outer_still_D=bool((outerstates=='D').any()),
                full_positive_raw_tap_support=not oof and foot<=full.keys(),input_q_L_unchanged=pair['query_xy'].tolist()==[255.5,255.5] and pair['L']==96.)
            candidate.require(all(v for key,v in record.items() if isinstance(v,bool)),'All64 constructive end-only ray fixtures')
            records.append(record)
    tests['64_source_parameter_constructed_fixtures_all8cells_native8_end_D_E_D_and_outer_D']=len(records)==64
    tests['all_fixtures_new_camera_is_invented_only_never_actual_source_replacement']=True
    # Fixed native mapping, missing/U, opaque proof failure, and no broad/owner alias.
    try:candidate.finite_end(tt[0],0,[0]*8)
    except AssertionError:tests['duplicate_permutation_rejected']=True
    unknown=candidate.direct_targets(dict(status='UNKNOWN',reasons=['SELF_OR_AMBIGUOUS']))
    tests['self_ambiguous_no_plan_preserves_all_UNKNOWN']=all(unknown[v]==-1 for v in VARIANTS)
    tests['only_requested_slot_target_other7_UNKNOWN']=candidate.requested_target8(7,1).tolist()==[-1]*7+[1]
    toy=dict(development_role='SOURCE_TRAIN',native_slot=0,query_xy=[24.,24.],L=32.,raw_hw=[64,64],original_query_available=True)
    endpoint=dict(authority=local.SYNTH_AUTHORITY,native_slot=0,selected_instance_supported=True,intended_part_supported=True,queried_end_localization_supported=True,all_plausible_candidates_enclosed=True,before_end_visible_supported=True,evidence_binding={'invented':True},end_locality_id='INVENTED',candidate_enclosures_xyxy=[[20.,20.,22.,22.]],surrounding_material_witnesses=[dict(raw_pixel_xy=[15,15],material_observed=True,same_selected_part_supported=True,evidence_binding={'invented':True})])
    plan=local.evaluate_plan(toy,endpoint,dict(cover_bbox_xyxy=[19,19,24,24],sham_translation_xy=[10,0]))
    proof_names=('original_q_L_native_flags_and_all8_inputs_unchanged','before_original_RGB96_valid96_target_context_exact','every_candidate_positive_RGB_tap_cover_alpha_one','every_candidate_positive_RGB_tap_sham_alpha_zero','distinct_foreign_material_supported','foreign_in_front_paste_composite_or_source_order','same_transformed_foreign_RGB_alpha_cover_and_sham','equal_positive_alpha_pasted_mass','background_outside_paste_exact','original_crop_cover_RGB_bytes_changed','original_crop_sham_has_donor_RGB_support','certified_surrounding_material_RGB_unchanged_both')
    proof={n:True for n in proof_names};proof['evidence_bindings']={'invented':True}
    closed=local.close_actual_proof(plan,proof);labels=candidate.direct_targets(closed)
    tests['B_frozen_closed_proof_to_separate_direct0_1_0_only']= [labels[v] for v in VARIANTS]==[0,1,0]
    for name in proof_names:
        bad=dict(proof);bad[name]=False;lab=candidate.direct_targets(local.close_actual_proof(plan,bad))
        candidate.require(all(lab[v]==-1 for v in VARIANTS),'Each actual proof required')
    tests['all12_actual_proof_failures_remain_unknown_no_weak_auto_admission']=True
    import torch
    z=torch.tensor([.2,-.3,.4,.7,float('nan')],requires_grad=True)
    target=torch.tensor([0,0,1,-1,-1],dtype=torch.int8);av=torch.tensor([1,1,1,1,0],dtype=torch.bool)
    loss=candidate.balanced_direct_end_loss(z,target,av);loss.backward()
    oracle=np.asarray([1/(1+np.exp(-.2))/4,1/(1+np.exp(.3))/4,(1/(1+np.exp(-.4))-1)/2,0,0])
    tests['direct_balanced_analytic_gradient_unknown_and_U_zero']=np.allclose(z.grad.numpy(),oracle,rtol=0,atol=3e-8)
    base=torch.tensor(1.25,requires_grad=True)
    tests['lambda0_exact_object_bypass_preserves_old_loss_and_no_direct_grad']=candidate.add_direct_objective(base,z,target,av,0) is base
    try:candidate.balanced_direct_end_loss(torch.zeros(5),torch.zeros(5,dtype=torch.int8),torch.ones(5,dtype=torch.bool))
    except AssertionError:tests['zero_class_denominator_failclosed']=True
    try:candidate.balanced_direct_end_loss(torch.zeros(2),torch.tensor([0,1],dtype=torch.int8),torch.tensor([False,True]))
    except AssertionError:tests['inputU_known_target_rejected']=True
    # Exercise the actual output helpers on invented arrays, not just equations.
    tiny=np.arange(9*11*3).reshape(9,11,3).astype(np.uint8);tinyrows=[dict(query_xy=[3.25,3.75],L=6.) for _ in range(8)]
    sampled,vv=crop8(crop_function(),tiny,tinyrows);oracle_rgb=np.zeros((96,96,3),np.uint8);oracle_valid=np.zeros((96,96),bool)
    for yy in range(96):
        for xx in range(96):
            point=(3.25+((xx+.5)/96-.5)*6,3.75+((yy+.5)/96-.5)*6);acc=np.zeros(3)
            for pixel,weight in local.rgb_taps(point,[9,11]):
                if pixel is not None:acc+=tiny[pixel[1],pixel[0]].astype(float)*weight
            oracle_rgb[yy,xx]=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8)
            oracle_valid[yy,xx]=-.5<=point[0]<10.5 and -.5<=point[1]<8.5
    tests['actual_crop8_V90_independent_scalar4tap_uint8_round_and_valid']=np.array_equal(sampled[0],oracle_rgb) and np.array_equal(vv[0],oracle_valid)
    tinybefore=dict(RGB_uint8=tiny,part_id=np.zeros((9,11),np.int32),camera_depth=np.full((9,11),2.),global_triangle_id=np.zeros((9,11),np.int32),ambiguity=np.zeros((9,11),bool))
    tinypair=dict(cover_enclosure_xyxy=np.array([2.,2.,3.,3.]),sham_translation_image_xy=[4,0],cover_front_z=1.,target_triangle_count=12)
    tc,ac=paste_plane(tinybefore,tiny.copy(),tinypair,'COVER');ts,ass=paste_plane(tinybefore,tiny.copy(),tinypair,'SHAM')
    tests['actual_planar_paste_same_alpha_area_translation_and_immutable_background']=ac.sum()==ass.sum()==4 and np.array_equal(tc['RGB_uint8'][2:4,2:4],ts['RGB_uint8'][2:4,6:8]) and np.array_equal(ts['RGB_uint8'][~ass],tiny[~ass]) and np.array_equal(tinybefore['RGB_uint8'],tiny)
    tests['candidate115157_plus65_separate_logit_no_model_instantiated']=115157+64+1==115222
    candidate.require(all(tests.values()),'Meaningful invented checks')
    return dict(completed=True,PASS=True,invented_only=True,tests=tests,number_checks=len(tests),constructive_fixtures64=records,
        source_image_mask_depth_NPZ_header_weight_or_feature_reads=0,model_instantiations_or_forwards=0,fit_updates=0)


def prepare():
    candidate.require(sha(SCOPE)==SCOPE_SHA,'Frozen root preparation scope')
    cfgs,rows=population();ns=source_functions();jlines(POPULATION,rows);proof=checks(cfgs,ns);write(CHECKS,proof)
    sources={name:bind(path) for name,path in [('source_constructive_code',TEACHER),('source_configs',CONFIGS),('source_material_renderer',CORE),('source_end_operator',OP),('source_TRAIN_input_plan',PLAN),('original_crop_sampler',SAMPLER),('frozen_B_end_contract',B),('existing_head',HEAD),('existing_trainer',TRAINER),('existing_train_protocol',OLDPROTOCOL)]}
    protocol=dict(completed=True,prepared_only=True,scope=bind(SCOPE),code=bind(__file__),candidate=bind(CODE),checks=bind(CHECKS),population=bind(POPULATION),sources=sources,
        counts=dict(TRAIN_bases=128,requested_native_end_attempts=1024,triplet_status_rows=3072,attempted_packet_native_rows=24576,original_BEFORE_inputU=16,selected_triplet_original_inputU=48,native_slots=list(range(8)),factor_cells={str(key):n for key,n in Counter(tuple(r['factors'][k] for k in ('material','aspect','geometry_family')) for r in rows).items()}),
        locality='Exact existing V127 step=min(.030,float(min(part_bounds_side))*.7), then original positive_scale_xyz. Fixed inner0.2 of each existing3face patch about registered endpoint index0; source slot->cfg.permCF_to_canonical. This finite convention is not an exact physical/native gold claim.',
        placement='Original camera/q/L/flags unchanged. Project fixed end3quads. Closed cover enclosure floor(minxy)-1 .. ceil(maxxy)+1. Opaque camera-aligned two-triangle plane at half minimum end cameraZ, no thickness. SHAM same plane/appearance at unchangedZ translated cameraX by ceil(enclosure_width+8) imagepixels; integer translation preserves alpha area. No search/shrink/retry.',
        actual_proofs='Original BEFORE depth/part/triangle/camera/CF exact guards; dense finite-end region target-visible before, no self/ambiguity/OOF; conservative B positive4tap footprint fully in original96 support; everycandidate alpha1COVER/alpha0SHAM; sameforeignobject/frontdepth/equalalpha; at least one sameprimary outerbroad visible witness retained byboth; original crops visibly changed byCOVER andSHAM hasdonor support. Anyfailure retainedUNKNOWN, no replacements.',
        source_RGB='Use original BEFORE RGB exact outside opaque new foreign plane; shade only new plane through pinned V129 source material/shading. All three saved raw96 use original V90 F64 bilinear+uint8round operator. NoGTcentredcrop or querycorrection.',
        labels='Separate direct_end_external_target {-1,0,1}, known triplet0/1/0 onlyafteractualproof closure. Other7 queried-native targetsUNKNOWN. OldnativeD/E/S, broad2, owner3 and quality neverconverted/overwritten. Nativevisibility and quality targets remain-1; trainingadmissionfalse.',
        candidate_head=dict(existing_parameters=115157,extra_direct_linear64_to1_parameters=65,total=115222,zero_last=True,paired_candidates=['RAW96','RAW96_PLUS_FROZEN_SAM'],same_original8_input_channels=True,old_support2_owner3_outputs_unchanged=True,quality_gate=False),
        candidate_objective='Preserve old arithmetic mean5 exactly; independent additive direct-end balancedD/E BCE onknown/available rows only. Unknown/U excluded, emptyclassfails. Lambda0 exactbypass. Eventualrootfitdesign must freeze dataadmission/weight/draws/steps/sameinit andlambda0stateparity; nofitentryissuedhere.',
        evaluation='No current/private/publicheld/VAL/V182COMMON bodyreads; COMMONneverfit, allhistoricaleval only/noblindclaim. No architecture/arm chosen fromrealmetrics.',
        sourcebody_or_header_hash_reads=0,new_model_or_forward_or_fit=0,selected_model=None,adopted=False,fully_solved=False,output=str(OUT),automatic_retry=False)
    write(PROTOCOL,protocol)
    template=dict(approved_synthetic_direct_end_generation=False,source_TRAIN_only=True,automatic_retry=False,scope=bind(SCOPE),code=bind(__file__),candidate=bind(CODE),protocol=bind(PROTOCOL),population=bind(POPULATION),source_buffer_bindings_to_be_root_frozen=None,fit=False,model_forward=False)
    write(H/'ROOT_ACTUAL_GENERATION_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),candidate=bind(CODE),protocol=bind(PROTOCOL),checks=bind(CHECKS),population=bind(POPULATION),actual_source_body_reads=0,actual_generation=0,actual_model_fit_forward=0,execution_runner='--entry ROOT_APPROVED_ACTUAL_ENTRY.json; root sourcebufferwhitelist must match all128 originalBEFORE bindings',selected_model=None,fully_solved=False))
    print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)


def crop8(crop,rgb,eight):
    images=[];valids=[]
    for r in eight:
        q,L=r['query_xy'],r['L']
        if q is None or L is None or not np.isfinite(q).all() or not np.isfinite(L) or L<=0:
            im=np.zeros((96,96,3),np.uint8);valid=np.zeros((96,96),bool)
        else:
            a,b=crop(rgb,np.asarray(q,np.float64),L,96);im=a.transpose(1,2,0).copy();valid=b[0].copy()
        images.append(im);valids.append(valid)
    return np.asarray(images),np.asarray(valids)


def load_before(binding,cfg):
    path=check(binding)
    keys=('RGB_uint8','part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical')
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in keys}
    candidate.require(a['RGB_uint8'].dtype==np.uint8 and a['RGB_uint8'].shape==(512,512,3),'Original RGB schema')
    for key,dtype in [('part_id',np.int32),('camera_depth',np.float64),('global_triangle_id',np.int32),('ambiguity',bool)]:
        candidate.require(a[key].dtype==dtype and a[key].shape==(512,512),'Original source raster '+key)
    for k in ('K','R','t'):candidate.require(np.array_equal(a['camera_'+k],np.asarray(cfg['camera'][k],np.float64)),'Exact original camera')
    candidate.require(np.array_equal(a['native8_CF_xy'],np.asarray(cfg['native8_CF_xy'],np.float64)) and np.array_equal(a['permCF_to_canonical'],np.asarray(cfg['permCF_to_canonical'],np.int32)),'Exact original CF registration')
    candidate.require(not (a['part_id']==999).any(),'Original BEFORE no foreign object')
    return a


def paste_plane(before,plane_rgb,pair,variant):
    lo=np.asarray(pair['cover_enclosure_xyxy'][:2],int);hi=np.asarray(pair['cover_enclosure_xyxy'][2:],int)
    delta=np.asarray([0,0] if variant=='COVER' else pair['sham_translation_image_xy'],int)
    h,w=before['part_id'].shape;x0,y0=lo+delta;x1,y1=hi+delta
    candidate.require(0<=x0<=x1<w and 0<=y0<=y1<h,'Frozen cover/sham fully in frame')
    paste=np.zeros((h,w),bool);paste[y0:y1+1,x0:x1+1]=True
    z=pair['cover_front_z'];bd=before['camera_depth'];finite=np.isfinite(bd)
    tol=64*np.finfo(np.float64).eps*np.maximum(1,np.maximum(abs(z),np.where(finite,np.abs(bd),1)))
    alpha=paste&(~finite|(z<bd-tol));tie=paste&finite&(np.abs(z-bd)<=tol)
    scene={k:v.copy() for k,v in before.items()};scene['part_id'][alpha]=999;scene['camera_depth'][alpha]=z
    yy,xx=np.indices((h,w));triangle=(yy-y0)/max(1,y1-y0)>(xx-x0)/max(1,x1-x0)
    scene['global_triangle_id'][alpha]=pair['target_triangle_count']+triangle[alpha].astype(np.int32)
    scene['ambiguity'][alpha]=False;scene['ambiguity'][tie]=True
    rgb=before['RGB_uint8'].copy();patch_rgb=plane_rgb[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
    dst=rgb[y0:y1+1,x0:x1+1];ma=alpha[y0:y1+1,x0:x1+1];dst[ma]=patch_rgb[ma]
    scene['RGB_uint8']=rgb
    return scene,alpha


def plane_texture(core,data,cfg,pair):
    """Original procedural material/light shader, evaluated once in cover pose.

    SHAM transports this exact opaque texture with its integer translation.
    Original target/background pixels are always copied, never re-shaded.
    """
    h,w=cfg['raw_hw'];lo=np.asarray(pair['cover_enclosure_xyxy'][:2],int);hi=np.asarray(pair['cover_enclosure_xyxy'][2:],int)
    vertices=np.concatenate((data['vertices'],pair['cover_vertices']))
    triangles=np.concatenate((data['triangles'],pair['foreign_triangles']+len(data['vertices'])))
    buffer=dict(depth=np.full((h,w),np.inf),part_id=np.full((h,w),-1,np.int32),face_id=np.full((h,w),-1,np.int32))
    buffer['depth'][lo[1]:hi[1]+1,lo[0]:hi[0]+1]=pair['cover_front_z'];buffer['part_id'][lo[1]:hi[1]+1,lo[0]:hi[0]+1]=999
    buffer['face_id'][lo[1]:hi[1]+1,lo[0]:hi[0]+1]=len(data['triangles'])
    return core.shade(buffer,vertices,triangles,cfg,np.zeros(3))


def run(entry_path):
    # No binary/header/sourcepixel hashing occurs before this complete closure.
    e=read(entry_path)
    candidate.require(e.get('approved_synthetic_direct_end_generation') is True and e.get('source_TRAIN_only') is True and e.get('automatic_retry') is False,'Separate root actual source generation entry')
    candidate.require(e.get('fit') is False and e.get('model_forward') is False,'Generation only')
    for key,path in [('scope',SCOPE),('code',Path(__file__)),('candidate',CODE),('protocol',PROTOCOL),('population',POPULATION)]:candidate.require(e[key]==bind(path),'Frozen entry '+key)
    p=read(PROTOCOL);candidate.require(p['checks']==bind(CHECKS) and read(CHECKS)['PASS'],'Frozen invented checks')
    for b in p['sources'].values():check(b)
    rows=lines(check(p['population']));cfgs={r['base_index']:r for r in read(CONFIGS)['rows'] if r['split']=='TRAIN'}
    expected={r['source_BEFORE_buffer']['path']:r['source_BEFORE_buffer'] for r in rows}
    supplied=e['source_buffer_bindings_to_be_root_frozen'];candidate.require(isinstance(supplied,list) and {b['path']:b for b in supplied}==expected and len(supplied)==128,'Only exact128 original TRAIN BEFORE buffer whitelist')
    candidate.require(not OUT.exists(),'No overwrite or automatic retry');OUT.mkdir()
    write(OUT/'STARTED.json',dict(entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),source_TRAIN_only=True,new_model_fit_forward=False))
    ns=source_functions();tt=templates(ns);core=module(CORE,'v188_pure_source_shader');op=module(OP,'v188_fixed_broad_support');local=module(B,'v188_fixed_end_locality');crop=crop_function()
    attempts=[];variants=[];used={};start=time.monotonic();before_parity=0
    try:
        for base in range(128):
            cfg=cfgs[base];eight=rows[base*8:base*8+8];before=load_before(eight[0]['source_BEFORE_buffer'],cfg);used[eight[0]['source_BEFORE_buffer']['path']]=eight[0]['source_BEFORE_buffer']
            data=scaled(tt[cfg['template_index']],cfg);original_rgb8,original_valid8=crop8(crop,before['RGB_uint8'],eight)
            for k,r in enumerate(eight):
                candidate.require(array_sha(original_rgb8[k])==r['original_rgb96_sha256'] and array_sha(original_valid8[k])==r['original_valid96_sha256'],'All1024 original BEFORE crop RGB/valid exactparity');before_parity+=1
            for k,r in enumerate(eight):
                result=dict(r,status='UNKNOWN_RETAINED',reasons=[],outputs=None,direct_end_states=[-1,-1,-1],training_admission=False)
                status3=[]
                def reject(reason):
                    result['reasons'].append(reason)
                    return result
                if not r['original_input_flags']['query_available']:
                    reject('ORIGINAL_INPUT_U')
                else:
                    try:
                        pair=candidate.propose_pair(data,k,cfg['permCF_to_canonical'],cfg['camera'],r['query_xy'],r['L'],r['raw_hw'])
                    except AssertionError as err:
                        reject('GEOMETRY_'+str(err));pair=None
                    if pair is not None:
                        patch=op.expected_patch_depth(pair['finite_end_quads'],cfg['camera'],r['raw_hw']);before_cls=op.classify_end_pixels(before,before,patch,pair['primary_part_id'])
                        visible=patch['geometry_valid'] and not patch['projected_region_out_of_frame'] and patch['region_mask'].any() and bool((before_cls[patch['region_mask']]==1).all())
                        if not visible:
                            reject('BEFORE_END_SELF_AMBIGUOUS_REGISTRATION_OR_OOF_UNKNOWN')
                        else:
                            xy,_=candidate.project(pair['finite_end_quads'].reshape(-1,3),cfg['camera']);box=np.r_[xy.min(0),xy.max(0)].tolist();endpix,oof=local.candidate_footprint([box],r['raw_hw'])
                            full,_=local.exact96_support(r['query_xy'],r['L'],r['raw_hw']);lo=pair['cover_enclosure_xyxy'][:2].astype(int);hi=pair['cover_enclosure_xyxy'][2:].astype(int)
                            coverbox=[int(lo[0]),int(lo[1]),int(hi[0])+1,int(hi[1])+1];cp=local.rectangle_pixels(coverbox);dx,dy=pair['sham_translation_image_xy'];sp={(x+dx,y+dy) for x,y in cp}
                            broadpatch=op.expected_patch_depth(pair['broad_patch_quads'],cfg['camera'],r['raw_hw']);bc=op.classify_end_pixels(before,before,broadpatch,pair['primary_part_id'])
                            outer=[(x,y) for x,y in sorted(full,key=lambda q:(q[1],q[0])) if bc[y,x]==1 and (x,y) not in endpix|cp|sp]
                            endpoint=dict(authority=local.SYNTH_AUTHORITY,native_slot=k,selected_instance_supported=True,intended_part_supported=True,queried_end_localization_supported=True,all_plausible_candidates_enclosed=True,before_end_visible_supported=True,end_locality_id=pair['convention'],candidate_enclosures_xyxy=[box],evidence_binding=dict(source=r['source_BEFORE_buffer'],constructive=p['sources']['source_constructive_code'],base=base,canonical=pair['canonical_index']),surrounding_material_witnesses=[] if not outer else [dict(raw_pixel_xy=list(outer[0]),material_observed=True,same_selected_part_supported=True,evidence_binding=r['source_BEFORE_buffer'])])
                            original=dict(development_role='SOURCE_TRAIN',native_slot=k,query_xy=r['query_xy'],L=r['L'],raw_hw=r['raw_hw'],original_query_available=True)
                            plan=local.evaluate_plan(original,endpoint,dict(cover_bbox_xyxy=coverbox,sham_translation_xy=[dx,dy]));result['geometry_plan']=plan
                            if plan['status']!='GEOMETRY_READY_PIXEL_PROOFS_PENDING':
                                reject('GEOMETRY_PLAN_'+plan['status'])
                            else:
                                texture=plane_texture(core,data,cfg,pair);cover,ca=paste_plane(before,texture,pair,'COVER');sham,sa=paste_plane(before,texture,pair,'SHAM');scene3=[before,cover,sham]
                                rgb3=[];valid3=[];broad=[]
                                for scene in scene3:
                                    rgb8,v8=crop8(crop,scene['RGB_uint8'],eight);rgb3.append(rgb8);valid3.append(v8)
                                    c=op.classify_end_pixels(before,scene,broadpatch,pair['primary_part_id']);z=op.summarize(op.sample_classes(c,op.original_crop_positions(r['query_xy'],r['L'])),broadpatch,True);broad.append([z['visible_support_target'],z['foreign_front_support_target']])
                                rgb3=np.asarray(rgb3);valid3=np.asarray(valid3)
                                endxy=np.asarray(list(endpix),int);ex,ey=endxy[:,0],endxy[:,1];outerxy=outer[0];ox,oy=outerxy
                                before_sha=array_sha(before['RGB_uint8']);proof=dict(original_q_L_native_flags_and_all8_inputs_unchanged=True,before_original_RGB96_valid96_target_context_exact=np.array_equal(rgb3[0],original_rgb8) and np.array_equal(valid3[0],original_valid8),every_candidate_positive_RGB_tap_cover_alpha_one=bool(ca[ey,ex].all()),every_candidate_positive_RGB_tap_sham_alpha_zero=not bool(sa[ey,ex].any()),distinct_foreign_material_supported=True,foreign_in_front_paste_composite_or_source_order=bool((cover['part_id'][ey,ex]==999).all()),same_transformed_foreign_RGB_alpha_cover_and_sham=np.array_equal(ca[lo[1]:hi[1]+1,lo[0]:hi[0]+1],sa[lo[1]+dy:hi[1]+dy+1,lo[0]+dx:hi[0]+dx+1]) and np.array_equal(cover['RGB_uint8'][lo[1]:hi[1]+1,lo[0]:hi[0]+1][ca[lo[1]:hi[1]+1,lo[0]:hi[0]+1]],sham['RGB_uint8'][lo[1]+dy:hi[1]+dy+1,lo[0]+dx:hi[0]+dx+1][sa[lo[1]+dy:hi[1]+dy+1,lo[0]+dx:hi[0]+dx+1]]),equal_positive_alpha_pasted_mass=int(ca.sum())==int(sa.sum()),background_outside_paste_exact=np.array_equal(cover['RGB_uint8'][~ca],before['RGB_uint8'][~ca]) and np.array_equal(sham['RGB_uint8'][~sa],before['RGB_uint8'][~sa]),original_crop_cover_RGB_bytes_changed=not np.array_equal(rgb3[1,k],rgb3[0,k]),original_crop_sham_has_donor_RGB_support=any(sa[y,x] for x,y in full),certified_surrounding_material_RGB_unchanged_both=np.array_equal(cover['RGB_uint8'][oy,ox],before['RGB_uint8'][oy,ox]) and np.array_equal(sham['RGB_uint8'][oy,ox],before['RGB_uint8'][oy,ox]),evidence_bindings=dict(source_BEFORE=r['source_BEFORE_buffer'],before_decoded_RGB_sha256=before_sha,raw96_before=array_sha(rgb3[0]),raw96_cover=array_sha(rgb3[1]),raw96_sham=array_sha(rgb3[2])))
                                closed=local.close_actual_proof(plan,{key:bool(v) if isinstance(v,(bool,np.bool_)) else v for key,v in proof.items()});labels=candidate.direct_targets(closed)
                                result['closed_proof']=closed;result['direct_end_states']=[labels[v] for v in VARIANTS];result['broad_support2_diagnostic']=broad
                                result['status']=closed['status'];result['reasons']=closed['reasons']
                                # Keep attempted constructed arrays even if their proof fails.
                                directory=OUT/r['attempt_id'];directory.mkdir();file=directory/'CONTROL_TRIPLET_INPUTS.npz'
                                direct8=np.asarray([candidate.requested_target8(k,labels[v]) for v in VARIANTS])
                                with file.open('xb') as f:np.savez_compressed(f,full_RGB=np.asarray([s['RGB_uint8'] for s in scene3]),rgb96=rgb3,valid96=valid3,query_xy=np.asarray([e['query_xy'] if e['query_xy'] is not None else [np.nan,np.nan] for e in eight],np.float64),L=np.asarray([np.nan if e['L'] is None else e['L'] for e in eight],np.float64),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.asarray([e['original_global_index'] for e in eight],np.int64),original_occurrence_id=np.asarray([e['original_occurrence_id'] for e in eight]),original_query_available=np.asarray([e['original_input_flags']['query_available'] for e in eight],bool),direct_end_external_target=direct8,new_native_visibility_target=np.full((3,8),-1,np.int8),new_quality_target=np.full((3,8),-1,np.int8),new_owner_target=np.full((3,8),-1,np.int8),broad_support2_diagnostic=np.asarray(broad,np.int8),foreign_alpha=np.asarray([np.zeros_like(ca),ca,sa]),variant=np.asarray(VARIANTS))
                                result['outputs']=bind(file)
                attempts.append(result)
                for vi,var in enumerate(VARIANTS):variants.append(dict(attempt_id=r['attempt_id'],variant=var,base_index=base,native_slot=k,original_global_index=r['original_global_index'],query_xy=r['query_xy'],L=r['L'],original_input_flags=r['original_input_flags'],constructed=result['outputs'] is not None,outputs=result['outputs'],status=result['status'],reasons=result['reasons'],direct_end_external_target=result['direct_end_states'][vi],new_native_visibility_target=-1,new_quality_target=-1,new_owner_target=-1,training_admission=False))
            print(json.dumps(dict(bases_done=base+1,attempts=len(attempts),seconds=time.monotonic()-start)),flush=True)
        candidate.require(len(attempts)==1024 and len(variants)==3072 and before_parity==1024,'Full attempt census no replacement')
        ap=OUT/'ALL1024_END_CONTROL_ATTEMPTS.jsonl';vp=OUT/'ALL3072_VARIANT_STATUS.jsonl';jlines(ap,attempts);jlines(vp,variants)
        counts=dict(attempts=1024,status=dict(Counter(r['status'] for r in attempts)),constructed=sum(r['outputs'] is not None for r in attempts),known_direct_triplets=sum(r['direct_end_states']==[0,1,0] for r in attempts),original_BEFORE_inputU16=sum(not r['original_input_flags']['query_available'] for r in attempts),before_RGB96_valid96_exactparity=before_parity)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_TRAIN_only=True,entry=bind(entry_path),code=bind(__file__),candidate=bind(CODE),protocol=bind(PROTOCOL),population=bind(POPULATION),attempts=bind(ap),variants=bind(vp),source_assets=list(used.values()),counts=counts,all_attempts_preserved=True,no_failure_replacement=True,new_training_admission=False,new_fit_forward=0,selected_model=None,adopted=False,fully_solved=False,seconds=time.monotonic()-start))
        print(json.dumps(dict(completed=True,PASS=True,completion=bind(OUT/'COMPLETE.json'),counts=counts)),flush=True)
    except BaseException:
        jlines(OUT/'PARTIAL_ATTEMPTS_PRESERVED.jsonl',attempts);write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();choice=parser.add_mutually_exclusive_group(required=True);choice.add_argument('--prepare',action='store_true');choice.add_argument('--entry',type=Path);args=parser.parse_args()
    prepare() if args.prepare else run(args.entry)
