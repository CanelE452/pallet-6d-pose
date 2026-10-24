"""V164 source native-part intervention preparation and one-shot pixel construction.
--prepare reads metadata/text and uses invented arrays only. Actual source/donor
pixel bodies require a separate root entry bound to this frozen code/protocol.
"""
from pathlib import Path
import argparse, ast, copy, hashlib, json, math, sys, time, traceback
import numpy as np
from PIL import Image
import PIL, scipy, torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OLD=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
WIT=P/'query_visibility_train_native_part_witness_20261024_v163'
MR=P/'query_visibility_mr6d_full_public_source_20261024_v153'
DESIGN=V/'ROOT_DESIGN_BEFORE_NATIVE_PART_COUNTERFACTUAL_PIXELS.json'
DESIGN_SHA='49075db8a881e31812a407a8edc0f267bec04694ff62df0353ed1c61cbb8da30'
PILOT=WIT/'reviewer_a/part_region_pilot/ALL22_COARSE_PART_REGION_OBSERVATIONS.jsonl'
PILOT_SHA='7a6042e4b3cc79a85dd172617cbcd1d4f6d9cc64acd8d48f19e3fd408517b044'
PILOT_COMPLETE=WIT/'reviewer_a/part_region_pilot/COMPLETE.json'
PILOT_COMPLETE_SHA='5eeebe18cb3c97b0a4a1ef7d6ffb348ef2a188a52140f83b7d500c503383c05a'
BLINDED=WIT/'evidence_preparation/BLINDED_INPUT390_AND_FRAME88_BINDINGS.json'
FACTS=MR/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
QUERIES=MR/'actual_full230_initializer_v1/ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'
DONOR_COMPLETE=OLD/'donor_proof_v1/DONOR_PROOF_COMPLETE.json'
HELPER=OLD/'pixel_builder_preparation/build_coherent_source_pixels.py'
HELPER_SHA='835c2df24a1cad8bef97947bada0c28a2e8411b3115b757156bdaaa3d7a10875'
LOCAL=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
LOCAL_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
TARGET=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
TARGET_SHA='aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3'
C8=P/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'
C8_SHA='cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578'
C9=P/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py'
C9_SHA='8deae91931c12bcc1692c98ce74fa5e703cfb0d57da980b126d5ca45ca74873e'
PROTOCOL=H/'PROTOCOL_BEFORE_NATIVE_PART_SOURCE_PIXEL_BODIES.json'
INPUTS=H/'FIXED22_FRAME_INPUTS_AND_66_ATTEMPTS_BEFORE_PIXELS.json'
OUT=V/'source_part_pixels_v1'
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    p=Path(b['path']);assert sha(p)==b['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pure(path,names,ns):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::PURE_AST_ONLY','exec'),ns)
    return ns
def operators():
    for p,h in [(HELPER,HELPER_SHA),(LOCAL,LOCAL_SHA),(TARGET,TARGET_SHA),(C8,C8_SHA),(C9,C9_SHA)]:assert sha(p)==h,str(p)
    ns=dict(np=np,Image=Image,math=math,distance_transform_edt=distance_transform_edt,exact=exact,ash=ash)
    pure(HELPER,('erode1','transformed_cutout','opaque_anchor','paste','raw_footprint','materialize','verify_before'),ns)
    pure(LOCAL,('crop',),ns);pure(TARGET,('valid_box','target_context_grid','sample_target_context'),ns)
    n=dict(torch=torch,F=F,MEAN=(.485,.456,.406),STD=(.229,.224,.225))
    pure(C8,('preprocess_views',),n);pure(C9,('preprocess_target',),n)
    ns.update(preprocess_views=n['preprocess_views'],preprocess_target=n['preprocess_target'])
    return ns
def materialize(rgb,frame,op):
    r=op['materialize'](rgb,frame,op)
    r['query_available']=np.array(frame['native_continuous_valid'],bool)
    r['owner_target']=np.full(8,-1,np.int8)
    return r
def views17(a,op,device):
    local=torch.as_tensor(np.ascontiguousarray(a['rgb96'].transpose(0,3,1,2)),device=device)
    valid=torch.as_tensor(a['valid96'][:,None],device=device)
    lv,_=op['preprocess_views'](local,valid)
    tv=op['preprocess_target'](torch.as_tensor(a['target224'].transpose(2,0,1)[None].copy(),device=device),
        torch.as_tensor(a['target_valid'][None,None],device=device))
    return torch.cat((tv,lv),0).detach().cpu().contiguous().numpy()
def region_geometry(box,alpha,hw,op):
    """Exactly one source-alpha-based scale and one cover/sham placement; no search."""
    x0,y0,x1,y1=map(int,box);h,w=map(int,hw);rh=y1-y0;rw=x1-x0
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):return None,dict(reason='REGION_INVALID_OR_PADDED')
    if alpha.dtype!=bool or alpha.ndim!=2 or not alpha.any():return None,dict(reason='DONOR_ALPHA_EMPTY')
    dd=distance_transform_edt(np.pad(alpha,1,constant_values=False))[1:-1,1:-1]
    radius=float(dd.max())
    if radius<=1:return None,dict(reason='DONOR_NO_POSITIVE_OPAQUE_INTERIOR_MARGIN')
    required=.5*math.hypot(rw-1,rh-1)+4.
    longside=max(1,math.ceil(max(alpha.shape)*required/(radius-1.)))
    return longside,dict(region_xyxy=[x0,y0,x1,y1],region_wh=[rw,rh],
        region_center_xy=[(x0+x1-1)/2.,(y0+y1-1)/2.],native_EDT_max_radius=radius,
        required_radius_with_fixed_4px_margin=required,longside_pixels=longside)
def fixed_placements(box,alpha,hw,op):
    anchor=op['opaque_anchor'](alpha)
    if anchor is None:return None,dict(reason='TRANSFORMED_ALPHA_EMPTY')
    x0,y0,x1,y1=map(int,box);h,w=hw;ch,cw=alpha.shape
    center=np.array([(x0+x1-1)/2.,(y0+y1-1)/2.])
    cover=np.rint(center-anchor).astype(np.int64)
    direction=1 if center[0]<=(w-1)/2 else -1
    delta=cw+(x1-x0)+4
    sham=cover+np.array([direction*delta,0],np.int64)
    d=dict(opaque_anchor_xy=anchor.tolist(),cover_origin_xy=cover.tolist(),sham_origin_xy=sham.tolist(),
        horizontal_direction=direction,sham_shift_pixels=delta,transformed_hw=[int(ch),int(cw)])
    if not all(0<=int(z[0]) and int(z[0])+cw<=w for z in (cover,sham)):
        return None,{**d,'reason':'FULL_X_BOX_OUT_OF_BOUNDS_NO_RETRY'}
    # Same y clipping is explicit and must produce exactly equal pasted alpha mass.
    return (cover,sham),d
def prove_pair(before,cover,sham,ca,sa,box,k,op,device='cpu'):
    x0,y0,x1,y1=map(int,box);roi=np.s_[y0:y1,x0:x1]
    vb=views17(before,op,device);vc=views17(cover,op,device);vs=views17(sham,op,device)
    cchanged=[not exact(vc[j],vb[j]) for j in (0,1+k,9+k)]
    schanged=[not exact(vs[j],vb[j]) for j in (0,1+k,9+k)]
    checks=dict(region_all_pixels_in_raw_image=(x1>x0 and y1>y0),
        cover_alpha1_every_supported_box_pixel=bool(ca[roi].all()),
        sham_alpha0_every_supported_box_pixel=bool(not sa[roi].any()),
        cover_sham_equal_positive_pasted_alpha_mass=int(ca.sum())==int(sa.sum()) and int(ca.sum())>0,
        cover_changes_actual_region_RGB=bool((cover['full_RGB'][roi]!=before['full_RGB'][roi]).any()),
        cover_changes_selected_actual_C8_C9_input_views=any(cchanged),
        sham_changes_selected_actual_C8_C9_input_views=any(schanged))
    invariant=('valid96','target_valid','crop_constructed','native_continuous_valid','nearest_valid',
        'central4_available','context_available','fractions','query_xy','L','native_slot','query_available')
    for name,z in [('cover',cover),('sham',sham)]:
        checks[name+'_all_input_metadata_validity_invariant']=all(exact(z[f],before[f]) for f in invariant)
        checks[name+'_new_native_quality_owner_all8_unknown']=all((z[f]==-1).all() for f in ('native_visibility_target','point_quality_target','owner_target'))
        checks[name+'_central32_exact_raw_slice']=exact(z['central32'],z['rgb96'][:,32:64,32:64])
        checks[name+'_source_original_inputfacts_unchanged']=all(exact(z[f],before[f]) for f in before if f.startswith('original_') or f=='query_id')
    q=before['query_xy'][k];L=float(before['L'][k]);pts,inside=op['raw_footprint'](q,L,ca.shape)
    cp=op['crop'](np.repeat(ca[:,:,None],3,2).astype(np.uint8)*255,q,L,96)[0]
    sp=op['crop'](np.repeat(sa[:,:,None],3,2).astype(np.uint8)*255,q,L,96)[0]
    detail=dict(pasted_alpha_mass={'COVER':int(ca.sum()),'ADJACENT_SHAM':int(sa.sum())},
        part_region_pixel_count=(x1-x0)*(y1-y0),cover_region_opaque_count=int(ca[roi].sum()),sham_region_overlap_count=int(sa[roi].sum()),
        selected_preprocess_view_order=['target224','local96','central32'],cover_selected_view_changed=cchanged,sham_selected_view_changed=schanged,
        preprocess17_FP32_sha256={name:ash(a) for name,a in [('BEFORE',vb),('COVER',vc),('ADJACENT_SHAM',vs)]},
        preprocess17_each_view_changed_from_before={name:[not exact(z[j],vb[j]) for j in range(17)] for name,z in [('COVER',vc),('ADJACENT_SHAM',vs)]},
        literal_q_center_proof_separate_not_native_target=dict(raw_center4_footprint_inside=inside,raw_center4_footprint_xy=pts,
            cover_all_raw_center4_contributors_donor=inside and all(bool(ca[y,x]) for x,y in pts),
            sham_all_raw_center4_contributors_donor=inside and all(bool(sa[y,x]) for x,y in pts),
            cover_sampled_alpha_center4=cp[0,47:49,47:49].tolist(),sham_sampled_alpha_center4=sp[0,47:49,47:49].tolist(),
            cover_center_RGB_changed=not exact(cover['rgb96'][k,47:49,47:49],before['rgb96'][k,47:49,47:49]),
            sham_center_RGB_changed=not exact(sham['rgb96'][k,47:49,47:49],before['rgb96'][k,47:49,47:49])))
    return checks,detail
def invented_checks():
    op=operators();checks={}
    rgb=(np.indices((240,480)).sum(0)%127).astype(np.uint8);rgb=np.repeat(rgb[:,:,None],3,2)
    q=np.tile([190.25,120.5],(8,1));fr=dict(query_xy=q.tolist(),L=[150.]*8,feature_constructed=[True]*8,
        context_box_xyxy=[100.,70.,300.,170.],native_slot=list(range(8)),native_continuous_valid=[True]*8,nearest_valid=[True]*8)
    box=[175,104,206,133];dr=np.full((51,41,3),[201,71,91],np.uint8);alpha=np.ones((51,41),bool)
    n,g=region_geometry(box,alpha,rgb.shape[:2],op);cut,a=op['transformed_cutout'](dr,alpha,n)
    places,p=fixed_placements(box,a,rgb.shape[:2],op);assert places is not None
    co,ca=op['paste'](rgb,cut,a,places[0]);sh,sa=op['paste'](rgb,cut,a,places[1])
    b=materialize(rgb,fr,op);c=materialize(co,fr,op);s=materialize(sh,fr,op)
    proof,detail=prove_pair(b,c,s,ca,sa,box,0,op);checks.update(proof)
    checks['same_transformed_donor_for_cover_and_sham']=exact(c['full_RGB'][ca],s['full_RGB'][sa])
    checks['sham_allowed_to_change_literalq_independent_of_part']=not sa[104:133,175:206].any()
    broken=ca.copy();broken[box[1],box[0]]=False
    bad,_=prove_pair(b,c,s,broken,sa,box,0,op)
    checks['one_missing_region_corner_alpha_cell_fails']=not bad['cover_alpha1_every_supported_box_pixel']
    # Identical RGB despite alpha proof is unlearnable, never admitted.
    nochange,_=prove_pair(b,b,b,ca,sa,box,0,op)
    checks['identical_inputs_even_with_alpha_are_unlearnable']=not nochange['cover_changes_actual_region_RGB'] and not nochange['cover_changes_selected_actual_C8_C9_input_views']
    edge=copy.deepcopy(fr);edge['query_xy'][0]=[479.75,120.];edge['native_continuous_valid'][0]=True;edge['nearest_valid'][0]=False
    z=materialize(rgb,edge,op)
    checks['continuous_available_not_AND_nearest']=bool(z['query_available'][0]) and not bool(z['nearest_valid'][0])
    edge['native_continuous_valid'][1]=False;edge['query_xy'][1]=[-1.,-1.];edge['feature_constructed'][1]=False
    z=materialize(rgb,edge,op)
    checks['original_U_retained_separate_constructed']=not z['query_available'][1] and not z['crop_constructed'][1] and z['rgb96'].shape[0]==8
    checks['padded_region_rejected']=region_geometry([-1,2,5,7],alpha,rgb.shape[:2],op)[0] is None
    checks['empty_alpha_rejected']=region_geometry(box,np.zeros((4,4),bool),rgb.shape[:2],op)[0] is None
    checks['no_interior_alpha_rejected']=region_geometry(box,np.ones((1,4),bool),rgb.shape[:2],op)[0] is None
    checks['placement_out_of_bounds_failure_no_search']=fixed_placements([0,10,20,30],np.ones((100,400),bool),rgb.shape[:2],op)[0] is None
    unknown=[dict(region_status='UNKNOWN',variants={v:dict(constructed=False) for v in VARIANTS}) for _ in range(15)]
    checks['UNKNOWN15_retained_no_constructed_variants']=len(unknown)==15 and all(not x['variants'][v]['constructed'] for x in unknown for v in VARIANTS)
    # q is far from the selected coarse part; cover remains anchored to actual box.
    far=copy.deepcopy(fr);far['query_xy'][0]=[20.,120.];fb=materialize(rgb,far,op);fc=materialize(co,far,op);fs=materialize(sh,far,op)
    fp,fd=prove_pair(fb,fc,fs,ca,sa,box,0,op)
    checks['q_background_case_not_selection_gate']=fp['cover_alpha1_every_supported_box_pixel'] and not fd['literal_q_center_proof_separate_not_native_target']['cover_all_raw_center4_contributors_donor']
    checks['context_changed_even_if_selected_local_misses_part']=fp['cover_changes_selected_actual_C8_C9_input_views'] and fd['cover_selected_view_changed'][0]
    checks['BEFORE_array_reconstruction_deterministic']=all(exact(b[k],materialize(rgb.copy(),fr,op)[k]) for k in b)
    checks['all8_metadata_targets_unknown_for_BEFORE']=all((b[k]==-1).all() for k in ('native_visibility_target','point_quality_target','owner_target'))
    assert all(bool(v) for v in checks.values()),[k for k,v in checks.items() if not v]
    return dict(completed=True,PASS=True,invented_arrays_only=True,checks=checks,number_checks=len(checks),
        actual_source_donor_pixel_or_NPZ_bodies_read=0,encoder_network_weight_feature_initializer_fit_PnP_calls=0,
        runtime=dict(executable=sys.executable,numpy=np.__version__,torch=torch.__version__,Pillow=PIL.__version__,scipy=scipy.__version__))
def metadata():
    assert sha(DESIGN)==DESIGN_SHA and sha(PILOT)==PILOT_SHA and sha(PILOT_COMPLETE)==PILOT_COMPLETE_SHA
    d=read(DESIGN);pilot=lines(PILOT);pc=read(PILOT_COMPLETE)
    assert pc['completed'] and pc['PASS'] and len(pilot)==22 and [r['case_id'] for r in pilot]==d['base_case_ids_fixed']
    assert {s:sum(r['region_status']==s for r in pilot) for s in ['SUPPORTED','UNKNOWN']}=={'SUPPORTED':17,'UNKNOWN':5}
    blind=read(BLINDED);bc={r['case_id']:r for r in blind['cases']};bf={r['frame_index']:r for r in blind['frames']}
    facts=lines(FACTS);qq=lines(QUERIES);assert len(facts)==len(qq)==1840
    frames={}
    for region in pilot:
        c=bc[region['case_id']]
        for k in ('global_index','query_id','frame_index','source_id','split','native_slot','native_name','query_xy','L','selected_gt_instance_index'):assert c[k]==region[k]
        assert region['split']=='SOURCE_DEVELOPMENT' and region['training_admitted'] is False
        assert region['new_native_target']==region['new_quality_target']==-1
        if region['region_status']=='UNKNOWN':assert region['original_fullRGB_bbox_xyxy'] is None
        else:
            box=region['original_fullRGB_bbox_xyxy'];assert len(box)==4 and all(type(z) is int for z in box)
            assert np.array_equal(np.array(region['diagnostic_bbox_xyxy'])+np.tile(region['diagnostic_raw_xyxy_offset'][:2],2),box)
        i=c['frame_index']
        if i not in frames:
            ff=[r for r in facts if r['frame_index']==i];qs=[r for r in qq if r['frame_index']==i];bb=bf[i]
            assert len(ff)==len(qs)==8 and [r['native_slot'] for r in ff]==[r['native_slot'] for r in qs]==list(range(8))
            for f,q in zip(ff,qs):
                for k in ('query_id','frame_id','source_id','split','native_slot','query_xy','L'):assert f[k]==q[k]
                assert f['split']=='SOURCE_DEVELOPMENT' and f['native_continuous_valid']==q['native_continuous_valid']
                assert f['nearest_valid']==q['query_nearest_pixel_valid']
            fr={k:[r[k] for r in ff] for k in ('global_index','query_id','native_slot','query_xy','L','feature_constructed','native_continuous_valid','nearest_valid','local_central4_available','context_available')}
            fr.update(frame_index=i,frame_id=ff[0]['frame_id'],source_id=bb['source_id'],split=bb['split'],raw_hw=bb['raw_hw'],
                canonical_RGB_NPY=bb['raw_RGB'],decoded_RGB_sha256=bb['decoded_RGB_sha256'],existing_factual96_packet=bb['crop_packet'],
                context_box_xyxy=ff[0]['context_box_xyxy'],facts=ff,original_query_input_metadata=qs)
            frames[i]=fr
    dc=read(DONOR_COMPLETE);assert dc['completed'] and dc['PASS'] and dc['attempted']==dc['passed']==59 and dc['failed']==0
    donor_records=check(dc['records']);donors=lines(donor_records)
    assert len(donors)==59 and len({r['donor_id'] for r in donors})==59 and all(r['PASS'] and r['split']=='TRAIN' for r in donors)
    attempts=[]
    for region in pilot:
        ranked=sorted(donors,key=lambda r:hashlib.sha256((region['case_id']+'V164_PART_DONOR'+r['donor_id']).encode()).hexdigest())
        for j,donor in enumerate(ranked[:3]):
            attempts.append(dict(pair_id=region['case_id']+'_D'+str(j+1),case_id=region['case_id'],region=region,
                frame_index=region['frame_index'],native_slot=region['native_slot'],donor_id=donor['donor_id'],
                donor=dict(compact_RGBA_cutout=donor['compact_RGBA_cutout'],array_proof=donor['array_proof'],
                    original_RGB=donor['original_RGB'],source_certificate=donor['source_certificate'],split='TRAIN'),
                selection_sha256=hashlib.sha256((region['case_id']+'V164_PART_DONOR'+donor['donor_id']).encode()).hexdigest()))
    assert len(attempts)==66 and sum(a['region']['region_status']=='UNKNOWN' for a in attempts)==15
    return d,dict(completed=True,regions_count=22,attempts_count=66,frames=list(frames.values()),attempts=attempts,
        actual_source_donor_pixel_bodies_read=False,donor_records=dc['records'])
def prepare():
    d,inputs=metadata();op=operators();write(INPUTS,inputs)
    generated=H/'GENERATED_MEANINGFUL_CHECKS.json';write(generated,invented_checks())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),design=bind(DESIGN),region_pilot=bind(PILOT),
        region_complete=bind(PILOT_COMPLETE),blinded_input_bindings=bind(BLINDED),inputfacts=bind(FACTS),original_queries=bind(QUERIES),
        donor_proof_complete=bind(DONOR_COMPLETE),donor_proof_records=inputs['donor_records'],inputs=bind(INPUTS),
        generated_checks=bind(generated),safe_helper=bind(HELPER),local_crop=bind(LOCAL),context_crop=bind(TARGET),
        C8_pure_RGB_preprocess=bind(C8),C9_pure_RGB_preprocess=bind(C9),output=str(OUT),
        fixed22_count=22,fixed66_attempt_census=66,supported17_max51_pairs=True,UNKNOWN5_15_attempts_no_images=True,
        donor_selection='Per case sort all59 provedTRAIN donors by SHA256(case_id+V164_PART_DONOR+donor_id), first3. No pixel/outcome dependent replacement.',
        transform='V161 exact Pillow RGB BILINEAR/alpha NEAREST plus false-outside3x3 erode1. Once-only longside=ceil(max(nativeHW)*(0.5*hypot(regionW-1,regionH-1)+4)/(native alpha EDTmax-1)); reject EDTmax<=1.',
        placement='Cover transformed opaque EDT row-major first maximum at coarsebox pixel-center with numpy rint ties-even. Sham same donor/alpha/dimensions moved toward horizontal imagecenter by transformedW+regionW+4. Reject fullXoutofbounds; sameYclipping allowed onlyequalpositivealpha mass. No search/rescale/retry.',
        opacity='Every integer pixel in supported half-open coarsebbox must have coverhardalpha1; sham overlap0. Box is coarse weak material-end neighborhood, not exact material ownershipmask or whole nativepart.',
        input_math='All8 original q/L/native continuous availability and separate construct/nearest/central4/context masks preserved. Exact original crop and context reconstruction; pure C8/C9 preprocess actual17FP32 views computed without encoder/weights. Changed rawRGB alone insufficient.',
        unlearnable='If COVER or SHAM leaves all selected target/local/central preprocessed views byte-identical toBEFORE, retainproof failure UNLEARNABLE_OR_FAILED_UNKNOWN. No target/admission.',
        center_owner='Literalcenter4 rawbilinearfootprint+sampledalpha/RGBchanges recorded separately. Neither qowner invariance norqonpart is a selection/admissiongate.',
        before='All8 saved RGB96/valid/cropconstruct/globalindex parity; all8 inputfact arrayhashes, originaltarget224/valid/fractions/central4/context hashes exact. Anyfailure retained no retry.',
        new_targets='native_visibility_target/point_quality_target/owner_target all8 allvariants -1; includes selectedanchor. No labels admitted inrenderphase.',
        output_schema=dict(manifest='ALL66_PART_PIXEL_INPUT_PROOFS.jsonl',per_pair='<pair_id>/PROOF.json',
            variants='<pair_id>/{BEFORE,COVER,ADJACENT_SHAM}_INPUTS.npz',
            NPZ_keys=['full_RGB','rgb96','valid96','central32','target224','target_valid','fractions','query_xy','L','native_slot','query_id','crop_constructed','native_continuous_valid','nearest_valid','central4_available','context_available','query_available','native_visibility_target','point_quality_target','owner_target','pasted_alpha','original_no_detection','original_finite_non_sentinel','original_query_RGB96_centre_domain_valid','original_native_grid_any_valid'],
            unknown='All66 metadata rows retained; UNKNOWNregion no guessedROI or constructedvariant; exception/prooffailure retained, no replacement.'),
        actual_source_donor_pixels_or_NPZ_bodies_read_during_preparation=0,encoder_network_weight_feature_initializer_fit_PnP_calls=0,
        runtime=read(generated)['runtime'],actual_pure_preprocess_device='cuda',training_admitted=False,automatic_retry=False,
        scope_limits='MR EuroWOODdeck regions only; native7runner region absent/UNKNOWN and native6n2 broader witnesslimit; no SQUARE/fullnative8solution claim.')
    write(PROTOCOL,p)
    template=H/'ROOT_ACTUAL_PIXEL_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_native_part_pixel_generation=False,code=p['code'],protocol=bind(PROTOCOL),design=p['design'],
        inputs=p['inputs'],generated_checks=p['generated_checks'],region_complete=p['region_complete'],
        donor_proof_complete=p['donor_proof_complete'],source_only=True,training_allowed=False,encoder_allowed=False,
        model_allowed=False,initializer_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,automatic_retry=False,output=str(OUT)))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),
        generated=bind(generated),inputs=bind(INPUTS),entry_template=bind(template),actual_source_donor_pixel_or_NPZ_bodies_read=0,
        encoder_network_weight_feature_initializer_fit_PnP_calls=0))
    print(json.dumps(dict(completed=True,prepared_only=True,preparation=bind(H/'PREPARATION_COMPLETE.json'),code=p['code'],protocol=bind(PROTOCOL))))
def run(entry_path):
    e=read(entry_path);p=read(PROTOCOL)
    assert e['approved_actual_native_part_pixel_generation'] is True and e['source_only'] is True
    for key in ('training_allowed','encoder_allowed','model_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','automatic_retry'):assert e[key] is False
    for key in ('code','design','inputs','generated_checks','region_complete','donor_proof_complete'):assert e[key]==p[key];check(e[key])
    assert e['protocol']==bind(PROTOCOL) and e['output']==str(OUT)
    assert p['code']==bind(__file__) and read(check(p['generated_checks']))['PASS']
    for key in ('region_pilot','blinded_input_bindings','inputfacts','original_queries','donor_proof_records','safe_helper','local_crop','context_crop','C8_pure_RGB_preprocess','C9_pure_RGB_preprocess'):check(p[key])
    assert np.__version__=='1.26.4' and torch.__version__=='2.1.1+cu118' and sys.executable==p['runtime']['executable']
    assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=read(check(p['inputs']));frames={r['frame_index']:r for r in inputs['frames']}
    op=operators();OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',dict(entry=bind(entry_path),protocol=bind(PROTOCOL),attempts=66,automatic_retry=False))
    results=[];fcache={};dcache={};started=time.monotonic()
    try:
        for a in inputs['attempts']:
            r=dict(pair_id=a['pair_id'],case_id=a['case_id'],frame_index=a['frame_index'],native_slot=a['native_slot'],
                donor_id=a['donor_id'],region=a['region'],source_only=True,PASS=False,constructed=False,
                variants={},training_admitted=False,all_new_native_quality_owner_targets=-1)
            folder=OUT/a['pair_id'];folder.mkdir()
            if a['region']['region_status']=='UNKNOWN':
                r['reason']='REGION_UNKNOWN_NO_GUESSED_ROI_NO_CONSTRUCTED_IMAGES'
            else:
                try:
                    fr=frames[a['frame_index']];k=a['native_slot'];reg=a['region'];box=reg['original_fullRGB_bbox_xyxy']
                    assert reg['query_xy']==fr['query_xy'][k] and reg['L']==fr['L'][k] and reg['query_id']==fr['query_id'][k]
                    i=a['frame_index']
                    if i not in fcache:
                        try:
                            rgb=np.load(check(fr['canonical_RGB_NPY']),allow_pickle=False)
                            assert rgb.dtype==np.uint8 and list(rgb.shape)==fr['raw_hw']+[3] and ash(rgb)==fr['decoded_RGB_sha256']
                            with np.load(check(fr['existing_factual96_packet']),allow_pickle=False) as z:saved={key:z[key].copy() for key in ('rgb96','valid96','crop_constructed','global_index')}
                            before=materialize(rgb,fr,op)
                            anchor=dict(native_slot=k,existing_RGB96_array_sha256=fr['facts'][k]['RGB96_HWC_sha256'])
                            parity=op['verify_before'](before,fr,anchor,saved);assert all(parity.values()),'ORIGINAL_BEFORE_PARITY_FAILURE'
                            fcache[i]=(rgb,before,parity)
                        except Exception:fcache[i]=traceback.format_exc()
                    if isinstance(fcache[i],str):raise RuntimeError('CACHED_SOURCE_FAILURE_NO_RETRY: '+fcache[i])
                    rgb,before,parity=fcache[i];r['before_parity']=parity
                    # Selected-anchor hash independently checked even if frame was first loaded for another native.
                    assert ash(before['rgb96'][k])==fr['facts'][k]['RGB96_HWC_sha256']
                    bp=folder/'BEFORE_INPUTS.npz';np.savez_compressed(bp,**before,pasted_alpha=np.zeros(rgb.shape[:2],bool));r['variants']['BEFORE']=bind(bp)
                    did=a['donor_id']
                    if did not in dcache:
                        try:
                            db=a['donor'];assert db['split']=='TRAIN'
                            with np.load(check(db['compact_RGBA_cutout']),allow_pickle=False) as z:drgb=z['rgb'].copy();da=z['alpha'].copy()
                            assert drgb.dtype==np.uint8 and da.dtype==bool and drgb.shape==(*da.shape,3)
                            assert ash(drgb)==db['array_proof']['rgb']['array_sha256'] and ash(da)==db['array_proof']['alpha']['array_sha256']
                            dcache[did]=(drgb,da)
                        except Exception:dcache[did]=traceback.format_exc()
                    if isinstance(dcache[did],str):raise RuntimeError('CACHED_DONOR_FAILURE_NO_RETRY: '+dcache[did])
                    drgb,da=dcache[did];longside,geo=region_geometry(box,da,rgb.shape[:2],op);r['scale_geometry']=geo
                    assert longside is not None,geo['reason']
                    cut,alpha=op['transformed_cutout'](drgb,da,longside);tp=folder/'TRANSFORMED_DONOR.npz';np.savez_compressed(tp,rgb=cut,alpha=alpha);r['transformed_donor']=bind(tp)
                    places,g=fixed_placements(box,alpha,rgb.shape[:2],op);r['placement_geometry']=g
                    assert places is not None,g['reason']
                    co,ca=op['paste'](rgb,cut,alpha,places[0]);sh,sa=op['paste'](rgb,cut,alpha,places[1])
                    cover=materialize(co,fr,op);sham=materialize(sh,fr,op)
                    for name,arr,mask in [('COVER',cover,ca),('ADJACENT_SHAM',sham,sa)]:
                        path=folder/(name+'_INPUTS.npz');np.savez_compressed(path,**arr,pasted_alpha=mask);r['variants'][name]=bind(path)
                    proof,detail=prove_pair(before,cover,sham,ca,sa,box,k,op,'cuda')
                    r.update(proof=proof,proof_detail=detail,constructed=True,PASS=all(bool(x) for x in proof.values()))
                    r['reason']='PIXEL_INPUT_PROOF_PASS_NOT_TRAINING_ADMITTED' if r['PASS'] else 'UNLEARNABLE_OR_FAILED_PROOF_UNKNOWN_RETAINED'
                except Exception:
                    r['reason']='ATTEMPT_FAILED_UNKNOWN_RETAINED_NO_RETRY';r['traceback']=traceback.format_exc()
            for name in VARIANTS:
                if name not in r['variants']:r['variants'][name]=dict(constructed=False,reason=r['reason'])
            write(folder/'PROOF.json',r);results.append(r)
            if len(results)%6==0:print(json.dumps(dict(attempts=len(results),passed=sum(z['PASS'] for z in results))),flush=True)
        manifest=OUT/'ALL66_PART_PIXEL_INPUT_PROOFS.jsonl'
        with manifest.open('x') as f:
            for r in results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        unknown=sum(r['region']['region_status']=='UNKNOWN' for r in results)
        complete=dict(completed=True,PASS=len(results)==66 and all(r['PASS'] for r in results),
            PASS_means_all66_proofs_pass_not_census_completion=True,all66_attempts_retained=len(results)==66,
            fixed_UNKNOWN_region_attempts=unknown,supported_region_attempts=66-unknown,
            supported_region_all_proofs_pass=all(r['PASS'] for r in results if r['region']['region_status']=='SUPPORTED'),
            passed_pixel_input_proofs=sum(r['PASS'] for r in results),failed_or_UNKNOWN_attempts_retained=sum(not r['PASS'] for r in results),
            proofs=bind(manifest),entry=bind(entry_path),protocol=bind(PROTOCOL),inputs=p['inputs'],region_pilot=p['region_pilot'],
            region_complete=p['region_complete'],donor_proof_complete=p['donor_proof_complete'],
            source_only=True,encoder_network_weight_feature_initializer_fit_PnP_calls=0,
            original_qL_native_split_availability_unchanged=True,all_new_native_quality_owner_targets=-1,
            training_admitted=False,original_labels_read_or_changed=False,automatic_retry=False,seconds=time.monotonic()-started)
        write(OUT/'COMPLETE.json',complete);print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,attempts_completed=len(results),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--invented-checks',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare and not args.entry:prepare()
    elif args.invented_checks and not args.entry:print(json.dumps(invented_checks()))
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --prepare; actual pixels require separate root --entry')

