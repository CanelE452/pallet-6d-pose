"""V161 coherent pixel-only proof. --prepare uses metadata and invented arrays.

Actual source reads require the separate root entry and completed donor proof.
No encoder, network preprocessing, model, labels, feature or checkpoint reader.
"""
from pathlib import Path
import argparse, ast, hashlib, json, math, os, sys, time, traceback
import numpy as np
from PIL import Image
import PIL
import scipy
from scipy.ndimage import distance_transform_edt

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
DESIGN=V/'ROOT_SOURCE_GENERATION_DESIGN_BEFORE_PIXELS.json'
DESIGN_SHA='25f38eb0e8b098b681a8bb38e5a3f93086a1a24cdc9fdfa8f3300412382c332b'
ASSIGN=V/'metadata_assignment/ASSIGNMENTS159_BEFORE_PIXELS.jsonl'
ASSIGN_COMPLETE=V/'metadata_assignment/METADATA_ASSIGNMENT_COMPLETE.json'
SELECT=V/'metadata_assignment/FIXED12_VISUAL_SELECTION_BEFORE_PIXELS.json'
DONOR_PROTOCOL=V/'donor_proof_preparation/PROTOCOL_BEFORE_DONOR_PIXEL_OR_NPZ_BODIES.json'
DONOR_COMPLETE=V/'donor_proof_v1/DONOR_PROOF_COMPLETE.json'
MR=P/'query_visibility_mr6d_full_public_source_20261024_v153'
FACTS=MR/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
QUERIES=MR/'actual_full230_initializer_v1/ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'
LOCAL=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
TARGET=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
LOCAL_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
TARGET_SHA='aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json'
INPUTS=H/'SOURCE_FRAME_INPUTS46_BEFORE_PIXELS.json'
OUT=V/'source_pixels_v1'
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    p=Path(b['path']); assert sha(p)==b['sha256'],str(p); return p
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,d):
    with Path(p).open('x') as f: json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b): return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pure(path,names,ns):
    tree=ast.parse(Path(path).read_text()); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure_AST','exec'),ns)
    return ns
def operators():
    assert sha(LOCAL)==LOCAL_SHA and sha(TARGET)==TARGET_SHA
    ns={'np':np};pure(LOCAL,('crop',),ns)
    return pure(TARGET,('valid_box','target_context_grid','sample_target_context'),ns)
def erode1(a):
    a=np.asarray(a,bool); z=np.pad(a,1,constant_values=False)
    return np.logical_and.reduce([z[y:y+a.shape[0],x:x+a.shape[1]] for y in range(3) for x in range(3)])
def transformed_cutout(rgb,alpha,longside):
    assert rgb.dtype==np.uint8 and alpha.dtype==bool and rgb.shape==(*alpha.shape,3)
    h,w=alpha.shape; scale=max(1,int(longside))/max(h,w)
    hh=max(1,int(np.floor(h*scale+.5))); ww=max(1,int(np.floor(w*scale+.5)))
    r=np.array(Image.fromarray(rgb).resize((ww,hh),Image.Resampling.BILINEAR),copy=True)
    a=np.array(Image.fromarray(alpha).resize((ww,hh),Image.Resampling.NEAREST),dtype=bool)
    return r,erode1(a)
def opaque_anchor(alpha):
    # Explicit surrounding zero cells make the all-True mask case well defined.
    d=distance_transform_edt(np.pad(alpha,1,constant_values=False))[1:-1,1:-1]
    if not alpha.any(): return None
    y,x=np.unravel_index(int(np.argmax(d)),d.shape)
    return np.array([x,y],np.int64)
def placements(q,L,alpha,hw):
    anchor=opaque_anchor(alpha)
    if anchor is None:return None,{'reason':'TRANSFORMED_ALPHA_EMPTY'}
    cover=np.rint(np.asarray(q,np.float64)-anchor).astype(np.int64)
    h,w=hw; ch,cw=alpha.shape
    direction=1 if q[0] <= (w-1)/2 else -1
    delta=cw+math.ceil(L/96)+4
    sham=cover+np.array([direction*delta,0],np.int64)
    facts={'opaque_anchor_xy':anchor.tolist(),'cover_origin_xy':cover.tolist(),'sham_origin_xy':sham.tolist(),'horizontal_direction':direction,'sham_shift_pixels':delta,'transformed_hw':[ch,cw]}
    if not all(0 <= int(xy[0]) and int(xy[0])+cw <= w for xy in (cover,sham)):
        return None,{**facts,'reason':'X_BOX_NOT_FULLY_IN_BOUNDS'}
    return (cover,sham),facts
def paste(rgb,cut_rgb,cut_alpha,origin):
    h,w=rgb.shape[:2];ch,cw=cut_alpha.shape;x,y=map(int,origin)
    assert 0<=x and x+cw<=w
    lo=max(0,y);hi=min(h,y+ch); result=rgb.copy();mask=np.zeros((h,w),bool)
    if hi>lo:
        a=cut_alpha[lo-y:hi-y];r=cut_rgb[lo-y:hi-y]
        view=result[lo:hi,x:x+cw];view[a]=r[a];mask[lo:hi,x:x+cw]=a
    return result,mask
def raw_footprint(q,L,hw,indices=(47,48)):
    """Independent enumeration: only positive bilinear weights contribute."""
    h,w=hw;points=[];inside=True
    for j in indices:
        yy=float(q[1])+((float(j)+.5)/96-.5)*float(L)
        for i in indices:
            xx=float(q[0])+((float(i)+.5)/96-.5)*float(L)
            x0=math.floor(xx);y0=math.floor(yy);dx=xx-x0;dy=yy-y0
            for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
                if weight>0:
                    x,y=x0+ox,y0+oy;inside=inside and 0<=x<w and 0<=y<h
                    points.append((int(x),int(y)))
    return sorted(set(points),key=lambda z:(z[1],z[0])),bool(inside)
def materialize(rgb,frame,op):
    q=np.array(frame['query_xy'],np.float64);L=np.array(frame['L'],np.float64)
    construct=np.array(frame['feature_constructed'],bool)
    local=np.zeros((8,3,96,96),np.uint8);valid=np.zeros((8,1,96,96),bool)
    for k in range(8):
        if construct[k]:local[k],valid[k]=op['crop'](rgb,q[k],L[k],96)
    box=frame['context_box_xyxy']
    if op['valid_box'](box): target,tvalid=op['sample_target_context'](rgb,box)
    else:target=np.zeros((224,224,3),np.float32);tvalid=np.zeros((224,224),bool)
    fractions=np.array([[np.float32(tvalid.mean()),np.float32(valid[k].mean()),np.float32(valid[k,:,32:64,32:64].mean())] for k in range(8)],np.float32)
    result={'full_RGB':rgb,'rgb96':np.ascontiguousarray(local.transpose(0,2,3,1)),'valid96':valid[:,0].copy(),'central32':np.ascontiguousarray(local[:,:,32:64,32:64].transpose(0,2,3,1)),'target224':target,'target_valid':tvalid,'fractions':fractions,'query_xy':q,'L':L,'native_slot':np.array(frame['native_slot'],np.int8),'crop_constructed':construct,'native_continuous_valid':np.array(frame['native_continuous_valid'],bool),'nearest_valid':np.array(frame['nearest_valid'],bool),'central4_available':valid[:,:,47:49,47:49].all((1,2,3)),'context_available':np.full(8,tvalid.any(),bool),'native_visibility_target':np.full(8,-1,np.int8),'point_quality_target':np.full(8,-1,np.int8)}
    if 'original_query_input_metadata' in frame:
        for key in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):
            result['original_'+key]=np.array([r[key] for r in frame['original_query_input_metadata']],bool)
    if 'query_id' in frame:result['query_id']=np.array(frame['query_id'])
    return result
def verify_before(arr,frame,anchor,saved):
    checks={'all8_RGB96_equal_saved_packet':exact(arr['rgb96'],saved['rgb96']),'all8_valid96_equal_saved_packet':exact(arr['valid96'],saved['valid96']),'all8_construct_equal_saved_packet':exact(arr['crop_constructed'],saved['crop_constructed']),'saved_global_indices_equal':np.array_equal(saved['global_index'],frame['global_index'])}
    checks['all8_RGB96_fact_hash']=all(ash(arr['rgb96'][k])==frame['facts'][k]['RGB96_HWC_sha256'] for k in range(8))
    checks['all8_valid96_fact_hash']=all(ash(arr['valid96'][k])==frame['facts'][k]['valid96_sha256'] for k in range(8))
    checks['target224_original_hash']=all(ash(arr['target224'])==f['target224_F32_HWC_sha256'] for f in frame['facts'])
    checks['target_valid_original_hash']=all(ash(arr['target_valid'])==f['target_valid_sha256'] for f in frame['facts'])
    checks['all8_fractions_original_exact']=exact(arr['fractions'],np.array([f['fractions'] for f in frame['facts']],np.float32))
    checks['selected_RGB96_anchor_hash']=ash(arr['rgb96'][anchor['native_slot']])==anchor['existing_RGB96_array_sha256']
    checks['all8_center4_original_equal']=np.array_equal(arr['central4_available'],frame['local_central4_available'])
    checks['all8_context_original_equal']=np.array_equal(arr['context_available'],frame['context_available'])
    return checks
def center_proof(before,cover,sham,ca,sa,q,L,k,op):
    pts,inside=raw_footprint(q,L,ca.shape)
    c=op['crop'](np.repeat(ca[:,:,None],3,2).astype(np.uint8)*255,q,L,96)[0]
    s=op['crop'](np.repeat(sa[:,:,None],3,2).astype(np.uint8)*255,q,L,96)[0]
    checks={'center4_footprint_inside':inside,'all_contributing_raw_cover_alpha1':inside and all(ca[y,x] for x,y in pts),'all_contributing_raw_sham_alpha0':inside and all(not sa[y,x] for x,y in pts),'frozen_sampled_cover_alpha_center4_255':bool((c[:,47:49,47:49]==255).all()),'frozen_sampled_sham_alpha_center4_0':bool((s[:,47:49,47:49]==0).all()),'sham_center4_RGB_bit_equal_before':exact(sham['rgb96'][k,47:49,47:49],before['rgb96'][k,47:49,47:49]),'sham_some_selected_local_donor_support':bool(s.any()),'cover_sham_equal_pasted_alpha_mass':int(ca.sum())==int(sa.sum()),'positive_pasted_mass':int(ca.sum())>0}
    for label,z in (('cover',cover),('sham',sham)):
        for key in ('valid96','target_valid','crop_constructed','native_continuous_valid','nearest_valid','central4_available','context_available','fractions','query_xy','L','native_slot'):
            checks[label+'_'+key+'_unchanged']=exact(z[key],before[key])
        for key in before:
            if key.startswith('original_') or key=='query_id':checks[label+'_'+key+'_unchanged']=exact(z[key],before[key])
    return checks,{'raw_footprint_xy':pts,'raw_footprint_count':len(pts),'cover_pasted_alpha_mass':int(ca.sum()),'sham_pasted_alpha_mass':int(sa.sum()),'cover_sampled_alpha96_sha256':ash(c),'sham_sampled_alpha96_sha256':ash(s),'sham_local_sampled_alpha_positive_cells':int((s[0]>0).sum())}

def frame_metadata(assignments):
    facts=lines(FACTS);queries=lines(QUERIES);assert len(facts)==len(queries)==1840
    frames=[]
    for i in sorted({r['frame_index'] for r in assignments}):
        ff=[r for r in facts if r['frame_index']==i];qq=[r for r in queries if r['frame_index']==i]
        assert len(ff)==len(qq)==8 and [r['native_slot'] for r in ff]==[r['native_slot'] for r in qq]==list(range(8))
        aa=[r['anchor'] for r in assignments if r['frame_index']==i];a=aa[0]
        for k,(f,q) in enumerate(zip(ff,qq)):
            assert f['query_id']==q['query_id']==a['frame_id']+':native'+str(k)
            for key in ('query_xy','L','frame_id','source_id','split','native_slot'):assert f[key]==q[key]
            assert f['split']=='SOURCE_DEVELOPMENT' and f['native_continuous_valid']==q['native_continuous_valid'] and f['nearest_valid']==q['query_nearest_pixel_valid']
            assert f['context_box_xyxy']==ff[0]['context_box_xyxy']
        fr={key:[r[key] for r in ff] for key in ('global_index','query_id','native_slot','query_xy','L','feature_constructed','native_continuous_valid','nearest_valid','local_central4_available','context_available')}
        fr.update(frame_index=i,frame_id=a['frame_id'],source_id=a['source_id'],split=a['split'],raw_hw=a['raw_hw'],canonical_RGB_NPY=a['canonical_RGB_NPY'],decoded_RGB_sha256=a['decoded_RGB_sha256'],existing_factual96_packet=a['existing_factual96_packet'],context_box_xyxy=ff[0]['context_box_xyxy'],facts=ff,original_query_input_metadata=qq)
        for x in aa:
            k=x['native_slot'];assert x['query_xy']==fr['query_xy'][k] and x['L']==fr['L'][k] and x['existing_RGB96_array_sha256']==ff[k]['RGB96_HWC_sha256']
        frames.append(fr)
    assert len(frames)==46
    return frames

def invented_checks():
    op=operators();checks={}
    rgb=np.arange(160*320*3,dtype=np.uint8).reshape(160,320,3)
    q8=np.array([[90.25,80.75],[110.,80.],[110.,100.],[90.,100.],[92.,82.],[108.,82.],[108.,98.],[92.,98.]])
    frame=dict(query_xy=q8.tolist(),L=[60.]*8,feature_constructed=[True]*8,context_box_xyxy=[90.25,80.,110.,100.],native_slot=list(range(8)),native_continuous_valid=[True]*8,nearest_valid=[True]*8)
    before=materialize(rgb,frame,op);drgb=np.full((35,45,3),[91,131,201],np.uint8);alpha=np.ones((35,45),bool)
    cr,aa=transformed_cutout(drgb,alpha,21);places,pf=placements(q8[0],60.,aa,rgb.shape[:2]);assert places is not None
    co,ca=paste(rgb,cr,aa,places[0]);sh,sa=paste(rgb,cr,aa,places[1]);cover=materialize(co,frame,op);sham=materialize(sh,frame,op)
    proof,detail=center_proof(before,cover,sham,ca,sa,q8[0],60.,0,op)
    checks.update(proof);checks['exact_center32_direct_slice']=exact(before['central32'],before['rgb96'][:,32:64,32:64])
    checks['cover_and_sham_from_one_fullRGB_each']=exact(cover['full_RGB'],co) and exact(sham['full_RGB'],sh)
    checks['new_native_quality_all8_UNKNOWN']=all((z['native_visibility_target']==-1).all() and (z['point_quality_target']==-1).all() for z in (before,cover,sham))
    checks['empty_alpha_retains_failure']=placements(q8[0],60.,np.zeros((3,3),bool),rgb.shape[:2])[0] is None
    checks['horizontal_geometry_failure_retained']=placements([1.,60.],60.,np.ones((20,30),bool),rgb.shape[:2])[0] is None
    checks['erosion_zero_outside']=int(erode1(np.ones((5,5),bool)).sum())==9
    checks['row_major_first_EDT_tie']=opaque_anchor(np.ones((4,4),bool)).tolist()==[1,1]
    checks['rint_ties_even']=np.rint([10.5,11.5]).tolist()==[10.,12.]
    intpoints,intvalid=raw_footprint([100.,100.],192.,rgb.shape[:2])
    checks['zero_weight_neighbors_excluded_from_footprint']=intvalid and len(intpoints)==4
    holed=np.ones((9,9),bool);holed[4,4]=False
    _,ha=transformed_cutout(np.full((9,9,3),99,np.uint8),holed,9)
    checks['donor_hole_not_filled']=not ha[4,4] and not ha[3:6,3:6].any()
    invalid_frame={**frame,'query_xy':q8.tolist(),'feature_constructed':[False]+[True]*7,'native_continuous_valid':[False]+[True]*7,'nearest_valid':[False]+[True]*7}
    invalid_frame['query_xy'][0]=[-1.,-1.]
    invalid_variants=[materialize(z,invalid_frame,op) for z in (rgb,co,sh)]
    checks['all8_retained_with_original_inputU']=all(z['rgb96'].shape==(8,96,96,3) and not z['rgb96'][0].any() and not z['valid96'][0].any() for z in invalid_variants)
    checks['no_original_inputU_resurrection']=all(not z['crop_constructed'][0] and not z['native_continuous_valid'][0] and not z['nearest_valid'][0] for z in invalid_variants)
    # Same vertical clipping is allowed; mass must remain equal.
    cv,cm=paste(rgb,cr,aa,[50,-4]);sv,sm=paste(rgb,cr,aa,[100,-4]);checks['vertical_clipping_equal_mass']=int(cm.sum())==int(sm.sum())>0
    # Break one contributing cell: proof must detect loss of opacity.
    broken=ca.copy();x,y=detail['raw_footprint_xy'][0];broken[y,x]=False
    bproof,_=center_proof(before,cover,sham,broken,sa,q8[0],60.,0,op)
    checks['independent_footprint_catches_one_missing_alpha_cell']=not bproof['all_contributing_raw_cover_alpha1']
    changed=materialize(sh.copy(),frame,op);changed['rgb96'][0,47,47,0]^=1
    bad,_=center_proof(before,cover,changed,ca,sa,q8[0],60.,0,op)
    checks['sham_center_RGB_change_detected']=not bad['sham_center4_RGB_bit_equal_before']
    assert all(bool(v) for v in checks.values()),[k for k,v in checks.items() if not v]
    return dict(completed=True,PASS=True,invented_arrays_only=True,checks=checks,number_checks=len(checks),source_pixel_crop_donor_feature_checkpoint_bodies_read=0,learned_calls=0,runtime={'executable':sys.executable,'numpy':np.__version__,'Pillow':PIL.__version__,'scipy':scipy.__version__})

def prepare():
    assert sha(DESIGN)==DESIGN_SHA
    design=read(DESIGN);assert design['training_allowed'] is False
    assignments=lines(ASSIGN);assert len(assignments)==159 and len({r['anchor_query_id'] for r in assignments})==53
    frames=frame_metadata(assignments);write(INPUTS,dict(completed=True,frames=frames,source_pixel_bodies_read=False,all46_TRAIN=True,feature_inputfacts=bind(FACTS),original_queries=bind(QUERIES)))
    generated=H/'GENERATED_INVENTED_PIXEL_CHECKS.json';write(generated,invented_checks())
    dc=read(DONOR_COMPLETE);assert dc['completed'] and dc['PASS'] and dc['attempted']==dc['passed']==59 and dc['failed']==0
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),design=bind(DESIGN),assignment_complete=bind(ASSIGN_COMPLETE),assignments=bind(ASSIGN),fixed12=bind(SELECT),frame_inputs=bind(INPUTS),generated=bind(generated),local_crop_code=bind(LOCAL),target_crop_code=bind(TARGET),inputfacts=bind(FACTS),original_queries=bind(QUERIES),donor_proof_protocol=bind(DONOR_PROTOCOL),donor_proof_complete=bind(DONOR_COMPLETE),donor_proof_records=dc['records'],runtime={'executable':sys.executable,'numpy':np.__version__,'Pillow':PIL.__version__,'scipy':scipy.__version__},output=str(OUT),actual_source_pixel_body_reads=0,encoder_model_fit_PnP_calls=0,context_box='INHERITED_ORIGINAL_Q8_UNCLIPPED_MINMAX_FROM_V153_INPUTFACTS',resize='Pillow RGB BILINEAR/alpha NEAREST; dimensions floor(dim*longside/maxHW+.5), max1; bool erode1 3x3 with false outside after resize',anchor='Euclidean scipy distance_transform_edt with explicit zero pad; row-major first argmax',center_proof='Frozen crop sampled alpha central47:49 plus independent positive bilinear raw contributing footprint; sham alpha0 and RGB bit equal',sham_direction='toward horizontal pixel-coordinate midpoint (W-1)/2; exact midpoint chooses +x',donor_proof_required_before_construction=True,donor_proof_contract={'complete_path':str(DONOR_COMPLETE),'records_path':str(V/'donor_proof_v1/PROVED_TRAIN59_DONORS.jsonl'),'root_entry_must_bind_complete':True},all159_attempts_retained=True,no_replacement_no_retry=True,new_native_visibility_and_quality='UNKNOWN -1 all8 allvariants; original labels not loaded',owner_auxiliary='Proof eligibility only; original53 TARGET weak owner is lineage; no training admission',output_schema={'proof_manifest':'ALL159_PIXEL_INPUT_PROOFS.jsonl','per_pair':'<pair_id>/PROOF.json','variant_NPZ':'<pair_id>/{BEFORE,COVER,ADJACENT_SHAM}_INPUTS.npz','variant_keys':['full_RGB','rgb96','valid96','central32','target224','target_valid','fractions','query_xy','L','native_slot','query_id','crop_constructed','native_continuous_valid','nearest_valid','central4_available','context_available','native_visibility_target','point_quality_target','pasted_alpha','original_no_detection','original_finite_non_sentinel','original_query_RGB96_centre_domain_valid','original_native_grid_any_valid'],'failed_variants':'Missing construction explicitly retained as constructed:false, no retry'},training_admitted=False,automatic_retry=False)
    write(PROTOCOL,protocol)
    template=H/'ROOT_ACTUAL_PIXEL_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_source_pixel_generation=False,code=protocol['code'],protocol=bind(PROTOCOL),design=protocol['design'],assignments=protocol['assignments'],frame_inputs=protocol['frame_inputs'],generated=protocol['generated'],donor_proof_complete=protocol['donor_proof_complete'],source_only=True,training_allowed=False,encoder_allowed=False,model_allowed=False,initializer_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,automatic_retry=False,output=str(OUT)))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(generated),frame_inputs=bind(INPUTS),entry_template=bind(template),source_pixel_body_reads=0,learned_calls=0,actual_pixel_generation=False,donor_proof_actual_pending=False,donor_proof_complete=bind(DONOR_COMPLETE),training_admitted=False))
    print(json.dumps({'prepared_only':True,'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(generated)}),flush=True)

def run(entry_path):
    e=read(entry_path);assert e['approved_actual_source_pixel_generation'] is True and e['source_only'] is True
    for key in ('training_allowed','encoder_allowed','model_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','automatic_retry'):assert e[key] is False
    p=read(check(e['protocol']));assert e['protocol']==bind(PROTOCOL)
    for key in ('code','design','assignments','frame_inputs','generated'):assert e[key]==p[key];check(p[key])
    for key in ('assignment_complete','fixed12','inputfacts','original_queries','local_crop_code','target_crop_code','donor_proof_protocol','donor_proof_records'):check(p[key])
    assert read(check(p['generated']))['PASS']
    assert p['code']==bind(__file__) and sha(DESIGN)==DESIGN_SHA and not OUT.exists() and e['output']==p['output']==str(OUT)
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve() and np.__version__==p['runtime']['numpy'] and PIL.__version__==p['runtime']['Pillow'] and scipy.__version__==p['runtime']['scipy']
    assert e['donor_proof_complete']==p['donor_proof_complete']
    dc=read(check(e['donor_proof_complete']));assert dc['completed'] and dc['PASS'] and dc['attempted']==dc['passed']==59 and dc['failed']==0, 'All59 donor source proof required before construction'
    donors=lines(check(p['donor_proof_records']));assert len(donors)==59
    # Complete must bind the exact donor records; schema finalized with proof producer.
    donor_records_binding=bind(V/'donor_proof_v1/PROVED_TRAIN59_DONORS.jsonl')
    assert dc['records']==donor_records_binding==p['donor_proof_records'], 'Donor COMPLETE must bind exact proved records'
    donor_map={r.get('donor_id',r.get('id')):r for r in donors};assert len(donor_map)==59
    assignments=lines(check(p['assignments']));assert len(assignments)==159
    frames={r['frame_index']:r for r in read(check(p['frame_inputs']))['frames']};assert len(frames)==46
    op=operators();OUT.mkdir();write(OUT/'STARTED.json',dict(completed=False,pid=os.getpid(),entry=bind(entry_path),protocol=bind(PROTOCOL),donor_proof_complete=e['donor_proof_complete'],automatic_retry=False))
    results=[];started=time.monotonic()
    try:
        for a in assignments:
            folder=OUT/a['pair_id'];folder.mkdir();r={'pair_id':a['pair_id'],'pair_sha256':a['pair_sha256'],'assignment_index':a['assignment_index'],'anchor_query_id':a['anchor_query_id'],'donor_id':a['donor_id'],'original_q':a['query_xy'],'original_L':a['L'],'native_slot':a['native_slot'],'split':a['split'],'PASS':False,'attempted_once':True,'replacement_or_retry':False,'training_admitted':False,'new_native_visibility_target':[-1]*8,'new_point_quality_target':[-1]*8,'variants':{},'proof':{}}
            try:
                fr=frames[a['frame_index']];assert a['split']=='SOURCE_DEVELOPMENT'
                rgb=np.load(check(fr['canonical_RGB_NPY']),allow_pickle=False)
                assert rgb.dtype==np.uint8 and list(rgb.shape)==fr['raw_hw']+[3] and ash(rgb)==fr['decoded_RGB_sha256']
                with np.load(check(fr['existing_factual96_packet']),allow_pickle=False) as z:saved={k:z[k].copy() for k in ('rgb96','valid96','crop_constructed','global_index')}
                before=materialize(rgb,fr,op);parity=verify_before(before,fr,a['anchor'],saved);r['before_parity']=parity
                bpath=folder/'BEFORE_INPUTS.npz';np.savez_compressed(bpath,**before,pasted_alpha=np.zeros(rgb.shape[:2],bool));r['variants']['BEFORE']=bind(bpath)
                assert all(parity.values()),'BEFORE parity failed; no replacement'
                donor=donor_map[a['donor_id']];assert donor['PASS'] and donor['split']=='TRAIN'
                cb=a['donor']['compact_RGBA_cutout'];assert donor['compact_RGBA_cutout']==cb
                with np.load(check(cb),allow_pickle=False) as z:drgb=z['rgb'].copy();da=z['alpha'].copy()
                assert drgb.dtype==np.uint8 and da.dtype==bool and drgb.shape==(*da.shape,3)
                assert ash(drgb)==donor['array_proof']['rgb']['array_sha256'] and list(drgb.shape)==donor['array_proof']['rgb']['shape']
                assert ash(da)==donor['array_proof']['alpha']['array_sha256'] and list(da.shape)==donor['array_proof']['alpha']['shape']
                expected=max(1,min(math.ceil(a['scale']*a['L']),math.floor(min(a['query_xy'][0]-4,fr['raw_hw'][1]-5-a['query_xy'][0])),math.floor((fr['raw_hw'][1]-12)/4)))
                assert expected==a['longside_pixels']
                cr,alpha=transformed_cutout(drgb,da,a['longside_pixels']);np.savez_compressed(folder/'TRANSFORMED_DONOR.npz',rgb=cr,alpha=alpha)
                places,geometry=placements(a['query_xy'],a['L'],alpha,fr['raw_hw']);r['geometry']=geometry
                assert places is not None,geometry['reason']
                co,ca=paste(rgb,cr,alpha,places[0]);sh,sa=paste(rgb,cr,alpha,places[1])
                cover=materialize(co,fr,op);sham=materialize(sh,fr,op)
                for name,arr,mask in (('COVER',cover,ca),('ADJACENT_SHAM',sham,sa)):
                    path=folder/(name+'_INPUTS.npz');np.savez_compressed(path,**arr,pasted_alpha=mask);r['variants'][name]=bind(path)
                proof,detail=center_proof(before,cover,sham,ca,sa,a['query_xy'],a['L'],a['native_slot'],op);r['proof']=proof;r['proof_detail']=detail
                r['PASS']=all(bool(x) for x in proof.values());r['reason']='PIXEL_INPUT_PROOF_PASS' if r['PASS'] else 'PIXEL_INPUT_PROOF_FAILURE_RETAINED'
                r['weak_owner_auxiliary_eligibility_only']=r['PASS']
            except Exception:
                r['reason']='ATTEMPT_FAILED_RETAINED';r['traceback']=traceback.format_exc();r['weak_owner_auxiliary_eligibility_only']=False
            for name in VARIANTS:
                if name not in r['variants']:r['variants'][name]={'constructed':False,'reason':'FAILURE_RETAINED_NO_RETRY'}
            write(folder/'PROOF.json',r);results.append(r)
            write(OUT/('PROGRESS_%03d.json'%len(results)),dict(attempts=len(results),passed=sum(x['PASS'] for x in results),failed=sum(not x['PASS'] for x in results),elapsed_seconds=time.monotonic()-started))
            if len(results)%10==0:print(json.dumps(dict(attempts=len(results),passed=sum(x['PASS'] for x in results))),flush=True)
        path=OUT/'ALL159_PIXEL_INPUT_PROOFS.jsonl'
        with path.open('x') as f:
            for r in results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=len(results)==159 and all(r['PASS'] for r in results),completion_PASS_means_all159_proofs_pass=True,all159_attempts_retained=len(results)==159,passed_pixel_input_proofs=sum(r['PASS'] for r in results),failed_attempts_retained=sum(not r['PASS'] for r in results),all_proofs_pass=all(r['PASS'] for r in results),proofs=bind(path),entry=bind(entry_path),protocol=bind(PROTOCOL),assignments=p['assignments'],fixed12=p['fixed12'],frame_inputs=p['frame_inputs'],donor_proof_complete=e['donor_proof_complete'],source_only=True,encoder_model_fit_PnP_calls=0,original_qL_native_split_availability_unchanged=True,all_new_native_visibility_and_point_quality_UNKNOWN=True,original_labels_read_or_changed=False,training_admitted=False,automatic_retry=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,attempts_completed=len(results),traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--invented-checks',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare and not args.entry:prepare()
    elif args.invented_checks and not args.entry:print(json.dumps(invented_checks()))
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --prepare; actual source pixels require separate root --entry')
