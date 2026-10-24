"""V172 saved original mask/depth measurements; prepare uses invented pixels only."""
from pathlib import Path
import argparse,hashlib,json,sys,time,traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
READY=V/'INPUT_PATH_READINESS.json';READY_SHA='e0b3b8eacc12698103f1aee34fd914973adc83759db960335f2bc79576f16a2c'
SCOPE=V/'ROOT_SCOPE_BEFORE_SOURCE_MASK_OR_DEPTH_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_MASK_DEPTH_BODIES.json';TESTS=H/'GENERATED_INVENTED_PIXEL_CHECKS.json'
OUT=V/'saved_mask_semantics_measurement_v1'
CLASSES=('INCIDENT_TARGET','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v169','_v112','/private_evaluation/'))
    assert sha(p)==b['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def categories(owner,ambiguous,incident):
    assert owner.dtype==np.int32 and ambiguous.dtype==bool and owner.shape==ambiguous.shape
    incident=np.asarray(incident,np.int64);assert np.all(incident>=0) and not np.any(incident==999)
    out=np.where(owner<0,3,np.where(owner==999,2,np.where(np.isin(owner,incident),0,1))).astype(np.int8)
    out[ambiguous]=4;return out
def pixel(buf,xy,classes):
    x,y=map(int,xy);h,w=classes.shape
    if not (0<=x<w and 0<=y<h):return dict(x=x,y=y,in_image=False,category='INVALID_SUPPORT',part_id=None,global_triangle_id=None,camera_depth=None,ambiguity=None,tie_count=None)
    depth=float(buf['camera_depth'][y,x])
    return dict(x=x,y=y,in_image=True,category=CLASSES[int(classes[y,x])],part_id=int(buf['part_id'][y,x]),global_triangle_id=int(buf['global_triangle_id'][y,x]),camera_depth=depth if np.isfinite(depth) else None,ambiguity=bool(buf['ambiguity'][y,x]),tie_count=int(buf['tie_count'][y,x]))
def bilinear_weights(classes,points,apply_crop_domain=False):
    points=np.asarray(points,np.float64);assert points.ndim==2 and points.shape[1]==2 and np.isfinite(points).all()
    h,w=classes.shape;x,y=points.T;ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
    weights=np.zeros((len(points),len(CLASSES)),np.float64)
    for ax,ay,ww in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=ix+ax,iy+ay;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);cls=np.full(len(points),5,np.int8);cls[inside]=classes[yy[inside],xx[inside]]
        weights[np.arange(len(points)),cls]+=ww
    crop_valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    if apply_crop_domain:weights[~crop_valid]=0.;weights[~crop_valid,5]=1.
    assert np.all(weights>=0) and np.allclose(weights.sum(1),1.,rtol=0,atol=2e-15)
    return weights,crop_valid
def distribution(w):
    w=np.asarray(w,np.float64);assert w.shape==(6,) and np.isclose(w.sum(),1.,rtol=0,atol=2e-13)
    return dict(owner5_plus_INVALID={k:float(v) for k,v in zip(CLASSES,w)},basic4_plus_INVALID=dict(TARGET=float(w[0]+w[1]),FOREIGN=float(w[2]),BG=float(w[3]),AMBIGUOUS=float(w[4]),INVALID_SUPPORT=float(w[5])))
def point_measure(buf,classes,q):
    if q is None or len(q)!=2 or not np.isfinite(q).all() or np.array_equal(q,[-1.,-1.]):return dict(status='UNKNOWN_ORIGINAL_QUERY_MISSING',nearest=None,bilinear=None)
    q=np.asarray(q,np.float64);nearest=np.floor(q+.5).astype(np.int64);weight,_=bilinear_weights(classes,q[None]);lo=np.floor(q).astype(np.int64);dx,dy=q-lo
    footprint=[]
    for ax,ay,w in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        footprint.append(dict(weight=float(w),**pixel(buf,lo+[ax,ay],classes)))
    return dict(status='MEASURED_ORIGINAL_COORDINATE',coordinate_xy=q.tolist(),nearest=pixel(buf,nearest,classes),bilinear=distribution(weight[0]),four_source_pixels=footprint,reference_centered_crop_generated=False)
def crop_region(classes,q,L,lo,hi):
    if q is None or L is None or not np.isfinite(q).all() or np.array_equal(q,[-1.,-1.]) or not np.isfinite(L) or L<=0:return dict(status='UNKNOWN_ORIGINAL_Q_OR_L_MISSING',weights=None)
    q=np.asarray(q,np.float64);u=(np.arange(lo,hi,dtype=np.float64)+.5)/96-.5;xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);points=np.column_stack((xx.ravel(),yy.ravel()))
    w,valid=bilinear_weights(classes,points,apply_crop_domain=True)
    result=dict(status='MEASURED_ORIGINAL_Q_L_SAVED96_GRID',saved96_index_range=[lo,hi],sample_count=len(points),crop_sample_domain_valid_fraction=float(valid.mean()),**distribution(w.mean(0)))
    if hi-lo==2:result['sample_points_xy']=points.tolist();result['sample_weights']=[distribution(z) for z in w]
    return result
def invented():
    owner=np.array([[0,999],[7,-1]],np.int32);amb=np.array([[0,1],[0,0]],bool);cat=categories(owner,amb,[0])
    c={};c['ambiguity_priority_over_foreign']=int(cat[0,1])==4
    c['incident_other_background_distinct']=cat.tolist()==[[0,4],[1,3]]
    w,_=bilinear_weights(cat,np.array([[.5,.5]]));c['four_pixel_onehot_equal_weights_no_ID_interpolation']=np.array_equal(w[0],np.array([.25,.25,0,.25,.25,0]))
    w2,_=bilinear_weights(cat,np.array([[0.,0.]]));c['exact_integer_pixel_includes_zero_weight_neighbors']=np.array_equal(w2[0],np.array([1,0,0,0,0,0]))
    w3,_=bilinear_weights(cat,np.array([[-.25,0.]]));c['partial_padding_INVALID_not_BG']=w3[0,5]==.25 and w3[0,0]==.75
    w4,valid=bilinear_weights(cat,np.array([[-.75,0.]]),True);c['frozen_crop_domain_invalid_is_all_INVALID']=not valid[0] and w4[0,5]==1
    large=np.zeros((128,128),np.int8);g4=crop_region(large,[64.,64.],96.,47,49);g32=crop_region(large,[64.,64.],96.,32,64)
    c['central4_saved96_indices47_48_exact']=g4['sample_points_xy']==[[63.5,63.5],[64.5,63.5],[63.5,64.5],[64.5,64.5]]
    c['central32_exact1024_samples_and_unit_mass']=g32['sample_count']==1024 and g32['owner5_plus_INVALID']['INCIDENT_TARGET']==1
    dummy={'part_id':owner,'ambiguity':amb,'camera_depth':np.array([[1.,2.],[3.,np.nan]],np.float64),'global_triangle_id':np.array([[4,5],[6,-1]],np.int32),'tie_count':np.array([[1,2],[1,0]],np.uint16)}
    near=point_measure(dummy,cat,[.5,.5]);c['nearest_floor_q_plus_half']=near['nearest']['x']==1 and near['nearest']['y']==1
    c['background_NaN_depth_preserved_as_null']=near['nearest']['category']=='BG' and near['nearest']['camera_depth'] is None
    c['missing_original_query_preserved_UNKNOWN']=point_measure(dummy,cat,[-1.,-1.])['status'].startswith('UNKNOWN')
    c['reference_point_measurement_creates_no_crop']=not near['reference_centered_crop_generated']
    incident_changed=categories(owner,np.zeros_like(amb),[7]);c['native_incident_set_is_per_slot']=incident_changed[1,0]==0 and incident_changed[0,0]==1
    raw=owner.copy();categories(owner,amb,[0]);c['source_owner_ID_array_unmodified']=exact(raw,owner)
    assert all(c.values());return dict(completed=True,PASS=True,invented_only=True,checks={k:bool(v) for k,v in c.items()},number_checks=len(c),code=bind(__file__),source_mask_depth_RGB_bodies_or_headers_read=0)
def prepare():
    assert sha(READY)==READY_SHA;ready=read(READY);assert ready['PASS'] and ready['actual_measurements']==0
    write(TESTS,invented())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),readiness=bind(READY),selection=ready['source_selection'],tests=bind(TESTS),runtime=dict(executable=sys.executable,numpy=np.__version__),
        fixed_rows=[5353,5586,5609],fixed_scenes=[669,698,701],all8_per_scene=True,total_query_rows=24,
        qL='Each q inherits matching native slot of V171 original_initial_points8, checked against original V132 query metadata q/L; no new coordinates or filters.',
        priority='AMBIGUOUS if ambiguity true, else FOREIGN if part999, else BG if part<0, else INCIDENT_TARGET if part in this native recordCF.incident_part_ids, else OTHER_TARGET.',
        owner_measurement='Float64 bilinear onehot class weights over exact four source pixels; never interpolate integer owner IDs. Keep5 owner classes and INVALID_SUPPORT with total mass1; also retain basic4+INVALID.',
        nearest='floor(q+.5), with original part/triangle/depth/ambiguity/tie; out of image INVALID_SUPPORT, background nonfinite depth null.',
        crop_owner_weights='Original q/L96 sample positions: q+((index+.5)/96-.5)*L. Central4 indices47:49 and central32 indices32:64; average bilinear onehot weights. Outside raw source pixel mass remains INVALID; frozen96 sample domain outside[-.5,w-.5)x[-.5,h-.5) becomes all INVALID, never BG.',
        reference='Nominal original reference only point nearest+bilinear4 diagnostic in both scene and BEFORE; absolutely no reference-centered crop or reference-centered96/32 grid.',
        BEFORE='Same original q/L and incident set measured against saved BEFORE target-only resolved owner/depth, not selected-part-alone hidden surfaces.',
        RGB_correspondence='Selected scene RGB_uint8 decoded array SHA must equal inherited V171 original decoded RGB SHA before owner measurements; no PNG read/new rendering.',
        metadata_parity='Check saved state8_CF/native8_CF_xy/native8_canonical_xy[perm]/perm/camera against exact JSON metadata and recordsCF; original labels and inputU flags retained.',
        output=str(OUT),no_new_labels_training_admission_or_threshold_selection=True,actual_source_bodies_read_during_preparation=0,automatic_retry=False)
    write(PROTOCOL,p);write(H/'ROOT_ACTUAL_MEASUREMENT_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_source_mask_depth_measurement=False,source_only=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],readiness=p['readiness'],selection=p['selection'],tests=p['tests'],new_render_fit_forward_or_forbidden_access=False,automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),tests=p['tests'],readiness=p['readiness'],actual_measurements=0,source_mask_depth_RGB_body_header_hash_reads=0))
    print(json.dumps(dict(prepared=True,preparation=bind(H/'PREPARATION_COMPLETE.json'),protocol=bind(PROTOCOL),code=bind(__file__))),flush=True)
def load_buffer(b,declared):
    with np.load(checked(b),allow_pickle=False) as z:
        assert set(z.files)==set(declared);a={k:z[k].copy() for k in z.files}
    for k,d in declared.items():
        assert a[k].shape==tuple(d['shape']),k
        assert (a[k].dtype.kind=='U' and a[k].dtype.itemsize==4) if d['dtype']=='Unicode1' else a[k].dtype==np.dtype(d['dtype']),k
    return a
def run(entry):
    e=read(entry);assert e['approved_source_mask_depth_measurement'] and e['source_only'] and not e['new_render_fit_forward_or_forbidden_access'] and not e['automatic_retry']
    assert e['protocol']==bind(PROTOCOL);p=read(checked(e['protocol']))
    for k in ('code','scope','readiness','selection','tests'):assert e[k]==p[k];checked(p[k])
    assert p['code']==bind(__file__) and read(checked(p['tests']))['PASS'] and sha(READY)==READY_SHA
    assert sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'] and not OUT.exists()
    ready=read(READY);selection=read(checked(p['selection']));witnesses={r['global_index']:r for r in selection['unique_witnesses']};OUT.mkdir();started=time.monotonic();outputs=[];proofs=[];source_bindings=[]
    write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),source_only=True))
    try:
        for scene in ready['scenes']:
            witness=witnesses[scene['global_index']];metadata=scene['projected_native_metadata'];q8=scene['all8_original_input_queries'];assert len(q8)==8
            qscene=scene['selected_combined_scene_buffer']['inherited_binding'];qbefore=scene['saved_target_only_all_parts_buffer']['inherited_binding'];source_bindings.extend([qscene,qbefore])
            buffers={'SCENE':load_buffer(qscene,ready['declared_scene_buffer_schema']),'BEFORE_TARGET_ONLY':load_buffer(qbefore,ready['declared_scene_buffer_schema'])}
            assert ash(buffers['SCENE']['RGB_uint8'])==witness['decoded_RGB_sha256_inherited']
            for which,buf in buffers.items():
                assert np.array_equal(buf['native8_CF_xy'],metadata['native8_CF_xy']) and np.array_equal(buf['permCF_to_canonical'],metadata['permCF_to_canonical'])
                assert exact(buf['native8_canonical_xy'][buf['permCF_to_canonical']],buf['native8_CF_xy'])
                for key in ('K','R','t'):assert np.array_equal(buf['camera_'+key],metadata['camera'][key])
                gtfile=read(checked(scene['original_metadata']['GT_scenes']));variant=scene['variant'] if which=='SCENE' else 'BEFORE';gt=next(x for x in gtfile['rows'] if x['variant']==variant)
                assert buf['state8_CF'].tolist()==gt['state8_CF']==[r['state'] for r in gt['records_CF']]
            for slot,qrow in enumerate(q8):
                assert qrow['native_slot']==slot and qrow['query_xy']==witness['original_initial_points'][slot]
                record=metadata['records_CF_metadata'][slot];assert record['CF_index']==slot and record['canonical_index']==metadata['permCF_to_canonical'][slot]
                reference=metadata['native8_CF_xy'][slot];assert record['projected_reference_xy']==reference
                measured={}
                for which,buf in buffers.items():
                    cat=categories(buf['part_id'],buf['ambiguity'],record['incident_part_ids'])
                    measured[which]=dict(original_q_point=point_measure(buf,cat,qrow['query_xy']),original_q_L_central4=crop_region(cat,qrow['query_xy'],qrow['roi_side'],47,49),original_q_L_central32=crop_region(cat,qrow['query_xy'],qrow['roi_side'],32,64),nominal_reference_diagnostic_point_only=point_measure(buf,cat,reference))
                outputs.append(dict(original_global_index=int(qrow['scene_index']*8+slot),scene_index=scene['scene_index'],base_index=scene['base_index'],variant=scene['variant'],query_id=qrow['query_id'],native_slot=slot,is_original_selected_witness=slot==witness['native_slot'],original_qL_metadata=qrow,nominal_reference_xy=reference,canonical_index=record['canonical_index'],incident_part_ids=record['incident_part_ids'],original_semantic_record_CF=record,measurements=measured,new_label=None,training_admitted=False))
            proofs.append(dict(selected_global_index=scene['global_index'],scene_index=scene['scene_index'],selected_scene_RGB_decoded_SHA=ash(buffers['SCENE']['RGB_uint8']),inherited_V171_RGB_SHA=witness['decoded_RGB_sha256_inherited'],RGB_correspondence_PASS=True,all8_saved_state_projection_perm_camera_metadata_PASS=True,all8_original_qL_inherited_exact=True,scene_buffer=qscene,before_buffer=qbefore))
        assert len(outputs)==24 and sum(r['is_original_selected_witness'] for r in outputs)==3
        for b in source_bindings:checked(b)
        for b in (p['code'],p['scope'],p['readiness'],p['selection'],p['tests']):checked(b)
        rowfile=OUT/'ALL24_ORIGINAL_QUERY_AND_REFERENCE_DIAGNOSTICS.jsonl'
        with rowfile.open('x') as f:
            for r in outputs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,phase='V172',code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),readiness=p['readiness'],selection=p['selection'],tests=p['tests'],rows=bind(rowfile),row_count=24,selected_witness_rows=3,scenes=proofs,owner_class_order=list(CLASSES),all_saved_source_buffer_hashes_unchanged=True,reference_centered_crops_created=0,new_RGB_renders_models_fits_features_weights_current_V169_private_V112_access=0,no_labels_qL_or_thresholds_changed=True,no_training_admission=True,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and a.entry is None:prepare()
    elif a.entry is not None and not a.prepare:run(a.entry)
    else:raise SystemExit('Prepare first; actual --entry is root responsibility after complete review')
