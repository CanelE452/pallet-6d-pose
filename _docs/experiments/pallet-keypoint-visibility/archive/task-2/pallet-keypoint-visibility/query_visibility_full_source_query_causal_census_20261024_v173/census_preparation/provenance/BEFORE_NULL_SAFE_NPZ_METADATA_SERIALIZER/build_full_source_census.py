"""V173 metadata preparation and root-entry-only OLD6144 literal-raster census."""
from pathlib import Path
import argparse,ast,hashlib,json,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_FULL_SOURCE_MASK_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_FULL_SOURCE_MASK_BODIES.json';TESTS=H/'GENERATED_INVENTED_CHECKS.json'
ROWS=H/'ALL6144_ORIGINAL_QUERY_METADATA_BEFORE_MASKS.jsonl';SCENES=H/'ALL768_SCENE_PATHS_BEFORE_MASKS.jsonl'
OUT=V/'literal_raster_census_v1'
V129=P/'query_visibility_registered_native_diversity_20261023_v129'
RENDER=V129/'actual_corpus_v1/RENDER_COMPLETE.json'
OWNER=('INCIDENT_TARGET_DECLARED_CF','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT')
JOINT=('VISIBLE_TARGET','EXTERNAL_OVER_BEFORE_TARGET','FOREIGN_WITHOUT_BEFORE_TARGET','BACKGROUND_BOTH','AMBIGUOUS_EITHER','INCONSISTENT_CHANGE','INVALID_SUPPORT')
FLAGS=('query_available','factual_query_available','feature_constructed','crop_constructed','context_available','Q_inside','RGB_central4_available','native_continuous_valid','native_nearest_valid')
REASONS=('NONE','BEFORE_FOREIGN_UNEXPECTED','INVALID_FOREIGN_OR_BEFORE_DEPTH','EXTERNAL_NOT_STRICTLY_FRONT','TARGET_PART_CHANGED','TARGET_TRIANGLE_CHANGED','TARGET_DEPTH_CHANGED_OR_INVALID','TARGET_DISAPPEARED_TO_BG','BACKGROUND_TO_TARGET','OTHER_UNCLASSIFIED')
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
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_lines(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pure(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::PURE_SAMPLING_ONLY','exec'),ns);return ns
def joint_map(before,scene):
    bp,sp=before['part_id'],scene['part_id'];bd,sd=before['camera_depth'],scene['camera_depth'];bt,st=before['global_triangle_id'],scene['global_triangle_id']
    assert bp.dtype==sp.dtype==np.int32 and bp.shape==sp.shape
    ambiguous=before['ambiguity']|scene['ambiguity'];target_b=(bp>=0)&(bp!=999);target_s=(sp>=0)&(sp!=999);bg_b=bp<0;bg_s=sp<0
    out=np.full(bp.shape,5,np.int8);reason=np.full(bp.shape,9,np.int8)
    out[ambiguous]=4;reason[ambiguous]=0;active=~ambiguous
    # Background and external guards use exact original raster values, no epsilon.
    bb=active&bg_b&bg_s;out[bb]=3;reason[bb]=0
    ext=active&target_b&(sp==999);fw=active&bg_b&(sp==999)
    ext_depth=np.isfinite(bd)&np.isfinite(sd)&(bd>0)&(sd>0)
    valid_ext=ext&ext_depth&(sd<bd);out[valid_ext]=1;reason[valid_ext]=0
    reason[ext&~ext_depth]=2;reason[ext&ext_depth&~(sd<bd)]=3
    valid_fw=fw&np.isfinite(sd)&(sd>0);out[valid_fw]=2;reason[valid_fw]=0;reason[fw&~valid_fw]=2
    same=active&target_b&target_s;part_same=bp==sp;tri_same=bt==st;depth_same=np.isfinite(bd)&np.isfinite(sd)&(bd>0)&(sd>0)&(bd==sd)
    visible=same&part_same&tri_same&depth_same;out[visible]=0;reason[visible]=0
    reason[same&~part_same]=4;reason[same&part_same&~tri_same]=5;reason[same&part_same&tri_same&~depth_same]=6
    reason[active&target_b&bg_s]=7;reason[active&bg_b&target_s]=8;reason[active&(bp==999)]=1
    assert np.isin(out,np.arange(6)).all() and np.all(reason[out!=5]==0)
    return out,reason
def sample_weights(classes,points,nclasses,apply_crop_domain=False):
    """V172 exact coordinate/weight arithmetic, generalized INVALID index for joint7."""
    points=np.asarray(points,np.float64);assert points.ndim==2 and points.shape[1]==2 and np.isfinite(points).all()
    h,w=classes.shape;x,y=points.T;ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
    weights=np.zeros((len(points),nclasses),np.float64);invalid=nclasses-1
    for ax,ay,ww in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=ix+ax,iy+ay;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);cls=np.full(len(points),invalid,np.int8);cls[inside]=classes[yy[inside],xx[inside]]
        weights[np.arange(len(points)),cls]+=ww
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    if apply_crop_domain:weights[~valid]=0;weights[~valid,invalid]=1
    assert np.all(weights>=0) and np.allclose(weights.sum(1),1,rtol=0,atol=2e-15)
    return weights,valid
def positions(q,L,lo,hi):
    u=(np.arange(lo,hi,dtype=np.float64)+.5)/96-.5;xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);return np.column_stack((xx.ravel(),yy.ravel()))
def measure_soft(classes,q,L,nclasses):
    validq=q is not None and len(q)==2 and np.isfinite(q).all() and not np.array_equal(q,[-1.,-1.]);result={}
    for name,points,iscrop in [('point',np.asarray(q,np.float64)[None] if validq else None,False),('central4',positions(q,L,47,49) if validq and L is not None and np.isfinite(L) and L>0 else None,True),('central32',positions(q,L,32,64) if validq and L is not None and np.isfinite(L) and L>0 else None,True)]:
        if points is None:result[name]=None
        else:result[name]=sample_weights(classes,points,nclasses,iscrop)[0].mean(0)
    return result
def invented(pinned):
    def b(pid,dep=None,tri=None,amb=None):
        pid=np.asarray([pid],np.int32);return dict(part_id=pid,camera_depth=np.asarray([dep if dep is not None else [2.]*pid.shape[1]],np.float64),global_triangle_id=np.asarray([tri if tri is not None else list(range(pid.shape[1]))],np.int32),ambiguity=np.asarray([amb if amb is not None else [False]*pid.shape[1]],bool))
    before=b([0,0,-1,-1,0,0]);scene=b([0,999,999,-1,999,1],dep=[2.,1.,1.,np.nan,1.,2.],amb=[0,0,0,0,1,0]);jm,reason=joint_map(before,scene)
    c={'joint_six_inimage_classes_partition':jm.tolist()==[[0,1,2,3,4,5]],'ambiguity_either_priority':int(jm[0,4])==4,'target_identity_change_inconsistent':int(reason[0,5])==4}
    points=np.array([[-.25,0.],[0,0],[.5,0],[4.9,0],[10.,0]],np.float64);j,_=sample_weights(jm,points,7);c['outside_mass_separate_INVALID_joint7']=j[-1,6]==1 and j[0,6]==.25
    owner=np.array([[0,1,2,3,4,0]],np.int8)
    for crop in (False,True):
        a,av=sample_weights(owner,points,6,crop);z,zv=pinned['bilinear_weights'](owner,points,crop);assert exact(a,z) and exact(av,zv)
    c['general_sampler_byteexact_pinned_V172_owner6']=True
    for key,value in [('global_triangle_id',9),('camera_depth',3.)]:
        bb=b([0]);ss=b([0]);ss[key][0,0]=value;assert joint_map(bb,ss)[0][0,0]==5
    c['visible_requires_same_triangle_and_exact_finite_depth']=True
    bb=b([0]);ss=b([999],dep=[3.]);c['external_strictly_front_guard']=joint_map(bb,ss)[0][0,0]==5
    bb=b([999]);ss=b([999],dep=[1.]);c['BEFORE_foreign_not_foreign_without_target']=joint_map(bb,ss)[0][0,0]==5
    bb=b([7]);ss=b([999],dep=[1.]);c['external_uses_all_target_not_only_incident']=joint_map(bb,ss)[0][0,0]==1
    bb=b([0,-1]);s1=b([999,-1],dep=[1.,np.nan]);s2=b([-1,999],dep=[np.nan,1.]);j1,_=joint_map(bb,s1);j2,_=joint_map(bb,s2)
    c['equal_marginals_different_joint_not_probability_product']=sorted(s1['part_id'][0].tolist())==sorted(s2['part_id'][0].tolist()) and not np.array_equal(j1,j2)
    c['BEFORE_self_pair_target_visible_BG_background']=joint_map(before,before)[0].tolist()==[[0,0,3,3,0,0]]
    c['missing_sentinel_is_UNKNOWN_not_zero_or_BG']=all(x is None for x in measure_soft(owner,[-1.,-1.],10.,6).values())
    c['central4_original96_coordinates']=positions([64,64],96,47,49).tolist()==[[63.5,63.5],[64.5,63.5],[63.5,64.5],[64.5,64.5]]
    c['central32_1024_original96_positions']=positions([64,64],96,32,64).shape==(1024,2)
    bb=b([0]);ss=b([999],dep=[np.nan]);c['nonfinite_foreign_depth_inconsistent']=joint_map(bb,ss)[0][0,0]==5
    assert all(c.values());return dict(completed=True,PASS=True,invented_only=True,checks={k:bool(v) for k,v in c.items()},number_checks=len(c),source_mask_depth_or_RGB_body_header_hash_reads=0,code=bind(__file__))
def prepare():
    scope=read(SCOPE);assert scope['no_model_scores_features_weights_forward_or_training'];pinned_b=scope['original_pure_measurement_code'];pinned=pure(checked(pinned_b),{'categories','bilinear_weights'},{'np':np,'CLASSES':OWNER})
    # All below are original text metadata only; binary paths are inherited/stat'ed, never hashed.
    selpath=P/'query_visibility_public_constraint_witness_crops_20261024_v171/WITNESS_SELECTION_BEFORE_RGB.json';selection=read(selpath)
    qb=next(b for b in selection['bindings'] if b['path'].endswith('ORIGINAL_NATIVE_QUERY6144_INPUT_ONLY.jsonl'));fb=next(b for b in selection['bindings'] if b['path'].endswith('INPUT_ROWS_AND_CROP_FACTS.json'));gb=next(b for b in selection['bindings'] if b['path'].endswith('ALL6144_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl'))
    queries=lines(checked(qb));facts=read(checked(fb))['rows'];oldgt=lines(checked(gb))[0::2];assert len(queries)==len(facts)==len(oldgt)==6144
    render=read(RENDER);assert render['PASS'] and render['scenes']==768 and render['bases']==192
    gtrows=read(checked(render['GT_scenes']))['rows'];inputrows=read(checked(render['input_scenes']))['rows'];assert len(gtrows)==len(inputrows)==768
    checkpoints={read(checked(b))['base_index']:(b,read(b['path'])) for b in render['base_checkpoints']};assert set(checkpoints)==set(range(192))
    scene_rows=[];inventory=[]
    for si,(g,ir) in enumerate(zip(gtrows,inputrows)):
        assert g['scene_index']==ir['scene_index']==si;before=gtrows[(si//4)*4];base=checkpoints[g['base_index']][1];assert base['target_layers_before']==base['target_layers_after']
        for binding in (g['raw_buffers'],before['raw_buffers']):assert Path(binding['path']).is_file()
        scene_rows.append(dict(scene_index=si,base_index=g['base_index'],variant=g['variant'],split=g['split'],factors=ir['factors'],scene_buffer=g['raw_buffers'],BEFORE_buffer=before['raw_buffers'],decoded_RGB_SHA=ir['decoded_RGB_sha256'],raw_hw=ir['raw_hw'],native8_CF_xy=g['native8_CF_xy'],permCF_to_canonical=g['permCF_to_canonical'],state8_CF=g['state8_CF'],camera=g['camera'],base_checkpoint=checkpoints[g['base_index']][0],target_layers_immutable=base['target_layers_before']==base['target_layers_after']))
        for slot in range(8):
            i=si*8+slot;q,f,t=queries[i],facts[i],oldgt[i];record=g['records_CF'][slot]
            assert f['occurrence_index']==i and q['native_slot']==t['native_slot']==record['CF_index']==slot and q['scene_index']==si
            assert q['query_id']==f['occurrence_id']==t['query_id'] and q['query_xy']==f['query_xy']==t['query_xy'] and q['roi_side']==f['L']==t['roi_side']
            assert record['canonical_index']==g['permCF_to_canonical'][slot]==t['canonical_index'] and record['state']==g['state8_CF'][slot]==t['teacher_state']
            assert f['decoded_RGB_sha256']==ir['decoded_RGB_sha256'] and f['raw_RGB_PNG']==ir['raw_RGB_PNG']
            inventory.append(dict(original_global_index=i,original_occurrence_id=q['query_id'],scene_index=si,base_index=g['base_index'],native_slot=slot,split=g['split'],variant=g['variant'],factors=ir['factors'],query_xy=q['query_xy'],L=q['roi_side'],original_query_metadata=q,original_input_flags={k:f[k] for k in FLAGS},selected_index=f['selected_index'],original_semantic_state=t['teacher_state'],declared_canonical_index=t['canonical_index'],nominal_reference_xy=t['reference_xy'],correspondence_boundary=t['correspondence_boundary'],position_band=t['position_band'],q_reference_error_px=t['error_px'],incident_part_ids=record['incident_part_ids'],primary_part_id=record['primary_part_id'],active_witness_indices=record['active_witness_indices'],rgb96_SHA_inherited=f['rgb96_sha256'],valid96_SHA_inherited=f['valid96_sha256'],training_admission=False))
    assert len(inventory)==6144 and sum(r['split']=='TRAIN' for r in inventory)==4096 and sum(not r['original_input_flags']['query_available'] for r in inventory)==88
    write_lines(ROWS,inventory);write_lines(SCENES,scene_rows);write(TESTS,invented(pinned))
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),pinned_sampling_code=pinned_b,definition_authority=scope['definition_authority'],original_queries=qb,original_crop_facts=fb,original_GT=gb,render_complete=bind(RENDER),original_scene_GT=render['GT_scenes'],original_scene_input=render['input_scenes'],inventory=bind(ROWS),scene_inventory=bind(SCENES),tests=bind(TESTS),runtime=dict(executable=sys.executable,numpy=np.__version__),output=str(OUT),counts=dict(rows=6144,scenes=768,bases=192,TRAIN=4096,VALIDATION=2048,original_inputU=88),owner_class_order=OWNER,joint_class_order=JOINT,
        joint_definition='At each original raw source pixel: either ambiguity first; target BEFORE+same targetpart/triangle/exact finite positive depth SCENE =VISIBLE; BEFORE anytarget+SCENE999 with finite positive depths and SCENEdepth<BEFOREdepth=EXTERNAL; BEFOREBG+SCENE999 finite positive depth=FOREIGN_WITHOUT; BG/BG=BACKGROUND; all remaining=INCONSISTENT with reason. BEFORE999 never counts as target or BG. Invalid footprint support is separate.',
        joint_aggregation='Classify pixel conjunction before float64 bilinear sampling. Never multiply marginal owner distributions. Keep all inconsistent weights/scene reason pixel counts.',
        sampling='Pinned V172 categorical owner6; exact samefloat64 source-pixel footprint generalized tojoint7 with invented byte-parity. Original q point, originalq/L96 central4 indices47:49 andcentral32 indices32:64. No ref-centered crop.',
        invalid='Missing/sentinelq ormissingL produces JSONnull/NPZNaN+explicit missing flag, neverfakezero/BG. Finite out-of-frame support mass INVALID. Original inputU never filters census rows or becomes newlabels.',
        sidecar='Two fixedsplit NPZ TRAIN4096/VALIDATION2048; allsoftfloat64 sceneowner6/BEFOREowner6/joint7 atpoint/c4/c32 + originalglobalID/q/L/native/flags/semantic; training_admission=false.',
        authority='Literal original raster and declared source registration only; not initializer physical native identity certification. CF/V4 correspondence_boundary is ordering disagreement, not pixelboundary. Original semantic D/E/S/U unchanged. No hardlabel threshold, best permutation, row replacement, training admission or model selection.',
        binary_headers_bodies_or_hashes_read_during_preparation=0,score_feature_weight_or_forbidden_reads=0,automatic_retry=False)
    write(PROTOCOL,protocol);write(H/'ROOT_ACTUAL_CENSUS_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_full_source_census=False,source_only=True,code=protocol['code'],protocol=bind(PROTOCOL),scope=protocol['scope'],inventory=protocol['inventory'],scene_inventory=protocol['scene_inventory'],tests=protocol['tests'],pinned_sampling_code=pinned_b,new_model_fit_render_or_forbidden_access=False,automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=bind(PROTOCOL),tests=protocol['tests'],inventory=protocol['inventory'],scene_inventory=protocol['scene_inventory'],actual_source_mask_runs=0,binary_body_header_hash_reads=0))
    print(json.dumps(dict(prepared=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def load_buffer(binding,metadata):
    fields=('RGB_uint8','camera_depth','part_id','global_triangle_id','ambiguity','tie_count','native8_CF_xy','native8_canonical_xy','permCF_to_canonical','state8_CF','camera_K','camera_R','camera_t')
    with np.load(checked(binding),allow_pickle=False) as z:a={k:z[k].copy() for k in fields}
    for k,dtype in [('RGB_uint8',np.uint8),('camera_depth',np.float64),('part_id',np.int32),('global_triangle_id',np.int32),('ambiguity',bool),('tie_count',np.uint16)]:
        assert a[k].dtype==dtype and a[k].shape==((512,512,3) if k=='RGB_uint8' else (512,512)),k
    assert ash(a['RGB_uint8'])==metadata['decoded_RGB_SHA']
    assert a['state8_CF'].tolist()==metadata['state8_CF'] and np.array_equal(a['native8_CF_xy'],metadata['native8_CF_xy']) and np.array_equal(a['permCF_to_canonical'],metadata['permCF_to_canonical'])
    assert exact(a['native8_canonical_xy'][a['permCF_to_canonical']],a['native8_CF_xy'])
    for k in ('K','R','t'):assert np.array_equal(a['camera_'+k],metadata['camera'][k])
    return a
def aggregation(rows,soft,nearest_joint):
    result=[];splits=np.array([r['split'] for r in rows]);native=np.array([r['native_slot'] for r in rows]);variant=np.array([r['variant'] for r in rows]);semantic=np.array([r['original_semantic_state'] for r in rows]);u=np.array([not r['original_input_flags']['query_available'] for r in rows])
    factors={k:np.array([r['factors'][k] for r in rows]) for k in ('material','aspect','geometry_family','foreign_family','signed_quarter')}
    for split in ('ALL','TRAIN','VALIDATION'):
        base=np.ones(6144,bool) if split=='ALL' else splits==split;groups=[('ALL',base)]
        for k,values in {'native_slot':native,'variant':variant,'semantic':semantic,**factors}.items():
            for value in (range(8) if k=='native_slot' else np.unique(values)):groups.append((k+'::'+str(value),base&(values==value)))
        for slot in range(8):
            for v in ('BEFORE','COVER','NEAR_SHAM','FAR_SHAM'):groups.append((f'native_variant::{slot}::{v}',base&(native==slot)&(variant==v)))
        for name,mask in groups:
            measures={}
            for key,a in soft.items():
                valid=mask&np.isfinite(a).all(1);measures[key]=dict(measured_N=int(valid.sum()),missing_N=int((mask&~np.isfinite(a).all(1)).sum()),mean_weights=a[valid].mean(0).tolist() if valid.any() else None,sum_weights=a[valid].sum(0).tolist() if valid.any() else None)
            result.append(dict(split=split,group=name,N=int(mask.sum()),original_inputU=int((mask&u).sum()),semantic_counts={s:int((mask&(semantic==s)).sum()) for s in 'DESU'},nearest_joint_counts={('UNKNOWN_MISSING' if i<0 else JOINT[i]):int((mask&(nearest_joint==i)).sum()) for i in range(-1,7)},soft_measures=measures))
    assert next(r['N'] for r in result if r['split']=='ALL' and r['group']=='ALL')==6144
    return result
def run(entry):
    e=read(entry);assert e['approved_full_source_census'] and e['source_only'] and not e['new_model_fit_render_or_forbidden_access'] and not e['automatic_retry']
    assert e['protocol']==bind(PROTOCOL);p=read(checked(e['protocol']))
    for k in ('code','scope','inventory','scene_inventory','tests','pinned_sampling_code'):assert e[k]==p[k];checked(p[k])
    assert p['code']==bind(__file__) and read(checked(p['tests']))['PASS'] and not OUT.exists()
    assert sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy']
    for k in ('definition_authority','original_queries','original_crop_facts','original_GT','render_complete','original_scene_GT','original_scene_input'):checked(p[k])
    rows=lines(ROWS);scenes=lines(SCENES);assert len(rows)==6144 and len(scenes)==768
    pinned=pure(checked(p['pinned_sampling_code']),{'categories','bilinear_weights','pixel','distribution','point_measure'},{'np':np,'CLASSES':OWNER})
    soft={f'{a}_{m}':np.full((6144,len(OWNER) if a!='joint' else len(JOINT)),np.nan,np.float64) for a in ('scene_owner','BEFORE_owner','joint') for m in ('point','central4','central32')}
    nearest_joint=np.full(6144,-1,np.int8);outputs=[];diagnostics=[];used={};OUT.mkdir();started=time.monotonic();write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),source_only=True))
    try:
        for base in range(192):
            before_meta=scenes[base*4];assert before_meta['variant']=='BEFORE';before=load_buffer(before_meta['scene_buffer'],before_meta);used[before_meta['scene_buffer']['path']]=before_meta['scene_buffer']
            for vm in scenes[base*4:base*4+4]:
                si=vm['scene_index'];assert vm['BEFORE_buffer']==before_meta['scene_buffer'];scene=before if vm['variant']=='BEFORE' else load_buffer(vm['scene_buffer'],vm);used[vm['scene_buffer']['path']]=vm['scene_buffer']
                joint,reason=joint_map(before,scene);visible=joint==0
                diagnostics.append(dict(scene_index=si,base_index=base,variant=vm['variant'],scene_buffer=vm['scene_buffer'],BEFORE_buffer=vm['BEFORE_buffer'],raw_joint_pixel_counts={JOINT[k]:int((joint==k).sum()) for k in range(6)},inconsistent_reason_pixel_counts={REASONS[k]:int(((joint==5)&(reason==k)).sum()) for k in range(1,len(REASONS))},visible_target_RGB_unequal_pixels=int(np.any(scene['RGB_uint8'][visible]!=before['RGB_uint8'][visible],axis=1).sum()),decoded_RGB_SHA_verified=True,saved_native_state_projection_perm_camera_metadata_verified=True))
                cat_cache={}
                for r in rows[si*8:si*8+8]:
                    i=r['original_global_index'];assert i==len(outputs) and r['scene_index']==si;incident=tuple(r['incident_part_ids'])
                    if incident not in cat_cache:cat_cache[incident]=(pinned['categories'](scene['part_id'],scene['ambiguity'],incident),pinned['categories'](before['part_id'],before['ambiguity'],incident))
                    sc,bc=cat_cache[incident];q=r['query_xy'];L=r['L'];ms={}
                    for name,classes,nclasses in [('scene_owner',sc,6),('BEFORE_owner',bc,6),('joint',joint,7)]:
                        measures=measure_soft(classes,q,L,nclasses);ms[name]={k:None if z is None else z.tolist() for k,z in measures.items()}
                        for kind,z in measures.items():
                            if z is not None:soft[name+'_'+kind][i]=z
                    qp=pinned['point_measure'](scene,sc,q);bp=pinned['point_measure'](before,bc,q)
                    if qp['nearest'] is not None:
                        x,y=qp['nearest']['x'],qp['nearest']['y'];inside=qp['nearest']['in_image'];nearest_joint[i]=int(joint[y,x]) if inside else 6
                        joint_nearest=dict(class_name=JOINT[int(nearest_joint[i])],inconsistent_reason=REASONS[int(reason[y,x])] if inside else None,raw_pixel_xy=[x,y])
                    else:joint_nearest=None
                    outputs.append(dict(**r,scene_owner_point_details=qp,BEFORE_owner_point_details=bp,joint_nearest=joint_nearest,soft_distributions=ms,owner_class_order=OWNER,joint_class_order=JOINT,registration_authority=p['authority'],literal_raster_diagnostic_only=True))
            if (base+1)%16==0:print(json.dumps(dict(bases_completed=base+1,rows_completed=len(outputs),elapsed_seconds=time.monotonic()-started)),flush=True)
        assert len(outputs)==6144 and sum(not r['original_input_flags']['query_available'] for r in outputs)==88
        for a in soft.values():valid=np.isfinite(a).all(1);assert np.allclose(a[valid].sum(1),1,rtol=0,atol=2e-13)
        allrows=OUT/'ALL6144_LITERAL_RASTER_SOURCE_DIAGNOSTICS.jsonl';write_lines(allrows,outputs);df=OUT/'ALL768_SCENE_INCONSISTENCY_DIAGNOSTICS.jsonl';write_lines(df,diagnostics)
        sidecars={};splits=np.array([r['split'] for r in rows]);metadata=dict(original_global_index=np.arange(6144,dtype=np.int32),original_occurrence_id=np.array([r['original_occurrence_id'] for r in rows]),scene_index=np.array([r['scene_index'] for r in rows],np.int32),base_index=np.array([r['base_index'] for r in rows],np.int32),native_slot=np.array([r['native_slot'] for r in rows],np.int8),query_xy=np.asarray([r['query_xy'] for r in rows],np.float64),L=np.asarray([np.nan if r['L'] is None else r['L'] for r in rows],np.float64),split=splits,variant=np.array([r['variant'] for r in rows]),original_semantic_state=np.array([r['original_semantic_state'] for r in rows]),declared_canonical_index=np.array([r['declared_canonical_index'] for r in rows],np.int8),nominal_reference_xy=np.asarray([r['nominal_reference_xy'] for r in rows],np.float64),correspondence_boundary=np.array([r['correspondence_boundary'] for r in rows],bool),selected_index=np.array([r['selected_index'] for r in rows],np.int32),nearest_joint_class_index=nearest_joint)
        metadata.update({k:np.array([r['original_input_flags'][k] for r in rows],bool) for k in FLAGS});metadata.update({k:np.array([r['factors'][k] for r in rows]) for k in ('material','aspect','geometry_family','foreign_family','signed_quarter')})
        for split,N in [('TRAIN',4096),('VALIDATION',2048)]:
            mask=splits==split;assert int(mask.sum())==N;data={k:a[mask].copy() for k,a in {**metadata,**soft}.items()};data.update(owner_class_order=np.array(OWNER),joint_class_order=np.array(JOINT),training_admission=np.array(False),initializer_physical_native_identity_certified=np.array(False))
            path=OUT/f'{split}{N}_LITERAL_RASTER_SOFT_SIDECAR.npz';np.savez_compressed(path,**data)
            with np.load(path,allow_pickle=False) as z:assert set(z.files)==set(data) and all(exact(z[k],a) for k,a in data.items())
            sidecars[split]=bind(path)
        groups=OUT/'EXACT_SPLIT_GROUP_NATIVE_VARIANT_AGGREGATES.json';write(groups,dict(completed=True,owner_class_order=OWNER,joint_class_order=JOINT,groups=aggregation(rows,soft,nearest_joint)))
        for b in used.values():checked(b)
        for k in ('code','scope','inventory','scene_inventory','tests','pinned_sampling_code','definition_authority','original_queries','original_crop_facts','original_GT','render_complete','original_scene_GT','original_scene_input'):checked(p[k])
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,phase='V173',code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),scope=p['scope'],tests=p['tests'],inventory=p['inventory'],scene_inventory=p['scene_inventory'],rows=bind(allrows),scene_diagnostics=bind(df),sidecars=sidecars,aggregates=bind(groups),counts=dict(rows=6144,scenes=768,bases=192,TRAIN=4096,VALIDATION=2048,original_inputU=88,missing_point_soft=int(np.isnan(soft['joint_point']).all(1).sum()),inconsistent_point_weight_rows=int(np.nansum(soft['joint_point'][:,5]>0))),all_original_query_ID_qL_native_flags_semantic_preserved=True,all_768_source_buffer_hashes_unchanged=True,all_soft_distributions_unit_mass_or_explicit_missing=True,owner_class_order=OWNER,joint_class_order=JOINT,registration_authority=p['authority'],new_label_threshold_model_fit_render_admission=0,current_V169_private_V112_score_feature_weight_access=0,training_admission=False,selected_model=None,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and a.entry is None:prepare()
    elif a.entry is not None and not a.prepare:run(a.entry)
    else:raise SystemExit('Prepare first; actual root-approved entry required')
