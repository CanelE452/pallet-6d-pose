"""Inert V141 RGB-only exact frozen R0 initializer on fixed public16 images.

Preparation copies only RGB identity/hash/HW/fold metadata. Runtime reads this
whitelist and original public RGB, factory code and R0 checkpoint only. Source
pose/K/native reference/semantic files and historical predicted scores are not
opened by the forward producer. No classifier, mapping or coordinate update.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_SOURCE_INITIALIZER_ASSOCIATION_POLICY_BEFORE_FORWARD.json'
POLICY_SHA='99e3efb5693da7ecb0866bdb0b931d6c21d1f56b31351ceaf7e2db50c6866d41'
V140=P/'query_visibility_public_native_reference_feasibility_20261024_v140'
V94=P/'query_visibility_matched_query_scalar_20261023_v94'
REFERENCE=V94/'pallox_actual_initializer_preparation/infer_fixed_pallox_queries.py'
REFERENCE_SHA='8e87d71fc0fc56468b5fd4ce4f366211520d02286e15a3ddd96a9355d59b9ce2'
REFERENCE_PROTOCOL=V94/'pallox_actual_initializer_preparation/PROTOCOL_BEFORE_SOURCE_FORWARD.json'
ROI=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FORWARD.json';INPUTS=H/'INPUTS16_RGB_ONLY.jsonl';GENERATED=H/'GENERATED_BOUNDARY_AND_REFERENCE_PARITY.json'
OUT=V/'actual_public_initializer_v1'
FORBIDDEN=('classifier_forward','fit','current223','V112','PnP','pose_optimization','q_L_correction','GT_input','GT_centered_crop','detection_reselection','historical_score_read','automatic_retry')
INPUT_KEYS={'frame_index','id','source','source_id','split','image','image_sha256','decoded_RGB_sha256','raw_hw','decoder'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def lines(path):return[json.loads(l)for l in Path(path).read_text().splitlines()if l]
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pure_native():
    nodes=[n for n in ast.parse(ROI.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_grid'];assert len(nodes)==1
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROI),'exec'),ns);return ns['native_grid']
def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float):return value if np.isfinite(value)else None
    if isinstance(value,dict):return{k:clean(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return[clean(v)for v in value]
    return value
def serial(z):
    return clean(dict(candidates=z['candidates'],selected_index=z['selected_index'],input_shape=z['input_shape'],canvas_shape=z['canvas_shape'],added_border=z['added_border']))


def query_arrays(z,hw):
    candidates=z['candidates'];selected=None if not candidates else int(np.argmax([r['score']for r in candidates]))
    assert selected==z['selected_index'];Hh,Ww=hw
    if selected is None:
        return dict(query_xy=np.full((8,2),np.nan,np.float64),L=np.full(8,np.nan,np.float64),
            finite_non_sentinel=np.zeros(8,bool),continuous_valid=np.zeros(8,bool),nearest_valid=np.zeros(8,bool),RGBcentre_valid=np.zeros(8,bool),native_grid_valid=np.zeros((8,9,9),bool))
    q=np.asarray(candidates[selected]['keypoints_xy'],np.float64);assert q.shape==(9,2)
    _,valid,L=pure_native()(q,hw);q8=q[:8].copy();finite=np.isfinite(q8).all(1)&~(q8==[-1,-1]).all(1)
    continuous=finite&(q8[:,0]>=0)&(q8[:,0]<Ww)&(q8[:,1]>=0)&(q8[:,1]<Hh)
    with np.errstate(invalid='ignore'):
        nearest=finite&(np.floor(q8[:,0]+.5)>=0)&(np.floor(q8[:,0]+.5)<Ww)&(np.floor(q8[:,1]+.5)>=0)&(np.floor(q8[:,1]+.5)<Hh)
    centre=finite&(q8[:,0]>=-.5)&(q8[:,0]<Ww-.5)&(q8[:,1]>=-.5)&(q8[:,1]<Hh-.5)
    assert np.array_equal(nearest,centre)
    return dict(query_xy=q8,L=np.full(8,L,np.float64),finite_non_sentinel=finite,continuous_valid=continuous,nearest_valid=nearest,RGBcentre_valid=centre,native_grid_valid=valid)


def build_input_rows(policy):
    # Source records may have GT metadata elsewhere. Only these whitelisted
    # image fields are copied; no reference/pose/state enters INPUTS16.
    loco=read(check(policy['source_inputs_fixed_before_new_initialization']['LOCO12']))['rows'];assert len(loco)==12
    fixed_path=V140/'pallox_reference_preparation/FIXED_FOUR_FRAME_SELECTION_BEFORE_NEW_BODY_READS.json'
    fixed=read(fixed_path)['selected'];assert len(fixed)==4
    facts_path=V140/'pallox_reference_diagnostics_v1/SOURCE_GEOMETRY_FEASIBILITY.json'
    facts=read(facts_path)['fixed_four_records'];decoded={r['id']:r['RGB_decoded_sha256']for r in facts}
    rows=[]
    for i,r in enumerate(loco):
        assert r['selection_index']==i and r['official_split']=='TRAIN'and r['subset']==2
        rows.append(dict(frame_index=i,id=f'V141_LOCO_{i:02d}_ALL{r["all_COCO_image_id"]}',source='LOCO',source_id=r['source_id'],split='TRAIN',
            image=r['raw_JPEG']['path'],image_sha256=r['raw_JPEG']['sha256'],decoded_RGB_sha256=r['decoded_RGB_sha256'],raw_hw=r['raw_hw'],decoder='CANONICAL_PIL_RGB_TO_BGR'))
    for r in fixed:
        ident=r['sequence']+':'+r['stem'];image=r['files']['image'];split=r['original_split'];assert split in('TRAIN_LAB','VAL_HALL')
        rows.append(dict(frame_index=len(rows),id='V141_PALLOX_'+ident,source='PALLOX',source_id=ident,split=split,
            image=image['path'],image_sha256=image['sha256'],decoded_RGB_sha256=decoded[ident],raw_hw=[480,640],decoder='EXACT_V94_CV2_BGR'))
    assert len(rows)==16 and all(set(r)==INPUT_KEYS for r in rows)and len({r['image']for r in rows})==16
    return rows,dict(LOCO_input_manifest=policy['source_inputs_fixed_before_new_initialization']['LOCO12'],
        Pallox_fixed_selection=bind(fixed_path),Pallox_decoded_RGB_SHA_metadata_only=bind(facts_path),
        source_GT_metadata_numeric_fields_not_copied=True,source_GT_files_not_opened_at_runtime=True)


def generated():
    assert sha(REFERENCE)==REFERENCE_SHA
    nodes=[n for n in ast.parse(REFERENCE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('native_function','selected_queries','serial')]
    ns={'np':np,'ast':ast,'ROI':ROI};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REFERENCE),'exec'),ns)
    base=np.array([[90.+11*k,120.+3*k]for k in range(9)],np.float64)
    def candidate(score,q):return dict(candidate_index=0,score=score,box_xyxy=np.array([1.,2.,3.,4.]),keypoints_xy=q,keypoints_conf=np.full(9,.75))
    cases=0
    for xy in([-.5,0.],[-.25,10.],[0.,0.],[639.49,479.49],[639.5,10.],[639.75,10.],[-100.,-100.],[640.,480.]):
        q=base.copy();q[0]=xy;z=dict(candidates=[candidate(.8,q),candidate(.8,q+30)],selected_index=0,input_shape=(512,640),canvas_shape=(680,840),added_border=100)
        arrays=query_arrays(z,[480,640]);reference=ns['selected_queries'](z,[480,640])
        for k,r in enumerate(reference):
            assert arrays['query_xy'][k].tolist()==r['query_xy']and arrays['L'][k]==r['roi_side']
            assert arrays['continuous_valid'][k]==r['original_query_continuous_valid']and arrays['nearest_valid'][k]==r['query_nearest_pixel_valid']and arrays['RGBcentre_valid'][k]==r['query_rgb96_centre_domain_valid']
        assert serial(z)==ns['serial'](z);cases+=1
    null=query_arrays(dict(candidates=[],selected_index=None),[480,640]);assert np.isnan(null['query_xy']).all()and not null['native_grid_valid'].any()
    q=base.copy();q[8]=[1e8,-1e8];z=dict(candidates=[candidate(.8,q)],selected_index=0);one=query_arrays(z,[480,640])
    assert all(np.array_equal(one[k],query_arrays(dict(candidates=[candidate(.8,base)],selected_index=0),[480,640])[k],equal_nan=True)for k in one)
    q=base.copy();q[0]=[np.nan,15.];q[1]=[-1.,-1.];q[2]=[np.inf,-3.];bad=query_arrays(dict(candidates=[candidate(.8,q)],selected_index=0),[480,640])
    assert not bad['finite_non_sentinel'][:3].any()and not bad['native_grid_valid'][:3].any()and np.isnan(bad['query_xy'][0,0])and np.isinf(bad['query_xy'][2,0])
    assert clean(q)[0]==[None,15.]and clean(q)[2]==[None,-3.]
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),reference_code=bind(REFERENCE),native_grid=bind(ROI),invented_only=True,
        exact_V94_finite_boundary_serial_parity_cases=cases,firstargmax_tie_preserved=True,centroid_excluded_from_L_and_query=True,
        no_detection_all8_and_nonfinite_sentinel_OOB_raw_values_retained=True,continuous_nearest_RGBcentre_and_native9x9_masks_separate=True,
        raw_NPZ_preserves_nonfinite_JSON_uses_null_only=True,source_RGB_checkpoint_GT_savedscore_model_body_reads_or_forward=0))


def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(REFERENCE)==REFERENCE_SHA
    policy=read(POLICY);rows,flow=build_input_rows(policy);generated()
    with INPUTS.open('x')as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    previous=read(REFERENCE_PROTOCOL)
    # Checkpoint binding inherited as metadata; no checkpoint tensor/body read.
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),inputs=bind(INPUTS),input_source_flow=flow,
        reference_code=bind(REFERENCE),reference_protocol=bind(REFERENCE_PROTOCOL),backbone_code=previous['assets']['backbone_code'],checkpoint=previous['assets']['backbone_checkpoint'],native_grid=previous['native_grid'],
        runtime=previous['runtime'],input_fields=sorted(INPUT_KEYS),frames=16,native_slots=128,planned_new_in_fixed_inventory=14,planned_legacy2_repeat=2,
        frame_order='LOCO12 frozen order then Pallox4 frozen order',runtime_input_access='Only INPUTS16 and its original RGB paths; no source GT/label/pose/K/mask/native refs or old predictions.',
        decoders='LOCO canonical PILRGB->contiguous BGR as V74 decodedRGB-V2; Pallox exact V94 cv2 BGR; both must match inherited decodedRGB SHA.',
        selection='Exact R0 factory firstargmax .001 all candidates/raw9 retained, nativefirst8 unchanged; no reassociation/GT selection.',
        L='Exact V60 native_grid AST, first8 finite non-sentinel span quarter clipped16..maxHW; centroid excluded.',
        nonfinite='RawNPZ retains IEEE values; JSON represents nonfinite asnull with explicit flags; no replacement/drop.',
        source_folds=dict(TRAIN_LOCO=12,TRAIN_LAB=2,VAL_HALL=2),warmup='One blank first-HW image, then16 fixed RGBs and repeatfirst; no automatic retry.',
        P3='No saved fullP3 duplicate; repeatfirst FP16 P3 exact in memory only.',legacy_saved_q_comparison='Separate post-seal diagnosis only; producer never opens historic proposal/query bodies.',
        checkpoint_body_reads_during_preparation=0,RGB_body_reads_during_preparation=0,actual_forward=False,output=str(OUT))
    write(PROTOCOL,p)
    template=dict(approved_actual_public_initializer=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],**{k:False for k in FORBIDDEN})
    path=H/'ROOT_INITIALIZER_ENTRY_TEMPLATE.json';write(path,template)
    print(json.dumps(dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),inputs=bind(INPUTS),generated=bind(GENERATED),template=bind(path),RGB_checkpoint_model_forward_reads=0)),flush=True)


def state(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()


def run(entry):
    e=read(entry);assert e['approved_actual_public_initializer']is True and all(e[k]is False for k in FORBIDDEN)
    assert sha(POLICY)==POLICY_SHA and same(e['code'],bind(__file__))and same(e['protocol'],bind(PROTOCOL))
    p=read(PROTOCOL)
    for k in('code','policy','generated','inputs'):assert same(e[k],p[k]);check(p[k])
    for k in('reference_code','reference_protocol','backbone_code','checkpoint','native_grid'):check(p[k])
    assert read(GENERATED)['PASS']and not OUT.exists()and p['frames']==16 and p['native_slots']==128 and p['output']==str(OUT)
    rows=lines(INPUTS);assert len(rows)==16 and all(set(r)==INPUT_KEYS and r['frame_index']==i for i,r in enumerate(rows))
    OUT.mkdir();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['YOLO_CONFIG_DIR']=str(OUT/'runtime/Ultralytics');os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib')
    import cv2,torch
    assert str(torch.__version__)==p['runtime']['torch']and cv2.__version__==p['runtime']['cv2']and Path(sys.executable).resolve()==Path(p['runtime']['python']).resolve()
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('v141_exact_R0_factory',p['backbone_code']['path']);factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    model=factory.FrozenYoloFeatures(p['checkpoint']['path'],device=p['runtime']['device'])
    def image(row):
        assert sha(row['image'])==row['image_sha256']
        if row['decoder']=='EXACT_V94_CV2_BGR':
            bgr=cv2.imread(row['image'],cv2.IMREAD_COLOR);assert bgr is not None;rgb=np.ascontiguousarray(bgr[:,:,::-1])
        else:
            assert row['decoder']=='CANONICAL_PIL_RGB_TO_BGR'
            from PIL import Image
            rgb=np.asarray(Image.open(row['image']).convert('RGB'));bgr=np.ascontiguousarray(rgb[:,:,::-1])
        assert rgb.dtype==np.uint8 and list(rgb.shape)==row['raw_hw']+[3]and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256']
        return bgr
    hooks=[0];handle=model.yolo.model.register_forward_pre_hook(lambda *_:hooks.__setitem__(0,hooks[0]+1))
    loaded=state(model.yolo.model);model.predict(np.zeros(tuple(rows[0]['raw_hw'])+(3,),np.uint8),already_padded=False,cpu_features=True);warm=state(model.yolo.model);warmhooks=hooks[0]
    write(OUT/'STARTED.json',dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),inputs=p['inputs'],loaded_state_SHA=loaded,warmed_state_SHA=warm,generated_blank_warmup_hooks=warmhooks,source_forward_calls=0,GT_read=False))
    proposals=[];queries=[];qpack={k:[]for k in query_arrays(dict(candidates=[],selected_index=None),[1,1])};offsets=[0];candidate_arrays={k:[]for k in('box_xyxy','keypoints_xy','keypoints_conf','score')};first=None;firstp3=None;start=time.monotonic()
    try:
        for i,row in enumerate(rows):
            z=model.predict(image(row),already_padded=False,cpu_features=True);s=serial(z);gain,offset=factory.canvas_affine(z['canvas_shape'],z['input_shape']);qa=query_arrays(z,row['raw_hw'])
            proposals.append(dict(**row,**s,affine_gain=gain,affine_offset=offset.tolist(),actual_original_initializer=True,source_emulation=False))
            for key,value in qa.items():qpack[key].append(value)
            for c in z['candidates']:
                for key in candidate_arrays:candidate_arrays[key].append(np.asarray(c[key],np.float64))
            offsets.append(offsets[-1]+len(z['candidates']))
            for k in range(8):queries.append(clean(dict(query_id=row['id']+':native'+str(k),frame_index=i,frame_id=row['id'],source=row['source'],source_id=row['source_id'],split=row['split'],native_slot=k,
                query_xy=qa['query_xy'][k],L=qa['L'][k],no_detection=z['selected_index']is None,finite_non_sentinel=qa['finite_non_sentinel'][k],native_continuous_valid=qa['continuous_valid'][k],
                query_nearest_pixel_valid=qa['nearest_valid'][k],query_RGB96_centre_domain_valid=qa['RGBcentre_valid'][k],native_grid_any_valid=qa['native_grid_valid'][k].any(),
                association_status='UNASSIGNED_BY_INPUT_PRODUCER',actual_original_initializer=True,source_emulation=False)))
            if i==0:first=s;firstp3=z['p3'].numpy().copy()
        repeat=model.predict(image(rows[0]),already_padded=False,cpu_features=True);assert serial(repeat)==first and repeat['p3'].numpy().tobytes()==firstp3.tobytes()
        assert state(model.yolo.model)==warm and sha(p['checkpoint']['path'])==p['checkpoint']['sha256']and all(not m.training for m in model.yolo.model.modules())and all(not p.requires_grad for p in model.yolo.model.parameters())
        for name,records in(('FROZEN_PROPOSALS_ALL16_INPUT_ONLY.jsonl',proposals),('ORIGINAL_NATIVE_QUERY128_INPUT_ONLY.jsonl',queries)):
            with(OUT/name).open('x')as f:
                for row in records:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        arrays={key:np.concatenate(value,axis=0)for key,value in qpack.items()};arrays.update(global_index=np.arange(128,dtype=np.int32),frame_index=np.repeat(np.arange(16,dtype=np.int32),8),native_slot=np.tile(np.arange(8,dtype=np.int8),16),
            query_id=np.array([r['query_id']for r in queries]),frame_id=np.repeat([r['id']for r in rows],8),split=np.repeat([r['split']for r in rows],8),no_detection=np.repeat([z['selected_index']is None for z in proposals],8))
        np.savez_compressed(OUT/'ORIGINAL_NATIVE_QUERY128_AND_VALID9X9_INPUT_ONLY.npz',**arrays)
        sizes={'box_xyxy':(4,),'keypoints_xy':(9,2),'keypoints_conf':(9,),'score':()}
        allc={key:np.stack(value)if value else np.empty((0,)+sizes[key],np.float64)for key,value in candidate_arrays.items()}
        np.savez_compressed(OUT/'ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz',frame_offsets=np.array(offsets,np.int32),selected_index=np.array([-1 if z['selected_index']is None else z['selected_index']for z in proposals],np.int32),**allc)
        handle.remove();model.close()
        result=dict(completed=True,PASS=True,frames=16,native_slots=128,no_detection_frames=sum(r['selected_index']is None for r in proposals),candidate_count=offsets[-1],
            finite_non_sentinel_slots=int(arrays['finite_non_sentinel'].sum()),continuous_valid_slots=int(arrays['continuous_valid'].sum()),nearest_valid_slots=int(arrays['nearest_valid'].sum()),
            finite_OOB_slots=int((arrays['finite_non_sentinel']&~arrays['continuous_valid']).sum()),all16_and128_retained=True,all_candidates_raw9_and_firstargmax_unchanged=True,centroid_excluded_from_native_q_and_L=True,
            warmup_state_SHA=warm,after_state_unchanged=True,checkpoint_unchanged=True,repeat_first_serial_and_P3_exact=True,no_saved_fullP3_duplicate=True,explicit_predict_calls=18,total_forward_hooks=hooks[0],generated_warmup_hooks=warmhooks,
            input_seal=p['inputs'],GT_pose_K_reference_semantic_mask_or_historic_score_reads=0,visibility_classifier_fit_private_V112_PnP_q_L_correction_calls=0,association_not_inferred=True,seconds=time.monotonic()-start)
        write(OUT/'RESULT.json',result)
        outputs=[bind(OUT/name)for name in('FROZEN_PROPOSALS_ALL16_INPUT_ONLY.jsonl','ORIGINAL_NATIVE_QUERY128_INPUT_ONLY.jsonl','ORIGINAL_NATIVE_QUERY128_AND_VALID9X9_INPUT_ONLY.npz','ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz','RESULT.json')]
        write(OUT/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),policy=e['policy'],protocol=bind(PROTOCOL),entry=bind(entry),inputs=p['inputs'],outputs=outputs,frames=16,rows=128,GT_read=False,historical_predictions_read=False,all16_frames128_native_slots_preserved=True,actual_original_initializer=True,source_emulation=False))
        print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'PREDICTION_SEAL.json'),result=bind(OUT/'RESULT.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use metadata/invented-only --prepare or explicit root --entry')
