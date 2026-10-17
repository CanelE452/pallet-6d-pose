"""Existing Blender RGB/cache provisioning with separate inference/GT streams.

All scene selection is identity/asset based before new candidate labels. No
model forward, fitted correction, candidate generation, or training occurs.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import builtins
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import time
import cv2
import numpy as np

LINE=Path('data/pallet/results/pallet_line_pose_v1')
DIM=Path('data/pallet/results/pallet_dim_conditioned_p_v1')
GEOMETRY=Path('challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz')
VIS=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
CALMODEL=VIS/'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/calibration/final_adapter_CAL46/MODEL.json'
N3PRED=DIM/'predictions/SYNTH_HELDOUT/N3_DIM_SYM_seed1.json'
VAL_ASSET='woodpallet_block_jtoastie_ccby.glb'
SUBPIX=dict(winSize=[5,5],zeroZone=[-1,-1],minimum_image_extent=15,criteria=dict(maxCount=40,epsilon=.001))


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def finite(x):
    if isinstance(x,dict):return{str(k):finite(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return[finite(v)for v in x]
    if isinstance(x,np.ndarray):return finite(x.tolist())
    if isinstance(x,np.generic):return finite(x.item())
    if isinstance(x,float)and not math.isfinite(x):return None
    return x


def write(p,x):
    with Path(p).open('x')as f:json.dump(finite(x),f,indent=2,allow_nan=False);f.write('\n')


def rows(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)


def save(p,records):
    with Path(p).open('xb')as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as z:
            for r in records:z.write((json.dumps(finite(r),separators=(',',':'),allow_nan=False)+'\n').encode())


def cost(start,cpu):return dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_model_forwards=0,new_training=0,new_rendering=0)


def functions(path,names,namespace):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert set(names)=={n.name for n in nodes}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
    return namespace


def select(repo,private):
    start,cpu=time.monotonic(),time.process_time();out=private/'synthetic_v1';out.mkdir()
    source=read(repo/LINE/'SOURCE_MANIFEST.json');side=read(repo/DIM/'DIMENSION_SIDECAR.json')['records'];assets={r['frame_id']:r['source_asset']for r in side}
    eligible=[r for r in source['records']if r['source']=='G38'and r['partition']=='heldout']
    assert len(eligible)==1025
    bysplit={'train':[],'validation':[]}
    for r in eligible:bysplit['validation'if assets[r['id']]==VAL_ASSET else 'train'].append(r)
    for key in bysplit:bysplit[key].sort(key=lambda r:hashlib.sha256(('P6_ASSET_SPLIT_V1:'+r['id']).encode()).hexdigest())
    assert len(bysplit['train'])==790 and len(bysplit['validation'])==235
    selected=bysplit['train'][:768]+bysplit['validation'][:235]
    pilot=set(r['id']for r in bysplit['train'][:384]+bysplit['validation'][:128]);selected_ids=set(r['id']for r in selected)
    assert len(pilot)==512 and len(selected_ids)==1003
    trainassets={assets[r['id']]for r in bysplit['train'][:768]};valassets={assets[r['id']]for r in bysplit['validation'][:235]};assert not trainassets&valassets
    lock=dict(schema='P6_SYNTHETIC_ASSET_GROUP_SPLIT_LOCK_V1',created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        selection_before_new_candidate_labels=True,original_upstream_population='heldout; now reused as NEW HEAD development only, not independent test',
        rule='G38 existing heldout only; whole source_asset holdout; SHA256(P6_ASSET_SPLIT_V1:+id) ascending within split; no prediction error or pose GT used',
        validation_asset=VAL_ASSET,train_assets=sorted(trainassets),validation_assets=sorted(valassets),
        proposal1024_executed=False,availability_amendment='Heldout G38=1025; validation asset235 cannot supply256. Pre-result expansion fixed1003=768train+235validation; proposed1024 not launched',
        pilot=dict(frames=512,train=384,validation=128,ids=[r['id']for r in selected if r['id']in pilot]),
        expansion=dict(frames=1003,train=768,validation=235,ids=[r['id']for r in selected]),
        all_rows=[dict(id=r['id'],source_index=r['index'],split='validation'if assets[r['id']]==VAL_ASSET else 'train',pilot=r['id']in pilot,source_asset=assets[r['id']],scenario_id=r['scenario_id'],original_upstream_partition=r['partition'])for r in selected],
        backbone_and_N3_frozen=True,training_seeds=[20261018,20261019,20261020],representative_seed=20261018,
        maximum_training_runs=6,maximum_frames4096=4096,no_GT_jitter_proposals=True,real_CAL_or_PRIMARY_training=False,
        original_source_sha256=sha(repo/LINE/'SOURCE_MANIFEST.json'),dimension_sidecar_sha256=sha(repo/DIM/'DIMENSION_SIDECAR.json'))
    write(out/'DATA_SPLIT_LOCK.json',lock);write(out/'SELECTION_RECEIPT.json',dict(cost=cost(start,cpu),lock_sha256=sha(out/'DATA_SPLIT_LOCK.json'),no_new_candidate_generation=True))
    print(json.dumps(dict(status='SPLIT_LOCKED',pilot=512,expansion=1003,source_asset_disjoint=True,lock_sha256=sha(out/'DATA_SPLIT_LOCK.json'))))


def provision(repo,private):
    start,cpu=time.monotonic(),time.process_time();out=private/'synthetic_v1';lock=read(out/'DATA_SPLIT_LOCK.json');source=read(repo/LINE/'SOURCE_MANIFEST.json')['records'];sidemap={r['frame_id']:r for r in read(repo/DIM/'DIMENSION_SIDECAR.json')['records']}
    predictions={r['id']:r for r in read(repo/N3PRED)['records']};geo=np.load(repo/GEOMETRY,allow_pickle=False);gi={str(x):i for i,x in enumerate(geo['stems'])}
    arrays={k:np.load(repo/LINE/f'cache/{k}.npy',mmap_mode='r')for k in ['points','point_valid','input_shape','gain','record_index','detected']}
    inputs=[];targets=[];quality=[];failures=[];sourcefiles=set();selected_rows_hash=hashlib.sha256();rotmax=0.;projectionmax=0.;canvasmax=0.;upstream_errors=[]
    for n,declared in enumerate(lock['all_rows']):
        fid=declared['id'];index=declared['source_index'];s=source[index];i=gi[fid];assert s['id']==fid and int(arrays['record_index'][index])==index
        label_path=repo/s['renderer_annotation_locator_provenance_only'];annotation=read(label_path);obj=annotation['objects'][0];camera=annotation['camera_data'];assert obj['source_asset']==declared['source_asset']
        raw_image=label_path.parent.parent/'rgb'/label_path.name.replace('_label.json','_rgb.png');prepared_path=Path(s['image'])
        if not raw_image.is_file():raise FileNotFoundError('Original renderer RGB required: '+str(raw_image))
        raw=cv2.imread(str(raw_image));prepared=cv2.imread(str(prepared_path));pad=int(s['reflect_pad_px']);h,w=s['raw_shape_hw']
        assert raw is not None and prepared is not None and list(raw.shape[:2])==[h,w]
        assert sha(prepared_path)==s['image_sha256'] and np.array_equal(prepared[pad:pad+h,pad:pad+w],raw)
        baseline_path=repo/DIM/f'source_baseline/{index:05d}.json';baseline=read(baseline_path);prediction=predictions[fid];assert baseline['id']==fid and baseline['selected_index']==prediction['selected_index']
        idx=baseline['selected_index'];q0=None if idx is None else np.asarray(baseline['candidates'][idx]['keypoints_xy'],float);qn=None if idx is None else np.asarray(prediction['candidates'][idx]['keypoints_xy'],float)
        support=np.zeros(9,bool)if q0 is None else np.isfinite(q0).all(-1)&~(q0==-1).all(-1)
        gain=float(arrays['gain'][index]);ih,iw=arrays['input_shape'][index];ph,pw=s['prepared_shape_hw'];recomputed_gain=min(640/ph,640/pw);offset=np.array([round((iw-round(pw*gain))/2-.1),round((ih-round(ph*gain))/2-.1)])
        assert gain==recomputed_gain
        if q0 is not None:
            # The cache is in letterbox coordinates after exactly one reflect pad.
            # Existing saved q0/qn are ALREADY unpadded original RGB coordinates.
            restored=(np.asarray(arrays['points'][index],float)-offset)/gain-pad
            gap=float(np.max(np.abs(restored-q0)));canvasmax=max(canvasmax,gap);assert gap<1e-4
            assert np.array_equal(q0[8],qn[8],equal_nan=True)
        K4=geo['K'][i];K=np.array([[K4[0],0,K4[2]-pad],[0,K4[1],K4[3]-pad],[0,0,1.]])
        intr=camera['intrinsics'];assert np.array_equal(K,np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]]))
        R=geo['R'][i];t=geo['t'][i];xyz=geo['Xcf'][i];cam=xyz@R.T+t;hom=cam@K.T;projected=hom[:,:2]/hom[:,2:];original=np.asarray(obj['projected_cuboid'],float)[:8]
        err=float(np.max(np.linalg.norm(projected-original,axis=1)));projectionmax=max(projectionmax,err);assert err<.05 and abs(err-float(geo['match_err'][i]))<1e-6
        M=np.asarray(obj['pose_transform']);assert np.array_equal(R,M[:3,:3])and np.array_equal(t,M[:3,3]);propergap=float(np.max(np.abs(R.T@R-np.eye(3))));rotmax=max(rotmax,propergap);assert propergap<1e-5 and np.linalg.det(R)>0
        dimensions=np.asarray(geo['dims'][i]);assert np.array_equal(dimensions[[0,2,1]],np.asarray(sidemap[fid]['canonical_WDH']))
        # Real renderer labels may use a camera-facing dynamic native ordering.
        # Xcf is a GT-only table; it NEVER becomes candidate input geometry.
        target=dict(id=fid,split=declared['split'],pilot=declared['pilot'],R_fixed=R.tolist(),t=t.tolist(),physical_dimensions_xyz_m=dimensions.tolist(),
            native_GT_xyz=xyz.tolist(),projected_GT_native_xy=original.tolist(),symmetry_order=int(sidemap[fid]['symmetry_order']),
            source_asset=declared['source_asset'],source_contract=sidemap[fid]['contract_basis'],reference_kind='BLENDER_RENDERER_CAMERA_POSE_AND_REGISTERED_VERTEX_PROJECTION',
            original_renderer_R_retained=True,renderer_annotation_sha256=sha(label_path),geometry_sidetable_row=i,raw_hw=[h,w],K=K.tolist())
        targets.append(target)
        inputs.append(dict(id=fid,partition='SYNTHETIC',split=declared['split'],pilot=declared['pilot'],session='SYNTHETIC_NO_SESSION_FEATURE',
            image=str(raw_image),image_sha256=sha(raw_image),raw_hw=[h,w],K=K.tolist(),known_physical_dimensions_xyz_m=dimensions.tolist(),
            raw_YOLO_points=None if q0 is None else q0.tolist(),N3_points=None if qn is None else qn.tolist(),prediction_support=support.tolist(),
            upstream_source='EXACT_SAVED_YOLO_AND_N3_DIM_SYM_SEED1_RAW_COORDINATES',source_basis=True,
            original_detector_available=idx is not None,selected_candidate_score=None if idx is None else baseline['candidates'][idx]['score'],
            no_reference_or_human_or_GT_mask_inputs=True))
        if q0 is not None:upstream_errors.extend(np.linalg.norm(q0[:8]-original,axis=1).tolist())
        selected_rows_hash.update(np.asarray(arrays['points'][index]).tobytes());selected_rows_hash.update(np.asarray(arrays['point_valid'][index]).tobytes())
        sourcefiles.update([label_path,raw_image,prepared_path,baseline_path])
        quality.append(dict(id=fid,split=declared['split'],pilot=declared['pilot'],source_asset=declared['source_asset'],render_projection_max_px=err,renderer_R_orthogonality_max_abs=propergap,
            rawRGB_equals_prepared_crop=True,raw_coordinate_cache_roundtrip_max_px=gap if q0 is not None else None,detected=idx is not None,
            all_real_upstream_errors_retained=True,targets_used_in_candidate_inputs=False))
        if n%128==0:print('SYNTHETIC_CONTRACT_VERIFIED',n+1,len(lock['all_rows']),flush=True)
    save(out/'UPSTREAM_INPUTS_NO_GT.jsonl.gz',inputs);save(out/'SYNTH_GT_LABELS.jsonl.gz',targets);save(out/'DATA_QUALITY_ROWS.jsonl.gz',quality)
    relevant=[repo/LINE/'SOURCE_MANIFEST.json',repo/LINE/'cache/CACHE_MANIFEST.json',repo/LINE/'cache/CACHE_COMPLETE.json',repo/DIM/'DIMENSION_SIDECAR.json',repo/N3PRED,repo/GEOMETRY,repo/CALMODEL,
        repo/'scripts/research/pallet_translation_loss_v1/build_geometry_sidetable.py',repo/'scripts/data_prep/blender/render_blender_data.py',repo/'scripts/data_prep/blender/blender_config.py',repo/'data/pallet/training_data/paper_release/README.md']
    # Existing large feature arrays are not copied or rehashed in full. Exact
    # selected point/support bytes and bound completion manifests identify reuse.
    write(out/'PROVISION_SEAL.json',dict(schema='P6_SYNTHETIC_SOURCE_PROVISION_SEAL_V1',rows=len(inputs),pilot512=True,
        split_counts=dict(Counter(r['split']for r in inputs)),source_geometry_convention='Fixed rendererXYZ allowed as known dimensions. NativeGTxyz only target stream; originalcamera_dynamic channels separately authenticated byprojection',
        annotation_vs_projection_max_px=projectionmax,renderer_R_orthogonality_max_abs=rotmax,cache_to_raw_roundtrip_max_px=canvasmax,
        selected_cache_points_support_bytes_sha256=selected_rows_hash.hexdigest(),all_rawRGB_equals_exact_prepared_center_crop=True,
        no_GT_mask_features=True,no_primary_or_CAL_reference_reads=True,new_model_forwards=0,new_rendering=0,
        actual_error_support=dict(raw_YOLO_corners=len(upstream_errors),median_px=float(np.median(upstream_errors)),p90_px=float(np.quantile(upstream_errors,.9)),gt_jitter_substitution=False),
        artifacts={n:sha(out/n)for n in ['UPSTREAM_INPUTS_NO_GT.jsonl.gz','SYNTH_GT_LABELS.jsonl.gz','DATA_QUALITY_ROWS.jsonl.gz','DATA_SPLIT_LOCK.json']},
        source_sha256={str(p):sha(p)for p in relevant},individual_files=[dict(path=str(p),sha256=sha(p))for p in sorted(sourcefiles)],cost=cost(start,cpu),failures=failures))
    print(json.dumps(dict(status='PROVISIONED_SEPARATE_INPUT_GT',rows=len(inputs),cost=cost(start,cpu),projection_max_px=projectionmax,raw_roundtrip_max_px=canvasmax)))


@contextmanager
def no_answers():
    old=builtins.open,io.open,Path.open
    def wrap(fn):
        def guarded(p,*args,**kwargs):
            mode=kwargs.get('mode',args[0]if args else 'r')
            if isinstance(p,(str,Path))and not any(c in str(mode)for c in 'wax+')and any(t in str(p)for t in ['SYNTH_GT_LABELS','GEOMETRY_SIDETABLE','/labels/','ISOLATED_','FIXED_REFERENCES','SCORED_ROWS','FAIR_ROW_METRICS']):raise PermissionError('Candidate input production forbids GT read: '+str(p))
            return fn(p,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,old)
    try:yield
    finally:builtins.open,io.open,Path.open=old


def packets(repo,private,pilot_only=False):
    start,cpu=time.monotonic(),time.process_time();out=private/'synthetic_v1';seal=read(out/'PROVISION_SEAL.json');path=out/'UPSTREAM_INPUTS_NO_GT.jsonl.gz';assert sha(path)==seal['artifacts'][path.name]
    sys.path.insert(0,str(repo));cv2.setNumThreads(1)
    packets=[];fail=[];counts=Counter()
    with no_answers():
        from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
        from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_api_v2 import registered_geometry
        ns=dict(np=np,cv2=cv2,select_pnp_hypotheses=select_pnp_hypotheses)
        functions(repo/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',['cuboid','solve'],ns)
        functions(repo/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',['rotations','infer'],ns)
        sub=functions(repo/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py',['_inputs','cap_points','correct'],dict(np=np,cv2=cv2,Counter=Counter,METHODS=('SUBPIX','CVRANK'),_configuration=lambda:{'SUBPIX':SUBPIX}))
        model=read(repo/CALMODEL)
        for r in rows(path):
            if pilot_only and not r['pilot']:continue
            fid=r['id'];q0=np.asarray(r['raw_YOLO_points'],float)if r['raw_YOLO_points']is not None else None;qn=np.asarray(r['N3_points'],float)if r['N3_points']is not None else None;support=np.asarray(r['prediction_support'],bool);h,w=r['raw_hw'];K=np.asarray(r['K']);dims=np.asarray(r['known_physical_dimensions_xyz_m'])
            if sha(r['image'])!=r['image_sha256']:raise ValueError('Frozen RGB changed')
            image=cv2.imread(r['image']);assert image is not None and list(image.shape[:2])==[h,w]
            packet=dict(r,schema='SYNTHETIC_PREDICTION_ONLY_STATEWISE_INPUT_V1',reference_inputs=False,GT_mask_inputs=False,raw_coordinates_already_unpadded=True)
            if q0 is None:
                packet.update(initial_pose=dict(available=False),solver_input=None,technical_status='NO_DETECTION_RETAINED');packets.append(packet);counts['no_detection']+=1;continue
            native,diag=sub['correct'](cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),qn,support,'SUBPIX');q=sub['cap_points'](q0,native,w,h,support);counts['SubPix_images']+=1;counts['SubPix_corner_calls']+=diag['algorithm_corner_calls']
            pose=ns['infer'](q,K,dims,source=True);counts['F_calls']+=1;packet.update(initial_points=q.tolist(),initial_pose=pose,N3_SUBPIX_points=q.tolist(),SubPix=diag)
            if not pose.get('available'):
                packet.update(solver_input=None,technical_status='N0_F_UNAVAILABLE_RETAINED');packets.append(packet);counts['unavailable_F']+=1;continue
            xyz=ns['cuboid'](*pose['cf_extents']);g=registered_geometry([h,w],K,xyz,pose);states=[];visible=[];query=[]
            for k in range(8):
                if g['initial_self_hidden'][k]:state='S'
                elif not support[k]or not np.isfinite(q[k]).all()or np.array_equal(q[k],[-1,-1]):state='U'
                elif not(0<=q[k,0]<w and 0<=q[k,1]<h):state='T'
                else:state=g['initial_roles'][k]
                states.append(state)
                if state in ('B','I'):query.append(q[k])
            scale=float(np.linalg.norm(np.ptp(query,axis=0)))if len(query)>1 else 1.;ratio=float(np.clip(scale/model['reference_scale_px'],.5,2))
            for k,state in enumerate(states):
                if state not in ('B','I'):continue
                sigma=2. if state=='B'else 3.;cov=sigma*sigma*np.eye(2);group=model['groups'][state+':fallback'];effective=model['covariance_scale']*cov+group['floor_px2']*ratio*ratio*np.eye(2)
                visible.append(dict(native_corner=k,state=state,xy=q[k].tolist(),covariance_px2=cov.tolist(),fit_covariance_px2=effective.tolist(),fit_reliability_weight=group['reliability_weight'],source='VISIBLE_BACKBONE_FALLBACK',image_measurement=True,measurement_is_pose_projection=False,conditional_precision_calibrated=False,synthetic_state_policy='COARSE_GEOMETRY_ROLE_AND_NATIVE_QUERY_CANVAS; no independently learned external O classification'))
            packet.update(solver_input=dict(xyz=xyz.tolist(),K=K.tolist(),image_hw=[h,w],states=states,visible=visible,boundary_observations=[],domains=[],u_regions=[],outline_observation=dict(supported=False,reason='NO_RENDERER_GT_MASK_ALLOWED; automatic B producer may supply RGB-only support'),coarse_pose=dict(R=pose['R_cf'],t=pose['centroid'],physical_registration_Q=g['physical_registration_Q'].tolist()),config=dict(covariance_scale=1.,huber_scale=3.,prior_weight=0.)),
                registered_family=dict(name='N0_PREDICTION_SELECTED_WD',native_xyz=xyz.tolist(),cf_extents=pose['cf_extents'],physical_registration_Q=g['physical_registration_Q'].tolist()),
                states=states,activation=dict(complete_ids=[v['native_corner']for v in visible],all_partial_ids=[],typed_partial_ids=[]),technical_status='AVAILABLE',
                observation_policy=dict(S_from_coarse_geometry=True,B_I_geometry_role_not_actual_visibility=True,external_occlusion_classifier_executed=False,renderer_visibility_or_masks_used=False,source_covariance='inherited fallback B2px/I3px',effective_covariance='same frozen CAL adapter, no newfit',source_basis_true_Rxpi_physical=True))
            packets.append(packet);counts['available']+=1
            if len(packets)%128==0:print('SYNTHETIC_NO_GT_PACKETS',len(packets),flush=True)
        name='PILOT512_NO_GT_PACKETS.jsonl.gz'if pilot_only else 'EXPANSION1003_NO_GT_PACKETS.jsonl.gz';save(out/name,packets)
    write(out/(name.replace('.jsonl.gz','_SEAL.json')),dict(schema='SYNTHETIC_STATEWISE_INPUT_PRODUCTION_SEAL_V1',rows=len(packets),before_new_candidate_labels=True,reference_reads=0,GT_masks_used=False,predictions_sha256=sha(out/name),input_sha256=sha(path),provision_seal_sha256=sha(out/'PROVISION_SEAL.json'),model_sha256=sha(repo/CALMODEL),operation_counts=dict(counts),all_failures_retained=True,cost=cost(start,cpu)))
    print(json.dumps(dict(status='NO_GT_PACKETS_SEALED',rows=len(packets),file=name,counts=counts,cost=cost(start,cpu))))


def root_frames(repo,private,pilot_only=True):
    """Convert frozen predictor-only packets to the REAL generator's API.

    The original F's two W/D fits are actual prediction-only upstream outputs,
    not GT-derived jitter or successful-candidate insertions. The common root
    generator receives exactly this same bank for all arms.
    """
    from copy import deepcopy
    from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_two_wd_family_v4_9 import make_wd_families
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    start,cpu=time.monotonic(),time.process_time();out=private/'synthetic_v1';cv2.setNumThreads(1)
    input_name='PILOT512_NO_GT_PACKETS.jsonl.gz'if pilot_only else 'EXPANSION1003_NO_GT_PACKETS.jsonl.gz'
    source=out/input_name;inputseal=read(out/input_name.replace('.jsonl.gz','_SEAL.json'));assert sha(source)==inputseal['predictions_sha256']
    result=[];counts=Counter();RX=np.diag([1.,-1.,-1.]);RY=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    with no_answers():
        for r in rows(source):
            if r['solver_input']is None:raise ValueError('Explicit unavailable-frame adapter required; preserve row '+r['id'])
            packet=deepcopy(r)
            for measurement in packet['solver_input']['visible']:measurement.setdefault('protection_px',2.)
            # Existing OT parser: T has a broad outside-canvas union, with no
            # finite location or asserted known side. It is prediction-conditional.
            packet['solver_input']['domains']=[dict(native_corner=k,state='T',shape='halfplane',outside_canvas_side='unknown',outside_margin_px=0.,halfplanes=[],canvas_hw=r['raw_hw'],provenance=dict(source='PREDICTION_CONDITIONAL_NATIVE_QUERY_OUTSIDE_CANVAS',independent_physical_truncation_certified=False,no_finite_T_localization=True,GT_inputs=False))for k,state in enumerate(r['states'])if state=='T']
            pose=r['initial_pose'];Q=np.asarray(r['registered_family']['physical_registration_Q']);coarse=dict(R=pose['R_cf'],t=pose['centroid'],physical_registration_Q=Q.tolist(),R_physical=pose['R_physical'])
            families={f['name']:f for f in make_wd_families(r['solver_input']['xyz'],coarse)}
            packet.update(original_e_legacy_boundary_observations=[],original_e_stage1_family_seed_bank={name:dict(family=f)for name,f in families.items()},registered_family=families['ORIGINAL_WD'],provenance=dict(original_mask_binding=None,synthetic_predictor_only=True,no_GT_mask_features=True))
            packet['solver_input']['outline_observation']=dict(supported=False,configuration={},reason='NO_AUTOMATIC_MASK_YET; renderer target masks forbidden as inference features')
            candidates=[dict(candidate_id='N0',source='N0',R_physical=pose['R_physical'],centroid=pose['centroid'],original_native_family=None)]
            dims=np.asarray(r['known_physical_dimensions_xyz_m']);q=np.asarray(r['initial_points']);K=np.asarray(r['K'])
            selection=select_pnp_hypotheses(q,K,dict(x=max(dims[0],dims[2]),y=dims[1],z=min(dims[0],dims[2])),None);counts['upstream_F_WD_hypothesis_selector_calls']+=1
            for h in selection.hypotheses:
                if not h.success:counts['unavailable_WD_hypotheses']+=1;continue
                d=h.camera_facing_dimensions.as_dict();cf=np.array([d['width'],d['height'],d['depth']]);R=np.asarray(h.rotation_camera_facing);t=np.asarray(h.translation_camera_facing)
                if t[2]<=0 or not np.isfinite(R).all()or not np.isfinite(t).all():counts['invalid_WD_hypotheses']+=1;continue
                Qwd=np.eye(3)if abs(cf[0]-dims[0])<1e-6 else RY
                physical=R@Qwd@RX
                candidates.append(dict(candidate_id='UPSTREAM_F/'+h.name,source='F_WD_HYPOTHESIS',R_physical=physical.tolist(),centroid=t.tolist(),original_native_family=None))
            trace=dict(id=r['id'],states=r['states'],initial_points=r['initial_points'],native_xyz=r['solver_input']['xyz'],K=r['K'],raw_hw=r['raw_hw'],initial_pose=pose,mask_binding=None,rays=[])
            frame=dict(id=r['id'],partition='SYNTH_TRAIN'if r['split']=='train'else 'SYNTH_VAL',split=r['split'],pilot=r['pilot'],session='SYNTHETIC_OPAQUE_METADATA',cluster_id='SYNTHETIC_OPAQUE_METADATA',image=r['image'],image_sha256=r['image_sha256'],packet=packet,trace=trace,families=families,candidates=candidates,fallback=pose,data_role='synthetic_HEAD_development_asset_group_split',supervised_inputs=False)
            result.append(frame);counts['frames']+=1;counts['carried_before_shared_dedup']+=len(candidates)
        name='PILOT512_ROOT_FRAMES_NO_GT.jsonl.gz'if pilot_only else 'EXPANSION1003_ROOT_FRAMES_NO_GT.jsonl.gz';save(out/name,result)
    write(out/name.replace('.jsonl.gz','_SEAL.json'),dict(schema='SYNTHETIC_ROOT_FRAME_ADAPTER_SEAL_V1',rows=len(result),predictions_sha256=sha(out/name),input_sha256=sha(source),source_sha256=sha(__file__),before_candidate_label_reads=True,reference_reads=0,GT_masks_used=False,operation_counts=dict(counts),same_root_candidategenerator_required=True,cost=cost(start,cpu)))
    print(json.dumps(dict(status='ROOT_FRAMES_SEALED',name=name,rows=len(result),counts=counts,cost=cost(start,cpu))))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','provision','pilot_packets','expansion_packets','pilot_frames','expansion_frames']);p.add_argument('--repo',required=True,type=Path);p.add_argument('--private',required=True,type=Path);a=p.parse_args()
    if a.stage=='select':select(a.repo,a.private)
    elif a.stage=='provision':provision(a.repo,a.private)
    elif a.stage in ('pilot_frames','expansion_frames'):root_frames(a.repo,a.private,a.stage=='pilot_frames')
    else:packets(a.repo,a.private,a.stage=='pilot_packets')


if __name__=='__main__':main()
