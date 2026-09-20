"""Append-only consolidation of frozen DCP artifacts; no training or selection.

Run with the pallet-yolo26 Python environment: prepare, score, runtime, build.
Only runtime requires CUDA. Existing inputs and generated_tables are read-only.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / 'scripts/research/pallet_dim_conditioned_p_v1'
sys.path.insert(0, str(CODE))
import dcp_env as E
OUT = ROOT / '_docs/experiments/pallet_final_paper_tables_v1'
TEX = ROOT / '_docs/paper/sensors_submission_v1/generated_tables_v2'
ARMS = ['OLD_P', 'N2_DIM_ONLY', 'N3_DIM_SYM']
NAMES = [f'{a}_seed{s}' for a in ARMS for s in (1, 2, 3)]
SQUARE = [f'{a}_seed{s}' for a in ['S0_FIXED', 'S1_SYM'] for s in (1, 2, 3)]
read = E.read

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    assert path.is_relative_to(OUT) or path.is_relative_to(TEX)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text() == text, ('Refusing overwrite', str(path))
    else:
        with path.open('x') as f:
            f.write(text)

def bound(path):
    return E.bound(path)

def ckpath(name):
    arm, seed = name.rsplit('_seed', 1)
    return E.C.BRAW/f'runs/seed{seed}/last.pt' if arm == 'OLD_P' else E.RAW/f'runs/{name}/last.pt'

def verify_inputs():
    source = read(OUT/'SOURCE_BINDING.json')
    E.verify(source['inputs'])
    assert git('branch', '--show-current') == 'main'
    assert digest(git('diff', '--binary')) == source['initial_tracked_diff_sha256']
    assert digest(git('diff', '--cached', '--binary')) == source['initial_staged_diff_sha256']
    return source

def prepare():
    assert git('branch', '--show-current') == 'main'
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main')
    assert not OUT.exists() and not TEX.exists()
    paths = {E.R0, *(ckpath(n) for n in NAMES)}
    for dirname in [CODE, E.DOC, ROOT/'_docs/paper/sensors_submission_v1/generated_tables']:
        paths.update(p for p in dirname.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update([E.LINE/'baseline/FULL_CANDIDATES.json', E.C.B/'P_SELECTION.json',
        E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json', E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',
        E.C.POSE/'AXIS_REVIEW_MANIFEST.json', E.RAW/'REAL_DEV_METRICS.json', E.RAW/'PAPER_POSE_METRICS.json', E.RAW/'SQUARE_METRICS.json',
        ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json', ROOT/'challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json',
        ROOT/'data/pallet/results/paper_eval_v1/arms/R0.json', ROOT/'data/pallet/results/paper_eval_v1/arms/ARM_RESULTS.json',
        ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json', ROOT/'challenge/real_gt_v2/manifests/DEV_NEG2689.json'])
    for n in NAMES:
        paths.update([E.RAW/f'predictions/REAL_DEV/{n}.json', E.RAW/f'pose_predictions/REAL_DEV/{n}.json'])
    for n in SQUARE:
        paths.add(E.RAW/f'predictions/SQUARE_DEV/{n}.json')
    for dirname in ['scripts/research/pallet_line_pose_v1', 'scripts/research/pallet_final_ml_contribution_test_v1', 'scripts/paper/pose_metric_closure_v1', 'challenge/evaluation_v2']:
        paths.update((ROOT/dirname).glob('*.py'))
    manifest = read(ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    assert manifest['role'] == 'DEV' and len(manifest['items']) == 319
    for r in manifest['items']:
        paths.add(ROOT/r['gt_v2_path'])
    for r in read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']:
        paths.add(ROOT/r['annotation'])
    for r in read(E.DOC/'DEV_CACHE_COMPLETE.json')['records']:
        paths.add(ROOT/r['path'])
    for k in read(E.DOC/'RUNTIME_PROTOCOL.json')['keys']:
        assert k in {r['image_path'] for r in manifest['items']}
        paths.add(ROOT/k)
    # Verify historical artifact revisions contain precisely the current metric code.
    history = {}
    for artifact, codes in {
        'REAL_DEV_RESULTS.json': ['eval_math.py', 'dev_evaluate.py'],
        'PAPER_POSE_RESULTS.json': ['pose.py'],
        'SQUARE_RESULTS.json': ['eval_math.py', 'secondary_evaluate.py'],
    }.items():
        rel = str((E.DOC/artifact).relative_to(ROOT))
        rev = git('log', '-1', '--format=%H', '--', rel)
        checks = {}
        for name in codes:
            code = CODE/name
            old = subprocess.check_output(['git','show',f'{rev}:{code.relative_to(ROOT)}'], cwd=ROOT)
            checks[name] = hashlib.sha256(old).hexdigest() == E.sha(code)
            assert checks[name], (artifact, name, 'metric code changed')
        history[artifact] = dict(artifact_commit=rev, metric_code_exact_at_artifact_commit=checks)
    assert E.sha(E.R0) == E.R0_SHA
    write(OUT/'SOURCE_BINDING.json', dict(schema='pallet_final_paper_tables_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_origin_main=git('rev-parse','origin/main'), start_HEAD=git('rev-parse','HEAD'), branch='main',
        initial_status=git('status','--short'), initial_tracked_diff_sha256=digest(git('diff','--binary')), initial_staged_diff_sha256=digest(git('diff','--cached','--binary')),
        inputs=[bound(p) for p in sorted(paths)], historical_evaluator_verification=history,
        main_population='reused DEV319; no new FINAL/sealed membership accessed',
        legacy_final_named_paths='Some existing DEV319 members have final/positive in their pathname; authorization is membership-based, no new sealed population opened.',
        AP_population='Existing artifact only: same DEV319 positives + historical DEV_NEG2689; no negative image inference',
        no_training=True, optimizer_updates=0, no_parameter_selection=True))
    print('SOURCE_BOUND', len(paths), flush=True)

def summary_pose(rows):
    from pose import pose_auc
    available = [r for r in rows if r['available']]
    s = dict(frames=len(rows), available=len(available), coverage=len(available)/len(rows),
        ADDsym_AUC_full=pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows],1.),
        ADDsym_AUC_conditional=pose_auc([r['ADDsym_normalized'] for r in available],1.) if available else None)
    for k in ['translation_cm','rotation_deg','yaw_deg','IoU3D']:
        s[k] = dict(median=float(np.median([r[k] for r in available])), P90=float(np.quantile([r[k] for r in available],.9)))
    return s

def assert_close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for k in a: assert_close(a[k],b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x,y in zip(a,b): assert_close(x,y)
    elif isinstance(a, (int,float)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=0,atol=1e-12), (a,b)
    else: assert a == b, (a,b)

def score():
    import torch, cv2
    from dev_evaluate import population_metadata, iou
    from eval_math import measure, summary
    from inference import preservation
    from pose import metadata, infer, metric
    verify_inputs(); torch.set_num_threads(4); cv2.setNumThreads(1)
    pe,pop = population_metadata()
    ids = [item.frame_id for item,_ in pop]; assert len(ids) == len(set(ids)) == 319
    baseline = read(E.LINE/'baseline/FULL_CANDIDATES.json'); assert baseline['weights_sha256'] == E.R0_SHA
    groups = {r['object_type']:r for r in read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    targets = {}; bases = []; cache = {}
    for r in read(E.DOC/'DEV_CACHE_COMPLETE.json')['records']:
        cache[r['id']] = torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
    for item,m in pop:
        key = pe.canonical_key(item.image); cap = cache[item.frame_id]['captured']; c = baseline['frames'][key]
        captured_candidates = __import__('inference').serial(cap['candidates'])
        assert len(c) == len(captured_candidates)
        for old, current in zip(c, captured_candidates):
            for field in old: assert_close(old[field], current[field])
        c = captured_candidates
        idx = cap['selected_index']; assert idx == (int(np.argmax([x['score'] for x in c])) if c else None)
        bases.append(dict(id=item.frame_id,key=key,candidates=c,selected_index=idx))
        targets[item.frame_id] = (pe.E._legacy_forbidden_target(item), m)
    predictions = {'R0': bases}
    for n in NAMES:
        p = read(E.RAW/f'predictions/REAL_DEV/{n}.json')
        assert p['complete'] and not p['GT_input']
        assert [r['id'] for r in p['records']] == ids
        predictions[n] = p['records']
    allrows = {}; sums = {}; equality = {}
    for name,preds in predictions.items():
        rows=[]; det_before=[];det_after=[]
        for base,p in zip(bases,preds):
            assert p['selected_index'] == base['selected_index']
            preservation(base['candidates'],p['candidates'],p['selected_index'])
            def projection(r):
                return dict(id=r['id'],selected_index=r['selected_index'],candidates=[{k:v for k,v in c.items() if k!='keypoints_xy'} for c in r['candidates']])
            det_before.append(projection(base));det_after.append(projection(p))
            t,m = targets[p['id']]; idx=p['selected_index'];c=None if idx is None else p['candidates'][idx]
            matched = c is not None and iou(c['box_xyxy'],t.box_xyxy)>=.5
            g=groups[m['object_type']];assert g['group_order']==2
            row=measure(np.full((9,2),np.nan) if c is None else c['keypoints_xy'],t.keypoints_xy,t.keypoint_supervision_mask,
                g['permutations'],cache[p['id']]['raw_hw'],matched,idx is not None)
            row.update(id=p['id'],session=m['session_id'],group='C2',object=m['object_type'])
            rows.append(row)
        allrows[name]=rows;sums[name]=summary(rows)
        equality[name]=dict(frames=len(rows),exact=True,baseline_detection_sha256=digest(det_before),arm_detection_sha256=digest(det_after),
            fields='all candidate fields except selected first eight keypoint xy; order, score, box, selected index, center and nonselected keypoints exact',
            class_contract='single-class pallet implicit in serialized candidates; no class field is modified or introduced')
        assert equality[name]['baseline_detection_sha256']==equality[name]['arm_detection_sha256']
        if name!='R0': assert_close(sums[name],read(E.DOC/'REAL_DEV_RESULTS.json')['summary'][name])
        print('2D',name,sums[name]['PCK']['10'],flush=True)
    denominator_keys=['total_frames','evaluable_frames','detected','matched','missing','corners','observed_corners']
    denominators={k:sums['R0'][k] for k in denominator_keys}
    for s in sums.values():assert {k:s[k] for k in denominator_keys}==denominators
    write(OUT/'RESCORED_2D.json',dict(summary=sums,rows=allrows,denominators=denominators,population_order_sha256=digest(ids),all_existing_DCP_rows_exact=True))
    write(OUT/'DETECTION_PRESERVATION.json',dict(arms=equality,all_exact=True,code_review='replace_selected deep-copies candidates and changes only selected keypoints_xy; preservation checks every remaining field, order and argmax-selected instance',
        AP_inference='unchanged from R0 by output-preservation contract; not a separate N3 AP measurement',new_negative_inference=0))
    # Current DCP pose solver only for the missing R0 row. No old pose summary is merged.
    meta,gt=metadata('REAL_DEV');assert set(meta)==set(gt)==set(ids)
    r0poses={};r0metrics=[]
    for r in bases:
        idx=r['selected_index'];fid=r['id'];points=None if idx is None else r['candidates'][idx]['keypoints_xy']
        r0poses[fid]=infer(points,*meta[fid]);r0metrics.append(metric((fid,r0poses[fid],gt[fid])))
    poses={'R0':summary_pose(r0metrics)}
    existing=read(E.DOC/'PAPER_POSE_RESULTS.json')['summary']['REAL_DEV']
    oldmetrics=read(E.RAW/'PAPER_POSE_METRICS.json')['REAL_DEV']
    for n in NAMES:
        assert set(r['id'] for r in oldmetrics[n])==set(ids) and len(oldmetrics[n])==319
        assert_close(summary_pose(oldmetrics[n]),existing[n]);poses[n]=existing[n]
    write(OUT/'R0_POSE.json',dict(summary=poses['R0'],predictions=r0poses,metrics=r0metrics,reference='geometry-reconstructed 6D reference; NOT independent physical GT',GT_matching_gate=False))
    write(OUT/'POSE_SUMMARIES.json',dict(summary=poses,reused=NAMES,newly_computed=['R0'],all_population_ids_equal=True,reference='geometry-reconstructed 6D reference; NOT independent physical GT'))
    square=read(E.DOC/'SQUARE_RESULTS.json');sqmetrics=read(E.RAW/'SQUARE_METRICS.json');sq={};sqids=None
    for n in SQUARE:
        rr=sqmetrics[n];assert all(r['group']=='C4' for r in rr)
        current=[r['id'] for r in rr]
        if sqids is None:sqids=current
        assert current==sqids and len(current)==len(set(current))==155
        assert_close(summary(rr),square['summary'][n]);sq[n]=square['summary'][n]
        E.verify([square['predictions'][n]])
    assert square['constant_dimensions'] and square['real_supervised_secondary'] and not square['dimension_effect_identifiable']
    write(OUT/'SQUARE_REUSE_AUDIT.json',dict(summary=sq,population_order_sha256=digest(sqids),frames=155,all_summaries_exact=True,real_supervised_secondary=True,constant_dimensions=True,dimension_utility_identifiable=False))
    params={}
    from refiner import model
    for n in NAMES:
        ck=torch.load(ckpath(n),map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==6000 and ck['baseline_checkpoint_sha256']==E.R0_SHA
        arm=n.rsplit('_seed',1)[0];head=model(arm,ck['config']);head.load_state_dict(ck['model_state_dict'],strict=True)
        params[n]=sum(p.numel() for p in head.parameters() if p.requires_grad)
        if arm=='N3_DIM_SYM': assert params[n]==20259
    write(OUT/'PARAMETER_AUDIT.json',dict(trainable_architecture_parameters=params,new_optimizer_updates=0,new_training=0))
    verify_inputs();print('SCORE_COMPLETE',flush=True)

@__import__('torch').no_grad()
def runtime():
    import torch, cv2
    from inference import load_head, predict_captured, registry_input, serial, preservation
    from pose import infer
    verify_inputs();device=E.gpu();torch.set_num_threads(4);cv2.setNumThreads(1)
    protocol=read(E.DOC/'RUNTIME_PROTOCOL.json');keys=protocol['keys'];assert len(keys)==26
    assert (protocol['warmup'],protocol['repeats'],protocol['threads'],protocol['opencv'])==(20,5,4,1)
    pe=E.old('paper_evaluation');frames={pe.canonical_key(r['image']):r for r in read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']}
    images={};meta={}
    base=read(E.LINE/'baseline/FULL_CANDIDATES.json')
    baseline_full={}
    for entry in read(E.DOC/'DEV_CACHE_COMPLETE.json')['records']:
        if entry['key'] in keys:
            row=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False)
            baseline_full[entry['key']]=serial(row['captured']['candidates'])
    for k in keys:
        assert E.sha(ROOT/k)==base['frame_metadata'][k]['image_sha256']
        images[k]=cv2.imread(str(ROOT/k));assert images[k] is not None
        r=frames[k];intr=read(ROOT/r['annotation'])['camera_data']['intrinsics'];d,g=registry_input(r['object_type'])
        meta[k]=(np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],float),d,g)
    head,path=load_head('N3_DIM_SYM',1)
    assert not any(p.requires_grad for p in head.parameters())
    extractor=E.old('features').FrozenYoloFeatures(E.R0)
    norm=read(E.DOC/'DIM_NORMALIZATION_LOCK.json');selection=read(E.DOC/'CALIBRATION_AND_SELECTION.json')
    saved={r['key']:r for r in read(E.RAW/'predictions/REAL_DEV/N3_DIM_SYM_seed1.json')['records']}
    arms=['R0','N3_DIM_SYM'];records=[]
    def run(arm,k):
        torch.cuda.synchronize();begin=time.perf_counter();cap=extractor.predict(images[k]);torch.cuda.synchronize();captured=time.perf_counter()
        K,d,g=meta[k]
        if arm=='R0':
            result=dict(candidates=cap['candidates'],selected_index=cap['selected_index']);refined=captured
        else:
            result,_=predict_captured(head,arm,cap,d,g,selection['temperatures'][arm+'_seed1']['temperature'],selection['rule'],images[k].shape[:2],norm)
            torch.cuda.synchronize();refined=time.perf_counter()
        idx=result['selected_index'];p=infer(None if idx is None else result['candidates'][idx]['keypoints_xy'],K,d[[0,2,1]])
        torch.cuda.synchronize();end=time.perf_counter()
        return result,dict(full_ms=(end-begin)*1000,R0_ms=(captured-begin)*1000,P_only_ms=(refined-captured)*1000,PnP_ms=(end-refined)*1000,pose_available=p['available'])
    parity=[]
    try:
        for arm in arms:
            for k in keys:
                result,_=run(arm,k);expected=baseline_full[k] if arm=='R0' else saved[k]['candidates']
                actual=serial(result['candidates']);assert len(actual)==len(expected)
                for a,b in zip(actual,expected):
                    assert a.keys()==b.keys()
                    for field in a:assert np.array_equal(np.asarray(a[field]),np.asarray(b[field])),(arm,k,field)
                selected=int(np.argmax([c['score'] for c in expected])) if expected else None
                assert result['selected_index']==selected
                preservation(baseline_full[k],actual,selected)
                parity.append(dict(arm=arm,key=k,exact=True))
            for i in range(20):run(arm,keys[i%26])
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(5):
            for arm in arms if repeat%2==0 else arms[::-1]:
                for k in keys:
                    _,r=run(arm,k);records.append(dict(arm=arm,key=k,repeat=repeat,**r))
        sums={a:{k:dict(median=float(np.median([r[k] for r in records if r['arm']==a])),P90=float(np.quantile([r[k] for r in records if r['arm']==a],.9))) for k in ['full_ms','R0_ms','P_only_ms','PnP_ms']} for a in arms}
        write(OUT/'RUNTIME_RESULTS.json',dict(complete=True,device=device,protocol=dict(protocol,arms=arms,batch=1,
            isolation='shared R0 and one N3 head resident; same-session R0/N3 timing; not isolated per-arm memory',seed_policy='fixed historical runtime seed1 only, no accuracy selection; accuracy tables use three-seed means'),
            records=records,summary=sums,parity=parity,parity_exact=True,runtime_samples_per_arm=130,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,checkpoint=bound(path)))
        print('RUNTIME_COMPLETE',sums,flush=True)
    finally:extractor.close()
    verify_inputs()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','score','runtime','build']);args=parser.parse_args()
    if args.phase=='build':
        from render import build
        build()
    else:globals()[args.phase]()
