"""Read-only canonical PoseFix diagnosis. No optimizer, detector forward or training.

Stages: prepare, infer, score, report, verify. Only this experiment's roots writable.
"""
from pathlib import Path
import argparse, copy, gc, hashlib, json, math, subprocess, sys, time
from collections import Counter
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT/'_docs/experiments/pallet_posefix_replay_diagnosis_v1'
RAW = ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1'
PRIOR = ROOT/'scripts/research/pallet_sensors_submission_v1'
PDOC = ROOT/'_docs/experiments/pallet_sensors_submission_v1'
PRAW = ROOT/'data/pallet/results/pallet_sensors_submission_v1'
sys.path.insert(0, str(ROOT/'scripts/research/pallet_dim_conditioned_p_v1'))
import dcp_env as E
from dev_evaluate import population_metadata, iou
from eval_math import measure, summary
import pose
sys.path.insert(0, str(PRIOR))
from prior_model import PoseFixPallet9, expectation
from posefix_contract_math import axis_aligned_crop_matrix, transform_points
from prior_inference import correct

read = E.read
SPLITS = ['calibration', 'selection', 'heldout', 'REAL_DEV', 'SQUARE_DEV']
CHAINS = ['RAW', 'CAPPED']

def serial(x):
    if isinstance(x, dict): return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [serial(v) for v in x]
    if isinstance(x, np.ndarray): return serial(x.tolist())
    if isinstance(x, np.generic): return serial(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x

def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW), path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value,str) else json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temp = path.with_suffix(path.suffix+'.pending')
    temp.write_text(text); temp.replace(path)

def freeze(path, value):
    if path.exists():
        assert read(path)==serial(value), ('Refuse to replace locked artifact',path)
    else: write(path,value)

def bound(path): return E.bound(path)
def setup():
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False

def prepare():
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    if (DOC/'SOURCE_BINDING.json').exists(): verify_sources(); return
    selection=read(PDOC/'PRIOR_SELECTION.json'); training=read(PDOC/'TRAINING_AUDIT.json')
    assert selection['selected_rule']['lam']==1 and selection['selected_rule']['max_move_image_diagonal_fraction']==.01
    for r in training['runs']:
        assert r['updates']==6000 and r['parameters']==68661641 and r['real_training']==0
    groups={x['object_type']:x for x in read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    inputs=[]; targets={}; sources=set()
    # Input record contains no GT or GT-dependent matching decision.
    def add(fid, split, image, image_sha, points, box, valid, hw, pad, detected, obj, dims, perms,
            gt, gv, matched, K, xyz, source, truth, prediction=None, session=None):
        usable=bool(detected and np.isfinite(box).all() and (np.asarray(box)[2:]>np.asarray(box)[:2]).all())
        inputs.append(dict(id=fid,split=split,image=str(Path(image).relative_to(ROOT)),image_sha256=image_sha,
            points=points,box=box,valid=valid,raw_hw=hw,pad=pad,detected=bool(detected),usable=usable,
            prediction=prediction,session=session))
        targets[fid]=dict(split=split,object=obj,dimensions_WDH=dims,permutations=perms,gt=gt,valid=gv,
            matched=bool(matched),K=K,xyz=xyz,source=source,truth=truth)
    pe,pop=population_metadata(); baseline=read(E.LINE/'baseline/FULL_CANDIDATES.json')
    cache=read(E.DOC/'DEV_CACHE_COMPLETE.json'); cmap={r['id']:r for r in cache['records']}
    pmeta,truth=pose.metadata('REAL_DEV')
    for item,meta in pop:
        fid=item.frame_id; key=pe.canonical_key(item.image)
        c=torch.load(ROOT/cmap[fid]['path'],map_location='cpu',weights_only=False)
        candidates=baseline['frames'][key]; ix=c['captured']['selected_index']
        pred=dict(candidates=candidates,selected_index=ix)
        assert ix==(int(np.argmax([v['score'] for v in candidates])) if candidates else None)
        candidate=None if ix is None else candidates[ix]
        p=np.full((9,2),np.nan) if candidate is None else np.asarray(candidate['keypoints_xy'],float)
        box=np.full(4,np.nan) if candidate is None else np.asarray(candidate['box_xyxy'],float)
        v=np.isfinite(p).all(-1)&~(p==-1).all(-1)
        t=pe.E._legacy_forbidden_target(item); K,xyz,src=pmeta[fid]; group=groups[meta['object_type']]
        add(fid,'REAL_DEV',ROOT/item.image,baseline['frame_metadata'][key]['image_sha256'],p,box,v,c['raw_hw'],0,
            ix is not None,meta['object_type'],c['dimensions'],group['permutations'],t.keypoints_xy,t.keypoint_supervision_mask,
            candidate is not None and iou(box,t.box_xyxy)>=.5,K,xyz,src,truth[fid],pred,meta['session_id'])
    # Existing synthetic rows, including missing / unmatched detections. No resplit.
    d=E.old('train').FeatureDataset(E.LINE,E.LINE/'cache'); a=d.arrays
    side={r['frame_id']:r for r in read(E.RAW/'DIMENSION_SIDECAR.json')['records']}
    geom=dict(np.load(ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'))
    gi={str(s):i for i,s in enumerate(geom['stems'])}
    for row in d.validation_rows:
        r=d.source['records'][int(d.indices[row])]; fid=r['id']; s=side[fid]; pad=r['reflect_pad_px']
        gain,offset=E.old('features').canvas_affine(r['prepared_shape_hw'],a['input_shape'][row])
        p=(a['points'][row]-offset)/gain; box=((a['boxes'][row].reshape(2,2)-offset)/gain).reshape(4)
        gt=(a['gt_points'][row]-offset)/gain-pad; i=gi[fid]; fx,fy,cx,cy=geom['K'][i]
        K=np.array([[fx,0,cx-geom['pad'][i]],[0,fy,cy-geom['pad'][i]],[0,0,1.]])
        xyz=geom['dims'][i]; g=dict(R=geom['R'][i],t=geom['t'][i],xyz=xyz,body_R=geom['R'][i],body_xyz=xyz,order=s['symmetry_order'])
        add(fid,r['partition'],Path(r['image']),r['image_sha256'],p,box,a['point_valid'][row],r['raw_shape_hw'],pad,
            a['detected'][row],s['source_asset'],s['canonical_WDH'],s['allowed_permutations'],gt,a['gt_valid'][row],
            a['matched'][row],K,xyz,True,g)
    # Square contract: prepared images have +100 border; stored candidates are raw pixels.
    from square_data import membership, square_metadata
    smeta,sgt=square_metadata()
    for index,r in enumerate(membership()[696:],696):
        c=read(E.RAW/f'square_cache/{index:04d}.json'); fid=r['id']; assert c['id']==fid
        ix=c['selected_index']; pred=dict(candidates=c['candidates'],selected_index=ix)
        candidate=None if ix is None else c['candidates'][ix]
        p=np.full((9,2),np.nan) if candidate is None else np.asarray(candidate['keypoints_xy'],float)+100
        box=np.full(4,np.nan) if candidate is None else np.asarray(candidate['box_xyxy'],float)+100
        target=np.asarray(r['target'])[0]; wh=np.array(r['raw_hw'])[::-1]+200; kp=target[5:].reshape(9,3)
        gt=kp[:,:2]*wh-100; gb=np.r_[target[1:3]*wh-target[3:5]*wh/2,target[1:3]*wh+target[3:5]*wh/2]
        K,xyz,src=smeta[fid]; group=groups['plastic_standard_110x110x15']
        add(fid,'SQUARE_DEV',ROOT/r['image'],r['image_sha256'],p,box,np.isfinite(p).all(-1),r['raw_hw'],100,
            ix is not None,group['object_type'],xyz[[0,2,1]],group['permutations'],gt,kp[:,2]>0,
            candidate is not None and iou(box,gb)>=.5,K,xyz,src,sgt[fid],pred)
    inputs.sort(key=lambda r:(SPLITS.index(r['split']),r['id']))
    counts=dict(Counter(r['split'] for r in inputs)); assert counts==dict(calibration=1004,selection=1031,heldout=1985,REAL_DEV=319,SQUARE_DEV=155)
    assert len({r['id'] for r in inputs})==len(inputs)
    freeze(RAW/'INPUTS.json',inputs); freeze(RAW/'TARGETS.json',targets)
    sources.update([E.R0,E.LINE/'baseline/FULL_CANDIDATES.json',E.LINE/'SOURCE_MANIFEST.json',E.LINE/'cache/CACHE_MANIFEST.json',
        E.LINE/'cache/CACHE_COMPLETE.json',E.RAW/'DIMENSION_SIDECAR.json',E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json',
        E.SYM_RAW/'A/square_membership.json',E.SYM_RAW/'A/square_annotation_target_view.json',
        ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz',
        ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
        E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',E.C.POSE/'AXIS_REVIEW_MANIFEST.json',pe.POS,
        ROOT/'_docs/experiments/pallet_final_paper_tables_v1/RESCORED_2D.json',
        ROOT/'_docs/experiments/pallet_final_paper_tables_v1/R0_POSE.json'])
    sources.update(PRIOR.glob('*.py')); sources.update(E.HERE.glob('*.py'))
    sources.update((ROOT/'challenge/evaluation_v2').glob('*.py'))
    sources.update((ROOT/'scripts/paper/pose_metric_closure_v1').glob('*.py'))
    sources.update((ROOT/'scripts/research/pallet_line_pose_v1').glob('*.py'))
    sources.update([PDOC/(n+'.json') for n in ['PRIOR_PROTOCOL_LOCK','PRIOR_SELECTION','TRAINING_AUDIT','P_VS_PRIOR_PAIRED','UNIFIED_DEV_RESULTS','RUNTIME_NUMERIC_AMENDMENT']])
    for k in ('points','boxes','point_valid','input_shape','gt_points','gt_valid','matched','detected','gain','record_index'):
        sources.add(d.directory/d.manifest['arrays'][k]['file'])
    for seed in (1,2,3):
        cp=PRAW/f'runs/PRIOR{seed}/last.pt'; assert E.sha(cp)==selection['checkpoints'][str(seed)]; sources.add(cp)
        sources.add(PRAW/f'validation_PRIOR{seed}.npz')
        sources.add(PRAW/f'evaluation/PRIOR{seed}/RAW_POINT_REPLACEMENTS.json')
    for name in ['pallet_posefix_replay_v1','pallet_posefix_large_error_v1','pallet_posefix_corner_gate_v1','pallet_posefix_utility_selector_v1']:
        sources.update((ROOT/'scripts/research'/name).glob('*.py'))
        sources.update((ROOT/'_docs/experiments'/name).glob('*.md'))
        sources.update((ROOT/'_docs/experiments'/name).glob('*.json'))
    freeze(DOC/'REPLAY_PROTOCOL.json',dict(training_updates=0,checkpoint='canonical synthetic-only PRIOR1/2/3 last6000',seeds=[1,2,3],
        counts=counts,passes=[0,1,2,3],feedback='Two independent trajectories: RAW feeds RAW; CAPPED feeds CAPPED. Shared PASS1 forward. Each pass cap relative to current input, not R0.',
        cap=.01,lambda_value=1,cap_scale='original image diagonal, excluding any prepared border',
        inputs='Frozen R0 predicted bbox/crop/center/valid mask; pose corners only change. RGB always identical. No GT access in infer.',
        source_RGB='Use existing prepared image and predicted box exactly as SourceRGB; subtract existing pad only at output; no second padding.',
        square='155 existing C4 DEV; prepared +100 border accounted exactly. No square supervised checkpoint, diagnosis only.',
        metric='Unmodified current eval_math: observed matched med/P90, full-GT PCK/Esym/gross20; whole-object branch. No pointwise matching.',
        errors='Store canonical GT-aligned errors AND native-channel errors against locked PASS0 whole-object GT branch. Movement/cosine uses native channels; phase switches reported separately.',
        direction='Undefined if either movement <=0.001px. Reversal cos<0; nonreversal 0<=cos<=0.1 orthogonal-ish; cos>0.1 same-direction. These disjoint bins do not claim semantics near boundary.',
        types='Overlapping indicators. Monotonic all error diffs<-1e-9; pass1-best error1<error0 and error2 OR error3>error1; oscillation adjacent error increments change sign AND corresponding displacement cos<0; no-response ALL3 moves<=0.001px; repeated cap>=2 passes.',
        strata=['<=5','(5,10]','(10,20]','>20'],strata_population='matched and input-valid, fixed PASS0 native-branch GT support; missing/mismatched penalties remain in main metrics but not geometric movement counts',
        phase_only='Matched frames with mean fixed error>20px, mean symmetry error<=10px and branch!=0.',
        phase_B_gate='Strong CASE C only if phase-only>=10% of matched REAL_DEV frames in at least2seeds PASS1 RAW; square secondary alone does not trigger main training.',
        numeric='10 identical forward calls per seed on lexicographically first usable REAL_DEV frame; compare every call to first, batch1 FP32 cuDNN nondeterministic TF32 off; tolerance3e-4 crop component, not changed.',
        figures='seed1 REAL_DEV CAPPED: lexicographically first eligible frame/native-corner for monotonic/reversal/repeatedcap. Fallback absent category labelled absent; never choose largest gain.',
        pose='Existing pose.infer/metric/pose_auc with same K/geometry; real/square reconstructed references not independent measured physical GT.',
        no_selection=True,DEV_reused=True,no_sealed_access=True,system_changes=False,thermal_limit_C=80))
    freeze(DOC/'SOURCE_BINDING.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        origin_main=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip(),
        sources=[bound(p) for p in sorted(sources)],inputs=bound(RAW/'INPUTS.json'),targets=bound(RAW/'TARGETS.json'),
        code=bound(__file__),protocol=bound(DOC/'REPLAY_PROTOCOL.json')))
    print('PREPARED',counts,flush=True)

def verify_sources():
    b=read(DOC/'SOURCE_BINDING.json')
    for r in b['sources']+[b['inputs'],b['targets'],b['code'],b['protocol']]:
        assert E.sha(ROOT/r['path'])==r['sha256'],r['path']

def crop_input(r):
    im=cv2.imread(str(ROOT/r['image'])); assert im is not None
    assert E.sha(ROOT/r['image'])==r['image_sha256']
    assert tuple(im.shape[:2])==tuple(np.array(r['raw_hw'])+2*r['pad'])
    matrix=axis_aligned_crop_matrix(r['box'])
    crop=cv2.warpAffine(im,matrix[:2],(288,384),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    rgb=(crop[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)).transpose(2,0,1)
    return matrix,torch.as_tensor(rgb,device='cuda')[None]

@torch.no_grad()
def forward(model,rgb,matrix,points,valid,layers=False):
    cp=transform_points(np.where(valid[:,None],points,0),matrix).astype(np.float32)
    args=[rgb,torch.as_tensor(cp,device='cuda')[None],torch.as_tensor(valid,device='cuda')[None]]
    with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
        result=model(*args,return_layers=layers)
    logits,activations=result if layers else (result,None)
    q=expectation(logits)[0].cpu().numpy(); raw=transform_points(q,np.linalg.inv(matrix))
    raw[~valid]=points[~valid]; raw[8]=points[8]
    return raw,q,activations

def numerical(model,r,seed):
    matrix,rgb=crop_input(r); p=np.array(r['points'],float); v=np.array(r['valid'],bool)
    reference=None; comparisons=[]
    for repeat in range(10):
        raw,q,layers=forward(model,rgb,matrix,p,v,True)
        values={k:x.detach().cpu().numpy().copy() for k,x in layers.items()}
        values['expected_coordinates']=q.copy()
        # Include all9 restored heatmap outputs, prior to invalid/center restoration.
        values['restored_coordinates']=transform_points(q,np.linalg.inv(matrix))-r['pad']
        if reference is None: reference=values
        errors={k:float(np.max(np.abs(values[k]-reference[k]))) for k in values}
        comparisons.append(dict(repeat=repeat,max_abs=errors,bitwise={k:np.array_equal(values[k],reference[k]) for k in values},
            first_divergence=next((k for k in values if errors[k]>0),None)))
        del layers
    mx=max(x['max_abs']['expected_coordinates'] for x in comparisons)
    return dict(seed=seed,id=r['id'],forwards=10,comparisons=comparisons,crop_atol=3e-4,crop_max_abs=mx,
        restored_max_abs=max(x['max_abs']['restored_coordinates'] for x in comparisons),within_reference_tolerance=mx<=3e-4,
        first_divergence=next((x['first_divergence'] for x in comparisons if x['first_divergence']),None))

@torch.no_grad()
def infer():
    setup(); verify_sources(); assert torch.cuda.is_available(); initial=E.gpu()
    rows=read(RAW/'INPUTS.json'); proto_sha=E.sha(DOC/'REPLAY_PROTOCOL.json')
    selected=read(PDOC/'PRIOR_SELECTION.json'); start=time.monotonic(); number=0
    for seed in (1,2,3):
        ck=torch.load(PRAW/f'runs/PRIOR{seed}/last.pt',map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==6000
        model=PoseFixPallet9().cuda().eval().requires_grad_(False); model.load_state_dict(ck['model_state_dict']); del ck
        assert sum(p.numel() for p in model.parameters())==68661641
        numeric_path=RAW/f'numerical_seed{seed}.json'
        if not numeric_path.exists():
            r=next(r for r in rows if r['split']=='REAL_DEV' and r['usable'])
            freeze(numeric_path,numerical(model,r,seed))
        for split in SPLITS:
            subset=[r for r in rows if r['split']==split]; path=RAW/f'predictions/seed{seed}_{split}.npz'
            if path.exists():
                z=np.load(path); assert str(z['protocol_sha256'])==proto_sha and list(z['ids'])==[r['id'] for r in subset]
                print('REUSE',seed,split,len(subset),flush=True); continue
            # axes: frame, chain(RAW/CAPPED), pass, corner, xy. Raw/cap same-input candidates stored separately.
            points=np.empty((len(subset),2,4,9,2)); raw_all=np.empty((len(subset),2,3,9,2)); capped=np.empty_like(raw_all)
            for j,r in enumerate(subset):
                p0=np.array(r['points'],float); valid=np.array(r['valid'],bool)
                states=[p0.copy(),p0.copy()]; points[j,:,0]=p0-r['pad']
                if r['usable']: matrix,rgb=crop_input(r)
                first=None
                for step in range(3):
                    for chain in range(2):
                        p=states[chain]
                        if not r['usable']: raw=p.copy()
                        elif step==0 and chain==1: raw=first.copy()
                        else: raw,_,_=forward(model,rgb,matrix,p,valid)
                        if step==0 and chain==0: first=raw.copy()
                        out=correct(p,raw,valid,selected['selected_rule'],r['raw_hw'])
                        assert np.array_equal(out[8],p0[8],equal_nan=True)
                        assert np.array_equal(raw[8],p0[8],equal_nan=True)
                        raw_all[j,chain,step]=raw-r['pad']; capped[j,chain,step]=out-r['pad']
                        states[chain]=raw if chain==0 else out
                        points[j,chain,step+1]=states[chain]-r['pad']
                number+=1
                if j%100==0 or j+1==len(subset):
                    print('INFER',seed,split,j+1,len(subset),'seconds',round(time.monotonic()-start,1),E.gpu()['gpu'],flush=True)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.pending.npz')
            np.savez_compressed(temp,ids=[r['id'] for r in subset],points=points,raw=raw_all,capped=capped,
                checkpoint_sha256=selected['checkpoints'][str(seed)],protocol_sha256=proto_sha)
            temp.replace(path)
        del model; gc.collect(); torch.cuda.empty_cache()
    freeze(DOC/'NUMERICAL_REPLAY.json',dict(complete=True,runs=[read(RAW/f'numerical_seed{s}.json') for s in (1,2,3)],
        scope='10 identical forwards per seed on one preregistered image; no universal numerical guarantee',
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),TF32=False,batch=1),initial_gpu=initial))
    verify_sources()
    freeze(DOC/'INFERENCE_COMPLETE.json',dict(complete=True,training_updates=0,parameters=68661641,seconds_this_invocation=time.monotonic()-start,
        frames_this_invocation=number,outputs=[bound(p) for p in sorted((RAW/'predictions').glob('*.npz'))],
        GT_input=False,bbox_crop_fixed=True,center8_exact=True,R0_checkpoint_unchanged=True,prior_checkpoints_unchanged=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','infer','score','report','verify'])
    args=parser.parse_args()
    if args.stage in ('score','report','verify'):
        from analysis import score,report,verify
    globals()[args.stage]()
