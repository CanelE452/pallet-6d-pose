"""One cached-feature reverse run; identity checks are included in the 319 calls."""
from collections import Counter
import gzip
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import torch
from . import common as C
from .reverse import reverse_captured
from .test_contracts import run as functional_checks
from scripts.research.pallet_n3_subpix_20261008_v1.evaluate import score
from scripts.research.pallet_n3_subpix_20261008_v1.runtime import _interference


def load_head():
    E,_ = C.legacy()
    inf = importlib.import_module('inference')
    assert Path(inf.__file__).resolve() == E.POSE.E.HERE/'inference.py'
    selected = C.read(C.ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json')
    spec = selected['temperatures']['N3_DIM_SYM_seed1']
    assert selected['complete'] and selected['real_access'] is False
    assert selected['rule'] == dict(lam=1.,max_move_image_diagonal_fraction=.01)
    # The immutable baseline provides code, while checkpoints live at the
    # explicit source root. Its legacy load_head resolves a different RAW root.
    path=C.ROOT/spec['checkpoint']['path']
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert checkpoint['complete'] and checkpoint['step']==6000
    assert checkpoint['baseline_checkpoint_sha256']==inf.E.R0_SHA
    head=inf.model('N3_DIM_SYM',checkpoint['config']).cuda().eval()
    head.load_state_dict(checkpoint['model_state_dict']);head.requires_grad_(False)
    assert C.sha(inf.__file__)==C.sha(C.ROOT/'scripts/research/pallet_dim_conditioned_p_v1/inference.py')
    assert C.sha(path) == spec['checkpoint']['sha256']
    assert spec['temperature'] == C.read(C.PRIOR/'PROTOCOL.json')['N3_temperature']
    norm = C.read(C.ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json')
    return inf,head,spec['temperature'],selected['rule'],norm


def prepare(frames):
    assert (C.DOC/'EXISTING_GRADE_SUMMARY.json').exists(), 'Existing four-arm grade aggregation must happen first'
    assert not (C.DOC/'PROTOCOL.json').exists(), 'Preserve completed and interrupted executions'
    prior = C.read(C.PRIOR/'PROTOCOL.json'); checks = C.read(C.PRIOR/'CHECKS.json')
    assert checks['status']=='PASS' and checks['raw_rows_sha256']==C.sha(C.PRIOR/'PREDICTIONS.jsonl.gz')
    for row in prior['code']:
        assert C.sha(C.ROOT/row['path']) == row['sha256'], row['path']
    old_runtime=C.read(C.PRIOR/'RUNTIME.json')
    assert old_runtime['complete'] and old_runtime['numeric_parity']['RAW_bitexact_calls']==600
    assert all(v['checked_calls']==150 and v['unique_frames']==26 and v['max_abs_final_px']==0 for v in old_runtime['numeric_parity']['arms'].values())
    original={r['id']:r for r in prior['input_manifest']}
    rows=list(C.iter_rows(C.PRIOR/'PREDICTIONS.jsonl.gz'))
    controls={arm:{r['id']:r for r in rows if r['method']==arm} for arm in C.ARMS[:4]}
    ids=[f['id'] for f in frames]
    assert len(ids)==len(set(ids))==319 and len({f['session'] for f in frames})==13
    assert all(len(v)==319 and set(v)==set(ids) for v in controls.values())
    source_sub=C.ROOT/'_docs/experiments/pallet_training_free_compare_20261007_v1/PREDICTIONS.jsonl.gz'
    native={r['id']:r['native_points'] for r in C.iter_rows(source_sub) if r['method']=='SUBPIX_NATIVE'}
    inputs=[]; identity=[]
    for f in frames:
        p=original[f['id']]
        assert np.array_equal(f['q'], np.asarray(controls['BASE'][f['id']]['native_points']),equal_nan=True)
        assert C.digest(f['q'])==C.digest(p['q0']) and C.digest(f['point_valid'])==C.digest(p['prediction_support'])
        assert C.digest(f['K'])==C.digest(p['K']) and C.digest(f['xyz'])==C.digest(p['dimensions_whd_m'])
        image=C.ROOT/f['axis']['image'];cache=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DEV_cache'/(f['axis']['frame_id']+'.pt')
        assert C.sha(image)==p['image_sha256'] and C.sha(cache)==p['cache_sha256']
        sub=np.asarray(controls['SUBPIX'][f['id']]['native_points'],np.float64)
        assert np.array_equal(sub,np.asarray(native[f['id']]),equal_nan=True)
        if np.array_equal(sub,f['q'],equal_nan=True):identity.append(f['id'])
        inputs.append(dict(id=f['id'],session=f['session'],image_sha256=p['image_sha256'],cache_sha256=p['cache_sha256'],
            raw_hw=f['raw_hw'],K=f['K'],dimensions_whd_m=f['xyz'],prediction_support=f['point_valid'],
            selected_index=f['captured']['captured']['selected_index'],q0_sha256=C.digest(f['q']),SUBPIX_input_sha256=C.digest(sub)))
    assert len(identity)==4
    initial=C.read(C.OUTPUT/'INITIAL_STATE.json')
    from scripts.research.pallet_training_free_compare_20261007_v1.methods import method_configuration
    bindings=[C.binding(C.PRIOR/name) for name in ('PROTOCOL.json','PREDICTIONS.jsonl.gz','CHECKS.json','RUNTIME.json','RUNTIME_ROWS.jsonl.gz')]
    selection=C.read(C.ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json')
    checkpoint=selection['temperatures']['N3_DIM_SYM_seed1']['checkpoint']
    for p in [C.ROOT/checkpoint['path'], C.GRADE_PATH, C.DOC/'GRADE_BINDINGS.json', C.DOC/'EXISTING_GRADE_SUMMARY.json',
        C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json',
        C.ROOT/'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json',
        C.ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json',
        C.ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json']:
        bindings.append(C.binding(p))
    code=[C.binding(Path(__file__).with_name(name)) for name in ('common.py','reverse.py','evaluate.py','test_contracts.py','runtime.py')]
    for p in [C.ROOT/'scripts/research/pallet_dim_conditioned_p_v1/inference.py',C.ROOT/'scripts/research/pallet_dim_conditioned_p_v1/refiner.py',
        C.ROOT/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py',C.ROOT/'scripts/research/pallet_line_pose_v1/features.py',
        C.ROOT/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py']:
        code.append(C.binding(p))
    panel=C.read(C.ROOT/'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json')['selected']
    protocol=dict(schema='pallet_subpix_order_20261009_v1',date='2026-10-09',base_commit=initial['head'],
        investigated_commit='5a498f070566e4161b3d74a43c69b569ee30f07d',
        remote_main_at_start=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=C.WORKTREE,text=True).split()[0],
        branch='research/subpix-n3-order-20261009',main_publication=False,arms=list(C.ARMS),
        new_route='SUBPIX_N3',N3_seed=1,N3_checkpoint=checkpoint,N3_temperature=selection['temperatures']['N3_DIM_SYM_seed1']['temperature'],
        N3_rule=selection['rule'],SUBPIX=method_configuration()['SUBPIX'],
        route='qS_native=S(gray,q0); qS=cap(q0,qS_native); qSN=actual N3(qS,same features,box,registered dims); qFinal=cap(q0,qSN); actual F(qFinal)',
        population=dict(frames=319,sessions=13,grades={'clean':153,'moderate':92,'severe':74},reference_corners=2499,observed_corners=2445,matched=311),
        input_manifest=inputs,input_manifest_sha256=C.digest(inputs),bindings=bindings,code=code,
        parity=dict(identity_ids=identity,identity_cases_counted_in_accuracy319=True,absolute_tolerance=1e-7,
            existing_four_routes_fixed26='reused prior actual full-path coordinates/pose parity; 600 calls,150/route all26',
            new_control_F_calls=0,old_float32_cap='compare identity native to old qN; final to cap_q0(old qN)'),
        features=dict(backbone_accuracy_calls=0,cached_original_fp16_neck=True,actual_coordinate_dependent_sampling_recomputed=True,
            original_mask='native finite/non-sentinel prediction mask; all9 slots true on these319',
            branch_mask='existing finite network coordinates; not human visibility or matching'),
        statistics=dict(metrics=['corner_px','translation_cm','rotation_deg','ADDsym_cm'],ddof=1,bootstrap=dict(units=13,resamples=10000,seed=20260917,shared_draws=True,grade_mask_after_original_session_sampling=True)),
        runtime=dict(panel_ids=[r['frame_id'] for r in panel],warmup_each=20,measured_each=130,max_full_calls=750,
            boundary='RAM image -> single actual YOLO selection/features -> correction -> single F; decode/model load excluded'),
        budget=dict(reverse_accuracy_head=319,reverse_accuracy_F=319,new_control_parity_F_ceiling=104,runtime_F=750,total_F_ceiling=1173,active_seconds=1800),
        forbidden=dict(new_training=0,new_annotations=0,new_gates=0,tuning=0,additional_seeds=0,paper_changes=0),
        prior_grade_aggregation_completed_before_reverse=True,
        user_changes=dict(tracked_sha256=initial['tracked_sha256'],tracked_diff_sha256=initial['tracked_diff_sha256'],status_sha256=initial['status_sha256']))
    C.write(C.DOC/'PROTOCOL.json',protocol)
    return protocol,controls


@torch.no_grad()
def main():
    C.OUTPUT.mkdir(parents=True,exist_ok=True)
    assert not (C.DOC/'PREDICTIONS.jsonl.gz').exists(), 'Do not overwrite completed outcomes'
    assert not (C.OUTPUT/'ACCURACY_ATTEMPT.json').exists(), 'Preserve interrupted attempts; do not silently replay'
    started=time.monotonic();cv2.setNumThreads(1);torch.set_num_threads(4)
    E,frames,targets,_,_=C.load_real()
    protocol,controls=prepare(frames)
    checks=dict(status='IN_PROGRESS',functional=functional_checks(),protocol_sha256=C.sha(C.DOC/'PROTOCOL.json'))
    probe=_interference();assert probe['quiet'] and probe['gpu_temperature_under_80'],'Competing workload preserved; do not run accuracy'
    torch.manual_seed(1);np.random.seed(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    inf,head,T,rule,norm=load_head()
    head_calls=[0]
    hook=head.register_forward_pre_hook(lambda *_:head_calls.__setitem__(0,head_calls[0]+1))
    frame_map={f['id']:f for f in frames}
    order=protocol['parity']['identity_ids']+[f['id'] for f in frames if f['id'] not in protocol['parity']['identity_ids']]
    counts=Counter();sub_calls=0;completed=[];identity_results=[];sample_calls=0
    C.write(C.OUTPUT/'ACCURACY_ATTEMPT.json',dict(completed=0,head_calls=0,F_calls=0))
    path=C.DOC/'PREDICTIONS.jsonl.gz'
    try:
        with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as stream:
            for i,fid in enumerate(order):
                assert time.monotonic()-started<1800,'Active evaluation budget exhausted'
                f=frame_map[fid];original=f['captured']['captured']
                before=C.digest({k:v for k,v in original.items() if k not in ('p3','p4')})
                bgr=cv2.imread(str(C.ROOT/f['axis']['image']),cv2.IMREAD_COLOR)
                assert bgr is not None and list(bgr.shape[:2])==list(f['raw_hw']),fid
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                captured=dict(original)
                for key in ('p3','p4'):captured[key]=original[key].to('cuda')
                dims,group=inf.registry_input(f['axis']['object_type'])
                np.testing.assert_array_equal(dims,f['captured']['dimensions']);assert group==f['captured']['order']
                C.write(C.OUTPUT/'ACCURACY_ATTEMPT.json',dict(attempted_id=fid,completed=len(completed),head_calls=head_calls[0],F_calls=len(completed),head_attempt=i+1))
                prediction,diag,qSN=reverse_captured(inf,head,captured,dims,group,T,rule,f['raw_hw'],norm,gray,diagnostics=True)
                assert head_calls[0]==i+1
                assert np.array_equal(diag['qS'],np.asarray(controls['SUBPIX'][fid]['native_points']),equal_nan=True)
                points=diag['qFinal'];support=f['point_valid'];cap=diag['cap_px']
                assert np.array_equal(points[8],f['q'][8],equal_nan=True)
                assert np.array_equal(points[~support],f['q'][~support],equal_nan=True)
                assert np.max(diag['total_final_move_px8'][support[:8]],initial=0)<=cap+1e-10
                assert np.max(diag['N3_move_from_SUBPIX_px8'][support[:8]],initial=0)<=cap+1e-4
                assert before==C.digest({k:v for k,v in original.items() if k not in ('p3','p4')})
                if fid in protocol['parity']['identity_ids']:
                    qN=np.asarray(controls['N3'][fid]['native_points'],np.float64)
                    native_error=float(np.max(np.abs(qSN-qN)))
                    expected_final=C.cap_points(f['q'],qN,gray.shape[1],gray.shape[0],support)
                    final_error=float(np.max(np.abs(points-expected_final)))
                    assert native_error<=1e-7 and final_error<=1e-7,(fid,native_error,final_error)
                    identity_results.append(dict(id=fid,PASS=True,native_max_abs_px=native_error,final_max_abs_px=final_error))
                row=score(E,f,points,targets[fid],'SUBPIX_N3');counts.update(row['PnP_counts'])
                row.update(q0=diag['q0'],qS_native=diag['qS_native'],qS=diag['qS'],qSN=qSN,qFinal=points,
                    prediction_support=support,raw_hw=f['raw_hw'],correction=diag,
                    fixed_metadata=dict(selected_index=original['selected_index'],candidate_metadata={k:v for k,v in original['candidates'][original['selected_index']].items() if k!='keypoints_xy'},preserved=True),
                    feature_cache_sha256=next(p['cache_sha256'] for p in protocol['input_manifest'] if p['id']==fid),
                    GT_inference_inputs=False,new_head_forwards=1)
                stream.write(json.dumps(C.finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');completed.append(fid)
                sub_calls+=diag['algorithm_corner_calls'];sample_calls+=diag['input_trace']['sample_calls']
                if i==3:
                    checks.update(identity=dict(PASS=True,rows=identity_results),existing_parity_reused=True,
                        status='MINIMUM_PASS',reverse_input_trace_first4=all(r['PASS'] for r in identity_results))
                    C.write(C.DOC/'CHECKS.json',checks);stream.flush();print('MINIMUM_IDENTITY_CHECKS_PASS',4,'included_in319',flush=True)
                if i%40==0 or i==318:
                    stream.flush();print('REVERSE_ACCURACY',i+1,319,'head',head_calls[0],'F',len(completed),'seconds',round(time.monotonic()-started,2),flush=True)
    finally:hook.remove()
    assert len(completed)==len(set(completed))==head_calls[0]==319
    checks.update(status='PASS',complete=True,raw_rows_sha256=C.sha(path),
        contracts=dict(all319_original_features_preserved=True,input_points_candidates_patch_samples_context_recomputed=True,
            actual_sample_calls=sample_calls,shared_neck_fp16=True,metadata_center_missing_preserved=True,
            SUBPIX_native_equals_CAP1_all319=True,strict_final_cap_from_Base=True,GT_inputs=False),
        execution=dict(reverse_accuracy_head_calls=head_calls[0],reverse_accuracy_F_calls=319,new_control_parity_F_calls=0,
            backbone_accuracy_calls=0,SUBPIX_corner_calls=sub_calls,SUBPIX_image_calls=319,actual_candidate_sample_calls=sample_calls,
            accuracy_PnP_counts=dict(counts),training_updates=0,rows=319,elapsed_seconds=time.monotonic()-started),
        interference=probe,inference_signature=list(inspect.signature(reverse_captured).parameters))
    C.write(C.DOC/'CHECKS.json',checks)
    print('REVERSE_ACCURACY_DONE',checks['execution'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        C.write(C.DOC/'INCOMPLETE.json',dict(complete=False,type=type(error).__name__,message=str(error),
            execution=C.read(C.OUTPUT/'ACCURACY_ATTEMPT.json') if (C.OUTPUT/'ACCURACY_ATTEMPT.json').exists() else {},
            rule='Preserve partial rows and do not relax tolerance or silently replay'))
        raise
