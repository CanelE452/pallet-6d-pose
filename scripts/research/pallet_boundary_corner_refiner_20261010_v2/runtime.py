"""Fresh complete-path synchronized timing on the frozen eligible 26-image panel.

Every timed call captures the detector afresh. Models/images/parity references
are loaded before timing; all correction, initial/final pose and hidden-point
replacement operations run inside each interval. This is never cache timing.
"""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import time

from . import common as C

ARMS = ('BASE', 'N3_SUBPIX', 'N3_BASIN_ROBUST', C.PRIMARY)
FRAMES, WARMUP, REPEATS = 26, 20, 5
STAGES = ('detector', 'N3_correction', 'initial_pose', 'boundary_observation',
          'final_pose_reprojection_and_metadata')


def schedules():
    warm, measured = [], []
    for i in range(WARMUP):
        order = list(ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)])
        if i % 2:
            order.reverse()
        for position, arm in enumerate(order):
            warm.append(dict(phase='warmup', arm=arm, warmup_index=i, arm_position=position, image_index=i % FRAMES))
    for repeat in range(REPEATS):
        order = list(ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)])
        if repeat % 2:
            order.reverse()
        for image in range(FRAMES):
            for position, arm in enumerate(order):
                measured.append(dict(phase='measured', arm=arm, repeat=repeat, arm_position=position, image_index=image))
    C.require(len(warm) == 80 and len(measured) == 520 and
              Counter(j['arm'] for j in warm) == {a:20 for a in ARMS} and
              Counter(j['arm'] for j in measured) == {a:130 for a in ARMS}, 'fixed600 schedule differs')
    return warm, measured


def prereqs(args):
    C.verify_protocol(args)
    seal_path = Path(args.accuracy_output) / 'GEOMETRY_SEAL.json'
    seal = C.read(seal_path)
    C.require(seal['complete'] and seal['rows'] == 1225 and seal['fixed_rows'] == 490 and
              seal['protocol'] == C.binding(args.protocol), 'runtime needs complete frozen geometry')
    for key,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('fixed_geometry','FIXED_GEOMETRY_SEALED.jsonl.gz'),
                     ('observations','OBSERVATIONS.jsonl.gz'),('parity','BASE_N3_PARITY.json')):
        C.bound(Path(args.accuracy_output)/name,seal[key],'runtime reference '+key)
    score = C.read(Path(args.accuracy_output)/'SCORING_RECEIPT.json')
    C.require(score['complete'] and score['geometry_seal']==C.binding(seal_path), 'runtime needs scoped scoring receipt')
    panel = C.read(C.CORRECTED / 'RUNTIME_PANEL.json')['frames']
    eligible = {f['id']:f for f in C.cohort_frames(args)}
    C.require(len(panel)==len({f['frame_id'] for f in panel})==26 and
              Counter(f['session_id'] for f in panel)=={s:2 for s in {f['session_id'] for f in panel}} and
              len({f['session_id'] for f in panel})==13 and all(f['frame_id'] in eligible for f in panel),
              'runtime panel population/sessions/scope differs')
    refs = {a:{} for a in ARMS}
    for name in ('GEOMETRY_SEALED.jsonl.gz','FIXED_GEOMETRY_SEALED.jsonl.gz'):
        for row in C.rows(Path(args.accuracy_output)/name):
            if row['method'] in refs and row['id'] in {f['frame_id'] for f in panel}:
                C.require(row['id'] not in refs[row['method']], 'duplicate runtime reference')
                refs[row['method']][row['id']] = {k:row[k] for k in
                    ('native_points','actual_pose','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')}
    C.require(all(set(v)=={f['frame_id'] for f in panel} for v in refs.values()), 'runtime reference IDs incomplete')
    return panel,eligible,refs,seal


def describe(values):
    import numpy as np
    a = np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        sample_variance=float(a.var(ddof=1)) if len(a)>1 else None,
        sample_std=float(a.std(ddof=1)) if len(a)>1 else None,
        median=float(np.median(a)) if len(a) else None,
        P90=float(np.percentile(a,90)) if len(a) else None,
        maximum=float(a.max()) if len(a) else None,unit='ms',ddof=1)


def durable_owned(path, value):
    """Update this run's exclusive journal, never a research output/cache."""
    C.require(path.is_file() and not path.is_symlink(), 'runtime journal ownership lost')
    pending = path.with_suffix('.pending')
    C.require(not pending.exists() and not pending.is_symlink(), 'preserve interrupted journal pending file')
    with pending.open('x',encoding='utf-8') as stream:
        json.dump(C.finite(value),stream,allow_nan=False,sort_keys=True)
        stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def measure(args):
    import cv2
    import numpy as np
    from .pipeline import Pipeline
    panel,eligible,refs,seal = prereqs(args)
    prior = C.prior_snapshot(args.prior_bindings)
    destination = C.output_path(args,'RUNTIME.json')
    row_path = C.output_path(args,'RUNTIME_ROWS.jsonl.gz')
    journal = C.output_path(args,'RUNTIME_STARTED.json')
    warm,measured = schedules()
    images = []
    for frame in panel:
        canonical = eligible[frame['frame_id']]
        path = Path(args.source_root)/canonical['image']
        C.require(C.sha(path)==canonical['image_sha256'], 'runtime RGB differs')
        image = cv2.imread(str(path),cv2.IMREAD_COLOR)
        C.require(image is not None and list(image.shape[:2])==canonical['raw_hw'], 'runtime decode shape differs')
        images.append(image)
    result = dict(schema='validated_boundary_actual_complete_runtime_v2',complete=False,status='FAILED_RUNTIME',
        arms=list(ARMS),frames=26,panel_sessions=13,panel=[dict(id=f['frame_id'],session=f['session_id']) for f in panel],
        protocol=C.binding(args.protocol),accuracy_seal=C.binding(Path(args.accuracy_output)/'GEOMETRY_SEAL.json'),
        runtime_code=C.binding(__file__),warmup_per_arm=20,repeats=5,measured_per_arm=130,
        configured_pipeline_calls=600,schedule='same cyclic route rotation/odd-block reversal as original benchmark',
        boundaries=dict(full_pipeline='RAM native BGR,K,physical W,H,D -> fresh YOLO -> fixed N3 -> fresh initial N3 pose -> optional fresh Base-query ROLE -> evidence/admission -> fresh finite bank -> final pose -> H reprojection -> output metadata',
            all_initial_and_final_PnP_included=True,all_observation_and_admission_work_included=True,
            excluded=['model/checkpoint load','RGB file/decode','GT scoring','parity checks','durable journal','resource snapshots']),
        cached_coordinate_replay_used_for_timing=False,GT_inference_access=False,other_benchmarks_parallel=False,
        actual_image_decode_calls=26,image_decode_outside_intervals=True)
    environment = dict(interference_snapshots=[])
    state = Counter()
    rows = []
    initialized = False
    start = time.monotonic()
    C.write_new(journal,dict(status='STARTED',protocol=C.binding(args.protocol),configured_calls=600,counts=dict(state)))
    pipeline = None
    try:
        with C.legacy_context(args) as (_,_,R):
            probe = R._interference()
            environment['interference_snapshots'].append(dict(phase='before_models',**probe))
            C.require(probe['quiet'] and probe['gpu_temperature_under_80'],'RUNTIME_PENDING: resource/thermal guard; preserve competing jobs')
        pipeline = Pipeline(args)
        initialized = True
        torch = pipeline.torch
        C.require(torch.cuda.is_available(),'CUDA unavailable')
        environment.update(python=__import__('sys').version.split()[0],torch=torch.__version__,numpy=np.__version__,
            opencv=cv2.__version__,ultralytics=importlib.metadata.version('ultralytics'),batch=1,
            actual_torch_threads=torch.get_num_threads(),actual_opencv_threads=cv2.getNumThreads(),
            numeric=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                CUDA_version=torch.version.cuda,cuDNN_version=torch.backends.cudnn.version()),
            device=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability())),
            threadpools=[{k:v for k,v in p.items() if k!='filepath'} for p in __import__('threadpoolctl').threadpool_info()])
        torch.cuda.reset_peak_memory_stats()
        last_probe = time.monotonic()
        with pipeline.old.no_truth_reads(),torch.no_grad(),C.primitive_counter() as primitive:
            for job in warm+measured:
                C.require(time.monotonic()-start <= args.remaining_seconds,'RUNTIME_PENDING: operating time limit')
                if time.monotonic()-last_probe >= 5 or (job['phase']=='measured' and job['image_index']==0 and job['arm_position']==0):
                    probe = pipeline.runtime._interference()
                    environment['interference_snapshots'].append(dict(job=job,**probe));last_probe=time.monotonic()
                    C.require(probe['quiet'] and probe['gpu_temperature_under_80'],'RUNTIME_PENDING: competing workload/thermal guard')
                state['pipeline_calls_started'] += 1
                durable_owned(journal,dict(status='RUNNING',active_job=job,counts=dict(state),model_counts=dict(pipeline.counts)))
                frame = eligible[panel[job['image_index']]['frame_id']]
                image = images[job['image_index']]
                before = primitive.copy()
                ticks = []
                def mark(stage):
                    torch.cuda.synchronize()
                    ticks.append((stage,time.perf_counter_ns()))
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                output = pipeline.predict(image,frame['K'],frame['xyz'],frame,method=job['arm'],stage_callback=mark)
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                C.require([s for s,_ in ticks]==list(STAGES), 'runtime stage markers differ')
                times = {}; previous=t0
                for stage,tick in ticks:
                    times[stage+'_ms']=(tick-previous)/1e6;previous=tick
                times['return_ms']=(t1-previous)/1e6
                raw = dict(**job,id=frame['id'],session=frame['session'],full_ms=(t1-t0)/1e6,**times,
                    final_points=output['native_points'],actual_pose=output['actual_pose'],
                    hidden_initial=output['hidden_initial'],output_status=output['output_status'],
                    new_pose_estimated=output['new_pose_estimated'],fallback_used=output['fallback_used'],
                    solver=output.get('solver'),selected_index=output['selected_index'],GT_canary_active=True,
                    observation_raw_logits_sha256=output.get('observation_raw_logits_sha256'),
                    primitive_entry_calls={k:primitive[k]-before[k] for k in primitive},parity_status='PENDING')
                rows.append(raw)
                reference = refs[job['arm']][frame['id']]
                raw['points_parity']=pipeline.runtime._point_parity(output['native_points'],reference['native_points'],1e-7,(job['arm'],frame['id']))
                raw['pose_parity']=pipeline.runtime._pose_parity(output['actual_pose'],reference['actual_pose'])
                C.require(all(output[k]==reference[k] for k in ('hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')), 'runtime saved outcome metadata differs')
                raw['raw_BASE_points_parity']=pipeline.runtime._point_parity(output['original_base_points'],frame['points']['BASE'],1e-7,('BASE',frame['id']))
                if job['arm']!='BASE':
                    raw['raw_N3_points_parity']=pipeline.runtime._point_parity(output['native_N3_points'],frame['points']['N3_SUBPIX'],1e-7,('N3',frame['id']))
                if output['new_pose_estimated']:
                    C.require(set(output['hidden_initial']).isdisjoint(output['solver']['fit_input_ids']), 'runtime H fit leakage')
                    if output['hidden_initial']:
                        np.testing.assert_allclose(np.asarray(output['native_points'])[output['hidden_initial']],np.asarray(output['solver']['projected'])[output['hidden_initial']],rtol=0,atol=1e-12)
                raw.update(parity_status='PASS',candidate_center_score_box_preserved=True)
                state['pipeline_calls_complete'] += 1
                if state['pipeline_calls_complete']%50==0:
                    print('BOUNDARY_REFINER_RUNTIME',state['pipeline_calls_complete'],600,flush=True)
            result['actual_OpenCV_entry_calls']=dict(primitive)
        C.require(len(rows)==state['pipeline_calls_complete']==600 and pipeline.counts['detector_calls']==600 and
                  pipeline.counts['initial_pose_calls']==600 and pipeline.counts['final_pose_paths']==300,
                  'complete whole-path execution counts differ')
        probe=pipeline.runtime._interference()
        environment['interference_snapshots'].append(dict(phase='after_complete',**probe))
        C.require(probe['quiet'] and probe['gpu_temperature_under_80'],'RUNTIME_PENDING: endpoint interference')
        result.update(complete=True,status='DONE')
        environment['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    except Exception as error:
        result.update(status='RUNTIME_PENDING' if 'RUNTIME_PENDING' in str(error) else 'FAILED_RUNTIME',
                      reason=dict(type=type(error).__name__,message=str(error)))
    finally:
        if pipeline is not None:
            result['actual_pipeline_calls']=dict(pipeline.counts)
            result['model_forwards']=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards,
                ROLE=pipeline.counts['ROLE_head_calls'],
                detector_internal_initialization_calls=max(0,pipeline.models.detector_forwards-pipeline.counts['detector_calls']))
            result['model_bindings']=pipeline.bindings
            pipeline.close()
        result.update(model_initialization_complete=initialized,environment=environment,execution=dict(state),
            elapsed_seconds=time.monotonic()-start,statistics_official=result['complete'],new_training_updates=0,new_GT_evaluations=0)
        result['summaries']={arm:{stage:describe([r[stage+'_ms'] for r in rows if r['arm']==arm and
            r['phase']=='measured' and r['parity_status']=='PASS']) for stage in ('full',)+STAGES+('return',)} for arm in ARMS}
        result['warmup_accounting']={a:dict(configured=20,completed=sum(r['arm']==a and r['phase']=='warmup' and r['parity_status']=='PASS' for r in rows)) for a in ARMS}
        C.require(C.prior_snapshot(args.prior_bindings)==prior,'prior file preservation changed during runtime')
        C.save_rows(row_path,rows)
        result['raw_rows']=C.binding(row_path)
        C.write_new(destination,result)
        durable_owned(journal,dict(status=result['status'],counts=dict(state),elapsed_seconds=result['elapsed_seconds']))
    print('BOUNDARY_REFINER_RUNTIME',result['status'],len(rows),flush=True)
    C.require(result['complete'],'runtime incomplete; interrupted rows retained')
    return result


def main():
    parser=C.parser(__doc__,('schedule','preflight','measure'))
    parser.set_defaults(output=str(C.PRIVATE/'runtime'))
    parser.add_argument('--accuracy-output',default=str(C.PRIVATE/'accuracy'))
    parser.add_argument('--remaining-seconds',type=float,default=3600.)
    args=parser.parse_args()
    if args.stage=='schedule':
        warm,measured=schedules();print(dict(arms=ARMS,warmup=len(warm),measured=len(measured),total=len(warm+measured)))
    elif args.stage=='preflight':
        prereqs(args);C.prior_snapshot(args.prior_bindings);C.output_path(args,'RUNTIME_STARTED.json')
        print('BOUNDARY_RUNTIME_PREFLIGHT_PASS',flush=True)
    else:
        measure(args)


if __name__=='__main__':
    main()
