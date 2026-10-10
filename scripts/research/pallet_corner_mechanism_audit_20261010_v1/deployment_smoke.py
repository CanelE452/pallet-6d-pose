"""One fixed, two-lifecycle deployment usage check; no accuracy or timing study.

The public facade, checkpoints, calibration and prediction policy are unchanged.
Each fresh instance predicts the same first preselected visual case once. Only
GT-free sealed geometry is the parity reference. Existing outputs are preserved.
"""
from __future__ import annotations

import builtins
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts.research.pallet_boundary_corner_refiner_20261010_v2 import common as C

DOC = C.REPO / '_docs/experiments/pallet_corner_mechanism_audit_20261010_v1'
NAMES = ('DEPLOYMENT_SMOKE_PROTOCOL.json', 'DEPLOYMENT_SMOKE_STARTED.json',
         'DEPLOYMENT_SMOKE_CHECKS.json', 'DEPLOYMENT_SMOKE_ROWS.jsonl.gz')
STATE_KEYS = ('output_status', 'new_pose_estimated', 'fallback_used', 'pose_available',
              'no_pose', 'hidden_initial', 'hidden_after', 'hidden_set_changed',
              'hidden_reprojected', 'reprojected_ids', 'excluded',
              'reprojections_reused_as_observations', 'output_coordinate_sources',
              'selected_index', 'fixed_metadata')


def utc():
    return datetime.now(timezone.utc).isoformat()


def destination(args, name, exists=False):
    output = Path(args.output)
    C.require(not output.is_symlink(), 'smoke output directory is a symlink')
    output = output.resolve()
    roots = (Path(args.source_root).resolve(), Path(args.baseline_root).resolve(),
             Path(args.fits).resolve())
    C.require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in roots),
              'smoke output overlaps a source/baseline/checkpoint dependency')
    C.require(not output.is_relative_to(C.REPO) or output == DOC.resolve(),
              'public smoke output must be the new mechanism-audit directory')
    path = output / name
    C.require(name in NAMES and not path.is_symlink(), 'invalid smoke output')
    C.require(exists or not path.exists(), 'preserve completed/interrupted smoke output ' + name)
    return path


def quiet():
    """Read-only resource gate, excluding this process and desktop rustdesk."""
    fields = 'name,driver_version,temperature.gpu,utilization.gpu,memory.used'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields,
                                  '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True).strip()
    table = subprocess.check_output(['ps', '-eo', 'pid,ppid,pcpu,comm,args', '--no-headers'], text=True)
    parsed = [line.strip().split(None, 4) for line in table.splitlines()]
    parsed = [r for r in parsed if len(r) == 5]
    parents = {int(r[0]): int(r[1]) for r in parsed}
    excluded, cursor = {os.getpid()}, os.getppid()
    while cursor and cursor not in excluded:
        excluded.add(cursor)
        cursor = parents.get(cursor, 0)
    foreign_cpu = []
    for pid, ppid, cpu, command, args in parsed:
        if int(pid) in excluded or int(ppid) == os.getpid():
            continue
        if re.search(r'python|torchrun|jupyter', command, re.I) and re.search(
                r'pallet|benchmark|train|inference|evaluate|runtime', args, re.I):
            foreign_cpu.append(dict(pid=int(pid), command=command, lifetime_cpu_percent=float(cpu)))
    compute = []
    foreign_gpu = []
    for line in apps.splitlines():
        pid, process, memory = [p.strip() for p in line.split(',', 2)]
        compute.append(dict(pid=int(pid), process=Path(process).name, memory=memory))
        if int(pid) != os.getpid() and '/usr/share/rustdesk/rustdesk' not in process:
            foreign_gpu.append(int(pid))
    temperatures = [float(line.split(',')[2].strip()) for line in gpu.splitlines()]
    return dict(quiet=not foreign_cpu and not foreign_gpu,
                temperature_under_80=bool(temperatures) and max(temperatures) < 80,
                gpu=gpu, gpu_fields=fields, compute_apps=compute,
                foreign_cpu_workloads=foreign_cpu, foreign_gpu_pids=foreign_gpu,
                desktop_exception='rustdesk', system_changes=False,
                limits='Snapshots can miss transient competing work; this is not a latency measurement.')


def inputs(args):
    paths = dict(frozen_pipeline_protocol=C.DOC/'PROTOCOL.json',
                 visual_case_protocol=C.DOC/'VISUAL_CASE_PROTOCOL.json',
                 original_inputs=C.OLD/'INPUTS.json', geometry_seal=C.DOC/'GEOMETRY_SEAL.json',
                 sealed_geometry=C.DOC/'GEOMETRY_SEALED.jsonl.gz',
                 sealed_fixed_geometry=C.DOC/'FIXED_GEOMETRY_SEALED.jsonl.gz',
                 protection=DOC/'PROTECTION_BEFORE.json', calibration=args.calibration,
                 training_completion=Path(args.fits)/'TRAINING_COMPLETION.json',
                 corrected_ROLE=Path(args.fits)/'IMAGE_ROLE.pt',
                 base_weights=args.base_weights or Path(args.source_root)/C.BASE_WEIGHT_REL,
                 N3_weights=args.N3_weights or Path(args.source_root)/C.N3_WEIGHT_REL,
                 code=Path(__file__), deployment=C.CODE/'deployment.py')
    paths.update({name:C.CODE/name for name in ('common.py','pipeline.py','pose.py','observations.py')})
    return {name:C.binding(path) for name,path in paths.items()}


def fixed_case():
    case = C.read(C.DOC/'VISUAL_CASE_PROTOCOL.json')['cases'][0]
    matches = [r for r in C.read(C.OLD/'INPUTS.json')['frames'] if r['id'] == case['id']]
    C.require(len(matches) == 1, 'first preselected visual case is not unique')
    frame = matches[0]
    C.require(frame['image'] == case['image']['path'] and frame['image_sha256'] == case['image']['sha256'],
              'first visual case image identity differs')
    return case, frame


def references(frame):
    seal = C.read(C.DOC/'GEOMETRY_SEAL.json')
    C.require(seal['complete'] and seal['rows'] == 1225 and seal['fixed_rows'] == 490 and
              seal['GT_read_allowed'] is False, 'complete GT-free geometry seal required')
    C.bound(C.DOC/'GEOMETRY_SEALED.jsonl.gz', seal['geometry'], 'sealed geometry')
    C.bound(C.DOC/'FIXED_GEOMETRY_SEALED.jsonl.gz', seal['fixed_geometry'], 'fixed sealed geometry')
    selected = [r for r in C.rows(C.DOC/'GEOMETRY_SEALED.jsonl.gz')
                if r['id'] == frame['id'] and r['method'] == C.PRIMARY]
    n3 = [r for r in C.rows(C.DOC/'FIXED_GEOMETRY_SEALED.jsonl.gz')
          if r['id'] == frame['id'] and r['method'] == 'N3_SUBPIX']
    C.require(len(selected) == len(n3) == 1, 'single sealed smoke reference required')
    return selected[0], n3[0]


def preflight(args, frozen=True):
    C.require(args.source_root and args.baseline_root, 'explicit source-root/baseline-root required')
    for name in NAMES[1:]:
        destination(args, name)
    case, frame = fixed_case()
    reference, fixed = references(frame)
    path = Path(args.source_root)/frame['image']
    C.bound(path, case['image'], 'unchanged original RGB')
    identity = inputs(args)
    original = C.read(C.DOC/'PROTOCOL.json')['fixed_inputs']
    for name in ('common.py','pipeline.py','pose.py','observations.py'):
        C.require(identity[name]['sha256'] == original[name]['sha256'], 'frozen decision code differs ' + name)
    for actual, locked in (('calibration','calibration'),('training_completion','training_completion'),
                           ('corrected_ROLE','corrected_role_last'),('base_weights','base_weights'),
                           ('N3_weights','N3_weights')):
        C.require(identity[actual]['sha256'] == original[locked]['sha256'] and
                  identity[actual]['bytes'] == original[locked]['bytes'], 'fixed dependency differs ' + actual)
    completion = C.read(Path(args.fits)/'TRAINING_COMPLETION.json')
    C.require(completion['complete'] and completion['formal_updates'] == 9000,
              'corrected completed checkpoint required')
    from scripts.research.pallet_boundary_corner_refiner_20261010_v2 import protection
    preservation = protection.verify(C.read(DOC/'PROTECTION_BEFORE.json'), args.source_root)
    C.require(preservation['passed'], 'protected publication/source changed')
    probe = quiet()
    C.require(probe['quiet'] and probe['temperature_under_80'], 'SMOKE_PENDING: competing research/GPU or thermal guard')
    if frozen:
        protocol = C.read(destination(args, NAMES[0], exists=True))
        C.require(protocol['inputs'] == identity and protocol['case']['id'] == case['id'] and
                  protocol['lifecycles'] == [1,2] and protocol['predicts_each'] == 1 and
                  protocol['method'] == C.PRIMARY and protocol['metadata_argument'] is None,
                  'frozen smoke contract drift')
    return case, frame, reference, fixed, identity, preservation, probe


def freeze(args):
    destination(args, NAMES[0])
    case, frame, _, _, identity, _, probe = preflight(args, frozen=False)
    C.write_new(destination(args, NAMES[0]), dict(schema='fixed_public_deployment_two_lifecycle_usage_v1',
        inputs=identity, case=case, K=frame['K'], xyz_physical_WHD_m=frame['xyz'], raw_hw=frame['raw_hw'],
        selection='first existing frozen VISUAL_CASE_PROTOCOL case; no score-based selection',
        lifecycles=[1,2], predicts_each=1, method=C.PRIMARY, metadata_argument=None,
        parity_reference='existing GT-free sealed primary geometry and fixed N3 coordinates',
        atol_coordinates_px=1e-7, atol_pose=1e-7, rtol=0,
        expected_path_calls=dict(detector_calls=2,N3_route_calls=2,initial_pose_calls=2,
            ROLE_head_calls=2,feature_initial_pose_calls=2,final_pose_paths=2),
        expected_model_forwards=dict(detector=4,N3=2,ROLE=2,detector_internal_initialization_calls=2),
        expected_detector_initialization_note='Initialization may occur lazily in the first predict, not necessarily inside constructor.',
        canary='Original GT canary plus smoke cache-read denial around construct/predict/close',
        prefreeze_resource_snapshot=probe, new_training_updates=0, new_GT_evaluations=0,
        new_RGB_generated=0, timing_intervals=0, new_accuracy_aggregate=False,
        decision_or_checkpoint_changes=False, no_automatic_retry=True,
        limits=['One same-image sequential lifecycle smoke; no general deployment/domain claim.',
                'Parity is against stored inference geometry, not physical GT or an accuracy improvement.']))
    print('DEPLOYMENT_SMOKE_FROZEN', case['id'], flush=True)


@contextmanager
def canary(old):
    """Protect truth and cached evaluator outputs during the usable API body."""
    forbidden = ('INPUTS.json','COHORT.json','RUNTIME_PANEL.json','VISUAL_CASE_PROTOCOL.json',
                 'GEOMETRY_SEALED.jsonl.gz','FIXED_GEOMETRY_SEALED.jsonl.gz',
                 'PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz','OBSERVATIONS.jsonl.gz',
                 'POSTHOC_ROWS.jsonl.gz')
    originals = (builtins.open,io.open,Path.open)
    def guarded(fn):
        def wrapper(path,*a,**kw):
            if isinstance(path,(str,os.PathLike)) and Path(path).name in forbidden:
                raise AssertionError('DEPLOYMENT_SMOKE_CACHE_CANARY:' + Path(path).name)
            return fn(path,*a,**kw)
        return wrapper
    with old.no_truth_reads():
        inner = (builtins.open,io.open,Path.open)
        builtins.open,io.open,Path.open = map(guarded,inner)
        try:
            yield
        finally:
            builtins.open,io.open,Path.open = inner
    C.require((builtins.open,io.open,Path.open) == originals, 'smoke canary did not restore IO functions')


def compact_pose(pose):
    return {k:pose[k] for k in ('available','R_cf','R_physical','centroid','cf_extents',
                                'reprojection_px','selected_hypothesis','state','reason') if k in pose}


def run(args):
    case, frame, reference, fixed, identity, before, probe = preflight(args)
    started = destination(args, NAMES[1])
    C.write_new(started,dict(schema='public_deployment_smoke_execution_claim_v1',
        status='STARTED',started_at_UTC=utc(),protocol=C.binding(destination(args,NAMES[0],exists=True)),
        configured_lifecycles=2,configured_predicts=2,no_automatic_retry=True))
    result = dict(schema='actual_public_deployment_two_lifecycle_usage_checks_v1',
        passed=False,complete=False,status='FAILED',protocol=C.binding(destination(args,NAMES[0],exists=True)),
        input_bindings=identity,case_id=case['id'],actual_image_decode_calls=0,
        actual_constructor_attempts=0,actual_constructors_completed=0,actual_predicts_started=0,
        actual_predicts_complete=0,actual_closes_completed=0,
        resource_snapshots=[dict(phase='before_models',**probe)],new_GT_evaluations=0,
        new_training_updates=0,new_RGB_generated=0,timing_intervals=0,new_accuracy_aggregate=False)
    rows=[]; totals=Counter(); model_totals=Counter(); cv_calls=Counter()
    environment={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')}
    os.environ['PALLET_SOURCE_ROOT']=str(Path(args.source_root).resolve())
    os.environ['PALLET_BASELINE_ROOT']=str(Path(args.baseline_root).resolve())
    pipeline=None
    try:
        import cv2
        import numpy as np
        import torch
        import scripts.research as research
        from scripts.research.pallet_observation_refiner_20261009_v1 import common as old
        from scripts.research.pallet_boundary_corner_refiner_20261010_v2.deployment import Pipeline
        image=cv2.imread(str(Path(args.source_root)/frame['image']),cv2.IMREAD_COLOR)
        result['actual_image_decode_calls']=1
        C.require(image is not None and list(image.shape[:2])==frame['raw_hw'],'same original image decode differs')
        namespace=list(research.__path__)
        result['environment']=dict(python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__,
                                    opencv=cv2.__version__,CUDA=torch.version.cuda)
        C.require(torch.cuda.is_available(),'CUDA is unavailable')
        with C.primitive_counter() as primitive:
            try:
                for lifecycle in (1,2):
                    probe=quiet();result['resource_snapshots'].append(dict(lifecycle=lifecycle,phase='before_construct',**probe))
                    C.require(probe['quiet'] and probe['temperature_under_80'],'SMOKE_PENDING: resource gate')
                    one=dict(lifecycle=lifecycle,case_id=case['id'],method=C.PRIMARY,parity='PENDING',
                             GT_canary_active=True,cache_canary_active=True,metadata_argument=None)
                    rows.append(one)
                    previous=primitive.copy()
                    with canary(old),torch.no_grad():
                        result['actual_constructor_attempts']+=1
                        pipeline=Pipeline(args)
                        result['actual_constructors_completed']+=1
                        counters=Counter()
                        hooks=[pipeline.models.n3.register_forward_pre_hook(lambda *_,c=counters:c.update(N3=1)),
                               pipeline.head.register_forward_pre_hook(lambda *_,c=counters:c.update(ROLE=1))]
                        try:
                            one['model_forwards_before_predict']=dict(detector=pipeline.models.detector_forwards,
                                                                    N3=pipeline.models.n3_forwards)
                            result['actual_predicts_started']+=1
                            output=pipeline.predict(image,frame['K'],frame['xyz'],metadata=None)
                            torch.cuda.synchronize()
                            result['actual_predicts_complete']+=1
                            one['coordinate_parity']=pipeline.runtime._point_parity(output['native_points'],reference['native_points'],1e-7,case['id'])
                            one['fixed_N3_parity']=pipeline.runtime._point_parity(output['native_N3_points'],fixed['native_points'],1e-7,case['id'])
                            one['pose_parity']=pipeline.runtime._pose_parity(output['actual_pose'],reference['actual_pose'])
                            C.require(all(C.finite(output[k])==reference[k] for k in STATE_KEYS),'sealed outcome/metadata parity differs')
                            if output['new_pose_estimated']:
                                C.require(set(output['hidden_initial']).isdisjoint(output['solver']['fit_input_ids']),'hidden fit leakage')
                                H=output['hidden_initial']
                                if H:
                                    np.testing.assert_allclose(np.asarray(output['native_points'])[H],np.asarray(output['solver']['projected'])[H],rtol=0,atol=1e-12)
                            one.update(parity='PASS',state={k:output[k] for k in STATE_KEYS},
                                native_points=output['native_points'],native_N3_points=output['native_N3_points'],
                                actual_pose=compact_pose(output['actual_pose']),K=frame['K'],xyz=frame['xyz'],
                                fit_input_ids=output['solver'].get('fit_input_ids',[]),
                                observation_raw_logits_sha256=output.get('observation_raw_logits_sha256'),
                                detector_center_preserved=True,cache_reference_used_as_model_input=False)
                        finally:
                            one['actual_pipeline_calls']=dict(pipeline.counts)
                            one['actual_model_forwards']=dict(detector=pipeline.models.detector_forwards,
                                N3=pipeline.models.n3_forwards,ROLE=counters['ROLE'],
                                N3_independent_hook=counters['N3'],
                                detector_internal_initialization_calls=pipeline.models.detector_forwards-pipeline.counts['detector_calls'])
                            totals.update(pipeline.counts);model_totals.update(one['actual_model_forwards'])
                            for hook in hooks:hook.remove()
                            pipeline.close();result['actual_closes_completed']+=1;pipeline=None
                    one['OpenCV_entry_calls']={k:primitive[k]-previous[k] for k in primitive}
                    one['namespace_restored_after_close']=list(research.__path__)==namespace
                    one['facade_lifecycle_lock_released']=not Pipeline._active_lifecycle
                    C.require(one['namespace_restored_after_close'] and one['facade_lifecycle_lock_released'],'deployment lifecycle state not restored')
                    print('DEPLOYMENT_SMOKE_LIFECYCLE_PASS',lifecycle,case['id'],flush=True)
            finally:
                cv_calls.update(primitive)
        protocol=C.read(destination(args,NAMES[0],exists=True))
        C.require(dict(totals)==protocol['expected_path_calls'],'two complete prediction path counts differ')
        for k,v in protocol['expected_model_forwards'].items():
            C.require(model_totals[k]==v,'actual model forwards differ '+k)
        C.require(model_totals['N3_independent_hook']==2 and all(r['parity']=='PASS' for r in rows),'independent model/parity counts differ')
        probe=quiet();result['resource_snapshots'].append(dict(phase='after_two_lifecycles',**probe))
        C.require(probe['quiet'] and probe['temperature_under_80'],'SMOKE_PENDING: endpoint resource gate')
        result.update(passed=True,complete=True,status='PASS')
    except Exception as error:
        message=str(error)
        for root,label in ((args.source_root,'source'),(args.baseline_root,'baseline'),(args.fits,'checkpoints')):
            message=message.replace(str(Path(root).resolve()),'<'+label+'>')
        result['failure']=dict(type=type(error).__name__,message=message)
    finally:
        if pipeline is not None:
            pipeline.close()
        for name,value in environment.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        from scripts.research.pallet_boundary_corner_refiner_20261010_v2 import protection
        after=protection.verify(C.read(DOC/'PROTECTION_BEFORE.json'),args.source_root)
        result['protected_repository_files']=after['protected_repository_files']
        result['source_checkout_unchanged']=after['source_checkout_unchanged']
        result['protected_publication_unchanged']=after['passed']
        result['input_bindings_unchanged']=inputs(args)==identity
        if not after['passed'] or not result['input_bindings_unchanged']:
            result.update(passed=False,complete=False,status='FAILED_PROTECTION')
        C.save_rows(destination(args,NAMES[3]),rows)
        result.update(raw_rows=C.binding(destination(args,NAMES[3],exists=True)),
            actual_pipeline_calls=dict(totals),actual_model_forwards=dict(model_totals),
            actual_OpenCV_entry_calls=dict(cv_calls),completed_at_UTC=utc(),
            limits=['Exactly one preselected image, two sequential real public-facade lifecycles.',
                    'No GT scoring, accuracy aggregation, timed interval, new model or decision policy.',
                    'Parity against sealed inference output does not show better source selection or pose accuracy.'])
        C.write_new(destination(args,NAMES[2]),result)
    print('DEPLOYMENT_SMOKE',result['status'],result['actual_predicts_complete'],flush=True)
    C.require(result['passed'],'preserved deployment smoke failure; no automatic retry')


def main():
    parser=C.parser(__doc__,('freeze','preflight','run'))
    parser.set_defaults(output=str(DOC),protocol=str(DOC/NAMES[0]),
                        fits=str(C.DOC/'deployment'))
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args)
    elif args.stage=='preflight':
        case,*_=preflight(args);print('DEPLOYMENT_SMOKE_PREFLIGHT_PASS',case['id'],flush=True)
    else:run(args)


if __name__=='__main__':
    main()
