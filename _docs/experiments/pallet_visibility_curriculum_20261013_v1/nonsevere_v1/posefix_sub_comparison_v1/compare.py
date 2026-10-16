"""Frozen cached PoseFix95 -> inherited SubPix/F; post-seal matched scoring."""
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import builtins
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
ROOT = Path('/home/minjae/Documents/github/pallet-pose')
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

PF_CACHE = ROOT / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed1_REAL_DEV.npz'
PF_SELECTION = ROOT / '_docs/experiments/pallet_sensors_submission_v1/PRIOR_SELECTION.json'
POSE_SOURCE = ROOT / 'scripts/research/pallet_dim_conditioned_p_v1/pose.py'
SUB_SOURCE = ROOT / 'scripts/research/pallet_training_free_compare_20261007_v1/methods.py'
SOLVE_SOURCE = ROOT / 'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py'
ARMS = ('NATIVE_BASE', 'INITIAL_N3_SUBPIX', 'POSEFIX_RAW',
        'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR',
        'FUSED_ACTIVE', 'JOINT_SINGLE_FAMILY')
SUBPIX = dict(winSize=[5, 5], zeroZone=[-1, -1], minimum_image_extent=15,
              criteria=dict(maxCount=40, epsilon=.001))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    assert not Path(path).exists(), 'Preserve existing artifact: ' + str(path)
    Path(path).write_text(json.dumps(C.finite(value), indent=2, allow_nan=False) + '\n')


def rows(path):
    return [json.loads(line) for line in gzip.open(path, 'rt') if line.strip()]


@contextmanager
def no_labels():
    old = builtins.open, io.open, Path.open
    def wrap(fn):
        def guarded(path, *args, **kwargs):
            if any(value in str(path) for value in ('SCORING', 'SCORED_ROWS', 'ORACLE',
                    'TARGETS.json', '/annotations/', 'TRAINING_LABELS', 'FINAL', 'RESULTS.json')):
                raise AssertionError('Forbidden baseline inference label read: ' + str(path))
            return fn(path, *args, **kwargs)
        return guarded
    builtins.open, io.open, Path.open = map(wrap, old)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = old


def compile_functions(path, names, namespace):
    """Execute only unchanged inference definitions, never old GT loaders."""
    tree = ast.parse(Path(path).read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == set(names)
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace


def prepare():
    assert not (HERE / 'PROTOCOL.json').exists()
    frames = read(COHORT / 'PREDICTION_INPUTS.json')['frames']
    assert len(frames) == 95 and Counter(f['partition'] for f in frames) == {'CAL': 46, 'TEST': 49}
    assert all(f['population_role'] == 'DEV' for f in frames)
    ids = {f['id'] for f in frames}
    with np.load(PF_CACHE, allow_pickle=False) as packet:
        allids = [str(value) for value in packet['ids']]
        positions = [allids.index(f['id']) for f in frames]
        points = packet['points'][positions, 0, 1].copy()
        checkpoint = str(packet['checkpoint_sha256'].item())
    assert checkpoint == read(PF_SELECTION)['checkpoints']['1']
    inputs = []
    for frame, pf in zip(frames, points):
        q0 = np.asarray(frame['points']['BASE'], float)
        support = np.asarray(frame['prediction_support']['BASE'] if isinstance(frame['prediction_support'], dict)
                             else frame['prediction_support'], bool)
        assert support.shape == (9,)
        assert np.array_equal(pf[8], q0[8], equal_nan=True)
        assert np.array_equal(pf[~support], q0[~support], equal_nan=True)
        inputs.append(dict(frame=frame, PoseFix_points=pf.tolist(), support=support.tolist()))
    write(HERE / 'BASELINE_INPUTS.json', dict(prediction_only=True, frames=inputs))
    files = [PF_CACHE, PF_SELECTION, POSE_SOURCE, SUB_SOURCE, SOLVE_SOURCE,
        ROOT / 'challenge/evaluation_v2/pnp_selector.py', ROOT / 'scripts/annotate/pallet_geometry.py',
        ROOT / 'scripts/research/pallet_n3_subpix_20261008_v1/runtime.py',
        ROOT / 'scripts/research/pallet_n3_subpix_20261008_v1/common.py',
        ROOT / 'scripts/research/pallet_training_free_compare_20261007_v1/runtime.py',
        COHORT / 'PREDICTION_INPUTS.json', COHORT / 'COHORT_PROTOCOL.json',
        COHORT / 'progressive_support_guard_v1/PREDICTIONS.jsonl.gz',
        COHORT / 'progressive_support_guard_v1/PREDICTION_SEAL.json',
        COHORT / 'boundary_progressive_replay_v1/PREDICTIONS.jsonl.gz',
        COHORT / 'boundary_progressive_replay_v1/PREDICTION_SEAL.json',
        COHORT / 'test_reused_v1/PREDICTIONS.jsonl.gz', COHORT / 'test_reused_v1/PREDICTION_SEAL.json',
        COHORT / 'SAM_boundary_uncertainty_v1/PREDICTIONS.jsonl.gz',
        COHORT / 'SAM_boundary_uncertainty_v1/PREDICTION_SEAL.json',
        COHORT / 'joint_model/FULL_PREDICTION_CACHE.json', COHORT / 'joint_model/MODEL_FREEZE.json',
        ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/curriculum.py',
        ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/scoring.py',
        Path(__file__), HERE / 'BASELINE_INPUTS.json']
    snapshots = HERE / 'code_snapshot'
    snapshots.mkdir()
    for path in files:
        if path.suffix == '.py':
            target = snapshots / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write(HERE / 'PROTOCOL.json', dict(schema='POSEFIX_SUB_MATCHED_NONSEVERE95_V1',
        before_new_RGB_SubPix_pose_inference=True, frame_ids=sorted(ids),
        population=dict(CAL=46, TEST=49, severe=0, FINAL=0), arms=list(ARMS),
        source_bindings={str(path): sha(path) for path in files},
        PoseFix=dict(checkpoint_sha256=checkpoint, seed=1, step=6000,
            model='PoseFixPallet9 RGB ResNet152, synthetic-only PRIOR1',
            selected_cache_tensor='points[selected95,0,1]: RAW trajectory, first pass, uncapped',
            new_model_forwards=0, new_training=0, original_center_and_missing_support=True),
        new_combinations=dict(historical_PoseFix_SubPix_recipe_or_cache_found=False,
            SUBPIX=SUBPIX, cap_fraction_raw_image_diagonal=.01,
            POSEFIX_SUBPIX_PF_ANCHOR='Cap SubPix displacement relative to cached PoseFix q; learned PoseFix movement is uncapped',
            POSEFIX_SUBPIX_YOLO_ANCHOR='Cap total PoseFix+SubPix displacement relative to original YOLO q0, as N3+SubPix does',
            both_controls_fixed_before_scoring=True, no_best_anchor_selected=True),
        final_F='Unchanged pose.infer: prediction-only W/D selector then SQPnP+RefineLM on finite native8; original physical-axis conversion. No initial pose or human visibility/H supplied.',
        N3_initial='Original initial_pose field from N3_SUBPIX_NO_MASK_STANDARD, not its later actual_pose refit; YOLO→N3_DIM_SYM seed1→SubPix→YOLO-anchor cap→F',
        current_joint_CAL='Frozen FULL-CAL head P/delta cache plus unchanged fresh RGB/Tiny appearance from sealed SAM study; original-family curriculum, cutoff.45, bound16. In-sample CAL replay; not OOF.',
        current_joint_TEST='Already sealed JOINT_SINGLE_FAMILY49, no rerun or tuning',
        new_F_calls_maximum=475, new_image_SubPix_calls=95,
        pre_scoring_parity='Recompute nativeBASE and N3+SubPix F on95; verify stored initial poses to1e-10. No GT reference needed.',
        scoring='Only isolatedCAL46/TEST49 truth after665predictionseal; all rows/failures preserved, same physical symmetry metrics and paired session bootstrap10000',
        evidence='CAL in-sample and previously seen TEST exploratory; pipeline differences include learned upstream coordinates and chosen W/D initial family',
        main_API_solver_git_edits=0))
    print('COMPARISON_PREREGISTERED', sha(HERE / 'PROTOCOL.json'), flush=True)


def infer():
    assert not (HERE / 'PREDICTIONS.jsonl.gz').exists()
    p = read(HERE / 'PROTOCOL.json')
    bindings = dict(p['source_bindings'])
    repair_path = HERE / 'PRE_INFERENCE_REPAIR.json'
    if repair_path.exists():
        repair = read(repair_path)
        assert repair['protocol_sha256'] == sha(HERE / 'PROTOCOL.json')
        assert repair['old_runner_sha256'] == bindings[str(Path(__file__))]
        assert repair['new_runner_sha256'] == sha(__file__)
        bindings[str(Path(__file__))] = repair['new_runner_sha256']
    for path, digest in bindings.items():
        assert sha(path) == digest, path
    inputs = read(HERE / 'BASELINE_INPUTS.json')['frames']
    cv2.setNumThreads(1)
    output = []
    parity = {'BASE': 0., 'N3_SUBPIX': 0.}
    corner_calls = 0
    with no_labels():
        # Direct unchanged source definitions avoid importing its unused GT metadata loaders.
        solve_ns = compile_functions(SOLVE_SOURCE, ['cuboid', 'solve'], {'np': np, 'cv2': cv2})
        from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
        pose_ns = compile_functions(POSE_SOURCE, ['rotations', 'infer'],
            dict(np=np, cv2=cv2, cuboid=solve_ns['cuboid'], solve=solve_ns['solve'],
                 select_pnp_hypotheses=select_pnp_hypotheses))
        sub_ns = compile_functions(SUB_SOURCE, ['_inputs', 'cap_points', 'correct'],
            dict(np=np, cv2=cv2, Counter=Counter, METHODS=('SUBPIX', 'CVRANK'),
                 _configuration=lambda: {'SUBPIX': SUBPIX}))
        from scripts.research.pallet_visibility_curriculum_20261013_v1.curriculum import solve_curriculum
        def sealed(directory):
            seal = read(COHORT / directory / 'PREDICTION_SEAL.json')
            path = COHORT / directory / 'PREDICTIONS.jsonl.gz'
            assert seal['before_scoring'] and sha(path) == seal.get('sha256', seal.get('predictions_sha256'))
            return rows(path)
        cal_active = {r['id']: r for r in sealed('progressive_support_guard_v1') if r['arm'] == 'FUSED_SIGMA2_PROGRESSIVE_ACTIVE_SUPPORT'}
        test_active = {r['id']: r for r in sealed('boundary_progressive_replay_v1') if r['method'] == 'FUSED_ACTIVE'}
        test_joint = {r['id']: r for r in sealed('test_reused_v1') if r['method'] == 'JOINT_SINGLE_FAMILY'}
        cal_evidence = {r['id']: r for r in sealed('SAM_boundary_uncertainty_v1') if r['method'] == 'RAW_STAGED'}
        full = {(r['id'], r['corner']): r for r in read(COHORT / 'joint_model/FULL_PREDICTION_CACHE.json')}
        for index, entry in enumerate(inputs):
            f = entry['frame']; fid = f['id']; q0 = np.asarray(f['points']['BASE'], float)
            qn = np.asarray(f['points']['N3_SUBPIX'], float); pf = np.asarray(entry['PoseFix_points'], float)
            support = np.asarray(entry['support'], bool); K = np.asarray(f['K']); dims = np.asarray(f['xyz'])
            image = cv2.imread(str(ROOT / f['image']))
            assert image is not None and sha(ROOT / f['image']) == f['image_sha256']
            assert list(image.shape[:2]) == f['raw_hw']
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            native, diag = sub_ns['correct'](gray, pf, support, 'SUBPIX')
            corner_calls += diag['algorithm_corner_calls']
            pf_sub = sub_ns['cap_points'](pf, native, image.shape[1], image.shape[0], support)
            total_sub = sub_ns['cap_points'](q0, native, image.shape[1], image.shape[0], support)
            points = {'NATIVE_BASE': q0, 'INITIAL_N3_SUBPIX': qn, 'POSEFIX_RAW': pf,
                      'POSEFIX_SUBPIX_PF_ANCHOR': pf_sub, 'POSEFIX_SUBPIX_YOLO_ANCHOR': total_sub}
            for arm, q in points.items():
                assert np.array_equal(q[8], q0[8], equal_nan=True)
                assert np.array_equal(q[~support], q0[~support], equal_nan=True)
                actual = pose_ns['infer'](q, K, dims, source=False)
                if arm in ('NATIVE_BASE', 'INITIAL_N3_SUBPIX'):
                    key = 'BASE' if arm == 'NATIVE_BASE' else 'N3_SUBPIX'
                    reference = f['initial_poses'][key]
                    assert actual['available'] == reference['available']
                    for field in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
                        if actual['available']:
                            error = float(np.max(np.abs(np.asarray(actual[field]) - np.asarray(reference[field]))))
                            parity[key] = max(parity[key], error)
                            assert error < 1e-10, (fid, key, field, error)
                output.append(dict(id=fid, partition=f['partition'], session=f['session'], method=arm,
                    actual_pose=actual, native_points=q.tolist(), source='existing cached head/native points plus unchanged F',
                    SubPix=diag if 'POSEFIX_SUBPIX' in arm else None, new_training=0, GT_inputs=False))
            active = cal_active[fid] if f['partition'] == 'CAL' else test_active[fid]
            output.append(dict(id=fid, partition=f['partition'], session=f['session'], method='FUSED_ACTIVE',
                actual_pose=active['actual_pose'], native_points=active['observed_points'],
                source='already sealed progressive study/reused TEST replay', solution=active['solution'], GT_inputs=False))
            if f['partition'] == 'TEST':
                joint = test_joint[fid]
                actual = joint['actual_pose']; measured = joint['corrected_measurements']; solution = joint['solution']
            else:
                e = cal_evidence[fid]; H = np.asarray(e['self_hidden'], bool)
                app = deepcopy(e['appearance']); corrected = qn.copy()
                height, width = image.shape[:2]
                valid = np.isfinite(qn[:8]).all(1) & ~np.all(qn[:8] == -1, axis=1)
                inside = valid & (qn[:8,0] >= 0) & (qn[:8,0] < width) & (qn[:8,1] >= 0) & (qn[:8,1] < height)
                for k in range(8):
                    if not valid[k]:
                        app['states'][k] = 'UNKNOWN'; app['visibility_score'][k] = 0.; continue
                    if not inside[k]:
                        app['states'][k] = 'OUT_OF_FRAME'; app['visibility_score'][k] = 0.; continue
                    if H[k]:
                        app['states'][k] = 'SELF_OCCLUDED'; app['visibility_score'][k] = 0.; continue
                    head = full[fid, k]; prob = head['P_usable']; delta = np.asarray(head['RGB_delta_px'])
                    assert np.linalg.norm(delta) <= 16 + 1e-8
                    app['visibility_score'][k] = prob
                    if prob >= .45:
                        app['states'][k] = 'VISIBLE_BOUNDARY' if app['boundary_score'][k] >= .35 else 'VISIBLE_INTERIOR'
                        app['local_accuracy_score'][k] = max(.12, app['local_accuracy_score'][k])
                        candidate = qn[k] + delta
                        if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                            corrected[k] = candidate
                    else:
                        app['states'][k] = 'UNKNOWN'
                initial = f['initial_poses']['N3_SUBPIX']; R0 = np.asarray(initial['R_cf']); t0 = np.asarray(initial['centroid'])
                Q = R0.T @ np.asarray(initial['R_physical'])
                solution = solve_curriculum(corrected[:8], C.cuboid(*initial['cf_extents']), K, R0, t0, H, app,
                    dict(visible_threshold=.45, easy_visibility=.45, analytic_jacobian=False))
                actual = dict(initial, R_cf=solution['R'], centroid=solution['t'], R_physical=(np.asarray(solution['R']) @ Q).tolist())
                if not solution['accepted']:
                    actual = deepcopy(initial)
                measured = corrected.tolist()
            output.append(dict(id=fid, partition=f['partition'], session=f['session'], method='JOINT_SINGLE_FAMILY',
                actual_pose=actual, native_points=measured, solution=solution,
                source='frozen FULLCAL head/cache original-family replay' if f['partition']=='CAL' else 'already sealed JOINT_SINGLE_FAMILY TEST49', GT_inputs=False))
            if index % 20 == 0:
                print('BASELINE_INFERRED', index + 1, 95, flush=True)
    assert len(output) == 665
    for path, digest in bindings.items():
        assert sha(path) == digest, path
    with gzip.open(HERE / 'PREDICTIONS.jsonl.gz', 'wt') as stream:
        for row in output:
            stream.write(json.dumps(C.finite(row), separators=(',', ':'), allow_nan=False) + '\n')
    write(HERE / 'PREDICTION_SEAL.json', dict(before_scoring=True, complete=True, rows=665,
        predictions_sha256=sha(HERE / 'PREDICTIONS.jsonl.gz'), protocol_sha256=sha(HERE / 'PROTOCOL.json'),
        F_calls=475, SubPix_image_calls=95, SubPix_corner_calls=corner_calls,
        historical_BASE_N3_pose_parity_max_abs=parity, cached_PoseFix_tensor='points[:,0,1]',
        severe_frames=0, FINAL_frames=0, model_forwards=0, new_training=0, GT_inference_reads=0))
    print('BASELINES_SEALED', sha(HERE / 'PREDICTIONS.jsonl.gz'), flush=True)


def score():
    from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric, distribution
    seal = read(HERE / 'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and sha(HERE / 'PREDICTIONS.jsonl.gz') == seal['predictions_sha256']
    truth = {}
    for name, path in [('CAL', COHORT / 'OOF_pose_v1/ISOLATED_CAL_SCORING.json'),
                       ('TEST', COHORT / 'test_reused_v1/ISOLATED_TEST_SCORING.json')]:
        packet = read(path)['rows']; assert len(packet) == (46 if name == 'CAL' else 49)
        truth.update(packet)
    assert len(truth) == 95
    scored = rows(HERE / 'PREDICTIONS.jsonl.gz'); groups = defaultdict(list)
    for row in scored:
        row['metric'] = pose_metric(row['actual_pose'], truth[row['id']]['target']['truth']) if row['actual_pose']['available'] else dict(success_5cm_5deg=False, translation_cm=None, rotation_deg=None, ADDsym_cm=None)
        groups[row['partition'], row['method']].append(row)
    lookup = {(r['partition'], r['method'], r['id']): r for r in scored}
    summary = {}; paired = {}
    for (partition, method), group in groups.items():
        available = [r for r in group if r['actual_pose']['available']]
        summary[partition + '/' + method] = dict(frames=len(group), available=len(available), failures=len(group)-len(available),
            success_count=sum(r['metric']['success_5cm_5deg'] for r in group),
            metrics={key:distribution([r['metric'][key] for r in available]) for key in ('translation_cm','rotation_deg','ADDsym_cm')},
            metrics_are_all_frame=len(available)==len(group), selected_families=dict(Counter(r['actual_pose'].get('selected_hypothesis','unavailable') for r in group)))
    for partition in ('CAL', 'TEST'):
        for arm in ('FUSED_ACTIVE', 'JOINT_SINGLE_FAMILY'):
            for baseline in ('INITIAL_N3_SUBPIX', 'NATIVE_BASE', 'POSEFIX_RAW', 'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR'):
                group = groups[partition, arm]; sessions = sorted({r['session'] for r in group})
                rng = np.random.default_rng(20261013); draws = rng.integers(0,len(sessions),(10000,len(sessions)))
                report = {}; cases = []
                for row in group:
                    other = lookup[partition, baseline, row['id']]
                    a,b = row['metric'],other['metric']
                    cases.append(dict(id=row['id'],session=row['session'],success_before=b['success_5cm_5deg'],success_after=a['success_5cm_5deg'],
                        delta={key:a[key]-b[key] if a[key] is not None and b[key] is not None else None for key in ('translation_cm','rotation_deg','ADDsym_cm')}))
                for metric in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
                    valid = [r for r in group if metric=='success_5cm_5deg' or (r['metric'][metric] is not None and lookup[partition,baseline,r['id']]['metric'][metric] is not None)]
                    delta = np.asarray([float(r['metric'][metric])-float(lookup[partition,baseline,r['id']]['metric'][metric]) for r in valid])
                    sums = np.asarray([sum(d for d,r in zip(delta,valid) if r['session']==s) for s in sessions]);counts = np.asarray([sum(r['session']==s for r in valid) for s in sessions])
                    denominator = counts[draws].sum(1); boot = sums[draws].sum(1)[denominator>0]/denominator[denominator>0]
                    report[metric] = dict(paired_frames=len(valid),excluded_pairs=len(group)-len(valid),mean_difference=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist())
                paired[partition+'/'+arm+'-'+baseline] = dict(metrics=report,
                    success_rescues=[r['id'] for r in cases if r['success_after'] and not r['success_before']],
                    success_losses=[r['id'] for r in cases if r['success_before'] and not r['success_after']],
                    both_T_R_improved=[r['id'] for r in cases if all(r['delta'][k] is not None and r['delta'][k]<0 for k in ('translation_cm','rotation_deg'))],
                    both_T_R_worsened=[r['id'] for r in cases if all(r['delta'][k] is not None and r['delta'][k]>0 for k in ('translation_cm','rotation_deg'))],cases=cases)
    with gzip.open(HERE / 'SCORED_ROWS.jsonl.gz', 'wt') as stream:
        for row in scored:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE / 'COMPARISON.json', dict(schema='POSEFIX_SUB_MATCHED_COMPARISON_V1', summary=summary, paired=paired,
        primary='Same previously observed TEST49, no severity or success-case deletion',CAL='Full-fitted confidence head in-sample exploratory; independent OOF head outcomes reported elsewhere',
        new_PoseFix_Sub_combinations=True, anchors_selected_from_outcomes=False, new_training=0, severe_frames=0, FINAL_labels_read=0,
        prediction_seal_sha256=sha(HERE/'PREDICTION_SEAL.json'), isolated_scoring_sha256={str(path):sha(path) for path in (COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json')}))
    for name,value in summary.items():
        print(name, value['success_count'], '/',value['frames'], 'T',value['metrics']['translation_cm']['mean'],'R',value['metrics']['rotation_deg']['mean'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','infer','score'));args=parser.parse_args()
    globals()[args.phase]()
