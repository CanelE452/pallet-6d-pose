"""Immutable new-experiment contracts. Never write to historical experiment paths."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments/pallet_task_risk_active_v1'
RAW = ROOT / 'data/pallet/results/pallet_task_risk_active_v1'
OLD_DOC = ROOT / '_docs/experiments/pallet_active_learning_v1/retrospective_v1'
OLD_RAW = ROOT / 'data/pallet/results/pallet_active_learning_v1/retrospective_v1'
OLD_CODE = ROOT / 'scripts/research/pallet_active_learning_v1'
POSE = ROOT / 'data/pallet/results/paper_pose_metric_closure_v1'
R0 = ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
R0_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
BASELINE = ROOT / 'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'
REGISTRY = ROOT / 'challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json'
NEGATIVE = ROOT / 'challenge/real_gt_v2/manifests/DEV_NEG2689.json'
SEED = 20260913
BOOTSTRAPS = 10000


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda: f.read(1024 * 1024), b''):
            h.update(value)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    assert any(path.resolve().is_relative_to(p.resolve()) for p in (DOC, RAW)), path
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f'Immutable artifact differs: {path}'
    else:
        with path.open('x') as f:
            f.write(text)


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_modules():
    for p in (ROOT, ROOT / 'scripts/paper/pose_metric_closure_v1'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import run_pose_evaluation
    import build_geometry_resolved_pose_gt
    import symmetry_aware_pose_metrics
    return run_pose_evaluation, build_geometry_resolved_pose_gt, symmetry_aware_pose_metrics


def gpu():
    query = lambda kind, fields: subprocess.check_output(
        ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], text=True).strip()
    status = query('gpu', 'name,temperature.gpu,memory.used,utilization.gpu,power.draw')
    processes = query('compute-apps', 'pid,process_name,used_memory')
    other = [r for r in processes.splitlines() if r.split(',')[0].strip() != str(os.getpid())]
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), gpu=status, compute=processes,
                  foreign_compute=other, system_changes=False)
    if other:
        write(DOC / ('RESOURCE_BUSY_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json'), result)
        raise RuntimeError('GPU busy: stop without waiting or changing foreign processes')
    assert float(status.split(',')[1]) < 80, 'Thermal guard'
    return result


def camera_binding(row):
    image = ROOT / row['image_path']
    options = [image.parent.parent / 'cam_K.txt']
    for source in row['source_matches']:
        p = ROOT / source
        options.append(p.parent.parent / 'cam_K.txt')
        if '/wood/selected/' in source:
            options.append(ROOT / 'data/pallet/raw_data/wood' / ('_annotate_' + p.parent.name) / 'cam_K.txt')
    found = list(dict.fromkeys(p for p in options if p.is_file()))
    assert found, ('TASK_RISK_INPUT_CONTRACT_BLOCKED', row['frame_id'])
    matrices = [np.loadtxt(p).reshape(3, 3) for p in found]
    assert all(np.isfinite(m).all() and m[0, 0] > 0 and m[1, 1] > 0 for m in matrices)
    assert all(np.array_equal(matrices[0], m) for m in matrices)
    return dict(K=matrices[0].tolist(), sources=[str(p.relative_to(ROOT)) for p in found],
                source_sha256=[sha(p) for p in found])


def setup():
    assert not (DOC / 'PROTOCOL_LOCK.json').exists(), 'Already locked; audit rather than reinitialize'
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'main'
    assert sha(R0) == R0_SHA
    old_lock, split, old_results = [read(OLD_DOC / name) for name in ('PROTOCOL_LOCK.json', 'SPLIT.json', 'RESULTS.json')]
    for p, h in old_lock['source_bindings'].items():
        assert sha(ROOT / p) == h, p
    assert read(OLD_DOC / 'VERDICT.json')['verdict'] == 'RETROSPECTIVE_AL_NO_SIGNAL'
    assert len(split['pool']) == 174 and len(split['evaluation']) == 145
    assert len(read(NEGATIVE)['items']) == 2689
    for group in ('pool', 'evaluation'):
        for r in split[group]:
            assert sha(ROOT / r['image_path']) == r['image_sha256']
    assert not {r['image_sha256'] for r in split['pool']} & {r['image_sha256'] for r in split['evaluation']}
    assert not {r['capture_session'] for r in split['pool']} & {r['capture_session'] for r in split['evaluation']}
    write(DOC / 'SPLIT_BINDING.json', dict(source_sha256=sha(OLD_DOC / 'SPLIT.json'),
        pool=split['pool'], evaluation=split['evaluation'], pool_count=174, evaluation_count=145,
        negative_count=2689, exact_image_hash_overlap=0, session_groups_unchanged=True))
    previous = {}
    for method in old_lock['methods']:
        runs = [old_results['per_run'][f'{method}_seed{s}'] for s in (1, 2, 3)]
        supplied = old_results['seed_means'][method]
        assert abs(np.mean([r['metrics']['box_ap50_95'] for r in runs]) - supplied['ap50_95']) < 1e-12
        for key in ('median_px', 'p90_px'):
            assert abs(np.mean([r['geometry'][method][key] for r in old_results['paired']]) - supplied['geometry'][key]) < 1e-12
        previous[method] = supplied
    write(DOC / 'PREVIOUS_RESULT_BINDING.json', dict(old_verdict='RETROSPECTIVE_AL_NO_SIGNAL',
        unchanged=True, seed_means_recomputed=True, seed_means=previous, paired_per_seed=old_results['paired'],
        R0=old_results['per_run']['R0'], result_sha256=sha(OLD_DOC / 'RESULTS.json')))
    registry = {r['object_type']: r for r in read(REGISTRY)['objects']}
    inputs, meta_paths = [], []
    for r in split['pool']:
        binding = camera_binding(r)
        dims = registry[r['object_type']]['physical_dimensions_m']
        long, short = max(dims['x'], dims['z']), min(dims['x'], dims['z'])
        session_meta = ROOT / r['image_path']
        session_meta = session_meta.parent.parent / 'session.json'
        declared = read(session_meta).get('object_type') if session_meta.is_file() else None
        assert declared in (r['object_type'], 'wood' if 'wood' in r['object_type'] else 'plastic')
        meta_paths.extend(ROOT / p for p in binding['sources'])
        meta_paths.append(session_meta)
        inputs.append(dict(frame_id=r['frame_id'], image_path=r['image_path'], image_sha256=r['image_sha256'],
            object_type=r['object_type'], camera=binding, dimensions=[long, short, dims['y']],
            category_source=str(session_meta.relative_to(ROOT))))
    write(RAW / 'TASK_INPUTS.json', inputs)
    write(DOC / 'TASK_SPACE_INPUT_CONTRACT.json', dict(status='GT_FREE_INPUTS_RESOLVED', frames=174,
        intrinsics='Existing session/raw cam_K.txt, never annotation camera_data. Wood45 retains known scaled sensor profile; calibration quality is not upgraded.',
        dimensions='Existing fixed category OBJECT_GEOMETRY_REGISTRY physical dimensions; category from session.json verified against frozen pool metadata; no frame-specific GT dimensions.',
        no_new_nominal_size=True, category_registry_sha256=sha(REGISTRY), inputs_sha256=sha(RAW / 'TASK_INPUTS.json'),
        hypothesis='Unchanged select_pnp_hypotheses; CF_WIDTH/CF_DEPTH, then existing canonical SQPnP+RefineLM solve on first8 keypoints.',
        coordinates='Camera-frame x=t[0], z=t[2] in meters; heading atan2(R[0,2],R[2,2]) in radians from the canonical MAIN rotation. No new robot extrinsic assumption.',
        fields_requiring_gt=False, target_error_or_true_axis_used=False,
        independent_session_K_vs_annotation_parity='Checked later in Phase1 after formula/code lock; discrepancy stops, never changes K from GT.'))
    sources = [R0, BASELINE, REGISTRY, NEGATIVE, RAW / 'TASK_INPUTS.json',
        OLD_RAW / 'pool/FEATURES.npz', OLD_RAW / 'pool/ACQUISITION_SIGNALS.json',
        POSE / 'POSE_EVAL_OBJECT_CONTRACT.json', POSE / 'GT_AXIS_RESOLUTION_LOCK.json',
        POSE / 'GEOMETRY_RESOLVED_POSE_GT.json', POSE / 'AXIS_REVIEW_MANIFEST.json',
        ROOT / 'challenge/evaluation_v2/paper_real_eval.py', ROOT / 'challenge/evaluation_v2/pnp_selector.py',
        ROOT / 'scripts/annotate/annotate_sessions.py', ROOT / 'scripts/annotate/annotate_wood.py',
        ROOT / 'scripts/annotate/object_geometry_registry.py', *meta_paths]
    sources += list(OLD_CODE.glob('*.py')) + [p for p in OLD_DOC.rglob('*') if p.is_file()]
    sources += [ROOT / r['label_path'] for r in split['pool'] + split['evaluation']]
    for method in old_lock['methods']:
        for seed in (1, 2, 3):
            folder = OLD_RAW / 'runs' / f'{method}_seed{seed}'
            sources += [folder / n for n in ('last.pt', 'EXPOSURE.json', 'TRAINING_AUDIT.json')]
            folder = OLD_RAW / 'evaluation' / f'{method}_seed{seed}'
            sources += [folder / n for n in ('RESULT.json', 'PER_FRAME.json', 'PREDICTIONS.json')]
    sources += list((ROOT / 'scripts/paper/pose_metric_closure_v1').glob('*.py'))
    sources += [p for p in (ROOT / '_docs/paper/final').rglob('*') if p.is_file()]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(sources)}
    write(DOC / 'SOURCE_BINDING.json', dict(paths=bindings, source_GT_files_hashed_only=True,
        GT_target_values_deserialized=False, old_controls_retrained=False))
    write(DOC / 'INPUT_CONTRACT_AUDIT.json', dict(status='PASS', R0_exact=True,
        split_hash_exact=True, pool_eval_hash_overlap=0, old_verdict_preserved=True, input_frames=174,
        old_source_lock_verified=True, source_files_bound=len(bindings), no_target_GT_read=True))
    print('PHASE0_PASS', len(bindings), 'source bindings', flush=True)


def lock():
    from perturb import DEFINITION
    from risk import DEFINITION as RISK
    old = read(OLD_DOC / 'PROTOCOL_LOCK.json')
    write(DOC / 'PERTURBATION_LOCK.json', DEFINITION)
    write(DOC / 'TASK_RISK_DEFINITION.json', RISK)
    write(DOC / 'METRIC_LOCK.json', dict(primary='translation_CVaR90_cm',
        definition='Mean of worst ceil(0.10*N) per-frame errors. N145 ->15. Fixed full145 denominator; primary valid only when all compared arms pose coverage=1.',
        secondary=['yaw_CVaR90_deg','kp_frame_mean_CVaR90_px','common_pooled_kp_P90','common_pooled_kp_median','gross20','AP50_95','Det','AUROC','FPR95','translation_median','yaw_median','IoU3D','ADDsym_AUC','pose_coverage'],
        missing_pose='Do not redefine primary on common-valid. Proposed coverage loss -> safety FAIL.',
        student_gate=dict(A='proposed translation CVaR90 seed mean < diversity AND better in>=2/3 paired seeds',
            B='same versus old geometry_weighted_diversity', C='yaw CVaR90 mean <= max(diversity,old_geometry)',
            D='pooled common kp P90 mean <= BOTH random and diversity',
            E='Det loss versus best(random,diversity,old_geometry)<=1/145; pose coverage1',
            F='No primary swapping; AP/median gains do not override primary fail')))
    write(DOC / 'FULL174_REFERENCE_PROTOCOL.json', dict(name='FULL174_FIXED_COMPUTE_REFERENCE',
        seed=1, updates=300, real_labels=174, real_slots=2400, mean_slots_per_label=2400/174,
        selected30_mean_slots_per_label=80, role='one-seed descriptive reference, NOT upper bound',
        executable_only_if_mechanism_pass=True))
    files = list(HERE.glob('*.py'))
    write(DOC / 'PROTOCOL_LOCK.json', dict(status='FROZEN_BEFORE_POOL_GT_ERROR_READ',
        timestamp=datetime.now(timezone.utc).isoformat(), start_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch='main', old_verdict='RETROSPECTIVE_AL_NO_SIGNAL', old_primary_unchanged=True,
        source_binding_sha256=sha(DOC / 'SOURCE_BINDING.json'),
        code_bindings={str(p.relative_to(ROOT)):sha(p) for p in files},
        definition_bindings={p.name:sha(p) for p in [DOC/'PERTURBATION_LOCK.json',DOC/'TASK_RISK_DEFINITION.json',DOC/'METRIC_LOCK.json',DOC/'TASK_SPACE_INPUT_CONTRACT.json']},
        pipeline='Lock -> old-selection poolGT audit -> isolated GT-free1392 inference -> freeze risk -> poolGT predictive diagnostic -> conditional selection/training -> reserved145 scoring',
        GT_scope='Phase1 may access pool174 labels AFTER this lock. Extraction is a separate process with GT read denial. Reserved145 labels and mixed319 GT-value containers stay unopened for value parsing until Phase6.',
        task_GT='Rebuild pool-only geometry-resolved GT using unchanged canonical cuboid/solve/min-residual rule from pool label keypoint_annotations; do not parse mixed319 GT container.',
        stage0_gates=dict(G0_original_pose_min=.90,G0_at_least6of8_min=.90,G1_operational_AUROC_min=.65,
            G2_AUROC_gain_min=.05,G3_positive_spearman_sessions_min=6,within_session_min_valid_frames=8,
            G4_other_operational_AUROC_loss_max=.03, same_operational_target_for_G1_G2_G3=True),
        diagnostic=dict(hard='error>=pool80th percentile, ties all hard; missing error ranks worst',
            score_top30_ties='deterministic original frozen pool order; exactly30', bootstrap_draws=BOOTSTRAPS,
            seed=SEED, bootstrap_hard_labels='Original frozen pool hard20 labels retained when resampling pairs',
            bootstrap_single_class='Undefined AUROC draws excluded and their count reported',
            within_session='Finite-error entries, >=8, nonconstant score and error; report undefined explicitly'),
        acquisition=dict(budget=30,weight='1+R_task, NOT reranked', distance='existing squared-distance k-center',
            initial='same centroid nearest', temporal_gap_ns=2000000000, selection_seed=SEED),
        training=dict(proposed_seeds=[1,2,3],full174_seeds=[1],max_fits=4,max_updates=1200,
            initialization='same existing R0 exact SHA', updates_per_fit=300, synthetic_batch=24,real_batch=8,
            synthetic_slots=7200,real_slots=2400,training_hyp=old['training_hyp'],
            implementation='Exact old simulation.py train/loader/optimizer behavior in isolated new output roots; no BN-freeze change',
            same_seed_synthetic_tensor_parity_required=True, last_step_only=True, controls_retrained=False),
        inference_budget=dict(pool_images=174,views=8,new_real_image_forwards=1392,optimizer_updates=0,
            framework_dummy_warmup='Record separately; no extra real-image warmup',
            P0_cached_stock_parity=dict(coordinate_atol_px=1e-4,score_atol=1e-6,candidate_index_exact=True)),
        no_tuning=True,no_independent_confirmation=True,no_novelty_claim=True))
    print('PROTOCOL_LOCKED', flush=True)


def verify_lock():
    lock = read(DOC / 'PROTOCOL_LOCK.json')
    assert sha(DOC/'SOURCE_BINDING.json') == lock['source_binding_sha256']
    for p, h in lock['code_bindings'].items():
        assert sha(ROOT/p) == h, p
    for p, h in lock['definition_bindings'].items():
        assert sha(DOC/p) == h, p


if __name__ == '__main__':
    {'setup':setup,'lock':lock}[sys.argv[1]]()
