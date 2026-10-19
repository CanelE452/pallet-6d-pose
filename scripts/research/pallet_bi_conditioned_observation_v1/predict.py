"""Blind real observation selection and fresh common-backend pose prediction.

The historical five complete systems are copied only as external comparators.
The eight matched groups select actual image observations and solve poses;
their model forward accepts an explicit numeric feature allowlist only.  This
program never accepts a real reference, target, annotation or error table.
Run as a fresh process so the existing Python I/O guard can audit all reads.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import faulthandler
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import torch

from scripts.research.pallet_risk_information_v1.prediction_guard import PredictionGuard
from . import bank as B
from . import features as F
from . import models as M
from . import solver as S
from .registry import ARMS, LEARNED, SEEDS


HISTORICAL_SOURCE = dict(N0_HISTORY='N0', E0_HISTORY='E0',
    H1_REP_HISTORY='D1_H1_MLP_s20261018',
    POSEFIX_PF_ANCHOR_HISTORY='POSEFIX_SUBPIX_PF_ANCHOR',
    POSEFIX_YOLO_ANCHOR_HISTORY='POSEFIX_SUBPIX_YOLO_ANCHOR')
MATCHED = ('RAW_OBSERVATION_MATCHED', 'LEGACY_RULE_MATCHED') + LEARNED
IDENTITY = ('id', 'partition', 'session', 'cluster_id')
FRAME_INPUT_FIELDS = IDENTITY + ('split', 'K', 'native_xyz', 'raw_hw', 'initial_points', 'initial_pose', 'image_sha256')
STATUS_FIELDS = ('accepted', 'fallback', 'technical_failure', 'timeout', 'solver_rejected',
    'ambiguous_but_accepted', 'historical_common_accepted', 'historical_common_fallback',
    'none_selected', 'action_role', 'selected_candidate_id')


def digest(value):
    return hashlib.sha256(json.dumps(S._json(value), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def save_rows(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            for row in records:
                compressed.write((json.dumps(S._json(row), sort_keys=True, separators=(',', ':'),
                    allow_nan=False) + '\n').encode())
    pending.replace(path)


def _pose_valid(pose):
    if not isinstance(pose, dict):
        raise ValueError('Historical actual_pose must be a mapping')
    if not pose.get('available'):
        return
    R = np.asarray(pose.get('R_physical'), float); t = np.asarray(pose.get('centroid'), float)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('Available historical pose must preserve finite R_physical/centroid')
    # No GT/reference orthonormalization, clipping, symmetry or pose editing.


def historical_entry(source, source_arm):
    """Allowlisted actual pose and status; historical errors never propagate."""
    pose = deepcopy(source['actual_pose']); _pose_valid(pose)
    old_status = source.get('status', source.get('prediction_status', 'HISTORICAL_OUTPUT'))
    status = old_status if isinstance(old_status, str) else 'HISTORICAL_STRUCTURED_STATUS'
    return dict(actual_pose=pose, status=status, prediction_status=status,
        **{k: deepcopy(source[k]) for k in STATUS_FIELDS if k in source},
        historical_status=deepcopy(old_status), historical_copy=True,
        historical_source_arm=source_arm, external_complete_system=True,
        actual_new_pose_optimizer_calls=0, current_reference_inputs=False)


def _eligible_bank(bank):
    return dict(bank, corners=[c for c in bank['corners'] if bank['states'][c['native_corner']] in ('B', 'I')])


def raw_from_bank(bank):
    """Exactly the shared q0 candidates, including exact-match original CAL noise."""
    view = _eligible_bank(bank); corners = []; selected = {}
    for corner in view['corners']:
        keep = [c for c in corner['candidates'] if c['kind'] == 'keep']
        if len(keep) != 1 or keep[0]['applied_xy'] != corner['q0']:
            raise ValueError('One original N0 keep candidate is mandatory')
        corners.append(dict(corner, partial_line=None, partial_original_visible=None))
        selected[corner['native_corner']] = keep[0]['candidate_id']
    return S.selection_from_candidates(dict(view, corners=corners), selected)


def infer_selection(bank, feature, model, norm):
    """Only allowlisted feature arrays enter the scorer; automatic B/I is fixed."""
    if feature['id'] != bank['frame_id']:
        raise ValueError('Feature/bank frame identity mismatch')
    rows = {int(r['native_corner']): r for r in feature['corner_rows']}
    if len(rows) != len(feature['corner_rows']):
        raise ValueError('Duplicate feature native corner')
    view = _eligible_bank(bank); selected = {}; choices = []
    expected = {c['native_corner'] for c in view['corners']}
    if set(rows) != expected:
        raise ValueError('Feature rows must cover exactly the fixed automatic visible set')
    with torch.inference_mode():
        for corner in view['corners']:
            k = corner['native_corner']; candidates = corner['candidates']; row = rows[k]
            ids = [c['candidate_id'] for c in candidates]
            if (row['candidate_ids'] != ids or row['state'] != bank['states'][k]
                    or row['applied_xy'] != [c['applied_xy'] for c in candidates]
                    or bool(row['partial_replaces_full_likelihood']) != bool(corner['partial_line'])):
                raise ValueError('Common feature/candidate/state/partial join changed')
            partial = corner['partial_line'] is not None
            if partial:
                keep = [c for c in candidates if c['kind'] == 'keep']
                if len(candidates) != 1 or len(keep) != 1:
                    raise ValueError('Fixed partial applies only without full alternatives')
                index = 0; scores = [None]; condition = None
            else:
                # IDs, paths, partition, coordinates, references, and bank
                # metadata cannot enter this model call. M.normalized reads
                # only these two explicit matrices and the train normalizer.
                numeric = dict(x_values=row['x_values'], x_validity=row['x_validity'])
                x = torch.from_numpy(M.normalized(numeric, norm))
                condition = F.condition(bank['states'][k], model.arm)
                conditions = torch.from_numpy(np.repeat(condition[None], len(x), axis=0).astype(np.float32))
                predicted = model(x, conditions).detach().cpu().numpy()
                if predicted.shape != (len(candidates),) or not np.isfinite(predicted).all():
                    raise FloatingPointError('Scorer produced nonfinite or mis-shaped candidate scores')
                scores = predicted.astype(float).tolist()
                index = min(range(len(candidates)), key=lambda j: (scores[j], tuple(candidates[j]['tie_break_key'])))
            candidate = candidates[index]; selected[k] = candidate['candidate_id']
            choices.append(dict(native_corner=k, automatic_state=bank['states'][k],
                selected_candidate_id=candidate['candidate_id'], selected_candidate_index=index,
                selected_kind='partial_normal' if partial else candidate['kind'],
                selected_score=scores[index], candidate_ids=ids, candidate_scores=scores,
                scorer_executed=not partial, scorer_condition=condition.tolist() if condition is not None else None,
                q0=deepcopy(candidate['q0']), raw_intersection_xy=deepcopy(candidate['raw_intersection_xy']),
                blend_alpha=candidate['blend_alpha'], applied_xy=None if partial else deepcopy(candidate['applied_xy']),
                covariance_px2=None if partial else deepcopy(candidate['covariance_px2']),
                source_line_ids=deepcopy(candidate.get('source_line_ids', [])),
                physical_source_ids=deepcopy(candidate.get('physical_source_ids', [])),
                partial=partial, typed_partial=deepcopy(corner['partial_line'])))
    return S.selection_from_candidates(view, selected), choices


def exact_solve_key(frame, selection):
    # Frame-specific camera/model/initialization is included even though the
    # cache is local to one frame. Metadata equality is conservative: no cache
    # reuse is inferred from pose proximity or identical selected IDs alone.
    return digest(dict(frame={k: frame[k] for k in ('K', 'native_xyz', 'raw_hw', 'initial_pose')},
        observations={k: selection[k] for k in ('states', 'full_points', 'partial_lines')},
        backend=S.contract()))


def execute_matched(frame, bank, feature, legacy, models, norm):
    """Actual selection -> exact packet cache -> actual fresh R,t, all eight arms."""
    if bank['frame_id'] != frame['id'] or feature['id'] != frame['id']:
        raise ValueError('Frame/common-bank identity mismatch')
    if legacy['states'] != bank['states']:
        raise ValueError('Legacy and learned backend states must remain identical')
    cache = {}; output = {}; details = {}; proofs = []
    for name in MATCHED:
        tick = time.monotonic(); selection_cpu = time.process_time(); choices = []
        if name == 'RAW_OBSERVATION_MATCHED':
            selection = raw_from_bank(bank)
        elif name == 'LEGACY_RULE_MATCHED':
            selection = S.legacy_selection(legacy, partial_lines=legacy['boundary_observations'])
        else:
            selection, choices = infer_selection(bank, feature, models[name], norm)
        selection_wall_seconds = time.monotonic() - tick
        selection_CPU_seconds = time.process_time() - selection_cpu
        signature = exact_solve_key(frame, selection)
        solve_tick, solve_cpu = time.monotonic(), time.process_time()
        if signature in cache:
            origin, result, canonical = cache[signature]
            if canonical != {k: selection[k] for k in ('states', 'full_points', 'partial_lines')}:
                raise RuntimeError('Hash equality without exact observation equality')
            reused = True; actual_calls = 0
        else:
            result = S.solve_frame(frame, selection); origin = name
            cache[signature] = (name, result, deepcopy({k: selection[k] for k in ('states', 'full_points', 'partial_lines')}))
            reused = False; actual_calls = result['optimizer_calls']
        solve_wall_seconds = time.monotonic() - solve_tick
        solve_CPU_seconds = time.process_time() - solve_cpu
        output[name] = dict(actual_pose=deepcopy(result['actual_pose']), accepted=result['accepted'],
            fallback=result['fallback'], status=result['status'], prediction_status=result['status'],
            technical_failure=result['status'] in ('invalid_observation', 'numerical_failure', 'optimizer_failed'),
            solver_rejected=not result['accepted'], choices=choices,
            observation_signature=signature, exact_observation_cache_reused=reused,
            solver_origin_arm=origin, actual_new_pose_optimizer_calls=actual_calls,
            backend_optimizer_calls_for_unique_packet=result['optimizer_calls'],
            fallback_reason=result.get('fallback_reason'), family_statuses=result.get('family_statuses', {}),
            current_reference_inputs=False, historical_copy=False,
            model_choice_wall_seconds=selection_wall_seconds if name in LEARNED else 0.,
            model_choice_CPU_seconds=selection_CPU_seconds if name in LEARNED else 0.,
            selection_wall_seconds=selection_wall_seconds, selection_CPU_seconds=selection_CPU_seconds,
            fresh_solver_wall_seconds=solve_wall_seconds if not reused else 0.,
            fresh_solver_CPU_seconds=solve_CPU_seconds if not reused else 0.,
            elapsed_selection_and_solve_seconds=time.monotonic()-tick)
        details[name] = dict(selection=selection, choices=choices, solver_result=deepcopy(result),
            exact_observation_signature=signature, cache_origin_arm=origin, solver_executed_this_arm=not reused)
        proofs.append(dict(arm=name, signature=signature, origin_arm=origin, cache_reused=reused,
            exact_numeric_and_metadata_packet_equality_verified=True, initialization_and_backend_equal=True,
            actual_new_pose_optimizer_calls=actual_calls, solver_status=result['status'],
            fresh_solver_wall_seconds=solve_wall_seconds if not reused else 0.,
            fresh_solver_CPU_seconds=solve_CPU_seconds if not reused else 0.))
    return output, details, proofs


def _load_models(root, plan_hash, normalizer):
    result = {}; bindings = []
    for seed in SEEDS:
        for arm in ('COMMON', 'BI'):
            name = f'OBS_{arm}_s{seed}'; folder = root / 'training' / name
            cp_path = folder / 'best.pt'; receipt_path = folder / 'RUN_RECEIPT.json'
            receipt = B.read(receipt_path)
            if (receipt['status'] != 'COMPLETE' or receipt.get('complete') is not True
                    or receipt['arm'] != arm or receipt['seed'] != seed
                    or receipt['checkpoint_sha256'] != B.sha(cp_path)):
                raise ValueError('Every fixed model must have its completed, bound best checkpoint')
            checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
            bound = checkpoint['bindings']['scientific']
            if (checkpoint['arm'] != arm or checkpoint['seed'] != seed or checkpoint['normalizer'] != normalizer
                    or bound['normalizer_sha256'] != B.sha(root / 'TRAIN_ONLY_NORMALIZER.json')
                    or bound['experiment_plan_sha256'] != plan_hash
                    or bound['model_source_sha256'] != B.sha(M.__file__)
                    or bound['feature_source_sha256'] != B.sha(F.__file__)
                    or checkpoint['bindings'] != receipt['bindings']):
                raise ValueError('Checkpoint feature, model, plan, normalizer or receipt binding changed')
            model = M.Scorer(arm); model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
            result[name] = model
            bindings.append(dict(arm=name, checkpoint_path=str(cp_path), checkpoint_sha256=B.sha(cp_path),
                receipt_path=str(receipt_path), receipt_sha256=B.sha(receipt_path),
                loaded_state_sha256=M.state_hash(model), best_epoch=receipt['best_epoch']))
    return result, bindings


def _bind_frame(folder, frame, run_binding):
    seal = B.read(folder / 'SEAL.json')
    if (not seal['complete'] or seal['id'] != frame['id'] or seal['partition'] != frame['partition']
            or seal['split'] != frame['split'] or seal['binding'] != dict(run_binding,
                inputs=seal['binding']['inputs'])
            or seal['binding']['inputs']['upstream_row_sha256'] != B.O._hash(frame)):
        raise ValueError('Common frame seal/upstream/bank binding changed')
    for name, value in seal['outputs'].items():
        if B.sha(folder / name) != value:
            raise ValueError('Common bank frame bytes changed: ' + name)
    bank = B.read(folder / 'BANK.json.gz'); feature = B.read(folder / 'FEATURES.json.gz')
    legacy = B.read(folder / 'LEGACY.json.gz')
    if (bank['image_binding']['sha256'] != frame['image_sha256'] or bank['image_hw'] != frame['raw_hw']
            or bank['native_xyz'] != frame['native_xyz'] or bank['cf_extents'] != frame['initial_pose']['cf_extents']
            or feature['partition'] != frame['partition'] or feature['split'] != frame['split']):
        raise ValueError('Common bank image/model/canvas/split identity changed')
    return bank, feature, legacy, seal


def source_bindings():
    paths = [Path(__file__), Path(S.__file__), Path(M.__file__), Path(F.__file__),
        Path(B.__file__), Path(B.O.__file__), Path(__file__).with_name('registry.py')]
    return {str(path): B.sha(path) for path in paths}


def run(a):
    started, cpu = time.monotonic(), time.process_time(); root = Path(a.run_dir)
    faulthandler.enable()
    manifest_path = root / 'PREDICTION_MANIFEST.json'
    if manifest_path.exists():
        raise FileExistsError('Completed blind prediction seal is immutable')
    prediction_path = root / 'PREDICTIONS.jsonl.gz'
    old_path = a.historical_run / 'PREDICTIONS.jsonl.gz'
    old_manifest_path = a.historical_run / 'PREDICTION_MANIFEST.json'
    allowed = [a.plan, a.upstream, a.bank_run, old_path, old_manifest_path,
        root / 'TRAIN_ONLY_NORMALIZER.json', root / 'TRAINING_MANIFEST.json']
    for name in LEARNED:
        allowed.extend((root / 'training' / name / 'best.pt', root / 'training' / name / 'RUN_RECEIPT.json'))
    for ordinal in range(238):
        bank_folder = a.bank_dir / 'frames' / 'real' / f'{ordinal:04d}'
        allowed.extend(bank_folder / name for name in ('BANK.json.gz', 'FEATURES.json.gz', 'LEGACY.json.gz', 'SEAL.json'))
        folder = root / 'prediction_frames' / 'real' / f'{ordinal:04d}'
        allowed.extend((folder / 'FRAME_PREDICTIONS.json.gz', folder / 'FRAME_SEAL.json'))
    # SciPy Qhull's native MessageStream calls Python tempfile.mkstemp for
    # error capture. A denied default /tmp write can crash its native cleanup
    # before Python emits the original PermissionError. Keep that transient
    # implementation file inside this run's audited output directory.
    runtime_temporary = root / 'prediction_runtime_tmp'
    runtime_temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(runtime_temporary.resolve())
    class RuntimeGuard(PredictionGuard):
        def _deny(self, event, path, reason):
            print(json.dumps(dict(event='PREDICTION_IO_BLOCKED', operation=event,
                path=str(path), reason=reason)), file=sys.stderr, flush=True)
            return super()._deny(event, path, reason)
    guard = RuntimeGuard(allowed_files=allowed,
        allowed_read_roots=(Path(sys.prefix), Path('/usr/lib'), Path(__file__).resolve().parents[2], runtime_temporary),
        output_root=root, denied_files=(a.reference_dir / 'FIXED_REFERENCES.json',),
        denied_roots=(a.reference_dir, root / 'evaluation', root / 'evaluation_v1',
            root / 'diagnostics', root / 'diagnostics_v1'))
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = []; frame_receipts = []; proof_rows = []; model_bindings = []
    with guard:
        # Exercise only native diagnostic-stream construction, not a pose fit,
        # before reading model/input records or visiting any real frame.
        S.V3.registered_incident_normals(S.RT.C.cuboid(1., .2, .8))
        plan = B.read(a.plan)
        if B.sha(a.plan) != a.plan_sha256 or not plan['status'].startswith('FROZEN'):
            raise ValueError('Frozen pre-fit scientific plan hash required')
        training = B.read(root / 'TRAINING_MANIFEST.json')
        if training['completed_fits'] != 6 or len(training['runs']) != 6:
            raise ValueError('All six scientific fits must be completed before this all-arm seal')
        normalizer = B.read(root / 'TRAIN_ONLY_NORMALIZER.json')
        models, model_bindings = _load_models(root, a.plan_sha256, normalizer)
        bank_run = B.read(a.bank_run)
        if (bank_run['domain'] != 'real' or bank_run['frames'] != 238 or bank_run.get('limit') is not None
                or bank_run.get('partition') is not None or bank_run['reference_reads'] != 0):
            raise ValueError('Complete reference-free all238 common real bank required')
        binding_sources = bank_run['binding']['source_sha256']
        if any(binding_sources[name] != B.sha(Path(B.__file__).with_name(name))
                for name in ('observer.py', 'features.py', 'bank.py')):
            raise ValueError('Common bank scientific/execution source changed')
        original_frames = B.rows(a.upstream)
        frames = [{key: frame[key] for key in FRAME_INPUT_FIELDS} for frame in original_frames]
        ids = [f['id'] for f in frames]
        if (len(frames) != 238 or len(set(ids)) != 238 or
                Counter(f['partition'] for f in frames) != Counter(TEST=49, REMAINDER=143, CAL=46)):
            raise ValueError('Exact original real238 partition denominator required')
        old_manifest = B.read(old_manifest_path)
        if old_manifest['artifacts']['PREDICTIONS.jsonl.gz'] != B.sha(old_path):
            raise ValueError('Historical complete-system outputs changed')
        historical = {row['id']: {**{k: row[k] for k in IDENTITY},
            'predictions': {new: historical_entry(row['predictions'][old], old)
                for new, old in HISTORICAL_SOURCE.items()}} for row in B.rows(old_path)}
        if len(historical) != 238 or set(historical) != set(ids):
            raise ValueError('Historical outputs must cover the exact same real238')
        input_bindings = [dict(path=str(p), sha256=B.sha(p)) for p in allowed if p.exists()
            and 'prediction_frames' not in p.parts]
        run_binding = dict(experiment_plan_sha256=a.plan_sha256, upstream_sha256=B.sha(a.upstream),
            bank_run_sha256=B.sha(a.bank_run), historical_predictions_sha256=B.sha(old_path),
            normalizer_sha256=B.sha(root / 'TRAIN_ONLY_NORMALIZER.json'),
            checkpoints=model_bindings, source_sha256=source_bindings(), backend=S.contract())
        binding_hash = digest(run_binding)
        for ordinal, (frame, original_frame) in enumerate(zip(frames, original_frames)):
            folder = root / 'prediction_frames' / 'real' / f'{ordinal:04d}'
            path = folder / 'FRAME_PREDICTIONS.json.gz'; seal_path = folder / 'FRAME_SEAL.json'
            bank_folder = a.bank_dir / 'frames' / 'real' / f'{ordinal:04d}'
            bank, feature, legacy, bank_seal = _bind_frame(bank_folder, original_frame, bank_run['binding'])
            expected_seal = dict(run_binding_sha256=binding_hash, bank_frame_seal_sha256=B.sha(bank_folder / 'SEAL.json'), id=frame['id'])
            if seal_path.exists():
                seal = B.read(seal_path)
                if seal['binding'] != expected_seal or seal['predictions_sha256'] != B.sha(path):
                    raise ValueError('Existing partial prediction cannot be reused under changed inputs/source')
                detail = B.read(path); ledger = detail['ledger']; proof = detail['cache_proofs']; reused_frame = True
            else:
                before = time.monotonic(); current, details, proof = execute_matched(frame, bank, feature, legacy, models, normalizer)
                prior = historical[frame['id']]
                if any(prior[key] != frame[key] for key in IDENTITY):
                    raise ValueError('Historical identity metadata changed')
                if prior['predictions']['N0_HISTORY']['actual_pose'] != frame['initial_pose']:
                    raise ValueError('Historical N0 must be the exact saved initialization')
                predictions = dict(prior['predictions'], **current)
                if set(predictions) != set(ARMS):
                    raise ValueError('The fixed thirteen-arm inventory changed')
                ledger = dict(**{key: frame[key] for key in IDENTITY},
                    object_scale_m=float(np.linalg.norm(np.ptp(np.asarray(frame['native_xyz']), axis=0))),
                    predictions={name: predictions[name] for name in ARMS},
                    reference_inputs=False, observation_record_file=str(path))
                detail = dict(schema='BLIND_LOCAL_OBSERVATION_FRAME_PREDICTIONS_V1', ledger=ledger,
                    current_observation_and_solver_records=details, cache_proofs=proof,
                    input_frame_sha256=digest(original_frame), bank_frame_seal_sha256=B.sha(bank_folder / 'SEAL.json'),
                    before_scoring=True, current_reference_inputs=False)
                B.write(path, detail)
                seal = dict(schema='BLIND_OBSERVATION_FRAME_PREDICTION_SEAL_V1', complete=True,
                    binding=expected_seal, predictions_sha256=B.sha(path), elapsed_seconds=time.monotonic()-before,
                    actual_new_pose_optimizer_calls=sum(p['actual_new_pose_optimizer_calls'] for p in proof),
                    unique_observation_packets=sum(not p['cache_reused'] for p in proof),
                    exact_cache_reuses=sum(p['cache_reused'] for p in proof), reference_reads=0)
                B.write(seal_path, seal); reused_frame = False
            output.append(ledger); frame_receipts.append(dict(path=str(seal_path), sha256=B.sha(seal_path), reused_frame=reused_frame,
                actual_new_pose_optimizer_calls=seal['actual_new_pose_optimizer_calls'], unique_observation_packets=seal['unique_observation_packets'],
                exact_cache_reuses=seal['exact_cache_reuses'], elapsed_original_seconds=seal['elapsed_seconds']))
            proof_rows.extend(dict(id=frame['id'], **p) for p in proof)
            if ordinal % 16 == 0 or ordinal == 237:
                print(json.dumps(dict(event='BLIND_REAL_POSE_PROGRESS', frames=ordinal+1, total=238,
                    optimizer_calls=sum(r['actual_new_pose_optimizer_calls'] for r in frame_receipts),
                    elapsed_seconds=time.monotonic()-started)), flush=True)
        save_rows(prediction_path, output); save_rows(root / 'EXACT_OBSERVATION_CACHE_PROOFS.jsonl.gz', proof_rows)
        B.write(root / 'PREDICTION_FRAME_MANIFEST.json', dict(binding=run_binding, frame_receipts=frame_receipts))
        all_bindings = input_bindings + [dict(path=path, sha256=value) for path, value in source_bindings().items()]
    B.write(root / 'PREDICTION_GUARD.json', guard.receipt())
    artifacts = {p.name: B.sha(p) for p in (prediction_path, root / 'EXACT_OBSERVATION_CACHE_PROOFS.jsonl.gz',
        root / 'PREDICTION_FRAME_MANIFEST.json', root / 'PREDICTION_GUARD.json')}
    manifest = dict(schema='BI_CONDITIONED_LOCAL_OBSERVATION_ALL_OUTPUT_SEAL_V1',
        status='ALL_238_X_13_OUTPUTS_SEALED_BEFORE_CURRENT_REAL_REFERENCE_OPEN', complete=True,
        before_scoring=True, real_reference_inputs=False, experiment_plan_sha256=a.plan_sha256,
        rows=238, arms=list(ARMS), ordered_ids=ids, artifacts=artifacts,
        model_and_input_bindings=all_bindings, checkpoint_bindings=model_bindings,
        source_sha256=source_bindings(), backend=S.contract(),
        historical_complete_system_copies=list(HISTORICAL_SOURCE), matched_fresh_observation_groups=list(MATCHED),
        actual_new_pose_optimizer_calls=sum(r['actual_new_pose_optimizer_calls'] for r in frame_receipts),
        unique_observation_packets=sum(r['unique_observation_packets'] for r in frame_receipts),
        exact_observation_cache_reuses=sum(r['exact_cache_reuses'] for r in frame_receipts),
        cached_numeric_model_choice_cost={name: dict(rows=238,
            wall_seconds=sum(r['predictions'][name]['model_choice_wall_seconds'] for r in output),
            CPU_seconds=sum(r['predictions'][name]['model_choice_CPU_seconds'] for r in output)) for name in LEARNED},
        fresh_solver_cost=dict(wall_seconds=sum(p['fresh_solver_wall_seconds'] for p in proof_rows),
            CPU_seconds=sum(p['fresh_solver_CPU_seconds'] for p in proof_rows)),
        pose_status_counts={name: dict(Counter(r['predictions'][name]['status'] for r in output)) for name in ARMS},
        new_model_fits=0, old_pose_bank_reranking=False, GT_inputs=False, seal_epoch=time.time(),
        wall_seconds=time.monotonic()-started, CPU_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timing_scope='cached common numeric features + local head + actual fresh pose solving; original RGB/state/mask extraction excluded')
    B.write(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'rows', 'actual_new_pose_optimizer_calls', 'exact_observation_cache_reuses', 'wall_seconds')}), flush=True)
    return manifest


def parser():
    p = argparse.ArgumentParser()
    for name in ('run-dir', 'upstream', 'bank-dir', 'bank-run', 'historical-run', 'plan', 'reference-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
