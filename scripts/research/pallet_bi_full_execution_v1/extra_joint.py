"""Joint original-packet S3/S4 features with exact per-frame score memoization.

Each arm retains its own admitted bank, aliases, minima and five score gaps.
Only pure original candidate_scores calls for an identical physical R/t and
identical packet/configuration/bundle are reused. No rounded or symmetry key,
cross-frame cache, solver, training, image read or reference read is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from . import s1_union as U
from . import s1_extra as SINGLE
from . import extra_sources as SOURCES
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D


class FrameCosts:
    def __init__(self, frame, erun, bundle, config):
        self.inp = dict(frame, packet=erun['packet'], families=erun['families'], fallback=frame['initial_pose'])
        self.bundle = bundle; self.config = config; self.models = None; self.cache = {}
        self.counts = Counter()
        self.context_sha256 = U.payload_sha(dict(input=self.inp, bundle=bundle, configuration=config))

    def compile(self):
        if self.models is None:
            self.models = S.compile_frame(self.inp, self.config, bundle=self.bundle)
            self.counts['actual_model_compilations'] += 1
        return self.models

    def score(self, candidate):
        # Endianness/shape are fixed; all 12 doubles, including signed zero,
        # enter the key. There is no numerical tolerance or rounded hashing.
        R = np.asarray(candidate['R_physical'], dtype='<f8')
        t = np.asarray(candidate['centroid'], dtype='<f8')
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError('Exact cache only accepts finite complete poses')
        key = (R.tobytes(order='C'), t.tobytes(order='C'))
        self.counts['individual_cost_requests'] += 1
        if key not in self.cache:
            self.cache[key] = S.candidate_scores(candidate, self.compile())
            self.counts['actual_unique_individual_cost_evaluations'] += 1
        else:
            self.counts['exact_individual_cost_cache_hits'] += 1
        return deepcopy(self.cache[key])


def _raw(candidate):
    return dict(candidate_id=candidate['candidate_id'], source='COMPLETED_ACTION',
        R_physical=candidate['actual_pose']['R_physical'], centroid=candidate['actual_pose']['centroid'])


def _union(frame, erun, base, extra_rows, shared):
    scale = base['object_scale_m']; old = deepcopy(base['candidates'])
    minima, spread = U.recover_minima(old)
    additions = []; rejected = []; duplicates = []; seen = set(); methods = set()
    old_ids = {c['candidate_id'] for c in old}
    for row in extra_rows:
        if row['id'] != frame['id']:
            raise ValueError('Extra endpoint/frame identity differs')
        cid, method = row['candidate_id'], row['method']
        if cid in seen or cid in old_ids or cid in ('N0', 'E0', 'E0_REGENERATED'):
            raise ValueError('Unique non-baseline extra IDs required')
        seen.add(cid); methods.add(method)
        if row.get('image_sha256') not in (None, frame['image_sha256']) or ('K' in row and row['K'] != frame['K']):
            raise ValueError('Extra RGB or camera binding differs')
        pose = row['actual_pose']
        if not row.get('producer_available', pose.get('available', False)):
            valid, reason = False, 'PRODUCER_UNAVAILABLE'
        else:
            valid, reason = U.pose_valid(pose, erun['families']['ORIGINAL_WD'])
            if reason == 'ORIGINAL_F_UNAVAILABLE': reason = 'PRODUCER_UNAVAILABLE'
        if not valid:
            rejected.append(dict(candidate_id=cid, method=method, reason=reason)); continue
        match = next((c for c in old + additions if D.same_pose(c['actual_pose'], pose, scale)), None)
        if match is not None:
            match['source_aliases'] = sorted(set(match.get('source_aliases', [])) | {cid, method})
            match['complete_action_aliases'] = sorted(set(match.get('complete_action_aliases', [])) | {cid})
            match['complete_action_stage_aliases'] = sorted(set(match.get('complete_action_stage_aliases', [])) | {method})
            match['eligible'] = True
            duplicates.append(dict(extra_candidate_id=cid, method=method, candidate_id=match['candidate_id'])); continue
        additions.append(dict(candidate_id=cid, actual_pose=deepcopy(pose), source_aliases=sorted({cid, method}),
            source_baseline_aliases=[], complete_action_aliases=[cid], complete_action_stage_aliases=[method],
            eligible=True, object_scale_m=scale))
    old_requests = 0; new_scores = []; new_bank = []; new_min = {m: None for m in U.MODES}
    if additions:
        models = shared.compile()
        if any(value is None for value in minima.values()):
            old_scores = [shared.score(_raw(c)) for c in old]; old_requests = len(old)
            for mode in U.MODES:
                if minima[mode] is None:
                    values = [score[f]['scores'][mode] for score in old_scores for f in models
                        if S._finite_feature(score[f]['scores'][mode]) is not None]
                    minima[mode] = min(values) if values else None
        new_bank = [_raw(c) for c in additions]
        new_scores = [shared.score(c) for c in new_bank]
        for mode in U.MODES:
            values = [score[f]['scores'][mode] for score in new_scores for f in models
                if S._finite_feature(score[f]['scores'][mode]) is not None]
            new_min[mode] = min(values) if values else None
        new_features = S.features_for_bank(shared.inp, models, new_bank, new_scores)
        for source, candidate, score in zip(additions, new_features, new_scores):
            candidate.update(deepcopy(source)); candidate['admission_rule'] = 'COMMON_COMPLETE_ACTION_VALID'
            family = candidate['feature_context_family']
            for j, mode in enumerate(U.MODES):
                values = [v for v in (minima[mode], new_min[mode]) if v is not None]
                cost = score[family]['scores'][mode]
                valid = bool(values) and S._finite_feature(cost) is not None
                candidate['features'][26+j] = U.encode(cost-min(values)) if valid else 0.
                candidate['features'][57+j] = float(valid)
        additions = new_features
    result = deepcopy(base); result['candidates'] = U.update_gaps(old, minima, new_min) + additions
    result['extra_union'] = dict(original_bank_exact_pose_reuse=True, old_candidate_count=len(old),
        new_unique_complete_actions=len(additions), complete_action_duplicates=duplicates,
        new_complete_action_rejections=rejected, old_minima=minima, new_complete_action_minima=new_min,
        old_minima_inverse_transform_spread=spread, old_cost_reconstructions_for_missing_context=old_requests,
        numerical_feature_atol=U.FEATURE_ATOL, extra_visited_endpoints=len(extra_rows),
        complete_action_methods=sorted(methods), new_support_packets_used_as_H1_features=False,
        old_individual_feature_scalars_bitwise_reused=26, bank_relative_gap_features_recomputed=5,
        normalization_and_188D_eligible_bank_summaries='Original action_data.inputs with each arm own train normalizer',
        base='sealed S1 bank, identical original downstream packet and original boundary bundle',
        admission='producer available, finite proper R, original-native all-positive depth',
        exact_score_cache_context_sha256=shared.context_sha256)
    D.validate_frame(result)
    costs = dict(new_complete_action_scores=new_scores, new_complete_action_bank=new_bank, old_minima=minima,
        union_minima={m: min(v for v in (minima[m], new_min[m]) if v is not None)
            if any(v is not None for v in (minima[m], new_min[m])) else None for m in U.MODES})
    return result, U.finite(costs)


def union_frames(frame, erun, base_features, bundle, extras_by_arm, config):
    """Return ({arm: feature}, {arm: costs}, shared_cost_receipt) for one frame."""
    shared = FrameCosts(frame, erun, bundle, config); features = {}; costs = {}
    for arm, extra in extras_by_arm.items():
        before = Counter(shared.counts)
        features[arm], costs[arm] = _union(frame, erun, base_features, extra, shared)
        features[arm]['extra_union']['actual_joint_cost_calls_for_this_arm'] = dict(shared.counts-before)
    receipt = dict(context_sha256=shared.context_sha256, actual_calls=dict(shared.counts),
        exact_pose_cache_entries=len(shared.cache), arms=list(extras_by_arm),
        shared_candidate_union_used_for_arm_gap_or_pooling=False, current_real_reference_reads=0)
    return features, costs, receipt


def source_binding():
    from scripts.research.pallet_boundary_action_repair_v1.d1_batch import source_bindings
    result = source_bindings()
    for path in (U.__file__, SINGLE.__file__, SOURCES.__file__, __file__): result[str(Path(path).resolve())] = U.sha(path)
    return result


def batch_features(inputs_path, base_features_path, base_frame_dir, extras_by_arm,
                   config_path, outputs_by_arm, run, joint_output, *, root_GO=False,
                   extra_seals_by_arm=None):
    """One ordered original-bank pass; each output is heads.prepare compatible.

    extras_by_arm and outputs_by_arm are matching {arm: path} maps. The base
    feature file must have its own valid FEATURE_SEAL.json. All five arm input
    hashes enter the shared lock and every row cache key.
    """
    if root_GO is not True: raise PermissionError('Explicit joint extra feature production GO required')
    U.check_plan(run); cv2.setNumThreads(1); started = time.monotonic(); cpu = time.process_time()
    if not extras_by_arm or set(extras_by_arm) != set(outputs_by_arm): raise ValueError('Exact arm input/output maps required')
    if extra_seals_by_arm is not None and set(extra_seals_by_arm) != set(extras_by_arm):
        raise ValueError('Exact arm/producer seal maps required')
    if len(set(map(str, outputs_by_arm.values()))) != len(outputs_by_arm): raise ValueError('Distinct arm output roots required')
    D.bound(base_features_path, Path(base_features_path).parent/'FEATURE_SEAL.json')
    inputs = list(U.rows(inputs_path)); base = list(U.rows(base_features_path)); ids = [f['id'] for f in inputs]
    if ids != [f['id'] for f in base] or len(set(ids)) != len(ids): raise ValueError('Full ordered input/base IDs required')
    extras = {}; producer_bindings = {}
    for arm, path in extras_by_arm.items():
        producer_bindings[arm] = SOURCES.verify(path, extra_seals_by_arm[arm] if extra_seals_by_arm is not None else None)
        extras[arm] = {fid: [] for fid in ids}
        for row in U.rows(path):
            if row['id'] not in extras[arm]: raise ValueError('Unexpected extra frame ID')
            extras[arm][row['id']].append(row)
        if any(not values for values in extras[arm].values()): raise ValueError('Every arm/frame needs an explicit endpoint row')
    config = U.read(config_path); joint_output = Path(joint_output); closure = source_binding()
    U.write(joint_output/'SOURCE_DEPENDENCY_CLOSURE.json', closure)
    lock = dict(inputs_sha256=U.sha(inputs_path), base_features_sha256=U.sha(base_features_path),
        base_feature_seal_sha256=U.sha(Path(base_features_path).parent/'FEATURE_SEAL.json'),
        extra_rows_sha256={arm: U.sha(path) for arm, path in extras_by_arm.items()},
        extra_producer_bindings=producer_bindings,
        outputs={arm: str(Path(path).resolve()) for arm, path in outputs_by_arm.items()},
        configuration_sha256=U.sha(config_path), source_dependency_closure_sha256=U.payload_sha(closure),
        plan_sha256=U.sha(Path(run)/'CAMPAIGN_PLAN.json'))
    U.write(joint_output/'JOINT_LOCK.json', lock)
    for output in outputs_by_arm.values():
        U.write(Path(output)/'SOURCE_DEPENDENCY_CLOSURE.json', closure)
    all_features = {arm: [] for arm in extras}; all_seals = {arm: [] for arm in extras}; receipts = []; calls = Counter()
    for ordinal, (frame, baseframe) in enumerate(zip(inputs, base)):
        source = Path(base_frame_dir)/'frames'/f'{ordinal:04d}'
        ep = source/'E_RUN.jsonl.gz'; bp = source/'BOUNDARY_BUNDLE.jsonl.gz'
        erun = next(U.rows(ep)); bundle = next(U.rows(bp))
        if erun['id'] != frame['id']: raise ValueError('Original packet frame ID differs')
        key = dict(input_payload_sha256=U.payload_sha(frame), base_payload_sha256=U.payload_sha(baseframe),
            original_E_RUN_sha256=U.sha(ep), original_bundle_sha256=U.sha(bp),
            extras_payload_sha256={arm: U.payload_sha(extras[arm][frame['id']]) for arm in extras},
            joint_lock_sha256=U.sha(joint_output/'JOINT_LOCK.json'))
        joint_folder = joint_output/'frames'/f'{ordinal:04d}'; done = joint_folder/'SEAL.json'
        if done.exists():
            row_receipt = U.read(done)
            if row_receipt['dependencies'] != key: raise ValueError('Completed joint row dependencies differ')
            if U.sha(joint_folder/'ORIGINAL_E_RUN_EXACT.jsonl.gz') != row_receipt['original_E_RUN_archive_sha256']:
                raise ValueError('Completed original endpoint archive differs')
            for arm, root in outputs_by_arm.items():
                out = Path(root)/'frames'/f'{ordinal:04d}'; seal = U.read(out/'SEAL.json')
                if U.sha(out/'SEAL.json') != row_receipt['arm_seals'][arm] or seal['dependencies'] != key: raise ValueError('Completed arm seal differs')
                for name, digest in seal['artifacts'].items():
                    if U.sha(out/name) != digest: raise ValueError('Completed arm artifact differs')
                all_features[arm].extend(U.rows(out/'FEATURES.jsonl.gz'))
                all_seals[arm].append(dict(id=frame['id'], path=str(out/'SEAL.json'), sha256=U.sha(out/'SEAL.json')))
            calls['completed_joint_row_cache_hits'] += 1
        else:
            tick = time.monotonic()
            features, costs, resource = union_frames(frame, erun, baseframe, bundle,
                {arm: extras[arm][frame['id']] for arm in extras}, config)
            U.save(joint_folder/'ORIGINAL_E_RUN_EXACT.jsonl.gz', [erun])
            arm_seals = {}
            for arm, root in outputs_by_arm.items():
                out = Path(root)/'frames'/f'{ordinal:04d}'
                U.save(out/'ALL_EXTRA_ENDPOINTS.jsonl.gz', extras[arm][frame['id']])
                U.save(out/'INDIVIDUAL_COSTS.jsonl.gz', [costs[arm]]); U.save(out/'FEATURES.jsonl.gz', [features[arm]])
                seal = dict(complete=True, id=frame['id'], arm=arm, dependencies=key,
                    original_endpoint_archive=str(joint_folder/'ORIGINAL_E_RUN_EXACT.jsonl.gz'),
                    original_E_RUN_payload_sha256=U.payload_sha(erun),
                    artifacts={name: U.sha(out/name) for name in ('ALL_EXTRA_ENDPOINTS.jsonl.gz', 'INDIVIDUAL_COSTS.jsonl.gz', 'FEATURES.jsonl.gz')})
                U.write(out/'SEAL.json', seal); arm_seals[arm] = U.sha(out/'SEAL.json')
                all_features[arm].append(features[arm]); all_seals[arm].append(dict(id=frame['id'], path=str(out/'SEAL.json'), sha256=arm_seals[arm]))
            row_receipt = dict(complete=True, id=frame['id'], dependencies=key, arm_seals=arm_seals,
                original_E_RUN_archive_sha256=U.sha(joint_folder/'ORIGINAL_E_RUN_EXACT.jsonl.gz'),
                resource=resource, wall_seconds=time.monotonic()-tick)
            U.write(done, row_receipt); calls.update(resource['actual_calls']); calls['new_joint_rows'] += 1
        receipts.append(dict(id=frame['id'], path=str(done), sha256=U.sha(done)))
        if ordinal % 100 == 0: print(U.canonical(dict(stage='JOINT_EXTRA_FEATURES', done=ordinal+1, total=len(inputs), wall_seconds=time.monotonic()-started)), flush=True)
    result = {}
    for arm, root in outputs_by_arm.items():
        root = Path(root); U.save(root/'HEAD_FEATURES.jsonl.gz', all_features[arm]); sp = root/'FEATURE_SEAL.json'
        stable = dict(schema='JOINT_ORIGINAL_PACKET_EXTRA_FEATURE_SEAL_V1', complete=True, arm=arm, rows=len(inputs), ordered_ids=ids,
            head_features_sha256=U.sha(root/'HEAD_FEATURES.jsonl.gz'), dependencies=lock, frame_seals=all_seals[arm],
            shared_joint_row_seals=receipts, all_original_and_extra_visited_endpoints_preserved=True,
            current_real_reference_reads=0, new_pose_optimizer_calls=0)
        U.write(sp, stable); result[arm] = stable
    summary = joint_output/'SUMMARY.json'
    if summary.exists():
        saved = U.read(summary)
        if saved['dependencies'] != lock or saved['joint_row_seals'] != receipts: raise ValueError('Completed joint summary differs')
    else:
        U.write(summary, dict(complete=True, dependencies=lock, rows=len(inputs), arms=list(extras),
            actual_calls=dict(calls), joint_row_seals=receipts, wall_seconds=time.monotonic()-started,
            CPU_seconds=time.process_time()-cpu, current_real_reference_reads=0, new_pose_optimizer_calls=0))
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ('inputs-path', 'base-features-path', 'base-frame-dir', 'arm-paths', 'config-path', 'run', 'joint-output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--root-go', action='store_true'); args = parser.parse_args()
    arms = U.read(args.arm_paths)
    result = batch_features(args.inputs_path, args.base_features_path, args.base_frame_dir,
        {arm: value['extras'] for arm, value in arms.items()}, args.config_path,
        {arm: value['output'] for arm, value in arms.items()}, args.run, args.joint_output, root_GO=args.root_go,
        extra_seals_by_arm={arm: value['seal'] for arm, value in arms.items()} if all('seal' in value for value in arms.values()) else None)
    print(U.canonical(dict(complete=True, arms=list(result), rows={arm: seal['rows'] for arm, seal in result.items()})), flush=True)


if __name__ == '__main__': main()
