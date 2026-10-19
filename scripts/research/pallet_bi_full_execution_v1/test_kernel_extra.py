"""Constructed own-packet union parity and sealed-boundary failure tests."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import cv2
import numpy as np

from . import kernel_extra as K
from . import test_fullnormal_backend as FIXTURE


def constructed():
    frame, selection, coarse, cfg = FIXTURE.setup()
    frame.update(partition='SYNTH', split='train', session='constructed', cluster_id='constructed', image_sha256='0'*64)
    observation = dict(states=selection['states'], _original_source=dict(ot_domains=dict(domains=[])),
        u_region_observation=dict(u_regions=[]))
    mask = np.zeros(frame['raw_hw'], np.float32)
    points = K.B.V.project_native(np.asarray(frame['native_xyz']), np.asarray(coarse['R']),
        np.asarray(coarse['t']), np.asarray(frame['K']))[0]
    cv2.fillConvexPoly(mask, cv2.convexHull(points.astype(np.int32)), 1.)
    result = K.CORE.solve_frame(frame, selection, observation, mask, cfg)
    config = dict(none_prior=.05)
    old = K.CORE.feature_frame(frame, result, config)
    return frame, result, old, config


def endpoint(frame, pose, cid, method='CONSTRUCTED', available=True):
    return dict(id=frame['id'], candidate_id=cid, method=method,
        producer_available=available, actual_pose=deepcopy(pose))


def _bound(frame, result, old, config, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    rp = folder/'RESULT.json.gz'; fp = folder/'FEATURE.json.gz'
    K.Q.write(rp, result); K.Q.write(fp, old)
    sources = K.CORE.source_binding()
    lock = dict(feature_configuration=config, source_binding=sources, new_matched_full_normal_backend=True)
    seal = dict(complete=True, id=frame['id'], source_binding=sources,
        artifacts={p.name: K.U.sha(p) for p in (rp, fp)})
    return K.bound_core_feature(frame, result, old, config, lock, seal, rp, fp), lock, seal, rp, fp


def _fresh_union(frame, result, union, config):
    models = K.B.compile_models(result['packet'], config)
    bank = [K._raw(c) for c in union['candidates']]
    scores = [K.S.candidate_scores(c, models) for c in bank]
    inp = dict(frame, packet=result['packet'], families=result['families'], fallback=frame['initial_pose'])
    rebuilt = K.S.features_for_bank(inp, models, bank, scores)
    fresh = deepcopy(union)
    for old, candidate in zip(fresh['candidates'], rebuilt):
        K._mask_absent(candidate)
        old['features'] = candidate['features']
        assert old['feature_context_family'] == candidate['feature_context_family']
    return fresh


def run_constructed():
    frame, result, old, config = constructed()
    before_result = K.U.payload_sha(result); before_old = K.U.payload_sha(old)
    base_pose = deepcopy(old['candidates'][0]['actual_pose'])
    shifted = deepcopy(base_pose); shifted['centroid'][0] += .12
    c2 = deepcopy(shifted); c2['R_physical'] = (np.asarray(c2['R_physical']) @ np.diag([-1., 1., -1.])).tolist()
    improper = deepcopy(base_pose); improper['R_physical'][0][0] += .03
    behind = deepcopy(base_pose); behind['centroid'][2] = -3.
    nonfinite = deepcopy(base_pose); nonfinite['centroid'][0] = float('nan')
    malformed = dict(available=True, R_physical=[1, 2, 3], centroid=[0, 0, 3])
    extras = [endpoint(frame, base_pose, 'duplicate'), endpoint(frame, shifted, 'shifted'),
        endpoint(frame, shifted, 'duplicate_shifted', 'OTHER_METHOD'), endpoint(frame, c2, 'distinct_c2'),
        endpoint(frame, base_pose, 'visited_rejected', available=False), endpoint(frame, improper, 'improper'),
        endpoint(frame, behind, 'behind'), endpoint(frame, nonfinite, 'nonfinite'), endpoint(frame, malformed, 'malformed')]
    with TemporaryDirectory(prefix='kernel-extra-constructed-') as temporary:
        bound, lock, seal, rp, fp = _bound(frame, result, old, config, temporary)
        with mock.patch.object(K.S, 'candidate_scores', wraps=K.S.candidate_scores) as calls:
            union, costs = K._union(frame, result, bound, extras, config)
        assert calls.call_count == 2, calls.call_count
        info = union['kernel_extra']
        assert info['new_unique_complete_actions'] == 2
        assert len(info['complete_action_duplicates']) == 2
        assert len(info['complete_action_rejections']) == 5
        assert info['old_individual_cost_evaluations'] == 0
        assert info['old_cost_reconstructions_for_missing_context'] == 0
        assert all(i in {c['candidate_id'] for c in union['candidates']} for i in ('shifted', 'distinct_c2'))
        assert K.U.payload_sha(result) == before_result and K.U.payload_sha(old) == before_old
        for previous, current in zip(old['candidates'], union['candidates']):
            assert previous['actual_pose'] == current['actual_pose']
            assert previous['features'][:26] == current['features'][:26]
            assert previous['features'][31:] == current['features'][31:]
            assert set(previous['source_aliases']) <= set(current['source_aliases'])
            assert previous['source_baseline_aliases'] == current['source_baseline_aliases']
            assert previous['eligible'] == current['eligible'] or current['candidate_id'] == old['candidates'][0]['candidate_id']
        for candidate in union['candidates']:
            for index in K.ABSENT_INDICES:
                assert candidate['features'][index] == candidate['features'][index + 31] == 0.
        for score in costs['new_complete_action_scores']:
            for family in score.values():
                assert family['scores']['B'] is None and family['scores']['edge'] is None
                assert family['scores']['A'] == family['scores']['AB']
        fresh = _fresh_union(frame, result, union, config)
        errors = [np.max(np.abs(np.asarray(a['features']) - b['features']))
            for a, b in zip(union['candidates'], fresh['candidates'])]
        assert max(errors) < K.U.FEATURE_ATOL, max(errors)
        normalizer = K.D.fit_normalizer([union]); fresh_normalizer = K.D.fit_normalizer([fresh])
        inputs, actions = K.D.inputs(union, normalizer, 'H1')
        fresh_inputs, fresh_actions = K.D.inputs(fresh, fresh_normalizer, 'H1')
        assert inputs.shape[1] == 188
        assert [c['candidate_id'] for c in actions] == [c['candidate_id'] for c in fresh_actions]
        np.testing.assert_allclose(inputs, fresh_inputs, atol=2e-6, rtol=0)
        assert np.array_equal(inputs[:, 62:124], np.repeat(inputs[:1, 62:124], len(inputs), axis=0))
        assert np.array_equal(inputs[:, 124:186], np.repeat(inputs[:1, 124:186], len(inputs), axis=0))
        # Deliberately lose one recoverable minimum. A fresh old-cost pass must
        # recover it, and the entire resulting feature bank must still agree.
        original_recover = K.U.recover_minima
        def absent_A(candidates):
            minima, spread = original_recover(candidates); minima['A'] = None
            return minima, spread
        with mock.patch.object(K.U, 'recover_minima', side_effect=absent_A):
            fallback = K.union_frame(frame, result, bound, extras, config)
        assert fallback['kernel_extra']['old_cost_reconstructions_for_missing_context'] == len(old['candidates'])
        np.testing.assert_allclose([c['features'] for c in fallback['candidates']],
            [c['features'] for c in fresh['candidates']], atol=K.U.FEATURE_ATOL, rtol=0)
        # Bare calls must count and audit the old costs instead of fabricating
        # a sealed-old-context receipt; a changed old scalar is rejected.
        bare = K.union_frame(frame, result, old, extras[:1], config)
        assert bare['kernel_extra']['old_individual_cost_evaluations'] == len(old['candidates'])
        bad = deepcopy(old); bad['candidates'][0]['features'][0] += .01
        try: K.union_frame(frame, result, bad, extras, config)
        except ValueError: pass
        else: raise AssertionError('Changed old scalar accepted')
        # Genuine sealed dependency corruptions, rather than mirrored arithmetic.
        for changed_lock in (dict(lock, feature_configuration=dict(none_prior=.2)), dict(lock, source_binding={})):
            try: K.bound_core_feature(frame, result, old, config, changed_lock, seal, rp, fp)
            except ValueError: pass
            else: raise AssertionError('Changed source/config binding accepted')
        changed_result = deepcopy(result); changed_result['packet']['provenance']['adversarial_change'] = True
        try: K.bound_core_feature(frame, changed_result, old, config, lock, seal, rp, fp)
        except ValueError: pass
        else: raise AssertionError('Changed decoded core accepted')
        # Parallel-driver locks bind the feature config through their exact
        # row cache payload rather than a direct producer-lock config field.
        spec = result['packet']['solver_input']
        selection = dict(states=spec['states'], full_points=spec['visible'], partial_lines=spec['boundary_observations'])
        observation = dict(_original_source=dict(ot_domains=dict(domains=spec['domains'])),
            u_region_observation=dict(u_regions=spec['u_regions']))
        context = K.CORE.cache_payload(frame, selection, observation, '0'*64, spec['config'], config)
        from . import fullnormal_parallel as PARALLEL
        parallel_lock = dict(schema='BI_FULL_NORMAL_PARALLEL_EXECUTION_V1', numerical_kernel_source_binding=lock['source_binding'], driver_sha256=K.U.sha(PARALLEL.__file__))
        parallel_seal = dict(seal, cache_payload_sha256=K.Q.P.digest(context))
        K.bound_core_feature(frame, result, old, config, parallel_lock, parallel_seal, rp, fp, context_payload=context)
        wrong_context = deepcopy(context); wrong_context['feature_configuration']['none_prior'] = .2
        try: K.bound_core_feature(frame, result, old, config, parallel_lock, parallel_seal, rp, fp, context_payload=wrong_context)
        except ValueError: pass
        else: raise AssertionError('Changed parallel feature context accepted')
        # A one-row constructed batch exercises the real immutable writers,
        # heads' feature seal contract and completed-run resume, without data.
        batch_root = Path(temporary)/'batch'; core_root = batch_root/'core'
        core_folder = core_root/'CONSTRUCTED'/'synthetic'/'0000'
        _, batch_lock, batch_seal, _, _ = _bound(frame, result, old, config, core_folder)
        K.U.write(core_root/'PRODUCER_LOCK.json', batch_lock)
        K.U.write(core_folder/'FRAME_SEAL.json', batch_seal)
        K.U.save(batch_root/'INPUTS.jsonl.gz', [frame])
        K.U.save(batch_root/'EXTRAS.jsonl.gz', extras[:5])
        K.U.write(batch_root/'PREDICTION_SEAL.json', dict(complete=True, predictions_sha256=K.U.sha(batch_root/'EXTRAS.jsonl.gz')))
        K.U.write(batch_root/'CONFIG.json', config); K.U.write(batch_root/'CAMPAIGN_PLAN.json', dict(constructed_only=True))
        args = (batch_root/'INPUTS.jsonl.gz', core_root, 'CONSTRUCTED', 'synthetic',
            batch_root/'EXTRAS.jsonl.gz', batch_root/'CONFIG.json', batch_root/'output', batch_root)
        batch_result = K.batch_features(*args, root_GO=True)
        assert K.D.bound(batch_root/'output'/'HEAD_FEATURES.jsonl.gz', batch_root/'output'/'FEATURE_SEAL.json') == batch_result['head_features_sha256']
        resumed = K.batch_features(*args, root_GO=True)
        assert batch_result == resumed
        archive = next(K.U.rows(batch_root/'output'/'frames'/'0000'/'ACTUAL_CORE_RESULT_EXACT.jsonl.gz'))
        assert archive == result
        try: K.batch_features(*args)
        except PermissionError: pass
        else: raise AssertionError('Unapproved dataset production accepted')
        rp.write_bytes(rp.read_bytes()+b'corruption')
        try: K.bound_core_feature(frame, result, old, config, lock, seal, rp, fp)
        except ValueError: pass
        else: raise AssertionError('Changed RESULT bytes accepted')
    return dict(complete=True, fixture='constructed analytic cuboid; no real/synthetic dataset rows',
        new_pose_solves_for_fixture_only=result['operations']['pose_optimizer_calls'],
        scientific_dataset_pose_calls=0, scientific_H1_fits=0, real_reference_reads=0,
        old_unique_candidates=len(old['candidates']), new_unique_candidates=2,
        rejected_extra_endpoints=5, duplicate_aliases=2,
        fresh_full_bank_feature_max_error=float(max(errors)), H1_input_dimension=inputs.shape[1],
        active_missing_minimum_fallback_checked=True, invalid_B_edge_do_not_trigger_fallback=True,
        strict_C2_phase_retained=True, sealed_source_config_bytes_corruption_rejected=True,
        constructed_batch_heads_seal_and_idempotent_resume_checked=True)


def test_constructed_union_against_full_fresh_features_and_binding_failures():
    run_constructed()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); K.cv2.setNumThreads(1)
    receipt = run_constructed()
    if args.output: K.U.write(args.output, receipt)
    print(K.U.canonical(receipt), flush=True)
