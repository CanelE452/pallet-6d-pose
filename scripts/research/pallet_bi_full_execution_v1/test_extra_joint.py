"""Analytic five-arm comparison against the unmodified single-arm adapter."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import cv2
import numpy as np

from . import extra_joint as J
from . import test_fullnormal_backend as F
from scripts.research.pallet_bi_conditioned_observation_v1.test_solver import fixture
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import boundary_model as PROFILES


def constructed():
    frame, selection, _, _, _ = fixture()
    frame.update(partition='SYNTH', split='train', session='analytic', cluster_id='analytic', image_sha256='0'*64)
    geometry = F.B.LOCAL.RT.registered_geometry(frame['raw_hw'], frame['K'], frame['native_xyz'], frame['initial_pose'])
    coarse = dict(R=frame['initial_pose']['R_cf'], t=frame['initial_pose']['centroid'], physical_registration_Q=geometry['physical_registration_Q'].tolist())
    families = {f['name']: f for f in F.RUN.WD.make_wd_families(frame['native_xyz'], coarse)}
    mask = np.zeros(frame['raw_hw'], np.float32)
    points = F.B.V.project_native(np.asarray(frame['native_xyz']), np.asarray(coarse['R']), np.asarray(coarse['t']), np.asarray(frame['K']))[0]
    cv2.fillConvexPoly(mask, cv2.convexHull(points.astype(np.int32)), 1.)
    outline = F.B.G10.observe_global_outline(mask, frame['raw_hw'])
    spec = dict(xyz=frame['native_xyz'], K=frame['K'], states=selection['states'], visible=selection['full_points'],
        domains=[], image_hw=frame['raw_hw'], coarse_pose=coarse, config=F.B.configuration(dict(prior_weight=0.)),
        boundary_observations=[], u_regions=[], outline_observation=outline)
    packet = dict(id=frame['id'], solver_input=spec, activation=dict(complete_ids=[r['native_corner'] for r in selection['full_points']], all_partial_ids=[]),
        original_e_stage1_family_seed_bank={name: dict(family=f) for name, f in families.items()}, original_e_legacy_boundary_observations=[])
    erun = dict(id=frame['id'], packet=packet, families=families, all_visited_endpoints=[dict(constructed=True)])
    image = np.zeros((*frame['raw_hw'], 3), np.uint8); image[:] = [12, 25, 210]; image[mask > 0] = [210, 25, 12]
    bundle = PROFILES.build_boundary_bundle(image, mask, packet)
    config = dict(none_prior=.05)
    inp = dict(frame, packet=packet, families=families, fallback=frame['initial_pose'])
    models = J.S.compile_frame(inp, config, bundle=bundle)
    n0 = deepcopy(frame['initial_pose']); e0 = deepcopy(n0); e0['centroid'][0] += .03
    bank = [dict(candidate_id=name, source=source, R_physical=pose['R_physical'], centroid=pose['centroid'])
        for name, source, pose in [('N0', 'N0', n0), ('E0_REGENERATED', 'E0', e0)]]
    features = J.S.features_for_bank(inp, models, bank, [J.S.candidate_scores(c, models) for c in bank])
    scale = max(m.ctx.scale for m in models.values())
    for c, raw in zip(features, bank):
        c.update(object_scale_m=scale, source_aliases=[raw['candidate_id']], source_baseline_aliases=[raw['source']], eligible=True)
    base = dict(frame, candidates=features, object_scale_m=scale, fallback=n0, e0_regenerated=e0)
    return frame, erun, base, bundle, config


def run_constructed():
    frame, erun, base, bundle, config = constructed()
    def pose(dx, phase=False):
        p = deepcopy(frame['initial_pose']); p['centroid'][0] += dx
        if phase: p['R_physical'] = (np.asarray(p['R_physical']) @ np.diag([-1., 1., -1.])).tolist()
        return p
    def row(arm, index, p, available=True):
        return dict(id=frame['id'], candidate_id=arm+'/'+str(index), method=arm, producer_available=available, actual_pose=p)
    extra = dict(A=[row('A', 0, pose(.04))], B=[row('B', 0, pose(.04)), row('B', 1, pose(.10))],
        C=[row('C', 0, pose(.04, True))], D=[row('D', 0, pose(.04+1e-12))],
        E=[row('E', 0, pose(0.)), row('E', 1, pose(.2), available=False)])
    before = J.U.payload_sha(dict(frame=frame, erun=erun, base=base, bundle=bundle, extra=extra))
    with mock.patch.object(J.S, 'compile_frame', wraps=J.S.compile_frame) as compile_calls:
        features, costs, receipt = J.union_frames(frame, erun, base, bundle, extra, config)
    assert compile_calls.call_count == 1
    assert receipt['actual_calls']['exact_individual_cost_cache_hits'] >= 1
    # Shared .04 and distinct .04+1e-12 must occupy different exact entries.
    expected_new = 4
    expected_old = len(base['candidates']) if any(v is None for v in J.U.recover_minima(base['candidates'])[0].values()) else 0
    assert receipt['exact_pose_cache_entries'] == expected_new+expected_old, receipt
    errors = {}; input_errors = {}
    for arm, rows in extra.items():
        single, single_costs = J.SINGLE.union_frame(frame, erun, base, bundle, rows, config)
        a, b = features[arm]['candidates'], single['candidates']
        assert [c['candidate_id'] for c in a] == [c['candidate_id'] for c in b]
        assert [c['actual_pose'] for c in a] == [c['actual_pose'] for c in b]
        assert [c['eligible'] for c in a] == [c['eligible'] for c in b]
        assert [c['source_aliases'] for c in a] == [c['source_aliases'] for c in b]
        errors[arm] = float(np.max(np.abs(np.asarray([c['features'] for c in a])-np.asarray([c['features'] for c in b]))))
        assert errors[arm] == 0.
        normalizer = J.D.fit_normalizer([features[arm]])
        x, actions = J.D.inputs(features[arm], normalizer, 'H1'); y, single_actions = J.D.inputs(single, normalizer, 'H1')
        assert [c['candidate_id'] for c in actions] == [c['candidate_id'] for c in single_actions]
        input_errors[arm] = float(np.max(np.abs(x-y))); assert input_errors[arm] == 0.
        assert len(costs[arm]['new_complete_action_scores']) == len(single_costs['new_complete_action_scores'])
    assert before == J.U.payload_sha(dict(frame=frame, erun=erun, base=base, bundle=bundle, extra=extra))
    # A changed model context gets a fresh cache even for an identical pose.
    fresh = J.FrameCosts(frame, erun, bundle, dict(none_prior=.07))
    original = J.FrameCosts(frame, erun, bundle, config)
    candidate = dict(candidate_id='probe', R_physical=pose(.04)['R_physical'], centroid=pose(.04)['centroid'])
    original.score(candidate); fresh.score(candidate)
    assert original.context_sha256 != fresh.context_sha256
    assert fresh.counts['actual_unique_individual_cost_evaluations'] == 1
    with TemporaryDirectory(prefix='extra-joint-constructed-') as temporary:
        root = Path(temporary)
        J.U.save(root/'INPUTS.jsonl.gz', [frame]); J.U.save(root/'base'/'HEAD_FEATURES.jsonl.gz', [base])
        J.U.write(root/'base'/'FEATURE_SEAL.json', dict(complete=True, head_features_sha256=J.U.sha(root/'base'/'HEAD_FEATURES.jsonl.gz')))
        J.U.save(root/'core'/'frames'/'0000'/'E_RUN.jsonl.gz', [erun]); J.U.save(root/'core'/'frames'/'0000'/'BOUNDARY_BUNDLE.jsonl.gz', [bundle])
        paths = {}; outputs = {}
        producer_seals = {}
        for arm, rows in extra.items():
            paths[arm] = root/f'{arm}_EXTRAS.jsonl.gz'; outputs[arm] = root/f'{arm}_features'
            J.U.save(paths[arm], rows)
            producer_seals[arm] = root/f'{arm}_PRODUCER_SEAL.json'
            J.U.write(producer_seals[arm], dict(complete=True, predictions_sha256=J.U.sha(paths[arm])))
        J.U.write(root/'CONFIG.json', config); J.U.write(root/'CAMPAIGN_PLAN.json', dict(constructed_only=True))
        args = (root/'INPUTS.jsonl.gz', root/'base'/'HEAD_FEATURES.jsonl.gz', root/'core', paths,
            root/'CONFIG.json', outputs, root, root/'joint')
        seals = J.batch_features(*args, root_GO=True, extra_seals_by_arm=producer_seals)
        for arm, output in outputs.items():
            J.D.bound(output/'HEAD_FEATURES.jsonl.gz', output/'FEATURE_SEAL.json')
        assert seals == J.batch_features(*args, root_GO=True, extra_seals_by_arm=producer_seals)
        # The full producer receipt is an external boundary: require both its
        # exact output byte hash and its frozen resource/method-lock binding.
        method_lock = root/'METHOD_LOCK_RESOURCE_LP4.json'
        J.U.write(method_lock, dict(constructed_guard_fixture=True))
        producer_receipt = root/'FULL_RECEIPT.json'
        full = dict(complete=True, rows=1241, all1241_completed=True,
            output_artifacts={paths['A'].name: J.U.sha(paths['A'])}, method_lock_sha256=J.U.sha(method_lock))
        J.U.write(producer_receipt, full)
        J.SOURCES.verify(paths['A'], producer_receipt)
        method_lock.write_text('{"changed":true}\n')
        try: J.SOURCES.verify(paths['A'], producer_receipt)
        except ValueError: pass
        else: raise AssertionError('Changed extra producer method lock accepted')
        bad_full = dict(full, all1241_completed=False)
        producer_receipt.write_text(J.U.canonical(bad_full))
        try: J.SOURCES.verify(paths['A'], producer_receipt)
        except ValueError: pass
        else: raise AssertionError('Incomplete extra producer accepted')
        archive = root/'joint'/'frames'/'0000'/'ORIGINAL_E_RUN_EXACT.jsonl.gz'
        archive.write_bytes(archive.read_bytes()+b'corruption')
        try: J.batch_features(*args, root_GO=True, extra_seals_by_arm=producer_seals)
        except ValueError: pass
        else: raise AssertionError('Corrupt shared original endpoint archive accepted')
    return dict(complete=True, fixture='analytic constructed cuboid only', scientific_dataset_rows=0,
        actual_pose_solves=0, H1_fits=0, real_reference_reads=0, arms=5,
        exact_features_max_error_vs_unmodified_single_arm=errors,
        exact_188D_inputs_max_error_vs_unmodified_single_arm=input_errors,
        shared_cost_receipt=receipt, changed_context_has_fresh_cache=True,
        sub_tolerance_different_exact_pose_not_memoized=True, C2_phase_preserved=True,
        batch_heads_feature_seals_and_idempotent_resume_pass=True, corrupted_original_archive_rejected=True)


def test_joint_against_five_independent_original_adapters():
    run_constructed()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path); args = parser.parse_args()
    cv2.setNumThreads(1); receipt = run_constructed()
    if args.output: J.U.write(args.output, receipt)
    print(J.U.canonical(receipt), flush=True)
