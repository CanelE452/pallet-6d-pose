"""Pure-math / temporary synthetic-ledger tests; never open research data."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import analysis as A


def yaw(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def pose(R=None, t=(0., 0., 1.)):
    return dict(available=True, R_physical=(np.eye(3) if R is None else R).tolist(), centroid=list(t))


def reference():
    return dict(R=np.eye(3).tolist(), t=[0., 0., 1.], xyz=[1., .15, 1.2], order=2)


def entry():
    return dict(actual_pose=pose(), accepted=False, fallback=True, status='FIXTURE', reason='test only')


def measured(t, r):
    return dict(pose_available=t is not None, translation_cm=t, rotation_deg=r,
                success_le5=False if t is None else t <= 5 and r <= 5,
                success_strict_lt5=False if t is None else t < 5 and r < 5,
                reference_rotation_available=True)


class AnalysisContracts(unittest.TestCase):
    def test_center_and_physical_C2_not_cuboid_CF_or_C4(self):
        p = pose(yaw(180), (.03, .04, 1.))
        p['R_cf'] = yaw(90).tolist()
        m = A.metric(p, reference())
        self.assertAlmostEqual(m['translation_cm'], 5.)
        self.assertLess(m['rotation_deg'], 1e-5)
        self.assertTrue(m['success_le5'])
        self.assertFalse(m['success_strict_lt5'])
        m90 = A.metric(pose(yaw(90)), reference())
        self.assertAlmostEqual(m90['rotation_deg'], 90.)
        self.assertFalse(m90['success_le5'])
        with self.assertRaises(ValueError):
            A.metric(pose(), dict(reference(), order=4))

    def test_missing_pose_is_binary_failure_and_continuous_NA(self):
        m = A.metric(dict(available=False), reference())
        self.assertFalse(m['success_le5'])
        self.assertIsNone(m['translation_cm'])
        self.assertIsNone(m['rotation_deg'])
        unresolved = A.metric(pose(), dict(reference(), R=None))
        self.assertIsNone(unresolved['success_le5'])
        self.assertEqual(unresolved['translation_cm'], 0.)

    def test_bad_rotation_and_acceptance_are_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            A.metric(pose(np.diag([-1., 1., 1.])), reference())
        with self.assertRaises(ValueError):
            A.normalize_entry(dict(entry(), accepted=True, fallback=True))
        with self.assertRaises(ValueError):
            A.normalize_entry(dict(entry(), actual_pose=dict(available=True)))

    def test_unknown_solver_flags_and_single_ambiguity_scope_stay_distinct(self):
        e = dict(entry(), accepted=True, fallback=False,
                 single_start_does_not_prove_global_uniqueness=True,
                 solution=dict(status='accepted', fallback_reason=None,
                               ambiguity=dict(detected=False)))
        s = A.normalize_entry(e)['prediction_status']
        self.assertIsNone(s['solver_rejected'])
        self.assertIsNone(s['technical_failure'])
        self.assertEqual(s['ambiguity_assessment_scope'], 'SINGLE_START_LOCAL_RANK_NO_GLOBAL_ALTERNATIVE_TEST')

    def test_original_E0_status_restores_known_flags_preserving_historical_NA(self):
        original = dict(entry(), accepted=None, fallback=None, status='unchanged_frozen_E')
        branch = dict(coarse_pose=deepcopy(original['actual_pose']),
                      baseline_accepted=True, baseline_fallback=False)
        binding = dict(path='PRIVATE/guarded/e_inputs/INPUTS.json', sha256='a'*64)
        restored = A.restore_e0_status(original, branch, binding)
        normalized = A.normalize_entry(restored)
        self.assertEqual(restored['actual_pose'], original['actual_pose'])
        self.assertIsNone(original['accepted'])
        self.assertIsNone(original['fallback'])
        self.assertTrue(normalized['prediction_status']['accepted'])
        self.assertFalse(normalized['prediction_status']['fallback'])
        self.assertTrue(normalized['prediction_status']['original_status_restored'])
        audit = normalized['restored_original_status']
        self.assertIsNone(audit['historical_common']['accepted'])
        self.assertEqual(audit['source'], binding)
        self.assertTrue(audit['exact_saved_pose_bridge'])
        wrong = deepcopy(branch)
        wrong['coarse_pose']['centroid'][0] += .001
        with self.assertRaises(ValueError):
            A.restore_e0_status(original, wrong, binding)
        with self.assertRaises(ValueError):
            A.restore_e0_status(original, dict(branch, baseline_fallback=True), binding)
        with self.assertRaises(ValueError):
            A.restore_e0_status(dict(original, accepted=False), branch, binding)

    def test_typed_single_audit_scope_and_E0_usage_are_explicit(self):
        single = dict(entry(), audit=dict(E0_start_only=True, start_policy_comparison_NOT_ISOLATED=True),
                      solution=dict(ambiguity=dict(detected=False)))
        status = A.normalize_entry(single)['prediction_status']
        self.assertEqual(status['ambiguity_assessment_scope'], 'SINGLE_START_LOCAL_RANK_NO_GLOBAL_ALTERNATIVE_TEST')
        self.assertTrue(status['uses_e_initialization'])
        self.assertEqual(status['start_policy_comparison_scope'], 'NOT_ISOLATED')
        multi = dict(entry(), audit=dict(exact_original_hierarchical_family_choice=True))
        self.assertFalse(A.normalize_entry(multi)['prediction_status']['uses_e_initialization'])
        self.assertIsNone(A.normalize_entry(entry())['prediction_status']['uses_e_initialization'])
        with self.assertRaises(ValueError):
            A.normalize_entry(dict(single, uses_e_initialization=False))

    def test_fixed_candidate_zero_and_exact_centroid_copy(self):
        nominal = np.eye(4)
        nominal[:3, 3] = [1, 2, 3]
        second = nominal.copy()
        second[:3, :3] = np.diag([-1., 1., -1.])
        obj = dict(camera_facing_pnp=dict(pose_transform=nominal.tolist()),
                   physical_dimensions_m=dict(x=1, y=.15, z=1.2),
                   canonical_pose_candidates=[dict(pose_transform=nominal.tolist(), axis_assignment='YAW_0'),
                                              dict(pose_transform=second.tolist(), axis_assignment='YAW_180')])
        doc = dict(schema_version='real_pallet_gt_v2', objects=[obj])
        r = A.reference_document(doc)
        self.assertEqual(r['R'], np.eye(3).tolist())
        self.assertEqual(r['t'], [1., 2., 3.])
        bad = deepcopy(doc)
        bad['objects'][0]['canonical_pose_candidates'][1]['pose_transform'][0][3] += 1e-12
        self.assertIsNone(A.reference_document(bad)['R'])
        self.assertEqual(A.reference_document(bad)['t'], r['t'])

    def test_sample_variance_and_tail_not_trimmed(self):
        s = A.distribution([1., 3., 101., None])
        self.assertEqual(s['full_denominator'], 4)
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['missing'], 1)
        self.assertEqual(s['max'], 101.)
        self.assertAlmostEqual(s['sample_variance'], np.var([1, 3, 101], ddof=1))
        self.assertAlmostEqual(s['sample_std'], np.std([1, 3, 101], ddof=1))
        self.assertIsNone(A.distribution([1.])['sample_std'])
        with self.assertRaises(ValueError):
            A.distribution([float('nan')])

    def _paired_fixture(self):
        values = [(1, 2), (1, 10), (10, 1), (100, 20), (None, 20)]
        return [dict(id=str(i), partition='TEST', session='fixture-session',
                     cluster_id='fixture-recording-'+str(i % 2),
                     metrics={'A': measured(a, a), 'B': measured(b, b)},
                     prediction_status={'A': {}, 'B': {}}) for i, (a, b) in enumerate(values)]

    def test_bidirectional_transitions_and_failure_group_contributions(self):
        rows = self._paired_fixture()
        boot = A.RecordingBootstrap(rows)
        p = A.paired(rows, 'A', 'B', boot)
        counts = p['transition']['counts']
        self.assertEqual(counts, dict(both_success=1, failure_to_success=1, success_to_failure=1,
                                     both_failure=2, reference_unresolved=0))
        self.assertEqual(p['success_rate']['mean_difference'], 0.)
        d = p['continuous']['translation_cm']
        self.assertEqual(d['full_denominator'], 5)
        self.assertEqual(d['common_available_n'], 4)
        self.assertEqual(d['mean_difference'], 79/4)
        self.assertEqual(d['improvement_sum'], -10)
        self.assertEqual(d['worsening_sum'], 89)
        self.assertEqual(d['group_contributions']['both_failure']['signed_delta_sum'], 80)
        self.assertEqual(d['group_contributions']['both_failure']['common_available_n'], 1)
        self.assertAlmostEqual(sum(v['contribution_to_common_available_mean']
                                   for v in d['group_contributions'].values()), d['mean_difference'])

    def test_recording_resampling_retains_all_frames_and_missing_mask(self):
        rows = self._paired_fixture()
        boot = A.RecordingBootstrap(rows)
        values = [1., 2., 3., None, 5.]
        result = boot.contrast(values)
        weighted = []
        for draw in boot.draws:
            expanded = [v for r, v in zip(rows, values) if v is not None
                        for _ in range(int(draw[boot.units.index(r['cluster_id'])]))]
            if expanded:
                weighted.append(sum(expanded)/len(expanded))
        np.testing.assert_allclose(result['CI95'], np.quantile(weighted, [.025, .975]), atol=0, rtol=0)
        self.assertEqual(result['mean_difference'], 11/4)
        self.assertEqual(boot.metadata()['seed'], 20260917)
        self.assertEqual(boot.metadata()['resamples'], 10000)
        self.assertEqual(boot.metadata()['cluster_sizes'], {'fixture-recording-0': 3, 'fixture-recording-1': 2})
        bad = deepcopy(rows)
        del bad[0]['cluster_id']
        with self.assertRaises(ValueError):
            A.RecordingBootstrap(bad)

    def test_joint_success_never_combines_two_different_poses(self):
        first = A.metric(pose(yaw(10)), reference())
        second = A.metric(pose(t=(.10, 0, 1)), reference())
        self.assertFalse(first['success_le5'])
        self.assertFalse(second['success_le5'])
        self.assertEqual(first['translation_cm'], 0.)
        self.assertEqual(second['rotation_deg'], 0.)

    def test_no_scoring_GO_means_no_data_IO(self):
        with mock.patch.object(A, 'sha', side_effect=AssertionError('forbidden IO')):
            with self.assertRaises(PermissionError):
                A.score(folder='missing', source_root='missing', single_folder='missing',
                        guarded_folder='missing', reference_manifest='missing', candidate_file='missing',
                        expected_prediction_seal_sha256='x', expected_candidate_seal_sha256='x',
                        expected_reference_manifest_sha256='x', output='missing')

    def test_prediction_AND_candidate_seal_gate_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for part, n in A.PARTITIONS.items():
                for i in range(n):
                    rows.append(dict(id=part+str(i), partition=part, session='fixture-session',
                        cluster_id='fixture-recording', predictions={a: entry() for a in A.DIAGNOSTICS}))
            with gzip.open(root/'REPLAY_PREDICTIONS.jsonl.gz', 'wt') as f:
                for r in rows:
                    f.write(json.dumps(r)+'\n')
            candidate = root/'candidates.jsonl.gz'
            candidate.write_bytes(b'opaque predictor-only candidate fixture; never GT')
            ids = [r['id'] for r in rows]
            c = dict(rows=238, GT_inputs=False, before_new_scoring=True, ordered_ids=ids,
                     generated_candidates_sha256=A.sha(candidate), coverage='fixture')
            A.write(root/'CANDIDATE_SEAL.json', c)
            cs = A.sha(root/'CANDIDATE_SEAL.json')
            p = dict(rows=238, GT_inputs=False, before_new_scoring=True, ordered_ids=ids,
                     arms=list(A.DIAGNOSTICS), candidate_seal_sha256=cs,
                     predictions_sha256=A.sha(root/'REPLAY_PREDICTIONS.jsonl.gz'))
            A.write(root/'PREDICTION_SEAL.json', p)
            ps = A.sha(root/'PREDICTION_SEAL.json')
            valid, _, _ = A.prediction_gate(root, ps, cs, candidate)
            self.assertEqual(len(valid), 238)
            candidate.write_bytes(b'changed candidate bytes')
            with self.assertRaises(ValueError):
                A.prediction_gate(root, ps, cs, candidate)

    def test_missing_actual_recording_never_synthesized_from_session(self):
        rows = [dict(id='fixture', partition='TEST', session='recording-like-name')]
        with self.assertRaises(ValueError):
            A.identity_rows(rows, full=False)

    def test_both_existing_SubPix_caps_keep_common95_without_selecting(self):
        rows, old = [], []
        refs = {'CAL': {}, 'TEST': {}}
        for part, n in [('CAL', 46), ('TEST', 49)]:
            for i in range(n):
                fid = part+str(i)
                refs[part][fid] = reference()
                rows.append(dict(id=fid, partition=part, session='fixture-session',
                    cluster_id='fixture-recording-'+str(i % 2),
                    metrics={a: measured(1., 1.) for a in ('POSEFIX_RAW', 'N0', 'E0')},
                    prediction_status={a: {} for a in ('POSEFIX_RAW', 'N0', 'E0')}))
                for arm in A.PF_SUBPIX_ARMS:
                    old.append(dict(id=fid, partition=part, session='fixture-session', method=arm,
                        actual_pose=pose(t=(0., 0., 1.) if arm == A.PF_SUBPIX_ARMS[0] else (.10, 0., 1.))))
        seal = dict(predictions_sha256='sealed', protocol_sha256='sealed')
        protocol = dict(PoseFix=dict(checkpoint_sha256=A.PF_CHECKPOINT),
            new_combinations=dict(both_controls_fixed_before_scoring=True, no_best_anchor_selected=True,
                cap_fraction_raw_image_diagonal=.01, POSEFIX_SUBPIX_PF_ANCHOR='PF anchor fixed',
                POSEFIX_SUBPIX_YOLO_ANCHOR='YOLO anchor fixed'))
        with mock.patch.object(A, 'read', side_effect=lambda p: seal if p.name == 'PREDICTION_SEAL.json' else protocol), \
             mock.patch.object(A, 'sha', return_value='sealed'), mock.patch.object(A, 'journal', return_value=old):
            result, flat = A.posefix_subpix_common95('no-real-data', rows, refs)
        self.assertEqual(result['populations']['COMMON95']['rows'], 95)
        self.assertEqual(result['populations']['CAL']['rows'], 46)
        self.assertEqual(result['populations']['TEST']['rows'], 49)
        self.assertEqual(result['populations']['COMMON95']['arms'][A.PF_SUBPIX_ARMS[0]]['successes_le5'], 95)
        self.assertEqual(result['populations']['COMMON95']['arms'][A.PF_SUBPIX_ARMS[1]]['successes_le5'], 0)
        self.assertEqual(len(flat), 475)
        self.assertFalse(result['PRIMARY192_established'])
        self.assertTrue(result['no_cap_or_method_selection'])
        self.assertEqual(len(A.ARMS), 19)


if __name__ == '__main__':
    unittest.main()
