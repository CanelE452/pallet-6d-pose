"""Actual pose/order bindings and ID-independent guarded return contracts."""
from copy import deepcopy
import unittest
import numpy as np
from .predict import bind_scores, guarded_entry, archived_h1_index
from .registry import SEEDS, arms, contrasts


def fixture():
    poses = [dict(available=True, R_physical=np.eye(3).tolist(), centroid=[float(i), 0., 2.]) for i in range(2)]
    entries = [dict(candidate_id="arbitrary_" + str(i), actual_pose=p,
                    candidate_aliases=["arbitrary_" + str(i)],
                    source_baseline_aliases=["N0"] if i == 0 else ["E0"])
               for i, p in enumerate(poses)]
    predictions = {}
    for seed in SEEDS:
        actions = [dict(candidate_id=e["candidate_id"]) for e in entries]
        predictions["D1_H1_MLP_s" + str(seed)] = dict(logits=[2., 1.], actions=deepcopy(actions), actual_pose=deepcopy(poses[0]))
        predictions["R_CONT_F0_s" + str(seed)] = dict(model_outputs=[[3., 2.], [1., 1.]], actions=deepcopy(actions), actual_pose=deepcopy(poses[1]))
    return entries, predictions


class PredictionContract(unittest.TestCase):
    def test_exact_order_and_actual_pose_binding(self):
        entries, predictions = fixture()
        h, r = bind_scores({}, entries, predictions)
        self.assertTrue(np.array_equal(h[SEEDS[0]], [2., 1.]))
        self.assertEqual(r[SEEDS[0]].shape, (2, 2))
        predictions["D1_H1_MLP_s20261018"]["actual_pose"] = entries[1]["actual_pose"]
        with self.assertRaises(ValueError):
            bind_scores({}, entries, predictions)

    def test_reordered_scores_are_rejected(self):
        entries, predictions = fixture()
        predictions["R_CONT_F0_s20261018"]["actions"].reverse()
        with self.assertRaises(ValueError):
            bind_scores({}, entries, predictions)

    def test_renaming_ids_changes_no_returned_physical_pose(self):
        entries, predictions = fixture()
        h, r = bind_scores({}, entries, predictions)
        first = guarded_entry(dict(object_scale_m=1.), entries, h[SEEDS[0]], r[SEEDS[0]], 1., "CONFIRMED")
        for i, entry in enumerate(entries):
            entry["candidate_id"] = "renamed_" + str(100 - i)
        for pred in predictions.values():
            pred["actions"] = [dict(candidate_id=e["candidate_id"]) for e in entries]
        h, r = bind_scores({}, entries, predictions)
        second = guarded_entry(dict(object_scale_m=1.), entries, h[SEEDS[0]], r[SEEDS[0]], 1., "CONFIRMED")
        self.assertEqual(first["actual_pose"], second["actual_pose"])
        self.assertTrue(second["guard_decision"]["changed"])

    def test_identity_retains_actual_h1(self):
        entries, predictions = fixture()
        h, r = bind_scores({}, entries, predictions)
        value = guarded_entry(dict(object_scale_m=1.), entries, h[SEEDS[0]], r[SEEDS[0]], -np.inf, "FALLBACK")
        self.assertEqual(value["actual_pose"], predictions["D1_H1_MLP_s20261018"]["actual_pose"])
        self.assertTrue(value["fallback"])
        self.assertFalse(value["guard_decision"]["changed"])

    def test_trial_is_explicitly_unadopted(self):
        entries, predictions = fixture()
        h, r = bind_scores({}, entries, predictions)
        value = guarded_entry(dict(object_scale_m=1.), entries, h[SEEDS[0]], r[SEEDS[0]], 1., "FAILED_CONFIRMATION", True)
        self.assertTrue(value["diagnostic_trial"])
        self.assertEqual(value["adoption_scope"], "UNADOPTED_CALIBRATION_TRIAL")

    def test_finite_registry_contains_all_seeds_and_comparisons(self):
        self.assertEqual(len(arms()), 26)
        self.assertEqual(len(arms(True)), 32)
        self.assertEqual(len(set(arms(True))), 32)
        self.assertTrue(all(a in arms(True) and b in arms(True) for a, b in contrasts(True)))

    def test_invalid_scores_keep_archived_non_N0_h1(self):
        for h_bad in (None, [], [np.nan, np.nan], [np.inf, 2.], ['bad', 2.]):
            for r_bad in (None, [], [[np.nan, 1.], [2., 1.]], [[np.inf, 1.], [2., 1.]]):
                entries, predictions = fixture()
                h = predictions['D1_H1_MLP_s20261018'];h['actual_pose'] = deepcopy(entries[1]['actual_pose'])
                if h_bad is None:h.pop('logits')
                else:h['logits'] = h_bad
                r = predictions['R_CONT_F0_s20261018']
                if r_bad is None:r.pop('model_outputs')
                else:r['model_outputs'] = r_bad
                hs, rs = bind_scores({}, entries, predictions)
                index = archived_h1_index(entries, h)
                out = guarded_entry(dict(object_scale_m=1.), entries, hs[SEEDS[0]], rs[SEEDS[0]], 100., 'TEST', h1_selected_index=index)
                self.assertEqual(index, 1)
                self.assertEqual(out['actual_pose'], entries[1]['actual_pose'])
                self.assertEqual(out['guard_decision']['reason'], 'SCORE_UNAVAILABLE_OR_NONFINITE')
                self.assertFalse(out['guard_decision']['changed'])

    def test_missing_alternative_or_score_context_keeps_actual_h1(self):
        for mode in ('missing_R_record', 'missing_R_scores', 'missing_R_pose', 'missing_R_actions', 'missing_H_actions'):
            entries, predictions = fixture();h=predictions['D1_H1_MLP_s20261018']
            h['actual_pose']=deepcopy(entries[1]['actual_pose']);h['logits']=[1.,2.]
            if mode=='missing_R_record':predictions.pop('R_CONT_F0_s20261018')
            elif mode=='missing_R_scores':predictions['R_CONT_F0_s20261018'].pop('model_outputs')
            elif mode=='missing_R_pose':predictions['R_CONT_F0_s20261018']['actual_pose']=dict(available=False)
            elif mode=='missing_R_actions':predictions['R_CONT_F0_s20261018'].pop('actions')
            else:h.pop('actions')
            hs,rs=bind_scores({},entries,predictions)
            out=guarded_entry(dict(object_scale_m=1.),entries,hs[SEEDS[0]],rs[SEEDS[0]],100.,'TEST',h1_selected_index=archived_h1_index(entries,h))
            self.assertEqual(out['actual_pose'],entries[1]['actual_pose']);self.assertFalse(out['guard_decision']['changed'])

    def test_absent_h1_never_fabricates_N0(self):
        entries,predictions=fixture();predictions.pop('D1_H1_MLP_s20261018')
        hs,rs=bind_scores({},entries,predictions)
        out=guarded_entry(dict(object_scale_m=1.),entries,hs[SEEDS[0]],rs[SEEDS[0]],100.,'TEST',h1_selected_index=archived_h1_index(entries,None))
        self.assertEqual(out['actual_pose'],dict(available=False));self.assertIsNone(out['selected_candidate_id']);self.assertEqual(out['guard_decision']['reason'],'H1_UNAVAILABLE')

    def test_finite_score_corruption_still_rejected(self):
        entries,predictions=fixture();predictions['R_CONT_F0_s20261018']['actual_pose']=deepcopy(entries[0]['actual_pose'])
        with self.assertRaisesRegex(ValueError,'R_CONT returned pose/score mismatch'):bind_scores({},entries,predictions)
        entries,predictions=fixture();predictions['D1_H1_MLP_s20261018']['actual_pose']['centroid'][0]=99.
        with self.assertRaisesRegex(ValueError,'exactly one unchanged action'):bind_scores({},entries,predictions)
        entries,predictions=fixture();predictions['D1_H1_MLP_s20261018'].pop('logits');entries.append(deepcopy(entries[0]))
        for p in predictions.values():p['actions'].append(deepcopy(p['actions'][0]))
        with self.assertRaisesRegex(ValueError,'exactly one unchanged action'):bind_scores({},entries,predictions)


if __name__ == "__main__":
    unittest.main()
