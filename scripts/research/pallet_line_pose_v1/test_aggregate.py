"""Generated CPU checks for exact evaluator statistics and paired seed handling."""
import sys
import unittest
from pathlib import Path

import numpy as np

from aggregate_results import (KEYPOINT_METRICS, METRICS, POSE_METRICS, ROOT,
    make_verdict, paired_metric, pose_auc, weighted_quantile, weighted_statistic)

sys.path.insert(0,str(ROOT/'scripts/paper/pose_metric_closure_v1'))
from paired_bootstrap_pose import statistic as original_pose_statistic


class AggregateTests(unittest.TestCase):
    def test_weighted_linear_quantile_matches_explicit_repetition(self):
        values = np.array([.1,.3,.3,.91,.04])
        weights = np.array([[1,2,0,3,1],[0,1,1,0,0],[4,3,1,2,1]])
        for q in (.5,.9):
            expected = [np.quantile(np.repeat(values,row),q) for row in weights]
            np.testing.assert_allclose(weighted_quantile(values,weights,q),expected,atol=1e-14,rtol=0)

    def test_auc_optimization_matches_original_threshold_trapezoid(self):
        rng = np.random.default_rng(8)
        values = np.r_[0.,.17,.1701,rng.uniform(0,.2,27)]
        diameters = rng.choice([.98,1.42,1.70],len(values))
        groups = np.arange(len(values))
        counts = rng.multinomial(len(values),np.full(len(values),1/len(values)),size=101)
        actual = weighted_statistic(values,groups,counts,"add_sym_auc",diameters)
        expected = [original_pose_statistic(np.repeat(values,row),np.repeat(diameters,row),"auc") for row in counts]
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
        self.assertAlmostEqual(pose_auc(values,diameters),original_pose_statistic(values,diameters,"auc"),places=14)

    def test_three_seeds_share_resampled_frames_without_n_inflation(self):
        baseline = {str(i):dict(session_id=f"session{i//3}",errors=np.array([4+i,6+i],dtype=float)) for i in range(15)}
        left = [{key:dict(session_id=row["session_id"],errors=row["errors"]-shift) for key,row in baseline.items()}
                for shift in (.2,.5,.8)]
        for metric in KEYPOINT_METRICS:
            result = paired_metric(left,[baseline]*3,metric,resamples=200)
            self.assertEqual(result["paired_frames"],15)
            self.assertEqual(result["paired_sessions"],5)
            self.assertAlmostEqual(result["difference"],-.5,places=12)
            for scheme in ("frame_level","session_cluster"):
                self.assertAlmostEqual(result[scheme]["low"],-.5,places=12)
                self.assertAlmostEqual(result[scheme]["high"],-.5,places=12)

    def test_2d_gain_does_not_imply_6d_gain_or_overall_success(self):
        baseline = dict(two_d={name:2. for name in KEYPOINT_METRICS},
            main_6d={name:(.4 if name in ("iou3d_median","add_sym_auc") else 2.) for name in POSE_METRICS},coverage=1.)
        metrics = {name:dict(mean=.5 if name in ("iou3d_median","add_sym_auc") else 1.) for name in METRICS}
        comparisons = {name:dict(session_cluster=dict(confirmed_benefit=True)) for name in METRICS}
        comparisons["translation_median_cm"]["session_cluster"]["confirmed_benefit"] = False
        summary = dict(baseline=baseline,arms={"image_joint":dict(metrics=metrics)},
            comparisons={"image_joint_vs_R0":dict(metrics=comparisons)},
            runs=[dict(arm="image_joint",coverage=1.) for _ in range(3)],identity=dict(PASS=True))
        verdict = make_verdict(summary)
        self.assertTrue(verdict["PASS"])
        self.assertTrue(verdict["keypoint_gain_confirmed"])
        self.assertFalse(verdict["pose_gain_confirmed"])
        self.assertFalse(verdict["overall_accuracy_improved"])
        comparisons["translation_median_cm"]["session_cluster"]["confirmed_benefit"] = True
        self.assertTrue(make_verdict(summary)["overall_accuracy_improved"])
        summary["runs"][1]["coverage"] = .99
        self.assertFalse(make_verdict(summary)["pose_gain_confirmed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
