import unittest
import numpy as np
from . import metric_baseline as M


def row(t,r,valid=True):
    return dict(available=valid,translation_cm=t,rotation_deg=r,yaw_deg=r,IoU3D=.5,ADDsym_normalized=.05,axis_correct=True)


class PoseObjectiveContracts(unittest.TestCase):
    def test_extended_quantiles_keep_failures_and_match_finite_numpy(self):
        for q in (0,.1,.5,.9,1):
            self.assertAlmostEqual(M.extended_quantile([3,1,8,4],q),np.quantile([3,1,8,4],q))
        self.assertTrue(np.isinf(M.extended_quantile([1,float('inf')],.5)))
        out=M.summarize([row(1,2),row(0,0,False)])
        self.assertEqual(out['coverage'],.5)
        self.assertEqual(out['conditional']['translation_cm']['median'],1)
        self.assertIsNone(out['full_population']['translation_cm']['median'])
        self.assertEqual(out['full_population']['translation_cm']['median_status'],'POSITIVE_INFINITY')

    def test_pooled_median_is_not_median_of_group_medians(self):
        a=[row(1,1)];b=[row(3,3),row(5,5),row(9,9)]
        self.assertEqual(M.summarize(a+b)['conditional']['translation_cm']['median'],4)
        self.assertNotEqual(4,np.mean([1,5]))

    def test_full_rotation_C2_units_and_centroid(self):
        from scripts.paper.pose_metric_closure_v1.symmetry_aware_pose_metrics import rotation_error_degrees
        ry90=np.array([[0,0,1],[0,1,0],[-1,0,0]],float)
        ry180=np.diag([-1,1,-1]);rx90=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
        self.assertAlmostEqual(rotation_error_degrees(ry180,np.eye(3)),0)
        self.assertAlmostEqual(rotation_error_degrees(ry90,np.eye(3)),90)
        self.assertAlmostEqual(rotation_error_degrees(rx90,np.eye(3)),90)
        pose=dict(available=True,centroid=[.03,0,1.04],R_physical=np.eye(3))
        truth=dict(order=2,t=[0,0,1],R=np.eye(3))
        out=M.extend_metric(row(5,0),pose,truth)
        self.assertAlmostEqual(out['camera_x_signed_cm'],3)
        self.assertAlmostEqual(out['camera_z_signed_cm'],4)

    def test_paired_median_and_axis_directions_are_not_conflated(self):
        before={str(i):row(t,r) for i,(t,r) in enumerate([(1,5),(10,1),(11,9)])}
        after={str(i):row(t,r) for i,(t,r) in enumerate([(2,4),(9,2),(30,9)])}
        out=M.paired(before,after)
        self.assertEqual(out['difference_of_conditional_medians']['translation_cm'],-1)
        self.assertEqual(out['median_of_common_frame_differences']['translation_cm'],1)
        self.assertEqual(sum(out['paired_direction_counts'].values()),3)

    def test_coverage_guard_and_Pareto_cost_tie_break_not_AUC(self):
        baseline=M.summarize([row(5,5),row(6,6)])
        new=M.summarize([row(1,1),row(0,0,False)])
        self.assertFalse(M.classify_candidate(new,baseline)['eligible_joint'])
        cards=[dict(card_id=i,translation_cm=t,rotation_deg=r,eligible_joint=True,added_inference_seconds=s,new_training_GPU_seconds=g)
               for i,t,r,s,g in [('B',2,1,0,5),('A',1,2,0,5),('C',3,3,0,1)]]
        self.assertEqual([v['card_id'] for v in M.pareto_front(cards)],['A','B'])
        self.assertEqual(M.choose_repeat(cards),'A')

    def test_distinct_oracle_extrema_remain_complete_poses(self):
        choices={'f':[dict(name='A',metric=row(1,9)),dict(name='B',metric=row(8,2)),dict(name='BASE',metric=row(5,5))]}
        result,selected=M.oracle_for(['f'],choices,{'f':row(5,5)})
        self.assertEqual(result['translation_optimal']['conditional']['rotation_deg']['median'],9)
        self.assertEqual(result['rotation_optimal']['conditional']['translation_cm']['median'],8)
        self.assertEqual(result['frames_with_same_candidate_joint_gain'],0)
        self.assertEqual(result['T_R_optimal_choices_different'],1)
        reverse={'f':list(reversed(choices['f']))}
        self.assertEqual(M.oracle_for(['f'],reverse,{'f':row(5,5)})[1],selected)

    def test_frozen_actual_population_and_baseline_contract(self):
        path=M.DOC/'BASELINE_POSE_RESULTS.json'
        if not path.exists():self.skipTest('Frozen baseline has not run')
        lock=M.read(M.DOC/'METRIC_AND_SELECTION_LOCK.json')
        for binding in lock['inputs']:M.verify(binding)
        result=M.read(path)
        self.assertEqual(result['materials']['PLASTIC']['groups'][M.PRIMARY]['R0']['frames'],99)
        self.assertEqual(result['materials']['WOOD']['groups']['severity:SEVERE_OCCLUSION']['R0']['status'],'NA_EMPTY_POPULATION')
        self.assertEqual(result['numeric_parity_frame_arm_comparisons'],1384)


if __name__=='__main__':unittest.main()
