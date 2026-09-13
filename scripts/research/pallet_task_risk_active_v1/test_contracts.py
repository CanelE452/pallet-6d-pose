"""Synthetic-only regression suite; no target GT values or GPU forwards."""
import copy
import math
import unittest
import cv2
import numpy as np
from contracts import *
from perturb import views
from risk import wrap,midranks,summarize,calculate,cvar90
from stats import auc,hard20,corr
from acquisition import old_parity,select
from pose_adapter import pose,pool_truth


def example(**kwargs):
    return dict(dict(pose_valid=True,detection=True,candidate_index=5,axis_id='CF_WIDTH',
                     x_m=1.,z_m=3.,yaw_rad=0.),**kwargs)


class SyntheticTests(unittest.TestCase):
    def test_perturbation_deterministic_shape_dtype_and_P0(self):
        x=np.random.default_rng(7).integers(0,256,(47,61,3),dtype=np.uint8);a,b=views(x),views(x)
        self.assertEqual(len(a),8);self.assertTrue(np.array_equal(a[0],x))
        for aa,bb in zip(a,b):
            self.assertTrue(np.array_equal(aa,bb));self.assertEqual(aa.shape,x.shape);self.assertEqual(aa.dtype,x.dtype)
        self.assertTrue(np.array_equal(a[1],(x.astype(np.float32)*.8).astype(np.uint8)))
    def test_photometric_pixel_mapping_and_channel_order(self):
        x=np.zeros((11,11,3),np.uint8);x[5,5]=[70,130,220]
        for view in views(x):
            for c in range(3):self.assertEqual(np.unravel_index(view[:,:,c].argmax(),(11,11)),(5,5))
        for a,b in zip(views(x),views(x[:,:,::-1])):self.assertTrue(np.array_equal(a[:,:,::-1],b))
    def test_yaw_wrap_boundary(self):
        a=[example(yaw_rad=np.radians(v)) for v in [179,-179,178,-178,179,-179,180,-180]]
        r=summarize(a);self.assertLess(r['spread_yaw_deg'],3)
        self.assertAlmostEqual(float(wrap(np.radians(359))),np.radians(-1))
    def test_constant_risk(self):
        r=calculate([dict(frame_id='a',views=[example() for _ in range(8)])])[0]
        self.assertEqual(r['R_task'],.5);self.assertEqual(r['spread_x_m'],0)
    def test_ties_missing_and_bounds(self):
        self.assertTrue(np.array_equal(midranks([1,1,2,None]),[1/3,1/3,5/6,1]))
        self.assertTrue(np.array_equal(midranks([None,None]),[1,1]))
    def test_candidate_axis_and_failure_rates(self):
        a=[example() for _ in range(8)];a[1]['candidate_index']=6;a[2]['axis_id']='CF_DEPTH'
        a[7]=example(pose_valid=False,detection=False)
        r=summarize(a);self.assertEqual(r['candidate_switch_rate'],1/7);self.assertEqual(r['axis_switch_rate'],1/7)
        self.assertEqual(r['pnp_failure_rate'],1/8);self.assertEqual(r['detection_failure_rate'],1/8)
    def test_missing_anchor_and_all_missing(self):
        a=[example() for _ in range(8)];a[0]=example(pose_valid=False,detection=False)
        r=summarize(a);self.assertIsNone(r['spread_yaw_rad']);self.assertEqual(r['axis_switch_rate'],1)
        a=[example(pose_valid=False,detection=False) for _ in range(8)]
        r=calculate([dict(frame_id='x',views=a)])[0];self.assertEqual(r['R_task'],1);self.assertEqual(r['valid_poses'],0)
    def test_risk_recompute_and_max_not_sum(self):
        rows=[dict(frame_id=str(i),views=[example(x_m=i*j*.01,yaw_rad=i*j*.02) for j in range(8)]) for i in range(8)]
        self.assertEqual(calculate(rows),calculate(copy.deepcopy(rows)))
        for r in calculate(rows):self.assertGreaterEqual(r['R_task'],0);self.assertLessEqual(r['R_task'],1)
    def test_cvar_independent_145_worst15(self):
        values=list(range(145));self.assertAlmostEqual(cvar90(values),sum(range(130,145))/15)
        self.assertEqual(cvar90([5]),5)
        with self.assertRaises(AssertionError):cvar90([1,np.nan])
    def test_statistics_hard_ties_auc_and_rank(self):
        self.assertEqual(auc([0,1,2,3],[0,0,1,1]),1)
        self.assertEqual(auc([1,1,1,1],[0,0,1,1]),.5)
        self.assertIsNone(auc([0,1],[1,1]))
        h,t=hard20([0,1,2,3,3]);self.assertEqual(int(h.sum()),2)
        self.assertEqual(corr([1,2,3],[3,2,1]),-1)
    def test_canonical_pose_and_GT_solver_parity(self):
        evaluate,build,_=canonical_modules();long,short,height=1.3,1.1,.11
        k=np.array([[610,0,320],[0,610,240],[0,0,1]],float)
        model=evaluate.cuboid(long,height,short)
        point=cv2.projectPoints(model,np.array([.25,.15,.07]),np.array([.1,.2,3.]),k,None)[0].reshape(8,2)
        points=np.vstack([point,point.mean(0)])
        p=pose(points,k,[long,short,height]);self.assertTrue(p['pose_valid'])
        a,b=(long,short) if p['axis_id']=='CF_WIDTH' else (short,long)
        canonical=evaluate.solve(evaluate.cuboid(a,height,b),point,k,np.ones(8,bool))
        self.assertTrue(np.array_equal(p['rotation'],canonical[0]));self.assertTrue(np.array_equal(p['translation'],canonical[1]))
        t=pool_truth(points,k,[long,short,height]);self.assertEqual(t['axis_id'],'CF_WIDTH')
        self.assertLess(t['reprojection_px'],1e-6)
        self.assertTrue(np.array_equal(evaluate.cuboid(a,height,b),build.cuboid(a,height,b)))
    def test_pose_is_GT_free_and_missing_handled(self):
        self.assertFalse(pose(None,np.eye(3),[1.3,1.1,.11])['pose_valid'])
        import inspect,pose_adapter
        self.assertNotIn('read(',inspect.getsource(pose_adapter))
    def test_old_selection_exact_reproduction(self):
        self.assertTrue(all(r['exact_order'] for r in old_parity().values()))
    def test_previous_means_and_split_binding(self):
        old=read(OLD_DOC/'RESULTS.json');split=read(DOC/'SPLIT_BINDING.json')
        self.assertEqual((len(split['pool']),len(split['evaluation'])),(174,145))
        self.assertEqual(len(read(NEGATIVE)['items']),2689)
        for method,supplied in old['seed_means'].items():
            for k in ('translation_median_cm','yaw_median_deg'):
                self.assertAlmostEqual(np.mean([old['per_run'][f'{method}_seed{s}']['pose']['ALL'][k] for s in (1,2,3)]),supplied['pose'][k],places=12)
        self.assertEqual(sha(R0),R0_SHA)


if __name__=='__main__':unittest.main()
