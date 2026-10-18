"""Synthetic hand-checkable fixtures only; no real reference reads."""
from __future__ import annotations
import math,tempfile,unittest
from pathlib import Path
import numpy as np
from . import independent_evaluation as I


def pose(angle=0.,center=(0.,0.,2.)):
    c,s=math.cos(angle),math.sin(angle)
    return dict(available=True,R_physical=[[c,0.,s],[0.,1.,0.],[-s,0.,c]],centroid=list(center))
REF=dict(order=2,R=np.eye(3).tolist(),t=[0.,0.,2.])
def item(index,before,after):
    return dict(id=str(index),cluster_id='record'+str(index%2),session='session'+str(index%3),
        metrics={'a':dict(pose_available=True,translation_cm=float(index+1),rotation_deg=float(index+2),success=before,metric_error=None),
                 'b':dict(pose_available=True,translation_cm=float(index+2),rotation_deg=float(index+1),success=after,metric_error=None)},
        status={'a':I.status({}),'b':I.status({})})


class IndependentEvaluationTests(unittest.TestCase):
    def test_joint_success_same_pose_and_units(self):
        r=I.metric(pose(math.radians(3.),(.04,0.,2.)),REF)
        self.assertTrue(r['success']);self.assertAlmostEqual(r['translation_cm'],4.);self.assertAlmostEqual(r['rotation_deg'],3.)
        self.assertFalse(I.metric(pose(math.radians(6.),(.04,0.,2.)),REF)['success'])
    def test_C2_yes_90degree_no(self):
        self.assertAlmostEqual(I.metric(pose(math.pi),REF)['rotation_deg'],0.)
        self.assertAlmostEqual(I.metric(pose(math.pi/2),REF)['rotation_deg'],90.)
    def test_no_pose_and_improper_rotation_retained_as_failure(self):
        r=I.metric({'available':False},REF);self.assertFalse(r['pose_available']);self.assertIsNone(r['translation_cm'])
        p=pose();p['R_physical'][0][0]=2.;r=I.metric(p,REF);self.assertFalse(r['pose_available']);self.assertEqual(r['metric_error'],'ValueError')
    def test_raw_and_stable_are_separate(self):
        r=I.metric(pose(1e-9),REF);s=I.metric(pose(1e-9),REF,True)
        self.assertEqual(r['rotation_deg'],0.);self.assertGreater(s['rotation_deg'],0.);self.assertTrue(r['success']and s['success'])
    def test_distribution_full_denominator_and_ddof1(self):
        d=I.distribution([1.,2.,None,3.,4.],5)
        self.assertEqual((d['n'],d['missing'],d['full_denominator']),(4,1,5));self.assertEqual(d['mean'],2.5)
        self.assertAlmostEqual(d['sample_variance'],5./3.);self.assertAlmostEqual(d['P90'],3.7);self.assertAlmostEqual(d['P95'],3.85)
    def test_zero_one_available_distribution(self):
        self.assertIsNone(I.distribution([],0)['mean']);self.assertIsNone(I.distribution([1.],1)['sample_std'])
    def test_exact_strict_directions_and_paired_groups(self):
        rows=[item(0,True,True),item(1,False,True),item(2,True,False),item(3,False,False)]
        p=I.paired(rows,'a','b',I.Bootstrap())
        self.assertEqual(p['counts'],dict(BOTH_SUCCESS=1,RECOVERY=1,DAMAGE=1,BOTH_FAILURE=1));self.assertEqual(p['success_difference_count'],0)
        self.assertEqual(p['deltas']['translation_cm']['worsened'],4);self.assertEqual(p['deltas']['rotation_deg']['improved'],4)
        self.assertEqual(p['both_failure_continuous']['denominator'],1)
    def test_stable_six_record_eight_session_bootstrap_draws(self):
        records=[dict(cluster_id='r'+str(i%6),session='s'+str(i%8))for i in range(24)]
        first,second=I.Bootstrap(),I.Bootstrap()
        for unit,n in(('cluster_id',6),('session',8)):
            a=first.interval([1.]*24,records,unit);b=second.interval([1.]*24,records,unit)
            self.assertEqual(a,b);self.assertEqual(a['CI95'],[1.,1.]);self.assertEqual(a['cluster_count'],n)
        self.assertEqual(first.proofs,second.proofs)
        self.assertTrue(all(v['draws']==10000 for v in first.proofs.values()))
    def test_unavailable_continuous_pairs_keep_binary_denominator(self):
        records=[dict(cluster_id='r0',session='s0'),dict(cluster_id='r1',session='s1')]
        ci=I.Bootstrap().interval([None,None],records,'cluster_id');self.assertIsNone(ci['CI95']);self.assertEqual(ci['empty_resamples'],10000)
    def test_no_new_action_group_or_seed(self):
        self.assertEqual(len(I.registered(False)),26);self.assertEqual(len(I.registered(True)),32)
        self.assertEqual(len(I.contrasts(False)),105);self.assertEqual(len(I.contrasts(True)),159)
    def test_source_priority_and_no_symmetry_alias_collapse(self):
        f=dict(fallback=pose(),e0_regenerated=pose(math.pi),object_scale_m=1.)
        self.assertEqual(I.source(pose(),f),'N0');self.assertEqual(I.source(pose(math.pi),f),'E0');self.assertEqual(I.source(pose(math.pi/2),f),'OTHER')
    def test_reference_gate_precedes_all_input_reads(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)
            with self.assertRaisesRegex(PermissionError,'scoring-GO'):I.run(p,p,p/'features',p,p/'output',GO=False)
            self.assertFalse((p/'output').exists())


if __name__=='__main__':unittest.main()
