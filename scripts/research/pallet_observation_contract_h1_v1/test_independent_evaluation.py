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
        self.assertEqual(len(I.contrasts(False)),84);self.assertEqual(len(I.contrasts(True)),143)
    def test_source_priority_and_no_symmetry_alias_collapse(self):
        f=dict(fallback=pose(),e0_regenerated=pose(math.pi),object_scale_m=1.)
        self.assertEqual(I.source(pose(),f),'N0');self.assertEqual(I.source(pose(math.pi),f),'E0');self.assertEqual(I.source(pose(math.pi/2),f),'OTHER')
    def test_reference_gate_precedes_all_input_reads(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)
            with self.assertRaisesRegex(PermissionError,'scoring-GO'):I.run(p,p,p/'features',p,p/'output',GO=False)
            self.assertFalse((p/'output').exists())
    def test_independent_guard_closed_gap_and_strict_benefit(self):
        entries=[dict(actual_pose=pose(center=(0.,0.,2.))),dict(actual_pose=pose(center=(1.,0.,2.)))]
        a=I.independent_guard(entries,[2.,1.],[[3.,2.],[1.,1.]],1.,1.)
        self.assertTrue(a['changed']);self.assertEqual(a['selected_index'],1)
        self.assertFalse(I.independent_guard(entries,[2.,1.],[[3.,2.],[1.,1.]],.99,1.)['changed'])
        self.assertFalse(I.independent_guard(entries,[2.,1.],[[1.,1.],[1.,1.]],1.,1.)['changed'])
        self.assertFalse(I.independent_guard(entries,[2.,1.],[[3.,2.],[1.,1.]],-math.inf,1.)['changed'])
        self.assertEqual(I.independent_guard([],[],[],-math.inf,1.)['reason'],'H1_UNAVAILABLE')
    def test_guard_original_tie_order_and_pose_alias(self):
        e=[dict(actual_pose=pose()),dict(actual_pose=pose(center=(1.,0.,2.)))]
        a=I.independent_guard(e,[0.,0.],[[2.,2.],[1.,1.]],0.,1.)
        self.assertEqual(a['h1_index'],0);self.assertEqual(a['rcont_index'],1)
        e[1]['actual_pose']=e[0]['actual_pose'];self.assertEqual(I.independent_guard(e,[0.,0.],[[2.,2.],[1.,1.]],0.,1.)['reason'],'SAME_PHYSICAL_POSE')
    def test_new_protocol_and_augmentation_inventory(self):
        p=I.expected_protocol();self.assertEqual(p['campaign'],'OBSERVATION_CONTRACT_H1_V1')
        self.assertTrue(p['calibrated_trial_is_diagnostic_not_adopted_performance']);self.assertEqual(p['actual_PRIMARY_pose_solves'],0)
        methods=I.registered(False);self.assertEqual(len(set(methods)),26)
        self.assertEqual(set(I.registered(True))-set(methods),{g+'_s'+str(s)for g in I.AUGMENTATION for s in I.SEEDS})
    def test_complete_extra_table_fixture_without_reference(self):
        methods=I.registered(False);p0,p1=pose(),pose(center=(1.,0.,2.))
        feature=dict(fallback=p0,e0_regenerated=p1,object_scale_m=1.,candidates=[
            dict(candidate_id=name,actual_pose=p,eligible=True,features=[0.]*62,source_baseline_aliases=[name])for name,p in(('N0',p0),('E0',p1))])
        actions=I.independent_actions(feature);predictions={a:dict(actual_pose=p0)for a in methods};policies=dict(seeds={})
        for seed in I.SEEDS:
            policies['seeds'][str(seed)]=dict(status='CONFIRMED_INTERNAL_POLICY',active_tau=dict(kind='FINITE',value=1.),calibrated_trial_tau=dict(kind='FINITE',value=1.))
            predictions['D1_H1_MLP_s'+str(seed)].update(logits=[2.,1.],actions=actions)
            predictions['R_CONT_F0_s'+str(seed)].update(model_outputs=[[2.,2.],[1.,1.]],actions=actions)
            for group in('H1_GUARDED_SWITCH','H1_GUARD_CALIBRATED_TRIAL'):
                d=I.independent_guard(actions,[2.,1.],[[2.,2.],[1.,1.]],1.,1.)
                predictions[group+'_s'+str(seed)]=dict(actual_pose=p1,guard_decision=d,selected_candidate_id='E0',action_role='E0',
                    diagnostic_trial=group.endswith('TRIAL'),policy_status='CONFIRMED_INTERNAL_POLICY')
        record=dict(id='fixture',partition='TEST',metrics={a:dict(translation_cm=101. if a in I.BASELINES else 1.,rotation_deg=46. if a in I.BASELINES else 1.,success=a not in I.BASELINES)for a in methods})
        def check(name,value,detail=None):self.assertTrue(value,(name,detail))
        def compare(want,got,path=''):self.assertEqual(want,got,path)
        tails,cross,decisions,selected=I.extra_tables([record],[dict(id='fixture',partition='TEST',predictions=predictions)],methods,False,{'fixture':feature},policies,check,compare)
        # Empty CAL/REMAINDER populations still retain every zero-count bin.
        self.assertEqual((len(tails),len(cross),len(decisions),len(selected)),(504,1664,6,1))
        self.assertTrue(all(r['event']=='REMOVED_LARGE_ERROR'for r in tails));self.assertTrue(all(r['changed']for r in decisions))


if __name__=='__main__':unittest.main()
