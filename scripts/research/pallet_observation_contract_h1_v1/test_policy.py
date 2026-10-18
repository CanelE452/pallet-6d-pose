import math,unittest
from copy import deepcopy
from . import policy as P

def entry(x):return dict(actual_pose=dict(available=True,R_physical=[[1.,0,0],[0,1.,0],[0,0,1.]],centroid=[float(x),0.,2.]))
def row(gap=1.,before=(10.,10.),after=(4.,4.)):
    def m(x):return dict(translation_cm=x[0],rotation_deg=x[1],success=x[0]<=5 and x[1]<=5)
    return dict(entries=[entry(0),entry(1)],h1_logits=[gap,0.],rcont_axes=[[2.,2.],[1.,1.]],object_scale_m=1.,h1_selected_index=0,action_metrics=[m(before),m(after)])

class GuardPolicy(unittest.TestCase):
    def test_strict_benefit_and_closed_gap(self):
        r=row();self.assertFalse(P.select_guarded(r['entries'],[1.,0.],[[1.,1.],[1.,1.]],2.,1.)['changed'])
        self.assertFalse(P.select_guarded(r['entries'],[1.,0.],r['rcont_axes'],.999,1.)['changed'])
        self.assertTrue(P.select_guarded(r['entries'],[1.,0.],r['rcont_axes'],1.,1.)['changed'])
    def test_identity_missing_and_nonfinite(self):
        r=row()
        self.assertEqual(P.select_guarded(r['entries'],None,None,1.,1.,h1_selected_index=1)['selected_index'],1)
        self.assertEqual(P.select_guarded(r['entries'],[1.,0.],[[math.nan,1.],[1.,1.]],1.,1.)['selected_index'],0)
        self.assertEqual(P.select_guarded([],[],[],1.,1.)['reason'],'H1_UNAVAILABLE')
        self.assertFalse(P.select_guarded(r['entries'],r['h1_logits'],r['rcont_axes'],-math.inf,1.)['changed'])
    def test_ties_and_id_rename(self):
        r=row();a=P.select_guarded(r['entries'],[0.,0.],r['rcont_axes'],0.,1.)
        rr=deepcopy(r['entries'])
        for i,e in enumerate(rr):e['candidate_id']='renamed'+str(99-i)
        self.assertEqual(a,P.select_guarded(rr,[0.,0.],r['rcont_axes'],0.,1.))
        self.assertEqual(a['h1_index'],0)
    def test_same_pose_no_switch(self):
        e=[entry(0),entry(0)];self.assertEqual(P.select_guarded(e,[1.,0.],[[2.,2.],[1.,1.]],2.,1.)['reason'],'SAME_PHYSICAL_POSE')
    def test_all_unique_gap_thresholds_no_grid(self):
        self.assertEqual(P.candidate_thresholds([row(2.),row(.1),row(2.)]),[-math.inf,.1,2.])
        self.assertEqual(P.decode_tau(P.encode_tau(-math.inf)),-math.inf)
        with self.assertRaises(ValueError):P.encode_tau(math.inf)
    def test_confirmation_failure_never_reselects(self):
        d=[row(1.),row(2.)];c=[row(1.,before=(4.,4.),after=(100.,100.))]
        a=P.calibrate(d,c,calibration_GO=True)
        self.assertEqual(a['status'],'CONFIRMATION_FAILED_KEEP_H1');self.assertEqual(a['confirmation_applications'],1)
        self.assertEqual(P.decode_tau(a['active_tau']),-math.inf);self.assertGreater(len(a['all_thresholds']),1)
    def test_nonidentity_strict_and_no_groups(self):
        r=row(before=(10.,10.),after=(10.,10.));a=P.calibrate([r],[r],calibration_GO=True)
        self.assertEqual(a['status'],'NO_ADMISSIBLE_NONIDENTITY_POLICY')
        a=P.calibrate([row()],[row()],internal_check_available=False,calibration_GO=True)
        self.assertEqual(a['status'],'INTERNAL_POLICY_CHECK_UNAVAILABLE');self.assertEqual(a['confirmation_applications'],0)
    def test_GO_before_labels(self):
        with self.assertRaises(PermissionError):P.calibrate(None,None)

class GroupedSplit(unittest.TestCase):
    def frames(self):
        result=[]
        for group,n in enumerate([48,27,34,12,66,25,23]):
            for i in range(n):
                result.append(dict(id='%s_%s'%(group,i),split='validation',source='S',source_asset='one_shared_asset',
                    generation=dict(_src_root='merged',_src_shard='run%s'%group,attempt_seed=1000*group+i),generation_attempt_key='attempt%s_%s'%(group,i)))
        return result
    def test_hash_prefix_balance_and_no_run_overlap(self):
        from .coverage import split_groups
        rr=self.frames();out=split_groups(rr,group_mode='generation_run',group_semantics_evidence={})
        a=out['assignments'];sides={}
        for r in a:sides.setdefault(r['group'],set()).add(r['subset'])
        self.assertTrue(all(len(v)==1 for v in sides.values()));self.assertEqual(out['frames'],235)
        sizes={k:sum(x['group']==k for x in a)for k in sides};ordered=out['hash_ordered_groups'];candidates=[]
        for cutoff in range(1,len(ordered)):
            n=sum(sizes[k]for k in ordered[:cutoff]);candidates.append((abs(3*n-470),cutoff))
        self.assertEqual(out['selected_prefix_cutoff'],min(candidates)[1]);self.assertEqual(out['labels_read_for_split'],0)
        reverse=split_groups(list(reversed(rr)),group_mode='generation_run',group_semantics_evidence={})
        self.assertEqual({r['id']:r['subset']for r in a},{r['id']:r['subset']for r in reverse['assignments']})
    def test_one_run_and_uncertain_source_disable_adoption(self):
        from .coverage import split_groups
        rr=self.frames()
        for r in rr:r['generation']['_src_shard']='one_run'
        self.assertEqual(split_groups(rr,group_mode='generation_run',group_semantics_evidence={})['status'],'INTERNAL_POLICY_CHECK_UNAVAILABLE')
        rr=self.frames();rr[0]['generation']['_src_root']=None
        self.assertEqual(split_groups(rr,group_mode='generation_run',group_semantics_evidence={})['status'],'INTERNAL_POLICY_CHECK_UNAVAILABLE')
    def test_attempt_identity_needs_semantics_evidence(self):
        from .coverage import split_groups
        with self.assertRaises(ValueError):split_groups(self.frames(),group_mode='generation_attempt',group_semantics_evidence={})

if __name__=='__main__':unittest.main()
