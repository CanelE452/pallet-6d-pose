from __future__ import annotations
from copy import deepcopy
import os,tempfile,unittest
from pathlib import Path
import numpy as np
from .independent_checks import independent_actions,independent_inputs,independent_choose
from .prediction_guard import PredictionGuard


def pose(z=2.):return dict(available=True,R_physical=np.eye(3).tolist(),centroid=[0.,0.,z])
def item(name,z=2.,eligible=True,aliases=()):return dict(candidate_id=name,actual_pose=pose(z),eligible=eligible,features=[0.]*31+[1.]*31,source_baseline_aliases=list(aliases))
def frame(cs=(),fallback=None,e0=None):return dict(candidates=list(cs),fallback=fallback or{'available':False},e0_regenerated=e0 or{'available':False},object_scale_m=1.)
N=dict(mean=[0.]*31,scale=[1.]*31,valid_train_counts=[1]*31)


class IndependentContractTests(unittest.TestCase):
    def test_no_action_and_no_virtual_none(self):
        x,a=independent_inputs(frame(),N);self.assertEqual(x.shape,(0,188));self.assertEqual(a,[]);self.assertIsNone(independent_choose([],a))
    def test_N0_absent_and_physical_tie_order(self):
        x,a=independent_inputs(frame([item('z',3.),item('a',2.)]),N)
        self.assertEqual([c['candidate_id']for c in a],['a','z']);self.assertEqual(independent_choose([1.,1.],a),0);self.assertEqual(x[:,-1].tolist(),[0.,0.])
    def test_N0_E0_duplicate_alias(self):
        r=frame([item('first'),item('second',aliases=['E0_REGENERATED'])],pose(),pose())
        a=independent_actions(r);self.assertEqual(len(a),1);self.assertTrue(a[0]['is_N0']);self.assertEqual(set(a[0]['source_baseline_aliases']),{'N0','E0','E0_REGENERATED'})
        self.assertEqual(len(independent_actions(r,True)),1)
    def test_available_N0_restores_ineligible(self):
        r=frame([item('n',eligible=False),item('discard',4.,eligible=False)],pose())
        a=independent_actions(r);self.assertEqual(len(a),1);self.assertEqual(a[0]['candidate_id'],'n')
    def test_missing_baseline_features_explicit_invalid(self):
        x,a=independent_inputs(frame([item('other',3.)],pose()),N)
        self.assertEqual(a[0]['features'],[0.]*62);self.assertEqual(x[0,:62].tolist(),[0.]*62)
    def test_no_symmetry_dedup(self):
        c=item('half_turn');c['actual_pose']['R_physical']=np.diag([-1.,1.,-1.]).tolist()
        self.assertEqual(len(independent_actions(frame([item('identity'),c]))),2)
    def test_missing_scalar_and_unseen_training_validity(self):
        c=item('x');c['features'][0]=100.;c['features'][31]=0.;c['features'][1]=500.
        n=deepcopy(N);n['valid_train_counts'][1]=0
        x,a=independent_inputs(frame([c]),n);self.assertEqual(float(x[0,0]),0.);self.assertEqual(float(x[0,1]),0.);self.assertEqual(float(x[0,32]),1.)
    def test_renamed_reordered_actions_preserve_pose_and_inputs(self):
        r=frame([item('a',4.),item('b',2.)],pose(3.))
        x,a=independent_inputs(r,N);r['candidates'].reverse()
        for i,c in enumerate(r['candidates']):c['candidate_id']='renamed_'+str(i)
        y,b=independent_inputs(r,N);self.assertTrue(np.array_equal(x,y));self.assertEqual([c['actual_pose']for c in a],[c['actual_pose']for c in b])
    def test_padding_not_in_pool(self):
        c=item('in',3.);bad=item('out',8.,False);bad['features']=[100.]*31+[1.]*31
        x,_=independent_inputs(frame([c,bad]),N);self.assertEqual(len(x),1);self.assertTrue(np.array_equal(x[0,:62],x[0,62:124]))
    def test_guard_blocks_arbitrary_name_alias_and_subprocess(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);secret=root/'innocent.bin';secret.write_text('reference fixture');ok=root/'allowed.json';ok.write_text('{}')
            linked=root/'hardlink.bin';os.link(secret,linked);sym=root/'symlink.bin';sym.symlink_to(secret)
            out=root/'new';out.mkdir()
            guard=PredictionGuard(allowed_files=[ok],allowed_read_roots=[root],denied_files=[secret],output_root=out)
            with guard:
                self.assertEqual(ok.read_text(),'{}')
                for p in(secret,linked,sym):
                    with self.assertRaises(PermissionError):p.read_text()
                with self.assertRaises(PermissionError):os.system('true')
                (out/'prediction.json').write_text('{}')
                with self.assertRaises(PermissionError):ok.write_text('mutation')
            self.assertEqual(len(guard.blocked),5)


if __name__=='__main__':unittest.main()
