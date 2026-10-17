"""Small artificial-data contract checks; no campaign outcomes are opened."""
from copy import deepcopy
import subprocess
import sys
import unittest
from unittest.mock import patch

from . import mechanism_diagnosis as M
from . import synthetic_head_evaluation as S


def metric(success,T=1.,R=1.):
    return dict(success=success,translation_cm=T,rotation_deg=R,pose_available=T is not None,metric_error=None)


class PostsealContracts(unittest.TestCase):
    def test_both_numeric_evaluators_reject_before_io(self):
        with patch.object(M,'read',side_effect=AssertionError('IO before GO')):
            with self.assertRaises(PermissionError):M.run('/absent',original_oracle_root='/absent')
        with patch.object(S,'read',side_effect=AssertionError('IO before GO')):
            with self.assertRaises(PermissionError):S.evaluate('/absent')
        # Other fixtures may legitimately have imported torch in the parent
        # interpreter. Check this module's cold-import contract independently.
        script='\n'.join(['import importlib, sys',
            f'importlib.import_module({M.__name__!r})',
            f'importlib.import_module({S.__name__!r})',
            "assert 'torch' not in sys.modules, 'Postseal evaluator import loaded torch'"])
        result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_original_strata_are_exact_aliases_and_unknown_stays_unresolved(self):
        r=dict(final_joint_success=False,original_consumer_generated_bank_complete=True,
               earliest_observed_failure_stage='family_selection')
        self.assertEqual(M.original_group(r),'FAMILY_SELECTION_LOSS')
        r['earliest_observed_failure_stage']='unknown_unrecorded'
        with self.assertRaises(ValueError):M.original_group(r)
        r['earliest_observed_failure_stage']='confirmed_original_consumer_observation_or_proposal_limit'
        r['original_consumer_generated_bank_complete']=False
        with self.assertRaises(ValueError):M.original_group(r)

    def test_archive_gap_is_unknown_instead_of_zero_oracle(self):
        row=dict(id='x',metrics={'A':metric(False)})
        data={'x':dict(carried={'has_success':False},arms={'A':dict(status='ARCHIVE_GAP',final_success=False)})}
        result=M.stage_summary([row],'A',data)
        self.assertEqual(result['generated_before_constraints_oracle'],dict(known_n=0,unknown_n=1,success_n=None))
        self.assertEqual(result['final'],dict(known_n=1,unknown_n=0,success_n=0))

    def test_paired_full_denominator_survives_missing_pose(self):
        pairs=[(True,True),(False,True),(True,False),(False,False)];rows=[]
        for i,(a,b) in enumerate(pairs):
            x=metric(a);y=metric(b)
            if i==3:x=metric(False,None,None);y=metric(False,None,None)
            rows.append(dict(id=str(i),metrics={'old':x,'new':y}))
        result=M.paired_summary(rows,'old','new')
        self.assertEqual(result['denominator'],4)
        self.assertEqual(result['counts'],dict(BOTH_SUCCESS=1,RECOVERY=1,DAMAGE=1,BOTH_FAILURE=1))
        self.assertEqual(result['continuous']['translation_cm']['common_available_n'],3)

    def test_source_C1_and_C2_do_not_share_a_hardcoded_label(self):
        eye=[[1.,0,0],[0,1.,0],[0,0,1.]];halfturn=[[-1.,0,0],[0,1.,0],[0,0,-1.]]
        target=dict(R_fixed=eye,t=[0.,0.,2.],symmetry_order=1)
        pose=dict(available=True,R_physical=halfturn,centroid=[0.,0.,2.])
        before=deepcopy(target)
        self.assertFalse(S.source_metric(pose,target)['success'])
        self.assertEqual(target,before)
        target['symmetry_order']=2
        self.assertTrue(S.source_metric(pose,target)['success'])


if __name__=='__main__':unittest.main()
