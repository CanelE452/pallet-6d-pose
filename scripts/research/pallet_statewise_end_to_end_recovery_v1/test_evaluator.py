"""Postseal safety and oracle-stage semantics on artificial poses only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import evaluator as E


REF=dict(R=np.eye(3).tolist(),t=[0.,0.,2.],order=2)


def candidate(name,angle=0.,translation=0.):
    a=np.deg2rad(angle)
    R=np.array([[np.cos(a),0.,np.sin(a)],[0.,1.,0.],[-np.sin(a),0.,np.cos(a)]])
    return dict(candidate_id=name,R_physical=R.tolist(),centroid=[translation,0.,2.])


class EvaluatorTests(unittest.TestCase):
    def test_no_GO_cannot_open_references_or_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            missing=Path(folder)/'nonexistent'
            with self.assertRaises(PermissionError):
                E.run(missing,baseline_folder=missing,baseline_comparison_folder=missing)
            self.assertFalse(missing.exists())

    def test_candidate_oracle_must_jointly_satisfy_both_metrics(self):
        # T-good/R-bad and T-bad/R-good cannot be joined into an oracle success.
        bank=[candidate('good_translation',angle=20.),candidate('good_rotation',translation=.2)]
        self.assertFalse(E._bank_success(bank,REF)['has_success'])
        bank.append(candidate('joint_good',angle=2.,translation=.02))
        self.assertEqual(E._bank_success(bank,REF)['success_ids'],['joint_good'])

    def test_constraint_filter_loss_is_distinct_from_candidate_absence(self):
        good=candidate('good');bad=candidate('bad',angle=20.)
        row=dict(carried=[good,bad],carried_scores=[
            {'F':dict(scores={'A':1.},feasible={'A':False})},
            {'F':dict(scores={'A':2.},feasible={'A':True})}],
            predictions={'A_FIXED_BANK':dict(actual_pose=E._candidate_pose(bad),selected_candidate_id='bad')})
        value=E.stage_diagnosis(row,REF,['A_FIXED_BANK'])['arms']['A_FIXED_BANK']
        self.assertTrue(value['generated_before_feasibility']['has_success'])
        self.assertFalse(value['after_constraints']['has_success'])
        self.assertEqual(value['stage_loss_class'],'SUCCESS_LOST_BEFORE_ADMISSIBLE_BANK_OR_BY_CONSTRAINTS')

    def test_CTRL_FIXED_never_borrows_budget_candidates(self):
        good=candidate('only_in_budget');bad=candidate('fixed_bad',angle=20.)
        bad['solve']=dict(start_id='N0_LOCAL_START',mode='fixed',feasible=True)
        row=dict(carried=[good],banks={'fixed':[good]},bank_scores={'fixed':[]},attempts=[bad],
            predictions={'CTRL_FIXED':dict(actual_pose=E._candidate_pose(bad),selected_candidate_id='fixed_bad')})
        value=E.stage_diagnosis(row,REF,['CTRL_FIXED'],baseline_poses={'N0':E._candidate_pose(bad)})['arms']['CTRL_FIXED']
        self.assertEqual(value['new_attempt_count'],1)
        self.assertFalse(value['generated_before_feasibility']['has_success'])
        self.assertFalse(value['after_constraints']['has_success'])
        self.assertIn('ARCHIVE_GAP',value['family_intermediate_status'])

    def test_RANSAC_archive_gap_does_not_invent_an_oracle_zero(self):
        row=dict(carried=[],predictions={'CTRL_RANSAC':dict(actual_pose=E._candidate_pose(candidate('selected')))})
        value=E.stage_diagnosis(row,REF,['CTRL_RANSAC'])['arms']['CTRL_RANSAC']
        self.assertEqual(value['status'],'ARCHIVE_GAP')
        self.assertTrue(value['final_success'])
        self.assertNotIn('generated_before_feasibility',value)

    def test_learned_eligibility_pool_has_no_family_stage(self):
        good=candidate('not_eligible');bad=candidate('eligible_bad',angle=20.)
        row=dict(carried=[],selection_banks={'L_HEAD_s1':dict(candidates=[good,bad],eligible_ids=['eligible_bad'])},
            predictions={'L_HEAD_s1':dict(actual_pose=E._candidate_pose(bad),selected_candidate_id='eligible_bad')})
        value=E.stage_diagnosis(row,REF,['L_HEAD_s1'])['arms']['L_HEAD_s1']
        self.assertTrue(value['generated_before_feasibility']['has_success'])
        self.assertFalse(value['after_constraints']['has_success'])
        self.assertEqual(value['stage_loss_class'],'LEARNED_ELIGIBILITY_FILTER_LOSS')
        self.assertEqual(value['family_intermediate_status'],'NOT_APPLICABLE_DIRECT_SOFTMAX_SELECTION')

    def test_learning_na_retains_actual_support_gate_reason(self):
        registry={'arms':[dict(name='L_HEAD_s1',status='REGISTERED',notes='fixed specification')]}
        value=E._unexecuted_arms(registry,[],{'L_HEAD_s1':dict(status='NOT_APPLICABLE',
            reason={'status':'NOT_APPLICABLE_LABEL_SUPPORT_GATE','train_success_frames':12})})
        self.assertEqual(value['L_HEAD_s1']['status'],'NOT_APPLICABLE')
        self.assertEqual(value['L_HEAD_s1']['reason']['train_success_frames'],12)
        with self.assertRaises(ValueError):
            E._unexecuted_arms(registry,[],{'L_HEAD_s1':dict(status='EXECUTED')})


if __name__=='__main__':unittest.main()
