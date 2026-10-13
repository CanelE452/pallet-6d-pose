"""Checks for the matched family control and branch acceptance accounting."""
import unittest
import tempfile
from pathlib import Path
import numpy as np

from . import common as C
from .joint_replay import _row, original_family_control, population_ids


class ReplayControls(unittest.TestCase):
    def setUp(self):
        self.initial = dict(available=True, R_cf=np.eye(3).tolist(), R_physical=np.eye(3).tolist(),
                            centroid=[0.,0.,4.], cf_extents=[1.1,.11,1.3], selected_hypothesis='ORIGINAL')
        self.K = np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
        self.points = np.vstack([C.project(C.cuboid(*self.initial['cf_extents']), np.eye(3), [0,0,4], self.K),
                                 [320.,240.]])
        corrected = self.points.copy(); corrected[0] += [1.,2.]
        self.result = dict(actual_pose=self.initial, solution=dict(accepted=False, reason='motion_guard'),
            branch_result=dict(accepted=True, selection_reason='alternate_shared_RGB_visible_cost_improved',
                candidates=[dict(original=True, available=True, R_cf=np.eye(3).tolist(),
                    R_physical=np.eye(3).tolist(), centroid=[0.,0.,4.], dimensions=[1.1,.11,1.3],
                    curriculum=dict(accepted=True, reason='accepted'))]),
            self_hidden=[False]*7+[True], appearance={}, corrected_measurements=corrected.tolist(),
            completed_points=corrected.tolist(), projected_points=self.points[:8].tolist(), provenance={}, timing={})

    def test_control_retains_physical_family_and_center(self):
        result = original_family_control(self.result, self.initial, self.points, self.K)
        self.assertEqual(result['actual_pose']['cf_extents'], self.initial['cf_extents'])
        self.assertEqual(result['actual_pose']['R_physical'], self.initial['R_physical'])
        np.testing.assert_array_equal(result['completed_points'][8], self.points[8])
        np.testing.assert_array_equal(result['completed_points'][0], self.result['corrected_measurements'][0])
        np.testing.assert_array_equal(self.points[0], np.asarray(self.result['corrected_measurements'][0])-[1,2])

    def test_branch_flip_counts_even_when_selected_local_update_rejected(self):
        frame = dict(id='synthetic', session='synthetic', points={'N3_SUBPIX':self.points.tolist()})
        row = _row(frame, 'JOINT_WD_BRANCH', self.result)
        self.assertTrue(row['solution']['accepted'])
        self.assertFalse(row['solution']['selected_curriculum_accepted'])
        self.assertFalse(self.result['solution']['accepted'])

    def test_nonsevere_contract_blocks_excluded_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            inputs, split, cohort = [directory/name for name in ('inputs.json','split.json','cohort.json')]
            C.write(inputs, dict(frames=[dict(id='clean'),dict(id='moderate')]))
            C.write(split, dict(real_split=dict(frame_ids=dict(CAL=['clean'],TEST=['moderate']))))
            contract=dict(eligible_frames=2,CAL_frames=1,TEST_frames=1,excluded_severe_ids=['severe'],
                hashes={'PREDICTION_INPUTS.json':C.sha(inputs),'SPLIT.json':C.sha(split)})
            C.write(cohort,contract)
            self.assertEqual(population_ids(inputs,split,'TEST',cohort),{'moderate'})
            C.write(inputs,dict(frames=[dict(id='clean'),dict(id='moderate'),dict(id='severe')]))
            with self.assertRaisesRegex(ValueError,'hash changed'):
                population_ids(inputs,split,'TEST',cohort)
            contract['hashes']['PREDICTION_INPUTS.json']=C.sha(inputs);C.write(cohort,contract)
            with self.assertRaisesRegex(ValueError,'Severe frame'):
                population_ids(inputs,split,'TEST',cohort)


if __name__ == '__main__':
    unittest.main()
