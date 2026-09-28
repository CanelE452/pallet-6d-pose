import copy
import inspect
import unittest
from unittest.mock import patch

from . import eval_student as E
from . import metric_baseline as M


class PairedStudentEvaluationContracts(unittest.TestCase):
    def test_cycle_paths_cannot_escape_v2_namespace(self):
        for cycle in ('A_INPUT_OCCLUSION','BASELINE_REPEAT'):
            paths=E.paths(cycle,'PLASTIC',43)
            self.assertTrue(paths['raw'].is_relative_to(E.C.RAW))
            self.assertTrue(paths['result'].is_relative_to(E.C.DOC))
            self.assertEqual(paths['result'].name,'RESULTS_PLASTIC_S43.json')
        for name in ('../old','a/b','/tmp/elsewhere',''):
            with self.assertRaises(AssertionError):E.paths(name,'PLASTIC',42)

    def test_inference_has_no_scoring_call_or_reference_reader(self):
        source=inspect.getsource(E.infer)
        for token in ('C.P.TRUTH','C.P.FINAL','Pose.metadata','score_frame','verified66('):
            self.assertNotIn(token,source)
        self.assertIn('for target in TARGETS',source)
        self.assertIn('both_targets_locked=True',source)

    def test_detector_parity_rechecks_resumed_native_outputs(self):
        records,predictions,_=E.E.metadata('PLASTIC');record=records[0];fid=record['id']
        reference={fid:predictions['R0'][fid]};values=copy.deepcopy(reference)
        E.assert_native_parity([record],reference,values)
        values[fid]['candidates'][0]['box_xyxy'][0]+=1
        with self.assertRaises(AssertionError):E.assert_native_parity([record],reference,values)

    def test_frozen_baseline_and_group_helpers_match_primary99(self):
        rows,_,_=E.E.metadata('PLASTIC');old=E.C.P.RAW
        arm='REF_LR5';frames={'REF':E.C.read(old/'FRAME_METRICS.json')[arm]}
        fixed={'REF':E.C.read(old/'FIXED_ID_METRICS.json')[arm]}
        pose={'REF':M.read(M.RAW/'FRAME_METRICS_PRIVATE.json')['PLASTIC']['OLD_REF']}
        groups,result=E.group_summaries(rows,'PLASTIC',frames,fixed,pose)
        baseline=M.read(M.DOC/'BASELINE_POSE_RESULTS.json')['materials']['PLASTIC']['groups'][M.PRIMARY]['OLD_REF']
        self.assertEqual(len(groups[M.PRIMARY]),99)
        self.assertEqual(result[M.PRIMARY]['REF']['conditional'],baseline['conditional'])
        self.assertEqual(result['ALL']['REF']['twoD']['corners'],985)
        self.assertEqual(result['ALL']['REF']['fixed_ID']['corners'],985)

    def test_score_resume_verifies_lock_before_existing_result(self):
        source=inspect.getsource(E.score)
        self.assertLess(source.index('inference_binding_checks(p)'),source.index("p['result'].exists()"))
        self.assertLess(source.index('SCORING_START_'),source.index('truth=C.read(C.P.TRUTH)'))
        self.assertNotIn('C.resource(',source)
        self.assertNotIn('C.state(',source)


if __name__=='__main__':unittest.main()
