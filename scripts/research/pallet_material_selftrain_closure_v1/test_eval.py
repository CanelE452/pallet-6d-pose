"""CPU-only contract tests using invented coordinates, never evaluation labels."""
import ast
import copy
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from .infer_eval import validate_metadata, image_order_hash, assert_detector_parity, precision_contract, set_precision
from .score_eval import groups, score_frame, summary


def metadata():
    return [dict(id='a:1', recording='REC_A', severity='CLEAN', image={'sha256': 'a'},
                 K=np.eye(3).tolist(), xyz=[.8, .14, .59], hw=[480, 640]),
            dict(id='b:2', recording='REC_B', severity='MODERATE_OCCLUSION', image={'sha256': 'b'},
                 K=np.eye(3).tolist(), xyz=[.8, .14, .59], hw=[480, 640])]


def prediction():
    return dict(selected_index=0, candidates=[dict(candidate_index=0, score=.9, box_xyxy=[0, 0, 100, 100],
                                                  keypoints_xy=[[float(i), float(i)] for i in range(9)], keypoints_conf=[.9] * 9)])


class TestWoodEvaluation(unittest.TestCase):
    def test_teacher_precision_does_not_inherit_student_tf32(self):
        fake = SimpleNamespace(backends=SimpleNamespace(
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
        set_precision('TEACHER', fake)
        self.assertFalse(fake.backends.cudnn.allow_tf32)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake.backends.cudnn.benchmark)
        self.assertFalse(precision_contract('TEACHER')['half'])
        set_precision('WOOD_RAW_LR5', fake)
        self.assertTrue(fake.backends.cudnn.allow_tf32)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)

    def test_metadata_has_no_references(self):
        rows = metadata()
        self.assertEqual(validate_metadata(rows), ['a:1', 'b:2'])
        rows[0]['gt'] = [[0, 0]] * 9
        with self.assertRaises(AssertionError): validate_metadata(rows)

    def test_metadata_rejects_duplicate_id(self):
        rows = metadata()
        rows[1]['id'] = rows[0]['id']
        with self.assertRaises(AssertionError): validate_metadata(rows)

    def test_order_hash_includes_image_identity_and_order(self):
        rows = metadata()
        self.assertNotEqual(image_order_hash(rows), image_order_hash(rows[::-1]))
        changed = copy.deepcopy(rows); changed[0]['image']['sha256'] = 'other'
        self.assertNotEqual(image_order_hash(rows), image_order_hash(changed))

    def test_empty_severity_not_fabricated(self):
        g = groups(metadata())
        self.assertNotIn('SEVERE', g)
        self.assertEqual(g['CLEAN'], ['a:1'])
        self.assertEqual(g['MODERATE'], ['b:2'])
        self.assertEqual(g['SESSION_a'], ['a:1'])

    def test_student_coordinates_may_change_detector_cannot(self):
        p = prediction(); q = copy.deepcopy(p)
        q['candidates'][0]['keypoints_xy'][0][0] += 30
        assert_detector_parity(p, q)
        q['candidates'][0]['score'] = .8
        with self.assertRaises(AssertionError): assert_detector_parity(p, q)

    def test_teacher_confidence_and_center_are_preserved(self):
        p = prediction(); q = copy.deepcopy(p)
        q['candidates'][0]['keypoints_xy'][0][0] += 30
        assert_detector_parity(p, q, teacher=True)
        q['candidates'][0]['keypoints_conf'][0] = .8
        with self.assertRaises(AssertionError): assert_detector_parity(p, q, teacher=True)
        q = copy.deepcopy(p); q['candidates'][0]['keypoints_xy'][8][0] += 1
        with self.assertRaises(AssertionError): assert_detector_parity(p, q, teacher=True)

    def test_missing_detection_retains_full_corner_denominator(self):
        p = prediction()
        truth = dict(gt=p['candidates'][0]['keypoints_xy'], valid=[True] * 9,
                     permutations=[list(range(9))], box=[0, 0, 100, 100], hw=[480, 640])
        row = score_frame('x', {'selected_index': None, 'candidates': []}, truth)
        result = summary([row])
        self.assertEqual(result['corners'], 8)
        self.assertEqual(result['correct']['10'], 0)
        self.assertEqual(result['full_penalty_median_px'], 800)

    def test_mismatch_not_replaced_with_other_candidate(self):
        p = prediction()
        truth = dict(gt=p['candidates'][0]['keypoints_xy'], valid=[True] * 9,
                     permutations=[list(range(9))], box=[200, 200, 300, 300], hw=[480, 640])
        alternate = copy.deepcopy(p['candidates'][0]); alternate.update(candidate_index=1, score=.8, box_xyxy=truth['box'])
        p['candidates'].append(alternate)
        row = score_frame('x', p, truth)
        self.assertFalse(row['matched'])
        self.assertEqual(row['errors'], [800] * 8)

    def test_inference_never_calls_reference_metadata_or_load(self):
        path = Path(__file__).with_name('infer_eval.py')
        tree = ast.parse(path.read_text())
        forbidden = {'D.Pose.metadata', 'D.load', 'D.metric', 'D.Pose.metric', 'score_frame'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertNotIn(ast.unparse(node.func), forbidden)
        self.assertNotIn('C.P.TRUTH', path.read_text())
        self.assertNotIn('annotation_path', path.read_text())


if __name__ == '__main__':
    unittest.main()
