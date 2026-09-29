import tempfile
from pathlib import Path
import unittest

from . import audit


class AuditTests(unittest.TestCase):
    def test_status_never_promotes_missing_to_pass(self):
        self.assertEqual(audit.final_status({'a': {'status': 'PASS'}, 'b': {'status': 'NOT_READY'}}), 'NOT_READY')
        self.assertEqual(audit.final_status({'a': {'status': 'FAIL'}, 'b': {'status': 'NOT_READY'}}), 'FAIL')
        self.assertEqual(audit.final_status({'a': {'status': 'PASS'}}), 'PASS')

    def test_binding_traversal(self):
        row = {'path': 'example.json', 'sha256': 'abc'}
        self.assertEqual(list(audit.bindings({'rows': [row], 'path': 'not_a_binding'})), [row])

    def test_numeric_comparison_checks_denominators(self):
        audit.numeric_close({'n': 99, 'value': 1.}, {'n': 99, 'value': 1.+1e-9})
        with self.assertRaises(AssertionError): audit.numeric_close({'n': 99}, {'n': 98})
        with self.assertRaises(AssertionError): audit.numeric_close({'n': 99}, {'n': 99, 'extra': 1})

    def test_private_publication_paths(self):
        name = audit.C.NAME
        safe = f'_docs/experiments/{name}/figures/aggregate.png'
        unsafe = [f'data/pallet/results/{name}/PREDICTIONS.json',
                  f'outputs/{name}/contact_sheet.png',
                  f'_docs/experiments/{name}/DETAILS_PRIVATE.json',
                  f'_docs/experiments/{name}/model.pt']
        self.assertEqual(audit.forbidden_public_paths([safe, *unsafe]), unsafe)

    def test_named_private_arrays_not_hash_metadata(self):
        self.assertEqual(audit.privacy_fields({'coordinates': 'sha256', 'image': {'path': 'x', 'sha256': 'y'}}), [])
        self.assertEqual(audit.privacy_fields({'nested': {'keypoints_xy': [[1, 2]]}}), ['$.nested.keypoints_xy'])

    def test_links_and_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(audit.local_links('[external](https://example.com) [bad](missing.md)', root), ['missing.md'])
            with self.assertRaises(audit.NotReady): audit.need(root/'missing.json')


if __name__ == '__main__':
    unittest.main()
