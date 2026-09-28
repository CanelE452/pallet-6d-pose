"""Fail-closed aggregate-audit helpers, with no GPU or training side effects."""
import copy
import unittest
from . import final_audit as A


class FinalAuditContracts(unittest.TestCase):
    def test_private_array_scan_does_not_publish_coordinates(self):
        data = {'path': 'private/coordinates.json', 'coordinates': 'sha256 binding only',
                'metric': {'errors': [1., 2.], 'PCK': .5}, 'nested': [{'K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}
        self.assertEqual(A.private_array_paths(data), ['$.nested[0].K'])
        self.assertEqual(A.private_array_paths({'xy': [1, 2], 'GT_DEPENDENT': True}), ['$.xy'])
        self.assertEqual(A.private_array_paths({'manual_target': [[1, 2]], 'gt': False}), ['$.manual_target'])

    def test_hash_bindings_are_recursively_found_not_fabricated(self):
        binding = {'path': 'one', 'sha256': 'abc', 'bytes': 2}
        self.assertEqual(list(A.bindings({'a': [binding], 'b': {'path': 'unbound'}})), [binding])

    def test_pending_limited_and_failed_never_become_pass(self):
        self.assertEqual(A.overall({'a': {'status': 'PASS'}}), 'PASS')
        self.assertEqual(A.overall({'a': {'status': 'PENDING'}}), 'PENDING')
        self.assertEqual(A.overall({'a': {'status': 'PENDING'}, 'b': {'status': 'FAIL'}}), 'FAIL')
        self.assertEqual(A.overall({'a': {'status': 'NOT_RUN'}}), 'PASS_WITH_EXPLICIT_LIMITATIONS')

    def test_local_figure_links_skip_external_and_resolve_spaces(self):
        text = '![a](figures/a.png) [b](../figures/b.svg "title") ![c](https://example.org/c.png) ![d](<figures/my plot.png>)'
        self.assertEqual(list(A.markdown_image_targets(text)), ['figures/a.png', '../figures/b.svg', 'figures/my plot.png'])

    def test_resource_caps_and_event_totals_fail_closed(self):
        ledger = dict(caps=dict(cycles=3, fits=12, optimizer_updates=7680, per_fit_updates=640, gpu_seconds=21600, wall_seconds=36000),
                      events=[dict(gpu_seconds=10., fits=1, optimizer_updates=320)],
                      totals=dict(gpu_seconds=10., fits=1, optimizer_updates=320, elapsed_wall_seconds=30.))
        self.assertTrue(A.cap_checks(ledger, [dict(optimizer_steps=320)])['ledger_caught_up'])
        self.assertFalse(A.cap_checks(ledger, [dict(optimizer_steps=320)]*2)['ledger_caught_up'])
        bad = copy.deepcopy(ledger); bad['totals']['fits'] = 2
        with self.assertRaises(AssertionError): A.cap_checks(bad)
        bad = copy.deepcopy(ledger); bad['caps']['fits'] = 13
        with self.assertRaises(AssertionError): A.cap_checks(bad)
        bad = copy.deepcopy(ledger); bad['events'][0]['gpu_seconds'] = -1
        with self.assertRaises(AssertionError): A.cap_checks(bad)
        with self.assertRaises(AssertionError): A.cap_checks(ledger, [dict(optimizer_steps=641)])


if __name__ == '__main__':
    unittest.main()
