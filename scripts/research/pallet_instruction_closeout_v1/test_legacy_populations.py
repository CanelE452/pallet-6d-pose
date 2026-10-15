"""Accounting checks for the source-only closeout, without model execution."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.pallet_instruction_closeout_v1.legacy_populations import resource_closeout


class ResourceCloseoutTests(unittest.TestCase):
    def fixture(self, folder, *, gpu_bound=40.0):
        resources = {
            'instrumented_GPU_related_stage_elapsed_seconds': 3.0,
            'instrumented_CPU_solver_analysis_stage_elapsed_seconds': 7.0,
            'GPU_whole_process_containing_interval_upper_bound_seconds': gpu_bound,
            'GPU_cap_seconds': 60.0,
            'conservative_CPU_with_hybrid_GPU_jobs_elapsed_seconds': 10.0,
            'CPU_cap_seconds': 120.0,
        }
        (folder / 'EXECUTION_RECEIPT.json').write_text(json.dumps({'resources': resources}))
        rows = [
            {'kind': 'gpu_process', 'elapsed_seconds': 1.0},
            {'kind': 'gpu_process', 'elapsed_seconds': 2.0},
            {'kind': 'cpu_solver_analysis', 'elapsed_seconds': 7.0},
            {'kind': 'duplicate_receipt_metadata', 'elapsed_seconds': 999.0},
        ]
        (folder / 'RESOURCE_LEDGER.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (folder / 'COST_AND_VALIDATION_KO.md').write_text('Saved accounting fixture.\n')

    def test_duplicate_receipt_is_preserved_without_recounting_work(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture(folder)
            result = resource_closeout(folder)
            self.assertEqual(result['recomputed_kind_totals_seconds']['gpu_process'], 3.0)
            self.assertEqual(result['recomputed_kind_totals_seconds']['cpu_solver_analysis'], 7.0)
            self.assertEqual(result['recomputed_kind_totals_seconds']['duplicate_receipt_metadata'], 999.0)
            self.assertTrue(result['duplicate_metadata_is_not_an_executed_stage'])
            self.assertFalse(result['exact_initial_small_IO_startup_teardown_and_full_publication_wall_recoverable'])
            self.assertEqual(result['status'], 'CLOSED_WITH_DOCUMENTED_MEASUREMENT_EXCEPTION')

    def test_cap_overrun_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture(folder, gpu_bound=61.0)
            with self.assertRaisesRegex(ValueError, 'exceeds the declared cap'):
                resource_closeout(folder)

    def test_receipt_ledger_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture(folder)
            path = folder / 'EXECUTION_RECEIPT.json'
            receipt = json.loads(path.read_text())
            receipt['resources']['instrumented_CPU_solver_analysis_stage_elapsed_seconds'] = 8.0
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'ledger/receipt differs'):
                resource_closeout(folder)


if __name__ == '__main__':
    unittest.main()
