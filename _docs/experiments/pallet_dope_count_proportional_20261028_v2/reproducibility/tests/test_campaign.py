"""Crash-boundary/integrity checks with manufactured bytes in this experiment."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'code/campaign.py'
spec = importlib.util.spec_from_file_location('proportional_campaign_tests', PATH)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class CampaignRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.original = C.ROOT
        self.tmp = tempfile.TemporaryDirectory(dir=self.original / 'tests')
        C.ROOT = Path(self.tmp.name)
        (C.ROOT / 'state').mkdir()
        (C.ROOT / 'logs').mkdir()
        (C.ROOT / 'preflight').mkdir()
        self.cp = C.ROOT / 'R2.pth'
        self.cp.write_bytes(b'manufactured_final_state')
        self.active = C.ROOT / 'state/ACTIVE_RESUME.pth'
        self.active.write_bytes(b'manufactured_resume_state')
        self.receipt = dict(complete=True, checkpoint_sha256=C.sha(self.cp),
            active_resume_sha256_at_commit=C.sha(self.active), stats=dict(
                updates_completed=828, fingerprint='same',
                run_identity=dict(output_checkpoint=str(self.cp), fingerprint='same')))

    def tearDown(self):
        C.ROOT = self.original
        self.tmp.cleanup()

    def test_receipt_before_cleanup_is_recoverable(self):
        C.cleanup_committed_resume(self.receipt, self.cp, preserve_unrelated=True)
        self.assertFalse(self.active.exists())
        self.assertEqual(self.cp.read_bytes(), b'manufactured_final_state')

    def test_earlier_completed_run_preserves_later_partial_state(self):
        self.active.write_bytes(b'different_later_arm_partial_state')
        C.cleanup_committed_resume(self.receipt, self.cp, preserve_unrelated=True)
        self.assertEqual(self.active.read_bytes(), b'different_later_arm_partial_state')

    def test_changed_final_checkpoint_cannot_authorize_deletion(self):
        self.cp.write_bytes(b'changed_final_state')
        with self.assertRaises(RuntimeError):
            C.cleanup_committed_resume(self.receipt, self.cp)
        self.assertTrue(self.active.exists())

    def test_changed_active_state_is_preserved_on_strict_cleanup(self):
        self.active.write_bytes(b'changed_active_state')
        with self.assertRaises(RuntimeError):
            C.cleanup_committed_resume(self.receipt, self.cp)
        self.assertTrue(self.active.exists())

    def test_symlink_cannot_be_removed_as_active_state(self):
        self.active.unlink()
        self.active.symlink_to(self.cp)
        with self.assertRaises(RuntimeError):
            C.cleanup_committed_resume(self.receipt, self.cp)
        self.assertTrue(self.cp.exists())

    def test_changed_frozen_source_stops_before_runtime(self):
        f = C.ROOT / 'frozen.py'
        f.write_text('original')
        go = dict(passed=True, training_authorized=True, frozen_files={str(f): C.sha(f)})
        (C.ROOT / 'preflight/TRAINING_GO.json').write_text(json.dumps(go))
        C.verify_go()
        f.write_text('changed')
        with self.assertRaises(RuntimeError):
            C.verify_go()

    def test_failed_child_is_logged_and_not_automatically_retried(self):
        with self.assertRaises(RuntimeError):
            C.logged_command([sys.executable, '-B', '-c', 'raise SystemExit(7)'], 'fake_failure')
        rows = [json.loads(x) for x in (C.ROOT / 'logs/COMMANDS.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['exit_code'], 7)
        self.assertEqual(rows[0]['attempt'], 1)


if __name__ == '__main__':
    unittest.main()
