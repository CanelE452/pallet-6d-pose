"""Synthetic approval callbacks and temporary evidence; never actual human labels."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_native_review_batch as batchfixtures
import annotate
from approve_existing_clicks import approve_existing_clicks, ACTION, AXES
from serve import ValidationError, utc_now


class ExistingClicksApprovalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = batchfixtures.NativeReviewBatchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.native = self.fixture.native()
        self.native.reviewer = None
        self.native.editor = annotate
        self.state = annotate.State()
        self.state.img = np.zeros((10, 20, 3), np.uint8)
        self.state.img_shape = self.state.img.shape
        self.state.mode = 'click'
        self.state.locked_pose = None
        self.ids = self.fixture.batch_ids
        self.ctx = self.fixture.context
        self.proofs = self.fixture.write_saved(self.ids)
        counts = (6, 6, 6, 5, 5, 6, 6, 5, 6, 5, 5, 6)
        for fid, count in zip(self.ids, counts):
            path = self.fixture.root / 'native/primary' / (fid.replace(':', '_') + '.json')
            self.native.path_map[str(path)] = (fid, 'primary')
            self.native.load(self.state, path)
            record = self.native.record
            record['object'].update(presence='present', target_identity_confirmed=True,
                target_object_id='SYNTHETIC_' + fid)
            for i in range(count):
                xy = [float(2 + i), float(2 + i / 4)]
                record['corners'][i].update(visibility='direct_visible',
                    definition_confirmed=True, x=xy[0], y=xy[1])
                self.state.kps_2d[i] = xy
                annotate._set_keypoint_state(self.state, i, xy,
                    source='manual_click', visibility=2, reason='visible')
                record['interaction_log'].append(dict(action='manual_raw_click_and_physical_confirmation',
                    performed_at=utc_now(), synthetic_test_only=True))
            record['native_pnp_assistance'] = [dict(source='SYNTHETIC TEST ONLY')]
            if fid == self.ids[-1]:
                for i in (6, 7):
                    record['corners'][i].update(visibility='not_direct_visible',
                        self_occlusion=True, geometry_confirmation=dict(
                            source='human_confirmed_manual_click_geometry', confirmed_at=utc_now(),
                            input_action='explicit_C_or_geometry_button', synthetic_test_only=True))
                record['interaction_log'].append(dict(action='human_confirmed_manual_geometry_proposals',
                    performed_at=utc_now(), synthetic_test_only=True))
            self.native.persist_recovery(self.state)
        self.before = copy.deepcopy(self.native.recovery['drafts'])
        self.proof_bytes = {path: path.read_bytes() for path in self.proofs.values()}
        self.choose = Mock(return_value=copy.deepcopy(self.fixture.fixture.reviewer))

    def assert_sources_unchanged(self):
        self.fixture.assert_originals_unchanged()
        for path, raw in self.proof_bytes.items():
            self.assertEqual(path.read_bytes(), raw)

    def alter_draft(self, change):
        recovery = json.loads(self.native.recovery_path.read_text())
        change(recovery['drafts'][self.ids[0] + '|primary'])
        self.native.recovery_path.write_text(json.dumps(recovery))

    def test_real_callback_approval_reuses_67_direct_and_two_c_states_defers_only_27(self):
        confirm = Mock(return_value=True)
        self.assertTrue(approve_existing_clicks(self.native, self.state, confirm, self.choose))
        self.choose.assert_called_once()
        confirm.assert_called_once()
        self.assertIn('67', confirm.call_args.args[1])
        self.assertIn('27', confirm.call_args.args[1])
        records = [self.ctx.record_for(fid, 'primary') for fid in self.ids]
        self.assertEqual(sum(c['visibility'] == 'direct_visible' for r in records for c in r['corners']), 67)
        self.assertEqual(sum(c['visibility'] == 'uncertain' for r in records for c in r['corners']), 27)
        for record in records:
            original = self.before[record['frame_id'] + '|primary']['record']
            for before, after in zip(original['corners'], record['corners']):
                if before['visibility'] is not None:
                    self.assertEqual(after, before)
                else:
                    self.assertEqual(after['visibility'], 'uncertain')
                    self.assertIsNone(after['x']); self.assertIsNone(after['y'])
                    self.assertTrue(all(after[axis] is False for axis in AXES))
                    self.assertIn('판단 보류', after['reason'])
            self.assertEqual(record['interaction_log'][:-1], original['interaction_log'])
            self.assertEqual(record['interaction_log'][-1]['action'], ACTION)
            self.assertFalse(record['independently_repeated'])
            self.assertFalse(record['independent_repeat'])
            self.assertTrue(record['reviewer']['previous_annotation_exposure'])
            self.assertTrue(record['reviewer']['machine_assistance'])
            self.assertEqual(record['source_kind'], 'human_reviewed')
            self.assertEqual(record['review_time']['clock_source'], 'server_wall_and_monotonic')
            self.assertEqual(record['existing_input_approval']['review_time_scope'],
                'post_confirmation_submission_only')
            self.assertFalse(record['existing_input_approval']['review_time_is_original_annotation_time'])
        receipt = json.loads((self.fixture.root / 'EXISTING_CLICKS_APPROVAL_RECEIPT.json').read_text())
        self.assertEqual(receipt['summary'], dict(frame_count=12, direct_points=67,
            confirmed_states=69, pending_states=27))
        self.assertFalse(receipt['full_population_claimed_complete'])
        self.assertEqual(self.native.requested_counts()['primary_completed'], 12)
        self.assertEqual(self.native.requested_counts()['primary_required'], 115)
        self.assert_sources_unchanged()

    def test_profile_cancel_creates_no_review_store_or_labels(self):
        confirm = Mock(return_value=True)
        self.choose.return_value = None
        self.assertFalse(approve_existing_clicks(self.native, self.state, confirm, self.choose))
        self.choose.assert_called_once(); confirm.assert_not_called()
        self.assertFalse(self.ctx.store_path.exists())
        self.assertEqual(self.ctx.export()['records'], [])
        self.assertEqual(self.native.record['corners'], self.before[self.ids[-1] + '|primary']['record']['corners'])
        self.assert_sources_unchanged()

    def test_confirmation_cancel_keeps_unentered_states_and_original_store_absent(self):
        self.assertFalse(approve_existing_clicks(self.native, self.state, Mock(return_value=False), self.choose))
        self.assertFalse(self.ctx.store_path.exists())
        self.assertEqual(self.ctx.export()['records'], [])
        recovery = json.loads(self.native.recovery_path.read_text())
        self.assertEqual(recovery['drafts'], self.before)
        self.assert_sources_unchanged()

    def test_stale_revision_foreign_owner_and_invalid_coordinates_fail_before_submission(self):
        changes = (
            lambda entry: entry.update(base_revision=999),
            lambda entry: entry['record'].update(reviewer=dict(self.fixture.fixture.reviewer, id='ANOTHER_PERSON')),
            lambda entry: entry['record']['corners'][0].update(x=1000.),
        )
        original = self.native.recovery_path.read_bytes()
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.native.recovery_path.write_bytes(original)
                self.alter_draft(change)
                confirm = Mock(return_value=True)
                with self.assertRaises(ValidationError):
                    approve_existing_clicks(self.native, self.state, confirm, self.choose)
                confirm.assert_not_called()
                self.assertEqual(self.ctx.export()['records'], [])
                self.assertFalse(self.ctx.store_path.exists())
        self.assert_sources_unchanged()

    def test_proof_and_recovery_bindings_must_match_before_any_human_submit(self):
        proof = self.proofs[self.ids[0]]
        doc = json.loads(proof.read_text()); doc['image_sha256'] = 'bad'
        proof.write_text(json.dumps(doc))
        confirm = Mock(return_value=True)
        with self.assertRaises(ValidationError):
            approve_existing_clicks(self.native, self.state, confirm, self.choose)
        confirm.assert_not_called()
        self.assertFalse(self.ctx.store_path.exists())
        proof.write_bytes(self.proof_bytes[proof])
        recovery = json.loads(self.native.recovery_path.read_text())
        recovery['bindings']['plan_sha256'] = 'bad'
        self.native.recovery_path.write_text(json.dumps(recovery))
        with self.assertRaises(ValidationError):
            approve_existing_clicks(self.native, self.state, confirm, self.choose)
        self.assertFalse(self.ctx.store_path.exists())
        self.assert_sources_unchanged()

    def test_inputs_changing_during_confirmation_are_rejected_without_partial_submission(self):
        def change_source(title, message):
            proof = self.proofs[self.ids[0]]
            doc = json.loads(proof.read_text()); doc['synthetic_extra_change'] = True
            proof.write_text(json.dumps(doc))
            return True
        with self.assertRaises(ValidationError):
            approve_existing_clicks(self.native, self.state, change_source, self.choose)
        self.assertEqual(self.ctx.export()['records'], [])
        self.assertFalse(self.ctx.store_path.exists())


if __name__ == '__main__':
    unittest.main()
