"""Synthetic chat categories never become coordinate references or GUI saves."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'review'))
from serve import ValidationError
from visibility_only_status import (DECLARATIONS_FILENAME, source_corner_sha256,
                                    visibility_only_status)
from save_visibility_locally import validate_document


class VisibilityOnlyStatusTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bindings = dict(manifest_sha256='synthetic-manifest', plan_sha256='synthetic-plan',
                             corner_contract_sha256='synthetic-contract',
                             corner_definition_version='synthetic-corners')
        self.ids = [f'synthetic:{i}' for i in range(12)]
        frames = {fid: dict(frame_id=fid, image_sha256=f'synthetic-image-{i}', width=640,
                           height=480, repeat_review=False) for i, fid in enumerate(self.ids)}
        self.ctx = SimpleNamespace(bindings=self.bindings, frames=frames,
                                   store_path=self.root/'annotations_in_progress.json',
                                   record_for=lambda fid, review_pass: None)
        self.records = {}
        drafts = {}
        for j, fid in enumerate(self.ids):
            corners = []
            for i in range(8):
                corners.append(dict(id=i, visibility='direct_visible' if i < 6 else 'not_direct_visible',
                    x=float(20+i) if i < 6 else None, y=30. if i < 6 else None,
                    definition_confirmed=i < 6, external_occlusion=False,
                    self_occlusion=i >= 6, out_of_frame=False, definition_uncertain=False,
                    reason='SYNTHETIC ONLY'))
            if j < 5:
                corners[3 if j == 0 else 5].update(visibility=None, x=None, y=None,
                                                 definition_confirmed=False)
            record = dict(frame_id=fid, review_pass='primary', corners=corners,
                          status='draft', reviewer=None, interaction_log=[])
            self.records[fid] = record
            drafts[fid+'|primary'] = dict(frame_id=fid, review_pass='primary',
                                          base_revision=None, record=record)
        self.native = SimpleNamespace(ctx=self.ctx, record=None, exclusions=set(),
            visibility_progress_path=self.root/'NATIVE_VISIBILITY_PROGRESS.json',
            recovery=dict(schema='lifter_native_viewer_recovery_v1', bindings=self.bindings,
                          evaluation_use=False, drafts=drafts))
        self.parent_path = self.root/'ORIGINAL_SYNTHETIC_12.json'
        self.queue_path = self.root/'REMAINING_SYNTHETIC_5.json'
        parent = dict(schema='lifter_human_review_batch_v1', source_kind='partial_human_review_queue',
                      bindings=self.bindings, frozen_plan_modified=False, frame_ids=self.ids)
        self.parent_path.write_text(json.dumps(parent))
        queue = dict(parent, frame_ids=self.ids[:5],
                     task_scope='remaining_missing_corners_of_existing_batch',
                     parent_batch_file=str(self.parent_path),
                     parent_batch_sha256=self.sha(self.parent_path),
                     focus_corner_ids={fid:[3 if i == 0 else 5] for i,fid in enumerate(self.ids[:5])})
        self.queue_path.write_text(json.dumps(queue))
        self.overlay = dict(schema='lifter_user_visibility_declarations_v1',
            source_kind='human_chat_visibility_declaration', bindings=self.bindings,
            source_message='SYNTHETIC USER: all five remaining corners are visible',
            source_queue_file=str(self.queue_path), source_queue_sha256=self.sha(self.queue_path),
            evaluation_use=False, human_reference_review_complete=False,
            previous_prediction_exposure=None, declarations=[])
        for i,fid in enumerate(self.ids[:5]):
            index = 3 if i == 0 else 5
            self.overlay['declarations'].append(dict(frame_id=fid, review_pass='primary',
                corner_id=index, visibility='direct_visible', x=None, y=None,
                coordinate_source='none', source_corner_sha256=source_corner_sha256(
                    self.records[fid]['corners'][index]), human_visibility_declared=True,
                source_kind='human_chat_visibility_declaration'))
        self.overlay_path = self.root/DECLARATIONS_FILENAME
        self.write_overlay()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_overlay(self):
        self.overlay_path.write_text(json.dumps(self.overlay))

    def status(self, fid=None, **kwargs):
        return visibility_only_status(self.native, fid or self.ids[0], 'primary', **kwargs)

    def test_exact_96_categories_keep_67_coordinates_and_24_hidden_unchanged(self):
        before = copy.deepcopy(self.native.recovery)
        files_before = {p:p.read_bytes() for p in self.root.iterdir()}
        statuses = [self.status(fid) for fid in self.ids]
        self.assertTrue(all(s['complete'] for s in statuses))
        self.assertEqual(sum(s['confirmed_category_count'] for s in statuses), 96)
        self.assertEqual(sum(s['manual_coordinate_count'] for s in statuses), 67)
        self.assertEqual(sum(len(s['declared_point_ids']) for s in statuses), 5)
        self.assertEqual(sum(c['self_occlusion'] for s in statuses for c in s['corners']), 24)
        self.assertEqual(sum(s['raw_document_complete'] for s in statuses), 7)
        self.assertEqual(self.native.recovery, before)
        self.assertFalse(self.native.visibility_progress_path.exists())
        self.assertFalse(self.ctx.store_path.exists())
        self.assertEqual({p:p.read_bytes() for p in self.root.iterdir()}, files_before)
        self.assertTrue(all(not s['evaluation_use'] and not s['human_reference_review_complete']
                            for s in statuses))

    def test_chat_corner_has_no_coordinates_and_cannot_pass_strict_local_save(self):
        status = self.status()
        corner = status['corners'][3]
        self.assertEqual(corner['visibility'], 'direct_visible')
        self.assertIsNone(corner['x'])
        self.assertIsNone(corner['y'])
        self.assertFalse(corner['definition_confirmed'])
        record = dict(self.records[self.ids[0]], corners=status['corners'])
        doc = dict(schema='lifter_native_visibility_save_v1',
                   source_kind='actual_human_native_save', evaluation_use=False,
                   provenance_status='UNVERIFIED', previous_prediction_exposure=None,
                   bindings=self.bindings, frame_id=self.ids[0], review_pass='primary',
                   image_sha256=self.ctx.frames[self.ids[0]]['image_sha256'], record=record)
        with self.assertRaises(ValidationError):
            validate_document(self.native, doc)

    def test_stale_corner_hash_never_marks_complete(self):
        self.records[self.ids[0]]['corners'][3]['reason'] = 'Later user edit'
        status = self.status()
        self.assertFalse(status['complete'])
        self.assertEqual(status['declared_point_ids'], [])
        self.assertEqual(status['confirmed_category_count'], 7)

    def test_later_actual_point_or_hidden_state_has_priority(self):
        corner = self.records[self.ids[0]]['corners'][3]
        corner.update(visibility='direct_visible', x=35., y=45., definition_confirmed=True)
        status = self.status()
        self.assertTrue(status['complete'])
        self.assertEqual(status['manual_coordinate_count'], 6)
        self.assertEqual(status['declared_point_ids'], [])
        self.assertEqual(status['corners'][3]['x'], 35.)
        corner.update(visibility='not_direct_visible', x=None, y=None,
                      definition_confirmed=False, external_occlusion=True)
        status = self.status()
        self.assertTrue(status['complete'])
        self.assertEqual(status['declared_point_ids'], [])
        self.assertEqual(status['corners'][3]['visibility'], 'not_direct_visible')

    def test_missing_invalid_or_revision_stale_record_cannot_be_completed(self):
        del self.native.recovery['drafts'][self.ids[0]+'|primary']
        self.assertFalse(self.status()['complete'])
        self.assertEqual(self.status()['corners'], [])
        self.assertFalse(self.status(current_record={'frame_id':self.ids[0]})['complete'])
        self.native.recovery['drafts'][self.ids[0]+'|primary'] = dict(
            frame_id=self.ids[0], review_pass='primary', base_revision=999,
            record=self.records[self.ids[0]])
        self.assertFalse(self.status()['complete'])

    def test_wrong_binding_duplicate_coordinates_and_fake_approval_rejected(self):
        original = copy.deepcopy(self.overlay)
        mutations = [lambda d:d['bindings'].update(manifest_sha256='wrong'),
                     lambda d:d['declarations'].append(copy.deepcopy(d['declarations'][0])),
                     lambda d:d['declarations'][0].update(x=20.),
                     lambda d:d.update(human_reference_review_complete=True),
                     lambda d:d.update(evaluation_use=True),
                     lambda d:d['declarations'][0].update(corner_id=6),
                     lambda d:d['declarations'][0].update(frame_id='unplanned'),
                     lambda d:d.update(previous_prediction_exposure=False)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.overlay = copy.deepcopy(original)
                mutate(self.overlay)
                self.write_overlay()
                with self.assertRaises(ValidationError):
                    self.status()

    def test_changed_queue_or_parent_hash_rejected(self):
        queue_before = self.queue_path.read_bytes()
        self.queue_path.write_bytes(queue_before+b'\n')
        with self.assertRaises(ValidationError):
            self.status()
        self.queue_path.write_bytes(queue_before)
        self.parent_path.write_bytes(self.parent_path.read_bytes()+b'\n')
        with self.assertRaises(ValidationError):
            self.status()

    def test_without_chat_overlay_missing_stays_missing(self):
        self.overlay_path.unlink()
        self.assertFalse(self.status()['complete'])
        self.assertEqual(self.status()['manual_coordinate_count'], 5)
        self.assertTrue(self.status(self.ids[11])['complete'])

    def test_valid_hash_checked_S_fallback_is_read_only(self):
        fid = self.ids[11]
        del self.native.recovery['drafts'][fid+'|primary']
        saved_path = self.root/'SYNTHETIC.VISIBILITY_SAVED.json'
        document = dict(schema='lifter_native_visibility_save_v1',
                        source_kind='actual_human_native_save', evaluation_use=False,
                        provenance_status='UNVERIFIED', previous_prediction_exposure=None,
                        bindings=self.bindings, frame_id=fid, review_pass='primary',
                        image_sha256=self.ctx.frames[fid]['image_sha256'],
                        base_revision=None, record=self.records[fid])
        saved_path.write_text(json.dumps(document))
        progress = dict(schema='lifter_native_visibility_progress_v1', bindings=self.bindings,
                        records={fid+'|primary':dict(saved_file=str(saved_path),
                            file_sha256=self.sha(saved_path))})
        self.native.visibility_progress_path.write_text(json.dumps(progress))
        originals = {p:p.read_bytes() for p in self.root.iterdir()}
        self.assertTrue(self.status(fid)['complete'])
        self.assertEqual(self.status(fid)['manual_coordinate_count'], 6)
        self.assertEqual({p:p.read_bytes() for p in self.root.iterdir()}, originals)
        self.native.recovery['drafts'][fid+'|primary'] = dict(frame_id=fid,
            review_pass='primary', base_revision=999, record=self.records[fid])
        self.assertFalse(self.status(fid)['complete'])
        del self.native.recovery['drafts'][fid+'|primary']
        saved_path.write_bytes(saved_path.read_bytes()+b'\n')
        with self.assertRaises(ValidationError):
            self.status(fid)


if __name__ == '__main__':
    unittest.main()
