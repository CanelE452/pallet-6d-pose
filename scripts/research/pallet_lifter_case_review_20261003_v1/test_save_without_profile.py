"""Popup-free native saves in temporary synthetic files; no real labels or GUI."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_native_review as fixtures
import open_existing_annotation as adapter
from object_geometry_registry import load_object_geometry_registry


class SaveWithoutProfileTests(unittest.TestCase):
    def setUp(self):
        # Composition avoids discovering the fixture's unrelated native tests.
        self.fixture = fixtures.NativeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.native, self.state = self.fixture.native, self.fixture.state
        self.native.reviewer = None
        self.native.load(self.state, self.fixture.path)
        self.context = self.fixture.fixture.ctx
        self.root = self.fixture.fixture.root
        self.registry = load_object_geometry_registry(
            adapter.ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')
        self.originals = {path: path.read_bytes() for path in (
            self.fixture.fixture.manifest, self.fixture.fixture.plan,
            self.fixture.fixture.contract, self.context.images['test:0'])}

    def complete(self):
        self.fixture.click(4, 3)
        self.fixture.click(6, 4)
        for index, axis in enumerate((
                'external_occlusion', 'self_occlusion', 'out_of_frame',
                'definition_uncertain', 'self_occlusion', 'definition_uncertain'), 2):
            self.native.mark(self.state, index, axis)

    def save(self):
        with patch.object(adapter, 'choose_profile',
                side_effect=AssertionError('S must not open a history popup')):
            return self.fixture.key('s')

    def document(self):
        return json.loads(self.saved_path().read_text())

    def saved_path(self):
        return Path(self.progress()['records']['test:0|primary']['saved_file'])

    def progress(self):
        return json.loads(self.native.visibility_progress_path.read_text())

    def fresh(self, *, revisit_saved=False):
        return adapter.NativeReview(self.context, None, self.root / 'native',
            revisit_saved=revisit_saved)

    def queue_ids(self, native, review_pass='primary'):
        _, contexts = native.contexts('', argparse.Namespace(), self.registry, adapter.ROOT)
        context = contexts.get('review:' + review_pass)
        if context is None:
            return []
        reverse = {str(path): fid for fid, path in self.context.images.items()}
        return [reverse[path] for path in context['frame_paths']]

    def assert_not_evaluator_evidence(self):
        self.assertIsNone(self.native.reviewer)
        self.assertIsNone(self.native.record['reviewer'])
        self.assertFalse(self.native.profile_path.exists())
        self.assertFalse(self.fixture.fixture.store.exists())
        self.assertEqual(self.context.export()['records'], [])
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original, str(path))

    def assert_not_reused(self):
        try:
            completed = self.fresh().saved_visibility_ids('primary')
        except adapter.ValidationError:
            return
        self.assertNotIn('test:0', completed)

    def test_complete_s_saves_actual_coordinates_and_flags_without_profile_popup(self):
        self.complete()
        before = copy.deepcopy(self.native.record['corners'])
        self.assertEqual(self.save(), 'save-next')
        document = self.document()
        self.assertEqual(document['schema'], 'lifter_native_visibility_save_v1')
        self.assertEqual(document['source_kind'], 'actual_human_native_save')
        self.assertIs(document['evaluation_use'], False)
        self.assertEqual(document['bindings'], self.context.bindings)
        self.assertEqual(document['frame_id'], 'test:0')
        self.assertEqual(document['review_pass'], 'primary')
        self.assertEqual(document['image_sha256'], self.context.frames['test:0']['image_sha256'])
        self.assertIsNone(document['previous_prediction_exposure'])
        self.assertIsNone(document['record']['reviewer'])
        self.assertEqual(document['provenance_status'], 'UNVERIFIED')
        self.assertEqual(document['annotation_state'], 'saved_complete')
        self.assertEqual(document['record']['status'], 'draft')
        self.assertEqual(document['record']['corners'], before)
        self.assertEqual(document['record']['corners'][0]['x'], 4.)
        self.assertEqual(document['record']['corners'][1]['y'], 4.)
        self.assertTrue(all(c['x'] is None and c['y'] is None for c in before[2:]))
        self.assertEqual(self.native.saved_visibility_ids('primary'), {'test:0'})
        self.assertEqual(self.native.saved_visibility_ids('repeat'), set())
        entry = self.progress()['records']['test:0|primary']
        self.assertEqual(Path(entry['saved_file']), self.fixture.path.with_name('raw.VISIBILITY_SAVED.json'))
        self.assertEqual(entry['file_sha256'], hashlib.sha256(self.saved_path().read_bytes()).hexdigest())
        self.assertFalse(self.fixture.path.exists())
        self.assert_not_evaluator_evidence()

    def test_complete_unknown_states_save_without_inventing_visible_points(self):
        for index in range(8):
            self.native.mark(self.state, index, 'definition_uncertain')
        self.assertEqual(self.save(), 'save-next')
        corners = self.document()['record']['corners']
        self.assertTrue(all(c['visibility'] == 'uncertain' for c in corners))
        self.assertTrue(all(c['x'] is None and c['y'] is None for c in corners))
        self.assertEqual(self.state.kps_2d, [None] * 9)
        self.assertFalse(self.document()['record']['object']['target_identity_confirmed'])
        self.assert_not_evaluator_evidence()

    def test_local_save_preserves_existing_native_annotation_file(self):
        self.fixture.path.parent.mkdir(parents=True)
        original = b'{"schema":"synthetic_native_original","test_only":true}\n'
        self.fixture.path.write_bytes(original)
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        self.assertEqual(self.fixture.path.read_bytes(), original)
        self.assertNotEqual(self.saved_path(), self.fixture.path)
        self.assert_not_evaluator_evidence()

    def test_incomplete_s_keeps_actual_draft_and_does_not_advance(self):
        self.fixture.click(4, 3)
        self.native.mark(self.state, 1, 'self_occlusion')
        before = copy.deepcopy(self.native.record['corners'])
        self.assertIsNone(self.save())
        self.assertFalse(self.fixture.path.exists())
        self.assertFalse(self.native.visibility_progress_path.exists())
        recovery = json.loads(self.native.recovery_path.read_text())
        record = recovery['drafts']['test:0|primary']['record']
        self.assertEqual(record['corners'], before)
        self.assertEqual(sum(c['visibility'] is None for c in record['corners']), 6)
        self.assertIsNone(record['reviewer'])
        self.assert_not_evaluator_evidence()

    def test_invalid_completed_corner_data_cannot_be_saved(self):
        self.complete()
        valid = copy.deepcopy(self.native.record)
        mutations = (
            lambda r: r['corners'][0].update(x=None),
            lambda r: r['corners'][0].update(x=20),
            lambda r: r['corners'][0].update(x=float('nan')),
            lambda r: r['corners'][0].update(x=True),
            lambda r: r['corners'][0].update(external_occlusion=True),
            lambda r: r['corners'][2].update(x=4., y=3.),
            lambda r: r['corners'][7].update(id=6),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.native.record = copy.deepcopy(valid)
                mutate(self.native.record)
                self.assertIsNone(self.save())
                self.assertFalse(self.fixture.path.exists())
                self.assertFalse(self.native.visibility_progress_path.exists())
        self.assert_not_evaluator_evidence()

    def test_reload_skips_saved_primary_and_revisit_reopens_it(self):
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        self.assertEqual(self.queue_ids(self.fresh()), [])
        self.assertEqual(self.queue_ids(self.fresh(), 'repeat'), ['test:0'])
        self.assertEqual(self.queue_ids(self.fresh(revisit_saved=True)), ['test:0'])
        self.native.load(self.state, self.fixture.path)
        self.assertEqual(self.state.kps_2d[:2], [[4., 3.], [6., 4.]])
        self.assertTrue(self.native.record['corners'][3]['self_occlusion'])
        self.assert_not_evaluator_evidence()

    def test_revisit_restores_saved_actions_when_recovery_file_is_missing(self):
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        self.native.recovery_path.unlink()
        fresh = self.fresh(revisit_saved=True)
        fresh.path_map[str(self.fixture.path)] = ('test:0', 'primary')
        fresh.install(fixtures.annotate)
        fresh.load(self.state, self.fixture.path)
        self.assertEqual(fresh.record['corners'], self.document()['record']['corners'])
        self.assertEqual(self.state.kps_2d[:2], [[4., 3.], [6., 4.]])
        self.assertTrue(fresh.record['corners'][3]['self_occlusion'])
        self.assertIsNone(fresh.record['reviewer'])
        self.assert_not_evaluator_evidence()

    def test_saved_file_hash_is_required_before_completion_can_be_reused(self):
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        document = self.document()
        document['record']['corners'][0]['x'] = 5.
        self.saved_path().write_text(json.dumps(document))
        self.assert_not_reused()
        self.assert_not_evaluator_evidence()

    def test_saved_document_binding_image_frame_and_pass_are_verified_even_with_matching_hash(self):
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        valid_document, valid_progress = self.document(), self.progress()
        mutations = (
            lambda d: d['bindings'].update(manifest_sha256='bad'),
            lambda d: d.update(image_sha256='bad'),
            lambda d: d.update(frame_id='unknown:0'),
            lambda d: d.update(review_pass='repeat'),
            lambda d: d.update(schema='other'),
            lambda d: d.update(evaluation_use=True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                document = copy.deepcopy(valid_document)
                mutate(document)
                self.saved_path().write_text(json.dumps(document))
                progress = copy.deepcopy(valid_progress)
                progress['records']['test:0|primary']['file_sha256'] = hashlib.sha256(
                    self.saved_path().read_bytes()).hexdigest()
                self.native.visibility_progress_path.write_text(json.dumps(progress))
                self.assert_not_reused()
        self.assert_not_evaluator_evidence()

    def test_progress_binding_or_missing_saved_file_cannot_mark_completion(self):
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        progress = self.progress()
        invalid = copy.deepcopy(progress)
        invalid['bindings']['plan_sha256'] = 'bad'
        self.native.visibility_progress_path.write_text(json.dumps(invalid))
        self.assert_not_reused()
        self.native.visibility_progress_path.write_text(json.dumps(progress))
        self.saved_path().unlink()
        self.assert_not_reused()
        self.assert_not_evaluator_evidence()

    def test_undo_after_save_reopens_frame_until_changed_draft_is_saved(self):
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        self.fixture.key('z')
        # The display loop preserves this draft after an undo on every render.
        self.native.persist_recovery(self.state)
        self.assertIsNone(self.native.record['corners'][7]['visibility'])
        self.assertEqual(self.native.saved_visibility_ids('primary'), set())
        self.assertEqual(self.queue_ids(self.fresh()), ['test:0'])
        self.assertIsNone(self.save())
        self.assert_not_evaluator_evidence()

    def test_changing_complete_saved_state_reopens_it_until_explicit_resave(self):
        self.complete()
        self.assertEqual(self.save(), 'save-next')
        self.fixture.key('0')
        self.fixture.key('i')
        self.assertTrue(self.native.record['corners'][0]['self_occlusion'])
        self.assertEqual(self.native.saved_visibility_ids('primary'), set())
        self.assertEqual(self.queue_ids(self.fresh()), ['test:0'])
        self.assertEqual(self.save(), 'save-next')
        self.assertEqual(self.native.saved_visibility_ids('primary'), {'test:0'})
        self.assertTrue(self.document()['record']['corners'][0]['self_occlusion'])
        self.assertIsNone(self.document()['record']['corners'][0]['x'])
        self.assert_not_evaluator_evidence()


if __name__ == '__main__':
    unittest.main()
