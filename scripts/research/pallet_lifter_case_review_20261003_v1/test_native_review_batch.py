"""Partial review queues on temporary synthetic frames; no GUI or real labels."""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_native_review as nativefixtures
import open_existing_annotation as adapter
import annotate
from object_geometry_registry import load_object_geometry_registry


class NativeReviewBatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = nativefixtures.fixtures.ReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        manifest = json.loads(self.fixture.manifest.read_text())
        rows = []
        for index in range(120):
            image = self.root / f'frame_{index:03d}.png'
            image.write_bytes(nativefixtures.fixtures.png())
            rows.append(dict(frame_id=f's{index // 30}:{index}',
                session_id=f's{index // 30}', saved_frame_index=index,
                camera_sensor_timestamp_ms=1000. + index,
                width=20, height=10, image_path=image.name,
                image_sha256=nativefixtures.fixtures.sha256(image),
                repeat_review=index % 5 == 0))
        manifest['frames'] = rows
        self.fixture.manifest.write_text(json.dumps(manifest))
        self.context = nativefixtures.fixtures.Context(self.fixture.manifest,
            self.fixture.plan, self.fixture.contract, self.fixture.store)
        self.all_ids = [row['frame_id'] for row in rows]
        self.excluded = self.all_ids[:5]
        self.exclusions_path = self.root / 'USER_EXCLUSIONS.json'
        self.exclusions_path.write_text(json.dumps(dict(schema='lifter_user_exclusions_v1',
            status='EXCLUDED_BY_USER', input_bindings=self.context.bindings,
            excluded_frame_ids=self.excluded)))
        indexes = (5, 17, 29, 30, 44, 59, 60, 74, 89, 90, 104, 119)
        self.batch_ids = [self.all_ids[index] for index in indexes]
        self.batch_path = self.root / 'SMALL_BATCH_12_V1.json'
        self.batch = dict(schema='lifter_human_review_batch_v1',
            source_kind='partial_human_review_queue', bindings=self.context.bindings,
            frame_ids=self.batch_ids, repeat_frame_ids=[], frozen_plan_modified=False,
            synthetic_test_only=True)
        self.batch_path.write_text(json.dumps(self.batch))
        self.registry = load_object_geometry_registry(
            adapter.ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')
        self.camera = dict(K=[[100., 0., 10.], [0., 100., 5.], [0., 0., 1.]],
            metadata_sha256='0' * 64, image_hw=[10, 20],
            dimensions_wdh_m=[1.1, 1.1, .15], pnp_xyz_m=[1.1, .15, 1.1])
        self.camera_patch = patch.object(adapter, 'recorded_camera', return_value=self.camera)
        self.camera_patch.start()
        self.addCleanup(self.camera_patch.stop)
        self.originals = {path: path.read_bytes() for path in (
            self.fixture.manifest, self.fixture.plan, self.fixture.contract, self.exclusions_path)}

    def native(self, *, batch=True, native_pnp=True, revisit_saved=False):
        return adapter.NativeReview(self.context, self.fixture.reviewer,
            self.root / 'native', native_pnp=native_pnp,
            batch_plan=self.batch_path if batch else None,
            revisit_saved=revisit_saved)

    def queue_ids(self, native, review_pass='primary'):
        _, contexts = native.contexts('', argparse.Namespace(), self.registry, adapter.ROOT)
        context = contexts.get('review:' + review_pass)
        if context is None:
            return []
        reverse = {str(path): fid for fid, path in self.context.images.items()}
        return [reverse[path] for path in context['frame_paths']]

    def write_saved(self, frame_ids, review_pass='primary'):
        """Validator fixture only: never claim these files are measured results."""
        progress = dict(schema='lifter_native_pnp_progress_v1',
            bindings=copy.deepcopy(self.context.bindings), records={}, synthetic_test_only=True)
        documents = {}
        for fid in frame_ids:
            path = self.root / 'synthetic_saved' / (fid.replace(':', '_') + '.PNP_ASSISTED.json')
            path.parent.mkdir(exist_ok=True)
            doc = dict(schema='lifter_native_pnp_assistance_v1',
                bindings=copy.deepcopy(self.context.bindings), frame_id=fid,
                review_pass=review_pass, image_sha256=self.context.frames[fid]['image_sha256'],
                source_kind='human_assisted_annotation', evaluation_use=False,
                independent_reference=False, synthetic_test_only=True,
                annotation={'objects': [{'pose_transform': np.eye(4).tolist(),
                    'projected_cuboid': [[4., 3.]] * 8,
                    'manual_kps': [[4., 3.]] * 4 + [None] * 5}]},
                editor_kps_2d=[[4., 3.]] * 4 + [None] * 5,
                editor_keypoint_annotations=[dict(source='manual_click' if i < 4 else 'unknown',
                    xy=[4., 3.] if i < 4 else None, visibility=2 if i < 4 else 0)
                    for i in range(9)])
            path.write_text(json.dumps(doc))
            documents[fid] = path
            progress['records'][fid + '|' + review_pass] = dict(frame_id=fid,
                review_pass=review_pass, assisted_file=str(path),
                full_visibility_review_complete=False)
        (self.root / 'NATIVE_PNP_PROGRESS.json').write_text(json.dumps(progress))
        return documents

    def assert_originals_unchanged(self):
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original, str(path))

    def test_batch_rejects_mismatched_bindings_duplicates_exclusions_and_bad_ids(self):
        changes = (
            lambda b: b['bindings'].update(plan_sha256='bad'),
            lambda b: b.update(frame_ids=[self.batch_ids[0], self.batch_ids[0]]),
            lambda b: b.update(frame_ids=[self.excluded[0]]),
            lambda b: b.update(frame_ids=['unknown:999']),
            lambda b: b.update(frame_ids=[]),
            lambda b: b.update(frame_ids=[123]),
            lambda b: b.update(repeat_frame_ids=[self.batch_ids[0]]),
            lambda b: b.update(frozen_plan_modified=True),
            lambda b: b.update(source_kind='human_reviewed'),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                invalid = copy.deepcopy(self.batch)
                change(invalid)
                self.batch_path.write_text(json.dumps(invalid))
                with self.assertRaises(adapter.ValidationError):
                    self.native()
        self.assert_originals_unchanged()
        self.assertFalse(self.fixture.store.exists())

    def test_full_counts_115_remain_separate_from_partial_12_and_repeat_deferred(self):
        full = self.native(batch=False)
        self.assertEqual(len(self.queue_ids(full)), 115)
        self.assertEqual(len(self.queue_ids(full, 'repeat')), 23)
        partial = self.native()
        self.assertEqual(self.queue_ids(partial), self.batch_ids)
        self.assertEqual(self.queue_ids(partial, 'repeat'), [])
        self.assertEqual(partial.requested_counts()['primary_required'], 115)
        counts = partial.batch_counts()
        self.assertEqual(counts['primary_required'], 12)
        self.assertEqual(counts['full_retained_primary'], 115)
        self.assertEqual(counts['primary_pnp_remaining'], 12)
        self.assertFalse(counts['full_population_completed'])
        self.assertTrue(counts['repeat_deferred'])
        self.assertEqual(self.context.counts()['total_primary'], 120)
        self.assertEqual(self.context.counts()['primary_reviewed'], 0)
        self.assert_originals_unchanged()

    def test_saved_pnp_skips_partial_stage_only_and_visibility_or_revisit_reopens(self):
        saved = self.batch_ids[:2]
        self.write_saved(saved)
        partial = self.native()
        self.assertEqual(self.queue_ids(partial), self.batch_ids[2:])
        self.assertEqual(partial.batch_counts()['primary_pnp_saved'], 2)
        self.assertEqual(partial.batch_counts()['primary_pnp_remaining'], 10)
        self.assertEqual(partial.batch_counts()['primary_reviewed'], 0)
        self.assertEqual(self.queue_ids(self.native(native_pnp=False)), self.batch_ids)
        self.assertEqual(self.queue_ids(self.native(revisit_saved=True)), self.batch_ids)
        self.assertEqual(len(self.queue_ids(self.native(batch=False))), 115)
        self.assertEqual(self.context.export()['records'], [])
        self.assertFalse(self.fixture.store.exists())
        self.assert_originals_unchanged()

    def test_proof_bindings_image_frame_pass_schema_and_missing_file_are_checked(self):
        fid = self.batch_ids[0]
        changes = (
            lambda d: d['bindings'].update(manifest_sha256='bad'),
            lambda d: d.update(image_sha256='bad'),
            lambda d: d.update(frame_id=self.batch_ids[1]),
            lambda d: d.update(review_pass='repeat'),
            lambda d: d.update(schema='other'),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                path = self.write_saved([fid])[fid]
                invalid = json.loads(path.read_text())
                change(invalid)
                path.write_text(json.dumps(invalid))
                self.assertIn(fid, self.queue_ids(self.native()))
        self.write_saved([fid])[fid].unlink()
        self.assertIn(fid, self.queue_ids(self.native()))
        self.assert_originals_unchanged()

    def test_progress_binding_mismatch_is_not_accepted_as_completed(self):
        self.write_saved([self.batch_ids[0]])
        path = self.root / 'NATIVE_PNP_PROGRESS.json'
        progress = json.loads(path.read_text())
        progress['bindings']['plan_sha256'] = 'bad'
        path.write_text(json.dumps(progress))
        with self.assertRaises(adapter.ValidationError):
            self.queue_ids(self.native())
        self.assertFalse(self.fixture.store.exists())

    def test_same_envelope_incomplete_or_malformed_assistance_does_not_skip_frame(self):
        fid = self.batch_ids[0]
        changes = (
            lambda d: d.update(source_kind='machine_proposed'),
            lambda d: d.update(evaluation_use=True),
            lambda d: d.pop('annotation'),
            lambda d: d['annotation'].update(objects=[]),
            lambda d: d['annotation']['objects'][0].update(pose_transform=[[1.]]),
            lambda d: d['annotation']['objects'][0].update(
                pose_transform=[[float('nan')] * 4] * 4),
            lambda d: d.update(annotation=None),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                path = self.write_saved([fid])[fid]
                invalid = json.loads(path.read_text())
                change(invalid)
                path.write_text(json.dumps(invalid))
                self.assertIn(fid, self.queue_ids(self.native()))
        for content in ('[]', '{"schema":'):
            with self.subTest(content=content):
                path = self.write_saved([fid])[fid]
                path.write_text(content)
                self.assertIn(fid, self.queue_ids(self.native()))
        self.assert_originals_unchanged()

    def test_all_12_saved_does_not_claim_full_population_or_human_visibility_complete(self):
        self.write_saved(self.batch_ids)
        partial = self.native()
        self.assertEqual(self.queue_ids(partial), [])
        counts = partial.batch_counts()
        self.assertEqual(counts['primary_pnp_saved'], 12)
        self.assertEqual(counts['primary_pnp_remaining'], 0)
        self.assertEqual(counts['primary_reviewed'], 0)
        self.assertFalse(counts['full_population_completed'])
        self.assertEqual(partial.requested_counts()['primary_remaining'], 115)
        self.assertEqual(self.queue_ids(self.native(native_pnp=False)), self.batch_ids)
        self.assertEqual(self.context.export()['records'], [])
        self.assert_originals_unchanged()

    def test_recovered_draft_outside_batch_is_preserved_when_batch_view_is_saved(self):
        outside_id = self.all_ids[6]
        state = annotate.State()
        state.img = np.zeros((10, 20, 3), np.uint8)
        state.img_shape = state.img.shape
        state.mode = 'click'
        state.locked_pose = None
        original = self.native(batch=False, native_pnp=False)
        original.editor = annotate
        outside_path = self.root / 'native/primary/outside.json'
        original.path_map[str(outside_path)] = (outside_id, 'primary')
        original.load(state, outside_path)
        original.record['corners'][0].update(visibility='direct_visible',
            definition_confirmed=True, x=4., y=3.)
        original.record['object'].update(presence='present', target_identity_confirmed=True,
            target_object_id='SYNTHETIC_ONLY')
        state.kps_2d[0] = [4., 3.]
        annotate._set_keypoint_state(state, 0, [4., 3.],
            source='manual_click', visibility=2, reason='visible')
        original.persist_recovery(state)
        key = outside_id + '|primary'
        before = copy.deepcopy(original.recovery['drafts'][key])
        partial = self.native()
        self.assertNotIn(outside_id, self.queue_ids(partial))
        self.assertEqual(partial.recovery['drafts'][key], before)
        partial.editor = annotate
        fid = self.batch_ids[0]
        current_path = self.root / 'native/primary/frame_005.json'
        partial.path_map[str(current_path)] = (fid, 'primary')
        partial.load(state, current_path)
        partial.persist_recovery(state)
        after = json.loads(partial.recovery_path.read_text())
        self.assertEqual(after['drafts'][key], before)
        self.assertEqual(after['drafts'][key]['record']['corners'][0]['x'], 4.)
        self.assertIn(fid + '|primary', after['drafts'])
        self.assertFalse(self.fixture.store.exists())
        self.assert_originals_unchanged()


if __name__ == '__main__':
    unittest.main()
