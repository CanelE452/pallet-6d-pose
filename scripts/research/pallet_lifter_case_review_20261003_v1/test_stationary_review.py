"""Synthetic, temporary-only stationary review actions; no human production labels."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_stationary_review import (HERE, SESSION_IDS, StationaryReview, Viewer,
                                    main, sha256, validate_profile, write_json)

SPEC = importlib.util.spec_from_file_location('stationary_test_evaluator', HERE/'metrics/evaluate.py')
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def profile(prior=False, blind=True):
    return dict(id='SYNTHETIC_TEST_ONLY', entered_by='human', confirmation=True,
                confirmed_at='2000-01-01T00:00:00Z', previous_prediction_exposure=prior,
                blind_selection_confirmation=blind, fixed_before_predictions=not prior and blind)


class StationaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan_path, self.store = self.root/'plan.json', self.root/'review/store.json'
        frames = []
        # Deliberately nonuniform sensor times: uniform video-FPS interpolation is wrong.
        for sid in SESSION_IDS:
            video = self.root/('forklift_v4_recording_20260901_' + sid + '_raw.mp4')
            video.write_bytes(b'SYNTHETIC_TEST_VIDEO_BINDING_ONLY_' + sid.encode())
            for i, timestamp in enumerate((1000., 1130., 1680., 2700.)):
                frames.append(dict(frame_id=sid + ':' + str(i), session_id=sid,
                    saved_frame_index=i, camera_sensor_timestamp_ms=timestamp,
                    raw_video_sha256=sha256(video), decoded_bgr_sha256='SYNTHETIC_PIXELS_ONLY'))
        write_json(self.plan_path, dict(frames=frames))
        self.review = StationaryReview(self.plan_path, self.root, self.store)

    def test_prepare_only_reads_sources_without_creating_human_artifact(self):
        with patch('builtins.print'):
            self.assertEqual(main(['--plan', str(self.plan_path), '--video-root', str(self.root),
                                   '--store', str(self.store), '--prepare-only']), 0)
        self.assertFalse(self.store.exists())
        self.assertFalse((self.store.parent/'STATIONARY_REVIEWER_PROFILE.json').exists())
        self.assertEqual(self.review.counts()['completed_sessions'], 0)

    def test_sensor_mapping_restart_and_actual_evaluator_path(self):
        self.review.set_profile(profile())
        self.review.start(SESSION_IDS[0], 1)
        # A genuine in-progress start survives a close/restart; no interval is manufactured.
        reopened = StationaryReview(self.plan_path, self.root, self.store)
        self.assertEqual(reopened.store['draft_interval']['saved_start_index'], 1)
        self.assertEqual(reopened.store['intervals'], [])
        interval = reopened.finish(SESSION_IDS[0], 3)
        self.assertEqual((interval['sensor_start_ms'], interval['sensor_end_ms']), (1130., 2700.))
        self.assertEqual(reopened.store['strict_evaluator_status'], 'ELIGIBLE')
        frames = reopened.plan['frames']
        predictions = {f['frame_id']: {'methods': {m: dict(pose_state='fresh', pose=dict(
            x_m=f['saved_frame_index']/10, z_m=1., yaw_deg=179. if f['saved_frame_index'] % 2 else -179.))
            for m in ('Base', 'N3')}} for f in frames}
        result = evaluator.evaluate_stop_intervals(frames, predictions, reopened.store,
                                                   {'plan_sha256': reopened.plan_hash})
        self.assertEqual(result['intervals'][0]['stored_frame_count'], 3)
        self.assertAlmostEqual(result['intervals'][0]['duration_s'], 1.570)
        self.assertEqual(result['intervals'][0]['methods']['Base']['fresh_output_frame_count'], 3)
        self.assertTrue(self.store.with_suffix('.history.jsonl').is_file())

    def test_prior_prediction_exposure_blocks_strict_metric_without_losing_labels(self):
        self.review.set_profile(profile(prior=True, blind=True))
        self.review.start(SESSION_IDS[0], 0)
        self.review.finish(SESSION_IDS[0], 2)
        self.assertEqual(len(self.review.store['intervals']), 1)
        self.assertFalse(self.review.store['fixed_before_predictions'])
        self.assertEqual(self.review.store['strict_evaluator_status'], 'BLOCKED_EXPOSURE')
        with self.assertRaises(evaluator.ContractError):
            evaluator.evaluate_stop_intervals(self.review.plan['frames'], {}, self.review.store,
                                              {'plan_sha256': self.review.plan_hash})
        # A forged true is rejected even if a separate blindness checkbox was checked.
        forged = profile(prior=True, blind=True)
        forged['fixed_before_predictions'] = True
        with self.assertRaises(ValueError):
            validate_profile(forged)

    def test_no_blind_confirmation_is_not_implicitly_promoted(self):
        self.review.set_profile(profile(prior=False, blind=False))
        self.review.complete_session(SESSION_IDS[0], no_stop=True)
        self.assertFalse(self.review.store['fixed_before_predictions'])
        self.assertFalse(self.review.store['confirmed_no_stop_intervals'])
        self.assertEqual(self.review.store['strict_evaluator_status'], 'BLOCKED_EXPOSURE')

    def test_global_no_stop_requires_four_explicit_session_actions(self):
        self.review.set_profile(profile())
        for sid in SESSION_IDS[:3]:
            self.review.complete_session(sid, no_stop=True)
        self.assertFalse(self.review.store['confirmed_no_stop_intervals'])
        self.assertEqual(self.review.store['status'], 'WAITING_HUMAN')
        self.review.complete_session(SESSION_IDS[-1], no_stop=True)
        self.assertTrue(self.review.store['confirmed_no_stop_intervals'])
        result = evaluator.evaluate_stop_intervals(self.review.plan['frames'], {}, self.review.store,
                                                   {'plan_sha256': self.review.plan_hash})
        self.assertEqual(result['intervals'], 'NA')
        self.assertEqual(result['std_x_m'], 'NA')
        self.assertEqual(self.review.counts()['completed_sessions'], 4)

    def test_wrong_session_zero_duration_overlap_and_absent_reviewer_rejected(self):
        with self.assertRaises(ValueError):
            self.review.start(SESSION_IDS[0], 0)
        self.review.set_profile(profile())
        self.review.start(SESSION_IDS[0], 0)
        for sid, index in ((SESSION_IDS[1], 2), (SESSION_IDS[0], 0)):
            with self.assertRaises(ValueError):
                self.review.finish(sid, index)
        self.review.finish(SESSION_IDS[0], 2)
        self.review.start(SESSION_IDS[0], 1)
        with self.assertRaises(ValueError):
            self.review.finish(SESSION_IDS[0], 3)
        self.review.cancel_start()
        with self.assertRaises(ValueError):
            self.review.complete_session(SESSION_IDS[0], no_stop=True)

    def test_delete_preserves_history_and_requires_new_session_completion(self):
        self.review.set_profile(profile())
        self.review.start(SESSION_IDS[0], 0)
        self.review.finish(SESSION_IDS[0], 2)
        self.review.complete_session(SESSION_IDS[0])
        self.review.delete_last(SESSION_IDS[0])
        self.assertEqual(self.review.store['intervals'], [])
        self.assertNotIn(SESSION_IDS[0], self.review.store['session_reviews'])
        revisions = [json.loads(line) for line in self.store.with_suffix('.history.jsonl').read_text().splitlines()]
        self.assertTrue(any(v['previous']['intervals'] for v in revisions))
        self.assertFalse(self.review.store['confirmed_no_stop_intervals'])

    def test_changed_media_or_sensor_mapping_cannot_reuse_store(self):
        self.review.set_profile(profile())
        self.review.start(SESSION_IDS[0], 0)
        self.review.finish(SESSION_IDS[0], 2)
        changed = json.loads(self.store.read_text())
        changed['intervals'][0]['sensor_end_ms'] += 1
        write_json(self.store, changed)
        with self.assertRaises(ValueError):
            StationaryReview(self.plan_path, self.root, self.store)
        self.review.videos[SESSION_IDS[0]].write_bytes(b'CORRUPTED_SYNTHETIC_VIDEO')
        with self.assertRaises(ValueError):
            StationaryReview(self.plan_path, self.root, self.root/'new_store.json')

    def test_initial_native_slider_does_not_seek_before_video_open(self):
        # Native createTrackbar may call its callback before the capture is ready.
        instance = object.__new__(Viewer)
        instance.setting_slider, instance.cap = False, None
        with patch.object(instance, 'seek') as seek:
            instance.slider(0)
            seek.assert_not_called()


if __name__ == '__main__':
    unittest.main()
