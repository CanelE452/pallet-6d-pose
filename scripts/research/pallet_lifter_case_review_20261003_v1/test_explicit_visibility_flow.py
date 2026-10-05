"""Temporary synthetic state/UI callbacks; no desktop windows or real profiles."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_native_review as fixtures
import open_existing_annotation as adapter


class _Variable:
    def __init__(self, value=False, **kwargs): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class _Widget:
    def __init__(self, *args, **kwargs): self.options = kwargs
    def pack(self, **kwargs): return None


class _Window:
    def __init__(self, on_loop): self.on_loop = on_loop
    def title(self, value): self.title_text = value
    def geometry(self, value): self.geometry_text = value
    def attributes(self, *args): return None
    def update_idletasks(self): return None
    def winfo_reqwidth(self): return 660
    def winfo_reqheight(self): return 330
    def winfo_screenwidth(self): return 1920
    def winfo_screenheight(self): return 1080
    def destroy(self): self.closed = True
    def mainloop(self): self.on_loop()


class ExplicitVisibilityFlowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.native, self.state = self.fixture.native, self.fixture.state

    def test_i_on_last_corner_keeps_draft_without_queued_save_or_profile(self):
        self.native.reviewer = None
        for index in range(7):
            self.native.mark(self.state, index, 'definition_uncertain')
        self.state.active = 7
        with (patch.object(self.native, 'save', side_effect=AssertionError('No automatic submit')),
                patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No popup'))):
            self.assertIsNone(self.fixture.key('i'))
        self.assertIsNone(self.native.queued_key)
        self.assertTrue(self.native.recovery_path.is_file())
        recovery = json.loads(self.native.recovery_path.read_text())
        current = recovery['drafts']['test:0|primary']['record']
        self.assertEqual(current['status'], 'draft')
        self.assertTrue(current['corners'][7]['self_occlusion'])
        self.assertFalse(self.fixture.fixture.store.exists())
        self.assertEqual(self.native.ctx.export()['records'], [])

    def test_i_on_all_filled_frame_still_does_not_submit(self):
        for index in range(8):
            self.native.mark(self.state, index, 'definition_uncertain')
        self.state.active = 2
        with (patch.object(self.native, 'save', side_effect=AssertionError('No automatic submit')),
                patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No popup'))):
            self.fixture.key('i')
        self.assertIsNone(self.native.queued_key)
        self.assertTrue(self.native.record['corners'][2]['self_occlusion'])
        self.assertFalse(self.fixture.fixture.store.exists())

    def test_explicit_uppercase_v_delegates_without_submission_or_profile(self):
        before = copy.deepcopy(self.native.record)
        with (patch('select_visible_corner.select_visible_corner', return_value=False) as selected,
                patch.object(self.native, 'save', side_effect=AssertionError('No submit')),
                patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No popup'))):
            self.assertIsNone(self.fixture.key('V'))
        selected.assert_called_once_with(self.native, self.state)
        self.assertEqual(self.native.record, before)
        self.assertIsNone(self.native.queued_key)
        self.assertFalse(self.fixture.fixture.store.exists())

    def test_known_actual_assistance_and_previous_manual_display_are_forwarded_only_as_known_flags(self):
        self.native.record['native_pnp_assistance'] = [dict(source='SYNTHETIC INPUT HISTORY')]
        self.native.loaded_existing_manual = True
        with patch('open_existing_annotation.choose_profile', return_value=None) as choose:
            self.assertIsNone(self.native.choose_history_profile(self.native.profile_path, preview=self.state.img))
        known = choose.call_args.kwargs['known_history']
        self.assertEqual(known, dict(machine_assistance=True, previous_annotation_exposure=True))
        self.assertNotIn('previous_prediction_exposure', known)
        sources = choose.call_args.kwargs['history_sources']
        self.assertEqual(sources['machine_assistance']['record_fields'], ['native_pnp_assistance'])
        self.assertEqual(sources['previous_annotation_exposure']['frame_id'], 'test:0')
        self.assertFalse(self.native.profile_path.exists())

    def _profile_without_desktop(self, prediction_choice):
        import tkinter as tk
        buttons, checkboxes = [], []
        def button(*args, **kwargs):
            widget = _Widget(*args, **kwargs); buttons.append(widget); return widget
        def checkbox(*args, **kwargs):
            widget = _Widget(*args, **kwargs); checkboxes.append(widget); return widget
        def loop():
            if prediction_choice is not None:
                marker = '본 적 있음' if prediction_choice else '본 적 없음'
                next(widget for widget in buttons if marker in widget.options['text']).options['command']()
        window = _Window(loop)
        profile = self.fixture.fixture.root / 'SYNTHETIC_HISTORY_PROFILE_ONLY.json'
        with (patch('scripts.annotate.korean_tk.install'), patch.object(tk, 'Tk', return_value=window),
                patch.object(tk, 'Frame', _Widget), patch.object(tk, 'Label', _Widget),
                patch.object(tk, 'Button', side_effect=button), patch.object(tk, 'Checkbutton', side_effect=checkbox),
                patch.object(tk, 'BooleanVar', _Variable)):
            result = adapter.choose_profile(profile, known_history=dict(machine_assistance=True,
                previous_annotation_exposure=True), history_sources={'synthetic_test_only': True})
        self.assertEqual(checkboxes, [])
        self.assertEqual(len(buttons), 3)
        return result, profile

    def test_prediction_exposure_is_not_written_when_no_human_answer_is_given(self):
        result, profile = self._profile_without_desktop(None)
        self.assertIsNone(result)
        self.assertFalse(profile.exists())
        self.assertFalse(self.fixture.fixture.store.exists())

    def test_only_explicit_prediction_answer_completes_temporary_profile(self):
        result, profile = self._profile_without_desktop(False)
        self.assertTrue(result['machine_assistance'])
        self.assertTrue(result['previous_annotation_exposure'])
        self.assertIs(result['previous_prediction_exposure'], False)
        saved = json.loads(profile.read_text())
        self.assertNotIn('previous_prediction_exposure', saved['recorded_history_flags'])
        self.assertFalse(self.fixture.fixture.store.exists())


if __name__ == '__main__':
    unittest.main()
