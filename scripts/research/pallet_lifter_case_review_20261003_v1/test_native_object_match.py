"""Synthetic temporary UI fixtures; never human or experimental evidence."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/'combined_integration'))
import test_bridge as fixtures
from open_object_match_annotation import NativeObjectReview, primary_snapshot
from object_match_review import Context
from bridge import GateError, read_json, read_jsonl


class NativeObjectTests(unittest.TestCase):
    def setUp(self):
        fixtures.BridgeTests.setUp(self)
        self.ctx = Context(self.queue, self.manifest, self.predictions,
                           self.reference, self.root/'native_store.json')
        self.native = NativeObjectReview(self.ctx)
        self.native.reviewer = fixtures.BridgeTests.reviewer(self)
        self.state = SimpleNamespace(img=cv2.imread(str(self.ctx.images['s:0'])))
        self.navigation = self.root/'view.json'
        self.native.paths[str(self.navigation.resolve())] = 's:0'
        self.native.load(self.state, self.navigation)
        self.editor = SimpleNamespace(cv2=SimpleNamespace(waitKey=lambda delay:255,
            resize=cv2.resize, EVENT_LBUTTONDOWN=cv2.EVENT_LBUTTONDOWN),
            _toast=lambda *args:None)
        self.native.install(self.editor)

    def test_native_button_real_save_restart_and_evaluator_link(self):
        image = self.editor.render(self.state, 0, 1, 'fixture')
        self.assertEqual(image.shape, (770,1290,3))
        self.assertFalse(self.ctx.store_path.exists())
        box, value = self.native.buttons[0]
        self.editor.on_mouse(cv2.EVENT_LBUTTONDOWN, box[0]+5,box[1]+5,0,self.state)
        key = self.editor.cv2.waitKey(20)
        self.assertEqual(key, ord('1'))
        self.assertEqual(self.editor._handle_click_key(key,self.state), 'save-next')
        self.assertEqual(self.ctx.counts()['same'],1)
        restarted = Context(self.queue,self.manifest,self.predictions,self.reference,self.ctx.store_path)
        self.assertEqual(restarted.counts()['pending'],0)
        export = read_json(self.root/'LIFTER_OBJECT_MATCH_REVIEWED.json')
        self.assertEqual(export['records'][0]['source_kind'],'human_reviewed')
        derived = read_jsonl(self.root/'ALL_STORED_FRAMES_EVALUATOR.jsonl')[0]
        original = read_jsonl(self.predictions)[0]
        for method in ('Base','N3'):
            self.assertTrue(derived['methods'][method]['object_match'])
            self.assertEqual(derived['methods'][method]['keypoints'],original['methods'][method]['keypoints'])
        self.assertTrue((self.root/'L4_EVALUATOR_DERIVATION.json').is_file())

    def test_reselection_keeps_history_and_other_decisions(self):
        self.native.choose('1')
        self.native.load(self.state,self.navigation)
        self.native.choose('2')
        self.assertEqual(self.ctx.counts()['different'],1)
        self.assertTrue(self.ctx.store_path.with_suffix('.json.history.jsonl').exists())
        self.native.load(self.state,self.navigation)
        self.native.choose('3')
        row = read_jsonl(self.root/'ALL_STORED_FRAMES_EVALUATOR.jsonl')[0]
        self.assertIsNone(row['methods']['Base']['object_match'])
        self.assertEqual(self.ctx.counts()['undetermined'],1)

    def test_navigation_and_opening_never_submit_decision(self):
        self.editor.render(self.state,0,1,'fixture')
        self.assertEqual(self.editor._handle_click_key(ord('n'),self.state),'next')
        self.assertEqual(self.editor._handle_click_key(ord('q'),self.state),'quit')
        self.assertFalse(self.ctx.store_path.exists())
        self.assertEqual(self.ctx.counts()['reviewed'],0)

    def test_snapshot_is_stable_when_repeat_export_changes(self):
        ref = read_json(self.reference)
        a = primary_snapshot(ref,{'s:0'},self.bindings)
        repeat = copy.deepcopy(ref['records'][0]);repeat['review_pass']='repeat'
        repeat['corners'][0]['x']=8
        ref['records'].append(repeat)
        ref['exported_at']='changed later'
        self.assertEqual(a,primary_snapshot(ref,{'s:0'},self.bindings))
        ref['records'][0]['corners'][0]['x']=9
        self.assertNotEqual(a,primary_snapshot(ref,{'s:0'},self.bindings))

    def test_snapshot_missing_primary_or_binding_rejected(self):
        ref = read_json(self.reference)
        with self.assertRaisesRegex(GateError,'1장'):
            primary_snapshot(ref,{'s:0','s:1'},self.bindings)
        ref['bindings']['plan_sha256']='changed'
        with self.assertRaises(GateError):
            primary_snapshot(ref,{'s:0'},self.bindings)


if __name__=='__main__':
    unittest.main()
