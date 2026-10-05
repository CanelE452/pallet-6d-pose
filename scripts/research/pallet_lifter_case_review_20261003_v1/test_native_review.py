"""Synthetic native-editor actions in temporary files; never production labels."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'review'))
import test_review as fixtures
from open_existing_annotation import NativeReview, ROOT, choose_profile
sys.path.insert(0, str(ROOT / 'scripts/annotate'))
import annotate
import annotate_draw
from annotate_pnp import make_pallet_keypoints_3d


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.ctx.contract['corners'] = [dict(id=i, xyz_m=xyz.tolist()) for i,xyz in enumerate(make_pallet_keypoints_3d(1.1,1.1,.15)[:8])]
        self.native = NativeReview(f.ctx, f.reviewer, f.root/'native')
        self.path = f.root/'native/primary/raw.json'
        self.native.path_map[str(self.path)] = ('test:0', 'primary')
        old = {k:getattr(annotate,k) for k in ('on_mouse','_handle_click_key','render','load_existing_annotation','update_pose')}
        old_panel = annotate_draw.build_panel
        old_wait = annotate.cv2.waitKey
        def restore():
            for k,v in old.items(): setattr(annotate,k,v)
            annotate_draw.build_panel=old_panel
            annotate.cv2.waitKey=old_wait
        self.addCleanup(restore)
        self.native.install(annotate)
        self.state=annotate.State()
        self.state.img=np.zeros((10,20,3),np.uint8)
        self.state.img_shape=self.state.img.shape
        self.state.zoom=1.; self.state.pan=[0,0];self.state.mode='click';self.state.line_mode=False
        self.native.load(self.state,self.path)

    def click(self,x=4,y=3):
        # Native canvas zoom/pan must map clicks back to original-image pixels.
        screen_x=(annotate.MARGIN_L+x-self.state.pan[0])*self.state.zoom
        screen_y=(annotate.MARGIN_T+y-self.state.pan[1])*self.state.zoom
        annotate.on_mouse(annotate.cv2.EVENT_LBUTTONDOWN,screen_x,screen_y,0,self.state)

    def key(self,c):
        return annotate._handle_click_key(ord(c),self.state,str(self.path),'','',np.eye(3))

    def test_partial_click_save_export_repeat_blind(self):
        self.click()
        for i in range(1,8): self.native.mark(self.state,i,'self_occlusion')
        self.assertEqual(self.key('s'),'save-next')
        out=self.fixture.ctx.export()['records'][0]
        self.assertEqual(out['corners'][0]['x'],4.)
        self.assertEqual(sum(c['visibility']=='direct_visible' for c in out['corners']),1)
        self.assertTrue(all(c['x'] is None for c in out['corners'][1:]))
        self.assertIsNone(self.state.pose)
        self.native.load(self.state,self.path)
        self.assertEqual(self.state.kps_2d[0],[4.,3.])
        repeated=self.fixture.root/'native/repeat/raw.json'
        self.native.path_map[str(repeated)]=('test:0','repeat')
        self.native.load(self.state,repeated)
        self.assertEqual(self.state.kps_2d,[None]*9)
        self.assertTrue(all(c['visibility'] is None for c in self.native.record['corners']))

    def test_repeat_actual_own_annotation_exposure_preserves_blindness_and_profile(self):
        self.click()
        for i in range(1,8): self.native.mark(self.state,i,'self_occlusion')
        self.key('s')
        repeated=self.fixture.root/'native/repeat/raw.json'
        self.native.path_map[str(repeated)]=('test:0','repeat')
        self.native.load(self.state,repeated)
        self.assertFalse(self.native.reviewer['previous_annotation_exposure'])
        self.assertTrue(self.native.record['reviewer']['previous_annotation_exposure'])
        self.assertEqual(self.state.kps_2d,[None]*9)
        self.assertTrue(all(c['visibility'] is None for c in self.native.record['corners']))
        evidence=self.native.record['actual_annotation_exposure_provenance']
        self.assertEqual(len(evidence),1)
        self.assertEqual(evidence[0]['previous_review_pass'],'primary')
        self.assertEqual(evidence[0]['previous_revision'],1)
        self.assertTrue(evidence[0]['first_pass_coordinates_not_prefilled'])
        for i in range(8): self.native.mark(self.state,i,'definition_uncertain')
        self.assertTrue(self.native.save(self.state,repeated,'reviewed'))
        record=self.fixture.ctx.record_for('test:0','repeat')
        self.assertTrue(record['reviewer']['previous_annotation_exposure'])
        self.assertEqual(record['repeat_relationship'],'same_person_repeat')
        self.assertFalse(record['independent_repeat'])
        self.assertTrue(all(c['x'] is None for c in record['corners']))
        # Another person is never assigned exposure simply because a primary record exists.
        other=copy.deepcopy(self.fixture.reviewer);other['id']='OTHER_TEST_FIXTURE'
        reviewer,evidence=self.native.reviewer_for_frame('test:0','repeat',other)
        self.assertFalse(reviewer['previous_annotation_exposure'])
        self.assertEqual(evidence,[])

    def test_outside_click_rejected_and_no_auto_projection(self):
        self.click(21,3)
        self.assertEqual(self.state.kps_2d,[None]*9)
        self.assertIsNone(self.native.record['corners'][0]['visibility'])
        self.click()
        before=copy.deepcopy(self.state.kps_2d)
        self.key('g');self.key('f');self.key('t');self.key('x')
        annotate.update_pose(self.state,np.eye(3))
        self.assertEqual(self.state.kps_2d,before)
        self.assertIsNone(self.state.pose)

    def test_unreviewed_cannot_submit_and_draft_not_exported(self):
        self.click()
        self.assertIsNone(self.key('s'))
        self.assertEqual(self.fixture.ctx.export()['records'],[])
        self.key('S')
        self.assertEqual(self.fixture.ctx.counts()['drafts'],1)
        self.assertEqual(self.fixture.ctx.export()['records'],[])

    def test_explicit_absent_preserved(self):
        with patch('tkinter.messagebox.askyesno',return_value=True) as confirm:
            self.key('a')
            self.assertEqual(self.key('s'),'save-next')
            self.assertEqual(confirm.call_count,1)
        record=self.fixture.ctx.export()['records'][0]
        self.assertEqual(record['object']['presence'],'absent')
        self.assertFalse(record['object']['target_identity_confirmed'])
        self.assertTrue(all(c['x'] is None for c in record['corners']))

    def test_unknown_requires_human_action_and_does_not_create_clicks(self):
        with patch('tkinter.messagebox.askyesno',return_value=False): self.key('U')
        self.assertTrue(all(c['visibility'] is None for c in self.native.record['corners']))
        with patch('tkinter.messagebox.askyesno',return_value=True): self.key('U')
        self.assertTrue(all(c['visibility']=='uncertain' and c['x'] is None for c in self.native.record['corners']))

    def test_panel_renders_eight_statuses(self):
        image=annotate_draw.build_panel(1000,0,self.state.kps_2d,None,0,120,1.,False)
        self.assertEqual(image.shape,(1000,annotate_draw.PANEL_W,3))

    def test_initial_enlarged_raw_view_not_cut_by_header_or_footer(self):
        # Pure synthetic view geometry; no clicks, labels or store writes.
        self.state.img=np.zeros((480,640,3),np.uint8)
        self.native.load(self.state,self.path)
        zoom=self.state.zoom
        px,py=self.state.pan
        h,w=self.state.img.shape[:2]
        canvas_w=w+annotate.MARGIN_L+annotate.MARGIN_R
        canvas_h=h+annotate.MARGIN_T+annotate.MARGIN_B
        self.assertGreater(zoom,1.)
        self.assertGreaterEqual((annotate.MARGIN_L-px)*zoom,0.)
        self.assertGreaterEqual((annotate.MARGIN_T-py)*zoom,28.)
        self.assertLessEqual((annotate.MARGIN_L+w-px)*zoom,canvas_w)
        self.assertLessEqual((annotate.MARGIN_T+h-py)*zoom,canvas_h-72.)
        self.assertEqual(self.state.kps_2d,[None]*9)
        self.assertFalse(self.fixture.store.exists())

    def test_visibility_reset_preserves_clicks_and_cancels_new_states_without_submitting(self):
        self.click()
        self.native.mark(self.state,1,'self_occlusion')
        self.key('r')
        self.assertEqual(self.state.kps_2d,[[4.,3.]]+[None]*8)
        self.assertEqual(self.native.record['corners'][0]['visibility'],'direct_visible')
        self.assertTrue(all(c['visibility'] is None and not c['self_occlusion'] for c in self.native.record['corners'][1:]))
        self.assertFalse(self.fixture.store.exists())

    def test_visibility_cancel_and_undo_preserve_manual_point_without_submitting(self):
        self.click();self.click(6,4)
        self.key('0');self.key('d')
        self.assertEqual(self.state.kps_2d[0],[4.,3.])
        self.assertEqual(self.state.kps_2d[1],[6.,4.])
        self.assertEqual(self.native.record['corners'][0]['visibility'],'direct_visible')
        self.key('z')
        self.assertEqual(self.state.kps_2d[0],[4.,3.])
        self.assertEqual(self.state.kps_2d[1],[6.,4.])
        self.assertEqual(self.native.record['corners'][0]['visibility'],'direct_visible')
        self.key('z')
        self.assertEqual(self.state.kps_2d[0],[4.,3.])
        self.assertIsNone(self.state.kps_2d[1])
        self.assertFalse(self.fixture.store.exists())

    def test_visibility_delete_undo_and_reset_undo(self):
        self.key('i')
        self.assertTrue(self.native.record['corners'][0]['self_occlusion'])
        self.key('z')
        self.assertIsNone(self.native.record['corners'][0]['visibility'])
        self.key('o'); self.key('0'); self.key('d')
        self.assertIsNone(self.native.record['corners'][0]['visibility'])
        self.key('z')
        self.assertTrue(self.native.record['corners'][0]['out_of_frame'])
        self.key('r'); self.key('z')
        self.assertTrue(self.native.record['corners'][0]['out_of_frame'])
        self.assertIsNone(self.state.kps_2d[0])
        self.assertFalse(self.fixture.store.exists())

    def test_true_last_action_undo_overwrite_and_object_confirmation(self):
        self.click(); self.key('0'); self.click(7,5)
        self.assertEqual(self.state.kps_2d[0],[7.,5.])
        self.key('z')
        self.assertEqual(self.state.kps_2d[0],[4.,3.])
        self.key('z')
        self.assertIsNone(self.state.kps_2d[0])
        self.assertFalse(self.native.record['object']['target_identity_confirmed'])

    def test_eight_unknown_states_require_explicit_save_without_target_dialog(self):
        for _ in range(8): self.key('u')
        self.assertIsNone(self.native.queued_key)
        self.assertEqual(self.fixture.ctx.export()['records'],[])
        self.assertTrue(self.native.recovery_path.exists())
        with patch('tkinter.messagebox.askyesno',side_effect=AssertionError('No per-frame target popup')):
            self.assertEqual(self.key('s'),'save-next')
        record=self.fixture.ctx.export()['records'][0]
        self.assertEqual(record['object']['presence'],'uncertain')
        self.assertFalse(record['object']['target_identity_confirmed'])
        self.assertTrue(all(c['x'] is None for c in record['corners']))

    def test_panel_number_and_visibility_buttons(self):
        annotate_draw.build_panel(1000,0,self.state.kps_2d,None,0,120,1.,False)
        rect,value=next((r,v) for r,v in self.native.panel_hits if v==ord('5'))
        x,y=(rect[0]+rect[2])//2,(rect[1]+rect[3])//2
        annotate.on_mouse(annotate.cv2.EVENT_LBUTTONDOWN,self.state.img.shape[1]+annotate.MARGIN_L+annotate.MARGIN_R+x,y,0,self.state)
        self.assertEqual(self.state.active,5)
        rect,value=next((r,v) for r,v in self.native.panel_hits if v==ord('e'))
        x,y=(rect[0]+rect[2])//2,(rect[1]+rect[3])//2
        annotate.on_mouse(annotate.cv2.EVENT_LBUTTONDOWN,self.state.img.shape[1]+annotate.MARGIN_L+annotate.MARGIN_R+x,y,0,self.state)
        self.assertEqual(annotate.cv2.waitKey(20),ord('e'))
        self.key('e')
        self.assertTrue(self.native.record['corners'][5]['external_occlusion'])
        self.assertEqual(self.state.kps_2d,[None]*9)

    def test_exclusion_task_list_preserves_frozen_plan_and_records(self):
        path=self.fixture.root/'USER_EXCLUSIONS.json'
        path.write_text(json.dumps(dict(schema='lifter_user_exclusions_v1',status='EXCLUDED_BY_USER',
            input_bindings=self.fixture.ctx.bindings,excluded_frame_ids=['test:0'])))
        native=NativeReview(self.fixture.ctx,self.fixture.reviewer,self.fixture.root/'other',exclusions_path=path)
        from object_geometry_registry import load_object_geometry_registry
        registry=load_object_geometry_registry(ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')
        sessions,contexts=native.contexts('',argparse.Namespace(),registry,ROOT)
        self.assertEqual(sessions,[])
        self.assertEqual(contexts,{})
        self.assertEqual(native.requested_counts()['primary_required'],0)
        self.assertEqual(native.requested_counts()['repeat_required'],0)
        self.assertEqual(self.fixture.ctx.counts()['total_primary'],1)
        self.assertEqual(self.fixture.ctx.counts()['total_repeat'],1)
        self.assertFalse(self.fixture.store.exists())
        bad=json.loads(path.read_text());bad['input_bindings']['plan_sha256']='wrong'
        path.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            NativeReview(self.fixture.ctx,self.fixture.reviewer,self.fixture.root/'other',exclusions_path=path)

    def test_completed_record_edit_without_typing_keeps_history(self):
        self.click()
        for i in range(1,8): self.native.mark(self.state,i,'self_occlusion')
        self.key('s')
        self.native.load(self.state,self.path)
        self.key('0'); self.click(8,5)
        with patch('tkinter.simpledialog.askstring',side_effect=AssertionError('No typing required')):
            self.assertEqual(self.key('s'),'save-next')
        history=self.fixture.store.with_name(self.fixture.store.stem+'.history.jsonl')
        self.assertTrue(history.exists())
        self.assertEqual(json.loads(history.read_text())['record']['corners'][0]['x'],4.)
        self.assertEqual(self.fixture.ctx.export()['records'][0]['corners'][0]['x'],8.)

    def test_image_clicks_and_draft_save_do_not_require_profile(self):
        self.native.reviewer=None
        with patch('open_existing_annotation.choose_profile') as profile:
            self.native.load(self.state,self.path)
            self.click()
            profile.assert_not_called()
        before=copy.deepcopy(self.native.record['corners'])
        with patch('open_existing_annotation.choose_profile',side_effect=AssertionError('No profile popup')):
            self.assertTrue(self.native.save(self.state,self.path,'draft'))
        self.assertEqual(self.native.record['corners'],before)
        self.assertFalse(self.fixture.store.exists())
        self.assertTrue(self.native.recovery_path.is_file())
        self.assertIsNone(self.native.record['reviewer'])
        self.assertFalse(self.native.profile_path.exists())
        self.assertEqual(self.fixture.ctx.counts()['drafts'],0)
        self.assertEqual(self.fixture.ctx.export()['records'],[])

    @unittest.skipUnless(os.environ.get('DISPLAY'),'Nonvisible Tk layout requires an X display')
    def test_first_profile_all_controls_fit_without_mapping_or_saving(self):
        import tkinter as tk
        original_init=tk.Tk.__init__
        measurements=[]
        def hidden_init(window,*args,**kwargs):
            original_init(window,*args,**kwargs)
            window.withdraw()
        def inspect_without_loop(window,*args,**kwargs):
            try:
                window.update_idletasks()
                self.assertEqual(window.state(),'withdrawn')
                requested=(window.winfo_reqwidth(),window.winfo_reqheight())
                allocated=tuple(map(int,window.geometry().split('+')[0].split('x')))
                self.assertGreaterEqual(allocated[0],requested[0])
                self.assertGreaterEqual(allocated[1],requested[1]+24)
                descendants=[]
                def collect(widget):
                    for child in widget.winfo_children():
                        descendants.append(child);collect(child)
                collect(window)
                self.assertEqual(sum(w.winfo_class()=='Checkbutton' for w in descendants),3)
                self.assertFalse(any(w.winfo_class()=='Entry' for w in descendants))
                confirmations=[w for w in descendants if w.winfo_class()=='Button'
                    and getattr(w,'_korean_text_source','').startswith('확인하고 저장')]
                self.assertEqual(len(confirmations),1)
                self.assertGreater(confirmations[0].winfo_reqheight(),20)
                measurements.append(dict(requested=requested,allocated=allocated,
                    confirmation_button_y=confirmations[0].winfo_y(),
                    confirmation_button_height=confirmations[0].winfo_height()))
            finally:
                window.destroy()
        profile=self.fixture.root/'synthetic_profile_layout_only.json'
        with patch.object(tk.Tk,'__init__',hidden_init),patch.object(tk.Tk,'mainloop',inspect_without_loop):
            result=choose_profile(profile,preview=np.zeros((480,640,3),np.uint8))
        self.assertIsNone(result)
        self.assertTrue(measurements)
        self.assertFalse(profile.exists())
        self.assertFalse(self.fixture.store.exists())

    def test_actual_editor_main_renders_then_quits_without_annotation(self):
        import annotate_review
        frames=[]
        registry=ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
        with ExitStack() as stack:
            stack.enter_context(patch.object(annotate_review,'load_review_contexts',self.native.contexts))
            for name in ('namedWindow','resizeWindow','setMouseCallback','createTrackbar','setTrackbarPos','destroyAllWindows'):
                stack.enter_context(patch.object(annotate.cv2,name))
            stack.enter_context(patch.object(annotate.cv2,'imshow',side_effect=lambda name,image:frames.append(image.copy())))
            stack.enter_context(patch.object(annotate.cv2,'waitKey',return_value=ord('q')))
            annotate.main(['--review-manifest',str(self.fixture.manifest),'--stride','1',
                '--population-role','DEV','--default_split','eval','--object-type','plastic_square',
                '--geometry-registry',str(registry)])
        self.assertTrue(frames)
        self.assertFalse(self.fixture.store.exists())
        self.assertEqual(self.fixture.ctx.export()['records'],[])


if __name__=='__main__': unittest.main()
