"""Interaction and immutable-source checks, using temporary sidecars only."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

from .open_corner_visibility import VisibilityReview, ValidationError
from . import open_corner_visibility as visibility_module


def fixture():
    points=[]
    for index in range(8):
        points.append(dict(corner_id=index, metric_reference=index<3,
            locked=index==0, status='DIRECT_VISIBLE' if index==0 else 'UNREVIEWED',
            xy=[200+index*13,240], reference_source='locked_DEV_coordinate',
            annotation_coordinate_source='legacy',
            coordinate_origin='existing_frozen_annotation_not_new_independent_reference'))
    fid='DEV319::fixture'
    frame=dict(case_id=fid,frame_id='fixture',population='DEV319',image={'sha256':'image-fixture'},
        annotation={'sha256':'annotation-fixture'},current_frame_grade='clean',corners=points)
    return dict(frames={fid:frame},images={fid:Path('/tmp/fixture-unused.png')},
        bindings={'static_manifest_sha256':'manifest-fixture','severity_snapshot_sha256':'severity-fixture'})


class NativeVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'state.json'
        self.context=fixture();self.original=copy.deepcopy(self.context)
        self.native=VisibilityReview(self.context,self.path)
        self.nav=Path(self.tmp.name)/'navigation.json'
        self.native.path_map[str(self.nav.resolve())]='DEV319::fixture'
        self.state=SimpleNamespace(img=np.zeros((480,640,3),np.uint8),zoom=1.,pan=[0,0])
        self.native.load(self.state,self.nav)
        self.editor=SimpleNamespace(cv2=SimpleNamespace(waitKey=lambda _:255,
            resize=cv2.resize,INTER_LINEAR=cv2.INTER_LINEAR,INTER_NEAREST=cv2.INTER_NEAREST,
            EVENT_LBUTTONDOWN=cv2.EVENT_LBUTTONDOWN))
        self.native.install(self.editor)

    def tearDown(self):
        self.tmp.cleanup()

    def add_geometry_fixture(self):
        points = self.context['frames']['DEV319::fixture']['corners']
        points[1]['xy'] = [-2., 240.]
        for i, status in [(1,'OUT_OF_FRAME'), (2,'SELF_OCCLUDED')]:
            points[i]['geometry_proposal'] = dict(status=status,
                source='temporary_geometry_fixture', evidence={'reference_xy':points[i]['xy']},
                machine_proposal_only=True, human_confirmation_required=True)

    def test_save_restart_restores_image_corner_and_view_without_new_labels(self):
        self.editor.render(self.state,0,1,'fixture')
        self.native.choose('2')
        before=copy.deepcopy(self.native.store['records'])
        history=copy.deepcopy(self.native.store['history'])
        self.state.zoom,self.state.pan=2.,[17,31]
        self.assertIsNone(self.editor._handle_click_key(ord('s'),self.state))
        self.assertEqual(self.editor._handle_click_key(ord('t'),self.state),'quit')
        self.assertTrue(self.native.restart_requested)
        reloaded=VisibilityReview(self.context,self.path)
        registry=SimpleNamespace(resolve=lambda _:SimpleNamespace(object_type='fixture'))
        sessions,contexts=reloaded.contexts(None,SimpleNamespace(),registry,None)
        image=Path(contexts[sessions[0][2]]['frame_paths'][0])
        navigation=Path(contexts[sessions[0][2]]['out_dir'])/(image.stem+'.json')
        fresh=SimpleNamespace(img=np.zeros((480,640,3),np.uint8))
        reloaded.load(fresh,navigation)
        self.assertEqual(reloaded.current,'DEV319::fixture')
        self.assertEqual(reloaded.cursor,2)
        self.assertEqual(fresh.zoom,2.)
        self.assertEqual(fresh.pan,[17,31])
        self.assertEqual(reloaded.store['records'],before)
        self.assertEqual(reloaded.store['history'],history)
        reloaded.install(self.editor)
        self.editor.render(fresh,0,1,'fixture')  # Restored pan values remain valid slice indices.

    def test_resume_later_frame_in_preferred_population_then_next_pending(self):
        context=fixture()
        for fid,population in [('GREEN0918::square','GREEN0918'),('DEV319::later','DEV319')]:
            row=copy.deepcopy(context['frames']['DEV319::fixture'])
            row.update(case_id=fid,frame_id=fid.split('::')[1],population=population)
            context['frames'][fid]=row
            context['images'][fid]=Path('/tmp/'+fid.replace('::','-')+'.png')
        native=VisibilityReview(context,self.path)
        later=Path(self.tmp.name)/'later.json'
        native.path_map[str(later.resolve())]='DEV319::later'
        native.load(self.state,later)
        native.save_view(self.state)
        registry=SimpleNamespace(resolve=lambda _:SimpleNamespace(object_type='fixture'))
        sessions,contexts=native.contexts(None,SimpleNamespace(),registry,None)
        self.assertEqual(native.population_sessions['DEV319'],0)
        self.assertEqual(contexts[sessions[0][2]]['frame_paths'][0],'/tmp/DEV319-later.png')
        native.install(self.editor)
        self.editor.render(self.state,0,1,'fixture')
        native.choose('1',all_remaining=True)
        native.save_view(self.state)
        resumed=VisibilityReview(context,self.path)
        sessions,contexts=resumed.contexts(None,SimpleNamespace(),registry,None)
        self.assertEqual(resumed.population_sessions['DEV319'],0)
        self.assertEqual(contexts[sessions[0][2]]['frame_paths'][0],'/tmp/fixture-unused.png')

    def test_invalid_bookmark_is_ignored_and_quit_saves_without_grading(self):
        self.editor.render(self.state,0,1,'fixture')
        bookmark=json.loads(self.native.resume_path.read_text())
        bookmark['input_bindings']['static_manifest_sha256']='stale'
        self.native.resume_path.write_text(json.dumps(bookmark))
        resumed=VisibilityReview(self.context,self.path)
        self.assertEqual(resumed.resume,{})
        self.assertEqual(resumed.store['records'],{})
        self.assertEqual(self.editor._handle_click_key(ord('q'),self.state),'quit')
        self.assertEqual(self.native.store['records'],{})
        self.assertEqual(self.native.store['history'],[])

    def test_geometry_exposure_alone_never_accepts_proposals(self):
        self.add_geometry_fixture()
        self.editor.render(self.state,0,1,'fixture')
        self.assertEqual(self.native.counts()['pending_geometry_proposals'],2)
        self.assertEqual(self.native.store['records'],{})
        self.assertTrue(self.native.store['exposures']['DEV319::fixture']['geometry_proposals_displayed'])
        self.assertEqual(self.native.counts()['human_input_points'],0)

    def test_explicit_geometry_button_accepts_mixed_states_and_can_undo(self):
        self.add_geometry_fixture()
        before = copy.deepcopy(self.context)
        self.editor.render(self.state,0,1,'fixture')
        box,_ = next(row for row in self.native.buttons if row[1]=='c')
        self.editor.on_mouse(cv2.EVENT_LBUTTONDOWN,box[0]+5,box[1]+5,0,self.state)
        self.assertEqual(self.editor.cv2.waitKey(20),ord('c'))
        self.assertEqual(self.editor._handle_click_key(ord('c'),self.state),'save-next')
        rows = self.native.store['records']['DEV319::fixture']['corners']
        self.assertEqual({k:r['status'] for k,r in rows.items()}, {'1':'OUT_OF_FRAME','2':'SELF_OCCLUDED'})
        self.assertTrue(all(r['machine_assistance'] for r in rows.values()))
        self.assertTrue(all(r['human_input_status']=='HUMAN_CONFIRMED_GEOMETRY_PROPOSAL' for r in rows.values()))
        self.assertTrue(all(r['geometry_proposal']['machine_proposal_only'] for r in rows.values()))
        self.assertEqual(self.context,before)
        self.native.undo()
        self.assertEqual(self.native.counts()['pending'],2)
        self.assertEqual(self.native.state_for('DEV319::fixture',0),'DIRECT_VISIBLE')

    def test_geometry_confirmation_preserves_existing_human_choice(self):
        self.add_geometry_fixture()
        self.editor.render(self.state,0,1,'fixture')
        self.native.cursor=2
        self.native.choose('2')
        previous=copy.deepcopy(self.native.store['records']['DEV319::fixture']['corners']['2'])
        self.editor._handle_click_key(ord('c'),self.state)
        rows = self.native.store['records']['DEV319::fixture']['corners']
        self.assertEqual(rows['2'],previous)
        self.assertEqual(rows['1']['status'],'OUT_OF_FRAME')
        self.native.undo()
        self.assertEqual(self.native.store['records']['DEV319::fixture']['corners'],{'2':previous})

    def test_open_and_render_never_create_visibility(self):
        self.assertFalse(self.path.exists())
        frame=self.editor.render(self.state,0,1,'fixture')
        self.assertEqual(frame.shape,(950,1430,3))
        self.assertEqual(self.native.store['records'],{})
        self.assertTrue(self.native.store['exposures']['DEV319::fixture']['reference_overlay_exposure'])
        self.assertEqual(self.native.counts()['pending'],2)
        self.assertEqual(self.context,self.original)

    def test_no_selection_without_display(self):
        with self.assertRaisesRegex(ValidationError,'not displayed'):
            self.native.choose('1')
        self.assertEqual(self.native.store['records'],{})

    def test_keys_save_each_point_then_advance_without_changing_coordinates(self):
        self.editor.render(self.state,0,1,'fixture')
        self.assertIsNone(self.editor._handle_click_key(ord('2'),self.state))
        self.assertEqual(self.native.cursor,2)
        action=self.editor._handle_click_key(ord('3'),self.state)
        self.assertEqual(action,'save-next')
        saved=json.loads(self.path.read_text())
        rows=saved['records']['DEV319::fixture']['corners']
        self.assertEqual(rows['1']['status'],'EXTERNAL_OCCLUDED')
        self.assertEqual(rows['2']['status'],'SELF_OCCLUDED')
        self.assertNotIn('0',rows)
        self.assertEqual(self.context,self.original)
        reloaded=VisibilityReview(self.context,self.path)
        self.assertEqual(reloaded.counts()['pending'],0)

    def test_mouse_button_has_same_save_route(self):
        self.editor.render(self.state,0,1,'fixture')
        self.editor.on_mouse(cv2.EVENT_LBUTTONDOWN,1000,165,0,self.state)
        self.assertEqual(self.editor.cv2.waitKey(20),ord('1'))
        self.editor._handle_click_key(ord('1'),self.state)
        row=self.native.store['records']['DEV319::fixture']['corners']['1']
        self.assertEqual(row['input_action'],'human_mouse_button')

    def test_explicit_all_visible_preserves_locked_and_optional_points(self):
        self.editor.render(self.state,0,1,'fixture')
        self.assertEqual(self.editor._handle_click_key(ord('a'),self.state),'save-next')
        rows=self.native.store['records']['DEV319::fixture']['corners']
        self.assertEqual(set(rows),{'1','2'})
        self.assertEqual({r['status'] for r in rows.values()},{'DIRECT_VISIBLE'})
        self.assertEqual(self.native.state_for('DEV319::fixture',0),'DIRECT_VISIBLE')

    def test_undo_reset_and_unknown_distinction(self):
        self.editor.render(self.state,0,1,'fixture')
        self.editor._handle_click_key(ord('5'),self.state)
        self.assertEqual(self.native.counts()['unresolved_human_unknown'],1)
        self.editor._handle_click_key(ord('z'),self.state)
        self.assertEqual(self.native.counts()['pending'],2)
        self.editor._handle_click_key(ord('a'),self.state)
        self.editor._handle_click_key(ord('r'),self.state)
        self.assertEqual(self.native.counts()['pending'],2)
        self.assertEqual(self.native.state_for('DEV319::fixture',0),'DIRECT_VISIBLE')
        self.editor._handle_click_key(ord('z'),self.state)
        self.assertEqual(self.native.counts()['pending'],0)

    def test_locked_edit_and_resumed_contract_tampering_rejected(self):
        self.editor.render(self.state,0,1,'fixture')
        self.native.cursor=0
        with self.assertRaisesRegex(ValidationError,'Locked'):
            self.native.choose('2')
        # An explicit rejected action must not leave a phantom record.
        self.native.cursor=1;self.native.choose('1')
        changed=json.loads(self.path.read_text())
        changed['records']['DEV319::fixture']['corners']['1']['reference_xy']=[999,999]
        self.path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValidationError,'Invalid saved corner'):
            VisibilityReview(self.context,self.path)

    def test_resumed_queue_skips_entered_frames_and_maps_population_buttons(self):
        context=fixture()
        for fid,population in [('GREEN0918::entered','GREEN0918'),
                               ('DEV319::entered','DEV319')]:
            row=copy.deepcopy(context['frames']['DEV319::fixture'])
            row.update(case_id=fid,frame_id=fid.split('::')[1],population=population)
            context['frames'][fid]=row
            context['images'][fid]=Path('/tmp/'+population+'-entered-unused.png')
        native=VisibilityReview(context,self.path)
        native.install(self.editor)
        for fid in ('GREEN0918::entered','DEV319::entered'):
            path=Path(self.tmp.name)/(fid.replace('::','-')+'.json')
            native.path_map[str(path.resolve())]=fid
            native.load(self.state,path)
            self.editor.render(self.state,0,1,'temporary fixture')
            if fid.startswith('GREEN'):
                # Explicit UNKNOWN inputs are entered, but remain unresolved.
                native.choose('5');native.choose('5')
            else:
                native.choose('1',all_remaining=True)
        registry=SimpleNamespace(resolve=lambda _:SimpleNamespace(object_type='fixture'))
        sessions,contexts=native.contexts(None,SimpleNamespace(),registry,None)
        self.assertEqual(len(sessions),1)
        self.assertEqual(native.population_sessions,{'DEV319':0})
        self.assertEqual(contexts[sessions[0][2]]['frame_paths'],
                         [str(context['images']['DEV319::fixture'])])
        buttons=native.population_buttons()
        self.assertFalse(buttons[0]['enabled'])
        self.assertTrue(buttons[0]['complete'])
        self.assertEqual(buttons[1]['session_index'],0)
        self.assertTrue(buttons[1]['enabled'])
        nav=native.workspace/'DEV319'/'fixture-unused.json'
        native.load(self.state,nav)
        self.editor.render(self.state,0,1,'temporary fixture')
        self.state.sess_pick=None
        self.editor.on_mouse(cv2.EVENT_LBUTTONDOWN,990,20,0,self.state)
        self.assertIsNone(self.state.sess_pick)  # Completed square button is disabled.
        self.editor.on_mouse(cv2.EVENT_LBUTTONDOWN,1200,20,0,self.state)
        self.assertEqual(self.state.sess_pick,0)  # Rectangular is now native session zero.
        self.assertEqual(native.counts()['unresolved_human_unknown'],2)
        optional=VisibilityReview(context,self.path,show_completed=True)
        all_sessions,all_contexts=optional.contexts(None,SimpleNamespace(),registry,None)
        self.assertEqual(optional.population_sessions,{'DEV319':0,'GREEN0918':1})
        self.assertEqual(sum(len(c['frame_paths']) for c in all_contexts.values()),3)
        self.assertEqual(len(all_sessions),2)
        self.assertTrue(all(button['enabled'] for button in optional.population_buttons()))
        self.assertEqual(optional.state_for('GREEN0918::entered',1),'UNKNOWN')
        self.assertEqual(optional.counts()['unresolved_human_unknown'],2)

    def test_all_entered_reports_no_pending_without_opening_gui(self):
        self.editor.render(self.state,0,1,'fixture')
        self.native.choose('5');self.native.choose('5')
        output=Path(self.tmp.name)/'STATIC_CORNER_VISIBILITY_INPUTS.json'
        output.write_bytes(self.path.read_bytes())
        before=output.read_bytes()
        stdout=io.StringIO()
        with mock.patch.object(visibility_module,'OUT',Path(self.tmp.name)), \
             mock.patch.object(visibility_module,'load_context',return_value=self.context), \
             mock.patch.object(visibility_module.sys,'argv',['open_corner_visibility.py']), \
             mock.patch.object(visibility_module.fcntl,'flock') as lock, \
             redirect_stdout(stdout):
            visibility_module.main()
        result=json.loads(stdout.getvalue())
        self.assertEqual(result['status'],'NO_PENDING_VISIBILITY_INPUT')
        self.assertEqual(result['counts']['pending'],0)
        self.assertEqual(result['counts']['unresolved_human_unknown'],2)
        lock.assert_not_called()
        self.assertEqual(output.read_bytes(),before)


if __name__=='__main__':unittest.main()
