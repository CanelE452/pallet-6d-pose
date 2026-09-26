"""Use the user's existing annotate.py for the locked eight-frame click pass.

Only process-local hooks: raw/manual points, partial save without PnP,
isolated outputs. No changes to the shared editor or historical annotations.
Role/visibility/bbox confirmation remains a separate later human step.
"""
import argparse
import copy
from pathlib import Path
import sys
import tempfile
import numpy as np
from . import common as C

WORK=C.RAW/'existing_annotation'

def rows():
    lock=C.read(C.DOC/'HARD_SELECTION_LOCK.json');C.verify(lock['private_selection'])
    return C.read(C.ROOT/lock['private_selection']['path'])['rows'][:C.state()['active_frames']]

def contexts(manifest,cli_args,registry,repo):
    selected=rows();spec=registry.resolve('plastic');args=copy.copy(cli_args)
    args.object_type=spec.object_type;args.population_role='DEV';args.default_split='train'
    args.capture_session_id='MIN_HARD8_DIRECT_CLICKS';args.lighting_condition=None
    args.intrinsics_quality='UNKNOWN';args.intrinsics_source='per-frame cam_K metadata; PnP disabled in this click pass'
    mapping=[]
    for r in selected:
        C.verify(r['image']);image=C.ROOT/r['image']['path'];kp=image.parent.parent/'cam_K.txt'
        K=np.loadtxt(kp).reshape(3,3)
        mapping.append(dict(frame_id=r['frame_id'],image=r['image'],recording=r['recording'],
                            K=K.tolist(),camera=C.bind(kp),output=str((WORK/'annotations'/(image.stem+'.json')).relative_to(C.ROOT))))
    assert len({Path(r['output']).name for r in mapping})==len(mapping)
    key='review:MIN_HARD8_DIRECT_CLICKS'
    context=dict(args=args,metadata=dict(population_role='DEV',object_type=spec.object_type),
        geometry_spec=spec,out_dir=str(WORK/'annotations'),K=np.array(mapping[0]['K']),K_source=args.intrinsics_source,
        frame_paths=[str(C.ROOT/r['image']['path']) for r in selected],frame_count=len(selected),
        writable=True,workspace_scope=None,display_role='DEV',source_session_dir=str(WORK),
        refresh_evaluation=False,force_explicit_object_type=True,active_evaluation_member=False)
    payload=dict(selection=C.bind(C.DOC/'HARD_SELECTION_LOCK.json'),frames=mapping,
                 editor='scripts/annotate/annotate.py',PnP_enabled=False,legacy_loaded=False,predictions_loaded=False,
                 phase='DIRECT_CLICKS_FIRST; not final training labels; role/bbox/visibility remain pending')
    path=WORK/'WORKSPACE.json'
    if path.exists():assert C.read(path)==payload
    else:C.save(path,payload,immutable=True)
    return [('MIN_HARD8_DIRECT_CLICKS',str(WORK),key)],{key:context}

def install(editor):
    original_key=editor._handle_click_key;original_mouse=editor.on_mouse;original_render=editor.render
    lookup={str((C.ROOT/r['image']['path']).resolve()):r for r in rows()}
    def no_pose(s,K,force=False):s.pose=None;s.locked_pose=None
    def load(s,path,read_only=False):
        p=Path(path)
        assert p.resolve().is_relative_to(WORK.resolve())
        if not p.exists():return False
        d=C.read(p);assert d['schema']=='hard_direct_clicks_v1'
        s.kps_2d=copy.deepcopy(d['manual_kps']);s.keypoint_annotations=copy.deepcopy(d['keypoint_annotations'])
        s.extrap_mask=[False]*9;s.pose=None;s.split='train';s.population_role='DEV'
        return True
    def handle(key,s,out_json,out_png,src_png,K):
        if key in map(ord,'fgctxw8/'):
            editor._toast(s,'DIRECT CLICKS ONLY: auto-fill/pose disabled',seconds=1.5);return None
        if key!=ord('s'):return original_key(key,s,out_json,out_png,src_png,K)
        target=Path(out_json);assert target.resolve().is_relative_to(WORK.resolve())
        r=lookup[str(Path(src_png).resolve())];points=copy.deepcopy(s.kps_2d);points[8]=None
        ann=copy.deepcopy(editor._ensure_keypoint_annotations(s));ann[8]=dict(xy=None,source='unknown',visibility=0,reason='unknown',in_frame=False)
        for i,p in enumerate(points[:8]):
            if p is not None:
                assert ann[i]['source']=='manual_click','Only direct clicks are accepted'
                assert 0<=p[0]<s.img_shape[1] and 0<=p[1]<s.img_shape[0],'Click must be inside raw image'
        C.save(target,dict(schema='hard_direct_clicks_v1',frame_id=r['frame_id'],image=r['image'],
               selection_sha256=C.sha(C.DOC/'HARD_SELECTION_LOCK.json'),manual_kps=points,
               keypoint_annotations=ann,pose=None,PnP_used=False,metadata_confirmation_pending=True,updated_at=C.now()))
        s.dirty=False;s.annotation_dirty=False;s.frame_tags_dirty=False;s.discard_armed=None
        print('SAVED_DIRECT_CLICKS',r['frame_id'],sum(p is not None for p in points),flush=True)
        return 'save-next'
    def mouse(event,x,y,flags,s):
        if s.active==8:s.active=7
        original_mouse(event,x,y,flags,s)
        s.active=min(7,s.active)
    def render(s,*args):
        vis=original_render(s,*args)
        editor.cv2.putText(vis,'HARD8 | click visible 0..7 | numbers: select | S: save+next | no auto-fill',
                           (15,vis.shape[0]-12),editor.cv2.FONT_HERSHEY_SIMPLEX,.55,(0,240,240),1,editor.cv2.LINE_AA)
        return vis
    editor.update_pose=no_pose;editor.load_existing_annotation=load
    editor._handle_click_key=handle;editor.on_mouse=mouse;editor.render=render

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if (C.DOC/'HARD_LABEL_LOCK.json').exists():raise SystemExit('Human labels locked; no editing.')
    sys.path.insert(0,str(C.ROOT/'scripts/annotate'))
    import annotate
    import annotate_review
    from object_geometry_registry import load_object_geometry_registry,DEFAULT_REGISTRY_PATH
    with C.exclusive('annotating'):
        sessions,result=contexts('',argparse.Namespace(),load_object_geometry_registry(DEFAULT_REGISTRY_PATH),C.ROOT)
        if args.prepare_only:print('READY existing annotate.py',len(rows()),'fixed frames; isolated direct-click outputs');return
        install(annotate)
        if args.smoke:
            global WORK
            original=WORK;C.OUT.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='existing_editor_smoke_',dir=C.OUT) as tmp:
                WORK=Path(tmp);s=annotate.State();s.img_shape=(480,640,3);s.kps_2d=[[20.,30.],[80.,30.]]+[None]*7
                s.keypoint_annotations=None;s.extrap_mask=[False]*9
                for i in (0,1):annotate._set_keypoint_state(s,i,s.kps_2d[i],source='manual_click',visibility=2,reason='visible')
                annotate.update_pose(s,np.eye(3));assert s.pose is None
                path=WORK/'test.json';src=str(C.ROOT/rows()[0]['image']['path'])
                assert annotate._handle_click_key(ord('s'),s,str(path),'',src,np.eye(3))=='save-next'
                d=C.read(path);assert d['manual_kps'][2:]==[None]*7 and d['PnP_used'] is False
                s.kps_2d=[None]*9;assert annotate.load_existing_annotation(s,str(path));assert s.kps_2d[0]==[20.,30.]
                assert annotate._handle_click_key(ord('g'),s,str(path),'',src,np.eye(3)) is None
            WORK=original;print('PASS existing editor partial2-point save/reload, no hidden fill, no PnP; actual human labels untouched');return
        annotate_review.load_review_contexts=contexts
        annotate.main(['--review-manifest',str(C.DOC/'HARD_SELECTION_LOCK.json'),'--stride','1',
                       '--population-role','DEV','--default_split','train','--object-type','plastic'])

if __name__=='__main__':main()
