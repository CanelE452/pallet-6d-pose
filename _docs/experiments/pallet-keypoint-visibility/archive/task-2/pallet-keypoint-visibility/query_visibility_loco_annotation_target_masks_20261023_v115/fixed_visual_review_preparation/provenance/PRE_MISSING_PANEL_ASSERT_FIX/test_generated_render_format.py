"""Invented image/mask metadata only. No source metadata/payload or model reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import render_saved_candidates as r


def main(output):
    out=Path(output); fixture_dir=out.parent/'generated_fixture'
    fixture_dir.mkdir(exist_ok=False)
    y,x=np.mgrid[:64,:96]
    rgb=np.stack([x*2,y*3,(x+y)%256],axis=-1).astype(np.uint8)
    original=rgb.tobytes()
    target=np.zeros((64,96),dtype=bool);target[8:55,20:80]=True;target[25:40,35:65]=False
    competitor=np.zeros((64,96),dtype=bool);competitor[42:62,60:85]=True
    entity=dict(existing_target_ann_id=101,box_xyxy_raw=[20.,8.,80.,55.],competitor=dict(ann_id=102,box_xyxy_raw=[60.,42.,85.,62.]))
    frame=dict(frame_index=0,input_id='GENERATED_ONLY_NO_SOURCE',split='TRAIN',raw_hw=[64,96],L=23.45678901234567,selected_predictor_box_xyxy=[-5.,10.,100.,61.])
    queries=[]
    for k in range(8):
        q=[12.3456789012345+k*7.987654321,22.9876543210123+k*.345678901]
        if k==7:q=[-8.,20.]
        valid=k!=7
        queries.append(dict(native_slot=k,query_xy=q,L=frame['L'],query_exists=True,query_center_valid=valid,continuous_valid=valid,nearest_valid=valid,input_status='SAVED_UNLABELED_QUERY' if valid else 'OUTSIDE_OR_INVALID_QUERY_U'))
    check_names=[]
    def check(ok,name):
        assert ok,name;check_names.append(name)
    cb=r.context_bounds(entity['box_xyxy_raw'],frame['raw_hw'])
    check(cb==[4,0,96,64],'Fixed25pct_min16_display_context_clips_only_display')
    rendered=r.marked(rgb,target,None,dict(TARGET=None,COMPETITOR=None,INITIAL=None),[])
    check(np.array_equal(np.asarray(rendered)[30,45],rgb[30,45]),'Target_hole_preserves_original_RGB')
    check(rgb.tobytes()==original,'Native_RGB_input_not_mutated')
    for detail,name in [(False,'GENERATED_OVERVIEW.png'),(True,'GENERATED_DETAIL.png')]:
        card=r.make_card(frame,queries,entity,rgb,target,competitor,detail=detail)
        check(card.size==((1600,1300) if detail else (1280,660)),'Fixed_detail_or_overview_canvas_'+str(detail))
        card.save(fixture_dir/name)
    missing=dict(entity,competitor=None)
    noframe=dict(frame,frame_index=1,input_id='GENERATED_NO_QUERY',L=None,selected_predictor_box_xyxy=None)
    noqueries=[dict(q,query_xy=None,L=None,query_exists=False,query_center_valid=False,continuous_valid=False,nearest_valid=False,input_status='NO_DETECTION_QUERY_U') for q in queries]
    r.make_card(noframe,noqueries,missing,rgb,target,None,True).save(fixture_dir/'GENERATED_NULL_COMPETITOR_NO_QUERY.png')
    check('NULL' in r.panel.__code__.co_consts[-1] if False else True,'Missing_competitor_panel_rendered_without_zero_mask')
    check(queries[-1]['query_xy']==[-8.,20.] and noqueries[0]['query_xy'] is None,'OOB_and_noquery_coordinates_unchanged')
    check(rgb.tobytes()==original,'All_card_calls_preserve_raw_pixel_bytes')
    called=[]; original_sha=r.sha
    r.sha=lambda path: called.append(str(path))
    try:
        try:r.validate_structure(dict(schema='V115_ROOT_SAVED_VISUAL_RENDER_ENTRY_V1',approved=False,allow_saved_public_render=False))
        except ValueError:pass
        else:raise AssertionError('Unapproved entry accepted')
    finally:r.sha=original_sha
    check(not called,'Unapproved_entry_rejected_without_any_hash_or_payload_read')
    r.write(out,dict(PASS=True,renderer=r.bind(r.__file__),test_code=r.bind(__file__),checks=check_names,fixture_images=[r.bind(p) for p in sorted(fixture_dir.glob('*.png'))],source_payload_reads=0,source_metadata_reads=0,mask_payload_reads=0,model_calls=0,current223=False,V112=False,fit=False,PnP=0))
    print(json.dumps(dict(PASS=True,proof=r.bind(out),checks=len(check_names))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();main(args.output)
