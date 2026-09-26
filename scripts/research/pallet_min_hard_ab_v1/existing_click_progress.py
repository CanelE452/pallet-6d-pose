"""Check the existing editor's direct-click pass; never infer role or hidden labels."""
from . import common as C
from .open_existing_annotation import WORK,rows

def summarize():
    with C.exclusive('annotating'):
        selected=rows();saved=0;clicks=0;assisted=0
        for r in selected:
            path=WORK/'annotations'/((C.ROOT/r['image']['path']).stem+'.json')
            if not path.exists():continue
            d=C.read(path)
            assert d['schema'] in ('hard_direct_clicks_v1','hard_pnp_assisted_v1') and d['image']==r['image']
            assert d['selection_sha256']==C.sha(C.DOC/'HARD_SELECTION_LOCK.json')
            for p,a in zip(d['manual_kps'][:8],d['keypoint_annotations'][:8]):
                if p is not None:
                    assert a['source'] in ('manual_click','pnp_projected','extrapolated')
                    clicks+=int(a['source']=='manual_click');assisted+=int(a['source']!='manual_click')
            saved+=1
        summary=dict(saved_frames=saved,required_frames=len(selected),direct_clicks=clicks,
                     assisted_points=assisted,reference_protocol='HUMAN_PNP_ASSISTED_NOT_BLIND',
                     final_labels_locked=False,role_visibility_bbox_confirmation='PENDING',training='NOT_RUN')
        C.save(C.DOC/'EXISTING_CLICK_PROGRESS_PUBLIC.json',summary)
        s=C.state();s['status']='WAITING_FOR_HUMAN_HARD_METADATA' if saved==len(selected) else 'WAITING_FOR_EXISTING_ANNOTATION_KEYPOINTS'
        s['command']='python -m scripts.research.pallet_min_hard_ab_v1.open_existing_annotation'
        C.save(C.RAW/'STATE.json',s);C.save(C.DOC/'STATUS.json',s)
        print('EXISTING_EDITOR_CLICKS',summary)
