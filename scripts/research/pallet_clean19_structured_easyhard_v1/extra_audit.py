"""Additional artifact-level audits, no optimization or selection."""
from collections import Counter
import json
import numpy as np
from . import common as C
from . import augmentation as A


def main():
    plans=[json.loads(x) for x in (C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()]
    targets={r['index']:r for r in C.read(C.DOC/'TARGETS.json')};stats={}
    max_difference=0.;max_roundtrip=0.;checked=0
    for mat in C.MATERIALS:
        stats[mat]={arm:dict(masked_channels=Counter(),masked_edges=Counter(),masked_count_hist=Counter(),remaining_count_hist=Counter(),applied=0) for arm in ('S1','S2')}
    for p in plans:
        if not p['real']:continue
        r=targets[p['target_index']];z=np.load(C.ROOT/p['cache']['path']);y=np.array(r['target']);transform=np.array(p['native_to_model'])
        q=np.c_[y,np.ones(9)]@transform.T
        back=np.c_[q[:,:2],np.ones(9)]@np.linalg.inv(transform).T
        max_roundtrip=max(max_roundtrip,float(np.max(np.abs(back[:,:2]-y))))
        assert np.allclose(np.clip(q[:,:2],0,640),z['keypoints'][0,:,:2]*640,atol=.001)
        mask=z['keypoints'][0,:,2]==2
        for arm in ('S1','S2'):
            new=A.apply(z['img'],p['plan'],arm)
            if p['plan']['applied']:
                rect=p['plan'][arm];l,t,w,h=rect;canvas=p['plan']['native_canvas']
                assert l>=max(0,canvas[0]) and t>=max(0,canvas[1]) and l+w<=min(640,canvas[2]) and t+h<=min(640,canvas[3])
                mm=A.cover(z['keypoints'][0,:,:2]*640,rect)&mask
                assert mm.sum()>=1 and mask.sum()-mm.sum()>=2
                if arm=='S2':assert any(mm[a] and mm[b] for a,b in A.EDGES)
                d=stats[p['material']][arm];d['applied']+=1
                d['masked_channels'].update(map(str,np.flatnonzero(mm)));d['masked_count_hist'].update([str(int(mm.sum()))]);d['remaining_count_hist'].update([str(int(mask.sum()-mm.sum()))])
                d['masked_edges'].update(f'{a}-{b}' for a,b in A.EDGES if mm[a] and mm[b])
                assert np.array_equal(new[:,t:t+h,l:l+w],A.fill(p['plan']))
                # RGB outside rectangle is exactly unchanged.
                outside=np.ones((640,640),bool);outside[t:t+h,l:l+w]=False
                assert np.array_equal(new[:,outside],z['img'][:,outside])
                max_difference=max(max_difference,p['plan']['bbox_overlap_difference'])
            else:assert np.array_equal(new,z['img'])
        checked+=1
    C.save(C.DOC/'AUGMENTATION_FREQUENCY.json',dict(material=stats,real_occurrences_checked=checked,max_bbox_overlap_difference=max_difference,max_unclipped_roundtrip_px=max_roundtrip,
        all_rectangles_inside_native_canvas=True,all_outside_RGB_unchanged=True,all_masks_after_occlusion_identical=True))
    C.save(C.DOC/'EVALUATION_CODE_LOCK.json',dict(files=[C.bind(p) for p in sorted(C.HERE.glob('*.py'))],reporting_code_locked=True))
    print('EXTRA_AUDIT_PASS',checked,flush=True)


if __name__=='__main__':main()
