"""Rescore saved R0/P/D/L only, separating metric changes from model changes."""
import math
import numpy as np
import cv2
import env as E
from metrics import measure,summarize
def main():
    pe=E.C.old('paper_evaluation');population=pe.population();base=E.read(E.C.LINE/'baseline/FULL_CANDIDATES.json')
    perms=np.array(E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][0]['permutations'])
    targets={};meta=E.read(pe.POS)['items'];meta={r['frame_id']:r for r in meta}
    for item in population.positive.items:
        t=pe.E._legacy_forbidden_target(item)
        im=cv2.imread(str(E.ROOT/item.image));assert im is not None
        targets[item.frame_id]=(t,im.shape[:2])
    stores={};summary={}
    for arm in ['R0']+[f'{a}{s}' for a in ['P','D','L'] for s in [1,2,3]]:
        if arm=='R0':path=E.C.LINE/'baseline/FULL_CANDIDATES.json'
        elif arm[0]=='P':path=E.C.BRAW/f'evaluation/{arm}/PREDICTIONS.json'
        elif arm[0]=='D':path=E.ROOT/f'data/pallet/results/pallet_sensors_refinement_closeout_v1/evaluation/{arm}/PREDICTIONS.json'
        else:path=E.C.LINE/f'evaluation/image_line_only_seed{arm[1:]}/PREDICTIONS.json'
        if not path.exists():summary[arm]=dict(status='MISSING_STORED_PREDICTIONS');continue
        frames=E.read(path)['frames'];rows=[];legacy=[]
        for item in population.positive.items:
            t,hw=targets[item.frame_id];key=pe.canonical_key(item.image);cs=frames[key]
            top=max(cs,key=lambda c:c['score']) if cs else None
            matched=top is not None and top['keypoints_xy'] is not None and pe.E._box_iou(np.array(top['box_xyxy']),t.box_xyxy)>=.5
            pred=np.array(top['keypoints_xy']) if matched else np.full((9,2),np.nan)
            r=measure(pred,t.keypoints_xy,t.keypoint_supervision_mask,perms,hw)
            r.update(frame_id=item.frame_id,session_id=meta[item.frame_id]['session_id'],matched=matched)
            if matched:legacy.extend(np.linalg.norm(pred-t.keypoints_xy,axis=-1)[t.keypoint_supervision_mask].tolist())
            rows.append(r)
        stores[arm]=rows;summary[arm]=dict(status='COMPLETE',**summarize(rows),
            LEGACY_matched_pooled9_median=float(np.median(legacy)),matched_frames=sum(r['matched'] for r in rows),
            predictions=E.bound(path),new_inferences=0,metric_change_only=True)
    E.write(E.RAW/'A/A0_PAPER_FULL_METRICS.json',stores)
    E.write(E.DOC/'A/A0_PAPER.json',dict(arms=summary,frames=319,negative_population=2689,
      no_new_inference=True,GT_denominator_preserved=True,paper_modified=False,square_population_mixed=False))
    print('A0 PAPER', {k:(v.get('primary_E_sym'),v.get('pooled_point_median_px')) for k,v in summary.items()},flush=True)
if __name__=='__main__':main()
