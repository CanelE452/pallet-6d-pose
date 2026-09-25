"""Reproduce locked S1+GEO_LINEAR reference before adaptation."""
import numpy as np
from . import common as C
def main():
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary
    from scripts.research.pallet_verified_anchor_v1.evaluate import metrics
    rr=V.records();groups=V.groups(rr);raw=C.read(C.PREV_RAW/'PREDICTIONS.json')['S1'];fm=C.read(C.PREV_RAW/'FRAME_METRICS.json')['S1'];pm=C.read(C.PREV_RAW/'POSE_METRICS.json')['S1'];dec=C.read(C.P.sraw(3)/'REAL_SCORER_DECISIONS.json')['S1'];published=C.read(C.P.sdoc(3)/'REAL_SCORER_RESULTS.json')['groups'];truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');out={};rebuilt={}
    for fid,p in raw.items():
        t=truth[fid];c=C.selected(p);matched=c is not None and V.E.C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
        rebuilt[fid]=dict(id=fid,**V.E.P.M.measure(np.full((9,2),np.nan) if c is None else c['keypoints_xy'],t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None));V.E.D.close(rebuilt[fid],fm[fid])
    for g,ids in groups.items():
        chosen=[next(h['metric'] for h in pm[i]['hypotheses'] if h['name']==dec[i]['selected']) for i in ids];cur=V.E.D.aggregate(chosen);oracle=V.E.D.aggregate([pm[i]['oracle'] for i in ids]);V.E.D.close(cur,published[g]['S1']['scorer']);V.E.D.close(oracle,published[g]['S1']['oracle'])
        out[g]=dict(twoD=summary([rebuilt[i] for i in ids]),current=cur,oracle=oracle,selection_loss=oracle['ADDsym_AUC']-cur['ADDsym_AUC'])
    anchors=C.read(C.PREV_RAW/'ANCHOR_POINT_METRICS.json');published_anchor=C.read(C.PREV_DOC/'VERIFIED_ANCHOR_TRANSFER.json');aa={}
    for g,sev in [('ALL',None),('HARD','HARD'),('CLEAN','CLEAN'),('MODERATE','MODERATE_OCCLUSION'),('SEVERE','SEVERE_OCCLUSION')]:
        rows=[r for r in anchors if sev is None or (r['severity']!='CLEAN' if sev=='HARD' else r['severity']==sev)];aa[g]=metrics([r['errors']['S1'] for r in rows]);V.E.D.close(aa[g],published_anchor['groups'][g]['S1'])
    C.freeze(C.DOC/'BASELINE_LOCK.json',dict(created_at=C.now(),status='BASELINE_REPRODUCED',model='S1',selector='frozen GEO_LINEAR',groups=out,anchors=aa,reference=published_anchor['reference'],bindings=C.read(C.RAW/'INFERENCE_BINDINGS.json'),tolerance=1e-7))
    print('BASELINE_REPRODUCED',{g:out[g]['current']['ADDsym_AUC'] for g in ('CLEAN','MODERATE','SEVERE')},flush=True)
if __name__=='__main__':main()
