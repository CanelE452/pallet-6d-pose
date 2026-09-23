from pathlib import Path
import json
import numpy as np
from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as C

ROOT=C.ROOT
NAME='pallet_clean19_pose_sensitive_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
read=C.read
save=D.save
bind=C.bind
verify=C.verify

def plans():return [json.loads(x) for x in (C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()]

def rank(hypotheses,scores):
    valid=[h for h in hypotheses if h['success']]
    if not valid:return None
    if len(valid)==1:return valid[0]['name']
    cfg=D.Selector.SelectorConfig()
    if abs(scores[valid[0]['name']]-scores[valid[1]['name']])<=cfg.parity_tie_tolerance:return None
    return min(valid,key=lambda h:scores[h['name']])['name']

def d8_scores(result,q):
    cfg=D.Selector.SelectorConfig();out={}
    for h in result['hypotheses']:
        if not h['success']:continue
        residual=np.linalg.norm(np.array(h['projected_keypoints'])-q,axis=1)
        rmse9=float(np.sqrt(np.mean(residual**2)));rmse8=float(np.sqrt(np.mean(residual[:8]**2)))
        assert abs(rmse9-h['score_components']['reprojection_rmse_px'])<1e-10
        out[h['name']]=dict(D9=h['score'],D8=h['score']+cfg.reprojection_weight*(rmse8-rmse9),RMSE9=rmse9,RMSE8=rmse8,P8_error=float(residual[8]))
    return out
