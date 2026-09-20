"""Shared immutable roots and unchanged metrics for the persistent recovery goal."""
import argparse
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from . import common as C
from . import evaluate as E
from . import paper_metrics_plastic as M
from .pseudo import top

BASE_DOC=C.DOC
BASE_RAW=C.RAW
DOC=BASE_DOC/'selftrain_recovery_v1'
RAW=BASE_RAW/'selftrain_recovery_v1'


@contextmanager
def scope(phase):
    with patch.object(C,'DOC',DOC/phase),patch.object(C,'RAW',RAW/phase),patch.object(M,'DOC',DOC/phase/'metrics'),patch.object(M,'RAW',RAW/phase/'metrics'):
        yield


def evaluation_protocol(phase,arms,sources):
    with scope(phase):
        ev=C.read(BASE_DOC/'EVAL_PROTOCOL.json')
        ev['records']=[r for r in ev['records'] if r['kind']=='PLASTIC'];assert len(ev['records'])==194
        ev['arms']=arms;ev['sources']+=sources+[C.bound(__file__)]
        ev['scope']='All ordinary plastic194; unfiltered standalone predictions; source student and all evaluation history preserved; reused DEV not independent confirmation'
        C.freeze(C.DOC/'EVAL_PROTOCOL.json',ev)
        met=C.read(BASE_DOC/'paper_metrics_plastic/PROTOCOL.json');met['arms']=arms
        met['sources']+=sources+[C.bound(__file__)]
        C.freeze(M.DOC/'PROTOCOL.json',met)


def score(phase,arm,with_detection=True):
    with scope(phase):
        for b in C.read(C.DOC/'EVAL_PROTOCOL.json')['sources']:C.verify(b)
        mapping=C.read(BASE_DOC/'paper_metrics_plastic/POSE_ID_BRIDGE.json')['mapping']
        original=M.positive
        def mapped(a):
            payload,pred=original(a);assert set(pred)==set(mapping)
            return payload,{mapping[k]:v for k,v in pred.items()}
        f=M.f0(arm)
        with patch.object(M,'positive',mapped):p=M.pose(arm)
        pe,pop=E.O.population_metadata()
        from scripts.research.pallet_dim_conditioned_p_v1 import eval_math
        groups={r['object_type']:r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
        targets={item.frame_id:pe.E._legacy_forbidden_target(item) for item,meta in pop if meta['object_type']==C.TYPES['PLASTIC']}
        rows=[]
        for r in C.read(C.RAW/f'EVAL_PREDICTIONS_{arm}.json')['records']:
            t=targets[r['id']];pred=top(r['prediction']);matched=pred is not None and E.O.iou(pred['box_xyxy'],t.box_xyxy)>=.5
            q=np.full((9,2),np.nan) if pred is None else pred['keypoints_xy']
            measure=eval_math.measure(q,np.array(t.keypoints_xy),np.array(t.keypoint_supervision_mask),groups[C.TYPES['PLASTIC']],r['raw_hw'],matched,pred is not None)
            rows.append(dict(id=r['id'],kind='PLASTIC',**measure))
        result=dict(arm=arm,f0=f,pose=p,symmetry=eval_math.summary(rows),metrics=rows,
                    prediction=C.bound(C.RAW/f'EVAL_PREDICTIONS_{arm}.json'))
        if with_detection:result['detection']=M.detection(arm)
        C.freeze(C.RAW/f'{"RESULTS" if with_detection else "SCREEN"}_{arm}.json',result)
        print('RECOVERY_RESULT',arm,'PCK20',result['symmetry']['PCK']['20'],'kp8med',result['symmetry']['matched_pooled_corner8_median_px'],
              'IoU3D',p['iou3d_median'],'ADDsym',p['add_sym_auc'],'tcm',p['translation_median_cm'],flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('action',choices=['eval','negative','score','screen']);p.add_argument('arm');a=p.parse_args()
    if a.action in ['score','screen']:score(a.phase,a.arm,a.action=='score')
    else:
        with scope(a.phase):
            E.infer(a.arm) if a.action=='eval' else M.infer_negative(a.arm)
