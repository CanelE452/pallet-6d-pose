"""Freeze all GT-free learned decisions before scoring reused real evaluation."""
import argparse
import copy
import time
import cv2
import numpy as np
import torch
from . import core as P
from .model import UtilitySelector,features,choose_pairs,apply_pairs
from .train import KEYS

ARMS=('A_N2','REPLAY','CORNER_FULL_PAIR','LEARNED_PAIR')


def top(pred):
    i=pred['selected_index'];return None if i is None else pred['candidates'][i]


@torch.no_grad()
def apply():
    protocol=P.verify();P.setup();runtime=[P.gpu()]
    if (P.DOC/'OUTPUTS_LOCK.json').exists():
        for b in P.read(P.DOC/'OUTPUTS_LOCK.json')['artifacts']:P.verify_binding(b)
        print('REAL_OUTPUTS_ALREADY_FROZEN',flush=True);return
    fit=P.read(P.DOC/'FIT.json');assert fit['complete'] and fit['steps']==1500
    P.verify_binding(fit['checkpoint'])
    ck=torch.load(P.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['complete'] and ck['step']==1500 and ck['protocol']==P.bound(P.DOC/'PROTOCOL.json')
    model=UtilitySelector(ck['numeric_dim'],ck['global_dim']);model.load_state_dict(ck['model_state_dict'])
    model=model.cuda().eval().requires_grad_(False)
    saved=P.read(P.N.RAW/'PREDICTIONS.json');gated=P.read(P.GATE_RAW/'PREDICTIONS.json')
    infer=P.read(P.GATE_RAW/'INFERENCE.json');assert infer['complete'] and infer['count']==222
    caches={}
    for binding in infer['cache_bindings']:
        P.verify_binding(binding);row=P.read(P.ROOT/binding['path']);caches[(row['dataset'],row['id'])]=(row,binding)
    outputs={ds:[] for ds in saved};decisions={ds:[] for ds in saved};feature_rows=[]
    start=time.monotonic();count=0
    for ds,records in saved.items():
        historical={r['id']:r for r in gated[ds]}
        for row in records:
            # Allowlisted RGB, model outputs and prediction heatmaps only; no annotation read.
            P.verify_binding(row['image']);im=cv2.imread(str(P.ROOT/row['image']['path']))
            assert im is not None and list(im.shape[:2])==row['raw_hw']
            predictions=row['predictions'];a=top(predictions['A_N2']);r=top(predictions['POSEFIX_RAW']);raw=top(predictions['R0'])
            points={k:np.asarray(v['keypoints_xy'],float) for k,v in [('n2',a),('replay',r),('r0',raw)]}
            valid=np.isfinite(points['r0']).all(-1)&~(points['r0']==-1).all(-1)
            valid &= np.isfinite(points['n2']).all(-1)&np.isfinite(points['replay']).all(-1)
            valid &= ~(points['n2']==-1).all(-1)&~(points['replay']==-1).all(-1)
            cache,binding=caches[(ds,row['id'])]
            assert cache['normal']['prediction']==predictions['POSEFIX_RAW']
            x=features(im,points['r0'],points['n2'],points['replay'],np.asarray(raw['box_xyxy']),valid,cache['normal']['heatmap'])
            y=model({k:torch.as_tensor(x[k],device='cuda')[None] for k in KEYS})[0].cpu().numpy()
            assert y.shape==(8,) and np.isfinite(y).all()
            mask=choose_pairs(y,valid);q=apply_pairs(points['n2'],points['replay'],mask)
            learned=copy.deepcopy(predictions['A_N2']);top(learned)['keypoints_xy']=q.tolist()
            base=predictions['A_N2']
            assert base['selected_index']==learned['selected_index']
            assert len(base['candidates'])==len(learned['candidates'])
            for i,(before,after) in enumerate(zip(base['candidates'],learned['candidates'])):
                for key in before:
                    if key!='keypoints_xy' or i!=base['selected_index']:assert before[key]==after[key],(row['id'],key)
            assert top(learned)['keypoints_xy'][8]==a['keypoints_xy'][8]
            assert (np.all(q[:8]==points['n2'][:8],-1)|np.all(q[:8]==points['replay'][:8],-1)).all()
            variants={arm:copy.deepcopy(historical[row['id']]['predictions'][arm]) for arm in ARMS[:-1]}
            variants['LEARNED_PAIR']=learned
            outputs[ds].append({**{k:row[k] for k in ('id','image','annotation','session','raw_hw')},'predictions':variants})
            decisions[ds].append(dict(id=row['id'],utility=y.tolist(),pair_accept=mask.tolist(),valid=valid.tolist(),
                accepted_corners=int(mask.sum())*2,heatmap_cache=binding,GT_input=False))
            feature_rows.append(x);count+=1
            if count%64==0:runtime.append(P.gpu());print(dict(stage='REAL_GT_FREE_SELECTION',count=count,total=222),flush=True)
    assert count==222
    P.save_npz(P.RAW/'REAL_FEATURES.npz',**{k:np.stack([r[k] for r in feature_rows]) for k in KEYS})
    P.freeze(P.RAW/'PREDICTIONS.json',outputs);P.freeze(P.RAW/'DECISIONS.json',decisions)
    runtime.append(P.gpu())
    P.freeze(P.DOC/'OUTPUTS_LOCK.json',dict(complete=True,frames=222,GT_read_for_selection=False,
        protocol=P.bound(P.DOC/'PROTOCOL.json'),fit=P.bound(P.DOC/'FIT.json'),checkpoint=fit['checkpoint'],
        artifacts=[P.bound(P.RAW/n) for n in ('PREDICTIONS.json','DECISIONS.json','REAL_FEATURES.npz')],
        elapsed_seconds=time.monotonic()-start,runtime_checks=runtime,model_training=False))
    print('ALL222_LEARNED_OUTPUTS_FROZEN_BEFORE_GT',flush=True)


def score():
    P.verify();lock=P.read(P.DOC/'OUTPUTS_LOCK.json');assert lock['complete']
    for b in [*lock['artifacts'],lock['fit'],lock['checkpoint']]:P.verify_binding(b)
    # First real GT-coordinate access, after immutable outputs for all222.
    from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
    from scripts.research.pallet_posefix_corner_gate_v1.evaluate import paired
    _,old,records,groups,pe,targets,_=O.evaluation_inputs()
    predictions=P.read(P.RAW/'PREDICTIONS.json')
    history=P.read(P.GATE_RAW/'PER_FRAME_METRICS.json')
    rows={ds:{a:[] for a in ARMS} for ds in O.DATASETS};checks=0
    for ds,pop in predictions.items():
        for row,record in zip(pop,records[ds]):
            assert row['id']==record['id']
            gt,box,modes,kind=O.read_targets(ds,record,row['raw_hw'],pe,targets)
            for mode,valid in modes:
                for arm,pred in row['predictions'].items():
                    metric=O.score_prediction(pred,gt,box,valid,groups[kind]['permutations'],row['raw_hw'],record)
                    rows[mode][arm].append(metric)
                    if arm!='LEARNED_PAIR':
                        before=next(r for r in history[mode][arm] if r['id']==row['id'])
                        O.assert_same(before,metric,mode+'/'+arm+'/'+row['id']);checks+=1
    assert checks==1116
    summaries={};comparisons={}
    for ds,arms in rows.items():
        summaries[ds]={};comparisons[ds]={}
        for arm,rr in arms.items():
            result=O.summary(rr);obs=[e for r in rr for e in r['observed_errors']]
            result.update(matched_mean_px=float(np.mean(obs)),correct10=sum(e<=10 for r in rr for e in r['errors']))
            summaries[ds][arm]=result;comparisons[ds][arm]=paired(arms['A_N2'],rr,arms['REPLAY'])
    decisions=P.read(P.RAW/'DECISIONS.json')
    acceptance={ds:dict(frames=len(rr),accepted_pairs=sum(sum(r['pair_accept']) for r in rr),
        frames_any_accepted=sum(any(r['pair_accept']) for r in rr)) for ds,rr in decisions.items()}
    P.freeze(P.RAW/'PER_FRAME_METRICS.json',rows)
    P.freeze(P.DOC/'RESULTS.json',dict(complete=True,summary=summaries,comparisons=comparisons,acceptance=acceptance,
        baseline_parity_checks=checks,output_lock=P.bound(P.DOC/'OUTPUTS_LOCK.json'),
        metrics=P.bound(P.RAW/'PER_FRAME_METRICS.json'),final_model_modified=False,independent_test=False,
        thresholds_retuned=False,evaluation_used_for_training=False,selector_trained=True))
    print(dict(stage='LEARNED_SELECTOR_SCORE_COMPLETE',summary={k:summaries[k]['LEARNED_PAIR'] for k in ('DEV72','GREEN150_MANUAL')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['apply','score']);args=parser.parse_args()
    apply() if args.stage=='apply' else score()
