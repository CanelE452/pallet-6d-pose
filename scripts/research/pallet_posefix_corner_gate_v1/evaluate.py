"""Apply no-GT gates first, freeze outputs, THEN score all unchanged frames."""
import copy
import json
import time
import cv2
import numpy as np
from . import protocol as P
from . import gates as G

ARMS=('A_N2','REPLAY','LEGACY_FRAME','CORNER_GEOMETRY_PAIR','CORNER_FULL_PAIR')


def top(pred):
    i=pred['selected_index'];return None if i is None else pred['candidates'][i]


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    return x


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter,1e-12))


def replace(pred,points):
    new=copy.deepcopy(pred);top(new)['keypoints_xy']=np.asarray(points).tolist();return new


def apply():
    code=P.lock_code();protocol=P.read(P.DOC/'PROTOCOL.json')
    if (P.DOC/'OUTPUTS_LOCK.json').exists():
        done=P.read(P.DOC/'OUTPUTS_LOCK.json')
        for binding in done['artifacts']+[done['protocol'],done['code']]:P.verify(binding)
        print('OUTPUTS_ALREADY_COMPLETE',flush=True);return
    inference=P.read(P.RAW/'INFERENCE.json')
    assert inference['complete'] and inference['count']==222 and not inference['GT_used']
    assert inference['bindings']['protocol']==P.bound(P.DOC/'PROTOCOL.json')
    for binding in inference['bindings'].values():P.verify(binding)
    saved=P.read(P.OLD/'PREDICTIONS.json')
    sources={
        'DEV72':{r['id']:r for r in P.read(P.ROOT/'_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json')['evaluation']},
        'GREEN150':{r['id']:r for r in P.read(P.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json')['records']}}
    registry={r['object_type']:r['physical_dimensions_m'] for r in P.read(P.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')['objects']}
    # Load only inference cache, original source K and registry, not annotation coordinates.
    by_id={}
    for binding in inference['cache_bindings']:
        P.verify(binding);frame=P.read(P.ROOT/binding['path'])
        assert frame['bindings']==inference['bindings']
        key=(frame['dataset'],frame['id']);assert key not in by_id
        by_id[key]=(frame,binding)
    assert len(by_id)==222,len(by_id)
    outputs={ds:[] for ds in saved};diagnostics={ds:[] for ds in saved};begin=time.monotonic()
    from scripts.self_training_yolo.pseudo_label_filters import geometry_scores
    for ds,rows in saved.items():
        for row in rows:
            meta=sources[ds][row['id']];cache,binding=by_id[(ds,row['id'])]
            assert cache['id']==row['id'] and cache['image']==row['image']
            raw=row['predictions']['POSEFIX_RAW'];base=row['predictions']['A_N2'];a=top(raw);b=top(base)
            q=np.asarray(a['keypoints_xy'],float);n2=np.asarray(b['keypoints_xy'],float)
            confidence=np.asarray(a['keypoints_conf'],float)
            valid=(confidence>=.5)&np.isfinite(q).all(-1)&~(q==-1).all(-1)
            K=np.asarray(meta['K'] if ds=='DEV72' else meta['source_K'],float)
            kind=meta['object_type'] if ds=='DEV72' else 'plastic_standard_110x110x15';dims=registry[kind]
            geometry=G.geometry_details(q,valid,K,dims)
            diagonal=geometry['projected_diagonal_px']
            mode=np.asarray(cache['normal']['heatmap']['mode_original_xy9'],float)
            mode_separation=np.linalg.norm(q[:8]-mode[:8],axis=-1)/diagonal if diagonal>0 else np.full(8,np.inf)
            flip=cache['flip'];flipped=top(flip['refined_unflipped_prediction']) if flip.get('refiner_available') else None
            fq=np.full((9,2),np.nan) if flipped is None else np.asarray(flipped['keypoints_xy'],float)
            fv=np.zeros(9,bool) if flipped is None else (np.asarray(flipped['keypoints_conf'])>=.5)&np.isfinite(fq).all(-1)&~(fq==-1).all(-1)
            flip_res=G.normalized_distances(q,fq,valid,fv,diagonal)
            flip_iou=0.0 if flipped is None else iou(a['box_xyxy'],flipped['box_xyxy'])
            full_flip_res=np.asarray(flip_res).copy()
            if flip_iou<.5:full_flip_res[:]=np.inf
            try:
                legacy_scores=geometry_scores(q,valid,K,dims,None if flipped is None else fq,None if flipped is None else fv)
            except cv2.error as exc:
                legacy_scores=dict(s_remove=float('inf'),s_flip=None,s_reproj=float('inf'),solver_error=str(exc))
            legacy_pass=bool(flipped is not None and a['score']>=.85 and valid[:8].sum()>=6
                and np.isfinite(legacy_scores['s_remove']) and legacy_scores['s_remove']<=.05
                and legacy_scores['s_flip'] is not None and np.isfinite(legacy_scores['s_flip']) and legacy_scores['s_flip']<=.05)
            geom=G.gate_with_recheck(n2,q,valid,K,dims,a['score'],confidence,policy='geometry_only',geometry=geometry)
            full=G.gate_with_recheck(n2,q,valid,K,dims,a['score'],confidence,policy='full',geometry=geometry,
                flip_residuals=full_flip_res,mode_separation=mode_separation)
            variants=dict(A_N2=copy.deepcopy(base),REPLAY=copy.deepcopy(raw),
                LEGACY_FRAME=copy.deepcopy(raw if legacy_pass else base),
                CORNER_GEOMETRY_PAIR=replace(base,geom['points']),CORNER_FULL_PAIR=replace(base,full['points']))
            for arm,pred in variants.items():
                assert pred['selected_index']==base['selected_index']
                for i,(p0,p1) in enumerate(zip(base['candidates'],pred['candidates'])):
                    for key in p0:
                        if key!='keypoints_xy' or i!=base['selected_index']:assert p0[key]==p1[key],(row['id'],arm,key)
                assert top(pred)['keypoints_xy'][8]==b['keypoints_xy'][8]
                pp=np.asarray(top(pred)['keypoints_xy'])
                allowed=np.all(pp[:8]==n2[:8],axis=1)|np.all(pp[:8]==q[:8],axis=1)
                assert allowed.all(),('Only original candidates allowed',arm,row['id'])
            outputs[ds].append(dict(id=row['id'],image=row['image'],annotation=row['annotation'],session=row['session'],raw_hw=row['raw_hw'],predictions=variants))
            diagnostics[ds].append(clean(dict(id=row['id'],inference_cache=binding,geometry=geometry,
                mode_separation=mode_separation,flip_residuals=flip_res,flip_iou=flip_iou,
                legacy_scores=legacy_scores,legacy_pass=legacy_pass,
                geometry_pair={k:v for k,v in geom.items() if k!='points'},
                full_pair={k:v for k,v in full.items() if k!='points'})))
    P.write(P.RAW/'PREDICTIONS.json',outputs);P.write(P.RAW/'GATES.json',diagnostics)
    P.write(P.DOC/'OUTPUTS_LOCK.json',dict(complete=True,frames=222,GT_read_for_gating=False,model_training=False,
        protocol=P.bound(P.DOC/'PROTOCOL.json'),code=P.bound(P.DOC/'CODE_LOCK.json'),
        artifacts=[P.bound(P.RAW/'PREDICTIONS.json'),P.bound(P.RAW/'GATES.json')],elapsed_seconds=time.monotonic()-begin))
    print('OUTPUTS_FROZEN_BEFORE_GT',222,flush=True)


def paired(base,new,raw):
    a=[];b=[];r=[];branches=0
    for x,y,z in zip(base,new,raw):
        assert x['id']==y['id']==z['id'] and x['canonical_valid']==y['canonical_valid']==z['canonical_valid']
        branches+=x['branch']!=y['branch']
        for i,v in enumerate(x['canonical_valid']):
            if v:a.append(x['canonical_errors'][i]);b.append(y['canonical_errors'][i]);r.append(z['canonical_errors'][i])
    a,b,r=map(np.asarray,(a,b,r));damage=(a<=10)&(r>10);gain=(a>10)&(r<=10)
    return dict(corners=len(a),gain_over_N2=int(((a>10)&(b<=10)).sum()),damage_over_N2=int(((a<=10)&(b>10)).sum()),
        good5_to_bad10=int(((a<5)&(b>10)).sum()),hard20_to_good10=int(((a>20)&(b<=10)).sum()),
        raw_Replay_gains=int(gain.sum()),raw_Replay_gains_retained=int((gain&(b<=10)).sum()),
        raw_Replay_gains_lost=int((gain&(b>10)).sum()),raw_Replay_damages=int(damage.sum()),
        raw_Replay_damages_prevented=int((damage&(b<=10)).sum()),raw_Replay_damages_remaining=int((damage&(b>10)).sum()),
        symmetry_branch_changes=branches)


def score():
    lock=P.read(P.DOC/'OUTPUTS_LOCK.json')
    for binding in lock['artifacts']+[lock['protocol'],lock['code']]:P.verify(binding)
    # First evaluation-only access to annotation targets, after ALL222 gate choices fixed.
    from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
    _,old,records,groups,pe,targets,_=O.evaluation_inputs()
    outputs=P.read(P.RAW/'PREDICTIONS.json');rows={ds:{a:[] for a in ARMS} for ds in O.DATASETS}
    history=P.read(P.OLD/'PER_FRAME_METRICS.json');checked=0
    for ds,pop in outputs.items():
        for row,record in zip(pop,records[ds]):
            assert row['id']==record['id']
            gt,box,modes,kind=O.read_targets(ds,record,row['raw_hw'],pe,targets)
            for mode,valid in modes:
                for arm,pred in row['predictions'].items():
                    metric=O.score_prediction(pred,gt,box,valid,groups[kind]['permutations'],row['raw_hw'],record)
                    rows[mode][arm].append(metric)
                    if arm in ('A_N2','REPLAY'):
                        original=next(r for r in history[mode]['A_N2' if arm=='A_N2' else 'POSEFIX_RAW'] if r['id']==row['id'])
                        O.assert_same(original,metric,mode+'/'+arm+'/'+row['id']);checked+=1
    summaries={};comparisons={}
    for ds,arms in rows.items():
        summaries[ds]={};comparisons[ds]={}
        for arm,rr in arms.items():
            result=O.summary(rr);obs=[e for r in rr for e in r['observed_errors']]
            result.update(matched_mean_px=float(np.mean(obs)),correct10=sum(e<=10 for r in rr for e in r['errors']))
            summaries[ds][arm]=result;comparisons[ds][arm]=paired(arms['A_N2'],rr,arms['REPLAY'])
    gates=P.read(P.RAW/'GATES.json');acceptance={}
    for ds,rr in gates.items():
        acceptance[ds]=dict(frames=len(rr),legacy_frame_accept=sum(r['legacy_pass'] for r in rr))
        for name in ('geometry_pair','full_pair'):
            acceptance[ds][name]=dict(accepted_pairs=sum(sum(r[name]['pair_accept']) for r in rr),
                accepted_corners=sum(sum(r[name]['corner_accept']) for r in rr),frames_any_accepted=sum(any(r[name]['pair_accept']) for r in rr))
    P.write(P.RAW/'PER_FRAME_METRICS.json',rows)
    P.write(P.DOC/'RESULTS.json',dict(complete=True,baseline_parity_checks=checked,summary=summaries,
        comparisons=comparisons,acceptance=acceptance,output_lock=P.bound(P.DOC/'OUTPUTS_LOCK.json'),
        final_model_modified=False,trained_selector=False,thresholds_retuned=False,independent_test=False))
    print(json.dumps(dict(summary={k:summaries[k] for k in ('DEV72','GREEN150_MANUAL')},comparisons={k:comparisons[k] for k in ('DEV72','GREEN150_MANUAL')},acceptance=acceptance),ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('apply','score'));args=p.parse_args()
    apply() if args.stage=='apply' else score()
