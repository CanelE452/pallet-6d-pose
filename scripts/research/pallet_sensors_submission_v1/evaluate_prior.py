"""Source-only selection, actual DEV inference, canonical scores and paired analysis."""
import math,gc
import numpy as np,torch
from env import *
from prior_data import SourceRGB
from prior_model import PoseFixPallet9,expectation
from posefix_contract_math import transform_points
from prior_inference import PriorInference,correct

@torch.no_grad()
def selection():
    src=SourceRGB();data=src.data;rules=read(OLD_DOC/'D_PROTOCOL_LOCK.json')['selection_rules'];legacy=old('select_synthetic');allmetrics={};bindings={}
    for seed in (1,2,3):
        ckpath=RAW/f'runs/PRIOR{seed}/last.pt';cache=RAW/f'validation_PRIOR{seed}.npz';model=PoseFixPallet9().cuda().eval();ck=torch.load(ckpath,map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict']);del ck
        if not cache.exists():
            values=[]
            for row in data.validation_rows:
                box=data.arrays['boxes'][row];points=np.array(data.arrays['points'][row],copy=True)
                if np.isfinite(box).all() and (box[2:]>box[:2]).all() and data.arrays['detected'][row]:
                    x=src.item(int(row));args=[torch.as_tensor(x[k],device='cuda')[None] for k in ('rgb','points','valid')]
                    with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
                        q=expectation(model(*args))[0].cpu().numpy()
                    restored=transform_points(q,np.linalg.inv(x['matrix']))
                    record=data.source['records'][int(data.indices[row])];gain,offset=old('features').canvas_affine(record['prepared_shape_hw'],data.arrays['input_shape'][row])
                    mask=x['valid'].copy();mask[8]=False;points[mask]=(restored*gain+offset)[mask]
                values.append(points)
            np.savez(cache,rows=data.validation_rows,points=np.stack(values),checkpoint_sha256=sha(ckpath))
        z=np.load(cache);assert str(z['checkpoint_sha256'])==sha(ckpath) and np.array_equal(z['rows'],data.validation_rows);bindings[str(seed)]=bound(cache)
        rowmap={int(r):i for i,r in enumerate(data.validation_rows)};allmetrics[seed]={}
        # Raw plus every fixed wrapper rule; evaluate all source partitions without selection leakage.
        for partition in ('calibration','selection','heldout'):
            rows=np.flatnonzero(data.partitions==partition);a={k:np.array(v[rows],copy=True) for k,v in data.arrays.items() if k not in ('p3','p4')};records=[data.source['records'][int(data.indices[r])] for r in rows];raw=z['points'][[rowmap[int(r)] for r in rows]]
            full=[]
            for rule in rules+[dict(lam=1.,max_move_image_diagonal_fraction=None)]:
                q=[]
                for i,r in enumerate(records):
                    # Correct in input-canvas pixels; original image cap transformed by scalar letterbox gain.
                    fraction=rule['max_move_image_diagonal_fraction'];delta=(raw[i]-a['points'][i])*rule['lam']
                    if fraction is not None:
                        cap=fraction*math.hypot(*r['raw_shape_hw'])*float(a['gain'][i]);delta*=np.minimum(1,cap/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[:,None]
                    p=a['points'][i].copy();v=a['point_valid'][i].astype(bool);v[8]=False;p[v]+=delta[v];q.append(p)
                full.append(legacy.frame_metrics(np.stack(q),a['point_valid'],a['gt_points'],a['gt_valid'],a['gain'].reshape(-1),records,a['matched'],a['detected'],a['matched_gt_index']))
            allmetrics[seed][partition]=full
        del model;gc.collect();torch.cuda.empty_cache();print('PRIOR_SOURCE_SCORED',seed,flush=True)
    candidates=[]
    for i,rule in enumerate(rules):
        candidates.append(dict(**rule,score=float(np.mean([[v['score'] for v in allmetrics[s]['selection'][i]] for s in (1,2,3)])),per_seed={str(s):legacy.summarize(allmetrics[s]['selection'][i]) for s in (1,2,3)}))
    if not (DOC/'PRIOR_SELECTION.json').exists():freeze(DOC/'PRIOR_SELECTION.json',dict(complete=True,selected_at=now(),candidates=candidates,selected_rule=legacy.choose_rule(candidates),checkpoints={str(s):sha(RAW/f'runs/PRIOR{s}/last.pt') for s in (1,2,3)},cache=bindings,code={p.name:sha(p) for p in [HERE/'prior_model.py',HERE/'prior_data.py',HERE/'evaluate_prior.py']},no_real_selection=True,no_temperature=True,source_partitions=dict(calibration=1004,selection=1031,heldout=1985)))
    chosen=read(DOC/'PRIOR_SELECTION.json')['selected_rule'];index=next(i for i,x in enumerate(rules) if all(x[k]==chosen[k] for k in x))
    write(DOC/'PRIOR_SOURCE_RESULTS.json',{str(s):{p:dict(selected=legacy.summarize(v[index]),raw=legacy.summarize(v[-1])) for p,v in allmetrics[s].items()} for s in (1,2,3)})

def infer():
    er=old('evaluate_real');_,cache,positive=er.baseline_inputs(LINE);positive=set(positive);jsonable=old('inference').jsonable
    freeze(DOC/'PRIOR_DEV_LOCK.json',dict(selection=bound(DOC/'PRIOR_SELECTION.json'),code={p.name:sha(p) for p in [HERE/'prior_inference.py',HERE/'evaluate_prior.py']},population=bound(OLD_DOC/'POPULATION_AND_METRIC_LOCK.json')))
    for seed in (1,2,3):
        name=f'PRIOR{seed}';dst=RAW/f'evaluation/{name}';dst.mkdir(parents=True,exist_ok=True)
        if complete(name+'_INFERENCE'):continue
        start=now();model=PriorInference(seed);records=[];replacements={};raw_replacements={};changed=0
        try:
            for count,(key,reference) in enumerate(cache['frames'].items(),1):
                image=er.load_bgr(key,cache['frame_metadata'][key]);pred=model.predict(image);changed+=er.check_prediction(pred,reference,frame_key=key);selected=pred['selected_index']
                records.append(dict(image_key=key,prediction=jsonable(pred)))
                if key in positive:
                    replacements[key]=[] if selected is None else [dict(candidate_index=selected,keypoints_xy=jsonable(pred['candidates'][selected]['keypoints_xy']))]
                    raw_replacements[key]=[] if selected is None else [dict(candidate_index=selected,keypoints_xy=jsonable(pred.get('raw_keypoints_xy',pred['candidates'][selected]['keypoints_xy'])))]
                if count%300==0:print('PRIOR_DEV',seed,count,3008,flush=True)
        finally:model.close()
        assert len(records)==len(cache['frames'])==3008 and len(replacements)==len(raw_replacements)==319
        write(dst/'IMAGE_PREDICTIONS.json',dict(complete=True,records=records,selection_sha256=sha(DOC/'PRIOR_SELECTION.json')))
        for label,values in [('POINT',replacements),('RAW_POINT',raw_replacements)]:
            write(dst/(label+'_REPLACEMENTS.json'),dict(schema='pallet_line_pose_point_replacements_v1',complete=True,baseline_cache_sha256=sha(LINE/'baseline/FULL_CANDIDATES.json'),arm=name+('_raw' if label=='RAW_POINT' else ''),selection_artifact=dict(path=str(DOC/'PRIOR_SELECTION.json'),sha256=sha(DOC/'PRIOR_SELECTION.json')),frames=values))
        receipt(name+'_INFERENCE',[DOC/'PRIOR_DEV_LOCK.json',RAW/f'runs/{name}/last.pt'],[dst/'IMAGE_PREDICTIONS.json',dst/'POINT_REPLACEMENTS.json',dst/'RAW_POINT_REPLACEMENTS.json'],start,actual_forwards=3008,positives=319,negatives=2689,changed_instances=changed,box_score_order_center_exact=True)
        del model;gc.collect();torch.cuda.empty_cache()

def score():
    pe=old('paper_evaluation')
    for seed in (1,2,3):
        for raw in (False,True):
            name=f'PRIOR{seed}'+('_raw' if raw else '');start=now()
            if complete(name+'_SCORED'):continue
            path=RAW/f'evaluation/PRIOR{seed}';rep=path/('RAW_POINT_REPLACEMENTS.json' if raw else 'POINT_REPLACEMENTS.json')
            dst,frames,altered=pe.replace_points(RAW,LINE/'baseline/FULL_CANDIDATES.json',rep,name)
            pe.paper_2d(dst,dst/'PREDICTIONS.json',str(R0));pe.paper_pose(dst,frames,name)
            receipt(name+'_SCORED',[rep],[dst/f for f in ('PREDICTIONS.json','PAPER_2D.json','PAPER_2D_per_frame.csv','POSE_PER_FRAME_BY_ARM.json')],start,altered=altered)

def run():
    assert complete('TRAIN_COMPLETE'),'Prior3 training must finish before source selection or DEV'
    if complete('EVALUATE_COMPLETE'):verify();return
    start=now();verify();assert not gpu()['foreign_compute'];selection();infer();score()
    from prior_analysis import run as analysis
    analysis()
    from submission_runtime import run as runtime
    runtime()
    receipt('EVALUATE_COMPLETE',[DOC/'PRIOR_DEV_LOCK.json'],[DOC/'UNIFIED_DEV_RESULTS.json',DOC/'P_VS_PRIOR_PAIRED.json',DOC/'RUNTIME_PANEL.json'],start)
