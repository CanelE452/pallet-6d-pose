"""Actual image forwards, exact detection parity and unchanged paper scorers."""
import copy
import gc
import numpy as np
import torch
from common import *
from point_inference import PointInference


def inputs():
    er=old('evaluate_real');binding,cache,keys=er.baseline_inputs(LINE)
    assert read(B/'P_SELECTION.json')['complete'] and read(B/'SYNTH_HELDOUT_RESULTS.json')['complete']
    return er,cache,keys


def lock():
    er,cache,keys=inputs()
    value=dict(timestamp=datetime.now(timezone.utc).isoformat(),complete=True,
        selection_sha256=sha(B/'P_SELECTION.json'),sources=old('paper_evaluation').fixed_inputs(),
        code={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'evaluate_point.py',HERE/'point_inference.py',HERE/'generic_point_refiner.py',HERE/'select_point.py',HERE/'statistics_and_mechanism.py']},
        baseline_cache_sha256=sha(LINE/'baseline/FULL_CANDIDATES.json'),positive319=True,negative2689=True,
        actual_negative_forward=True,real_for_selection=False,role='historically reused DEV',
        timing=read(LINE/'REAL_EVALUATION_PLAN.json')['timing'])
    if (B/'REAL_EVALUATION_LOCK.json').exists():
        oldlock=read(B/'REAL_EVALUATION_LOCK.json');value['timestamp']=oldlock['timestamp']
    write(B/'REAL_EVALUATION_LOCK.json',value)
    return value


def verify_lock():
    v=read(B/'REAL_EVALUATION_LOCK.json');assert sha(B/'P_SELECTION.json')==v['selection_sha256']
    for p,h in v['sources'].items():assert sha(p)==h
    for p,h in v['code'].items():assert sha(ROOT/p)==h


def infer():
    lock();er,cache,positives=inputs();pe=old('paper_evaluation');jsonable=old('inference').jsonable
    write(B/'GPU_REAL_START.json',gpu())
    for seed in (1,2,3):
        destination=BRAW/f'evaluation/P{seed}';destination.mkdir(parents=True,exist_ok=True)
        if (destination/'POINT_REPLACEMENTS.json').exists():
            assert read(B/f'P{seed}_INFERENCE_AUDIT.json')['PASS'];continue
        model=PointInference(seed);records=[];replacements={};changed_pos=0;changed_neg=0
        try:
            for count,(key,references) in enumerate(cache['frames'].items(),1):
                image=er.load_bgr(key,cache['frame_metadata'][key]);prediction=model.predict(image)
                try:changed=er.check_prediction(prediction,references,frame_key=key)
                except Exception as error:
                    write(destination/'CONTRACT_FAILURE.json',dict(key=key,error=str(error),prediction=jsonable(prediction),reference=references));raise
                selected=prediction['selected_index']
                if key in positives:
                    changed_pos+=changed
                    replacements[key]=[] if selected is None else [dict(candidate_index=selected,keypoints_xy=jsonable(prediction['candidates'][selected]['keypoints_xy']))]
                else:changed_neg+=changed
                records.append(dict(image_key=key,prediction=jsonable(prediction)))
                if count%100==0:print('P_REAL',seed,count,3008,'GPU',gpu()['gpu'],flush=True)
        finally:model.close()
        verify_lock()
        write(destination/'IMAGE_PREDICTIONS.json',dict(complete=True,records=records,selection_sha256=sha(B/'P_SELECTION.json')))
        write(destination/'POINT_REPLACEMENTS.json',dict(schema='pallet_line_pose_point_replacements_v1',complete=True,
            baseline_cache_sha256=sha(LINE/'baseline/FULL_CANDIDATES.json'),arm=f'P{seed}',
            selection_artifact=dict(path=str(B/'P_SELECTION.json'),sha256=sha(B/'P_SELECTION.json')),frames=replacements))
        write(B/f'P{seed}_INFERENCE_AUDIT.json',dict(PASS=True,complete=True,actual_positive_forwards=319,actual_negative_forwards=2689,
            box_max_diff=0,score_max_diff=0,candidate_order_exact=True,selected_instance_exact=True,baseline_points_exact=True,
            center8_max_diff=0,nonselected_exact=True,changed_positive_instances=changed_pos,changed_negative_instances=changed_neg,
            image_predictions_sha256=sha(destination/'IMAGE_PREDICTIONS.json'),replacements_sha256=sha(destination/'POINT_REPLACEMENTS.json'),
            negative_metric_replay='All negative box/score/order verified by actual P inference. Canonical scorer reuses original negative points because negative keypoints have no supervised endpoint; actual P point outputs retained separately.'))
        del model;gc.collect();torch.cuda.empty_cache()
    print('ACTUAL_REAL_INFERENCE_COMPLETE',flush=True)


def score():
    verify_lock();pe=old('paper_evaluation')
    for seed in (1,2,3):
        label=f'P{seed}';path=BRAW/f'evaluation/{label}/POINT_REPLACEMENTS.json'
        assert read(B/f'{label}_INFERENCE_AUDIT.json')['PASS']
        destination,frames,altered=pe.replace_points(BRAW,LINE/'baseline/FULL_CANDIDATES.json',path,label)
        pe.paper_2d(destination,destination/'PREDICTIONS.json',str(R0))
        pe.paper_pose(destination,frames,label)
        write(B/f'{label}_PAPER_EVALUATION_COMPLETE.json',dict(complete=True,
            outputs={name:sha(destination/name) for name in ['PAPER_2D.json','PAPER_2D_per_frame.csv','POSE_PER_FRAME_BY_ARM.json',f'POSE_EVALUATION_{label}.json']},
            unchanged_evaluators=True,positive=319,negative=2689,altered_positive_candidates=altered))
        print('P_CANONICAL_SCORING_COMPLETE',seed,flush=True)
    verify_lock()


def runtime():
    verify_lock();er,cache,keys=inputs();timing=read(B/'REAL_EVALUATION_LOCK.json')['timing']
    write(B/'GPU_RUNTIME_START.json',gpu())
    images={key:er.load_bgr(key,cache['frame_metadata'][key]) for key in timing['keys']}
    baseline=er.PlainBaseline(R0);results={};records=[]
    for method in ('L','P'):
        for seed in (1,2,3):
            model=(old('inference').PalletLinePoseInference(LINE/f'runs/image_line_only_seed{seed}/last.pt',LINE/'SELECTION.json') if method=='L' else PointInference(seed))
            previous=read((LINE/f'evaluation/image_line_only_seed{seed}' if method=='L' else BRAW/f'evaluation/P{seed}')/'IMAGE_PREDICTIONS.json')
            prior={r['image_key']:r['prediction'] for r in previous['records']};current=[]
            try:
                for i in range(timing['warmup']):
                    image=images[timing['keys'][i%len(images)]];baseline.predict(image);model.predict(image)
                for repeat in range(timing['repeats']):
                    for index,key in enumerate(timing['keys']):
                        image=images[key]
                        if (repeat+index)%2:
                            pred,ms=er.timed(lambda:model.predict(image));ref,base=er.timed(lambda:baseline.predict(image))
                        else:
                            ref,base=er.timed(lambda:baseline.predict(image));pred,ms=er.timed(lambda:model.predict(image))
                        for actual,stored in zip(ref,cache['frames'][key]):
                            for field in ('score','box_xyxy','keypoints_xy'):er.exact(actual[field],stored[field],f'runtime_baseline/{key}/{field}')
                        assert len(ref)==len(cache['frames'][key])
                        er.check_prediction(pred,cache['frames'][key],frame_key=key)
                        for actual,stored in zip(pred['candidates'],prior[key]['candidates']):er.exact(actual['keypoints_xy'],stored['keypoints_xy'],f'runtime_accuracy/{key}')
                        current.append(dict(method=method,seed=seed,key=key,repeat=repeat,baseline_ms=base,integrated_ms=ms,added_ms=ms-base,head_used=pred['head_used']))
            finally:model.close()
            records+=current
            results[f'{method}{seed}']=dict(baseline_ms=er.stats([r['baseline_ms'] for r in current]),
                integrated_ms=er.stats([r['integrated_ms'] for r in current]),added_ms=er.stats([r['added_ms'] for r in current]),
                head_used_fraction=float(np.mean([r['head_used'] for r in current])))
            print('RUNTIME_COMPLETE',method,seed,results[f'{method}{seed}'],flush=True)
            del model;gc.collect();torch.cuda.empty_cache();gpu()
    verify_lock()
    write(B/'RUNTIME.json',dict(complete=True,PASS=True,parity_PASS=True,protocol=timing,by_run=results,records=records,
        R0=er.stats([r['baseline_ms'] for r in records]),scope='Exact old image-to-original-2D scope; excludes file decoding/loading/PnP; paired alternating measurements',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))
    write(B/'PARAM_COMPUTE_FINAL.json',dict(parameters=read(B/'PARAMETER_BUDGET_LOCK.json'),evidence=read(B/'EVIDENCE_BUDGET_AUDIT.json'),runtime_sha256=sha(B/'RUNTIME.json'),equal_FLOPs_claim=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['infer','score','runtime']);args=p.parse_args()
    globals()[args.phase]()
