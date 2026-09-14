"""Actual positive/negative inference, then unmodified canonical CPU scorers."""
import numpy as np
import torch
from env import *
from direct_inference import DirectInference
def infer():
    verify();assert not gpu()['foreign_compute'];er=old('evaluate_real');_,cache,positive=er.baseline_inputs(LINE);positive=set(positive);jsonable=old('inference').jsonable
    freeze(DOC/'D_DEV_EVALUATION_LOCK.json',dict(selection=bound(DOC/'D_SELECTION.json'),sources={p.name:sha(p) for p in [HERE/'direct_inference.py',HERE/'evaluate_direct.py',HERE/'direct_residual_control.py']},baseline_cache_sha256=sha(LINE/'baseline/FULL_CANDIDATES.json'),actual_frames=3008))
    for s in (1,2,3):
        dst=RAW/f'evaluation/D{s}';dst.mkdir(parents=True,exist_ok=True)
        if (DOC/f'D{s}_INFERENCE_AUDIT.json').exists():
            assert sha(dst/'IMAGE_PREDICTIONS.json')==read(DOC/f'D{s}_INFERENCE_AUDIT.json')['image_predictions_sha256'];continue
        model=DirectInference(s);records=[];replacement={};changed=0
        try:
            for count,(key,reference) in enumerate(cache['frames'].items(),1):
                image=er.load_bgr(key,cache['frame_metadata'][key]);pred=model.predict(image)
                changed+=er.check_prediction(pred,reference,frame_key=key);selected=pred['selected_index']
                records.append(dict(image_key=key,prediction=jsonable(pred)))
                if key in positive:replacement[key]=[] if selected is None else [dict(candidate_index=selected,keypoints_xy=jsonable(pred['candidates'][selected]['keypoints_xy']))]
                if count%300==0:print('D_INFER',s,count,3008,flush=True)
        finally:model.close()
        write(dst/'IMAGE_PREDICTIONS.json',dict(complete=True,records=records,selection_sha256=sha(DOC/'D_SELECTION.json')))
        write(dst/'POINT_REPLACEMENTS.json',dict(schema='pallet_line_pose_point_replacements_v1',complete=True,baseline_cache_sha256=sha(LINE/'baseline/FULL_CANDIDATES.json'),arm=f'D{s}',selection_artifact=dict(path=str(DOC/'D_SELECTION.json'),sha256=sha(DOC/'D_SELECTION.json')),frames=replacement))
        write(DOC/f'D{s}_INFERENCE_AUDIT.json',dict(PASS=True,positive=319,negative=2689,actual_forwards=3008,changed_instances=changed,center8_exact=True,box_score_order_selected_baseline_exact=True,
            image_predictions_sha256=sha(dst/'IMAGE_PREDICTIONS.json'),point_replacements_sha256=sha(dst/'POINT_REPLACEMENTS.json'),
            negative_policy='Actual forward verified detection contract. Canonical classifier scoring reuses frozen negative candidates; negative points have no supervised endpoint.'))
        del model;torch.cuda.empty_cache()
def score():
    verify();pe=old('paper_evaluation')
    for s in (1,2,3):
        assert read(DOC/f'D{s}_INFERENCE_AUDIT.json')['PASS'];arm=f'D{s}';path=RAW/f'evaluation/{arm}'
        if (DOC/f'{arm}_SCORE_COMPLETE.json').exists():continue
        dst,frames,altered=pe.replace_points(RAW,LINE/'baseline/FULL_CANDIDATES.json',path/'POINT_REPLACEMENTS.json',arm)
        pe.paper_2d(dst,dst/'PREDICTIONS.json',str(R0));pe.paper_pose(dst,frames,arm)
        write(DOC/f'{arm}_SCORE_COMPLETE.json',dict(complete=True,altered=altered,outputs=[bound(dst/f) for f in ('PREDICTIONS.json','PAPER_2D.json','PAPER_2D_per_frame.csv','POSE_PER_FRAME_BY_ARM.json')]))
        print('D_CANONICAL_SCORE_COMPLETE',s,flush=True)
if __name__=='__main__':infer() if 'infer' in sys.argv else score()
