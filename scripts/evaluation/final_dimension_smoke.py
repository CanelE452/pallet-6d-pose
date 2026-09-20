"""Forward-only parity check on one ALREADY evaluated historical DEV image."""
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from scripts.evaluation import final_dimension_release as R


def main():
    torch.set_num_threads(4)
    lock=R.checked_lock(); E=R.setup()
    from inference import load_head,predict_captured
    gpu=E.gpu()
    cache=R.read(R.DCP/'DEV_CACHE_COMPLETE.json')['records'][0]
    row=torch.load(R.ROOT/cache['path'],map_location='cpu',weights_only=False)
    cap=row['captured']
    for k in ('p3','p4'): cap[k]=cap[k].cuda()
    checks=[]
    for arm in R.ARMS:
        for seed in (1,2,3):
            head,_=load_head(arm,seed);name=f'{arm}_seed{seed}'
            pred,_=predict_captured(head,arm,cap,row['dimensions'],row['order'],
                lock['temperatures'][name],lock['decode_rule'],row['raw_hw'],
                R.read(R.DCP/'DIM_NORMALIZATION_LOCK.json'))
            old=next(r for r in R.read(E.RAW/f'predictions/REAL_DEV/{name}.json')['records'] if r['id']==row['id'])
            assert pred['selected_index']==old['selected_index'] and len(pred['candidates'])==len(old['candidates'])
            diffs=[float(np.abs(np.array(a['keypoints_xy'])-np.array(b['keypoints_xy'])).max()) for a,b in zip(pred['candidates'],old['candidates'])]
            maximum=max(diffs,default=0.)
            assert maximum<=1e-3,(name,maximum)
            checks.append(dict(model=name,max_abs_corner_difference_px=maximum,allowed_absolute_px=.001))
            del head
    # Validate the detector/image path on the same historical frame, never new green labels.
    from dev_evaluate import population_metadata
    pe,pop=population_metadata()
    item=next(item for item,meta in pop if item.frame_id==row['id'])
    image_path=R.ROOT/item.image
    extractor=E.old('features').FrozenYoloFeatures(E.R0)
    try: fresh=extractor.predict(cv2.imread(str(image_path)))
    finally: extractor.close()
    assert fresh['selected_index']==cap['selected_index']
    assert len(fresh['candidates'])==len(cap['candidates'])
    difference=max((float(np.abs(a['keypoints_xy']-b['keypoints_xy']).max()) for a,b in zip(fresh['candidates'],cap['candidates'])),default=0.)
    assert difference<=1e-3,difference
    result=dict(PASS=True,training_updates=0,green_predictions=0,historical_DEV_frame=row['id'],
        model_lock=R.binding(R.DOC/'MODEL_LOCK.json'),gpu=gpu,heads=checks,
        full_image_baseline_parity_max_abs_px=difference,code=R.binding(Path(__file__)))
    R.freeze(R.DOC/'INFERENCE_SMOKE.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
