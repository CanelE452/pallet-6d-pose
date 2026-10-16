"""Replay the live API on prediction-only inputs, then seal before scoring."""
import argparse
from pathlib import Path
from collections import Counter
import time

import cv2
import numpy as np

from . import common as C
from .api import VisibilityCurriculum
from .run import no_annotations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition',choices=('calibration','test'),required=True)
    parser.add_argument('--config-v1',type=Path,required=True)
    parser.add_argument('--config-v2',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    partition={'calibration':'CAL','test':'TEST'}[args.partition]
    ids=set(C.read(C.DOC/'SPLIT.json')['real_split']['frame_ids'][partition])
    paths=[args.config_v1,args.config_v2]
    files=[p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')]
    code={p.name:C.sha(p) for p in files}
    model_bindings={}
    for path in paths:
        config=C.read(path)
        for name in ('visibility_model','segmentation_weights'):
            if config.get(name):
                weight=(path.resolve().parent/config[name]).resolve()
                model_bindings[str(weight)]=C.sha(weight)
    C.write(args.output/'PROTOCOL.json',dict(partition=args.partition,ids=sorted(ids),
        config_sha256={str(p):C.sha(p) for p in paths},model_sha256=model_bindings,code_sha256=code,
        prediction_inputs_sha256=C.sha(C.DOC/'PREDICTION_INPUTS.json'),contract_sha256=C.sha(C.DOC/'DATA_CONTRACT.json'),
        prior_TEST_metrics_already_observed=args.partition=='test',
        evaluation_status='reused historical held-session exploratory evaluation' if args.partition=='test' else 'CAL model-selection replay',
        candidate_selection='fixed strict RGB boundary arm selected on CAL, source and RGB heads unchanged',
        new_detector_forward=0,reprojected_points_used_for_fit=False,FINAL_accessed=False))
    results=[];cv2.setNumThreads(1);started=time.perf_counter()
    with no_annotations():
        frames=[f for f in C.prediction_inputs() if f['id'] in ids]
        assert len(frames)==len(ids)
        engines={'SEGMENTED_RGB_CURRICULUM':VisibilityCurriculum.from_config(args.config_v1),
                 'SEGMENTED_RGB_BOUNDARY_CURRICULUM':VisibilityCurriculum.from_config(args.config_v2)}
        for index,frame in enumerate(frames):
            path=C.ROOT/frame['image']
            if C.sha(path)!=frame['image_sha256']:raise ValueError('Image binding changed: '+frame['id'])
            image=cv2.imread(str(path))
            if image is None:raise FileNotFoundError(str(path))
            initial=frame['initial_poses']['N3_SUBPIX'];q=np.asarray(frame['points']['N3_SUBPIX'],float)
            for name,engine in engines.items():
                result=engine.refine(image,q,frame['K'],initial)
                results.append(dict(id=frame['id'],session=frame['session'],arm='N3_SUBPIX',method=name,
                    actual_pose=result['actual_pose'],solution=result['solution'],self_hidden=result['self_hidden'],
                    appearance=result['appearance'],original_points=q[:8].tolist(),
                    native_points=result['completed_points'],projected_points=result['projected_points'],
                    correction=result.get('correction'),raw_rgb_appearance=result.get('raw_rgb_appearance'),
                    corrected_measurements=result.get('corrected_measurements'),provenance=result['provenance'],timing=result['timing']))
                if name=='SEGMENTED_RGB_CURRICULUM':
                    results.append(dict(id=frame['id'],session=frame['session'],arm='N3_SUBPIX',method='INITIAL',
                        actual_pose=initial,solution=dict(accepted=False,R=initial['R_cf'],t=initial['centroid'],
                            used_ids=list(range(8)),reason='frozen_initial'),self_hidden=result['self_hidden'],
                        appearance=result['appearance'],original_points=q[:8].tolist(),native_points=q.tolist(),
                        projected_points=C.project(C.cuboid(*initial['cf_extents']),initial['R_cf'],initial['centroid'],frame['K']).tolist()))
            if (index+1)%10==0:print('LIVE_INFER',index+1,len(frames),round(time.perf_counter()-started,1),flush=True)
    assert code=={p.name:C.sha(p) for p in files},'Code changed during inference'
    C.save_rows(args.output/'PREDICTIONS.jsonl.gz',results)
    C.write(args.output/'PREDICTION_SEAL.json',dict(before_scoring=True,complete=True,
        sha256=C.sha(args.output/'PREDICTIONS.jsonl.gz'),rows=len(results),
        methods=dict(Counter(r['method'] for r in results)),GT_read_canary='PASS',seconds=time.perf_counter()-started))
    print('SEALED',len(results),args.output,flush=True)


if __name__=='__main__':main()
