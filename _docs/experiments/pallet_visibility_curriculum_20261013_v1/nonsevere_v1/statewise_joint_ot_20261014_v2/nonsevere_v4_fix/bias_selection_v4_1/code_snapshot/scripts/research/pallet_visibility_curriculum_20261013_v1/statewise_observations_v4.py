"""Fresh RGB observations without pose solving or annotation access."""
import os
for _setting in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_setting]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2
import torch

from . import common as C
from .statewise_joint_experiment_v2 import _frames,source_bindings,crop_bindings,STUDY,COHORT
from .state_visibility_v2 import load_state_model
from .statewise_masks_v2 import StatewiseMaskProvider
from .statewise_joint_api_v4 import StatewiseJointV4


_CONTEXT=None


def bindings():
    result=source_bindings()
    for name in ('state_visibility_v4.py','statewise_joint_api_v4.py',
                 'statewise_observations_v4.py','test_state_visibility_v4.py'):
        path=Path(__file__).parent/name
        result[str(path.relative_to(C.ROOT))]=C.sha(path)
    return result


def init_worker(device):
    global _CONTEXT
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=C.read(STUDY/'RUNTIME_CONFIG_V3.json')
    model_path=STUDY/config['state_model'];model=load_state_model(str(model_path),device)
    masks=StatewiseMaskProvider.from_configs(C.read(STUDY/config['mask_descriptor']),
                                           COHORT/'BOUNDARY_OBSERVATION_CONFIG.json',device)
    _CONTEXT=(model,C.read(STUDY/config['OT_model']),masks,device)


def observe_frame(frame):
    model,ot,provider,device=_CONTEXT
    path=C.ROOT/frame['image']
    if frame['image_sha256'] and C.sha(path)!=frame['image_sha256']:
        raise ValueError('RGB bytes changed: '+frame['id'])
    image=cv2.imread(str(path))
    if image is None or list(image.shape[:2])!=frame['raw_hw']:
        raise ValueError('Actual RGB canvas mismatch')
    api=StatewiseJointV4(model,ot,provider,config=dict(prediction_source='N3_SUBPIX',
         record_id=frame['id'],device=device))
    observed=api.observe(image,frame['raw_points'],frame['K'],
        C.cuboid(*frame['initial_pose']['cf_extents']),frame['initial_pose'],
        crop_transform=frame['crop_transform'])
    return dict(id=frame['id'],source_record_id=frame['source_record_id'],
        partition=frame['partition'],session=frame['session'],cohort=frame['cohort'],
        image=frame['image'],image_sha256=C.sha(path),K=frame['K'],raw_hw=frame['raw_hw'],
        crop_transform=frame['crop_transform'],initial_pose=frame['initial_pose'],runtime=observed)


def predict(output,workers=3,device='cuda'):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve completed observation outputs')
    if workers not in (1,2,3):raise ValueError('One to three workers')
    frames=_frames();output.mkdir(parents=True)
    protocol=dict(schema='FRESH_RGB_OBSERVATIONS_V4',ids=[f['id']for f in frames],
        primary_population=dict(CAL=46,TEST=49,severe=0),crop_primary=False,
        frozen_upstream='N3_SUBPIX',annotations_read=False,new_RGB_forwards=True,
        pose_solver_calls=0,state_gate_revision='available_in_canvas_nonS_RGB_query_survives_coarse_T',
        initial_S_unchanged=True,S_image_queries_allowed=0,
        initial_projection_is_guidance_not_coordinate_observation=True,
        source_sha256=bindings(),crop_source_sha256=crop_bindings(),
        frozen_runtime_descriptor=dict(path=str((STUDY/'RUNTIME_CONFIG_V3.json').relative_to(C.ROOT)),
                                       sha256=C.sha(STUDY/'RUNTIME_CONFIG_V3.json')),
        prediction_input_sha256=C.sha(COHORT/'PREDICTION_INPUTS.json'),
        TEST_status='REUSED_EXPLORATORY_NOT_NEW_UNSEEN',workers=workers)
    C.write(output/'PROTOCOL.json',protocol)
    for rel,h in protocol['source_sha256'].items():
        dest=output/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/rel,dest)
        if C.sha(dest)!=h:raise ValueError('Snapshot mismatch')
    start=time.monotonic();by_id={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),
           initializer=init_worker,initargs=(device,)) as pool:
        futures={pool.submit(observe_frame,f):f['id'] for f in frames}
        for future in as_completed(futures):
            row=future.result()
            if row['id']!=futures[future] or row['id'] in by_id:raise ValueError('Record binding mismatch')
            by_id[row['id']]=row
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(by_id),total=len(frames),
                last_id=row['id'],elapsed_seconds=time.monotonic()-start))
            print(f"{len(by_id)}/{len(frames)} {row['id']} states={''.join(row['runtime']['states'])}",flush=True)
    if bindings()!=protocol['source_sha256']:raise ValueError('Source changed during extraction')
    values=[by_id[f['id']]for f in frames]
    C.save_rows(output/'OBSERVATIONS.jsonl.gz',values)
    C.save_rows(output/'CAL_OBSERVATIONS.jsonl.gz',[
        r for r in values if r['partition']=='CAL'and r['cohort']=='PRIMARY_NONSEVERE'])
    C.write(output/'OBSERVATION_SEAL.json',dict(schema='RGB_OBSERVATION_SEAL_V4',
        before_new_uncertainty_fitting=True,annotations_read=False,rows=len(values),
        observations_sha256=C.sha(output/'OBSERVATIONS.jsonl.gz'),
        CAL_observations_sha256=C.sha(output/'CAL_OBSERVATIONS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=protocol['source_sha256'],
        elapsed_seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--device',default='cuda')
    a=p.parse_args();predict(a.output,a.workers,a.device)


if __name__=='__main__':main()
