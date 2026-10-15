"""One real fixed CAL R1 CLI check with N3/SubPix/unused imports blocked."""
from __future__ import annotations
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
import torch
from . import cli, runtime as RT
from .batch import MODEL_PATHS, Budget, _input, check_binding, compare, read, write
from .instrumentation import instrument


def run(root,input_dir,folder):
    root,folder=Path(root),Path(folder);protocol=check_binding(root,folder)
    inputs,_=_input(input_dir,protocol)
    prediction_seal=read(folder/'PREDICTION_SEAL.json')
    if prediction_seal['predictions_sha256']!=RT.sha256(folder/'PREDICTIONS.jsonl.gz'):
        raise ValueError('Whole accuracy payload binding required')
    fid=protocol['pilot_ids'][0]
    i,f=next((i,f)for i,f in enumerate(inputs['frames'])if f['id']==fid)
    source=f['arms']['RAW_YOLO'];output=folder/'real_R1_cli';input_path=folder/'REAL_R1_CLI_INPUT.json'
    write(input_path,dict(id=fid,prediction_source='RAW_YOLO',image=str(root/f['image']),
        image_sha256=f['image_sha256'],raw_hw=f['raw_hw'],K=f['K'],
        native_points=source['native_points'],coarse_pose=source['coarse_pose']))
    budget=Budget(folder.parent);budget.check('gpu_process');start=time.monotonic();status='technical_incomplete'
    try:
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with instrument(None) as meter:
            argv=['run','--input',str(input_path),'--output',str(output),
                  '--asset-root',str(root),'--device','cuda']
            for key,path in MODEL_PATHS.items():argv.extend(['--'+key.replace('_','-'),str(root/path)])
            cli.main(argv);torch.cuda.synchronize();counts=meter.snapshot()
        expected_obs=read(folder/'observations'/f'{i:03d}_RAW_YOLO.json')['result']
        observed=read(output/'OBSERVATION.json')
        for key in ('states','initial_geometry','state_observation','mask_audit','visible_correction','frozen_visible_measurements'):
            compare(expected_obs['observation'][key],observed['observation'][key],key,atol=1e-8,rtol=1e-8)
        expected_pose=read(folder/'poses'/f'{i:03d}_RAW_YOLO.json')['result']
        compare(expected_pose,read(output/'POSE.json'),'real_cli_pose',atol=1e-9,rtol=0)
        with np.load(output/'MASKS.npz')as a,np.load(folder/expected_obs['mask_binding']['path'])as b:
            if any(not np.array_equal(a[k],b[k])for k in a.files):raise AssertionError('CLI masks differ')
        write(folder/'REAL_R1_CLI_CHECK.json',dict(status='PASS',id=fid,selection='first prespecified CAL pilot ID',
            observations_and_mask_arrays_match_accuracy=True,final_pose_and_hypothesis_acceptance_match=True,
            actual_calls=counts,N3_assets_and_SubPix_blocked=True,GT_inputs=False,
            elapsed_seconds=time.monotonic()-start,elapsed_scope='cold CLI including decode and model load; no YOLO',
            source_sha256={'cli.py':RT.sha256(cli.__file__),'runtime.py':RT.sha256(RT.__file__),
                Path(__file__).name:RT.sha256(__file__)}))
        status='complete'
    except Exception:
        (folder/'REAL_R1_CLI_TECHNICAL_FAILURE.txt').write_text(traceback.format_exc());raise
    finally:budget.record('real_R1_CLI_check','gpu_process',time.monotonic()-start,status=status)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in('root','input-dir','folder'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.root,a.input_dir,a.folder)
