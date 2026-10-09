"""Pre-run fixed 26-frame solver cost measurement, without reading scoring truth."""
from collections import Counter
import time
import numpy as np
import cv2
from . import common as C
from .inference import hidden_mask,finish
from .evaluate import initial_pose

def run():
    from .solver import HypothesisBank
    assert C.read(C.DOC/'SOLVER_CHECKS.json')['passed']
    assert not (C.DOC/'CPU_PILOT.json').exists()
    C.source_modules()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy
    E,_=legacy();cv2.setNumThreads(1)
    panel=C.read(C.ROOT/'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json')['selected']
    inputs={r['id']:r for r in C.read(C.DOC/'INPUTS.json')['frames']}
    rows=[];counts=Counter();begin=time.monotonic()
    with C.no_truth_reads():
        for p in panel:
            f=inputs[p['frame_id']];t0=time.monotonic()
            for arm in ('BASE','N3_SUBPIX'):
                q=np.asarray(f['points'][arm],float);initial=initial_pose(E,q,f,counts);H,_=hidden_mask(initial)
                bank=HypothesisBank(q,f['K'],f['xyz'],image_size=(f['raw_hw'][1],f['raw_hw'][0]))
                for suffix in C.SUFFIXES[:4]:
                    hidden=H if suffix.startswith('GEOM') else []
                    result=finish(bank,q,initial,excluded=hidden,hidden=hidden,robust=suffix.endswith('ROBUST'))
                    rows.append(dict(id=f['id'],method=arm+'_'+suffix,new_pose_estimated=result['new_pose_estimated'],counts=result['solver']['operation_counts']))
            print('CPU PILOT',len(rows)//8,round(time.monotonic()-t0,3),flush=True)
    elapsed=time.monotonic()-begin
    C.write(C.DOC/'CPU_PILOT.json',dict(complete=True,panel_ids=[p['frame_id'] for p in panel],sessions=13,
        rows=rows,initial_counts=dict(counts),seconds=elapsed,rough_full_12path_seconds=elapsed*319/26*1.5,
        accuracy_score_read=False,not_deployment_latency=True,extra_pilot_logical_paths=len(rows),initial_calls=52,
        maximum_CPU_main_seconds=3600))

if __name__=='__main__':run()
