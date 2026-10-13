"""Synchronized CAL-only live postprocessing latency, without detector timing."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from .api import VisibilityCurriculum
from .run import no_annotations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:raise ValueError('Positive repeats required')
    args.output.mkdir(exist_ok=False,parents=True)
    sources=[p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')]
    hashes={p.name:C.sha(p) for p in sources}
    C.write(args.output/'PROTOCOL.json',dict(partition='CAL_ONLY',repeats=args.repeats,warmup=5,
        code_sha256=hashes,config=C.read(args.config),config_sha256=C.sha(args.config),
        timing_scope='resident RGB and upstream KP/initial pose through live postprocessor, batch1',
        detector_forward=False, upstream_N3_forward=False,image_decode_timed=False,GT_read=False,
        clock='time.perf_counter with CUDA synchronize before/after each call'))
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    with no_annotations():
        frames=C.read(C.SCRATCH/'CAL_INPUTS.json')['frames']
        images=[cv2.imread(str(C.ROOT/f['image'])) for f in frames]
        assert all(x is not None for x in images)
        engine=VisibilityCurriculum.from_config(args.config)
        def operation(index):
            frame=frames[index]
            if torch.cuda.is_available():torch.cuda.synchronize()
            begin=time.perf_counter()
            result=engine.refine(images[index],frame['points']['N3_SUBPIX'],frame['K'],frame['initial_poses']['N3_SUBPIX'])
            if torch.cuda.is_available():torch.cuda.synchronize()
            return (time.perf_counter()-begin)*1000,result
        cold_ms,_=operation(0)
        for i in range(5):operation(i)
        rows=[]
        for repeat in range(args.repeats):
            for i,frame in enumerate(frames):
                elapsed,result=operation(i)
                rows.append(dict(id=frame['id'],session=frame['session'],repeat=repeat,
                    synchronized_wall_ms=elapsed,accepted=result['solution']['accepted'],
                    evidence_ms=result['timing']['evidence_seconds']*1000,
                    solve_ms=result['timing']['solve_seconds']*1000))
            print('REPEAT',repeat+1,len(frames),flush=True)
    assert hashes=={p.name:C.sha(p) for p in sources},'Code changed during benchmark'
    C.save_rows(args.output/'TIMINGS.jsonl.gz',rows)
    ms=np.asarray([r['synchronized_wall_ms'] for r in rows])
    C.write(args.output/'RESULTS.json',dict(cold_first_call_ms=cold_ms,frames=len(frames),rows=len(rows),
        median_ms=float(np.median(ms)),P90_ms=float(np.quantile(ms,.9)),mean_ms=float(ms.mean()),
        min_ms=float(ms.min()),max_ms=float(ms.max()),
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        scope='postprocessing only, not end-to-end detector/N3 FPS',source_changed=False,
        TIMINGS_sha256=C.sha(args.output/'TIMINGS.jsonl.gz')))
    print('LATENCY',np.median(ms),np.quantile(ms,.9),flush=True)


if __name__=='__main__':main()
