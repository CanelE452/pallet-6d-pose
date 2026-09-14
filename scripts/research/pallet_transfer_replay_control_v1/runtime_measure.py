"""One fixed, non-selective end-to-end runtime panel after performance evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from evaluate import NAMES,E,checkpoint,read
from runtime import ROOT,DOC,atomic_json,sha


def main():
    assert read(DOC/'TRAINING_COMPLETE.json')['status']=='PASS'
    rows=read(ROOT/'_docs/experiments/pallet_active_learning_v1/retrospective_v1/SPLIT.json')['evaluation']
    order=sorted(rows,key=lambda r:hashlib.sha256(('replay-runtime-v1\n'+r['image_sha256']).encode()).hexdigest())[:26]
    result={};gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip()
    for name in NAMES:
        predictor=E._UltralyticsPredictor(checkpoint(name),'0')
        for row in order[:5]:predictor.predict(ROOT/row['image_path'])
        torch.cuda.synchronize();repeats=[]
        for repeat in range(3):
            values=[]
            for row in order:
                torch.cuda.synchronize();start=time.perf_counter();predictor.predict(ROOT/row['image_path']);torch.cuda.synchronize()
                values.append((time.perf_counter()-start)*1000)
            repeats.append(values)
        flat=[v for rr in repeats for v in rr]
        result[name]=dict(checkpoint_sha256=sha(checkpoint(name)),repeat_frame_ms=repeats,
            mean_ms=sum(flat)/len(flat),median_ms=float(np.median(flat)),n_frames=26,repeats=3,warmup=5)
        del predictor;torch.cuda.empty_cache();print(name,'runtime complete',flush=True)
    atomic_json(DOC/'RUNTIME.json',dict(status='COMPLETE',gpu=gpu,protocol='Fixed26 hash-ordered target DEV frames; 5 warmup then3 full repeats; synchronized end-to-end predictor calls; every repeat retained',
        selection='No fastest repeat/model selected; summaries pool all78 calls per model',results=result,
        caveat='Single workstation run in fixed model order; display/RustDesk remained active; not Jetson/export latency and not causal architecture speed isolation.'))


if __name__=='__main__':main()
