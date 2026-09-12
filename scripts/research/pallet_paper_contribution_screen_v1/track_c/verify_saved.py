"""Independent saved-checkpoint and row-level checks, not optimizer receipts only."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,R0,sha,tensor_sha,write
from track_c.wiring import load_model

OUT=DOC/'C_geometry_preserving_da/RESUME_COMPLETE'

def detection_state_keys(model):
    """Module-object ownership, independently enumerated state keys."""
    head=model.model[-1]
    modules=set()
    for branch in (head.one2many,head.one2one):
        for key in ('box_head','cls_head'):
            modules.update(id(m) for m in branch[key].modules())
    keys=set()
    for prefix,module in model.named_modules():
        if id(module) in modules:
            keys.update(f'{prefix}.{name}' for name,_ in module.named_parameters(recurse=False))
            keys.update(f'{prefix}.{name}' for name,_ in module.named_buffers(recurse=False))
    return keys

def dense(model,x):
    model.eval();model.training=True;model.model[-1].training=True
    with torch.no_grad():
        outputs=model(x)
    return {k:v['kpts'] for k,v in outputs.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[1,2,3]);args=parser.parse_args()
    torch.set_num_threads(2)
    baseline=load_model().eval();base_state=baseline.state_dict();mutable=detection_state_keys(baseline)
    checks=[]
    for seed in ([args.seed] if args.seed else [1,2,3]):
        path=RAW/f'C_geometry_preserving_da/C2_seed{seed}/last.pt'
        ck=torch.load(path,map_location='cpu');model=ck['model'].float()
        assert ck['train_args']['task']=='pose'
        assert set(model.state_dict())==set(base_state)
        frozen=[k for k in base_state if k not in mutable]
        changed=[k for k in frozen if not torch.equal(base_state[k],model.state_dict()[k])]
        assert not changed,changed
        assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in baseline.parameters())
        assert all(not p.requires_grad for n,p in model.named_parameters() if n not in mutable)
        actual_hash=tensor_sha(model.state_dict())
        torch.manual_seed(913)
        for size in [64,128]:
            x=torch.rand(2,3,size,size)
            a,b=dense(baseline,x),dense(model,x)
            assert set(a)==set(b)
            assert all(torch.equal(a[k],b[k]) for k in a)
        row=dict(seed=seed,status='PASS',checkpoint_sha256=sha(path),state_sha256=actual_hash,
            frozen_state_keys=len(frozen),frozen_keys_equal=True,raw_pose_bit_exact=True,
            probe_shapes=[[2,3,64,64],[2,3,128,128]],device='CPU',
            deployment_parameter_count=sum(p.numel() for p in model.parameters()),
            comparison_scope='Dense pose only; final detection selection is evaluated separately')
        checks.append(row);write(OUT/f'SAVED_C2_seed{seed}_AUDIT.json',row)
    if args.seed:
        print(checks,flush=True);return
    recomputed=[]
    for seed in [1,2,3]:
        for arm in ['C0','C1','C2']:
            name=f'{arm}_seed{seed}';base=RAW/'C_geometry_preserving_da'/name
            report=json.loads((base/'evaluation_pose/PAPER_2D.json').read_text())['metrics']['box_and_keypoint_2d']
            rows=list(csv.DictReader((base/'evaluation_pose/PAPER_2D_per_frame.csv').open()))
            errors=[float(v) for r in rows for v in r['top_keypoint_supervised_errors_px'].split(';') if v]
            med=float(np.median(errors));p90=float(np.quantile(errors,.9))
            assert abs(med-report['keypoint_location_median_px'])<1e-5
            assert abs(p90-report['keypoint_location_p90_px'])<1e-5
            audit=json.loads((base/'TRAINING_AUDIT.json').read_text())
            assert audit['optimizer_updates']==900 and sha(base/'last.pt')==audit['checkpoint_sha256']
            assert len(rows)==3008
            recomputed.append(dict(name=name,median_px=med,p90_px=p90,status='PASS',evaluated_frames=3008))
    write(OUT/'INDEPENDENT_SAVED_AUDIT.json',dict(status='PASS',C2_checkpoints=checks,
        CSV_recomputations=recomputed,valid_fit_updates=8100,lost_prior_fit_updates=900,cumulative_updates=9000,
        permitted_repair_updates=900))
    print('Independent saved-state checks and all9 CSV recomputations PASS',flush=True)

if __name__=='__main__':main()
