"""Portable CPU checks: five full-frame mask-head forwards and1784 map lookups.

No real RGB, annotations, backbone weights, optimizer or PnP are read.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import types
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
ARMS={'FULL_CLIP':'PLAIN_CLIP','FULL_CLIP_SURGERY_ARCH':'CLIP_SURGERY_ARCH'}


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in ('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def relocate(binding):
    original=Path(binding['path']);marker='V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
    assert marker in original.parts
    p=safe(HERE/Path(*original.parts[original.parts.index(marker)+1:]))
    raw=p.read_bytes()
    assert len(raw)==binding['size_bytes']and hashlib.sha256(raw).hexdigest()==binding['sha256'],str(p)
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='fullframe_replay_v1')
    args=parser.parse_args();assert Path(args.out).name==args.out and'..'not in args.out
    out=safe(HERE/args.out);assert not out.exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    receipt=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    freeze=json.loads(relocate(receipt['configuration']));assert freeze['PASS']and freeze['threshold']==.5
    mb=freeze['model_code'];m=types.ModuleType('v267_portable_head');m.__file__=str(HERE/'model_mask_head_v267.py');sys.modules[m.__name__]=m
    exec(compile(relocate(mb),m.__file__,'exec'),m.__dict__)
    fixture=np.load(io.BytesIO(relocate(receipt['forward_replay_inputs'])),allow_pickle=False)
    maps=np.load(io.BytesIO(relocate(receipt['frame_masks'])),allow_pickle=False)
    queries=[json.loads(x)for x in relocate(receipt['current_query_inputs']).splitlines()]
    pred=[json.loads(x)for x in relocate(receipt['predictions']).splitlines()]
    assert len(queries)==len(pred)==1784
    reports={}
    with torch.inference_mode():
        for arm,sourcearm in ARMS.items():
            model=m.FullMaskHead().eval();cp=torch.load(io.BytesIO(relocate(freeze['checkpoints'][sourcearm])),map_location='cpu')
            model.load_state_dict(cp['state_dict'],strict=True)
            actual=model(torch.as_tensor(fixture[arm+'_tokens']).float()).numpy()[:,0]
            expected=fixture[arm+'_logit']
            delta=float(np.max(np.abs(actual-expected)));assert delta<1e-4
            full_map_delta=float(np.max(np.abs(expected-maps[arm+'_logit'][fixture['frame_index']])));assert full_map_delta==0
            maximum=0;labels=0
            for i,row in enumerate(queries):
                assert row['row_id']==pred[i]['row_id']and row['global_index']==i
                if row['available']:
                    q=np.asarray(row['original_q0'],np.float64);h,w=row['raw_hw']
                    grid=torch.as_tensor((q+.5)/np.array([w,h])*2-1,dtype=torch.float32).reshape(1,1,1,2)
                    field=torch.as_tensor(maps[arm+'_logit'][row['frame_index']]).reshape(1,1,32,32)
                    score=float(F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0,0])
                    label='D'if score>=0 else'E';expected_score=-pred[i][arm]['E_logit']
                    maximum=max(maximum,abs(score-expected_score))
                else:label='U'
                assert label==pred[i][arm]['pred'];labels+=1
            assert maximum<1e-5
            reports[arm]=dict(full_frame_head_forward_rows=len(fixture['frame_index']),full_head_logit_max_abs_diff=delta,
                fixture_map_byte_values_match=True,all1784_proxy_labels_identical=True,query_logit_max_abs_diff=maximum,
                original_U=sum(not r['available']for r in queries))
    out.mkdir();result=dict(completed=True,PASS=True,arms=reports,source_RGB_or_GT_or_backbone_reads=0,
        new_optimizer_or_PnP=0,scope='Mask-headforwardfivefixedfullframes/arm plus all1784rawmaplookups; no fullbackbone refoward.',
        runtime=dict(torch=torch.__version__,numpy=np.__version__))
    with safe(out/'COMPLETE.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
