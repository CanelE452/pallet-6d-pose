"""Portable CPU:5fixed full-frame refiner forwards/arm +1784map lookups.

No real RGB decoding, target annotation, CLIP weights/backbone, fit or PnP.
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
ARMS=('HIGHRES_REFINE','BOUNDARY_REFINE')


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def relocate(b):
    original=Path(b['path']);marker=HERE.name
    if marker in original.parts:
        p=HERE/Path(*original.parts[original.parts.index(marker)+1:])
    else:
        old='V267_FULL_IMAGE_CLIP_VISIBLE_MASK';assert old in original.parts
        tail=Path(*original.parts[original.parts.index(old)+1:])
        local=HERE/'dependencies'/old/tail
        p=local if safe(local).exists()else HERE.parent/old/tail
    raw=safe(p).read_bytes()
    assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256'],str(p)
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='fullframe_replay_v1')
    args=parser.parse_args();assert Path(args.out).name==args.out and'..'not in args.out
    out=safe(HERE/args.out);assert not out.exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    receipt=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    freeze=json.loads(relocate(receipt['configuration']));assert freeze['PASS']and freeze['threshold']==.5
    mb=freeze['model_code'];m=types.ModuleType('v268_portable_refiner');m.__file__=str(HERE/'model_mask_refine_v268.py');sys.modules[m.__name__]=m
    exec(compile(relocate(mb),m.__file__,'exec'),m.__dict__)
    fixture=np.load(io.BytesIO(relocate(receipt['forward_replay_inputs'])),allow_pickle=False)
    index=json.loads(relocate(receipt['frame_masks']));assert index['resolution']==256
    queries=[json.loads(x)for x in relocate(receipt['current_query_inputs']).splitlines()]
    pred=[json.loads(x)for x in relocate(receipt['predictions']).splitlines()]
    assert len(queries)==len(pred)==1784
    inputs=[torch.as_tensor(fixture[k]).float()for k in('rgb256','tokens','coarse_logits')]
    reports={}
    with torch.inference_mode():
        for arm in ARMS:
            model=m.FullFrameRefiner().eval();cp=torch.load(io.BytesIO(relocate(freeze['checkpoints'][arm])),map_location='cpu')
            model.load_state_dict(cp['state_dict'],strict=True)
            actual=model(*inputs)
            expected=fixture[arm+'_logit'];actualraw=actual['mask_logits'].numpy()[:,0]
            delta=float(np.max(np.abs(actualraw-expected)));assert delta<1e-4
            assert float(actual['bounded_delta'].abs().max())<=6.000001
            with np.load(io.BytesIO(relocate(index['arms'][arm])),allow_pickle=False)as z:
                maps=np.array(z['logits'],copy=True)
                assert np.array_equal(z['frame_index'],np.arange(223))
            assert maps.shape==(223,256,256)
            assert np.array_equal(expected,maps[fixture['frame_index']])
            maximum=0.; labels=0
            for i,row in enumerate(queries):
                assert row['row_id']==pred[i]['row_id']and row['global_index']==i
                if row['available']:
                    q=np.asarray(row['original_q0'],np.float64);h,w=row['raw_hw']
                    grid=torch.tensor((q+.5)/np.array([w,h])*2-1,dtype=torch.float32).reshape(1,1,1,2)
                    field=torch.from_numpy(maps[row['frame_index']]).reshape(1,1,256,256)
                    score=float(F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0,0])
                    label='D'if score>=0 else'E';maximum=max(maximum,abs(score+pred[i][arm]['E_logit']))
                else:label='U'
                assert label==pred[i][arm]['pred'];labels+=1
            assert maximum<1e-5
            reports[arm]=dict(full_frame_refiner_forward_rows=len(fixture['frame_index']),full_frame_logit_max_abs_diff=delta,
                fixture_matches_frozen_full_maps_exactly=True,all1784_proxy_labels_identical=True,query_logit_max_abs_diff=maximum,
                original_U=sum(not r['available']for r in queries))
    out.mkdir();result=dict(completed=True,PASS=True,arms=reports,original_RGB_or_GT_or_backbone_reads=0,
        new_optimizer_or_PnP=0,scope='Fivefixedfull-frameRGBfeature/CLIPcontextrefinerforwardsperarmand1784rawmaplookups;nofullCLIPbackbonereforward.',
        runtime=dict(torch=torch.__version__,numpy=np.__version__))
    with safe(out/'COMPLETE.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
