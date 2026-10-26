#!/usr/bin/env python3
"""Portable CPU: 5 fixed head forwards/arm and all1784 frozen-map query lookups.

Reads released-feature fixtures and small downstream head checkpoints only.
No full RGB, annotation/GT, CLIP/EVA/DeCLIP encoder weights, fit or PnP.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import stat
import sys
import types
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
MARKER='V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE'
ARMS=('EVA_STANDARD','EVA_CSA','EVA_DECLIP')
FIXTURE_FRAMES=(0,56,112,168,222)


def safe(value,fresh=False):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any(x in str(p).casefold() for x in ('v112','cache','.pyc')):
        raise ValueError(str(p))
    for a in [*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:continue
        if stat.S_ISLNK(mode):raise ValueError('Symlink: '+str(a))
    if fresh and p.exists():raise FileExistsError(str(p))
    return p


def relocate(binding):
    original=Path(binding['path'])
    if '..' in original.parts or any(x in str(original).casefold() for x in ('v112','cache','.pyc')):
        raise ValueError('Forbidden serialized binding')
    if MARKER in original.parts:
        p=HERE/Path(*original.parts[original.parts.index(MARKER)+1:])
    else:
        marker='V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
        if marker not in original.parts:raise ValueError('Undeclared dependency root: '+str(original))
        tail=Path(*original.parts[original.parts.index(marker)+1:])
        dep=HERE/'dependencies'/marker/tail
        p=dep if safe(dep).exists() else HERE.parent/marker/tail
    data=safe(p).read_bytes()
    if len(data)!=binding['size_bytes'] or hashlib.sha256(data).hexdigest()!=binding['sha256']:
        raise ValueError('SHA/size mismatch: '+str(p))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='fullframe_replay_v2')
    args=parser.parse_args()
    if Path(args.out).name!=args.out or '..' in args.out:raise ValueError('Single output name')
    out=safe(HERE/args.out,True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    assert receipt['completed'] and receipt.get('PASS',True)
    config=json.loads(relocate(receipt['configuration']))
    assert config['PASS'] and config['threshold']==.5 and tuple(config['train_arms'])==ARMS
    model_binding=config['model_code']
    model_path=HERE/'model_mask_head_v269.py'
    m=types.ModuleType('v269_portable_head_only')
    m.__file__=str(model_path);sys.modules[m.__name__]=m
    exec(compile(relocate(model_binding),str(model_path),'exec'),m.__dict__)
    with np.load(io.BytesIO(relocate(receipt['forward_replay_inputs'])),allow_pickle=False) as z:
        fixture={k:np.array(z[k],copy=True) for k in z.files}
    assert np.array_equal(fixture['frame_index'],np.array(FIXTURE_FRAMES))
    with np.load(io.BytesIO(relocate(receipt['frame_masks'])),allow_pickle=False) as z:
        assert np.array_equal(z['frame_index'],np.arange(223))
        maps={a:np.array(z[a+'_logit'],copy=True) for a in ARMS}
    queries=[json.loads(x) for x in relocate(receipt['current_query_inputs']).splitlines()]
    predictions=[json.loads(x) for x in relocate(receipt['predictions']).splitlines()]
    assert len(queries)==len(predictions)==1784
    reports={}
    with torch.inference_mode():
        for arm in ARMS:
            tokens=fixture['tokens'+arm]
            expected=fixture[arm+'_logit']
            assert tokens.shape==(5,1024,512) and expected.shape==(5,32,32)
            assert maps[arm].shape==(223,32,32)
            assert np.array_equal(expected,maps[arm][fixture['frame_index']])
            model=m.FullMaskHead().eval()
            checkpoint=torch.load(io.BytesIO(relocate(config['checkpoints'][arm])),map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            before=m.state_sha(model.state_dict())
            actual=model(torch.as_tensor(tokens).float()).numpy()[:,0]
            delta=float(np.max(np.abs(actual-expected)))
            assert delta<1e-4,(arm,delta)
            assert before==m.state_sha(model.state_dict())
            maximum=0.;count=0
            for i,(row,pred) in enumerate(zip(queries,predictions)):
                assert row['global_index']==pred['global_index']==i and row['row_id']==pred['row_id']
                assert row['native_slot']==pred['native_slot']==i%8 and row['available']==pred['available']
                if row['available']:
                    q=np.asarray(row['original_q0'],np.float64);h,w=row['raw_hw']
                    grid=torch.tensor((q+.5)/np.array([w,h])*2-1,dtype=torch.float32).reshape(1,1,1,2)
                    field=torch.from_numpy(maps[arm][row['frame_index']]).reshape(1,1,32,32)
                    logit=float(F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0,0])
                    label='D' if logit>=0 else 'E'
                    maximum=max(maximum,abs(logit+pred[arm]['E_logit']))
                else:label='U'
                assert label==pred[arm]['pred'],(arm,i,label,pred[arm]['pred'])
                count+=1
            assert maximum<1e-5
            reports[arm]=dict(head_forward_rows=5,fixture_frame_indices=list(FIXTURE_FRAMES),
                head_logit_max_abs_difference=delta,fixture_exactly_matches_frozen_fullmaps=True,
                head_state_unchanged=True,query_rows_checked=count,all1784_proxy_labels_identical=True,
                query_logit_max_abs_difference=maximum,original_U=sum(not r['available'] for r in queries))
    out.mkdir()
    result=dict(completed=True,PASS=True,arms=reports,encoder_or_full_RGB_or_GT_reads=0,
        optimizer_updates_or_PnP=0,head_forward_rows_total=15,
        scope='Fixed feature fixtures -> small SYN pallet head only; not a full encoder reproduction.',
        runtime=dict(torch=torch.__version__,numpy=np.__version__))
    safe(out/'COMPLETE.json',True).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
