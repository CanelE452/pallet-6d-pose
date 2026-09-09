"""Creates explicitly GENERATED software-test data, never pallet results."""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import json,sys
import torch
from conftest import make_observation
from pointline_v4.cache_io import sha256,write_json,SCHEMA


def build(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);records=[]
    for i in range(8):
        o=make_observation(seed=i+10)
        op=root/f'obs_{i}.pt';tp=root/f'gt_{i}.pt'
        torch.save({f.name:getattr(o,f.name) for f in fields(o)},op)
        torch.save(dict(points=o.baseline+.25,supervised=o.point_valid,matched=torch.ones(1,dtype=torch.bool)),tp)
        records.append(dict(frame_id=f'generated_{i}',session_id=f'generated_session_{i%3}',origin='generated_fixture',state='native',
                            observation=op.name,observation_sha256=sha256(op),supervision=tp.name,supervision_sha256=sha256(tp)))
    for role,subset in [('train',records[:4]),('calibration',records[4:6]),('generated_test',records[6:])]:
        write_json(root/f'{role}.json',dict(schema=SCHEMA,role=role,channels=[2,3],records=subset))
    core=Path(__file__).resolve().parents[1]/'pointline_v4'
    protocol=dict(locked=True,seeds=[1],channels=[2,3],train_manifest_sha256=sha256(root/'train.json'),
        core_source_sha256={p.name:sha256(p) for p in sorted(core.glob('*.py'))},
        model=dict(visual=4,width=16,layers=1),training=dict(steps=2000,batch=2,lr=.001,weight_decay=.0001,gradient_clip=5.),
        loss=dict(temperature=.25,beta=.1,regression_weight=1.,corner_weight=.25),role='generated software smoke only')
    write_json(root/'protocol.json',protocol)
    return root

if __name__=='__main__':build(sys.argv[1])
