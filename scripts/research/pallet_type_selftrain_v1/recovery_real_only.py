"""Two-arm replay ablation under a frozen detector. No new labels or selection."""
import argparse
from pathlib import Path
from unittest.mock import patch
from . import recovery_common as R
from . import recovery_pose as P
from .recovery_repeat import TracedPoseTrainer

C=R.C
PHASE='pose_real_only'
ARMS={f'{t}_REAL_ONLY':dict(target=t,lr=1e-5) for t in ['RAW','REF']}


def real_slots(parent):
    assert len(parent)==1024
    first,second=parent[:512],parent[512:]
    assert all(Path(p).name.startswith('syn__') for p in first)
    assert all(Path(p).name.startswith('PLASTIC__') for p in second)
    assert len(set(second))==217
    return second+second


def prepare():
    base=C.read(R.DOC/'pose_only/PROTOCOL.json')
    sources=base['sources']+[C.bound(R.DOC/'pose_only/PROTOCOL.json'),C.bound(__file__),
        C.bound(Path(__file__).with_name('test_recovery_real_only.py')),C.bound(Path(__file__).with_name('recovery_repeat.py')),
        C.bound(R.RAW/'pose_flip/SCREEN_AUDIT.json')]
    for b in sources+base['inputs']:C.verify(b)
    with R.scope(PHASE):
        datasets={}
        for target in ['RAW','REF']:
            old=C.ROOT/base['datasets'][target]['train_list']['path']
            slots=real_slots(old.read_text().splitlines())
            folder=C.RAW/'dataset'/target
            C.write_text(folder/'train.txt','\n'.join(slots)+'\n')
            C.write_text(folder/'val.txt',(R.BASE_RAW/'dataset/val.txt').read_text())
            C.write_text(folder/'data.yaml',f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
            datasets[target]=dict(data=C.bound(folder/'data.yaml'),train_list=C.bound(folder/'train.txt'),
                sampled_real_unique=217,real_slots=1024,synthetic_slots=0,original_unique_membership_unchanged=True)
        a,b=[(C.ROOT/datasets[t]['train_list']['path']).read_text().splitlines() for t in ['RAW','REF']]
        assert [Path(x).name for x in a]==[Path(x).name for x in b]
        protocol=dict(**{k:v for k,v in base.items() if k not in ['arms','datasets','sources','controls','real','source_negatives']},
            arms=ARMS,datasets=datasets,sources=sources,source_negatives=0,
            real='Exactly the same217 unique pseudo images and trusted masks; each original512-real-slot multiset duplicated to1024; no new frame selection or teacher refresh.',
            controls='RAW_REAL_ONLY vs REF_REAL_ONLY; existing RAW_LR5/REF_LR5 are50percent synthetic controls. Existing SYN_LR5 is100percent synthetic control. Same R0 initialization, lr1e-5,5epochs/320updates, flip disabled.',
            rationale='Frozen detector/backbone eliminate detector forgetting in this stage; synthetic pose gradients may help or interfere. Ablate source replay while holding optimizer budget and real image membership fixed. Real exposure doubles, which is part of this intervention, not a separately isolated mechanism.',
            predeclared_decision='All2 run regardless of intermediate result. Repeat only if REF_REAL_ONLY PCK20 exceeds REF_LR5 and R0, with MAIN IoU3D and matched8cornerP90 no worse than REF_LR5. Otherwise preserve previous candidate. No learning-rate/step sweep.',
            constraints='No new GT, no final-model promotion, no geometry/shape/X/IoU selection, officialC2 unchanged; repeated DEV194 with teacher overlap is not independent confirmation.')
        C.freeze(C.DOC/'PROTOCOL.json',protocol)
    R.evaluation_protocol(PHASE,list(ARMS),sources+[C.bound(R.DOC/PHASE/'PROTOCOL.json')])
    print('REAL_ONLY_PROTOCOL_LOCKED',list(ARMS),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare',*ARMS]);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:
        with patch.object(P,'PHASE',PHASE),patch.object(P,'PoseOnlyTrainer',TracedPoseTrainer):P.train(a.action)
