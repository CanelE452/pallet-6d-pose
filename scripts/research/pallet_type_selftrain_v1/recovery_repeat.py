"""Actual order replicates: indexed aliases survive Ultralytics path sorting."""
import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import recovery_common as R
from . import recovery_pose as P
from .recovery_pose_trainer import PoseOnlyTrainer
from . import train as T
C=R.C
PHASE='pose_repeat'
ARMS={f'{target}_ORDER{seed}':dict(target=f'{target}_ORDER{seed}',base_target=target,lr=1e-5,order_seed=seed)
      for seed in [43,44] for target in ['SYN','RAW','REF']}


class TracedPoseTrainer(PoseOnlyTrainer):
    def preprocess_batch(self,batch):
        out=super().preprocess_batch(batch)
        if not hasattr(self,'recovery_first_batch'):
            self.recovery_first_batch=dict(files=out['im_file'],
                image_tensor_sha256=hashlib.sha256(out['img'].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                keypoint_tensor_sha256=hashlib.sha256(out['keypoints'].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            C.freeze(C.DOC/f'TRACE_{self.args.name}.json',self.recovery_first_batch)
        return out


def prepare():
    base=C.read(R.DOC/'pose_only/PROTOCOL.json')
    selection={}
    for name in base['arms']:
        r=C.read(R.RAW/'pose_only'/f'SCREEN_{name}.json')
        selection[name]=dict(PCK20=r['symmetry']['PCK']['20'],IoU3D=r['pose']['iou3d_median'])
    assert selection['REF_LR5']['PCK20']>max(selection[n]['PCK20'] for n in selection if n!='REF_LR5')
    sources=base['sources']+[C.bound(R.DOC/'pose_only/PROTOCOL.json'),C.bound(R.RAW/'pose_only/SCREEN_REF_LR5.json'),C.bound(__file__)]
    for binding in sources:C.verify(binding)
    with R.scope(PHASE):
        datasets={};inputs=[];aliases={}
        for arm,spec in ARMS.items():
            old=(C.ROOT/base['datasets'][spec['base_target']]['train_list']['path']).read_text().splitlines()
            assert len(old)==1024
            indices=np.random.default_rng(spec['order_seed']).permutation(len(old));ordered=[Path(old[i]) for i in indices]
            folder=C.RAW/'dataset'/arm;slots=[];mapping=[]
            for i,image in enumerate(ordered):
                name=f'{i:04d}__{image.name}';dest=folder/'images'/name;lbl=image.parent.parent/'labels'/image.with_suffix('.txt').name
                target=folder/'labels'/Path(name).with_suffix('.txt').name
                T.link(image,dest);T.link(lbl,target)
                slots.append(str(dest));mapping.append(dict(slot=i,source=str(image.relative_to(C.ROOT)),image=C.bound(dest),label=C.bound(target)))
                inputs.extend([C.bound(dest),C.bound(target)])
            assert slots==sorted(slots),'Indexed aliases must survive dataset sorting'
            C.write_text(folder/'train.txt','\n'.join(slots)+'\n')
            C.write_text(folder/'val.txt',(R.BASE_RAW/'dataset/val.txt').read_text())
            C.write_text(folder/'data.yaml',f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
            datasets[arm]=dict(data=C.bound(folder/'data.yaml'),train_list=C.bound(folder/'train.txt'),
                              sampled_real_unique=base['datasets'][spec['base_target']]['sampled_real_unique'])
            aliases[arm]=mapping
        for seed in [43,44]:
            a,b=[aliases[f'{t}_ORDER{seed}'] for t in ['RAW','REF']]
            assert [x['image']['sha256'] for x in a]==[x['image']['sha256'] for x in b]
        assert [x['image']['sha256'] for x in aliases['REF_ORDER43']]!=[x['image']['sha256'] for x in aliases['REF_ORDER44']]
        protocol=dict(**{k:v for k,v in base.items() if k not in ['arms','datasets','inputs','sources']},
            arms=ARMS,datasets=datasets,inputs=inputs,sources=sources,
            selected_lr=1e-5,screen_selection=selection,selection_claim='Highest predeclared primary PCK20 among six exploratory arms, not independent validation',
            repeats='Original optimizer seed42 fixed. Two additional order seeds43/44 permute the exact1024 exposures. Indexed path aliases prevent BaseDataset.sorted() undoing this permutation.',
            stochasticity='Image membership/multiplicity/targets and budget unchanged; ordering changes which transformations/samples meet in a batch. First-batch tensor fingerprints and final checkpoint differences must be verified.',
            detection='Same R0 detector/backbone/BN frozen as screening; no per-image test filter',
            stop='Run both order replicates and matched raw/source controls regardless of intermediate results; no best-repeat selection')
        C.freeze(C.DOC/'PROTOCOL.json',protocol);C.freeze(C.RAW/'ALIASES.json',aliases)
    R.evaluation_protocol(PHASE,list(ARMS),sources+[C.bound(R.DOC/PHASE/'PROTOCOL.json')])
    print('REPEAT_PROTOCOL_LOCKED',list(ARMS),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare',*ARMS]);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        with patch.object(P,'PHASE',PHASE),patch.object(P,'PoseOnlyTrainer',TracedPoseTrainer):P.train(a.action)
