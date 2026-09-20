"""Predeclared 2 learning rates x source/raw/refined controls, pose-only updates."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
from . import recovery_common as R
from . import train as T
from .pseudo import top
from .recovery_pose_trainer import PoseOnlyTrainer,pose_parameter
C=R.C
PHASE='pose_only'
ARMS={f'{target}_LR{power}':dict(target=target,lr=10.**(-power)) for power in [4,5] for target in ['SYN','RAW','REF']}


def paired_labels(row):
    raw=np.array(T.label(top(row['raw']),row['raw_hw']).split(),float)
    ref=np.array(T.label(top(row['refined']),row['raw_hw']).split(),float)
    assert np.array_equal(raw[:5],ref[:5])
    a,b=raw[5:].reshape(9,3),ref[5:].reshape(9,3)
    both=(a[:,2]==2)&(b[:,2]==2)
    for p in [a,b]:p[~both,:2]=.5;p[~both,2]=1
    assert np.array_equal(a[:,2],b[:,2])
    encode=lambda x:' '.join(f'{v:.9f}' for v in x)+'\n'
    return {'RAW':encode(raw),'REF':encode(ref)},int(both.sum())


def prepare():
    parent=C.read(R.BASE_DOC/'TRAIN_PROTOCOL.json')
    for binding in parent['inputs']+parent['sources']+[parent['code'],parent['initialization']]:C.verify(binding)
    candidates=[r for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC'];assert len(candidates)==249
    old_slots=(C.ROOT/parent['datasets']['PLASTIC']['train_list']['path']).read_text().splitlines()
    old_syn=[Path(p) for p in old_slots[:512]];assert len(set(old_syn))==512
    sources=[C.bound(R.BASE_DOC/'TRAIN_PROTOCOL.json'),C.bound(R.BASE_RAW/'PSEUDO_ACCEPTED.json'),
             C.bound(R.BASE_DOC/'DATA_VERIFICATION.json'),C.bound(__file__),C.bound(R.__file__),
             C.bound(Path(__file__).with_name('recovery_pose_trainer.py')),C.bound(T.__file__),
             C.bound(Path(__file__).with_name('test_recovery_pose.py'))]
    with R.scope(PHASE):
        datasets={};inputs=[]
        for target in ['SYN','RAW','REF']:
            folder=C.RAW/'dataset'/target;syn=[];real=[]
            for p in old_syn:
                dest=folder/'images'/p.name;lbl=p.parent.parent/'labels'/p.with_suffix('.txt').name
                T.link(p,dest);T.link(lbl,folder/'labels'/lbl.name);syn.append(str(dest));inputs.extend([C.bound(p),C.bound(lbl)])
            if target!='SYN':
                for row in candidates:
                    labels,_=paired_labels(row);image=R.BASE_RAW/'dataset/images'/f'{row["id"]}.png'
                    dest=folder/'images'/image.name;T.link(image,dest)
                    C.write_text(folder/'labels'/f'{row["id"]}.txt',labels[target]);real.append(str(dest))
                    inputs.extend([C.bound(image),C.bound(folder/'labels'/f'{row["id"]}.txt')])
            replacement=syn if target=='SYN' else np.random.default_rng(9021).choice(sorted(real),512,replace=True).tolist()
            C.write_text(folder/'train.txt','\n'.join(syn+replacement)+'\n')
            C.write_text(folder/'val.txt',(R.BASE_RAW/'dataset/val.txt').read_text())
            C.write_text(folder/'data.yaml',f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
            datasets[target]=dict(data=C.bound(folder/'data.yaml'),train_list=C.bound(folder/'train.txt'),
                                  selected_real=0 if target=='SYN' else 249,sampled_real_unique=0 if target=='SYN' else len(set(replacement)))
        raw_names=[Path(x).name for x in (C.ROOT/datasets['RAW']['train_list']['path']).read_text().splitlines()]
        ref_names=[Path(x).name for x in (C.ROOT/datasets['REF']['train_list']['path']).read_text().splitlines()]
        assert raw_names==ref_names
        protocol=dict(arms=ARMS,args=T.ARGS,initialization=parent['initialization'],sources=sources,inputs=inputs,datasets=datasets,
            budget='Each R0 initialized independently;5epochs x1024slots,320optimizer updates; same seed42, final last.pt only',
            frozen='Backbone/neck/box/class branches and ALL buffers exact. Pose branches+flow trainable. BN affine in pose branch trainable, running stats fixed.',
            loss='Existing true-ignore pose loss, original hyperparameters except the two predeclared learning rates. No new label filters.',
            real='RAW and REF share the same217 sampled real images, order, boxes, common trusted keypoint masks and exposures. Only supervised coordinates differ.',
            controls='SYN replaces pseudo512 with same synthetic512. New source-only control under each learning rate.',
            source_negatives=0,GT_training=False,auto_promote=False,
            selection='Screen fixed final outputs against prespecified goal metrics. No epoch choice. Any promising setting needs repeat-seed and control verification.',
            prior_bn_probe='Student weights + R0 BN worsened; R0 weights + student BN improved several pose metrics. This stage freezes detector, uses original R0 BN to isolate pose-target learning, not the mixed-statistics probe.')
        C.freeze(C.DOC/'PROTOCOL.json',protocol)
    R.evaluation_protocol(PHASE,list(ARMS),sources+[C.bound(R.DOC/PHASE/'PROTOCOL.json')])
    print('POSE_PROTOCOL_LOCKED',datasets,flush=True)


def train(arm):
    with R.scope(PHASE):
        protocol=C.read(C.DOC/'PROTOCOL.json');spec=protocol['arms'][arm]
        for b in protocol['sources']+protocol['inputs']+[protocol['initialization']]:C.verify(b)
        fit=C.DOC/f'FIT_{arm}.json'
        if fit.exists():C.verify(C.read(fit)['checkpoint']);print('ALREADY_COMPLETE',arm,flush=True);return
        assert not (C.RAW/'runs'/arm).exists(),'Preserve incomplete run'
        available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))//1024
        assert available>=6000
        C.N.setup();torch.set_num_interop_threads(1);C.N.E.gpu();assert torch.cuda.is_available()
        args=dict(protocol['args'],lr0=spec['lr'],model=str(C.N.E.R0),data=str(C.ROOT/protocol['datasets'][spec['target']]['data']['path']),
                  project=str(C.RAW/'runs'),name=arm,exist_ok=False)
        trainer=PoseOnlyTrainer(overrides=args);steps=[];history=[];start=time.monotonic()
        base=torch.load(C.N.E.R0,map_location='cpu',weights_only=False)['model'].float().state_dict()
        def begin(t):
            state=t.model.state_dict();shared=[k for k,v in base.items() if k in state and v.shape==state[k].shape]
            assert all(torch.equal(base[k],state[k].cpu()) for k in shared)
            t.optimizer.register_step_post_hook(lambda opt,args,kwargs:steps.append(1))
            print('POSE_EXACT_R0_START',arm,'trainable_tensors',len(t.recovery_trainable),flush=True)
        def epoch(t):
            fixed=t.check_frozen();history.append(dict(epoch=t.epoch+1,steps=len(steps),fixed_tensors=fixed,gpu=C.N.E.gpu()))
            print('POSE_TRAIN',arm,t.epoch+1,'/5','steps',len(steps),flush=True)
        trainer.add_callback('on_train_start',begin);trainer.add_callback('on_train_epoch_end',epoch)
        trainer.train()
        checkpoint=C.RAW/'runs'/arm/'weights/last.pt';final=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
        protected=[k for k in base if not pose_parameter(k) or k.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
        assert all(torch.equal(base[k],final[k]) for k in protected),'Saved frozen state differs'
        changed=[k for k in base if not torch.equal(base[k],final[k])]
        assert changed and all(pose_parameter(k) for k in changed)
        rows=list(csv.DictReader((C.RAW/'runs'/arm/'results.csv').open()));assert len(rows)==5 and len(steps)==320
        C.freeze(fit,dict(complete=True,arm=arm,checkpoint=C.bound(checkpoint),protocol=C.bound(C.DOC/'PROTOCOL.json'),
            epochs=5,optimizer_steps=320,seconds=time.monotonic()-start,history=history,exact_R0_initialization=True,
            protected_state_exact=True,protected_tensors=len(protected),changed_tensors=changed,
            results_csv=C.bound(C.RAW/'runs'/arm/'results.csv')))
        print('POSE_FIT_COMPLETE',arm,round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare',*ARMS]);a=p.parse_args()
    prepare() if a.phase=='prepare' else train(a.phase)
