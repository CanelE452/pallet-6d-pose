"""Three conditional protected pose/flow fits, exclusively in the new namespace.

Old training entrypoints and mutable old module contexts are never invoked.
All three targets share one accepted population, slot order, image transform,
and post-affine trusted-support intersection.  No evaluation is run here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import gc
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.utils import RANK

from scripts.research.pallet_clean_to_pose_transfer_v1.augmentation import (
    digest, rng_state, restore_rng, state_seed,
)
from scripts.research.pallet_clean_to_pose_transfer_v1.trainer import (
    assert_protected_state, batch_trace, protected_state_keys,
)
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer

from . import common as C
from . import teacher as T

ARMS = ('RAW_TARGET', 'GLOBAL_TARGET', 'DEPTH_TARGET')
DATASET = C.PRIVATE / 'dataset'
OLD_DOC = C.ROOT / '_docs/experiments/pallet_clean_to_pose_transfer_v1'
OLD_RAW = C.ROOT / 'data/pallet/results/pallet_clean_to_pose_transfer_v1'
GENERATOR_SEED = 6148914691236517205


def label_path(image):
    return Path(str(image).replace('/images/', '/labels/')).with_suffix('.txt')


def local_link(source, destination):
    source, destination = Path(source), Path(destination)
    assert source.is_file() and destination.parent.resolve().is_relative_to(C.PRIVATE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        assert destination.is_symlink() and destination.resolve() == source.resolve()
    else:
        destination.symlink_to(source.resolve())


def prepare():
    gate = C.read(C.DOC / 'SUMMARY.json')
    assert gate['status'] == 'PILOT_GO' and gate['student_training_eligible']
    inputs = C.read(C.PRIVATE / 'INPUTS_PRIVATE.json')
    teachers = C.read(C.PRIVATE / 'TEACHER_PRIVATE.json')
    selected = sorted([r for r in teachers['TRAIN'] if r['result']['accepted']], key=lambda r:r['id'])
    source_inputs = {r['id']:r for r in inputs['TRAIN']}
    assert len(selected) == 119 and len({r['original_recording'] for r in selected}) >= 2
    m = float(np.median([np.log(r['result']['zD'] / r['R0']['centroid'][2]) for r in selected]))
    assert m == gate['GLOBAL_POST']['train_only_log_z_median']
    previous = C.read(OLD_DOC / 'PRIMARY_PROTOCOL.json')
    C.verify(previous['initialization'])
    private_old = C.read(OLD_RAW / 'PRIMARY_INPUT_BINDINGS_PRIVATE.json')
    sources = [Path(p) for p in private_old['source_slot_order']]
    assert len(sources) == len(set(sources)) == 512
    targets, bindings = {}, []
    for selected_row in selected:
        row = source_inputs[selected_row['id']]
        C.verify(row['old_label'])
        old_label = C.ROOT / row['old_label']['path']
        image = Path(str(old_label).replace('/labels/', '/images/')).with_suffix('.png')
        assert image.is_file()
        old_text = old_label.read_text()
        values = np.asarray(old_text.split(), dtype=np.float64)
        assert values.shape == (32,)
        points = values[5:].reshape(9, 3)
        assert np.array_equal(points[:, 2], row['oldvisibility'])
        q0, qd = np.asarray(selected_row['q0']), np.asarray(selected_row['qD'])
        pose = selected_row['R0']
        qg = T.transfer(q0, np.asarray(row['support']), np.asarray(row['K']),
            np.asarray(pose['R_cf']), np.asarray(pose['centroid']),
            np.asarray(pose['centroid']) * np.exp(m), np.asarray(pose['cf_extents']))
        support = np.asarray(row['support'], bool); support[8] = False
        arm_points = {}
        for arm, q in [('RAW_TARGET', q0), ('GLOBAL_TARGET', qg), ('DEPTH_TARGET', qd)]:
            target = values.copy()
            if arm != 'RAW_TARGET':
                xy = target[5:].reshape(9, 3)
                xy[support, :2] = (q[support] + 100.) / [row['hw'][1]+200, row['hw'][0]+200]
            text = old_text if arm == 'RAW_TARGET' else ' '.join(f'{v:.9f}' for v in target)+'\n'
            stored = np.asarray(text.split(), dtype=np.float32).reshape(1, 32)
            arm_points[arm] = stored[:, 5:].reshape(1, 9, 3)
            assert np.array_equal(stored[:, :5], np.asarray(old_text.split(), dtype=np.float32)[None, :5])
            assert np.array_equal(arm_points[arm][..., 2], arm_points['RAW_TARGET'][..., 2])
            assert np.array_equal(arm_points[arm][:, 8], arm_points['RAW_TARGET'][:, 8])
            assert np.array_equal(arm_points[arm][:, ~support], arm_points['RAW_TARGET'][:, ~support])
            folder = DATASET / arm
            local_link(image, folder / 'images' / image.name)
            C.save(folder / 'labels' / old_label.name, text)
            bindings += [C.bind(folder / 'labels' / old_label.name)]
        targets[image.name] = dict(points={arm:arm_points[arm] for arm in ARMS},
            recording=row['original_recording'], old_label=row['old_label'], image=C.bind(image),
            native_qD=qd.tolist(), native_qG=qg.tolist(), native_q0=q0.tolist())
    names = sorted(targets)
    real_order = np.random.default_rng(9021).choice(names, 512, replace=True).tolist()
    val_images = [Path(p) for p in (OLD_RAW / 'dataset/RAW/val.txt').read_text().splitlines()]
    assert len(val_images) == 32
    validation = []
    for index, source in enumerate(val_images):
        image = C.PRIVATE / 'validation/images' / (f'{index:04d}_'+source.name)
        local_link(source, image)
        C.save(label_path(image), label_path(source).read_text())
        validation.append(str(image))
    datasets = {}
    for arm in ARMS:
        folder = DATASET / arm
        for source in sources:
            local_link(source, folder / 'images' / source.name)
            C.save(folder / 'labels' / source.with_suffix('.txt').name, label_path(source).read_text())
        paths = [str(folder/'images'/p.name) for p in sources] + [str(folder/'images'/name) for name in real_order]
        C.save(folder/'train.txt', '\n'.join(paths)+'\n')
        C.save(folder/'val.txt', '\n'.join(validation)+'\n')
        C.save(folder/'data.yaml', f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\n'
            'nc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        datasets[arm] = {k:C.bind(folder/name) for k,name in [('data','data.yaml'),('train_list','train.txt'),('val_list','val.txt')]}
    C.save(C.PRIVATE/'TARGETS_PRIVATE.json', dict(targets=targets, real_slot_order=real_order,
        source_slot_order=[p.name for p in sources], global_log_scale=m))
    files = [Path(__file__), Path(T.__file__),
        C.ROOT/'scripts/research/pallet_type_selftrain_v1/recovery_pose_trainer.py',
        C.ROOT/'scripts/self_training_yolo/v3/true_ignore_trainer.py',
        C.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py',
        C.ROOT/'scripts/research/pallet_clean_to_pose_transfer_v1/trainer.py',
        C.ROOT/'scripts/research/pallet_clean_to_pose_transfer_v1/augmentation.py']
    protocol = dict(locked_before_fit=True, teacher_gate=C.bind(C.DOC/'SUMMARY.json'),
        parent_protocol=C.bind(C.DOC/'PROTOCOL.json'), initialization=previous['initialization'],
        args=dict(previous['args']), arms=list(ARMS), accepted_train=len(selected),
        accepted_recordings=dict(Counter(r['original_recording'] for r in selected)),
        sampled_real_unique=len(set(real_order)), real_slots_per_epoch=512, source_slots_per_epoch=512,
        epochs=5, updates_per_arm=320, seed=42, global_log_scale=m, target_map=C.bind(C.PRIVATE/'TARGETS_PRIVATE.json'),
        datasets=datasets, targets_labels=bindings, sources=[C.bind(p) for p in files],
        target_semantics='RAW exact original export bytes; GLOBAL/DEPTH change supported corners0..7 only; bbox/class/center8 and original true-ignore support unchanged.',
        augmentation='Existing CLEAR affine/HSV, exactly one RNG advance; each REAL target transformed with same RNG, common post-affine v2 mask intersects all3 and original trusted mask. No occlusion.',
        loss='Unmodified TrueIgnorePoseLoss26; v1 excluded from location/RLE/keypoint objectness; synthetic0/2 stock-equivalent.',
        output_weights_private_only=True, checkpoint_selection='last only',
        source_bookkeeping_validation=dict(frames=32, selection='Original fixed synthetic32, framework final epoch only; no real/eval reference or checkpoint selection'),
        no_old_entrypoint_or_mutable_module_context=True, no_evaluation_reference_coordinates_read=True)
    path=C.DOC/'STUDENT_TRAIN_PROTOCOL.json'
    if path.exists():
        assert C.read(path)==C.clean(protocol)
    else:
        C.save(path, protocol)
    return protocol, targets


class TripleTransform:
    def __init__(self, transform, targets, arm):
        self.transform, self.targets, self.arm = transform, targets, arm

    def __call__(self, labels):
        name=Path(labels['im_file']).name
        if name.startswith('syn__'):
            out=self.transform(labels); h=digest(out['img'])
            out['transfer_info']=dict(name=name,role='SOURCE',recording='SYNTHETIC',condition='CLEAR',
                target=self.arm,before_image=h,after_image=h,applied=False,scheduled=False,plan=None,
                actual_covered=0,reason='source_unchanged',geometric_demotions=0)
            return out
        entry=self.targets[name]; pp={k:np.asarray(v,dtype=np.float32) for k,v in entry['points'].items()}
        assert np.array_equal(labels['instances'].keypoints,pp[self.arm])
        before=rng_state(); outs={}; after=None
        for arm in ARMS:
            alternate=copy.deepcopy(labels);alternate['instances'].keypoints=pp[arm].copy()
            restore_rng(before);outs[arm]=self.transform(alternate)
            current=rng_state()
            if after is None: after=current
            else: assert state_seed(current,'after')==state_seed(after,'after')
        restore_rng(after); ref=outs[ARMS[0]]
        for value in outs.values():
            for key in ('img','bboxes','cls','batch_idx'):
                assert torch.equal(value[key],ref[key]), ('Different RGB/bbox/class/batch',key)
            assert value['keypoints'].shape==ref['keypoints'].shape
        out=outs[self.arm]; trusted=np.asarray(pp['RAW_TARGET'][...,2]==2)
        assert len(out['keypoints']) in (0,1)
        demotions=0; supervised=0; supervised9=0
        if len(out['keypoints']):
            original=torch.as_tensor(trusted,device=out['keypoints'].device)
            common=original.clone()
            for value in outs.values():common &= value['keypoints'][...,2]==2
            demotions=int((original&~common).sum())
            out['keypoints'][...,2]=torch.where(common,2.,1.)
            supervised=int(common[0,:8].sum());supervised9=int(common.sum())
        h=digest(out['img'])
        out['transfer_info']=dict(name=name,role='REAL',recording=entry['recording'],condition='CLEAR',
            target=self.arm,before_image=h,after_image=h,applied=False,scheduled=False,plan=None,
            actual_covered=0,reason='clear_three_target_common_support',geometric_demotions=demotions,
            original_supervised=int(trusted.sum()),original_ignored=int((pp['RAW_TARGET'][...,2]==1).sum()),
            actual_supervised=supervised,actual_supervised_all9=supervised9,actual_remaining=supervised,
            boxes=digest(out['bboxes']),support=digest(out['keypoints'][...,2]),coordinates=digest(out['keypoints'][...,:2]))
        return out


class DepthStudentTrainer(PoseOnlyTrainer):
    def __init__(self,*args,targets,arm,**kwargs):
        self.targets,self.arm,self.trace=targets,arm,[]
        super().__init__(*args,**kwargs)

    def build_dataset(self,img_path,mode='train',batch=None):
        if mode=='train':
            listing=Path(img_path).absolute()
            assert listing.is_relative_to(DATASET)
            assert all(Path(x).absolute().is_relative_to(DATASET) for x in listing.read_text().splitlines())
        data=super().build_dataset(img_path,mode,batch)
        if mode=='train':data.transforms=TripleTransform(data.transforms,self.targets,self.arm)
        return data

    def get_dataloader(self,dataset_path,batch_size=16,rank=-1,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        assert rank==RANK==-1 and not self.args.compile and not self.args.rect
        data=self.build_dataset(dataset_path,mode,batch_size)
        generator=torch.Generator().manual_seed(GENERATOR_SEED+RANK+int(self.args.seed)-42)
        return InfiniteDataLoader(dataset=data,batch_size=min(batch_size,len(data)),shuffle=True,
            num_workers=self.args.workers,sampler=None,prefetch_factor=4 if self.args.workers else None,
            pin_memory=self.device.type=='cuda',collate_fn=data.collate_fn,worker_init_fn=seed_worker,
            generator=generator,drop_last=False)

    def _setup_train(self):
        super()._setup_train();self.transfer_protected_keys=protected_state_keys(self.model)
        assert set(self.transfer_protected_keys)==set(self.recovery_fixed)

    def preprocess_batch(self,batch):
        info=batch.pop('transfer_info');batch=super().preprocess_batch(batch)
        self.trace.append(batch_trace(batch,info,int(self.epoch),len(self.trace)))
        return batch


def gpu_guard():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines()
    import os
    foreign=[r for r in rows if r.split(',')[0].strip()!=str(os.getpid()) and 'rustdesk' not in r]
    assert not foreign, ('Foreign GPU process; no automatic wait/kill',foreign)
    temperature=float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    assert temperature<80
    return temperature


def train_all():
    p,targets=prepare()
    for b in p['sources']+[p['initialization'],p['teacher_gate'],p['target_map']]:C.verify(b)
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    base=torch.load(C.ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    fits={}; total_seconds=0.; whole_start=time.monotonic()
    for arm in ARMS:
        fitpath=C.DOC/f'FIT_{arm}_S42.json'
        if fitpath.exists():
            fit=C.read(fitpath);assert fit['complete'];C.verify(fit['checkpoint']);fits[arm]=fit;total_seconds+=fit['seconds'];continue
        name=arm+'_S42';run=C.PRIVATE/'runs'/name
        assert not run.exists(), 'Incomplete prior fit preserved; no implicit replay'
        gpu_guard()
        args=dict(p['args'],model=str(C.ROOT/p['initialization']['path']),data=str(C.ROOT/p['datasets'][arm]['data']['path']),
            project=str(C.PRIVATE/'runs'),name=name,exist_ok=False)
        trainer=DepthStudentTrainer(overrides=args,targets=targets,arm=arm)
        steps=[];history=[];start=time.monotonic()
        C.save(C.PRIVATE/f'START_{arm}.json',dict(status='STARTED',at=C.now(),protocol=C.bind(C.DOC/'STUDENT_TRAIN_PROTOCOL.json')))
        def step_done(opt,args,kwargs):
            steps.append(1)
            assert len(steps)<=320 and total_seconds+time.monotonic()-start<1200, 'Fixed total GPU-fit wall budget'
        def on_start(t):
            actual=t.model.state_dict();assert set(actual)==set(base)
            assert all(torch.equal(base[k],actual[k].detach().cpu()) for k in base)
            assert len(t.recovery_trainable)==132 and len(t.transfer_protected_keys)==747
            t.optimizer.register_step_post_hook(step_done)
            print('EXACT_R0',arm,'trainable132 protected747',flush=True)
        def on_epoch(t):
            fixed=t.check_frozen();temp=gpu_guard()
            history.append(dict(epoch=t.epoch+1,optimizer_steps=len(steps),protected_tensors=fixed,temperature_C=temp))
            C.save(C.PRIVATE/f'RUN_STATE_{arm}.json',dict(status='RUNNING',optimizer_steps=len(steps),history=history),freeze=False)
            print('DEPTH_FIT_EPOCH',arm,t.epoch+1,len(steps),flush=True)
        trainer.add_callback('on_train_start',on_start);trainer.add_callback('on_train_epoch_end',on_epoch)
        try:
            trainer.train()
            checkpoint=run/'weights/last.pt';final=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
            assert_protected_state(final,base,trainer.transfer_protected_keys)
            changed=[k for k in base if not torch.equal(base[k],final[k])]
            assert changed and set(changed)<=set(trainer.recovery_trainable)
            assert len(steps)==len(trainer.trace)==320 and len(list(csv.DictReader((run/'results.csv').open())))==5
            assert sum(r['roles']['REAL']['images'] for r in trainer.trace)==2560
            assert sum(r['roles']['SOURCE']['images'] for r in trainer.trace)==2560
            trace=C.PRIVATE/f'TRACE_{arm}_S42.json';C.save(trace,trainer.trace)
            seconds=time.monotonic()-start;total_seconds+=seconds
            fit=dict(status='DONE',complete=True,arm=arm,seed=42,epochs=5,optimizer_steps=320,steps=320,
                checkpoint=C.bind(checkpoint),protocol=C.bind(C.DOC/'STUDENT_TRAIN_PROTOCOL.json'),
                initialization=p['initialization'],exact_R0_initialization=True,protected_state_exact=True,
                trainable_tensors=132,protected_tensors=747,changed_tensors=changed,trace=C.bind(trace),
                results_csv=C.bind(run/'results.csv'),history=history,seconds=seconds,GPU_training_wall_seconds=seconds,
                real_exposures=2560,source_exposures=2560,framework_source32_bookkeeping=True,
                checkpoint_selection='last only',manual_added=0,evaluation_reference_read=False)
            C.save(fitpath,fit);fits[arm]=fit
            C.save(C.PRIVATE/f'RUN_STATE_{arm}.json',dict(status='COMPLETED',fit=C.bind(fitpath)),freeze=False)
            print('DEPTH_FIT_DONE',arm,seconds,flush=True)
        except BaseException as exc:
            C.save(C.PRIVATE/f'FAILURE_{arm}.json',dict(status='FAILED',error=repr(exc),optimizer_updates=len(steps),
                seconds=time.monotonic()-start,at=C.now()))
            raise
        del trainer,final;gc.collect();torch.cuda.empty_cache()
    traces={arm:C.read(C.ROOT/fits[arm]['trace']['path']) for arm in ARMS}
    checks=0
    for index in range(320):
        ref=traces[ARMS[0]][index]
        for arm in ARMS:
            row=traces[arm][index]
            for key in ('names','images','before_images','after_images','boxes','support','batch_idx','roles'):
                assert row[key]==ref[key],(index,arm,key);checks+=1
            for left,right in zip(ref['transfer'],row['transfer']):
                for key in ('recording','role','before_image','after_image','geometric_demotions'):
                    assert left[key]==right[key];checks+=1
    C.save(C.DOC/'TRAIN_RECEIPTS.json',dict(status='DONE',complete=True,arms=fits,fits=3,optimizer_updates=960,
        GPU_training_seconds=total_seconds,three_fit_wall_seconds=time.monotonic()-whole_start,
        real_exposures=7680,source_exposures=7680,exposures=15360,three_arm_trace_checks=checks,
        same_RGB_sample_order_affine_bbox_support=True,global_log_scale=p['global_log_scale'],
        framework_source_validation='Same original32 source examples at final epoch for framework bookkeeping only; no student evaluator or real reference access',
        student_formal_evaluation_examples=0,actual_F_calls=0,evaluation_reference_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train'])
    args=parser.parse_args();prepare() if args.stage=='prepare' else train_all()
