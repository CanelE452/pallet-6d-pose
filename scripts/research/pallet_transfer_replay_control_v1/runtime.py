"""Deterministic domain-isolated batch8 streams and the fixed replay trainer."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / 'data/pallet/results/pallet_transfer_replay_control_v1'
DOC = ROOT / '_docs/experiments/pallet_transfer_replay_control_v1'
R0 = ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
TARGET = ROOT / 'data/pallet/results/pallet_active_learning_v1/retrospective_v1/dataset/random'
SOURCE = ROOT / 'data/pallet/results/pallet_paper_contribution_screen_v1/C_geometry_preserving_da/dataset/synthetic'
from contracts import ARM_WEIGHTS, backward_coefficients

HYP = dict(task='pose', imgsz=640, batch=32, epochs=10, optimizer='SGD', lr0=.002, lrf=.01,
    momentum=.937, weight_decay=.0005, warmup_epochs=1., box=7.5, cls=.5, dfl=1.5,
    pose=12., kobj=1., hsv_h=.015, hsv_s=.5, hsv_v=.35, degrees=0., translate=.1,
    scale=.25, shear=0., perspective=0., flipud=0., fliplr=0., mosaic=.15,
    close_mosaic=3, mixup=0., copy_paste=0., erasing=.4, deterministic=True)
DATA = dict(names={0:'item'}, kpt_shape=[9,3], flip_idx=[1,0,3,2,5,4,7,6,8], channels=3)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    h = hashlib.sha256()
    values = value if isinstance(value, dict) else {'tensor': value}
    for key, tensor in sorted(values.items()):
        h.update(key.encode()); h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    torch.save(value, tmp); tmp.replace(path)


def load_model(device='cuda'):
    ck = torch.load(R0, map_location='cpu')
    model = copy.deepcopy(ck.get('ema') or ck['model']).float().to(device)
    model.requires_grad_(True); model.args = get_cfg(overrides=HYP)
    # This YOLO26n has reg_max=1 and an Identity DFL module (zero parameters).
    assert model.model[-1].reg_max == 1 and sum(p.numel() for p in model.model[-1].dfl.parameters()) == 0
    return model


def optimizer(model):
    bias, norm, weight = [], [], []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad: continue
            (bias if name == 'bias' else norm if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
             else weight).append(parameter)
    return torch.optim.SGD([dict(params=bias, weight_decay=0., is_bias=True),
        dict(params=weight, weight_decay=.0005, is_bias=False),
        dict(params=norm, weight_decay=0., is_bias=False)], lr=.002, momentum=.937, nesterov=True)


def freeze_bn(model):
    model.train()
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval(); count += 1
    assert count == 126


def bn_buffers(model):
    return {f'{name}.{key}': value.detach().cpu().clone() for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)}


def device_batch(batch, device='cuda'):
    return {key: ((value.to(device, non_blocking=False).float() / 255) if key == 'img'
                  else value.to(device, non_blocking=False)) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def batch_digest(batch):
    return tensor_sha({key: batch[key] for key in ('img','batch_idx','cls','bboxes','keypoints')})


class TracedDataset(YOLODataset):
    def __init__(self, domain, epoch):
        self.domain = domain; self.references = []
        hyp = get_cfg(overrides=HYP)
        if epoch >= 7: hyp.mosaic = 0.
        root = SOURCE if domain == 'source' else TARGET
        super().__init__(img_path=str(root/'images'), imgsz=640, batch_size=8, augment=True,
            hyp=hyp, rect=False, cache=False, stride=32, pad=0., data=DATA, task='pose', prefix=domain+': ')
        assert len(self) == (1440 if domain == 'source' else 30)
        self.allowlist = {str(Path(p).resolve()) for p in self.im_files}

    def get_image_and_label(self, index):
        self.references.append(str(Path(self.im_files[index]).resolve()))
        return super().get_image_and_label(index)

    def seeded_item(self, index, seed):
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        try:
            random.seed(seed); np.random.seed(seed % (2**32));
            generator = torch.Generator().manual_seed(seed); torch.random.set_rng_state(generator.get_state())
            start = len(self.references); item = super().__getitem__(index)
            refs = self.references[start:]
            assert refs and set(refs) <= self.allowlist
            return item, refs
        finally:
            random.setstate(py_state); np.random.set_state(np_state); torch.random.set_rng_state(torch_state)


def order_key(seed, epoch, stream, cycle, index):
    text = f'replay-control-v1-order\n{seed}\n{epoch}\n{stream}\n{cycle}\n{index}'
    return hashlib.sha256(text.encode()).digest()


class BatchStream:
    def __init__(self, domain, stream, seed, epoch):
        self.domain, self.stream, self.seed, self.epoch = domain, stream, seed, epoch
        self.dataset = TracedDataset(domain, epoch); self.cursor = 0; self.indices = []

    def _fill(self):
        cycle = len(self.indices) // len(self.dataset)
        self.indices.extend(sorted(range(len(self.dataset)), key=lambda i: order_key(self.seed,self.epoch,self.stream,cycle,i)))

    def batch(self, step_in_epoch):
        while len(self.indices) < self.cursor + 8: self._fill()
        indices = self.indices[self.cursor:self.cursor+8]; self.cursor += 8
        items, refs = [], []
        for slot, index in enumerate(indices):
            text = f'replay-control-v1-augment\n{self.seed}\n{self.epoch}\n{step_in_epoch}\n{self.stream}\n{slot}'
            value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')
            item, used = self.dataset.seeded_item(index, value); items.append(item); refs.append(used)
        batch = YOLODataset.collate_fn(items)
        return batch, dict(stream=self.stream, domain=self.domain, indices=indices,
            tensor_sha256=batch_digest(batch), original_references=refs,
            total_original_references=sum(map(len, refs)), unique_original_references=len(set().union(*map(set,refs))))


def epoch_streams(arm, seed, epoch):
    result = {}
    for stream in ARM_WEIGHTS[arm]:
        domain = 'source' if stream.startswith('source_') else 'target'
        result[stream] = BatchStream(domain, stream, seed, epoch)
    return result


def learning_rate(opt, step, epoch):
    ratio = (1 + math.cos(math.pi * epoch / 10)) / 2 * .99 + .01
    for group in opt.param_groups:
        group['lr'] = float(np.interp(step,[0,30],[.1 if group['is_bias'] else 0.,.002*ratio])) if step <= 30 else .002*ratio
        group['momentum'] = float(np.interp(step,[0,30],[.8,.937])) if step <= 30 else .937
    return [dict(lr=float(g['lr']), momentum=float(g['momentum']), weight_decay=float(g['weight_decay'])) for g in opt.param_groups]


def state_sha(model):
    return tensor_sha(model.state_dict())


def train(arm, seed):
    if arm not in ARM_WEIGHTS or seed not in (1,2,3): raise ValueError((arm,seed))
    gate = json.loads((DOC/'ACTUAL_MODEL_GATE.json').read_text()); assert gate['status'] == 'PASS'
    torch.set_num_threads(4); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True, warn_only=True)
    out = RAW/'runs'/f'{arm}_seed{seed}'; out.mkdir(parents=True, exist_ok=True)
    final = out/'last.pt'; state_path = out/'resume_state.pt'
    if final.exists():
        audit=json.loads((out/'TRAINING_AUDIT.json').read_text()); assert audit['optimizer_updates']==300 and sha(final)==audit['checkpoint_sha256']; return
    model=load_model(); freeze_bn(model); initial=state_sha(model); frozen_bn=bn_buffers(model)
    criterion=model.init_criterion(); opt=optimizer(model)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in model.parameters() if p.requires_grad}
    trace=[]; next_step=0; consumed_before_resume=0
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu'); assert saved['arm']==arm and saved['seed']==seed
        assert saved['initial_state_sha256']==initial
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        criterion.updates=saved['criterion']['updates']; criterion.o2m=saved['criterion']['o2m']; criterion.o2o=saved['criterion']['o2o']
        trace=saved['trace']; next_step=saved['next_step']; consumed_before_resume=next_step
        random.setstate(saved['random']); np.random.set_state(saved['numpy']); torch.random.set_rng_state(saved['torch_cpu']); torch.cuda.set_rng_state(saved['torch_cuda'])
    start=time.time(); coefficients=backward_coefficients(arm)
    for epoch in range(next_step//30,10):
        streams=epoch_streams(arm,seed,epoch)
        # Recreate each epoch's mutable Mosaic buffer state up to the resume point.
        skip=max(0,next_step-epoch*30) if epoch==next_step//30 else 0
        for local_step in range(skip):
            reconstructed={name:streams[name].batch(local_step)[1] for name in streams}
            old=trace[epoch*30+local_step]['streams']
            assert all(reconstructed[n]['tensor_sha256']==old[n]['tensor_sha256'] for n in streams)
        for local_step in range(skip,30):
            step=epoch*30+local_step; batches={}; stream_records={}
            for name, stream in streams.items(): batches[name],stream_records[name]=stream.batch(local_step)
            freeze_bn(model); opt.zero_grad(set_to_none=True); total=0.; components={}
            for name, coefficient in coefficients.items():
                batch=device_batch(batches[name]); c8=criterion(model(batch['img']),batch)[0].sum()
                assert torch.isfinite(c8); (coefficient*c8).backward()
                components[name]=float(c8.detach()); total+=coefficient*components[name]
            preclip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
            schedule=learning_rate(opt,step,epoch); opt.step()
            assert all(torch.equal(value,model.state_dict()[name].detach().cpu()) for name,value in frozen_bn.items())
            record=dict(step=step,epoch=epoch,local_step=local_step,streams=stream_records,
                C8=components,backward_objective=total,preclip_gradient_norm=preclip,
                clipped=preclip>10.,optimizer=schedule,criterion=dict(updates=criterion.updates,o2m=criterion.o2m,o2o=criterion.o2o))
            if len(trace)==step: trace.append(record)
            else: assert trace[step]==record
            resume=dict(schema='replay_control_resume_v1',arm=arm,seed=seed,next_step=step+1,
                initial_state_sha256=initial,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer=opt.state_dict(),criterion=dict(updates=criterion.updates,o2m=criterion.o2m,o2o=criterion.o2o),
                random=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.random.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(),trace=trace)
            atomic_torch(state_path,resume)
            if (step+1)%30==0: print(json.dumps(dict(arm=arm,seed=seed,updates=step+1,
                objective=round(total,6),grad=round(preclip,4),elapsed_s=round(time.time()-start),
                allocated_mb=round(torch.cuda.memory_allocated()/2**20))),flush=True)
        criterion.update()
        # Persist criterion schedule state after each virtual epoch.
        saved=torch.load(state_path,map_location='cpu'); saved['criterion']=dict(updates=criterion.updates,o2m=criterion.o2m,o2o=criterion.o2o)
        atomic_torch(state_path,saved)
    assert len(trace)==300 and criterion.updates==10
    saved_model=copy.deepcopy(model).cpu().eval(); saved_model.args=vars(saved_model.args)
    if hasattr(saved_model,'criterion'): delattr(saved_model,'criterion')
    pending=out/'last_unverified.pt'; assert not pending.exists()
    torch.save(dict(model=saved_model,ema=None,train_args=HYP,epoch=9,optimizer=None),pending)
    pending.replace(final)
    atomic_json(out/'EXPOSURE.json',trace)
    refs={stream:sum(r['streams'].get(stream,{}).get('total_original_references',0) for r in trace) for stream in set().union(*(set(r['streams']) for r in trace))}
    atomic_json(out/'TRAINING_AUDIT.json',dict(status='PASS',arm=arm,seed=seed,optimizer_updates=300,
        valid_comparison_updates=300,consumed_before_resume=consumed_before_resume,
        planned_slot_exposure=dict(real=2400 if arm!='T32_COMPUTE' else 9600,
            synthetic=7200 if arm=='REPLAY' else 0),actual_stream_slots={s:2400 for s in ARM_WEIGHTS[arm]},
        actual_mosaic_original_references=refs,initial_state_sha256=initial,final_state_sha256=state_sha(model),
        checkpoint_sha256=sha(final),resume_state_sha256=sha(state_path),criterion_updates=criterion.updates,
        BN_buffers_equal=True,BN_affine_trainable=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        clipped_updates=sum(r['clipped'] for r in trace),elapsed_this_invocation_s=time.time()-start))
    print(f'{arm}_seed{seed} COMPLETE',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--arm',choices=list(ARM_WEIGHTS),required=True);parser.add_argument('--seed',type=int,choices=[1,2,3],required=True)
    args=parser.parse_args(); train(args.arm,args.seed)
