"""C2 matched real-only affine-off fits; no historical artifacts are rewritten."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics.data.augment import RandomPerspective
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer, pose_parameter
from . import common as C

IDENT = 'C2_REAL_AFFINE_OFF'
RAW = C.RAW / 'cycles' / IDENT
DOC = C.DOC / 'cycles' / IDENT
ARMS = ('PLASTIC_RAW', 'PLASTIC_REF', 'WOOD_RAW', 'WOOD_REF')


def save(path, value, freeze=False):
    assert Path(path).is_relative_to(RAW) or Path(path).is_relative_to(DOC)
    C.save(path, value, freeze)


def protocol(material):
    path = C.P.REC / 'pose_only/PROTOCOL.json' if material == 'PLASTIC' else C.M.DOC / 'WOOD_TRAIN_PROTOCOL.json'
    return path, C.read(path)


def digest(value):
    if torch.is_tensor(value): value = value.detach().cpu().contiguous().numpy()
    encoded = value.tobytes() if isinstance(value, np.ndarray) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def rng_digest():
    return digest(dict(python=repr(random.getstate()), numpy=repr(np.random.get_state()), torch=digest(torch.get_rng_state())))


class RealAffineOff:
    """Keep the exact transform and RNG calls, changing only REAL ranges."""
    def __init__(self, transform):
        assert isinstance(transform, RandomPerspective)
        self.transform = transform

    def __call__(self, labels):
        assert 'im_file' in labels, 'Cannot infer data role without original file identity'
        if Path(labels['im_file']).name.startswith('syn__'):
            return self.transform(labels)
        previous = self.transform.translate, self.transform.scale
        try:
            self.transform.translate = self.transform.scale = 0.
            return self.transform(labels)
        finally:
            self.transform.translate, self.transform.scale = previous


def wrap_affine(transform):
    """Visit installed Compose/pre-transform containers without replacing others."""
    if isinstance(transform, RealAffineOff): return transform, 0
    if isinstance(transform, RandomPerspective): return RealAffineOff(transform), 1
    count = 0
    if hasattr(transform, 'transforms'):
        items = []
        for child in transform.transforms:
            child, n = wrap_affine(child); items.append(child); count += n
        transform.transforms = items
    if hasattr(transform, 'pre_transform') and transform.pre_transform is not None:
        transform.pre_transform, n = wrap_affine(transform.pre_transform); count += n
    return transform, count


class AffineOffTrainer(PoseOnlyTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        result = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode == 'train':
            result.transforms, count = wrap_affine(result.transforms)
            assert count >= 1, 'No RandomPerspective found in installed train transform tree'
            self.affine_wrapped_count = count
        return result

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        names = [Path(p).name for p in batch['im_file']]
        keypoints = batch['keypoints']; mask = keypoints[..., 2]
        role = torch.tensor([n.startswith('syn__') for n in names], device=mask.device)
        instance_role = role[batch['batch_idx'].long()]
        record = dict(batch=len(self.affine_trace), epoch=int(self.epoch), names=names,
                      images=digest(batch['img']), boxes=digest(batch['bboxes']), support=digest(mask),
                      coordinates=digest(keypoints[..., :2]), batch_idx=digest(batch['batch_idx']),
                      roles={})
        for label, selected in [('SOURCE', instance_role), ('REAL', ~instance_role)]:
            values = mask[selected]
            record['roles'][label] = dict(images=sum(n.startswith('syn__') == (label == 'SOURCE') for n in names),
                                         instances=int(selected.sum()), supervised=int((values == 2).sum()),
                                         ignore=int((values == 1).sum()), invisible=int((values == 0).sum()))
        self.affine_trace.append(record)
        return batch


def make_dataset(material, target):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    _, p = protocol(material)
    args = get_cfg(overrides=dict(p['args'], lr0=1e-5))
    return YOLODataset(img_path=str(C.ROOT / p['datasets'][target]['train_list']['path']), imgsz=640,
                       batch_size=16, augment=True, hyp=args, rect=False, cache=False, stride=32, pad=0.,
                       task='pose', data=dict(names={0: 'pallet'}, nc=1, kpt_shape=[9, 3], flip_idx=[1,0,3,2,5,4,7,6,8]),
                       prefix='C2_PREFLIGHT ')


def sample_fingerprint(sample):
    return dict(image=digest(sample['img']), boxes=digest(sample['bboxes']),
                support=digest(sample['keypoints'][..., 2]), xy=digest(sample['keypoints'][..., :2]))


def preflight():
    destination = DOC / 'PREFLIGHT.json'
    if destination.exists():
        assert C.read(destination)['passed']; print('C2_PREFLIGHT_ALREADY_COMPLETE'); return
    start = time.monotonic(); torch.set_num_threads(4); cv2.setNumThreads(1)
    assert (DOC / 'SPEC.md').exists()
    results = {}
    from scripts.research.pallet_material_selftrain_closure_v1.train_pair import parity_signature, assert_pair
    for material in ('PLASTIC', 'WOOD'):
        pp, p = protocol(material)
        for b in p['inputs'] + p['sources'] + [p['initialization']]: C.verify(b)
        paths = {a: (C.ROOT / p['datasets'][a]['train_list']['path']).read_text().splitlines() for a in ('RAW','REF')}
        signatures = {a: parity_signature(v) for a, v in paths.items()}; assert_pair(signatures['RAW'], signatures['REF'])
        datasets = {a: make_dataset(material, a) for a in ('RAW','REF')}
        assert [Path(x).name for x in datasets['RAW'].im_files] == [Path(x).name for x in datasets['REF'].im_files]
        base = copy.deepcopy(datasets['RAW'].transforms)
        for dataset in datasets.values():
            dataset.transforms, count = wrap_affine(dataset.transforms); assert count >= 1
        rows = []; candidates = {}
        for role in ('SOURCE','REAL'):
            ii = [i for i, p in enumerate(datasets['RAW'].im_files) if Path(p).name.startswith('syn__') == (role == 'SOURCE')]
            candidates[role] = [ii[int((j+.5)*len(ii)/32)] for j in range(32)]
            for index in candidates[role]:
                sample_seed = 20260928 + index
                seed(sample_seed); old = base(copy.deepcopy(datasets['RAW'].get_image_and_label(index))); old_rng = rng_digest()
                seed(sample_seed); new = datasets['RAW'][index]; new_rng = rng_digest()
                seed(sample_seed); ref = datasets['REF'][index]
                a, b, c = map(sample_fingerprint, (old, new, ref))
                assert old_rng == new_rng, 'RNG draw schedule changed'
                if role == 'SOURCE': assert a == b, 'Synthetic augmentation changed'
                assert all(b[k] == c[k] for k in ('image','boxes','support')), 'Paired augmented input/mask/bbox mismatch'
                rows.append(dict(role=role, index=index, old=a, new=b, ref=c, rng_same=True))
        changed = sum(r['old']['image'] != r['new']['image'] for r in rows if r['role'] == 'REAL')
        assert changed == 32
        results[material] = dict(protocol=C.bind(pp), signatures=signatures, source_bit_exact=32,
                                 real_changed=changed, RNG_same=64, paired_RGB_mask_boxes_exact=64,
                                 sampled_real_unique=p['datasets']['RAW']['sampled_real_unique'])
        save(RAW / f'PREFLIGHT_{material}_PRIVATE.json', rows, True)
    transform_path=Path(inspect.getsourcefile(RandomPerspective))
    save(destination, dict(passed=True, spec=C.bind(DOC / 'SPEC.md'), materials=results,
                           seconds=time.monotonic()-start, fits=0, optimizer_updates=0, GPU_seconds=0,
                           source_transform=dict(path=str(transform_path),sha256=C.sha(transform_path),bytes=transform_path.stat().st_size),
                           caveat='32source+32real deterministic exposures/material; actual all-batch pairing checked after fits'), True)
    print('C2_PREFLIGHT_PASS', results, flush=True)


def gpu():
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    return thermal_guard()


def train(arm):
    assert arm in ARMS
    material, target = arm.split('_'); pp, p = protocol(material)
    fit_path = RAW / f'FIT_{arm}.json'
    if fit_path.exists():
        fit = C.read(fit_path); assert fit['complete'] and fit['optimizer_steps'] == 320
        C.verify(fit['checkpoint']); print('FIT_ALREADY_COMPLETE', arm); return
    assert C.read(DOC / 'PREFLIGHT.json')['passed']
    for b in p['inputs'] + p['sources'] + [p['initialization']]: C.verify(b)
    run_dir = RAW / 'runs' / arm
    assert not run_dir.exists(), 'Incomplete run preserved: no implicit restart'
    from scripts.research.pallet_type_selftrain_v1 import train as T
    T.C.N.setup(); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), 'Host GPU required'
    gpu()
    args = dict(p['args'], lr0=1e-5, model=str(C.ROOT/p['initialization']['path']),
                data=str(C.ROOT/p['datasets'][target]['data']['path']), project=str(RAW/'runs'), name=arm, exist_ok=False)
    trainer = AffineOffTrainer(overrides=args); trainer.affine_trace = []
    base = torch.load(C.ROOT/p['initialization']['path'], map_location='cpu', weights_only=False)['model'].float().state_dict()
    steps, epochs = [], []; begin_time = time.monotonic(); state_path = RAW / f'RUN_STATE_{arm}.json'
    save(state_path, dict(status='STARTED', arm=arm, utc=C.now(), spec=C.bind(DOC/'SPEC.md'), optimizer_steps=0))
    def step_done(opt, args, kwargs):
        steps.append(1); assert len(steps) <= 320
    def on_start(t):
        state = t.model.state_dict()
        assert set(base) == set(state) and all(torch.equal(base[k],state[k].detach().cpu()) for k in base)
        t.optimizer.register_step_post_hook(step_done)
        assert all(pose_parameter(n) for n in t.recovery_trainable)
        print('C2_EXACT_R0', arm, len(t.recovery_trainable), flush=True)
    def on_epoch(t):
        fixed = t.check_frozen(); temperature = gpu()
        epochs.append(dict(epoch=t.epoch+1, steps=len(steps), protected_tensors=fixed, temperature_C=temperature))
        save(state_path, dict(status='RUNNING', arm=arm, optimizer_steps=len(steps), epochs=epochs,
                             seconds=time.monotonic()-begin_time, utc=C.now()))
        save(RAW/f'TRACE_PROGRESS_{arm}.json', t.affine_trace)
        print('C2_EPOCH', arm, t.epoch+1, len(steps), flush=True)
    trainer.add_callback('on_train_start', on_start); trainer.add_callback('on_train_epoch_end', on_epoch)
    try:
        trainer.train()
        checkpoint = run_dir/'weights/last.pt'; final = torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
        protected = [k for k in base if not pose_parameter(k) or k.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
        assert all(torch.equal(base[k], final[k]) for k in protected)
        changed = [k for k in base if not torch.equal(base[k],final[k])]
        assert changed and all(pose_parameter(k) for k in changed)
        assert len(steps) == len(trainer.affine_trace) == 320
        csv_path = run_dir/'results.csv'; assert len(list(csv.DictReader(csv_path.open()))) == 5
        save(RAW/f'TRACE_{arm}.json', trainer.affine_trace, True)
        fit = dict(complete=True, arm=arm, target=target, material=material, epochs=5, optimizer_steps=320,
                   checkpoint=C.bind(checkpoint), initialization=p['initialization'], protocol=C.bind(pp),
                   spec=C.bind(DOC/'SPEC.md'), implementation=C.bind(Path(__file__)),
                   exact_R0_initialization=True, frozen_state_exact=True, protected_tensors=len(protected), changed_tensors=changed,
                   trace=C.bind(RAW/f'TRACE_{arm}.json'), results_csv=C.bind(csv_path), history=epochs,
                   seconds=time.monotonic()-begin_time, GPU_seconds=time.monotonic()-begin_time,
                   manual_supervision_added=0, GT_training=False, checkpoint_selection='last after exactly320updates',
                   data_rewritten=False, source_augmentation_unchanged=True)
        save(fit_path, fit, True); save(state_path, dict(status='COMPLETE', fit=C.bind(fit_path), optimizer_steps=320))
        print('C2_FIT_COMPLETE',arm,fit['seconds'],flush=True)
    except BaseException as error:
        save(state_path, dict(status='FAILED_PRESERVED', arm=arm, error=repr(error), optimizer_steps=len(steps),
                             seconds=time.monotonic()-begin_time, history=epochs, utc=C.now()))
        save(RAW/f'FAILED_TRACE_{arm}.json',trainer.affine_trace,True)
        raise


def train_all():
    for arm in ARMS:
        if (RAW/f'FIT_{arm}.json').exists(): continue
        log = RAW/f'TRAIN_{arm}.log'; assert not log.exists(), 'Preserve existing run log'
        log.parent.mkdir(parents=True,exist_ok=True)
        print('C2_START',arm,flush=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable,'-u','-m','scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine','train','--arm',arm],stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode == 0, (arm,result.returncode,str(log))
        fit=C.read(RAW/f'FIT_{arm}.json');print('C2_DONE',arm,fit['optimizer_steps'],fit['seconds'],flush=True)
    paired_trace()


def paired_trace():
    out={}
    for material in ('PLASTIC','WOOD'):
        a,b=[C.read(RAW/f'TRACE_{material}_{target}.json') for target in ('RAW','REF')]
        assert len(a)==len(b)==320
        protected_keys=('batch','epoch','names','images','boxes','support','batch_idx','roles')
        for x,y in zip(a,b): assert all(x[k]==y[k] for k in protected_keys), (material,x['batch'],'paired training trace mismatch')
        totals={role:{k:sum(x['roles'][role][k] for x in a) for k in ('images','instances','supervised','ignore','invisible')} for role in ('SOURCE','REAL')}
        assert totals['SOURCE']['images']==totals['REAL']['images']==2560
        out[material]=dict(batches=320,images_boxes_support_order_exact=True,supervision_exposure=totals,
                           differing_coordinate_batches=sum(x['coordinates']!=y['coordinates'] for x,y in zip(a,b)))
    save(DOC/'TRAINING_PARITY.json',out,True); print('C2_TRAINING_PARITY_PASS',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train','train-all','paired-trace']);p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.stage=='train':train(a.arm)
    elif a.stage=='train-all':train_all()
    elif a.stage=='paired-trace':paired_trace()
    else:preflight()


if __name__=='__main__':main()
