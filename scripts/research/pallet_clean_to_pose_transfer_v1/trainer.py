"""Namespace-local paired trainer and seed-aware loader; no training entrypoint.

Factory construction, not reseeding an already running InfiniteDataLoader,
fixes the installed constant-generator seed bug. Single-process/single-GPU only.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import os

import torch
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.utils import RANK
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer

from .augmentation import PairedTransform, digest
from . import common as C

ORIGINAL_GENERATOR_SEED = 6148914691236517205


def loader_seed(training_seed):
    return ORIGINAL_GENERATOR_SEED + RANK + int(training_seed)-42


def assert_new_train_list(train_list):
    """YOLO label-cache writes must follow NEW aliases, never old directories."""
    listing = Path(train_list).absolute()
    assert listing.is_relative_to(C.RAW), ('Train listing outside new namespace', listing)
    paths = [Path(os.path.abspath(line)) for line in listing.read_text().splitlines() if line]
    assert paths and all(path.is_relative_to(C.RAW) for path in paths), 'Old image aliases would redirect label-cache writes'


def make_loader(dataset, batch=16, workers=2, training_seed=42, rank=-1,
                pin_memory=False, shuffle=True):
    """Used unchanged by actual training and CPU preflight (including workers2)."""
    assert rank == -1 and RANK == -1, 'Only locked single-process training is supported'
    batch = min(batch, len(dataset))
    workers = min(os.cpu_count() or 1, workers)
    generator = torch.Generator().manual_seed(loader_seed(training_seed))
    return InfiniteDataLoader(dataset=dataset, batch_size=batch, shuffle=shuffle,
        num_workers=workers, sampler=None, prefetch_factor=4 if workers else None,
        pin_memory=pin_memory, collate_fn=getattr(dataset, 'collate_fn', None),
        worker_init_fn=seed_worker, generator=generator, drop_last=False)


def make_dataset(train_list, args, paired_labels, target, condition, recordings=None):
    """CPU-callable actual YOLODataset path; the caller owns new-namespace caches."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    assert_new_train_list(train_list)
    dataset = YOLODataset(img_path=str(train_list), imgsz=640, batch_size=16,
        augment=True, hyp=get_cfg(overrides=dict(args)), rect=False, cache=False,
        stride=32, pad=0., task='pose', data=dict(names={0: 'pallet'}, nc=1,
        kpt_shape=[9, 3], flip_idx=[1, 0, 3, 2, 5, 4, 7, 6, 8]), prefix='CLEAN_TRANSFER ')
    dataset.transforms = PairedTransform(dataset.transforms, paired_labels, target, condition, recordings)
    return dataset


def protected_state_keys(model):
    """All non-trainable state, including flow loc/cov/mask and every BN buffer."""
    allowed = {name for name, value in model.named_parameters() if value.requires_grad}
    assert allowed
    return sorted(set(model.state_dict())-allowed)


def assert_protected_state(state, initial, protected):
    assert set(state) == set(initial)
    assert all(torch.equal(state[name].detach().cpu(), initial[name].detach().cpu()) for name in protected)
    return len(protected)


def batch_trace(batch, info, epoch=0, index=0):
    """JSON-safe trace; RGB and coordinates remain tensors only in memory."""
    names = [Path(path).name for path in batch['im_file']]
    assert len(info) == len(names) and [row['name'] for row in info] == names
    mask = batch['keypoints'][..., 2]
    source_images = torch.tensor([name.startswith('syn__') for name in names], device=mask.device)
    source_instances = source_images[batch['batch_idx'].long()]
    roles = {}
    for role, select in [('SOURCE', source_instances), ('REAL', ~source_instances)]:
        values = mask[select]
        roles[role] = dict(images=sum(name.startswith('syn__') == (role == 'SOURCE') for name in names),
            instances=int(select.sum()), supervised=int((values == 2).sum()),
            ignore=int((values == 1).sum()), invisible=int((values == 0).sum()))
    recordings = {}
    for row in info:
        if row['role'] != 'REAL':
            continue
        group = recordings.setdefault(row['recording'], Counter())
        group.update(images=1, scheduled=int(row['scheduled']), applied=int(row['applied']),
                     supervised=row['actual_supervised'], covered=row['actual_covered'],
                     remaining=row['actual_remaining'], geometric_demotions=row['geometric_demotions'])
    return dict(batch=index, epoch=epoch, names=names, images=digest(batch['img']),
        before_images=[row['before_image'] for row in info], after_images=[row['after_image'] for row in info],
        boxes=digest(batch['bboxes']), support=digest(mask), coordinates=digest(batch['keypoints'][..., :2]),
        batch_idx=digest(batch['batch_idx']), roles=roles,
        recording_exposure={key: dict(value) for key, value in recordings.items()}, transfer=list(info))


class CleanPoseTrainer(PoseOnlyTrainer):
    def __init__(self, *args, paired_labels, target, condition, recordings=None, **kwargs):
        assert target in ('RAW', 'REF') and condition in ('CLEAR', 'OCC')
        self.paired_labels, self.target, self.condition = paired_labels, target, condition
        self.recordings, self.trace = recordings, []
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode='train', batch=None):
        if mode == 'train':
            assert_new_train_list(img_path)
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode == 'train':
            dataset.transforms = PairedTransform(dataset.transforms, self.paired_labels,
                                                 self.target, self.condition, self.recordings)
        return dataset

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        assert not self.args.compile and not self.args.rect
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return make_loader(dataset, batch_size, self.args.workers, self.args.seed,
                           rank=rank, pin_memory=self.device.type == 'cuda', shuffle=True)

    def _setup_train(self):
        super()._setup_train()
        assert set(protected_state_keys(self.model)) == set(self.recovery_fixed)
        self.transfer_protected_keys = protected_state_keys(self.model)

    def preprocess_batch(self, batch):
        info = batch.pop('transfer_info')
        batch = super().preprocess_batch(batch)
        self.trace.append(batch_trace(batch, info, epoch=int(self.epoch), index=len(self.trace)))
        return batch
