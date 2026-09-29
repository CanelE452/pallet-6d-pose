"""Paired RAW/REF geometry, then optional input-only random occlusion.

Rectangle size/aspect/noise and cover-one/leave-two rules follow old Clean19
S1. Unlike its historical S1/S2 paired planner, this random-only transfer uses
the full current canvas, accepts the first valid of32 proposals, and does not
condition application on a structured S2 placement. No target is manufactured.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import pickle
import random

import numpy as np
import torch

from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import cover, overlap, fill


def digest(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def state_seed(state, name):
    h = hashlib.sha256(pickle.dumps((state[0], state[1], name), protocol=4))
    h.update(state[2].numpy().tobytes())
    return int.from_bytes(h.digest()[:8], 'little')


def load_paired_labels(raw_train_list, ref_train_list):
    """Read existing label aliases only; never resolve away the syn__ role name."""
    paths = {key: [Path(p) for p in Path(path).read_text().splitlines() if p]
             for key, path in [('RAW', raw_train_list), ('REF', ref_train_list)]}
    assert [p.name for p in paths['RAW']] == [p.name for p in paths['REF']]
    paired = {}
    for raw, ref in zip(paths['RAW'], paths['REF']):
        if raw.name.startswith('syn__'):
            continue
        rows = {}
        for arm, path in [('RAW', raw), ('REF', ref)]:
            label = path.parent.parent/'labels'/path.with_suffix('.txt').name
            values = np.asarray([line.split() for line in label.read_text().splitlines() if line.strip()], dtype=np.float32)
            assert values.shape == (1, 32) and np.isfinite(values).all()
            rows[arm] = values
        assert np.array_equal(rows['RAW'][:, :5], rows['REF'][:, :5])
        points = {key: value[:, 5:].reshape(1, 9, 3).copy() for key, value in rows.items()}
        assert np.array_equal(points['RAW'][..., 2], points['REF'][..., 2])
        assert set(np.unique(points['RAW'][..., 2])) <= {1., 2.}
        if raw.name in paired:
            assert all(np.array_equal(paired[raw.name][key], points[key]) for key in points)
        paired[raw.name] = points
    assert paired
    return paired


def random_plan(points, mask, box, shape, seed):
    """TRAIN REF-conditioned plan shared by both targets and CLEAR/OCC."""
    rng = np.random.default_rng(seed)
    points = np.asarray(points)
    mask = np.asarray(mask, dtype=bool).copy()
    assert points.shape == (9, 2) and mask.shape == (9,)
    mask[8] = False  # Center is not used to make placement feasible.
    area = float(np.prod(np.maximum(0., np.asarray(box)[2:]-np.asarray(box)[:2])))
    aspect = float(rng.choice([.5, 1., 2.]))
    fraction = float(rng.choice([.1, .2, .3]))
    w = max(1, round(np.sqrt(area*fraction*aspect)))
    h = max(1, round(np.sqrt(area*fraction/aspect)))
    plan = dict(seed=int(seed), scheduled=bool(rng.random() < .5),
                area_fraction=fraction, aspect=aspect, size=[w, h],
                fill_seed=int(rng.integers(0, 2**31-1)), applied=False,
                reason='not_scheduled', rectangle=None, covered=[],
                supervised=int(mask.sum()), remaining=int(mask.sum()),
                candidates_checked=0, bbox_fraction=0.)
    if not plan['scheduled']:
        return plan
    if mask.sum() < 3:
        plan['reason'] = 'cannot_cover1_leave2'
        return plan
    if w > shape[1] or h > shape[0]:
        plan['reason'] = 'shape_exceeds_input'
        return plan
    for _ in range(32):
        l = int(rng.integers(0, shape[1]-w+1))
        t = int(rng.integers(0, shape[0]-h+1))
        rectangle = [l, t, w, h]
        covered = cover(points, rectangle) & mask
        plan['candidates_checked'] += 1
        if covered.sum() >= 1 and mask.sum()-covered.sum() >= 2 and overlap(rectangle, box) > 0:
            plan.update(applied=True, reason='random_valid', rectangle=rectangle,
                        covered=np.flatnonzero(covered).tolist(), remaining=int(mask.sum()-covered.sum()),
                        bbox_fraction=overlap(rectangle, box)/max(area, 1e-12))
            return plan
    plan['reason'] = 'no_valid_random_position_32'
    return plan


class PairedTransform:
    """Two same-RNG base transforms, one effective RNG advance, common support.

    A point is supervised only if BOTH transformed targets remain supervised.
    All others use true-ignore1, including originally ignored points that stock
    clipping would turn into0. Only REAL support is intersected; source is stock.
    CLEAR computes the identical plan but never applies it.
    """
    def __init__(self, transform, paired_labels, target, condition, recordings=None):
        assert target in ('RAW', 'REF') and condition in ('CLEAR', 'OCC')
        self.transform, self.paired_labels = transform, paired_labels
        self.target, self.condition, self.recordings = target, condition, recordings

    def __call__(self, labels):
        name = Path(labels['im_file']).name
        if name.startswith('syn__'):
            out = self.transform(labels)
            h = digest(out['img'])
            out['transfer_info'] = dict(name=name, role='SOURCE', recording='SYNTHETIC',
                condition=self.condition, target=self.target, before_image=h, after_image=h,
                applied=False, scheduled=False, plan=None, actual_covered=0,
                reason='source_unchanged', geometric_demotions=0)
            return out

        assert name in self.paired_labels and labels['instances'].normalized
        paired = self.paired_labels[name]
        original = np.asarray(labels['instances'].keypoints)
        assert original.shape == (1, 9, 3)
        assert np.array_equal(original, paired[self.target]), 'Dataset label differs from locked paired map'
        assert np.array_equal(paired['RAW'][..., 2], paired['REF'][..., 2])
        original_support = original[..., 2].copy()
        assert set(np.unique(original_support)) <= {1., 2.}
        alternate = copy.deepcopy(labels)
        other = 'REF' if self.target == 'RAW' else 'RAW'
        alternate['instances'].keypoints = paired[other].copy()
        before_rng = rng_state()
        out = self.transform(labels)
        after_rng = rng_state()
        try:
            restore_rng(before_rng)
            counterpart = self.transform(alternate)
            assert state_seed(rng_state(), 'after') == state_seed(after_rng, 'after'), 'Target changed base RNG consumption'
        finally:
            restore_rng(after_rng)
        for key in ('img', 'bboxes', 'cls', 'batch_idx'):
            assert torch.equal(out[key], counterpart[key]), ('Target changed base geometry/RGB', key)
        assert out['keypoints'].shape == counterpart['keypoints'].shape
        assert len(out['keypoints']) in (0, 1)
        before_image = digest(out['img'])
        recording = self.recordings[name] if self.recordings is not None else 'UNSPECIFIED'
        info = dict(name=name, role='REAL', recording=recording, target=self.target,
                    condition=self.condition, before_image=before_image,
                    applied=False, scheduled=False, plan=None, reason='no_transformed_instance',
                    original_supervised=int((original_support == 2).sum()),
                    original_ignored=int((original_support == 1).sum()),
                    geometric_demotions=0, actual_supervised=0, actual_supervised_all9=0,
                    actual_covered=0, actual_remaining=0, planned_target_covered=0)
        if len(out['keypoints']):
            both = (out['keypoints'][..., 2] == 2) & (counterpart['keypoints'][..., 2] == 2)
            original_trusted = torch.as_tensor(original_support == 2, device=both.device)
            both &= original_trusted
            common = torch.where(both, 2., 1.).to(out['keypoints'].dtype)
            out['keypoints'][..., 2] = common
            counterpart['keypoints'][..., 2] = common
            info['geometric_demotions'] = int((original_trusted & ~both).sum())
            reference = out if self.target == 'REF' else counterpart
            hh, ww = out['img'].shape[-2:]
            center, size = reference['bboxes'][0, :2].numpy(), reference['bboxes'][0, 2:].numpy()
            box = np.r_[center-size/2, center+size/2]*[ww, hh, ww, hh]
            ref_points = reference['keypoints'][0, :, :2].numpy()*[ww, hh]
            plan = random_plan(ref_points, both[0].numpy(), box, (hh, ww), state_seed(before_rng, name))
            actual_mask = both[0, :8].numpy()
            actual_points = out['keypoints'][0, :8, :2].numpy()*[ww, hh]
            covered = int((cover(actual_points, plan['rectangle']) & actual_mask).sum()) if plan['applied'] else 0
            applied = self.condition == 'OCC' and plan['applied']
            before_targets = {key: digest(out[key]) for key in ('keypoints', 'bboxes', 'cls', 'batch_idx')}
            if applied:
                l, t, w, h = plan['rectangle']
                out['img'][:, t:t+h, l:l+w] = torch.from_numpy(fill(plan))
            assert before_targets == {key: digest(out[key]) for key in before_targets}
            info.update(plan=plan, scheduled=plan['scheduled'], applied=bool(applied),
                        reason=plan['reason'] if self.condition == 'OCC' else 'clear_control',
                        actual_supervised=int(actual_mask.sum()), actual_supervised_all9=int(both.sum()),
                        planned_target_covered=covered, actual_covered=covered if applied else 0,
                        actual_remaining=int(actual_mask.sum())-(covered if applied else 0))
        info['after_image'] = digest(out['img'])
        info['boxes'] = digest(out['bboxes'])
        info['support'] = digest(out['keypoints'][..., 2])
        info['coordinates'] = digest(out['keypoints'][..., :2])
        out['transfer_info'] = info
        return out
