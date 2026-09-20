"""Mask conditioning hints only; never mask target supervision or move labels."""
import numpy as np
import torch
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation


def mask_one(item,rng,force=False):
    out=dict(item,points=item['points'].copy(),valid=item['valid'].copy())
    if not force and rng.random()<.5:return out
    choices=np.flatnonzero(np.asarray(item['target_valid'][:8],bool)&np.asarray(item['valid'][:8],bool))
    if len(choices):out['valid'][int(rng.choice(choices))]=False
    return out


def without_own_hint(valid,corner):
    if corner not in range(8):raise ValueError('Only the eight corner hints may be masked')
    out=valid.clone();out[:,corner]=False
    return out


@torch.no_grad()
def predict_crop(model,rgb,points,valid,blind=False):
    if not blind:return expectation(model(rgb,points,valid))
    result=points.clone()
    # Every corner is read from a pass where ONLY its own hint is absent.
    # No PnP, frame rejection, GT, uncertainty threshold or native-index change.
    for corner in range(8):
        q=expectation(model(rgb,points,without_own_hint(valid,corner)))
        result[:,corner]=q[:,corner]
    return result
