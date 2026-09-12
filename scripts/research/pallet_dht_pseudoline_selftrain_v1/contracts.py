"""Observation-only boundary and paired exposure plan; not a student trainer."""
import json
from pathlib import Path

import numpy as np


def reject_real_gt(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.lower().startswith(('gt','ground_truth','target_points','target_pose','annotation','manual_label')):
                raise ValueError(f'real GT field forbidden: {key}')
            reject_real_gt(item)
    elif isinstance(value,list):
        for item in value:reject_real_gt(item)


def load_frozen_real_teacher_cache(path, allowed_path):
    if Path(path).resolve()!=Path(allowed_path).resolve():raise ValueError('cache path not allowlisted')
    cache=json.loads(Path(path).read_text());reject_real_gt(cache)
    return cache


def paired_exposure(seed,synth_count,real_count,steps=900):
    rng=np.random.default_rng(seed)
    synth=rng.integers(0,synth_count,(steps,24)).tolist()
    real=rng.integers(0,real_count,(steps,8)).tolist()
    augmentation_seeds=rng.integers(0,2**31-1,(steps,32)).tolist()
    return {arm:dict(synthetic=synth,real=real if arm!='C0' else None,augmentation=augmentation_seeds,
                    updates=steps,synthetic_exposures=steps*24,real_exposures=0 if arm=='C0' else steps*8)
            for arm in ('C0','C1','C2')}
