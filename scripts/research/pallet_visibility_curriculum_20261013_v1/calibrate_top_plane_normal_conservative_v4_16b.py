"""Separate empirical CAL44 max/3 top-normal scale sensitivity.

Consume the immutable CAL angular-error journal. No new references, image
queries or pose fits. The CAL maximum is a finite observed envelope, never a
future coverage or Gaussian 3-sigma guarantee. All upstream models are fixed.
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import hashlib

import numpy as np

from . import common as C
from . import calibrate_top_plane_normal_v4_16 as BASE
from .scoring import distribution

OUTPUT=BASE.OUTPUT.parent/'CAL_top_plane_normal_conservative_v4_16b'
SCHEMA='CAL_TOP_PLANE_NORMAL_EMPIRICAL_ENVELOPE_V4_16B'
RECIPE={**deepcopy(BASE.RECIPE),
    'angular_scale':'max(existing2deg geometry-tolerance floor, untrimmed compatible CAL maximum oriented angular error / 3)',
    'CAL_maximum_divisor':3.,'fit_source':'sealed CAL44 angular-error records; no annotation reopening',
    'finite_CAL_maximum_is_future_probability_or_3sigma_coverage_guarantee':False}


def _digest(value):
    return hashlib.sha256(json.dumps(C.finite(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _sigma(records):
    if not records:raise ValueError('Compatible CAL training population required')
    return max(np.deg2rad(RECIPE['floor_angle_deg']),max(r['angle_rad']for r in records)/3.)


def normal_factor_parameters(model):
    if (model.get('schema')!=SCHEMA or model.get('fit_partition')!='CAL'
        or model.get('compatible_frame_count')!=44 or model.get('recipe')!=RECIPE):
        raise ValueError('Frozen empirical CAL44 normal envelope required')
    payload={k:v for k,v in model.items()if k!='model_digest_sha256'}
    if _digest(payload)!=model.get('model_digest_sha256'):raise ValueError('Conservative model digest changed')
    sigma=float(model['sigma_angle_rad'])
    expected=max(np.deg2rad(RECIPE['floor_angle_deg']),float(model['CAL_maximum_angular_error_rad'])/3)
    if not np.isfinite(sigma) or sigma!=expected:raise ValueError('Fixed untrimmed CAL max/3 recipe changed')
    return dict(sigma_angle_rad=sigma,weight=RECIPE['weight'],huber_scale=RECIPE['huber_scale'])


def fit(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every prior normal uncertainty model')
    source_seal=C.read(BASE.OUTPUT/'MODEL_SEAL.json');diagnostic_path=BASE.OUTPUT/'CAL_NORMAL_DIAGNOSTICS.json'
    if (C.sha(BASE.OUTPUT/'MODEL.json')!=source_seal['model_sha256']
        or C.sha(diagnostic_path)!=source_seal['diagnostics_sha256']
        or source_seal['fit_partition']!='CAL' or source_seal['fit_count']!=44):
        raise ValueError('Immutable original CAL44 angular-error journal required')
    diagnostic=C.read(diagnostic_path);records=diagnostic['compatible_records']
    if (len(records)!=44 or len({r['id']for r in records})!=44
        or any(r['partition']!='CAL' or not r['compatible']for r in records)):
        raise ValueError('All44 original compatible CAL records, no tail trimming')
    source={str(Path(p).relative_to(C.ROOT)):C.sha(p)for p in (__file__,BASE.__file__)}
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in (BASE.OUTPUT/'MODEL.json',BASE.OUTPUT/'MODEL_SEAL.json',diagnostic_path)}
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(schema='CAL_TOP_NORMAL_MAX_ENVELOPE_PROTOCOL_V4_16B',
        recipe=RECIPE,source_sha256=source,input_sha256=inputs,rows=44,
        new_reference_or_image_or_pose_reads=False,remaining_or_TEST_used_for_sigma=False,
        no_default_adoption_or_unique_pose_claim=True))
    for relative,digest in source.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Conservative source snapshot changed')
    sigma=_sigma(records);sessions=sorted({r['session']for r in records})
    model=dict(schema=SCHEMA,recipe=RECIPE,fit_partition='CAL',CAL_frame_count=46,
        compatible_frame_count=44,incompatible_frame_count=2,fit_ids=[r['id']for r in records],fit_sessions=sessions,
        sigma_angle_rad=sigma,sigma_angle_deg=float(np.degrees(sigma)),
        CAL_maximum_angular_error_rad=max(r['angle_rad']for r in records),
        model_fit_inputs=inputs,model_parent_sha256=source_seal['model_sha256'],
        finite_empirical_envelope_only=True,future_or_Gaussian_coverage_certificate=False)
    model['model_digest_sha256']=_digest(model)
    folds=[]
    for session in sessions:
        train=[r for r in records if r['session']!=session];held=[r for r in records if r['session']==session]
        local=_sigma(train);ratios=np.array([r['angle_rad']/local for r in held])
        folds.append(dict(heldout_session=session,train_ids=[r['id']for r in train],heldout_ids=[r['id']for r in held],
            sigma_angle_deg=float(np.degrees(local)),train_maximum_angular_error_deg=max(r['angle_deg']for r in train),
            heldout_angular_error_deg=distribution([r['angle_deg']for r in held]),
            descriptive_fraction_le2=float(np.mean(ratios<=2)),descriptive_fraction_le3=float(np.mean(ratios<=3)),
            heldout_maximum_standardized_angle=float(ratios.max()),future_probability_guarantee=False))
    C.write(output/'MODEL.json',model)
    C.write(output/'CAL_LOGO_DIAGNOSTIC.json',dict(folds=folds,all44_untrimmed=True,
        full_CAL_descriptive_fraction_le3=float(np.mean([r['angle_rad']<=3*sigma for r in records])),
        model_parameters=normal_factor_parameters(model),CAL_population_independent_truth_claimed=False))
    for relative,digest in {**source,**inputs}.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Calibration source/input changed')
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),
        diagnostic_sha256=C.sha(output/'CAL_LOGO_DIAGNOSTIC.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        source_sha256=source,input_sha256=inputs,fit_partition='CAL',fit_count=44,
        before_any_new_conservative_normal_prior_pose_fits_or_metric=True,original_sigma2_model_unchanged=True))
    print(dict(sigma_angle_deg=model['sigma_angle_deg'],CAL_maximum_deg=float(np.degrees(model['CAL_maximum_angular_error_rad'])),
        folds=[{k:f[k]for k in ('heldout_session','sigma_angle_deg','descriptive_fraction_le3','heldout_maximum_standardized_angle')}for f in folds]),flush=True)


if __name__=='__main__':fit()
