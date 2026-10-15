"""Postmodel held143 coarse-top-normal errors; never recalibrate the model."""
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from . import score_remainder_derived_reference_v4_14 as REF
from . import score_remainder_closed_sources_v4_14 as CLOSED
from . import calibrate_top_plane_normal_v4_16 as CAL
from . import top_plane_normal_factor_v4_16 as NORMAL
from .scoring import distribution

OUTPUT=CAL.OUTPUT.parent/'held_top_plane_normal_v4_16'


def diagnose(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every previous held-reference diagnosis')
    model_seal=C.read(CAL.OUTPUT/'MODEL_SEAL.json');model_path=CAL.OUTPUT/'MODEL.json'
    if C.sha(model_path)!=model_seal['model_sha256'] or not model_seal['before_any_new_normal_prior_pose_run_or_metric']:
        raise ValueError('CAL-only normal model must be frozen before held143 diagnosis')
    model=C.read(model_path);parameters=CAL.normal_factor_parameters(model);sigma=parameters['sigma_angle_rad']
    closure=CLOSED.verify_additional_source_closure()
    pose=REF.REMAINDER/'pose_evaluation_v4_14';seal=C.read(pose/'PREDICTION_SEAL.json')
    if (seal['rows']!=143 or not seal['before_remaining_GT_or_metric_read']
        or C.sha(pose/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']):
        raise ValueError('Complete143 pose seal required before any held numeric reference')
    score_seal=C.read(REF.OUTPUT/'SCORE_SEAL.json')
    if C.sha(REF.OUTPUT/'POSE_REFERENCE_SCORE.json')!=score_seal['score_sha256']:
        raise ValueError('Same already sealed nominal-reference score required')
    plan=C.read(REF.OUTPUT/'PLAN.json');annotations={r['id']:r for r in plan['annotations']}
    observation=REF.REMAINDER/'fresh_observations_v4_14';obs_seal=C.read(observation/'OBSERVATION_SEAL.json')
    if C.sha(observation/'OBSERVATIONS.jsonl.gz')!=obs_seal['observations_sha256'] or obs_seal['ordered_ids']!=plan['ids']:
        raise ValueError('Frozen actual143 initial geometry source required')
    output.mkdir(parents=True)
    source={str(Path(p).relative_to(C.ROOT)):C.sha(p)for p in (__file__,CAL.__file__,NORMAL.__file__,REF.__file__,CLOSED.__file__)}
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in (model_path,CAL.OUTPUT/'MODEL_SEAL.json',
        pose/'PREDICTION_SEAL.json',REF.OUTPUT/'PLAN.json',REF.OUTPUT/'SCORE_SEAL.json',
        observation/'OBSERVATION_SEAL.json',observation/'OBSERVATIONS.jsonl.gz')}
    C.write(output/'PROTOCOL.json',dict(schema='POSTMODEL_HELD143_TOP_NORMAL_DIAGNOSTIC_V4_16',
        source_sha256=source,input_sha256=inputs,source_closure=closure,
        CAL_model_fitted_and_sealed_before_this_diagnostic=True,
        sigma_or_floor_or_weight_not_refitted_from_143=True,new_pose_fits_or_image_queries=0,
        all143_kept=True,model_parameters=parameters,provided_reference_is_derived_unconfirmed=True))
    for relative,digest in source.items():
        p=output/'code_snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,p)
        if C.sha(p)!=digest:raise ValueError('Diagnostic source snapshot changed')
    records=[]
    for row in C.rows(observation/'OBSERVATIONS.jsonl.gz'):
        geometry=row['runtime']['initial_geometry']
        R0=NORMAL._rotation(geometry['R_cf']);Q=NORMAL._rotation(geometry['physical_registration_Q'])
        annotation=annotations[row['id']]
        if C.sha(C.ROOT/annotation['annotation'])!=annotation['annotation_sha256']:raise ValueError('Fixed reference changed')
        reference=REF.reference_from_document(C.read(C.ROOT/annotation['annotation']))
        result=dict(id=row['id'],session=row['session'],rotation_reference_available=reference['R'] is not None,
            sigma_deg=float(np.degrees(sigma)),reference_metadata=reference['metadata'])
        if reference['R'] is not None:
            n0=R0@NORMAL.TOP_NORMAL_CF;nt=NORMAL._rotation(reference['R'])@(Q.T@NORMAL.TOP_NORMAL_CF)
            angle=float(np.arctan2(np.linalg.norm(np.cross(n0,nt)),np.clip(n0@nt,-1.,1.)))
            result.update(angle_deg=float(np.degrees(angle)),standardized_angle=angle/sigma,
                coarse_top_normal_camera=n0.tolist(),reference_top_normal_camera=nt.tolist())
        records.append(result)
    if [r['id']for r in records]!=plan['ids']:raise ValueError('Every original143 required')
    eligible=[r for r in records if r['rotation_reference_available']]
    ratios=np.array([r['standardized_angle']for r in eligible]);summary=dict(rows=143,rotation_eligible=len(eligible),
        angle_deg=distribution([r['angle_deg']for r in eligible]),
        standardized_angle=distribution(ratios),descriptive_fraction_le2=float(np.mean(ratios<=2)),
        descriptive_fraction_le3=float(np.mean(ratios<=3)),
        by_session={s:distribution([r['angle_deg']for r in eligible if r['session']==s])for s in sorted({r['session']for r in records})})
    CLOSED.verify_additional_source_closure()
    if C.sha(model_path)!=model_seal['model_sha256']:raise ValueError('CAL model was changed during diagnosis')
    C.write(output/'DIAGNOSTIC.json',dict(summary=summary,records=records,
        OOD_diagnostic_not_model_selection_or_new_probability_certificate=True,
        CAL_sigma_floor_weight_unchanged=True,model_sha256=model_seal['model_sha256']))
    C.write(output/'SEAL.json',dict(diagnostic_sha256=C.sha(output/'DIAGNOSTIC.json'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),CAL_model_sha256=model_seal['model_sha256'],
        all143_post_complete_pose_and_score_seals=True,no_fit_or_parameter_change=True))
    print(summary,flush=True)


if __name__=='__main__':diagnose()
