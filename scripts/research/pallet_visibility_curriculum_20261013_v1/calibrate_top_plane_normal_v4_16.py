"""CAL44-only top-normal total angular scale, no new pose/image fitting."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import shutil

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import select_frozen_wd_partial_refinement_v4_13e as OLD
from . import statewise_uncertainty_calibration_v4 as ALIGN
from . import top_plane_normal_factor_v4_16 as FACTOR
from .scoring import distribution

OUTPUT=OLD.OUTPUT.parent/'CAL_top_plane_normal_v4_16'
SCHEMA='CAL_TOP_PLANE_NORMAL_UNCERTAINTY_V4_16'
RECIPE=dict(fit_partition='CAL',fit_population='fixed-native compatible original CAL44; all CAL46 inventoried',
    normal_cf=[0.,-1.,0.],coarse_normal='initial R_cf @ normal_cf',
    target_normal='fixed reference R @ (archived Q.T @ normal_cf)',
    native_phase_choice='registered 3D geometry only, never pose-error minimum',
    angular_scale='max(existing2deg geometry-tolerance floor, median oriented angular error / sqrt(2*log(2)))',
    floor_angle_deg=2.,rayleigh_relation_is_scale_heuristic_not_coverage_certificate=True,
    prior_mean='coarse top-normal direction; no angular bias correction',weight=1.,huber_scale=3.,
    yaw_position_and_depth_direct_penalty=False,WD_quarter_turn_and_C2_invariant=True,
    outliers_trimmed=False,future_probability_or_95percent_coverage_claim=False,
    upstream_all_CAL_models_frozen_conditional_LOGO=True,TEST_or_remaining_references_fitted=False,
    runtime_GT_or_frameID_or_session_routing=False,S_raw_coordinates_or_image_reads=0,new_pose_fits=0,
    backbone_correlated_regularization_not_independent_image_information=True,
    ambiguity_or_uniqueness_not_resolved_by_prior=True)


def _sigma(records):
    if not records:raise ValueError('Nonempty compatible training population required')
    return max(np.deg2rad(RECIPE['floor_angle_deg']),
               float(np.median([r['angle_rad']for r in records]))/np.sqrt(2*np.log(2)))


def _digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(C.finite(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def normal_factor_parameters(model):
    if (model.get('schema')!=SCHEMA or model.get('fit_partition')!='CAL'
        or model.get('compatible_frame_count')!=44 or model.get('recipe')!=RECIPE):
        raise ValueError('Frozen compatible CAL44 normal model required')
    expected=model.get('model_digest_sha256');payload={k:v for k,v in model.items()if k!='model_digest_sha256'}
    if _digest(payload)!=expected:raise ValueError('CAL model digest changed')
    sigma=float(model['sigma_angle_rad'])
    if not np.isfinite(sigma) or sigma<np.deg2rad(RECIPE['floor_angle_deg']):raise ValueError('Positive declared scale floor required')
    return dict(sigma_angle_rad=sigma,weight=RECIPE['weight'],huber_scale=RECIPE['huber_scale'])


def fit(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every earlier calibration')
    protocol=C.read(OLD.OLD/'PROTOCOL.json');seal=C.read(OLD.OLD/'PREDICTION_SEAL.json');journal=OLD.OLD/'PREDICTIONS.jsonl.gz'
    if seal['rows']!=95 or C.sha(journal)!=seal['predictions_sha256'] or C.sha(OLD.OLD/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('Complete immutable original95 source required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    rows=[r for r in C.rows(journal)if r['partition']=='CAL']
    if len(rows)!=46:raise ValueError('All original CAL46 must be retained in calibration inventory')
    target=E.COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
    sources={str(Path(p).relative_to(C.ROOT)):C.sha(p)for p in (__file__,FACTOR.__file__,ALIGN.__file__)}
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in (journal,OLD.OLD/'PROTOCOL.json',OLD.OLD/'PREDICTION_SEAL.json',target)}
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(schema='CAL_TOP_PLANE_NORMAL_FIT_PROTOCOL_V4_16',recipe=RECIPE,
        rows=46,ids=[r['id']for r in rows],source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protocol['protected_user_sha256'],labels='CAL reference rotations only for calibration',
        no_TEST_or_remaining_label_paths=True,no_new_image_or_pose_fit=True))
    for relative,digest in sources.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Source snapshot changed')
    labels=C.read(target)['rows'];records=[];inventory=[]
    yaw90=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    for row in rows:
        family=next(f for f in row['families']if f['name']=='ORIGINAL_WD')
        R0=FACTOR._rotation(family['coarse_pose']['R']);Q=FACTOR._rotation(family['physical_registration_Q'])
        geometry=dict(native_xyz=family['native_xyz'],physical_registration_Q=family['physical_registration_Q'])
        alignment=ALIGN._reference_alignment({'runtime':{'initial_geometry':geometry}},labels[row['id']])
        case=dict(id=row['id'],partition='CAL',session=row['session'],compatible=alignment['compatible'],
            geometry_gap_m=alignment['geometry_gap_m'],fixed_3D_phase=alignment['phase'],
            exclusion_reason=None if alignment['compatible']else'fixed-native 3D registration incompatible')
        inventory.append(case)
        if not alignment['compatible']:continue
        reference=FACTOR._rotation(labels[row['id']]['target']['truth']['R'])
        n0=R0@FACTOR.TOP_NORMAL_CF;nt=reference@(Q.T@FACTOR.TOP_NORMAL_CF)
        if not np.allclose(nt,alignment['R_reference_native']@FACTOR.TOP_NORMAL_CF,rtol=0,atol=1e-9):
            raise ValueError('Fixed 3D C2 phase changed the oriented physical top normal')
        if not np.allclose((R0@yaw90.T)@FACTOR.TOP_NORMAL_CF,n0,rtol=0,atol=1e-12):
            raise ValueError('W/D quarter-turn unexpectedly changed the normal center')
        if not np.allclose(reference@(yaw90@yaw90)@(Q.T@FACTOR.TOP_NORMAL_CF),nt,rtol=0,atol=1e-12):
            raise ValueError('C2 physical symmetry unexpectedly changed the top normal')
        angle=float(np.arctan2(np.linalg.norm(np.cross(n0,nt)),np.clip(n0@nt,-1.,1.)))
        records.append({**case,'coarse_R_cf':R0.tolist(),'archived_Q':Q.tolist(),
            'coarse_top_normal_camera':n0.tolist(),'reference_top_normal_camera':nt.tolist(),
            'angle_rad':angle,'angle_deg':float(np.degrees(angle))})
    if len(records)!=44:raise ValueError('Expected original fixed-compatible CAL44, no positive-only or error trimming')
    sigma=_sigma(records);sessions=sorted({r['session']for r in records})
    model=dict(schema=SCHEMA,fit_partition='CAL',recipe=deepcopy(RECIPE),CAL_frame_count=46,
        compatible_frame_count=44,incompatible_frame_count=2,fit_ids=[r['id']for r in records],fit_sessions=sessions,
        sigma_angle_rad=sigma,sigma_angle_deg=float(np.degrees(sigma)),
        median_angular_error_deg=float(np.median([r['angle_deg']for r in records])),
        normalization='oriented sphere angle in radians; two isotropic tangent dimensions',
        model_fit_inputs=inputs,independent_physical_reference=False)
    model['model_digest_sha256']=_digest(model)
    folds=[]
    for session in sessions:
        train=[r for r in records if r['session']!=session];held=[r for r in records if r['session']==session]
        local=_sigma(train);ratios=np.array([r['angle_rad']/local for r in held])
        folds.append(dict(heldout_session=session,train_ids=[r['id']for r in train],heldout_ids=[r['id']for r in held],
            sigma_angle_deg=float(np.degrees(local)),heldout_error_deg=distribution([r['angle_deg']for r in held]),
            standardized_angle=distribution(ratios),descriptive_fraction_le2=float(np.mean(ratios<=2)),
            descriptive_fraction_le3=float(np.mean(ratios<=3)),future_probability_guarantee=False))
    C.write(output/'MODEL.json',model)
    C.write(output/'CAL_NORMAL_DIAGNOSTICS.json',dict(inventory=inventory,compatible_records=records,
        angular_error_deg=distribution([r['angle_deg']for r in records]),model_parameters=normal_factor_parameters(model),
        fixedCAL44_fit_noTEST_or_remaining_parameters=True,LOGO_folds=folds,
        yaw_or_translation_or_depth_residuals_not_fitted=True))
    for value in (sources,inputs,protocol['protected_user_sha256']):E._verify_sources(value)
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),
        diagnostics_sha256=C.sha(output/'CAL_NORMAL_DIAGNOSTICS.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        source_sha256=sources,input_sha256=inputs,fit_partition='CAL',fit_count=44,
        before_any_new_normal_prior_pose_run_or_metric=True,no_default_method_adoption=True))
    print(dict(sigma_deg=model['sigma_angle_deg'],angular_error_deg=distribution([r['angle_deg']for r in records]),
        CAL46=46,fit44=44,incompatible2=2,folds=folds),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();fit(args.output)
