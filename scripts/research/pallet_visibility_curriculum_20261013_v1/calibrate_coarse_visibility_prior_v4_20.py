"""Sealed CAL-only frame-event floor for the bounded coarse visibility prior.

Prepare whitelists only original model/pose/initial-S geometry. Fit opens the
isolated CAL labels after preparation checks; no TEST or remainder labels,
images, raw/observed 2D keypoints, model forward or pose optimizer are used.
CAL labels were historically exposed in diagnosis; this is not a holdout.
"""
from copy import deepcopy
from pathlib import Path
from collections import Counter
import argparse
import shutil
import sys
import numpy as np

from . import evaluate_o_localization_abstention_v4_19 as V
from . import bounded_coarse_visibility_prior_v4_20 as PRIOR
from . import statewise_uncertainty_calibration_v4 as ALIGN

C,E=V.C,V.E
OUTPUT=V.OUTPUT.parent/'CAL_coarse_visibility_prior_v4_20'
LABELS=V.BASE.ARCHIVE.E.COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
RECIPE=dict(schema='BOUNDED_COARSE_VISIBILITY_CAL_RECIPE_V4_20',population_CAL=46,
    fixed_native_compatible_fit_population=44,native_incompatible2_kept_in_diagnostics=True,
    initial_S='fixed original coarse model only; never physical visibility truth',
    eligible_initial_S='initial_self_hidden and maximum incident cosine < -sin(existing2deg)',
    reference_phase='existing ALIGN._reference_alignment geometry-only native registration; no minimum pose-error phase',
    frame_wrong_event='nonempty eligible initial S and any fixed-native reference maximum incident cosine > +sin2deg',
    frame_consistent_event='nonempty eligible initial S and all fixed-native reference maximum incident cosines < -sin2deg',
    frame_grazing_event='neither strong wrong nor strong consistent; recorded separately, not informative fit count',
    no_eligible_S='cost0 at runtime; no event fit count',
    event_unit='one frame, not independent S corners or incident faces',
    fixed_pseudocounts=dict(wrong=1,consistent=1),epsilon='(wrong_frames+1)/(wrong_frames+consistent_frames+2)',
    geometry_tolerance_deg=2.,smoothing_cosine_scale='sin(existing2deg)',
    h_k='1/(1+sum_j exp(a_kj/s))',H='product_k h_k; empty product1',
    loss='-2 log(epsilon+(1-epsilon)H)',factor_weight=1.,one_frame_cap='-2 log(epsilon)',
    LOGO='same four original CAL sessions; every fold epsilon uses only training sessions',
    frozen_upstream_all_CAL_models_conditional_subsystem_LOGO=True,
    calibrated_probability_or_future_coverage=False,independent_new_image_information=False,
    S_image_raw_2D_or_region_queries=0,initial_S_relabelled=False,new_hard_constraints=0,
    all_other_solver_factors_and_constraints_unchanged=True,
    TEST_or_REMAINDER_fit_or_ID_parameter_routing=False,default_adoption=False,
    historical_CAL_reference_exposure_acknowledged=True)


def _verify(output):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if (seal['protocol_sha256']!=C.sha(output/'PROTOCOL.json')
            or seal['CAL_inputs_sha256']!=C.sha(output/'CAL_INPUTS.jsonl.gz')
            or protocol['recipe']!=RECIPE):
        raise ValueError('Immutable prefit recipe and geometry inputs required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(protocol[key])
    rows=list(C.rows(output/'CAL_INPUTS.jsonl.gz'))
    if len(rows)!=46 or [r['id']for r in rows]!=protocol['ids']or any(r['partition']!='CAL'for r in rows):
        raise ValueError('Exactly all ordered CAL46 geometry rows required')
    return protocol,rows


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every earlier preparation')
    parent=C.read(V.OUTPUT/'PROTOCOL.json');seal=C.read(V.OUTPUT/'INPUT_SEAL.json')
    if seal['rows']!=238 or seal['input_sha256']!=C.sha(V.OUTPUT/'INPUTS.jsonl.gz'):
        raise ValueError('Frozen all238 input binding required')
    rows=[]
    for task in C.rows(V.OUTPUT/'INPUTS.jsonl.gz'):
        if task['partition']!='CAL':continue
        family=task['families']['ORIGINAL_WD'];pose=family['coarse_pose']
        _,audit=PRIOR.initial_confident_S_geometry(family['native_xyz'],pose['R'],pose['t'],
                                                  np.asarray(task['initial_self_hidden'],dtype=bool))
        rows.append(dict(id=task['id'],partition='CAL',session=task['session'],
            native_xyz=family['native_xyz'],physical_registration_Q=family['physical_registration_Q'],
            original_coarse_pose=dict(R=deepcopy(pose['R']),t=deepcopy(pose['t'])),
            initial_self_hidden=task['initial_self_hidden'],
            initial_S_geometry=audit))
    if len(rows)!=46 or len({r['session']for r in rows})!=4:
        raise ValueError('The unchanged CAL46/four-session population is required')
    sources=dict(parent['source_sha256'])
    for module in (sys.modules[__name__],PRIOR,ALIGN):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in
            (V.OUTPUT/'PROTOCOL.json',V.OUTPUT/'INPUT_SEAL.json',V.OUTPUT/'INPUTS.jsonl.gz',
             V.OUTPUT/'PREDICTION_SEAL.json',V.OUTPUT/'SCORE_SEAL.json',LABELS)}
    for mapping in (sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(schema='BOUNDED_COARSE_VISIBILITY_PREFIT_PROTOCOL_V4_20',
        recipe=RECIPE,ids=[r['id']for r in rows],source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],labels_path=str(LABELS.relative_to(C.ROOT)),
        reference_bytes_hashed_without_numeric_decode_in_prepare=True,
        historical_CAL_diagnostic_label_exposure=True,new_fit_label_decode_only_after_this_seal=True,
        raw_2D_or_image_values_in_CAL_inputs=False,no_TEST_or_REMAINDER_records=True,
        pending_root_and_peer_essential_review_before_fit=True,new_pose_fits=0))
    C.save_rows(output/'CAL_INPUTS.jsonl.gz',rows)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'PREPARATION_SEAL.json',dict(rows=46,protocol_sha256=C.sha(output/'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output/'CAL_INPUTS.jsonl.gz'),before_new_fit_reference_decode=True,
        historical_CAL_diagnostic_exposure=True,model_not_fitted=True))
    print(dict(prepared=46,path=str(output),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json')),flush=True)


def frame_event(row,label):
    if row['partition']!='CAL':raise ValueError('Non-CAL reference rejected before any target access')
    g=dict(native_xyz=row['native_xyz'],physical_registration_Q=row['physical_registration_Q'])
    alignment=ALIGN._reference_alignment({'runtime':{'initial_geometry':g}},label)
    eligible=[r for r in row['initial_S_geometry']if r['eligible']]
    # Recompute membership without any reference-dependent eligibility choice.
    _,original=PRIOR.initial_confident_S_geometry(row['native_xyz'],row['original_coarse_pose']['R'],
        row['original_coarse_pose']['t'],np.asarray(row['initial_self_hidden'],dtype=bool))
    if original!=row['initial_S_geometry']:raise ValueError('Initial-only eligibility changed')
    xyz=np.asarray(row['native_xyz']);camera=-alignment['R_reference_native'].T@alignment['t_reference']
    limit=float(np.sin(np.deg2rad(2.)));reference=[]
    for r in eligible:
        k=r['native_corner'];ray=camera-xyz[k]
        maximum=float(np.max(np.asarray(r['incident_normals'])@ray/np.linalg.norm(ray)))
        reference.append(dict(native_corner=k,maximum_incident_cosine=maximum,
            maximum_incident_angle_deg=float(np.degrees(np.arcsin(np.clip(maximum,-1.,1.))))))
    if not alignment['compatible']:event='EXCLUDED_NATIVE_INCOMPATIBLE'
    elif not eligible:event='NO_ELIGIBLE_INITIAL_S'
    elif any(r['maximum_incident_cosine']>limit for r in reference):event='WRONG'
    elif all(r['maximum_incident_cosine']<-limit for r in reference):event='CONSISTENT'
    else:event='REFERENCE_GRAZING'
    return dict(id=row['id'],partition='CAL',session=row['session'],event=event,
        reference_native_compatible=bool(alignment['compatible']),fixed_geometry_phase=int(alignment['phase']),
        fixed_geometry_gap_m=float(alignment['geometry_gap_m']),eligible_initial_S_ids=[r['native_corner']for r in eligible],
        initial_S_geometry=original,reference_geometry=reference,
        reference_R_native=alignment['R_reference_native'].tolist(),reference_t=alignment['t_reference'].tolist())


def model_from_events(events,heldout_session=None):
    if any(r['partition']!='CAL'for r in events):raise ValueError('Only CAL events may fit a floor')
    training=[r for r in events if r['session']!=heldout_session]
    counts=Counter(r['event']for r in training)
    informative=[r for r in training if r['event']in('WRONG','CONSISTENT')]
    wrong=counts['WRONG'];consistent=counts['CONSISTENT'];epsilon=(wrong+1.)/(wrong+consistent+2.)
    model=dict(schema=PRIOR.MODEL_SCHEMA,fit_partition='CAL',epsilon=epsilon,
        geometry_tolerance_deg=2.,fixed_pseudocounts=dict(PRIOR.PSEUDOCOUNTS),recipe=deepcopy(RECIPE),
        heldout_session=heldout_session,training_CAL_ids=[r['id']for r in training],
        compatible_training_ids=[r['id']for r in training if r['reference_native_compatible']],
        event_fit_ids=[r['id']for r in informative],event_counts=dict(counts),
        fit_sessions=sorted({r['session']for r in training}),event_unit='frame',
        calibrated_physical_visibility_probability=False,factor_weight=1.)
    model['model_digest_sha256']=PRIOR.model_digest(model);PRIOR.validate_model(model)
    return model


def reference_penalty(row,event,model):
    p0=row['original_coarse_pose']
    prepared=PRIOR.prepare_bounded_coarse_visibility_prior(row['native_xyz'],p0['R'],p0['t'],
        np.asarray(row['initial_self_hidden'],dtype=bool),model)
    p=PRIOR.G._pose_vector(np.asarray(event['reference_R_native']),np.asarray(event['reference_t']),1.)
    value,gradient,audit=PRIOR.coarse_visibility_prior_value_and_gradient(p,row['native_xyz'],1.,prepared)
    if not np.isfinite(value)or not np.isfinite(gradient).all():raise ValueError('Nominal reference penalty must remain finite')
    return dict(id=event['id'],event=event['event'],fixed_native_compatible=event['reference_native_compatible'],
        reference_penalty=value,maximum_frame_penalty=audit['maximum_frame_energy'],hard_rejected=False)


def fit(output=OUTPUT):
    output=Path(output).resolve();protocol,rows=_verify(output)
    if (output/'MODEL.json').exists():raise FileExistsError('Preserve every earlier fitted model')
    labels=C.read(LABELS)['rows']
    if len(labels)!=46 or set(labels)!={r['id']for r in rows}:raise ValueError('Isolated CAL46 only')
    events=[frame_event(r,labels[r['id']])for r in rows]
    if sum(r['reference_native_compatible']for r in events)!=44:raise ValueError('Keep compatible44 and incompatible2 explicit')
    full=model_from_events(events);folds=[]
    byid={r['id']:r for r in rows}
    for session in sorted({r['session']for r in events}):
        model=model_from_events(events,session);held=[r for r in events if r['session']==session]
        if set(model['training_CAL_ids'])&{r['id']for r in held}:raise ValueError('LOGO session leakage')
        C.write(output/'folds'/session/'MODEL.json',model)
        folds.append(dict(heldout_session=session,train_ids=model['training_CAL_ids'],heldout_ids=[r['id']for r in held],
            epsilon=model['epsilon'],training_event_counts=model['event_counts'],heldout_event_counts=dict(Counter(r['event']for r in held)),
            heldout_reference_penalties=[reference_penalty(byid[r['id']],r,model)for r in held]))
    penalties=[reference_penalty(byid[event['id']],event,full)for event in events]
    C.write(output/'MODEL.json',full)
    C.write(output/'CAL_EVENT_DIAGNOSTICS.json',dict(rows=events,reference_penalties=penalties,all46_kept=True,
        event_counts=dict(Counter(r['event']for r in events)),pose_fits=0,S_image_or_2D_queries=0))
    C.write(output/'CAL_LOGO_REPORT.json',dict(rows46=46,compatible44=44,folds=folds,
        frame_event_evaluation_not_independent_probability_calibration=True,
        frozen_upstream_all_CAL_conditioning=True,no_TEST_or_REMAINDER_fit=True))
    _verify(output)
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),
        CAL_events_sha256=C.sha(output/'CAL_EVENT_DIAGNOSTICS.json'),LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        fold_model_sha256={f['heldout_session']:C.sha(output/'folds'/f['heldout_session']/'MODEL.json')for f in folds},
        source_sha256=protocol['source_sha256'],input_sha256=protocol['input_sha256'],
        fit_partition='CAL',pose_solver_calls=0,not_adopted_and_no_full_population_run=True))
    print(dict(full_epsilon=full['epsilon'],events=full['event_counts'],folds=folds,path=str(output)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','fit'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    (prepare if args.command=='prepare'else fit)(args.output)
