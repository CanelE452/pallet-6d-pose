"""CAL-only empirical floor for a bounded coarse O-region regularizer.

Replay the existing max-error radius recipe leaving one frame out, count a
frame event when any eligible O target misses, and use Beta(1,1) smoothing.
Five frames and two sessions describe a tiny conditional plastic-only sample;
epsilon is a regularization floor, never a pose/visibility probability.
The original runtime ellipses and their radius model remain unchanged.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import ot_domains_v2 as SOURCE
from . import bounded_o_localization_prior_v4_21 as PRIOR

SOURCE_MODEL=E.STUDY/'ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json'
PARENT=E.STUDY/'nonsevere_v4_fix/boundary_recovery_v1/refined_grazing_image_chain_mixtures_v4_18c'
OUTPUT=PARENT.parent/'CAL_bounded_o_localization_prior_v4_21'
SCHEMA='CAL_BOUNDED_COARSE_O_LOCALIZATION_FLOOR_V21'
RECIPE=dict(fit_partition='CAL',source='unchanged genuine HUMAN external-O error archive',
    recipe='leave one whole frame out; replay original normalized max radius and original pixel floor',
    event='any held-out eligible O error exceeds its fold radius',
    estimator='(mismatch_frames+1)/(informative_frames+2)',pseudocount_alpha=1.,pseudocount_beta=1.,
    same_frame_points_not_independent=True,LOGO='two available O-support sessions',
    runtime_ellipse_model_unchanged=True,new_runtime_radius_fit=False,
    crossvalidation_radius_replays=True,source_exclusions_unchanged=True,
    runtime_loss='one frame capped outside-ellipse pseudo-Huber regularizer',huber_scale=3.,
    probability_coverage_or_native_identity_guarantee=False,
    wood_or_false_O_classifier_calibration=False,new_pose_fits_or_RGB_or_NN=0)


def _source():
    model=C.read(SOURCE_MODEL)
    SOURCE._validate_model(model,'N3_SUBPIX')
    if (model['quantile']!=1. or model['self_hidden_samples_used']or model['TEST_inputs_used']
            or model['candidate_pose_inputs_used']or model['sample_count']!=7
            or len(model['training_record_ids'])!=5 or model['recording_count']!=2):
        raise ValueError('Exact existing genuine-O CAL archive required')
    return model


def _samples(model,cal_ids):
    result=[];seen=set()
    for item in model['samples']:
        r=deepcopy(item);ident=r['record_id'];key=(ident,r['native_corner'])
        if (ident not in cal_ids or key in seen or r['session']!=ident.split(':')[0]
                or not isinstance(r['native_corner'],int)or isinstance(r['native_corner'],bool)
                or not 0<=r['native_corner']<8):
            raise ValueError('Unique declared CAL frame/corner/session required')
        seen.add(key)
        values=np.array([r['error_px'],r['scale_px'],r['normalized_error']],float)
        if (not np.isfinite(values).all()or values[0]<0 or values[1]<16
                or not np.isclose(values[0]/values[1],values[2],atol=1e-12,rtol=1e-12)):
            raise ValueError('Bound nonnegative CAL error and existing nonself scale required')
        result.append(r)
    if len(result)!=7 or {r['record_id']for r in result}!=set(model['training_record_ids']):
        raise ValueError('All seven original samples and five frames must remain')
    return result


def _events(train,test,floor):
    if not train or not test:raise ValueError('Nonempty train/heldout source samples required')
    ratio=max(r['normalized_error']for r in train)
    frames=[]
    for ident in sorted({r['record_id']for r in test}):
        values=[]
        for r in test:
            if r['record_id']!=ident:continue
            radius=max(floor,r['scale_px']*ratio)
            values.append(dict(native_corner=r['native_corner'],error_px=r['error_px'],
                fold_radius_px=radius,outside=r['error_px']>radius))
        frames.append(dict(id=ident,event=any(v['outside']for v in values),points=values))
    return frames


def fit_event_floor(samples,minimum_radius_px,heldout_session=None):
    selected=[r for r in samples if r['session']!=heldout_session]
    ids=sorted({r['record_id']for r in selected})
    if len(ids)<2:raise ValueError('At least two O frames needed for frame-heldout replay')
    folds=[]
    for ident in ids:
        train=[r for r in selected if r['record_id']!=ident]
        held=[r for r in selected if r['record_id']==ident]
        event=_events(train,held,minimum_radius_px)[0]
        folds.append(dict(**event,training_ids=sorted({r['record_id']for r in train})))
    mismatch=sum(r['event']for r in folds);epsilon=(mismatch+1.)/(len(folds)+2.)
    return dict(heldout_session=heldout_session,fit_ids=ids,fit_sample_count=len(selected),
        informative_frames=len(folds),mismatch_frames=mismatch,epsilon=epsilon,
        maximum_frame_energy=float(-2*np.log(epsilon)),frame_heldout_replays=folds,
        confidence_or_prediction_probability_claim=False)


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior CAL preparations')
    parent=C.read(PARENT/'PROTOCOL.json');seal=C.read(PARENT/'INPUT_SEAL.json')
    if C.sha(PARENT/'PROTOCOL.json')!=seal['protocol_sha256']or seal['rows']!=238:
        raise ValueError('Exact parent CAL46/TEST49/R143 declaration required')
    cal_ids=parent['ids'][:46]
    if parent['population']!=dict(CAL=46,TEST=49,REMAINDER=143,severe=0):raise ValueError('Fixed cohort required')
    sources={str(Path(p).resolve().relative_to(C.ROOT)):C.sha(Path(p))
        for p in (__file__,PRIOR.__file__,SOURCE.__file__,C.__file__,E.__file__)}
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in (SOURCE_MODEL,PARENT/'PROTOCOL.json',PARENT/'INPUT_SEAL.json')}
    for mapping in (sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol=dict(schema='BOUNDED_O_FLOOR_PREFIT_PROTOCOL_V21',recipe=RECIPE,
        prepared_before_new_floor_model_fit_UTC=datetime.now(timezone.utc).isoformat(),
        prior_CAL_error_archive_already_exploratorily_inspected=True,cal_ids=cal_ids,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=parent['protected_user_sha256'],
        source_model_path=str(SOURCE_MODEL.relative_to(C.ROOT)),no_primary_reference_decode=True)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot changed')
    C.write(output/'PREPARATION_SEAL.json',dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),
        before_new_epsilon_fit=True,source_sha256=sources,input_sha256=inputs))


def _verify(output):
    protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']or protocol['recipe']!=RECIPE:
        raise ValueError('Prepared floor recipe changed')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    return protocol


def fit(output=OUTPUT):
    output=Path(output).resolve();protocol=_verify(output)
    if (output/'MODEL.json').exists():raise FileExistsError('Preserve prior model')
    source=_source();samples=_samples(source,set(protocol['cal_ids']));floor=source['minimum_radius_px']
    full=fit_event_floor(samples,floor);folds=[]
    for session in sorted({r['session']for r in samples}):
        local=fit_event_floor(samples,floor,heldout_session=session)
        held=[r for r in samples if r['session']==session];train=[r for r in samples if r['session']!=session]
        held_events=_events(train,held,floor)
        folds.append(dict(model=local,heldout_session=session,heldout_events=held_events,
            heldout_mismatch_frames=sum(r['event']for r in held_events),heldout_frames=len(held_events)))
    model=dict(schema=SCHEMA,recipe=RECIPE,fit_partition='CAL',source_O_model_digest=source['model_sha256'],
        source_excluded=source['excluded'],source_samples=7,source_frames=5,source_sessions=2,
        source_is_plastic_only=True,wood_examples=0,full_fit=full,epsilon=full['epsilon'],
        new_runtime_radius_or_point_fit=False,only_correlated_regularization_floor=True,
        probability_or_independent_generalization_claim=False)
    model['model_digest_sha256']=SOURCE._digest(model)
    C.write(output/'MODEL.json',model);C.write(output/'CAL_LOGO_REPORT.json',dict(folds=folds,
        source_samples=samples,tiny_conditional_sample=True,primary_reference_reads=0,new_pose_or_RGB_or_NN=0))
    _verify(output)
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),
        LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        source_sha256=protocol['source_sha256'],input_sha256=protocol['input_sha256'],fit_partition='CAL'))
    print(dict(epsilon=model['epsilon'],frame_cap=full['maximum_frame_energy'],frames=5,samples=7),flush=True)


def prior_parameters(model):
    if model.get('schema')!=SCHEMA or model.get('recipe')!=RECIPE or model.get('fit_partition')!='CAL':
        raise ValueError('Explicit new CAL coarse-O floor model required')
    body={k:v for k,v in model.items()if k!='model_digest_sha256'}
    if SOURCE._digest(body)!=model.get('model_digest_sha256'):raise ValueError('Floor model digest changed')
    epsilon=float(model['epsilon'])
    if not 0<epsilon<1:raise ValueError('Nondegenerate bounded floor required')
    return dict(epsilon=epsilon,huber_scale=RECIPE['huber_scale'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','fit'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    else:fit(args.output)
