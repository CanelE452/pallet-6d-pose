"""Isolated cached PoseFix RAW143 supplement; no forward or new pose fit.

Whitelist only cached prediction objects. Their historical GT metric siblings
are skipped without JSON decoding. Camera/registered-size metadata alone is
whitelisted from the mixed target journal, never poses/corners/human states.
Scoring follows the full remaining143 prediction and owned reference seals.
"""
import argparse
from pathlib import Path
import json
import gzip
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist

from . import common as C
from . import observe_remainder_rgb_v4_14 as O
from . import audit_remainder_pipeline_v4_14 as A
from . import remainder_native_inputs_v1 as N
from . import score_remainder_derived_reference_v4_14 as REF

OUTPUT=O.REMAINDER/'posefix_baseline_supplement_v4_14'
PF=C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1'
CACHE=PF/'predictions/seed1_REAL_DEV.npz'
POSE_CACHE=PF/'pose/seed1_REAL_DEV_RAW.json'
OLD=C.DOC/'nonsevere_v1/posefix_sub_comparison_v1'


def _array_records(text):
    i=1
    while True:
        while text[i].isspace()or text[i]==',':i+=1
        if text[i]==']':break
        end=N._skip_value(text,i);yield text[i:end];i=end


def _first_pass_predictions():
    text=POSE_CACHE.read_text();i=1
    while text[i].isspace():i+=1
    i=N._skip_value(text,i)  # PASS0 is not decoded.
    while text[i].isspace()or text[i]==',':i+=1
    end=N._skip_value(text,i)
    return [N.select_prediction_fields(item,fields={'prediction'})['prediction']
            for item in _array_records(text[i:end])]


def _target_metadata(wanted):
    text=(PF/'TARGETS.json').read_text();i=1;decoder=json.JSONDecoder();out={}
    while True:
        while text[i].isspace()or text[i]==',':i+=1
        if text[i]=='}':break
        fid,i=decoder.raw_decode(text,i)
        while text[i].isspace():i+=1
        if text[i]!=':':raise ValueError('Mixed metadata journal key malformed')
        i+=1
        while text[i].isspace():i+=1
        end=N._skip_value(text,i)
        if fid in wanted:
            out[fid]=N.select_prediction_fields(text[i:end],fields={'K','xyz','source'})
        i=end
    return out


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior PF baseline supplement')
    native=list(C.rows(O.INPUT/'NATIVE_INPUTS143.jsonl.gz'));wanted={r['id']:r for r in native}
    old_protocol=C.read(OLD/'PROTOCOL.json');expected=old_protocol['source_bindings'][str(CACHE)]
    if C.sha(CACHE)!=expected:raise ValueError('Exact original95 PoseFix neural cache required')
    with np.load(CACHE,allow_pickle=False)as packet:
        ids=[str(x)for x in packet['ids']];points=packet['points'][:,0,1].copy()
        checkpoint=str(packet['checkpoint_sha256'].item())
    if checkpoint!=old_protocol['PoseFix']['checkpoint_sha256']:
        raise ValueError('Frozen seed1 PRIOR6000 checkpoint changed')
    poses=dict(zip(ids,_first_pass_predictions()))
    if len(poses)!=319 or not set(wanted)<=set(poses):raise ValueError('All143 cached poses required')
    images={}
    for value in _array_records((PF/'INPUTS.json').read_text()):
        record=N.select_prediction_fields(value,fields={'id','split','image','image_sha256','raw_hw'})
        if record['id']in wanted:images[record['id']]=record
    metadata=_target_metadata(wanted);records=[];phases=Counter()
    Ry90=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    for original in native:
        fid=original['id'];frame=original['frame'];image=images[fid];meta=metadata[fid];p=poses[fid]
        if image['split']!='REAL_DEV'or any(image[k]!=frame[k]for k in ('image','image_sha256','raw_hw')):
            raise ValueError('PF/current actual RGB differs')
        if meta['K']!=frame['K']or meta['xyz']!=frame['known_physical_dimensions_xyz_m']or meta['source']is not False:
            raise ValueError('PF/current camera, known size or source=False differs')
        if not p.get('available'):raise ValueError('Unavailable PF pose must be reported, not dropped')
        Q=np.asarray(p['R_cf']).T@np.asarray(p['R_physical'])
        tag='I'if np.allclose(Q,np.eye(3),atol=1e-7,rtol=0)else 'Ry90'
        if tag=='Ry90'and not np.allclose(Q,Ry90,atol=1e-7,rtol=0):raise ValueError('PF native basis incompatible')
        phases[tag]+=1
        gap=cdist(C.cuboid(*p['cf_extents'])@Q,C.cuboid(*meta['xyz']))
        if max(gap.min(0).max(),gap.min(1).max())>1e-7:raise ValueError('PF known physical size mismatch')
        records.append(dict(id=fid,session=frame['session'],partition='REMAINDER',image=frame['image'],
            image_sha256=frame['image_sha256'],method='POSEFIX_RAW',actual_pose=p,
            native_points=points[ids.index(fid)].tolist(),physical_registration_Q=Q.tolist(),
            metadata_K=meta['K'],metadata_registered_size=meta['xyz']))
    # Exact parity with the previously reported95 RAW baseline establishes the
    # same cached F variant, not a newly chosen pass/cap/SubPix configuration.
    checked=0
    with gzip.open(OLD/'PREDICTIONS.jsonl.gz','rt')as stream:
        for line in stream:
            old=N.select_prediction_fields(line,fields={'id','method','actual_pose'})
            if old['method']!='POSEFIX_RAW':continue
            for key in ('R_cf','R_physical','centroid','cf_extents'):
                if poses[old['id']][key]!=old['actual_pose'][key]:raise ValueError('Original95 RAW variant changed')
            checked+=1
    if checked!=95:raise ValueError('Original95 exact parity missing')
    output.mkdir(parents=True)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in (CACHE,POSE_CACHE,PF/'INPUTS.json',PF/'TARGETS.json',
        OLD/'PROTOCOL.json',OLD/'PREDICTIONS.jsonl.gz',O.INPUT/'INPUT_SEAL.json',O.INPUT/'NATIVE_INPUTS143.jsonl.gz')}
    sources={str(Path(p).relative_to(C.ROOT)):C.sha(p)for p in (__file__,N.__file__,A.__file__,REF.__file__)}
    C.write(output/'PROTOCOL.json',dict(schema='CACHED_POSEFIX_RAW_REMAINDER143_SUPPLEMENT_V1',rows=143,
        ids=[r['id']for r in records],input_sha256=inputs,source_sha256=sources,
        checkpoint_sha256=checkpoint,PoseFix_variant=old_protocol['PoseFix'],cached_F_pose_variant='source=False, unmodified width/depth selector SQPnP+RefineLM',
        original95_pose_parity_rows=95,original95_pose_max_abs_difference=0.,Q_counts=dict(phases),
        same143_RGB_camera_knownsize_and_centroid_origin=True,source_TRUE_Rxpi_used=False,
        missing_cached_poses=0,new_neural_forwards=0,new_PnP_fits=0,new_training=0,
        numeric_GT_pose_corner_humanstate_or_metric_values_decoded=False,
        metadata_fields_whitelisted_from_mixed_targets=['K','xyz','source'],
        prediction_fields_whitelisted_from_mixed_pose_cache=['prediction'],
        different_RGB_head_full_pipeline_comparator_not_same_upstream_postprocessor_only=True,
        appended_SubPix=False,cap_applied=False,first_RAW_pass_only=True,
        existing_four_arm_comparison_plan_unchanged=True,reference_plan=A.binding(REF.OUTPUT/'PLAN.json')))
    C.save_rows(output/'PREDICTIONS.jsonl.gz',records)
    C.write(output/'BASELINE_SEAL.json',dict(rows=143,predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_remaining_numeric_GT_decode=True,
        before_remaining143_postseal_baseline_comparison=True))
    print('sealed exact cached PoseFix RAW143; no GT numeric decode/inference',flush=True)


def postseal(output=OUTPUT):
    output=Path(output).resolve();p=C.read(output/'PROTOCOL.json');seal=C.read(output/'BASELINE_SEAL.json')
    for name in ('input_sha256','source_sha256'):E=A.E;E._verify_sources(p[name])
    if C.sha(output/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('PF cached baseline seal changed')
    if (output/'POSTSEAL_SCORE.json').exists():raise FileExistsError('Preserve PF postseal comparison')
    final=A.C.read(A.OUTPUT/'POSTSEAL_COMPARISON_V1.json')
    own=C.read(REF.OUTPUT/'SCORE_SEAL.json');full=C.read(A.POSE/'PREDICTION_SEAL.json')
    if full['rows']!=143 or own['prediction_seal_sha256']!=C.sha(A.POSE/'PREDICTION_SEAL.json'):
        raise ValueError('Complete143 prediction and owned reference score seals required')
    A.verify_source_closure();annotations={r['id']:r for r in C.read(REF.OUTPUT/'PLAN.json')['annotations']}
    baseline={r['id']:r for r in C.rows(output/'PREDICTIONS.jsonl.gz')};rows=[]
    for row in final['records']:
        fid=row['id'];record=annotations[fid];path=C.ROOT/record['annotation']
        if C.sha(path)!=record['annotation_sha256']:raise ValueError('Fixed reference document changed')
        reference=REF.reference_from_document(C.read(path));metric=A.independent_metric(baseline[fid]['actual_pose'],reference)
        rows.append(dict(id=fid,session=row['session'],metrics={**row['metrics'],'POSEFIX_RAW':metric}))
    summary=dict(rows=143,translation_cm=A._distribution([r['metrics']['POSEFIX_RAW']['translation_cm']for r in rows]),
        rotation_deg=A._distribution([r['metrics']['POSEFIX_RAW']['rotation_deg']for r in rows if r['metrics']['POSEFIX_RAW']['rotation_deg']is not None]),
        translation_le5=sum(r['metrics']['POSEFIX_RAW']['translation_cm']<=5 for r in rows),
        successes=sum(bool(r['metrics']['POSEFIX_RAW']['success_5cm_5deg'])for r in rows),
        max_translation_id=max(rows,key=lambda r:r['metrics']['POSEFIX_RAW']['translation_cm'])['id'])
    C.write(output/'POSTSEAL_SCORE.json',dict(summary=summary,rows=rows,
        paired_final_minus_PoseFix_RAW=A._paired(rows,'POSEFIX_RAW',A.ARMS[-1]),
        initial_four_arm_comparison_plan_unchanged=True,all143_kept=True,
        derived_reference_policy=p['reference_plan'],independent_physical_GT_or_unseen_claim=False,
        new_model_or_PnP_fits=0,baseline_seal=A.binding(output/'BASELINE_SEAL.json'),
        full143_prediction_seal=A.binding(A.POSE/'PREDICTION_SEAL.json')))
    C.write(output/'SCORE_SEAL.json',dict(score_sha256=C.sha(output/'POSTSEAL_SCORE.json'),
        all143=True,numeric_GT_decode_after_full143_prediction_and_owned_score_seals=True))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','postseal'))
    args=parser.parse_args();(prepare if args.stage=='prepare'else postseal)()
