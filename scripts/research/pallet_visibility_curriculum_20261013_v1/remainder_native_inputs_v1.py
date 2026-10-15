"""Prepare cached predictor-only native143 inputs; never open GT annotations.

The historical journal mixes predictions and scores. Only an explicit
top-level whitelist is decoded; corner/branch/reference/metric values are
textually skipped. Known size comes from frozen prediction registry metadata,
and Q is authenticated against the prediction-only coarse solver convention.
No image decoding, backbone forward, line fit or new PnP occurs here.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import gzip
import json
import shutil

import numpy as np
from scipy.spatial.distance import cdist

from . import common as C
from . import statewise_evaluation_v4 as E
from .statewise_joint_api_v2 import registered_geometry

READINESS=E.COHORT/'remainder_after_test49_v1/run_input_readiness_v1'
OUTPUT=READINESS.parent/'native_input_preparation_v1'
ALLOWED_FIELDS=frozenset(('id','method','raw_hw','native_points','qFinal','actual_pose','pose_parameters_saved'))


def _skip_value(s,i):
    while s[i].isspace():i+=1
    if s[i] in '[{':
        depth=0;quoted=False;escaped=False
        while i<len(s):
            char=s[i]
            if quoted:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char=='"':quoted=False
            elif char=='"':quoted=True
            elif char in '[{':depth+=1
            elif char in ']}':
                depth-=1
                if depth==0:return i+1
            i+=1
        raise ValueError('Malformed skipped JSON value')
    if s[i]=='"':
        i+=1;escaped=False
        while i<len(s):
            char=s[i]
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':return i+1
            i+=1
        raise ValueError('Malformed skipped JSON string')
    while i<len(s) and s[i] not in ',}':i+=1
    return i


def select_prediction_fields(line,fields=ALLOWED_FIELDS):
    decoder=json.JSONDecoder();i=0;result={}
    while line[i].isspace():i+=1
    if line[i]!='{':raise ValueError('Prediction journal object required')
    i+=1
    while True:
        while line[i].isspace() or line[i]==',':i+=1
        if line[i]=='}':break
        key,i=decoder.raw_decode(line,i)
        while line[i].isspace():i+=1
        if line[i]!=':':raise ValueError('Malformed prediction journal key')
        i+=1
        while line[i].isspace():i+=1
        if key in fields:result[key],i=decoder.raw_decode(line,i)
        else:i=_skip_value(line,i)
    return result


def predictor_record(reference,prediction,registry_record):
    """Pure API argument builder; no reference pose/order/corner values."""
    if prediction['id']!=reference['id'] or prediction['method']!='N3_SUBPIX':raise ValueError('Exact cached N3_SUBPIX id required')
    if prediction['raw_hw']!=reference['image_hw']:raise ValueError('Cached raw camera canvas mismatch')
    q=np.asarray(prediction['qFinal'],float)
    if q.shape!=(9,2) or not np.array_equal(q,np.asarray(prediction['native_points'],float),equal_nan=True):
        raise ValueError('Exact unchanged native9 combined prediction required')
    pose=deepcopy(prediction['actual_pose'])
    if not isinstance(pose,dict) or not pose.get('available'):raise ValueError('Available cached predictor-only coarse pose required')
    R=np.asarray(pose['R_cf'],float);physical=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
    extents=np.asarray(pose['cf_extents'],float)
    for rotation in (R,physical):
        if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7,rtol=0) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-7):
            raise ValueError('Proper cached native/physical coarse rotations required')
    if t.shape!=(3,) or not np.isfinite(t).all() or t[2]<=0 or extents.shape!=(3,) or not np.isfinite(extents).all() or np.any(extents<=0):
        raise ValueError('Finite positive-depth coarse pose and registered extents required')
    registered=np.array([registry_record['physical_dimensions_m'][k] for k in ('x','y','z')],float)
    if not np.array_equal(registered,np.asarray(reference['known_physical_dimensions_xyz_m'],float)):
        raise ValueError('External known size differs from historical prediction registry')
    Q=R.T@physical
    Ry90=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    if np.allclose(extents,registered,atol=1e-6,rtol=0):expected=np.eye(3)
    elif np.allclose(extents,registered[[2,1,0]],atol=1e-6,rtol=0):expected=Ry90
    else:raise ValueError('Coarse extents not a registered prediction-only W/D hypothesis')
    if not np.allclose(Q,expected,atol=1e-7,rtol=0):raise ValueError('Saved coarse Q differs from fixed source=False native convention')
    xyz=C.cuboid(*extents);body=C.cuboid(*registered);distances=cdist(xyz@Q,body)
    gap=max(float(distances.min(0).max()),float(distances.min(1).max()))
    if gap>1e-7:raise ValueError('Native/physical registered unordered cuboid mismatch')
    K=np.asarray(reference['K'],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Finite camera intrinsics required')
    geometry=registered_geometry(reference['image_hw'],K,xyz,pose)
    gated=q.copy();gated[:8][np.asarray(geometry['initial_self_hidden'],bool)]=[-1.,-1.]
    frame=dict(id=reference['id'],source_record_id=reference['id'],partition='REMAINDER',session=reference['session'],
        cohort='REMAINDER_FINAL_CLEAN_MODERATE143',image=reference['image'],image_sha256=reference['image_sha256'],
        raw_hw=deepcopy(reference['image_hw']),K=K.tolist(),raw_points=q.tolist(),initial_pose=pose,crop_transform=None,
        native_xyz=xyz.tolist(),physical_registration_Q=Q.tolist(),
        known_physical_dimensions_xyz_m=registered.tolist(),prediction_object_type=reference['historical_prediction_metadata_object_type'])
    return C.finite(dict(schema='REMAINDER_CACHED_NATIVE9_PREDICTION_INPUT_V1',id=reference['id'],frame=frame,
        original_source=deepcopy(frame),initial_geometry=geometry,raw_points_initial_S_gated=gated.tolist(),
        provenance=dict(native9_center_preserved=True,native_channels_permuted=False,GT_coordinates_or_pose_or_human_state_inputs=False,
            reference_order_or_phase_used=False,external_known_size=True,source_basis=False,
            initial_S_geometry_only=True,S_raw_queries_after_initial_geometry=0,
            cached_coarse_pose_not_recomputed=True,new_image_NN_line_or_PnP_calls=0,
            unordered_native_physical_cube_gap_m=gap,registry_Q_convention_verified=True,
            source_prediction_journal_sha256=reference['upstream_prediction']['sha256'],
            source_feature_cache_sha256=reference['upstream_cache']['sha256'])))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prepared143 input artifacts')
    seal=C.read(READINESS/'INPUT_SEAL.json');bindings=C.read(READINESS/'INPUT_BINDINGS.json')
    for name,digest in seal['outputs'].items():
        if C.sha(READINESS/name)!=digest:raise ValueError('Readiness output binding changed '+name)
    E._verify_sources(bindings);protected=seal['protected_user_sha256'];E._verify_sources(protected)
    references=C.read(READINESS/'RUN_INPUTS143.json')['rows']
    if len(references)!=143 or len({r['id'] for r in references})!=143:raise ValueError('Exact original143 population required')
    desired={r['id'] for r in references};journal=C.ROOT/references[0]['upstream_prediction']['path']
    if C.sha(journal)!=references[0]['upstream_prediction']['sha256']:raise ValueError('Cached N3 journal changed')
    predictions={}
    with gzip.open(journal,'rt') as stream:
        for line in stream:
            header=select_prediction_fields(line,{'id','method'})
            if header.get('id') in desired and header.get('method')=='N3_SUBPIX':
                record=select_prediction_fields(line)
                if record['id'] in predictions:raise ValueError('Duplicate N3 cached record')
                predictions[record['id']]=record
    if set(predictions)!=desired:raise ValueError('Complete exact cached143 required')
    registry_path=C.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
    registry={r['object_type']:r for r in C.read(registry_path)['objects']}
    records=[]
    for reference in references:
        if reference['severity'] not in ('CLEAN','MODERATE_OCCLUSION'):raise ValueError('Severe input excluded')
        if C.sha(C.ROOT/reference['image'])!=reference['image_sha256']:raise ValueError('Actual RGB binding changed')
        if C.sha(C.ROOT/reference['upstream_cache']['path'])!=reference['upstream_cache']['sha256']:raise ValueError('Actual feature cache changed')
        records.append(predictor_record(reference,predictions[reference['id']],registry[reference['historical_prediction_metadata_object_type']]))
    output.mkdir(parents=True)
    sources={str(Path(__file__).relative_to(C.ROOT)):C.sha(__file__),str(Path(registered_geometry.__code__.co_filename).relative_to(C.ROOT)):C.sha(registered_geometry.__code__.co_filename)}
    inputs={**bindings,**{str((READINESS/n).relative_to(C.ROOT)):C.sha(READINESS/n) for n in ('INPUT_SEAL.json','RUN_INPUTS143.json','READINESS_PACKET_V1.json')}}
    C.write(output/'PROTOCOL.json',dict(schema='REMAINDER143_NATIVE_INPUT_PREPARATION_V1',ids=[r['id'] for r in references],rows=143,severe=0,
        whitelist=sorted(ALLOWED_FIELDS),GT_annotation_files_or_scored_values_decoded=False,
        known_dimension_source='frozen historical prediction registry + external size condition',
        reference_signed_axis_quality_not_part_of_inference=True,new_RGB_NN_line_or_PnP_calls=0,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected))
    C.save_rows(output/'NATIVE_INPUTS143.jsonl.gz',records)
    C.write(output/'READINESS_NOTES.json',dict(rows=143,native_and_registered_Q_ready=143,
        actual_native_annotation_phase_unconfirmed=143,independent_physical_GT_ready=0,
        source_readiness_packet=str((READINESS/'READINESS_PACKET_V1.json').relative_to(C.ROOT)),
        prediction_input_excludes_annotation_state_order_pose_corner_values=True,
        fresh_state_mask_correction_partial_and_pose_stages_not_executed_here=True))
    for relative,digest in sources.items():
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,dest);assert C.sha(dest)==digest
    E._verify_sources(inputs);E._verify_sources(protected)
    C.write(output/'INPUT_SEAL.json',dict(rows=143,ordered_ids=[r['id'] for r in references],native_inputs_sha256=C.sha(output/'NATIVE_INPUTS143.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        source_Q_shape_checks_all_PASS=True,new143_RGB_or_NN_or_line_or_PnP_calls=0,GT_values_decoded=False))
    print('NATIVE143_PREPARED',str(output),C.sha(output/'INPUT_SEAL.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    prepare(parser.parse_args().output)
