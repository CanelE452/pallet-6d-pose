"""Fresh17c queries after sealed initial-grazing B/I->U adaptation.

The image/geometry/mask/configuration and NMS/DAG/topology implementations
are unchanged. Only the independently sealed authoritative state gate,
filtered B/I archive and appended broad U regions replace their original
counterparts before any new local query. CAL46 and REST192 share one worker.
This distinct observation generation needs a new CAL noise fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import types

import numpy as np

from . import common as C
from . import observe_native_edge_chains_v4_17b as OLD


OUTPUT = OLD.BOUNDARY/'grazing_native_edge_observations_v4_18b'
GRAZING = OLD.BOUNDARY/'grazing_native_candidates_v4_18'
OBSERVATION_GENERATION = 'INITIAL_GRAZING_U_THEN_ACTUAL_IMAGE_NMS17C_WITH_UNCHANGED_DAG_AND_TOPOLOGY'
NMS = OLD.NMS


def _same_numbers(a, b, name):
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape or not np.allclose(aa,bb,atol=1e-12,rtol=0):
        raise ValueError('Exact initial registered geometry required: '+name)


def _join_task(original, adapted):
    """Whitelist the sealed adaptation; do not copy proposals/raw/targets."""
    for key in ('id','partition','session','image','image_sha256'):
        if original[key] != adapted[key]:
            raise ValueError('Original/adapted frame or RGB identity differs: '+key)
    geometry = original['initial_geometry']
    old_states, states = original['states'], adapted['states']
    gate = adapted['grazing_adaptation']['geometry_gate']
    if (len(old_states) != 8 or len(states) != 8 or states != adapted['spec']['states']
            or old_states != adapted['grazing_adaptation']['original_states']
            or states != gate['states'] or len(geometry['initial_self_hidden']) != 8):
        raise ValueError('Eight original and adapted authoritative state gates required')
    changed = []
    for k, (before, after) in enumerate(zip(old_states,states)):
        # Permanent S and previous U/O/T are checked before any measurement.
        if geometry['initial_self_hidden'][k] or before in ('S','U','O','T'):
            if after != before:
                raise ValueError('Permanent S and previous U/O/T must stay unchanged')
        elif before != after:
            if before not in ('B','I') or after != 'U':
                raise ValueError('Only sealed initial-grazing B/I->U changes are allowed')
            changed.append(k)
    if changed != gate['new_U_native_ids']:
        raise ValueError('Changed states differ from the sealed geometry-only gate')
    if geometry['initial_self_hidden'] != adapted['initial_self_hidden']:
        raise ValueError('Initial permanent S geometry differs')
    expected_visible = []
    for record in original['visible_measurements']:
        k = NMS.V3._id(record)
        if geometry['initial_self_hidden'][k] or old_states[k] not in ('B','I'):
            raise ValueError('Original visible archive contains excluded evidence')
        if k not in changed:
            expected_visible.append(record)
    if expected_visible != adapted['spec']['visible']:
        raise ValueError('Retained B/I q/covariance/fit covariance/reliability must be exact')
    if original['domains'] != adapted['spec']['domains']:
        raise ValueError('Original O/T numeric domains changed')
    expected_regions = deepcopy(original['u_regions'])
    new_regions = adapted['grazing_adaptation']['new_U_regions']
    if any(NMS.V3._id(r) not in changed for r in new_regions):
        raise ValueError('Only new grazing-U regions may be appended')
    expected_regions.extend(new_regions)
    if expected_regions != adapted['u_regions']:
        raise ValueError('Previous U regions must be preserved exactly; append sealed new regions only')
    missing = adapted['grazing_adaptation']['missing_new_U_raw_native_ids']
    if sorted([NMS.V3._id(r) for r in new_regions]+list(missing)) != sorted(changed):
        raise ValueError('Every new U must have its sealed region or explicit missing-center abstention')
    if original['nonself_prediction_scale_px'] != adapted['grazing_adaptation']['U_radius_px']/adapted['grazing_adaptation']['borrowed_CAL264_radius_ratio']:
        # Multiplication/division can round once, without changing any pixels.
        if not np.isclose(original['nonself_prediction_scale_px'],adapted['grazing_adaptation']['U_radius_px']/adapted['grazing_adaptation']['borrowed_CAL264_radius_ratio'],atol=1e-12,rtol=1e-14):
            raise ValueError('Frozen nonS prediction scale changed')
    family = adapted['families']['ORIGINAL_WD']
    for a,b,name in ((geometry['native_xyz'],family['native_xyz'],'native XYZ'),
            (geometry['physical_registration_Q'],family['physical_registration_Q'],'physical Q'),
            (geometry['cf_extents'],family['cf_extents'],'known dimensions'),
            (geometry['R_cf'],family['coarse_pose']['R'],'initial R'),
            (geometry['centroid'],family['coarse_pose']['t'],'initial centroid'),
            (geometry['K'],adapted['spec']['K'],'K')):
        _same_numbers(a,b,name)
    if geometry['image_hw'] != adapted['spec']['image_hw']:
        raise ValueError('Initial image canvas differs')
    task = deepcopy(original)
    task.update(states=deepcopy(states),visible_measurements=deepcopy(expected_visible),
        u_regions=deepcopy(adapted['u_regions']),original_states=deepcopy(old_states),
        new_grazing_U_native_ids=changed,
        original_observation_math_sha256=original['unchanged_observation_math_sha256'],
        sealed_grazing_adaptation_sha256=OLD._digest(adapted['grazing_adaptation']),
        sealed_grazing_filtered_boundary_sha256=OLD._digest(adapted['boundary_merged']),
        observation_generation=OBSERVATION_GENERATION)
    task['unchanged_observation_math_sha256'] = OLD._digest(OLD._math_fields(task))
    return task


def _merge_maps(*maps):
    result = {}
    for mapping in maps:
        for key,value in mapping.items():
            if key in result and result[key] != value:
                raise ValueError('Conflicting frozen source/input binding: '+key)
            result[key] = value
    return result


def _inputs():
    originals, old_protocol, old_seal = OLD._load(OLD.OUTPUT)
    protocol = C.read(GRAZING/'PROTOCOL.json')
    seal = C.read(GRAZING/'INPUT_SEAL.json')
    if (seal['rows'] != 238 or C.sha(GRAZING/'INPUTS.jsonl.gz') != seal['input_sha256']
            or C.sha(GRAZING/'PROTOCOL.json') != seal['protocol_sha256']):
        raise ValueError('Sealed initial-grazing adapted238 inputs required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        OLD._verify_map(protocol[key])
    adapted = list(C.rows(GRAZING/'INPUTS.jsonl.gz'))
    ids = [r['id'] for r in adapted]
    if (len(ids) != 238 or len(set(ids)) != 238 or ids != seal['ordered_ids']
            or ids != old_seal['ordered_ids'] or ids != protocol['ids']):
        raise ValueError('Exact unique ordered original238 join required')
    tasks = [_join_task(o,a) for o,a in zip(originals,adapted)]
    if (Counter(t['partition'] for t in tasks) != Counter(CAL=46,TEST=49,REMAINDER=143)
            or sum(len(t['new_grazing_U_native_ids']) for t in tasks) != protocol['predeclared_grazing_points']):
        raise ValueError('Original nonsevere population and predeclared grazing inventory differ')
    for task in tasks:
        binding = task['mask_binding']
        if protocol['input_sha256'].get(binding['path']) != binding['sha256']:
            raise ValueError('Adapted source does not bind the identical original mask file')
    sources = _merge_maps(old_protocol['source_sha256'],protocol['source_sha256'])
    for path in (Path(__file__),Path(OLD.__file__),Path(NMS.__file__),Path(OLD.TOPOLOGY.__file__)):
        sources[str(path.resolve().relative_to(C.ROOT))] = C.sha(path)
    inputs = _merge_maps(old_protocol['input_sha256'],protocol['input_sha256'])
    for folder,names in ((OLD.OUTPUT,('PROTOCOL.json','PREPARATION_SEAL.json','INPUTS.jsonl.gz')),
            (GRAZING,('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz'))):
        for name in names:
            path = folder/name
            inputs[str(path.relative_to(C.ROOT))] = C.sha(path)
    if old_protocol['protected_user_sha256'] != protocol['protected_user_sha256']:
        raise ValueError('Protected user file inventories differ')
    return tasks,sources,inputs,old_protocol['protected_user_sha256'],old_protocol


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every original or partial observation archive')
    tasks,sources,inputs,protected,parent = _inputs()
    for mapping in (sources,inputs,protected):
        OLD._verify_map(mapping)
    output.mkdir(parents=True)
    protocol = dict(schema='GRAZING_U_UNIFORM_NATIVE_EDGE_OBSERVATION_PROTOCOL_V4_18B',
        ordered_ids=[t['id'] for t in tasks],population=dict(CAL=46,TEST=49,REMAINDER=143,total=238,severe=0),
        prepared_before_new_RGB_observations_UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        inherited_parent_population=parent['population'],configuration=deepcopy(NMS.DEFAULT_CONFIGURATION),
        observation_generation=OBSERVATION_GENERATION,initial_grazing_U_count=sum(len(t['new_grazing_U_native_ids']) for t in tasks),
        initial_grazing_U_frames=sum(bool(t['new_grazing_U_native_ids']) for t in tasks),
        query_order='Use sealed adapted states/filtered visible before frozen17c actual NMS; unknown mates receive existing14px protected exclusion.',
        unchanged_quantities=['original RGB and full clean SAM mask','initial ORIGINAL XYZ/K/Q/R/centroid/canvas',
            'all retained B/I q and noise/reliability','all original S/O/T/U states','O/T numeric domains','previous U regions','nonS scale'],
        changes='Only sealed initial-grazing B/I become U; new broad U regions appended; fresh actual image queries after new state gate.',
        full_RGB_pipeline_state_classification_or_mask_NN_forward=False,
        old_grazing_endpoint_RGB_samples_or_old_NMS_chains_reused=False,
        same_worker_and_configuration_for_CAL_and_REST=True,DAG_or_NMS_or_topology_math_changes=0,
        all_returned_alternatives_and_zero_edge_abstentions_preserved=True,
        combinatorial_overflow_remains_explicit_whole_edge_abstention=True,
        precise_keypoint_adoptions=0,old_q_covariance_or_retained_state_changes=0,
        S_local_queries=0,U_corner_queries=0,new_NN_forwards=0,pose_calls=0,reference_coordinate_decodes=0,
        existing_CAL_noise_models_validated_for_this_generation=False,new_CAL_noise_refit_required=True,
        engineering_angle_floor_is_not_CAL_total_error_model=True,
        default_pipeline_replacement=False,reused_data_is_exploratory=True)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        path = output/'code_snapshot'/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path) != digest:
            raise ValueError('Exact frozen source snapshot required')
    C.save_rows(output/'INPUTS.jsonl.gz',tasks)
    for mapping in (sources,inputs,protected):
        OLD._verify_map(mapping)
    C.write(output/'PREPARATION_SEAL.json',dict(schema='GRAZING_U_UNIFORM_OBSERVATION_PREPARATION_SEAL_V4_18B',
        rows=238,ordered_ids=protocol['ordered_ids'],protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),source_sha256=sources,
        observation_generation=OBSERVATION_GENERATION,prepared_before_new_RGB_observation=True,
        no_reference_numeric_decode=True))
    print(json.dumps(dict(prepared=238,output=str(output),grazing_U_count=protocol['initial_grazing_U_count'])),flush=True)


def observe_one(task):
    if task['observation_generation'] != OBSERVATION_GENERATION:
        raise ValueError('New grazing-gated observation generation required')
    for k in task['new_grazing_U_native_ids']:
        if task['states'][k] != 'U' or task['initial_geometry']['initial_self_hidden'][k]:
            raise ValueError('New U gate must precede any local image query')
    result = OLD.observe_one(task)
    result.update(schema='GRAZING_U_UNIFORM_NATIVE_EDGE_OBSERVATION_ROW_V4_18B',
        observation_generation=OBSERVATION_GENERATION,original_states=task['original_states'],
        new_grazing_U_native_ids=task['new_grazing_U_native_ids'],
        sealed_grazing_adaptation_sha256=task['sealed_grazing_adaptation_sha256'],
        original_observation_math_sha256=task['original_observation_math_sha256'])
    result['contract'].update(fresh_NMS_after_new_U_gate=True,old_grazing_endpoint_RGB_reused=False,
        all_retained_BI_measurements_exact=True,old_noise_model_not_used=True,new_CAL_noise_refit_required=True)
    return result


def _summary(rows):
    result = OLD._summary(rows)
    result.update(observation_generation=OBSERVATION_GENERATION,
        initial_grazing_U_count=sum(len(r['new_grazing_U_native_ids']) for r in rows),
        frames_with_initial_grazing_U=sum(bool(r['new_grazing_U_native_ids']) for r in rows),
        new_CAL_noise_refit_required=True)
    return result


def observe(output=OUTPUT,partition='CAL',workers=3):
    # Private facade changes dispatch/metadata only; the frozen NMS operation
    # and original robust cancellation/checkpoint implementation are untouched.
    runner = types.FunctionType(OLD.observe.__code__,{**OLD.observe.__globals__,
        'observe_one':observe_one,'_summary':_summary},OLD.observe.__name__,OLD.observe.__defaults__)
    return runner(Path(output),partition,workers)


def finalize(output=OUTPUT):
    output = Path(output).resolve()
    tasks,protocol,_ = OLD._load(output)
    if (output/'OBSERVATION_SEAL.json').exists():
        raise FileExistsError('Preserve complete grazing-gated238 seal')
    for name in ('CAL','REST'):
        OLD._verify_observation_seal(output,name)
    rows = [r for name in ('CAL','REST') for r in C.rows(output/(name+'_OBSERVATIONS.jsonl.gz'))]
    by_id = {r['id']:r for r in rows}
    if len(rows)!=238 or len(by_id)!=238 or set(by_id)!={t['id'] for t in tasks}:
        raise ValueError('Every original238, including all zero-edge abstentions, is required')
    ordered = []
    for task in tasks:
        row = by_id[task['id']]
        for key in ('partition','states','original_states','visible_measurements','domains','u_regions',
                'initial_geometry','new_grazing_U_native_ids','unchanged_observation_math_sha256',
                'sealed_grazing_adaptation_sha256','observation_generation'):
            if row[key] != task[key]:
                raise ValueError('Exact sealed adapted observation quantities changed: '+key)
        ordered.append(row)
    C.save_rows(output/'OBSERVATIONS.jsonl.gz',ordered)
    summary = _summary(ordered)
    C.write(output/'SUMMARY.json',summary)
    C.write(output/'OBSERVATION_SEAL.json',dict(schema='GRAZING_U_UNIFORM_NATIVE_EDGE_OBSERVATION_SEAL_V4_18B',
        rows=238,ordered_ids=[t['id'] for t in tasks],observation_generation=OBSERVATION_GENERATION,
        observations_sha256=C.sha(output/'OBSERVATIONS.jsonl.gz'),summary_sha256=C.sha(output/'SUMMARY.json'),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        partition_seals={name:C.sha(output/(name+'_OBSERVATION_SEAL.json')) for name in ('CAL','REST')},
        source_sha256=protocol['source_sha256'],all_frames_including_abstentions_preserved=True,
        sealed_initial_grazing_U_applied_before_new_RGB=True,retained_BI_q_and_covariance_unchanged=True,
        initial_S_and_previous_U_O_T_unchanged=True,new_CAL_noise_refit_required=True,
        no_numeric_reference_decodes=True,no_NN_or_pose_run=True,default_pipeline_replaced=False))
    print(json.dumps(dict(grazing_generation_sealed=238,summary=summary)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('check','prepare','observe','finalize'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--partition',choices=('CAL','REST'),default='CAL')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.stage=='check':
        tasks,*_ = _inputs()
        print(json.dumps(dict(checked=238,new_grazing_U_count=sum(len(t['new_grazing_U_native_ids']) for t in tasks),new_RGB_queries=0)))
    elif args.stage=='prepare':prepare(args.output)
    elif args.stage=='observe':observe(args.output,args.partition,args.workers)
    else:finalize(args.output)
