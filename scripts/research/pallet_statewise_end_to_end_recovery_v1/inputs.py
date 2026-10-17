"""Join original prediction-only packets and actual earlier candidate poses.

No oracle candidate metrics, successful-candidate membership, six-case assisted
boundaries, or evaluation reference is opened. G10/G13 old objective values are
not ranked against each other: every carried pose is rescored under one model.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from .common import read, rows, save_rows, sha, write

FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')


def _pose_from_record(pose, family, *, source, name):
    if pose is None or pose.get('available') is False:
        return None
    Q = np.asarray(family['physical_registration_Q'], float)
    xyz = np.asarray(family['native_xyz'], float)
    R = np.asarray(pose.get('R_cf', pose.get('R')), float)
    if R.shape != (3, 3):
        return None
    center = pose.get('centroid')
    t = pose.get('native_t', pose.get('t'))
    if center is None and t is not None:
        center = R @ xyz.mean(0) + np.asarray(t, float)
    if center is None:
        return None
    return dict(candidate_id=name, source=source, R_physical=(R@Q).tolist(),
                centroid=np.asarray(center, float).tolist(), original_native_family=family['name'])


def real_frames(config, output):
    old = Path(config['old_private'])
    packet_path = old/'e_trace_v2/TYPED_PACKETS.jsonl.gz'
    trace_path = old/'e_trace_v2/E_TRACE.jsonl.gz'
    bank_path = old/'original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz'
    endpoint_path = old/'GENERATED_CANDIDATES.jsonl.gz'
    trace_seal = read(packet_path.parent/'EXTRACTION_SEAL.json')
    # The existing seal uses the exact original relative filenames.
    for p in (packet_path, trace_path):
        entry = trace_seal['files'][p.name]
        expected = entry.get('sha256') if isinstance(entry, dict) else entry
        if sha(p) != expected:
            raise ValueError('Original typed/trace seal mismatch: '+p.name)
    bseal = read(bank_path.parent/'CANDIDATE_MANIFEST.json')
    if sha(bank_path) != bseal['artifacts'][bank_path.name]:
        raise ValueError('Original candidate bank changed')
    endpoint_seal=read(old/'CANDIDATE_SEAL.json')
    if sha(endpoint_path)!=endpoint_seal['generated_candidates_sha256']:
        raise ValueError('Existing logged candidate endpoints changed')
    baselines = {r['id']: r for r in rows(old/'score_v2/BASELINE_PREDICTIONS.jsonl.gz')}
    traces = {r['id']: r for r in rows(trace_path)}
    binding = {str(p): sha(p) for p in (packet_path, trace_path, bank_path,endpoint_path,
                                      old/'score_v2/BASELINE_PREDICTIONS.jsonl.gz')}
    packets = rows(packet_path)
    banks = rows(bank_path)
    endpoints = rows(endpoint_path)
    frames = []
    for packet, bank, endpoint in zip(packets, banks, endpoints, strict=True):
        fid = packet['id']
        if bank['id'] != fid or bank['GT_reference_or_success_inputs'] is not False:
            raise ValueError('Prediction-only exact ID bank required')
        trace, baseline = traces[fid], baselines[fid]
        if packet['image_sha256'] != trace['image_sha256']:
            raise ValueError('Image bindings differ')
        families = {n: deepcopy(packet['original_e_stage1_family_seed_bank'][n]['family']) for n in FAMILIES}
        candidates = []
        source_counts=Counter()
        for stage in ('G10', 'G13'):
            for family in bank['stages'][stage]['families']:
                for sample in family['feasible_samples']:
                    p = _pose_from_record(sample['pose'], family, source='ORIGINAL_'+stage,
                        name=f'{stage}/{family["name"]}/sample/{sample["original_index"]}')
                    if p is not None:
                        candidates.append(p)
                for key in ('selected_solver_pose', 'published_actual_pose'):
                    p = _pose_from_record(family.get(key), family, source='ORIGINAL_'+stage,
                                         name=f'{stage}/{family["name"]}/{key}')
                    if p is not None:
                        candidates.append(p)
                for outcome in family['original_outcomes']:
                    p=_pose_from_record(outcome.get('original_record'),family,
                        source='ORIGINAL_'+stage+'_OUTCOME',
                        name=f'{stage}/{family["name"]}/actual_outcome/{outcome["original_index"]}')
                    source_counts[stage+'_outcome_pose_available' if p is not None else stage+'_outcome_Rt_unavailable']+=1
                    if p is not None:
                        candidates.append(p)
        if endpoint['id']!=fid or endpoint['GT_inputs'] is not False:
            raise ValueError('Exact prediction-only logged endpoint population required')
        for j,e in enumerate(endpoint['candidates']):
            if e['stage'] not in ('G10','G13') or e['family'] not in families:
                raise ValueError('Unknown original endpoint stage/family')
            p=_pose_from_record(dict(R=e['R'],t=e['t']),families[e['family']],
                 source='EXISTING_LOGGED_'+e['stage']+'_ENDPOINT',
                 name=f'LOGGED/{e["stage"]}/{e["family"]}/{e["kind"]}/{j}')
            source_counts['logged_endpoint_pose_available' if p is not None else 'logged_endpoint_Rt_unavailable']+=1
            if p is not None:
                candidates.append(p)
        for arm in ('N0', 'E0'):
            p = baseline['predictions'][arm]['actual_pose']
            if p['available']:
                candidates.append(dict(candidate_id=arm, source=arm,
                    R_physical=deepcopy(p['R_physical']), centroid=deepcopy(p['centroid']),
                    original_native_family=None))
        image = Path(config['source_root'])/packet['image']
        if not image.is_file() or sha(image) != packet['image_sha256']:
            raise ValueError('Exact bound RGB absent or changed: '+fid)
        # Preserve every available pose in the archive. A physically invalid
        # or behind-camera endpoint is not an eligible image-projection seed;
        # it remains explicit instead of disappearing from provenance/diagnosis.
        archived=deepcopy(candidates)
        admitted=[]
        for candidate in archived:
            R=np.asarray(candidate['R_physical'],float);center=np.asarray(candidate['centroid'],float)
            proper=bool(np.isfinite(R).all()and np.isfinite(center).all()and
                np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)and abs(np.linalg.det(R)-1)<1e-6)
            positive=False
            if proper:
                f=families['ORIGINAL_WD'];Q=np.asarray(f['physical_registration_Q']);xyz=np.asarray(f['native_xyz'])
                native_R=R@Q.T
                depth=((xyz-xyz.mean(0))@native_R.T+center)[:,2]
                positive=bool(np.min(depth)>packet['solver_input']['config'].get('positive_depth_m',1e-5))
            candidate['initial_projection_admitted']=proper and positive
            candidate['initial_projection_rejection_reason']=None if proper and positive else (
                'INVALID_NONFINITE_OR_NONPROPER_POSE' if not proper else 'NONPOSITIVE_MODEL_VERTEX_DEPTH')
            source_counts['projection_admitted'if proper and positive else 'projection_ineligible_preserved']+=1
            if proper and positive:
                admitted.append(deepcopy(candidate))
        frame = dict(id=fid, partition=packet['partition'], session=packet['session'],
                     cluster_id=baseline['cluster_id'], image=str(image),
                     image_sha256=packet['image_sha256'], packet=packet,
                     trace=trace, families=families, candidates=admitted,
                     candidate_archive_all=archived,candidate_source_counts=dict(source_counts),
                     fallback=deepcopy(baseline['predictions']['N0']['actual_pose']),
                     data_role='exploratory_development_evaluation', supervised_inputs=False)
        frames.append(frame)
    if len(frames)!=238 or Counter(f['partition'] for f in frames)!=Counter(CAL=46, TEST=49, REMAINDER=143):
        raise ValueError('Exact original238 population required')
    output = Path(output)
    save_rows(output/'PREDICTION_INPUTS238.jsonl.gz', frames)
    result = dict(schema='PREDICTION_ONLY_INPUT_JOIN_V1', frames=238,
                  partitions=dict(Counter(f['partition'] for f in frames)),
                  ordered_ids=[f['id'] for f in frames], inputs_sha256=binding,
                  output_sha256=sha(output/'PREDICTION_INPUTS238.jsonl.gz'),
                  original_candidate_references_read=False, reference_or_oracle_inputs=False,
                  old_G10_G13_objective_values_not_exported=True,
                  all_previous_pose_sources_carried=True,
                  scope='retained feasible + original available outcome poses + existing prediction-only logged endpoints + N0/E0; invalid projection poses archived explicitly',
                  source_counts=dict(sum((Counter(f['candidate_source_counts'])for f in frames),Counter())))
    write(output/'INPUT_BINDING.json', result)
    return result
