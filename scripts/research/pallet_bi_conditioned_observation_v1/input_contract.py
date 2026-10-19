"""Read-only independent audit of frozen RGB/N0/state/mask ancestry.

Only synthetic target coordinates are decoded. Real reference files are hashed
as opaque bytes. This module runs no network, local observation fit, or PnP.
The full report is private because it contains original RGB paths and IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def cuboid(extents):
    a, b, c = np.asarray(extents, float) / 2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def project(xyz, R, t, K):
    camera = np.asarray(xyz) @ np.asarray(R).T + np.asarray(t)
    homogeneous = camera @ np.asarray(K).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def audit_domain(prior, domain, frames):
    base = prior / ('d1_' + domain + '_observations_v1')
    seal = read(base / 'OBSERVATION_SEAL.json')
    upstream = prior / 'd1_upstream_v1' / (
        'REAL238_UPSTREAM.jsonl.gz' if domain == 'real' else 'SYNTHETIC1003_UPSTREAM.jsonl.gz')
    assert seal['input_sha256'] == sha(upstream)
    assert seal['ordered_ids'] == [f['id'] for f in frames]
    assert seal['complete'] and seal['rows'] == len(frames)
    state_counts, checks = Counter(), Counter()
    source_variants, inventory = defaultdict(int), []
    max_projection_gap = 0.
    partial_count = 0
    for index, frame in enumerate(frames):
        folder = base / 'frames' / f'{index:04d}'
        fs = read(folder / 'OBSERVATION_SEAL.json')
        assert sha(folder / 'OBSERVATION_SEAL.json') == seal['frame_seals_sha256'][f'{index:04d}']
        assert fs['complete'] and fs['id'] == frame['id']
        assert fs['input_row_sha256'] == hashlib.sha256(canonical(frame).encode()).hexdigest()
        assert fs['reference_reads'] == 0 and fs['e0_executed'] is False
        assert sha(folder / 'OBSERVATION.jsonl.gz') == fs['observation_sha256']
        assert sha(folder / 'MASK.npz') == fs['mask_sha256']
        observation = next(rows(folder / 'OBSERVATION.jsonl.gz'))
        source = observation['_original_source']
        geometry = observation['runtime']['initial_geometry']
        assert observation['id'] == frame['id']
        assert source['original_coarse_pose'] == frame['initial_pose']
        assert geometry['K'] == frame['K'] and geometry['native_xyz'] == frame['native_xyz']
        assert geometry['image_hw'] == frame['raw_hw']
        for name in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
            assert geometry[name] == frame['initial_pose'][name]
        q = np.array(frame['initial_points'])
        hidden = np.array(geometry['initial_self_hidden'], bool)
        expected = q.copy(); expected[:8][hidden] = [-1., -1.]
        assert np.array_equal(expected, source['original_raw_points'])
        assert observation['states'] == source['states']
        assert np.array_equal(hidden, [s == 'S' for s in observation['states']])
        assert np.array_equal(cuboid(frame['initial_pose']['cf_extents']), frame['native_xyz'])
        projection = project(frame['native_xyz'], geometry['R_cf'], geometry['centroid'], frame['K'])
        gap = float(np.max(np.abs(projection - geometry['initial_projection_xy'])))
        max_projection_gap = max(max_projection_gap, gap)
        assert gap < 1e-9
        for k in source['mask_audit']['prompt_native_ids']:
            assert not hidden[k]
        assert np.array_equal(q[source['mask_audit']['prompt_native_ids']],
                              source['mask_audit']['prompt_points'])
        for record in observation['correction']['records']:
            if observation['states'][record['native_corner']] not in ('B', 'I'):
                assert not record.get('pixel_queried', False)
        assert Path(frame['image']).is_file()
        assert sha(frame['image']) == frame['image_sha256'] == observation['image_sha256']
        with np.load(folder / 'MASK.npz', allow_pickle=False) as masks:
            keys = masks.files
            assert {'pallet_mask_tight', 'pallet_mask_clean', 'state_pallet_mask'} <= set(keys)
            tight = masks['pallet_mask_tight']
            assert list(tight.shape) == frame['raw_hw']
            assert np.isfinite(tight).all() and tight.min() >= 0 and tight.max() <= 1
        # FACE was measured using this N0 geometry before its old E0 solve.
        # Decode the prediction-only E_RUN archive solely to inspect FACE input.
        cfolder = prior / ('d1_' + domain + '_candidates_v1') / 'frames' / f'{index:04d}'
        cs = read(cfolder / 'FRAME_SEAL.json')
        assert cs['id'] == frame['id']
        assert cs['observation_seal_sha256'] == sha(folder / 'OBSERVATION_SEAL.json')
        assert cs['e_run_sha256'] == sha(cfolder / 'E_RUN.jsonl.gz')
        erun = next(rows(cfolder / 'E_RUN.jsonl.gz'))
        assert erun['id'] == frame['id'] and erun['fallback'] == frame['initial_pose']
        face = erun['face_observation']
        assert face['GT_inputs'] is False
        for item in face['boundary_observations']:
            assert item['image_binding']['sha256'] == frame['image_sha256']
            assert observation['states'][item['native_corner']] in ('B', 'I')
        partial_count += len(face['boundary_observations'])
        source_variants[hashlib.sha256(canonical(fs['source_sha256']).encode()).hexdigest()] += 1
        state_counts.update(observation['states'])
        for key in ('N0_pose_exact', 'S_gated_N0_points_exact', 'geometry_exact',
                    'input_row_digest_exact', 'observation_digest_exact',
                    'mask_digest_exact', 'RGB_digest_exact', 'FACE_N0_binding_exact'):
            checks[key] += 1
        inventory.append(dict(id=frame['id'], partition=frame['partition'], split=frame['split'],
            image=frame['image'], image_sha256=frame['image_sha256'], image_exists=True,
            raw_hw=frame['raw_hw'], observation_folder=str(folder), mask_keys=keys,
            initial_pose=frame['initial_pose'], K=frame['K'], native_xyz=frame['native_xyz'],
            states=observation['states'], FACE_partial_count=len(face['boundary_observations'])))
        if index % 128 == 0:
            print(json.dumps(dict(domain=domain, verified=index+1, total=len(frames))), flush=True)
    return dict(rows=len(frames), partitions=dict(Counter(f['partition'] for f in frames)),
        splits=dict(Counter(str(f['split']) for f in frames)), checks=dict(checks),
        state_counts=dict(state_counts), FACE_partial_records=partial_count,
        max_initial_projection_recomputation_gap_px=max_projection_gap,
        source_binding_variants=dict(source_variants), inventory=inventory,
        observation_seal_sha256=sha(base / 'OBSERVATION_SEAL.json'))


def audit_synthetic(synthetic, frames):
    targets = list(rows(synthetic / 'SYNTH_GT_LABELS.jsonl.gz'))
    lock = read(synthetic / 'DATA_SPLIT_LOCK.json')
    assert [r['id'] for r in targets] == [r['id'] for r in frames] == lock['expansion']['ids']
    assert len(targets) == 1003
    provision = read(synthetic / 'PROVISION_SEAL.json')
    assert sha(synthetic / 'SYNTH_GT_LABELS.jsonl.gz') == provision['artifacts']['SYNTH_GT_LABELS.jsonl.gz']
    assets, shapes, symmetry, mappings = Counter(), Counter(), Counter(), Counter()
    maximum_projection_gap, maximum_body_gap, maximum_rotation_gap = 0., 0., 0.
    maximum_mapped_body_projection_gap = 0.
    for frame, target in zip(frames, targets):
        assert frame['split'] == target['split']
        assert frame['K'] == target['K'] and frame['raw_hw'] == target['raw_hw']
        assert target['symmetry_order'] in (1, 2)
        assert np.allclose(frame['known_physical_dimensions_xyz_m'], target['physical_dimensions_xyz_m'], atol=1e-12, rtol=0)
        pose = frame['initial_pose']
        Q = np.asarray(pose['R_cf']).T @ np.asarray(pose['R_physical'])
        body = np.asarray(frame['native_xyz']) @ Q
        gtxyz = np.asarray(target['native_GT_xyz'])
        distance = np.linalg.norm(body[:, None] - gtxyz[None], axis=-1)
        ids = distance.argmin(1)
        assert len(set(ids.tolist())) == 8
        bodygap = float(distance[np.arange(8), ids].max())
        maximum_body_gap = max(maximum_body_gap, bodygap)
        assert bodygap < 1e-6
        mappings[str(ids.tolist())] += 1
        body_projection = project(body, target['R_fixed'], target['t'], target['K'])
        mapped_annotation = np.asarray(target['projected_GT_native_xy'])[ids]
        mapped_gap = float(np.linalg.norm(body_projection-mapped_annotation, axis=-1).max())
        maximum_mapped_body_projection_gap = max(maximum_mapped_body_projection_gap, mapped_gap)
        assert mapped_gap < .05
        projection = project(gtxyz, target['R_fixed'], target['t'], target['K'])
        maximum_projection_gap = max(maximum_projection_gap, float(np.linalg.norm(projection-target['projected_GT_native_xy'],axis=-1).max()))
        R = np.asarray(target['R_fixed'])
        maximum_rotation_gap = max(maximum_rotation_gap, float(np.abs(R.T@R-np.eye(3)).max()))
        assets[(target['source_asset'], target['split'])] += 1
        symmetry[(target['split'], target['symmetry_order'])] += 1
        shapes[tuple(frame['raw_hw'])] += 1
    assert maximum_projection_gap < .05
    return dict(rows=len(targets), source=str(synthetic / 'SYNTH_GT_LABELS.jsonl.gz'),
        sha256=sha(synthetic / 'SYNTH_GT_LABELS.jsonl.gz'), split_lock_sha256=sha(synthetic / 'DATA_SPLIT_LOCK.json'),
        asset_split_counts=[dict(asset=a, split=s, n=n) for (a,s),n in sorted(assets.items())],
        symmetry_counts=[dict(split=s,order=o,n=n) for (s,o),n in sorted(symmetry.items())],
        raw_shape_counts={str(k):v for k,v in shapes.items()}, nativeGT_body_bijections=dict(mappings),
        maximum_body_vertex_mapping_gap_m=maximum_body_gap,
        maximum_body_projection_vs_same_bijection_annotation_gap_px=maximum_mapped_body_projection_gap,
        maximum_original_projection_vs_annotation_gap_px=maximum_projection_gap,
        original_renderer_R_orthogonality_gap=maximum_rotation_gap, renderer_R_modified=False,
        source_role='virtual registered cuboid corner reference; no physical-material-edge certification',
        native_rule='physical body row coordinates = prediction native_xyz @ (R_cf.T @ R_physical)',
        target_rule='Project physical body vertices with ORIGINAL R_fixed @ whole-object allowed Y symmetry and t; choose one correspondence using all finite original N0 first-eight queries, never each corner separately',
        symmetry_rule='C1 identity; C2 identity or diag(-1,1,-1), no 90-degree rectangular symmetry',
        origin='registered cuboid geometric center', translation_and_dimensions_units='metres', pixel_units='raw original RGB pixels',
        train_assets=lock['train_assets'], validation_assets=lock['validation_assets'],
        asset_disjoint=not bool(set(lock['train_assets']) & set(lock['validation_assets'])),
        original_selection_rule=lock['rule'], source_reference_fields=list(targets[0]))


def run(args):
    start, cpu = time.monotonic(), time.process_time()
    prior, synthetic, source_root = map(Path, (args.prior_d1,args.synthetic,args.source_root))
    upstream = prior / 'd1_upstream_v1'
    us = read(upstream / 'UPSTREAM_SEAL.json')
    for name, expected in us['artifacts'].items():
        assert sha(upstream/name) == expected
    real = list(rows(upstream/'REAL238_UPSTREAM.jsonl.gz'))
    synth = list(rows(upstream/'SYNTHETIC1003_UPSTREAM.jsonl.gz'))
    assert len(real) == 238 and len(synth) == 1003
    assert Counter(f['partition'] for f in real) == Counter(CAL=46, TEST=49, REMAINDER=143)
    assert Counter(f['split'] for f in synth) == Counter(train=768, validation=235)
    history = {}
    for name, path in [('OBS', args.history_obs), ('RISK', args.history_risk)]:
        p = Path(path)
        predictions = list(rows(p/'PREDICTIONS.jsonl.gz'))
        assert [r['id'] for r in predictions] == [f['id'] for f in real]
        assert all(r['predictions']['N0']['actual_pose'] == f['initial_pose'] for r,f in zip(predictions, real))
        es = read(p/'evaluation_v1/EVALUATION_SEAL.json')
        history[name] = dict(rows=238, N0_actual_pose_exact_rows=238,
            predictions_sha256=sha(p/'PREDICTIONS.jsonl.gz'), fixed_reference_sha256=es['reference_sha256'],
            real_metrics_recomputed=False)
    real_report = audit_domain(prior,'real',real)
    synth_report = audit_domain(prior,'synthetic',synth)
    target_report = audit_synthetic(synthetic,synth)
    current = Path(args.real_reference)
    current_digest = sha(current)
    assert all(h['fixed_reference_sha256'] == current_digest for h in history.values())
    # Historical reference metadata, with no coordinate or pose-value decode.
    relative_refs = [
        '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/OOF_pose_v1/ISOLATED_CAL_SCORING.json',
        '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/test_reused_v1/ISOLATED_TEST_SCORING.json',
        '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1/derived_reference_score_v4_14/PLAN.json']
    reference_bytes = {r:dict(sha256=sha(source_root/r),bytes=(source_root/r).stat().st_size) for r in relative_refs}
    base = source_root/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2'
    state = read(base/'state_visibility/bounded_ownership_spatial_v3/MODEL.json')
    covariance = read(base/'nonsevere_v4_fix/calibration/final_adapter_CAL46/MODEL.json')
    report = dict(schema='BI_CONDITIONED_OBSERVATION_P0_INPUT_CONTRACT_V1',status='PASS',
        real=real_report, synthetic=synth_report, synthetic_targets=target_report, historical_N0=history,
        real_reference=dict(path=str(current),sha256=current_digest,decoded=False,old_minimal_separate_reference_files=reference_bytes,
            reference_conventions_identical_claimed=False,
            reason='Current OBS/RISK canonical fixed reference is the scoring target. Old minimal used three older roots; differing CAL summaries must not be treated as changed N0 poses.'),
        inherited_supervision=dict(state_classifier=dict(scope=state['train_scope'],target=state['target'],DIRECT=state['DIRECT'],EXTERNAL=state['EXTERNAL'],
            frozen_ImageNet_feature_backbone=True,existing_CAL_segmenter=True),
            covariance=dict(fit_partition=covariance['fit_partition'],fit_ids_n=len(covariance['fit_ids']),compatible_n=len(covariance['compatible_fit_ids']),
                excluded_ids=covariance['incompatible_ids'],recipe=covariance['recipe'],group_counts={k:v['count'] for k,v in covariance['groups'].items()}),
            newly_added_real_supervision=0),
        physical_oracles=dict(real_independent_BI_state_available=False, real_physical_line_available=False,
            existing_CAL_human_labels='DIRECT_VISIBLE/EXTERNAL_OCCLUDED/SELF_OCCLUDED, not B/I',
            direct_point_annotations='Existing isolated CAL/TEST targets may contain human DIRECT clicks; numeric decode and eligibility only after prediction seal',
            synthetic_state_reference='GEOMETRIC_ROLE_REFERENCE only',synthetic_point_reference='PROJECTED_CORNER_REFERENCE_DIAGNOSTIC only',
            evidence_source='pallet_minimal_bi_ablation_v1/postseal_diagnostics.py: human_boundary_vs_interior_ground_truth NA'),
        operations=dict(network_forwards=0,new_local_line_fits=0,new_PnP=0,real_reference_numeric_decodes=0,real_new_metric_calculations=0),
        cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu),source_sha256=sha(__file__))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],real=238,synthetic=1003,output=str(output),cost=report['cost'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('prior-d1','synthetic','history-obs','history-risk','source-root','real-reference','output'):
        parser.add_argument('--'+field,required=True)
    run(parser.parse_args())
