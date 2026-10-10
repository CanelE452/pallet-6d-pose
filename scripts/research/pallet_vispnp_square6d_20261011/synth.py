"""A1: existing1985 RGB/predictions -> sealed coordinates/masks -> actual F.

No model is constructed or forwarded. Prepared RGB is cropped by the existing
100px reflection border. All pose reference fields are accessed only after
the complete coordinates/masks file and its immutable seal have been saved.
"""
from collections import Counter
import copy
import time

import cv2
import numpy as np

from . import common as C
from .adapter import correspondence_mask
from .visibility import visibility
from .statistics import aggregate
from . import verdict as V

METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
N = 1985
LINE = 'data/pallet/results/pallet_line_pose_v1'
RAW = 'data/pallet/results/pallet_dim_conditioned_p_v1'
DOC = '_docs/experiments/pallet_dim_conditioned_p_v1'
GEOMETRY = 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'


def _file(path):
    return dict(path=str(path.relative_to(C.SOURCE)), sha256=C.sha(path), bytes=path.stat().st_size)


def _unchanged_candidates(base, refined):
    assert base['selected_index'] == refined['selected_index']
    assert len(base['candidates']) == len(refined['candidates'])
    selected = base['selected_index']
    for i, (b, n) in enumerate(zip(base['candidates'], refined['candidates'])):
        assert C.digest({k: v for k, v in b.items() if k != 'keypoints_xy'}) == C.digest(
            {k: v for k, v in n.items() if k != 'keypoints_xy'})
        if i != selected:
            assert C.digest(b) == C.digest(n)
    assert refined['candidates'][selected]['keypoints_xy'][8] == base['candidates'][selected]['keypoints_xy'][8]


def _same_metric(old, new, tolerance=1e-7):
    assert old['id'] == new['id'] and old['available'] == new['available']
    if old['available']:
        for key in ('translation_cm', 'rotation_deg', 'yaw_deg', 'IoU3D', 'ADDsym_m', 'ADDsym_normalized'):
            assert abs(old[key] - new[key]) <= tolerance, (old['id'], key, old[key], new[key])


def _gate():
    assert C.read_json(C.DOC/'A0.json')['status'] == 'PASS', 'A0 prerequisite failed'
    method = C.read_json(C.DOC/'METHOD_LOCK.json')
    assert method['status'].startswith('LOCKED')
    assert method['primary_definitions']['confusion'] == 'rotation_deg > 45 and abs(yaw_deg) >= 60'
    assert method['primary_definitions']['success'] == 'translation_cm < 5 and rotation_deg < 5'
    assert method['bootstrap']['draws'] == V.BOOTSTRAP_DRAWS
    assert method['bootstrap']['seed'] == V.BOOTSTRAP_SEED
    source_lock = C.read_json(C.DOC/'SOURCE_LOCK.json')
    for binding in source_lock['new_core']:
        assert C.sha(C.ROOT/binding['path']) == binding['sha256'], ('Code changed after source lock', binding['path'])
    outputs = ['SYNTH_COORDINATES_AND_MASKS.jsonl.gz', 'SYNTH_COORDINATE_SEAL.json',
               'SYNTH_ALL.jsonl.gz', 'SYNTH_PREDICTIONS.jsonl.gz', 'SYNTH_METRICS.json',
               'SYNTH_PAIRED.json', 'SYNTH_FAILURES.json', 'SYNTH_VERDICT.json', 'SYNTH_EXECUTION.json']
    assert not any((C.DOC/name).exists() for name in outputs), 'Preserve an existing execution'


def run():
    _gate()
    started = time.monotonic()
    cv2.setNumThreads(1)
    # This resolves historical definitions only. legacy() does not call load_real
    # and does not open real targets or instantiate a synthetic feature bank.
    _, correct, cap_points = C.existing()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy
    E, _ = legacy()
    source_manifest_path = C.SOURCE/LINE/'SOURCE_MANIFEST.json'
    manifest = C.read_json(source_manifest_path)
    source_rows = [r for r in manifest['records'] if r['partition'] == 'heldout']
    ids = [r['id'] for r in source_rows]
    assert len(ids) == len(set(ids)) == N
    assert len({r['scenario_id'] for r in source_rows}) == 1505
    assert Counter(r['source'] for r in source_rows) == {'G38': 1025, 'P0': 480, 'TEX': 480}
    cache_manifest_path = C.SOURCE/LINE/'cache/CACHE_MANIFEST.json'
    cache_manifest = C.read_json(cache_manifest_path)
    assert cache_manifest['source_manifest_sha256'] == C.sha(source_manifest_path)
    cache_rows = {manifest['records'][j]['id']: i for i, j in enumerate(cache_manifest['record_indices'])}
    audit_path = C.SOURCE/DOC/'SYNTH_DETECTION_AUDIT.json'
    audit = C.read_json(audit_path)
    assert audit['complete'] and audit['frames'] == N
    bound_base = {b['path']: b['sha256'] for b in audit['files']}
    result_path = C.SOURCE/DOC/'SYNTH_HELDOUT_RESULTS.json'
    result = C.read_json(result_path)
    assert result['complete']
    n3, bindings = {}, [_file(p) for p in (source_manifest_path, cache_manifest_path, audit_path, result_path,
                                          C.SOURCE/GEOMETRY, C.SOURCE/RAW/'DIMENSION_SIDECAR.npz')]
    for seed in V.SEEDS:
        bound = result['predictions'][f'N3_DIM_SYM_seed{seed}']
        path = C.SOURCE/bound['path']
        assert C.sha(path) == bound['sha256']
        packet = C.read_json(path)
        assert packet['complete'] and not packet['GT_input']
        assert [r['id'] for r in packet['records']] == ids
        n3[seed] = {r['id']: r for r in packet['records']}
        bindings.append(_file(path))
    geometry = np.load(C.SOURCE/GEOMETRY, allow_pickle=False)
    geometry_index = {str(s): i for i, s in enumerate(geometry['stems'])}
    # Explicit field list: R/t/reference/permutations are not accessed here.
    K_array, dims_array, pads = geometry['K'], geometry['dims'], geometry['pad']
    support_cache = np.load(C.SOURCE/LINE/'cache/point_valid.npy', mmap_mode='r')
    locked_rows = []
    subpix_corners = Counter()
    image_bindings = []
    for j, record in enumerate(source_rows):
        fid = record['id']
        row_index = cache_rows[fid]
        base_path = C.SOURCE/RAW/f'source_baseline/{row_index:05d}.json'
        assert C.sha(base_path) == bound_base[str(base_path.relative_to(C.SOURCE))]
        base = C.read_json(base_path)
        assert base['id'] == fid and base['cache_adapter_exact'] and not base['GT_input']
        assert base['selected_index'] is not None
        q0 = np.asarray(base['candidates'][base['selected_index']]['keypoints_xy'], float)
        support = np.array(support_cache[row_index], bool)
        assert q0.shape == (9, 2) and support.shape == (9,) and support.all()
        assert np.isfinite(q0).all() and not (q0 == -1).all(-1).any()
        image = C.SOURCE/record['image'] if not str(record['image']).startswith('/') else record['image']
        assert C.sha(image) == record['image_sha256']
        prepared = cv2.imread(str(image), cv2.IMREAD_COLOR)
        assert prepared is not None and list(prepared.shape[:2]) == record['prepared_shape_hw']
        pad = int(record['reflect_pad_px'])
        assert pad == 100
        raw = prepared[pad:-pad, pad:-pad]
        assert list(raw.shape[:2]) == record['raw_shape_hw']
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        h, w = raw.shape[:2]
        gi = geometry_index[fid]
        fx, fy, cx, cy = K_array[gi]
        assert pads[gi] == pad
        K = np.array([[fx, 0, cx-pad], [0, fy, cy-pad], [0, 0, 1]], float)
        xyz = dims_array[gi]
        assert np.isfinite(K).all() and np.isfinite(xyz).all() and (xyz > 0).all()
        qSub_native, sub_diag = correct(gray, q0, support, 'SUBPIX')
        qSub = cap_points(q0, qSub_native, w, h, support)
        subpix_corners['SUBPIX'] += sub_diag['algorithm_corner_calls']
        for seed in V.SEEDS:
            refined = n3[seed][fid]
            _unchanged_candidates(base, refined)
            qN = np.asarray(refined['candidates'][refined['selected_index']]['keypoints_xy'], float)
            qS, seq_diag = correct(gray, qN, support, 'SUBPIX')
            qFinal = cap_points(q0, qS, w, h, support)
            subpix_corners[f'N3_THEN_SUBPIX_seed{seed}'] += seq_diag['algorithm_corner_calls']
            for method, q, native, diag in [('BASE', q0, q0, None), ('N3_DIM_SYM', qN, qN, None),
                                           ('SUBPIX', qSub, qSub_native, sub_diag),
                                           ('N3_THEN_SUBPIX', qFinal, qS, seq_diag)]:
                assert np.array_equal(q[8], q0[8])
                if method in ('SUBPIX', 'N3_THEN_SUBPIX'):
                    assert np.max(np.linalg.norm(q[:8]-q0[:8], axis=-1)) <= .01*np.hypot(w, h)+1e-10
                vis = visibility(q, support[:8])
                locked_rows.append(dict(id=fid, seed=seed, method=method, session=record['scenario_id'],
                    source=record['source'], q0=q0, qN=qN if method in ('N3_DIM_SYM', 'N3_THEN_SUBPIX') else q0,
                    qS=native, qFinal=q, prediction_support=support, raw_hw=[h, w], visibility=vis,
                    fixed_metadata=dict(K=K, dimensions_pnp_WH_D_m=xyz,
                        selected_index=base['selected_index'],
                        candidate_metadata=[{k:v for k,v in c.items() if k != 'keypoints_xy'} for c in base['candidates']],
                        center_preserved=True, source_basis_transform='fixed Rx(pi) after F'),
                    correction=dict(diagnostics=diag, cap_px=.01*np.hypot(w,h),
                        cap_active8=(np.linalg.norm(native[:8]-q0[:8],axis=-1) > .01*np.hypot(w,h)).tolist(),
                        total_final_px8=np.linalg.norm(q[:8]-q0[:8], axis=-1))))
        image_bindings.append(dict(id=fid, sha256=record['image_sha256']))
        if j % 100 == 0 or j+1 == N:
            print('SYNTH_COORDINATES', j+1, N, 'seconds', round(time.monotonic()-started, 1), flush=True)
    coordinate_path = C.DOC/'SYNTH_COORDINATES_AND_MASKS.jsonl.gz'
    C.write_rows(coordinate_path, locked_rows)
    seal = dict(status='SEALED_BEFORE_POSE_REFERENCES', rows=len(locked_rows), frames=N, seeds=list(V.SEEDS),
        methods=list(METHODS), coordinates_sha256=C.sha(coordinate_path),
        method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'), A0_sha256=C.sha(C.DOC/'A0.json'),
        input_bindings=bindings, baseline_manifest_sha256=C.sha(audit_path),
        image_bindings=image_bindings, code=[dict(path=str(p.relative_to(C.ROOT)),sha256=C.sha(p))
            for p in (C.ROOT/'scripts/research/pallet_vispnp_square6d_20261011/synth.py',
                      C.ROOT/'scripts/research/pallet_vispnp_square6d_20261011/statistics.py',
                      C.ROOT/'scripts/research/pallet_vispnp_square6d_20261011/verdict.py')],
        primary_rules=V.rules(), accessed_geometry_fields_before_seal=['stems','K','dims','pad'],
        pose_reference_fields_accessed_before_seal=False, human_visibility_used=False,
        detector_calls=0, N3_forwards=0, training_updates=0, generated_synthetic_images=0)
    seal_path = C.DOC/'SYNTH_COORDINATE_SEAL.json'
    C.write_json(seal_path, seal)
    assert C.sha(coordinate_path) == seal['coordinates_sha256']
    seal_sha = C.sha(seal_path)
    # References are first accessed after both complete sealed artifacts exist.
    reference_R, reference_t = geometry['R'], geometry['t']
    side = np.load(C.SOURCE/RAW/'DIMENSION_SIDECAR.npz', allow_pickle=False)
    matched = np.load(C.SOURCE/LINE/'cache/matched.npy', mmap_mode='r')
    source_by_id = {r['id']:r for r in source_rows}
    # Existing N3 ALL metrics are an additional immutable arithmetic parity audit.
    old_metrics_path = C.SOURCE/RAW/'PAPER_POSE_METRICS.json'
    old_metrics = C.read_json(old_metrics_path)['SYNTH_HELDOUT']
    old_n3 = {s:{r['id']:r for r in old_metrics[f'N3_DIM_SYM_seed{s}']} for s in V.SEEDS}
    all_rows, vis_rows = [], []
    calls = Counter()
    pairs_complete = 0
    for locked in locked_rows:
        fid, seed, method = locked['id'], locked['seed'], locked['method']
        record = source_by_id[fid]
        gi, ci = geometry_index[fid], cache_rows[fid]
        xyz = dims_array[gi]
        order = int(side['order'][ci])
        truth = dict(R=reference_R[gi], t=reference_t[gi], xyz=xyz,
                     body_R=reference_R[gi], body_xyz=xyz, order=order)
        keypoints = np.asarray(record['targets'][0]['keypoints_normalized'])
        target = keypoints[:,:2]*np.asarray(record['prepared_shape_hw'])[::-1]-100
        valid = keypoints[:,2] > 0
        permutations = side['permutations'][ci,:order]
        q = np.asarray(locked['qFinal'], float)
        K = np.asarray(locked['fixed_metadata']['K'], float)
        corner = E.M.measure(q, target, valid, permutations, locked['raw_hw'], bool(matched[ci]), True)
        corner.update(id=fid, session=locked['session'])
        for variant, mask, destination in [('ALL', np.ones(8,bool), all_rows),
                                            ('VIS', locked['visibility']['effective_mask'], vis_rows)]:
            before = time.monotonic()
            with correspondence_mask(q, K, mask) as audit_calls:
                actual = E.POSE.infer(q, K, xyz, True)
            pose = E.POSE.metric((fid, actual, truth))
            calls.update({k:audit_calls[k] for k in ('solvePnP','solvePnPRefineLM')})
            row = copy.deepcopy(locked)
            row.update(method=method if variant == 'ALL' else method+'_VIS', variant=variant,
                actual_pose=actual, pose=pose, corner=corner, F_attempt=True, F_complete=True,
                F_seconds=time.monotonic()-before, PnP_counts=audit_calls,
                final_hypothesis=actual.get('selected_hypothesis'),
                evaluation_reference_points=target, evaluation_reference_valid=valid,
                evaluation_reference_used_in_inference=False, canonical_symmetry_order=order,
                reference_seal_sha256=seal_sha)
            if variant == 'ALL' and method == 'N3_DIM_SYM':
                _same_metric(old_n3[seed][fid], pose)
            destination.append(row)
        assert all_rows[-1]['corner'] == vis_rows[-1]['corner']
        pairs_complete += 1
        if pairs_complete % 100 == 0 or pairs_complete == len(locked_rows):
            C.write_json(C.DOC/'SYNTH_EXECUTION.json', dict(status='RUNNING', pairs_complete=pairs_complete,
                pairs_expected=len(locked_rows), actual_F_calls=pairs_complete*2,
                elapsed_seconds=time.monotonic()-started, PnP_counts=dict(calls)))
            print('SYNTH_F', pairs_complete, len(locked_rows), 'seconds', round(time.monotonic()-started,1), flush=True)
    C.write_rows(C.DOC/'SYNTH_ALL.jsonl.gz', all_rows)
    C.write_rows(C.DOC/'SYNTH_PREDICTIONS.jsonl.gz', vis_rows)
    # Serialize first, then aggregate exactly the public numeric rows.
    all_rows = list(C.rows(C.DOC/'SYNTH_ALL.jsonl.gz'))
    vis_rows = list(C.rows(C.DOC/'SYNTH_PREDICTIONS.jsonl.gz'))
    scopes = {'ALL': ids}
    for source in ('G38','P0','TEX'):
        scopes[source] = [r['id'] for r in source_rows if r['source'] == source]
    for order in (1,2):
        scopes[f'C{order}'] = [fid for fid in ids if int(side['order'][cache_rows[fid]]) == order]
    print('SYNTH_STATISTICS_BEGIN', flush=True)
    metrics, paired, failures = aggregate(all_rows, vis_rows, population='SYNTH_HELDOUT',
                                         bootstrap_level='frame', scopes=scopes)
    conclusion = V.evaluate(paired, phase='A1')
    for name, payload in [('METRICS',metrics),('PAIRED',paired),('FAILURES',failures),('VERDICT',conclusion)]:
        C.write_json(C.DOC/f'SYNTH_{name}.json', payload)
    assert C.sha(coordinate_path) == seal['coordinates_sha256']
    assert all(C.sha(C.SOURCE/b['path']) == b['sha256'] for b in bindings)
    assert all(C.sha(C.SOURCE/b['path']) == b['sha256'] for b in audit['files'])
    assert all(C.sha(r['image']) == r['image_sha256'] for r in source_rows)
    C.write_json(C.DOC/'SYNTH_EXECUTION.json', dict(status='COMPLETE', frames=N, seeds=list(V.SEEDS),
        methods=list(METHODS), ALL_rows=len(all_rows), VIS_rows=len(vis_rows), actual_F_calls=pairs_complete*2,
        coordinates_sealed_before_reference=True, corner_metrics_ALL_VIS_exact=True,
        N3_ALL_existing_metric_parity=True, N3_ALL_metric_absolute_tolerance=1e-7,
        baseline_and_subpix_coordinates_shared_but_F_evaluated_each_seed=True,
        PnP_counts=dict(calls), subpix_corner_calls=dict(subpix_corners),
        detector_calls=0, N3_forwards=0, training_updates=0, new_synthetic_images=0,
        original_source_bindings_unchanged=True, elapsed_seconds=time.monotonic()-started,
        ALL_sha256=C.sha(C.DOC/'SYNTH_ALL.jsonl.gz'), VIS_sha256=C.sha(C.DOC/'SYNTH_PREDICTIONS.jsonl.gz'),
        verdict=conclusion['verdict'], continue_to_A2=conclusion['continue_to_A2']))
    print('SYNTH_COMPLETE', conclusion['verdict'], 'continue_to_A2', conclusion['continue_to_A2'], flush=True)
    return conclusion


if __name__ == '__main__':
    run()
