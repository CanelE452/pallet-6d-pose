"""Audit existing CAL RGB stencils against cached FP16 channels 0:3 only.

No original model imports are performed. Two small NumPy/OpenCV AST fragments
are extracted from the bound original model.py, excluding poses and necks.
Freeze the protocol first, then run one check. Existing receipts are preserved.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_corner_mechanism_audit_20261010_v1'
OLD = REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1'
REPAIR = REPO / '_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
PRIVATE = Path('/dev/shm/pallet-observation-private-20261009/learned_cache')
INDICES = list(range(768, 896))
CHANNELS = ['brightness_gray_times2_minus1', 'Sobel_gradient_dot_normal', 'Sobel_gradient_dot_tangent']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve()
    public = p.is_relative_to(REPO)
    return dict(path=str(p.relative_to(REPO)) if public else p.name,
                origin='public_repository' if public else 'private_readonly_dependency',
                sha256=sha(p), bytes=p.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def paths(args):
    return dict(code=Path(__file__),
                original_model=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/model.py',
                original_edge_definition=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/source_audit.py',
                original_cache_preparation=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/training.py',
                source_family_split=OLD / 'SOURCE_FAMILY_SPLIT.json',
                preparation_receipt=OLD / 'SUPERVISION_PREPARATION.json',
                ready_rows=REPAIR / 'READY_SOURCE_TARGET_ROWS.jsonl.gz',
                previous_source_contract=DOC / 'SOURCE_CONTRACT_CHECKS.json',
                cache_manifest=Path(args.cache_manifest), features=Path(args.features))


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                historical_preparation_versions='not recorded in the original preparation receipt',
                opencv_optimized=cv2.useOptimized(), opencv_threads=cv2.getNumThreads())


def fragments(p):
    """Compile only bound array geometry and BGR stencil; no module imports."""
    edge_tree = ast.parse(p['original_edge_definition'].read_text())
    assignments = [n for n in edge_tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'EDGES' for t in n.targets)]
    assert len(assignments) == 1
    edges = ast.literal_eval(assignments[0].value)
    assert edges == [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                     (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    tree = ast.parse(p['original_model'].read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    query = funcs['query_geometry']
    body = funcs['inputs'].body
    def assigns(n, key):
        return isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in n.targets)
    first = next(i for i, n in enumerate(body) if assigns(n, 'gray'))
    last = next(i for i, n in enumerate(body) if assigns(n, 'rgb'))
    selected = body[first:last + 1]
    # Gray, gx, gy, local sample(), sx/sy, normal, tangent, rgb only.
    assert [n.name for n in selected if isinstance(n, ast.FunctionDef)] == ['sample']
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(ast.Module(body=selected, type_ignores=[])))
    prefix = ast.parse("cc=q['candidate'];x=cc[:,:,0].astype(np.float32);y=cc[:,:,1].astype(np.float32)").body
    stencil = ast.FunctionDef(name='rgb_stencil', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg='image'), ast.arg(arg='q')], vararg=None, kwonlyargs=[],
        kw_defaults=[], kwarg=None, defaults=[]), body=prefix + selected +
        [ast.Return(value=ast.Name(id='rgb', ctx=ast.Load()))], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[query, stencil], type_ignores=[]))
    spec = dict(query_geometry_ast_sha256=hashlib.sha256(ast.dump(query, include_attributes=False).encode()).hexdigest(),
                RGB_stencil_ast_sha256=hashlib.sha256(ast.dump(stencil, include_attributes=False).encode()).hexdigest(),
                query_geometry_lines=[query.lineno, query.end_lineno],
                original_RGB_stencil_lines=[body[first].lineno, body[last].end_lineno],
                edges=edges)
    namespace = dict(np=np, cv2=cv2, EDGES=edges)
    exec(compile(module, '<bound_original_NumPy_OpenCV_fragments>', 'exec'), namespace)
    return namespace['query_geometry'], namespace['rgb_stencil'], spec


def metadata(p):
    manifest = read(p['cache_manifest'])
    split = read(p['source_family_split'])
    assert manifest['complete'] and len(manifest['records']) == 1024
    assert manifest['specs']['features'] == dict(shape=[1024, 84, 28, 65], dtype='float16')
    assert split['counts'] == dict(train=768, calibration=128, source_test=128)
    with gzip.open(p['ready_rows'], 'rt') as f:
        # Only split/identity, raw shape and selected Base coordinates are used.
        ready = [json.loads(line) for line in f]
    assert len(ready) == len(split['records']) == 1024
    assert Counter(r['partition'] for r in ready) == dict(train=768, calibration=128, source_test=128)
    selected = []
    for i in INDICES:
        m, r, s = manifest['records'][i], ready[i], split['records'][i]
        assert m['index'] == r['index'] == r['feature_cache_index'] == i
        assert m['id'] == r['id'] == s['id'] and m['family'] == r['family'] == s['family']
        assert m['partition'] == r['partition'] == s['partition'] == 'calibration'
        assert m['variant'] == 0 and m['composition'] == dict(kind='existing_P0_original', generated=False)
        assert m['selected_detector_candidate'] is not None
        points = np.array(m['selected_points'], dtype=np.float64)
        assert points.shape == (9, 2) and np.isfinite(points).all()
        assert np.array_equal(points, np.array(r['frozen_selected_points'], dtype=np.float64))
        assert m['original_rgb']['origin'] == 'source' and m['original_rgb']['path'] == s['raw']['rgb']
        relative = Path(m['original_rgb']['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        selected.append(dict(index=i, id=m['id'], family=m['family'], partition='calibration',
            raw_hw=r['raw_hw'], original_rgb=m['original_rgb'], decoded_BGR_sha256=m['rgb_sha256'],
            selected_points_float64_sha256=hashlib.sha256(points.tobytes()).hexdigest()))
    assert len({r['id'] for r in selected}) == len({r['family'] for r in selected}) == 128
    return manifest, selected


def RGB_bindings(source_root, selected):
    root = Path(source_root).resolve()
    result = []
    for row in selected:
        declared = row['original_rgb']
        p = (root / declared['path']).resolve()
        assert p.is_relative_to(root), 'Original RGB must remain inside the explicit source root'
        b = dict(path=declared['path'], origin='source', sha256=sha(p), bytes=p.stat().st_size)
        assert b == declared, 'Actual original PNG differs from the cache manifest declaration'
        result.append(dict(index=row['index'], id=row['id'], original_rgb=b))
    return result


def freeze(args):
    begin = time.monotonic()
    assert not Path(args.protocol).exists() and not Path(args.output).exists(), 'Preserve existing receipts'
    cv2.setNumThreads(1)
    p = paths(args)
    before = {k: binding(v) for k, v in p.items()}
    _, selected = metadata(p)
    _, _, spec = fragments(p)
    previous = read(p['previous_source_contract'])['authoritative_inputs']
    assert before['features']['sha256'] == previous['features']['sha256']
    assert before['cache_manifest']['sha256'] == previous['cache_manifest']['sha256']
    a = np.load(p['features'], mmap_mode='r')
    assert a.shape == (1024, 84, 28, 65) and a.dtype == np.float16
    original_RGB = RGB_bindings(args.source_root, selected)
    assert before == {k: binding(v) for k, v in p.items()}
    result = dict(schema='source_CAL_RGB_channels_contract_protocol_v1', frozen_before_comparison=True,
        authoritative_inputs=before, selected_CAL=selected, original_RGB=original_RGB,
        compiled_scope=spec, runtime_versions=versions(), source_root_argument_required=True,
        population=dict(split='calibration', indices=INDICES, original_RGB_frames=128,
            selected_Base_points_shape=[9, 2], feature_shape=[1024, 84, 28, 65],
            compared_slice='features[768:896,:,0:3,:]', queries_per_frame=84,
            bins_per_query=65, channels_compared=3, compared_FP16_values=128 * 84 * 3 * 65),
        fixed_policy=dict(read='cv2.imread(original_RGB,cv2.IMREAD_COLOR); BGR uint8',
            query='exact original query_geometry AST;12 EDGES*7 u=1/8..7/8;normal=(-ty,tx);candidate=center+(bin-32)*normal',
            channels=CHANNELS,
            grayscale='cv2.COLOR_BGR2GRAY;astype(np.float32)/255',
            gradients='cv2.Sobel(gray,CV_32F,1,0,ksize=3)/4 and (0,1)/4; unchanged OpenCV default border',
            remap='INTER_LINEAR;BORDER_REFLECT_101; maps candidate x/y.astype(float32)',
            arithmetic='extract original RGB expression AST including float64 normals/tangents; no forced float32 intermediate RGB cast',
            cache_rounding='NumPy float64 RGB -> float16 nearest rounding; compared to original CUDA full-tensor FP16 rounding then original stored FP16',
            exact_definition='all uint16 FP16 bit patterns equal, including signed zero; numeric equality/max difference separately recorded',
            decision='PASS only if all shape/identity/raw-file/decoded-BGR/protected-input checks pass and every FP16 bit equals; any failure preserved, no threshold or corrective rerun',
            allowed_runs=1, save_new_RGB=False, recompute_neck=False, execute_original_model_inputs=False),
        cost=dict(protocol_freeze_calls=1, RGB_decodes_before_freeze=0,
            expected_new_original_RGB_decodes=128, expected_recomputed_RGB_stencils=128,
            new_detector_forwards=0, new_neck_forwards=0, new_head_forwards=0,
            initial_pose_calls=0, new_PnP=0, new_rays=0, optimizer_updates=0,
            new_or_modified_RGB=0, feature_cache_writes=0),
        limits=['Three brightness/gradient channels only; this does not certify all19 image/neck channels',
            'Historical OpenCV/NumPy versions are not recorded; current versions are explicit and mismatch is preserved',
            'Source CAL only; no new real245 accuracy or GT inspection',
            'Original RGB files, feature cache, models and all prior receipts remain unchanged'],
        freeze_elapsed_seconds=time.monotonic() - begin)
    write_new(args.protocol, result)
    print(json.dumps(dict(stage='protocol_frozen', binding=binding(args.protocol),
                          source_CAL_frames=128, RGB_decodes=0)))


def check(args):
    begin = time.monotonic()
    assert not Path(args.output).exists(), 'Preserve completed or failed receipt'
    cv2.setNumThreads(1)
    protocol = read(args.protocol)
    protocol_binding = binding(args.protocol)
    p = paths(args)
    actual = dict(original_RGB_decodes=0, recomputed_RGB_stencils=0, compared_FP16_values=0,
        new_detector_forwards=0, new_neck_forwards=0, new_head_forwards=0,
        initial_pose_calls=0, new_PnP=0, new_rays=0, optimizer_updates=0,
        new_or_modified_RGB=0, feature_cache_writes=0, arithmetic_audit_runs=1)
    receipt = dict(schema='source_CAL_RGB_channels_contract_checks_v1', passed=False,
        protocol=protocol_binding, actual_execution=actual, authoritative_inputs=protocol['authoritative_inputs'],
        source_root_argument_required=True, real_GT_or_pose_scores_read=False,
        source_GT_used=False, source_GT_geometry_and_target_fields_loaded_but_unused=True,
        source_metadata_scope='READY identity/split/raw_hw/selected Base only; K,R,t,targets unused',
        private_access='Original128 source PNG, immutable FP16 features.npy, immutable cache manifest; paths source-relative or basenames only',
        limits=protocol['limits'])
    try:
        before = {k: binding(v) for k, v in p.items()}
        assert before == protocol['authoritative_inputs'], 'Frozen input changed before comparison'
        manifest, selected = metadata(p)
        assert selected == protocol['selected_CAL']
        query_geometry, stencil, spec = fragments(p)
        assert spec == protocol['compiled_scope'] and versions() == protocol['runtime_versions']
        RGB_before = RGB_bindings(args.source_root, selected)
        assert RGB_before == protocol['original_RGB']
        features = np.load(p['features'], mmap_mode='r')
        assert features.shape == (1024, 84, 28, 65) and features.dtype == np.float16
        channel = [dict(channel=i, name=name, n=0, bit_mismatch=0, numeric_mismatch=0,
            signed_zero_only_mismatch=0, max_abs_diff=0.) for i, name in enumerate(CHANNELS)]
        rows = []
        examples = []
        for row in selected:
            i = row['index']
            image = cv2.imread(str(Path(args.source_root) / row['original_rgb']['path']), cv2.IMREAD_COLOR)
            actual['original_RGB_decodes'] += 1
            assert image is not None and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
            assert list(image.shape[:2]) == row['raw_hw']
            assert hashlib.sha256(image.tobytes()).hexdigest() == row['decoded_BGR_sha256']
            points = np.array(manifest['records'][i]['selected_points'], dtype=np.float64)
            q = query_geometry(points)
            assert q['candidate'].shape == (84, 65, 2) and q['normal'].shape == q['tangent'].shape == (84, 2)
            assert len(q['identity']) == 84
            assert all(ident[0] == j // 7 and ident[3] == (j % 7 + 1) / 8 for j, ident in enumerate(q['identity']))
            rgb = stencil(image, q)
            actual['recomputed_RGB_stencils'] += 1
            assert rgb.shape == (84, 3, 65) and rgb.dtype == np.float64 and np.isfinite(rgb).all()
            replay = rgb.astype(np.float16)
            cached = np.asarray(features[i, :, 0:3, :])
            assert np.isfinite(replay).all() and np.isfinite(cached).all()
            bits = replay.view(np.uint16) != cached.view(np.uint16)
            numeric = replay != cached
            difference = np.abs(replay.astype(np.float64) - cached.astype(np.float64))
            actual['compared_FP16_values'] += replay.size
            for c, ch in enumerate(channel):
                ch['n'] += replay[:, c, :].size
                ch['bit_mismatch'] += int(bits[:, c, :].sum())
                ch['numeric_mismatch'] += int(numeric[:, c, :].sum())
                ch['signed_zero_only_mismatch'] += int((bits[:, c, :] & ~numeric[:, c, :]).sum())
                ch['max_abs_diff'] = max(ch['max_abs_diff'], float(difference[:, c, :].max()))
            rows.append(dict(index=i, id=row['id'], partition='calibration', raw_hw=row['raw_hw'],
                original_png_SHA_verified=True, decoded_BGR_SHA_verified=True,
                selected_Base_and_READY_identity_verified=True, shape=[84, 3, 65],
                bit_mismatch=int(bits.sum()), numeric_mismatch=int(numeric.sum()),
                max_abs_diff=float(difference.max())))
            for j, c, b in np.argwhere(bits)[:max(0, 20 - len(examples))]:
                examples.append(dict(index=i, id=row['id'], query=int(j), edge=int(j) // 7,
                    channel=int(c), bin=int(b), replay=float(replay[j, c, b]), cached=float(cached[j, c, b]),
                    replay_FP16_uint16=int(replay.view(np.uint16)[j, c, b]),
                    cached_FP16_uint16=int(cached.view(np.uint16)[j, c, b]),
                    normal=q['normal'][j].tolist(), tangent=q['tangent'][j].tolist(),
                    candidate=q['candidate'][j, b].tolist()))
        after = {k: binding(v) for k, v in p.items()}
        RGB_after = RGB_bindings(args.source_root, selected)
        assert before == after and RGB_before == RGB_after
        assert binding(args.protocol) == protocol_binding
        assert actual['original_RGB_decodes'] == actual['recomputed_RGB_stencils'] == 128
        assert actual['compared_FP16_values'] == protocol['population']['compared_FP16_values']
        receipt.update(channels=channel, rows=rows, mismatch_examples=examples,
            runtime_versions=versions(), compiled_scope=spec, inputs_before_after_equal=True,
            original_RGB_before_after_equal=True, all128_original_png_SHA_verified=True,
            all128_decoded_BGR_SHA_verified=True, all128_CAL_Base_READY_identity_verified=True,
            all128_queries84_bins65_verified=True, cached_dtype='float16', replay_dtype='float16',
            pre_round_RGB_dtype='float64', bit_exact=all(ch['bit_mismatch'] == 0 for ch in channel),
            numeric_exact=all(ch['numeric_mismatch'] == 0 for ch in channel),
            max_abs_diff=max(ch['max_abs_diff'] for ch in channel),
            channels_compared=[0, 1, 2], channels3_through18_not_recomputed=True,
            entire19_channel_parity_claimed=False)
        receipt['passed'] = receipt['bit_exact']
    except Exception as error:
        message = str(error)
        for value in [args.source_root, args.features, args.cache_manifest, str(Path.home())]:
            message = message.replace(str(value), '<private_readonly_path>')
        receipt['failure'] = dict(type=type(error).__name__, message=message)
        # Preserve actual work even on an input/implementation failure.
        receipt['failure_preserved_no_corrective_rerun'] = True
    receipt['elapsed_seconds'] = time.monotonic() - begin
    write_new(args.output, receipt)
    print(json.dumps(dict(passed=receipt['passed'], actual_execution=actual,
        channels=receipt.get('channels'), max_abs_diff=receipt.get('max_abs_diff'),
        failure=receipt.get('failure'), output=binding(args.output))))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'check'])
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--features', default=str(PRIVATE / 'features.npy'))
    parser.add_argument('--cache-manifest', default=str(PRIVATE / 'CACHE_MANIFEST.json'))
    parser.add_argument('--protocol', default=str(DOC / 'RGB_CONTRACT_PROTOCOL.json'))
    parser.add_argument('--output', default=str(DOC / 'RGB_CONTRACT_CHECKS.json'))
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else check(args)
