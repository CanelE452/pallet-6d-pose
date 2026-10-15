"""Fixed238 predictor-only RAW/N3_SUBPIX inputs from the original same-seed cache.

No reference loader, metric, state classifier, image decoder or NN is imported.
Only missing RAW F poses are newly generated. Historical mixed prediction/score
JSONL is scanned with a lexical top-level whitelist before JSON decoding.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(os.environ.get('PALLET_SOURCE_ROOT', str(REPOSITORY_ROOT))).resolve()
OUTPUT = Path('artifacts/pallet_minimal_bi_ablation_v1/inputs')
DOC_ROOT = REPOSITORY_ROOT / '_docs/experiments/pallet_minimal_bi_ablation_v1/contract'
D = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
B = D / 'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
N3 = Path('_docs/experiments/pallet_n3_subpix_20261008_v1')
OLD95 = D / 'PREDICTION_INPUTS.json'
OLD143 = D / 'remainder_after_test49_v1/native_input_preparation_v1/NATIVE_INPUTS143.jsonl.gz'
ORDER_SEAL = B / 'clean_fresh_BI_pose_eval_v4_32_r2/INPUT_SEAL.json'
GAIN10 = B / 'raw_ns_BI_gain10_input_v1/INPUTS.json'
DEV_CACHE = Path('_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json')
RECORDINGS = Path('data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
DETECTOR = Path('challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
POSE_SOURCE = Path('scripts/research/pallet_dim_conditioned_p_v1/pose.py')
SOLVE_SOURCE = Path('scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py')
SELECTOR_SOURCE = Path('challenge/evaluation_v2/pnp_selector.py')
GEOMETRY_SOURCE = Path('scripts/annotate/pallet_geometry.py')
PARSER_SOURCE = Path('scripts/research/pallet_visibility_curriculum_20261013_v1/remainder_native_inputs_v1.py')
BRANCHES = ('RAW_YOLO', 'N3_SUBPIX')
PARTITIONS = {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}
SCHEMA = 'MINIMAL_BI_ABLATION_PREDICTOR_ONLY_FIXED238_INPUTS_V1'
POSE_ATOL = 1e-9


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing artifact: ' + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _defs(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Missing exact original function definitions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def prediction_parser(source_root=SOURCE_ROOT):
    ns = {'json': json, 'ALLOWED_FIELDS': frozenset()}
    return _defs(source_root / PARSER_SOURCE, {'_skip_value', 'select_prediction_fields'}, ns)['select_prediction_fields']


def existing_F(source_root=SOURCE_ROOT):
    """Compile only original prediction functions; metric/metadata/main excluded."""
    name = '_minimal_bi_original_geometry'
    spec = importlib.util.spec_from_file_location(name, source_root / GEOMETRY_SOURCE)
    geometry = importlib.util.module_from_spec(spec)
    sys.modules[name] = geometry
    spec.loader.exec_module(geometry)
    selector = types.ModuleType('_minimal_bi_original_selector')
    selector.__file__ = str(source_root / SELECTOR_SOURCE)
    selector.__dict__['geometry'] = geometry
    sys.modules[selector.__name__] = selector
    tree = ast.parse((source_root / SELECTOR_SOURCE).read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'scripts.annotate')]
    exec(compile(tree, selector.__file__, 'exec'), selector.__dict__)
    ns = {'np': np, 'cv2': cv2, 'select_pnp_hypotheses': selector.select_pnp_hypotheses}
    _defs(source_root / SOLVE_SOURCE, {'cuboid', 'solve'}, ns)
    _defs(source_root / POSE_SOURCE, {'rotations', 'infer'}, ns)
    return ns['infer']


def assert_pose_equal(a, b, atol=POSE_ATOL):
    if a.get('available') is not b.get('available'):
        raise ValueError('Historical F availability mismatch')
    if not a.get('available'):
        return 0.0
    if a['selected_hypothesis'] != b['selected_hypothesis']:
        raise ValueError('Historical F W/D selector mismatch')
    maximum = 0.0
    for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'reprojection_px'):
        x, y = np.asarray(a[key], float), np.asarray(b[key], float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Finite matching historical pose fields required')
        maximum = max(maximum, float(np.max(np.abs(x-y))))
        if not np.allclose(x, y, atol=atol, rtol=0):
            raise ValueError('Historical F numerical mismatch: ' + key)
    return maximum


def recording_map(document):
    out = {}
    for group in document['groups']:
        for session in group['sessions']:
            key = session['session_key']
            if '/sessions/' not in key or group['is_collection']:
                continue
            name = key.rsplit('/sessions/', 1)[-1]
            old = out.setdefault(name, group['recording_id'])
            if old != group['recording_id']:
                raise ValueError('Ambiguous recording session')
    return out


def load_frames(output=OUTPUT, *, verify=True):
    output = Path(output)
    seal = read(output / 'INPUT_SEAL.json')
    if verify and (sha(output / 'INPUTS.json') != seal['inputs_sha256'] or sha(output / 'INPUTS.jsonl.gz') != seal['inputs_gzip_sha256']):
        raise ValueError('Predictor input journal SHA mismatch')
    with gzip.open(output / 'INPUTS.jsonl.gz', 'rt') as stream:
        frames = [json.loads(line) for line in stream]
    if [r['id'] for r in frames] != seal['ordered_ids'] or len(frames) != 238:
        raise ValueError('Fixed ordered238 required')
    if [digest(r) for r in frames] != seal['canonical_row_sha256']:
        raise ValueError('Canonical input row mismatch')
    return frames


def prepare_inputs(output=OUTPUT, source_root=SOURCE_ROOT, *, docs=DOC_ROOT):
    start, cpu = time.monotonic(), time.process_time()
    output, source_root, docs = Path(output), Path(source_root), Path(docs)
    if output.exists():
        raise FileExistsError('Preserve every input preparation attempt')
    cv2.setNumThreads(1)
    inputs, sources = {}, {}
    def bind(relative, expected=None):
        value = sha(source_root / relative)
        if expected is not None and value != expected:
            raise ValueError('Frozen input mismatch: ' + str(relative))
        inputs[str(relative)] = value
        return source_root / relative
    for p in (POSE_SOURCE, SOLVE_SOURCE, SELECTOR_SOURCE, GEOMETRY_SOURCE, PARSER_SOURCE):
        sources[str(p)] = sha(source_root / p)
    protocol = read(bind(N3 / 'PROTOCOL.json'))
    order = read(bind(ORDER_SEAL))['ordered_ids']
    if len(order) != 238 or len(set(order)) != 238:
        raise ValueError('Fixed original ordered238 required')
    if protocol['N3_seed'] != 1 or protocol['N3_temperature'] != 1.0:
        raise ValueError('Original N3 seed1/temperature1 only')
    metadata = {r['id']: r for r in protocol['input_manifest']}
    if not set(order) <= set(metadata):
        raise ValueError('Missing original same-execution raw/N3 metadata')
    old95 = read(bind(OLD95))
    if old95['GT_input'] is not False or old95['human_visibility_input'] is not False:
        raise ValueError('Predictor-only95 required')
    frames95 = {r['id']: r for r in old95['frames']}
    with gzip.open(bind(OLD143), 'rt') as stream:
        frames143 = {r['id']: r['frame'] for r in map(json.loads, stream)}
    if len(frames95) != 95 or len(frames143) != 143 or set(frames95) & set(frames143):
        raise ValueError('95/143 exact disjoint input population required')
    if set(order) != set(frames95) | set(frames143):
        raise ValueError('Authoritative238 mismatch')
    cache = read(bind(DEV_CACHE))
    if cache['R0_candidates_exact'] is not True:
        raise ValueError('Original detector candidate cache equality required')
    cache_by_id = {r['id']: r for r in cache['records']}
    clusters = recording_map(read(bind(RECORDINGS, 'f0be41054f8b4c955a0f004d7b6f57b6eeabd0e3ab6409086c39eb42fd4bd75a')))
    n3ckpt = protocol['N3_checkpoint']
    bind(Path(n3ckpt['path']), n3ckpt['sha256'])
    bind(DETECTOR)
    select = prediction_parser(source_root)
    predictions = {}
    journal = bind(N3 / 'PREDICTIONS.jsonl.gz', '041afb757b3e2f410ee2c10138ad114d542c5e11c7c1281ef6a18cb57425ad7c')
    with gzip.open(journal, 'rt') as stream:
        for line in stream:
            header = select(line, {'id', 'method'})
            if header.get('id') in metadata and header.get('id') in set(order) and header.get('method') in ('BASE', 'N3_SUBPIX'):
                row = select(line, {'id', 'method', 'raw_hw', 'native_points', 'qFinal', 'actual_pose'})
                key = (row['id'], row['method'])
                if key in predictions:
                    raise ValueError('Duplicate predictor row')
                predictions[key] = row
    infer = existing_F(source_root)
    result, parity, recomputed, cached = [], [], [], []
    for i, fid in enumerate(order):
        meta = metadata[fid]
        source = frames95.get(fid, frames143.get(fid))
        partition = source['partition']
        if partition != ('CAL' if i < 46 else 'TEST' if i < 95 else 'REMAINDER'):
            raise ValueError('Original partition/order mismatch')
        xyz = source.get('xyz', source.get('known_physical_dimensions_xyz_m'))
        for key in ('session', 'raw_hw', 'K', 'image_sha256', 'selected_index', 'candidate_metadata'):
            if key in source and source[key] != meta[key]:
                raise ValueError('Original metadata mismatch: ' + key)
        if xyz != meta['dimensions_whd_m']:
            raise ValueError('Archive dimensions are physical XYZ; do not reorder')
        ns = predictions[(fid, 'N3_SUBPIX')]
        if ns['qFinal'] != ns['native_points'] or ns['raw_hw'] != meta['raw_hw']:
            raise ValueError('Unchanged raw-camera native9 required')
        if fid in frames95:
            if source['points']['BASE'] != meta['q0'] or source['points']['N3_SUBPIX'] != ns['native_points']:
                raise ValueError('Original95 coordinate mismatch')
            ns_pose = source['initial_poses']['N3_SUBPIX']
            raw_pose = source['initial_poses']['BASE']
            assert_pose_equal(ns_pose, ns['actual_pose'], atol=0)
        else:
            if source['raw_points'] != ns['native_points']:
                raise ValueError('Historical143 raw_points means N3_SUBPIX, not RAW')
            ns_pose = source['initial_pose']
            assert_pose_equal(ns_pose, ns['actual_pose'], atol=0)
            raw_pose = predictions.get((fid, 'BASE'), {}).get('actual_pose')
        computed = infer(meta['q0'], np.asarray(meta['K'], float), np.asarray(xyz, float), False)
        if raw_pose is None:
            raw_pose = computed
            recomputed.append(fid)
            coarse_source = 'NEW_CPU1_EXACT_OLD_F_ON_ORIGINAL_RAW_Q0'
        else:
            parity.append(assert_pose_equal(raw_pose, computed))
            cached.append(fid)
            coarse_source = 'UNCHANGED_HISTORICAL_OWN_RAW_F_VERIFIED_BY_EXACT_OLD_F'
        c = cache_by_id[fid]
        if c['key'] != source['image'] or c['sha256'] != meta['cache_sha256']:
            raise ValueError('Original image/cache identity mismatch')
        bind(Path(c['path']), c['sha256'])
        bind(Path(source['image']), meta['image_sha256'])
        frame = {key: deepcopy(meta[key]) for key in ('id', 'session', 'image_sha256', 'raw_hw', 'K', 'selected_index', 'candidate_metadata')}
        frame.update(partition=partition, cluster_id=clusters[meta['session']], image=source['image'], xyz=deepcopy(xyz),
                     prediction_support=deepcopy(meta['prediction_support']), arms={},
                     selected_detection_cache={'path': c['path'], 'sha256': c['sha256']})
        for branch, q, pose, src in ((BRANCHES[0], meta['q0'], raw_pose, coarse_source),
                                    (BRANCHES[1], ns['native_points'], ns_pose, 'UNCHANGED_HISTORICAL_OWN_N3_SUBPIX_F')):
            support = source.get('prediction_support', meta['prediction_support'])
            if isinstance(support, dict):
                support = support['BASE' if branch == 'RAW_YOLO' else branch]
            frame['arms'][branch] = dict(native_points=deepcopy(q), keypoints_conf=deepcopy(meta['candidate_metadata']['keypoints_conf']),
                prediction_support=deepcopy(support), coarse_pose=deepcopy(pose), source_point_key='BASE' if branch == 'RAW_YOLO' else branch,
                confidence_source='Original selected YOLO detection confidence reused; no separately calibrated N3 confidence',
                coarse_source=src, baseline_accepted=None, baseline_fallback=None)
            if len(q) != 9 or len(support) != 9 or len(frame['arms'][branch]['keypoints_conf']) != 9:
                raise ValueError('Native9 archive required')
        result.append(frame)
        if (i+1) % 25 == 0:
            print('INPUT', i+1, '/238', flush=True)
    if Counter(r['partition'] for r in result) != Counter(PARTITIONS):
        raise ValueError('All238 partition counts required')
    if len(cached) != 102 or len(recomputed) != 136:
        raise ValueError('Frozen102 cached /136 missing RAW F contract changed')
    gain = read(bind(GAIN10, 'da7888b529939ea64dd20a78dcd4fc0493943f64cea46ca9f0a77a5f92891212'))
    lookup = {r['id']: r for r in result}
    gain_checks = []
    for old in gain['frames']:
        new = lookup[old['id']]
        for key in ('id', 'session', 'image', 'image_sha256', 'raw_hw', 'K', 'xyz', 'selected_index', 'candidate_metadata'):
            if old[key] != new[key]:
                raise ValueError('Gain10 common source mismatch: ' + key)
        for branch in BRANCHES:
            for key in ('native_points', 'keypoints_conf', 'prediction_support', 'coarse_pose', 'source_point_key', 'confidence_source'):
                if old['arms'][branch][key] != new['arms'][branch][key]:
                    raise ValueError('Gain10 exact branch mismatch: ' + key)
        gain_checks.append(old['id'])
    image_groups = defaultdict(list)
    for row in result:
        image_groups[row['image_sha256']].append(row['id'])
    duplicates = [ids for ids in image_groups.values() if len(ids) > 1]
    session_counts = Counter(r['session'] for r in result)
    cluster_counts = {p: len({r['cluster_id'] for r in result if r['partition'] == p}) for p in PARTITIONS}
    cluster_counts['PRIMARY'] = len({r['cluster_id'] for r in result if r['partition'] != 'CAL'})
    resource = dict(wall_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu,
        process_count=1, cv2_threads=1, env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        new_YOLO_N3_forwards=0, new_RAW_F_poses=136, cached_RAW_F_parity_calls=102, total_F_calls=238,
        GT_or_reference_document_opens=0, image_decodes=0, GPU_calls=0,
        note='Read-only metadata investigation commands precede this timed preparation; no GPU/NN in those commands')
    contract = dict(schema=SCHEMA, rows=238, ordered_ids=order, partitions=PARTITIONS, primary_rows=192, severe_rows=0,
        branches=list(BRANCHES), N3_seed=1, N3_temperature=1.0, N3_checkpoint=n3ckpt,
        detector_checkpoint=dict(path=str(DETECTOR), sha256=inputs[str(DETECTOR)]),
        selected_candidate_and_RAW_YOLO_execution_shared=True, original_native9_whole_image_coordinates=True,
        original_xyz_semantics='physical XYZ; archival dimensions_whd_m key stores XYZ here; no reorder',
        original_F_source_basis=False, same_F_prediction_functions_only=True, source_AST_allowlist={str(POSE_SOURCE):['rotations','infer'],str(SOLVE_SOURCE):['cuboid','solve']},
        missing_RAW_F_ids=recomputed, cached_RAW_F_ids=cached, cached_RAW_F_parity_max_abs=max(parity), pose_parity_atol=POSE_ATOL, pose_parity_rtol=0,
        original95_and_original143_exact=True, gain10_exact_ids=gain_checks, gain10_source_difference='coarse_source description only; all numeric/sourcepoint/confidence values exact',
        session_counts=dict(session_counts), recording_cluster_counts=cluster_counts, recording_map_sha256=inputs[str(RECORDINGS)],
        recording_group_limit='Content/provenance grouping, not proof of statistically independent captures',
        duplicate_image_SHA_groups=duplicates, original245_and110_status='No authoritative245/110 ID manifest established yet; no inferred removal reason',
        reference_numeric_or_human_state_inputs=False, reference_manifest_owned_by_separate_scorer=True,
        resources=resource, source_sha256=sources, input_sha256=inputs)
    output.mkdir(parents=True)
    with gzip.open(output/'INPUTS.jsonl.gz','wt',compresslevel=1,encoding='utf-8') as stream:
        for row in result:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    write(output/'INPUTS.json',dict(schema=SCHEMA,ordered_ids=order,branches=list(BRANCHES),frames=result,
        reference_or_GT_inputs=False,baseline_accepted_and_fallback_not_stored=True))
    write(output/'CONTRACT.json', contract)
    seal = dict(schema='MINIMAL_BI_PREDICTOR_INPUT_SEAL_V1', rows=238, ordered_ids=order, partitions=PARTITIONS,
        arms=list(BRANCHES), inputs_sha256=sha(output/'INPUTS.json'), inputs_gzip_sha256=sha(output/'INPUTS.jsonl.gz'), canonical_row_sha256=[digest(r) for r in result],
        contract_sha256=sha(output/'CONTRACT.json'), source_sha256={str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)):sha(Path(__file__).resolve()), **sources}, input_sha256=inputs,
        resources=resource, baseline_acceptance='Not stored in historical F; null/NA, not manufactured accepted=True')
    write(output/'INPUT_SEAL.json',seal)
    if load_frames(output) != result:
        raise ValueError('Serialized input canonical roundtrip mismatch')
    docs.mkdir(parents=True, exist_ok=True)
    write(docs/'INPUT_CONTRACT_V1.json',dict(contract, artifact_input_seal_sha256=sha(output/'INPUT_SEAL.json'),
         inputs_path='INPUT_ARTIFACT_DIR/INPUTS.json', input_seal_path='INPUT_ARTIFACT_DIR/INPUT_SEAL.json'))
    write(docs/'INPUT_PREPARATION_RECEIPT_V1.json',dict(rows=238, partitions=PARTITIONS, gain10_exact=10,
        input_seal_sha256=sha(output/'INPUT_SEAL.json'), contract_sha256=sha(output/'CONTRACT.json'),
        resources=resource, outputs_unchanged_originals=True,
        source_path=str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT))))
    return dict(output=str(output), rows=238, cached_RAW_F=102, new_RAW_F=136, max_parity_abs=max(parity),
                input_seal_sha256=sha(output/'INPUT_SEAL.json'), resources=resource)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','verify'])
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    p.add_argument('--docs',type=Path,default=DOC_ROOT)
    args=p.parse_args()
    print(prepare_inputs(args.output,args.source_root,docs=args.docs) if args.stage=='prepare' else dict(rows=len(load_frames(args.output))),flush=True)


if __name__=='__main__':
    main()
