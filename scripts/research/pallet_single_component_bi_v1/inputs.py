"""Predictor-only component inputs on the existing fixed238 population.

The 20261008 experiment already saved original pre-SubPix N3 coordinates and
RAW->SubPix+cap coordinates for all319 images. This builder takes the exact238
subset, reproduces all cached F poses and computes only missing SubPix F poses.
Mixed historical journals are lexically projected before JSON decoding. Their
score/label objects, evaluators, metadata/GT loaders and NN APIs are not used.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np

from scripts.research.pallet_minimal_bi_ablation_v1 import inputs as P


SCHEMA = 'SINGLE_COMPONENT_BI_FIXED238_INPUTS_V1'
BRANCHES = ('RAW_YOLO', 'N3_SUBPIX', 'N3_ONLY', 'SUBPIX_ONLY')
NEW_BRANCHES = ('N3_ONLY', 'SUBPIX_ONLY')
OLD = Path('_docs/experiments/pallet_n3_subpix_20261008_v1')
METHODS_SOURCE = Path('scripts/research/pallet_training_free_compare_20261007_v1/methods.py')
CHECKPOINT = Path('data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed1/last.pt')
FROZEN = {
    str(OLD/'PROTOCOL.json'): '91167ab626a6c02b3d5f425b51d483bd57ef5a46cc73341a827335d1a2a1b197',
    str(OLD/'PREDICTIONS.jsonl.gz'): '041afb757b3e2f410ee2c10138ad114d542c5e11c7c1281ef6a18cb57425ad7c',
    str(CHECKPOINT): 'ceea743b7b8467cf43eef66582b923dd980e5b2fb27f5575af5df185109f22dd',
}
POSE_ATOL = 1e-9
POSE_RTOL = 0.
ALLOWED_PROTOCOL = frozenset({'N3_seed','N3_temperature','N3_checkpoint','SUBPIX','cap',
    'input_manifest','input_manifest_sha256','code'})
ALLOWED_ROW = frozenset({'id','method','raw_hw','native_points','qFinal','qN','qS',
    'actual_pose','prediction_support','fixed_metadata'})


def _equal(a, b, message):
    if P.canonical(a) != P.canonical(b):
        raise ValueError(message)


def predictor_rows(path, wanted_ids, parser):
    """Skip all score/reference fields even in the historical mixed journal."""
    selected = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            header = parser(line, {'id','method'})
            fid, method = header.get('id'), header.get('method')
            if fid not in wanted_ids or method not in ('BASE','N3','SUBPIX','N3_SUBPIX'):
                continue
            row = parser(line, ALLOWED_ROW)
            key = (fid, method)
            if key in selected:
                raise ValueError('Duplicate original prediction row')
            selected[key] = row
    if set(selected) != {(fid, method) for fid in wanted_ids for method in ('BASE','N3','SUBPIX','N3_SUBPIX')}:
        raise ValueError('All four historical predictor rows required for every target image')
    return selected


def cap_function(source_root):
    """Compile exactly the existing pure cap, excluding unrelated CV rank config."""
    path = Path(source_root) / METHODS_SOURCE
    names = {'_inputs', 'cap_points'}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Exact historical cap definitions missing')
    namespace = {'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'), namespace)
    return namespace['cap_points']


def validate_frame(frame, metadata, records, cap):
    """Return native qN and qSubPix with the immutable own RAW/combined join."""
    fid = frame['id']
    for name in ('id','session','image_sha256','raw_hw','K','selected_index','candidate_metadata'):
        _equal(frame[name], metadata[name], 'Original detector/image metadata changed: '+name)
    _equal(frame['xyz'], metadata['dimensions_whd_m'], 'Physical XYZ dimensions must not be reordered')
    raw, ns = frame['arms']['RAW_YOLO'], frame['arms']['N3_SUBPIX']
    _equal(raw['native_points'], metadata['q0'], 'Own prior RAW q0 differs from original execution')
    _equal(raw['prediction_support'], metadata['prediction_support'], 'Own original support changed')
    _equal(raw['keypoints_conf'], metadata['candidate_metadata']['keypoints_conf'], 'Original confidence changed')
    q0, support = np.asarray(metadata['q0'],float), np.asarray(metadata['prediction_support'],bool)
    if q0.shape != (9,2) or support.shape != (9,):
        raise ValueError('Native9 predictor contract required')
    for method in ('BASE','N3','SUBPIX','N3_SUBPIX'):
        row = records[(fid,method)]
        if np.asarray(row['native_points'],float).shape != (9,2):
            raise ValueError('Historical native9 array changed')
        _equal(row['native_points'], row['qFinal'], 'Historical final readout differs from native image coordinates')
        _equal(row['raw_hw'], frame['raw_hw'], 'Historical original camera canvas changed')
        _equal(row['prediction_support'], metadata['prediction_support'], 'Historical detection support changed')
        fixed = row['fixed_metadata']
        if fixed.get('preserved') is not True:
            raise ValueError('Historical metadata preservation proof missing')
        _equal(fixed['selected_index'], frame['selected_index'], 'Selected detection changed')
        _equal(fixed['candidate_metadata'], frame['candidate_metadata'], 'Candidate metadata changed')
    _equal(records[(fid,'BASE')]['native_points'], raw['native_points'], 'Historical BASE does not bind prior RAW')
    _equal(records[(fid,'N3_SUBPIX')]['native_points'], ns['native_points'], 'Historical combined final does not bind prior N3/SubPix')
    _equal(records[(fid,'N3')]['native_points'], metadata['qN'], 'Pre-SubPix qN cache changed')
    _equal(ns['prediction_support'], metadata['prediction_support'], 'Prior combined support differs')
    height,width = frame['raw_hw']
    for method in ('SUBPIX','N3_SUBPIX'):
        uncapped = np.asarray(records[(fid,method)]['qS'],float)
        final = cap(q0,uncapped,width,height,support)
        if not np.array_equal(final, np.asarray(records[(fid,method)]['native_points'],float),equal_nan=True):
            raise ValueError('Historical strict total cap must be around original RAW q0: '+method)
    out = {}
    for branch,method in (('N3_ONLY','N3'),('SUBPIX_ONLY','SUBPIX')):
        q = np.asarray(records[(fid,method)]['native_points'],float)
        unusable = ~support | ~np.isfinite(q0).all(axis=1) | np.all(q0 == -1,axis=1)
        if not np.array_equal(q[8],q0[8],equal_nan=True) or not np.array_equal(q[unusable],q0[unusable],equal_nan=True):
            raise ValueError('Center or unsupported/missing prediction changed')
        if not np.isfinite(q[~unusable]).all():
            raise ValueError('Nonfinite supported coordinate')
        out[branch] = q.tolist()
    return out


def prepare(prior_input_dir, source_root, output):
    """Prepare exact238 frames and return (frames, INPUT_SEAL dictionary)."""
    prior_input_dir, source_root, output = map(Path,(prior_input_dir,source_root,output))
    if output.exists():
        raise FileExistsError('Preserve every input preparation attempt')
    output.mkdir(parents=True)
    started, cpu = time.monotonic(), time.process_time()
    cv2.setNumThreads(1)
    sources, bindings, parity, created = {}, {}, [], []
    completed = 0
    try:
        prior = P.load_frames(prior_input_dir)
        prior_seal = P.read(prior_input_dir/'INPUT_SEAL.json')
        ids = [r['id'] for r in prior]
        if Counter(r['partition'] for r in prior) != Counter(P.PARTITIONS):
            raise ValueError('Fixed CAL46/TEST49/REMAINDER143 required')
        for relative,expected in FROZEN.items():
            actual = P.sha(source_root/relative)
            if actual != expected:
                raise ValueError('Frozen source input changed: '+relative)
            bindings[relative] = actual
        for relative in (P.POSE_SOURCE,P.SOLVE_SOURCE,P.SELECTOR_SOURCE,P.GEOMETRY_SOURCE,P.PARSER_SOURCE,METHODS_SOURCE):
            relative = str(relative)
            actual = P.sha(source_root/relative)
            if relative in prior_seal['source_sha256'] and actual != prior_seal['source_sha256'][relative]:
                raise ValueError('Existing F/parser source changed')
            sources[relative] = actual
        sources['scripts/research/pallet_minimal_bi_ablation_v1/inputs.py'] = P.sha(P.__file__)
        parser = P.prediction_parser(source_root)
        protocol = parser((source_root/OLD/'PROTOCOL.json').read_text(),ALLOWED_PROTOCOL)
        if (protocol['N3_seed'] != 1 or protocol['N3_temperature'] != 1.0
                or protocol['N3_checkpoint']['path'] != str(CHECKPOINT)
                or protocol['N3_checkpoint']['sha256'] != FROZEN[str(CHECKPOINT)]
                or protocol['cap'] != {'fraction':.01,'reference':'original Base q0 and raw image diagonal','arithmetic':'float64'}):
            raise ValueError('Original frozen seed1/temperature1/raw-q0 cap contract changed')
        expected_subpix = dict(winSize=[5,5],zeroZone=[-1,-1],
            criteria=dict(type=3,maxCount=40,epsilon=.001),
            initial_and_refined_arithmetic='OpenCV float32; returned native array float64',
            minimum_image_extent=15,initial_inside_required=True,
            fallback='unsupported/missing/outside initial, too-small image, OpenCV exception, nonfinite or outside refined output')
        _equal(protocol['SUBPIX'],expected_subpix,'Existing OpenCV SubPix definition changed')
        for entry in protocol['code']:
            if P.sha(source_root/entry['path']) != entry['sha256']:
                raise ValueError('Historical component builder source changed')
            sources[entry['path']] = entry['sha256']
        meta = {r['id']:r for r in protocol['input_manifest']}
        if len(meta) != 319 or not set(ids) <= set(meta) or P.digest(protocol['input_manifest']) != protocol['input_manifest_sha256']:
            raise ValueError('Original319 detector/N3 predictor manifest required')
        cached = predictor_rows(source_root/OLD/'PREDICTIONS.jsonl.gz',set(ids),parser)
        cap, infer = cap_function(source_root), P.existing_F(source_root)
        result = []
        for frame in prior:
            fid = frame['id']
            selected = validate_frame(frame,meta[fid],cached,cap)
            image = frame['image']
            cache = frame['selected_detection_cache']
            for relative,expected in ((image,frame['image_sha256']),(cache['path'],cache['sha256'])):
                if P.sha(source_root/relative) != expected:
                    raise ValueError('Original RGB/detector cache changed')
                bindings[relative] = expected
            updated = deepcopy(frame)
            for branch, method in (('N3_ONLY','N3'),('SUBPIX_ONLY','SUBPIX')):
                old_pose = cached[(fid,method)].get('actual_pose')
                computed = infer(selected[branch],np.asarray(frame['K'],float),np.asarray(frame['xyz'],float),False)
                if old_pose is not None:
                    maximum = P.assert_pose_equal(old_pose,computed,atol=POSE_ATOL)
                    if old_pose.get('available'):
                        _equal(old_pose['cf_extents'],computed['cf_extents'],'Exact discrete native CF geometry changed')
                    parity.append(dict(id=fid,branch=branch,max_abs_difference=maximum,
                        available_and_selected_hypothesis_exact=True,passed=True))
                    pose = deepcopy(old_pose)
                    pose_source = 'UNCHANGED_HISTORICAL_OWN_F_VERIFIED_BY_EXACT_EXISTING_F'
                else:
                    if branch != 'SUBPIX_ONLY':
                        raise ValueError('All238 N3 cached F vectors are required')
                    pose = computed
                    created.append(fid)
                    pose_source = 'NEW_CPU1_EXISTING_F_ON_ORIGINAL_SUBPIX_CAP1_POINTS'
                updated['arms'][branch] = dict(native_points=selected[branch],
                    keypoints_conf=deepcopy(frame['arms']['RAW_YOLO']['keypoints_conf']),
                    prediction_support=deepcopy(frame['arms']['RAW_YOLO']['prediction_support']),
                    coarse_pose=pose,source_point_key='ORIGINAL_N3_PRE_SUBPIX_QN' if branch == 'N3_ONLY' else 'ORIGINAL_RAW_SUBPIX_CAP1',
                    confidence_source='Original selected YOLO confidence; no new component confidence calibration',
                    coarse_source=pose_source,baseline_accepted=None,baseline_fallback=None,
                    original_component_method=method)
            for old in ('RAW_YOLO','N3_SUBPIX'):
                _equal(updated['arms'][old],frame['arms'][old],'Existing input branch mutated')
            result.append(updated)
            completed += 1
            if completed % 25 == 0 or completed == 238:
                print('COMPONENT_INPUT',completed,'/238','cached_F_checks',len(parity),'new_F',len(created),flush=True)
        counts = Counter(r['branch'] for r in parity)
        if counts != Counter({'N3_ONLY':238,'SUBPIX_ONLY':19}) or len(created) != 219:
            raise ValueError('Frozen238 N3 /19 SubPix cached /219 missing SubPix F counts changed')
        resources = dict(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
            workers=1,cv2_threads=1,new_NN_forwards=0,new_GPU_calls=0,new_YOLO_or_N3_forwards=0,
            image_decodes=0,cornerSubPix_calls=0,F_calls=476,cached_F_parity_calls=257,new_missing_F=219,
            GT_or_score_or_reference_objects_decoded=0)
        contract = dict(schema=SCHEMA,rows=238,primary_rows=192,partitions=P.PARTITIONS,
            old_experiment_population=319,current_population='exact existing CAL46+TEST49+REMAINDER143 subset; severe0',
            branches=list(BRANCHES),new_branches=list(NEW_BRANCHES),original_RAW_and_N3_SUBPIX_exact=True,
            N3_seed=1,N3_temperature=1.0,N3_checkpoint=protocol['N3_checkpoint'],
            same_original_YOLO_detection_candidate_metadata=True,
            N3_ONLY='original cached qN; no SubPix and no extra strict float64 cap',
            SUBPIX_ONLY='existing correct(gray,q0,support,SUBPIX), then existing strict cap centered on RAW q0',
            N3_SUBPIX='existing SubPix initialized at qN; final strict total cap centered on RAW q0, not qN',
            tiny_original_N3_float32_cap_roundoff_preserved=True,SUBPIX_configuration=protocol['SUBPIX'],
            cap=protocol['cap'],coordinate_order='original native9; no semantic reindexing or pose projection as observation',
            F='unchanged existing_F(q,K,physical_xyz,False), own W/D hypothesis and own CF geometry',
            cached_F_parity=dict(calls=257,counts=dict(counts),numeric_atol=POSE_ATOL,numeric_rtol=POSE_RTOL,
                max_abs=max(r['max_abs_difference'] for r in parity),discrete_exact=True),
            new_missing_SubPix_F_ids=created,source_sha256=sources,input_sha256=bindings,
            resources=resources,new_training_or_seeds_or_GT_modifications=0,reference_or_GT_inputs=False)
        P.write(output/'CONTRACT.json',contract)
        P.write(output/'F_PARITY.json',dict(rows=parity,atol=POSE_ATOL,rtol=POSE_RTOL,failures=0))
        P.write(output/'INPUTS.json',dict(schema=SCHEMA,ordered_ids=ids,branches=list(BRANCHES),frames=result,
            reference_or_GT_inputs=False,baseline_accepted_and_fallback_not_stored=True))
        with gzip.open(output/'INPUTS.jsonl.gz','wt',compresslevel=1) as stream:
            for row in result:
                stream.write(P.canonical(row)+'\n')
        seal = dict(schema=SCHEMA+'_SEAL',rows=238,ordered_ids=ids,partitions=P.PARTITIONS,arms=list(BRANCHES),
            inputs_sha256=P.sha(output/'INPUTS.json'),inputs_gzip_sha256=P.sha(output/'INPUTS.jsonl.gz'),
            canonical_row_sha256=[P.digest(r) for r in result],contract_sha256=P.sha(output/'CONTRACT.json'),
            F_parity_sha256=P.sha(output/'F_PARITY.json'),prior_input_seal_sha256=P.sha(prior_input_dir/'INPUT_SEAL.json'),
            prior_inputs_sha256=P.sha(prior_input_dir/'INPUTS.json'),source_sha256={
                'scripts/research/pallet_single_component_bi_v1/inputs.py':P.sha(__file__),**sources},
            input_sha256=bindings,resources=resources,GT_inputs=False,all_rows_and_failures_kept=True)
        P.write(output/'INPUT_SEAL.json',seal)
        _equal(P.load_frames(output),result,'Roundtrip predictor inputs changed')
        return result, seal
    except BaseException as error:
        P.write(output/'PREPARATION_FAILURE.json',dict(error_type=type(error).__name__,reason=str(error),
            completed_frames=completed,cached_F_parity_completed=len(parity),new_F_completed=len(created),
            wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
            attempt_preserved=True,original_sources_or_caches_modified=False))
        raise


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-input-dir','source-root','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    args = cli.parse_args()
    frames, seal = prepare(args.prior_input_dir,args.source_root,args.output)
    print(json.dumps(dict(rows=len(frames),F_calls=seal['resources']['F_calls'],
        parity=seal['resources']['cached_F_parity_calls'],new_F=seal['resources']['new_missing_F'],
        input_seal_sha256=P.sha(args.output/'INPUT_SEAL.json'))))
