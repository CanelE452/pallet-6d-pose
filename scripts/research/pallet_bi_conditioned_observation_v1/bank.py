"""Reference-free, manifest-bound cached N0 observation bank production."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import observer as O
from . import features as F


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    payload = json.dumps(O._json(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    if path.suffix == '.gz':
        with pending.open('wb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                compressed.write(payload.encode())
    else:
        pending.write_text(payload)
    pending.replace(path)


def source_bindings():
    return {p.name: sha(p) for p in (Path(O.__file__), Path(F.__file__), Path(__file__))}


def validate_lock(extraction_path, producer_path, p0_path):
    lock = read(extraction_path); producer = read(producer_path)
    if lock['P0_input_contract_sha256'] != sha(p0_path):
        raise ValueError('P0 input contract changed')
    if lock['configuration'] != O.CONFIGURATION or lock['feature_contract'] != F.feature_contract():
        raise ValueError('Scientific extraction configuration/feature contract changed')
    actual = source_bindings()
    for name in ('observer.py', 'features.py'):
        if lock['source_sha256'][name] != actual[name]:
            raise ValueError('Frozen extraction source changed: ' + name)
    if producer['bank_py_sha256'] != actual['bank.py'] or producer['extraction_lock_sha256'] != sha(extraction_path):
        raise ValueError('Bank producer source or extraction lock changed')
    if lock['mask_key'] != 'pallet_mask_tight':
        raise ValueError('The frozen original v2 correction mask must be tight')
    return lock, producer


def _same(value, expected, name):
    if not np.array_equal(np.asarray(value), np.asarray(expected)):
        raise ValueError('Cached N0 input mismatch: ' + name)


def _legacy_source(observation, face_rows):
    typed = deepcopy(face_rows)
    for item in typed:
        for line in item['lines']:
            record = line.get('provenance', {}).get('line_record')
            if record is not None:
                line['physical_source_ids'] = [O.physical_line_source_id(record)]
    return dict(schema='FROZEN_EXISTING_D1_RULE_OBSERVATIONS_V1', states=list(observation['states']),
        frozen_visible_measurements=deepcopy(observation['frozen_BI_fit']),
        boundary_observations=typed,
        provenance=dict(source='existing D1 recover_stages/fixed_fit plus cached FACE typed replacement',
                        new_solver_or_pose_bank_used=False, new_full_candidate_union=False))


def build_frame(task):
    """One frame in; only original RGB, automatic cache and N0 are read."""
    started = time.monotonic(); cv2.setNumThreads(1)
    frame, inventory, output, lock_binding = task
    folder = Path(inventory['observation_folder']); destination = Path(output)
    observation_path = folder / 'OBSERVATION.jsonl.gz'; mask_path = folder / 'MASK.npz'
    observation_seal_path = folder / 'OBSERVATION_SEAL.json'
    cache_root = folder.parents[1]
    face_path = cache_root.parent / cache_root.name.replace('_observations_v1', '_candidates_v1') / 'frames' / folder.name / 'E_RUN.jsonl.gz'
    inputs = dict(upstream_row_sha256=O._hash(frame), observation_sha256=sha(observation_path),
        mask_sha256=sha(mask_path), observation_seal_sha256=sha(observation_seal_path),
        FACE_cache_sha256=sha(face_path), FACE_seal_sha256=sha(face_path.parent/'FRAME_SEAL.json'),
        image_sha256=sha(frame['image']))
    if inputs['observation_seal_sha256'] != inventory['observation_seal_sha256']:
        raise ValueError('Per-frame cache seal differs from frozen P0 domain seal')
    expected_binding = dict(lock_binding, inputs=inputs)
    seal_path = destination / 'SEAL.json'
    if seal_path.exists():
        seal = read(seal_path)
        if seal.get('binding') != expected_binding:
            raise ValueError('Existing frame bank lock/input mismatch: ' + frame['id'])
        for name, digest in seal['outputs'].items():
            if sha(destination / name) != digest:
                raise ValueError('Cached common bank bytes changed')
        return dict(seal, reused=True, elapsed_this_call_seconds=time.monotonic()-started)
    seal = read(observation_seal_path)
    if (seal['id'] != frame['id'] or not seal['complete'] or seal['observation_sha256'] != inputs['observation_sha256']
            or seal['mask_sha256'] != inputs['mask_sha256']):
        raise ValueError('Existing observation seal mismatch')
    original_input_hash=hashlib.sha256(json.dumps(frame,sort_keys=True,ensure_ascii=False,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    if seal['input_row_sha256'] != original_input_hash:
        raise ValueError('Original exact automatic-cache upstream row changed')
    face_seal=read(face_path.parent/'FRAME_SEAL.json')
    if (face_seal['id']!=frame['id'] or face_seal['observation_seal_sha256']!=inputs['observation_seal_sha256']
            or face_seal['e_run_sha256']!=inputs['FACE_cache_sha256']):
        raise ValueError('FACE cache seal no longer matches this N0 observation')
    if inputs['image_sha256'] != frame['image_sha256'] or frame['image_sha256'] != inventory['image_sha256']:
        raise ValueError('Original RGB bytes changed')
    observation = rows(observation_path)
    if len(observation) != 1:
        raise ValueError('Exactly one cached automatic observation row required')
    observation = observation[0]
    if (observation['id'] != frame['id'] or observation['image_sha256'] != frame['image_sha256']
            or observation['states'] != inventory['states']):
        raise ValueError('Automatic observation row binding mismatch')
    geometry = observation['runtime']['initial_geometry']
    _same(geometry['native_xyz'], frame['native_xyz'], 'native_xyz')
    _same(geometry['K'], frame['K'], 'K')
    _same(geometry['image_hw'], frame['raw_hw'], 'raw_hw')
    for key in ('R_cf', 'centroid', 'cf_extents'):
        _same(geometry[key], frame['initial_pose'][key], key)
    for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
        _same(observation['_original_source']['original_coarse_pose'][key], frame['initial_pose'][key], 'original_pose.' + key)
    gated = np.asarray(frame['initial_points'], float).copy()
    gated[:8][np.asarray(geometry['initial_self_hidden'], bool)] = [-1., -1.]
    _same(observation['_original_source']['original_raw_points'], gated, 'gated original raw points')
    face_records = rows(face_path)
    if len(face_records) != 1 or face_records[0]['id'] != frame['id']:
        raise ValueError('Exactly one same-N0 FACE cache required')
    face = face_records[0]['face_observation']
    if face['image_binding']['sha256'] != frame['image_sha256'] or face['image_binding']['image_hw'] != frame['raw_hw']:
        raise ValueError('Cached FACE original RGB binding mismatch')
    # The P0 audit independently checked every FACE record's geometry/raw N0
    # linkage. The selected row's raw archive is also checked by observe().
    image = cv2.imread(frame['image'])
    if image is None or list(image.shape[:2]) != frame['raw_hw']:
        raise ValueError('Exact original BGR canvas required')
    with np.load(mask_path, allow_pickle=False) as archive:
        mask = archive['pallet_mask_tight'].copy()
    bank = O.observe(image, mask, frame['initial_points'], geometry, observation['states'],
        frame_id=frame['id'], image_binding=dict(sha256=frame['image_sha256'], image_hw=frame['raw_hw']),
        shared_partial=face['boundary_observations'], raw_measurements=observation['original_CAL_fit'])
    feature = dict(schema='COMMON_CANDIDATE_FEATURE_ROWS_V1', id=frame['id'], split=frame['split'],
        partition=frame['partition'], corner_rows=F.feature_rows(bank))
    legacy = _legacy_source(observation, face['boundary_observations'])
    write(destination/'BANK.json.gz', bank); write(destination/'FEATURES.json.gz', feature)
    write(destination/'LEGACY.json.gz', legacy)
    coverage = dict(visible_corners=len(bank['queried_native_ids']),
        full_alternative_corners=sum(c['full_alternative_count'] > 0 for c in bank['corners'] if c['candidates']),
        full_alternatives=sum(c['full_alternative_count'] for c in bank['corners'] if c['candidates']),
        candidates=sum(len(c['candidates']) for c in bank['corners']),
        fixed_partial_corners=sum(c['partial_line'] is not None for c in bank['corners']),
        q0_only_corners=sum(c['full_alternative_count'] == 0 and c['partial_line'] is None for c in bank['corners'] if c['candidates']),
        lines=sum(len(c['rays']) for c in bank['corners']),
        supported_lines=sum(r.get('supported', False) for c in bank['corners'] for r in c['rays']))
    result = dict(schema='COMMON_BANK_FRAME_SEAL_V1', id=frame['id'], partition=frame['partition'], split=frame['split'],
        complete=True, binding=expected_binding, coverage=coverage,
        outputs={name:sha(destination/name) for name in ('BANK.json.gz','FEATURES.json.gz','LEGACY.json.gz')},
        elapsed_seconds=time.monotonic()-started, reference_reads=0, neural_forward_calls=0)
    write(seal_path,result)
    return dict(result,reused=False,elapsed_this_call_seconds=time.monotonic()-started)


def run(args):
    started=time.monotonic(); extraction,producer=validate_lock(args.extraction_lock,args.producer_lock,args.p0)
    p0=read(args.p0)
    # The only P0 fields consumed are the explicitly enumerated automatic
    # inventory fields. No synthetic target or real reference section is used.
    allowed=('id','partition','split','image','image_sha256','raw_hw','observation_folder','states')
    inventory=[{key:item[key] for key in allowed} for item in p0[args.domain]['inventory']]
    frames=rows(args.upstream)
    if len(frames)!=len(inventory) or any(frame['id']!=item['id'] for frame,item in zip(frames,inventory)):
        raise ValueError('Upstream/P0 inventory order differs')
    original_domain_seal=Path(inventory[0]['observation_folder']).parents[1]/'OBSERVATION_SEAL.json'
    if sha(original_domain_seal)!=p0[args.domain]['observation_seal_sha256']:
        raise ValueError('Frozen P0 automatic-cache domain seal changed')
    domain_seal=read(original_domain_seal)
    if (domain_seal['input_sha256']!=sha(args.upstream) or domain_seal['ordered_ids']!=[f['id'] for f in frames]
            or not domain_seal['complete'] or domain_seal['rows']!=len(frames)):
        raise ValueError('Actual upstream file differs from frozen automatic-cache input')
    for ordinal,(frame,item) in enumerate(zip(frames,inventory)):
        if any(frame[key]!=item[key] for key in ('id','partition','split','image','image_sha256','raw_hw')):
            raise ValueError('Frozen P0 image or split metadata changed')
        item['observation_seal_sha256']=domain_seal['frame_seals_sha256'][f'{ordinal:04d}']
    selected=[(ordinal,frame,item) for ordinal,(frame,item) in enumerate(zip(frames,inventory))
              if not args.partition or frame['partition']==args.partition]
    if args.limit is not None:
        selected=selected[:args.limit]
    output=Path(args.output)
    lock_binding=dict(extraction_lock_sha256=sha(args.extraction_lock), bank_producer_lock_sha256=sha(args.producer_lock),
        P0_input_contract_sha256=sha(args.p0), source_sha256=source_bindings())
    tasks=[(frame,item,str(output/'frames'/args.domain/f'{ordinal:04d}'),lock_binding) for ordinal,frame,item in selected]
    if not 1<=args.workers<=3:
        raise ValueError('CPU worker count must be 1..3')
    if args.workers==1:
        seals=[build_frame(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            seals=list(executor.map(build_frame,tasks,chunksize=1))
    feature_path=output/f'{args.domain}_{args.partition or "all"}_{args.limit or "all"}_FEATURES.jsonl.gz'
    feature_lines=[]
    for (ordinal,frame,_),seal in zip(selected,seals):
        feature=read(output/'frames'/args.domain/f'{ordinal:04d}'/'FEATURES.json.gz')
        feature_lines.append(json.dumps(feature,sort_keys=True,separators=(',', ':'),allow_nan=False))
    feature_path.parent.mkdir(parents=True,exist_ok=True)
    with feature_path.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
            compressed.write(('\n'.join(feature_lines)+'\n').encode())
    coverage={key:sum(s['coverage'][key] for s in seals) for key in seals[0]['coverage']} if seals else {}
    report=dict(schema='COMMON_BANK_RUN_V1',domain=args.domain,partition=args.partition,limit=args.limit,workers=args.workers,
        frames=len(seals),new_frames=sum(not s['reused'] for s in seals),reused_frames=sum(s['reused'] for s in seals),
        elapsed_wall_seconds=time.monotonic()-started,cpu_batch_seconds=sum(s['elapsed_this_call_seconds'] for s in seals),
        original_extraction_seconds=sum(s['elapsed_seconds'] for s in seals),coverage=coverage,
        feature_file=str(feature_path),feature_sha256=sha(feature_path),binding=lock_binding,
        frame_seals=[str(output/'frames'/args.domain/f'{ordinal:04d}'/'SEAL.json') for ordinal,_,_ in selected],
        new_state_or_mask_forwards=0,pose_solves=0,reference_reads=0)
    if seals:
        report['estimated_1241_frame_serial_seconds']=report['original_extraction_seconds']/len(seals)*1241
    write(output/f'{args.domain}_{args.partition or "all"}_{args.limit or "all"}_RUN.json',report)
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return report


def parser():
    result=argparse.ArgumentParser()
    for name in ('p0','upstream','extraction-lock','producer-lock','output'):
        result.add_argument('--'+name,required=True)
    result.add_argument('--domain',choices=('real','synthetic'),required=True)
    result.add_argument('--partition'); result.add_argument('--limit',type=int)
    result.add_argument('--workers',type=int,default=1)
    return result


if __name__=='__main__':
    run(parser().parse_args())
