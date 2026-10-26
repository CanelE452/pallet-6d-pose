#!/usr/bin/env python3
"""Input-only access/provenance probe for the fixed223 full RGB frames.

Preserves all1784 original queries, native slots, availability and order.
Does not read annotations, evaluator GT, checkpoints, models or segmentation labels.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import stat
from collections import Counter
import numpy as np
from PIL import Image

W = Path(__file__).absolute().parent
R = W.parent
COMPLETE = R / 'V263_SPATIAL_HEAD_C/current223_frozen_inference_v1/COMPLETE.json'
COMPLETE_SHA = '8fcfa2761b26e3f2fa236f525bf98267d4964d7a650fef5bceb99b7d2ac3e881'
INPUT_SHA = '3f244d7c7715278cd36062937ff952c0fa6f347c46afbf8a4810c12640ee5780'
ACCESS_SHA = 'd4e07313988b71dd71950848987d11925036b19e29d1a6ce2158a94bebb79137'


def safe(value, fresh=False):
    p = Path(value)
    if not p.is_absolute() or '..' in p.parts or any(x in str(p).casefold() for x in ('v112','cache','.pyc')):
        raise ValueError('Forbidden artifact path: ' + str(p))
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError('Symlink or symlink ancestor: ' + str(a))
    if fresh and p.exists():
        raise FileExistsError(str(p))
    return p


def binding(path):
    p = safe(path)
    b = p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def checked(b):
    p = safe(b['path'])
    data = p.read_bytes()
    if len(data) != b['size_bytes'] or hashlib.sha256(data).hexdigest() != b['sha256']:
        raise ValueError('SHA/size mismatch: ' + str(p))
    return data


def read_rgb(frame):
    """Decode the canonical file; always recheck encoded and decoded RGB hashes."""
    data = checked(frame['image_binding'])
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    if list(rgb.shape[:2]) != frame['raw_hw']:
        raise ValueError('Full RGB shape changed: ' + frame['frame_id'])
    if hashlib.sha256(rgb.tobytes()).hexdigest() != frame['decoded_RGB_sha256']:
        raise ValueError('Decoded RGB hash changed: ' + frame['frame_id'])
    return rgb


def input_metadata():
    b = safe(COMPLETE).read_bytes()
    if hashlib.sha256(b).hexdigest() != COMPLETE_SHA:
        raise ValueError('Pinned V263 current COMPLETE changed')
    complete = json.loads(b)
    rows_binding = complete['outputs']['INPUT_ROWS1784.jsonl']
    access_binding = complete['outputs']['RGB223_ACCESS.json']
    if rows_binding['sha256'] != INPUT_SHA or access_binding['sha256'] != ACCESS_SHA:
        raise ValueError('Pinned input-only metadata changed')
    rows = [json.loads(line) for line in checked(rows_binding).splitlines()]
    access = json.loads(checked(access_binding))
    if len(rows) != 1784 or len(access) != 223:
        raise ValueError('Expected223 frames/1784queries')
    for i, row in enumerate(rows):
        if row['global_index'] != i or row['query_index'] != i or row['frame_index'] != i//8 or row['native_slot'] != i%8:
            raise ValueError('Original query/native order mismatch')
        if row['teacher_or_GT_in_inputs'] is not False:
            raise ValueError('Input metadata must not contain teacher/GT inputs')
        if row['frame_id'] != access[i//8]['id']:
            raise ValueError('Frame/query correspondence mismatch')
    return rows, access, dict(complete=binding(COMPLETE), input_rows=rows_binding, RGB_access=access_binding)


def write_json(path, value):
    p = safe(path, True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return binding(p)


def write_jsonl(path, rows):
    p = safe(path, True)
    p.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
    return binding(p)


def probe(out):
    safe(out, True).mkdir()
    rows, access, sources = input_metadata()
    frames, queries, failures = [], [], []
    for fi, a in enumerate(access):
        original = rows[fi*8:fi*8+8]
        try:
            encoded = checked(a['canonical_binding'])
            with Image.open(io.BytesIO(encoded)) as image:
                rgb = np.asarray(image.convert('RGB'),dtype=np.uint8).copy()
            decoded_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            if decoded_sha != a['decoded_RGB_sha256']:
                raise ValueError('Decoded canonical RGB differs from original V263 input')
            frame = dict(frame_index=fi,frame_id=a['id'],
                image_binding=a['canonical_binding'],decoded_RGB_sha256=decoded_sha,
                raw_hw=list(rgb.shape[:2]),native_slots=list(range(8)),
                query_indices8=[r['global_index'] for r in original],
                original_query_xy8=[r['original_q0'] for r in original],
                original_query_available8=[r['available'] for r in original],
                original_crop_side8=[r['crop_side'] for r in original],
                query_order_preserved=True,teacher_or_annotation_or_GT_inputs=False)
            frames.append(frame)
            for row in original:
                query = dict(global_index=row['global_index'],query_index=row['query_index'],
                    frame_index=fi,frame_id=a['id'],row_id=a['id']+':native'+str(row['native_slot']),
                    native_slot=row['native_slot'],original_q0=row['original_q0'],available=row['available'],
                    original_input_supported=row['input_supported'],original_crop_side=row['crop_side'],
                    raw_hw=list(rgb.shape[:2]),image_encoded_sha256=a['canonical_binding']['sha256'],
                    image_decoded_RGB_sha256=decoded_sha,teacher_or_annotation_or_GT_inputs=False)
                queries.append(query)
        except Exception as exc:
            failures.append(dict(frame_index=fi,frame_id=a['id'],error_type=type(exc).__name__,error=str(exc)))
    frame_binding = write_jsonl(out/'FRAMES223_INPUT_ONLY.jsonl',frames)
    query_binding = write_jsonl(out/'QUERIES1784_INPUT_ONLY.jsonl',queries)
    result = dict(completed=True,PASS=len(frames)==223 and len(queries)==1784 and not failures,
        requested_frames=223,accessible_verified_frames=len(frames),missing_or_changed_frames=len(failures),
        queries=len(queries),original_available_queries=sum(r['available'] for r in queries),
        source_session_counts=dict(Counter(f['frame_id'].split(':')[0] for f in frames)),
        raw_hw_counts={str(k):v for k,v in Counter(tuple(f['raw_hw']) for f in frames).items()},
        original_query_order_native_slots_and_availability_preserved=True,
        sources=sources,frames=frame_binding,queries_manifest=query_binding,failures=failures,
        code=binding(Path(__file__).absolute()),new_annotation_GT_label_or_evaluator_reads=0,
        new_model_checkpoint_or_training_or_PnP=0,full_RGB_decodes=len(frames),
        purpose='Full-image inference inputs and provenance only; no visible/occluded or pixel training labels.',
        warning='A non-target pixel at a noisy query is not automatically EXTERNAL_OCCLUDED: wrong position, background, out-of-frame/search and other pallets remain possible.')
    return write_json(out/'COMPLETE.json',result), result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='current_fullframe_inputs_v1')
    args=parser.parse_args()
    if Path(args.out).name != args.out or '..' in args.out:
        raise ValueError('Single output directory name required')
    complete,result=probe(safe(W/args.out,True))
    print(json.dumps({'PASS':result['PASS'],'frames':result['accessible_verified_frames'],
        'queries':result['queries'],'failures':result['failures'],'COMPLETE':complete},ensure_ascii=False))


if __name__=='__main__':
    main()
