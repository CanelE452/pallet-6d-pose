"""Input-only original-Pallox-query pack. Actual pixels/arrays need root entry.

No cue, annotation, mask, pose, depth, head, checkpoint or detector import.
Only cached V90 P3 is sampled; no backbone or learned-model forward is called.
"""
from pathlib import Path
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import shutil
import sys
import time

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import torch

ROOT = Path(__file__).resolve().parent
V94 = ROOT.parent
P = V94.parent
V90 = P/'query_visibility_public_cue_inputs_20261023_v90'
INTAKE = V94/'pallox_actual_initializer/FIXED100'
MATH = V90/'producer/spatial_input_math.py'
MATH_SHA = '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
LEGACY = P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
V74 = P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
OUT = V94/'pallox_actual_query_input_pack'
PROTOCOL = ROOT/'INPUT_PACK_PROTOCOL_BEFORE_ARRAYS.json'
QUERY_FIELDS = {'query_id','frame_index','frame_input_id','pair_id','view','split','native_slot',
                'query_xy','roi_side','original_query_continuous_valid','query_nearest_pixel_valid',
                'query_rgb96_centre_domain_valid','reason','native_semantic_state','human_gold'}
FRAME_FIELDS = {'frame_input_id','image','image_sha256','pair_id','raw_hw','seed_V84_reuse',
                'session','source_family','split','view'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def check(desc):
    if sha(desc['path']) != desc['sha256']:
        raise ValueError('Binding changed: ' + desc['path'])
    return Path(desc['path'])


def readjson(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def math_module():
    assert sha(MATH) == MATH_SHA
    spec = importlib.util.spec_from_file_location('v90_frozen_input_math', MATH)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def metadata_join(frames, queries, records, original_frames):
    """Metadata only; do not reinterpret or choose a target/query using GT."""
    if not len(frames) == len(records) == len(original_frames) == 100 or len(queries) != 800:
        raise ValueError('All100 frames/all800 slots are mandatory')
    math = math_module()
    for i, (f, r, original) in enumerate(zip(frames, records, original_frames)):
        assert set(original) == FRAME_FIELDS
        assert set(f) == FRAME_FIELDS | {'frame_index','candidates','selected_index','input_shape',
                                        'canvas_shape','added_border','affine_gain','affine_offset',
                                        'V90_P3_exact','reusable_full_P3'}
        assert f['frame_index'] == r['frame_index'] == i
        for key in original:
            assert f[key] == r[key] == original[key]
        assert f['V90_P3_exact'] is True and r['P3_dtype'] == 'float16'
        assert f['reusable_full_P3'] == r['full_P3']
        a = math.expected_affine(f['raw_hw'])
        assert f['canvas_shape'] == r['canvas_shape'] == a['canvas_shape']
        assert f['input_shape'] == r['actual_input_shape'] == a['input_shape']
        assert f['added_border'] == r['added_border'] == 100
        assert f['affine_gain'] == r['gain'] == a['gain']
        assert np.array_equal(f['affine_offset'], r['offset']) and np.array_equal(r['offset'], a['offset'])
        assert r['P3_shape'] == [64] + a['p3_actual_hw']
        assert r['P3_zero_padding_archived'] is False
        selected = None if not f['candidates'] else int(np.argmax([c['score'] for c in f['candidates']]))
        assert f['selected_index'] == selected
        for k in range(8):
            q = queries[i*8+k]
            assert set(q) == QUERY_FIELDS
            assert q['frame_index'] == i and q['frame_input_id'] == f['frame_input_id']
            assert q['native_slot'] == k and q['query_id'] == f['frame_input_id']+f':native{k}'
            assert all(q[key] == f[key] for key in ['pair_id', 'view', 'split'])
            assert q['native_semantic_state'] == 'UNKNOWN' and q['human_gold'] is False
            if selected is None:
                assert q['query_xy'] is None and q['roi_side'] is None and q['reason'] == 'NO_DETECTION_U'
            else:
                assert q['query_xy'] == f['candidates'][selected]['keypoints_xy'][k]
                assert np.isfinite(q['query_xy']).all() and q['roi_side'] > 0
                x, y = q['query_xy']; h, w = f['raw_hw']
                assert q['original_query_continuous_valid'] == bool(0 <= x < w and 0 <= y < h)
                assert q['query_nearest_pixel_valid'] == bool(0 <= np.floor(x+.5) < w and 0 <= np.floor(y+.5) < h)
                assert q['query_rgb96_centre_domain_valid'] == bool(-.5 <= x < w-.5 and -.5 <= y < h-.5)


def pack_frame(rgb, p3, frame_queries, record):
    """Pure operator; even centre-invalid finite q keeps original crop and ROI."""
    math = math_module()
    assert rgb.dtype == np.uint8 and list(rgb.shape) == record['raw_hw']+[3]
    assert p3.dtype == np.float16 and list(p3.shape) == record['P3_shape']
    assert len(frame_queries) == 8
    present = np.array([r['query_xy'] is not None for r in frame_queries])
    rawq = np.full((8, 2), np.nan, np.float64); rawL = np.full(8, np.nan, np.float64)
    x = np.zeros((8, 65, 9, 9), np.float16)
    anyvalid = np.zeros(8, bool); valid9 = np.zeros((8, 9, 9), bool)
    rgbs = np.zeros((8, 3, 96, 96), np.uint8); valids = np.zeros((8, 1, 96, 96), bool)
    for k in np.flatnonzero(present):
        rawq[k] = frame_queries[k]['query_xy']; rawL[k] = frame_queries[k]['roi_side']
    if present.any():
        rois, av, vv = math.sample_p3(p3, rawq[present], rawL[present], record['raw_hw'],
                                    record['gain'], record['offset'], record['added_border'])
        x[present] = rois; anyvalid[present] = av; valid9[present] = vv
        for k in np.flatnonzero(present):
            rgbs[k], valids[k] = math.crop(rgb, rawq[k], rawL[k], 96)
    # Safe finite coordinates are only for factual validity computation. Raw null
    # query/L remain null in lineage and NaN in explicitly raw NPZ metadata.
    safeq = np.where(present[:, None], rawq, -1e8)
    centre, continuous, nearest = math.query_validity(safeq, record['raw_hw'])
    centre &= present; continuous &= present; nearest &= present
    rgb_centre = present & np.array([bool(r['query_rgb96_centre_domain_valid']) for r in frame_queries])
    central4 = valids[:, 0, 47:49, 47:49].reshape(8, -1).all(1)
    assert np.array_equal(x[:, 64].astype(bool), valid9)
    assert np.array_equal(continuous, [r['original_query_continuous_valid'] for r in frame_queries])
    assert np.array_equal(nearest, [r['query_nearest_pixel_valid'] for r in frame_queries])
    return dict(p3_roi=x, rgb96=rgbs, crop_valid96=valids, query_xy_raw=rawq, L_raw=rawL,
                query_present=present, query_continuous_valid=continuous, query_nearest_valid=nearest,
                query_center_valid=centre, query_rgb96_centre_domain_valid=rgb_centre,
                ROI_anyvalid=anyvalid, ROI_center_valid=valid9[:, 4, 4], RGB_central4_valid=central4,
                operator_centre_support_v94_draft=valid9[:, 4, 4] & central4,
                original_query_continuous_valid=np.array([r['original_query_continuous_valid'] for r in frame_queries]),
                original_reason=np.array([r['reason'] for r in frame_queries]),
                query_id=np.array([r['query_id'] for r in frame_queries]),
                native_slot=np.arange(8, dtype=np.int8), frame_index=np.full(8, record['frame_index'], np.int32),
                frame_input_id=np.array([record['frame_input_id']]*8), split=np.array([record['split']]*8),
                image_sha256=np.array([record['image_sha256']]*8))


def freeze():
    """Only code and previously saved source JSON metadata are read here."""
    paths = dict(intake_seal=INTAKE/'INTAKE_SEAL.json', intake_result=INTAKE/'RESULT.json',
                 proposals=INTAKE/'FROZEN_PROPOSALS_ALL100.jsonl', queries=INTAKE/'ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl',
                 original_frames=P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl',
                 p3_records=V90/'input_pack/FRAME_FULL_P3_RECORDS.jsonl', prior_pack_seal=V90/'input_pack/INPUT_PACK_SEAL.json',
                 geometry_protocol=V90/'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json')
    seal = readjson(paths['intake_seal']); result = readjson(paths['intake_result'])
    assert seal['completed'] and seal['PASS'] and result['completed'] and result['PASS']
    assert result['frames'] == 100 and result['slots'] == 800 and result['all100_P3_exact_V90']
    for d in seal['files']: check(d)
    metadata_join(lines(paths['proposals']), lines(paths['queries']), lines(paths['p3_records']), lines(paths['original_frames']))
    generated = readjson(ROOT/'GENERATED_PACK_TESTS.json'); assert generated['PASS']
    write(PROTOCOL, dict(completed=True, schema='V94_ACTUAL_PALLOX800_QUERY_INPUT_PACK_BEFORE_ARRAYS',
        code=bind(Path(__file__)), generated=bind(ROOT/'GENERATED_PACK_TESTS.json'),
        math_code=bind(MATH), legacy_roi_code=bind(LEGACY), crop_reference_code=bind(V74),
        bindings={k:bind(v) for k,v in paths.items()}, source_frames=100, slots=800,
        original_reason_counts=result['reason_counts'], source_splits={'TRAIN_LAB':44,'VAL_HALL':56},
        source='Actual frozen initializer firstargmax q0..7; no centroid8/no GTbest/no emulation/no qL modification.',
        P3='Reuse only exact-bound cached actual FP16 64x68x80, temporary80pad in RAM, V90 sample_p3 CPU FP32 grid_sample; no new backbone.',
        RGB='Canonical PNG decode SHA exact against V90 record; exact V74 crop AST, uint8/valid96. Finite out-of-bounds q retain actual crop/ROI. Absent q only gets zero tensors.',
        metadata_nulls='Missing rawq/L: JSONnull and NPZ raw NaN; modeltensors remain finite. Native logical slots are lineage only, not semantic IDs or head inputs.',
        validity='Keep original continuous flag/reason, continuous/nativeNN, combinedcentre, RGBcentre-domain, RGBvalid96, ROIany/centre, RGBcentral4 and draftoperator support all separately. None is a visibility label.',
        runtime={'python':'/home/minjae/anaconda3/envs/pallet-yolo26/bin/python','torch':'2.1.1+cu118','device':'CPU','threads':1,'PIL_decode':'Decoded RGB bytes must match V90 canonical SHA; no tolerance.'},
        output=str(OUT), chunk_queries=128, byte_budget=100000000, minimum_free_before_build=150000000,
        no_GT_pose_mask_cue_label_or_semantic_native_mapping=True, no_backbone_or_head_or_weights=True,
        no_current_fit_PnP=True, training_admission=False))
    write(ROOT/'ROOT_ENTRYLOCK_TEMPLATE.json',dict(root_approved=False, purpose='INPUT_PACK_ONLY_NO_MODEL_OR_GT',
        code=bind(Path(__file__)),protocol=bind(PROTOCOL),generated=bind(ROOT/'GENERATED_PACK_TESTS.json'),
        independent_review={'path':'ROOT_FILL_ACTUAL_REVIEW_PATH','sha256':'ROOT_FILL_EXACT_SHA'},output=str(OUT)))


def run(entry_path):
    # Root entry, exact code/protocol/proof/review before opening ANY RGB/P3 body.
    entry = readjson(entry_path)
    assert entry['root_approved'] is True and entry['purpose'] == 'INPUT_PACK_ONLY_NO_MODEL_OR_GT'
    assert check(entry['code']).resolve() == Path(__file__).resolve()
    assert check(entry['protocol']).resolve() == PROTOCOL.resolve()
    assert check(entry['generated']).resolve() == (ROOT/'GENERATED_PACK_TESTS.json').resolve()
    review = readjson(check(entry['independent_review'])); assert review['completed'] and review['PASS']
    assert entry['output'] == str(OUT) and not OUT.exists()
    protocol = readjson(PROTOCOL)
    for d in [protocol['code'], protocol['generated'], protocol['math_code'], protocol['legacy_roi_code'], protocol['crop_reference_code']]:check(d)
    for d in protocol['bindings'].values():check(d)
    assert sys.version_info[:2] == (3,10) and torch.__version__ == '2.1.1+cu118'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    assert shutil.disk_usage(V94).free >= protocol['minimum_free_before_build']
    frames = lines(protocol['bindings']['proposals']['path']); queries = lines(protocol['bindings']['queries']['path'])
    records = lines(protocol['bindings']['p3_records']['path']); originals=lines(protocol['bindings']['original_frames']['path'])
    metadata_join(frames, queries, records, originals)
    OUT.mkdir(); (OUT/'chunks').mkdir(); start=time.monotonic()
    write(OUT/'STARTED.json',dict(entry=bind(Path(entry_path)),protocol=bind(PROTOCOL),arrays_opened_before_entry=0))
    chunks=[]; buffer=[]; lineage=[]; firstproof={}; counts=Counter()
    for i,(frame,record) in enumerate(zip(frames, records)):
        assert sha(frame['image']) == frame['image_sha256']
        rgb=np.array(Image.open(frame['image']).convert('RGB'),copy=True)
        assert list(rgb.shape)==frame['raw_hw']+[3]
        assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==record['decoded_RGB_sha256']
        with np.load(check(record['full_P3']),allow_pickle=False) as data:
            expected={'p3','original_hw','added_border','canvas_shape','input_shape','gain','offset','p3_actual_hw','image_sha256','decoded_RGB_sha256','backbone_sha256','frame_input_id'}
            assert set(data.files)==expected
            for key, val in [('original_hw',record['raw_hw']),('canvas_shape',record['canvas_shape']),('input_shape',record['actual_input_shape']),('p3_actual_hw',record['P3_shape'][1:]),('offset',record['offset'])]:assert np.array_equal(data[key],val)
            assert float(data['gain'])==record['gain'] and int(data['added_border'])==100
            assert str(data['image_sha256'])==record['image_sha256'] and str(data['decoded_RGB_sha256'])==record['decoded_RGB_sha256']
            assert str(data['frame_input_id'])==record['frame_input_id']
            assert str(data['backbone_sha256'])=='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
            pack=pack_frame(rgb,data['p3'],queries[i*8:i*8+8],record)
            if i==0:
                repeat=pack_frame(rgb,data['p3'],queries[:8],record)
                assert all(np.array_equal(pack[k],repeat[k],equal_nan=True) if pack[k].dtype.kind=='f' else np.array_equal(pack[k],repeat[k]) for k in pack)
                firstproof={'same_saved_arrays_operator_repeat_bitexact':True,'new_model_calls':0}
        buffer.append(pack)
        for k in range(8):
            qi=queries[i*8+k]; lineage.append(dict(query_index=i*8+k,**qi,image=frame['image'],image_sha256=frame['image_sha256'],raw_hw=frame['raw_hw'],selected_index=frame['selected_index'],full_P3=record['full_P3']))
            counts[qi['reason']]+=1
        if len(buffer)==16 or i==99:
            merged={k:np.concatenate([x[k]for x in buffer])for k in buffer[0]}
            begin=i*8+8-len(merged['query_id']);end=i*8+8
            cp=OUT/'chunks'/f'queries_{begin:04d}_{end:04d}.npz';np.savez_compressed(cp,**merged)
            chunks.append({'start':begin,'end':end,'count':end-begin,'file':bind(cp)});buffer=[]
        if i%16==0:print(json.dumps({'completed_frames':i+1,'total':100}),flush=True)
    lp=OUT/'QUERY_INPUT_LINEAGE_ONLY.jsonl';lp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n'for r in lineage))
    assert len(lineage)==800 and dict(counts)==protocol['original_reason_counts']
    manifest=OUT/'PACK_MANIFEST.json';write(manifest,dict(completed=True,frame_count=100,query_count=800,protocol=bind(PROTOCOL),chunks=chunks,lineage=bind(lp),tensor_fields=['p3_roi','rgb96','crop_valid96'],all800_order_kept=True,no_semantic_labels=True,training_admission=False))
    qa=OUT/'INPUT_PACK_QA.json';write(qa,dict(completed=True,PASS=True,manifest=bind(manifest),original_counts=dict(counts),first_frame_repeat=firstproof,canonical_RGB_SHA_all100_exact=True,all_P3_archives_exactbound=True,actual_P3_array_reads=100,actual_RGB_decodes=100,new_backbone_head_fit_current_PnP=0,runtime={'torch':torch.__version__,'PIL':PIL_VERSION,'CPU_threads':1},seconds=time.monotonic()-start))
    total=sum(f.stat().st_size for f in OUT.rglob('*')if f.is_file());assert total+10000<protocol['byte_budget']
    write(OUT/'INPUT_PACK_SEAL.json',dict(completed=True,PASS=True,entry=bind(Path(entry_path)),protocol=bind(PROTOCOL),manifest=bind(manifest),QA=bind(qa),chunks=[c['file']for c in chunks],lineage=bind(lp),frame_count=100,query_count=800,bytes_before_seal=total,source_only=True,training_allowed=False,current_allowed=False,new_backbone_head_fit_PnP=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);parser.add_argument('--entry');args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        if not args.entry:parser.error('Actual source pixels/arrays require approved root entry')
        run(args.entry)
