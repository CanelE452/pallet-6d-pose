"""Scenario-disjoint synthetic utility data with genuinely frozen candidates.

Only synthetic GT enters ``utility_targets``. RGB/candidate construction and
feature extraction receive no targets. Real image identities are read solely
to exclude their hashes; no real annotation target is read or used here.
"""
from __future__ import annotations

from collections import Counter
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from . import core as P
from scripts.research.pallet_posefix_replay_v1.source import SourceData
from scripts.research.pallet_posefix_replay_v1 import core as N

ROOT = P.ROOT
DCP_RAW = ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1'
DCP_DOC = ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1'
REAL_SPLIT = ROOT / '_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json'
GREEN = ROOT / '_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'
N2_PREDICTIONS = DCP_RAW / 'predictions/SYNTH_HELDOUT/N2_DIM_ONLY_seed1.json'
PAIRS = np.asarray([[0, 3], [1, 2], [4, 7], [5, 6]], dtype=np.int64)
PARITY_ATOL_PX = .001
CHUNK_ROWS = 32


def row_record(data, row):
    return data.source['records'][int(data.indices[int(row)])]


def cache_stats(data):
    result = {}
    for key, spec in data.manifest['arrays'].items():
        path = data.directory / spec['file']
        stat = path.stat()
        result[key] = dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def prepare_split():
    """No inference or scoring: fixed RNG8101 scenario split, metadata only."""
    source = SourceData()
    data = source.data
    record = lambda row: row_record(data, row)
    orders = np.load(N.RAW / 'ORDERS.npz')
    previous = np.load(N.E.C.BRAW / 'order_seed1.npy')
    assert previous.shape == (6000, 16)
    n2_fit = P.read(DCP_DOC / 'fits/N2_DIM_ONLY_seed1.json')
    assert hashlib.sha256(previous.astype('<i8').tobytes()).hexdigest() == n2_fit['batch_order_sha256']
    prior_fit = P.read(ROOT / '_docs/experiments/pallet_sensors_submission_v1/PRIOR1_COMPLETE.json')
    P.verify_binding(prior_fit['order_file'])
    assert prior_fit['order_file'] == P.bound(N.E.C.BRAW / 'order_seed1.npy')
    assert np.array_equal(np.unique(previous), np.sort(source.train_rows))
    assert len(source.train_rows) == 55915 and len(source.held_rows) == 1978
    assert not np.intersect1d(source.held_rows, previous).size
    assert not np.intersect1d(source.held_rows, orders['source_rows']).size
    probe_groups = {record(row)['scenario_id'] for row in orders['held_rows']}
    pool = np.asarray([row for row in source.held_rows
                       if record(row)['scenario_id'] not in probe_groups], dtype=np.int64)
    groups = sorted({record(row)['scenario_id'] for row in pool})
    assert len(pool) == 1616 and len(groups) == 1254 and len(probe_groups) == 249
    assert hashlib.sha256(pool.astype('<i8').tobytes()).hexdigest() == 'bd8a12089b8dfb78aad6234288f624cd569af186273b524a9e80b2fd428e8bda'
    shuffled = np.random.default_rng(8101).permutation(groups).tolist()
    train_groups = set(shuffled[:int(.8 * len(groups))])
    val_groups = set(shuffled[int(.8 * len(groups)):])
    assert not train_groups & val_groups and train_groups | val_groups == set(groups)
    train_rows = [int(row) for row in pool if record(row)['scenario_id'] in train_groups]
    val_rows = [int(row) for row in pool if record(row)['scenario_id'] in val_groups]

    # Metadata-only exclusion check; no real GT coordinate file is opened.
    real = P.read(REAL_SPLIT)
    green = P.read(GREEN)
    forbidden = real['train'] + real['evaluation'] + green['records']
    assert len(real['train']) == 9 and len(real['evaluation']) == 72 and len(green['records']) == 150
    forbidden_hashes = {r['image']['sha256'] for r in forbidden}
    train_source_hashes = {record(row)['image_sha256'] for row in source.train_rows}
    train_source_groups = {record(row)['scenario_id'] for row in source.train_rows}
    by_hash, records = {}, []
    for row in pool:
        r = record(row)
        assert r['source_kind'] == 'synthetic' and r['source_split'] == 'val' and r['partition'] == 'heldout'
        assert r['scenario_id'] not in train_source_groups
        assert r['image_sha256'] not in forbidden_hashes | train_source_hashes
        if r['image_sha256'] in by_hash:
            assert by_hash[r['image_sha256']] == r['scenario_id'], 'Same pixels assigned to different scenarios'
        by_hash[r['image_sha256']] = r['scenario_id']
        binding = P.bound(Path(r['image']))
        assert binding['sha256'] == r['image_sha256']
        baseline = DCP_RAW / f'source_baseline/{int(row):05d}.json'
        records.append(dict(row=int(row), id=r['id'], scenario=r['scenario_id'],
                            split='train' if int(row) in train_rows else 'val',
                            image=binding, source=r['source'], baseline=P.bound(baseline)))
    assert not ({r['image']['sha256'] for r in records if r['split'] == 'train'} &
                {r['image']['sha256'] for r in records if r['split'] == 'val'})
    side = np.load(DCP_RAW / 'DIMENSION_SIDECAR.npz')
    assert np.array_equal(side['record_index'], data.indices)
    historical_cache_stats = P.read(DCP_DOC / 'SOURCE_BINDINGS.json')['cache_stat']
    current_cache_stats = cache_stats(data)
    assert current_cache_stats == historical_cache_stats
    files = [data.run_dir / 'SOURCE_MANIFEST.json', data.directory / 'CACHE_MANIFEST.json',
             data.directory / 'CACHE_COMPLETE.json', N.RAW / 'ORDERS.npz',
             N.E.C.BRAW / 'order_seed1.npy', DCP_RAW / 'DIMENSION_SIDECAR.npz',
             DCP_DOC / 'DIM_NORMALIZATION_LOCK.json', DCP_DOC / 'SOURCE_BINDINGS.json',
             DCP_DOC / 'CALIBRATION_AND_SELECTION.json', DCP_DOC / 'fits/N2_DIM_ONLY_seed1.json',
             DCP_DOC / 'SYNTH_DETECTION_AUDIT.json', N2_PREDICTIONS, REAL_SPLIT, GREEN,
             ROOT / '_docs/experiments/pallet_sensors_submission_v1/TRAIN_CODE_LOCK.json',
             ROOT / '_docs/experiments/pallet_sensors_submission_v1/PRIOR1_COMPLETE.json',
             ROOT / '_docs/experiments/pallet_sensors_submission_v1/PRIOR_SOURCE_RESULTS.json',
             DCP_DOC / 'SYNTH_HELDOUT_RESULTS.json']
    return dict(complete=True, seed=8101, train_fraction_scenarios=.8,
                split_method='default_rng(8101).permutation(sorted(scenario_ids)); first floor(.8*1254) groups train',
                pool_rows=pool.tolist(), train_rows=train_rows, val_rows=val_rows, records=records,
                counts=dict(pool_rows=len(pool), train_rows=len(train_rows), val_rows=len(val_rows),
                            pool_scenarios=len(groups), train_scenarios=len(train_groups),
                            val_scenarios=len(val_groups), variants_per_row=2,
                            source_examples=2*len(pool), train_examples=2*len(train_rows),
                            val_examples=2*len(val_rows),
                            native_GT_corners=int(data.arrays['gt_valid'][pool, :8].sum()),
                            symmetry_orders={str(k):int(v) for k,v in Counter(side['order'][pool].tolist()).items()}),
                exclusions=dict(real9_images=9,real_DEV72_images=72,green_images=150,
                                any_real_hash_overlap=False, generator_gradient_overlap=False,
                                replay_probe_rows=256,replay_probe_scenarios=249,
                                retained_original_heldout_rows=1978),
                native_GT_groups='Per-row approved sidecar permutations[:order], one N2-anchored branch for supervision only',
                selected_cache_signature=N.source_cache_signature(data,pool),
                selected_sidecar_signature={key:N.array_sha(side[key][pool]) for key in side.files},
                cache_arrays_stat=current_cache_stats,bindings=[P.bound(path) for path in files],
                pool_row_i8_sha256=hashlib.sha256(pool.astype('<i8').tobytes()).hexdigest(),
                caveats=['R0 used original validation/checkpoint selection; not globally unseen data',
                         'Historical N2/PRIOR/source development evaluation exists; not new independent confirmation',
                         'All 55915 usable original training rows seen by N2 and PRIOR1; none used here',
                         'No C4 training images; no claim of square-green-domain coverage',
                         'P0/TEX versions share scenario_id and cannot cross train/validation'])


def stress_points(points, valid, box, row):
    """No GT: move two vertical pairs coherently; fixed per-row RNG."""
    result = np.asarray(points, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    diagonal = float(np.linalg.norm(np.asarray(box)[2:] - np.asarray(box)[:2]))
    rng = np.random.default_rng(np.random.SeedSequence([8102, int(row)]))
    pairs = rng.choice(len(PAIRS), size=2, replace=False)
    angles = rng.uniform(0, 2*np.pi, 2)
    radii = rng.uniform(.05, .15, 2) * diagonal
    for pair, angle, radius in zip(pairs, angles, radii):
        indices = PAIRS[pair][valid[PAIRS[pair]]]
        result[indices] += np.array([math.cos(angle), math.sin(angle)]) * radius
    assert np.array_equal(result[8], points[8])
    return result


def top(prediction):
    return prediction['candidates'][prediction['selected_index']]


def prepared_prediction(baseline):
    """Historical source JSON is raw-image coordinates; RGB has 100px border."""
    prediction = copy.deepcopy(baseline)
    for candidate in prediction['candidates']:
        candidate['keypoints_xy'] = (np.asarray(candidate['keypoints_xy'],dtype=np.float64)+100).tolist()
        candidate['box_xyxy'] = (np.asarray(candidate['box_xyxy'],dtype=np.float64)+100).tolist()
    return prediction


def load_n2():
    from scripts.research.pallet_dim_conditioned_p_v1 import refiner as D
    fit = P.read(DCP_DOC / 'fits/N2_DIM_ONLY_seed1.json')
    P.verify_binding(fit['checkpoint'])
    ck = torch.load(ROOT / fit['checkpoint']['path'], map_location='cpu', weights_only=False)
    assert ck['complete'] and ck['step'] == 6000 and ck['baseline_checkpoint_sha256'] == N.E.R0_SHA
    model = D.model('N2_DIM_ONLY',ck['config'])
    model.load_state_dict(ck['model_state_dict'])
    del ck
    return model.cuda().eval().requires_grad_(False)


@torch.inference_mode()
def n2_candidate(head, data, row, original_points, variant, raw_hw, gain, offset):
    """Frozen visual features, actual forward on each changed point input."""
    from scripts.research.pallet_dim_conditioned_p_v1 import refiner as D
    batch = data.batch(np.asarray([row]),'N2_DIM_ONLY',device='cuda',supervision=False)
    input_points = (np.asarray(data.arrays['points'][row],dtype=np.float32).copy() if variant == 0
                    else (np.asarray(original_points)*gain+offset).astype(np.float32))
    batch['points'] = torch.from_numpy(input_points[None]).cuda()
    output = D.forward(head,batch)
    selection = n2_candidate.selection
    rule = selection['rule']
    temperature = selection['temperatures']['N2_DIM_ONLY_seed1']['temperature']
    fraction = rule['max_move_image_diagonal_fraction']
    # Historic source decoder caps in original raw-image pixels, excluding border.
    cap = None if fraction is None else fraction*math.hypot(*raw_hw)*gain
    decoded = D.decode(output,temperature,rule['lam'],cap)[0].cpu().numpy()
    delta = decoded.astype(np.float64)-input_points.astype(np.float64)
    candidate = np.asarray(original_points,dtype=np.float64).copy()
    changed = np.isfinite(input_points).all(-1)&np.any(delta != 0,axis=-1)
    changed[8] = False
    if rule['lam'] != 0:
        candidate[changed] += delta[changed]/gain
    return candidate


def source_bindings():
    return dict(protocol=P.bound(P.DOC/'PROTOCOL.json'),split=P.bound(P.DOC/'SPLIT.json'),
                generator_code=P.bound(Path(__file__)),feature_code=P.bound(P.HERE/'model.py'),
                N2_checkpoint=P.read(DCP_DOC/'fits/N2_DIM_ONLY_seed1.json')['checkpoint'],
                Replay_checkpoint=P.read(N.DOC/'FIT.json')['checkpoint'],
                original_N2_predictions=P.bound(N2_PREDICTIONS))


def source_arrays(records):
    keys = records[0].keys()
    assert all(set(row)==set(keys) for row in records)
    return {key:np.stack([record[key] for record in records]) for key in keys}


@torch.inference_mode()
def generate_source():
    """Run only after source split, code and protocol are frozen by the root."""
    from .model import features, utility_targets
    from scripts.research.pallet_dim_conditioned_p_v1.data import PaperData
    from scripts.research.pallet_posefix_corner_gate_v1.infer import refine_summary
    P.setup();protocol=P.verify();split=P.read(P.DOC/'SPLIT.json');bindings=source_bindings()
    done=P.DOC/'SOURCE_COMPLETE.json'
    if done.exists():
        result=P.read(done);assert result['complete'] and result['bindings']==bindings
        P.verify_binding(result['artifact'])
        for b in result['chunk_receipts']:P.verify_binding(b)
        print('SOURCE_ALREADY_COMPLETE',result['examples'],flush=True);return result
    assert protocol['stress_seed']==8102 and protocol['split_seed']==8101
    data=PaperData();source=SourceData(data.base);rows=np.asarray(split['pool_rows'],dtype=np.int64)
    assert cache_stats(data.base)==split['cache_arrays_stat']
    assert N.source_cache_signature(data.base,rows)==split['selected_cache_signature']
    assert {key:N.array_sha(data.side[key][rows]) for key in data.side.files}==split['selected_sidecar_signature']
    metas={int(r['row']):r for r in split['records']}
    assert len(rows)==len(metas)==1616 and np.isin(rows,source.held_rows).all()
    n2_original={r['id']:r for r in P.read(N2_PREDICTIONS)['records']}
    n2_candidate.selection=P.read(DCP_DOC/'CALIBRATION_AND_SELECTION.json')
    gpu_checks=[P.gpu()];head=load_n2();replay=N.load_model();started=time.monotonic()
    chunks=[];receipts=[];parity_max=0.;parity_count=0;processed=0
    try:
        for start in range(0,len(rows),CHUNK_ROWS):
            current_rows=rows[start:start+CHUNK_ROWS]
            dst=P.RAW/'source_chunks'/f'{start:05d}.npz'
            receipt_path=dst.with_suffix('.json')
            if receipt_path.exists():
                receipt=P.read(receipt_path)
                assert receipt['bindings']==bindings and receipt['rows']==current_rows.tolist()
                P.verify_binding(receipt['artifact'])
                for row in current_rows:
                    P.verify_binding(metas[int(row)]['image']);P.verify_binding(metas[int(row)]['baseline'])
            else:
                assert not dst.exists(),'Unreceipted incomplete chunk; preserve it and inspect before resuming'
                examples=[];chunk_parity=0.
                for row0 in current_rows:
                    row=int(row0);meta=metas[row];record=row_record(data.base,row)
                    assert record['id']==meta['id'] and record['scenario_id']==meta['scenario']
                    P.verify_binding(meta['image']);P.verify_binding(meta['baseline'])
                    image=cv2.imread(str(ROOT/meta['image']['path']))
                    assert image is not None and list(image.shape[:2])==record['prepared_shape_hw']
                    baseline=P.read(ROOT/meta['baseline']['path'])
                    assert baseline['id']==record['id'] and baseline['cache_adapter_exact']
                    base=prepared_prediction(baseline);selected=top(base)
                    points=np.asarray(selected['keypoints_xy'],dtype=np.float64)
                    box=np.asarray(selected['box_xyxy'],dtype=np.float64)
                    valid=np.asarray(data.arrays['point_valid'][row],dtype=bool)&np.isfinite(points).all(-1)&~(points==-1).all(-1)
                    gain,offset=N.E.old('features').canvas_affine(record['prepared_shape_hw'],data.arrays['input_shape'][row])
                    assert np.allclose((points*gain+offset).astype(np.float32),data.arrays['points'][row],atol=1e-4,rtol=0)
                    D=float(np.linalg.norm(box[2:]-box[:2]));assert D>0 and np.isfinite(D)
                    for variant in (0,1):
                        current=points.copy() if variant==0 else stress_points(points,valid,box,row)
                        current_pred=copy.deepcopy(base);top(current_pred)['keypoints_xy']=current.tolist()
                        n2=n2_candidate(head,data,row,current,variant,record['raw_shape_hw'],gain,offset)
                        if variant==0:
                            old=np.asarray(top(n2_original[record['id']])['keypoints_xy'],dtype=np.float64)+100
                            np.testing.assert_allclose(n2,old,atol=PARITY_ATOL_PX,rtol=0)
                            chunk_parity=max(chunk_parity,float(np.max(np.abs(n2-old))))
                        corrected,heatmap=refine_summary(replay,image,current_pred)
                        assert heatmap is not None
                        candidate=np.asarray(top(corrected)['keypoints_xy'],dtype=np.float64)
                        np.testing.assert_array_equal(n2[8],points[8]);np.testing.assert_array_equal(candidate[8],points[8])
                        # No GT is passed to either generator or this feature call.
                        x=features(image,current,n2,candidate,box,valid,heatmap)
                        # Supervision begins ONLY after candidates and features exist.
                        gt=(np.asarray(data.arrays['gt_points'][row],dtype=np.float64)-offset)/gain
                        gt_valid=np.asarray(data.arrays['gt_valid'][row],dtype=bool)&np.isfinite(gt).all(-1)
                        gt_valid[8]=False
                        order=int(data.side['order'][row]);permutations=data.side['permutations'][row,:order]
                        target=utility_targets(n2,candidate,gt,gt_valid,permutations,valid,D)
                        x.update(target=target['target'],mask=target['mask'],n2=n2,replay=candidate,
                                 gt_aligned=target['gt_aligned'],gt_valid=target['gt_valid'],valid=valid,
                                 D=np.float64(D),row=np.int64(row),variant=np.int8(variant),
                                 train=np.bool_(meta['split']=='train'),branch=np.int64(target['branch']))
                        examples.append(x)
                    processed+=1
                    if processed%50==0:
                        check=P.gpu();gpu_checks.append(check)
                        print(json.dumps(dict(stage='GENERATE_SOURCE',rows=processed,total=1616,
                                              seconds=time.monotonic()-started,gpu=check)),flush=True)
                arrays=source_arrays(examples)
                assert len(arrays['row'])==2*len(current_rows)
                P.save_npz(dst,**arrays)
                receipt=dict(complete=True,rows=current_rows.tolist(),examples=len(examples),bindings=bindings,
                             artifact=P.bound(dst),clean_N2_parity_max_abs_px=chunk_parity,
                             clean_N2_parity_count=len(current_rows),GT_in_features=False)
                P.freeze(receipt_path,receipt)
            chunks.append(dst);receipts.append(P.bound(receipt_path))
            parity_max=max(parity_max,receipt['clean_N2_parity_max_abs_px'])
            parity_count+=receipt['clean_N2_parity_count']
    finally:
        del head,replay
        gc.collect();torch.cuda.empty_cache()
    gpu_checks.append(P.gpu())
    pieces=[np.load(path) for path in chunks]
    arrays={key:np.concatenate([piece[key] for piece in pieces]) for key in pieces[0].files}
    for piece in pieces:piece.close()
    assert len(arrays['row'])==3232 and parity_count==1616
    np.testing.assert_array_equal(arrays['row'][::2],rows)
    np.testing.assert_array_equal(arrays['row'][1::2],rows)
    assert np.array_equal(arrays['variant'],np.tile([0,1],1616))
    assert np.array_equal(arrays['train'][::2],arrays['train'][1::2])
    assert int(arrays['train'].sum())==split['counts']['train_examples']
    destination=P.RAW/'SOURCE.npz'
    P.save_npz(destination,**arrays)
    result=dict(complete=True,examples=3232,source_rows=1616,train_examples=int(arrays['train'].sum()),
                validation_examples=int((~arrays['train']).sum()),bindings=bindings,
                artifact=P.bound(destination),chunk_receipts=receipts,
                fields={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()},
                clean_N2_parity_count=parity_count,clean_N2_parity_max_abs_px=parity_max,
                parity_tolerance_px=PARITY_ATOL_PX,GT_in_features=False,GT_used_for_utility_supervision_only=True,
                generators_frozen=True,real_images_used=False,gpu_checks=gpu_checks,
                elapsed_seconds=time.monotonic()-started)
    P.freeze(done,result)
    print(json.dumps(dict(stage='SOURCE_COMPLETE',examples=3232,train=result['train_examples'],
                          validation=result['validation_examples'],N2_parity=parity_max,
                          seconds=result['elapsed_seconds'])),flush=True)
    return result


if __name__=='__main__':
    generate_source()
