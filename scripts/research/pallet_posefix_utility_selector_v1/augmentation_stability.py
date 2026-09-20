"""Bounded GT-free GPU augmentation-stability audit; frozen labels stay unchanged."""
import argparse
import copy
import json
import math
import time

import cv2
import numpy as np
import torch

from . import core as P
from . import all8_replay_frame_filter as A
from .x_crossing_filter import stats

DOC = P.DOC / 'augmentation_stability'
RAW = P.RAW / 'augmentation_stability'
VARIANTS = ['identity', 'brightness085', 'brightness115', 'resize090', 'resize110', 'crop_left_top', 'crop_right_bottom']
RETENTIONS = [.8, .6, .9]
PROTOCOL = DOC / 'PROTOCOL_RUNTIME_CORRECTED.json'


def transform(image, variant):
    h, w = image.shape[:2]
    affine = np.eye(3)
    if variant == 'identity':
        return image.copy(), affine
    if variant.startswith('brightness'):
        factor = .85 if variant == 'brightness085' else 1.15
        return np.clip(image.astype(float)*factor, 0, 255).round().astype(np.uint8), affine
    if variant.startswith('resize'):
        factor = .9 if variant == 'resize090' else 1.1
        nw, nh = round(w*factor), round(h*factor)
        affine[0,0], affine[1,1] = nw/w, nh/h
        affine[:2,2] = (np.array([nw/w, nh/h])-1)/2
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR), affine
    dx, dy = max(1, round(.02*w)), max(1, round(.02*h))
    if variant == 'crop_left_top':
        affine[:2,2] = [-dx, -dy]
        return image[dy:, dx:].copy(), affine
    assert variant == 'crop_right_bottom'
    return image[:h-dy, :w-dx].copy(), affine


def map_points(points, affine):
    p = np.asarray(points, float)
    return np.c_[p, np.ones(len(p))].dot(affine.T)[:,:2]


def instability(original, variants, diagonal):
    p, q = np.asarray(original, float)[:8], np.asarray(variants, float)[:,:8]
    assert p.shape == (8,2) and q.shape[1:] == (8,2)
    assert np.isfinite(p).all() and np.isfinite(q).all() and diagonal > 0
    shifts = np.linalg.norm(q-p[None], axis=-1)
    rms = np.sqrt(np.mean(shifts**2, axis=0))
    return dict(score=float(rms.max()/diagonal), max_corner_rms_px=float(rms.max()),
                per_corner_rms_px=rms.tolist(), per_variant_shift_px=shifts.tolist())


def rank_keep(rows, fraction, key='score'):
    n = math.floor(len(rows)*fraction)
    return {r['id'] for r in sorted(rows, key=lambda r: (r[key] is None, r[key] if r[key] is not None else 0, r['id']))[:n]}


def serial(value):
    if isinstance(value, dict): return {k: serial(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def prepare():
    parent = P.read(A.DOC/'OUTPUTS_LOCK.json')
    for binding in parent['artifacts']+[parent['protocol']]: P.verify_binding(binding)
    N = P.N
    fit = P.read(N.DOC/'FIT.json')
    protocol = dict(experiment='augmentation_stability_v1', scope='All179 existing confidence+all8-median-LOO survivors; no flip in this parent',
        disabled=['physical_shape_filter', 'x_crossing_filter'], variants=VARIANTS,
        inference='Each transformed RGB is rerun through frozen R0 and frozen Replay. Inverse-map coordinates. Top-confidence candidate each time, no GT matching.',
        primary_score='max over8 native corners of RMS displacement from original Replay across6 augmentations / original R0 box diagonal',
        missing='Missing or invalid predicted corners -> score null, sort most unstable',
        correspondence='Fixed native corner identity; NO symmetry/GT reassignment. Equivalent identity switches may count as instability.',
        output='Keep/drop whole image; saved original Replay prediction copied exactly, no averaging, fallback or replacement',
        primary_retention=.8, secondary_retention=[.6,.9], retention_rounding='floor per population; deterministic id tie-break',
        comparisons=['same-count lowest original all8 LOO', 'same-count highest R0 confidence', 'uniform random subset expectation'],
        evaluation='Decisions frozen before reading saved GT errors; PCK20 and bad20 image detection AUROC, matched-only separately',
        reuse='DEV unknown-origin reference and GREEN manual; reused development audit, not independent confirmation',
        threshold_tuning=False, training=False, promotion=False, output_coordinates_changed=False,
        resize_mapping='OpenCV pixel center: x_new=(x_old+0.5)*scale-0.5',
        crop='Remove2% left+top or right+bottom; do not use GT to choose crop',
        brightness='Multiply uint8 channels by0.85 or1.15; clip+round',
        parity_tolerance_px=.001, device='cuda', thermal_limit_C=80,
        precision='Historical R0 cuDNN TF32=True; Replay cuDNN TF32=False. matmul TF32=False for both. Restore before each model.',
        initial_parity_stop=P.bound(DOC/'INITIAL_PARITY_STOP.json'),
        sources=[P.bound(A.DOC/'OUTPUTS_LOCK.json'), P.bound(A.RAW/'DECISIONS.json'), P.bound(A.RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json'),
                 P.bound(N.RAW/'PREDICTIONS.json'), P.bound(N.DOC/'FIT.json'), fit['checkpoint'], P.bound(N.E.R0),
                 P.bound(N.C.__file__), P.bound(N.__file__), P.bound(P.ROOT/'scripts/research/pallet_line_pose_v1/features.py')],
        code=P.bound(__file__), tests=P.bound(P.HERE/'test_augmentation_stability.py'))
    for binding in protocol['sources']: P.verify_binding(binding)
    P.freeze(PROTOCOL, protocol)
    return protocol


@torch.no_grad()
def run():
    protocol = prepare()
    P.setup()
    assert torch.cuda.is_available(), 'This audit requires host CUDA access, not CPU fallback'
    gpu = P.N.E.gpu(); print('GPU', gpu, flush=True)
    model = P.N.load_model()
    extractor = P.N.E.old('features').FrozenYoloFeatures(P.N.E.R0)
    saved = P.read(P.N.RAW/'PREDICTIONS.json')
    parent = P.read(A.RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json')
    parent_decisions = P.read(A.RAW/'DECISIONS.json')
    start = time.monotonic(); done = 0; inference = {}; accepted = {}
    try:
        for ds, rows in parent.items():
            old = {r['id']:r for r in saved[ds]}
            loo = {r['id']:r['s_remove_all8'] for r in parent_decisions[ds]}
            inference[ds] = []
            for row in rows:
                P.verify_binding(row['image'])
                index = next(i for i,r in enumerate(rows) if r['id']==row['id'])
                cache = RAW/'frames'/f'{ds}_{index:03d}.json'
                if cache.exists():
                    result = P.read(cache)
                    assert result['id']==row['id'] and result['protocol_sha256']==P.bound(PROTOCOL)['sha256']
                else:
                    image = cv2.imread(str(P.ROOT/row['image']['path'])); assert image is not None
                    outputs = []; parity = {}; original = old[row['id']]['predictions']
                    for variant in VARIANTS:
                        aug, affine = transform(image, variant)
                        torch.backends.cudnn.allow_tf32 = True
                        captured = extractor.predict(aug)
                        torch.backends.cudnn.allow_tf32 = False
                        raw = dict(candidates=serial(captured['candidates']), selected_index=captured['selected_index'])
                        refined = P.N.C.predict(model, aug, raw)
                        top = A.N.top(refined)
                        valid = top is not None
                        if valid:
                            coords = np.asarray(top['keypoints_xy'], float)
                            valid = bool(np.isfinite(coords[:8]).all() and not (coords[:8]==-1).all(1).any())
                        restored = map_points(coords, np.linalg.inv(affine)).tolist() if valid else None
                        if variant == 'identity':
                            for arm, pred in [('R0',raw), ('POSEFIX_RAW',refined)]:
                                current, previous = A.N.top(pred), A.N.top(original[arm])
                                assert current is not None and previous is not None
                                delta = float(np.abs(np.asarray(current['keypoints_xy'])-previous['keypoints_xy']).max())
                                assert delta <= .001, (row['id'], arm, 'identity parity', delta)
                                assert abs(current['score']-previous['score']) <= 1e-6
                                assert np.max(np.abs(np.array(current['box_xyxy'])-previous['box_xyxy'])) <= .001
                                parity[arm] = delta
                        outputs.append(dict(variant=variant, valid=valid, affine=affine.tolist(), points_original_frame=restored,
                                            detector_score=None if top is None else top['score']))
                    candidate = A.N.top(original['R0'])
                    diagonal = float(np.linalg.norm(np.array(candidate['box_xyxy'])[2:]-np.array(candidate['box_xyxy'])[:2]))
                    values = [o['points_original_frame'] for o in outputs[1:]]
                    measures = instability(A.N.top(row['prediction'])['keypoints_xy'], values, diagonal) if all(v is not None for v in values) else dict(score=None, max_corner_rms_px=None)
                    result = dict(id=row['id'], image=row['image'], protocol_sha256=P.bound(PROTOCOL)['sha256'],
                                  **measures, negative_confidence=-candidate['score'], loo=loo[row['id']], outputs=outputs, identity_parity_px=parity)
                    P.freeze(cache, result)
                inference[ds].append(result); done += 1
                if done%10==0 or done==179:
                    print(json.dumps(dict(done=done,total=179,elapsed_seconds=time.monotonic()-start,gpu=P.N.E.gpu())),flush=True)
            accepted[ds] = {}
            for retention in RETENTIONS:
                selected = rank_keep(inference[ds],retention)
                accepted[ds][str(retention)] = [copy.deepcopy(r) for r in rows if r['id'] in selected]
        for binding in protocol['sources']+[protocol['code'],protocol['tests']]: P.verify_binding(binding)
        P.freeze(RAW/'INFERENCE.json', inference)
        P.freeze(RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json', accepted)
        P.freeze(DOC/'OUTPUTS_LOCK.json', dict(complete=True, decisions_before_GT=True, initial_gpu=gpu,
            protocol=P.bound(PROTOCOL), artifacts=[P.bound(RAW/p) for p in ['INFERENCE.json','ACCEPTED_UNCHANGED_PREDICTIONS.json']],
            evaluated_images=done, forward_variants_per_image=7, coordinates_changed=0))
    finally:
        extractor.close()


def auc(labels, scores):
    a = [s for y,s in zip(labels,scores) if y]; b = [s for y,s in zip(labels,scores) if not y]
    if not a or not b: return None
    return float(np.mean([float(x>y)+.5*float(x==y) for x in a for y in b]))


def score():
    lock = P.read(DOC/'OUTPUTS_LOCK.json')
    for binding in lock['artifacts']+[lock['protocol']]: P.verify_binding(binding)
    metric_source = P.N.RAW/'PER_FRAME_METRICS.json'
    parent_result = P.read(A.DOC/'RESULTS.json'); P.verify_binding(parent_result['metrics_source'])
    data = P.read(RAW/'INFERENCE.json'); metrics = P.read(metric_source)
    results = {}
    for ds, mode in [('DEV72','DEV72'),('GREEN150','GREEN150_MANUAL')]:
        rows = data[ds]; ids = {r['id'] for r in rows}
        mm = [r for r in metrics[mode]['POSEFIX_RAW'] if r['id'] in ids]
        byid = {r['id']:r for r in mm}; allstats = stats(mm)
        comparisons = {}
        for fraction in RETENTIONS:
            methods = {}
            for key in ['score','loo','negative_confidence']:
                kept = rank_keep(rows,fraction,key); rejected = ids-kept
                methods[key] = dict(kept=stats([r for r in mm if r['id'] in kept]),
                    rejected=stats([r for r in mm if r['id'] in rejected]), rejected_ids=sorted(rejected))
            comparisons[str(fraction)] = dict(methods=methods,random_expected_bad20_rejected=allstats['bad20_images']*(len(rows)-len(kept))/len(rows))
        discrimination = {}
        for matched in [False,True]:
            subset = [r for r in rows if not matched or byid[r['id']]['matched']]
            labels = [any(e>20 for e in byid[r['id']]['errors']) for r in subset]
            discrimination['matched_only' if matched else 'all'] = dict(images=len(subset), bad20=sum(labels),
                AUROC={key:auc(labels,[float('inf') if r[key] is None else r[key] for r in subset]) for key in ['score','loo','negative_confidence']})
        results[ds] = dict(before=allstats, comparisons=comparisons, discrimination=discrimination,
                           missing_variant_images=sum(r['score'] is None for r in rows))
    result = dict(complete=True, results=results, output_lock=P.bound(DOC/'OUTPUTS_LOCK.json'), metrics_source=P.bound(metric_source),
                  coordinates_changed=False, self_training=False, independent_confirmation=False, tuned=False)
    P.freeze(DOC/'RESULTS.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','score'])
    args=parser.parse_args();run() if args.stage=='run' else score()
