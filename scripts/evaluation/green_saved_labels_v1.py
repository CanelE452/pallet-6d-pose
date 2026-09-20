"""Authorized saved-label GREEN150 evaluation; no assertion of finished label QA.

Separate version from the immutable final_dimension_release implementation.
Uses its frozen R0/N0/N2 models and decoding, but snapshots exactly the proposed
150 IDs. Calibration discrepancies are reported, never repaired. Only 2D is run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.evaluation import final_dimension_release as F

ROOT = F.ROOT
DOC = F.DOC / 'green150_saved_labels_v1'
RAW = F.RAW / 'green150_saved_labels_v1'
DATASET = DOC / 'DATASET_SNAPSHOT.json'
PROTOCOL = dict(
    status='ADDITIONAL_2D_SAVED_LABEL_EVALUATION_NOT_FINAL_HUMAN_QA',
    user_authorization='2026-09-19: include green pallet using available data; do not drop green from paper',
    annotation_review_complete=False, independent_confirmation=False,
    model='unchanged frozen R0 and N0/N2 seeds 1/2/3; no new training',
    dimensions='registry canonical 1.1 x 1.1 x 0.15 m; external input, no GT pose swap',
    primary_green='manual_click corners with known visibility; same whole-object C4 mean alignment',
    secondary_green='all known corners, including PnP-derived points, explicitly labeled proxy',
    primary_amendment_reason='Before predictions: 128/150 camera metadata mismatches and 519/1200 PnP-derived corners; do not treat generated points as independent observations',
    matching='highest-confidence candidate, IoU>=0.5 against bounding box of all known in-image annotation points; this box can contain PnP-derived points',
    missing='existing image-diagonal penalty; PCK5/10/20 full eligible-corner denominator',
    clusters='descriptive capture-family results; no CI from seven correlated session folders',
    physical_6D=False, source_labels_modified=False,
)


def annotation_arrays(doc, manual_only=False):
    import numpy as np
    entries = doc['objects'][0]['keypoint_annotations']
    if len(entries) != 9:
        raise ValueError('Expected 9 annotation entries')
    gt = np.array([e['xy'] if e.get('xy') is not None else [np.nan, np.nan] for e in entries], float)
    valid = np.array([e.get('visibility', 0) != 0 and e.get('xy') is not None
                      and (not manual_only or e.get('source') == 'manual_click') for e in entries])
    valid &= np.isfinite(gt).all(1) & ~(gt == -1).all(1)
    return gt, valid


def snapshot():
    import numpy as np
    from PIL import Image
    F.checked_lock()
    if DATASET.exists():
        load_dataset()
        print('Existing immutable snapshot verified; no current labels overwritten', flush=True)
        return
    proposal_path = F.REVIEW / 'REVIEW_150_PROPOSAL.json'
    proposal = F.read(proposal_path)
    if proposal['union_frames'] != 150 or len(proposal['records']) != 150:
        raise ValueError('Expected precisely the pre-prediction 150-image proposal')
    records = []
    buffers = []
    for candidate in proposal['records']:
        image = ROOT / candidate['image']
        annotation = ROOT / candidate['annotation']
        raw = annotation.read_bytes()
        doc = json.loads(raw)
        obj = doc['objects'][0]
        if obj.get('split') != 'eval':
            raise ValueError('A selected frame was excluded by user; do not silently replace: ' + candidate['id'])
        if obj.get('object_type', doc.get('object_type')) != 'plastic_standard_110x110x15':
            raise ValueError('Unexpected object type')
        gt, valid = annotation_arrays(doc)
        _, manual = annotation_arrays(doc, True)
        if manual[:8].sum() < 4:
            raise ValueError('Fewer than four known manual corners: ' + candidate['id'])
        with Image.open(image) as im:
            w, h = im.size
        if (w, h) != (doc['camera_data']['width'], doc['camera_data']['height']):
            raise ValueError('Image and annotation resolution mismatch')
        inside = valid & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
        if not inside.any():
            raise ValueError('No in-image reference points')
        camera = image.parent.parent / 'cam_K.txt'
        k = np.loadtxt(camera).reshape(3, 3)
        intr = doc['camera_data']['intrinsics']
        ak = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
        dest = DOC / 'annotations' / (candidate['id'] + '.json')
        digest = hashlib.sha256(raw).hexdigest()
        ib = F.binding(image)
        if ib['sha256'] != candidate['image_sha256']:
            raise ValueError('Proposal image changed: ' + candidate['id'])
        groups = [g for g in candidate['groups'] if g != 'truncation']
        if obj.get('truncation', {}).get('is_truncated') is True:
            groups.append('truncation')
        records.append(dict(
            id=candidate['id'], session=candidate['session'], groups=groups,
            capture_family=image.parents[3].name, image=ib,
            annotation=dict(path=str(dest.relative_to(ROOT)), sha256=digest, bytes=len(raw)),
            original_annotation=dict(path=candidate['annotation'], sha256=digest, bytes=len(raw)),
            camera=F.binding(camera), camera_intrinsics_match=bool(np.allclose(k, ak, atol=1e-5, rtol=0)),
            source_K=k.tolist(), annotation_K=ak.tolist(), known_corners=int(valid[:8].sum()),
            manual_corners=int(manual[:8].sum()),
            annotation_sources=dict(Counter(e.get('source', 'unknown') for e in obj['keypoint_annotations'][:8])),
            canonical_WDH_m=[1.1, 1.1, .15], original_hw=[h, w],
            prior_training_development_overlap='NOT_CLEARED; historical square experiments used related captures',
        ))
        buffers.append((annotation, dest, raw))
    if len({r['id'] for r in records}) != 150 or len({r['image']['sha256'] for r in records}) != 150:
        raise ValueError('Duplicate membership/image content')
    for original, _, raw in buffers:
        if original.read_bytes() != raw:
            raise ValueError('Concurrent annotation save; retry snapshot')
    # Preserve exact saved bytes; the editor continues to use the original labels.
    for _, dest, raw in buffers:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.read_bytes() != raw:
                raise ValueError('Partial snapshot differs; create a new version')
        else:
            with dest.open('xb') as handle:
                handle.write(raw)
    F.freeze(DATASET, dict(protocol=PROTOCOL, records=records,
                          proposal=F.binding(proposal_path), model_lock=F.binding(F.DOC / 'MODEL_LOCK.json'),
                          evaluator=F.binding(Path(__file__)),
                          counts=dict(frames=150, camera_mismatches=sum(not r['camera_intrinsics_match'] for r in records),
                                      manual_corners=sum(r['manual_corners'] for r in records),
                                      source_counts=dict(sum((Counter(r['annotation_sources']) for r in records), Counter())))))
    print(json.dumps(F.read(DATASET)['counts'], indent=2), flush=True)


def load_dataset():
    F.checked_lock()
    data = F.read(DATASET)
    for key in ('model_lock', 'evaluator', 'proposal'):
        F.verify(data[key])
    for row in data['records']:
        for key in ('image', 'annotation', 'camera'):
            F.verify(row[key])
    return data


def infer():
    import cv2
    import torch
    data = load_dataset()
    path = RAW / 'PREDICTIONS.json'
    if path.exists():
        saved = F.read(path)
        F.verify(saved['dataset']); F.verify(saved['model'])
        print('Immutable predictions already present; no rerun', flush=True)
        return
    E = F.setup()
    from inference import load_head, predict_captured, serial, registry_input
    torch.set_num_threads(4)
    gpu_start = E.gpu()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA access required; do not fall back to CPU')
    dimensions, order = registry_input('plastic_standard_110x110x15')
    assert dimensions.tolist() == [1.1, 1.1, .15] and order == 4
    lock = F.checked_lock()
    heads = {(r['arm'], r['seed']): load_head(r['arm'], r['seed'])[0] for r in lock['heads']}
    extractor = E.old('features').FrozenYoloFeatures(E.R0)
    norm = F.read(F.DCP / 'DIM_NORMALIZATION_LOCK.json')
    result = {name: [] for name in ['R0', *(f'{a}_seed{s}' for a, s in heads)]}
    try:
        for i, row in enumerate(data['records']):
            im = cv2.imread(str(ROOT / row['image']['path']))
            if im is None:
                raise ValueError(row['id'])
            captured = extractor.predict(im)
            result['R0'].append(dict(id=row['id'], raw_hw=list(im.shape[:2]),
                                    candidates=serial(captured['candidates']), selected_index=captured['selected_index']))
            for (arm, seed), head in heads.items():
                name = f'{arm}_seed{seed}'
                pred, _ = predict_captured(head, arm, captured, dimensions, order,
                                           lock['temperatures'][name], lock['decode_rule'], im.shape[:2], norm)
                result[name].append(dict(id=row['id'], raw_hw=list(im.shape[:2]), **pred))
            if i % 20 == 0:
                E.gpu()
                print(f'GREEN150 INFERENCE {i + 1}/150', flush=True)
    finally:
        extractor.close()
    F.freeze(path, dict(complete=True, GT_input=False, camera_input=False,
                        dataset=F.binding(DATASET), model=F.binding(F.DOC / 'MODEL_LOCK.json'),
                        predictions=result, gpu_start=gpu_start, gpu_end=E.gpu()))


def score():
    import numpy as np
    data = load_dataset()
    payload = F.read(RAW / 'PREDICTIONS.json')
    F.verify(payload['dataset']); F.verify(payload['model'])
    E = F.setup()
    from eval_math import measure, summary, damage
    from dev_evaluate import iou
    objects = F.read(E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    perms = next(r['permutations'] for r in objects if r['object_type'] == 'plastic_standard_110x110x15')
    modes = {}
    for mode in ('manual_only', 'all_known_proxy'):
        rows = {}
        for name, predictions in payload['predictions'].items():
            if len(predictions) != len(data['records']):
                raise ValueError('Incomplete predictions')
            scored = []
            for row, pred in zip(data['records'], predictions):
                assert row['id'] == pred['id']
                doc = F.read(ROOT / row['annotation']['path'])
                gt, all_valid = annotation_arrays(doc)
                _, valid = annotation_arrays(doc, mode == 'manual_only')
                h, w = pred['raw_hw']
                inside = all_valid & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
                box = np.r_[gt[inside].min(0), gt[inside].max(0)]
                idx = pred['selected_index']
                candidate = None if idx is None else pred['candidates'][idx]
                matched = candidate is not None and iou(candidate['box_xyxy'], box) >= .5
                points = np.full((9, 2), np.nan) if candidate is None else candidate['keypoints_xy']
                metrics = measure(points, gt, valid, perms, (h, w), matched, idx is not None)
                scored.append(dict(id=row['id'], session=row['session'], capture_family=row['capture_family'], **metrics))
            rows[name] = scored
        contrasts = {}
        for reference in ('R0', 'N0_BASE_REPLAY'):
            contrasts[reference] = {}
            for seed in (1, 2, 3):
                a = rows[f'N2_DIM_ONLY_seed{seed}']
                b = rows['R0' if reference == 'R0' else f'{reference}_seed{seed}']
                contrasts[reference][str(seed)] = dict(
                    delta_E_sym=float(np.mean([x['E_sym'] - y['E_sym'] for x, y in zip(a, b)])),
                    **damage(b, a))
        condition = {}
        for group in ('truncation', 'handheld_multiview', 'evening_capture_candidate', 'darker_dusk_candidate'):
            ids = {r['id'] for r in data['records'] if group in r['groups']}
            condition[group] = {name: summary([r for r in rr if r['id'] in ids]) for name, rr in rows.items()}
        captures = {group: {name: summary([r for r in rr if r['capture_family'] == group]) for name, rr in rows.items()}
                    for group in sorted({r['capture_family'] for r in data['records']})}
        modes[mode] = dict(rows=rows, summary={k: summary(v) for k, v in rows.items()},
                           damage_and_delta=contrasts, condition=condition, capture_family=captures)
    F.freeze(RAW / 'METRICS.json', dict(protocol=PROTOCOL, modes=modes,
               predictions=F.binding(RAW / 'PREDICTIONS.json'), counts=data['counts'],
               confidence_interval='NOT_ESTIMATED_FEW_DEPENDENT_CAPTURE_FAMILIES'))
    print('GREEN150_SAVED_LABEL_2D_SCORING_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('snapshot', 'infer', 'score'))
    args = parser.parse_args()
    {'snapshot': snapshot, 'infer': infer, 'score': score}[args.stage]()


if __name__ == '__main__':
    main()
