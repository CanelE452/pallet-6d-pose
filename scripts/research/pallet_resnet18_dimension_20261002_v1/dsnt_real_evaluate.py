"""Evaluate a completed DSNT ResNet18 arm on frozen real development sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import dsnt_diagnostic as D
import dsnt_train as T
from input_data import prepare, transform_points
from scripts.evaluation import final_dimension_release as F
from scripts.evaluation import green0918_square_audit_v1 as Q
from scripts.evaluation import green_saved_labels_v1 as G


GREEN150 = F.DOC / 'green150_saved_labels_v1/DATASET_SNAPSHOT.json'
DEV319 = F.ROOT / 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'


def features(dimensions):
    dimensions = np.asarray(dimensions, np.float64)
    if dimensions.ndim != 2 or dimensions.shape[1] != 3 or not np.isfinite(dimensions).all():
        raise ValueError('Expected finite canonical W,D,H rows')
    log = np.log(dimensions)
    raw = np.c_[log, log[:, 0] - log[:, 1], log[:, 2] - .5 * (log[:, 0] + log[:, 1])]
    normalization = json.loads(D.DIMENSION_NORMALIZATION.read_text())
    return ((raw - np.asarray(normalization['mean'])) / np.asarray(normalization['scale'])).astype(np.float32)


def dimension_support(rows):
    """Disclose whether each real canonical dimension is inside source TRAIN support."""
    source = json.loads(D.SOURCE_MANIFEST.read_text())
    sidecar = np.load(D.DIMENSION_SIDECAR, allow_pickle=False)
    if not np.array_equal(sidecar['record_index'], np.arange(len(source['records']))):
        raise ValueError('Dimension sidecar/source-manifest order mismatch')
    train_indices = np.asarray([index for index, row in enumerate(source['records'])
                                if row['partition'] == 'train'], np.int64)
    if len(train_indices) != 55980:
        raise ValueError('Unexpected source TRAIN population')
    train_dimensions = np.asarray(sidecar['dimensions'][train_indices], np.float64)
    train_context = features(train_dimensions)
    real = []
    seen = set()
    for row in rows:
        key = (row['object_type'], *map(float, row['dimensions']))
        if key in seen:
            continue
        seen.add(key)
        value = np.asarray(row['dimensions'], np.float64)
        context = features(value[None])[0]
        below = value < train_dimensions.min(0)
        above = value > train_dimensions.max(0)
        real.append(dict(object_type=row['object_type'], canonical_WDH_m=value.tolist(),
            normalized_context=context.tolist(),
            outside_TRAIN_canonical_axes=np.flatnonzero(below | above).tolist(),
            below_TRAIN_canonical_min=below.tolist(), above_TRAIN_canonical_max=above.tolist(),
            inside_TRAIN_axis_aligned_WDH_box=bool(not (below | above).any())))
    return dict(
        source_manifest=D.bound(D.SOURCE_MANIFEST), dimension_sidecar=D.bound(D.DIMENSION_SIDECAR),
        normalization=D.bound(D.DIMENSION_NORMALIZATION), train_rows=len(train_indices),
        canonical_WDH_m=dict(minimum=train_dimensions.min(0).tolist(),
                             maximum=train_dimensions.max(0).tolist()),
        normalized_context=dict(minimum=train_context.min(0).tolist(),
                                maximum=train_context.max(0).tolist()),
        real=real,
        interpretation='Axis-aligned support diagnostic only; being inside does not establish distributional equivalence.')


def records():
    E = F.setup()
    from dev_evaluate import population_metadata
    from inference import registry_input
    pe, population = population_metadata()
    green150 = G.load_dataset()
    green0918 = Q.load_dataset()
    rows = []
    for item, metadata in population:
        dimensions, _ = registry_input(metadata['object_type'])
        rows.append(dict(id=item.frame_id, dataset='DEV319', image=str(F.ROOT / item.image),
            dimensions=dimensions.tolist(), object_type=metadata['object_type']))
    for row in green150['records']:
        rows.append(dict(id=row['id'], dataset='GREEN150', image=str(F.ROOT / row['image']['path']),
            dimensions=row['canonical_WDH_m'], object_type='plastic_standard_110x110x15'))
    for row in green0918['records']:
        rows.append(dict(id=row['id'], dataset='GREEN0918_119', image=str(F.ROOT / row['image']['path']),
            dimensions=row['canonical_WDH_m'], object_type='plastic_standard_110x110x15'))
    if len(rows) != 319 + 150 + 119 or len({(r['dataset'], r['id']) for r in rows}) != len(rows):
        raise ValueError('Unexpected evaluation population')
    return E, pe, population, rows, dict(GREEN150=green150, GREEN0918_119=green0918)


def checkpoint(arm):
    complete = T.complete_path(arm)
    payload = json.loads(complete.read_text())
    if not payload.get('complete') or payload.get('arm') != arm:
        raise ValueError('Training completion receipt does not match requested arm')
    D.verify(payload['protocol']); D.verify(payload['final_checkpoint'])
    return D.ROOT / payload['final_checkpoint']['path'], payload


@torch.no_grad()
def infer(arm):
    E, _, _, rows, _ = records()
    model_path, training = checkpoint(arm)
    destination = D.RAW / f'DSNT_FULL_{arm}_REAL_PREDICTIONS.json'
    if destination.exists():
        print('Existing immutable real predictions present', flush=True)
        return
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    if state.get('arm') != arm or state.get('epoch') != T.EPOCHS:
        raise ValueError('Final checkpoint arm/epoch mismatch')
    D.verify(state['protocol'])
    model = D.initialize(); model.load_state_dict(state['model_state_dict']); model.eval()
    normalized = features(np.asarray([row['dimensions'] for row in rows]))
    normalized *= D.ARMS[arm][None]
    predictions = {dataset: [] for dataset in ('DEV319', 'GREEN150', 'GREEN0918_119')}
    for start in range(0, len(rows), 16):
        selected = rows[start:start + 16]
        prepared = []
        images = []
        for row in selected:
            image = cv2.imread(row['image'])
            if image is None:
                raise ValueError(row['image'])
            value = prepare(image, source_pre_padded=False)
            prepared.append(value); images.append(value['tensor'])
        batch = torch.stack(images).cuda()
        context = torch.from_numpy(normalized[start:start + len(selected)]).cuda()
        logits = model(batch, context)
        coordinates, _, probability = D.spatial_coordinates(logits)
        coordinates = coordinates.cpu().numpy(); probability = probability.cpu().numpy()
        grid = np.empty_like(coordinates)
        grid[..., 0] = (coordinates[..., 0] + 1.) * .5 * (D.HEATMAP_HW[1] - 1)
        grid[..., 1] = (coordinates[..., 1] + 1.) * .5 * (D.HEATMAP_HW[0] - 1)
        for index, row in enumerate(selected):
            network = grid[index] * D.STRIDE
            points = transform_points(network, prepared[index]['affine_net_to_input'])
            box = np.r_[points[:8].min(0), points[:8].max(0)]
            predictions[row['dataset']].append(dict(id=row['id'], raw_hw=prepared[index]['input_hw'],
                points=points.tolist(), box_xyxy=box.tolist(), dimensions=row['dimensions'],
                mean_softmax_peak=float(probability[index].max(-1).mean())))
        if start == 0 or start % 64 == 0:
            print('DSNT_REAL_INFER', arm, min(start + 16, len(rows)), len(rows), flush=True)
    D.write(destination, dict(complete=True, arm=arm, GT_input=False, camera_input=False,
        checkpoint=D.bound(model_path), training=D.bound(T.complete_path(arm)),
        datasets=dict(DEV319=D.bound(DEV319), GREEN150=D.bound(G.DATASET),
                      GREEN0918_119=D.bound(Q.DATASET)),
        contracts=dict(evaluator=D.bound(Path(__file__)),
                       symmetry=D.bound(E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')),
        inputs=dict(DEV319=319, GREEN150=150, GREEN0918_119=119), predictions=predictions,
        context='Canonical W,D,H fixed object axes; no pose-selected swap.', gpu=E.gpu()))
    print('DSNT_REAL_INFERENCE_COMPLETE', arm, flush=True)


def manual_target(row, prediction, in_frame_only):
    annotation = F.read(F.ROOT / row['annotation']['path'])
    gt, all_valid = G.annotation_arrays(annotation)
    _, valid = G.annotation_arrays(annotation, True)
    h, w = prediction['raw_hw']
    inside = np.isfinite(gt).all(1) & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
    if in_frame_only:
        valid &= inside
    match_points = all_valid & inside
    box = np.r_[gt[match_points].min(0), gt[match_points].max(0)]
    return gt, valid, box


def score(arm):
    E, pe, population, rows, datasets = records()
    from dev_evaluate import iou
    from eval_math import measure, summary
    payload = json.loads((D.RAW / f'DSNT_FULL_{arm}_REAL_PREDICTIONS.json').read_text())
    if not payload.get('complete') or payload.get('arm') != arm:
        raise ValueError('Incomplete or wrong-arm real predictions')
    D.verify(payload['checkpoint']); D.verify(payload['training'])
    for binding in payload['datasets'].values():
        D.verify(binding)
    for binding in payload['contracts'].values():
        D.verify(binding)
    expected_ids = {
        'DEV319': [item.frame_id for item, _ in population],
        'GREEN150': [row['id'] for row in datasets['GREEN150']['records']],
        'GREEN0918_119': [row['id'] for row in datasets['GREEN0918_119']['records']],
    }
    for dataset, ids in expected_ids.items():
        predictions = payload['predictions'].get(dataset)
        if predictions is None or [row['id'] for row in predictions] != ids:
            raise ValueError(f'Incomplete or reordered predictions: {dataset}')
    objects = F.read(E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    permutations = {row['object_type']: row['permutations'] for row in objects}
    green150 = {row['id']: row for row in datasets['GREEN150']['records']}
    green0918 = {row['id']: row for row in datasets['GREEN0918_119']['records']}
    results = {}
    dev = []
    for (item, metadata), prediction in zip(population, payload['predictions']['DEV319']):
        if item.frame_id != prediction['id']:
            raise ValueError('DEV order mismatch')
        target = pe.E._legacy_forbidden_target(item)
        points = np.asarray(prediction['points'])
        matched = iou(prediction['box_xyxy'], target.box_xyxy) >= .5
        dev.append(dict(id=prediction['id'], session=metadata['session_id'], object_type=metadata['object_type'],
            **measure(points, target.keypoints_xy, target.keypoint_supervision_mask,
                      permutations[metadata['object_type']], prediction['raw_hw'], matched, True)))
    results['DEV319'] = dict(rows=dev, summary=summary(dev))
    results['DEV319_BY_OBJECT'] = {
        object_type: dict(rows=selected, summary=summary(selected))
        for object_type in sorted({row['object_type'] for row in dev})
        for selected in [[row for row in dev if row['object_type'] == object_type]]
    }
    square_permutations = permutations['plastic_standard_110x110x15']
    for dataset, metadata_rows in (('GREEN150', green150), ('GREEN0918_119', green0918)):
        for suffix, in_frame in (('MANUAL_DECLARED', False), ('MANUAL_IN_FRAME', True)):
            scored = []
            for prediction in payload['predictions'][dataset]:
                row = metadata_rows[prediction['id']]
                gt, valid, box = manual_target(row, prediction, in_frame)
                matched = iou(prediction['box_xyxy'], box) >= .5
                scored.append(dict(id=prediction['id'], session=row['session'],
                    **measure(np.asarray(prediction['points']), gt, valid, square_permutations,
                              prediction['raw_hw'], matched, True)))
            results[f'{dataset}_{suffix}'] = dict(rows=scored, summary=summary(scored))
    destination = D.DOC / f'DSNT_FULL_{arm}_REAL_RESULTS.json'
    D.write(destination, dict(complete=True, arm=arm, checkpoint=payload['checkpoint'],
        predictions=D.bound(D.RAW / f'DSNT_FULL_{arm}_REAL_PREDICTIONS.json'),
        datasets=payload['datasets'], dimension_support=dimension_support(rows), results=results,
        independent_confirmation=False, endpoint='First eight corner channels; center channel excluded.',
        coverage_interpretation='Predicted-hull IoU>=0.5 for a forced single-pallet output; not detector recall.',
        note='Reused development references. GREEN manual errors use all-known in-image hull only for predicted-hull matching.'))
    print('DSNT_REAL_SCORE_COMPLETE', arm, flush=True)


def pose_score(arm):
    _, _, population, _, _ = records()
    import pose
    payload = json.loads((D.RAW / f'DSNT_FULL_{arm}_REAL_PREDICTIONS.json').read_text())
    if not payload.get('complete') or payload.get('arm') != arm:
        raise ValueError('Incomplete or wrong-arm real predictions')
    for binding in (*payload['datasets'].values(), *payload['contracts'].values()):
        D.verify(binding)
    expected_ids = [item.frame_id for item, _ in population]
    if [row['id'] for row in payload['predictions']['DEV319']] != expected_ids:
        raise ValueError('DEV319 prediction order mismatch')
    metadata, truth = pose.metadata('REAL_DEV')
    if set(expected_ids) != set(metadata) or set(expected_ids) != set(truth):
        raise ValueError('Pose metadata/reference population mismatch')
    rows = []
    for prediction in payload['predictions']['DEV319']:
        inferred = pose.infer(prediction['points'], *metadata[prediction['id']])
        rows.append(pose.metric((prediction['id'], inferred, truth[prediction['id']])))
    available = [row for row in rows if row['available']]
    result = dict(frames=len(rows), available=len(available), coverage=len(available) / len(rows),
        ADDsym_AUC_full=pose.pose_auc([row['ADDsym_normalized'] if row['available'] else float('inf') for row in rows], 1.))
    for key in ('translation_cm', 'rotation_deg', 'yaw_deg', 'IoU3D'):
        values = [row[key] for row in available]
        result[key] = (dict(median=float(np.median(values)), p90=float(np.quantile(values, .9)))
                       if values else dict(median=None, p90=None))
    D.write(D.DOC / f'DSNT_FULL_{arm}_POSE_RESULTS.json', dict(complete=True, arm=arm,
        predictions=D.bound(D.RAW / f'DSNT_FULL_{arm}_REAL_PREDICTIONS.json'),
        summary=result, rows=rows, physical_GT=False, forced_single_pallet_output=True,
        coverage_interpretation='PnP solver availability from nine forced softargmax points; not detection coverage.',
        solver='Existing reconstructed reference and identical registered dimensions/SQPnP-LM contract.'))
    print('DSNT_REAL_POSE_SCORE_COMPLETE', arm, json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('infer', 'score', 'pose_score', 'all'))
    parser.add_argument('arm', choices=tuple(D.ARMS))
    arguments = parser.parse_args()
    if arguments.stage == 'all':
        infer(arguments.arm); score(arguments.arm); pose_score(arguments.arm)
    else:
        globals()[arguments.stage](arguments.arm)
