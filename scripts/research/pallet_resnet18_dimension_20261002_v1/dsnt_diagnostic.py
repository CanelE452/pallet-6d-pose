"""Bounded source-only functional diagnostic for dimension-conditioned ResNet18.

The prior MSE heatmap baseline converged to near-zero maps.  This diagnostic
keeps the same image encoder/decoder and replaces only the training objective
and decoding contract with a normalized spatial-distribution (DSNT-style)
objective.  Three equal-budget arms vary only the supplied dimension context.
No real evaluation or paper-performance claim is made here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data._utils.collate import default_collate

from model import SimpleBaselineResNet18
from input_data import (DIMENSION_NORMALIZATION, DIMENSION_SIDECAR, HEATMAP_HW,
                        SOURCE_MANIFEST, STRIDE, SourceDataset)


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
RAW = ROOT / 'data/pallet/results' / HERE.name
PRETRAINED = ROOT / 'data/pallet/results/pallet_resnet18_refiner_20261001_v1/pretrained/resnet18-f37072fd.pth'
PROTOCOL = DOC / 'DSNT_DIAGNOSTIC_PROTOCOL.json'
RESULT = DOC / 'DSNT_DIAGNOSTIC_RESULTS.json'
ORDER = RAW / 'DSNT_DIAGNOSTIC_ORDER.npy'

ARMS = {
    'CONSTANT': np.array([0., 0., 0., 0., 0.], np.float32),
    'SHAPE': np.array([0., 0., 0., 1., 1.], np.float32),
    'FULL': np.array([1., 1., 1., 1., 1.], np.float32),
}
UPDATES = 500
BATCH = 16
EVALUATE = (0, 100, 500)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        pending = path.with_name(path.name + '.pending')
        pending.write_text(text)
        pending.replace(path)


def verify(binding):
    path = ROOT / binding['path']
    if sha(path) != binding['sha256'] or path.stat().st_size != binding['bytes']:
        raise ValueError(f'Frozen binding changed: {binding["path"]}')


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def gpu_status():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.used,temperature.gpu,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()


def order256():
    batches = []
    for cycle in range(32):
        batches.append(np.random.default_rng(20261002 + cycle).permutation(256))
    return np.concatenate(batches)[:UPDATES * BATCH].reshape(UPDATES, BATCH).astype(np.int64)


def evaluation_indices(dataset):
    groups = defaultdict(list)
    for local, source_index in enumerate(dataset.indices[:256]):
        shape = tuple(dataset.records[int(source_index)]['prepared_shape_hw'])
        if len(groups[shape]) < 4:
            groups[shape].append(local)
    if len(groups) != 4 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError('Expected four prepared shapes with four rows each in TRAIN256')
    return [row for shape in sorted(groups) for row in groups[shape]]


def spatial_coordinates(logits):
    if logits.ndim != 4 or logits.shape[1:] != (9, *HEATMAP_HW):
        raise ValueError('Expected Bx9x96x128 logits')
    batch, channels, height, width = logits.shape
    flat = logits.reshape(batch, channels, -1)
    log_probability = functional.log_softmax(flat, dim=-1)
    probability = log_probability.exp()
    xs = torch.linspace(-1., 1., width, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(-1., 1., height, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    coordinates = probability @ grid
    return coordinates, log_probability, probability


def distribution_loss(logits, target_heatmaps, valid):
    coordinates, log_probability, probability = spatial_coordinates(logits)
    target = target_heatmaps.to(logits).reshape(len(logits), 9, -1)
    mass = target.sum(-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).eps)
    target_probability = target / mass
    target_coordinates = target_probability @ torch.stack(torch.meshgrid(
        torch.linspace(-1., 1., HEATMAP_HW[0], device=logits.device, dtype=logits.dtype),
        torch.linspace(-1., 1., HEATMAP_HW[1], device=logits.device, dtype=logits.dtype),
        indexing='ij'), dim=-1).reshape(-1, 2).flip(-1)
    mask = valid.to(device=logits.device, dtype=torch.bool)
    if not mask.any():
        raise ValueError('Batch has no supervised keypoints')
    cross_entropy = -(target_probability * log_probability).sum(-1)
    coordinate = functional.smooth_l1_loss(coordinates, target_coordinates, reduction='none').sum(-1)
    loss = (0.1 * cross_entropy + coordinate)[mask].mean()
    return loss, dict(cross_entropy=cross_entropy[mask].mean(),
        coordinate=coordinate[mask].mean(), coordinates=coordinates,
        probability=probability, target_coordinates=target_coordinates)


def photometric(image):
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    rgb = image * std + mean
    factors = .8 + .4 * torch.rand((3, len(image), 1, 1, 1), device=image.device)
    rgb = rgb * factors[0]
    grey = (rgb * rgb.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
    center = grey.mean((2, 3), keepdim=True)
    rgb = (rgb - center) * factors[1] + center
    grey = (rgb * rgb.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
    rgb = (rgb - grey) * factors[2] + grey
    return (rgb.clamp(0, 1) - mean) / std


def context(batch, arm):
    mask = batch['dimension_context'].new_tensor(ARMS[arm])
    return batch['dimension_context'] * mask


def iou(a, b):
    left = np.maximum(a[:2], b[:2]); right = np.minimum(a[2:], b[2:])
    intersection = float(np.maximum(right - left, 0.).prod())
    union = float(np.maximum(a[2:] - a[:2], 0.).prod() +
                  np.maximum(b[2:] - b[:2], 0.).prod() - intersection)
    return 0. if union <= 0 else intersection / union


def stats(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return None
    return dict(minimum=float(values.min()), maximum=float(values.max()),
        mean=float(values.mean()), median=float(np.median(values)),
        p90=float(np.quantile(values, .9)))


@torch.no_grad()
def evaluate(model, cached, indices, arm, step):
    prior = model.training
    model.eval()
    rows = []
    logits_all = []
    coordinates_all = []
    losses = []
    components = []
    for start in range(0, len(indices), 4):
        batch = default_collate([cached[index] for index in indices[start:start + 4]])
        logits = model(batch['image'].cuda(), context(batch, arm).cuda())
        value, detail = distribution_loss(logits, batch['heatmaps'].cuda(), batch['target_valid'].cuda())
        losses.append(float(value)); components.append((float(detail['cross_entropy']), float(detail['coordinate'])))
        coordinates = detail['coordinates'].cpu().numpy()
        probability = detail['probability'].cpu().numpy()
        grid = np.empty_like(coordinates)
        grid[..., 0] = (coordinates[..., 0] + 1.) * .5 * (HEATMAP_HW[1] - 1)
        grid[..., 1] = (coordinates[..., 1] + 1.) * .5 * (HEATMAP_HW[0] - 1)
        points = grid * STRIDE
        argmax = probability.argmax(-1)
        argmax_grid = np.stack((argmax % HEATMAP_HW[1], argmax // HEATMAP_HW[1]), axis=-1) * STRIDE
        for j in range(len(points)):
            valid = batch['target_valid'][j].numpy().astype(bool)
            gt = batch['points_net'][j].numpy()
            errors = np.linalg.norm(points[j][valid] - gt[valid], axis=-1)
            argmax_errors = np.linalg.norm(argmax_grid[j][valid] - gt[valid], axis=-1)
            gt_corners = gt[:8][valid[:8]]
            predicted_box = np.r_[points[j, :8].min(0), points[j, :8].max(0)]
            gt_box = np.r_[gt_corners.min(0), gt_corners.max(0)]
            overlap = iou(predicted_box, gt_box)
            rows.append(dict(source_index=int(batch['index'][j]), id=batch['id'][j],
                errors=errors.tolist(), argmax_errors=argmax_errors.tolist(),
                iou=overlap, matched=overlap >= .5,
                softmax_peak=probability[j].max(-1).tolist()))
        logits_all.append(logits.cpu().numpy()); coordinates_all.append(points)
    logits_all = np.concatenate(logits_all)
    coordinates_all = np.concatenate(coordinates_all)
    errors = [value for row in rows for value in row['errors']]
    argmax_errors = [value for row in rows for value in row['argmax_errors']]
    peaks = [value for row in rows for value in row['softmax_peak']]
    result = dict(arm=arm, step=step, frames=len(rows), supervised_points=len(errors),
        loss=float(np.mean(losses)), cross_entropy=float(np.mean([v[0] for v in components])),
        coordinate_loss=float(np.mean([v[1] for v in components])),
        softargmax_network_px=stats(errors), argmax_network_px=stats(argmax_errors),
        softmax_peak=stats(peaks), matched_frames=sum(row['matched'] for row in rows),
        mean_coordinate_variance=float(coordinates_all.astype(np.float64).var(0).mean()),
        mean_logit_spatial_variance=float(logits_all.astype(np.float64).var((-2, -1)).mean()),
        rows=rows)
    model.train(prior)
    return result


def seal():
    if PROTOCOL.exists():
        payload = json.loads(PROTOCOL.read_text())
        for binding in payload['bindings'] + [payload['batch_order']]:
            verify(binding)
        print('Existing DSNT diagnostic protocol verified', flush=True)
        return
    order = order256()
    ORDER.parent.mkdir(parents=True, exist_ok=True)
    with ORDER.open('xb') as stream:
        np.save(stream, order)
    payload = dict(schema='resnet18_dimension_dsnt_functional_diagnostic_v1',
        scope='SOURCE_TRAIN256_REPEATED_DIAGNOSTIC_ONLY', arms=list(ARMS),
        arm_context_masks={name: value.tolist() for name, value in ARMS.items()},
        same_initialization=True, same_batch_order=True, same_photometric_rng=True,
        updates_per_arm=UPDATES, batch=BATCH, distinct_train_images=256,
        exposures_per_arm=UPDATES * BATCH, evaluation_steps=list(EVALUATE),
        evaluation='Fixed first four TRAIN rows per prepared image shape; 16 rows total.',
        objective='mean valid keypoints of 0.1*spatial Gaussian cross entropy + smooth-L1 softargmax coordinate error',
        decoder='Spatial softmax expectation; all nine keypoints required because each source frame contains one pallet.',
        optimizer=dict(name='Adam', lr=.001, weight_decay=0., betas=[.9, .999]),
        augmentation='Same brightness/contrast/saturation range and ordering as prior baseline.',
        gate='At step500 each arm must have at least one IoU>=0.5 frame and softargmax median error <=80% of its step0 value.',
        real_images=0, real_labels=0, checkpoint_selection=False,
        batch_order=bound(ORDER), batch_order_array_sha256=array_sha(order),
        bindings=[bound(path) for path in (SOURCE_MANIFEST, DIMENSION_SIDECAR,
            DIMENSION_NORMALIZATION, PRETRAINED, HERE / 'model.py', HERE / 'input_data.py', Path(__file__))])
    write(PROTOCOL, payload)
    print('DSNT_DIAGNOSTIC_SEALED', sha(PROTOCOL), flush=True)


def initialize():
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    state = torch.load(PRETRAINED, map_location='cpu', weights_only=True)
    return SimpleBaselineResNet18(pretrained_state=state).cuda()


def run():
    protocol = json.loads(PROTOCOL.read_text())
    for binding in protocol['bindings'] + [protocol['batch_order']]:
        verify(binding)
    if RESULT.exists():
        print('Existing DSNT diagnostic result present; no automatic rerun', flush=True)
        return
    order = np.load(ORDER)
    if order.shape != (UPDATES, BATCH) or array_sha(order) != protocol['batch_order_array_sha256']:
        raise ValueError('Batch order changed')
    torch.set_num_threads(1); cv2.setNumThreads(1)
    dataset = SourceDataset(SOURCE_MANIFEST, partition='train')
    cached = [dataset[index] for index in range(256)]
    indices = evaluation_indices(dataset)
    outcomes = []
    for arm in ARMS:
        model = initialize()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        evaluations = [evaluate(model, cached, indices, arm, 0)]
        trace = []
        begin = time.perf_counter()
        model.train()
        for step, rows in enumerate(order, 1):
            batch = default_collate([cached[int(index)] for index in rows])
            optimizer.zero_grad(set_to_none=True)
            logits = model(photometric(batch['image'].cuda()), context(batch, arm).cuda())
            value, detail = distribution_loss(logits, batch['heatmaps'].cuda(), batch['target_valid'].cuda())
            if not torch.isfinite(value):
                raise ValueError('Nonfinite loss')
            value.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 50 == 0:
                trace.append(dict(step=step, loss=float(value.detach()),
                    cross_entropy=float(detail['cross_entropy'].detach()),
                    coordinate=float(detail['coordinate'].detach()), gradient=float(gradient),
                    seconds=time.perf_counter() - begin))
                print('DSNT_UPDATE', arm, step, json.dumps(trace[-1]), flush=True)
            if step in EVALUATE:
                evaluation = evaluate(model, cached, indices, arm, step)
                evaluations.append(evaluation)
                print('DSNT_EVAL', arm, step, json.dumps({key: evaluation[key] for key in
                    ('loss', 'softargmax_network_px', 'argmax_network_px', 'matched_frames',
                     'softmax_peak', 'mean_coordinate_variance')}), flush=True)
        initial, final = evaluations[0], evaluations[-1]
        gate = dict(matched=final['matched_frames'] > 0,
            localization=final['softargmax_network_px']['median'] <= .8 * initial['softargmax_network_px']['median'])
        gate['PASS'] = all(gate.values())
        checkpoint = RAW / f'DSNT_{arm}_step0500_discarded.pt'
        torch.save(dict(discarded=True, arm=arm, step=UPDATES,
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            protocol=bound(PROTOCOL)), checkpoint)
        outcomes.append(dict(arm=arm, evaluations=evaluations, trace=trace, gate=gate,
            elapsed_seconds=time.perf_counter() - begin, checkpoint=bound(checkpoint)))
        del model, optimizer
        torch.cuda.empty_cache()
    result = dict(schema='resnet18_dimension_dsnt_functional_diagnostic_v1', complete=True,
        PASS=all(row['gate']['PASS'] for row in outcomes), protocol=bound(PROTOCOL),
        outcomes=outcomes, gpu=gpu_status(), real_images=0, generalization_established=False,
        paper_backbone_result=False, full_training_started=False)
    write(RESULT, result)
    lines = ['# ResNet18 치수 입력 기능 진단', '',
        '합성 TRAIN256만 반복한 폐기용 500-update 진단이다. 실사 성능이나 일반화를 뜻하지 않는다.', '',
        '| arm | step | softargmax median/P90 network px | argmax median/P90 | matched/16 | peak median |',
        '|---|---:|---:|---:|---:|---:|']
    for outcome in outcomes:
        for row in outcome['evaluations']:
            soft = row['softargmax_network_px']; hard = row['argmax_network_px']
            lines.append(f"| {outcome['arm']} | {row['step']} | {soft['median']:.3f}/{soft['p90']:.3f} | "
                         f"{hard['median']:.3f}/{hard['p90']:.3f} | {row['matched_frames']}/16 | "
                         f"{row['softmax_peak']['median']:.6f} |")
    lines += ['', f"기능 gate 전체 통과: **{result['PASS']}**.", '',
        '통과해도 어느 치수 arm이 일반화가 좋은지는 정하지 않는다. 다음 단계는 같은 전체 TRAIN budget의 CONSTANT/SHAPE/FULL 비교다.', '',
        '[전수 JSON](DSNT_DIAGNOSTIC_RESULTS.json)']
    report = DOC / 'DSNT_DIAGNOSTIC_RESULTS_KO.md'
    text = '\n'.join(lines) + '\n'
    if report.exists() and report.read_text() != text:
        raise ValueError('Diagnostic report differs')
    if not report.exists():
        report.write_text(text)
    print('DSNT_DIAGNOSTIC_COMPLETE', result['PASS'], flush=True)


def selfcheck():
    logits = torch.zeros((2, 9, *HEATMAP_HW), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 20, 30] = 1.; target[1, 8, 70, 100] = 1.
    valid = torch.zeros((2, 9), dtype=torch.bool); valid[0, 0] = True; valid[1, 8] = True
    value, detail = distribution_loss(logits, target, valid)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()
    assert detail['coordinates'].shape == (2, 9, 2)
    assert order256().shape == (UPDATES, BATCH) and np.array_equal(order256(), order256())
    assert set(ARMS) == {'CONSTANT', 'SHAPE', 'FULL'}
    print('DSNT_DIAGNOSTIC_SELFCHECK_PASS; actual images0 GPU0 updates0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('selfcheck', 'seal', 'run'))
    arguments = parser.parse_args()
    globals()[arguments.stage]()
