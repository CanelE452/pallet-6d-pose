"""Resumable full-source training for the ResNet18 dimension input arms."""
from __future__ import annotations

import argparse
import json
import math
import signal
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dsnt_diagnostic as D
from input_data import SourceDataset, worker_init_fn


PROTOCOL = D.DOC / 'DSNT_FULL_TRAIN_PROTOCOL.json'
SMOKE = D.DOC / 'DSNT_FULL_GPU_SMOKE.json'
EPOCHS = 10
STOP = False


def request_stop(signum, frame):
    del frame
    global STOP
    STOP = True
    print('DSNT_FULL_GRACEFUL_STOP_REQUESTED', signum, flush=True)


def read(path):
    return json.loads(path.read_text())


def replace_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_protocol():
    protocol = read(PROTOCOL)
    for binding in protocol['bindings']:
        D.verify(binding)
    return protocol


def one_update(model, optimizer, batch, arm):
    optimizer.zero_grad(set_to_none=True)
    logits = model(D.photometric(batch['image'].cuda(non_blocking=True)),
                   D.context(batch, arm).cuda(non_blocking=True))
    value, detail = D.distribution_loss(logits, batch['heatmaps'].cuda(non_blocking=True),
                                        batch['target_valid'].cuda(non_blocking=True))
    if not torch.isfinite(value):
        raise ValueError('Nonfinite training loss')
    value.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(value.detach()), cross_entropy=float(detail['cross_entropy'].detach()),
                coordinate=float(detail['coordinate'].detach()), gradient=float(gradient))


def box_iou(points, target, valid):
    target_corners = target[:8][valid[:8]]
    if len(target_corners) < 3:
        return 0.
    predicted = np.r_[points[:8].min(0), points[:8].max(0)]
    truth = np.r_[target_corners.min(0), target_corners.max(0)]
    return D.iou(predicted, truth)


@torch.no_grad()
def validate(model, loader, arm):
    prior = model.training
    model.eval()
    losses = []
    soft_errors = []
    hard_errors = []
    matched = 0
    frames = 0
    points_count = 0
    for batch in loader:
        logits = model(batch['image'].cuda(non_blocking=True),
                       D.context(batch, arm).cuda(non_blocking=True))
        value, detail = D.distribution_loss(logits, batch['heatmaps'].cuda(non_blocking=True),
                                            batch['target_valid'].cuda(non_blocking=True))
        losses.append((float(value), len(batch['image'])))
        coordinates = detail['coordinates'].cpu().numpy()
        probability = detail['probability'].cpu().numpy()
        grid = np.empty_like(coordinates)
        grid[..., 0] = (coordinates[..., 0] + 1.) * .5 * (D.HEATMAP_HW[1] - 1)
        grid[..., 1] = (coordinates[..., 1] + 1.) * .5 * (D.HEATMAP_HW[0] - 1)
        points = grid * D.STRIDE
        argmax = probability.argmax(-1)
        hard = np.stack((argmax % D.HEATMAP_HW[1], argmax // D.HEATMAP_HW[1]), axis=-1) * D.STRIDE
        for index in range(len(points)):
            valid = batch['target_valid'][index].numpy().astype(bool)
            target = batch['points_net'][index].numpy()
            soft_errors.extend(np.linalg.norm(points[index][valid] - target[valid], axis=-1).tolist())
            hard_errors.extend(np.linalg.norm(hard[index][valid] - target[valid], axis=-1).tolist())
            matched += box_iou(points[index], target, valid) >= .5
            points_count += int(valid.sum())
            frames += 1
    model.train(prior)
    return dict(frames=frames, supervised_points=points_count,
        loss=sum(value * count for value, count in losses) / sum(count for _, count in losses),
        softargmax_network_px=D.stats(soft_errors), argmax_network_px=D.stats(hard_errors),
        matched_frames=matched, matched_fraction=matched / frames)


def smoke():
    if SMOKE.exists():
        payload = read(SMOKE)
        for binding in payload['bindings']:
            D.verify(binding)
        print('Existing full-training smoke verified', flush=True)
        return
    torch.set_num_threads(1); cv2.setNumThreads(1)
    dataset = SourceDataset(D.SOURCE_MANIFEST, partition='train')
    loader = DataLoader(dataset, batch_size=16, sampler=list(range(16)), num_workers=0)
    batch = next(iter(loader))
    values = {}
    for arm in D.ARMS:
        model = D.initialize()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        before = D.evaluate(model, [dataset[index] for index in range(16)], list(range(16)), arm, 0)
        update = one_update(model, optimizer, batch, arm)
        after = D.evaluate(model, [dataset[index] for index in range(16)], list(range(16)), arm, 1)
        values[arm] = dict(update=update, before=before, after=after)
        del model, optimizer
        torch.cuda.empty_cache()
    D.write(SMOKE, dict(PASS=True, discarded_updates=len(D.ARMS), values=values,
        bindings=[D.bound(path) for path in (D.PRETRAINED, D.SOURCE_MANIFEST,
            D.DIMENSION_SIDECAR, D.DIMENSION_NORMALIZATION, D.HERE / 'model.py',
            D.HERE / 'input_data.py', D.HERE / 'dsnt_diagnostic.py', __file__)]))
    print('DSNT_FULL_GPU_SMOKE_PASS', flush=True)


def seal():
    if PROTOCOL.exists():
        verify_protocol()
        print('Existing DSNT full-training protocol verified', flush=True)
        return
    diagnostic = read(D.RESULT)
    if not diagnostic['PASS']:
        raise ValueError('Functional diagnostic did not pass')
    smoke_result = read(SMOKE)
    if not smoke_result['PASS']:
        raise ValueError('GPU smoke did not pass')
    payload = dict(schema='resnet18_dimension_dsnt_full_source_train_v1',
        purpose='Third RGB backbone requested for IEEE Sensors; controlled dimension-input comparison.',
        arms=list(D.ARMS), arm_context_masks={name: value.tolist() for name, value in D.ARMS.items()},
        source=dict(train=55980, calibration=1004, selection=1031, heldout=1985),
        real_training_images=0, epochs=EPOCHS, batch=16,
        total_updates_per_arm=math.ceil(55980 / 16) * EPOCHS,
        model='ImageNet ResNet18 plus three deconvolutions, nine spatial-distribution heatmaps and final decoder FiLM.',
        objective='0.1*normalized Gaussian spatial cross entropy + smooth-L1 spatial-expectation coordinate loss.',
        optimizer=dict(name='Adam', lr=.001, weight_decay=0., betas=[.9, .999],
                       schedule='x0.1 at epochs7 and9'),
        initialization='Same seed42 ImageNet body and randomly initialized decoder/FiLM for each arm.',
        order='Per-epoch TRAIN permutation numpy seed42+epoch; same order all arms.',
        augmentation='Same GPU brightness/contrast/saturation RNG reset at arm initialization.',
        validation='Synthetic calibration1004 after every epoch; fixed epoch10 checkpoint, no real metric selection.',
        checkpoint='Exact model/optimizer/CPU+CUDA RNG/epoch/batch resume every500 updates and epoch finalization.',
        decoder='Spatial softmax expectation for all nine keypoints; single-pallet source contract.',
        interpretation='CONSTANT controls architecture; SHAPE uses only two ratios; FULL uses absolute W,D,H plus ratios.',
        diagnostic=D.bound(D.RESULT), smoke=D.bound(SMOKE),
        bindings=[D.bound(path) for path in (D.RESULT, SMOKE, D.PRETRAINED, D.SOURCE_MANIFEST,
            D.DIMENSION_SIDECAR, D.DIMENSION_NORMALIZATION, D.HERE / 'model.py',
            D.HERE / 'input_data.py', D.HERE / 'dsnt_diagnostic.py', __file__)])
    D.write(PROTOCOL, payload)
    print('DSNT_FULL_TRAIN_PROTOCOL_SEALED', D.sha(PROTOCOL), flush=True)


def checkpoint_path(arm):
    return D.RAW / f'DSNT_FULL_{arm}_last.pt'


def complete_path(arm):
    return D.DOC / f'DSNT_FULL_{arm}_COMPLETE.json'


def train(arm):
    if arm not in D.ARMS:
        raise ValueError(arm)
    protocol = verify_protocol()
    complete = complete_path(arm)
    if complete.exists():
        print('DSNT_FULL_ALREADY_COMPLETE', arm, flush=True)
        return
    torch.set_num_threads(1); cv2.setNumThreads(1)
    dataset = SourceDataset(D.SOURCE_MANIFEST, partition='train')
    calibration = SourceDataset(D.SOURCE_MANIFEST, partition='calibration')
    validation_loader = DataLoader(calibration, batch_size=16, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(900042))
    path = checkpoint_path(arm)
    begin_epoch = 1; begin_batch = 0; global_step = 0; images_seen = 0
    elapsed_before = 0.; curves = []; trace = []; resumes = []
    epoch_numerator = 0.; epoch_frames = 0
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        D.verify(state['protocol'])
        model = D.initialize(); model.load_state_dict(state['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.cuda()
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        begin_epoch = state['next_epoch']; begin_batch = state['next_batch']
        global_step = state['step']; images_seen = state['images_seen']
        elapsed_before = state['elapsed_seconds']; curves = state['curves']; trace = state['trace']
        epoch_numerator = state['epoch_loss_numerator']; epoch_frames = state['epoch_frames']
        resumes = state['resumes'] + [dict(time=time.time(), step=global_step,
            epoch=begin_epoch, batch=begin_batch)]
    else:
        model = D.initialize()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    begin = time.perf_counter()

    def save(next_epoch, next_batch):
        state = dict(arm=arm, model=model.state_dict(), optimizer=optimizer.state_dict(),
            next_epoch=next_epoch, next_batch=next_batch, step=global_step, images_seen=images_seen,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            curves=curves, trace=trace, resumes=resumes,
            epoch_loss_numerator=epoch_numerator, epoch_frames=epoch_frames,
            elapsed_seconds=elapsed_before + time.perf_counter() - begin,
            protocol=D.bound(PROTOCOL))
        temporary = path.with_name(path.name + '.pending')
        torch.save(state, temporary); temporary.replace(path)
        return state

    for epoch in range(begin_epoch, EPOCHS + 1):
        order = np.random.default_rng(42 + epoch).permutation(len(dataset)).tolist()
        batches = [order[index:index + 16] for index in range(0, len(order), 16)]
        offset = begin_batch if epoch == begin_epoch else 0
        loader = DataLoader(dataset, batch_sampler=batches[offset:], num_workers=4,
            pin_memory=True, worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(100042 + epoch))
        lr = .001 * (.1 ** sum(epoch >= milestone for milestone in (7, 9)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        for batch_index, batch in enumerate(loader, offset):
            values = one_update(model, optimizer, batch, arm)
            global_step += 1; images_seen += len(batch['image'])
            epoch_numerator += values['loss'] * len(batch['image']); epoch_frames += len(batch['image'])
            if global_step == 1 or global_step % 500 == 0:
                row = dict(arm=arm, epoch=epoch, batch=batch_index + 1, step=global_step,
                    images_seen=images_seen, lr=lr, **values,
                    elapsed_seconds=elapsed_before + time.perf_counter() - begin)
                trace.append(row)
                replace_json(D.RAW / f'DSNT_FULL_{arm}_PROGRESS.json', row)
                print('DSNT_FULL_UPDATE', json.dumps(row), flush=True)
            if global_step % 500 == 0 or STOP:
                save(epoch, batch_index + 1)
            if STOP:
                raise SystemExit('Graceful stop after durable checkpoint')
        if epoch_frames != len(dataset):
            raise ValueError(f'Unexpected epoch frame count: {epoch_frames}')
        # Preserve the completed epoch before potentially expensive validation.
        save(epoch, len(batches))
        metrics = validate(model, validation_loader, arm)
        curve = dict(arm=arm, epoch=epoch, step=global_step, lr=lr,
            training_loss=epoch_numerator / epoch_frames, calibration=metrics)
        curves.append(curve)
        replace_json(D.RAW / f'DSNT_FULL_{arm}_CURVES.json', curves)
        epoch_numerator = 0.; epoch_frames = 0
        state = save(epoch + 1, 0)
        print('DSNT_FULL_EPOCH_COMPLETE', json.dumps(curve), flush=True)
    final = D.RAW / f'DSNT_FULL_{arm}_epoch{EPOCHS:02d}.pt'
    temporary = final.with_name(final.name + '.pending')
    torch.save(dict(arm=arm, epoch=EPOCHS, step=global_step,
        model_state_dict=model.state_dict(), protocol=D.bound(PROTOCOL)), temporary)
    temporary.replace(final)
    result = dict(complete=True, arm=arm, epochs=EPOCHS, updates=global_step,
        images_seen=images_seen, final_checkpoint=D.bound(final), resume_checkpoint=D.bound(path),
        protocol=D.bound(PROTOCOL), curves=curves, trace=trace, resumes=resumes,
        elapsed_seconds=state['elapsed_seconds'], real_training_images=0,
        final_epoch_fixed=True, generalization_to_real_not_yet_measured=True)
    D.write(complete, result)
    print('DSNT_FULL_TRAIN_COMPLETE', arm, result['elapsed_seconds'], flush=True)


def status():
    rows = {}
    for arm in D.ARMS:
        progress = D.RAW / f'DSNT_FULL_{arm}_PROGRESS.json'
        complete = complete_path(arm)
        rows[arm] = dict(complete=complete.exists(), progress=None if not progress.exists() else read(progress))
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'seal', 'train', 'status'))
    parser.add_argument('arm', nargs='?', choices=tuple(D.ARMS))
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    if arguments.stage == 'train':
        if arguments.arm is None:
            raise SystemExit('train requires an arm')
        train(arguments.arm)
    else:
        globals()[arguments.stage]()
