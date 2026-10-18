"""Fixed eight-train-frame visual forward/backward probe; no optimizer exists."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

from .common import sha, write


def run(args):
    if not args.probe_GO:
        raise PermissionError('Root visual technical probe GO is required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Fresh CUDA process requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    # Torch and CUDA initialization occur only after the explicit guards.
    from .train_visual import configure_cuda, validate_cache
    from .train import prepare_items, attach_extra, forward_item, item_loss, state_hash
    from .risk_models import RiskModel, mean_frame_losses
    from .visual_patches import PatchProvider

    started = time.monotonic()
    items, bindings, diagnostics = prepare_items(
        args.features, args.feature_seal, args.labels_dir, args.targets,
        args.target_seal, args.split_lock)
    bindings.update(attach_extra(items, args.extra_features, args.extra_seal,
                                 args.extra_normalizer))
    cache = validate_cache(args.visual_inputs, args.synthetic_cache, args.features, items)
    torch, backend = configure_cuda()
    provider = PatchProvider(args.visual_inputs, args.synthetic_cache)
    fixed = items['train'][:8]
    assert len(fixed) == 8 and sum(len(i['entries']) for i in fixed) == 311
    results = []
    initial = None
    for arm in ('V_ORDER_NULL', 'V_ORDER_RGB'):
        torch.manual_seed(20261018)
        model = RiskModel(arm, 288).to('cuda:0')
        current = state_hash(model)
        if initial is None:
            initial = current
        elif current != initial:
            raise AssertionError('Matched visual probe initial states differ')
        # A fixed first-frame warm-up initializes the selected CUDA kernels.
        warm = forward_item(model, fixed[0], 'cuda:0', provider)
        warm_loss = item_loss(warm, fixed[0], arm, 20261018, 0, True)
        if warm_loss is not None:
            warm_loss.backward()
        model.zero_grad(set_to_none=True)
        del warm, warm_loss
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        wall, cpu = time.monotonic(), time.process_time()
        first.record()
        losses = []
        for item in fixed:
            output = forward_item(model, item, 'cuda:0', provider)
            losses.append(item_loss(output, item, arm, 20261018, 0, True))
        loss, valid = mean_frame_losses(losses)
        if loss is None:
            raise ValueError('Fixed probe unexpectedly has no rank supervision')
        loss.backward()
        last.record()
        torch.cuda.synchronize()
        finite = all(torch.isfinite(p.grad).all().item()
                     for p in model.parameters() if p.grad is not None)
        if not finite:
            raise ValueError('Visual analytic gradients are nonfinite')
        results.append(dict(
            arm=arm, parameters=sum(p.numel() for p in model.parameters()),
            initial_state_sha256=current, frames=8, actions=311, loss_frames=valid,
            forward_backward_host_wall_seconds=time.monotonic()-wall,
            CPU_seconds=time.process_time()-cpu,
            CUDA_stream_elapsed_seconds=first.elapsed_time(last)/1000,
            CUDA_time_scope='Elapsed CUDA stream interval, including dispatch gaps; not an active-kernel integral',
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            gradient_finite=True, optimizer_created=False, optimizer_steps=0,
            warmup_frames=1, candidate_chunk=8))
        del model, output, losses, loss
        torch.cuda.empty_cache()
    receipt = dict(
        schema='VISUAL_FIXED_TRAIN_FORWARD_BACKWARD_THROUGHPUT_V1', status='PASS',
        results=results, fixed_frame_ids=[i['id'] for i in fixed],
        backend=backend, bindings=bindings, cache_validation=cache,
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in
                       ('throughput_visual.py','train_visual.py','train.py',
                        'risk_models.py','supervision.py','visual_patches.py')},
        whole_probe_wall_seconds=time.monotonic()-started,
        optimizer_created=False, optimizer_steps=0, real_reference_decodes=0,
        real_RGB_or_cache_reads=0, original_archive_modified=False,
        inference_FPS_claim=False,
        note='Both use the same eight original train frames, actual311 actions and fixed seed; this probe does not estimate end-to-end inference FPS.')
    write(args.output, receipt)
    return receipt


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','feature-seal','labels-dir','targets','target-seal',
                 'split-lock','extra-features','extra-seal','extra-normalizer',
                 'visual-inputs','synthetic-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--probe-GO',action='store_true')
    return p


if __name__ == '__main__':
    result=run(parser().parse_args())
    print(json.dumps(dict(status=result['status'],results=result['results']),allow_nan=False))
