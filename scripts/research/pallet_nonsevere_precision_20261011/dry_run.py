"""Clean-replay entry point for the original no-update forward/backward check.

Added after the recorded original inline check; it was not an extra screen fit.
It creates a fresh receipt only and never replaces an existing receipt.
"""
import argparse, time
from pathlib import Path
import torch
import precision_model as M
import train as T


def run(args):
    out = Path(args.public_dir) / 'MODEL_DRY_RUN.json'
    assert not out.exists(), 'Existing original dry-run retained'
    T.configure()
    begin = time.monotonic()
    env, refiner, data, model, order, path, targets, decode, loss = T.setup(args, 1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    active = time.monotonic()
    rows = order[0]
    batch = data.batch(rows, 'N3_DIM_SYM')
    tiles = M.native_tiles(data, rows, batch, model.base.displacements)
    output = model(batch, True, tiles)
    ce, coord, geo, counts = T.components(output, batch, data, rows, refiner,
                                         targets, decode, loss, None)
    (ce + coord).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.norm()) for g in gradients) > 0
    torch.cuda.synchronize()
    T.write(out, dict(status='FORWARD_BACKWARD_NO_OPTIMIZER_UPDATES',
        total_setup_seconds=time.monotonic() - begin,
        actual_device_seconds=time.monotonic() - active,
        CE=float(ce.detach()), COORD=float(coord.detach()),
        tiles_shape=list(tiles[0].shape),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        counts=counts, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        optimizer_updates=0, gradients_finite=True,
        checkpoint_sha256=T.sha(path), source_files_changed=0,
        provenance='Clean replay of documented original inline check; this helper added after original check',
        first_failed_attempt=None))
    print('CLEAN_REPLAY_DRY_RUN_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--private-dir', required=True)
    parser.add_argument('--public-dir', required=True)
    run(parser.parse_args())
