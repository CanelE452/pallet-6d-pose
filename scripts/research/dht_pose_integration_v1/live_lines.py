"""Generate the actual batch1 image DHT predictions in an isolated live result.

This is an inference parity correction, with unchanged checkpoints, input recipe,
lambda grid and synthetic-only selection rule. Original cached-feature results
are preserved. No GT values or support masks enter prediction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dht_lines as D


def source_hashes():
    return D.source_hashes() | {str(Path(__file__)): D.C.sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Original integration result directory; output goes into live/')
    args = parser.parse_args()
    original = args.run_dir.resolve()
    live = original/'live'
    live.mkdir(exist_ok=True)
    names = ('CONFIG.json', 'manifest.json', 'BASELINE_DOPE.json', 'BASELINE_YOLO.json', 'AUXILIARY_PROTOCOL.json', 'PURPOSE.md')
    for name in names:
        source, destination = original/name, live/name
        if destination.exists():
            if D.C.sha(source) != D.C.sha(destination):
                raise ValueError(f'Existing live input differs from original bytes: {name}')
        else:
            shutil.copyfile(source, destination)
    frozen_inputs = {name: D.C.sha(live/name) for name in names}
    cfg, manifest = D.read(live/'CONFIG.json'), D.read(live/'manifest.json')
    if cfg['schema'] != 'dht_pose_integration_v1' or cfg['dht']['seeds'] != [1, 2, 3]:
        raise ValueError('Expected original integration config with three frozen DHT checkpoints')
    records = manifest['records']
    if len(records) != 692 or len({r['id'] for r in records}) != 692:
        raise ValueError('Expected original692 unique evaluation records')
    identity = dict(config_sha256=frozen_inputs['CONFIG.json'], manifest_sha256=frozen_inputs['manifest.json'],
                    checkpoint_sha256=cfg['dht']['checkpoint_sha256'], source_sha256=source_hashes())
    output_path = live/'DHT_LINES.json'
    if output_path.exists():
        previous = D.read(output_path)
        if previous.get('complete') and all(previous.get(k) == v for k, v in identity.items()):
            print('Verified existing completed live predictions', flush=True)
            return
        raise ValueError('Existing live predictions differ in provenance')
    protocol = dict(schema='dht_pose_live_path_protocol_v1', registered_before_live_accuracy_analysis=True,
                    reason='Original runtime parity found2 of192 sampled role lines changed by image-feature re-extraction or batch12-to-batch1 head inference; timed live predictions must be the accuracy source.',
                    config_sha256=identity['config_sha256'], manifest_sha256=identity['manifest_sha256'],
                    parent_runtime_sha256=D.C.sha(original/'RUNTIME.json'),
                    original_results_preserved=str(original), copied_input_sha256=frozen_inputs,
                    intervention='Extract original uint8reflect100/square400 VGG features once per image atbatch1, quantizefloat16 thenfloat32 as in ImageDHT, evaluate allthree frozen DHT heads atbatch1.',
                    selection='Retain original lambda grid and select each baseline lambda on synthetic validation only; no real-driven change.',
                    prediction_mask='All8 predicted lines; no GT validity/support/facing mask.',
                    runtime='Actual image pipeline, source identical to ImageDHT used in combined benchmark; no cached-feature input.')
    if (live/'LIVE_PROTOCOL.json').exists() and D.read(live/'LIVE_PROTOCOL.json') != protocol:
        raise ValueError('Existing live protocol differs')
    D.write(live/'LIVE_PROTOCOL.json', protocol)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = D.ImageDHT(cfg, seed=1, device='cuda')
    heads = {1: (pipeline.head, pipeline.lattice)}
    for seed in (2, 3):
        heads[seed] = D.load_head(cfg, seed, torch.device('cuda'))
    output = []
    with torch.no_grad():
        for index, record in enumerate(records):
            if D.C.sha(Path(record['image'])) != record['image_sha256']:
                raise ValueError(f"Image source bytes changed: {record['id']}")
            image = cv2.imread(record['image'])
            if image is None or image.shape[:2] != (record['height'], record['width']):
                raise ValueError('Image shape mismatch')
            features = pipeline.features(image)
            row = {k: record[k] for k in ('id', 'population', 'group', 'width', 'height')}
            row['seeds'] = {}
            for seed, (head, lattice) in heads.items():
                scores = head(features)
                theta, rho = [v[0].cpu().numpy() for v in D.decode(scores, lattice)]
                lines = D.T.line_pixels(theta, rho, record['width'], record['height'])
                peak = scores.masked_fill(~lattice.valid.flatten()[None, None], -1e9).softmax(-1).max(-1).values[0].cpu().numpy()
                if not np.isfinite(lines).all() or not np.isfinite(peak).all():
                    raise ValueError('Nonfinite live predicted lines or probabilities')
                row['seeds'][str(seed)] = dict(lines=lines.tolist(), theta_deg=theta.tolist(), rho=rho.tolist(), peak_probability=peak.tolist())
            output.append(row)
            if (index+1) % 100 == 0 or index+1 == len(records):
                print(f'Live image DHT {index+1}/{len(records)} x3seeds', flush=True)
    if source_hashes() != identity['source_sha256'] or any(D.C.sha(live/name) != value for name, value in frozen_inputs.items()):
        raise ValueError('Frozen source or inputs changed during live inference')
    cached = {r['id']: r for r in D.read(original/'DHT_LINES.json')['records']}
    changes = []
    for row in output:
        for seed in cfg['dht']['seeds']:
            before, after = cached[row['id']]['seeds'][str(seed)], row['seeds'][str(seed)]
            different = (np.asarray(before['theta_deg']) != np.asarray(after['theta_deg'])) | (np.asarray(before['rho']) != np.asarray(after['rho']))
            if different.any():
                changes.append(dict(id=row['id'], population=row['population'], seed=seed, roles=np.flatnonzero(different).tolist()))
    D.write(output_path, dict(schema='dht_pose_lines_v1', complete=True, **identity, n_frames=len(records),
                             seeds=cfg['dht']['seeds'], records=output, source_cache=None,
                             inference_path='Fresh image feature extraction atbatch1; all3 frozen heads atbatch1; no cached features',
                             cached_comparison=dict(changed_frame_seed_pairs=len(changes), changed_roles=sum(len(r['roles']) for r in changes),
                                                    total_frame_seed_pairs=692*3, total_roles=692*3*8, changes=changes),
                             runtime=dict(torch=torch.__version__, opencv=cv2.__version__, torch_threads=4, opencv_threads=1,
                                          cudnn_benchmark=False, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                          cudnn_tf32=torch.backends.cudnn.allow_tf32),
                             live_protocol_sha256=D.C.sha(live/'LIVE_PROTOCOL.json'),
                             scope='Existing final6000step checkpoints, actual image path; no training, fitting, GT masking or real-based model selection.'))
    print(f'Complete live predictions; changed roles={sum(len(r["roles"]) for r in changes)}/16608; output={output_path}', flush=True)


if __name__ == '__main__':
    main()
