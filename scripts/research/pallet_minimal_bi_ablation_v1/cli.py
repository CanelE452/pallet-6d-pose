"""Single-image minimal B/I observe/solve CLI; no reference/scoring interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import runtime as R


INPUT_KEYS = {'schema', 'id', 'prediction_source', 'image', 'image_sha256',
              'raw_hw', 'K', 'native_points', 'coarse_pose'}
COARSE_KEYS = {'available', 'R_cf', 'R_physical', 'centroid', 'cf_extents',
               'physical_registration_Q', 'source', 'selected_hypothesis', 'reprojection_px'}


def validate_input(value):
    """Only predictor fields are admitted; reference/annotation paths have no API."""
    if not isinstance(value, dict) or set(value) - INPUT_KEYS:
        raise ValueError('Only explicit predictor input fields are permitted')
    if not {'K', 'native_points', 'coarse_pose'} <= set(value):
        raise ValueError('K, native_points and own coarse_pose are required')
    coarse = value['coarse_pose']
    if not isinstance(coarse, dict) or set(coarse) - COARSE_KEYS:
        raise ValueError('Only own predictor coarse-pose fields are permitted')
    if not {'R_cf', 'R_physical', 'centroid', 'cf_extents'} <= set(coarse):
        raise ValueError('Own native and physical pose basis must be explicit')
    return value


def _write_new(path, value):
    path = Path(path)
    # The caller creates a new output directory; files never overwrite attempts.
    with path.open('x') as stream:
        json.dump(R._json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('observe', 'solve', 'run'))
    parser.add_argument('--input', type=Path, required=True,
                        help='Predictor-only JSON: K/native_points/own coarse_pose/image')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory; every existing attempt is preserved')
    parser.add_argument('--observation', type=Path,
                        help='Saved observation JSON, required for solve')
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--state-descriptor', type=Path)
    parser.add_argument('--mask-descriptor', type=Path)
    parser.add_argument('--boundary-config', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Preserve completed and partial output directories')
    # Validation precedes image/model IO. This command never opens refs or scores.
    source = validate_input(json.loads(args.input.read_text()))
    input_sha = R.sha256(args.input)
    if args.stage == 'solve':
        if args.observation is None:
            parser.error('solve requires --observation')
        stored = json.loads(args.observation.read_text())
        observation = stored.get('observation', stored)
        hw = source.get('raw_hw', observation['initial_geometry']['image_hw'])
        args.output.mkdir(parents=True)
        result = R.solve(observation, source['coarse_pose'], source['K'], hw)
        _write_new(args.output / 'POSE.json', result)
        bindings = dict(input_sha256=input_sha,
                        observation_sha256=R.sha256(args.observation))
    else:
        required = ('asset_root', 'state_descriptor', 'mask_descriptor', 'boundary_config')
        if any(getattr(args, key) is None for key in required) or 'image' not in source:
            parser.error('observe/run require image in input and all explicit model configuration paths')
        image_path = Path(source['image'])
        if not image_path.is_absolute():
            image_path = args.input.parent / image_path
        image_sha = R.sha256(image_path)
        if 'image_sha256' in source and image_sha != source['image_sha256']:
            raise ValueError('Predictor image SHA changed')
        image = cv2.imread(str(image_path))
        if image is None or ('raw_hw' in source and list(image.shape[:2]) != source['raw_hw']):
            raise ValueError('Actual source RGB and canvas are required')
        components = R.load_components(asset_root=args.asset_root,
            state_descriptor=args.state_descriptor, mask_descriptor=args.mask_descriptor,
            boundary_config=args.boundary_config, device=args.device)
        cv2.setNumThreads(1)
        import torch
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        args.output.mkdir(parents=True)
        observed = R.observe(image, source['native_points'], source['K'], source['coarse_pose'],
            components=components, prediction_source=source.get('prediction_source', 'RAW_YOLO'),
            record_id=source.get('id', 'current'))
        np.savez_compressed(args.output / 'MASKS.npz', **observed.pop('masks'))
        _write_new(args.output / 'OBSERVATION.json', observed)
        if args.stage == 'run':
            _write_new(args.output / 'POSE.json', R.solve(observed['observation'],
                source['coarse_pose'], source['K'], image.shape[:2]))
        bindings = dict(input_sha256=input_sha, image_sha256=image_sha,
                        component_bindings=components.bindings)
    _write_new(args.output / 'RUN_RECEIPT.json', dict(schema=R.SCHEMA,
        stage=args.stage, parity_policy=R.PARITY_POLICY, bindings=bindings,
        assets={p.name: R.sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        reference_or_GT_inputs=False, new_N3_or_YOLO_forwards=0,
        U_display_regions=False, OT_regions=False, C_or_old_bank=False))
    return args.output


if __name__ == '__main__':
    main()
