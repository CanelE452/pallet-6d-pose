"""Run the validated CAL interval S/B/I/O/T method on one actual image."""
import argparse
from pathlib import Path

import cv2

from . import common as C
from .evaluation_interval_v4_2 import load_interval_configuration, MODEL, source_bindings
from .statewise_evaluation_v4 import _runtime_models, FRESH, _verify_sources
from .state_visibility_v2 import load_state_model
from .statewise_masks_v2 import StatewiseMaskProvider
from .statewise_interval_api_v4_2 import StatewiseIntervalV4_2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing inference output')
    selected, uncertainty, solver, interval, bindings = load_interval_configuration(args.model)
    _, paths, model_bindings = _runtime_models()
    observation_seal = C.read(FRESH / 'OBSERVATION_SEAL.json')
    if C.sha(FRESH / 'PROTOCOL.json') != observation_seal['protocol_sha256']:
        raise ValueError('Frozen fresh-observation protocol changed')
    expected_runtime = C.read(FRESH / 'PROTOCOL.json')['frozen_runtime_descriptor']
    if model_bindings.get(expected_runtime['path']) != expected_runtime['sha256']:
        raise ValueError('Runtime descriptor differs from the frozen observation pipeline')
    input_sha, image_sha = C.sha(args.input), C.sha(args.image)
    sources = source_bindings()
    sources[str(Path(__file__).relative_to(C.ROOT))] = C.sha(__file__)
    _verify_sources(sources)
    inputs = C.read(args.input)
    pose = inputs['initial_pose']
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError('Actual image could not be decoded')
    head = load_state_model(str(paths['state_model']), args.device)
    masks = StatewiseMaskProvider.from_configs(
        C.read(paths['mask_descriptor']), paths['boundary_mask_config'], args.device)
    api = StatewiseIntervalV4_2(head, C.read(paths['OT_model']), masks,
        uncertainty_model=uncertainty, truncation_model=interval,
        config={'solver': solver, 'device': args.device, 'prediction_source': 'N3_SUBPIX',
                'record_id': inputs.get('record_id', 'actual_image')})
    result = api.refine(image, inputs['raw_points'], inputs['K'],
        inputs.get('native_xyz', C.cuboid(*pose['cf_extents'])), pose,
        independent_truncation=inputs.get('independent_truncation'),
        crop_transform=inputs.get('crop_transform'))
    after_selection = load_interval_configuration(args.model)
    if after_selection[-1] != bindings:
        raise RuntimeError('CAL model/validation bindings changed during inference')
    _, _, after_models = _runtime_models()
    if model_bindings != after_models:
        raise RuntimeError('Runtime model bindings changed during inference')
    if C.sha(args.input) != input_sha or C.sha(args.image) != image_sha:
        raise RuntimeError('Input/image file changed during inference')
    _verify_sources(sources)
    result['runtime_execution_bindings'] = {
        **bindings,
        'CAL_uncertainty_model': selected['calibration_model'],
        'frozen_observation_runtime_descriptor': expected_runtime,
        'RGB_provider_model_sha256': model_bindings,
        'image_file_sha256': image_sha, 'input_file_sha256': input_sha,
        'parameters_from_CAL_only': True,
        'adapter_and_cli_source_sha256': sources}
    C.write(args.output, result)
    print('states='+''.join(result['states'])+' joint_feasible='+str(result['accepted'])
          +' pose_source='+result['actual_pose']['source'])


if __name__ == '__main__':
    main()
