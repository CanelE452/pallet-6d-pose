"""Fresh remaining143 RGB observations for the frozen G10 -> d -> e pipeline.

Only observations are produced here. There is no pose fit, annotation read,
training, or reuse of the original95 masks/measurements. Successful B/I are
preserved at each historical recovery stage, including its original CAL fit
scale. Global geometry guides SAM boxes; initial S never supplies a prompt.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import capture_boundary_recovery_masks_v4_6 as MASK
from . import visible_bi_boundary_recovery_v4_3b as RECOVER
from . import visible_bi_corner_reach_v4_7 as REACH
from .statewise_interval_api_v4_2 import StatewiseIntervalV4_2
from .statewise_masks_v2 import StatewiseMaskProvider
from .state_visibility_v2 import load_state_model
from .statewise_uncertainty_calibration_v4 import load_uncertainty_model, apply_calibrated_measurements
from .truncation_interval_v4_2 import load_truncation_interval_model
from .boundary_partial_observations_v4_4 import fixed_fit_measurements
from .boundary_shared_edge_observations_v4_7 import observe_shared_supported_b_lines
from .uncertain_u_regions_v4_5 import build_u_regions

REMAINDER = C.DOC / 'nonsevere_v1/remainder_after_test49_v1'
INPUT = REMAINDER / 'native_input_preparation_v1'
OUTPUT = REMAINDER / 'fresh_observations_v4_14'
BOUNDARY = E.FIX / 'boundary_recovery_v1'
UNCERTAINTY_MODEL = E.FIX / 'calibration/final_adapter_CAL46/MODEL.json'
INTERVAL_MODEL = E.FIX / 'truncation_interval_model_CAL_v4_2/MODEL.json'
LEGACY_PROTOCOL = BOUNDARY / 'shared_edge_partial_b_eval_v4_7/PROTOCOL.json'
STAGE_NAMES = ('TIGHT_SAM_SUPPORT_GUARDED_V4_3B', 'FULL_SAM_SUPPORT_GUARDED_V4_3B',
               'FULL_SAM_CORNER_REACH_SUPPORT_FIRST_V4_7')
_CONTEXT = None


class RecordingMaskProvider:
    """Record the fresh original provider result, without changing its call."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.last = None

    def __call__(self, image_bgr, **kwargs):
        self.last = self.delegate(image_bgr, **kwargs)
        return self.last


def recover_stages(image, source, tight_mask, full_mask):
    """Keep every already successful B/I exactly at every recovery stage."""
    geometry = source['initial_geometry']; states = source['states']
    initial = {name: deepcopy(geometry[name]) for name in ('R_cf', 'centroid', 'cf_extents')}
    correction = deepcopy(source['visible_correction'])
    raw = deepcopy(correction['raw_points'])
    hidden = set(k for k, flag in enumerate(geometry['initial_self_hidden']) if flag)
    if any(raw[k] != [-1., -1.] for k in hidden):
        raise ValueError('Initial S must be erased before all recovery stages')
    audits = []
    for name, method, mask in zip(STAGE_NAMES, (RECOVER, RECOVER, REACH),
                                  (tight_mask, full_mask, full_mask)):
        previous = correction
        correction = method.recover_visible_boundary(image, raw, geometry['K'], initial,
            states, pallet_mask=mask, base_correction=previous, native_xyz=geometry['native_xyz'])
        before = {m['native_corner']: m for m in previous['visible_measurements']}
        after = {m['native_corner']: m for m in correction['visible_measurements']}
        records = {r['native_corner']: r for r in previous['records']}
        eligible = {k for k in before if states[k] == 'B' and not records[k]['corrected']}
        changed = {k for k in before if before[k] != after.get(k)}
        if set(before) != set(after) or not changed <= eligible:
            raise ValueError('Recovery changed an existing successful B or I observation')
        queried = set(correction.get('recovery_queried_native_ids', []))
        if queried & hidden or any(states[k] != 'B' for k in queried):
            raise ValueError('Recovery queried a non-B or initial S corner')
        if correction['raw_points'] != raw or correction['measured_points'][8:] != previous['measured_points'][8:]:
            raise ValueError('Recovery changed original raw or ninth metadata')
        audits.append(dict(stage=name, eligible_fallback_B_ids=sorted(eligible),
            changed_complete_B_ids=sorted(changed), queried_B_ids=sorted(queried),
            successful_B_and_all_I_measurements_exact=True, S_image_queries=0,
            correction_sha256=E._digest(correction)))
    return correction, audits


def initialize(device):
    global _CONTEXT
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    _, paths, _ = E._runtime_models()
    head = load_state_model(str(paths['state_model']), device)
    tight = RecordingMaskProvider(StatewiseMaskProvider.from_configs(
        C.read(paths['mask_descriptor']), paths['boundary_mask_config'], device))
    full = MASK.GeometryPromptSAMCapture.from_configs(
        C.read(paths['mask_descriptor']), paths['boundary_mask_config'], device)
    model = load_uncertainty_model(UNCERTAINTY_MODEL)
    interval = load_truncation_interval_model(INTERVAL_MODEL)
    config = C.read(LEGACY_PROTOCOL)['solver_configuration']
    _CONTEXT = dict(head=head, ot=C.read(paths['OT_model']), tight=tight, full=full,
                    model=model, interval=interval, config=config, device=device)


def observe_remainder_native_frame_v4_14(native_input, bundle):
    """Return fresh solver-compatible observations and mask arrays; no pose."""
    frame = native_input['frame']; path = C.ROOT / frame['image']
    if frame['partition'] != 'REMAINDER' or C.sha(path) != frame['image_sha256']:
        raise ValueError('Sealed remaining-frame RGB binding required')
    image = cv2.imread(str(path))
    if image is None or list(image.shape[:2]) != frame['raw_hw']:
        raise ValueError('Actual remaining-frame RGB canvas changed')
    bundle['tight'].last = None
    api = StatewiseIntervalV4_2(bundle['head'], bundle['ot'], bundle['tight'],
        uncertainty_model=bundle['model'], truncation_model=bundle['interval'],
        config=dict(prediction_source='N3_SUBPIX', record_id=frame['id'],
                    device=bundle['device'], solver=bundle['config']))
    # The upstream builder already erased initial S before constructing this
    # carrier. Even the frozen API's initial array copy cannot access S raw.
    gated_raw = native_input['raw_points_initial_S_gated']
    if any(gated_raw[k] != [-1., -1.] for k,h in
           enumerate(native_input['initial_geometry']['initial_self_hidden']) if h):
        raise ValueError('Native143 initial S carrier must already be erased')
    source = api.observe(image, gated_raw, frame['K'], frame['native_xyz'],
                         frame['initial_pose'], crop_transform=frame['crop_transform'])
    geometry = source['initial_geometry']
    for key in ('native_xyz', 'K', 'R_cf', 'centroid', 'physical_registration_Q',
                'initial_projection_xy', 'initial_roles', 'initial_self_hidden', 'image_hw'):
        if geometry[key] != native_input['initial_geometry'][key]:
            raise ValueError('Fresh RGB changed the initial registered geometry: ' + key)
    tight = bundle['tight'].last
    if tight is None:
        # The frozen API intentionally skips masks when no non-S RGB query
        # exists. No B can then request local recovery or partial evidence.
        zero = np.zeros(image.shape[:2], np.float32)
        tight = dict(pallet_mask=zero, state_pallet_mask=zero.copy(),
                     provenance=dict(reason='frozen_API_no_gated_RGB_queries', mask_forward_calls=0))
    prompt = MASK.build_geometry_prompt(geometry, source['original_raw_points'], source['states'])
    # Reuse the loaded network object; the full-box call still executes its
    # own actual image encoder and decoder with the frozen numerical flags.
    bundle['full'].predictor = bundle['tight'].delegate.predictor
    full = bundle['full'](image, prompt=prompt, state_pallet_mask=tight['state_pallet_mask'])
    if not np.array_equal(full['state_pallet_mask'], tight['state_pallet_mask']):
        raise ValueError('Tiny state mask changed in full-box capture')
    original_fit = apply_calibrated_measurements(source['frozen_visible_measurements'],
        bundle['model'], states=source['states'],
        nonself_scale_px=source['ot_domains']['nonself_prediction_scale_px'])
    source['calibrated_visible_measurements'] = deepcopy(original_fit)
    correction, stages = recover_stages(image, source, tight['pallet_mask'], full['pallet_mask_clean'])
    fit = fixed_fit_measurements(original_fit, correction)
    regions = build_u_regions(source, bundle['interval'])
    partial = observe_shared_supported_b_lines(correction, geometry, source['states'], fit,
        localization_model=bundle['interval'],
        nonself_scale_px=source['ot_domains']['nonself_prediction_scale_px'],
        image_binding=dict(sha256=frame['image_sha256'], image_hw=geometry['image_hw']))
    hidden = {k for k, h in enumerate(geometry['initial_self_hidden']) if h}
    if (hidden.intersection(source['mask_audit']['prompt_native_ids'])
            or hidden.intersection(source['state_observation'].get('rgb_queried_ids', []))
            or any(m['native_corner'] in hidden for m in fit)
            or any(d['native_corner'] in hidden for d in regions['u_regions'])):
        raise ValueError('Initial S leaked into an image observation or region')
    row = dict(schema='FRESH_REMAINDER_RGB_LEGACY_V47_OBSERVATIONS_V4_14',
        id=frame['id'], partition='REMAINDER', session=frame['session'], cohort=frame['cohort'],
        image=frame['image'], image_sha256=frame['image_sha256'], states=source['states'],
        known_physical_dimensions_xyz_m=deepcopy(frame['known_physical_dimensions_xyz_m']),
        prediction_object_type=frame['prediction_object_type'],
        runtime={'initial_geometry':geometry}, _original_source=source,
        native_initial_S_gated_input_sha256=E._digest(dict(id=frame['id'],
            initial_geometry=native_input['initial_geometry'], gated_native9=gated_raw,
            image_sha256=frame['image_sha256'])), old_correction=source['visible_correction'],
        correction=correction, frozen_BI_fit=fit, original_CAL_fit=original_fit,
        partial_boundary_observation=partial, u_region_observation=regions,
        recovery_stage_audit=stages,
        full_mask_provenance=full['provenance'], tight_mask_provenance=tight['provenance'],
        contract=dict(initial_geometry_native_Q_and_ninth_fixed=True, S_image_queries=0,
            initial_S_raw_coordinates_not_supplied_to_observer=True,
            effective_CAL_covariance_and_reliability_held_fixed=True,
            GT_inputs_to_inference=False, annotations_read=False, pose_solver_calls=0,
            old95_masks_or_observations_reused=False, actual_current_RGB_forward=True))
    arrays = {k:v for k,v in full.items() if k != 'provenance'}
    arrays['pallet_mask_tight'] = tight['pallet_mask']
    return C.finite(row), arrays


def _run(task):
    native_input, output = task; start = time.monotonic()
    row, arrays = observe_remainder_native_frame_v4_14(native_input, _CONTEXT)
    folder = Path(output); stem = row['id'].replace(':', '__')
    mask = folder / 'masks' / (stem + '.npz')
    if mask.exists(): raise FileExistsError('Preserve mask evidence')
    np.savez_compressed(mask, **arrays)
    row['mask_evidence'] = dict(mask_npz=mask.name, mask_file_sha256=C.sha(mask),
        image_sha256=row['image_sha256'], image=row['image'],
        tight_mask_pixel_sha256=E._digest(arrays['pallet_mask_tight']),
        clean_mask_pixel_sha256=E._digest(arrays['pallet_mask_clean']))
    row['elapsed_seconds'] = time.monotonic()-start
    C.write(folder / 'frames' / (stem + '.json'), row)
    return row


def _inputs():
    seal = C.read(INPUT / 'INPUT_SEAL.json'); protocol = C.read(INPUT / 'PROTOCOL.json')
    journal = INPUT / 'NATIVE_INPUTS143.jsonl.gz'
    if (seal['rows'] != 143 or seal['GT_values_decoded'] or
            C.sha(journal) != seal['native_inputs_sha256'] or
            C.sha(INPUT / 'PROTOCOL.json') != seal['protocol_sha256']):
        raise ValueError('Exact sealed annotation-free native143 required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(seal[key])
    rows = list(C.rows(journal))
    if [r['id'] for r in rows] != seal['ordered_ids'] or len({r['id'] for r in rows}) != 143:
        raise ValueError('Exact ordered unique143 required')
    return rows, seal, protocol


def predict(output=OUTPUT, workers=3, device='cuda'):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve completed or partial observation run')
    if workers not in (1,2,3): raise ValueError('One to three bounded workers')
    rows, native_seal, _ = _inputs()
    _, paths, inputs = E._runtime_models()
    sources = E.source_bindings()
    for name in ('observe_remainder_rgb_v4_14.py', 'statewise_interval_api_v4_2.py',
                 'statewise_bias_api_v4_1.py', 'truncation_interval_v4_2.py',
                 'capture_boundary_recovery_masks_v4_6.py', 'visible_bi_boundary_recovery_v4_3.py',
                 'visible_bi_boundary_recovery_v4_3b.py', 'visible_bi_corner_reach_v4_7.py',
                 'boundary_partial_observations_v4_4.py', 'boundary_partial_observations_v4_6.py',
                 'boundary_shared_edge_observations_v4_7.py', 'uncertain_u_regions_v4_5.py'):
        path = Path(__file__).parent / name; sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    for path in (INPUT / 'NATIVE_INPUTS143.jsonl.gz', INPUT / 'INPUT_SEAL.json',
                 INPUT / 'PROTOCOL.json', UNCERTAINTY_MODEL, INTERVAL_MODEL, LEGACY_PROTOCOL):
        inputs[str(path.relative_to(C.ROOT))] = C.sha(path)
    assets_path = C.DOC / 'MOBILESAM_ASSET_MANIFEST.json'; assets = C.read(assets_path)
    inputs[str(assets_path.relative_to(C.ROOT))] = C.sha(assets_path)
    boundary = C.read(paths['boundary_mask_config'])
    vendor = (paths['boundary_mask_config'].parent / boundary['mobile_sam_vendor']).resolve()
    for name, expected in assets['runtime_files_sha256'].items():
        path = vendor / name
        if C.sha(path) != expected: raise ValueError('Frozen MobileSAM vendor changed')
        inputs[str(path.relative_to(C.ROOT))] = expected
    for row in rows:
        frame = row['frame']; inputs[frame['image']] = frame['image_sha256']
    protected = native_seal['protected_user_sha256']
    for bindings in (sources, inputs, protected): E._verify_sources(bindings)
    output.mkdir(parents=True); (output/'masks').mkdir(); (output/'frames').mkdir()
    for relative, digest in sources.items():
        dest = output/'code_snapshot'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative, dest)
        if C.sha(dest) != digest: raise ValueError('Observation source snapshot mismatch')
    import torch
    protocol = dict(schema='FRESH_REMAINDER143_RGB_OBSERVATION_PROTOCOL_V4_14',
        ids=[r['id'] for r in rows], rows=143, severe_frames=0, workers=workers, device=device,
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        annotations_read=False, GT_or_pose_labels_read=False, pose_solver_calls=0,
        pretrained_models_and_CAL_models_refitted=False, historical143_is_not_new_unseen_holdout=True,
        state_API='StatewiseIntervalV4_2.observe ONLY', recovery_stages=list(STAGE_NAMES),
        solver_configuration=C.read(LEGACY_PROTOCOL)['solver_configuration'],
        uncertainty_model=E._binding(UNCERTAINTY_MODEL), interval_model=E._binding(INTERVAL_MODEL),
        original_CAL_effective_covariance_and_reliability_preserved=True,
        full_mask_configuration=MASK.configuration(), old95_cache_reused=False,
        S_image_queries=0, native9_channels_and_Q_preserved=True,
        runtime_versions=dict(torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__),
        torch_flags=dict(cudnn_benchmark=False,cudnn_deterministic=True,cudnn_allow_tf32=False,matmul_allow_tf32=False))
    C.write(output/'PROTOCOL.json', protocol)
    start=time.monotonic(); completed={}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=initialize, initargs=(device,)) as pool:
        futures={pool.submit(_run,(row,str(output))):row['id'] for row in rows}
        try:
            for future in as_completed(futures):
                row=future.result(); completed[row['id']]=row
                print(f"observed {len(completed)}/143 {row['id']} {time.monotonic()-start:.1f}s", flush=True)
        except BaseException:
            for future in futures: future.cancel()
            C.write(output/'ABORTED_BEFORE_POSE_OR_SCORING.json',
                dict(completed=len(completed), completed_ids=list(completed), prediction_seal_created=False,
                     annotations_read=False, pose_solver_calls=0))
            raise
    ordered=[completed[r['id']] for r in rows]
    C.save_rows(output/'OBSERVATIONS.jsonl.gz', ordered)
    records=[dict(id=r['id'], **r['mask_evidence'], provenance=r['full_mask_provenance']) for r in ordered]
    C.write(output/'masks/RECORDS.json', records)
    C.write(output/'masks/PROTOCOL.json', protocol)
    for bindings in (sources, inputs, protected): E._verify_sources(bindings)
    for record in records:
        if C.sha(output/'masks'/record['mask_npz']) != record['mask_file_sha256']:
            raise ValueError('Fresh mask evidence changed')
    mask_seal=dict(records=143, ordered_ids=protocol['ids'], GT_or_pose_labels_read=False,
        records_sha256=C.sha(output/'masks/RECORDS.json'), protocol_sha256=C.sha(output/'masks/PROTOCOL.json'),
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected)
    C.write(output/'masks/SEAL.json', mask_seal)
    C.write(output/'OBSERVATION_SEAL.json', dict(rows=143, ordered_ids=protocol['ids'],
        observations_sha256=C.sha(output/'OBSERVATIONS.jsonl.gz'), protocol_sha256=C.sha(output/'PROTOCOL.json'),
        masks_seal=E._binding(output/'masks/SEAL.json'), source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=protected, annotations_read=False, GT_or_pose_labels_read=False,
        before_remaining_pose_prediction_and_scoring=True, pose_solver_calls=0,
        old95_cache_reused=False, all143_real_RGB_observations=True,
        elapsed_seconds=time.monotonic()-start))
    print('sealed fresh143 observations and masks; no poses or labels', flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(); predict(args.output,args.workers,args.device)


if __name__ == '__main__': main()
