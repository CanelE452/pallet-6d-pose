"""Opt-in global-grid MobileSAM diagnostic after a detector missed a stool.

One fixed CAL image; no detector, semantic point, pose, state or keypoint input.
All vendor-filtered masks remain candidates with identity and depth UNKNOWN.
Preparation is metadata-only. Actual model/image loading needs a separate GO.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from . import capture_generic_instance_mask_candidates_v1 as P

C = P.C
B = P.B
CASE_ID = 'eval_outside:1778651519235162880'
PARENT_PLAN = B / 'generic_instance_mask_population_CAL_plan_v1/PREDECLARATION.json'
PARENT_PLAN_SHA = 'd8e3a79b1fa96cfd2972f6759ec334858280e8ed86fbd7ff67b347c71029bb4e'
PARENT_OUTPUT = B / 'generic_instance_mask_candidates_CAL46_v1'
PARENT_SEAL_SHA = '1065a9834226a6155eaaff49c902f1ca54ad9b733409883dde9a2b060d30b453'
PLAN = B / 'class_agnostic_AMG_case_plan_v1/PREDECLARATION.json'
OUTPUT = B / 'class_agnostic_AMG_CAL_case_v1'
SCHEMA = 'STANDALONE_GLOBAL_GRID_MOBILESAM_CANDIDATES_FIXED_CAL_CASE_V1'
AMG = P.VENDOR / 'mobile_sam/automatic_mask_generator.py'
CONFIGURATION = dict(points_per_side=16, points_per_batch=64, pred_iou_thresh=.88,
    stability_score_thresh=.95, stability_score_offset=1., box_nms_thresh=.7,
    crop_n_layers=0, crop_nms_thresh=.7, crop_overlap_ratio=512 / 1500,
    crop_n_points_downscale_factor=1, point_grids=None, min_mask_region_area=0,
    output_mode='binary_mask')


def global_grid(image_hw):
    """Same cell-centered whole-image grid as the pinned vendor; no local input."""
    h, w = image_hw
    axis = np.linspace(1 / 32, 1 - 1 / 32, 16)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx, yy], axis=-1).reshape(-1, 2) * [w, h]


def infer_case(image_bgr, target_mask, generator, *, save_mask):
    """Injected AMG for pure tests; the target is only post-inference context."""
    image, target = np.asarray(image_bgr), np.asarray(target_mask, float)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Unmodified full-canvas uint8 BGR required')
    h, w = image.shape[:2]
    if target.shape != (h, w) or not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError('Frozen same-canvas target occupancy required')
    old_image, old_target = image.copy(), target.copy()
    annotations = list(generator.generate(image[:, :, ::-1].copy()))
    if len(annotations) > 768:
        raise ValueError('Returned masks exceed declared256x3 raw slots')
    records = []
    grid = global_grid((h, w))
    for i, annotation in enumerate(annotations):
        mask = np.asarray(annotation['segmentation'])
        if mask.shape != (h, w) or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError('Vendor returned a nonbinary or non-native-canvas mask')
        mask = np.asarray(mask, np.uint8)
        points = np.asarray(annotation['point_coords'], float)
        box, crop = np.asarray(annotation['bbox'], float), np.asarray(annotation['crop_box'], float)
        scores = np.asarray([annotation['predicted_iou'], annotation['stability_score']], float)
        if (points.shape != (1, 2) or box.shape != (4,) or crop.shape != (4,)
                or not np.isfinite([*points.ravel(), *box, *crop, *scores]).all()
                or np.min(np.linalg.norm(grid - points[0], axis=1)) > 1e-4
                or not np.array_equal(crop, [0, 0, w, h])):
            raise ValueError('Exact finite full-image uniform-grid AMG provenance required')
        if int(annotation['area']) != int(mask.sum()):
            raise ValueError('Vendor mask area disagrees with actual returned pixels')
        overlap = P.overlap_metrics(mask, target)
        status, reason = P.status_from_overlap(overlap)
        records.append(dict(candidate_index=i, candidate_id=f'{CASE_ID}/AMG_{i:03d}',
            candidate_file_binding=save_mask(i, mask.copy()), mask_pixel_sha256=P.mask_pixel_sha256(mask),
            image_hw=[h, w], area_pixels=int(mask.sum()), bbox_xywh=box.tolist(),
            crop_box_xywh=crop.tolist(), global_prompt_point_xy=points.tolist(),
            prompt_source='UNIFORM_FULL_IMAGE_16_BY_16_GRID_NOT_A_KEYPOINT',
            predicted_iou=float(scores[0]), stability_score=float(scores[1]),
            scores_are_semantic_or_depth_probabilities=False, target_overlap=overlap,
            status=status, status_reason=reason, target_identity='UNKNOWN',
            other_object_confirmed=False, foreground_occlusion_confirmed=False,
            precise_keypoint_created=False, supplied_to_current_pipeline=False))
    if not np.array_equal(image, old_image) or not np.array_equal(target, old_target):
        raise ValueError('Input image/target changed')
    return dict(schema=SCHEMA, id=CASE_ID, partition='CAL', image_hw=[h, w],
        status='CANDIDATES_AVAILABLE' if records else 'UNAVAILABLE',
        unavailable_reason=None if records else 'no_masks_survived_fixed_vendor_filters',
        candidates=records, returned_candidates=len(records), all_returned_masks_preserved=True,
        preserved_population='vendor post-quality/stability/crop-edge/box-NMS output, not all raw768 slots',
        expected_image_encoders=1, expected_prompt_decoder_batches=4, global_grid_prompts=256,
        raw_multimask_slots_before_vendor_filters=768, detector_calls=0,
        target_mask_used_only_for_postprediction_overlap=True,
        target_complement_used_as_occluder=False, KP_S_U_O_local_prompt_queries=0,
        global_uniform_point_prompts_are_not_S_U_local_queries=True,
        current_pipeline_mask_supply=0, new_GT_reference_or_primary_reads=0,
        other_object_or_depth_certificates=0, pose_noise_fits=0)


def _literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _literal(node.left) / _literal(node.right)
    raise ValueError('Unexpected vendor default expression')


def verify_vendor_defaults():
    tree = ast.parse(AMG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SamAutomaticMaskGenerator')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    names = [n.arg for n in init.args.args][-len(init.args.defaults):]
    defaults = {key: _literal(value) for key, value in zip(names, init.args.defaults)}
    expected = {**CONFIGURATION, 'points_per_side': 32}
    if defaults != expected:
        raise ValueError('Pinned vendor AMG defaults differ; do not silently adjust recipe')
    return defaults


def _sources():
    paths = [Path(__file__), Path(__file__).with_name('test_capture_class_agnostic_AMG_case_v1.py'),
             Path(P.__file__), Path(C.__file__)]
    manifest = C.read(C.DOC / 'MOBILESAM_ASSET_MANIFEST.json')
    for name, digest in manifest['runtime_files_sha256'].items():
        path = P.VENDOR / name
        if C.sha(path) != digest:
            raise ValueError('Pinned MobileSAM bytes changed')
        if name.endswith('.py'):
            paths.append(path)
    return {P._binding(path)['path']: C.sha(path) for path in paths}


def prepare(*, root_metadata_GO=False):
    if root_metadata_GO is not True:
        raise PermissionError('Source/metadata GO before IO')
    if PLAN.exists():
        raise FileExistsError('Preserve prior declaration')
    if C.sha(PARENT_PLAN) != PARENT_PLAN_SHA or C.sha(PARENT_OUTPUT / 'EXECUTION_SEAL.json') != PARENT_SEAL_SHA:
        raise ValueError('Exact completed CAL46 diagnostic ancestry required')
    parent = C.read(PARENT_PLAN)
    case = next(row for row in parent['cases'] if row['id'] == CASE_ID)
    if parent['ordered_ids'].index(CASE_ID) != 21 or case['partition'] != 'CAL':
        raise ValueError('Same explicit postdiagnostic case21 required')
    for mapping in (parent['source_sha256'], parent['input_sha256'], parent['protected_user_sha256']):
        P._verify(mapping)
    defaults = verify_vendor_defaults()
    sources = _sources()
    inputs = {P._binding(path)['path']: C.sha(path) for path in (
        PARENT_PLAN, PARENT_PLAN.parent / 'PLAN_SEAL.json',
        PARENT_OUTPUT / 'EXECUTION_SEAL.json', PARENT_OUTPUT / 'case21.json',
        PARENT_OUTPUT / 'case21_candidates.png', P.SAM,
        C.DOC / 'MOBILESAM_ASSET_MANIFEST.json',
        P._resolve(case['image_binding']['path']), P._resolve(case['target_mask_binding']['path']))}
    value = dict(schema=SCHEMA, UTC=datetime.now(timezone.utc).isoformat(),
        case=case, case_selection='POSTDIAGNOSTIC CAL46 case21: visible blue stool missed by fixedgeneric detector; no GT/pose/error routing',
        configuration=deepcopy(CONFIGURATION), verified_vendor_defaults=defaults,
        checkpoint=P._binding(P.SAM), vendor_root=str(P.VENDOR.relative_to(C.ROOT)),
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=parent['protected_user_sha256'],
        workers=1, device='cpu', BLAS_threads=1, max_actual_images_in_memory=1,
        expected_encoder_calls=1, expected_decoder_batches=4, global_points=256,
        expected_raw_multimask_slots=768,
        proposal_semantics='class-agnostic UNKNOWN candidates, no target/non-target/foreground-depth certification',
        retained_masks='all vendor-filtered/NMS output in original vendor order; no new mask scoring/ranking',
        target_context='same frozen pallet mask only for postprediction overlap; no complement, identity or inference input',
        prompt_contract='uniformglobal grid only; no KP/S/U/O coordinates or boxes, detector, clicks or ROI',
        actual_image_target_mask_model_decodes_during_prepare=0,
        separate_actual_root_GO_required=True, current_pipeline_changes=0,
        prior_frozen_artifacts_changed=False, primary_labels_or_pose_noise_fit=0,
        expected_output=str(OUTPUT.relative_to(C.ROOT)))
    C.write(PLAN, value)
    for path, digest in sources.items():
        target = PLAN.parent / 'code_snapshot' / path.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(P._resolve(path), target)
        if C.sha(target) != digest:
            raise ValueError('Snapshot changed')
    for mapping in (sources, inputs, parent['protected_user_sha256']):
        P._verify(mapping)
    C.write(PLAN.parent / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN),
        source_sha256=sources, actual_image_model_or_mask_decodes=0))
    return dict(plan_sha256=C.sha(PLAN), sources=len(sources), inputs=len(inputs),
                protected=len(parent['protected_user_sha256']))


def load_generator():
    """Local pinned checkpoint only. Called after the separate actual GO."""
    sys.path.insert(0, str(P.VENDOR))
    import torch
    import mobile_sam
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    if Path(mobile_sam.__file__).resolve() != (P.VENDOR / 'mobile_sam/__init__.py').resolve():
        raise ValueError('Unexpected MobileSAM module')
    if C.sha(P.SAM) != P.SAM_SHA:
        raise ValueError('Pinned local checkpoint required; no downloads')
    torch.set_num_threads(1)
    model = sam_model_registry['vit_t'](checkpoint=str(P.SAM)).to(device='cpu')
    model.eval()
    if inspect.signature(SamAutomaticMaskGenerator.__init__).parameters['points_per_side'].default != 32:
        raise ValueError('Unexpected imported AMG API')
    return SamAutomaticMaskGenerator(model, **deepcopy(CONFIGURATION))


def _render(image_bgr, row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb = image_bgr[:, :, ::-1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.imshow(rgb); ax.axis('off')
    axes[0].set_title('Original RGB; unchanged brightness')
    grid = global_grid(image_bgr.shape[:2])
    axes[1].scatter(grid[:, 0], grid[:, 1], s=5, color='cyan')
    axes[1].set_title('Uniform global256 prompts; not keypoints')
    for candidate in row['candidates']:
        with np.load(P._resolve(candidate['candidate_file_binding']['path']), allow_pickle=False) as arrays:
            mask = arrays['segmentation'].astype(bool)
        if mask.any() and not mask.all():
            axes[2].contour(mask, levels=[.5], linewidths=.6)
    axes[2].set_title(f"All {row['returned_candidates']} vendor-filtered masks; identity/depth UNKNOWN")
    fig.suptitle(CASE_ID + ' | standalone global AMG diagnostic, no pose pipeline supply')
    path = OUTPUT / 'overview_candidates.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    assets = [P._binding(path)]
    for start in range(0, len(row['candidates']), 12):
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for ax in axes.ravel():
            ax.axis('off')
        for ax, candidate in zip(axes.ravel(), row['candidates'][start:start + 12]):
            with np.load(P._resolve(candidate['candidate_file_binding']['path']), allow_pickle=False) as arrays:
                mask = arrays['segmentation'].astype(bool)
            ax.imshow(rgb); overlay = np.zeros((*mask.shape, 4)); overlay[mask] = [.1, .8, .2, .45]
            ax.imshow(overlay); ax.set_title(f"AMG{candidate['candidate_index']} {candidate['status']}\nIoUq={candidate['predicted_iou']:.3f} stab={candidate['stability_score']:.3f}")
        path = OUTPUT / f'gallery_{start // 12:02d}.png'
        fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig); assets.append(P._binding(path))
    return assets


def run(*, expected_plan_sha256, root_actual_AMG_GO=False):
    if root_actual_AMG_GO is not True:
        raise PermissionError('Separate actual AMG GO required before IO')
    if C.sha(PLAN) != expected_plan_sha256 or C.read(PLAN.parent / 'PLAN_SEAL.json')['plan_sha256'] != expected_plan_sha256:
        raise ValueError('Exact declared plan required')
    plan = C.read(PLAN)
    if plan['configuration'] != CONFIGURATION or plan['case']['id'] != CASE_ID:
        raise ValueError('Frozen case/configuration required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        P._verify(plan[key])
    if OUTPUT.exists():
        raise FileExistsError('Preserve every attempt')
    OUTPUT.mkdir(parents=True); (OUTPUT / 'masks').mkdir(); started = time.monotonic()
    C.write(OUTPUT / 'EXECUTION_START.json', dict(plan_sha256=expected_plan_sha256,
        UTC=datetime.now(timezone.utc).isoformat()))
    try:
        generator = load_generator()
        import cv2
        cv2.setNumThreads(1)
        case = plan['case']
        image = cv2.imread(str(P._resolve(case['image_binding']['path'])))
        if image is None:
            raise ValueError('Original RGB failed to decode')
        with np.load(P._resolve(case['target_mask_binding']['path']), allow_pickle=False) as arrays:
            target = arrays['pallet_mask_clean'].copy()
        def save(index, mask):
            path = OUTPUT / 'masks' / f'candidate_{index:03d}.npz'
            if path.exists():
                raise FileExistsError('Do not replace returned candidates')
            np.savez_compressed(path, segmentation=mask)
            return P._binding(path)
        row = infer_case(image, target, generator, save_mask=save)
        row.update(image_binding=case['image_binding'], target_mask_binding=case['target_mask_binding'],
                   figures=_render(image, row))
        C.write(OUTPUT / 'RESULT.json', row)
        text = '# 전역 class-agnostic AMG 후보 진단\n\n고정 CAL case21의 detector 스툴 누락 이후 선언한 진단입니다. 전체 grid 후보이며 object identity/target/앞뒤 관계의 정답이 아닙니다. 기존 pose pipeline에는 공급하지 않았습니다.\n\n'
        for figure in row['figures']:
            text += '- [actual candidates](' + Path(figure['path']).name + ')\n'
        (OUTPUT / 'REPORT_KO.md').write_text(text)
        for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
            P._verify(plan[key])
        C.write(OUTPUT / 'EXECUTION_SEAL.json', dict(complete=True,
            plan_sha256=expected_plan_sha256, source_sha256=plan['source_sha256'],
            input_sha256=plan['input_sha256'], protected_user_sha256=plan['protected_user_sha256'],
            returned_candidates=row['returned_candidates'], elapsed_seconds=time.monotonic() - started,
            assets={str(path.relative_to(OUTPUT)): C.sha(path) for path in OUTPUT.rglob('*') if path.is_file()},
            current_pipeline_changes=0, labels_or_primary_or_pose_noise_fit=0))
    except BaseException as exc:
        C.write(OUTPUT / 'FAILURE_RECEIPT.json', dict(complete=False,
            error_type=type(exc).__name__, error=str(exc), attempt_preserved=True))
        raise
    return dict(output=str(OUTPUT.relative_to(C.ROOT)), seal_sha256=C.sha(OUTPUT / 'EXECUTION_SEAL.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--root-metadata-GO', action='store_true')
    parser.add_argument('--root-actual-AMG-GO', action='store_true')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    print(json.dumps(prepare(root_metadata_GO=args.root_metadata_GO) if args.stage == 'prepare'
        else run(expected_plan_sha256=args.expected_plan_sha256, root_actual_AMG_GO=args.root_actual_AMG_GO)), flush=True)
