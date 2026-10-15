"""Standalone fixed-CAL3 generic detector -> box-prompted SAM candidates.

No keypoint/state/pose input exists in the supplier API. A predicted class and
mask are candidates, not a non-target ownership or foreground-depth certificate.
Preparation reads metadata/opaque hashes only; inference needs a separate GO.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

for _thread_key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_key] = '1'

import numpy as np

from . import common as C

B = C.DOC / 'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
PLAN = B / 'generic_instance_mask_candidates_plan_v1/PREDECLARATION.json'
OUTPUT = B / 'generic_instance_mask_candidates_CAL3_v1'
PARENT = B / 'observed_bi_line_association_CAL_process_v4_30_v2'
ASSETS = B / 'automatic_non_target_instance_mask_feasibility_v1'
VENDOR = C.DOC / 'foundation_mask_v1/vendor/MobileSAM'
ULTRA = Path('/home/minjae/anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics')
CASES = ('wood_184309:000925', 'wood_184309:000804', 'eval_cad:1778653017736058368')
DETECTOR = C.ROOT / 'challenge/weights/pretrained_yolo/yolo26n.pt'
DETECTOR_SHA = '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'
SAM = VENDOR / 'weights/mobile_sam.pt'
SAM_SHA = '6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'
CONFIGURATION = dict(conf=.25, imgsz=640, max_det=20, iou=.7, batch=1,
    half=False, device='cpu', agnostic_nms=False, save=False, verbose=False)
SCHEMA = 'GENERIC_DETECTOR_BOX_SAM_INSTANCE_CANDIDATES_FIXED_CAL3_V1'


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(C.ROOT)) if path.is_relative_to(C.ROOT) else str(path),
                sha256=C.sha(path), bytes=path.stat().st_size)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else C.ROOT / path


def _verify(mapping):
    for path, expected in mapping.items():
        if C.sha(_resolve(path)) != expected:
            raise ValueError('Pinned source/input changed: ' + path)


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def mask_pixel_sha256(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all() or np.any((mask != 0) & (mask != 1)):
        raise ValueError('Binary full-canvas mask required')
    values = np.ascontiguousarray(mask, dtype=np.uint8)
    header = json.dumps(dict(shape=list(values.shape), dtype='uint8'), sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + values.tobytes()).hexdigest()


def overlap_metrics(mask, target_mask):
    """Describe overlap only; never turn target negatives into another mask."""
    mask, target = np.asarray(mask, bool), np.asarray(target_mask, float)
    if target.shape != mask.shape or not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError('Same-canvas finite target occupancy required')
    target = target >= .5
    count, target_count = int(mask.sum()), int(target.sum())
    intersection, union = int(np.count_nonzero(mask & target)), int(np.count_nonzero(mask | target))
    return dict(candidate_pixels=count, target_pixels=target_count, intersection_pixels=intersection,
        union_pixels=union, candidate_fraction_in_target=intersection / count if count else None,
        target_fraction_in_candidate=intersection / target_count if target_count else None,
        IoU=intersection / union if union else None, any_positive_target_overlap=intersection > 0,
        overlap_is_not_independent_other_object_or_depth_evidence=True)


def status_from_overlap(overlap):
    if not overlap['candidate_pixels']:
        return 'UNKNOWN', 'empty_selected_mask'
    if not overlap['target_pixels']:
        return 'UNKNOWN', 'target_context_unavailable_empty_target'
    if overlap['any_positive_target_overlap']:
        return 'suspected_target', 'any_positive_target_overlap_diagnostic_only'
    return 'CANDIDATE', 'no_target_overlap_does_not_prove_other_instance'


def infer_case(image_bgr, target_mask, detector, predictor, *, case_id, save_candidates):
    """Injected models permit pure tests. Production callers must hold actual GO.

    All detections and SAM alternatives survive. Only valid, in-canvas,
    confidence-admitted boxes receive SAM box prompts. There is no automatic
    occluder-mask union or handoff to a keypoint/pose/visibility consumer.
    """
    image, target = np.asarray(image_bgr), np.asarray(target_mask, float)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Unmodified uint8 full-frame BGR image required')
    h, w = image.shape[:2]
    if target.shape != (h, w) or not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError('Frozen same-image target occupancy required')
    before_image, before_target = image.copy(), target.copy()
    results = list(detector.predict(source=image.copy(), **deepcopy(CONFIGURATION)))
    if len(results) != 1:
        raise ValueError('One full-frame detector result required')
    result = results[0]
    boxes = np.empty((0, 6)) if result.boxes is None else _numpy(result.boxes.data)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or len(boxes) > CONFIGURATION['max_det']:
        raise ValueError('At most declared20 untracked xyxy/conf/class detections required')
    names = result.names
    if not isinstance(names, dict):
        raise ValueError('Explicit detector class-name map required')
    records, initialized, calls = [], False, 0
    for index, values in enumerate(boxes):
        original_box = np.asarray(values[:4], float)
        confidence, class_value = float(values[4]), float(values[5])
        class_id = int(class_value) if np.isfinite(class_value) and int(class_value) == class_value else None
        record = dict(detection_index=index, instance_id=f'{case_id}/DETECTION_{index:03d}',
            detector_box_xyxy=C.finite(original_box), detector_confidence=C.finite(confidence),
            detector_class_id=class_id, detector_class_name=names.get(class_id),
            status='UNKNOWN', available=False, SAM_forward_calls=0, selected_mask_index=None,
            alternatives=[], other_object_confirmed=False, foreground_occlusion_confirmed=False,
            precise_point_created=False, supplied_to_existing_pipeline=False)
        reasons = []
        if not np.isfinite(original_box).all(): reasons.append('nonfinite_detector_box')
        if not np.isfinite(confidence) or confidence < CONFIGURATION['conf'] or confidence > 1:
            reasons.append('low_or_invalid_detector_confidence')
        if class_id not in names or class_id not in range(80): reasons.append('unknown_COCO_class_id')
        if not reasons:
            box = np.clip(original_box, [0., 0., 0., 0.], [w, h, w, h])
            record.update(SAM_box_xyxy=box.tolist(), prompt_box_canvas_clipped=bool(np.any(box != original_box)))
            if np.any(box[2:] <= box[:2]): reasons.append('empty_or_invalid_prompt_box')
        if reasons:
            record['unavailable_reasons'] = reasons
            records.append(record)
            continue
        if not initialized:
            predictor.set_image(image.copy(), image_format='BGR')
            initialized = True
        masks, quality, _ = predictor.predict(box=box.copy(), multimask_output=True)
        calls += 1
        masks, quality = np.asarray(masks), np.asarray(quality, float)
        if masks.shape != (3, h, w) or quality.shape != (3,):
            raise ValueError('All three original-canvas SAM alternatives and scores required')
        if not np.isfinite(masks).all() or np.any((masks != 0) & (masks != 1)):
            raise ValueError('SAM binary alternatives must not be fabricated or resized')
        masks = np.asarray(masks, np.uint8)
        stored = save_candidates(index, masks.copy(), quality.copy())
        record.update(candidate_file_binding=stored, SAM_forward_calls=1,
            SAM_selection='highest uncalibrated SAM quality; first exact tie',
            SAM_quality_is_semantic_or_depth_probability=False)
        for alternative, (mask, score) in enumerate(zip(masks, quality)):
            record['alternatives'].append(dict(index=alternative, uncalibrated_SAM_quality=C.finite(float(score)),
                mask_pixel_sha256=mask_pixel_sha256(mask), image_hw=[h, w], dtype='uint8',
                target_overlap=overlap_metrics(mask, target)))
        if not np.isfinite(quality).all():
            record['unavailable_reasons'] = ['nonfinite_SAM_quality_all_alternatives_preserved']
        else:
            chosen = int(np.argmax(quality))
            record['selected_mask_index'] = chosen
            overlap = record['alternatives'][chosen]['target_overlap']
            record['status'], record['status_reason'] = status_from_overlap(overlap)
            record['available'] = record['status'] != 'UNKNOWN'
        records.append(record)
    if not np.array_equal(before_image, image) or not np.array_equal(before_target, target):
        raise ValueError('Original image/targetmask changed')
    return dict(schema=SCHEMA, id=case_id, partition='CAL', image_hw=[h, w],
        status='CANDIDATES_AVAILABLE' if any(r['available'] for r in records) else 'UNAVAILABLE',
        unavailable_reason=None if any(r['available'] for r in records) else ('no_detections' if not records else 'all_detections_unavailable'),
        detections=records, detector_forward_calls=1, SAM_image_encoder_calls=int(initialized),
        SAM_prompt_decoder_calls=calls, returned_detections=len(boxes),
        zero_rows_retained=True, target_mask_used_only_for_postprediction_overlap=True,
        target_complement_or_O_circle_or_same_target_alternative_used_as_other_mask=False,
        new_KP_S_U_O_coordinate_prompts=0, masks_supplied_to_current_method=0,
        other_object_or_depth_confirmations=0, pose_or_noise_fits=0)


def extract_case_bindings(text):
    """Decode only flat image/mask metadata, not saved line/reference arrays."""
    image_matches = list(re.finditer(r'"image_binding"\s*:\s*(\{[^{}]*\})', text))
    mask_matches = list(re.finditer(r'"mask_binding"\s*:\s*(\{[^{}]*\})', text))
    if len(image_matches) != 3 or len(mask_matches) != 3:
        raise ValueError('Exactly the previously sealed three metadata bindings required')
    rows = []
    for im, mm in zip(image_matches, mask_matches):
        ids = re.findall(r'"id"\s*:\s*"([^"\n]+)"', text[:im.start()])
        image, mask = json.loads(im.group(1)), json.loads(mm.group(1))
        if not ids or mm.start() < im.end(): raise ValueError('Ordered case/image/mask binding required')
        rows.append(dict(id=ids[-1], partition='CAL', image_binding={k:image[k] for k in ('path','sha256')},
            target_mask_binding={k:mask[k] for k in ('path','sha256')}))
    if tuple(r['id'] for r in rows) != CASES:
        raise ValueError('Same immutable fixed CAL3 IDs/order required')
    return rows


def _sources():
    paths = [Path(__file__), Path(__file__).with_name('test_capture_generic_instance_mask_candidates_v1.py'),
        Path(C.__file__), VENDOR/'mobile_sam/predictor.py', VENDOR/'mobile_sam/build_sam.py',
        VENDOR/'mobile_sam/modeling/sam.py', VENDOR/'mobile_sam/modeling/mask_decoder.py',
        ULTRA/'models/yolo/model.py', ULTRA/'models/yolo/detect/predict.py',
        ULTRA/'engine/model.py', ULTRA/'engine/results.py', ULTRA/'nn/tasks.py',
        ULTRA/'cfg/default.yaml', ULTRA/'cfg/datasets/coco.yaml',
        ULTRA/'utils/__init__.py', ULTRA/'utils/checks.py', ULTRA/'utils/downloads.py',
        ULTRA.parent/'ultralytics-8.4.60.dist-info/METADATA']
    manifest = C.read(C.DOC/'MOBILESAM_ASSET_MANIFEST.json')
    for name, expected in manifest['runtime_files_sha256'].items():
        if C.sha(VENDOR/name) != expected:
            raise ValueError('Pinned MobileSAM vendor/weight changed')
        if name.endswith('.py'): paths.append(VENDOR/name)
    return {_binding(p)['path']:C.sha(p) for p in paths}


def prepare(*, root_metadata_GO=False):
    if root_metadata_GO is not True: raise PermissionError('Metadata GO required before IO')
    if PLAN.exists(): raise FileExistsError('Preserve prior declaration')
    parent_seal = C.read(PARENT/'RENDER_SEAL.json')
    process = PARENT/'PROCESS_DATA.json'
    if C.sha(process) != parent_seal['assets']['PROCESS_DATA.json']:
        raise ValueError('Original fixed CAL3 metadata bytes changed')
    cases = extract_case_bindings(process.read_text())
    asset_seal = C.read(ASSETS/'FEASIBILITY_SEAL_V1.json')
    if C.sha(ASSETS/'STATIC_CHECKPOINT_METADATA.json') != asset_seal['assets_sha256']['STATIC_CHECKPOINT_METADATA.json']:
        raise ValueError('Static detector metadata evidence changed')
    static = C.read(ASSETS/'STATIC_CHECKPOINT_METADATA.json')['static_checkpoint_metadata']
    names = {key:value for key,value in re.findall(r'^  (\d+): ([^\n]+)$',
        (ULTRA/'cfg/datasets/coco.yaml').read_text(),re.M)}
    if (static[1]['path'] != str(DETECTOR.relative_to(C.ROOT)) or not static[1]['names_equal_local_COCO80']
            or static[1]['sha256'] != DETECTOR_SHA or static[1]['task'] != 'detect'
            or static[1]['nc'] != 80 or static[1]['names'] != names or len(names) != 80):
        raise ValueError('Existing generic COCO80 detector metadata required')
    if C.sha(DETECTOR) != DETECTOR_SHA or C.sha(SAM) != SAM_SHA:
        raise ValueError('Existing explicit local checkpoints required; no automatic download')
    sources = _sources()
    paths = [DETECTOR, SAM, process, PARENT/'RENDER_SEAL.json',
        ASSETS/'STATIC_CHECKPOINT_METADATA.json', ASSETS/'FEASIBILITY_SEAL_V1.json',
        C.DOC/'MOBILESAM_ASSET_MANIFEST.json']
    inputs = {_binding(p)['path']:C.sha(p) for p in paths}
    for case in cases:
        for key in ('image_binding','target_mask_binding'):
            binding = case[key]; path = _resolve(binding['path'])
            if C.sha(path) != binding['sha256']: raise ValueError('Frozen original image/mask bytes changed')
            inputs[binding['path']] = binding['sha256']
    protected = C.read(B/'occluder_mask_supply_source_audit_v1/SOURCE_INPUT_BINDINGS.json')['protected_user_sha256']
    for mapping in (sources, inputs, protected): _verify(mapping)
    PLAN.parent.mkdir(parents=True)
    value = dict(schema='STANDALONE_GENERIC_INSTANCE_CANDIDATES_FIXED_CAL3_PREDECLARATION_V1',
        UTC=datetime.now(timezone.utc).isoformat(),cases=cases,configuration=deepcopy(CONFIGURATION),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        detector=_binding(DETECTOR),SAM_checkpoint=_binding(SAM),SAM_vendor=str(VENDOR.relative_to(C.ROOT)),
        installed_generic_API='ultralytics8.4.60 source pinned; actual load compatibility not yet tested',
        COCO_names=static[1]['names'],fixed_case_selection='same previously declared CAL3 order, not new success/instance selection',
        records='all returned detector boxes/class/conf; all three SAM alternatives/quality/file+pixel SHA; first max-quality display selection',
        semantics='CANDIDATE or UNKNOWN or suspected_target only; no other-instance, O or foreground-depth confirmation',
        target_overlap='postprediction binary mask occupancy overlap at existing.5; any positive overlap flags suspected_target, not hard truth',
        mask_source='new full-frame generic detector boxes only; no target complement, O circle, point/native-keypoint prompts or same-target candidate substitution',
        inference_budget=dict(workers=1,rows_in_memory=1,SAM_encoder_per_frame_at_most=1,
            decoder='sequential one box/all three alternatives; max20 detector boxes',BLAS_OMP_MKL_threads=1),
        preparation_image_mask_array_model_label_or_primary_decodes=0,
        current31_32_sources_models_observations_unchanged=True,existing_method_consumer_connection=False,
        actual_root_GO_separate_required=True,default_adoption=False,new_pose_noise_fits=0)
    C.write(PLAN,value)
    for relative,digest in sources.items():
        destination=PLAN.parent/'code_snapshot'/relative.lstrip('/')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(_resolve(relative),destination)
        if C.sha(destination)!=digest:raise ValueError('Source snapshot changed')
    for mapping in (sources,inputs,protected):_verify(mapping)
    C.write(PLAN.parent/'PLAN_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=sources,
        image_mask_array_model_label_primary_decodes=0))
    return dict(plan=str(PLAN.relative_to(C.ROOT)),sha256=C.sha(PLAN),cases=3,sources=len(sources),inputs=len(inputs))


def _load_models(plan):
    """Only called after actual GO, source/plan/input checks and output creation."""
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_OFFLINE']='true'
    os.environ['YOLO_CONFIG_DIR']=str(OUTPUT/'ultralytics_runtime_config')
    import torch
    torch.set_num_threads(1)
    from ultralytics import YOLO
    import ultralytics
    if (not Path(ultralytics.__file__).resolve().is_relative_to(ULTRA)
            or ultralytics.__version__!='8.4.60'):
        raise ValueError('Pinned installed generic detector API required')
    detector=YOLO(str(_resolve(plan['detector']['path'])),task='detect')
    if detector.task!='detect' or {str(k):v for k,v in detector.names.items()}!=plan['COCO_names']:
        raise ValueError('Generic COCO80 runtime checkpoint must match static metadata')
    sys.path.insert(0,str(VENDOR))
    import mobile_sam
    if not Path(mobile_sam.__file__).resolve().is_relative_to(VENDOR):
        raise ValueError('Pinned MobileSAM vendor required')
    model=mobile_sam.sam_model_registry['vit_t']()
    model.load_state_dict(torch.load(_resolve(plan['SAM_checkpoint']['path']),map_location='cpu',weights_only=True))
    predictor=mobile_sam.SamPredictor(model.to('cpu').eval().requires_grad_(False))
    return detector,predictor


def _render_case(image, row, output, number):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,6));rgb=image[...,::-1]
    for ax in axes:ax.imshow(rgb);ax.set_axis_off()
    axes[0].set_title('Original RGB; unchanged brightness')
    axes[1].set_title('All generic detector boxes/classes/scores')
    axes[2].set_title('Selected SAM candidates; identity/depth UNKNOWN')
    for i,r in enumerate(row['detections']):
        color=plt.get_cmap('tab20')(i%20)
        box=r['detector_box_xyxy']
        if all(v is not None for v in box):
            x,y,xx,yy=box;axes[1].add_patch(plt.Rectangle((x,y),xx-x,yy-y,fill=False,color=color,lw=1.2))
            confidence='unknown'if r['detector_confidence']is None else f"{r['detector_confidence']:.3f}"
            axes[1].text(x,y,f"{i}: {r['detector_class_name']} {confidence}",fontsize=7,color='black',bbox=dict(facecolor=color,alpha=.8,pad=1))
        if r['selected_mask_index'] is not None:
            with np.load(_resolve(r['candidate_file_binding']['path']),allow_pickle=False)as data:
                mask=data['sam_candidate_masks'][r['selected_mask_index']]
            rgba=np.zeros((*mask.shape,4));rgba[...,:3]=color[:3];rgba[...,3]=mask*.28
            axes[2].imshow(rgba)
            axes[2].text(.01,.98-.045*i,f"{i}: {r['detector_class_name']} / {r['status']}",transform=axes[2].transAxes,va='top',fontsize=7,color='black',bbox=dict(facecolor=color,alpha=.8,pad=1))
    fig.suptitle(row['id']+' | standalone candidates, not supplied to current pose pipeline',fontsize=11)
    fig.tight_layout();path=output/f'case{number}_candidates.png';fig.savefig(path,dpi=150);plt.close(fig)
    return _binding(path)


def run(*,expected_plan_sha256,root_actual_inference_GO=False):
    if root_actual_inference_GO is not True:raise PermissionError('Separate root actual inference GO before IO')
    if not isinstance(expected_plan_sha256,str)or len(expected_plan_sha256)!=64:raise ValueError('Explicit frozen plan SHA required')
    if C.sha(PLAN)!=expected_plan_sha256 or C.read(PLAN.parent/'PLAN_SEAL.json')['plan_sha256']!=expected_plan_sha256:
        raise ValueError('Exact standalone source/fixture plan required')
    plan=C.read(PLAN)
    if plan['configuration']!=CONFIGURATION or tuple(r['id']for r in plan['cases'])!=CASES:raise ValueError('Exact declared fixed CAL3/configuration required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(plan[key])
    if OUTPUT.exists():raise FileExistsError('Preserve every completed/failed attempt')
    OUTPUT.mkdir(parents=True);(OUTPUT/'masks').mkdir();started=time.monotonic()
    C.write(OUTPUT/'EXECUTION_START.json',dict(plan_sha256=expected_plan_sha256,UTC=datetime.now(timezone.utc).isoformat(),rows_completed=0))
    try:
        detector,predictor=_load_models(plan)
        import cv2
        cv2.setNumThreads(1)
        rows=[]
        for number,case in enumerate(plan['cases']):
            image=cv2.imread(str(_resolve(case['image_binding']['path'])))
            if image is None:raise ValueError('Bound original RGB decode failed')
            with np.load(_resolve(case['target_mask_binding']['path']),allow_pickle=False)as data:target=data['pallet_mask_clean'].copy()
            def save(index,masks,quality):
                path=OUTPUT/'masks'/f'case{number}_instance{index:03d}.npz'
                if path.exists():raise FileExistsError('No mask overwrite')
                np.savez_compressed(path,sam_candidate_masks=masks,sam_candidate_quality=quality)
                return _binding(path)
            row=infer_case(image,target,detector,predictor,case_id=case['id'],save_candidates=save)
            row.update(image_binding=case['image_binding'],target_mask_binding=case['target_mask_binding'])
            row['figure_binding']=_render_case(image,row,OUTPUT,number)
            C.write(OUTPUT/f'case{number}.json',row);rows.append(row)
            print(json.dumps(dict(completed=len(rows),id=row['id'],detections=len(row['detections']))),flush=True)
            del image,target
        C.write(OUTPUT/'RESULTS.json',dict(schema=SCHEMA,cases=rows,plan_sha256=expected_plan_sha256))
        text='# 고정 CAL3 자동 instance 후보\n\nGeneric COCO80 detector와 별도 box-prompt MobileSAM 후보입니다. other object/앞가림/물리 O 확정이 아니며 기존 pose pipeline에 공급하지 않았습니다. target 겹침은 진단일 뿐입니다. 모든 box·SAM3대안·0/unknown 행은 RESULTS에 보존했습니다. GT·KP/S/U/O point prompt·포즈/잡음 적합은 없습니다.\n\n'
        for i,row in enumerate(rows):text+=f"## {row['id']}\n\n{row['returned_detections']} detections; {row['status']}.\n\n![actual candidates](case{i}_candidates.png)\n\n"
        (OUTPUT/'REPORT_KO.md').write_text(text)
        for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(plan[key])
        C.write(OUTPUT/'EXECUTION_SEAL.json',dict(schema=SCHEMA,complete=True,cases=3,ordered_ids=list(CASES),
            plan_sha256=expected_plan_sha256,source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],
            protected_user_sha256=plan['protected_user_sha256'],elapsed_seconds=time.monotonic()-started,
            assets={str(p.relative_to(OUTPUT)):C.sha(p)for p in OUTPUT.rglob('*')if p.is_file()},
            masks_supplied_to_current_pipeline=0,other_object_or_depth_confirmations=0,GT_or_primary_reads=0,pose_noise_fits=0))
    except BaseException as exc:
        C.write(OUTPUT/'FAILURE_RECEIPT.json',dict(plan_sha256=expected_plan_sha256,error_type=type(exc).__name__,error=str(exc),complete=False,attempt_preserved=True))
        raise
    return dict(output=str(OUTPUT.relative_to(C.ROOT)),seal_sha256=C.sha(OUTPUT/'EXECUTION_SEAL.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--root-metadata-GO',action='store_true');parser.add_argument('--root-actual-inference-GO',action='store_true')
    parser.add_argument('--expected-plan-sha256');args=parser.parse_args()
    result=prepare(root_metadata_GO=args.root_metadata_GO)if args.stage=='prepare'else run(expected_plan_sha256=args.expected_plan_sha256,root_actual_inference_GO=args.root_actual_inference_GO)
    print(json.dumps(result,ensure_ascii=False),flush=True)
