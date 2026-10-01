"""Read-only full-candidate source audit; no real references, forward or fit.

Overlap categories are descriptive diagnostics, never semantic negative labels.
All original 5120 source frames remain included, independent of pose eligibility.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1'
OLD = ROOT / '_docs/experiments/pallet_pose_union_selection_20261001_v1'
RAW = ROOT / 'data/pallet/results/pallet_pose_union_selection_20261001_v1'
READS = set()


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        denied = ('/data/evaluation/', 'GEOMETRY_RESOLVED_POSE_GT', 'TRUTH_FOR_DISPLAY',
                  'AXIS_REVIEW_MANIFEST', '/real_gt_v2/annotations/', '/negative_real_',
                  '/E1_POSE_METRICS', '/pallet_pose_stable_improvement_20261001_v1/POSE_METRICS')
        assert not any(x in path for x in denied), ('REAL_REFERENCE_DENIED', path)
        if path.startswith(str(ROOT)):
            READS.add(path)


sys.addaudithook(guard)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def verify(binding):
    assert sha(ROOT / binding['path']) == binding['sha256'], binding['path']


def box_relation(box, target):
    box, target = np.asarray(box, float), np.asarray(target, float)
    assert box.shape == target.shape == (4,) and np.isfinite(box).all()
    area = float(np.prod(np.maximum(box[2:] - box[:2], 0.)))
    gtarea = float(np.prod(target[2:] - target[:2]))
    assert gtarea > 0
    intersection = float(np.prod(np.maximum(np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2]), 0.)))
    iou = intersection / (area + gtarea - intersection) if area else 0.
    containment = intersection / area if area else 0.
    ratio = area / gtarea
    if iou >= .5:
        kind = 'MATCH_IOU_GE_0_5'
    elif intersection == 0:
        kind = 'TARGET_BOX_DISJOINT_NOT_SEMANTIC_NEGATIVE'
    elif containment >= .8 and ratio <= .5:
        kind = 'CONTAINED_SMALL_PARTLIKE_PROXY_NOT_SEMANTIC_LABEL'
    else:
        kind = 'OTHER_LOW_IOU_OVERLAP'
    return dict(iou=iou, intersection_over_candidate=containment, candidate_over_target_area=ratio,
                category=kind, positive_box_area=area > 0)


def aggregate(frames, candidates):
    summary = dict(frames=len(frames), candidates=len(candidates),
                   candidate_count_histogram=dict(Counter(str(r['candidate_count']) for r in frames)),
                   categories=dict(Counter(r['category'] for r in candidates)),
                   selected_categories=dict(Counter(r['category'] for r in candidates if r['selected'])),
                   nonselected_categories=dict(Counter(r['category'] for r in candidates if not r['selected'])))
    for threshold in (.5, .75):
        key = str(threshold)
        summary['IoU_' + key] = dict(
            selected_match=sum(r['selected_iou'] >= threshold for r in frames),
            any_candidate_match=sum(r['best_iou'] >= threshold for r in frames),
            selected_miss_with_matching_alternative=sum(r['selected_iou'] < threshold <= r['best_iou'] for r in frames),
            no_matching_candidate=sum(r['best_iou'] < threshold for r in frames))
    summary['no_detection'] = sum(r['candidate_count'] == 0 for r in frames)
    summary['frames_with_match_and_low_iou_competitor'] = sum(r['matched_and_low_iou'] for r in frames)
    summary['frames_with_match_and_low_iou_competitor_score_ge_0_1'] = sum(r['matched_and_low_iou_high_score'] for r in frames)
    summary['selected_score_is_argmax'] = all(r['top1_is_argmax'] for r in frames)
    summary['candidate_positive_area'] = sum(r['positive_box_area'] for r in candidates)
    summary['by_category_pose_validity'] = {}
    for category in sorted({r['category'] for r in candidates}):
        selected = [r for r in candidates if r['category'] == category]
        summary['by_category_pose_validity'][category] = dict(
            candidates=len(selected), any_available_pose=sum(r['any_available_pose'] for r in selected),
            any_pose_all_corners_positive_depth=sum(r['any_full_cheirality_pose'] for r in selected),
            score_ge_0_1=sum(r['score'] >= .1 for r in selected),
            score_ge_0_4=sum(r['score'] >= .4 for r in selected),
            center_in_reflect_border=sum(r['center_in_reflect_border'] for r in selected),
            majority_box_in_reflect_border=sum(r['native_canvas_area_fraction'] < .5 for r in selected),
            minimum_reprojection_median_px=float(np.median([r['minimum_reprojection_px'] for r in selected if r['minimum_reprojection_px'] is not None])) if any(r['minimum_reprojection_px'] is not None for r in selected) else None)
    return summary


def negative_inventory():
    base = ROOT / 'data/pallet/training_data/paper_release/negative/extracted/negative_synth_v1_train'
    records = [json.loads(line) for line in (base / 'records.jsonl').read_text().splitlines()]
    assert len(records) == len({r['sample_id'] for r in records}) == 9000
    counts = Counter()
    bad = []
    labels_digest = hashlib.sha256()
    for row in records:
        path = base / 'labels' / (row['file_id'] + '_label.json')
        data = path.read_bytes()
        labels_digest.update(row['file_id'].encode() + hashlib.sha256(data).digest())
        label = json.loads(data)
        okay = (label.get('object_present') is False and label.get('pose_valid') is False
                and label.get('keypoints') == [] and label.get('structural_lines') == [] and label.get('objects') == [])
        if not okay:
            bad.append(row['file_id'])
        assert label['release']['sample_id'] == row['sample_id']
        counts[(row['negative_type'], row['origin'], row['metadata_provenance'])] += 1
    return dict(frames=9000, label_contract_bad_ids=bad, label_contract_ok=not bad,
                label_ordered_digest=labels_digest.hexdigest(),
                metadata_counts=[dict(type=k[0], origin=k[1], provenance=k[2], count=v) for k, v in sorted(counts.items())],
                RGB_files_present=len(list((base / 'rgb').glob('*_rgb.png'))),
                records_binding=bind(base / 'records.jsonl'), index_binding=bind(base / 'index.csv'),
                README_binding=bind(base / 'README.txt'),
                limitation='Semantic-absence labels, not within-positive-image part labels. README explicitly states legacy N0 background pallet exclusion is UNVERIFIED; a label check is not independent visual semantic verification.',
                current_R0_full_candidate_cache_authenticated=False,
                current_R0_new_forwards=0)


def main():
    start = time.monotonic()
    receipt_path = OLD / 'R0_SOURCE_COMPLETE.json'
    receipt = read(receipt_path)
    assert receipt['complete'] and receipt['frames'] == 5120
    assert receipt['checkpoint']['sha256'] == '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
    source_lock = read(OLD / 'SOURCE_PREDICTIONS_LOCK.json')
    verify(source_lock['receipts']['R0'])
    verify(source_lock['metadata'])
    verify(source_lock['protocol'])
    verify(source_lock['runtime_amendment'])
    contract = read(OLD / 'SOURCE_CONTRACT.json')
    manifest_path = ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
    manifest_binding = next(b for b in contract['bindings'] if b['path'] == str(manifest_path.relative_to(ROOT)))
    verify(manifest_binding)
    source = {r['id']: r for r in read(manifest_path)['records']}
    inputs = read(RAW / 'SOURCE_INPUTS.json')
    assert len(inputs) == len(receipt['files']) == 5120
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    O.D.cv2.setNumThreads(1)
    frames, candidates = [], []
    label_checks = Counter()
    bounds_max = 0.
    for j, row in enumerate(inputs):
        fid = row['id']
        src = source[fid]
        assert src['source_kind'] == 'synthetic' and len(src['targets']) == 1
        assert src['image_sha256'] == row['image']['sha256'] and src['prepared_shape_hw'] == row['hw']
        content = Path(src['label']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == src['label_sha256']
        parsed = np.fromstring(content.decode(), sep=' ')
        target = src['targets'][0]
        assert parsed.shape == (32,) and parsed[0] == target['class_id'] == 0
        np.testing.assert_array_equal(parsed[1:5], target['box_xywh_normalized'])
        np.testing.assert_array_equal(parsed[5:].reshape(9, 3), target['keypoints_normalized'])
        h, w = row['hw']
        xywh = parsed[1:5] * [w, h, w, h]
        box = np.r_[xywh[:2] - xywh[2:] / 2, xywh[:2] + xywh[2:] / 2]
        keypoints = parsed[5:].reshape(9, 3)
        visible = keypoints[:, 2] == 2
        points = keypoints[visible, :2] * [w, h]
        envelope = np.r_[points.min(0), points.max(0)]
        bounds_max = max(bounds_max, float(np.max(abs(box - envelope))))
        label_checks['single_class0_positive_labels_verified'] += 1
        label_checks['all9_projection_points_inside_prepared_canvas'] += int(visible.sum() == 9)
        label_checks['target_box_extends_reflect_border'] += int(np.any(box[:2] < 100) or box[2] > w-100 or box[3] > h-100)
        verify(receipt['files'][j])
        saved = read(ROOT / receipt['files'][j]['path'])
        assert saved['id'] == fid and saved['checkpoint_sha'] == receipt['checkpoint']['sha256']
        pred = saved['prediction']
        cc = []
        for index, candidate in enumerate(pred['candidates']):
            assert index == candidate['candidate_index'] and candidate['score'] > .001
            relation = box_relation(candidate['box_xyxy'], box)
            native = np.array([100., 100., w-100., h-100.])
            cb = np.asarray(candidate['box_xyxy'], float)
            native_relation = box_relation(cb, native)
            center = (cb[:2] + cb[2:]) / 2
            # Only predicted points plus existing K/dimensions enter this solve.
            one = dict(candidates=[dict(candidate, candidate_index=0)], selected_index=0)
            poses = O.candidate_record(one, dict(K=row['K'], xyz=row['dims']))['hypotheses']
            available = [p for p in poses if p['pose']['available']]
            observation = dict(id=fid, split=row['split'], source=src['source'], candidate_index=index,
                selected=index == pred['selected_index'], score=candidate['score'], box_xyxy=candidate['box_xyxy'],
                **relation, any_available_pose=bool(available),
                native_canvas_area_fraction=native_relation['intersection_over_candidate'],
                center_in_reflect_border=bool(np.any(center < native[:2]) or np.any(center > native[2:])),
                any_full_cheirality_pose=any(p['inference_cues']['corner8_positive_depth_fraction'] == 1. for p in available),
                minimum_reprojection_px=min(p['pose']['reprojection_px'] for p in available) if available else None,
                semantic_negative_verified=False)
            cc.append(observation)
            candidates.append(observation)
        selected = pred['selected_index']
        assert selected is None if not cc else selected == int(np.argmax([r['score'] for r in cc]))
        matched = any(c['iou'] >= .5 for c in cc)
        frames.append(dict(id=fid, split=row['split'], source=src['source'], image_sha256=row['image']['sha256'],
            candidate_count=len(cc), selected_index=selected, selected_iou=cc[selected]['iou'] if cc else 0.,
            best_iou=max((c['iou'] for c in cc), default=0.), target_box_xyxy=box.tolist(),
            top1_is_argmax=True, matched_and_low_iou=matched and any(c['iou'] < .5 for c in cc),
            matched_and_low_iou_high_score=matched and any(c['iou'] < .5 and c['score'] >= .1 for c in cc)))
        if (j + 1) % 1024 == 0:
            print('DETECTION_SOURCE_AUDIT', j + 1, len(candidates), flush=True)
    assert len(frames) == 5120 and len(candidates) == 7173
    negative = negative_inventory()
    result = dict(complete=True, created_at=datetime.now(timezone.utc).isoformat(),
        scope='SOURCE5120 full saved candidate pool; CPU diagnostic only; all C1 and improper-pose rows retained.',
        categories='IoU>=.5 matched; IoU<.5 intersection/candidate-area>=.8 and candidate/target-area<=.5 contained-small proxy; exactly zero intersection target-box-disjoint; remaining low-IoU overlap. These are not semantic classes or training eligibility.',
        threshold_selection='Fixed descriptive bins from prior matching convention; no classifier or threshold search.',
        all=aggregate(frames, candidates),
        by_split={s: aggregate([r for r in frames if r['split'] == s], [r for r in candidates if r['split'] == s]) for s in ('TRAIN', 'VAL')},
        by_source={s: aggregate([r for r in frames if r['source'] == s], [r for r in candidates if r['source'] == s]) for s in ('G38', 'P0', 'TEX')},
        source_label_check=dict(label_checks, bbox_vs_known_projected9_envelope_max_px=bounds_max,
            bbox_definition='Envelope of v=2 projected corner8+center points inside prepared canvas, not visible object mask or exhaustive scene annotation.',
            source_manifest=manifest_binding, labels_rechecked=5120),
        current_pool=dict(checkpoint=receipt['checkpoint'], score_floor_strict=.001, max_det=300,
            end2end_one2one=True, IoU_NMS_performed=False, dense_pre_threshold_anchor_cache=False,
            pixel_predictions_unchanged=True),
        physical_validity_caveat='A PnP solution with finite positive-depth corners only proves geometric fit to the supplied pallet dimensions; it does not prove that the RGB region is a pallet.',
        semantic_negative_caveat='Low IoU may be a duplicate, localization/truncation, partial target, another real pallet, or nonpallet. Single target annotation does not exhaust scene semantics.',
        negative9k=negative,
        mismatched_selected_frames=[r for r in frames if r['selected_iou'] < .5],
        all_frame_diagnostics=frames,
        nonmatched_candidates=[r for r in candidates if r['iou'] < .5],
        selected_smoke_for_visual_review=contract['smoke_selection'],
        bindings=[bind(p) for p in (Path(__file__), receipt_path, OLD/'SOURCE_PREDICTIONS_LOCK.json',
            OLD/'SOURCE_CONTRACT.json', RAW/'SOURCE_INPUTS.json',
            ROOT/'scripts/research/pallet_line_pose_v1/source_data.py',
            ROOT/'scripts/research/pallet_line_pose_v1/features.py',
            ROOT/'challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py',
            Path(O.__file__), Path(O.D.__file__), Path(O.D.Pose.__file__))],
        neural_forwards=0, fits=0, backward_calls=0, real_reference_reads=0,
        source_candidate_PnP_calls=len(candidates),
        audit_read_paths=dict(unique_count=len(READS),
            sorted_newline_utf8_sha256=hashlib.sha256('\n'.join(sorted(READS)).encode()).hexdigest(),
            real_reference_guard_active=True),
        wall_seconds=time.monotonic()-start)
    DOC.mkdir(parents=True, exist_ok=True)
    output = DOC/'DETECTION_SOURCE_FEASIBILITY.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(all=result['all'], by_split=result['by_split'], source_label_check=result['source_label_check'], negative9k=negative), indent=2), flush=True)
    print('DETECTION_SOURCE_COMPLETE', bind(output), flush=True)


if __name__ == '__main__':
    main()
