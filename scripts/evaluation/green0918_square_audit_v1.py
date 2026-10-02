"""Frozen 2D audit of the separate 0918 square-pallet annotation set.

This evaluator deliberately keeps the 0918 set separate from GREEN150.  It
uses the already frozen R0, N0 and N2 models, scores direct manual clicks as
the primary observation, and does not create a 3D-ground-truth claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.evaluation import final_dimension_release as F
from scripts.evaluation import green_saved_labels_v1 as G


ROOT = F.ROOT
NAME = 'pallet_green0918_dimension_audit_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
SOURCE = ROOT / 'outputs/annotations/0918_dataset_square'
GREEN150 = F.DOC / 'green150_saved_labels_v1/DATASET_SNAPSHOT.json'
SOURCE_MANIFEST = ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
DATASET = DOC / 'DATASET_SNAPSHOT.json'

PROTOCOL = dict(
    status='REUSED_DEVELOPMENT_2D_AUDIT_NOT_INDEPENDENT_TEST',
    purpose='Check the recent separate 0918 green square annotations with frozen R0/N0/N2.',
    dimensions='Externally supplied fixed canonical W,D,H = 1.1,1.1,0.15 metres.',
    primary='All declared-visible manual_click points; an in-image-only sensitivity is also retained.',
    matching='Selected detector candidate; IoU>=0.5 against all-known in-image annotation-point box.',
    symmetry='Approved whole-object C4 permutations; center remains fixed.',
    inference_inputs='RGB plus registry dimensions for N2; no annotation, GT pose, selected phase or camera-facing W/D swap.',
    no_3D_claim=True,
    no_training=True,
    no_model_selection=True,
    independent_confirmation=False,
)


def decoded_sha(image):
    """Hash decoded BGR pixels as well as encoded bytes for overlap auditing."""
    import cv2
    value = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f'Unreadable image: {image}')
    return hashlib.sha256(value.tobytes()).hexdigest(), list(value.shape[:2])


def snapshot():
    import numpy as np

    F.checked_lock()
    if DATASET.exists():
        load_dataset()
        print('Existing immutable 0918 snapshot verified', flush=True)
        return
    labels = sorted(SOURCE.glob('*.json'))
    if len(labels) != 119:
        raise ValueError(f'Expected 119 annotations, got {len(labels)}')
    green = F.read(GREEN150)
    green_raw = {row['image']['sha256'] for row in green['records']}
    green_decoded = set()
    for row in green['records']:
        digest, _ = decoded_sha(ROOT / row['image']['path'])
        green_decoded.add(digest)

    rows = []
    manual_distribution = Counter()
    source_counts = Counter()
    split_counts = Counter()
    population_counts = Counter()
    outside = []
    reprojection = []
    for label in labels:
        image = label.with_suffix('.png')
        overlay = SOURCE / '_overlays' / image.name
        if not image.is_file() or not overlay.is_file():
            raise ValueError(f'Missing image/overlay for {label.stem}')
        doc = F.read(label)
        if doc.get('schema_version') != 'real_pallet_gt_v2' or len(doc.get('objects', [])) != 1:
            raise ValueError(f'Unexpected annotation schema: {label}')
        obj = doc['objects'][0]
        object_type = obj.get('object_type', doc.get('object_type'))
        if object_type != 'plastic_standard_110x110x15':
            raise ValueError(f'Unexpected object type: {label}')
        dims = obj.get('physical_dimensions_m', {})
        if [dims.get('x'), dims.get('z'), dims.get('y')] != [1.1, 1.1, .15]:
            raise ValueError(f'Unexpected canonical dimensions: {label}')
        entries = obj.get('keypoint_annotations', [])
        if len(entries) != 9:
            raise ValueError(f'Expected nine annotations: {label}')
        raw_sha = F.binding(image)['sha256']
        pixel_sha, hw = decoded_sha(image)
        if hw != [doc['camera_data']['height'], doc['camera_data']['width']]:
            raise ValueError(f'Image/camera shape mismatch: {label}')
        gt, known = G.annotation_arrays(doc)
        _, manual = G.annotation_arrays(doc, True)
        h, w = hw
        inside = np.isfinite(gt).all(1) & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
        manual_distribution[int(manual[:8].sum())] += 1
        for keypoint, entry in enumerate(entries):
            source_counts[entry.get('source', 'unknown')] += 1
            if known[keypoint] and not inside[keypoint]:
                outside.append(dict(id=label.stem, keypoint=keypoint,
                    source=entry.get('source', 'unknown'), xy=entry.get('xy'),
                    visibility=entry.get('visibility'), truncation=obj.get('truncation')))
        split_counts[obj.get('split', 'missing')] += 1
        population_counts[doc.get('population_role', 'missing')] += 1
        reprojection.append(float(obj['reproj_error_px']))
        rows.append(dict(
            id=label.stem,
            session=doc.get('capture_session_id'),
            image=F.binding(image),
            annotation=F.binding(label),
            overlay=F.binding(overlay),
            image_decoded_sha256=pixel_sha,
            original_hw=hw,
            canonical_WDH_m=[1.1, 1.1, .15],
            manual_corners=int(manual[:8].sum()),
            known_corners=int(known[:8].sum()),
            manual_in_frame_corners=int((manual[:8] & inside[:8]).sum()),
            split=obj.get('split'),
            population_role=doc.get('population_role'),
            pose_status=obj.get('pose_status'),
            migration_status=obj.get('migration_status'),
            canonical_pose_present=obj.get('canonical_pose') is not None,
            reproj_error_px=float(obj['reproj_error_px']),
        ))
    if len({r['image']['sha256'] for r in rows}) != 119 or len({r['image_decoded_sha256'] for r in rows}) != 119:
        raise ValueError('Duplicate 0918 image content')
    raw_overlap = sorted({r['image']['sha256'] for r in rows} & green_raw)
    decoded_overlap = sorted({r['image_decoded_sha256'] for r in rows} & green_decoded)
    if raw_overlap or decoded_overlap:
        raise ValueError('0918 must remain separate from GREEN150')
    source = F.read(SOURCE_MANIFEST)
    source_training_sha = {row['image_sha256'] for row in source['records'] if row['partition'] == 'train'}
    training_overlap = sorted({row['image']['sha256'] for row in rows} & source_training_sha)
    if training_overlap:
        raise ValueError('Unexpected overlap with frozen N0/N2 synthetic TRAIN images')
    reprojection = np.asarray(reprojection, np.float64)
    counts = dict(
        frames=len(rows),
        source_counts=dict(sorted(source_counts.items())),
        manual_corner_distribution={str(k): v for k, v in sorted(manual_distribution.items())},
        manual_corners=sum(r['manual_corners'] for r in rows),
        manual_in_frame_corners=sum(r['manual_in_frame_corners'] for r in rows),
        split_counts=dict(split_counts),
        population_role_counts=dict(population_counts),
        canonical_pose_missing=sum(not r['canonical_pose_present'] for r in rows),
        unconfirmed_signed_axis=sum(r['pose_status'] == 'UNCONFIRMED_SIGNED_AXIS' for r in rows),
        manual_review_required=sum(r['migration_status'] == 'MANUAL_REVIEW_REQUIRED' for r in rows),
        outside_known_points=len(outside),
        outside_manual_points=sum(r['source'] == 'manual_click' for r in outside),
        reprojection_px=dict(minimum=float(reprojection.min()), median=float(np.median(reprojection)),
                             p90=float(np.quantile(reprojection, .9)), maximum=float(reprojection.max())),
    )
    F.freeze(DATASET, dict(protocol=PROTOCOL, records=rows, counts=counts,
        outside_points=outside,
        overlap_with_GREEN150=dict(encoded_sha256=len(raw_overlap), decoded_pixel_sha256=len(decoded_overlap)),
        overlap_with_frozen_model_synthetic_TRAIN=dict(encoded_sha256=len(training_overlap),
            training_rows=sum(row['partition'] == 'train' for row in source['records'])),
        source_directory=str(SOURCE.relative_to(ROOT)),
        GREEN150=F.binding(GREEN150), source_manifest=F.binding(SOURCE_MANIFEST),
        model_lock=F.binding(F.DOC / 'MODEL_LOCK.json'),
        evaluator=F.binding(Path(__file__))))
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


def load_dataset():
    data = F.read(DATASET)
    for key in ('GREEN150', 'source_manifest', 'model_lock', 'evaluator'):
        F.verify(data[key])
    for row in data['records']:
        for key in ('image', 'annotation', 'overlay'):
            F.verify(row[key])
    return data


def serial(captured):
    return dict(candidates=[{k: v.tolist() if hasattr(v, 'tolist') else v for k, v in candidate.items()}
                            for candidate in captured['candidates']],
                selected_index=captured['selected_index'])


def infer():
    import cv2
    import torch

    data = load_dataset()
    destination = RAW / 'PREDICTIONS.json'
    if destination.exists():
        payload = F.read(destination)
        F.verify(payload['dataset']); F.verify(payload['model'])
        print('Existing immutable 0918 predictions verified', flush=True)
        return
    E = F.setup()
    from inference import load_head, predict_captured, registry_input

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA access required; no CPU fallback')
    dimensions, order = registry_input('plastic_standard_110x110x15')
    if dimensions.tolist() != [1.1, 1.1, .15] or order != 4:
        raise ValueError('Registry square-pallet contract changed')
    lock = F.checked_lock()
    heads = {(row['arm'], row['seed']): load_head(row['arm'], row['seed'])[0] for row in lock['heads']}
    extractor = E.old('features').FrozenYoloFeatures(E.R0)
    normalization = F.read(F.DCP / 'DIM_NORMALIZATION_LOCK.json')
    output = {name: [] for name in ['R0', *(f'{arm}_seed{seed}' for arm, seed in heads)]}
    start_gpu = E.gpu()
    try:
        for index, row in enumerate(data['records']):
            image = cv2.imread(str(ROOT / row['image']['path']))
            if image is None or list(image.shape[:2]) != row['original_hw']:
                raise ValueError(row['id'])
            captured = extractor.predict(image)
            output['R0'].append(dict(id=row['id'], raw_hw=list(image.shape[:2]), **serial(captured)))
            for (arm, seed), head in heads.items():
                name = f'{arm}_seed{seed}'
                prediction, _ = predict_captured(head, arm, captured, dimensions, order,
                    lock['temperatures'][name], lock['decode_rule'], image.shape[:2], normalization)
                output[name].append(dict(id=row['id'], raw_hw=list(image.shape[:2]), **prediction))
            if index == 0 or (index + 1) % 20 == 0:
                E.gpu()
                print(f'GREEN0918 INFERENCE {index + 1}/119', flush=True)
    finally:
        extractor.close()
    F.freeze(destination, dict(complete=True, dataset=F.binding(DATASET),
        model=F.binding(F.DOC / 'MODEL_LOCK.json'), GT_input=False, camera_input=False,
        dimensions_input=[1.1, 1.1, .15], predictions=output,
        gpu_start=start_gpu, gpu_end=E.gpu()))


def score():
    import numpy as np

    data = load_dataset()
    payload = F.read(RAW / 'PREDICTIONS.json')
    F.verify(payload['dataset']); F.verify(payload['model'])
    E = F.setup()
    from eval_math import measure, summary, damage
    from dev_evaluate import iou

    objects = F.read(E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    permutations = next(row['permutations'] for row in objects
                        if row['object_type'] == 'plastic_standard_110x110x15')
    modes = {}
    for mode in ('manual_declared', 'manual_in_frame', 'all_known_proxy'):
        model_rows = {}
        for name, predictions in payload['predictions'].items():
            if len(predictions) != len(data['records']):
                raise ValueError('Incomplete predictions')
            scored = []
            for row, prediction in zip(data['records'], predictions):
                if row['id'] != prediction['id']:
                    raise ValueError('Prediction order mismatch')
                annotation = F.read(ROOT / row['annotation']['path'])
                gt, all_valid = G.annotation_arrays(annotation)
                if mode == 'all_known_proxy':
                    valid = all_valid.copy()
                else:
                    _, valid = G.annotation_arrays(annotation, True)
                h, w = prediction['raw_hw']
                inside = np.isfinite(gt).all(1) & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
                if mode == 'manual_in_frame':
                    valid &= inside
                box_points = all_valid & inside
                if not box_points.any():
                    raise ValueError(f'No in-image matching points: {row["id"]}')
                box = np.r_[gt[box_points].min(0), gt[box_points].max(0)]
                selected = prediction['selected_index']
                candidate = None if selected is None else prediction['candidates'][selected]
                matched = candidate is not None and iou(candidate['box_xyxy'], box) >= .5
                points = np.full((9, 2), np.nan) if candidate is None else candidate['keypoints_xy']
                metrics = measure(points, gt, valid, permutations, (h, w), matched, selected is not None)
                scored.append(dict(id=row['id'], session=row['session'], **metrics))
            model_rows[name] = scored
        comparisons = {}
        for reference in ('R0', 'N0_BASE_REPLAY'):
            comparisons[reference] = {}
            for seed in (1, 2, 3):
                after = model_rows[f'N2_DIM_ONLY_seed{seed}']
                before = model_rows['R0' if reference == 'R0' else f'N0_BASE_REPLAY_seed{seed}']
                comparisons[reference][str(seed)] = dict(
                    delta_E_sym=float(np.mean([a['E_sym'] - b['E_sym'] for a, b in zip(after, before)])),
                    **damage(before, after))
        modes[mode] = dict(rows=model_rows,
            summary={name: summary(rows) for name, rows in model_rows.items()},
            N2_comparisons=comparisons)
    F.freeze(RAW / 'METRICS.json', dict(protocol=PROTOCOL, counts=data['counts'], modes=modes,
        predictions=F.binding(RAW / 'PREDICTIONS.json'),
        dimension_effect_identifiable=False,
        dimension_effect_note='All 119 frames have one identical W,D,H vector. N2 versus N0 measures whole trained-model transfer, not causal value of per-frame dimension variation.',
        confidence_interval='NOT_ESTIMATED_SINGLE_CORRELATED_CAPTURE_SESSION'))
    print('GREEN0918_2D_SCORE_COMPLETE', flush=True)


def fmt(value):
    return f'{value:.4f}'


def report():
    data = load_dataset()
    result = F.read(RAW / 'METRICS.json')
    summary = result['modes']['manual_declared']['summary']
    names = ['R0', *(f'N0_BASE_REPLAY_seed{s}' for s in (1, 2, 3)), *(f'N2_DIM_ONLY_seed{s}' for s in (1, 2, 3))]
    lines = [
        '# 0918 초록 정사각형 팔레트 119장 감사', '',
        '이 자료는 기존 GREEN150과 encoded SHA 및 decoded pixel SHA가 모두 0장 겹치는 별도 집단이다. '
        '동결된 R0와 치수 없는 N0, 치수 입력 N2만 평가했으며 새 학습이나 checkpoint 선택은 하지 않았다.', '',
        '## 2D 직접 클릭점 결과', '',
        '| 모델 | median px | P90 px | PCK5 | PCK10 | PCK20 | matched/119 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in names:
        row = summary[name]
        lines.append(f"| {name} | {fmt(row['matched_pooled_corner8_median_px'])} | "
                     f"{fmt(row['matched_pooled_corner8_P90_px'])} | {100*row['PCK']['5']:.2f}% | "
                     f"{100*row['PCK']['10']:.2f}% | {100*row['PCK']['20']:.2f}% | {row['matched']}/119 |")
    counts = data['counts']
    lines += [
        '', '주 표는 기존 GREEN 계약과 같이 declared-visible `manual_click` 602점을 모두 유지한다. '
        '그중 이미지 밖 수동점 2개를 제외한 600점 민감도와 PnP 생성점을 포함한 proxy 결과도 `METRICS.json`에 함께 보존했다.', '',
        '## 자료 상태', '',
        f"- 119장, 수동 클릭 {counts['manual_corners']}점(이미지 안 {counts['manual_in_frame_corners']}점), "
        f"PnP 생성 {counts['source_counts'].get('pnp_projected', 0)}점, 자동 중심 {counts['source_counts'].get('centroid_auto', 0)}점.",
        f"- 모든 파일은 `split=train`, `population_role=DEV`로 함께 표기되어 있다. 실제 학습 사용 여부는 이 플래그만으로 확정하지 않았다.",
        '- 동결 N0/N2가 사용한 합성 TRAIN55,980장의 encoded image SHA와는 0장 겹친다.',
        f"- canonical pose 없음 {counts['canonical_pose_missing']}/119, signed axis 미확정 {counts['unconfirmed_signed_axis']}/119, 수동 검토 필요 {counts['manual_review_required']}/119.",
        f"- 저장 reprojection error 중앙값 {counts['reprojection_px']['median']:.4f}px, 최대 {counts['reprojection_px']['maximum']:.4f}px.",
        '- 3D T/R 정답 완료 자료로 사용하지 않았다. 이번 결과는 재사용 개발 2D 감사다.', '',
        '## 해석', '',
        '119장 모두 같은 1.10×1.10×0.15m 치수를 가진다. 따라서 이 집단 안에서는 치수 벡터가 상수이고, '
        '치수 정보 자체의 인과 효과를 식별할 수 없다. 같은 seed의 N2와 N0 차이는 학습된 전체 모델의 전이 결과로만 읽어야 한다.', '',
        '## 주석 검토 예시', '',
        '- [026500 overlay](../../../outputs/annotations/0918_dataset_square/_overlays/026500.png): 저장 reprojection error 7.7320px.',
        '- [028924 overlay](../../../outputs/annotations/0918_dataset_square/_overlays/028924.png): 저장 reprojection error 8.1448px.',
        '- [029710 annotation](../../../outputs/annotations/0918_dataset_square/029710.json), '
        '[029844 annotation](../../../outputs/annotations/0918_dataset_square/029844.json): visible 수동 클릭점이 이미지 밖이라 QA 민감도에서 분리했다.', '',
        '[고정 데이터 스냅샷](DATASET_SNAPSHOT.json) · '
        '[전수 예측](../../../data/pallet/results/pallet_green0918_dimension_audit_v1/PREDICTIONS.json) · '
        '[전수 지표](../../../data/pallet/results/pallet_green0918_dimension_audit_v1/METRICS.json)',
    ]
    destination = DOC / 'REPORT_KO.md'
    text = '\n'.join(lines) + '\n'
    if destination.exists() and destination.read_text() != text:
        raise ValueError('Immutable report differs')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(text)
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('snapshot', 'infer', 'score', 'report', 'all'))
    args = parser.parse_args()
    if args.stage == 'all':
        for action in (snapshot, infer, score, report):
            action()
    else:
        globals()[args.stage]()


if __name__ == '__main__':
    main()
