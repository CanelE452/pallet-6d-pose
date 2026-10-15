"""Replay saved supplier arrays/metadata only; never run models or open labels."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'scripts/research').is_dir())
HERE = Path(__file__).resolve().parent
B = HERE.parent
OUT = B / 'generic_instance_mask_candidates_CAL46_v1'
PLAN = B / 'generic_instance_mask_population_CAL_plan_v1/PREDECLARATION.json'
EXPECTED_PLAN = 'd8e3a79b1fa96cfd2972f6759ec334858280e8ed86fbd7ff67b347c71029bb4e'
EXPECTED_SEAL = '1065a9834226a6155eaaff49c902f1ca54ad9b733409883dde9a2b060d30b453'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def binding(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def main():
    checks = []
    def check(name, value):
        checks.append(dict(check=name, passes=bool(value)))
        if not value:
            raise AssertionError(name)

    check('repository root', (ROOT / 'scripts').is_dir())
    check('declared plan exact', sha(PLAN) == EXPECTED_PLAN)
    check('execution seal exact', sha(OUT / 'EXECUTION_SEAL.json') == EXPECTED_SEAL)
    plan, seal = read(PLAN), read(OUT / 'EXECUTION_SEAL.json')
    rows = read(OUT / 'RESULTS.json')['cases']
    check('complete all46 order', seal['complete'] and seal['cases'] == len(rows) == 46
          and [r['id'] for r in rows] == plan['ordered_ids'] == seal['ordered_ids'])
    check('same fixed supplier configuration', plan['configuration'] == dict(conf=.25,
          imgsz=640, max_det=20, iou=.7, batch=1, half=False, device='cpu',
          agnostic_nms=False, save=False, verbose=False))
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        check(key + ' execution equals plan', seal[key] == plan[key])
        for path, digest in plan[key].items():
            check(key + ':' + path, sha(resolve(path)) == digest)
    for path, digest in seal['assets'].items():
        check('sealed asset:' + path, sha(OUT / path) == digest)

    status, classes, calls = Counter(), Counter(), Counter()
    by_id = {}
    for i, (row, case) in enumerate(zip(rows, plan['cases'])):
        check(f'row{i} exact saved JSON', row == read(OUT / f'case{i:02d}.json'))
        check(f'row{i} CAL identity/input bindings', row['partition'] == 'CAL'
              and row['id'] == case['id'] and row['image_binding'] == case['image_binding']
              and row['target_mask_binding'] == case['target_mask_binding'])
        for key in ('image_binding', 'target_mask_binding', 'figure_binding'):
            check(f'row{i} {key}', sha(resolve(row[key]['path'])) == row[key]['sha256'])
        with np.load(resolve(row['target_mask_binding']['path']), allow_pickle=False) as arrays:
            target = arrays['pallet_mask_clean'] >= .5
        check(f'row{i} image canvas', list(target.shape) == row['image_hw'])
        detections = row['detections']
        check(f'row{i} detector count', len(detections) == row['returned_detections'] <= 20)
        check(f'row{i} one encoder and detector', row['detector_forward_calls'] == 1
              and row['SAM_image_encoder_calls'] == int(bool(detections)))
        check(f'row{i} standalone zero contracts', row['zero_rows_retained']
              and row['target_mask_used_only_for_postprediction_overlap']
              and row['new_KP_S_U_O_coordinate_prompts'] == row['masks_supplied_to_current_method']
              == row['other_object_or_depth_confirmations'] == row['pose_or_noise_fits'] == 0
              and not row['target_complement_or_O_circle_or_same_target_alternative_used_as_other_mask'])
        check(f'row{i} unavailable retained', row['status'] == ('CANDIDATES_AVAILABLE' if detections else 'UNAVAILABLE'))
        for key in ('detector_forward_calls', 'SAM_image_encoder_calls', 'SAM_prompt_decoder_calls', 'returned_detections'):
            calls[key] += row[key]
        status['rows_' + row['status']] += 1
        for j, d in enumerate(detections):
            prefix = f'row{i} detection{j}'
            check(prefix + ' returned order/confidence', d['detection_index'] == j
                  and .25 <= d['detector_confidence'] <= 1 and 0 <= d['detector_class_id'] < 80)
            check(prefix + ' no semantic/depth/adoption claim', not d['other_object_confirmed']
                  and not d['foreground_occlusion_confirmed'] and not d['precise_point_created']
                  and not d['supplied_to_existing_pipeline'])
            file_binding = d['candidate_file_binding']
            check(prefix + ' candidate file SHA', sha(resolve(file_binding['path'])) == file_binding['sha256'])
            with np.load(resolve(file_binding['path']), allow_pickle=False) as arrays:
                masks, quality = arrays['sam_candidate_masks'], arrays['sam_candidate_quality']
                check(prefix + ' three native binary masks', masks.dtype == np.uint8
                      and masks.shape == (3, *target.shape) and np.isin(masks, [0, 1]).all())
                check(prefix + ' quality shape/finite', quality.shape == (3,) and np.isfinite(quality).all())
                check(prefix + ' saved first maximum', d['selected_mask_index'] == int(np.argmax(quality)))
                for k, (mask, score, alt) in enumerate(zip(masks, quality, d['alternatives'])):
                    check(prefix + f' alternative{k} score', alt['index'] == k
                          and alt['uncalibrated_SAM_quality'] == float(score))
                    header = json.dumps(dict(shape=list(mask.shape), dtype='uint8'), sort_keys=True).encode()
                    pixel_sha = hashlib.sha256(header + b'\0' + np.ascontiguousarray(mask).tobytes()).hexdigest()
                    check(prefix + f' alternative{k} pixel SHA', pixel_sha == alt['mask_pixel_sha256'])
                    binary = mask.astype(bool)
                    n, nt = int(binary.sum()), int(target.sum())
                    inter, union = int((binary & target).sum()), int((binary | target).sum())
                    independent = dict(candidate_pixels=n, target_pixels=nt, intersection_pixels=inter,
                        union_pixels=union, candidate_fraction_in_target=inter / n if n else None,
                        target_fraction_in_candidate=inter / nt if nt else None,
                        IoU=inter / union if union else None, any_positive_target_overlap=inter > 0,
                        overlap_is_not_independent_other_object_or_depth_evidence=True)
                    check(prefix + f' alternative{k} independent overlap arithmetic', alt['target_overlap'] == independent)
                    status['SAM_alternatives'] += 1
            selected = d['alternatives'][d['selected_mask_index']]['target_overlap']
            expected = ('UNKNOWN' if not selected['candidate_pixels'] or not selected['target_pixels']
                        else 'suspected_target' if selected['intersection_pixels'] else 'CANDIDATE')
            check(prefix + ' diagnostic status', d['status'] == expected)
            status[d['status']] += 1
            classes[d['detector_class_name']] += 1
        by_id[row['id']] = row

    old = read(B / 'generic_instance_mask_candidates_CAL3_v1/RESULTS.json')['cases']
    for row in old:
        current = by_id[row['id']]
        keys = ('detector_box_xyxy', 'detector_confidence', 'detector_class_id', 'detector_class_name',
                'status', 'selected_mask_index', 'alternatives', 'SAM_box_xyxy')
        check('CAL3 exact saved predictions:' + row['id'], len(row['detections']) == len(current['detections'])
              and all({k: a[k] for k in keys} == {k: b[k] for k in keys}
                      for a, b in zip(row['detections'], current['detections'])))
    report = (OUT / 'REPORT_KO.md').read_text()
    links = re.findall(r'\]\(([^)]+)\)', report)
    check('all46 report links', len(links) == 46 and all((OUT / p).is_file() for p in links))
    check('runtime settings inside output', (OUT / 'ultralytics_runtime_config/Ultralytics/settings.json').is_file())
    check('fixed aggregate counts', dict(calls) == dict(detector_forward_calls=46,
          SAM_image_encoder_calls=27, SAM_prompt_decoder_calls=67, returned_detections=67)
          and status['SAM_alternatives'] == 201 and status['suspected_target'] == 22
          and status['CANDIDATE'] == 45 and status['rows_UNAVAILABLE'] == 19)
    result = dict(schema='GENERIC_INSTANCE_MASK_CAL46_SAVED_ARRAY_QA_V1',
        UTC=datetime.now(timezone.utc).isoformat(), checks=len(checks), failures=0,
        check_results=checks, execution=dict(exec_session=50457, first_yield_ms=1000,
            exit_code=0, elapsed_seconds=seal['elapsed_seconds'], actual_inference_attempts=1,
            OOM_or_failure=False), plan=binding(PLAN), execution_seal=binding(OUT / 'EXECUTION_SEAL.json'),
        counts=dict(calls), statuses=dict(status), classes=dict(classes),
        direct_visual_review=dict(all46_viewed=True, via_all8_saved_contact_sheets=True,
            contact_sheets={str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / 'actual_visual_contact_sheets_v1').glob('*.png'))}),
        visual_findings_KO=[
            'CAD18 중16은 검출0, 2장은 팔레트 자체 bench/target 의심. 빈 결과는 가림 없음 증거가 아니다.',
            'night08 세 장의 실제 팔레트 앞/위 원뿔은 검출되지 않아 모두 UNAVAILABLE이다.',
            'outside 다섯 장은 배경 사람/우산 후보만 검출했다. 첫 세 장의 팔레트 위 파란 의자와 추가 표지/원뿔은 후보가 없어 가림 공급의 양성 검증이 되지 않았다.',
            'wood20장의 후보는 주로 팔레트 bench와 떨어진 자전거다. 같은 팔레트 중복 bench 및 자전거/오토바이 명칭 혼동도 있어 독립 instance 또는 깊이의 정답이 아니다.'
        ], usefulness_KO='전체 CAL46 동일 자동 공급은 실행됐으나 현재 candidate45를 앞가림 mask로 채택할 근거는 없다. 실제 주요 가림 물체 누락과 target 자체 검출이 확인되어 기존 관측/pose pipeline 공급0을 유지한다.',
        limitations=dict(target_overlap_diagnostic_only=True, independent_other_object_or_depth_certificates=0,
            metrics_or_GT_reference_reads=0, pose_noise_fits=0, source_or_frozen_artifact_edits=0,
            no_instance_quality_or_occlusion_accuracy_metric_claim=True,
            contact_sheet_review_not_pixel_ground_truth_annotation=True),
        scope=dict(original_target_NPZ_arrays=46, saved_candidate_NPZ_arrays=67,
            model_inference_during_QA=0, RGB_redecodes_during_QA=0, published_CAL3_saved_prediction_comparisons=3))
    path = HERE / 'ACTUAL_EXECUTION_QA_V1.json'
    if path.exists():
        raise FileExistsError('Append-only QA receipt already exists')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'ACTUAL_QA_SEAL_V1.json').write_text(json.dumps(dict(receipt_sha256=sha(path),
        audit_source=binding(Path(__file__)), execution_seal_sha256=EXPECTED_SEAL), indent=2) + '\n')
    print(json.dumps(dict(checks=len(checks), failures=0, receipt=binding(path), counts=dict(calls), statuses=dict(status)), ensure_ascii=False))


if __name__ == '__main__':
    main()
