"""Independent saved-array arithmetic only; no model/GT/image inference."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'scripts/research').is_dir())
OUT = HERE.parent / 'class_agnostic_AMG_CAL_case_v1'
PLAN = HERE.parent / 'class_agnostic_AMG_case_plan_v1/PREDECLARATION.json'
PLAN_SHA = 'dc27006791204864ddfaf573dd42108975bbb7b336451a2c293224f1c3ba8a72'
SEAL_SHA = 'e26979a2da604990512289a259e82afc59908e254d4787aa96a0ab1394a9a088'


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
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha(path))


def main():
    checks = []
    def check(name, passes):
        checks.append(dict(check=name, passes=bool(passes)))
        if not passes:
            raise AssertionError(name)
    check('exact plan', sha(PLAN) == PLAN_SHA)
    check('exact execution seal', sha(OUT / 'EXECUTION_SEAL.json') == SEAL_SHA)
    plan, seal, row = read(PLAN), read(OUT / 'EXECUTION_SEAL.json'), read(OUT / 'RESULT.json')
    check('complete fixed case', seal['complete'] and row['id'] == plan['case']['id']
          == 'eval_outside:1778651519235162880' and row['partition'] == 'CAL')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        check(key + ' bound', seal[key] == plan[key])
        for name, digest in plan[key].items():
            check(key + ':' + name, sha(resolve(name)) == digest)
    for name, digest in seal['assets'].items():
        check('sealed asset:' + name, sha(OUT / name) == digest)
    check('complete asset count', len(seal['assets']) == 50)
    check('all42 returned preserved', len(row['candidates']) == row['returned_candidates']
          == seal['returned_candidates'] == 42 and row['all_returned_masks_preserved'])
    check('uniform standalone budget', row['detector_calls'] == 0
          and row['global_grid_prompts'] == 256 and row['expected_image_encoders'] == 1
          and row['expected_prompt_decoder_batches'] == 4
          and row['raw_multimask_slots_before_vendor_filters'] == 768)
    check('no production or semantic/GT claims', row['current_pipeline_mask_supply']
          == row['new_GT_reference_or_primary_reads'] == row['other_object_or_depth_certificates']
          == row['KP_S_U_O_local_prompt_queries'] == row['pose_noise_fits'] == 0
          and not row['target_complement_used_as_occluder'])
    for key in ('image_binding', 'target_mask_binding'):
        check(key + ' exact original', row[key] == plan['case'][key]
              and sha(resolve(row[key]['path'])) == row[key]['sha256'])
    with np.load(resolve(row['target_mask_binding']['path']), allow_pickle=False) as arrays:
        target = arrays['pallet_mask_clean'] >= .5
    h, w = target.shape
    check('same full native canvas', row['image_hw'] == [h, w])
    x_grid = (np.arange(16) + .5) * w / 16
    y_grid = (np.arange(16) + .5) * h / 16
    statuses, quality = Counter(), []
    summaries = []
    for i, candidate in enumerate(row['candidates']):
        name = f'candidate{i}'
        check(name + ' original vendor order', candidate['candidate_index'] == i)
        file = candidate['candidate_file_binding']
        check(name + ' file SHA', sha(resolve(file['path'])) == file['sha256'])
        with np.load(resolve(file['path']), allow_pickle=False) as arrays:
            mask = arrays['segmentation']
            check(name + ' binary native pixels', mask.dtype == np.uint8
                  and mask.shape == target.shape and np.isin(mask, [0, 1]).all())
            check(name + ' area', candidate['area_pixels'] == int(mask.sum()))
            header = json.dumps(dict(shape=[h, w], dtype='uint8'), sort_keys=True).encode()
            digest = hashlib.sha256(header + b'\0' + np.ascontiguousarray(mask).tobytes()).hexdigest()
            check(name + ' independent pixel SHA', digest == candidate['mask_pixel_sha256'])
            yy, xx = np.nonzero(mask)
            expected_bbox = [float(xx.min()), float(yy.min()), float(xx.max() - xx.min()), float(yy.max() - yy.min())]
            check(name + ' bbox from actual pixels', candidate['bbox_xywh'] == expected_bbox)
            binary = mask.astype(bool)
            n, nt = int(binary.sum()), int(target.sum())
            inter, union = int((binary & target).sum()), int((binary | target).sum())
            expected = dict(candidate_pixels=n, target_pixels=nt, intersection_pixels=inter,
                union_pixels=union, candidate_fraction_in_target=inter / n if n else None,
                target_fraction_in_candidate=inter / nt if nt else None,
                IoU=inter / union if union else None, any_positive_target_overlap=inter > 0,
                overlap_is_not_independent_other_object_or_depth_evidence=True)
            check(name + ' independent overlap', candidate['target_overlap'] == expected)
        point = np.asarray(candidate['global_prompt_point_xy'])
        check(name + ' uniform global prompt', point.shape == (1, 2)
              and np.min(abs(x_grid - point[0, 0])) <= 1e-4
              and np.min(abs(y_grid - point[0, 1])) <= 1e-4)
        check(name + ' one full crop', candidate['crop_box_xywh'] == [0, 0, w, h])
        check(name + ' fixed vendor quality filters', np.isfinite(candidate['predicted_iou'])
              and candidate['predicted_iou'] > plan['configuration']['pred_iou_thresh']
              and candidate['stability_score'] >= plan['configuration']['stability_score_thresh'])
        expected_status = 'UNKNOWN' if not n or not nt else 'suspected_target' if inter else 'CANDIDATE'
        check(name + ' no identity/depth certification', candidate['status'] == expected_status
              and candidate['target_identity'] == 'UNKNOWN'
              and not candidate['scores_are_semantic_or_depth_probabilities']
              and not candidate['other_object_confirmed'] and not candidate['foreground_occlusion_confirmed']
              and not candidate['precise_keypoint_created'] and not candidate['supplied_to_current_pipeline'])
        statuses[candidate['status']] += 1
        quality.append(candidate['predicted_iou'])
        summaries.append({k: candidate[k] for k in ('candidate_index', 'bbox_xywh', 'area_pixels',
            'predicted_iou', 'stability_score', 'status', 'mask_pixel_sha256')})
    check('saved statuses', dict(statuses) == {'CANDIDATE': 40, 'suspected_target': 2})
    check('full five figures', len(row['figures']) == 5
          and [Path(f['path']).name for f in row['figures']] == ['overview_candidates.png']
          + [f'gallery_{i:02d}.png' for i in range(4)])
    for figure in row['figures']:
        check('figure:' + figure['path'], sha(resolve(figure['path'])) == figure['sha256'])
    links = re.findall(r'\]\(([^)]+)\)', (OUT / 'REPORT_KO.md').read_text())
    check('report five local links', len(links) == 5 and all((OUT / link).is_file() for link in links))
    result = dict(schema='STANDALONE_AMG_FIXED_CAL_ACTUAL_QA_V1', UTC=datetime.now(timezone.utc).isoformat(),
        checks=len(checks), failures=0, check_results=checks,
        execution=dict(exec_session=45116, PID=429656, first_yield_ms=1000, exit_code=0,
            actual_inference_attempts=1, elapsed_seconds=seal['elapsed_seconds'],
            sampled_RSS_KiB=[1897760, 1832356], sampled_RSS_is_not_peak=True,
            encoder_and_decoder_counts_are_source_expected_not_instrumented=True),
        plan=binding(PLAN), execution_seal=binding(OUT / 'EXECUTION_SEAL.json'),
        source_counts=dict(source=len(plan['source_sha256']), input=len(plan['input_sha256']),
            protected=len(plan['protected_user_sha256'])),
        counts=dict(returned_masks=42, CANDIDATE=40, suspected_target=2, UNKNOWN=0, saved_mask_NPZ=42, figures=5),
        all5_figures_directly_viewed=True, candidates=summaries,
        visual_findings_KO=[
            '42개 전체 갤러리에서 단독 파란 스툴 마스크는 보이지 않았다. 수동 pixel 정답이나 정확도 점수를 만든 판정은 아니다.',
            'AMG5는 스툴과 팔레트가 합쳐진 mask, AMG6는 주로 팔레트 mask다. 둘 다 target과 겹쳐 suspected_target으로 남았다.',
            '나머지40 CANDIDATE는 지면/건물/창문/배경 사람 등의 세그먼트로 보이며 모든 후보의 실제 object identity와 앞뒤 관계는 UNKNOWN이다.',
            'predicted_iou 최고값은1을 넘는다. 보관한 uncalibrated model score이며 semantic/depth probability로 간주하거나 범위를 잘라 쓰지 않았다.',
            'AMG5에서 AMG6를 뺀 차집합이나 target complement를 independent other-object/occluder mask로 만들거나 채택하지 않았다.'
        ], max_uncalibrated_predicted_iou=max(quality),
        scope=dict(saved_mask_arrays_opened=42, original_target_array_opened=1,
            RGB_redecodes_during_QA=0, NN_forward_during_QA=0, GT_or_primary_reads=0,
            new_pose_noise_fit=0, frozen_source_PLAN_report_or_result_edits=0,
            production_pipeline_supply=0, independent_occlusion_certificate=0))
    path = HERE / 'ACTUAL_EXECUTION_QA_V1.json'
    if path.exists():
        raise FileExistsError('Preserve prior QA')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'ACTUAL_QA_SEAL_V1.json').write_text(json.dumps(dict(receipt_sha256=sha(path),
        audit_source=binding(Path(__file__)), execution_seal_sha256=SEAL_SHA), indent=2) + '\n')
    print(json.dumps(dict(checks=len(checks), failures=0, receipt=binding(path), statuses=dict(statuses)), ensure_ascii=False))


if __name__ == '__main__':
    main()
