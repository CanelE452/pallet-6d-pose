"""Prepare a read-only, hash-bound route from existing lifter inputs to evaluation.

No human action, reference coordinate, object-match decision, or model output is
created. The original editor and all previous closeout files are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'scripts/research/pallet_lifter_case_review_20261003_v1'
RAW = ROOT / 'data/pallet/results/pallet_lifter_case_review_20261003_v1'
REVIEW = RAW / 'review'
DOC = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1'
DEFAULT_OUTPUT = DOC / 'lifter_connection_20261006_v1'
EXPECTED_PENDING = {'174126:13': 3, '174126:419': 5, '174342:1190': 5,
                    '174925:32': 5, '174925:1002': 5}
AXES = ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size, sha256=digest(path))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def check(condition, description, checks):
    if not condition:
        raise ValueError(description)
    checks[description] += 1


def output_folder(proposed):
    if not proposed.exists():
        return proposed
    number = 2
    while True:
        candidate = proposed.with_name(proposed.name.rsplit('_v', 1)[0] + f'_v{number}')
        if not candidate.exists():
            return candidate
        number += 1


def inventory():
    return {str(path.relative_to(ROOT)): bind(path) for path in sorted(REVIEW.rglob('*'))
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.png')}


def prepare_command(batch):
    return [sys.executable, str(SCRIPTS / 'open_existing_annotation.py'),
            '--pass', 'primary', '--visibility-only', '--revisit-saved',
            '--batch-plan', str(REVIEW / batch)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    checks = Counter()
    output = output_folder(args.output.resolve())
    before_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    before_tracked = subprocess.check_output(
        ['git', 'status', '--short', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
    packet_path = DOC / 'closeout_20261006_v1/final_review/FINAL_REVIEW_PACKET.json'
    packet = read(packet_path)
    protected = list(packet['inputs_and_updated_paper'])
    protected += [bind(packet_path), bind(ROOT / 'scripts/annotate/annotate.py')]
    review_before = inventory()
    protected += list(review_before.values())
    raw_path = RAW / 'raw_predictions/ALL_STORED_FRAMES.jsonl'
    identity_path = RAW / 'raw_predictions/RUN_IDENTITY.json'
    integration_path = DOC / 'INTEGRATION_MANIFEST.json'
    time_path = RAW / 'LIFTER_INPUT_AND_TIME_MAP.json'
    protected += [bind(path) for path in (raw_path, identity_path, integration_path, time_path)]
    expected_raw = read(integration_path)['inputs']['lifter_raw']
    actual_raw = bind(raw_path)
    check(actual_raw['sha256'] == expected_raw['sha256'] and
          actual_raw['bytes'] == expected_raw['bytes'], 'original_8910_raw_receipt_binding', checks)
    protected_by_path = {entry['path']: entry for entry in protected}
    for entry in protected_by_path.values():
        check(bind(resolve(entry['path'])) == entry, 'protected_input_hash_before', checks)

    sys.path.insert(0, str(SCRIPTS))
    from open_existing_annotation import Context, NativeReview
    import cv2
    context = Context(REVIEW / 'MANIFEST.json', RAW / 'LIFTER_EVALUATION_PLAN.json',
                      REVIEW / 'CORNER_CONTRACT.json', REVIEW / 'annotations_in_progress.json')
    native = NativeReview(context, None, REVIEW / 'native_annotations', passes=('primary',),
                          native_pnp=False, batch_plan=REVIEW / 'SMALL_BATCH_12_V1.json')
    batch = read(REVIEW / 'SMALL_BATCH_12_V1.json')
    queue = read(REVIEW / 'REMAINING_CORNERS_5_V1.json')
    recovery = read(REVIEW / 'VIEWER_DRAFT_RECOVERY.json')
    plan = read(RAW / 'LIFTER_EVALUATION_PLAN.json')
    planned = {frame['frame_id']: frame for frame in plan['frames']}
    check(len(planned) == 8910 and len(batch['frame_ids']) == 12, 'unchanged_8910_and_12_populations', checks)
    check(queue['frame_ids'] == list(EXPECTED_PENDING) and
          queue['original_evaluation_frame_ids'] == batch['frame_ids'], 'five_tasks_retain_original_12_scope', checks)
    check(queue['parent_batch_sha256'] == digest(REVIEW / 'SMALL_BATCH_12_V1.json'),
          'five_tasks_bind_original_12_batch', checks)
    check(recovery['bindings'] == context.bindings and batch['bindings'] == context.bindings
          and queue['bindings'] == context.bindings, 'review_input_contract_bindings', checks)
    rows, sessions, pose_states = {}, Counter(), {method: Counter() for method in ('Base', 'N3')}
    max_move = 0.0
    identity_keys = {'session_id': 'session_id', 'stored_index': 'saved_frame_index',
                     'sensor_timestamp_ms': 'camera_sensor_timestamp_ms',
                     'camera_frame_number': 'camera_frame_number', 'decoded_bgr_sha256': 'decoded_bgr_sha256'}
    for line in raw_path.open(encoding='utf-8'):
        row = json.loads(line)
        fid = row['frame_id']
        check(fid in planned and fid not in rows, 'raw_unique_frame_id_in_frozen_plan', checks)
        frame = planned[fid]
        for key, plan_key in identity_keys.items():
            check(row[key] == frame[plan_key], 'raw_plan_frame_time_pixel_identity', checks)
        check(fid == f"{row['session_id']}:{row['stored_index']}", 'raw_frame_id_matches_stored_index', checks)
        before, after = row['methods']['Base'], row['methods']['N3']
        check(before['keypoints_mask'] == after['keypoints_mask'], 'Base_N3_same_point_valid_mask', checks)
        check(before['selected_object'] == after['selected_object'] and
              before.get('selected_index') == after.get('selected_index'), 'Base_N3_same_frozen_object_selection', checks)
        for method in ('Base', 'N3'):
            prediction = row['methods'][method]
            check(prediction.get('object_id') is None and prediction.get('object_match') is None,
                  'raw_human_object_match_not_synthesized', checks)
            pose_states[method][prediction['pose_state']] += 1
        for index, valid in enumerate(before['keypoints_mask']):
            if valid:
                p, q = before['keypoints'][index], after['keypoints'][index]
                move = math.hypot(p[0] - q[0], p[1] - q[1])
                check(math.isfinite(move) and move <= 8.00001, 'unchanged_8px_N3_move_cap', checks)
                max_move = max(max_move, move)
        rows[fid] = row
        sessions[row['session_id']] += 1
    check(set(rows) == set(planned), 'all_8910_raw_plan_frames_present', checks)
    check(dict(sessions) == {'173507': 3729, '174126': 757, '174342': 2501, '174925': 1923},
          'four_original_session_counts', checks)

    totals = Counter()
    selected = []
    for fid in batch['frame_ids']:
        frame = context.frames[fid]
        draft = recovery['drafts'][fid + '|primary']['record']
        check(draft['frame_id'] == fid and draft['review_pass'] == 'primary'
              and draft.get('reviewer') is None, 'actual_draft_frame_without_formal_approval', checks)
        check(digest(context.images[fid]) == frame['image_sha256'], '12_original_PNG_file_hashes', checks)
        image = cv2.imread(str(context.images[fid]), cv2.IMREAD_COLOR)
        check(image is not None and image.shape == (frame['height'], frame['width'], 3),
              '12_original_image_dimensions', checks)
        check(hashlib.sha256(image.tobytes()).hexdigest() == frame['decoded_bgr_sha256']
              == rows[fid]['decoded_bgr_sha256'], '12_PNG_decoded_pixel_to_original_prediction', checks)
        status = native.visibility_only_status(fid, 'primary')
        check(status['complete'] and status['confirmed_category_count'] == 8,
              'actual_96_category_labels_with_chat_declarations', checks)
        manual_ids, hidden_ids, pending_ids = [], [], []
        for corner in draft['corners']:
            visibility = corner['visibility']
            if visibility == 'direct_visible':
                check(corner['definition_confirmed'] and not any(corner[axis] for axis in AXES)
                      and all(type(corner[k]) in (int, float) and math.isfinite(corner[k]) for k in ('x', 'y')),
                      'actual_manual_reference_xy_and_states', checks)
                totals['manual_coordinates'] += 1
                manual_ids.append(corner['id'])
            elif visibility == 'not_direct_visible':
                check(corner['self_occlusion'] and not any(corner[axis] for axis in AXES if axis != 'self_occlusion')
                      and corner['x'] is None and corner['y'] is None, 'actual_self_hidden_no_reference_xy', checks)
                totals['self_hidden'] += 1
                hidden_ids.append(corner['id'])
            else:
                check(visibility is None and corner['x'] is None and corner['y'] is None
                      and not any(corner[axis] for axis in AXES), 'five_raw_pending_corners_remain_null', checks)
                check(EXPECTED_PENDING.get(fid) == corner['id'], 'only_expected_five_corners_pending', checks)
                declared = status['corners'][corner['id']]
                check(declared.get('category_label_only') is True and declared['visibility'] == 'direct_visible'
                      and declared['x'] is None and declared['y'] is None,
                      'chat_visible_declaration_never_creates_coordinate', checks)
                hint = queue['position_hints'][fid]
                source = Path(hint['source_pnp'])
                pnp = read(source)
                check(digest(source) == hint['source_pnp_sha256'] and pnp['bindings'] == context.bindings
                      and pnp['frame_id'] == fid, 'own_PnP_hint_input_hash_identity', checks)
                check(pnp['editor_keypoint_annotations'][corner['id']]['source'] == 'pnp_projected'
                      and pnp['editor_kps_2d'][corner['id']] == hint['xy']
                      and pnp['manual_reference_corners'][corner['id']]['x'] is None
                      and pnp['manual_reference_corners'][corner['id']]['y'] is None,
                      'generated_hint_separate_from_manual_reference', checks)
                totals['visible_label_without_coordinates'] += 1
                pending_ids.append(corner['id'])
        selected.append(dict(frame_id=fid, sensor_timestamp_ms=rows[fid]['sensor_timestamp_ms'],
            image_sha256=frame['image_sha256'], decoded_bgr_sha256=frame['decoded_bgr_sha256'],
            manual_corner_ids=manual_ids, self_hidden_corner_ids=hidden_ids,
            missing_reference_corner_ids=pending_ids, reference_object= draft['object'],
            selected_object=rows[fid]['methods']['Base']['selected_object'],
            object_match='NOT_HUMAN_REVIEWED', evaluation_use=False))
    check(dict(totals) == {'manual_coordinates': 67, 'self_hidden': 24,
                          'visible_label_without_coordinates': 5}, '67_manual_24_self_5_label_only_inventory', checks)

    prepare_runs = []
    for batch_name, expected_count in [('REMAINING_CORNERS_5_V1.json', 5), ('SMALL_BATCH_12_V1.json', 12)]:
        command = prepare_command(batch_name) + ['--prepare-only']
        process_started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        duration = time.perf_counter() - process_started
        check(process.returncode == 0, 'existing_annotation_prepare_only_exit_zero', checks)
        payload = json.loads(process.stdout.strip().splitlines()[-1])
        check(payload['editor'] == 'scripts/annotate/annotate.py' and payload['PnP'] is False
              and payload['sessions'] == [f'01_PRIMARY_{expected_count}'], 'existing_editor_5_and_12_actual_queues', checks)
        check(payload['partial_batch_counts']['primary_required'] == expected_count,
              'prepare_only_original_partial_queue_counts', checks)
        prepare_runs.append(dict(command=command, returncode=process.returncode, cpu_wall_seconds=duration,
                                 stdout=payload, stderr=process.stderr, human_actions_created=0))

    for path in (REVIEW / 'annotations_in_progress.json', REVIEW / 'LIFTER_REFERENCE_REVIEWED.json',
                 REVIEW / 'NATIVE_REVIEWER_PROFILE.json'):
        check(not path.exists(), 'prepare_did_not_create_official_store_export_profile', checks)
    check(inventory() == review_before, 'all_original_review_files_unchanged_after_prepare', checks)
    for entry in protected_by_path.values():
        check(bind(resolve(entry['path'])) == entry, 'protected_input_hash_after', checks)
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == before_head,
          'git_HEAD_unchanged', checks)
    check(subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=ROOT,
                                  text=True).splitlines() == before_tracked, 'tracked_worktree_unchanged', checks)

    commands = dict(five_missing_coordinate_clicks=prepare_command('REMAINING_CORNERS_5_V1.json'),
                    approve_original_12_inputs=prepare_command('SMALL_BATCH_12_V1.json'),
                    human_selected_box_matches=[sys.executable, str(SCRIPTS / 'open_object_match_annotation.py'),
                        '--batch-plan', str(REVIEW / 'SMALL_BATCH_12_V1.json')])
    text = '\n\n'.join([
        '# 기존 리프터 자료를 정확도 평가에 연결하기',
        '**새 촬영이나 전체 재어노테이션 없이, 기존 12장의 남은 5좌표를 먼저 채우고 기존 입력을 제출한 뒤 선택 박스만 확인합니다.**',
        '## 1. 사진마다 남은 점 한 개만 클릭',
        '아래 명령은 기존 `scripts/annotate/annotate.py`를 사용합니다. 기존 67좌표와 자체 가림 24상태를 유지하며 5장만 엽니다. 각 사진에서 선택된 실제 코너 한 번 클릭 → **S 저장·다음**입니다. 보임 여부는 이미 답변했으므로 다시 묻지 않습니다. 노란 빈 원은 본인 PnP 위치 참고이며 그 좌표를 정답으로 자동 저장하지 않습니다.',
        '```bash\nDISPLAY=:0 ' + shlex.join(commands['five_missing_coordinate_clicks']) + '\n```',
        '| 프레임 | 추가할 코너 |\n|---|---:|\n' + '\n'.join(f'| {fid} | {index} |' for fid, index in EXPECTED_PENDING.items()),
        '`--revisit-saved`가 필요합니다. 이 옵션이 없으면 기존 보임 분류가 완료된 것으로 인식해 창이 열리지 않습니다. 완료한 좌표를 다시 찍는 옵션이 아니라 남은 좌표 작업과 기존 입력 제출 화면을 열기 위한 옵션입니다.',
        '## 2. 기존 12장 입력을 한 번 제출',
        '다섯 좌표의 실제 저장을 확인한 다음 아래 명령을 실행합니다. 기존 점은 다시 찍지 않습니다. **B 기존 입력 제출**을 누르고 실제 작업 이력을 한 번 답한 뒤 기존 입력을 사용한다는 확인을 합니다. 아직 다섯 좌표가 없는데 B를 먼저 눌러 판단 보류로 바꾸지 않습니다. 제출 뒤 공식 참조 파일과 원시 입력 보존 아카이브가 생성됩니다.',
        '```bash\nDISPLAY=:0 ' + shlex.join(commands['approve_original_12_inputs']) + '\n```',
        '## 3. 모델이 선택한 박스만 12장 확인',
        '참조를 제출한 다음 아래 명령을 실행합니다. 원이미지와 고정 선택 박스만 표시되며 모델 코너·방법명·오차는 표시하지 않습니다. **1 같은 파렛트 / 2 다른 대상 / 3 판단하기 어려움** 중 실제 판정을 선택하면 저장하고 다음으로 넘어갑니다. Base와 N3는 동일한 선택을 사용합니다.',
        '```bash\nDISPLAY=:0 ' + shlex.join(commands['human_selected_box_matches']) + '\n```',
        '## 평가와 범위',
        '코너·대상 입력이 완료되면 기존 8,910프레임 예측을 그대로 재사용하여 12장의 가시 코너 오차 중앙값/P90과 PCK@10px를 계산합니다. 잘못 선택한 객체와 예측 결측은 평가 실패로 남깁니다. **부분 12장 평가**이며 전체 120장 검수 완료, 보류 103장 완료, 반복 23장 완료로 보고하지 않습니다. 독립 물리 T/R 참조는 없으므로 cm·각도 정확도는 x입니다. 정지 잡음은 별도로 실제 정지 구간을 사람이 확인해야 합니다.',
        '사람 입력이 세 단계 모두 저장되면 작업자가 아래 명령으로 평가와 결과 생성을 이어갑니다. 승인·좌표·대응이 미완료면 해당 평가만 대기로 남기며 사람 답변을 자동 생성하지 않습니다.',
        '```bash\n' + shlex.join([sys.executable, str(Path(__file__).with_name('evaluate_lifter_connection_20261006.py'))]) + '\n```',
        '이번 준비 검산은 새 학습·모델 추론·사람 승인·대상 대응을 만들지 않았습니다. 원시 JSON의 다섯 좌표는 여전히 null이고 공식 참조 승인 0장입니다. `--prepare-only` 두 번은 큐와 모듈 연결만 읽으며 창과 승인 파일을 만들지 않았습니다. 향후 실제 사용자 클릭/S/B는 해당 검수 저장소에 새 입력·승인 파일을 씁니다.',
        '[실제 연결 검산·입력 해시·실행 비용](CONNECTION_AUDIT.json)',
    ]) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'CONNECT_KO.md').write_text(text, encoding='utf-8')
    report = dict(schema='lifter_existing_12_reference_connection_preparation_v1',
        status='PASS_READY_FOR_MINIMUM_HUMAN_INPUT', generated_at=datetime.now(timezone.utc).isoformat(),
        script=bind(Path(__file__)), git_head=before_head, tracked_changes=before_tracked,
        scope=dict(raw_frames=8910, partial_reference_frames=12, original_primary_frames=120,
                   retained_primary_frames=115, deferred_primary_frames=103, deferred_repeat_frames=23,
                   full_120_reference_complete=False),
        actual_reference_inventory=dict(totals), frame_connections=selected,
        fixed_predictions=dict(source=actual_raw, sessions=dict(sessions),
            pose_states={method: dict(states) for method, states in pose_states.items()},
            maximum_N3_move_px=max_move, raw_point_masks_and_selection_preserved=True),
        missing_coordinate_fields_remain_null=True, PnP_hint_is_human_reference=False,
        official_reference_approved_frames=0, human_object_match_decisions_created=0,
        automatic_human_review_promotion=False, original_review_file_count=len(review_before),
        protected_previous_final_packet_files=len(packet['inputs_and_updated_paper']),
        original_inputs_and_previous_results_unchanged=True,
        input_bindings=list(protected_by_path.values()), prepare_only_runs=prepare_runs,
        commands=commands, checks=dict(checks), successful_check_count=sum(checks.values()),
        cpu_wall_seconds=time.perf_counter() - started,
        cpu_cost_includes_hash_inventory_and_two_prepare_subprocesses=True,
        training_runs=0, optimizer_updates=0, model_forward_frames=0, gpu_seconds=0,
        PDF_compilations=0, remote_pushes=0, actual_lifter_control=0,
        output=str(output.relative_to(ROOT)), document=bind(output / 'CONNECT_KO.md'))
    (output / 'CONNECTION_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                            allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], output=str(output),
                         checks=report['successful_check_count'], cpu_wall_seconds=report['cpu_wall_seconds'],
                         manual_coordinates=67, self_hidden=24, label_only_without_xy=5,
                         prepare_queues=[5, 12], new_training=0, new_model_forward=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
