"""Read-only input audit and closeout evidence for completed human labels.

This wrapper never imports a model, creates human approvals, or changes the
frozen reference/evaluator. Every output goes to the new dated closeout folder.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1'
LIFTER = ROOT / 'scripts/research/pallet_lifter_case_review_20261003_v1'
RAW = ROOT / 'data/pallet/results/pallet_lifter_case_review_20261003_v1'
REVIEW = RAW / 'review'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size, sha256=sha256(path))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def lifter_audit():
    started = time.perf_counter()
    sys.path.insert(0, str(LIFTER))
    from open_existing_annotation import Context, NativeReview
    context = Context(REVIEW/'MANIFEST.json', RAW/'LIFTER_EVALUATION_PLAN.json',
                      REVIEW/'CORNER_CONTRACT.json', REVIEW/'annotations_in_progress.json')
    native = NativeReview(context, None, REVIEW/'native_annotations',
                          passes=('primary',), native_pnp=False,
                          batch_plan=REVIEW/'SMALL_BATCH_12_V1.json')
    batch = read(REVIEW/'SMALL_BATCH_12_V1.json')
    statuses, labels, manual, missing_xy = [], Counter(), 0, []
    for fid in batch['frame_ids']:
        status = native.visibility_only_status(fid, 'primary')
        if not status['complete']:
            raise ValueError('Previously completed category labels are missing: '+fid)
        manual += status['manual_coordinate_count']
        local = Counter()
        for corner in status['corners']:
            category = corner['visibility']
            if category == 'not_direct_visible':
                category = next(key for key in ('external_occlusion', 'self_occlusion',
                    'out_of_frame') if corner[key])
            labels[category] += 1
            local[category] += 1
            if corner.get('category_label_only'):
                missing_xy.append(dict(frame_id=fid, corner_id=corner['id'],
                    visibility=corner['visibility'], x=None, y=None,
                    source='human_chat_visibility_declaration'))
        statuses.append(dict(frame_id=fid, categories=dict(local),
            categories_complete=status['complete'],
            manual_coordinate_count=status['manual_coordinate_count'],
            visible_without_coordinates=status['declared_point_ids'],
            raw_document_complete=status['raw_document_complete'],
            official_evaluation_approval=False))
    if sum(labels.values()) != 96 or manual != 67 or len(missing_xy) != 5:
        raise ValueError('Human label inventory changed; review the new inputs')

    raw_path = RAW/'raw_predictions/ALL_STORED_FRAMES.jsonl'
    original = read(ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/INTEGRATION_MANIFEST.json')
    expected = original['inputs']['lifter_raw']
    if (raw_path.stat().st_size != expected['bytes'] or sha256(raw_path) != expected['sha256']):
        raise ValueError('Fixed predictions changed; do not reuse their receipt')
    states = {name: Counter() for name in ('Base', 'N3')}
    sessions, selected = Counter(), {}
    ids = set()
    max_move = 0.0
    for line in raw_path.open(encoding='utf-8'):
        row = json.loads(line)
        fid = row['frame_id']
        if fid in ids:
            raise ValueError('Duplicate fixed prediction frame '+fid)
        ids.add(fid)
        sessions[row['session_id']] += 1
        methods = row['methods']
        before, after = methods['Base'], methods['N3']
        if before['keypoints_mask'] != after['keypoints_mask']:
            raise ValueError('Base/N3 point-valid masks differ: '+fid)
        if before['selected_object'] != after['selected_object']:
            raise ValueError('Base/N3 frozen object selection differs: '+fid)
        for name in states:
            states[name][methods[name]['pose_state']] += 1
        for i, valid in enumerate(before['keypoints_mask']):
            if valid:
                p, q = before['keypoints'][i], after['keypoints'][i]
                move = math.hypot(p[0]-q[0], p[1]-q[1])
                if not math.isfinite(move) or move > 8.00001:
                    raise ValueError('Frozen displacement cap failed: '+fid)
                max_move = max(max_move, move)
        if fid in batch['frame_ids']:
            selected[fid] = dict(frame_id=fid, selected_object=before['selected_object'],
                                 target_match_status='NOT_HUMAN_REVIEWED')
    if dict(sessions) != {'173507':3729, '174126':757, '174342':2501, '174925':1923}:
        raise ValueError('The completed 8910-frame cohort changed')
    for name in states:
        if dict(states[name]) != {'fresh':8772, 'no_pose':138}:
            raise ValueError('Output-state receipt regression: '+name)
    input_paths = [REVIEW/'MANIFEST.json', RAW/'LIFTER_EVALUATION_PLAN.json',
        REVIEW/'CORNER_CONTRACT.json', REVIEW/'SMALL_BATCH_12_V1.json',
        REVIEW/'VIEWER_DRAFT_RECOVERY.json',
        REVIEW/'USER_VISIBILITY_DECLARATIONS_20261005_V1.json', raw_path]
    outputs = dict(schema='lifter_minimum_human_closeout_audit_v1',
        generated_at=datetime.now(timezone.utc).isoformat(),
        classification=dict(frames=12, states_entered=96, states_expected=96,
            categories=dict(labels), manual_coordinates=manual,
            visible_without_coordinates_count=len(missing_xy),
            visible_without_coordinates=missing_xy, per_frame=statuses),
        formal_reference=dict(approved_frames=0, target_match_reviewed_frames=0,
            visible_corner_accuracy='x', reason='No formally approved reference or human object-match sidecar; five visible labels lack coordinates',
            annotations_in_progress_present=(REVIEW/'annotations_in_progress.json').is_file(),
            approved_export_present=(REVIEW/'LIFTER_REFERENCE_REVIEWED.json').is_file(),
            human_visibility_classification_complete=True,
            cli_promoted_draft_to_human_reviewed=False),
        inference=dict(reused_frames=len(ids), new_forward_frames=0,
            sessions=dict(sessions), methods={name:dict(counts) for name, counts in states.items()},
            point_mask_equality='PASS', selected_object_equality='PASS',
            maximum_observed_N3_move_px=max_move, fixed_move_cap_px=8,
            existing_continuity_source=bind(ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/INTEGRATION_MANIFEST.json')),
        remaining_optional_human_tasks=dict(same_target_frames=12,
            previous_exposure_confirmation_count=1,
            stationary_sessions=4,
            deferred_primary_frames=103, deferred_repeat_frames=23,
            repeat_or_reclick_classification_requested=False,
            first_reference_five_missing_coordinates_are_not_required_for_static_closeout=True),
        independent_physical_pose_reference=dict(available=False, translation='x', rotation='x'),
        source_bindings=[bind(path) for path in input_paths],
        cpu_wall_seconds=time.perf_counter()-started,
        training_runs=0, optimizer_updates=0, new_model_forward_frames=0,
        validation_status='PASS')
    write(DOC/'lifter/AUDIT.json', outputs)
    write(DOC/'lifter/PREPARED_TARGET_INVENTORY.json', dict(
        schema='machine_prepared_selected_target_inventory_v1',
        source_kind='machine_prepared_not_human_reviewed', evaluation_use=False,
        predictions_binding=bind(raw_path), batch_binding=bind(REVIEW/'SMALL_BATCH_12_V1.json'),
        records=[selected[fid] for fid in batch['frame_ids']],
        decisions_created=0, note='Inventory only. Frozen human-reference approval is still required for the actual match evaluator.'))
    print(json.dumps(dict(validation='PASS',category_labels=sum(labels.values()),
        manual_coordinates=manual, reused_frames=len(ids), new_forward_frames=0,
        seconds=outputs['cpu_wall_seconds']), ensure_ascii=False))


def finalize_closeout():
    started = time.perf_counter()
    previous_manifest=read(DOC/'INTEGRATION_MANIFEST.json') if (DOC/'INTEGRATION_MANIFEST.json').is_file() else None
    required = [DOC/'static/EXECUTION_RECEIPT.json',
        DOC/'static/STATIC_INVARIANCE_CHECK.json',
        DOC/'static/COUNTS_AND_PROVENANCE_POSTCHECK.json',
        DOC/'static/EXECUTION_COST_LEDGER.json',
        DOC/'visibility_square/EXECUTION_VALIDATION.json',
        DOC/'visibility_square/ACTUAL_CPU_COST_LEDGER.json',
        DOC/'lifter/AUDIT.json', DOC/'REUSED_RESULT_AUDIT.json',
        DOC/'paper_patch/CLOSEOUT_VALIDATION.json',
        DOC/'paper_patch/EXECUTION_RECEIPT.json',
        DOC/'paper_patch/PAPER_CELL_MAP.json', DOC/'paper_patch/PAPER_GAP_MATRIX.md',
        DOC/'final_review/INDEPENDENT_PAPER_AUDIT.json',
        DOC/'final_review/INDEPENDENT_REFERENCE_AUDIT.json']
    if any(not path.is_file() for path in required):
        raise FileNotFoundError('Component output still pending: '+', '.join(
            str(path) for path in required if not path.is_file()))
    original = read(DOC/'START_STATE.json')
    protected = []
    for relative, expected in original['files'].items():
        path = ROOT/relative if not Path(relative).is_absolute() else Path(relative)
        actual = bind(path)
        protected.append(dict(path=relative,
            unchanged=actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes']))
    if not all(row['unchanged'] for row in protected):
        raise ValueError('An original input or earlier paper copy changed')
    preservation = dict(status='PASS', checked_files=len(protected), all_unchanged=True,
        files=protected, git_head_unchanged=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==original['git_head'],
        original_tracked_changes_unchanged=subprocess.check_output(
            ['git','status','--short','--untracked-files=no'],cwd=ROOT,text=True).splitlines()==original['tracked_changes'])
    if not preservation['git_head_unchanged'] or not preservation['original_tracked_changes_unchanged']:
        raise ValueError('Repository basis or unrelated tracked files changed')
    write(DOC/'PRESERVATION_CHECK.json', preservation)
    static = read(DOC/'static/STATIC_REAGGREGATION.json')
    invariant = read(DOC/'static/STATIC_INVARIANCE_CHECK.json')
    labels = read(DOC/'static/LABEL_PROVENANCE_AUDIT.json')
    lifter = read(DOC/'lifter/AUDIT.json')
    reused = read(DOC/'REUSED_RESULT_AUDIT.json')
    paper = read(DOC/'paper_patch/CLOSEOUT_VALIDATION.json')
    visibility = read(DOC/'visibility_square/EXECUTION_VALIDATION.json')
    if invariant['status']!='PASS' or lifter['validation_status']!='PASS' or reused['failed']:
        raise ValueError('A required numeric audit failed')
    if paper.get('status') not in ('PASS','VERIFIED_COMPLETE') or visibility.get('status') not in ('PASS','VERIFIED_COMPLETE'):
        raise ValueError('Paper or visibility validation did not pass')
    for name in ('INDEPENDENT_PAPER_AUDIT.json','INDEPENDENT_REFERENCE_AUDIT.json'):
        if read(DOC/'final_review'/name).get('status') not in ('PASS','PASS_WITH_DOCUMENTED_LIMITATIONS'):
            raise ValueError('Independent final audit is not ready: '+name)
    cell_map=read(DOC/'paper_patch/PAPER_CELL_MAP.json')
    insertion_checks={}
    def inserted_status(target):
        cells=[cell for cell in cell_map['cells'] if cell['target']==target]
        if not cells or any(cell['status'] not in ('INSERTED_IN_COPY','X_REFERENCE_OR_CONTRACT_MISSING') for cell in cells):
            raise ValueError('Table insertion was not verified: '+target)
        status='PARTIAL_INSERTED_IN_COPY' if any(cell['status'].startswith('X_') for cell in cells) else 'INSERTED_IN_COPY'
        insertion_checks[target]=dict(status=status, mapped_cells=len(cells))
        return status
    grade_status=inserted_status('tab:occlusion_results')
    visibility_status=inserted_status('sup:visibility')
    square_status=inserted_status('tab:square_results')
    prior_prefix='_docs/experiments/pallet_combined_closeout_20261003_v1/paper_updated/'
    for relative in ('tables/cost_results.tex','tables/backbone_results.tex'):
        entry=original['files'][prior_prefix+relative]
        path=DOC/'paper_updated'/relative
        if sha256(path)!=entry['sha256']:
            raise ValueError('An inherited table changed without new evidence: '+relative)
        insertion_checks[relative]=dict(status='REUSED_INSERTED_IN_COPY', source_sha256=entry['sha256'])
    write(DOC/'MANUSCRIPT_LINK_CHECK.json',dict(status='PASS',
        checks=insertion_checks, actual_numeric_cells=paper['replaced_numeric_cells_main']+
        paper['replaced_numeric_cells_supplement'],
        remaining_detailed_fragments_status=paper['detailed_fragments']))
    phases = []
    for path in (DOC/'static/EXECUTION_COST_LEDGER.json',
                 DOC/'visibility_square/EXECUTION_VALIDATION.json',
                 DOC/'visibility_square/ACTUAL_CPU_COST_LEDGER.json',
                 DOC/'lifter/AUDIT.json',DOC/'REUSED_RESULT_AUDIT.json',
                 DOC/'paper_patch/CLOSEOUT_VALIDATION.json',
                 DOC/'paper_patch/EXECUTION_RECEIPT.json'):
        phases.append(dict(source=bind(path), receipt=read(path)))
    tasks = [
        ('latest static grades and composition',grade_status,'DEV319 153/92/74; GREEN119 3/85/31; criterion remains explicit in source metadata'),
        ('three backbone and ablation regrouping','VERIFIED_COMPLETE','fixed per-seed predictions; all319 values exactly unchanged'),
        ('latest visibility states and metrics',visibility_status,'3030 current+71 prior reference states; no reference-coordinate edits'),
        ('square 602/600 evaluation',square_status,'same mode and denominator for every method; independent6D x'),
        ('other refiners319 and student128','REUSED_VERIFIED_COMPLETE','36 checkpoint/protocol/raw source bindings rechecked; panels kept separate'),
        ('existing runtime measurements','REUSED_INSERTED_IN_COPY','fixed26/warmup20/repeat5; no GPU benchmark rerun'),
        ('lifter fixed8910 inference and continuity','REUSED_INSERTED_IN_COPY','8772fresh/138no_pose per method; masks/selection/cap rechecked'),
        ('lifter12 visibility labels','VERIFIED_COMPLETE','96labels=67manualvisible+5visiblewithoutXY+24selfhidden'),
        ('lifter visible corner accuracy','WAITING_HUMAN_REFERENCE_AND_MATCH','officialapproved0; coordinates absent at5visible points; leave x without more annotation requests'),
        ('lifter stationary variation','WAITING_OPTIONAL_HUMAN','0reviewed intervals; x retained'),
        ('independent square/lifter physical T/R','BLOCKED_REFERENCE','no independent reference; x retained'),
        ('additional detailed supporting tables','AVAILABLE_AS_SUPPORTING_EVIDENCE','required three-backbone grade claims inserted; optional detailed tables fully formatted'),
        ('PDF/PPT,control,training,push,upload','OUT_OF_SCOPE_USER','all0 in this closeout'),
    ]
    with (DOC/'TASK_STATUS.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['task','status','evidence_or_reason']);writer.writerows(tasks)
    user_actions = """# 사용자 작업을 최소화한 마감 경로

완료한 정적 분류·코너 상태·리프터12장 입력을 그대로 사용했다. 같은 이미지를 다시 분류하거나 기존67점을 다시 클릭할 필요가 없다. 이번 정적 결과와 리프터 연속 출력에 근거한 원고 마감에는 추가 코너 클릭을 요구하지 않는다.

기존 없음/중간/어려움 분류는 사용자 입력 등급으로 설명했다. 외부 가림·전체 난이도 중 어느 기준인지 추정해 확정하지 않았으며, 추가 기준 답변이나 재분류 없이 보조 분석으로 사용할 수 있게 했다. 마지막에는 [최종 확인 화면](final_review/FINAL_REVIEW.html)의 결과와 제한을 한 번 확인하면 된다. 숫자를 맞추기 위한 재분류는 없다.

## 추가 정확도를 채우고 싶을 때만 하는 작업

- 리프터 코너 정확도: 기존67점은 재사용한다. 보임 여부만 선언한5점에는 실제 참조 좌표가 없으므로 이 칸까지 계산하려면 그5점 좌표, 실제 노출 이력 확인, 고정 선택 객체의 대상 대응 최대12장 확인이 필요하다. 현재는 x를 유지하며 이 추가 작업을 강제하지 않는다.
- 정지 잡음:4세션의 실제 정지 구간이 사람에 의해 확인돼야 한다. 현재는 x를 유지한다.
- 나머지103주프레임·23반복프레임은 사용자 부담 때문에 보류한 범위이며 완료로 바꾸지 않았다.
- 독립 물리 T/R 정답은 클릭이나 CLI 계산으로 대신할 수 없다. 독립 참조 확보 전에는 x다.

수정된 로컬 메뉴는 완료한 정적·PnP·가시성 버튼을 비활성화한다. 다시 저장하라는 B 안내도 제거했다.

![완료 입력을 다시 요구하지 않는 실제 화면](minimal_user_menu.png)

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_combined_closeout_20261003_v1/open_human_review.py
```

이 메뉴를 열기만 해서는 사람 검수 완료·노출 이력·대상 일치가 생성되지 않는다. 원본 주석과 frozen evaluator는 보존된다.
"""
    (DOC/'USER_ACTIONS_KO.md').write_text(user_actions,encoding='utf-8')
    table=['| 기반 | 코너 중앙값 px Base → N3 | T 중앙값 cm | R 중앙값 ° | 자세 산출 장수 |',
           '|---|---:|---:|---:|---:|']
    for name, shown in [('yolo','YOLO'),('dope','DOPE'),('resnet18','ResNet-18')]:
        before=static['backbones'][name]['Base']['result']['all']
        after=static['backbones'][name]['N3']['result']['all']
        table.append(f"| {shown} | {before['corner_median_px']:.3f} → {after['corner_median_px']:.3f} | "
            f"{before['translation_median_cm']:.3f} → {after['translation_median_cm']:.3f} | "
            f"{before['rotation_median_deg']:.3f} → {after['rotation_median_deg']:.3f} | "
            f"{before['pose_available_frames']} /319 → {after['pose_available_frames']} /319 |")
    summary_figure=list((DOC/'paper_updated/figures').glob('*review*summary*.png'))
    if not summary_figure:
        summary_figure=list((DOC/'paper_updated/figures').glob('*closeout*.png'))
    image_section=''
    for path in summary_figure[:2]:
        image_section+=f'![완료한 입력과 평가 범위]({path.relative_to(DOC).as_posix()})\n\n'
    image_section+='![기존 고정8910프레임 실제 예측](paper_updated/figures/lifter_prediction_timeseries.png)\n'
    image_section+='\n기존 원고의 실제 영상 예시는 [예시1](paper_updated/figures/example_1_frame.png), [예시2](paper_updated/figures/example_2_frame.png), [예시3](paper_updated/figures/example_3_frame.png)에 보존했다. 이번 재집계로 새로 골라낸 성능 좋은 사례라고 기록하지 않았다.\n'
    ledger=read(DOC/'static/EXECUTION_COST_LEDGER.json')
    paper_cost=read(DOC/'paper_patch/EXECUTION_RECEIPT.json')
    report = f"""# 추가 사람 작업을 줄인 최신 원고 마감 결과

정적 입력을 재사용해 재집계·검산했고 실제 LaTeX 복사본의 표·본문·그림을 갱신했다. 리프터는 이미 완료된 고정8910장 예측을 재사용했다. 새 학습·optimizer update·추론은 모두0회다. 원고 원본과 이전 결과는 보존했다.

실제 계약은 세 Base가 RGB 추정기이고, 학습된 N3 후단이 이미지 특징·초기 코너·박스·물리 치수를 함께 받는 구조다. DOPE·ResNet N3의 기존 각3seed 학습 결과를 재사용했으며, 모든 Base에 직접 치수를 넣어 새로 학습했다고 기록하지 않았다. ResNet Base는 검증된10-epoch CONSTANT-fold RGB 모델이다.

## 실제 계산과 원고 반영

- 정적 DEV319 등급153/92/74, GREEN119 등급3/85/31을 그대로 연결했다. 새 프레임 등급435건과 기존 승인3건을 사용했다. 등급 의미는 `{labels['classification_semantics_status']}`이며 답변 없는 기준을 CLI가 확정하지 않았다.
- 정적 참조 상태3101개=이번3030+기존71. 참조 없는 슬롯53/350은 가림으로 합치지 않았다. 기준 좌표를 바꾸지 않고 세 기반·seed·상태별 지표를 계산했다.
- 세 기반과 YOLO 절제N0/N1/P/N2/N3의2D/6D, D/L/PoseFix319, 학생128을 분리해 재집계했다. 세션 짝지은 분석과 기존 비용표는 검산 완료본을 재사용했다.
- 정사각형119는602점/600점 모드를 나란히 유지했다. 과거150장은 별도 자료·결과로 분리했다. 독립6D 참조가 없는 정확도는x다.
- 리프터12장 가시성96개는 완료이며 수동 좌표67점·좌표 없는 보임5점·자체가림24점을 구분했다. 공식 참조승인과 대상 대응은0건이므로 코너 정확도x를 유지했다.
- 실제 삽입된 셀과 별도 교체 조각만 준비된 셀은 [셀 출처표](paper_patch/PAPER_CELL_MAP.json)와 [미완료 행렬](paper_patch/PAPER_GAP_MATRIX.md)에 각각 기록했다.
- 본문 {paper['replaced_numeric_cells_main']}개·보충 {paper['replaced_numeric_cells_supplement']}개 숫자 셀을 실제 교체했고, 근거 없는 {paper['x_cells_preserved']}개 x 셀은 유지했다. 상세 재질×등급 행은 별도 CSV/LaTeX 조각으로 제공하며 모두 본문에 삽입했다고 주장하지 않는다.
- 최종 독립 검토에서 발견한 결론·재질 설명의 오래된 문구를 정정했다. 본문에서 설명한 세 기반×세 등급은 보충 표9행에 실제 넣었고, 추가 세부 행은 완성된 선택적 보조 자료다. 사용자가 직접 표를 편집하거나 새로 삽입할 필요가 없다.

## 전체319장 결과

{chr(10).join(table)}

N3는 각seed 지표의 평균이다. 서로 다른seed 예측을 합쳐 새 중앙값을 만든 값이 아니다. 코너 중앙값은 유효 예측의 조건부 값이며 PCK는 참조2499점의 전체 분모를 유지한다. DOPE의 자세 실패109장도 유지했다.

중앙값 개선과 어려운 사례의 개선은 다르다. DOPE의 코너P90은51.282→53.570px, 회전P90은80.583→82.173°로 악화된다. ResNet의 이동P90은97.545→100.304cm로 악화된다. YOLO의 사용자 입력 심함 집단에서도 이동 중앙값16.047→17.481cm로 악화된다. 이 값들을 숨기거나 좋은seed만 선택하지 않았다.

{image_section}

## 검산과 원본 보존

- 전체 지표 불변 검산 `{invariant['check_count']}/{invariant['check_count']} PASS`; 상태 집단 재분류가 전체2D/6D 수치를 바꾸지 않았다.
- 기존 ResNet10-epoch CONSTANT-fold checkpoint·protocol·receipt와 대조군·학생 원시 결과의 해시 `{reused['passed']}/{reused['passed']} PASS`.
- 원본/이전복사본 입력 `{len(protected)}/{len(protected)}개` 보존. HEAD는 `{original['git_head']}`이며 기존 수정 파일도 유지했다.
- 리프터8910행의 Base/N3 결측 마스크·고정 선택 객체·8px 이동 상한 검산PASS, 새forward0.
- 실제 patch 적용·LaTeX 참조·그림·표 연결 검증은 [CLOSEOUT_VALIDATION](paper_patch/CLOSEOUT_VALIDATION.json)에 있다. PDF를 컴파일하지 않았으므로 최종 페이지 수는NA다.

## 남은 x와 사람 작업

이번 정적 결과와 연속 출력 원고 마감에는 추가 분류·코너 클릭·식별자 입력·등급 기준 답변을 요구하지 않는다. 기존3등급은 사용자 입력 등급으로 제한하여 보조 분석으로 사용했다. 마지막에는 [최종 확인 화면](final_review/FINAL_REVIEW.html)의 결과·근거 제한·원고를 한 번 확인하면 된다. 이 문서 확인을 주석 승인이나 새 정확도 결과로 바꾸지 않는다.

리프터 코너 정확도는 공식 참조·대상 대응·일부 실제 좌표가 없어x, 정지 구간 변동은 실제 정지 확인이 없어x, 독립 물리T/R은 별도 참조가 없어x다. 이를 없애는 추가 작업은 [최소 사용자 작업](USER_ACTIONS_KO.md)에 분리했다.103주프레임·23반복프레임을 완료라고 기록하지 않았다.

## 실제 실행과 비용

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_static_registry_review_20261003_v1.reaggregate_native_closeout_20261006
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_combined_closeout_20261003_v1/closeout_latest.py lifter-audit
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_combined_closeout_20261003_v1/closeout_latest.py finalize
```

정적 재집계는 출처 연결 정정 전후2회 실행했으며 합계{ledger['measured_reaggregation_wall_seconds_total']:.3f}초다. 가시성/정사각형 재집계2회는9.273초, 의미 검증의 계측된wall0.150초이며 각 실행을 [가시성 비용 ledger](visibility_square/ACTUAL_CPU_COST_LEDGER.json)에 보존했다. 리프터 재사용 감사는{lifter['cpu_wall_seconds']:.3f}초, 기존 결과36파일 해시 감사는{reused['cpu_wall_seconds']:.3f}초다. 원고 검증 명령·실제 시간은receipt에 연결했다. 병행한 각CPU전용단계의벽시계를전체작업시간이나CPU사용초로합치지않았다. CPU사용초는NA다. 기존 GPU 측정값은 새 측정값으로 복사하지 않았다. 이번 GPU 학습·추론·재측정은0이다.

- [가시성·정사각형 계산](visibility_square/EXECUTION_VALIDATION.json)
- [원고 처리 {len(paper_cost['phases'])}회 합계 {paper_cost['total_executed_script_wall_seconds']:.3f}초 실행 기록](paper_patch/EXECUTION_RECEIPT.json)
- [직접 독립 검산](final_review/INDEPENDENT_PAPER_AUDIT_KO.md)
- [정적 결과와 seed별 CSV](static/DEV319_HEADLINE_AND_SEED.csv)
- [원고 복사본](paper_updated/main.tex), [본문 Markdown](paper_updated/manuscript_ko.md)
- [통합 patch](paper_patch/INTEGRATED.patch)
- [전체 실행 출처](INTEGRATION_MANIFEST.json)

PDF/PPT 생성, 리프터 제어, 새 학습, push와 외부 업로드는 실행하지 않았다.
"""
    (DOC/'FINAL_REPORT_KO.md').write_text(report,encoding='utf-8')
    elapsed=time.perf_counter()-started
    history=[]
    if previous_manifest:
        history=previous_manifest.get('finalize_invocations', [dict(
            generated_at=previous_manifest['generated_at'],
            wall_seconds=previous_manifest['finalize_cpu_wall_seconds'],
            reason='Initial final packaging before the paper execution-cost receipt became available')])
    history.append(dict(generated_at=datetime.now(timezone.utc).isoformat(),
        wall_seconds=elapsed, reason='Final packaging including every available component receipt'))
    manifest=dict(schema='pallet_latest_closeout_manifest_v1',
        generated_at=datetime.now(timezone.utc).isoformat(),
        status='CLI_CLOSEOUT_COMPLETE_WITH_DECLARED_REFERENCE_GAPS',
        git_head=original['git_head'], input_and_prior_paper_preservation=bind(DOC/'PRESERVATION_CHECK.json'),
        components=[bind(path) for path in required], component_receipts=phases,
        manuscript_link_check=bind(DOC/'MANUSCRIPT_LINK_CHECK.json'),
        updated_paper=str((DOC/'paper_updated').relative_to(ROOT)),
        gap_matrix=str((DOC/'paper_patch/PAPER_GAP_MATRIX.md').relative_to(ROOT)),
        human_classification_repeated=False, human_approvals_fabricated=False,
        new_training=0, optimizer_updates=0, new_model_forward_frames=0,
        pdf_compilation=0, external_upload=0, git_push=0, actual_lifter_control=0,
        finalize_cpu_wall_seconds=elapsed,
        finalize_invocations=history,
        elapsed_since_state_capture_seconds=time.monotonic()-original['start_monotonic'],
        process_cpu_seconds='NA_NOT_INSTRUMENTED',
        actual_cost_note='Use each component receipt; static ledger includes both runs. Concurrent phase seconds are not overall wall time.')
    write(DOC/'INTEGRATION_MANIFEST.json',manifest)
    print(json.dumps(dict(status=manifest['status'], protected_files=len(protected),
        numeric_regression_checks=invariant['check_count'], new_training=0,
        new_model_forward_frames=0, finalize_seconds=elapsed), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['lifter-audit','finalize'])
    args = parser.parse_args()
    if args.phase == 'lifter-audit':
        lifter_audit()
    else:
        finalize_closeout()


if __name__ == '__main__':
    main()
