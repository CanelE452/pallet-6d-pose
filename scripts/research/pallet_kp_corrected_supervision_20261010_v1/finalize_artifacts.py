"""Assemble documentation/execution ledger and bind public artifacts; no fits.

The manifest deliberately excludes itself and the two receipts that depend on
it. Normal Git publication is performed separately after independent review.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_kp_corrected_supervision_20261010_v1'
DOC = ROOT / '_docs/experiments' / NAME
CODE = ROOT / 'scripts/research' / NAME


def read(name):
    return json.loads((DOC / name).read_text())


def bind(path):
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(name, value):
    (DOC / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n')


def documents():
    training = read('TRAINING_COMPLETION.json')
    inference = read('SUBSET_INFER_ADAPTER_RECEIPT.json')
    geometry = read('LEARNED_POSE_EXECUTION.json')
    stress = read('REAL_STRESS_EXECUTION.json')
    runtime = read('RUNTIME.json')
    audit = read('REVIEW_DEVELOPMENT_AUDIT.json')
    assert all(item['complete'] for item in (training, inference, geometry, stress, runtime, audit))
    assert training['formal_updates'] == 9000 and inference['frames'] == 245
    assert geometry['rows'] == 1470 and stress['rows'] == 6860
    assert runtime['execution']['pipeline_calls_complete'] == 600
    sources = ['TRAINING_COMPLETION.json', 'SUBSET_INFER_ADAPTER_RECEIPT.json',
               'LEARNED_POSE_EXECUTION.json', 'REAL_STRESS_EXECUTION.json',
               'RUNTIME.json', 'REVIEW_DEVELOPMENT_AUDIT.json', 'REVIEW_FINAL_CODE_BINDING.json']
    write('BUILD_LEDGER.json', dict(
        schema='corrected_supervision_completed_execution_ledger_v1', complete=True,
        scope=dict(latest_user_request='319로하지말 고 쉬움 중간만 해 가림 어려움하지말고',
                   selected=245, easy=153, medium=92, excluded_hard=74,
                   new_hard_frame_paths=0, prior319_experiments_preserved=True),
        authorization=bind(DOC/'APPROVED_ADDITIONAL_9000.json'),
        scope_protocol=bind(DOC/'SUBSET_PROTOCOL.json'),
        formal_training=dict(updates=9000, updates_each=3000, arms=3, batch=16,
                             seed=1, throwaway_updates=0, image_exposures=144000,
                             source_probe_head_calls=training['source_probe_head_calls'],
                             source_probe_exposures=training['source_probe_image_exposures'],
                             head_calls=training['total_head_forward_calls'],
                             wall_seconds=training['wall_seconds'],
                             source_selection_or_retuning=False,
                             cumulative_original_plus_corrected_formal_updates=18000,
                             historical_throwaway_not_reclassified_as_formal=True),
        real_observations=dict(frames=245, detector=245, learned_head_calls=735,
                               feature_initial_pose_calls=245,
                               primitive_counts=inference['primitive_counts'],
                               adapter_wall_seconds=inference['wall_seconds'],
                               benchmark=False, GT_free_sealed=True),
        real_geometry=dict(paths=1470, point_paths=1225, point_line_paths=245,
                           fits_and_optimizer_ledger=geometry['fit_counts'],
                           initial_Base_pose_reused=True, GT_free_sealed=True,
                           wall_seconds=None,
                           wall_status=geometry['seconds_status']),
        scoring_only_resume=dict(guard_failure_preserved=True, exact_image_session_bijection=319,
                                 selected_posthoc_frames=245, paths_scored=1470,
                                 detector=0, heads=0, PnP=0, optimizers=0,
                                 wall_seconds=geometry['scoring_resume_wall_seconds'],
                                 geometry_SHA_unchanged=True),
        real_mask_stress=dict(paths=6860, frames=245, coordinates=2, families=2,
                              conditions=7, shared_coordinate_banks=490,
                              actual_OpenCV_counts=stress['actual_OpenCV_counts'],
                              counts=stress['counts'], wall_seconds=stress['wall_seconds'],
                              detector=0, heads=0, training=0, oracle_only=True,
                              nontransformable=1799, nontransformable_not_pose_failure=True),
        whole_path_benchmark=dict(pipeline_calls=600, warmup=80, measured=520,
                                  arms=4, measured_each=130, frames=26, sessions=13,
                                  model_forwards=runtime['model_forwards'],
                                  execution=runtime['execution'], parity_passes=600,
                                  timing_cached_coordinates=False,
                                  batch_latency_summation=False,
                                  actual_latency_rows='RUNTIME_ROWS.jsonl.gz',
                                  elapsed_wall_not_mean_latency=True),
        publication_math=dict(development_numeric_invocations=3,
                              semantic_negative_invocations=2, output_guard_only_invocations=2,
                              final_strict_numeric_invocation_index=6,
                              final_strict_receipt='REVIEW_CHECKS.json',
                              final_status_not_inferred_before_receipt=True,
                              second_development_receipt_ENOSPC=True,
                              all_math_model_PnP_optimizer_ray_calls=0),
        generated_assets=dict(new_RGB=0, new_real_capture=0, manual_annotations=0,
                              new_N3_updates=0, extra_seeds=0, physical_GT_created=0,
                              numerical_figures=9, actual_case_panels=12,
                              cases_frozen_before_accuracy=True),
        source_bindings=[bind(DOC/n) for n in sources],
        unavailable_measurements=dict(failed_geometry_process_wall=None,
                                      point_line_residual_callback_count=None,
                                      independent_physical_pose_GT=None),
        protected_prior_files=331,
        assembly_actual_execution=dict(detector=0, head=0, training=0, PnP=0,
                                       optimizer=0, rays=0, private_GT=0),
        boundary='This ledger covers the corrected run and its declared adapters. Prior experiments/repair diagnostics retain their own immutable ledgers; their rows are not counted as fresh executions here.'))
    (DOC/'README.md').write_text('''가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**이번 고정 실험에서는 아니요.** 최신 요청에 따라 쉬움 153장·중간 92장만 사용했습니다. 어려움 74장은 새 평가에서 제외했습니다. IMAGE_ROLE은 위치 15.20239cm·회전 14.56122°, 고정 N3→cornerSubPix는 9.75455cm·10.91484°였습니다. 기존 GEOMETRIC_PROXY 참조의 결과이며 독립 실측 자세 인증은 아닙니다.

읽기 시작할 파일은 [상세 결과 보고서](RESULT_KO.md)입니다. 24방법의 전체 운용 결과, 쉬움/중간 분리, 평균·분산·SD·중앙값·P90, 새 자세/기본 반환/실패, 마스크 오판과 자세 성능, 직접 가시점 손상과 숨은 점 재투영, source 학습과 전이 실패를 설명합니다. 보고서에는 **실제 영상 12패널과 수치 그림 9개**가 있습니다. 사전에 고른 사례이므로 새 모델의 성공 사례만 골라 보여 주지 않았습니다.

검토자는 [검산 방법](REPRODUCE.md)의 표준 Python 명령으로 공개 원행을 다시 계산할 수 있습니다. 핵심 자료는 다음과 같습니다.

| 검토할 내용 | 파일 |
| --- | --- |
| 고정된 포함/제외 영상, 분할, 시간 패널, 사례 선정 | [COHORT.json](COHORT.json), [SUBSET_PROTOCOL.json](SUBSET_PROTOCOL.json), [RUNTIME_PANEL.json](RUNTIME_PANEL.json), [VISUAL_CASE_PROTOCOL.json](VISUAL_CASE_PROTOCOL.json) |
| 실제 추가 3×3000 학습과 같은 초기화/배치 순서 | [FORMAL_UPDATE_ROWS.jsonl.gz](FORMAL_UPDATE_ROWS.jsonl.gz), [TRAINING_COMPLETION.json](TRAINING_COMPLETION.json), [CHECKPOINT_METADATA.json](CHECKPOINT_METADATA.json) |
| 정답을 읽기 전 관측 735개와 geometry 1470개 | [LEARNED_OBSERVATIONS.jsonl.gz](LEARNED_OBSERVATIONS.jsonl.gz), [LEARNED_GEOMETRY_SEALED.jsonl.gz](LEARNED_GEOMETRY_SEALED.jsonl.gz) |
| 실제 채점 원행과 대응점 검산 | [LEARNED_PREDICTIONS.jsonl.gz](LEARNED_PREDICTIONS.jsonl.gz), [POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz](POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz) |
| 같은 Base/N3 좌표의 필수 무학습 12대조 | [HISTORICAL_FILTERED_ROWS.jsonl.gz](HISTORICAL_FILTERED_ROWS.jsonl.gz), [METRICS.json](METRICS.json) |
| 실제 1·2점/동시오류 6860경로와 oracle 한계 | [REAL_STRESS_ROWS.jsonl.gz](REAL_STRESS_ROWS.jsonl.gz), [REAL_STRESS_STATISTICS.json](REAL_STRESS_STATISTICS.json) |
| 새 전체 경로 600회, 캐시 재생 없는 시간 측정 | [RUNTIME_ROWS.jsonl.gz](RUNTIME_ROWS.jsonl.gz), [RUNTIME.json](RUNTIME.json) |
| 실패와 채점만 재개한 기록 | [EVALUATION_INTERRUPTION.json](EVALUATION_INTERRUPTION.json), [SCORING_RESUME_PROTOCOL.json](SCORING_RESUME_PROTOCOL.json) |
| 실제 실행량·독립 검산·변조 거부 검사 | [BUILD_LEDGER.json](BUILD_LEDGER.json), [REVIEW_CHECKS.json](REVIEW_CHECKS.json), [REVIEW_VALIDATION_TESTS.json](REVIEW_VALIDATION_TESTS.json) |
| 모든 새 공개 파일 SHA 및 Git 게시 증거 | [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json), [PUBLICATION.json](PUBLICATION.json) |

이 폴더는 이전 319장 결과와 감독 수정의 실패 기록을 덮어쓰지 않은 후속 실험입니다. 원본 보호 331파일의 목록은 [PRIOR_PUBLICATION_BINDINGS.json](PRIOR_PUBLICATION_BINDINGS.json)에 있습니다. 배포 방법과 사람이 정답으로 점을 고르는 oracle 진단을 구분했습니다. 새 자세가 나오면 예측 자기 가림으로 제외한 코너를 최종 재투영 좌표로 교체했고, 이 재투영점으로 다시 PnP를 수행하지 않았습니다.
''', encoding='utf-8')
    (DOC/'REPRODUCE.md').write_text('''공개 파일만으로 원행의 수치와 내부 일관성을 검산할 수 있습니다. Python 3.9 이상 표준 라이브러리를 사용하며 Torch/OpenCV/모델/비공개 영상·가중치·GT는 필요하지 않습니다. 새 학습이나 자세 계산, 시간 측정을 시작하지 않습니다.

저장소를 해당 연구 브랜치의 최종 게시 commit으로 checkout한 뒤 루트에서 실행합니다. sparse checkout을 사용한다면 이 새 폴더뿐 아니라 `PRIOR_PUBLICATION_BINDINGS.json`에 지정한 기존 331파일도 있어야 합니다. 전체 clone을 사용하면 됩니다.

```bash
python3 -I -S scripts/research/pallet_kp_corrected_supervision_20261010_v1/review_verify.py --require-manifest
```

이 명령은 이 폴더의 전용 `REVIEW_CHECKS.json`만 새 로컬 검산 영수증으로 갱신합니다. Git의 게시 영수증을 보존하고 싶다면 저장소 밖의 새 파일로 출력합니다.

```bash
python3 -I -S scripts/research/pallet_kp_corrected_supervision_20261010_v1/review_verify.py --require-manifest --output /dev/shm/pallet-review-independent.json
```

15개 그룹의 PASS와 전체 `complete: true`가 완료 조건입니다. 실패하면 첫 실패 그룹과 원인이 저장됩니다. 다른 저장소 파일을 `--output`으로 덮어쓰기 또는 심볼릭 링크를 통해 덮어쓰기는 거부합니다. 공개 행에서 직접 확인하는 내용은 다음과 같습니다.

1. 기존 331파일의 byte/SHA 보존, 승인·학습 protocol·체크포인트 metadata의 연결.
2. 실제 9000행의 3×3000 업데이트, batch16, 같은 초기화/배치 순서·노출 수. 체크포인트 tensor 자체는 공개하지 않아 그 바이트를 직접 열어 검증하지 않습니다.
3. 쉬움153/중간92의 고정245 ID, 제외74 ID, 26시간 패널·6사례의 적합성.
4. 735관측의 저장된66-way logits를 실제 decoder 정책으로 재계산, 선택한 관측 ID/수 연결. 학습 head를 다시 실행하지 않습니다.
5. 봉인된1470 geometry와 채점 원행의 연결, 기본 반환 상태, 최종 fit의 H 제외, 최종 R,t로 H 좌표 재투영, per-solve 호출량. 새 PnP를 하지 않습니다.
6. 채점 ID 불일치 중단과 exact image+session mapping의 일대일성, 채점만 이어간 증거. 어려움 영상의 좌표/GT 채점을 추가하지 않습니다.
7. 24방법·41대응 대조·쉬움/중간/전체의 전체 운용·새 자세·공통집합, 평균·ddof1 분산/SD·linear 중앙값/P90·최대값.
8. 보존된10000×13 bootstrap multiplicity로 paired95% 구간과 성공 주장, 새 seed나 draw 없음.
9. 대응점 정확도/2D·3D 배치/최종 inlier, 가림 오판과 자세 개선·악화의 분리, 직접 가시점 손상과 실제 재투영 SELF의 비교.
10. 6860실사 오판 경로의2oracle×2좌표×7조건×245분모, 제거/잔류 ID, shared490banks, 변형불가를 실패로 처리하지 않는 상태, 전체 운용과 변형 가능 집합의 통계.
11. 실제 source 곡선, 600시간 원행의 warmup/측정 반복·실행량·parity와20분포, 최종 manifest의 모든 공개 입력 SHA.

[REVIEW_VALIDATION_TESTS.json](REVIEW_VALIDATION_TESTS.json)은 공개 파일만 복사하고 site-packages를 끈 격리 환경에서 수행한 두 의미적 오류 검사와 두 출력 보호 검사입니다. 전체/combined 위치 평균을 함께1cm 바꾼 입력은 평균 재계산에서 거부됐고, seal과 채점 행의 H 표시 좌표를 함께1px 바꾼 입력도 R,t 재투영에서 거부됐습니다. 해당 검사에서 실제 사용한 코드 바이트는 `review_verify_control.py`로 보존했습니다. 마지막 verifier는 scope/상태 연결을 강화했으며 최종 실행 코드는 게시된 SHA와 [REVIEW_CHECKS.json](REVIEW_CHECKS.json)에서 확인합니다. 검산 개발 중 저장 실패와 실행 횟수도 [REVIEW_DEVELOPMENT_AUDIT.json](REVIEW_DEVELOPMENT_AUDIT.json)에 남았습니다.

공개 검산으로 인증하지 못하는 범위도 분명합니다. 저장된 GPU 실행 증거와 타이밍 행의 수학을 확인하지만 과거 GPU 동작 자체, transient 자원 상태, 비공개 checkpoint tensor나 cached scorer를 독립적으로 다시 실행하지 않습니다. 참조는 기존 GEOMETRIC_PROXY이며 물리 실측 GT가 아닙니다. DEV와 source-test는 이미 진단에 사용한 자료로 새 holdout이 아닙니다.

전체 실행을 새 환경에서 재현하려면 원본 SHA가 일치하는 RGB·Base/N3 가중치·기존 registry/K·source P0 features·실제 USD/mask/depth·fixed initial tensor/order·수정 supervision·cached proxy reference와 원본 dependency 환경이 별도로 필요합니다. 이 자료를 다른 RGB/박스 mask로 대체하면 같은 실험이 아닙니다. 데이터/전체 가중치/mesh를 Git에 새로 올리지 않았습니다. 비공개 의존성 이름과 SHA는 각 protocol, completion, adapter receipt에 있습니다. 9000회 승인 claim은 이미 소비한 한 번의 실행이며 이 README가 새 학습을 자동 승인하거나 실행하지 않습니다.

실행 코드는 새 폴더의 `subset_downstream.py`, `scoring_resume.py`, `subset_stress.py`, `subset_runtime.py`입니다. 최초 geometry driver는 ID guard 실패를 포함한 실제 바이트로 보존했습니다. 채점 mapping 수정은 별도 코드이므로 실패를 없앤 새 driver로 교체하지 않았습니다. 통계·그림·보고서 생성기는 `statistics.py`, `supplementary_statistics.py`, `stress_statistics.py`, `visuals.py`, `build_report.py`입니다. 무거운 실행은 완료된 결과를 가정하며 import만으로 학습/추론을 수행하지 않습니다.

게시 SHA는 [PUBLICATION.json](PUBLICATION.json)의 payload commit을 기준으로 검토할 수 있습니다. manifest/검산 영수증/publication 영수증은 자기 자신의 SHA를 포함하는 순환을 만들지 않습니다. 최종 receipt commit은 원격 research 브랜치와 실제로 비교한 뒤 사용자에게 전달합니다. main push/자동 merge/force push는 수행하지 않습니다.
''', encoding='utf-8')
    print('DOCUMENTS_LEDGER_ASSEMBLED_NO_SCIENTIFIC_EXECUTION')


def manifest():
    excluded = {'REVIEW_MANIFEST.json', 'REVIEW_CHECKS.json', 'PUBLICATION.json'}
    files = sorted(p for base in (DOC,CODE) for p in base.rglob('*')
                   if p.is_file() and p.name not in excluded
                   and '__pycache__' not in p.parts and p.suffix != '.pyc')
    assert len(files) > 70 and all(not p.is_symlink() for p in files)
    assert not (DOC/'REVIEW_CHECKS.json').exists(), 'Freeze before final strict review.'
    write('REVIEW_MANIFEST.json', dict(
        schema='corrected_supervision_publication_bindings_v1',
        complete=True, files=[bind(p) for p in files],
        excluded_receipts=sorted(excluded), protected_prior_files=331,
        all_new_phase_payload_files_bound=True,
        rationale='Exclude self and dependent review/publication receipts to avoid hash cycles; prior331 have immutable separate bindings. No weights/rawRGB/private features are payload.'))
    print('PUBLIC_MANIFEST_FROZEN',len(files),'files',sum(p.stat().st_size for p in files),'bytes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('documents','manifest'))
    args=parser.parse_args()
    (documents if args.stage == 'documents' else manifest)()


if __name__ == '__main__':
    main()
