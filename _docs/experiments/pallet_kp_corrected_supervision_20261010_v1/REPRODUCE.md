공개 파일만으로 원행의 수치와 내부 일관성을 검산할 수 있습니다. Python 3.9 이상 표준 라이브러리를 사용하며 Torch/OpenCV/모델/비공개 영상·가중치·GT는 필요하지 않습니다. 새 학습이나 자세 계산, 시간 측정을 시작하지 않습니다.

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
