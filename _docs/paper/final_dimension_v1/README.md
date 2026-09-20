# 월요일 마감용 최종 모델 고정 — 2026-09-19

## 현재 결정: 기존 결과 + 초록150 저장 라벨 평가 포함

초록 제외는 이전 응답의 오해였다. 사용자는 초록 포함을 다시 명확히 요청했다.
`manuscript.tex`는 기존 DEV319와 실측 처리시간에 초록150의 실제 GPU 추론/2D 점수 결과를 별도로 포함한다.
모든 표: `ALL_EXPERIMENT_TABLES.md`. 초록 상세: `green150_saved_labels_v1/RESULTS_KO.md`.
수동 클릭681점 주 분석과 PnP포함1200점 보조 분석을 분리하며 미검수/카메라/독립성 한계를 명시했다.
재현: `python -m scripts.evaluation.green_saved_labels_v1 snapshot|infer|score`를 단계별로 실행한다.
스냅샷과 예측은 불변이며 이미 있으면 재학습·재추론하지 않는다. infer는 정상 CUDA 권한이 필요하다.
표 생성: `python -m scripts.evaluation.build_green_saved_tables`.
`EXISTING_EVIDENCE_TABLES.md` / `existing_evidence_tables.tex`는 기존 JSON에서 생성한 PCK·coverage·seed별 개선/악화 표다.
생성: `python -m scripts.evaluation.build_existing_evidence_tables` (추론·학습 없음).
아래 기존 초록 실행 절차는 이전 버전이다. 이번 150장에는 위의 별도 버전을 사용했다.
PDF 빌드·시각 검수와 저자 정보 확인은 별도로 남아 있다.

최종 구성은 **동결된 YOLO26n R0 + 외부 치수 입력 + P 점 정제기(N2_DIM_ONLY)**.
새 학습, seed/temperature/반경 탐색, DHT, 명시적 C1/C2/C4 입력은 하지 않는다.
치수는 canonical W/D/H(m)이며 GT pose 기반 W/D 교환을 하지 않는다.
N2는 fixed-index 학습 target이다. symmetry-aware loss를 사용한 N3/N4와 구분한다.

## 고정 및 검증

- `MODEL_LOCK.json`: 배포 seed1, 재현성 seed2/3, R0/N0/N2 체크포인트·코드·정규화·temperature 해시.
- seed1은 작은 번호를 택한 고정 대표이며 최고 DEV 점수로 골랐다고 주장하지 않는다.
  기존 DEV는 이미 보았으므로 이 결정을 과거 실험의 사전등록으로 소급하지 않는다.
- `INFERENCE_SMOKE.json`: 기존 DEV 1장만으로 모든 head와 실제 YOLO 추론 경로를 확인.
- `DEV_COMPARISON.md/json`: 동일 corner8 지표의 R0 / 기본 P / 치수 P, seed 평균과 대표 seed 구분.
- `manuscript.tex`: 최종 치수 입력 모델을 중심으로 한 새 원고 초안.
  과거 `sensors_submission_v1` 원고 및 성능은 보존한다.

## 어노테이션이 끝나기 전

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.evaluation.final_dimension_release preflight
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.evaluation.final_dimension_release review-status
```

위 명령은 모델 해시/입력 상태만 확인한다. 새 초록 이미지의 모델 추론은 하지 않는다.
기존 851장 전체를 편입하지 않고 검토 복사본에서 EVAL로 저장한 파일만 후보가 된다.
이미지·어노테이션·카메라 해시, 9점 visibility, 같은 이미지 중복을 확인한다.
과거 square 실험과 같은 세션이면 독립 확인이라고 부르지 않는다.
촬영 권한/실측 치수/카메라 실제 보정 정확도는 파일 일치 검사만으로 입증되지 않는다.

## 향후 초록 평가를 별도로 재개할 경우에만

주의: 아래 기존 freeze 명령은 전체 검토 폴더의 EVAL을 읽는다. 150장 제안을 평가하려면
승인된 ID 목록을 강제하는 별도 버전의 실행 경로가 먼저 필요하다. 현재 그대로 실행하지 않는다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.evaluation.final_dimension_release freeze-eval --confirm-annotation-complete
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.evaluation.final_dimension_release infer
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.evaluation.final_dimension_release score
```

`freeze-eval`은 파일별 SHA를 잠근다. 이후 라벨이 바뀌면 기존 결과에 덮어쓰지 않고 실패한다.
GPU 추론은 R0 1회/프레임 + N0/N2 각 3개 head이며 재학습은 없다.
추론에는 RGB와 등록 치수만 사용하고 annotation은 별도 score 단계에서만 읽는다
(추론 전 무결성 검사의 파일 바이트 해싱은 예외이며 GT를 네트워크에 전달하지 않는다).
최고 score 후보 선택, box/score/order/center 유지, 전체 GT 분모 누락 penalty를 유지한다.

2D 주 지표는 frame-mean E_sym, 부 지표는 matched corner8 median/P90,
full-denominator PCK5/10/20, coverage, good→bad damage다.
C4는 평가에서 전체 물체 순열로만 사용하며 네트워크 입력이나 GT 기반 추론 선택은 아니다.
신규 bbox는 알려진 in-image 9점의 경계로 만들고 IoU≥0.5를 사용한다.
이는 새로 명시한 raw-image 평가 계약으로, 과거 square padded YOLO bbox와 완전히 같다고 주장하지 않는다.
세션 단위 paired CI를 쓰며 세션이 1개뿐이면 CI를 추정하지 않는다.
6D 신규 물리 실측 검증과 N2 전용 latency는 이 실행기에서 완료했다고 주장하지 않는다.

## 실제 남은 마감 항목

1. 기존/초록 수치·원고 교차 검수와 저장 라벨 품질 한계에 대한 저자 확인.
2. 기존 개선과 초록의 혼합 결과, 재사용 DEV의 해석 제한을 유지.
3. 동일 세션 R0/N0/N2 처리시간 실측 완료: `RUNTIME_RESULTS.json`, `RUNTIME.md` (기존 N4 값을 대체하지 않음).
4. LaTeX 환경에서 PDF 빌드 및 페이지/표/참고문헌 검수.
5. 저자/소속/연구비/데이터 권한/제출 동의는 실제 저자 확인.

현재 원고는 내부 초안이며 투고 완료가 아니다. Git commit/push와 학술지 제출은 수행하지 않았다.
