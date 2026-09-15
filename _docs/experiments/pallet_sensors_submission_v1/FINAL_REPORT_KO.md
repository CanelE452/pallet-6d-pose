# Sensors 원고 마감 결과

- 시작 main: `0b8487b87b55d117d2f0aa4b1f3ef5b9dc6573ad`.
- 게시 SHA는 작업 완료 메시지와 ignored raw/PUBLISH_COMPLETE.json에서 확인한다. 자기 SHA를 커밋 내부에 억지로 넣지 않는다.
- 실행: PARTIAL_TECHNICAL; 선행 비교: MATCHED_BUDGET_COMPLETE; 독립 확인: AWAITING_INDEPENDENT_DATA.
- 기존 R0/P/D/L의 학습·가중치·선택·평가 결과는 그대로 보존했다. 기존 모델 추가 학습은 0회다.
- 선행은 실제 공식 TF1 CPU 네트워크를 기준으로 연산/가중치를 검증한 PyTorch PoseFix-derived pallet9다. RGB 및 pallet9 target/support 변경, BN의 실제 epsilon 보정을 공개했다. torchvision 대체나 D 재명명 결과가 아니다.

## 새 학습

실제 18000 updates, disposable smoke 1 update(s).

- seed1: 6000회 / 노출 96000 / 재개 0 / 2498.0초.
- seed2: 6000회 / 노출 96000 / 재개 0 / 2455.1초.
- seed3: 6000회 / 노출 96000 / 재개 0 / 2394.2초.

실제 학습 코드 SHA는 TRAIN_CODE_LOCK.json에 기록했다. 6,000회 최종 checkpoint만 사용했으며 원 논문의 140 epochs 수렴 성능을 재현했다고 주장하지 않는다.

- `prior_model.py`: `83e96d5071c1bd4903e429fefa23c9b965e48d0ece2bdca2314e1dad2938014e`
- `prior_data.py`: `52e37b409808014d71ff37a53a145d2b4d4cffb42c94fc058373f7ff314d1ded`
- `train_prior.py`: `14d97d072191943fb1a3c3326bd22c6f394cfb52230b234a82af801b8470758c`

## 결과와 한계

DEV319/13 sessions 및 negative2689이며 기존 조건부 매칭311장/2,756점, 전체 GT2,818점 분모를 보존했다. 새 prior의 실제 분모·비유한값·pose coverage는 UNIFIED_DEV_RESULTS.json에 따로 기록했다.

새 전체 panel 속도 측정은 외부 finetune_green_zip.py의 GPU 점유 때문에 보류했다. GPU는 정상 동작 중이며 no-CUDA/드라이버 오류가 아니다. 지시문의 반복 확인 상한에 따라 다른 작업을 종료하거나 무기한 대기하지 않았다. 기존 속도는 역사적 기록으로만 보존하며 PRIOR와의 동일 세션 속도 비교라고 하지 않는다.

GPU 작업 담당자가 다른 학습 종료를 확인한 뒤 `python scripts/research/pallet_sensors_submission_v1/run.py evaluate`로 비용 단계부터 재개한다. 이어 confirmation → manuscript → 모든 새 PDF 페이지 검사/visual receipt → audit → publish를 실행한다. 완료된 18,000 updates와 DEV 추론/통계를 재수행하지 않는다. 백그라운드 대기 작업은 남기지 않았다.

- P_minus_R0: -0.710767px, paired-session95% interval [-1.173744, -0.411083]. 음수는 P의 낮은 오차를 뜻한다.
- P_minus_D: -0.525361px, paired-session95% interval [-0.980016, -0.210116]. 음수는 P의 낮은 오차를 뜻한다.
- P_minus_PRIOR: 0.336232px, paired-session95% interval [0.062853, 0.584639]. 음수는 P의 낮은 오차를 뜻한다.

P가 모든 지표에서 최상은 아니다. 기존 D/L의 관측 P90는 P보다 낮으며 D의 지연도 더 낮았다. eval_noapril 세션에서는 P−R0의 중앙값이 악화됐다. 개발 자료 재사용 및 보정하지 않은 복수 secondary interval이므로 독립 확인이나 수렴된 방법 우월성으로 확대하지 않는다.

R0 초기 checkpoint에는 COCO-pose 사전학습 이력이 있다. 합성 전용 주장은 추가 P 학습에 한정한다. cm→mm로 표시한 pose 오차 통계 변화는 독립 물리 6D 실측이나 삽입 성공률이 아니다.

## 산출물과 다음 의존성

- 영문 전체 원고: `_docs/paper/sensors_submission_v1/manuscript.pdf` 및 `.tex`.
- 보조자료: 같은 디렉터리의 `supplementary.pdf` 및 `.tex`.
- JSON 기반 숫자/도표와 원천 hash: `NUMBER_SOURCES.json`, `NUMBERS_MANIFEST.json`.
- 현재 미완료 의존성은 외부 사람 검증 자료다. 데이터 담당자는 HUMAN_INPUTS_REQUIRED.txt의 촬영/블라인드 이중 어노테이션/치수·카메라 검증 자료를 지정 incoming 경로에 제공하고, 공저자는 AUTHOR_REVIEW.md의 저자·연구비·권리·동의 항목을 검토해야 한다.
- 자료 도착 후 `run.py confirmation --panel <검증된 panel.json>` → `run.py manuscript` → PDF 각 페이지 재검토 → `run.py audit` → `run.py publish`.
- 독립 새 자료가 없는 상태를 모든 실험 완료나 투고 준비 완료라고 부르지 않는다. 학술지 제출은 하지 않았다.

## 동일 DEV 및 같은 세션 속도 요약

| 모델 | median px | P90 px | 전체 PCK10 % | translation cm | pose coverage | 대표 full-path ms |
|---|---:|---:|---:|---:|---:|---:|
| R0 single | 6.615678 | 38.670038 | 63.7331 | 7.896852 | 1.0000 | NM (새 panel 보류) |
| P seed mean | 5.904910 | 37.766432 | 67.3409 | 7.152979 | 1.0000 | NM (새 panel 보류) |
| D seed mean | 6.430271 | 37.566241 | 64.5730 | 7.597128 | 1.0000 | NM (새 panel 보류) |
| L seed mean | 6.069932 | 37.319600 | 66.7850 | 7.548493 | 1.0000 | NM (새 panel 보류) |
| PRIOR seed mean | 5.568679 | 37.226827 | 68.4055 | 6.895311 | 1.0000 | NM (새 panel 보류) |

정확도는 seed별 통계의 평균이고 속도 대표는 seed1이다. 같은 열을 ensemble 또는 seed1 정확도로 해석하지 않는다. Raw prior는 UNIFIED_DEV_RESULTS와 보조자료의 별도 행이다. Rotation/yaw/IoU3D/ADDsym AUC, 전체 frame/point 분모와 모든 latency 반복도 해당 JSON/PDF에 보존했다.


이번 고정 예산의 주 중앙오차 비교에서는 PRIOR가 P보다 낮은 오차를 보였다. P를 가장 정확한 비교군으로 결론내리지 않는다. 알려진 파라미터 수와 미측정인 새 동일 세션 속도를 구분해야 한다.
