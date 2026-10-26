# 졸업논문 분석 Batch 1 — 재학습 없음

## 목적

파렛트 종류별 DOPE 자기학습(pallet-6d-pose 79ba5b3 결과)의 논문 문장을 확정하기 전에, 기존 체크포인트·예측으로 T0–T6을 계산한다.\
학습·재학습·임계값·하이퍼파라미터 변경 금지. 추론(forward)은 T1(b)·T3·T4에서만 허용.

## 확인된 사실 (계획자가 기록에서 본 것)

- [확인] 주 실험 폴더 /home/minjae/Documents/pallet-research/dope-loo-flip-per-type-20261026-v1 :\
  SUMMARY.json, SCORES.json, PREDICTIONS.json, 종류별(GREEN/PLASTIC/WOOD) POOL.json·PSEUDO_R1.json·PSEUDO_R2.json, R1·R2 체크포인트\
  (출처: 79ba5b3의 \_docs/experiments/pallet_dope_selftrain_graduation_20261026_v1/reproducibility/methods_verified.json, source_results/loo_flip/COMPLETION.json)
- [확인] flip 제거 폴더 /home/minjae/Documents/pallet-research/dope-loo-no-flip-per-type-20261026-v1 :\
  SCORES.json, PLASTIC184_PREDICTIONS.json, PLASTIC184_SCORES.json, 종류별 RAW_R0_PREDICTIONS.json·PSEUDO_R1/R2.json
- [확인] 주 평가 DEV292 = PLASTIC 128장/985점, WOOD 45/346, GREEN 119/943.\
  PCK@τ = 원본 기준 오차 ≤ τ px인 점 수 / 전체 평가 기준점 수(예측 누락은 실패). 의미 채널 0–7, 중심점 제외, 대칭 재정렬 없음
- [확인] 대조군(합성만 추가학습)은 종류 공통 arm 하나(sharedcontrol R1/R2)
- [확인] R0 = /home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057_noseg.pth (sha256 f4b455b3…).\
  같은 저장소 data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/INVENTORY.md에\
  R0 학습 입력 640×480→400×400 비등방 squash, 좌표 추출 7×7 soft-argmax(T=0.1)로 기록
- [확인] 이번 평가·미세조정 전처리: 반사 패딩 100px → 원래 크기 → 비율 유지 짧은 변 400(640×480→400×528).\
  디코더: 가우시안 σ2, 국소 최대, 응답>0.3, 반경 5(11×11) 가중 중심, 0.4395칸 오프셋 (methods_verified.json)
- [확인] \_docs/audits/eval56_summary/decoder_reconciliation/DECODER_COORDINATE_COMPARISON.md:\
  11×11 가중평균 추출이 ep57 코너를 7×7 soft-argmax 대비 중앙값 +3.4px(eval56), +5.1px(wood) 이동

## 확인이 필요한 전제 — 실행 전에 확인하고, 틀리면 멈추고 보고

- [추정] 두 폴더의 PREDICTIONS.json·SCORES.json에 영상별·기준점별 예측/오차가 있어 재추론 없이 영상 단위 집계가 된다\
  → 한 모델의 영상별 합계가 SUMMARY.json 정답 수(예: PLASTIC R0 PCK10 258/985)와 정확히 일치하는지 먼저 확인. 불일치·부재면 중단
- [추정] R0의 학습 당시 추론 경로(400×400 squash + 7×7 soft-argmax)를 재현하는 코드가 있다(decoder_reconciliation 문서의 D0/P0 경로)\
  → 없거나 학습 당시와 같다는 근거가 없으면 T1(b)는 실행하지 말고 무엇이 없는지 보고. 비슷한 경로를 임의로 만들지 말 것
- [추정] 필터 구현(methods_verified.json evidence_anchors의 main_filter·ablation_filter·LOO·RANSAC 위치)에서 단계별 통과 여부를 원본 수정 없이 얻을 수 있다\
  → 새 폴더에 래퍼를 만들고, 최종 승인 결과가 원 함수와 DEV292 전 프레임·전 코너에서 100% 일치함을 먼저 보고. 일치 전엔 T3 수치 보고 금지
- [추정] 녹화 ID와 촬영 조건(실내/실외, 주간/야간)이 메타데이터에 있다 → 없으면 "없음"으로 보고

## 작업 (순서: 전제 확인 → T0 → T1 → T2–T6)

T0 데이터 구성 — 메타데이터만, 영상·정답 내용은 열지 말 것

- DEV292와 종류별 학습 풀 1,000장: 녹화 ID별 장수, 촬영 조건 분포
- 보호 FINAL(물리 FINAL 역할 자료): 종류별 장수, 독립 6D/자세 정답 존재 여부, 학습 풀과 녹화 겹침\
  T1 R0 기준선\
  &#x20;(a) DEV292 관측점에서 모델별 평균 부호 오차(dx, dy = 예측−기준)와 |오차| 중앙값:\
  &#x20; R0, 대조군 R1·R2, LOO+flip R1·R2, LOO만 R1·R2 — 종류별\
  &#x20;(b) 전제가 확인되면 R0만 원래 경로로 재평가: 같은 기준점·같은 PCK 규칙으로 PCK`@5/10/20`, coverage, 조건부 중앙값\
  T2 paired bootstrap 95% CI
- 영상 단위: 영상을 복원추출해 두 모델에 같은 표본 적용, 10,000회, seed 20261026, percentile
- 녹화 단위: 녹화 ID를 복원추출. 종류별 녹화 수를 같이 표기하고, 3개 미만이면 계산하지 말고 표기만
- 지표: ΔPCK`@5/10/20`, Δcoverage, Δ조건부 정밀도(= PCK@10 정답 수 / 관측점 수)
- 쌍(종류별 주 평가): LOO+flip vs 대조군(R1, R2) / LOO+flip vs LOO만(R1, R2) / R2 vs R1(LOO+flip, 대조군 각각)
- 보조: GREEN manual-click 600점, PLASTIC184 — 주 결과와 섞지 말고 별도 표
- 판정(계산 전 고정): 하한>0 "높았다", 0 포함 "차이 불명확", 상한<0 "낮았다". 다중 비교 보정 없음을 표 아래 명시\
  T3 필터 정밀도 — DEV292 정답은 채점에만, 임계값 변경 금지
- 교사: R0, 종류별 LOO+flip R1, 종류별 LOO만 R1. 원본 + 좌우 반전 추론
- 누적 단계: s0 신뢰도≥0.3·영상 안 → s1 RANSAC 합의(인라이어≥6) → s2 +LOO(점·중앙값) → s3 +flip(점·중앙값) → 최종(≥6점)
- 비교 선별 "신뢰도만": s0 점이 6개 이상인 영상의 s0 점 전부
- 단계별 보고: 통과 영상 수, 통과 점 수, 통과 점 정밀도(오차 ≤5/10/20px), 탈락 점 정밀도. GREEN은 manual-click 점만으로도 따로\
  T4 PnP 가능 프레임 비율: 모델별·종류별로 DEV292에서 모서리 6개 이상 검출 + 등록 치수·K로 PnP 성립(양의 깊이)한 비율. 정답 불필요\
  T5 정성 그림 후보 — 선정 규칙 고정
- 종류별로 영상당 PCK@10(LOO+flip R2 − 대조군 R2) → 최대·중앙값·최소 각 1장(총 9장). 동률이면 영상 ID 오름차순
- 오버레이: 기준점 + R0 + 대조군 R2 + LOO+flip R2, 모서리 번호별 고정 색. 영상별 차이 목록 CSV도 저장\
  T6 교차 종류: 두 SCORES.json에 이미 있는 교차 종류 수치만 표로 정리(새 추론 없음)

## 출력

- 새 폴더 /home/minjae/Documents/pallet-research/dope-thesis-analysis-batch1-v1/ (이미 있으면 중단하고 보고)
- 작업별 CSV + MD 표, 그림 PNG + PDF, RUN_LOG(명령, seed, 입력 파일·코드 sha256)
- 작업 전후 두 실험 폴더와 R0 체크포인트의 해시 비교 결과

## 이 계획이 틀렸다면 어디서

- 가장 약한 고리는 T1(b): R0 원래 경로를 정확히 재현 못 하면 숫자가 오히려 오해를 만든다 → 근거가 없으면 (a)만
- T3 래퍼가 원 함수와 다르게 동작할 위험 → 100% 일치 확인이 먼저
- T2 영상 단위 CI는 같은 녹화의 연속 프레임을 독립으로 봐서 좁게 나온다 → 녹화 단위를 같이 보고, 녹화가 5개 이상인 종류는 녹화 단위를 본문 기준으로
- 실패 증상: SUMMARY.json 정답 수 재현 실패, 래퍼 불일치, 녹화 ID 누락

## 게이트

아래 항목을 실행 전에 확인하고, 실패하면 그 자리에서 멈추고 보고해줘. 멈추는 게 기본값이야.

1. 이 지시문이 지목한 파일·키·함수 위치가 실제로 존재하는가 → 없거나 의미가 다르면 중단하고 실제 구조를 보고
2. 이 지시문이 전제한 값의 일치(SUMMARY.json 정답 수 재현, 래퍼–원 함수 일치)가 수치로 성립하는가 → 불일치하면 어디서 얼마나 벗어나는지 보고
3. DEV292 정답이 임계값·체크포인트·사례 선택에 쓰이지 않는가 → 쓰이게 되면 중단하고 경로를 보고
4. 만들라고 한 산출물(특히 CI·필터 단계별 통계)이 이미 존재하는가 → 있으면 만들지 말고 경로를 보고
5. 보호 FINAL의 영상·정답 내용을 열지 않는다(메타데이터 집계만)

## 제약

- 기존 데이터·체크포인트·결과 파일 수정 금지, 추가 학습 금지, 출력은 새 폴더에만
- 작업 전후 기존 파일 해시를 비교해 보고
- 사례·seed를 유리한 쪽으로 다시 고르지 말고 전체를 그대로 보고

## 보고

- [확인]/[추정] 태그를 달아줘. [확인]에는 경로:라인 또는 수치를 인용
- 내 전제가 틀렸으면 틀렸다고 말해줘. 지시문을 따르는 것보다 오류를 잡는 게 우선이야

이거 해서 main에 push해줄수있어?