# 최종 인계 — B 완료, A 본학습은 수식 결정 대기

2026-09-16. 전체 요청을 완료했다고 보고하지 않는다. **B의 선 보조 이득은 입증되지 않았고, A의 대칭 지도 효과는 아직 평가하지 않았다.**

## 실제 실행 범위

| 항목 | 실행 | 상태 |
|---|---:|---|
| 첨부 수학 + adapter CPU 회귀검사 | 31/31 | PASS |
| A 실제 identity loss/gradient 검사 | RECT 16장 + SQUARE 16장 | bit-exact; optimizer 없음 |
| A0 square 기존 5모델 추론 | 5 × 155 = 775회 | 완료 |
| A0 기존 paper prediction 재채점 | 10 × 319 = 3,190개 | 완료; 새 추론 아님 |
| A0 원 annotation 정정 후 재채점 | 기존 예측 775개 | 완료; 추가 추론 아님 |
| A 본학습 | 0/12 fits, 0/24,000 updates | 미실행 |
| B source 관측 | P 2,304회, DHT 768회 | 완료; 기존 R0 cache 사용 |
| B 실사 관측 | R0 3,008회, P 5,574회, DHT 1,858회 | 완료 |
| B 군별·seed별 예측 생성 | 합성 7,680개, 실사 45,120개 | cache 재사용; neural forward 횟수와 다름 |
| B 6D 보조 평가 | 12,465 pose-frame 평가 | 완료 |
| B runtime | 520 timed samples | 완료; 준비·검산 호출 별도 |

추가 DHT 수치 진단 1,132회. Runtime 전체 호출은 R0 808 / P 808 / DHT 606회로, 검산·warmup·stage 계측을 포함한다. B 새 학습은 0회다. 상세 분모와 실행량은 `RUN_MANIFEST.json`을 따른다.

## A: 효과 없음이 아니라 미평가

객체 전체 C1/C2/C4 순열, 좌표·mask 공동 순열, corner8 고정 분모, centroid 별도 평가를 검증했다. RECT source와 승인된 SQUARE task-equivalent cohort는 섞지 않았다.

설치된 YOLO26의 RLE는 배치 전체 평균 뒤 clamp하는 비분리 손실이다. 따라서 **물체별 독립 complete-loss min**과 **원래 stock 손실 완전 보존**을 일반적으로 동시에 보장할 수 없다. 위치비용만 쓰거나 RLE를 빼는 임의 변경을 하지 않았다.

필요한 사용자 결정: 원래 손실을 보존하면서 두 head가 공유하는 물체별 대칭 조합을 **공동 선택**하는 프로토콜 변경을 허용할 것인가? 허용 후에도 EQUIV adapter·공유 branch 배선 검사부터 구현·검증해야 하며, 현재 준비 완료된 학습기로 가장하지 않는다. 동일 2,000-step 예산은 유지한다.

추가로 원본 annotation 대조에서 square train 1장·val 1장의 prepared label이 미표시 코너/중심을 visible로 저장한 문제가 발견됐다. 851장 원영상 픽셀은 모두 정확했다. 원본은 보존하고 두 행의 tuple·bbox만 별도 target view에서 정정했다. 향후 두 학습군은 이 동일 view를 사용해야 한다. 재채점 변화는 모델 개선이 아니다. 자세한 사항은 `A/REPORT_KO.md`와 `A/ANNOTATION_TARGET_AMENDMENT.json`에 있다.

## B: 현 고정 구성에서는 최종 추론에 넣을 근거 부족

B3−B0의 주지표 E_sym 차이:

- 합성: −0.000001219, 95% CI [−0.000005159, +0.000002977]. 예측 512장 중 corner annotation 없는 2장을 명시적으로 분리한 평가 510장.
- 재사용 실사 DEV: −0.000004359, 95% CI [−0.000021313, +0.000013411]. 319장, 13세션 cluster bootstrap; unmatched 8장도 실패 penalty로 분모에 유지.

둘 다 `LINE_AUXILIARY_NOT_ESTABLISHED`. B1/B2 대조도 세 선 또는 모호성 가중의 필수성을 입증하지 못했다. 실사 pooled P90은 61.636→61.150px로 소폭 감소했지만, 개선·악화 프레임이 공존하고 주지표 CI가 0을 포함한다.

동일 RTX3080 전체 RGB→PnP median은 P **12.73ms**, B3 **39.33ms**, 약 **3.09배**다. 뚜렷한 정확도 이득 없이 현재 구현 비용이 늘었다. P/DHT와 이 수식의 bounded 결과이며 Hough 자체의 불가능성이나 A의 실패로 일반화하지 않는다. 자동 추가 탐색은 하지 않는다.

## 변경·보존 경계

- 새 코드와 보고서는 `scripts/research/pallet_symmetry_three_line_v1` 및 이 문서 폴더에만 추가했다. 대용량 cache/예측/그림 원본은 별도 ignored 결과 경로에 둔다.
- 기존 R0/P/DHT 가중치·원본 이미지·원본 GT·논문·배포 설정은 변경하지 않았다. 사용자 소유의 기존 dirty 파일은 staging에서 제외한다.
- 새 평가는 task-equivalent corner8 지표다. 기존 논문의 fixed-index/matched9 결과를 덮어쓰거나 같은 지표인 것처럼 혼합하지 않았다.
- GT는 평가와 calibration 선택에만 사용했다. B inference role alignment/가중/후보 선택에는 입력하지 않았다. WLS 및 utility 출력 호출은 0이다.
- 정사각형 task C4는 물리·시각적 외관의 완전한 대칭 주장과 다르다. 실사 pose GT는 독립 계측이 아니다. DEV 재사용, physical visibility 미검증, 단일 DHT seed라는 한계를 유지한다.
- GPU는 정상 사용했고 재부팅·드라이버/시스템 변경·타인 작업 종료·외부 알림·FINAL 열람은 하지 않았다.

다음 단계는 **A의 공동 branch 선택 허용 여부 결정**이다. B의 실패 판정을 이유로 A를 실패 처리하지 않는다.
