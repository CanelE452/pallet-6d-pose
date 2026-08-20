# SEED REPLICATION PROTOCOL — 미실행, HARD_BLOCK

```
상태   NOT RUN
사유   seed3~6 에 대응하는 E3@18k source 가 존재하지 않는다 (PHASE 4A HARD BLOCK)
```

## 확인한 것

```
weights/paper_s2/paper_s2_multihead/screen_A1_CORNER_LINE_e3confirm25k_seed1/  step_{06,12,18,25}000.pth
                                                              .../seed2/       step_{06,12,18,25}000.pth
                                                              .../seed3~6/     없음
```

`mh_poseaware.SOURCE_RUN` 이 이 run 을 가리키고, C0·C1_RESCUE 4개 arm 모두
`step_18000.pth` 를 source 로 기록하고 있다(결과 JSON 의 `source_checkpoint`).
seed3~6 을 만들려면 **E3 base 를 그 seed 로 새로 학습**해야 하며, 브리프의 규정대로
새 initialization convention 을 발명하지 않는다.

## 비용 (자동 실행하지 않음)

```
E3 base 18,000 step × 4 seed     seed 당 약 2.5~3시간   → 10~12시간   [추정]
연속학습 3,000 step × 2 arm × 4  8 런 × 29분            → 약 4시간     [확인 기반]
6-seed 재평가                                            → 약 1시간
────────────────────────────────────────────────────────────────
합계                                                      약 15~17시간
```

29분/런은 C0/C1 실측이다. E3 base 는 이 프로젝트에서 재측정한 적이 없어 `[추정]`이다.

## 그런데 이 지출을 권하지 않는다

`PARAMETER_DRIFT_AUDIT.md` 가 보인 것: 두 seed 의 corner branch drift 궤적이
3,000 step 에서 0.4429 vs 0.4383 으로 **1% 차이**다. 파라미터 수준의 between-run
분산은 이미 작다.

반면 `LEARNING_CURVE_AUDIT.md` 가 보인 것: 같은 run 안에서 인접 마크 간 지표 변동이
arm 간 최종 효과와 같거나 더 크다(12개 중 5개는 더 큼, seed1 LA_HARD obs_rms 는
2000→3000 에 42.6 포인트 이동).

**seed 확장은 between-run 분산을 다루는 도구인데, 지금 지배적인 잡음은 within-run 이다.**
seed 6개로 늘려도 각 run 의 3000-마크가 ±20~40% 흔들리면 평균의 표준오차는 그만큼 크다.

## 대신 권하는 것 (다음 실험 하나, 자동 실행 안 함)

**평가 쪽을 먼저 고친다.**

```
1. LA cell 의 n 을 키운다.  현재 LA_FRONTAL 134 / LA_EASY 170 / LA_HARD 185 는
   dev(6,242)에서 뽑을 수 있는 전부다. train split 에는 같은 cell 이
   706 / 950 / 931 장 있으나 E3@18k 가 이미 본 프레임이라 쓸 수 없다.
   → 늘리려면 held-out 저앙각 프레임을 새로 렌더하거나, dev 비율을 재설계해야 한다.
2. 또는 마지막 마크 하나가 아니라 여러 마크를 묶어 읽는 추정량으로 바꾼다.
   단 이건 사전등록 대상이라 이번 결과를 보고 바꾸지 않는다.
```

두 방향 모두 **새 학습이 아니라 평가 설계** 문제다. seed 확장보다 싸고 직접적이다.
