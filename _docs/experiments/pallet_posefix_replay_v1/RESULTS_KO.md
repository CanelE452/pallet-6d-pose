# PoseFix 합성 GT replay 보존 통제 실험

판정: `REPLAY_SCREEN_FAILED_DO_NOT_START_SELFTTRAIN`.

## 실제 실행과 비교 조건

- 추가학습 전의 같은 PRIOR1에서 시작. 기존 real-only와 seed1/300step/TFAdam1e-4/전층 학습/BN통계고정 동일.
- 매 step 실사8장의 index와 입력 교란을 기존 RNG6401로 재생성해 bit-exact 검증. 실사 노출2400회 그대로 유지.
- 합성8장/step, 총2400노출 추가. 합성 GT supervision만 추가하며 새 pseudo-label/teacher anchor/추가 유지loss 없음.
- 목적함수=real 데이터loss + source 데이터loss + 기존 L2 한 번. real loss 또는 L2를 절반으로 줄이거나 두 배로 늘리지 않음.
- 합성은 정상R0입력과 GT+큰교란을50:50확률로 사용. 정상/교란replay의 개별효과는 분리하지 않음.
- 전체 계산량과 노출은 증가하므로 compute-matched 주장이 아님. replay가 더 적은 real노출 때문에 좋아졌다는 설명은 배제.
- 같은9장/38manual감독, 미확인/PnP/중심정답제외. RGB+R0점/박스만 입력; 깊이/CAD/PnP/치수는 추론 입력 아님.
- 새 source heldout256은 replay train과분리. 기존R0/과거연구노출은배제하지않아독립최종검증아님.
- 실사평가는같은DEV72+GREEN150, 촬영분리·고정R0검출·GT분모·전체물체대칭평가유지. 평가로checkpoint/cap/threshold를선택하지않음.

학습 및 최종 probe/저장 경과 184.6초, peak allocated 2182.9MiB.

## 같은 학습9장: 학습능력 검사일 뿐 일반화 아님

| 입력 | 입력평균 px | replay 출력평균 px | PCK10 | 복구/hard |
|---|---:|---:|---:|---:|
| original_R0 | 18.661 | 0.319 | 100.000% | 3/3 |
| noisy_manual | 27.017 | 0.562 | 100.000% | 193/193 |

noisy_manual304개는38개정답×8교란이며새로운304개실사정답이아님.

## 합성 heldout256: 원본 prepared RGB px, fixed index

| 모델 | 입력 | 평균 px | P90 px | PCK10 | 복구/hard | 훼손/good |
|---|---|---:|---:|---:|---:|---:|
| 기존합성PoseFix | clean | 4.718 | 6.655 | 94.016% | 1/64 | 1/1669 |
| 기존합성PoseFix | stress | 25.694 | 49.928 | 19.535% | 56/1292 | 0/11 |
| 실사만 | clean | 32.207 | 100.622 | 51.978% | 2/64 | 679/1669 |
| 실사만 | stress | 48.726 | 128.676 | 29.426% | 284/1292 | 5/11 |
| 실사+합성replay | clean | 5.653 | 9.320 | 90.950% | 3/64 | 39/1669 |
| 실사+합성replay | stress | 9.710 | 24.716 | 68.843% | 772/1292 | 0/11 |

## 실사 촬영분리 개발평가

| 모델 | DEV PCK10 | DEV P90 px | DEV 복구/hard | DEV 훼손/good | GREEN manual PCK10 | GREEN P90 px | GREEN 복구/hard | GREEN 훼손/good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 53.514% | 51.375 | 0/156 | 0/138 | 82.232% | 9.213 | 0/4 | 0/344 |
| 현재 DIM-only | 59.459% | 50.910 | 0/156 | 0/138 | 81.351% | 9.749 | 0/4 | 0/344 |
| 기존 PoseFix raw | 57.838% | 50.524 | 0/156 | 0/138 | 80.176% | 10.172 | 0/4 | 0/344 |
| 기존 PoseFix 1%cap | 57.838% | 50.524 | 0/156 | 0/138 | 80.323% | 10.090 | 0/4 | 0/344 |
| 실사만 raw | 45.225% | 144.866 | 10/156 | 35/138 | 68.135% | 37.452 | 3/4 | 76/344 |
| 실사만 1%cap | 54.234% | 53.427 | 0/156 | 12/138 | 79.442% | 10.404 | 0/4 | 20/344 |
| 실사+replay raw | 60.180% | 52.982 | 4/156 | 1/138 | 77.386% | 11.065 | 1/4 | 14/344 |
| 실사+replay 1%cap | 59.820% | 50.497 | 0/156 | 0/138 | 78.561% | 10.769 | 0/4 | 5/344 |

GREEN matched hard4개뿐. 전체hard75중71은박스매칭실패패널티라점보정으로복구불가. 1%cap은640×480에서8px로, 같은GT대응의>20→<=10복구에필요한이동보다작음. 그러므로 raw가복구검사의주출력,cap은안전성참고이며좋은모드선택금지.

## 실행 전 고정한 후속 self-training 진입 기준

| 검사 | 관측값 | 조건 | 통과 |
|---|---:|---|---|
| trainability | — |  same training sanity criteria | True |
| source_clean_PCK10_delta_pp | -3.06627 | >= -1.0 | False |
| source_clean_P90_ratio | 1.40043 | <= 1.1 | False |
| DEV72_raw_damage_rate | 0.00724638 | <= 0.01 | True |
| DEV72_raw_PCK10_delta_vs_N2_pp | 0.720721 | >= -0.5 | True |
| DEV72_raw_P90_ratio_vs_N2 | 1.04071 | <= 1.05 | True |
| GREEN150_MANUAL_raw_damage_rate | 0.0406977 | <= 0.01 | False |
| GREEN150_MANUAL_raw_PCK10_delta_vs_N2_pp | -3.96476 | >= -0.5 | False |
| GREEN150_MANUAL_raw_P90_ratio_vs_N2 | 1.13493 | <= 1.05 | False |
| DEV72_raw_recovered | 4 | >= 5 | False |

모든 기준을 만족해야 후속 선택적self-training에진입한다. 기준은개발screen의사전안전범위이지통계적비열등성검증이아니다.

## 실행 범위와 보존

- 후속 선택적self-training 시작 여부: False.
- 실패 시 새 pseudo-label을만들거나학생을추가학습하지않음. threshold/seed/lr/cap변경으로결과를구제하는추가탐색없음.
- 원본 R0/PoseFix/N2 checkpoint, 기존 결과, 최종 모델, 어노테이션과 논문 표를 보존함. 자동 교체/commit/push 없음.
- 222장 전체평가,baseline metric parity744개,검출/중심등보존888개검사.
- PCK10은 전체GT유효코너분모,매칭실패패널티포함. P90은매칭코너. 복구는R0>20→<=10,훼손은R0<5→>10.
- 동일9장의부분/불균형감독과개발자료재사용은여전히한계다. 실패를RGB보정/PoseFix/self-training전체의불가능으로확대하지않음.
- 정상합성GT replay와큰교란source둘다추가된개입이다. 계산량증가,source분포,정규화효과를각각분리한실험은아님.

산출물: PROTOCOL.json, INPUT_LOCK.json, BEFORE.json, FIT.json, REAL_RESULTS.json, DECISION.json, AUDIT.json.
그림: outputs/pallet_posefix_replay_v1/user_example_right_zoom.png 및 사용자전체/가장개선/가장악화사례.
