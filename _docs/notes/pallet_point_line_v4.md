# 점·선 증거 분리 실험 v4 (pallet_point_line_v4)

소스: `scripts/research/pallet_point_line_v4_bundle/` (외부 제공 번들, repo commit
`6a68452` 기준으로 작성됨). CLI 연결 코드는 같은 폴더 `cli/`.
결과 루트: `data/pallet/results/pallet_point_line_v4/`.

## 1. 제안

- 가설: 동일한 고정 backbone·동일한 후보 bank 조건에서 (a) 유한 선분 내부 관측과
  (b) 역할별 Deep Hough 단서를 읽기에 추가하면, 원래 맞던 점을 망가뜨리지 않으면서
  큰 위치·번호 오류가 줄어든다. (c) 같은-ID baseline reference 의존을 없애면 번호
  오류가 줄어든다.
- 방법: 네 군 P(끝점만) / S(+선분 내부) / H(+명시 DHT, 주 방법) / HA(H + 같은-ID
  reference) × seed 1,2,3. 각 2,000 AdamW updates, batch16, FP32. 네 군 모두
  P3+P4 를 공유하므로 해상도 단독 효과는 식별하지 않는다. **P/S 는 no-Hough 모델이
  아니다** — Hough 로 학습된 backbone 과 Hough 후보를 공유한다.
- 판정 지표: 유효 프레임별 8코너 평균 오차 / raw 대각선 의 평균(primary).
  진입 기준 = H 가 세 seed 각각 baseline·P 대비 1% 이상 낮고, pooled median/P90
  비악화, 원래 ≤10px 점의 >10px 전환율 ≤1%.
- 예상 실패 모드: (i) 후보 bank 에 애초에 더 나은 배치가 없음 → G4 차단,
  (ii) 좋은 후보는 있는데 scorer 가 못 고름, (iii) 합성에서만 좋아지고 실사 전이 실패
  (PAPER_S2 6연속 REJECT 와 같은 벽).
- 중단 기준: G4 미달 → `NATIVE_PROPOSAL_SIGNAL_INSUFFICIENT`. 합성 기준 미달 →
  `NO_SYNTHETIC_ADVANCEMENT_SIGNAL`, 실사 추론·학습 확대 없이 종료.

기존 해석 정정: structured-v2 는 이미 끝점 패치·유한 선분 내부 읽기·전체 배치
scoring·후보별 정답 오차 감독을 갖고 있다. 이번 기여는 그 기능들의 신설이 아니라
증거 축(선분 내부 / 명시 DHT / same-ID reference)의 분리다.

## 2. 결과

- 판정: `NO_SYNTHETIC_ADVANCEMENT_SIGNAL`. Stage A(4군×3seed 캐시 학습)와 Stage B(합성
  calibration + 합성 검증)까지 실행했고, 사전등록한 합성 진입 기준을 통과하지 못해
  **실사 DEV 추론과 단계 C(전체 네트워크 재학습)는 미실행**이다. 근거 파일:
  `data/pallet/results/pallet_point_line_v4/VERDICT.json`,
  `COMPARISON_synth_val.json`, `FINAL_AUDIT.json`. 상세 서술은
  [REPORT_KO.md](../../data/pallet/results/pallet_point_line_v4/REPORT_KO.md).

### 주지표 — 프레임평균 8코너 오차 ÷ raw 대각선, synth_val 512장

네 군(§1 제안 참조): P = 끝점만 읽음(no explicit Hough cue, 점 집합 reference) /
S = P + 유한 선분 내부 읽기 / H = S + 역할별 명시 Deep Hough 단서(주 방법) /
HA = H + 같은-ID baseline reference 의존.

| | seed1 | seed2 | seed3 |
|---|---:|---:|---:|
| baseline (고정 backbone 원래 출력) | 0.005755 | 0.005755 | 0.005755 |
| P (끝점만) | 0.005755 | 0.005755 | 0.005755 |
| S (+선분 내부) | 0.005429 | 0.006152 | 0.005436 |
| H (+명시 DHT, 주 방법) | 0.005789 | 0.005436 | 0.005436 |
| HA (H + 같은-ID reference) | 0.005793 | 0.005436 | 0.005436 |

사전등록 기준(H 가 세 seed 각각 baseline·P 대비 1% 이상 낮고 median/P90 비악화, 원래
≤10px 점의 손상률 ≤1%)은 **H seed1 에서 실패**한다.

### 핵심 진단 — 512장 중 1~5장의 파국 뒤집힘이 전부를 결정했다

모든 개선은 `G38__G__f7377` 단 한 장(133.606px → 5.340px, 8코너 인덱스 오프셋
8/8 → 2/8)의 수리다. 악화는 `f5154`(3.175px → 158.419px, 오프셋 0/8 → 8/8)와
`f3141`(8.841px → 157.864px)이다. 두 변화 모두 좌표가 조금씩 움직인 게 아니라
**8코너 인덱스 순열**이 통째로 바뀐 것이다(아래 그림 참조). 512장×8코너 중 한
프레임(8점)이 ±150px 움직이면 평균이 약 ±0.3px(≈6%) 흔들리며, 이는 관측된
군간·seed간 폭과 일치한다.

![H_seed1 candidate selection on synthetic val — decisive frames](../assets/pallet_point_line_v4/decisive_frames.png)

*이 그림은 H_seed1 이 어떤 후보를 골랐는지 보여주는 기전 설명용이며, 정확도
향상의 증거가 아니다(그림 상단 캡션에도 명시). 왼쪽은 수리된 프레임(오차 감소),
오른쪽은 파괴된 프레임(오차 급증)이고, 두 경우 모두 8코너 인덱스가 통째로
재배정된 것이지 점이 조금 움직인 게 아니다. 원본:
`data/pallet/results/pallet_point_line_v4/figures/decisive_frames.png`.*

### 그 뒤집힘의 정체 — 위치 오차가 아니라 C4 위상 선택 (사후 독립 확인)

최적 인덱스 할당으로 다시 재면 `f7377` 원래 예측 133.606 → **5.024px**,
`f5154` 선택 후보 158.419 → **3.186px** 다. 두 프레임이 요구하는 순열은 **동일**하며
`proposals.py` 의 `C4[3]`(= `YAW90` 3회, 270° 사분면)이다. 선택된 후보도 각각
quarter1·quarter3 의 C4 재인덱싱 후보였다.

즉 133px·158px 중 97~98%가 라벨링이고 실제 점 이동분은 3~5px 뿐이다. 코너는
제자리에 있고 번호만 돈다 — 메모리 `catastrophic-corner-error-is-90deg-axis-permutation`
과 같은 현상이다. 따라서 이 검증기는 이 프레임들에서 위치추정을 하는 게 아니라
**C4 위상을 고르고 있고, 한 번 맞히고 한 번 틀린다.** 정사각 4면 진입 팔레트
(`square-pallet-c4-symmetry-contract-2026-09-06`)에서 그 위상은 영상 기하만으로
결정되지 않을 수 있으므로, 관측된 "개선/악화"는 4지선다 동전던지기로 읽는 것이 맞다.

보고 지표가 8코너(중심 제외)임도 이 재계산으로 확증했다 — 9점으로 재계산하면
119.112/140.852 로 어긋나고 8코너면 소수 3자리까지 일치한다.

### 세 가설에 대한 답

1. **S − P (유한 선분 내부 읽기가 추가로 유용한가)**: 확인되지 않음(−8.3e-5,
   seed 산포보다 작음). P 는 세 seed 모두 calibration 에서 identity-only 가
   선택됐다 — finite margin 후보들이 전부 median/P90 비악화 제약을 위반했기 때문이다.
2. **H − S (명시 DHT 단서가 추가로 유용한가)**: 확인되지 않음(−1.2e-4, seed
   산포보다 작음). 단일 최고 셀은 오히려 S seed1 이다.
3. **H − HA (같은-ID baseline reference 의존이 번호 오류를 유지시키는 지름길인가)**:
   차이 −1.3e-6 = 주지표의 0.02% — **효과 없음.** seed2·3 은 두 군이 동일 프레임을
   동일하게 선택했다. 다만 이는 배선 문제 때문이 아니다 — 무결성 게이트(G3)에서
   HA 만 baseline ID 재라벨에 실제로 반응함을 별도 확인했다(H 2.6e-8 vs HA 4.0e-4).

### 병목 위치 — 후보 생성이 아니라 랭킹(선택)

train 분할에서 native oracle 여유는 **42.19%**, 프레임 평균 1px 이상 개선 가능한
후보가 있는 프레임은 **159개**(기준 32개, G4 통과)다. 즉 더 나은 배치 후보는
충분히 있는데, 학습된 검증기는 512장 중 1~5장만 실제로 바꾼다. 실패 위치는
후보 생성이 아니라 그중에서 고르는 랭킹이다.

### 한계

- 사전등록한 "양호점 손상률 ≤1%" 기준은 **통과했는데도** `f5154`·`f3141` 파괴가
  일어났다 — 한 프레임 8점은 전체 감독점(약 3,700여 개)의 0.2%뿐이라 비율 기준이
  파국 프레임을 걸러내지 못한다.
- calibration 256장에서 고른 margin 이 synth_val 512장에서 뒤집혔다(H seed1:
  calibration 주지표 0.005327 → val 0.005789). 파국 프레임 소수가 목적함수를
  지배하는 구조에서는 calibration 256장이 부족하다.
- `official_metric_parity_checked=false` — 실사 단계에 진입하지 않아 canonical
  evaluator 대조를 수행하지 않았다.
- 합성 단계에서 진입 기준을 통과하지 못했으므로, 이 결과는 실사 전이에 대해
  아무 것도 말하지 않는다.

### 외부 독립검토 반영 (정정)

외부 검토(431 검사 전부 통과, 원본 픽셀 지표 최대차 0.0)가 해석 오류 4건을 지적했고
저장 좌표에서 재계산해 **전부 사실임을 확인**했다. 전체 내역은
[CORRECTIONS_KO.md](../../data/pallet/results/pallet_point_line_v4/CORRECTIONS_KO.md),
검토 원문은 `data/pallet/results/pallet_point_line_v4/external_audit/`.

- **"C4 위상 동전던지기" 철회.** 관측 509장 중 순수 C4 후보가 엄격히 더 나은 8장에서
  H 세 seed 는 `f7377` 1장만 맞혔다(균등무작위 기대 2장). 첫 4후보 raw argmin 의 GT
  최선 불일치 H1=8/H2=9/H3=8. 무작위가 아니라 **원래 번호 배치 쪽으로 치우친 계통적
  랭킹 실패**다. 영상 식별 불가능성은 [미검증 가설]로 낮춘다.
- **"점수가 나은 후보를 싫어했다" 정정.** raw argmin 은 P1 498/512·H1 494/512 에서
  비identity 를 선호한다. 억제한 것은 calibration margin 이다 — 랭킹과 수락 규칙은 다르다.
- **S_seed3 을 H2/H3 와 묶은 것은 오류.** S1·S3 는 후보 1(순수 재배열, 5.024px),
  H2/H3/HA2/HA3 는 후보 28(재배열+국소 snap, 5.340px)로 서로 다르다.
- **"실제 이동 3~5px" 는 양의 혼동.** 잔여 정답오차가 아니라 좌표로 재면 실제 이동은
  `f7377` 코너4 9.104px, `f5154` 코너0 1.716px 한 코너씩이다. f7377 에서 H 는 순수
  재배열보다 0.315px 나쁜 배치를 골랐다.
- train oracle 42.19% 는 train 진단이므로 validation 병목의 근거로 쓰지 않는다.
  무결성 주장의 보증 범위도 좁혔다(정책 동결 순서, P4 parity 의 범위, 이미지 바이트
  비중복, supervision open 계측 범위, `missing_required_artifacts=[]` 의 한계).

### 다음으로 허용되는 행동 (정정)

"GT 오차 동률 세기" 계획은 폐기한다 — 동률이라는 사실은 영상 식별 가능성을 검사하지
못한다. 대신 순수 재배열로 고칠 수 있었는데 놓친 7장과 정상에서 파괴된 `f5154`·`f3141`
에서 **렌더러/어노테이션의 앞면 선택 규칙과 후보 점수 선호가 어디서 어긋나는지** 보는
제한된 무학습 진단 1건. 새 학습·새 렌더·실사 진입 없이 수행한다.
