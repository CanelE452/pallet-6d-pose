# 팔레트 키포인트 보임·가림 판별: V193까지의 추가 진행 검토와 다음 작업 조언

작업 유형: 최신 결과 진단·연구 방향 검토 / 적용 경로: 목적 확인 → 근거 대조 → 제한된 후속 설계.

- 작성일: 2026-10-24.
- 확인한 저장소: `CanelE452/pallet-6d-pose`.
- 고정한 main snapshot: `d2fe21c6861bde8e148cec991345dedfaa2b2865` — 2026-10-24 12:44:03 UTC, 한국시간 21:44:03.
- 이번 추가 검토의 중심: **V189–V193**. 이전 원고·OCPR 결과와 개발 버전은 변경하지 않았다.
- 산출물 성격: **근거에 기반한 조언 문서**. 새 모델·학습·추론·실험을 실행하거나 승인하는 문서가 아니다.
- 표기: **[확인]**은 이번에 읽은 원문·코드·저장 집계 또는 명시한 사용자 계약, **[추정]**은 그로부터의 해석, **[추정·미검증]**은 아직 실행하지 않은 설계다. 공개 보고서의 PASS와 이번 검산은 구분한다.

## 1. 목적을 다시 고정하면 지금 해야 할 일이 달라진다

**[확인·사용자 계약]** 현재 목표는 초기 추론에서 지정한 팔레트 키포인트에 대해 **직접 보이는지(DIRECT_VISIBLE, D), 외부 물체에 가려졌는지(EXTERNAL_OCCLUDED, E)를 구별하는 모델**을 만드는 것이다. 실제 팔레트 이미지뿐 아니라 그 특징·레이블로 classifier, calibration, normalizer, threshold 또는 모델 선택을 적합하지 않는다. 범용 사전학습 모델의 동결 사용과 합성 학습·검증은 허용된다. RGB 추론에는 실제로 얻을 수 있는 초기 추정, 카메라 내부 파라미터 K, 알려진 종류별 치수만 쓸 수 있다. 이 문서가 그 계약을 변경하지 않는다.

**[추정·미검증] 권고하는 목적의 순서:**

1. **판별 대상 일치:** 사람이 평가하는 native 키포인트와 모델이 학습하는 부위가 같은 대상을 가리키는가?
2. **입력에서 판별 가능:** 실제 초기 query와 정규화 crop에 그 대상과 가리는 물체의 관계가 남아 있는가?
3. **허용된 합성 자료에서 일반화:** 학습에 쓰지 않은 장면·물체 부위·가림 자산에서도 D/E를 구별하는가?
4. **실사 전이:** 모든 설정을 고정한 뒤 실사에서 D 보존과 E 검출을 함께 달성하는가?
5. **후속 활용:** 검증된 출력이 관측 선택이나 pose 계산에 도움이 되는가?

**[추정] 지금은 1–4가 주목표다. pose 성공률 개선을 첫 단계의 필수 통과 조건으로 넣으면 분류기 자체의 실패와 후단 사용 방식의 실패가 다시 섞인다.** 반대로 D/E 분류만 잘됐다고 B/I 관측의 물리적 대응이나 6D 자세 개선까지 입증되는 것도 아니다.

## 2. 이번 추가 진행을 반영한 핵심 판단

**[확인] “직접 가림 출력을 만들어 학습하자”는 작업은 이미 했다.** V190에서 C54/C55를 실제로 학습했고, V191에서 기존 실사 개발 집합에 적용했다. C55는 이전 C53보다 E를 맞힌 수가 20→37개로 늘었지만 D를 맞힌 수는 868→831개로 줄었다. V193은 그 다음의 source 대조 자료 확대이며 새 모델 성능 결과는 아니다. [S1–S5]

**[추정] 따라서 지금 가장 타당한 조언은 “head를 하나 더 붙이고 더 오래 학습하자”가 아니다.** 이미 직접 출력을 추가한 뒤에도 남은 **학습 목표와 평가 목표의 대응, 독립 source 검증 부재, 감독 범위의 편중, 실사 학습 금지 계약과 실제 학습 자료의 차이**를 먼저 정리해야 한다. 이 중 어느 하나가 단독 원인이라고 확정할 근거는 아직 없다.

| 이전에 필요했던 일 | 최신 확인 상태 | 지금 조언의 변화 |
|---|---|---|
| 원영상의 모든 픽셀을 96 crop이 읽어야 한다는 과한 조건 검토 | V189에서 실제 샘플링 지지 기준으로 수정 | 같은 수정 작업을 다시 지시하지 않는다 |
| 넓은 영역의 support와 특정 끝부위의 가림을 구분 | 별도 direct-end target과 logit을 구현하고 V190 학습 완료 | 직접감독이 왜 실사 D/E와 충분히 연결되지 않는지 본다 |
| RGB와 frozen SAM 조건을 맞춘 비교 | C54/C55 동일 초기화·샘플 순서·각 8,000 update 기록 | SAM을 무조건 추가하거나 제거하는 결론을 내리지 않는다 |
| SHAM 배치 때문에 탈락한 대조쌍 확인 | V192 진단 후 V193 실제 공개 source 쌍 5→27 | 데이터 생성 진전과 모델 성능 진전을 구분한다 |
| 실사 가시성 판별 해결 | V191 미해결, V193 새 모델 평가 없음 | 채택·일반화·전체 문제 해결을 선언하지 않는다 |

표의 실행 사실은 [확인], 마지막 열은 [추정]이다. 근거: [S1–S5].

## 3. V189–V193에서 실제로 한 일

### 3.1 V189: 샘플링 조건을 고쳤지만 지원 자료 수는 같았다

**[확인]** V189는 end 영역이 원래 96×96 샘플 좌표 범위 안에 있고, 유효한 bilinear 보간에 양의 기여를 하는지를 확인하도록 바꿨다. 원 query, crop 크기 L, native 순서를 보존하고 COVER는 질문한 끝부분을 가리며 SHAM은 그 끝부분을 보존하도록 했다. 결과는 합성 **97/1,024 triplet**, 공개 실사 기반 **5/66 triplet**으로 V188과 같았다. 새로 검사 가능해진 공개 source 33건도 다른 기하·주변 근거 보존 조건에서 탈락했다. 생성 조건 실패를 모델 오분류로 세면 안 된다. [S1]

**[확인]** 여기서 triplet은 같은 장면의 **BEFORE / COVER / ADJACENT_SHAM** 세 변형이다. SHAM은 가리는 물체를 주변에 두되 질문한 부위를 가리지 않는 대조다. 독립된 세 장면을 뜻하지 않는다. 합성 direct target은 native anchor 주변의 유한한 부위를 사용하는 별도 목표이며 코드가 `physical_native_gold=False`를 명시한다. [S1, S10]

### 3.2 V190: 직접 끝부위 감독으로 두 모델을 실제 학습했다

**[확인]** C54는 RGB96, C55는 RGB96와 frozen SAM 신호를 썼다. 각 8,000 update, 총 16,000 update를 수행했고 초기화와 표본 순서를 맞췄다. 기존 support·owner 손실을 유지하면서 direct-end 이진 출력을 추가했다. 직접 레이블은 **306행 = 합성 291행 + 공개 실사 기반 weak 15행**, D204/E102다. 이는 **97개 합성 triplet과 5개 공개 실사 triplet**에서 나온 상관된 표본이다. [S2]

**[확인]** 새 출력층은 65개 파라미터지만 **65개만 학습한 것은 아니다.** trainer는 전체 raw CNN과 공유 층을 포함한 **115,222개 파라미터**를 학습한다. 외부 SAM만 동결한다. 입력은 RGB, 유효 픽셀, 고정 crop 상대 XY, SAM mask와 가용성으로 구성된다. 따라서 이 모델을 “동결된 범용 공간 특징 위의 65개 파라미터만 학습한 모델”이라고 설명하면 틀린다. [S2, S7, S11]

**[확인]** 알려진 direct 학습 레이블에서 C54는 305/306, C55는 306/306을 맞혔다. **학습 적합 결과**다. V192는 이 306행 모두 학습에 사용했고 별도의 direct-end source holdout이 없다고 명시한다. 이 데이터를 지금 나누어도 이미 학습한 C54/C55의 독립 검증이 되지 않는다. [S2, S4]

### 3.3 V191: 실사에서 일부 E를 회복했지만 충분하지 않았다

**[확인]** 다음은 같은 고정 D/E query **1,237개: D1,147/E90**에서 읽은 CSV다. 목재105프레임의 645점과 정사각형 등록118프레임 중 유효117프레임의 592점이다. SELF·화면 밖·미주석·대응 UNKNOWN은 이 D/E 표에 합치지 않았다. 이번 검토에서 **18개 집계행의 혼동행렬 합, 분모, precision/recall을 직접 산술 대조했고 저장값과 최대 차이는 0**이었다. 전체 원시 ledger 재집계나 모델 재실행은 아니다. [S3, S6]

| 모델 | 실제 D를 D로 판정 | D 재현율 | 실제 E를 E로 판정 | E 재현율 | E 정밀도 | 실제 E→D | primary 예측 U |
|---|---:|---:|---:|---:|---:|---:|---:|
| v61 | 260/1,147 | 22.67% | 88/90 | 97.78% | 9.03% | 2 | 0 |
| legacy | 655/1,147 | 57.11% | 83/90 | 92.22% | 20.65% | 3 | 177 |
| C52 RGB | 821/1,147 | 71.58% | 33/90 | 36.67% | 9.85% | 52 | 29 |
| C53 RGB+SAM | 868/1,147 | 75.68% | 20/90 | 22.22% | 6.83% | 70 | 6 |
| C54 RGB+direct-end | 698/1,147 | 60.85% | 35/90 | 38.89% | 7.23% | 55 | 0 |
| **C55 RGB+SAM+direct-end** | **831/1,147** | **72.45%** | **37/90** | **41.11%** | **10.48%** | **53** | **0** |

**[확인]** 이 표는 같은 사람 D/E 정답에 대한 응답 비교이며, 모든 모델이 같은 목표·자료로 학습됐다는 표가 아니다. legacy는 과거 real CAL로 학습된 진단 참조군이다. v61의 E 열은 query-pixel presence의 기존 비가시 이진 응답(`ABSENT_NONVISIBLE`)을 평가에 연결한 것이며, 외부 가림을 직접 학습한 출력으로 해석하지 않는다. 둘 다 이번 엄격 합성 전용 direct-end 목표의 등조건 학습 대조로 취급하지 않는다. [S3, S6, S9]

**[확인]** C53→C55에서는 E 정답이 17개 늘고 D 정답은 37개 줄었다. 전체 정답은 888→868개다. 같은 query를 연결한 paired CSV에서는 기존 정답→오답 174개, 기존 오답→정답 153개, 기존 U→정답 1개, 기존 U→오답 5개였다. E만 보면 회복20/손상3, D만 보면 회복133와 U에서 회복1/손상171이다. [S6, S8]

**[추정]** 이것은 **가림을 더 찾아내면서 보이는 코너도 더 많이 가림으로 처리하는 trade-off**다. E 재현율 하나나 전체 정확도 하나로 해결 여부를 판단하면 안 된다. C55가 E라고 한 353개 중 실제 E는 37개이므로, 이 출력을 그대로 관측 제거에 쓰는 것은 별도 검증이 필요하다.

**[확인]** C53과 C55는 단순히 같은 출력에 loss 하나만 추가한 비교가 아니다. C52/C53은 지역 support의 규칙으로 D/E/U를 만들었고, C54/C55는 direct logit이 0 이상이면 E, 미만이면 D를 반환한다. 유효한 유한 입력에 대한 신뢰 보류 규칙은 없다. 따라서 **primary U0은 불확실성이 사라졌다는 결과가 아니다.** 전체 원입력 1,784개 중 input U45는 별도로 남는다. [S2, S3, S9]

#### 팔레트 유형별 차이는 반드시 남겨야 한다

**[확인] C55 유형별 결과:** [S6]

| 유형 | D/E 평가 분모 | D 정답 / D 재현율 | E 정답 / E 재현율 | 실제 E→D | 실제 D→E |
|---|---:|---:|---:|---:|---:|
| 목재105 | 616 / 29 | 492 / 79.87% | 6 / 20.69% | 23 | 124 |
| 정사각형118 중117유효 | 531 / 61 | 339 / 63.84% | 31 / 50.82% | 30 | 192 |

**[추정]** 목재에서도 E29개 중23개를 D로 통과시키므로 정사각형의 외관 모호성만으로 현재 실패를 설명할 수는 없다. 다만 촬영 세션·크기·재질·가림 물체가 함께 달라졌으므로 이 차이를 형상 효과로 확정할 수도 없다. 이번 표는 가시성 분류 표이며, 약90도 회전·W/D 대응 오류를 직접 계수한 결과가 아니다.

### 3.4 V192: 독립 표본과 감독 범위가 좁다는 점이 구체화됐다

**[확인]** 합성 97 triplet은 47개 base에서 왔고, 공개 실사 5 triplet은 2개 프레임에서 왔다. 합성 지원 native 분포는 **[29,24,17,23,2,2,0,0]**, 각 native의 시도 분모는128이다. native6/7은 이 direct 감독에 포함되지 않았다. 65,536개의 증강 crop을 같은 수의 독립 장면으로 셀 수 없다. [S4]

**[확인]** 합성 지원 쌍의 end-related sample은 96×96 중16–49개였다. 이는 끝부분 원픽셀의 양의 보간 기여가 있는 output sample 수다. 사람이 식별할 수 있는 신호량이나 CNN이 보존한 정보량을 직접 측정한 것은 아니다. [S4, S12]

**[확인·해석 주의]** V192의 짧은 필드명 `SHAM_changed_RGB96_samples`는 실제로 `end_related96_SHAM_changed_RGB_samples`를 읽는다. **끝부분 관련 subset에서만** 변화 수를 센다. 97쌍에서 이 값이0인 것은 질문한 끝부분을 SHAM이 보존했다는 의미다. **전체 crop이 BEFORE와 같거나 주변 가림 물체가 입력에 없다는 뜻으로 읽으면 틀린다.** 이 문서의 판단에도 그런 잘못된 해석을 사용하지 않았다. [S12]

### 3.5 V193: 공개 source 대조쌍은 늘었지만 학습·채택은 없었다

**[확인]** 한 번의 반대 방향 SHAM 배치와 실제 픽셀 조건 검사를 통해 기존5개를 유지하면서 지원 triplet이 **27개**, 원본 프레임이 **12개**, weak D/E/D 행이 **81개**로 늘었다. 지원 native는0·1·4·5다. 전체66시도 중 배치 실패21개와 UNKNOWN18개를 남겼다. 새 학습·모델 추론·실사 평가·PnP는 없고 `training_admission=false`, `adopted=false`다. [S5]

**[추정]** 이는 필요한 데이터 제작상의 진전이다. 그러나 **81개의 독립 자연 가림 정답**, native8 전체의 감독, 엄격 합성 전용 모델의 개선으로 해석하면 안 된다. 원 사진이 실제 팔레트 RGB이면 그 위에 가림 물체를 합성해도 전체 입력이 합성 전용으로 바뀌지는 않는다.

## 4. 이번에 실제 그림을 열어 확인한 범위

**[확인]** 아래 네 PNG를 바이트로 읽어 표시하고 확인했다. 전체 갤러리의 이미지 검수, 원 RGB/NPZ 재해시, 물리 owner 정답 생성은 수행하지 않았다.

| 직접 본 이미지 | 그림에 기록된 사실 | 이 그림에서 가능한 해석과 한계 |
|---|---|---|
| V191 `CASE_0002/CARD.png` | 목재 native2, 사람 D, 참조와 query 차이 약2.007px, C54 D/C55 E | 작은 참조 오차에서도 D 오탐이 남는 사례. 참조의 물리 정확도나 실패 원인을 확정하지 않음 |
| V191 `CASE_0996/CARD.png` | 정사각형 native4, 사람 E, 참조 차이 약1.138px, C54/C55 D | E 누락을 큰 crop 위치 오차만으로 설명할 수 없는 사례. crop에 녹색 재료가 보인다는 사실을 native D 정답으로 바꾸지 않음 |
| V193 `Q0277_D2/SELECTED_END_TRIPLET.png` | BEFORE/COVER/SHAM, 원 query·L 유지, finite-end weak 표적 | 주변에 foreign 물체가 존재하는가와 특정 끝부분을 가리는가는 다른 질문임을 보여 주는 대조 |
| V193 `Q0277_D2/PRODUCER_RAW_CONTEXT_CONTACT.png` | 전체 원 사진 문맥과 세 crop, 끝부분 enclosure와 주변 witness 표시 | crop 중심 query와 감독하는 끝부분의 관계를 확인할 수 있음. 실제 깊이·native 물리 GT의 증명이 아님 |

원 이미지 링크: [S13–S16]. 이미지는 새로 만들거나 수정하지 않았다. **[추정]** 앞의 두 사례는 실패가 남았다는 반례이며 전체 실패 원인의 빈도를 대표하는 무작위 표본은 아니다.

## 5. 왜 아직 잘 안 되는가: 확정 사실과 가설을 분리한 진단

### 5.1 감독 목표가 가까워졌지만 평가 목표와 같아진 것은 아니다

**[확인]** 새 direct-end 목표는 특정 유한 부위가 합성 가림에 의해 가려졌는지다. 실사 평가는 등록된 native 코너의 사람 D/E 레이블이다. 현재 head 입력에는 RGB96, valid, 고정 상대 XY, 선택적인 SAM mask가 있지만 **native ID, 지정 부위 위치, 전체 팔레트 문맥, K/치수/pose가 직접 들어가지는 않는다.** q와 L을 만드는 upstream에는 기하 정보가 있을 수 있지만 그 정보가 head의 별도 조건으로 전달되는 것과는 다르다. [S1, S7, S10]

**[추정]** “화면에 목재가 보인다” 또는 “주변에 다른 물체가 있다”를 학습하는 것이 쉬워도, “지금 질문한 native에 대응하는 끝부분이 가렸는가”는 더 어려울 수 있다. 같은 crop 안에 여러 판재·앞뒤 모서리·다른 팔레트가 있으면 중심과 주변 재질만으로 질문 대상을 유일하게 정하기 어려울 수 있다. 현재 코드가 이런 정보를 명시적으로 주지 않는다는 것은 사실이지만, 그것이 실사 오분류의 단독 원인이라는 것은 아직 가설이다.

**[추정·미검증]** 다음에는 목표를 새 이름으로 바꾸기보다, **native ID → 현재 초기 query → 의도한 물리 부위 또는 가상 코너 → 합성 direct-end 감독**의 대응을 한 행에서 설명할 수 있어야 한다. 배포 시 얻을 수 없는 GT 끝점이나 GT pose를 입력 조건으로 넣어 해결한 것처럼 만들면 안 된다.

### 5.2 높은 학습 적합과 전이 사이에 독립 검증이 없다

**[확인]** C55는 direct TRAIN306개를 모두 맞혔지만 독립 direct-end source 검증이 없다. 실사에서는 E37/90이었다. 합성 지원은 일부 native에 몰려 있고 동일 base에서 여러 변형이 나온다. [S2–S4]

**[추정]** 현재 자료로는 다음 설명을 구분할 수 없다.

- 새로운 부위·배치에도 같은 가림 관계를 이해하지만 실사 외관에서만 실패한다.
- 학습에 본 장면·가림 물체·붙여넣기 특징을 이용하고 새로운 합성 장면에서도 실패한다.
- finite-end target과 native D/E가 다른 대상을 지칭해 평가에서 실패한다.
- crop에 필요한 문맥이 없거나, 초기 W/D·native 대응 오류로 다른 부위를 보고 있다.

**[추정]** 그래서 다음 큰 학습 전에 필요한 것은 더 높은 TRAIN accuracy가 아니라 **학습에서 보지 않은 합성 그룹에서의 direct 관계 일반화**다. 기존 source 데이터를 사후 분할해 기존 checkpoint를 검증하는 것으로 대체할 수 없다.

### 5.3 현재 C54/C55는 사용자가 요구한 엄격 합성 전용 모델이 아니다

**[확인]** V190은 합성 증강뿐 아니라 `PUBLIC_TRAIN1480`, `CF3816`, 공개 RGB 기반 direct weak 레이블을 사용한다. 기존 owner 보조 감독도 공유 층에 gradient를 전달한다. **current223을 fit에 쓰지 않았음**과 **실제 팔레트 이미지로 task-specific fit을 하지 않았음**은 서로 다른 조건이다. [S2, S11]

**[추정]** 이 차이는 연구 결과를 폐기할 이유가 아니라 **어느 계약을 검증한 결과인지 바로잡아야 할 이유**다. C54/C55와 V193 자료 제작은 역사적 진단으로 보존한다. 엄격 계약을 유지한다면 공개 실사 팔레트의 weak 레이블도 새 학습·보정·문턱·모델 선택에 쓰지 않는 별도 경로가 필요하다. 이번 문서는 공개 실사 사용으로 계약이 바뀌었다고 추정하지 않는다.

### 5.4 이진 출력을 강제하는 것과 UNKNOWN을 해결하는 것은 다르다

**[확인]** C54/C55는 유한 direct logit에서 D 또는 E를 반환한다. 새 logit으로 “신뢰가 부족하면 보류”가 학습된 것은 아니다. [S2, S9]

**[추정·미검증]** 다음 기록은 서로 분리해야 한다.

| 상태 | 뜻 | 평가에서의 처리 |
|---|---|---|
| input unavailable | crop/특징을 만들 수 없음 | 입력 실패율과 원래 분모를 보존 |
| truth UNKNOWN | 사람/생성 계약이 참값을 확정하지 못함 | D/E 정확도에는 넣지 않되 건수·원인을 별도 보고 |
| predicted U | 유효 입력이지만 모델/규칙이 보류 | coverage와 클래스별 보류·누락을 함께 보고 |
| SELF | 자체 형상 때문에 직접 보이지 않는 별도 상태 | E와 합치지 않음; 초기 기하 판정은 가설로 유지 |
| OOF | 영상 밖 | 가림 E와 합치지 않음 |

**[추정]** U를 많이 내어 어려운 점을 지우거나, U를 없애 모든 점을 강제 분류하는 방법 모두 단일 coverage 숫자로 우수성을 주장할 수 없다. 실제 사용 목적에 맞는 E 누락과 D 손상의 비용을 먼저 정하고, 허용된 합성 validation에서만 규칙을 고정해야 한다.

### 5.5 물리 GT가 없다는 이유로 이미 있는 D/E 평가를 없애면 안 된다

**[확인]** V191은 사람의 D/E 의미 레이블과 좌표 reference의 권위를 구분한다. 작은 reference 차이는 정확한 물리 코너 인증이 아니며, 큰 차이도 기존 사람 E 레이블을 자동 무효화하지 않는다. [S3, S13, S14]

**[추정]** 사람의 의미 판독이 유효한 D/E 평가는 계속 할 수 있다. 독립 물리 대응 정답이 필요한 것은 **정밀한 위치 보정의 정답성, 실제 경계 owner, 가상 cuboid 코너와 재료 모서리의 일치** 같은 더 강한 주장이다. 없는 정답 때문에 모든 학습을 영구 BLOCKED로 두거나, 반대로 weak 자료를 물리 정답으로 승격하는 양쪽 모두 피해야 한다.

## 6. 선행연구가 지지하는 원칙과 여기서 새로 검증해야 할 것

**[확인]** 아래 원문/공식 기록을 이번에 열어 확인했다. 많은 방법을 추가하기보다 현재 설계 판단에 필요한 원칙만 쓴다.

| 1차 출처 | 확인한 핵심 | 이번 연구에 적용할 범위 |
|---|---|---|
| Lian et al., **VAPO: Visibility-Aware Keypoint Localization for Efficient 6DoF Object Pose Estimation**, IROS2025. 본문 III-A/B; 학회는 공저자 publication 목록으로 확인 | 표면 키포인트의 외부 가시성과 self-visibility를 나누고 대칭에 따른 레이블 일관성을 다룸. non-convex에서 normal 조건은 충분조건이 아님 | keypoint별 목표·대응·자기 가림을 구분할 근거. 가상 cuboid 코너를 동일한 표면점으로 간주하거나 synthetic-only 해법이라고 부를 근거는 아님 |
| 공식 BOP **PVNet-CVPR19/LM-O**, 2020-08-11 submission | 물체별 CenterNet/PVNet을 BOP2020 PBR RGB만으로 학습한 변형. 원 논문과 detector/offset field/PBR-only 차이를 명시 | 합성만으로 RGB keypoint 모델을 학습한 사례. frozen 범용 native 가림 분류기가 이미 있다는 증거는 아님 |
| Geirhos et al., **Shortcut Learning in Deep Neural Networks**, Nature Machine Intelligence2020 | 표준 평가에 맞는 상관 규칙이 다른 조건으로 전이되지 않는 문제와 평가 설계 논의 | donor/붙여넣기/장면 외관에 의존하는지 반증 대조를 둘 이유. 현재 모델의 shortcut 원인을 입증한 것은 아님 |
| Cawley & Talbot, **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation**, JMLR2010 | 유한 평가 자료로 선택을 반복하는 과정 자체의 과적합·선택 편향 | 같은 real223 결과를 보고 head/threshold를 고른 뒤 독립 평가라고 부를 수 없다는 근거 |

공식 링크와 확인 범위는 [L1–L4]. **[추정]** keypoint visibility 자체, 합성 학습 자체, SAM 사용 자체를 신규성으로 삼기 어렵다. 가능한 연구 질문은 **실제 추론 오차가 있는 query와 크기 정규화 조건에서, 의미 대상이 일치하는 합성 감독이 실사 native D/E로 전이되는가**다. 이것도 아직 성능이나 신규성을 입증한 결론은 아니다.

## 7. 내가 권하는 다음 경로: 큰 학습보다 먼저 세 가지를 고정하기

### 7.1 첫째 — “어떤 점의 가림인가”를 바꾸지 않고 연결한다

**[추정·미검증]** 한 query의 계약에 최소한 다음이 있어야 한다.

- 물체 instance, native ID와 실제 의미, 허용 permutation, 초기 추정의 q/L.
- 학습 레이블의 대상: 표면 native / 가상 cuboid corner / 연결된 finite material end 중 무엇인지.
- crop에서 그 대상의 위치를 알 수 있는 근거와 불확실성. GT 위치는 감독·진단에만 사용.
- D/E/SELF/OOF/UNKNOWN의 정의와 레이블 출처.
- BEFORE/COVER/SHAM에서 고정한 것과 바꾼 것, 주변 같은 부위의 근거가 유지되는지.
- 실제 배포 가능한 prompt·mask·기하 정보의 출처. 합성 GT mask를 입력하고 실사에서 SAM mask로 바꾸지 않음.

**[추정·미검증]** B는 투영 외곽의 물체–배경 경계 관측, I는 내부 RGB gradient/색 전환 관측으로 유지한다. B/I는 가시성의 정답 분류가 아니다. B뿐 아니라 의미적·물리적 대응이 확인된 I도 위치 감독을 받을 수 있다. 한 방향 부분선만 있는 경우에는 선의 수직 방향 제약만 주고, 관측되지 않은 접선 방향까지 정확한2D점처럼 감독하지 않는다. 현재 D/E 판별 검증에 이 확장 목표를 몰래 합치지 않는다.

### 7.2 둘째 — 허용된 학습 계보와 독립 합성 split을 먼저 만든다

**[추정·미검증]** 엄격 계약에서는 public-real source와 그 파생 CF/weak 레이블도 새 학습에서 제외한다. 기존 real-CAL state head·잔차 통계·H1 입력 의존성을 새 가시성 모델에 가져오지 않는다. 범용 frozen 모델의 사전학습과 팔레트 task-specific fit은 별도 칸으로 공개한다. 초기 검출기 자체가 어떤 학습 이력을 갖는지도 별도로 확인해야 하며, head만 합성으로 바꿨다는 이유로 전체 pipeline의 합성 전용 출처를 확정하지 않는다.

**[추정·미검증]** 분할 단위는 개별 증강 row가 아니라 **원 장면/base·원 프레임·물체 instance·그 native 전체·모든 paired variant·jitter의 계보**다. 같은 원영상에서 나온 support/owner 보조학습 표본이 validation 장면을 공유해도 누수다. donor 자산·장면·시점·재질·크기별 미관측 조건도 별도 검증 축으로 둔다. 새 분할의 독립성을 주장하려면 그 split에 맞춰 새 checkpoint를 학습해야 한다.

**[추정·미검증]** 기존 `occlusion_fraction`은 생성 의미·분모·native 순서·self/external 구분이 확인되기 전에는 D/E 정답으로 쓰지 않는다. 실제 표면 mesh와 renderer/object-ID/depth를 확인할 수 있으면 외부 가림 레이블을 그 계약에 맞춰 정의한다. RGB+2D GT만으로 정확한 hidden surface owner를 복원할 수 있다고 가정하지 않는다. 새 학습에는 허용된 합성 출처에서 근거가 있는 의미 D/E 또는 명시적인 weak finite-end 감독만 사용한다. 공개 실사 의미/weak 레이블은 기존 개발 진단·평가 근거로 분리한다.

**[추정]** 이미 실사 개발 결과를 보고 연구 문제를 진단한 이력도 남겨야 한다. 새 가중치·문턱·모델 선택을 합성에서만 고정하는 것과 연구 설계가 실사 실패 관찰에서 출발했다는 것은 별도로 공개한다. 현재 C55가 특정 실사 지표에서 높다는 이유로 다음 모델이나 문턱을 선택하지 않는다.

### 7.3 셋째 — 최소 baseline은 유지하고, 확인된 정보 부족 하나만 바꾼다

**[추정·미검증] 권고 baseline:** 현재의 작은 RGB96 공간 모델 구조를 **새 합성 전용 split에서** 기준으로 삼는다. 기존 C54의 실사 학습 가중치를 이어 학습해 합성 전용이라고 부르지 않는다. SAM2·DINOv2·이상탐지를 한꺼번에 추가하지 않는다. 범용 frozen 특징으로 바꾸는 비교는 입력과 감독의 문제를 먼저 분리한 뒤 선택할 후속 대조다.

**[추정·미검증] 첫 변경 후보는 하나다.** 같은 crop으로 의도한 native/end를 식별할 수 없는 사례가 확인되면, 배포 가능한 초기 기하의 **질문 부위 정보**를 전달하는 작은 조건 입력을 검토한다. 예를 들어 초기 projection의 주변 방향·인접 부위 관계를 사용할 수 있지만, native ID를 넣는 것만으로90도 W/D 혼동이 해결된다고 보장하지 않는다. 초기 pose가 잘못됐을 때 조건 입력도 틀리므로 그 조건 오류를 별도로 검증해야 한다.

**[추정·미검증]** 반대로 같은 부위는 분명하지만 crop이 자주 빗나간다면, 먼저 위치/범위 문제를 다룬다. object-scale 정규화는 유지하고 pose-conditioned rectification은 기본값으로 추가하지 않는다. 확대가 원영상에 없던 세부를 복원하지 않으므로 작은 물체의 유효 픽셀·crop 밖 비율을 별도 보고한다. 더 큰 문맥을 쓰는 대조는 총 입력 픽셀과 계산량을 맞춘다.

**[추정·미검증]** localization 공동학습은 위 진단에서 필요한 경우에만 추가한다. 보이는 물리 부위의 유효 GT에만 위치 loss를 주고, GT 기반 valid mask와 고정 분모로 정규화한다. 예측 visibility로 loss를 꺼서 모든 점을 숨기는 회피를 허용하지 않는다. 가린/가상 코너의 문맥 기반 좌표 추정은 별도 amodal 목표이며 직접 관측이라고 취급하지 않는다. 이는 기존 direct head의 단순 동등 교체가 아니라 목표 확장이다.

## 8. 제안하는 최소 확인 순서와 중단 조건

**[추정·미검증]** 아래는 후속 작업을 별도로 합의했을 때 사용할 순서다. 이 문서 작성 중에는 실행하지 않았다. 현재 8,000 update나 이전 실행시간을 새 예산으로 자동 복사하지 않는다.

**[추정·미검증] 재사용 원칙:** V192의 기존 계보·지원 census와 V193의 q/L/native·sampler·alpha/RGB·witness 검산을 먼저 재사용한다. 같은66시도의 검산을 처음부터 반복할 이유는 없다. 추가로 필요한 것은 엄격 합성 전용 경로의 사전 그룹 분할, finite-end와 native 평가 목표의 연결, 그 구별에 필요한 정보가 실제 head 입력에 남는지에 대한 확인이다.

| 순서 | 입력·고정 요소 | 확인하거나 바꿀 것 | 필요한 출력 | 진행 / 중단 기준 |
|---|---|---|---|---|
| 0. 목표·출처 확정 | 현재 사용자 계약, 기존 source 목록·코드 | native D/E와 finite-end의 대응, public-real와 CAL 의존 | 목표/입력/레이블/출처 표 | 계약 위반 source가 남으면 새 합성 전용 fit은 보류 |
| 1. 기존 자료의 계보 확인 | 기존 저장 메타데이터·분할·실제 query | base/frame/variant/보조loss 간 중복, native·크기·가림 자산별 지원 | 그룹 분할과 지원/UNKNOWN/실패 census | 독립 source holdout을 만들 수 없으면 일반화 평가를 보류; 증강 row 수로 충분성을 대체하지 않음 |
| 2. 입력–목표 판독 확인 | 저장된 원 crop·전체 문맥·원 레이블, 고정 q/L | 질문 부위를 찾을 수 있는지, COVER/SHAM이 그 관계만 바꾸는지 | 고정 사례별 target association·input validity 기록 | target을 유일하게 연결할 수 없으면 그 행은 강한 native 감독으로 쓰지 않음; 기존 사람 D/E 평가 레이블을 임의 삭제하지 않음 |
| 3. 작은 합성 일반화 확인 | 사전 group split, 같은 crop·target·학습 예산 | RGB 기준 모델 한 개부터; 필요할 때 조건 입력 대조 한 개 | TRAIN/VAL/TEST D/E 혼동행렬, coverage, paired 반응, 비용 | TRAIN만 좋고 새로운 합성 그룹에서 실패하면 실사 평가·새 backbone 추가보다 source/입력 진단으로 돌아감 |
| 4. 설정 동결 | 합성 validation에서 정한 모델·threshold·U 규칙 | checkpoint·normalizer·prompt·crop·fallback 버전 고정 | 동결 계약과 재현 manifest | 실사 결과로 다시 선택하면 엄격 계약 위반 또는 명시적 계약 변경으로 기록 |
| 5. 실사 전이 확인 | 같은 동결 설정, 촬영 세션 분리, 유효 의미 D/E | 목재·정사각형·영상 크기·가림 조건별 판별 | 클래스별 지표·손상/회복·불확실성·미산출 | 반복 노출223은 개발 진단. 독립 세션 없으면 독립 일반화 결론을 보류 |
| 6. 후단 사용 | 분류 판정이 충분히 검증된 뒤 고정 backend | 점·부분선의 사용/가중·초기 자세 fallback | pose 손상/회복·평균/꼬리·전체 비용 | 분류 성공을 pose 성공으로 자동 확장하지 않음; 원좌표와 보정좌표를 독립 관측처럼 중복 투입하지 않음 |

**[추정·미검증] 처리량과 예산:** 첫 학습 전에 작은 고정 배치로 실제 데이터 준비·frozen feature 또는 mask 생성·학습 step·저장을 각각 측정하고, 측정한 처리량으로 실행량을 정한다. GPU 시간이나 충분한 표본 수를 지금 확정하지 않는다. 현재 저장 결과만으로 가능한0–2번의 점검을 먼저 끝내는 편이 낭비가 적다.

### 반드시 보는 분류 지표

**[추정·미검증]** D/E confusion, E 재현율, E 정밀도, D 재현율, 클래스별 예측 U와 input unavailable, native 의미 D/E 레이블이 유효하고 사전 고정 query cohort에 속하는 분모, 기존 정답 손상·오답 회복을 함께 보고한다. 좌표 reference 유효성은 위치 오차의 별도 subset으로 두며, 좌표가 불확실하다는 이유로 유효한 의미 D/E까지 제외하지 않는다. weak target와 native target을 한 표의 동일 정답으로 합치지 않는다. 인접 프레임과 같은 triplet 변형을 독립 표본으로 세지 않고 원 장면·촬영 세션 단위 불확실성을 보고한다. 자료가 적으면 넓은 불확실성을 그대로 남긴다.

**[추정·미검증]** UNKNOWN은 성능을 좋게 보이게 하는 삭제 도구가 아니어야 한다. truth UNKNOWN, 입력 실패, 모델 보류를 별도 분모로 보이고, 보류를 실패로 포함한 coverage-aware 결과도 함께 제시한다. 사후적으로 쉬운 native만 남겨 전체8개 문제를 풀었다고 하지 않는다.

### 향후 pose에 연결할 때만 추가할 항목

**[추정·미검증]** 같은 reference-valid 영상에서 공동 성공 `T<5cm`와 사전 작업 동치에 맞춘 `R<5°`, 평균/중앙값/P95 T/R, 실제 pose 산출 분모, 미산출, 회복·손상, 전체 runtime을 보고한다. 목재와 정사각형은 분리한다. canonical 약90도 오류와 symmetry-aware 오류를 나누고, native ID/허용 permutation 근거가 없으면 semantic W/D 오류 원인은 UNKNOWN으로 남긴다. 실사 유형별 결과를 보고 유리한 threshold나 모델을 고르지 않는다.

## 9. 지금 유지할 것과 보류할 것

| 항목 | 권고 | 이유 |
|---|---|---|
| V189의 실제 sampler·q/L/native 보존 검사 | 유지 | 입력을 바꾸어 레이블만 좋아지는 문제를 막는 확인 장치 |
| direct-end와 broad support/owner의 목표 분리 | 유지 | 서로 다른 질문을 한 출력 이름으로 덮지 않음 |
| V191 동일 query 비교·실패 사례 공개 | 유지 | D/E trade-off와 실제 남은 실패를 볼 수 있음 |
| V193 원쌍 보존·UNKNOWN/실패 보존 | 유지 | 자료 제작의 성공 범위와 누락을 정직하게 남김 |
| C55의 즉시 채택 | 보류 | E53/90 누락, D316/1,147 오탐, 엄격 합성 전용 계약 불충족 |
| 공개 weak27쌍을 자동으로 다음 학습에 투입 | 보류 | 현재 실사 팔레트 fit 금지 계약과 다름; `training_admission=false` |
| 동일 설정의 학습시간·seed만 대폭 확대 | 보류 | direct TRAIN은 이미 거의 완벽하고 독립 source 검증이 없음 |
| SAM2·DINOv2·이상탐지·새 selector 동시 추가 | 보류 | 어떤 문제를 해결했는지 귀속하기 어려워짐 |
| `source-only`를 `synthetic-only`로 설명 | 사용하지 않음 | 공개 실사 source 포함 여부가 다른 조건 |
| 물리 GT 부재를 이유로 사람 D/E 평가 전체 폐기 | 사용하지 않음 | 의미 가시성 평가와 정밀한 물리 대응 인증은 별개 |
| query 중심 owner를 native 가시성으로 치환 | 사용하지 않음 | 중심에 재료가 보여도 질문한 native는 가려질 수 있음 |

이 표는 [추정] 권고이며, 사실 근거는 앞 절의 코드·결과에 연결된다.

## 10. 교수님께 현재 말할 수 있는 내용

**[확인+추정] 설명 초안:**

> 현재는 팔레트의 초기 키포인트가 직접 보이는지 외부 물체에 가려졌는지를 구별하는 문제를 먼저 검증하고 있습니다. 넓은 영역의 재료·가림 물체 존재를 예측하던 모델에 특정 끝부위의 직접 가림 감독을 추가해 실제 학습·평가했습니다. 실사 개발 자료에서 가려진 점의 검출은 일부 늘었지만 보이는 점의 오분류도 늘어, 아직 신뢰할 수 있는 분류기로 채택하지 않았습니다. 최근에는 source 대조쌍의 입력 보존과 가림 위치를 검증했으나, 이 결과는 모델의 일반화 증거는 아닙니다. 다음에는 학습에서 질문한 부위와 평가 native의 대응을 고정하고, 실사 팔레트 학습을 제외한 독립 합성 분할에서 먼저 일반화를 확인하려 합니다.

**[확인] 현재 말할 수 없는 것:** native8 전체 해결, 엄격 합성 전용 조건으로 검증된 C55, 독립 실사 일반화, SAM의 보편적 우위, B/I 정책의 독립 효과, 물리 owner 오류 해결, pose 개선, 통계적 유의성. V190–V193 원문도 채택·전체 해결을 선언하지 않는다. [S2–S5]

**[추정] 다음 의사결정의 핵심은 모델 이름이 아니다. “무엇을 물었는지 입력에서 식별할 수 있고, 그 질문과 같은 합성 정답으로 새로운 장면에서도 학습되는가”를 먼저 확인하는 것이다.** 그 연결이 없으면 direct loss·SAM·crop 확대를 계속 더해도 서로 다른 문제를 함께 맞추는 실험이 반복될 수 있다.

## 11. 다음 인수인계에서 필요한 최소 자료

**[추정·미검증]** 다음 목록은 새로 실행하라는 명령이 아니라, 후속 결과를 판단할 때 필요한 근거다. 파일명은 기존 산출물로 가장하지 않고 내용 계약으로만 적는다.

1. **출처·학습 계약:** 모든 loss의 입력/레이블 출처, frozen/학습 여부, public-real·private-real·synthetic 구분, upstream/CAL 간접 의존.
2. **target 매핑:** native·물체 부위·finite-end·query·crop과 감독의 관계, weak/직접/UNKNOWN 권위.
3. **독립 split:** base/frame/instance/variant/donor 계보와 train/validation/test 중복 검사. 보조 loss의 입력도 포함.
4. **실제 입력 계약:** crop 생성식·q/L 출처·원해상도·valid/padding·mask/prompt·조건 정보. GT 입력과 배포 입력 구분.
5. **동결된 기준:** 사용할 모델, threshold/U/fallback과 합성 validation 선택 근거, 실사 평가 전에 고정한 시각.
6. **저장 예측과 집계:** 동일 ID에 D/E/U truth/prediction/logit, input availability, 유형·세션·native·크기·crop 밖 여부, 회복/손상 연결.
7. **실제 개선·실패 그림:** 전체 문맥과 원 crop을 함께, query를 GT로 옮기지 않고 표시. 선택 기준·이미지 provenance 보존.
8. **계산 범위:** 전처리·mask/특징 생성·head·I/O·후단 포함/제외와 cache 조건. 작은 head 시간으로 전체 시스템 비용을 대신하지 않음.

## 12. 실제 검토·검산 범위와 한계

**[확인] 이번에 한 것:**

- 고정 main snapshot과 V189–V193 보고서·결과 JSON 확인.
- V190 학습 완료·사전 계약·trainer/head·레이블 admission 관련 코드를 분담해 읽고 대조.
- V191 18행 집계 CSV를 읽고 혼동행렬·분모·precision/recall 산술 직접 확인: 최대 차이0.
- 분담 검토에서 V191 paired CSV 신규 C54/C55 관련54행의 합과 주집계 연결 확인.
- V192 필드의 원 정의를 코드까지 추적해 SHAM subset/전체 crop 해석 구분.
- V193 생성·보존 조건과 별도 검산 보고서 확인. 위 네 실제 PNG를 직접 열어 확인.
- 공식 선행연구/공식 benchmark 기록 읽기, 이 새 Markdown 조언 문서 작성.

**[확인] 하지 않은 것:**

- 모델·checkpoint 다운로드/설치/실행, 학습·추론·렌더·PnP·새 실험.
- raw ledger 압축 파일의 전량 재집계, 원 NPZ/RGB·checkpoint의 재해시, 공개 PASS 전부의 독립 재실행.
- 모든 사례의 시각 판독, 독립 물리 GT 검증, CI/유의성 검정, 새 real 평가.
- 기존 코드·감사 기록·원고·PDF·ZIP 수정, GitHub push, 외부 공유·투고, 다른 작업의 화면 조작.

**[확인]** 공개 보고서의 PASS는 그 보고서와 코드를 읽어 확인한 실행 기록이다. 이 문서에서 직접 검산했다고 한 것은 위의 저장 CSV 산술 범위다. 최신 main이 이후 바뀌면 이 문서는 계속 이 고정 snapshot의 검토로 남는다.

## 13. 근거 링크

아래 저장소 링크는 모두 동일 commit에 고정했다. 별도 표시가 없는 페이지 범위·물리 정답·미실행 실험을 추정하지 않았다.

### 저장소 원문

- [S1 — V189 보고서](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_sampled_domain_direct_end_20261024_v189/PORTABLE_REPORT_KO.md)
- [S2 — V190 실제 source 학습 보고서](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_paired_source_training_20261024_v190/PORTABLE_REPORT_KO.md)
- [S3 — V191 동결 실사 평가 보고서](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191/PORTABLE_REPORT_KO.md)
- [S4 — V192 source 범위·계보 감사](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_direct_end_coverage_audit_20261024_v192/PORTABLE_REPORT_KO.md)
- [S5 — V193 실제 source 대조쌍 확대 보고서](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_end_pairs_opposite_sham_20261024_v193/PORTABLE_REPORT_KO.md)
- [S6 — V191 18행 혼동행렬·precision/recall CSV](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191/fixed_direct_end_real_visual_v1/PRIMARY_ALL6_CM_PRECISION_RECALL_U.csv)
- [S7 — 실제 RGB96 spatial head](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_local_support_raw_head_20261024_v181/head_preparation/local_support_head.py)
- [S8 — V191 paired 회복·손상 CSV](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191/saved_real_head_evaluation_v1/current1784/PAIRED_RECOVERY_AND_DAMAGE.csv)
- [S9 — V191 평가 코드](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191/evaluation_preparation_a/evaluate_saved_direct_end_real.py)
- [S10 — V189 synthetic direct-end 정의](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/synthetic_direct_end_candidate.py)
- [S11 — V190에서 사용한 paired trainer](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_sampled_domain_direct_end_20261024_v189/paired_direct_end_fit_preparation_c/train_paired_direct_end.py)
- [S12 — V192 실제 sample 수 필드 정의, quantities()](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_direct_end_coverage_audit_20261024_v192/source_metadata_diagnostic_preparation_a/diagnose_stored_source_metadata.py)
- [S13 — 직접 본 V191 CASE_0002](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191/fixed_direct_end_real_visual_v1/CASE_0002/CARD.png)
- [S14 — 직접 본 V191 CASE_0996](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191/fixed_direct_end_real_visual_v1/CASE_0996/CARD.png)
- [S15 — 직접 본 V193 Q0277_D2 triplet](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_end_pairs_opposite_sham_20261024_v193/source_pair_visual_readout_v2_caption_fit/Q0277_D2/SELECTED_END_TRIPLET.png)
- [S16 — 직접 본 V193 Q0277_D2 전체 문맥](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_end_pairs_opposite_sham_20261024_v193/source_pair_visual_readout_v2_caption_fit/Q0277_D2/PRODUCER_RAW_CONTEXT_CONTACT.png)
- [S17 — V193 생성 코드](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_end_pairs_opposite_sham_20261024_v193/real_end_opposite_sham_preparation_c/build_real_end_opposite_sham_pairs.py)
- [S18 — V193 별도 저장결과 검산 보고](https://github.com/CanelE452/pallet-6d-pose/blob/d2fe21c6861bde8e148cec991345dedfaa2b2865/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_end_pairs_opposite_sham_20261024_v193/independent_actual_source_review_a/actual_review_v1/REVIEW.json)

### 공식 선행연구·benchmark

- **L1.** Ruyi Lian, Yuewei Lin, Longin Jan Latecki, Haibin Ling. *VAPO: Visibility-Aware Keypoint Localization for Efficient 6DoF Object Pose Estimation*. IROS2025. [저자 공개 본문, III-A/B](https://arxiv.org/html/2403.14559v2), [공저자 학회 정보](https://ywlincq.github.io/publications.html). 이 문서에서는 표면점·가시성·대칭 레이블의 원칙만 사용했다.
- **L2.** BOP official submission *PVNet-CVPR19/LM-O*, 2020-08-11. [PBR-only RGB 학습 계약](https://bop.felk.cvut.cz/sub_info/1147/). 원 CVPR2019 논문과 동일한 학습 계약이라고 쓰지 않았다.
- **L3.** Robert Geirhos et al. *Shortcut Learning in Deep Neural Networks*. Nature Machine Intelligence, 2020. [저자 공개 기록과 초록](https://arxiv.org/abs/2004.07780), DOI [10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z). 이번에는 공식 저자 공개 레코드/초록을 확인했으며 본문 전체를 새로 정독했다고 주장하지 않는다.
- **L4.** Gavin C. Cawley, Nicola L. C. Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR11(70):2079–2107, 2010. [공식 논문 레코드·초록](https://www.jmlr.org/papers/v11/cawley10a.html). 이번에는 이 공식 레코드를 확인했으며 전체 PDF 재독을 수행했다고 주장하지 않는다.
