# Pairwise source VAL 독립 검산 및 해석

새 pairwise selector는 **source VAL 45개 조건 중 29개 통과, 16개 실패**로 탈락했다. 세 refiner seed 모두 R0_ONLY보다 회전 오차 P90이 커졌다. 따라서 이 방법의 새 real routing·real GT 평가를 진행할 수 없고, real T/R 개선을 입증한 결과도 아니다. 검산 자체는 PASS이며 방법의 성능 판정 FAIL과 구별한다.

근거는 [동결 평가 결과](SOURCE_VAL_GATE.json), [독립 검산 JSON](SOURCE_VAL_VERIFICATION.json), [선택 동결 기록](SOURCE_VAL_ROUTING_LOCK.json), [학습 완료 기록](TRAINING_COMPLETE.json)이다. 이 문서는 기존 결과의 확인이며, 모델·gate·threshold·seed를 새로 선택하지 않았다.

## 독립 검산 범위

- repository 평가·학습 helper를 import하지 않고, Python 정렬 후 순위 보간으로 median/P90을 다시 계산했다. `F.gate`와 `np.quantile`은 사용하지 않았다.
- 고정된 VAL 1,024개 전체 ID, TRAIN/VAL 분리, 8개 모델의 T/R 오차 배열, 45개 개별 조건을 검사했다. 모든 모델은 1,024개 pose가 유효하고 실패는 0개이며, conditional과 전체 모집단 요약은 동일하다.
- 저장된 float32 정규화 feature를 float64로 변환한 뒤 독립 broadcast 합산으로 4개 selector의 점수·선택 4,096건을 재현했다. 합산 순서의 수치 차이는 허용 오차 내이며 모든 선택 identity가 일치했다.
- protocol·checkpoint·receipt·feature·routing·metric 등 66개 고유 파일 binding의 SHA/크기를 검증했다. 새 fit은 3개, R0_ONLY 재사용은 1개이며 신규 최적화 objective 호출 합계는 1,257회였다.
- 이번 검산의 추가 학습·이미지 forward·PnP solve·GT 재평가는 각각 0회다. GT container는 기존 binding 확인을 위한 byte hash만 검증했으며 geometry 배열을 열어 GT 값을 계산하지 않았다. real reference는 읽지 않았다.

## 전체 VAL 수치

T 단위는 cm, R 단위는 degree이다. 낮을수록 좋다. 모든 행의 N=1,024, pose 실패=0이며, 특정 seed나 성공 프레임만 고르지 않았다.

| 모델 | T median | T P90 | R median | R P90 |
|---|---:|---:|---:|---:|
| R0_ONLY | 1.679493903 | 11.423339872 | 0.608126750 | 4.195164907 |
| UNION_s1 | 1.646521957 | 11.689973534 | 0.642442199 | 5.565992109 |
| UNION_s2 | 1.646304609 | 11.839805234 | 0.604724650 | 6.299050130 |
| UNION_s3 | 1.774252839 | 12.012423319 | 0.647675332 | 7.577449719 |
| R0_GEO | 1.674594149 | 10.932544237 | 0.613334153 | 4.328167849 |
| DIVERSE251_s1_GEO | 1.818473940 | 19.343094305 | 0.805513017 | 79.453475298 |
| DIVERSE251_s2_GEO | 1.900039532 | 18.537748907 | 0.775323011 | 73.676611147 |
| DIVERSE251_s3_GEO | 2.051196123 | 18.074772829 | 0.765876728 | 38.153660603 |

## 45개 조건의 확인

각 UNION seed를 R0_ONLY, 기존 R0_GEO, 해당 DIVERSE_GEO에 각각 비교한다. 비교당 T/R median 엄격 개선 2개, T/R P90 1.05배 이하 2개, 실패 수 비증가 1개로 총 45개다. 독립 계산한 각 Boolean과 요약 수치는 저장된 결과와 정확히 일치했다.

| Seed | R0_ONLY 실패 조건 | R0_GEO 실패 조건 | 해당 DIVERSE_GEO 실패 조건 |
|---|---|---|---|
| s1 | R median, R P90 | T P90, R median, R P90 | 없음 |
| s2 | R P90 | T P90, R P90 | 없음 |
| s3 | T median, T P90, R median, R P90 | T median, T P90, R median, R P90 | 없음 |

R0_ONLY의 R P90은 4.195164907°이고 허용 상한은 4.404923153°다. UNION s1/s2/s3는 각각 5.565992109° / 6.299050130° / 7.577449719°로, 기준 대비 각각 32.68% / 50.15% / 80.62% 증가했다. 세 seed 모두 회전 tail guard를 충족하지 못한다.

## 이전 수렴 CE 단계와의 동일성·선택 변화

[이전 수렴 단계 결과](../pallet_pose_selector_convergence_20261001_v1/SOURCE_VAL_ANALYSIS_KO.md)와 같은 source 입력·후보 pose를 사용했다. R0_ONLY와 4개 GEO baseline의 오차 배열 10,240개 scalar는 이전 단계와 **bit-identical**이다. 재사용 R0_ONLY의 weight float64, mean/std float32도 byte 단위로 동일하고, VAL 1,024개 선택 record 전체가 동일하다. 새 wrapper JSON에는 현재 protocol·재사용 provenance가 들어가므로 파일 byte가 원본과 같다는 주장은 하지 않는다. R0_ONLY는 R0_GEO와 별도 선택기이며 두 모델끼리 동일하다는 뜻도 아니다.

| Seed | 이전 DIVERSE 선택 | 현재 DIVERSE 선택 | 전체 pose 선택 변경 | Parent 변경 | W/D 이름 변경 |
|---|---:|---:|---:|---:|---:|
| s1 | 271 | 345 | 200 | 188 | 25 |
| s2 | 281 | 353 | 193 | 176 | 28 |
| s3 | 275 | 393 | 239 | 228 | 23 |

변경된 프레임에서 이전 수렴 CE 단계 대비 T/R 방향을 비교한 결과는 다음과 같다. 두 지표 동시 개선·악화는 아래 각 단독 지표 집계에 포함된다.

| Seed | 변경 N | T 감소 / 증가 | R 감소 / 증가 | T·R 모두 감소 | T·R 모두 증가 |
|---|---:|---:|---:|---:|---:|
| s1 | 200 | 83 / 117 | 88 / 112 | 45 | 74 |
| s2 | 193 | 76 / 117 | 84 / 109 | 40 | 73 |
| s3 | 239 | 94 / 145 | 104 / 135 | 42 | 83 |

선택하지 않은 후보 pose는 바뀌지 않았고, 선택이 같은 프레임의 T/R는 이전 단계와 정확히 같다. DIVERSE 선택 증가와 오차 악화의 동반 관찰은 확인되지만, 이것만으로 DIVERSE 선택 자체가 악화의 원인이라고 단정할 수 없다. 본 비교는 사후 관찰이며 seed·threshold·후속 방법 선택에 사용한 runtime gate가 아니다.

## 판정의 범위

이 실험은 WD 두 edge와 expert 두 edge를 학습하는 정해진 pairwise 목적함수 및 공유 linear94 모델의 검증이다. 이전 four-way CE와 목적함수·분모가 다르므로 서로 다른 training loss 숫자를 직접 비교해 성능 향상이라고 해석하지 않는다. Source 성능이 기존 기준에 미달했다는 결론은 명확하지만, 모든 pairwise 방법 또는 모든 후보 결합 방법이 불가능하다는 증명은 아니다.

같은 source VAL이 여러 사전 고정 방법의 통과 여부 확인에 재사용되었다. 독립 unseen test 성능으로 해석하지 않으며, 실제 작업 현장의 개선 주장에는 별도 검증이 필요하다. 현재 namespace에 새 real choices/routing/metrics/results 실행 산출물은 없다.

## 기록 순서와 검산 산출물

| 사건 | UTC 시각 |
|---|---|
| 마지막 fit/reuse receipt | 2026-10-01T00:55:09.691901+00:00 |
| source routing lock | 2026-10-01T00:55:27.422636+00:00 |
| 최초 source reference 값 접근 | 2026-10-01T00:55:42.919275+00:00 |
| source gate 완료 | 2026-10-01T00:55:43.483229+00:00 |

선택 동결이 source VAL GT 값 접근보다 앞선 순서를 검증했다. [독립 검산 JSON](SOURCE_VAL_VERIFICATION.json)에 45개 상세 판정, 66개 hash binding, 8모델 통계, 선택 변경 방향, 재사용 동일성 및 읽은 파일 경로를 저장했다. 원시 `data/pallet/results/.../SOURCE_VAL_METRICS.npz`와 feature·checkpoint cache는 로컬 연구 산출물이며, GitHub 문서에는 공개 JSON의 수치·hash로 추적할 수 있게 했다.

검산용 임시 스크립트는 `/tmp/pallet_pairwise_source_val_verify.py`이고 SHA256은 `ae9c2984c5bb644d24d3b1f68d6ae99c0bea41bdbcbb9b65d1e890a620565be4`이다. 구현 코드는 [source 평가기](../../../scripts/research/pallet_pose_selector_pairwise_20261001_v1/evaluate_source.py) 및 [학습기](../../../scripts/research/pallet_pose_selector_pairwise_20261001_v1/train.py)를 참조한다.
