# 완료된 seed 1, 2 gradient 해석

완료된 8개 run만 읽었다. 각 run은 6,998 update·111,960장 노출이다. 각 checkpoint 내부 완료/EMA update/프로토콜 바인딩과 파일 SHA를 CPU에서 확인했다. 전체 6,998행 trace는 같은 seed의 네 군 및 v1 joint와 정확히 같고, 서로 다른 seed의 trace는 다르다.

## 전체 step PCGrad 기록

| 군 | seed | 적용 / 6,998 | E1 적용 / 3,499 | E2 적용 / 3,499 | 전역 line/stock norm 비 중앙값 E1→E2 |
|---|---:|---:|---:|---:|---:|
| pcgrad | 1 | 2,316 (33.10%) | 1,140 | 1,176 | 0.00621 → 0.04470 |
| pcgrad | 2 | 2,060 (29.44%) | 1,117 | 943 | 0.00610 → 0.03998 |
| balanced_pcgrad | 1 | 2,230 (31.87%) | 1,124 | 1,106 | 0.00642 → 0.00566 |
| balanced_pcgrad | 2 | 1,998 (28.55%) | 1,082 | 916 | 0.00604 → 0.00499 |

PCGrad 두 군의 적용 수는 실제 모든 step을 센 값이다. balanced/incidence는 수술하지 않고 8개 지정 지점만 측정했으므로 그 기록을 전체 충돌률로 해석하지 않는다. 관측된 두 PCGrad 군의 전역 norm 비는 고정 line 계수에서 epoch2에 커지고, balanced 계수에서는 비슷하거나 낮아졌다. 이는 규모 관찰이며 정확도 개선을 뜻하지 않는다.

## 국소 진단의 분모와 차이

각 run의 고정 step은 [2, 16, 256, 1750, 3499, 3500, 5249, 6998]이다(E1 5개/E2 3개). Step2 Hough stock norm0을 제외하면 Hough cosine은 7개만 유효하다. 아래 합계는 seed별 고정 지점의 단순 집계이며 무작위 표본의 빈도 추정이 아니다.

| 군 | 전역 음수 / 16 | Hough 음수 / 14 | pose+RLE Hough 음수 / 14 | E2 Hough line/stock norm 비 중앙값, seed 순서 |
|---|---:|---:|---:|---:|
| balanced | 8 | 5 | 6 | 0.334 / 0.273 |
| pcgrad | 10 | 7 | 7 | 3.473 / 1.370 |
| balanced_pcgrad | 7 | 8 | 8 | 0.361 / 0.321 |
| incidence | 5 | 8 | 6 | 21.378 / 23.446 |

전역 양수와 Hough 음수가 동시에 관측된다. Seed1 PCGrad step5249는 전역 +0.00703/Hough −0.59768이다. Seed2 incidence step5249도 전역 +0.01480/Hough −0.11123이다. 반대로 seed2 PCGrad 마지막 지점은 전역 −0.00832/Hough +0.14978이다. 등록한 전역 규칙과 국소 방향을 같은 것으로 취급하면 안 된다. Full stock에는 검출·visibility 등의 항도 포함되므로 pose+RLE만의 방향과도 구별한다.

## Incidence 및 측정 한계

Main 기록에는 incidence 자체의 분리된 gradient norm이 없다. 위 큰 비율은 semantic line/원래 stock의 비율이며 incidence/stock 비율이 아니다. 분리된 incidence 측정은 별도 seed1 smoke epoch1의 고정16장뿐(stock60.970, weighted incidence2.876, weighted line0.747)이다. 이를 다른 seed나 main 후기로 확장하지 않았다.

비-incidence 48개 고정 probe에서 효율 재구성과 독립 계산의 전역 충돌 부호가 모두 일치했다. 수치 오차는 JSON에 보존했으며 CUDA bit-exact나 나머지 step의 부호 일치는 주장하지 않는다.

성능·인과·실사 일반화 결론은 없다. 원본 seed1 해석과 학습 소스를 보존했으며 새 forward/GPU 사용도 없다. 모든 source/input SHA는 [SUMMARY.json](SUMMARY.json)에 있다.

세 번째 seed 완료 후에는 같은 스크립트에 `--seeds 1 2 3 --output-dir ../three_seed_gradient_summary`를 지정한다. 미완료 run 또는 기존 결과 덮어쓰기는 거부한다.
