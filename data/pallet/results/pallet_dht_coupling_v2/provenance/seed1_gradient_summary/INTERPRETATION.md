# Seed 1 gradient 기록 해석

완료된 네 실험군의 저장 기록만 비교했다. 각 군은 2 epoch·6,998 update·111,960장 노출이며, 전체 augmentation trace가 동일하다. 성능·인과·3seed 결론은 포함하지 않는다.

`full stock`은 box/class/점 위치/visibility/DFL/RLE 전체이고, `pose+RLE`는 별도 진단이다. 공유 범위는 실제 두 손실이 연결되는 215개 tensor·1,658,764개 파라미터이며 Hough 출력 projection/spatial_mix 등의 private 부분을 제외한다. 아래 값은 투영·global clipping 전이다.

## 전체 step을 기록한 PCGrad 두 군

| 실험군 | epoch1 적용 / 3,499 | epoch2 적용 / 3,499 | 전체 적용 / 6,998 | 선/stock 전역 norm 비 중앙값 E1→E2 |
|---|---:|---:|---:|---:|
| pcgrad | 1,140 | 1,176 | 2,316 (33.10%) | 0.00621 → 0.04470 |
| balanced_pcgrad | 1,124 | 1,106 | 2,230 (31.87%) | 0.00642 → 0.00566 |

단순 balanced/incidence 군은 수술하지 않았고 8개 지정 step만 기록했다. 이 두 군의 8개 기록으로 전체 학습 충돌률을 계산하면 안 된다. 고정 line 계수 PCGrad의 전역 norm 비 중앙값은 epoch2에 약 7.2배 커졌고, balanced_pcgrad에서는 비슷한 수준이었다. 이는 관측된 규모 차이이며 정확도 원인으로 단정하지 않는다.

## 같은 8개 지정 지점의 전역·Hough 비교

고정 step은 2, 16, 256, 1750, 3499, 3500, 5249, 6998이다(E1 5개/E2 3개). Hough stock gradient가 0인 step2는 cosine 분모에서 제외한다.

| 실험군 | full-stock 전역 음수 / 8 | full-stock Hough 음수 / 7 | pose+RLE Hough 음수 / 7 | Hough 선/stock norm 비, E2 세 지점 범위 |
|---|---:|---:|---:|---:|
| balanced | 3 | 3 | 2 | 0.225–0.821 |
| pcgrad | 4 | 3 | 3 | 2.656–3.552 |
| balanced_pcgrad | 2 | 4 | 3 | 0.136–0.509 |
| incidence | 2 | 4 | 3 | 14.848–28.493 |

초기 step16에서는 네 군 모두 Hough 부분의 선 norm이 stock의 약 24배이고 cosine은 약 −0.49였다. 이후 방향과 비율은 층·step에 따라 달라진다. 예를 들어 PCGrad step5249는 전역 +0.00703, Hough −0.59768이다. 등록된 전역 투영 규칙은 이런 국소 음수를 항상 작동 신호로 삼지는 않는다. 또한 balanced_pcgrad 마지막 step의 Hough cosine은 full-stock +0.12284, pose+RLE −0.55841로 서로 다르다. 점 손실만의 충돌과 전체 stock 손실의 충돌을 구별해야 한다.

## Incidence와 수치·범위 제한

Incidence의 실제 main epoch 평균 weighted incidence loss는 0.16207→0.05856, weighted line loss는 0.04274→0.03957이다. Main 기록에는 incidence 자체의 분리된 gradient가 없으므로, 위 표의 큰 line/stock 비를 incidence/stock 비로 읽으면 안 된다. 별도 preflight의 고정 16장·smoke epoch1 진단에서만 weighted incidence norm 2.876 / stock 60.970 / line 0.747이 측정됐다. 이는 main 후기의 규모 추정이 아니다.

세 비-incidence 군의 독립 재구성 검사는 24개 지점이다. 실제 relative-L2 차이와 absolute 차이를 JSON에 보존했고, 24개 모두 효율 계산과 독립 계산의 전역 충돌 부호가 일치했다. Incidence는 원 stock gradient를 직접 구하는 경로로 기록하며, 합손실에서 선을 빼 원 stock이라고 부르지 않는다. CUDA bit-exact 또는 미측정 step의 부호 일치를 주장하지 않는다.

모든 근거 파일 SHA와 source SHA는 [SUMMARY.json](SUMMARY.json)에 있다. 원 모델·학습·데이터·평가 파일을 수정하지 않았고 새 forward/GPU 사용도 없다.
