# 리프터 PnP 보조 8코너 부분 평가

고정 12장·96점의 기존 PnP 보조 주석과 원시 Base/N3 예측을 비교했습니다.
공식 120장·24장 반복 검수와는 별도인 탐색적 기하 참조 패널입니다. 모든 점이 직접 보인다는 의미나 독립 실측 정답이라는 의미가 아닙니다.

| 방법 | 유효 일치점 중앙값(px) | P90(px) | PCK≤10px 전체 96점(%) | 유효점 / 실패점 |
|---|---:|---:|---:|---:|
| Base | 3.974 | 9.643 | 91.667 | 96 / 0 |
| N3 | 4.184 | 8.977 | 94.792 | 96 / 0 |

잘못 선택한 대상과 결측 점은 전체 분모에 남겨 PCK 실패로 처리합니다. 중앙값/P90은 실제 계산 가능한 일치점에서만 계산하며 실패율을 함께 공개합니다.

- 중앙값의 차이 median(N3)−median(Base): 0.210 px.
- 짝지은 차이의 중앙값 median(N3−Base): 0.090 px.
- 점별 개선/악화/동일: 46 / 50 / 0.
- 대상 판정: {"same": 12}.
- 주석 출처: {"manual_click": 66, "pnp_projected": 30}.

이전 모델 예측 노출 여부는 UNKNOWN으로 남깁니다. 현재 대상 대응 화면에서 선택 박스를 본 이력은 실제 검수 기록에 별도로 저장됩니다.
리프터 정지 잡음과 독립 물리 T/R은 x이며, 공식 가시 코너 정확도도 기존 계약의 완료로 바꾸지 않습니다.
원고 삽입은 아직 하지 않았습니다. 새 학습·새 추론·장비 제어는 모두 0회입니다.

## 모든 고정 표본 이미지

왼쪽: 참조(청록=직접 클릭, 보라=PnP 생성), 가운데: Base, 오른쪽: N3. 12장 전부 표시하며 성능으로 골라내지 않았습니다.

### 173507:56

![173507:56](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/173507_56.png)

### 173507:1652

![173507:1652](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/173507_1652.png)

### 173507:3655

![173507:3655](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/173507_3655.png)

### 174126:13

![174126:13](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174126_13.png)

### 174126:419

![174126:419](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174126_419.png)

### 174126:744

![174126:744](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174126_744.png)

### 174342:41

![174342:41](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174342_41.png)

### 174342:1190

![174342:1190](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174342_1190.png)

### 174342:2447

![174342:2447](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174342_2447.png)

### 174925:32

![174925:32](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174925_32.png)

### 174925:1002

![174925:1002](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174925_1002.png)

### 174925:1892

![174925:1892](/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182152558451Z/overlays/174925_1892.png)

