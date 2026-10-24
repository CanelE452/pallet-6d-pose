# V191: 직접 끝부위 감독 이후의 고정 실사 벤치마크

직접 끝부위 감독을 더한 C55는 이전 C53보다 E 재현율이 22.2% → 41.1%로 높아졌지만, D 재현율은 75.7% → 72.4%로 낮아졌다. E90개 중 53개를 여전히 D로 통과시켜 전체 목표는 미해결이며 모델을 채택하지 않았다.

원 입력은 current223×8=1784와 public45×8=360이다. primary는 같은 원 q/RGB에 대응하는 인간 D1147/E90, 총1237행이다. 등록 프레임은 WOOD105/SQUARE118이며 SQUARE primary는117프레임이다. 원 q/L/native/가용성·crop을 바꾸지 않았다.

## 전체 6모델 primary 비교

CM의 행은 인간 D/E, 각 행의 열은 모델 D/E/U다. 정밀도는 결정된 해당 예측 분모, 재현율은 U를 포함한 해당 인간 D/E 전체 분모다. 기존 모델의 E 열은 원래 binary 응답이며 물리 원인 인증이 아니다.

|모델|N|D행 [D,E,U]|E행 [D,E,U]|D 정밀도|D 재현율|E 정밀도|E 재현율|U|
|---|---|---|---|---|---|---|---|---|
|v61|1237|[260, 887, 0]|[2, 88, 0]|99.2%|22.7%|9.0%|97.8%|0|
|legacy|1237|[655, 319, 173]|[3, 83, 4]|99.5%|57.1%|20.6%|92.2%|177|
|C52|1237|[821, 302, 24]|[52, 33, 5]|94.0%|71.6%|9.9%|36.7%|29|
|C53|1237|[868, 273, 6]|[70, 20, 0]|92.5%|75.7%|6.8%|22.2%|6|
|C54|1237|[698, 449, 0]|[55, 35, 0]|92.7%|60.9%|7.2%|38.9%|0|
|C55|1237|[831, 316, 0]|[53, 37, 0]|94.0%|72.4%|10.5%|41.1%|0|


C55는 E를 37/90 검출했고, 이전 C53의 20/90보다 늘었다. 그러나 53/90을 놓치며 legacy의 E 재현율 92.2%에도 미치지 못한다. 높은 D 통과율만으로 해결로 판단하지 않았다.

## C55 재질별 결과

|재질|primary N|D/E 분모|D행 [D,E,U]|E행 [D,E,U]|D 재현율|E 재현율|
|---|---|---|---|---|---|---|
|WOOD|645|616/29|[492, 124, 0]|[23, 6, 0]|79.9%|20.7%|
|SQUARE|592|531/61|[339, 192, 0]|[30, 31, 0]|63.8%|50.8%|


WOOD primary645(D616/E29)와 SQUARE primary592(D531/E61)를 각각 유지한다. [모든6모델·재질 표](fixed_direct_end_real_visual_v1/PRIMARY_ALL6_CM_PRECISION_RECALL_U.csv)와 [native8/재질 전체1784 분모](fixed_direct_end_real_visual_v1/ALL_ORIGINAL_NATIVE_MATERIAL_DENOMINATORS.csv)를 함께 제공한다.

## 위치·self·public weak는 별개다

기존 reference 오차≤3px인 human D는 461행이며 C55 응답은 D330/E131/U0다. 이461행은 manual-click near376 집단과 다르다. <=3px는 기존 참조 연관으로서 물리 좌표 gold나 인과 효과가 아니다.

|모델|별도 human S N|D 응답|binary hidden 응답|미결정|
|---|---|---|---|---|
|v61|21|9|12|0|
|legacy|21|13|3|5|
|C52|21|18|2|1|
|C53|21|15|6|0|
|C54|21|19|2|0|
|C55|21|16|5|0|


위 self 표는 별도 인간 S 집단에 대한 binary 응답이며 learned-self classifier가 아니다. [전체 self 표](fixed_direct_end_real_visual_v1/SELF_SEPARATE_BINARY_RESPONSES.csv)를 보존했다.

public360은 중심 owner 약관찰 known204/UNKNOWN156이며 원 inputU108, center4 invalid112가 별개다. 아래 T/non-T와 owner F 지표를 current native E90 성능과 동일시하지 않는다. legacy public owner 경로는 없어서 만들지 않았다.

|모델|T 정밀도|T 재현율|non-T 재현율|owner F 정밀도|owner F 재현율|
|---|---|---|---|---|---|
|C52|39.5%|70.8%|85.6%|82.1%|24.0%|
|C53|53.3%|66.7%|92.2%|75.4%|47.9%|
|C54|30.6%|79.2%|76.1%|77.4%|25.0%|
|C55|46.5%|83.3%|87.2%|77.8%|51.0%|
|v61|30.4%|29.2%|91.1%|미정|미정|


[public weak 전체 분모·혼동표](fixed_direct_end_real_visual_v1/SUMMARY_AND_TABLES.json) · [public weak 요약 CSV](fixed_direct_end_real_visual_v1/PUBLIC_WEAK_CENTER_OWNER_ONLY.csv)

## 고정 갤러리에서 볼 수 있는 것

기존 68개 고정 crop 중 아래6개는 설명용 링크다. 새 사례를 렌더하거나 input을 재중심화하지 않았다. 그림의 cyan q는 표시 사본에만 있고 clean96는 원 모델 입력 픽셀이다. 실제 원인 판단은 local crop만으로 확정하지 않는다.

### original 0

고정된 원 입력과 6모델 출력을 함께 비교한다. 무효 padding이 있으면 자연 배경으로 해석하지 않는다.

![fixed original 0](fixed_direct_end_real_visual_v1/CASE_0000/CARD.png)

### original 2

기존 human D인데 C55는 E다. 작은 reference 오차 집단에도 D 오탐이 남으며 local crop만으로 원인을 단정할 수 없다.

![fixed original 2](fixed_direct_end_real_visual_v1/CASE_0002/CARD.png)

### original 256

사전에 고정된 입력 예시다. 표의 인간 D/E와 각 모델 응답을 그대로 보며 새 원인·라벨을 만들지 않는다.

![fixed original 256](fixed_direct_end_real_visual_v1/CASE_0256/CARD.png)

### original 300

기존 reference 오차가 약103px인 사례다. E를 맞혔다는 수치만으로 현재 q의 의도한 끝부위 외부가림을 인증하지 않는다.

![fixed original 300](fixed_direct_end_real_visual_v1/CASE_0300/CARD.png)

### original 840

SQUARE 고정 crop이다. 색이나 중심의 literal owner를 native 외부가림 gold로 바꾸지 않는다.

![fixed original 840](fixed_direct_end_real_visual_v1/CASE_0840/CARD.png)

### original 996

기존 human E이며 C54/C55 모두 D다. reference 오차 약1.14px에서도 E miss가 남지만 그 참조가 물리 pixel gold라는 뜻은 아니다.

![fixed original 996](fixed_direct_end_real_visual_v1/CASE_0996/CARD.png)

[68개 전체 portable 갤러리](fixed_direct_end_real_visual_v1/PORTABLE_GALLERY_KO.md) · [HTML 갤러리 원문](fixed_direct_end_real_visual_v1/index.html) · [고정 선택표](fixed_direct_end_real_visual_v1/FIXED_CELL_SELECTION.csv)

## 확인한 범위와 남은 문제

- 이 벤치마크의 인간 D/E/S 의미 권위는 유지한다. 새 direct source 표적은 synthetic near-anchor 끝부위 규약과 public RGB 약관찰이며, 실제 native semantic label과의 동일성 gold로 승격하지 않았다.
- C55와 C53 비교에는 감독·학습 조건 변화가 함께 있다. 이 결과로 SAM, 특정 데이터, head 변화의 원인을 분리하거나 새 학습/문턱/모델/architecture/data를 선택하지 않았다.
- source 직접감독306행의 apparent fit은 학습 노출 결과다. 자연실사 성능 증거가 아니며 source 학습·보고 단계 V190과 이 frozen 실사 벤치마크를 구분한다.
- current223과 publicheld45는 반복 개발 노출되어 fresh blind 검증이 아니다. legacy의 과거 realCAL 이력을 숨기지 않으며 새 C54/C55는 source-only로 학습했다.
- 원 current1784의 input U45와 public360의 input U108을 보존했다. primary 표의 U는 해당1237행 안의 모델 응답 U이며, 전체 원 입력U 분모와 다르다. legacy U177과 기존 C52/C53 regional NONE도 자체 의미 그대로다.
- human self S는 E에 합치지 않고 별도 binary response로 보고한다. E 응답 자체가 물리 외부가림 원인이나 외부/self 분리 성공을 뜻하지 않는다.
- 전체 WOOD/SQUARE와 native0..7 목표는 유지한다. 고정68갤러리는 post-result strata의 사전 고정 first6 규칙이며 prevalence 대표표본이 아니다. Root24/C12 실뷰는 일부 중첩하며 전68/1784 이미지를 모두 봤다는 주장은 없다. 브라우저 렌더는 검증하지 않았다.

[독립 수치 검산](independent_saved_result_review_b/actual_review_v1/REVIEW.json) · [root24 실제 그림 QA](ROOT_ACTUAL_FIXED24_CARD_VISUAL_REVIEW.json) · [C12 실제 그림 QA](independent_actual_visual_review_c/REVIEW.json)

[예측 코드](prediction_preparation_a/infer_both_frozen_direct_end_heads.py) · [평가 코드](evaluation_preparation_a/evaluate_saved_direct_end_real.py) · [갤러리 코드](visual_readout_preparation_a/render_fixed_direct_end_real.py) · [보고서 생성 코드](report_preparation_a/build_top_level_readout.py)

[정확 수치·binding JSON](ROOT_RESULTS_KO.json) · [phase 범위](ROOT_PHASE_RESULT.json) · [오프라인 HTML](ROOT_READOUT_KO.html)
