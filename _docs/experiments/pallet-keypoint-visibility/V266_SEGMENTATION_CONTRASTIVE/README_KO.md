# V266 — 영역 분할과 픽셀 대조 학습 비교

**이번 구성은 외부 가림 개선에 실패했습니다.** 보임 recall은89.63→92.59/92.50%로 올랐지만 외부 가림 recall은25.56→22.22/22.22%로 내려갔습니다. 두 새 방식 모두 이전 누락67점 회복0, 잔존67, 새 누락3점입니다. 새 모델 채택과 이 결과 뒤의 추가 fit은 없습니다.

합성110개 base(학습86/검증24)만으로 두 추가 모듈을 각1000업데이트 학습했습니다. 실사223장은 새 학습·통계·문턱·체크포인트 선택에 사용하지 않고 final과1784예측을 동결한 뒤 평가했습니다. 이223장은 반복 개발 평가이며 독립 블라인드 검증은 아닙니다. PnP와 위치 출력 변경은 없습니다.

![실제 보임 복구·콘 가림 누락·새 누락](artifacts/V266_SEGMENTATION_CONTRASTIVE/RESULT_OVERVIEW_KO.png)

## 동일 평가의 혼동행렬과 precision/recall

PRIMARY1237점은D1147/E90, 목재D616/E29와정사각형D531/E61입니다. 행별 열은D/E/U입니다. D=보임,E=외부가림,U=보류이며 자기 가림은E에 합치지 않았습니다. 전체1784점의입력불가45U는모든방식에서유지됩니다. 아래PRIMARY는모두입력가능해U0입니다.

| 평가군 | 방식 | D→D/E/U | E→D/E/U | D P/R% | E P/R/F1% | U% |
|---|---|---|---|---|---|---|
| 전체 | 현재 V264 | 1028/119/0 | 67/23/0 | 93.88/89.63 | 16.20/25.56/19.83 | 0.00 |
| 전체 | 분할 추가 | 1062/85/0 | 70/20/0 | 93.82/92.59 | 19.05/22.22/20.51 | 0.00 |
| 전체 | 분할+대조 | 1061/86/0 | 70/20/0 | 93.81/92.50 | 18.87/22.22/20.41 | 0.00 |
| 목재 | 현재 V264 | 510/106/0 | 18/11/0 | 96.59/82.79 | 9.40/37.93/15.07 | 0.00 |
| 목재 | 분할 추가 | 541/75/0 | 18/11/0 | 96.78/87.82 | 12.79/37.93/19.13 | 0.00 |
| 목재 | 분할+대조 | 540/76/0 | 18/11/0 | 96.77/87.66 | 12.64/37.93/18.97 | 0.00 |
| 정사각형 | 현재 V264 | 518/13/0 | 49/12/0 | 91.36/97.55 | 48.00/19.67/27.91 | 0.00 |
| 정사각형 | 분할 추가 | 521/10/0 | 52/9/0 | 90.92/98.12 | 47.37/14.75/22.50 | 0.00 |
| 정사각형 | 분할+대조 | 521/10/0 | 52/9/0 | 90.92/98.12 | 47.37/14.75/22.50 | 0.00 |

## 실제로 확인한 두 병목

- **실사 분할 전이:** G370의 콘 흰띠, G908·G1515의 콘받침을 대상 팔레트로 나눕니다. 같은 콘의 표면을 외부 물체로 일관되게 유지하지 못합니다. 실사 dense GT가 없어 실사분할IoU는 주장하지 않습니다.
- **분할→키포인트 가림 판단:** 합성DEV1928은 q주변외부물체확률.751/.732인데도 P(E).220/.225로 두방식모두D입니다. 분할 자체가 맞아도 최종판단이 실패합니다. 이는 현재 처리층의 문제이며 분할방법 전체의 불가능성을 증명하지 않습니다.
- 전체1784개에서 바뀐 판정은 기존E→D51/50점뿐이며 새E발견은0입니다. 기존분류기 점수에 추가모듈이 기대는 경로가 남아있습니다. 추가가림누락은 정사각형G1515·1536·1677입니다.
- G996은 사용자의 시각판단과 기존E주석이 다른 사례로 유지했습니다. 주석을 바꾸거나 명백한오류근거로 단정하지 않았습니다.

## 정확히 무엇을 학습했는가

기존DINO·heatmap·초기q/crop·V264분류기는동결합니다. 추가분할처리층은16×16픽셀소유지도(target/foreign/background)와, 원래query·고정heatmap위치·전체에서 읽은소유확률로 기존E점수에보정값을더합니다. 두새방식은동일23204파라미터·같은초기값/학습표본이며 대조손실(.1,tau.1)만다릅니다. 기준선대비추가용량·감독·업데이트도달라집니다. CLIP Surgery 자체실행은아닙니다.

렌더러part_id0…18은대상팔레트,999는외부물체,-1은실제배경입니다. 모호함/영상밖은IGNORE입니다. 외부물체가crop에있다는이유만으로 nativeE정답을부여하지않습니다. 원래nativeD/E는유지합니다. 440개buffer와base분할을검산했고,7040행을같은crop좌표/resize로softlabel에맞췄습니다. FIT실제학습D4572/E188이며 학습배치는D16/E16입니다.

합성DEV외부물체분할IoU는65.88→67.43%로소폭올랐지만 실사가림개선으로이어지지않았습니다. 분할IoU는합성16×16fractional토큰기준이며 실사성능이아닙니다.

## 이미지·표·원인·재현

- [전체 설명 HTML](artifacts/V266_SEGMENTATION_CONTRASTIVE/ROOT_READOUT_KO.html): 폴더를다운로드해브라우저로열면연결된그림과표를볼수있습니다.
- [외부가림90점전체와고정97사례](artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/INDEX_KO.html), [목재콘흰띠G370](artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/CASE_G0370.png), [새누락G1515](artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/CASE_G1515.png).
- [합성RGB·정답·예측대조](artifacts/V266_SEGMENTATION_CONTRASTIVE/syn_visual_v1/INDEX_KO.html), [합성정답대조PNG](artifacts/V266_SEGMENTATION_CONTRASTIVE/syn_visual_v1/SYN_RGB_GT_PRED_FIXED6_KO.png), [합성readout실패1928](artifacts/V266_SEGMENTATION_CONTRASTIVE/syn_visual_v1/SYN_ROW1928_READOUT_CASE.json).
- [평가군36행CSV](artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/CLASSIFICATION.csv), [전체1784점](artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/ALL1784_COMPARISON.csv), [외부가림90점전이](artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/EXTERNAL90_TRANSITIONS.csv), [자기가림별도](artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/SELF_RESPONSES.csv).
- [학습전조건](artifacts/V266_SEGMENTATION_CONTRASTIVE/PROTOCOL_BEFORE_FIT.json), [두추가모델checkpoint·학습로그](artifacts/V266_SEGMENTATION_CONTRASTIVE/syn_training_v1), [실행안내](artifacts/V266_SEGMENTATION_CONTRASTIVE/RUN_GUIDE_KO.md).
- [독립데이터검토](artifacts/V266_SEGMENTATION_CONTRASTIVE/DATA_INDEPENDENT_REVIEW.json), [독립학습검토](artifacts/V266_SEGMENTATION_CONTRASTIVE/TRAINING_INDEPENDENT_REVIEW.json), [독립수치검산](artifacts/V266_SEGMENTATION_CONTRASTIVE/ROOT_INDEPENDENT_METRICS_CHECK.json).

CPU분류검산은전체1784개에서같은판정을재현했고 고정32개입력의전체denseforward도확인했습니다. 지표재현의주요6출력은바이트동일합니다. 모든학습/평가코드·checkpoint·실험조건·입력SHA·저장분할지도·실제crop·compact재현입력을포함합니다. 대형1.38GB합성/350MB실사특징과원래rendererbuffer는경로/SHA로참조하므로 저장소만으로전체학습을즉시재실행할수있다고주장하지않습니다. 저장소만으로compact분류/지표검산은가능합니다.

다음명령은`artifacts/V266_SEGMENTATION_CONTRASTIVE`에서실행합니다. 새출력폴더이름을사용하세요.

```bash
python -B replay_predictions_v266.py --out replay_cpu_check
python -B evaluate_v266.py --replay-metrics --inputs evaluation_v1/compact_inputs --out replay_metrics_check
```
