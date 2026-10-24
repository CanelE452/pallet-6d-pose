# V196 source HOLDOUT: 저장된 감독·입력대응 해석

고정 source weak D/E/D 과제의 실패는 남아 있다. C56은 COVER 4/15, C57은 7/15를 E로 응답했고, 두 모델 모두 BEFORE/SHAM은 28/30을 D로 응답했다. 이는 기존 source HOLDOUT에 대한 결과이며 자연실사/native 물리 gold 성능이 아니다.

전체360입력=15triplet×3variant×8native를 유지한다. known45/UNKNOWN315, 입력U0이다. 15triplet은6base·6component이며 독립15영상으로 세지 않는다.

## 전체15 triplet

B/C/S는 BEFORE/COVER/SHAM이며 저장 weak정답은 D/E/D다. `end96`은 원 RGB bilinear tap영향 sample수다. `거리96`은 q중심(47.5,47.5)에서 저장된 enclosure중심까지의 진단거리로 물리좌표 오차가 아니다.

|#|base|native|end96|COVER변화/SHAM변화|enclosure 폭×높이96|거리96|C56 B/C/S|C57 B/C/S|
|---|---|---|---|---|---|---|---|---|
|00|42|1|25|25/0|1.107×1.260|7.218|D/E/D|D/E/D|
|01|50|1|20|20/0|1.150×1.554|4.732|D/D/D|D/D/D|
|02|50|2|20|20/0|1.102×1.301|6.025|D/D/D|D/E/D|
|03|74|0|25|25/0|1.228×1.265|0.840|D/D/D|D/E/D|
|04|74|1|24|24/0|1.355×1.617|3.480|D/D/D|D/D/D|
|05|74|2|20|20/0|1.191×1.208|9.128|D/D/D|D/E/D|
|06|74|5|25|25/0|0.881×1.060|0.219|D/D/D|D/E/D|
|07|75|0|20|20/0|1.221×1.257|2.708|D/D/D|D/D/D|
|08|75|1|30|30/0|1.347×1.607|5.532|D/D/D|D/D/D|
|09|75|2|25|25/0|1.185×1.201|9.137|D/D/D|D/D/D|
|10|75|5|25|25/0|0.876×1.054|1.177|D/D/D|D/D/D|
|11|83|0|30|30/0|1.288×1.487|12.671|D/E/D|D/D/D|
|12|83|1|20|20/0|1.377×1.905|32.391|D/D/D|D/D/D|
|13|83|2|20|20/0|1.197×1.452|27.566|E/E/E|E/E/E|
|14|104|3|25|25/0|1.082×1.057|2.485|D/E/D|D/E/D|

## 확인된 점과 미확정 원인

- 샘플 부재나 SHAM의 동일끝 변경을 이번 15개 실패의 일반 설명으로 삼을 근거는 없다. 저장된 end-related tap20~30개가 모두 COVER에서 변경되고 SHAM에서는0개 변경이다. 생성기 기존 proof 및 C saved검산과 일치한다. tap영향샘플수는 continuous끝면적·RGB대비·사람판독성이나 실제 native정체성의 인증이 아니다.
- 고정 source weak D/E/D 과제에서 source-held 일반화 실패가 실제로 남는다. C56 COVER4/15, C57 COVER7/15만 E; 둘다 BEFORE/SHAM D28/30. 모든입력가용/U0이며 원source CM과 정확일치. 자연실사 가림전이 수치도 독립15영상 증거도 아니며 이비교만으로 SAM이나 한 조건의 인과효과를 분리하지 않는다.
- 등록 end enclosure와 원query 중심은 동일한 위치라는 보장이 없다. 기존96-space enclosure중심과 (47.5,47.5)의 거리가0.219~32.391px이다. 원q/L은그대로이며 native부위감독과 literalcenter정답은구분된다. enclosure중심거리 진단이며 물리corner오차/교정용GT가 아니다. 거리·크기가 실패원인이라는 추론은 하지 않는다.
- 제한된 source HOLDOUT 권한과 coverage를 유지해야 한다. 15triplet은6base/6component의45weaklabels이며 나머지315native행은UNKNOWN. knownnative4/6/7=0, real-endHOLDOUT=0이다. allnative0..7 원360입력 보존은 모든native에 감독이나 성능인증이 있다는 뜻이 아니다. 원source 설계는 이미 알려졌고 새blind검증은 아니다.

C56의 triplet패턴은 D/E/D3·D/D/D11·E/E/E1, C57은 D/E/D6·D/D/D8·E/E/E1이다. 모든15의 end20~30샘플은 COVER에서 변하고 SHAM에서 유지되므로 입력에 아무 차이도 없다는 설명은 맞지 않는다. 하지만 작은 continuous enclosure(폭0.876~1.377, 높이1.054~1.905)가 실패의 원인이라는 결론도 이 관찰에서 나오지 않는다.

known native 분모는 0:9, 1:15, 2:12, 3:3, 4:0, 5:6, 6:0, 7:0이다. 각 native 원입력45는 유지되지만 UNKNOWN을 D나 정답으로 채우지 않는다. 이 생성metadata에는 triplet별 material/aspect가 없어 base번호로 추정하지 않았다.

V194의 source TRAIN327 apparent응답과 component-HOLDOUT45는 다르다. HOLDOUT45는 전부 합성이며 real-end holdout0이다. 설계가 이미 알려진 source 검증이고 자연실사/unseen blind 검증·C54/C55 사후 독립검증으로 바꾸지 않는다.

## 다음에 구분할 수 있는 TRAIN-only 확인 제안

- 의도부위와 원crop의 감독대응이 실제 입력에서 일관적인가? 기존 TRAIN split의 전체 원순서를 먼저 고정해 원q96/등록finite-end/selectedpart 근거를 같은화면에서 대조하는 작은 read-only 검토를 제안한다. q/L·라벨·부위정의는 바꾸지 않고 correspondence UNKNOWN을 그대로 기록한다. HOLDOUT 행으로 새 ROI나 자료를 고르지 않는다. 선언된 part권한/입력대응의 공백과 올바른 약한과제에 대한 예측실패를 구분한다.
- positive tap존재가 실제 식별단서의 충분함까지 뜻하는가? 동일하게 미리 고정한 TRAIN 원쌍에 대해 기존 before/COVER/SHAM proof와 원96을 대조해 변화가 어디에 있으며 원부위가 주변과 식별되는지 확인한다. tapcount만으로 크기·대비 원인을 확정하지 않고 기록만 한다. 필요 수치의 정의/범위는 실행전 별도로 고정해야 하며 이번에는 새배열계산·파이프라인을 만들지 않는다. 실제RGB 가독성/semantic불확실성과 단순히 샘플이 없다는 주장 사이를 구분한다.

이 제안은 실행하거나 모델·문턱·아키텍처·자료·arm을 선택한 것이 아니다. 이번에는 생성된 CSV/JSON과 기존 source보고서만 읽었고 새배열계산·학습·추론·instrumentation 파이프라인은 만들지 않았다. A의 추가 실제 이미지뷰는0이며 root/C의 범위를 자신의 전수이미지검사로 주장하지 않는다.

[전체15 저장association](../all360_source_holdout_viewer_v1/ALL15_TARGET_ASSOCIATIONS.json) · [전체360 CSV](../all360_source_holdout_viewer_v1/ALL360_HOLDOUT_TARGET_MAPPING.csv) · [C 독립 저장검산](../independent_saved_viewer_review_c/REVIEW.json) · [V194 source보고서](../../query_visibility_source_group_split_direct_end_training_20261024_v194/ROOT_REPORT_KO.md)
