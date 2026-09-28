# LOSS SIGNAL AUDIT — 실제 TRAIN graph, optimizer 0

결론: MAIN 소표본에서는 RLE의 branch aggregate clamp가 실제로 켜졌지만 위치 신호는 남았다. C3 실패점 중 corner4의 위치 신호는 강하게 감쇠했으나 RLE/combined 신호가 남았다. 따라서 “C3가 전체 gradient 소실 때문에 실패했다”는 결론은 지지하지 않는다. 작은 보조 회귀항 B는 **능력 시험**이지 확정 원인 수정이 아니다.

## 범위와 검증

실제 설치 Ultralytics 8.4.60 + 기존 TrueIgnorePoseLoss26/E2ELoss를 hook했다. 복제한 유사 criterion으로 원래 결과를 대체하지 않았다. 3 checkpoint, clear/occluded 각 batch4(REAL2+SOURCE2), optimizer 생성/step/fit 0, GPU 구간 4.521s. 879개 state tensor/checkpoint 모두 bit-exact, 원 loss와 hook loss 6회 bit-exact, ignored 전 channel gradient 최대0, actual fixed-assignment finite difference64/64 및 leaf descent16/16 통과. CPU 단위시험5개 통과.

C3는 원래 real-affineOFF의 과거 조건이다. MAIN은 기존 vanilla affine/HSV를 유지하며 실제 A의 SharedOcclusion과 원 index seed를 재사용했다. MAIN REAL2는 이미 고정한 A preflight에서 처음 mask가 가능했던 사례이며 prediction/error/DEV로 고르지 않았다. batch4는 전체 batch16 학습 역학을 대체하지 않는다. 원 recipe AMP=False와 같은 FP32이며 BN/EMA checkpoint는 고정했다. raw arrays, RGB 및 실제 좌표는 private data namespace에만 있다.

## 실제 branch 결과

| context | branch | supervised assigned | e max | RLE before→after | location head norm | RLE head norm |
| --- | --- | --- | --- | --- | --- | --- |
| C3_MANUAL9 CLEAR | one2many | 260 | 6.4144 | 2.48689→2.48689 | 0.56009 | 5.83633 |
| C3_MANUAL9 CLEAR | one2one | 26 | 6.1093 | 1.94223→1.94223 | 4.87457 | 64.58013 |
| C3_MANUAL9 OCCLUDED | one2many | 260 | 6.4144 | 2.74534→2.74534 | 0.56297 | 6.03514 |
| C3_MANUAL9 OCCLUDED | one2one | 26 | 6.1093 | 2.26267→2.26267 | 4.91216 | 65.53862 |
| MAIN_R0 CLEAR | one2many | 340 | 0.5906 | -2.03209→0.00000 | 4.41885 | 0.00000 |
| MAIN_R0 CLEAR | one2one | 34 | 0.1009 | -2.97921→0.00000 | 0.81378 | 0.00000 |
| MAIN_R0 OCCLUDED | one2many | 340 | 0.6859 | -1.75923→0.00000 | 5.23379 | 0.00000 |
| MAIN_R0 OCCLUDED | one2one | 34 | 0.1871 | -2.79922→0.00000 | 0.88000 | 0.00000 |
| MAIN_REF CLEAR | one2many | 340 | 0.4660 | -2.18559→0.00000 | 0.45562 | 0.00000 |
| MAIN_REF CLEAR | one2one | 34 | 0.0427 | -3.05599→0.00000 | 3.07194 | 0.00000 |
| MAIN_REF OCCLUDED | one2many | 340 | 0.5382 | -2.02191→0.00000 | 0.48963 | 0.00000 |
| MAIN_REF OCCLUDED | one2one | 34 | 0.1678 | -2.87355→0.00000 | 3.87027 | 0.00000 |


MAIN 8/8 branch-context에서 pre-clamp RLE가 음수여서 coordinate뿐 아니라 sigma/flow gradient도0이었다. 이 8/8은 선택한 배치의 결과이지 전체 TRAIN 빈도가 아니다. C3는 RLE clamp가 꺼져 있고 location/RLE 둘 다 양의 target-radial gradient를 보였다. 실제 측정한 supervised point의 ±100 표준화 clamp는0건이다.

## C3 실패3의 identity/assignment 분리

원래 ID wood_day_01:002141의 corner0/4/5를 그대로 사용했다. native cached deployment error와 현재 padded640+HSV probe 오차는 입력·단위가 다르므로 동일 수치로 취급하지 않는다. 현재 입력 최고 confidence one2one anchor는 실제 TAL assigned anchor와 같다. corner의 nearest matching이나 GT 기반 candidate 교체를 하지 않았다.

| original corner | cached native error px | probe input640 error px | e | exp(-e) | location xy grad | RLE xy grad | combined xy grad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 129.185 | 96.427 | 2.15366 | 0.116058 | 0.149305 | 0.158370 | 0.264425 |
| 4 | 212.418 | 162.408 | 6.10929 | 0.002222 | 0.004815 | 0.125236 | 0.129695 |
| 5 | 147.259 | 111.198 | 2.86396 | 0.057042 | 0.084623 | 0.274885 | 0.328566 |


corner4 exp(-e)=0.00222는 위치항 감쇠의 실제 증거지만 RLE xy gradient0.12524와 combined0.12970이 남는다. 세 점 모두 RLE standardized clip이 없고 coordinate-leaf combined 신호가 target 방향이다. manual_click provenance는 물리 signed-axis/corner 정답을 확정하지 않는다.

## 단위·감쇠·clamp

GT는 normalized→입력640px→각 anchor stride 좌표로 변환되고 GT box area도 stride²로 나뉜다. 기존 location은 `e=||delta||²/((2σ_OKS)²·area·2)`, `1-exp(-e)`이며 9점 σ_OKS=1/9다. 이는 RLE의 예측 sigma와 다르다. RLE는 stride residual을 sigmoid(scale logit)으로 나눈 뒤 각 축 ±100 clamp, learned flow+residual density, supervised point 평균을 거친 뒤 **branch 전체 scalar**를 min0 clamp한다. visibility는 ignore(v1)를 분자·분모에서 빼고 v0 negative를 유지한다.

99개 artificial single-point curve는 실제 installed KeypointLoss/RLE/learned C3 flow를 호출했다. native256px 단일축 예시에서 e8.4406, exp(-e)0.0002159, location gradient0.0001495 대 normalizedSmoothL1 0.11916이었다. 이는 mechanism isolate이며 관측 빈도/실패 인과가 아니다. 한 축 ±100 clamp 시 그 축 residual 경로는0이어도 flow coupling으로 다른 축/scale gradient는 남을 수 있다. 실제 curve에는 축별 gradient도 저장했다.

## B 최소 후보와 TRAIN calibration

`z=(pred-target)/sqrt((2σ_OKS)²·detached(area)·2)`에 coordinate-wise SmoothL1을 더한다. β=1은 각 |z_axis|=1 전환이며 ||z||²=e; 일반2D에서 radial sqrt(e)=1 전환이라고 하지 않는다. mask와 K/n_supervised × mean(anchor,K) denominator는 기존 location과 같다. area/σ detach, v1은 xy/confidence/scale 모두0, v0 BCE 기존 동작 보존. coefficient0에서 원 location/visibility/RLE tuple bit-exact를 확인했다.

MAIN R0 clear TRAIN batch의 weighted head norm: 원 location 4.493162234907, unit supplement 2.723064633158. `λ=.1*original/unit=0.16500387762358062`로 초기 보조 신호를 원 위치항의10%로 잠근다. branch .8/.2, pose gain12, kobj1, rle1.0를 적용한 per-image objective 기준이며 실제 batch loss에는 batch4가 곱해진다. 양 branch location parameter support가 분리되어 joint norm은 branch norm 제곱합의 제곱근이다. PCK10/DEV나 후속 B 결과로 β/λ를 선택하지 않았다. B fit은 부모의 별도 protocol 승인 대상이다.

API: `make_supplement_criterion(model, 0.16500387762358062, beta=1.)`. top-level SupplementLoss가 기존 criterion을 먼저 호출하고 location 보조항만 추가한다. `_setup_train` 뒤 model.criterion을 직접 설치하면 이미 존재한 criterion 누락을 피한다. EMA validation에 설치할 때는 EMA 자신의 flow를 참조하는 criterion을 별도 생성한다. E2E update schedule, RLE clamp/gain, LR, mask/source ratio, assignment 및 detector는 유지한다.

## 해석 제한

가장 가까운 원인 후보를 분리한 작은 TRAIN 진단이며 보조항이 자연 Moderate+Severe의 T/R을 개선한다는 보장은 없다. aggregate clamp 활성만으로 제거가 옳다는 결론도 없다. frozen EMA/소배치 graph는 online optimizer 전 궤적이 아니다. coordinate-leaf target descent는 공유 head weight 업데이트 후 모든 점 개선을 보장하지 않는다. 실제 물리 정답은 여전히 미확정이다.

정확한 구현/설치본/입력/checkpoint/hash와 세부 값: [LOSS_SIGNAL_AUDIT.json](LOSS_SIGNAL_AUDIT.json). 실행 core는 GPU 진단 이후 변경하지 않았고 report/미사용 helper 정리 후 source 전체를 후속 B lock용으로 바인딩했다.
