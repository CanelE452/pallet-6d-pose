# B — 작은 normalized Smooth L1 좌표 보조항

## 선택 근거와 경쟁 설명

MAIN의 고정 TRAIN R0/REF·clear/occluded 8 branch에서는 RLE aggregate min0가 활성화되었다. 기존 location 경로는 살아 있다. 별도 C3 문제frame은 highest-score가 실제 assignment와 같았고 corner4 exp(-e)≈.00222로 location감쇠가 있지만 RLE gradient는 남는다. 따라서 전체 실패를 손실포화로 확정하지 않는다. identity 미확정·의사타깃 오류·적은 exposure·동결범위/표현한계가 경쟁 설명이다.

질문은 '손실 버그를 고치면 해결되는가'가 아니라 '실제 감쇠 가능성이 있는 좌표경로에 작은 비포화 신호를 더하면 학생 T/R이 개선되는가'이다. 설치criterion을 hook하여 값동등·finite difference·ignore0·state보존을 확인했다. 큰오차curve는 기전, 작은실제probe는 관찰이며 전수원인 증명이 아니다.

## 한 변경과 값의 근거

현재 location/presence/RLE를 그대로 유지하고 location에 λ×SmoothL1 하나만 더한다. σ는 현재 KeypointLoss의 고정 σ, area는 detached target box면적이다. z=(pred−target)/sqrt((2σ)^2·area·2), 원래OKS e=||z||². β=1은 각 |z_axis|=1 전환이며 pixel단위10px/12px 튜닝이 아니다. 동일 supervised(v=2)와 원래location의point-count 정규화를 사용하고 v1의 모든새gradient는0이다.

λ=0.16500387762358062: MAIN_R0 clear TRAIN probe에서 기존location 두branch의 weighted head-gradient norm의10%가 되도록 한 값. 두branchparameter support는분리되어 제곱합sqrt로측정했다. 평가 T/R/PCK/AUC는값선택에쓰지않았다. 이 비율은 제한적운영강도이며 최적값을주장하지않는다. 모든batch/후반epoch에서10%가되도록동적재조정하지않는다.

## 대조와 조건

NEW_RAW_LOSS / NEW_REF_LOSS 각320update, R0초기값·기존217/512/512·vanilla affine/HSV·AdamW lr1e−5·seed42·poseflowonly·last-only 동일. A의가림은끄고 extra항만변경한다. 기존클리핑·RLEclamp·학습률·source비중은안바꾼다. baseline과의전체parameter gradient배율변화는실제TRAIN진단으로기록하며 큰이득이면단순배율control의필요성을검토한다. 별도architecture/inference비용0.

새 줄: 동일감독점의normalizedSmoothL1보조항. 재사용 줄: 실제criterion을상속하며super계산과RLE/존재항/geometry를보존. 남은공백: kptlocation과RLE가결합된이현재계약에서단순좌표항의추가가치는아직검증안됨.

## 평가 / 다음 결정

동일사전lock의Plastic99 T/R와full128/66·severity/recording·tail·coverage·source를보고한다. A의관측은B계수조정에쓰지않는다. 반대방향결과도남긴다. 두원리가어느축에서든유효하고조합질문이정당하면세번째cycle에서A+B RAW/REF누락셀만실행한다. 무조건결합하지않으며효과없는축구제를위한sweep은없다. 의미있는gain은추가seed와matchedcontrol예산으로확인한다.
