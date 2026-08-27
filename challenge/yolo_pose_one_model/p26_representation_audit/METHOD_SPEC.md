# METHOD SPEC — P26 representation audit (training-0)

## 질문

YOLO26 내부에 correct candidate 와 wrong candidate 를 구분할 정보가 있는가.
있다면 어느 stage 까지 살아 있고 어디에서 사라지는가.

## 고정한 것 (결과 보기 전)

```
checkpoint   Y0 vanilla 하나. Y2/Y0E/YN 대체 금지
inference    imgsz 640 / conf 0.001 / iou 0.7 / max_det 300 / one2one E2E / pad100 REFLECT_101
fuse         호출하지 않는다 (Pose26.fuse 가 module 을 제거)
학습          0 회
CASE         A REPRESENTATION / B CLS_HEAD / C OBJECTIVE / D NEGATIVE_BOUNDARY
             E DOMAIN / F NOT_LOCALIZED  — 정의는 결과 후 변경 없음
```

## tap (실제 module graph 를 감사해 정한 것, 하드코딩 아님)

각 pyramid scale 별로 **독립적인 sequential 경로**다. P3→P4→P5 를 순차 layer 로 읽지 않는다.

```
scale i:  neck_in = one2one_cv3[i][0] 의 입력 (= Detect.forward 의 x_detach[i])
          cls1    = one2one_cv3[i][0] 출력      (C=64)
          cls_pen = one2one_cv3[i][1] 출력      (C=64)
          logit   = one2one_cv3[i][2] 출력      (C=nc=1)
          pose_pen= one2one_cv4[i]  출력        (C=45, secondary)
```

`neck_in` 은 level 마다 채널이 달라(64/128/256) ALL scope 로 합치지 않는다.

## provenance

`get_topk_index` 를 감싸 anchor flat index 만 기록한다(반환값 불변).
flat index → (level, y, x) 는 `Detect.forward_head` 의 concat 순서(P3→P4→P5)로 푼다.
**독립 검증**: 매핑된 cell 의 `sigmoid(logit)` 이 final confidence 와 일치해야 한다.

## 그룹

```
S+           G38 val 프레임에서 IoU>=0.5 후보 중 conf 최고
R+           real128 프레임에서 IoU>=0.5 후보 중 conf 최고
RW           같은 real 프레임에서 IoU<0.5 후보 중 conf 최고 (가장 강한 distractor)
RW_RANKFAIL  correct 가 존재하는데 top1 이 wrong 인 프레임의 그 top1
RN           real negative 프레임의 conf 최고 후보 (threshold 올리지 않음)
```

## 지표

5-NN AUROC / nearest-centroid balanced accuracy / Fisher ratio / centroid cosine distance /
within-class dispersion. balanced subsample = min(nA,nB), seed 42, 100 bootstrap 95% CI.

★ `logit` 은 1 차원이라 5-NN 이 나쁜 추정량이다 — **스칼라 직접 AUROC** 를 정본으로 쓴다
(`LOGIT_DIRECT_AUROC.json`). 다차원 tap 은 5-NN 유지.

★ PCA 는 visualization only. gate evidence 로 쓰지 않는다.
