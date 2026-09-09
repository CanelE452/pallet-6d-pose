# PURPOSE — joint G38 + legacy + legacy-tex one-shot training

[소비처] 논문 §data 의 "합성 데이터 구성" 절, 그리고 배포 모델 선택.
         지금까지 legacy v1/v2 는 **G38 를 먼저 학습한 뒤 finetune** 하는 형태로만
         들어갔다. 그 방식이 검출은 크게 올리고 keypoint 정밀도는 떨어뜨렸다
         (`legacy_v1v2_ft/RESULT.json`: cbox +13.6pp / corner median +18.1% 악화,
         guard 3개 전부 FAIL). **한 번에 같이 학습하면 그 trade 가 사라지는가**를 묻는다.

[문장]   "G38 · legacy v1/v2 · legacy-tex 를 **하나의 학습에 섞어** 처음부터 돌리면,
          순차 finetune 이 만든 '검출↑ / keypoint↓' trade 없이 검출 이득을 얻는다."
          이 문장이 참인지 거짓인지 이 run 하나로 가른다.

## 무엇이 다른가 — 순차 vs 동시

```
기존 (순차)   G38 38,002 로 학습  →  그 체크포인트에서 legacy 로 finetune
이번 (동시)   G38 38,002 + legacy 8,989 + legacy-tex 8,989 = 55,980 을
              pretrained yolo26n-pose 에서 한 번에 30ep
```

**치수 조건화는 넣지 않는다.** stock YOLO26n-Pose 그대로다.
치수 입력은 별도 트랙(`dimension_conditioning_probe`)에서 이미 STOP 판정이 났고
(oracle 천장 ADD-S AUC +0.05, 야간에서 B3 = B4 로 치수를 못 씀), 이 실험의 질문과
섞으면 두 변수가 한 번에 바뀌어 무엇이 원인인지 알 수 없게 된다.

## 데이터 (실측)

```
datasets/g38_legacy_v1v2_p0_tex20k
  train 55,980   G38 38,002 + P0 8,989 + TEX 8,989
  val    4,020   G38  1,998 + P0 1,011 + TEX 1,011
  라벨 32열 = 1 cls + 4 bbox + 27 (9kp x 3),  빈 라벨 0
  전부 심볼릭 링크 — 물리 복사 없음
```

★ 이 데이터셋은 **학습에 쓰인 적이 없다**(`args.yaml` 전수 스캔에서 참조 0건).
지금까지는 frozen feature 캐시 생성에만 쓰였다.

## 레시피 — Y0/Y0E 와 동일하게 고정

```
init      yolo26n-pose.pt (공식 pretrained)   ← 순차 FT 가 아니라는 뜻
epochs 30 · batch 32 · imgsz 640 · seed 42 · SGD
lr0 .01 · lrf .01 · cos_lr · warmup 3 · patience 0 · single_cls
fliplr 0 · flipud 0 · mosaic 0.3 · scale 0.25 · erasing 0.4 · close_mosaic 10
architecture 변경 0 · loss 변경 0
```

같은 레시피라 `Y26_G38_Y0_VANILLA_30EP_SEED42`(G38 만 38,002, 30ep)와
**데이터 구성만 다른 대조**가 된다.

## 판정지표 — 결과 보기 전 고정

주 비교는 **G38-only 30ep(Y0) 대비**다. legacy FT 는 epoch/init 이 달라 참고로만 둔다.

```
검출        det_recall_deploy(conf>=0.40) · top1-cbox(ALL / NIGHT)
keypoint    corner median · corner p90        ★순차 FT 가 악화시킨 축
FP 억제     neg AP · AUROC · FPR@TPR95
pose        정본 161 에서 ADD-S AUC (가능하면)

GAIN 판정
  G1  ALL top1-cbox      Y0 대비 +2pp 이상
  G2  corner median      Y0 대비 악화 <= +5%      ★순차 FT 는 +18.1% 였다
  G3  NIGHT top1-cbox    Y0 대비 하락 없음
  G4  neg AP             Y0 대비 하락 <= 0.02

  G1 통과 + G2 통과  ->  JOINT_BEATS_SEQUENTIAL   (검출 얻고 keypoint 안 잃음)
  G1 통과 + G2 실패  ->  SAME_TRADE_AS_FT         (순차와 같은 trade — 이득 없음)
  G1 실패            ->  JOINT_NO_DETECTION_GAIN
```

## 착수 시점에 아는 한계

- **legacy 는 v1/v2, 즉 사용자 팔레트다.** 이 모델은 평가셋 팔레트를 학습에서 본다 —
  논문 트랙(처음 본 팔레트 일반화)에는 쓸 수 없고, **challenge/배포 트랙 모델**이다.
- val 4,020 도 같은 세 소스에서 나왔다. 합성 val 은 이미 포화라 선택 근거로 쓰지 않는다.
- 30ep · seed 1개. seed 산포는 재지 않는다.
