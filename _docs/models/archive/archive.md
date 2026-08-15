# 초기 실험 모델 (Archive)

Step 1 합성 데이터 파이프라인 개발 과정에서 학습한 모델들.
mixed_v1 이전의 실험이며, 현재 active하게 사용하지 않음.

## 모델 목록

```
모델               Weight 경로                                      Epochs   학습 데이터                       이미지 수   초기 weight                        비고
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
pallet_category    weights/misc/pallet_category/final_net_epoch_0060.pth  60      Isaac Sim (train/)                ~2,000     scratch (VGG-19 pretrained)        최초 pretrain baseline
pallet_v11         weights/misc/pallet_v11/                               121     Isaac Sim (train/)                 4,000     pallet_category ep60 → ep115       v11 렌더링 개선 반영
pallet_v11_far     weights/misc/pallet_v11_far/                           121     Isaac Sim (train/ + far)           6,000     pallet_v11 ep105                   원거리 데이터 추가
blender_v1         weights/blender_v1/final_net_epoch_0060.pth       60      Blender (blender_v1_train/)        3,600     scratch                            Blender 단독
combined_v1        weights/misc/combined_v1/final_net_epoch_0060.pth      60      Isaac 6K + Blender 3.6K            9,600     scratch                            소스 비율 불균형 (1.67:1)
```

## 평가 결과 (mixed_v1_val 800장, fair eval)

```
모델               PCK@3px   PCK@10px   PnP Rate   Reproj mean   Vol Ratio   Vol<20%
──────────────────────────────────────────────────────────────────────────────────────────
blender_v1         0.540     0.810      70.8%      83.0 px       1.985       50.1%
combined_v1        0.570     0.830      94.4%      67.3 px       1.810       53.4%
```

> pallet_category, pallet_v11, pallet_v11_far는 fair eval 미실시

## 주요 교훈

- **blender_v1**: 자체 val에서는 최고 PCK/100% PnP, fair eval에서 Isaac 프레임에 약함. real 근거리 정면 약함
- **combined_v1**: PnP 성공률 높지만 소스 비율 불균형(Isaac:Blender=1.67:1)으로 Vol Ratio 불안정
- **pallet_v11 계열**: Isaac Sim 단독 → 렌더링 개선만으로는 한계. Blender 소스 추가가 효과적
- **결론**: Isaac:Blender 1:1 균형이 핵심 → mixed_v1로 수렴
