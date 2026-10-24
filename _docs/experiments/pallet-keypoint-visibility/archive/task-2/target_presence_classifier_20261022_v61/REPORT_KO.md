# v61: 예측 키포인트 위치의 대상 팔레트 presence 분류

**사용자가 정정한 목표에 맞춰 새 presence head 1개를 실제로 학습하고 검증했다.** 목표는 초기 모델의 query 영상 좌표에 보이는 대상 팔레트 픽셀이 있는지 여부다. 자세 개선, 올바른 코너 ID, 물리적 DIRECT 가시성, virtual ray 차단은 서로 다른 목표다. v60은 다른 목표의 결과로 그대로 보존하며, 그 자세 gate 실패를 이번 목표의 실패로 해석하지 않는다. 이번에는 PnP·자세 평가를 새로 실행하지 않았다.

## 정답·입력 계약

```python
q = initial_predicted_keypoint[k]       # native first8; centroid 제외
px, py = floor(q + 0.5)                # 원영상 정수 pixel center, 반올림 tie는 위쪽
presence = visible_target_mask_M4[py, px] > 127
```

- query와 crop 중심은 **실제 초기 예측 좌표**이며 GT 투영점으로 중앙을 바꾸지 않았다. 좌표 보정/회귀도 없다.
- finite query, 영상 내부, 가장 가까운 픽셀도 영상 내부, ROI의 유효 source pixel 존재를 valid로 고정했다. 밖의 query를 border로 clamp하거나 대상 부재 정답으로 강제하지 않는다. invalid는 coverage에서 따로 보고하며 확정 presence 판정을 내리지 않는다.
- 같은 원영상 해상도의 visible mask M4에서 **한 픽셀**을 읽는다. 넓은 patch 비율이나 3×3 majority vote가 정답이 아니다. 중심의 3×3이 모두1이면 CLEAR_INTERIOR, 모두0이면 CLEAR_EXTERIOR, 섞이면 BOUNDARY_UNCERTAIN, stencil이 영상 끝에서 잘리면 IMAGE_BORDER_UNCERTAIN으로 표시한다. 이 속성은 loss 가중치·정답을 바꾸지 않으며 전체 valid query를 학습한다.
- 1,003장 전부에서 PNG uint8/해상도/실제 값 분포를 읽었다. 값은0/1/2/254/255 범위였으며 `>127` 이진화 면적이 저장 annotation의 amodal/visible 면적과 일치하고 M4⊆M0임을 확인했다. grayscale 값은 확률로 해석하지 않았다. 모델 입력 resize/crop은 기존 affine/P3 ROI cache 계약을 그대로 썼고 label sampling은 resize 전 원좌표에서 수행한다.
- M0=1,M4=0은 대상 단독 실루엣에 속하지만 최종 장면에는 대상이 보이지 않는 픽셀의 보조 속성이다. M0=0은 배경·구멍 등을 포함한다. 이를 특정 가리는 물체 owner나 3D 코너별 external/self 정답으로 확대하지 않는다.
- 초기 기하 B/I/S는 초기 pose와 convex cuboid에서 계산한 **분석용 속성**이다. 실제 RGB의 대상 존재 정답이 아니다. 3D 자기 가림 코너의 투영 위치라도 앞쪽 다른 팔레트 면이 그 픽셀을 차지하면 이번 presence 정답은1일 수 있다.

## 실제 학습

- 기존 합성 TRAIN768/VAL235, 유효 TRAIN query5,673(존재2,342/부재3,331), VAL1,742(존재701/부재1,041). 전체 native query는 TRAIN6,144/VAL1,880이며 valid coverage92.33%/92.66%다.
- 기존 동결 YOLO26 P3의 9×9 ROI+pixel-valid 채널을 같은 query에서 재사용했다. backbone·BN은 갱신하지 않았다. 새 head는 Conv65→32/GELU/Conv32→32/GELU/GAP/Linear1, 28,033 parameter다. v60 ray head 가중치는 읽지 않고 새로 초기화했다.
- seed20261018, AdamW lr0.001/wd0.0001, batch32 frame, unweighted BCE, 20epoch/**480optimizer update**, **fit1회**다. 첫 update UTC2026-10-22 11:21:28, 합성 VAL BCE 최저 epoch9를 선택하고 threshold0.5와 함께 동결했다.
- 실사 이미지/파생 특징/레이블을 학습·calibration·normalizer·threshold·checkpoint 선택에 사용하지 않았다. 기존 real-CAL state/covariance/H1도 이 head의 입력에 사용하지 않았다. K·등록 치수는 기존 초기 query를 만든 경로의 물리 입력이며 새로운 실사 잔차 적합은 없다.
- 기본 YOLO/N3의 retained 합성 학습 출처는 v54 모듈별 감사와 v60 source binding을 유지한다. 범용 사전학습 전체가 합성이라고 주장하지 않는다. TRAIN/VAL의 source scene ID·RGB hash는 분리되지만, 전체 paired-render/자산 lineage와 upstream detector-validation 노출·반복 VAL 사용 한계는 남아 있다. 새 독립 test의 결과가 아니다.

## 주결과: presence 분류

threshold0.5이며 positive는 **보이는 대상 팔레트 존재**다.

| VAL 집합 | n | AP | Precision | Recall | TP | FP: 거짓 존재 | FN: 거짓 부재 | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 전체 valid |1,742|0.8081|0.7880|0.6205|435|117|266|924|
| 명확한 내부·외부 |896|0.9554|0.9122|0.8774|322|31|45|498|
| 경계·영상 끝 불확실 |846|0.5175|0.5678|0.3383|113|86|221|426|

전체 VAL AUROC0.8335, accuracy0.7801, balanced accuracy0.7541, BCE0.4911이다. TRAIN-prior 상수의 VAL BCE0.6742/AP0.4024, always-absent accuracy0.5976과 비교해 합성 분류 feasibility screening을 통과했다. 이는 실사에서 충분한 정확도 또는 배포 안전성 통과를 뜻하지 않는다. 목표 관련 신호를 학습할 수 있다는 첫 소규모 결과다.

**경계가 주된 약점이다.** 전체 valid query의846/1742(48.56%)가 경계·영상 끝 층이고, false absence221/266와 false presence86/117이 그 층에 있다. 명확한 내부·외부만 보여주며 전체 성능을 대체하지 않는다. 개별 point와 frame 내 query는 상관돼 있으므로1,742개 독립 장면처럼 통계적 일반화를 주장하지 않는다.

가능한 속성별 VAL 결과:

- M0-only 부재93개 중 FP8/TN85. M0 밖 배경·구멍 부재948개 중 FP109/TN839.
- 초기 기하 B:1,284개, precision0.4944/recall0.2649, FP91/FN247. 초기 기하 I:458개, precision0.9301/recall0.9479, FP26/FN19. B/I는 관측 정답이 아니라 초기 투영의 외곽/내부 역할이다.
- 초기 convex-self-hidden=True인280query에도 M4 존재224개가 있다. 이는 이번 pixel-presence 목표가3D 코너 가시성과 다르다는 점을 보여주며, target의 실제 자기 가림을 틀렸다고 판정한 결과가 아니다.

동일 valid VAL query 중 ray차단 positive인데 M4 존재인18개, ray차단 negative인데 M4 부재인833개도 있다. 따라서 v60 occlusion_fraction 결과를 presence 결과로 재명명할 수 없다.

## 실사 적용과 그림

동결된 동일 head를 목재105·정사각형118의 기존 검증된 query/ROI cache에 forward만 적용했다. 새 학습·새 backbone forward·좌표 보정·PnP는0회다. **실사 대상 pixel mask 정답이 없어 presence precision/recall/AP를 계산하지 않았다.** 실사 prediction1784개는 `REAL_QUERY_PREDICTIONS.csv`에 있으며 정답 상태는 UNAVAILABLE이다.

`Real_Presence_Overlays.pdf/png/svg`는 목재/정사각형 각1개를 예측 category 내 recording/ID 순으로 고른 사례다. 예측 결과에 따라 그림을 선택한 사후 진단이며 대표 성능이나 검증 통계로 해석하지 않는다. 녹색 원은 predicted presence, 자홍×는 predicted absence이고 숫자는 query ID와확률이다. 사람이 보기에 맞는 것처럼 보이는 점도 독립 정답이라고 선언하지 않았다.

`Synthetic_Presence_Cases_v2.pdf/png/svg`는 clear TP/TN/FP/FN과 boundary FP/FN6query를 같은 RGB/crop/scale에서 보여준다. 원 query 청록+, 실제 한 픽셀 노랑 사각형, M4 흰색=대상/검정=부재, 회색=원영상 밖 padding이다. 첫 figure의 흰 배경 padding이 target 흰색과 섞일 소지가 있어 표시만 수정한 v2를 만들었으며 원본 그림도 보존했다. 모델·label·선정·수치는 바꾸지 않았다.

## 검산·비용·보존

PIL `getpixel`로 전체 valid7,415query의 한 픽셀 정답을 독립 재생했고, scikit-learn AP/AUROC/혼동행렬이 원 집계와 일치했다. input mask/ROI/head/protocol hash와 모든 출력 행은 manifest에 연결한다.

마스크 계약·label 생성12.159초, GPU head480update+매epochVAL 및 checkpoint 선택1.394초, 실사 cached-feature head inference0.704초였다. 이는 캐시 이후의 측정 구간이고 Python 시작/모델 로딩/전체 영상→query 생성 비용을 포함하지 않는다. 기존 frozen SYNTH 특징 추출19.855초/REAL4.247초는 v60에서 이미 수행한 비용으로 별도다. 이번에 backbone을 다시 실행하지 않았다. end-to-end 가속을 주장하지 않는다.

원본 데이터, v60/이전 부정 결과와 원고는 보존했다. 현재 이presence 실험은 학습·합성 검증·실사 정성 forward까지 완료됐다. 업로드·공개·원고 변경은0회다. 남은 검증은 독립 합성 장면 또는 독립 실사 target pixel-mask 주석이며, 현재 실사 정량 정확도는 미확인이다. 이번 결과에 맞춰 자동으로 다음 모델이나 threshold 탐색을 시작하지 않는다.
