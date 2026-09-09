# ADAPTER_MAPPING — 확인한 소스 key → 새 export key

새 export 는 이 패키지가 정의한 인터페이스다. 아래 왼쪽은 **실제 파일을 읽어 확인한**
기존 key 이고 오른쪽이 새 계약이다.

| 새 `Observation` 필드 | 실제 출처 | 확인 위치 |
|---|---|---|
| `features` = (P3, P4) | `HoughFeatureFusion.forward(features)` 의 `features[0]`, `features[1]` — **블록 입력**, residual 되먹임 전 | `pallet_dht_joint_v1/hough_block.py:113-121` (`if len(features)!=3: raise ... expected P3, P4 and P5`), 채널 `channels=(64,128,256)` |
| `raw_to_feature` | `raw_to_input_affine` (structured-v2 packed, FP64) ÷ stride, offset `(-0.5,-0.5)` | `pallet_dht_structured_v2/cache.py` INPUT_SPECS `raw_to_input_affine`; offset 계약은 `pallet_dht_decoder_probe_v1/cache.py:sample_visual` (`grid=2*input/size-1`, `align_corners=False`) |
| `content_valid` | affine 역변환으로 raw 사각형 `[0,W-1]x[0,H-1]` 안에 드는 feature 중심 | `pointline_v4.adapter_helpers.content_mask_from_raw_rectangle` + 실제 raw H/W (`MANIFEST.json` `width`/`height`) |
| `baseline`, `point_valid`, `point_conf` | `baseline_points`, `point_valid`, `point_conf` | structured-v2 `INPUT_SPECS` (예측 기반 유효성, GT 아님) |
| `layouts`, `candidate_valid` | `build_proposals(...)['layouts'], ['valid']` — native(clean) 상태 | `pallet_dht_structured_v2/proposals.py`; 64슬롯·identity index0·centroid는 원본 복사 |
| `line_h`, `line_logits`, `line_valid` | `line_h`, `peak_logits`, `peak_valid` | structured-v2 `cache.pack_observation` (raw 공간 homogeneous line; 저장된 `line_fusion__line_peaks_h_raw` 와 동일성 검증됨) |
| `diagonal` | `diagonal` = `hypot(width,height)` (raw px) | structured-v2 `INPUT_SPECS` |
| supervision `points`/`supervised`/`matched` | `targets['points']` / `targets['valid']` / `targets['matched']` | structured-v2 `TARGET_SPECS`; **별도 파일**, Observation 으로 전달되지 않음 |

## 새로 수행한 것과 재사용한 것

- **새로 수행**: 동일한 검증된 predictor(`CanonicalPredictor`, checkpoint SHA
  `0960fb32…`)로 2,560장 실제 forward 를 다시 돌려 **native P3** 를 캡처했다.
  기존 캐시에는 P4 만 있었다.
- **재사용**: baseline 점·역할선·교점·합성 GT 는 frozen structured-v2 pack 그대로다.
- **엄격 parity**: 새로 캡처한 P4 를 frozen 캐시 P4 와 비교해 **2,560/2,560 bit-exact**
  (max abs diff 0.0). 따라서 새 P3 는 그 baseline 을 만든 것과 같은 forward 산물이다.
  이 결과는 `EXTRACTION_COMPLETE.json` 이 기록한 수치환경(`cudnn_allow_tf32=True`)을
  그대로 재현했을 때 얻어졌다. 더 엄격한 설정(TF32 off)에서는 max abs diff 6.2e-2 로
  bit-exact 가 아니며, 이 사실을 덮지 않는다.

## 알려진 한계

- feature 평면은 batch 처리를 위해 P3 `(64,80,80)`, P4 `(128,40,40)` 으로
  **오른쪽·아래 zero padding** 했다. affine 은 그대로 두고 `content_valid` 가
  padding 과 letterbox 여백을 배제한다 (실측: 416x640 입력에서 P4 유효 612 cell,
  기하 예상 616).
- `content_valid` 는 raw 화소가 존재하는 cell 표시일 뿐, backbone 수용영역에 이미
  들어온 padding 영향까지 제거하지 못한다. 가시성 감독도 아니다.
- feature index offset 은 `-0.5` (기존 검증된 repo sampler 계약). `AFFINE_PROBE.json`
  의 실측은 stride(1/8, 1/16)는 확인하지만 `-0.5` 와 수용영역 중심 기준 `-0.469`
  (0.03 cell 차)를 구분할 해상도가 없다. 이 상수는 네 군에 공통이라 군간 차이를
  만들 수 없다.
