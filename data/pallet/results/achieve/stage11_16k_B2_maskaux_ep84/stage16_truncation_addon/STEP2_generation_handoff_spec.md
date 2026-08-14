# STAGE16 Step2 — truncation_addon_v1 생성 핸드오프 스펙

> 목적: 이 Ubuntu 세션은 Blender 미설치 + addon/v3 생성기 부재로 Step2 실행 불가.
> 이 문서는 **Windows FoundationPose 세션(또는 자산 이관 후)** 에서 재유도 없이 즉시
> 생성하도록 모든 결정·파라미터·게이트를 고정한다. (작성 2026-06-29, Ubuntu 분석 세션)

## 0. 어디서 실행하나 (환경)

- ★ addon_v1/v3 를 만든 검증 파이프라인 = **Windows `C:/Users/User/Documents/GitHub/FoundationPose`**,
  Blender 5.1 + `synth_data_scene.blend` + camera_dynamic_0123_v4 생성기 + frame_meta randomizer(parking_lot/industrial BG·HDRI·cargo·floor).
- ★ **이 repo 의 `scripts/data_prep/blender/render_blender_data.py` 는 쓰지 말 것** —
  옛 Z=UP DOPE order(corner 0~3=top, L64-76)라 camera_dynamic_0123_v4 아님(v8 위험 레거시).
- 자산을 Ubuntu 로 이관하는 경우: `synth_data_scene.blend` + industrial glTF + HDRI(autoshop_01_2k.hdr 등) + occluder pool + SDG python 을
  `scripts/data_prep/blender/` 로 복사 → Blender headless(`blender -b ... --python`) 설치 후 실행(EEVEE ~0.3–0.5s/frame, 6000장 ~1시간).
- palletobj OBJ 는 이 repo 에 있음: `data/pallet/scan_cleanup/pallet_full.obj`.

## 1. 출력 계약 (addon_v1 schema 그대로 답습 — 새 설계 금지)

검증 기준 = `challenge/data/02_synthetic/training/addon_v1/000000.json`. 필드:
- `keypoint_convention = "camera_dynamic_0123_v4"` (front 0~3 = 카메라 근접 long side, {0,1,4,5}=top / {2,3,6,7}=bottom, 8=centroid)
- `cuboid_dimensions_m = [1.1, 1.3, 0.12]` (palletobj_v1)
- `projected_cuboid` (8점) — ★**clamp 금지**: 화면밖 음수/초과 좌표 그대로 저장(addon 예: corner5 = [-5.12, 293.39])
- `projected_cuboid_centroid`, `keypoints_3d_world`(9)
- `keypoint_in_frame` (corner별 bool), `num_corners_in_frame`
- per-frame `intrinsics` (RealSense D435i), `mask_rle`(COCO RLE, occlusion-aware visible), `frame_meta`(BG/HDRI/occluder/cargo/floor)
- projection convention(검증 `scripts/data_prep/validate/audit_addon_v1.py`): `Xc = Rwc^T (Xw − C)` → `FLIP = diag(1,−1,−1)`(USD→OpenCV) → per-frame K. round-trip p99 < 1px.
- 출력 위치: `challenge/data/02_synthetic/training/truncation_addon_v1/{i:06d}.png + .json + mask`.

## 2. 파이프라인 개조 (단 2가지)

addon 생성기를 그대로 쓰되:
1. **visibility gate 완화** — 기존: "전체 8코너 in-frame + unoccluded raycast"(FP 방지용).
   변경: **truncation 허용**(일부 코너 화면밖 OK). 단 FP 방지 유지 → 최소 가드(예: in-frame 코너 ≥ N AND centroid 부근 보임)일 때만 프레임 채택.
2. **per-corner in/out 라벨** — `keypoint_in_frame` + keypoint status:
   `in_front_of_camera`(z>0) / `inside_image`(0≤u<W,0≤v<H) / `heatmap_valid`(= inside AND in_front) / `V_geom` / `missing_corner_bitmask`. clamp 금지.

## 3. 생성 분포 (= "near왜곡 + truncation 통합", 사용자 결정 2026-06-29)

Phase A failure report(`v8lt_failure_aggregate.txt`, real V<8 N=17, **극소표본**) 기반. N=17 과적합 금지 → V4~V8 고루 커버.

- **near-distortion 을 base 외형으로 (V=8/V<8 공통)**: depth 1.4~3.5m(med~1.9 집중), elevation **저각**(아래 STAGE18 분포), projected size large 200~560px.
- **★elevation 분포 (STAGE18 정량 처방, 2026-06-29)**: rear 코너(4-7)가 ≲8°서 depth 붕괴/≳25°서 회복 확인. 원인=주로 **(가) 데이터갭**(학습 저앙각 거의0: v3<10°=0%, combined<10°=5.5%, 68%>25° vs real 90%+<10°). frsep↔err 무상관(0.06)=관측 degeneracy 아님=데이터로 풀림. → 분포 **elev <3°:10% / 3-8°:30% / 8-15°:30% / 15-25°:15% / 25+:15%** (부감 >25°는 이미 잘 되는 대역이라 축소). ★**전이대 8-15° 조밀 표집**(현 real·train 공백 → 이 데이터로 임계각 확정). ★rear 코너 병목이므로 저앙각+원거리+측면(edge-on) 최악조합 명시 표집.
- ⚠ **<3° 극저앙각은 진짜 (나) degenerate**(frsep≈10px, 앞뒤 2D 겹침=단안 원리적 한계) — 10%만, 과투자 금지. 운용선 카메라 살짝 올려 회피 or depth fusion.
- ★**"0,1 튐"은 전제오류**(STAGE17): 0,1(front-top)은 가장 정확, 진짜 병목=REAR(4-7). surface-point 아이디어는 front 아닌 **rear/깊이축**에 줄 것(주면). corner 0,1 겨냥 금지.
- **truncation 을 그 위에**: V_geom 분포 권장 (생성기가 분포 보고 ±조정, 근거 기록)
  ```
  V=8 ~20%   (근접 full-view, appearance 학습용 — 단순 anchor 아님)
  V=7 ~10%
  V=6 ~35%   (Phase A 지배적: real 65%)
  V=5 ~10%
  V=4 ~25%   (Phase A 2위: 한 면만 남는 최난 케이스)
  ```
- **잘리는 코너 = near face(idx 0~3) 위주** (Phase A: far face 거의 안 잘림).
- **border 방향**: L ≈ R 위주 + **Top 포함** + Bottom 드물게 (corner 누적 real L22/R18/T13/B3).
  (주의: 기존 memory `truncation-side-cut-bias` 는 "top 제외"였으나 Phase A 실측은 T13 으로 꽤 나옴 → top 포함으로 갱신.)
- bin 내 균형: missing-corner bitmask 다양, near/close-up/large/저각, azimuth 공백구간 보강(v3 audit: 4번째 긴변 0장), extreme close-up.
- appearance randomization: truncation 이 primary 지만 **B2 가 학습한 addon_v1/v3 수준의 BG/HDRI/material 다양성 유지**(과변경 금지 — Phase A 가 지목한 appearance gap 을 메우는 게 목적이므로 분포 일치가 핵심).

## 4. ★게이트 — 파일럿 먼저 (블라인드 6000장 금지)

1. **파일럿 300장** 생성 →
2. **자가 감사** (전부 통과해야 스케일업):
   - convention: projected_cuboid round-trip p99 < 1px, 0123 면 배치 정합
   - 분포: V_geom 히스토그램 / bitmask top-N / border 비율 / depth·elev·size 분포 = 의도분포 일치
   - 라벨: mask_rle area == json area, keypoint status ↔ 실제 in/out 검산, clamp 안 됨
   - ★눈검증: 오버레이 ≥20장(edge + 코너 in=초록/out=빨강 + centroid) 실제 near·저각·truncation 보이는지 육안
   - FP: 팔레트 없는데 라벨 있는 케이스 0
3. 감사 PASS → **6000장 스케일업**. FAIL → 멈추고 보고.
- 감사 산출: `data/pallet/results/stage16_truncation_addon/trunc_addon_v1_audit/`.

## 5. 다음 (Step3 학습) 주의 — BN freeze 사실 정정

- B2 recipe 의 "BN freeze" 는 **train.py 에 실제로는 없음** [확인]: `encoder_freeze_steps`(VGG params requires_grad 첫 N step 토글) + `encoder_lr_scale` 만 존재. train 모드라 BatchNorm running stats 는 계속 갱신됨.
  → Step3 에서 진짜 BN freeze 를 원하면 별도 구현 필요(현 recipe 는 안 함).
- Step3 replay: old 40 / v3 20 / addon 20 / **trunc_addon 20** (첫 run 보호적, 안정 시 25%까지). B2 init, heatmap+mask_aux(w0.01) 만, vector/offset 금지. val 4분리(Old/v3/addon/trunc).

## 6. 성공/실패 기준 (Step4, 재확인용)

- 성공: V=8 det 하락 ≤1%p, V=8 good 하락 ≤3%p, V<8 det 11.8%→**25%+**, honest full-8 reproj 개선, false accept 증가 無.
- 실패: V=8 det 3%p+ 하락 / V<8 honest 개선 無 / gate-only 만 개선(honest 그대로) / false accept 증가.
- ★ honest full-8 reproj 로 판정(Phase A: sel 7.2 vs honest 49.4 = coplanar gate 과대평가).
