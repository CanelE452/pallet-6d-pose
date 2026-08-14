# addon_v1 합성셋 검증 리포트 (16k 학습 전)

- 대상: `challenge/data/02_synthetic/training/addon_v1/` (numeric frame 6000장; `_summary*.json` 2개는 데이터 아님)
- 비교군: v3 10k (`challenge/data/02_synthetic/training/v3/`)
- 목적: 16k(=v3 10k + addon 6k) heatmap 학습 전 무결성 + 의도한 보강(작은/낮은카메라/외형) + v3 중복 판정
- 스크립트: `scripts/data_prep/validate/audit_addon_v1.py`, `overlay_addon_v1.py`
- 산출: `data/pallet/results/addon_v1_audit/` (json + 그림 + overlay)

## 판정: 16k 학습에 써도 됨 (PASS). 의도한 보강 = 작은 크기 + 낮은 elevation + 원거리 축에서 확실.

---

## 1. 무결성 (전수 6000장)

```
항목                         결과                              판정
──────────────────────────────────────────────────────────────────
RLE mask area 일치           6000/6000 = 100.00%               PASS
keypoint 재투영 (per-frame K) p50=0.010px / p99=0.056px /       PASS
                             worst=0.102px                      (<1px 기대 충족)
corner ordering (0123 v4)    6000/6000 = 100.00%               PASS
intrinsic 존재               6000/6000 (missing 0)             PASS
```

- [확인] 재투영 convention: `keypoints_3d_world`는 이미 world 좌표. `Xc = Rwc^T·(Xw - C)` (USD cam-to-world) 후 OpenCV flip `diag(1,-1,-1)`. 동일 코드로 v3도 p99=0.074px 통과 → convention 정합 확인.
- [확인] ordering: world 프레임에서 centroid 기준 상대좌표로 top{0,1,4,5} z=+0.114 / bot{2,3,6,7} z=-0.006, front{0,1,2,3}↔back{4,5,6,7} 반대편(~1.3m), 8=centroid. camera-facing X-flip(코너0이 카메라 근접측으로 스왑)은 face 그룹핑을 깨지 않음 — v3와 동일 `camera_dynamic_0123_v4`.
- ★ **intrinsic은 addon 내부에서 고정** (fx=605.91, cx=317.60, cy=256.29, 단일값). v3는 가변(fx 468~748). → 학습/평가 시 반드시 프레임별 K 사용(이미 검증 코드는 그렇게 함). addon은 1개 카메라 모델로만 렌더됨 — 다양성은 pose/거리/외형에서 나옴(intrinsic 다양성 X).

## 2. 분포 — addon vs v3 (★보강 확인)

```
지표                    addon (n=6000)          v3 (n=10000)         보강 방향
────────────────────────────────────────────────────────────────────────────
V<8 비율                20.97% (1258)           25.40% (2540)        ↓ (아래 주석)
  V hist (4/5/6/7/8)    373/82/669/134/4742     1427/191/711/211/7460
projected size           p5=51.6  med=244.3      p5=134.6 med=282.8   ★작은쪽 대폭 보강
  pc_diag (px)           p95=682.5               p95=864.5
  <134.6px(v3 p5) 비율   32.6%                   5% (정의상)          ★
  <100px 비율            24.6%                   ~0%                  ★
elevation (geometric)    p5=2.6  med=27.1        p5=9.4  med=31.7     ★낮은 카메라 보강
  (deg)                  min=-0.1                min=1.5
camera dist (from ctr)   p5=1.91 med=6.09        p5=1.80 med=4.72     원거리 보강(작은크기와 정합)
azimuth                  전 범위 균등 (보강 신규 클러스터 없음, v3도 이미 -180~180 커버)
```

해석:
- ★ **작은 크기 보강 = 확실.** addon pc_diag 분포가 <100px에 강한 피크(24.6%), v3는 사실상 0%. `cmp_pc_diag.png`에서 적색(addon)이 좌측(소형)에 집중 + 우측(대형>700px) 꼬리도 두꺼움 → 소형·대형 양쪽 확장.
- ★ **낮은 카메라 보강 = 확실.** addon elevation 0~10°에 v3 대비 큰 질량(p5 2.6° vs 9.4°, min -0.1° = 수평/약간 아래). 포크리프트 실배포 저카메라 시나리오에 정합. `cmp_elevation.png` 참조.
- **원거리 보강.** median dist 6.09m vs 4.72m → 작은 projected size의 원인. 의도된 "멀리서 작게 보이는 팔레트" 보강과 일치.
- **azimuth: 신규 클러스터 없음** — v3가 이미 전방위(-180~180) 커버. addon은 같은 범위 균등 샘플(보강 축 아님, 의도와 무관하면 OK).
- ⚠ **V<8 비율은 addon이 오히려 약간 낮음(21% vs 25%).** 단 구성이 다름: addon은 V=4(강한 truncation)가 적고(6.2% vs 14.3%) V=6(부분 가림/측면잘림)이 상대적으로 많음. → addon의 "truncation 보강"은 v3보다 *강하지 않다*. truncation을 더 늘리려는 의도였다면 미달. (작은크기·저elevation 보강이 주효과.)

## 3. v3 near-duplicate (누수)

- pose+scale 벡터(azimuth sin/cos, elevation, dist, projected scale)로 addon→v3 최근접거리.
- thresh=0.02(거의 동일 시점)에서 **22/6000 = 0.37%**. nearest median=0.146(전반적으로 충분히 다름).
- ★ 22장 직접 검증: 최근접쌍도 실제 카메라 상대위치가 0.06~2.0m 차이 + intrinsic 다름(fx 605.9 vs 490~748) + BG/cargo/조명 독립 랜덤. → **파일/프레임 복제 아님, 단지 유사 시점 우연 일치.** 누수 무시 가능(0.37%, 독립 렌더).
- 결론: **near-dup 누수 없음.** v3↔addon은 겹치는 시점 분포에서 독립 샘플.

## 4. 대표 overlay (눈검증 — 오케스트레이터 확인용)

`data/pallet/results/addon_v1_audit/` 아래:
- `montage_overlay_check.png` (5장 한눈에: RGB | mask_rle | cuboid 0123 v4 + centroid)
- 개별 full-res: `overlay_000924.png`(대형 clean), `overlay_001120.png`(소형 저카메라), `overlay_003920.png`(원거리 초소형 V=8), `overlay_001659.png`(occluder+cargo V=6), `overlay_003262.png`(강truncation V=4)
- 분포 그림: `cmp_pc_diag.png`, `cmp_elevation.png`, `cmp_mask_size.png`, `cmp_in_frame.png`, `cmp_dist.png`, `cmp_azimuth.png`, `cmp_neardup.png`
- 눈검증 결과: mask_rle가 팔레트 상판(보드 간격 보임)에 정합, cuboid wireframe이 팔레트를 정확히 감싸고 centroid 중앙, 0123 ordering 색상 일관. truncation/occlusion 케이스도 visible 코너 정합.

## 5. 종합 판정

- **무결성: PASS** (RLE 100%, reproj p99 0.056px, ordering 100%, intrinsic 정상).
- **의도 보강: 확인됨** — 작은 projected size(★대폭), 낮은 elevation(★), 원거리. azimuth는 신규 클러스터 없음(v3가 이미 커버). **truncation(V<8)은 보강 아님 — v3보다 오히려 적음**(작은크기·저카메라가 핵심 기여).
- **누수: 없음** (near-dup 0.37%, 전부 우연 유사시점·독립 렌더).
- → **16k(v3 10k + addon 6k) 학습에 사용 가능.** 단 두 가지 유의:
  1. addon intrinsic 단일 고정(fx=605.9) → 카메라 모델 다양성은 v3에서만 옴. per-frame K 학습/평가 필수(하드코딩 금지).
  2. truncation 강건성 추가가 목표였다면 addon은 기여 적음 → 별도 truncation 보강 필요할 수 있음.
```
