# win_search2k — Windows SEARCH2K 인수인계

작성 2026-08-14 (Ubuntu 머신에서). 대상: Windows `E:\CODING\GitHub\FoundationPose` 쪽.

## 결론 먼저

**이 zip 만으로는 SEARCH2K 가 돌지 않는다.** 산출물 3종은 전부 들어 있으나
**프레임(png/json/mask) 은 한 장도 없다.** 원본 데이터셋이 이 리눅스 머신에서
사라졌기 때문이고, 남아 있는 대체 셋은 **다른 이미지**로 판명됐다.

프레임은 Windows 쪽 원본에서 조립해야 한다 (아래 §3).

## 1. zip 내용 — `win_search2k_artifacts.zip` (22M, 22 files)

```
파일                        크기        SHA256(앞 16)      비고
──────────────────────────────────────────────────────────────────────
line_internal_split.csv    1,340,165   70ba7f1e8832bb0c   ★동결본, 기대값 일치
step_25545.pth            20,328,994   b37dfc2617a7e0f0   tag=P0_AUG_ONLY step=25545
line_search2k_manifest.csv    16,007   1bf294f31499ce9a   2000행 ★러너가 읽는 것
line_dev512_manifest.csv       4,103   93e5ce8567dea0ab    512행 ★러너가 읽는 것
search2k_manifest.csv         16,007   40402afed5beff39   2000행 (구세대, 아래 주의)
그 외 CSV 16개                                            audit/manifest 일체
```

zip 자체 SHA256 `e637e9bd461fdb7a0c08eb8df787bdd866075ab52d370ba0ac454b6b02fb7008`

### checkpoint 내용

```
keys   hough_decoder_sha, lambda_cons, late_a1, model, population_sha,
       runner_sha, seed, sigma_map100_pixel, split_sha, step, tag,
       target_semantics_sha
tag    P0_AUG_ONLY   step 25545   model ✅   late_a1 ✅   optimizer 없음(무관)
```

`split_sha` / `population_sha` / `runner_sha` 가 들어 있으니 Windows 쪽에서
정합성 대조가 가능하다.

### ⚠️ manifest 두 종이 다르다

`search2k_manifest.csv` 와 `line_search2k_manifest.csv` 는 **다른 파일**이다
(index 겹침 231/2000). `supporting_line_map_capacity.py:308-309` 는

```python
dev       = V2.manifest("line_dev512")
train_ids = V2.manifest("line_search2k")
```

즉 **`line_` 접두사 쪽**을 읽는다. `search2k_manifest.csv` 는 이전 세대 파일이며
참고용으로만 동봉했다.

## 2. 프레임이 없는 이유 — 원본 소실 + 대체 셋 비동일

### 2.1 원본은 2026-08-14 에 삭제됐다

```
data/pallet/training_data/pallet6d_v2_10k   17G  삭제 (사용자 직접, history:203)
~/Downloads/pallet6d_v2_clean_10k_part1~4.zip  27G  삭제 (history:218)
```

전 디스크 검색으로 재확인 [확인]:

```
all/ + index.csv 레이아웃 폴더     0건
*pallet6d* / *prod10k* 이름        0건
휴지통(6.6G)                       무관한 것들만
```

### 2.2 남은 40k 셋은 "같은 씬, 다른 렌더" 다 — 대체 불가

현재 유일본은
`data/pallet/training_data/paper_release/v2_prod40k_clean_merged/` (40,000 프레임).
`records.jsonl` 의 `_src_shard` 가 옛 셋과 같은 shard 이름
(`v2_prod10k2_s8101~8105`, `v2_prod10k3_s8201~8205`)을 가리켜서 복원 가능해
보였으나, **픽셀을 대조하니 다른 이미지였다.**

```
검증 항목                              결과              해석
────────────────────────────────────────────────────────────────────
shard / pallet_type (n=2429)          불일치 0          메타는 승계됨
visible_kp / bbox_min_side (n=2429)   불일치 0          메타는 승계됨
label 해상도 (n=2429)                 불일치 1588       ← 라벨이 다름
이미지 luma median gray (n=150)       일치 3            ← 이미지가 다름
```

`records.jsonl` 이 옛 셋의 메타를 그대로 승계했기 때문에 **값 대조로는 같아 보인다.**
메타 일치를 프레임 동일성 근거로 쓰면 안 된다 — 실제 픽셀(luma)이나 라벨 해상도로
검증해야 한다. 이번에 한 번 오판했다.

`pnp_conditioning` 키 추가 · mask 동봉 등 08-14 기록의 차이와도 정합한다
(= 재렌더링본).

### 2.3 그래서 지금 상태

`step_25545.pth` 는 옛 셋으로 학습됐고 `split_sha`·`population_sha` 가 옛 셋을
가리킨다. 새 셋 이미지를 넣으면 checkpoint·split 과 짝이 맞지 않아
SEARCH2K 결과가 무의미해진다. **틀린 데이터를 넣느니 비워 두는 쪽을 택했다.**

## 3. Windows 에서 프레임 조립하는 법 (권장 경로)

이 데이터는 **애초에 Windows 에서 생성됐다.** `records.jsonl` 의 원본 경로:

```
E:\CODING\GitHub\FoundationPose\data\pallet\runs\diagnostics\v2_prod10k2_s8101_public\rgb\f0000_rgb.png
```

아래 10개 shard 폴더가 Windows 에 남아 있으면 리눅스에서 보낼 필요가 없다.

```
v2_prod10k2_s8101_public   v2_prod10k3_s8201_public
v2_prod10k2_s8102_public   v2_prod10k3_s8202_public
v2_prod10k2_s8103_public   v2_prod10k3_s8203_public
v2_prod10k2_s8104_public   v2_prod10k3_s8204_public
v2_prod10k2_s8105_public   v2_prod10k3_s8205_public
```

### 조립 규칙

zip 안 `eligibility_audit.csv` 의 `frame_uid` 컬럼이 대응표다:

```
index    frame_uid
000000   run1|v2_prod10k2_s8101_public|f0000
000044   run1|v2_prod10k2_s8101_public|f0044
```

즉 `{index}` ← `{shard}/rgb/{frame_id}_rgb.png`.

러너(`edge_mandatory_fast_search.py`)가 요구하는 레이아웃:

```
pallet6d_v2_10k/
  all/{index:06d}.png        ← {shard}/rgb/{frame_id}_rgb.png
  all/{index:06d}.json       ← {shard}/labels/{frame_id}_label.json
  index.csv                  ← 17컬럼 (아래)
  {mask_visible 상대경로}     ← index.csv 의 mask_visible 컬럼이 가리키는 곳
  {mask_amodal  상대경로}
```

`index.csv` 에서 러너가 실제로 읽는 컬럼 (grep `m["..."]` 로 확인):

```
index, run, shard, frame_id, visible_kp_count, bbox_vis_min_side_px,
tiny_warning, diagnostic_mode, pallet_type, mask_visible, mask_amodal, trace
```

JSON 은 아래 키를 요구한다 (`phase_eligibility`):

```
camera_data.intrinsics{fx,fy,cx,cy} · camera_data.width/height
objects[0].cuboid (8,3) · projected_cuboid (8,2)
objects[0].projected_cuboid_centroid · dimensions_m{width,depth,height}
```

### ⚠️ 필요 프레임 수

`line_search2k`(2000) ∪ `line_dev512`(512) = **2512 장**.
`search2k-budget` 명령까지 돌리려면 `line_confirm6k`(6000) 도 필요하다.

단 `phase_eligibility` 는 `for i in range(20000)` 으로 **전수를 돈다.** 부분 셋으로
돌리려면 그 phase 를 건너뛰거나(이미 `eligibility_audit.csv` 가 zip 에 있음)
러너를 고쳐야 한다.

## 4. 정리 필요 — 잘못된 추출물

```
data/pallet/transfer/win_search2k/pallet6d_v2_10k/   1.7G
```

§2.2 의 잘못된 전제(새 40k = 옛 셋)로 만든 2,429 프레임 추출물이다. **쓰면 안 된다.**
삭제 권한이 막혀 있어 남겨 뒀다:

```
rm -rf /home/minjae/Documents/github/pallet-pose/data/pallet/transfer/win_search2k/pallet6d_v2_10k
```

## 5. 미확인

- 이 머신에 **마운트 안 된 465G NVMe**(`nvme1n1`, p1 100M EFI / p3 465G / p4 535M)
  가 있다. 파티션 구조상 듀얼부팅 Windows 디스크로 보이나 [추정] — 마운트해서
  확인하지 않았다. 여기 `FoundationPose` 원본이 있을 수 있다.
- `annotation_reprojection_audit.csv` 의 `res` 컬럼을 만드는 코드가 현재 repo 에
  없어, 그 값의 정의를 코드로 확정하지 못했다. 위 §2.2 의 해상도 대조는
  "res = label 의 width×height" 라는 전제 [추정] 위에 있다. 다만 luma 대조가
  독립 근거로 같은 결론을 준다.
