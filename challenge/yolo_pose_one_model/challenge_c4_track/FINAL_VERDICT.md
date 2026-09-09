# FINAL_VERDICT — 과제용 C4 symmetry-aware YOLO26n track

2026-09-05.  git HEAD `985201c`.  ultralytics 8.4.60 · torch 2.1.1+cu118 · RTX 3080.

---

## Stage A source

```
dataset    challenge/yolo_pose_one_model/datasets/broad40k
           train 39,500 / val 500 · 라벨 40,000 전 줄 32 토큰 · symlink 0 · 짝 불일치 0
           keypoint 규약 camera_dynamic_0123_v4 (camera-facing)
오염 검사   palletobj 0/40,000 · pallet_full 0 · real 파일명 패턴 0 · gt_source (none)
           source asset 4 종 전부 generic
           g38_legacy_v1v2_p0_tex20k 와 train stem 교집합 0
한계       mesh-hash 대조 불가(자산 4 중 0 개만 hash 가능) → 배제 근거는 asset 문자열 +
           dimensions_m signature.  타깃 두께비만큼 얇은 프레임 117/40,000 (0.29%)
           정사각(≤1.05) 프레임 2,766/40,000 (6.9%) — 실측
```

**새 synthetic 을 만들지 않았고, broad40k 에 C4 를 적용하지 않았다.**

## Stage A checkpoint

```
runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt
sha256  6a40a4d430fd205a427e38a1927aad2a0a0bef20b984484e0b77318e7bd355ea   6,552,679 B
```

**재학습하지 않고 재사용했다.**  §2-1 의 9 개 조건(membership · COCO init · 60 epoch
완주 · batch 32 · seed 42 · stock loss · real supervision 없음 · checkpoint 정상 ·
results.csv 정상)을 전부 대조해 충족했고, SHA 가 `dimension_conditioning_probe/
SOURCE_AUDIT.md` 의 기존 기록과 일치한다.

`best.pt` 가 아니라 `last.pt` 를 썼다 — broad40k val 500 장은 train 과 같은 pool 의
strided subsample 이라 holdout 이 아니다.

## F0 / F1 init 동일 여부

**동일하다.**  드라이버가 매 실행 시작에 SHA 를 확인하고 불일치면 중단한다.
recipe 도 `runs_live_gt/ft_live_gt_v4/args.yaml` 과 대조 후 진행했다.

한 가지 예외를 기록한다 — `workers` 를 4 에서 2 로 낮췄다.  첫 실행이 DataLoader
공유메모리 9.2 GB 로 OOM(SIGKILL) 됐기 때문이다(F1 8/40 에서 사망, 커널 로그 확인).
자원 설정이라 loss 와 무관하지만 augmentation 난수 시퀀스를 바꾸므로 **F0 도 함께 다시
돌렸다**.  batch 32 등 나머지는 그대로다.

## real train/val membership

```
datasets/live_gt_v4
train 3,087   원본 696 · truncation crop 999 · flip 696 · noise 696
val     155   원본만 (crop/aug 0)
split         interleave, val_every 6
leakage       0 (파생 stem 정규화 후 실제 교집합 계산, 규칙은 696/696 로 검증)
kpt_shape [9,3]   flip_idx [1,0,3,2,5,4,7,6,8]   라벨 3,242 개 전부 32 토큰
```

기존 v4 데이터·파생본을 그대로 썼다.  새 crop/flip/noise 를 만들지 않았다.

## C4 permutation 정의

`plastic_standard_110x110x15` 의 3D vertex 를 `Ry(θ)` 로 돌린 뒤 좌표 매칭(<1e-9)으로
index 를 복원했다.  기존 `PERMUTATION_GROUP.json` 을 복사하지 않았다.

```
  0도  [0, 1, 2, 3, 4, 5, 6, 7, 8]
 90도  [1, 5, 6, 2, 0, 4, 7, 3, 8]
180도  [5, 4, 7, 6, 1, 0, 3, 2, 8]
270도  [4, 0, 3, 7, 5, 1, 2, 6, 8]
```

bijection · centroid 8 고정 · top/bottom 보존 · cuboid 12-edge 보존 · 90x2=180 ·
180+90=270 · 90x4=identity · inverse 일치 — 8 항목 전수 PASS.
90도 값이 기존 `canonicalize_fourfold_yaw.ROT90_PERMUTATION` 과 일치했다(독립 유도의
교차검증).

unit/parity test **17/17 PASS** (`TEST_RESULTS.txt`).  smoke 에서 branch histogram 이
한쪽으로 쏠리지 않고 one2many·one2one 양쪽에서 실행됨을 확인했다.

---

## fixed-index 결과 (val 155)

```
        검출      median    p90      <5px     >20px    pose mAP50-95
F0     152/155    2.00px    5.23     88.8%     4.6%       0.9297
F1     155/155    3.54px  139.73     57.4%    40.0%       0.4112
```

## C4-equivalent 결과

```
        median    p90      <5px     >20px
F0      1.96px    4.32     91.4%     1.3%
F1      2.08px    3.60     96.8%     0.0%
```

## symmetry-rescued / true collapse

```
                                       F0      F1
symmetry-rescued (fixed>20 & C4<5)      4      61
true collapse    (C4 > 20px)            2       0
```

F1 의 fixed-index 실패 62 프레임은 **전부 270도**이고, 그 프레임들의 C4 오차는
median 1.97px / max 5.92px 다.  위치는 정확하고 번호만 돌았다.

## branch histogram

```
학습 (F1, epoch 39)   one2many  identity 19,269 · 90도 3,900 · 180도 0 · 270도 7,700
                      one2one   identity  1,927 · 90도   390 · 180도 0 · 270도   770
평가 (val 선택 perm)   F0  0도 147 / 90도 2 / 180도 0 / 270도 3
                      F1  0도  93 / 90도 0 / 180도 0 / 270도 62
```

F0 는 C4 가 꺼져 있어 학습 중 histogram 이 전부 0 이다(정상).

## newauto compatibility

```
계약   POSE_FACE_KPTS = (0,1,2,3),  _box_object_points 원점 = 전면 중심
위험   face phase 가 바뀌면 tvec 이 인접 면 중심으로 0.55·√2 ≈ 0.78 m 점프

측정 (GT 미사용, 5 세션)
       F0  638 pair 중 632 가 0도    안정률 99.06%
       F1  627 pair 중 625 가 0도    안정률 99.68%
```

우려했던 프레임 단위 phase 진동은 **관측되지 않았다**.  세션 사이에는 phase 가 다를 수
있으나(val 에서 0도 93 / 270도 62), 정사각 팔레트는 네 면 어디로도 포크가 들어가므로
그 자체가 물리적 오류는 아니다.

---

## 최종 판정

```
C4_LOCALIZATION_IMPROVED · DEPLOYABLE_WITH_CAVEAT · NOT_PROMOTED
```

§12 우선순위로 보면 F1 이 1·2·3 에서 앞선다 (진짜 collapse 2→0, phase 안정
99.06→99.68%, C4-equivalent p90 4.32→3.60 / gross20 1.3→0.0%).  4·5 에서 F0 가
앞서지만 **그 차이가 전부 270도 equivalent permutation** 임을 확인했으므로 localization
실패로 세지 않는다.

**그럼에도 기존 배포 weight 를 덮지 않는다.**  이유는 세 가지다.

1. **표본이 얇다.**  F1 의 우위는 collapse 2→0, 미검출 3→0, phase 불안정 6→2 —
   전부 한 자릿수 사건이고 seed 하나짜리다.  이 크기의 차이를 단일 run 으로 주장하지
   않는 것이 이 저장소의 기존 규율이기도 하다.
2. **가장 직접적인 반례 데이터가 없다.**  동료가 "많이 흔들렸다" 고 지목한 세션
   (`forklift_v4_recording_20260904_190700`)이 이 저장소에 반입되지 않았다.  그
   세션에서 재보지 못한 채로 배포 판정을 내리는 것은 이르다.
3. **흔들림의 더 싼 원인이 안 고쳐져 있다.**  `newauto` config 가
   `PALLET_FACE_W 1.000 / PALLET_DEPTH 1.200` 인데 실측은 1.10 / 1.10 이다.  정사각
   물체를 직사각 모델로 `solvePnP` 하는 중이고, 이건 keypoint 품질과 무관하게 자세를
   흔든다.  수정 비용은 config 두 줄이다.

## 최종 후보 checkpoint

```
F1   challenge_c4_track/F1/weights/best.pt
     sha256 42dbfc01ef181d5628f6b0a0e1a2f7c8...   (전체 값은 COMPARISON.json)
F0   challenge_c4_track/F0/weights/best.pt
     sha256 a4606fc5c39717e9...                    control, 비교용으로 보존
```

배포 후보는 **F1** 이되, 위 세 조건이 해소되기 전까지 `release/` 를 갱신하지 않는다.

## 다음 작업 (이번 실험의 성공으로 봉합하지 않는다)

```
1  newauto config 치수를 1.10 / 0.15 / 1.10 으로 고치고 같은 영상 재측정   비용 5 분
2  190700 세션 반입 후 F0/F1 phase·오차 재측정                            비용 10 분
3  seed 2~3 개로 F0/F1 반복 — 소표본 우위가 재현되는지                     비용 1 시간
4  (선택) 조건부 C4 로 Stage A 재학습 — 정사각 프레임(2.6~6.9%)에만 적용
       기대효과 낮음(신호 밀도 2.6% vs real FT 100%), 비용 4 시간 반
```
