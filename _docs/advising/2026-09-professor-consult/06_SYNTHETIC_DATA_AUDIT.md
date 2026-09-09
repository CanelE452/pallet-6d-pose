# 06 — 현재 synthetic 데이터 규모 실측

모든 수치는 **manifest·config·디렉토리 실측**이다.  추정한 숫자는 없다.
새 학습 0 · 새 추론 0.

## R0 (synthetic-only source model) 는 정확히 무엇으로 학습됐나

```text
checkpoint   challenge/yolo_pose_one_model/spatial_concat_scratch/runs/
             YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt
sha256       970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7
data.yaml    challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml
             (args.yaml 의 data 필드에서 확인)
epochs 60 · batch 32 · imgsz 640 · SGD · seed 42 · patience 0 · pretrained yolo26n-pose
```

[확인] **train 55,980 장 / val 4,020 장 = 60,000 장.**
`ls datasets/g38_legacy_v1v2_p0_tex20k/{images,labels}/{train,val}` 실측이며
images 와 labels 개수가 정확히 일치한다.

### 55,980 장의 구성 — symlink 타겟으로 실측

```text
source set                          count   비중    무엇인가
────────────────────────────────────────────────────────────────────────────────
stage_a  (G__ 접두)                38,002   67.9%   generic 계열 (G38).  범용 팔레트 렌더
legacy_v1v2_p0_10k  (P__ 접두)      8,989   16.1%   내 파렛트 v1/v2, 텍스처 없음
legacy_v1v2_p0_tex10k (TEX__ 접두)  8,989   16.1%   같은 v1/v2, 텍스처 20k 변형
────────────────────────────────────────────────────────────────────────────────
합계                               55,980  100.0%
```

[확인] 파일은 전부 symlink 이고, `readlink` 로 센 타겟 상위 폴더 분포가 위와 같다.
[확인] `stage_a/_composition.json` 은 train total 73,916 (generic 38,002 + target 35,914)
이라고 적는다.  **R0 는 그 중 generic 38,002 만** 가져갔다 — target 35,914 는 안 들어갔다.

### ★ 주의 — asset 다양성은 장수와 다르다

```text
broad_family_v2/CURRENT_ASSET_FAMILY_AUDIT.md 실측
  total_frames                    40,000
  unique_source_assets                 4
  unique_mesh_instances_verified       2
  effective_asset_count(exp entropy) 3.999
  single_asset_max_share          0.2545
  note: "frame 별 W/D/H 스케일 랜덤화는 mesh 다양성이 아니다"
```

[확인] **"G38" 은 38 개 family 가 아니라 약 38K 장이라는 뜻이다.**
generic 계열의 고유 mesh 는 4 개, 그중 실제 mesh 로 검증된 것은 2 개다.
상담에서 "38 종류 팔레트" 라고 말하면 안 된다.

## self-training arm 이 받은 노출

```text
source: data/pallet/results/paper_selftrain_v1/SELFTRAIN_EXPOSURE_LOCK.json
        (결과 보기 전에 얼린 계약)

pseudo_exposures_per_epoch        1,440
synthetic_exposures_per_epoch     1,440      (= replay 50%)
epochs                               10
total_pseudo_exposures           14,400
total_synthetic_exposures        14,400
updates_per_epoch                    90   · total_optimizer_updates 900
batch 32 · lr 0.002 · teacher_rounds 1 (static teacher, R0 캐시 고정)
small_pool_policy   "sampling with replacement to fill the fixed pseudo slots"
```

[확인] pseudo : synthetic replay = **50 : 50** 으로 얼려져 있고, 필터마다 통과 장수가
달라도 슬롯 수는 같다 — 그래서 필터 비교가 라벨 **수량**이 아니라 **품질** 비교가 된다.
[확인] R0-CONT 는 pseudo 슬롯을 synthetic replay 로 채운 통제군이다(pseudo 0).

## 실제로 학생이 본 real pseudo-label 은 몇 장인가

```text
pool 에서 필터를 통과한 프레임 수 (pl_review 덤프 실측, 학습이 읽은 라벨 파일 기준)
  R1_NAIVE          926
  R2_CONF           274
  R3_CONF_REPROJ    253
  R4_CONF_REMOVE    269
  R5_PROPOSED       261
```

[확인] 통과 장수는 261~926 인데 노출 슬롯은 전부 14,400 이다.  즉 **엄격한 필터일수록
같은 라벨을 더 여러 번 본다**(replacement 샘플링).  이것이 exposure lock 의 의도다.

## 정리 표

```text
Model/Arm   Synthetic source                    Synth img  Replay expo  Real PL frames  Notes
──────────────────────────────────────────────────────────────────────────────────────────────
R0          g38_legacy_v1v2_p0_tex20k            55,980         —              0        60 ep, source-only
            = G38 38,002 + v1v2 8,989 + tex 8,989
R0_CONT     same, replay only                    55,980      14,400            0        pseudo slots -> replay
R1_NAIVE    same + pseudo                        55,980      14,400          926        no filter
R2_CONF     same + pseudo                        55,980      14,400          274        confidence
R3          same + pseudo                        55,980      14,400          253        + reprojection
R4          same + pseudo                        55,980      14,400          269        + keypoint removal
R5_PROPOSED same + pseudo                        55,980      14,400          261        full consistency
```

## 더 작은 synthetic setup 의 전례가 이미 있다

[확인] 이건 새 아이디어가 아니라 **이미 측정된 적이 있다.**
`_docs/history/2026-08-24.md` 의 SAME REAL n=128 비교:

```text
model   composition                  cbox    med px   p90 px  gross20  night cbox
─────────────────────────────────────────────────────────────────────────────────
A42     generic 10K                 0.422    53.46    87.91   0.917      0.000
G38     generic 38K                 0.852    12.03    66.66   0.314      0.536
OLD     generic 38K + target x2     0.969     9.68    40.99   0.222      0.929
C43     V2 10K                      0.797    14.28    91.98   0.401      0.643
FT      OLD + real FT               0.984     6.47    25.40   0.135      0.964
```

[확인] 같은 문서의 판정: `GENERIC_SCALE_EFFECT = STRONG`,
`COVERAGE_EFFECT_10K = POSITIVE`, `EXTRA_V2_SIGNAL = NULL_OR_WORSE`.
[확인] 이 비교는 **n=128 real 프레임**에서 났고 PAPER_EVAL 319 와 모집단이 다르다 —
표 A/B 숫자와 같은 줄에 놓고 비교하면 안 된다.

## 이미 만들어져 있는 더 작은 데이터셋 (재사용 가능)

```text
73,916  stage_a                    generic + target 전체
55,980  g38_legacy_v1v2_p0_tex20k  ★ R0 가 쓴 것
39,500  paper_generic_v1 / broad40k
38,002  g38_generic_only           ★ R0 에서 v1/v2 를 뺀 것 (MID 후보)
10,000  fast_t10 / fast_m10
 9,867  v1_cf_matched10k           ★ 위 표의 A42 계열 (SMALL 후보)
 8,989  legacy_v1v2_p0_10k / _tex10k
```

[확인] `g38_generic_only`(38,002)와 10K 급 데이터셋이 **이미 빌드돼 있다.**
ablation 을 하려면 새로 렌더링할 필요가 없다.

## 아직 확인 못 한 것

- [ ] `g38_generic_only` 로 60ep 학습한 checkpoint 가 PAPER_EVAL 319 위에서 평가된 적이
      있는지 — 런 폴더(`OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42`)는 존재하나 이번 작업에서
      PAPER_EVAL 평가 여부를 끝까지 추적하지 않았다.  [추정] 없을 가능성이 높다
      (PAPER_EVAL 계약이 그 런보다 나중에 얼려졌다).
- [ ] scene 수 / 렌더 배치 수는 이미지 장수만 셌고 scene 단위 manifest 는 확인하지 않았다.
