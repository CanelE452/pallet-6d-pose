# architecture_reaudit — PHASE A

**GATE_A_PROVENANCE_MATCH = PASS · GATE_A_METRIC_REPRODUCTION = PASS**

training 0 · new inference 0. `mh_predcache_*.npz` 만 읽었다.

## 비교 대상

예산이 일치하는 25k 쌍만 비교한다. 두 run 은 `--split-late` **하나만** 다르다
(pool 33,758 / batch 8 / lr 1e-3 / wd 1e-4 / marks [6k,12k,18k,25k] /
ramp 500 / λ_corner 0.03518006215468158 / split_sha256 56b36d2a… 전부 동일).

`CAPACITY_MATCHED_3K`(E4)는 **3,000 step continuation**(source
`screen_A0_LINE_ONLY_long25k/step_18000.pth`)이라 25k 와 예산·source 가 모두
달라 **직접 비교에서 제외**했다. 기존 3k 결과는 `capacity_control_compare.json`
에 있으며 별도로만 인용한다.

## 결과 (population D2_MH_DEV512, n=512, paired, 10,000 bootstrap, seed 독립)

### seed 1
```
metric            shared     split     delta     rel%    CI low   CI high     0배제
---------------------------------------------------------------------------------
corner_rms        0.9006    0.7129   -0.1877   -20.84   -0.2358   -0.1174    True
line_angle        2.2788    2.2093   -0.0695    -3.05   -0.2607    0.1377   False
line_offset       0.9676    0.9885    0.0209     2.16   -0.0573    0.0901   False
pose_R            7.8297    7.2322   -0.5974    -7.63   -1.4483    0.0543   False
pose_t            0.2244    0.1825   -0.0419   -18.67   -0.0808   -0.0079    True
5cm5deg           0.0781    0.1465
```

### seed 2
```
metric            shared     split     delta     rel%    CI low   CI high     0배제
---------------------------------------------------------------------------------
corner_rms        0.8827    0.7243   -0.1584   -17.94   -0.2039   -0.1140    True
line_angle        2.3866    2.3186   -0.0680    -2.85   -0.2839    0.1303   False
line_offset       1.0599    1.0299   -0.0300    -2.83   -0.1051    0.0406   False
pose_R            8.0668    7.5387   -0.5281    -6.55   -1.4047    0.3822   False
pose_t            0.2397    0.1941   -0.0456   -19.01   -0.0775   -0.0092    True
5cm5deg           0.0938    0.1367
```

delta 는 `split − shared` 이므로 **음수 = SPLIT_LATE 가 좋음**.

## 읽기

1. **corner 는 두 seed 모두 명확히 개선된다** — RMS −20.84% / −17.94%,
   CI 가 0 을 배제. 이것이 SPLIT_LATE 의 실체다. `[확인]`
2. **line 은 사실상 변하지 않는다** — angle −3.05% / −2.85%, offset +2.16% /
   −2.83%, 네 경우 모두 CI 가 0 을 포함. 기존 메모리
   `line-branch-seed-variance-exceeds-effect`(seed 산포 15~19%)에 비추면 이
   크기는 **주장할 수 없는 범위**다. `[확인]`
3. **pose 는 t 만 유의하다** — R 은 −7.63% / −6.55% 인데 CI 가 0 을 포함하고,
   t 는 −18.67% / −19.01% 로 두 seed 모두 CI 가 0 을 배제한다.
   5cm5deg 는 0.0781→0.1465 (seed1), 0.0938→0.1367 (seed2). `[확인]`
   → corner 개선이 **translation 을 통해** pose 로 전이된다는 그림이며, 기존
   메모리 `corner-scale-error-is-the-translation-lever` 와 방향이 같다 `[추정]`.

**Q1 답: SPLIT_LATE 는 shared-late 보다 낫다. 단 그 이득은 corner(및 그로 인한
translation)에 한정되고 line 에는 없다.** `[확인]`

## METRIC_REPRODUCTION 상세

정본은 `mh_screen_A1_CORNER_LINE_{label}_seed*.json` 의 `["25000"]["D2_MH_DEV512"]["line"]`.
집계는 **전체 supported role 을 pool** 한 median(정본 `DH.summarise`)이다.

```
  SHARED_LATE_25K_seed1      angle 2.273491 vs 2.273491 OK   offset 0.960385 vs 0.960385 OK
  SHARED_LATE_25K_seed2      angle 2.451199 vs 2.451199 OK   offset 1.028682 vs 1.028682 OK
  SPLIT_LATE_25K_seed1       angle 2.205082 vs 2.205082 OK   offset 0.969264 vs 0.969264 OK
  SPLIT_LATE_25K_seed2       angle 2.336040 vs 2.336040 OK   offset 1.033146 vs 1.033145 OK
```

### 이 게이트가 실제로 잡아낸 것 두 가지

- **단위 오류** — predcache 는 canonical(θ=radian, ρ=canonical50)을 저장하는데
  `DH.measure` 는 centred MAP100(θ=degree)을 기대한다. 변환 없이 넣으면
  angle 2.30° 가 **0.0409** 로 읽힌다(실제로 그렇게 나왔다).
  정본 `DH.centred_from_canonical` 를 거쳐야 한다.
  근거: `mh_diagnose.py:1235` 가 `np.degrees(pred_theta - gt_theta)` 를 쓴다.
- **잘못된 정본 파일** — `mh_screen_meta_long25k_seed*.json` 에는
  `A2_CORNER_LINE_MASK` 만 들어 있다. 처음에 그 값(2.302364)과 비교해
  재현 실패로 오판했다. A1 의 정본은 `mh_screen_A1_CORNER_LINE_*` 쪽이다.

둘 다 봉합하지 않고 원인을 찾아 고쳤으며, 그 뒤 4/4 (run×seed) 가 소수점 6자리
까지 일치한다.
