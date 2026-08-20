# FINAL ARCHITECTURE DECISION

```
ARCHITECTURE   NOT_LOCKED
  backbone / 2-head          LOCK   SPLIT_LATE_2HEAD
  pose fusion                미확정  F3 가 유일하게 작동하나 사전등록 gate 는 NOT_ESTABLISHED
  negative path              미확정  NEGATIVE_HANDLING_SUPPORTED = False
```

## 확정된 것

```
SPLIT_LATE_2HEAD             corner 전용 late 경로가 A1/E2/E4 대비 corner +16~21%,
                             PATH-C 회전 +14~15%, line 은 구조적 무손실
line isolation               two-stream 커리큘럼에서 3,000 step 학습 후에도
                             line_late max|diff| = 0.000e+00  (C1·C1_RESCUE·N1 전부)
gradient-normalized negative q 로 개입 강도를 seed 간 통일하는 계약
presence score               score_4kp = 4번째 corner peak, centroid 제외
```

## 미확정이었던 두 가지 (2 는 종결됨)

### 1. pose fusion (F3)

ALL 에서 두 seed 모두 통과하고 rotation 이득이 CI 로 확립된다. 떨어뜨린 것은
`LA_HARD` seed2 하나(n=51, CI [-31.4,+16.0]로 0 포함)다.
→ 병목은 fusion 이 아니라 **평가 해상도**.

### 2. negative path

negative·detection 절은 두 seed 모두 통과한다(FP -40.7%/-88.1%, recall drop 0.00pp).
떨어뜨린 것은 pose safety 하나이고 seed 방향이 정반대다
(seed1 t/ADD-S 개선, seed2 전면 악화).
→ 병목은 negative 데이터가 아니라 **belief 를 dense 하게 누르는 방식이 pose 에 주는 부작용**
   [추정 — seed 2개, 인과 미확인].

**종결 (2026-08-19).** dense suppression 은 REJECT, 대신 gradient 가 pose 에 닿지
않는 `SCORE4KP_THRESHOLD` 를 채택했다. 아래 "DETACHED PRESENCE GATE — 최종" 절 참조.

## 공통 병목

두 미확정 모두 **seed 2개로는 못 가르는 상태**에서 멈췄었다(2 는 이후 종결). 앞선 감사에서
파라미터 drift 는 seed 간 1% 차이인데 cell 지표는 40% 씩 흔들린다는 것을 이미 확인했다.

## 남은 평가 과제 (자동 실행 금지)

```
1  평가 전용 저앙각 point-valid holdout 확보
   → F3 의 LA_HARD 판정을 n=51 에서 수백으로 올려 실제로 가른다
   → EDGE clean_untouched 1,000 이 후보이나 저앙각 분포 확인 필요
2  real positive/negative 평가셋 구축
   → negative 결론은 synthetic 한정. real FP/AP 는 아직 말할 수 없다
```

새 head·새 fusion·새 loss·λ 재튜닝을 자동으로 시작하지 않는다.

---

---

## DETACHED PRESENCE GATE — 최종 (2026-08-19)

dense negative suppression 이 pose 에 준 부작용을 피하려고, **gradient 가 pose
network 에 아예 닿지 않는** presence gate 를 마지막 후보로 검증했다.

### pose invariance — 논증이 아니라 측정

같은 모델 객체를 CPU 로 옮겨 presence 파이프라인 전후 출력을 비교했다.

```
                     param diff   corner out   line out   grad tensors
seed1                0.000e+00    0.000e+00    0.000e+00       0
seed2                0.000e+00    0.000e+00    0.000e+00       0
```

첫 판(GPU 비교)은 seed1 에서 corner 1.853e-03 이 나왔다. 이건 네트워크가
움직인 게 아니다 — 모델·presence 코드·seed 를 하나도 안 건드리고 **더미 GPU
블록 1 GiB 만** 두 forward 사이에 끼워도 정확히 같은 1.853e-03 이 나오고,
3 GiB 면 2.139e-03 로 커진다. cuDNN 이 그때그때 남은 workspace 를 보고 conv
알고리즘을 고르기 때문이고, `benchmark=False`·`deterministic=True` 로도 고정되지
않는다. 즉 옛 검사는 가중치가 아니라 allocator 상태를 재고 있었다. CPU conv 는
비트 단위 재현되므로 위 0 은 진짜 0 이다. GPU 수치는 gate 가 아니라 noise floor
로 `presence_pose_invariance.json` 에 남겼다.

### 사전등록 판정은 보존한다

```
ORIGINAL_PREREG_VERDICT            FAIL
파일                                presence_verdict.json (수정 안 함)
sha256                              ffa67e88d0c18e0eb33b586650382a42026add5d0…
FAILURE_CAUSE                       PROTOCOL_CONFLICT   ← MODEL_FAILURE 아님
```

threshold 선택은 Recall ≥ 95% 를, 자격심사는 Recall ≥ 98% 를 요구했다. FP 를
최소화하면 threshold 는 항상 95% 바닥으로 내려가므로 두 조항은 **FP 가 이미
동률일 때만** 동시에 성립한다. 모델이 못한 게 아니라 프로토콜이 충돌했다.

### 교정 프로토콜 (FROZEN)

```
choose threshold minimizing negative FP/image
subject to positive Recall >= 0.98
```

앞으로 **모든 real/synthetic confirmation 에 고정**한다. Recall ≥ 95% 바닥은
폐기. 이 규칙으로 낸 아래 수치는 secondary analysis 이며 사전등록 판정을
덮어쓰지 않는다 (`NEGATIVE_FINAL_DECISION.json`).

threshold 는 `np.quantile` 이 아니라 **관측된 positive score 중에서** 고른다.
분위수는 표본 사이에 떨어져 실현 recall 0.97997(= drop 2.003pp)을 주는데, 이건
성능이 아니라 이산화 때문에 제약을 어긴 것이다. 제약을 정확히 만족시키면
threshold 가 내려가 FP 는 오히려 불리해진다(보수적 방향).

### 수치 (synthetic dev — real 아님)

```
arm                    seed   Recall 0.9801 (drop 1.99pp)      Recall 0.95 참고
                              FP/img    감소                    FP/img    감소
──────────────────────────────────────────────────────────────────────────────
P0_NO_GATE               -    1.0000     —                     1.0000     —
P1_SCORE4KP              1    0.2760   -72.4%                  0.2040   -79.6%
P2_DETACHED_LINEAR       1    0.2680   -73.2%                  0.1950   -80.5%
P1_SCORE4KP              2    0.2680   -73.2%                  0.1430   -85.7%
P2_DETACHED_LINEAR       2    0.2670   -73.3%                  0.1260   -87.4%
```

교정 프로토콜에서 P1·P2 모두 두 seed 다 PASS (drop 1.99pp ≤ 2pp,
FP 감소 72.4~73.3% ≥ 30%).

카테고리별로 갈린다. `N0_MATCHED_EMPTY` 는 FP/img 0.005~0.0075 로 사실상
해결이고, 남는 건 전부 `N1_STRUCTURAL_HARD`(0.41~0.43) 와
`N2_PALLET_LIKE_HARD`(0.19~0.20) 다.

### P2 는 필요 없다

같은 negative dev 이미지에 대한 paired bootstrap(B=10,000).

```
seed   recall    delta(P2-P1)   CI95                 판정
───────────────────────────────────────────────────────────────
1      0.9801    -0.0080        [-0.0170, +0.0010]   0 포함 — 미확립
2      0.9801    -0.0011        [-0.0100, +0.0080]   0 포함 — 미확립
1      0.95      -0.0090        [-0.0180, -0.0010]   P2 우세
2      0.95      -0.0170        [-0.0270, -0.0080]   P2 우세
```

학습 파라미터 9개는 느슨한 동작점에서만 0.9~1.7pp 를 벌고, **gate 가 실제로 서는
동작점에서는 측정 가능한 이득이 없다.**

### 결정

```
FINAL_NEGATIVE_HANDLING       SCORE4KP_THRESHOLD
  근거                        Recall≈0.98 에서 FP/image -72.4~-73.2%
                              pose network 완전 불변 (param/output diff = 0)
                              P2 대비 이득 미확립 (CI 0 포함, 두 seed)
                              trainable parameter 0, 추가 추론비용 무시가능
P2 DETACHED_LINEAR_PRESENCE   REJECT_AS_UNNECESSARY_COMPLEXITY
DENSE NEGATIVE SUPPRESSION    REJECT  (FP 는 줄였으나 seed2 pose safety 대실패)
```

### 아키텍처 상태

```
ARCHITECTURE_SEARCH = CLOSED

FINAL CORE
  SPLIT_LATE_2HEAD
  + F3 ROTATION_ONLY_TREFIT
  + SCORE4KP REJECTION
```

### 적용범위

**synthetic negative validation only.**
real-world FP/AP claim = **PENDING REAL EVALUATION.**

### NEXT

real positive/negative evaluation protocol 준비.
새 architecture·training 실험은 실행하지 않는다.
