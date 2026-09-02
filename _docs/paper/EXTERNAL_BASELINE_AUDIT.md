# External keypoint baseline audit (M1)

작성 2026-09-01.  M1 에 넣을 수 있는 외부 비교 대상의 **현재 상태**만 적는다.
억지 wrapper 로 숫자를 만들지 않는다.

## 현재 M1 에 실제로 들어간 것

```text
YOLO26n-Pose (synthetic-only)   R0        평가 완료
Proposed                        R5        평가 완료
DOPE (same-data control)        DOPE      평가 완료 (2026-09-02, offline prediction 경로)
SingleShotPose                  —         INCOMPATIBLE (구현 없음)
PVNet                           —         NEEDS_TRAIN, 단 사전 negative 결과 있음
Real-FT upper bound             —         NEEDS_LEAKAGE_AUDIT
```

## DOPE — 완료 (2026-09-02)

evaluator 에 `--predictions` 경로를 열어 해결했다.  모델을 직접 올리는 대신 미리
덤프한 예측을 읽으므로, Ultralytics 가 아닌 baseline 도 **같은 evaluator·같은
population·같은 metric** 으로 채점된다.

```text
challenge/evaluation_v2/paper_real_eval.py   _CachedPredictor + --predictions
scripts/self_training_yolo/dump_dope_predictions.py
data/pallet/results/paper_eval_v1/baselines/DOPE_R0_PREDICTIONS.json
```

결과 (PAPER_EVAL 319 / 2689, 같은 채점 코드):

```text
              corner↓   det↑   AP50-95↑   AUROC↑   FPR95↓
──────────────────────────────────────────────────────────
DOPE           10.875  0.737     0.3412   0.9903   0.0409
YOLO26n R0      6.501  0.975     0.7688   0.9921   0.0417
Proposed        7.057  0.984     0.7585   0.9953   0.0283
```

**비대칭을 숨기지 않는다.**  DOPE 에는 box head 가 없어 box 를 검출된 cuboid 코너의
bounding box 로 유도했다.  따라서 `AP50-95` 는 같은 양이 아니고, score 도 DOPE 는
belief peak · YOLO 는 box confidence 라 `AUROC`/`FPR95` 척도가 다르다.
직접 비교가 성립하는 열은 **corner 와 det** 이다.

corner 는 2026-09-02 의 visibility 확정 이후 값이다.  그 전에는 319 장 중 216 장만
supervision mask 를 갖고 있어 (DOPE 10.083 / R0 4.420 / Proposed 4.180) 편향된
모집단 위에서 계산됐다.  정정 후 **Proposed 가 R0 보다 corner 가 나쁘다** — det 와
ranking 은 여전히 Proposed 가 낫다.

추론은 reflect-padding 을 썼다 (plain squash 는 truncation·근접에서 체계적 과소검출).

checkpoint 는 있다.

```text
weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth
sha256 0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f
```

막았던 것은 **evaluator 가 Ultralytics 전용**이라는 점이었다.
`paper_real_eval.py` 는 `YOLO(weights, task="pose")` 로만 모델을 올린다.

```text
challenge/evaluation_v2/paper_real_eval.py:1724   self.model = YOLO(str(weights), task="pose")
```

`scripts/stage0/model_compare/mc_dump_dope.py` 가 DOPE keypoint 를 YOLO 와 같은
형식으로 덤프하지만, 161 장 canonical 셋(`mc_frames`)과 다른 checkpoint 에 묶여 있어
PAPER_EVAL 319 장에 그대로 쓸 수 없다.

(해결됨) 필요한 작업은 evaluator 에 **offline prediction 경로**를 여는 것이었다.  모델을 직접
올리는 대신 미리 덤프한 keypoint/box 를 읽게 하면 DOPE 뿐 아니라 PVNet 같은
외부 baseline 도 **같은 evaluator·같은 population·같은 metric** 으로 채점된다.
그게 M1 의 공정성에 맞는 방향이라 wrapper 를 따로 만들지 않았다.

주의: DOPE 추론은 reflect-padding 이 필요하다.  plain squash 를 쓰면 truncation·
근접에서 체계적으로 과소검출되어 모델/도메인 오진을 부른다 (과거 2회 교정됨).

## SingleShotPose — INCOMPATIBLE

저장소에 구현이 없다.

```text
find . -iname "*singleshot*"   → 결과 없음
```

같은 9 keypoint 표현·같은 synthetic supervision 으로 공정 비교하려면 외부 구현을
들여와 데이터 변환까지 새로 해야 한다.  근거 없이 숫자를 만들지 않고, 이 상태를
Appendix 에 그대로 적는다.

## PVNet — NEEDS_TRAIN, 단 사전 결과가 부정적

구현 자산은 있다.

```text
scripts/stage0/finetune_pvnet.py
scripts/stage0/eval_harness/eval_pvnet_heads.py
Deep_Object_Pose/common/utils_pvnet.py
weights/stage_screens/stage4_pvnet
```

다만 이 저장소에는 **dense vector voting 이 negative 결과였다**는 기록이 있다.
재시도 전에 그 기록을 먼저 읽어야 하고, "외부 baseline 을 채운다" 는 이유만으로
이미 부정적으로 판정된 계열을 다시 돌리지 않는다.

M1 의 최소 요건(DOPE · YOLO26 · Proposed)이 DOPE bridge 로 충족되면, PVNet 은
Appendix 로 미룬다.

## Real-FT upper bound — NEEDS_LEAKAGE_AUDIT

supervised real fine-tune checkpoint 후보가 있다.

```text
challenge/yolo_pose_one_model/runs_ft/ft_a_real157_neg259_synth12k
challenge/yolo_pose_one_model/runs_ft/ft_b_patience0_ep40
challenge/yolo_pose_one_model/legacy_v1v2_ft/runs/LV1V2_FT_15EP_SEED42
```

평가하기 전에 **학습 GT membership 과 PAPER_EVAL 의 중복**을 SHA 로 감사해야 한다.
`ft_a` 는 이름에 real157 이 들어 있어 real GT 를 직접 썼을 가능성이 높고, 그
157 장이 PAPER_EVAL 319 와 겹치면 그 수치는 controlled comparison 이 아니다.

중복이 확인되면 표기를 `LEAKED_SUPERVISED_UPPER_BOUND` 로 고정하고, M1 의
controlled row 로 오해되지 않게 별도 절에 둔다.  중복이 없어도 `SUPERVISED UPPER
BOUND` 표기는 유지한다 — 같은 supervision 조건이 아니기 때문이다.

## 우선순위

```text
1. evaluator offline prediction 경로   DONE (2026-09-02) — DOPE 가 M1 에 들어감
2. Real-FT leakage 감사                다음 — upper bound 를 Appendix 에 채움
3. PVNet                               과거 negative 기록 검토 후에만
4. SingleShotPose                      구현 도입이 정당화될 때만
```

M1 의 최소 외부 비교 요건(DOPE · YOLO26 · Proposed)은 충족됐다.

core Proposed 결과는 이미 나와 있으므로, 위 넷 중 어느 것도 MAIN 표를 막지 않는다.
