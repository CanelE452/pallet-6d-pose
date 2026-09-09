# IMPLEMENTATION_AUDIT — C4 symmetry-aware pose loss

무엇을 바꿨고, 무엇을 **일부러 건드리지 않았는지**를 적는다.

---

## 1. Ultralytics 8.4.60 의 실제 흐름 (소스에서 확인)

```
E2ELoss(model, loss_fn)
    self.one2many = loss_fn(model, tal_topk=10)
    self.one2one  = loss_fn(model, tal_topk=7, tal_topk2=1)
```

`loss_fn` 하나로 **두 벌**을 만든다.  따라서 `loss_fn` 자리에 custom class 를 넣으면
one2many / one2one 양쪽에 자동으로 들어간다.  한쪽에만 들어가는 사고(§13-7)는
구조적으로 일어나지 않지만, 그래도 런타임에서 확인한다(T9 + smoke branch 히스토그램).

`PoseLoss26.calculate_keypoints_loss` 의 순서:

```
selected = _select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
selected[..., :2] /= stride
gt_kpt   = selected[masks]                 # (N_pos, 9, 3)
kpt_mask = gt_kpt[..., 2] != 0
kpts_loss     = keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
kpts_obj_loss = bce_pose(pred_kpt[..., 2], kpt_mask.float())
rle_loss      = calculate_rle_loss(...)     # flow_model 있을 때만
```

`KeypointLoss.forward` 는 마지막에 `.mean()` 으로 스칼라를 낸다.  branch 선택에는
instance 별 값이 필요한데, 기존 `pallet_yolo_loss/symmetry.py` 의
`per_instance_kpt_loss` 가 **같은 수식에서 마지막 mean 만 instance 축으로 남긴** 것이라
그대로 재사용했다.  새로 쓰지 않은 이유는 수식이 갈라지면 선택 기준과 실제 loss 가
어긋나기 때문이다.

---

## 2. 추가한 것 — 전부 새 이름, opt-in

```
pallet_yolo_loss/c4.py
    C4Config                 환경변수 C4_CONFIG 로만 켜진다
    validate_permutations()  bijection / centroid / 12-edge 를 검사
    ChallengeC4PoseLoss      PoseLoss26 상속

pallet_yolo_loss/model.py    ChallengeC4PoseModel      (기존 class 수정 없음, 뒤에 추가)
pallet_yolo_loss/trainer.py  ChallengeC4Trainer        (기존 class 수정 없음, 뒤에 추가)
```

기존 `PSPCPoseLoss26` · `A1SymmetryPoseLoss` · `ASCTrainer` 는 한 글자도 고치지
않았다.  과거 실험의 재현성이 깨지지 않는다.

---

## 3. C4 를 넣은 자리 — target 동치류 하나뿐

```python
def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, ...):
    if not self.c4.enabled or not masks.any():
        return super().calculate_keypoints_loss(...)        # ← stock 과 동일 경로

    with torch.no_grad():
        best = self._choose_branches(...)                   # object 별 argmin
        gather = perms[row_branch]                          # GT 행마다 순열
    keypoints = torch.gather(keypoints, 1, gather...)       # 좌표+visibility 동시
    return super().calculate_keypoints_loss(masks, ..., keypoints, ...)
```

설계 의도가 세 가지다.

**(a) disabled 경로는 stock 그 자체다.**  분기 하나로 곧장 `super()` 로 가므로
forward 도 gradient 도 bit-exact 하게 같다 (T1 에서 `max|diff| = 0.000e+00` 확인).
F0 control 이 이 성질에 의존한다 — F0 는 같은 코드·같은 trainer 로 돌되 `C4_CONFIG`
만 없다.

**(b) 좌표와 visibility 가 갈라질 수 없다.**  `[x, y, v]` 를 담은 텐서의 keypoint
축을 한 번의 `gather` 로 재배열한다.  좌표만 옮기고 visibility 를 원래 index 에 두는
사고(§5-1, §13-6)는 구조적으로 불가능하다.  kobj target 은 `super()` 안에서
`kpt_mask` 로부터 파생되므로 같은 branch 를 따라간다.

**(c) 선택은 object 단위다.**  positive anchor 를 `(batch, target_gt_idx)` 키로 묶어
`scatter_add_` 로 object 별 비용을 모은 뒤 branch 축에서 `argmin(0)` 한다.  keypoint
마다 따로 고르는 경로가 없다.

선택 자체에는 gradient 가 필요 없으므로 `torch.no_grad()` 안에서 네 branch 를 잰다.

---

## 4. 하지 않은 것 (금지 사항, §4)

```
pred[i] -> 가장 가까운 GT point              없음
Hungarian(pred[0:8], GT[0:8])                없음
keypoint 별 독립 permutation                  없음
```

AST 로 검사한다 — 문자열 grep 은 docstring 의 "금지" 설명문을 잡는다(실제로 처음에
그렇게 걸렸다).  `cdist` · `linear_sum_assignment` · `cKDTree` · `knn` 등이 **호출·
import 되지 않는지**를 본다 (T8).

## 5. 건드리지 않은 것 (§5-3)

```
box · cls · dfl · anchor assignment(TAL)     stock 그대로
pose 12.0 · kobj 1.0 · rle 1.0               v4 args.yaml 값 그대로
```

gain 은 결과를 보고 조정하지 않는다.

---

## 6. 검증 결과

`TEST_RESULTS.txt` — **17/17 PASS**.

```
T1  disabled == stock         forward 일치, gradient max|diff| 0.000e+00
T2  90도 등가                  indexed 0.389466 > 0,  C4 0.000e+00
T3  180 / 270도 등가           indexed 0.404619 / 0.389466,  C4 0.000e+00
T4  centroid 8 고정            네 permutation 전부
T5  0~7 bijection             네 permutation 전부
T6  cuboid 12-edge 보존        네 permutation 전부
T7  visibility 동반 이동        invisible 좌표가 loss 를 오염시키지 않음
T8  point-wise 매칭 없음        AST 기준, 호출 0
T9  one2many/one2one 양쪽      둘 다 ChallengeC4PoseLoss
T10 backward                   loss·gradient finite, sum|g| = 0.0474 (nonzero)
```

### smoke (1 epoch, F0/F1)

```
init SHA 6a40a4d430fd205a…  F0/F1 동일
recipe 대조 통과 (ref = runs_live_gt/ft_live_gt_v4/args.yaml)
NaN/Inf 0,  두 arm 모두 checkpoint 생성

branch histogram
  F0 (C4 OFF)   one2many 0 · one2one 0                    ← stock 경로만 탐
  F1 (C4 ON)    one2many  id 55.2% / 90 9.2% / 180 0.1% / 270 35.6%   n=38,352
                one2one   id 54.8% / 90 9.4% / 180 0.1% / 270 35.6%   n= 3,836
```

한 branch 로 100% 쏠리지 않았고, one2many·one2one 양쪽에서 실제로 실행됐다.
180도가 0.1% 로 희소한 것은 앞뒤가 통째로 뒤집히는 배치가 드물기 때문으로 보이며
`[추정]`, 0 이 아니므로 계산 자체는 도달한다.

---

## 7. 알려진 한계

* branch 선택은 매 forward 마다 새로 한다 — epoch 간 일관성을 강제하지 않는다.
  학습이 진행되며 특정 phase 로 수렴할 수도, 진동할 수도 있다.  진동하면 downstream
  face phase 가 불안정해지므로 `DOWNSTREAM_PHASE.json` 에서 따로 잰다.
* C4 는 **정사각 물체에만** 유효하다.  generic synthetic(broad40k)에는 직사각
  팔레트가 섞여 있어 적용하지 않는다 — `C4_SYMMETRY_CONTRACT.json` 의
  `does_not_apply_to` 에 명시했다.
* F1 이 F0 보다 느리다 (smoke 기준 1.6분 vs 0.6분).  branch 4 개를 재는 비용이다.
