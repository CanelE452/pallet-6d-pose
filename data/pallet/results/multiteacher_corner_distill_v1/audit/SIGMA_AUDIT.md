# Pose26 sigma factual audit

환경: conda `pallet-yolo26`, ultralytics **8.4.60**
(`/home/minjae/anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics`)
YOLO26 가중치는 `pallet-pose`(8.0.120)에서 로드 불가 — C3k2 부재. 이 트랙의 모든 YOLO26 추론은 `pallet-yolo26` 에서 돈다.

## 어디에 있나

- `nn/modules/head.py:666` `class Pose26(Pose)`
- `nn/modules/head.py:704-705`
  `self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint`
  `self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)`
- `nn/modules/head.py:710` end2end 이면 `one2one_cv4_sigma = deepcopy(cv4_sigma)` — one2many/one2one **각각 별도 sigma head** 존재.
- `utils/loss.py:800` `class PoseLoss26(v8PoseLoss)`

## 학습 시 실제로 loss 에 쓰이는가 — 쓰인다 [확인]

`utils/loss.py:835-838`
```
if self.rle_loss and preds.get("kpts_sigma", None) is not None:
    pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
    pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)
    pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)   # (b, hw, K, 5)
```
`self.rle_loss` 는 `model.model[-1].flow_model` 이 있으면 생성되고, `Pose26.__init__` 이 항상
`self.flow_model = RealNVP()` 를 만든다 → **Pose26 이면 rle_loss 는 언제나 활성**.
loss gain `rle: 1.0` (`cfg/default.yaml:108`), `loss[5] *= self.hyp.rle`.
gradient path 살아 있음 — sigma → RLE → backward.

## parameterisation [확인]

`utils/loss.py:900-901`
```
pred_sigma = pred_sigma.sigmoid()
error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
```
- sigma = conv raw logit 의 **sigmoid** → 값 범위 (0, 1). 표준편차처럼 쓰이지만 상한 1 로 잘림.
- 좌표 단위는 **stride 로 나눈 feature-grid 단위** (`calculate_keypoints_loss` 에서
  `selected_keypoints[..., :2] /= stride_tensor`). 따라서 sigma 의 물리 단위는
  "해당 anchor 가 속한 scale 의 grid cell" 이고, **P3/P4/P5 사이에서 서로 다른 픽셀 스케일**을 갖는다.
  픽셀로 환산하려면 sigma * stride 를 곱해야 하고, 그 stride 는 그 detection 을 낸 anchor 에 의존한다.
- `RLELoss.forward` (`utils/loss.py:193-208`) 는 `log_sigma - log_phi + log(2 sigma) + |error|`.
  sigma 는 절대 스케일이 아니라 flow 의 residual 분포와 결합된 값이다.

## inference/export 때 접근 가능한가 — 표준 경로로는 불가 [확인]

`nn/modules/head.py:748-752`
```
if self.training:
    preds["kpts_sigma"] = torch.cat([...])
```
→ `model.eval()` / `predict()` 경로에서는 `kpts_sigma` 키 자체가 생성되지 않는다.

`nn/modules/head.py:756-758`
```
def fuse(self):
    super().fuse()
    self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None
```
→ ultralytics 의 기본 predict 경로는 fuse 를 거치므로, 메모리상 sigma head 가 **None 으로 삭제**된다.
(체크포인트 .pt 안의 weight 자체는 남아 있다.)

## SIGMA_STATUS 판정

**SIGMA_STATUS = DIAGNOSTIC_ONLY**

근거:
1. 학습 loss 에는 실재하고 gradient 도 흐른다 → UNUSED 아님.
2. 그러나 배포 추론 경로에서 나오지 않고, fuse 가 head 를 지운다 → USABLE 아님.
   (커스텀 forward 로 head 를 직접 호출하면 뽑을 수는 있으나, 그건 배포 계약 밖의 개입이다.)
3. 값이 sigmoid 로 (0,1) 상한이고 단위가 anchor scale 의존 grid unit 이라,
   **서로 다른 teacher 사이에서 comparable scale 이 아니다.**
4. DOPE/heatmap 계열 teacher 는 sigma 자체가 없다.

따라서:
- Gate A 의 fusion arm **F3 (native uncertainty weighted mean) = BLOCKED_NOT_COMPARABLE**.
- Gate D2 student loss 는 sigma 기반 KL 을 쓰지 않고,
  METHOD_LOCK 에 사전 정의한 **weighted robust coordinate loss fallback** 을 쓴다.
- sigma 는 teacher-time diagnostic (커스텀 forward 로 추출 가능할 때) 으로만 기록하고,
  어떤 gate 판정에도 입력하지 않는다.
