# Pose26 one2one / one2many 경로 감사 — 설치본 ultralytics 8.4.60

경로: `.../site-packages/ultralytics/nn/modules/head.py`, `.../utils/nms.py`,
`.../models/yolo/detect/predict.py`, `.../nn/tasks.py`.
아래는 전부 **실제 설치 source 에서 확인한 것**이며 upstream 문서 추정이 아니다 `[확인]`.

## forward 구조 (Detect.forward, head.py:157-171)

```python
preds = self.forward_head(x, **self.one2many)          # one2many 는 항상 계산된다
if self.end2end:
    x_detach = [xi.detach() for xi in x]
    one2one  = self.forward_head(x_detach, **self.one2one)
    preds = {"one2many": preds, "one2one": one2one}
if self.training:
    return preds
y = self._inference(preds["one2one"] if self.end2end else preds)   # ★ 추론은 one2one 만
if self.end2end:
    y = self.postprocess(y.permute(0, 2, 1))
return y if self.export else (y, preds)
```

→ **질문 2 답**: stock 추론 경로는 one2one 만 쓴다 `[확인]`.
   단 one2many 도 forward 에서 계산은 된다(버려질 뿐).

## head 구성 (Pose26.one2many / one2one, head.py:723-737)

```
one2many : cv2(box) cv3(cls) cv4(pose) cv4_kpts cv4_sigma
one2one  : one2one_cv2 one2one_cv3 one2one_cv4 one2one_cv4_kpts one2one_cv4_sigma
```

→ **질문 1 답**: one2many 는 별도 parameter 를 가진 실재 branch 다. checkpoint 에
   실제로 들어있는지는 `SOURCE_AUDIT.json` 의 state_dict key 수로 실측한다 `[확인]`.

## ★ 함정 — fuse() 가 one2many 를 지운다

```python
# nn/tasks.py:253
if isinstance(m, Detect) and getattr(m, "end2end", False):
    m.fuse()                       # -> Detect.fuse: self.cv2 = self.cv3 = None
                                   #    Pose.fuse  : + self.cv4 = None
                                   #    Pose26.fuse: + cv4_kpts = cv4_sigma = flow_model = None
```

`YOLO.predict` 는 AutoBackend 에서 fuse 를 태우므로 **아무 조치 없이는 one2many 에
접근할 수 없다.** 이번 실험은 `Pose26.fuse` 를 no-op 으로 만들어 head 를 보존한다.
Conv+BN fusion 자체는 그대로 일어나므로 one2one 경로 수치는 바뀌지 않는다 —
이 주장은 M0 parity 로 검증한다.

## decode 경로 (질문 3)

```python
Detect._inference(x)   : dbox = decode_bboxes(dfl(x["boxes"]), anchors) * strides
                         torch.cat([dbox, x["scores"].sigmoid()], 1)
Pose._inference(x)     : + kpts_decode(x["kpts"])
Pose26.kpts_decode     : y[0::ndim] = (y[0::ndim] + anchors[0]) * strides   (Pose 와 다름)
decode_bboxes          : xywh = xywh and not self.end2end and not self.xyxy
                         -> end2end=True 이므로 두 branch 모두 **xyxy** 로 나온다
```

`_inference` / `kpts_decode` 는 branch 를 구분하지 않는 **같은 메서드**이고, 두 branch 의
채널 구성(4*reg_max, nc, nk)도 같다. anchors/strides 는 같은 feats 에서 만들어진다.
→ **질문 3 답**: one2many 를 동일 좌표계로 decode 할 수 있다 `[확인]`.

## NMS 가 적용되는 곳 / 안 되는 곳

```python
# models/yolo/detect/predict.py:47-58
preds = nms.non_max_suppression(preds, self.args.conf, self.args.iou, ...,
                                max_det=self.args.max_det,
                                nc=len(self.model.names),
                                end2end=getattr(self.model, "end2end", False))
```

predictor 는 **항상** 이 함수를 부른다. `end2end=True` 면 내부에서 IoU 억제를 건너뛰고
conf 필터 + max_det 만 한다. 즉 "NMS 없음" 은 이 플래그로 결정된다 `[확인]`.

모델 내부의 `Detect.postprocess` + `get_topk_index` 는 NMS 가 아니라 **클래스 최대 score
기준 top-k(max_det)** 선택이다 (head.py:219-257).

→ **질문 4 답**: v8/11 이 쓴 것과 **같은 stock 함수**(`utils/nms.non_max_suppression`)를
   그대로 재사용할 수 있다. 이번 실험은 그 함수의 `end2end` 인자만 False 로 강제하고
   conf/iou/max_det 는 predictor args 를 그대로 쓴다.

## 이번 실험이 고정하는 recipe (기존 artifact/source 에서 읽은 값)

```
conf     0.001     cf_real_eval.py:49 / neg_eval_one.py — 세 mode 공통
iou      0.7       cfg/default.yaml:54 — v8/11 이 arch baseline 에서 쓴 값과 동일
max_det  300       cfg/default.yaml:55 — head.max_det 도 300
imgsz    640
```

결과를 보고 이 값들을 바꾸지 않는다.
