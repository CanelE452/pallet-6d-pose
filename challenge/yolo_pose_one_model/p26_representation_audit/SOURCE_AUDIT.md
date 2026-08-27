# SOURCE AUDIT

```
checkpoint  challenge/yolo_pose_one_model/runs_posecls_g38/
            Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt
sha256      37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1   기존 audit 과 일치
mtime       2026-08-25 16:07:28
commit      96ddf1967ecee2759e5d36578a84f2e4eb021efe                          기대값과 일치
python      3.10.20   torch 2.1.1+cu118   CUDA 11.8   RTX 3080
ultralytics 8.4.60    .../envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics
```

## module 경로 (실측)

```
Pose26                    nn/modules/head.py:666        end2end True, nl 3, nc 1, reg_max 1
                                                        kpt_shape [9,3], max_det 300
                                                        stride [8, 16, 32]
Detect.postprocess        nn/modules/head.py:219
Detect.get_topk_index     nn/modules/head.py:235        NMS 가 아니라 class-max top-k
one2one cls branch        model.model[-1].one2one_cv3[i]
                          Sequential( (0) Sequential(DWConv, Conv)
                                      (1) Sequential(DWConv, Conv)
                                      (2) Conv2d(64 -> 1) )
one2one pose branch       model.model[-1].one2one_cv4[i]   (C=45)
neck -> head 입력          Detect.forward 의 x[0..2]; one2one 은 x_detach 를 받는다
```

## fuse 는 호출하지 않았다

`nn/tasks.py:253` 이 end2end 모델의 `head.fuse()` 를 불러 one2many module 을 지운다.
이번 audit 은 module graph 원형이 필요하므로 `Detect/Pose/Pose26.fuse` 를 no-op 으로 두고
모델을 로드했다. 이 조치가 one2one 수치를 바꾸지 않는다는 것은 M0 parity 로 증명했다.
