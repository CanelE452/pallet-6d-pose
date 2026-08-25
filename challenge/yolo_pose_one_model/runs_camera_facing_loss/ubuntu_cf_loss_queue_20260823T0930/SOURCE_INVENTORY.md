# SOURCE INVENTORY — camera-facing loss queue

```
role                                                 size              sha16
------------------------------------------------------------------------------
fixed matched10K manifest view                        512   6e3d79a7bf0c0008
★ CF dataset view (신규)                                215   162b51362cc96b9c
원본 camera-facing 라벨 소스                                213   966b79fb3c8b3cd6
broad40k 와 내용 동일(검증)                                  213   966b79fb3c8b3cd6
pretrained init                                   7878574   eb3bb8268828aeaf
60ep Standard recipe lock                            1953   b92e57b1be2d9c42
QA-clean real candidate                            263775   813ea97b6532591b
yolo26n-ft deployment reference                   6548455   ba4ef77da0eb3b0b
custom criterion (project-local)                    12002   061d40e18280237e
```

경로:
- `/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml`
- `/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/v1_cf_matched10k/data.yaml`
- `/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/broad40k/data.yaml`
- `/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/paper_generic_v1/data.yaml`
- `/home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo26n-pose.pt`
- `/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/args.yaml`
- `/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json`
- `/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt`
- `/home/minjae/Documents/github/pallet-pose/pallet_yolo_loss/symmetry.py`