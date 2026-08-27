
import os, sys, json
sys.path.insert(0, "/home/minjae/Documents/github/pallet-pose")
os.environ["A1_CONFIG"] = ""
os.environ["PSPC_CONFIG"] = ""
# 옛/신 custom loss 가 끼어들지 않는지 런타임에서 감시만 한다 (주입은 하지 않는다)
from pallet_yolo_loss.loss import PSPCPoseLoss26
import pallet_yolo_loss.symmetry as SY
from pallet_yolo_loss import posecls as PCL
CNT = {"pspc": 0}
_pc = PSPCPoseLoss26.projective_loss
def _spy(self, *a, **k):
    CNT["pspc"] += 1
    return _pc(self, *a, **k)
PSPCPoseLoss26.projective_loss = _spy
SY.ROLE_CALLS["n"] = 0
PCL.CALLS["align"] = 0
from ultralytics.models.yolo.pose import PoseTrainer
tr = PoseTrainer(overrides=dict(
    task="pose", mode="train", model="/home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo11n-pose.pt", data="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/g38_smoke256/data.yaml",
    epochs=1, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=3.0, patience=0,
    single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=10, device=0, workers=8, project="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_arch_baseline", name="SMOKE_B11",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
inner = getattr(c, "one2many", c)
m = tr.model.model[-1]
json.dump({"PSPC_CALLS": CNT["pspc"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "POSEALIGN_CALLS": PCL.CALLS["align"],
           "criterion": type(inner).__name__, "criterion_top": type(c).__name__,
           "head": type(m).__name__, "kpt_shape": list(getattr(m, "kpt_shape", [])),
           "end2end": bool(getattr(tr.model, "end2end", False)),
           "n_train_batches": len(tr.train_loader), "epochs": 1, "seed": 42,
           "init": "/home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo11n-pose.pt", "data": "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/g38_smoke256/data.yaml",
           "fliplr": tr.args.fliplr, "flipud": tr.args.flipud,
           "params": int(sum(p.numel() for p in tr.model.parameters()))},
          open("/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_arch_baseline/SMOKE_B11/RUNTIME_AUDIT.json", "w"), indent=2)
