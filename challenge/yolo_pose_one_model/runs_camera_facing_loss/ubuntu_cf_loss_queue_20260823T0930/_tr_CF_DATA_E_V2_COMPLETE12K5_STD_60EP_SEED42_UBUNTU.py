
import os, sys, json
sys.path.insert(0, "/home/minjae/Documents/github/pallet-pose")
os.environ["A1_CONFIG"] = ""          # STANDARD only
os.environ["PSPC_CONFIG"] = ""
from pallet_yolo_loss.loss import PSPCPoseLoss26
import pallet_yolo_loss.symmetry as SY
PC = {"n": 0}
_pc = PSPCPoseLoss26.projective_loss
def _spy(self, *a, **k):
    PC["n"] += 1
    return _pc(self, *a, **k)
PSPCPoseLoss26.projective_loss = _spy
SY.ROLE_CALLS["n"] = 0
from ultralytics.models.yolo.pose import PoseTrainer
tr = PoseTrainer(overrides=dict(
    task="pose", mode="train", model="/home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo26n-pose.pt", data="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/v2_cf_complete/data.yaml",
    epochs=60, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, pose=12.0, kobj=1.0, warmup_epochs=3.0,
    patience=0, single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015,
    hsv_s=0.5, hsv_v=0.35, fliplr=0.0, flipud=0.0, erasing=0.4,
    seed=42, deterministic=True, save_period=10, device=0, workers=8,
    project="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss", name="CF_DATA_E_V2_COMPLETE12K5_STD_60EP_SEED42_UBUNTU", exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
json.dump({"PC_CALL_COUNT": PC["n"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "criterion": type(getattr(c, "one2many", c)).__name__,
           "data": "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/v2_cf_complete/data.yaml", "epochs": 60, "seed": 42},
          open("/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss/CF_DATA_E_V2_COMPLETE12K5_STD_60EP_SEED42_UBUNTU/RUNTIME_AUDIT.json", "w"), indent=2)
