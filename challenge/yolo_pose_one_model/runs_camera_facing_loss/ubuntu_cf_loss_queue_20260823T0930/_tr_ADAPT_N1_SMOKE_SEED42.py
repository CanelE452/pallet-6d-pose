
import os, sys, json
sys.path.insert(0, "/home/minjae/Documents/github/pallet-pose")
os.environ["A1_CONFIG"] = ""
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
    task="pose", mode="train", model="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt", data="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/adapt_n1_negative/data.yaml",
    epochs=1, batch=32, imgsz=640, optimizer="SGD", lr0=0.002, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=1.0, patience=0,
    single_cls=True, mosaic=0.15, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=5, device=0, workers=8, project="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss", name="ADAPT_N1_SMOKE_SEED42",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
json.dump({"PC_CALL_COUNT": PC["n"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "criterion": type(getattr(c, "one2many", c)).__name__,
           "n_train_batches": len(tr.train_loader), "epochs": 1,
           "data": "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/adapt_n1_negative/data.yaml", "init": "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt"},
          open("/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss/ADAPT_N1_SMOKE_SEED42/RUNTIME_AUDIT.json", "w"), indent=2)
