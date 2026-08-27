
import os, sys, json
sys.path.insert(0, "/home/minjae/Documents/github/pallet-pose")
os.environ["A1_CONFIG"] = "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_pallet_loss/overnight_20260823/TAKL_LOSS_CONFIG.json"
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
from pallet_yolo_loss.trainer import ASCTrainer
tr = ASCTrainer(overrides=dict(
    task="pose", mode="train", model="/home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo26n-pose.pt", data="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml",
    epochs=60, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, pose=12.0, kobj=1.0, warmup_epochs=3.0,
    patience=0, single_cls=True, mosaic=0.3, scale=0.25, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=10, device=0, workers=8, project="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_pallet_loss", name="PSPC_TAKL_V1MATCHED10K_60EP_SEED42",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
crit = getattr(tr.model, "criterion", None)
inner = getattr(crit, "one2many", crit)
a1 = getattr(inner, "a1", None)
json.dump({"PC_CALL_COUNT": PC["n"], "ROLE_MARGIN_CALL_COUNT": SY.ROLE_CALLS["n"],
           "asc_enabled": bool(getattr(a1, "asc_enabled", False)),
           "takl_enabled": bool(getattr(a1, "takl_enabled", False)),
           "nrl_enabled": bool(getattr(a1, "nrl_enabled", False)),
           "takl_tau": getattr(a1, "takl_tau", None),
           "takl_lambda": getattr(a1, "takl_lambda", None),
           "nrl_beta": getattr(a1, "nrl_beta", None),
           "nrl_lambda": getattr(a1, "nrl_lambda", None),
           "sym_assets": list(getattr(a1, "sym_assets", ())),
           "seed": 42}, open("/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_pallet_loss/PSPC_TAKL_V1MATCHED10K_60EP_SEED42/RUNTIME_AUDIT.json", "w"), indent=2)
