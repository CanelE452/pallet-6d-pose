"""FAST SCREEN 단일 arm 학습 — 사전등록 계약 그대로. 30ep, patience 0, last.pt primary."""
import argparse, json, os, subprocess, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=30)
A = ap.parse_args()

NAME = f"ROOTFAST_{A.arm}_30EP_SEED42"
d = f"{CFR}/{NAME}"
_rp = f"{d}/results.csv"
if (os.path.exists(f"{d}/weights/last.pt") and os.path.exists(_rp)
        and len(open(_rp).read().strip().split("\n")) - 1 >= A.epochs):
    print(f"{NAME} 이미 완료 ({A.epochs}ep)")
    sys.exit(0)
os.makedirs(d, exist_ok=True)
if not os.path.exists(f"{d}/PURPOSE.md"):
    open(f"{d}/PURPOSE.md", "w").write(
        f"[소비처] 논문/Render 결정 — OLD target 정보가 identity 인지 support 인지 판정 ({A.arm})\n"
        "[문장]  target-free 로 geometry/viewpoint support 만 맞춰도 NIGHT ranking 이 회복된다.\n")
code = f'''
import os, sys, json
sys.path.insert(0, "{ROOT}")
os.environ["A1_CONFIG"] = ""
os.environ["PSPC_CONFIG"] = ""
from pallet_yolo_loss.loss import PSPCPoseLoss26
import pallet_yolo_loss.symmetry as SY
PC = {{"n": 0}}
_pc = PSPCPoseLoss26.projective_loss
def _spy(self, *a, **k):
    PC["n"] += 1
    return _pc(self, *a, **k)
PSPCPoseLoss26.projective_loss = _spy
SY.ROLE_CALLS["n"] = 0
from ultralytics.models.yolo.pose import PoseTrainer
tr = PoseTrainer(overrides=dict(
    task="pose", mode="train", model="{INIT}", data="{A.data}",
    epochs={A.epochs}, batch=32, imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=3.0, patience=0,
    single_cls=True, mosaic=0.3, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=42, deterministic=True,
    save_period=10, device=0, workers=8, project="{CFR}", name="{NAME}",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
c = getattr(tr.model, "criterion", None)
json.dump({{"PC_CALL_COUNT": PC["n"], "ROLE_CALLS": SY.ROLE_CALLS["n"],
           "criterion": type(getattr(c, "one2many", c)).__name__,
           "data": "{A.data}", "epochs": {A.epochs}, "seed": 42,
           "n_train_batches": len(tr.train_loader)}},
          open("{CFR}/{NAME}/RUNTIME_AUDIT.json", "w"), indent=2)
'''
sc = f"{Q}/_tr_{NAME}.py"
open(sc, "w").write(code)
lf = f"{Q}/_train_{NAME}.log"
with open(lf, "w") as fh:
    r = subprocess.run([sys.executable, "-u", sc], cwd=d, stdout=fh,
                       stderr=subprocess.STDOUT, text=True)
if not os.path.exists(f"{d}/weights/last.pt"):
    tail = "\n".join(open(lf).read().replace("\r", "\n").split("\n")[-20:])
    print(f"FAIL {NAME} rc={r.returncode}\n{tail[-900:]}")
    sys.exit(1)
a = json.load(open(f"{d}/RUNTIME_AUDIT.json"))
nep = len(open(f"{d}/results.csv").read().strip().split("\n")) - 1
print(f"{NAME} OK  {nep}ep  batches/epoch {a['n_train_batches']}  "
      f"PC {a['PC_CALL_COUNT']} role {a['ROLE_CALLS']}  {a['criterion']}")
