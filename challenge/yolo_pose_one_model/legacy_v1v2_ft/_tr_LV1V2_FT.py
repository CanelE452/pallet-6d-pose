
import os, sys, json
sys.path.insert(0, "/home/minjae/Documents/github/pallet-pose")
os.environ["A1_CONFIG"] = ""
os.environ["PSPC_CONFIG"] = ""

# ── seed 를 실제로 먹이기 위한 패치 ───────────────────────────────────────
# ultralytics 8.4.60 data/build.py 는 dataloader generator 를
#   generator.manual_seed(6148914691236517205 + RANK)
# 로 **하드코딩**한다. args.seed 는 여기에 들어가지 않는다. FT 는 고정 가중치에서
# 시작하므로 남는 무작위성은 데이터 순서 + worker 증강 RNG 뿐인데 둘 다 이 generator
# 에서 파생된다 → args.seed 만 바꾸면 결과가 비트 단위로 동일해진다(실측 확인).
#
# ★ build_dataloader 는 trainer 모듈이 import 시점에 **자기 네임스페이스로 바인딩**한다.
#   원본 모듈만 패치하면 안 먹는다(실측: dt.build_dataloader is patched -> False).
#   그래서 바인딩된 모든 곳을 갈아끼운다.
import json as _json
import torch as _torch
import ultralytics.data.build as _UB
import ultralytics.data as _UD
import ultralytics.models.yolo.detect.train as _DT
import ultralytics.models.yolo.pose.train as _PT

_FIRED = []
_REAL_GEN = _torch.Generator
class _SeededGen(_REAL_GEN):
    def manual_seed(self, x):
        _FIRED.append([int(x), int(x) + 43])
        return _REAL_GEN.manual_seed(self, x + 43)

_orig_build = _UB.build_dataloader
def _build(*a, **k):
    _UB.torch.Generator = _SeededGen
    try:
        return _orig_build(*a, **k)
    finally:
        _UB.torch.Generator = _REAL_GEN

for _m in (_UB, _UD, _DT, _PT):
    if hasattr(_m, "build_dataloader"):
        _m.build_dataloader = _build
assert _DT.build_dataloader is _build, "trainer 바인딩 패치 실패"
# ─────────────────────────────────────────────────────────────────────────

from ultralytics.models.yolo.pose import PoseTrainer
tr = PoseTrainer(overrides=dict(
    task="pose", mode="train", model="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt", data="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/legacy_v1v2_p0_10k/data.yaml",
    epochs=15, batch=32, imgsz=640, optimizer="SGD", lr0=0.002, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=1.0, patience=0,
    single_cls=True, mosaic=0.15, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed=43, deterministic=True,
    save_period=5, device=0, workers=2, project="/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/legacy_v1v2_ft/runs", name="LV1V2_FT_15EP_SEED43",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
_json.dump({"fired": _FIRED, "seed": 43},
           open("/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/legacy_v1v2_ft/runs/LV1V2_FT_15EP_SEED43/DATALOADER_SEED_PATCH.json", "w"), indent=2)
json.dump({"n_train_batches": len(tr.train_loader), "epochs": 15,
           "data": "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/legacy_v1v2_p0_10k/data.yaml", "init": "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt", "lr0": 0.002, "recipe": "ADAPT_N0_15EP_SEED42", "seed": 43,
           "dataloader_seed_patch": "generator.manual_seed(CONST + 43) — 8.4.60 하드코딩 우회",
           "deviation": "workers 8->2 (first attempt OOM-killed, shmem-rss 9.1GB). hyperparameters unchanged."},
          open("/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/legacy_v1v2_ft/runs/LV1V2_FT_15EP_SEED43/RUNTIME_AUDIT.json", "w"), indent=2)
