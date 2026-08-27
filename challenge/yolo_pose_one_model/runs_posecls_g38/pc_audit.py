"""Phase 0 — SOURCE_AUDIT / DATA_CONTRACT / init sha.  학습 0, overwrite 0."""
from __future__ import annotations
import hashlib, inspect, json, os, subprocess, sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/runs_posecls_g38"
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
POOL = f"{ROOT}/data/pallet/training_data/paper_release/v2_prod40k_clean_merged"


def sh(c):
    return subprocess.run(c, shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


import torch, ultralytics                                       # noqa: E402
from ultralytics.utils.loss import PoseLoss26, E2ELoss, KeypointLoss   # noqa: E402
from ultralytics.nn.modules.head import Pose26                  # noqa: E402

src = {
    "commit": sh("git rev-parse HEAD"),
    "branch": sh("git rev-parse --abbrev-ref HEAD"),
    "git_status_short_count": len([l for l in sh("git status --short").split("\n") if l]),
    "git_modified_tracked": [l for l in sh("git status --porcelain -uno").split("\n") if l],
    "isolation": ("git worktree 를 만들지 않았다 — 이 실험이 쓰는 dataset/weights 는 전부 "
                  "untracked/ignored 라 worktree 에는 딸려오지 않는다. 격리는 새 namespace "
                  "(runs_posecls_g38) 로 하고 기존 파일은 건드리지 않는다."),
    "python_executable": sys.executable, "python_version": sys.version.split()[0],
    "torch": torch.__version__, "cuda_torch": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "ultralytics_version": ultralytics.__version__,
    "ultralytics_path": os.path.dirname(ultralytics.__file__),
    "Pose26_source": inspect.getsourcefile(Pose26),
    "Pose26_line": inspect.getsourcelines(Pose26)[1],
    "PoseLoss26_source": inspect.getsourcefile(PoseLoss26),
    "PoseLoss26_line": inspect.getsourcelines(PoseLoss26)[1],
    "E2ELoss_source": inspect.getsourcefile(E2ELoss),
    "KeypointLoss_source": inspect.getsourcefile(KeypointLoss),
    "pip_install_performed": False,
    "loss_py_sha256": sha256(os.path.join(os.path.dirname(ultralytics.__file__),
                                          "utils", "loss.py")),
}
json.dump(src, open(f"{NS}/SOURCE_AUDIT.json", "w"), indent=2, ensure_ascii=False)

# -------------------------------------------------------------- DATA CONTRACT
man = json.load(open(f"{QY}/G38_GENERIC_ONLY_MANIFEST.json"))
tr, va = man["train"], man["val"]
DS = f"{Y}/datasets/g38_generic_only"
img_tr = sorted(os.listdir(f"{DS}/images/train"))
img_va = sorted(os.listdir(f"{DS}/images/val"))
lab_tr = sorted(os.listdir(f"{DS}/labels/train"))
gen_tr = [os.path.basename(l).replace("_rgb.png", "").strip()
          for l in open(f"{Y}/manifests/generic_train.txt") if l.strip()]
gen_va = [os.path.basename(l).replace("_rgb.png", "").strip()
          for l in open(f"{Y}/manifests/generic_val.txt") if l.strip()]

lab0 = json.load(open(f"{POOL}/labels/{gen_tr[0]}_label.json"))
conv = lab0["objects"][0]["keypoint_convention"]
# YOLO 라벨에서 kpt 수 실측
sample = open(f"{DS}/labels/train/{img_tr[0][:-4]}.txt").read().split("\n")[0].split()
nkp = (len(sample) - 5) // 3

dc = {
    "dataset": os.path.relpath(DS, ROOT),
    "manifest": os.path.relpath(f"{QY}/G38_GENERIC_ONLY_MANIFEST.json", ROOT),
    "train_declared": len(tr), "train_unique": len(set(tr)), "train_files": len(img_tr),
    "val_declared": len(va), "val_unique": len(set(va)), "val_files": len(img_va),
    "label_files_train": len(lab_tr),
    "train_int_val": len(set(tr) & set(va)),
    "stems_match_generic_manifest": (sorted(s.replace("G__", "") for s in tr) == sorted(gen_tr)
                                     and sorted(s.replace("G__", "") for s in va) == sorted(gen_va)),
    "keypoint_convention": conv, "n_keypoints_in_yolo_label": nkp,
    "target_specific_positive": 0, "real_positive": 0, "real_pseudo_label": 0,
    "synthetic_negative": 0,
    "train_sha256": hashlib.sha256("\n".join(sorted(tr)).encode()).hexdigest(),
    "val_sha256": hashlib.sha256("\n".join(sorted(va)).encode()).hexdigest(),
}
dc["PASS"] = bool(dc["train_unique"] == 38002 and dc["train_files"] == 38002
                  and dc["val_unique"] == dc["val_files"] and dc["train_int_val"] == 0
                  and dc["stems_match_generic_manifest"]
                  and conv == "camera_dynamic_0123_v4" and nkp == 9)
json.dump(dc, open(f"{NS}/DATA_CONTRACT.json", "w"), indent=2, ensure_ascii=False)

# -------------------------------------------------------------- INIT
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
init = {"path": os.path.relpath(INIT, ROOT), "exists": os.path.exists(INIT),
        "sha256": sha256(INIT) if os.path.exists(INIT) else None,
        "bytes": os.path.getsize(INIT) if os.path.exists(INIT) else None,
        "shared_by": ["Y0", "Y1"],
        "not_a_finetune_of": "G38 60ep checkpoint 아님 — clean pretrained 에서 30ep"}
json.dump(init, open(f"{NS}/INIT_AUDIT.json", "w"), indent=2, ensure_ascii=False)

print(json.dumps({"commit": src["commit"][:12], "ultralytics": src["ultralytics_version"],
                  "torch": src["torch"], "gpu": src["gpu"],
                  "DATA": {k: dc[k] for k in ("train_unique", "train_files", "val_unique",
                                              "val_files", "train_int_val",
                                              "keypoint_convention",
                                              "n_keypoints_in_yolo_label", "PASS")},
                  "INIT": {"exists": init["exists"], "sha16": (init["sha256"] or "")[:16]}},
                 indent=2, ensure_ascii=False))
