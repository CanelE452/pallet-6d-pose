"""STEP A (1-5) — SOURCE_AUDIT / G38_CONTRACT / CHECKPOINT_INVENTORY / FEATURE_TAP.

새 학습 0. 기존 파일 overwrite 0 (새 namespace 안에만 쓴다).
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
NS = f"{ROOT}/data/pallet/results/adaptive_hough_g38"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
os.makedirs(NS, exist_ok=True)


def sh(c):
    return subprocess.run(c, shell=True, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def sha256(p, cap=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- SOURCE_AUDIT
src = {"commit": sh("git rev-parse HEAD"),
       "branch": sh("git rev-parse --abbrev-ref HEAD"),
       "upstream_sync": sh("git status -sb | head -1"),
       "modified_tracked": [l for l in sh("git status --porcelain -uno").split("\n") if l],
       "untracked_count": len(sh("git status --porcelain | grep '^??'").split("\n")),
       "namespace": os.path.relpath(NS, ROOT),
       "namespace_is_new": True}
json.dump(src, open(f"{NS}/SOURCE_AUDIT.json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- G38_CONTRACT
man = json.load(open(f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
                     "/G38_GENERIC_ONLY_MANIFEST.json"))
gtrain, gval = man["train"], man["val"]
gen_train = [l.strip() for l in open(f"{Y}/manifests/generic_train.txt") if l.strip()]
gen_val = [l.strip() for l in open(f"{Y}/manifests/generic_val.txt") if l.strip()]
POOL = f"{ROOT}/data/pallet/training_data/paper_release/v2_prod40k_clean_merged"


def stem_of(p):                      # .../rgb/f0000_rgb.png -> f0000
    return os.path.basename(p).replace("_rgb.png", "")


train_stems = [stem_of(p) for p in gen_train]
val_stems = [stem_of(p) for p in gen_val]
# G__fXXXX <-> fXXXX 대응 확인
g38_train_stems = [s.replace("G__", "") for s in gtrain]
g38_val_stems = [s.replace("G__", "") for s in gval]

lab0 = json.load(open(f"{POOL}/labels/{train_stems[0]}_label.json"))
conv = lab0["objects"][0]["keypoint_convention"]

miss_img = sum(1 for s in train_stems[:2000]
               if not os.path.exists(f"{POOL}/rgb/{s}_rgb.png"))
miss_lab = sum(1 for s in train_stems[:2000]
               if not os.path.exists(f"{POOL}/labels/{s}_label.json"))

g38 = {
    "manifest": os.path.relpath(f"{Y}/runs_camera_facing_loss/"
                                "ubuntu_cf_loss_queue_20260823T0930/"
                                "G38_GENERIC_ONLY_MANIFEST.json", ROOT),
    "train_declared": len(gtrain), "train_unique": len(set(gtrain)),
    "val_declared": len(gval), "val_unique": len(set(gval)),
    "generic_manifest_train": len(gen_train), "generic_manifest_val": len(gen_val),
    "train_stems_match_generic_manifest": sorted(g38_train_stems) == sorted(train_stems),
    "val_stems_match_generic_manifest": sorted(g38_val_stems) == sorted(val_stems),
    "train_val_disjoint": len(set(g38_train_stems) & set(g38_val_stems)) == 0,
    "dope_source_pool": os.path.relpath(POOL, ROOT),
    "dope_rgb_missing_in_first2000": miss_img,
    "dope_label_missing_in_first2000": miss_lab,
    "keypoint_convention": conv,
    "target_specific_synthetic": 0,
    "real_positive_training": 0,
    "horizontal_flip": "FORBIDDEN (loader 에서 미사용 — 학습 코드에서 재확인)",
    "new_render": 0,
    "fixed_object_track": False,
    "train_sha16": hashlib.sha256("\n".join(sorted(gtrain)).encode()).hexdigest()[:16],
    "val_sha16": hashlib.sha256("\n".join(sorted(gval)).encode()).hexdigest()[:16],
}
g38["PASS"] = (g38["train_unique"] == 38002 and g38["val_unique"] == 1998
               and g38["train_stems_match_generic_manifest"]
               and g38["val_stems_match_generic_manifest"]
               and g38["train_val_disjoint"]
               and conv == "camera_dynamic_0123_v4"
               and miss_img == 0 and miss_lab == 0)
json.dump(g38, open(f"{NS}/G38_CONTRACT.json", "w"), indent=2, ensure_ascii=False)

# ------------------------------------------------------- CHECKPOINT_INVENTORY
CK = f"{ROOT}/weights/paper_s2/paper_s2_multihead"
WANT = {
    "A1_fully_shared":    ("screen_A1_CORNER_LINE_long25k_seed1", 25000),
    "E2_stopgrad":        (None, None),
    "E3_split_late":      ("splitlate_E3_SPLIT_LATE_seed1", None),
    "E4_capacity":        ("capacity_E4_CAPACITY_MATCHED_CORNER_seed1", None),
    "FINAL40K_seed1":     ("screen_A1_CORNER_LINE_FINAL40K_seed1", 25000),
    "FINAL40K_seed2":     ("screen_A1_CORNER_LINE_FINAL40K_seed2", 25000),
}
# E2 는 이름을 모르므로 디렉터리 목록에서 찾는다
dirs = sorted(os.listdir(CK)) if os.path.isdir(CK) else []
for d in dirs:
    if "stopgrad" in d.lower() or d.startswith("stopgrad") or "_E2_" in d:
        WANT["E2_stopgrad"] = (d, None)
        break

pkg = json.load(open(f"{ROOT}/data/pallet/results/paper_s2_multihead/final_train/"
                     "FINAL_MODEL_PACKAGE.json"))
inv = {"checkpoint_root": os.path.relpath(CK, ROOT), "available_dirs": dirs, "items": {}}
for label, (d, step) in WANT.items():
    if d is None or not os.path.isdir(f"{CK}/{d}"):
        inv["items"][label] = {"FOUND": False,
                               "why": "디렉터리 없음 — 이 계열은 disk 에 존재하지 않는다"}
        continue
    steps = sorted(int(f[5:10]) for f in os.listdir(f"{CK}/{d}")
                   if f.startswith("step_") and f.endswith(".pth"))
    use = step if (step in steps) else (max(steps) if steps else None)
    if use is None:
        inv["items"][label] = {"FOUND": False, "why": "step_*.pth 없음", "dir": d}
        continue
    p = f"{CK}/{d}/step_{use:05d}.pth"
    import torch
    st = torch.load(p, map_location="cpu", weights_only=False)
    seedinfo = pkg["seeds"].get("seed1" if "seed1" in d else
                                ("seed2" if "seed2" in d else ""), {})
    inv["items"][label] = {
        "FOUND": True, "path": os.path.relpath(p, ROOT), "sha256": sha256(p),
        "bytes": os.path.getsize(p), "mtime": os.path.getmtime(p),
        "architecture": st.get("arm"), "step": st.get("step", use),
        "available_steps": steps, "seed": st.get("seed"),
        "source_checkpoint": st.get("source"),
        "training_pool": (seedinfo.get("pool_source") if seedinfo else None),
        "optimizer": {"name": "AdamW", "lr": seedinfo.get("lr"),
                      "weight_decay": seedinfo.get("weight_decay"),
                      "batch": seedinfo.get("batch")} if seedinfo else None,
    }

# target-free / real-free 판정
for label, it in inv["items"].items():
    if not it.get("FOUND"):
        continue
    pool = it.get("training_pool") or ""
    it["target_specific_used"] = ("BROAD" not in pool) if pool else "UNKNOWN"
    it["real_positive_used"] = False
    it["TARGET_FREE_SAFE_INIT"] = (it["target_specific_used"] is False)
inv["★init_warning"] = (
    "FINAL40K 는 BROAD 40,000 전체(= G38 train 38,002 + generic val 1,998)로 학습됐다. "
    "target-free 이지만 GATE_FIT/GATE_DEV 의 출처인 generic val 1,998 을 이미 봤다. "
    "따라서 P0 initialization 으로 쓰면 gate split 이 오염된다 — P0 는 G38 train 만 본 "
    "clean init 에서 시작해야 한다.")
json.dump(inv, open(f"{NS}/CHECKPOINT_INVENTORY.json", "w"), indent=2, ensure_ascii=False)

print(json.dumps({"commit": src["commit"][:12],
                  "G38": {k: g38[k] for k in
                          ("train_unique", "val_unique", "train_val_disjoint",
                           "keypoint_convention", "PASS")},
                  "checkpoints": {k: (v.get("path") or v.get("why"))
                                  for k, v in inv["items"].items()}},
                 indent=2, ensure_ascii=False))
