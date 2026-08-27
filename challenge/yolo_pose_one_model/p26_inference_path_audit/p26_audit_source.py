"""SOURCE_AUDIT + DATA_CONTRACT + METHOD_SPEC.  학습 0, 파일 변경 0."""
from __future__ import annotations
import hashlib, inspect, json, os, subprocess, sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/p26_inference_path_audit"
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
AB = f"{Y}/runs_arch_baseline"
POOL_NEG = f"{ROOT}/data/pallet/raw_data/negative_real_20260823/rgb"
DS = f"{Y}/datasets/g38_generic_only"


def sh(c):
    return subprocess.run(c, shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


import torch, ultralytics                                          # noqa: E402
from ultralytics.nn.modules.head import Pose26                     # noqa: E402
from ultralytics.utils import nms as NMS                           # noqa: E402

ab = json.load(open(f"{AB}/RESULT_Y0.json"))
W = os.path.join(ROOT, ab["weights"])
sha = sha256(W)
if sha != ab["checkpoint_sha256"]:
    raise SystemExit(f"HARD BLOCK: checkpoint sha 불일치 {sha} vs {ab['checkpoint_sha256']}")

st = torch.load(W, map_location="cpu", weights_only=False)
sd = st["model"].state_dict() if hasattr(st.get("model"), "state_dict") else {}
o2m = [k for k in sd if any(t in k for t in (".cv2.", ".cv3.", ".cv4.", ".cv4_kpts.", ".cv4_sigma."))
       and "one2one" not in k]
o2o = [k for k in sd if "one2one" in k]

src = {
    "checkpoint": {"path": os.path.relpath(W, ROOT), "sha256": sha,
                   "bytes": os.path.getsize(W), "mtime": os.path.getmtime(W),
                   "matches_arch_baseline_Y0": True,
                   "epoch_in_ckpt": st.get("epoch"), "train_args_epochs":
                       (st.get("train_args") or {}).get("epochs")},
    "Y0_contract_from_artifact": {
        "real_ALL_cbox": ab["real128"]["ALL"]["cbox"],
        "real_ALL_median": ab["real128"]["ALL"]["median"],
        "real_ALL_p90": ab["real128"]["ALL"]["p90"],
        "NIGHT_top1_cbox": ab["real128"]["night_candidate"]["top1_cbox"],
        "source": "runs_arch_baseline/RESULT_Y0.json"},
    "git": {"commit": sh("git rev-parse HEAD"),
            "modified_tracked": [l for l in sh("git status --porcelain -uno").split("\n") if l]},
    "env": {"python": sys.executable, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "ultralytics": ultralytics.__version__,
            "ultralytics_path": os.path.dirname(ultralytics.__file__)},
    "Pose26_source": inspect.getsourcefile(Pose26),
    "Pose26_line": inspect.getsourcelines(Pose26)[1],
    "nms_source": inspect.getsourcefile(NMS.non_max_suppression),
    "branch_parameters_in_checkpoint": {
        "one2many_keys": len(o2m), "one2one_keys": len(o2o),
        "one2many_present": len(o2m) > 0,
        "example_o2m": sorted(o2m)[:3], "example_o2o": sorted(o2o)[:3]},
    "no_training_performed": True,
}
json.dump(src, open(f"{NS}/audits/P26_INFERENCE_PATH_SOURCE_AUDIT.json", "w"),
          indent=2, ensure_ascii=False)
json.dump(src, open(f"{NS}/SOURCE_AUDIT.json", "w"), indent=2, ensure_ascii=False)

LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
r0 = json.load(open(f"{QY}/REAL_PC_Y0.json"))["per_frame"]
pf = [r for r in r0 if r["frame"] not in LEAK]
dc = {"real_positive": {"n": len(pf),
                        "DAY": sum(1 for r in pf if r["domain"] == "DAY"),
                        "NIGHT": sum(1 for r in pf if r["domain"] == "NIGHT"),
                        "source": "REVIEWED_CLEAN_REALDEV_V2 140 - FT leak 12",
                        "membership_sha256": hashlib.sha256(
                            "\n".join(sorted(r["frame"] for r in pf)).encode()).hexdigest()},
      "real_negative": {"n": len(os.listdir(POOL_NEG)),
                        "path": os.path.relpath(POOL_NEG, ROOT),
                        "note": "FT-used 259 와 내용 교집합 0 (실측) — held-out = 2,689 전체"},
      "synthetic_val": {"n": len(os.listdir(f"{DS}/images/val")),
                        "path": os.path.relpath(DS, ROOT)},
      "matches_arch_baseline": {
          "real_n": len(pf) == 128,
          "neg_n": len(os.listdir(POOL_NEG)) == ab["negative"]["n_neg"],
          "synth_n": len(os.listdir(f"{DS}/images/val")) == 1998}}
dc["PASS"] = all(dc["matches_arch_baseline"].values())
json.dump(dc, open(f"{NS}/DATA_CONTRACT.json", "w"), indent=2, ensure_ascii=False)

spec = {
  "experiment": "P26_INFERENCE_PATH_ABLATION (training-0)",
  "question": ("YOLO26 의 약한 real-negative rejection 이 backbone/representation 때문인가, "
               "Pose26 의 one2one/end-to-end candidate-selection 경로 때문인가"),
  "modes": {
    "M0_P26_E2E": "one2one -> _inference -> postprocess (정본과 동일, NMS 없음)",
    "M1_P26_O2M_RAW": "one2many -> _inference -> postprocess (같은 top-k rule, NMS 없음)",
    "M2_P26_O2M_NMS": "one2many -> _inference -> xyxy2xywh -> stock non_max_suppression"},
  "fixed_recipe": {"conf": 0.001, "iou": 0.7, "max_det": 300, "imgsz": 640,
                   "source": "cf_real_eval.py:49 (conf) / cfg/default.yaml:54-55 (iou,max_det)",
                   "frozen_before_results": True},
  "patch_scope": ["Detect/Pose/Pose26.fuse -> no-op (one2many head 보존)",
                  "Pose26.forward -> stock 메서드 재조합 (forward_head/_inference/postprocess)",
                  "O2M_NMS 에서만 non_max_suppression 의 end2end 인자를 False 로 강제"],
  "not_patched": ["conf/iou/max_det", "evaluator 정의", "전처리", "가중치"],
  "gate": {
    "safety_all": {"ALL_cbox_drop_pp_max": 0.02, "NIGHT_any_cbox_drop_pp_max": 0.02,
                   "ALL_median_degrade_rel_max": 0.05},
    "need": 2,
    "benefits": {"A_NIGHT_top1_gain_pp": 0.07, "B_neg_AP_gain": 0.10,
                 "C_FPR95_rel_drop": 0.20, "D_neg_detect040_rel_drop": 0.20},
    "verdicts": ["INFERENCE_PATH_IS_MAJOR_FACTOR", "INFERENCE_PATH_PARTIAL_FACTOR",
                 "INFERENCE_PATH_NOT_FACTOR", "NMS_HARMS_LOCALIZATION"],
    "frozen_before_results": True},
  "forbidden": ["새 학습", "fine-tuning", "loss 변경", "threshold 사후 변경",
                "기존 artifact overwrite"],
}
json.dump(spec, open(f"{NS}/METHOD_SPEC.json", "w"), indent=2, ensure_ascii=False)

print(json.dumps({"sha_match": True, "sha16": sha[:16],
                  "o2m_keys": len(o2m), "o2o_keys": len(o2o),
                  "data": {k: (v if not isinstance(v, dict) else
                               {kk: vv for kk, vv in v.items() if kk in ("n", "DAY", "NIGHT")})
                           for k, v in dc.items() if k != "matches_arch_baseline"},
                  "DATA_PASS": dc["PASS"]}, indent=2, ensure_ascii=False))
