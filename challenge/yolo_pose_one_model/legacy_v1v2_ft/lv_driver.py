"""LEGACY_V1V2_P0_10K FT — train -> eval -> verdict -> notify 를 한 파일에서.

완료 판정은 exit code 나 프로세스 존재로 하지 않는다. 산출물(weights/last.pt,
결과 JSON)의 존재로만 한다.

gate 는 GATE_PREREG.json 에서 읽는다 — 이 파일 안에서 문턱을 만들지 않는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
YO = REPO / "challenge/yolo_pose_one_model"
Q = YO / "runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
PY_ENV = "/home/minjae/anaconda3/envs/pallet-yolo26/bin/python"

BASE = YO / "runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt"
DATA = YO / "datasets/legacy_v1v2_p0_10k/data.yaml"
SEED = int(__import__("os").environ.get("LV_SEED", "42"))
NAME = f"LV1V2_FT_15EP_SEED{SEED}"
TAG = "LV1V2FT" if SEED == 42 else f"LV1V2FT{SEED}"
RUN = HERE / "runs" / NAME
LOG = HERE / "DRIVER_LOG.txt"


def log(msg):
    line = f"[lv_driver] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, tag):
    log(f"RUN {tag}: {' '.join(str(c) for c in cmd)}")
    p = subprocess.run([str(c) for c in cmd], cwd=str(REPO))
    log(f"END {tag}: returncode={p.returncode}")
    return p.returncode


def train():
    if (RUN / "weights/last.pt").exists():
        log("train skipped — weights/last.pt already exists")
        return
    RUN.parent.mkdir(parents=True, exist_ok=True)
    # ADAPT_N0_15EP_SEED42 레시피 그대로. 바뀌는 건 data 하나뿐(단일변수).
    script = HERE / "_tr_LV1V2_FT.py"
    script.write_text(f'''
import os, sys, json
sys.path.insert(0, "{REPO}")
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
        _FIRED.append([int(x), int(x) + {SEED}])
        return _REAL_GEN.manual_seed(self, x + {SEED})

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
    task="pose", mode="train", model="{BASE}", data="{DATA}",
    epochs=15, batch=32, imgsz=640, optimizer="SGD", lr0=0.002, lrf=0.01,
    cos_lr=True, close_mosaic=10, warmup_epochs=1.0, patience=0,
    single_cls=True, mosaic=0.15, scale=0.25, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, erasing=0.4, seed={SEED}, deterministic=True,
    save_period=5, device=0, workers=2, project="{RUN.parent}", name="{NAME}",
    exist_ok=True, resume=False, val=True, plots=False))
tr.train()
_json.dump({{"fired": _FIRED, "seed": {SEED}}},
           open("{RUN}/DATALOADER_SEED_PATCH.json", "w"), indent=2)
json.dump({{"n_train_batches": len(tr.train_loader), "epochs": 15,
           "data": "{DATA}", "init": "{BASE}", "lr0": 0.002, "recipe": "ADAPT_N0_15EP_SEED42", "seed": {SEED},
           "dataloader_seed_patch": "generator.manual_seed(CONST + {SEED}) — 8.4.60 하드코딩 우회",
           "deviation": "workers 8->2 (first attempt OOM-killed, shmem-rss 9.1GB). hyperparameters unchanged."}},
          open("{RUN}/RUNTIME_AUDIT.json", "w"), indent=2)
''', encoding="utf-8")
    run([PY_ENV, script], "train")
    if not (RUN / "weights/last.pt").exists():
        log("FAIL: training produced no weights/last.pt")
        sys.exit(1)
    log("train done — weights/last.pt present")


def evaluate(tag, weights):
    """세 평가를 모두 돌린다. 이미 있는 결과는 건너뛴다(대조군 재사용)."""
    todo = [
        (Q / f"REAL_{tag}.json", [PY_ENV, Q / "cf_real_eval.py", "--weights", weights, "--tag", tag]),
        (Q / f"NIGHT_CAND_{tag}.json", [PY_ENV, Q / "night_cand_one.py", "--weights", weights, "--tag", tag]),
        (Q / f"NEGSCORE_{tag}.json", [PY_ENV, Q / "neg_eval_one.py", "--weights", weights, "--tag", tag]),
    ]
    for out, cmd in todo:
        if out.exists():
            log(f"eval skipped ({tag}) — {out.name} exists")
            continue
        run(cmd, f"eval:{tag}:{out.name}")
        if not out.exists():
            log(f"FAIL: {out.name} not produced")
            sys.exit(1)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def verdict():
    gate = load(HERE / "GATE_PREREG.json")
    a_real, b_real = load(Q / "REAL_G38.json"), load(Q / f"REAL_{TAG}.json")
    a_ni, b_ni = load(Q / "NIGHT_CAND_G38.json"), load(Q / f"NIGHT_CAND_{TAG}.json")

    A, B = a_real["correct_box"], b_real["correct_box"]
    d_cbox = b_real["correct_box_recall"] - a_real["correct_box_recall"]
    r_med = (A["corner_median"] - B["corner_median"]) / A["corner_median"]
    r_p90 = (A["corner_p90"] - B["corner_p90"]) / A["corner_p90"]
    d_top1 = round(b_ni["top1_cbox"] * b_ni["n"]) - round(a_ni["top1_cbox"] * a_ni["n"])
    d_any = round(b_ni["any_cbox"] * b_ni["n"]) - round(a_ni["any_cbox"] * a_ni["n"])
    r_ni_p90 = (A["night_p90"] - B["night_p90"]) / A["night_p90"]
    d_margin = b_ni["margin_median"] - a_ni["margin_median"]

    h = gate["hits"]
    hits = {
        "all_cbox_+2pp": d_cbox >= h["all_cbox_pp"],
        "all_median_-8%": r_med >= h["all_median_rel"],
        "all_p90_-10%": r_p90 >= h["all_p90_rel"],
        "night_top1_+3f": d_top1 >= h["night_top1_cbox_frames"],
        "night_p90_-15%": r_ni_p90 >= h["night_p90_rel"],
        "night_margin_+0.10": d_margin >= h["night_margin_abs"],
    }
    g = gate["guards"]
    guards = {
        "all_cbox": d_cbox <= g["all_cbox_pp"],
        "night_any_cbox": d_any <= g["night_any_cbox_frames"],
        "all_p90": (-r_p90) >= g["all_p90_rel_worse"],
    }
    n_hit = sum(hits.values())
    tripped = [k for k, v in guards.items() if v]
    if tripped:
        v = "HARM"
    elif n_hit >= gate["hits_need"]:
        v = "GAIN"
    else:
        v = "NO_GAIN"

    res = {
        "verdict": v, "hits_passed": n_hit, "hits_need": gate["hits_need"],
        "hits": hits, "guards": guards, "guards_tripped": tripped,
        "deltas": {
            "correct_box_recall": {"G38": a_real["correct_box_recall"],
                                   TAG: b_real["correct_box_recall"], "delta_pp": d_cbox},
            "corner_median": {"G38": A["corner_median"], TAG: B["corner_median"], "rel_gain": r_med},
            "corner_p90": {"G38": A["corner_p90"], TAG: B["corner_p90"], "rel_gain": r_p90},
            "night_top1_frames": {"G38": round(a_ni["top1_cbox"] * a_ni["n"]),
                                  TAG: round(b_ni["top1_cbox"] * b_ni["n"]), "delta": d_top1},
            "night_any_frames": {"G38": round(a_ni["any_cbox"] * a_ni["n"]),
                                 TAG: round(b_ni["any_cbox"] * b_ni["n"]), "delta": d_any},
            "night_p90": {"G38": A["night_p90"], TAG: B["night_p90"], "rel_gain": r_ni_p90},
            "night_margin": {"G38": a_ni["margin_median"], TAG: b_ni["margin_median"], "delta": d_margin},
            "detection_recall": {"G38": a_real["detection_recall"], TAG: b_real["detection_recall"]},
        },
        "scope": gate["scope"],
        "n": {"real": a_real["n_total"], "night": a_ni["n"]},
    }
    for tag in ("G38", TAG):
        p = Q / f"NEGSCORE_{tag}.json"
        if p.exists():
            res.setdefault("negative", {})[tag] = load(p)
    (HERE / f"RESULT_{TAG}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def report(res):
    d = res["deltas"]
    def row(name, a, b, extra=""):
        return f"{name:22s} {a:>10.4f} {b:>10.4f}   {extra}"

    def rel(x):
        """양수 = 개선, 음수 = 악화. 부호에 맞는 단어를 붙인다."""
        return f"{abs(x)*100:.1f}% {'개선' if x >= 0 else '악화'}"
    lines = [
        f"LEGACY_V1V2_P0_10K FT on G38 (seed {SEED}) — 사전등록 gate 판정",
        "",
        f"{'metric':22s} {'G38':>10s} {TAG:>10s}   변화",
        "─" * 72,
        row("correct_box_recall", d["correct_box_recall"]["G38"], d["correct_box_recall"][TAG],
            f"{d['correct_box_recall']['delta_pp']*100:+.2f} pp"),
        row("corner_median (px)", d["corner_median"]["G38"], d["corner_median"][TAG],
            rel(d['corner_median']['rel_gain'])),
        row("corner_p90 (px)", d["corner_p90"]["G38"], d["corner_p90"][TAG],
            rel(d['corner_p90']['rel_gain'])),
        row("night_p90 (px)", d["night_p90"]["G38"], d["night_p90"][TAG],
            rel(d['night_p90']['rel_gain'])),
        row("night_margin", d["night_margin"]["G38"], d["night_margin"][TAG],
            f"{d['night_margin']['delta']:+.4f}"),
        row("detection_recall", d["detection_recall"]["G38"], d["detection_recall"][TAG]),
        f"{'night top1 (frames)':22s} {d['night_top1_frames']['G38']:>10d} "
        f"{d['night_top1_frames'][TAG]:>10d}   {d['night_top1_frames']['delta']:+d}",
        f"{'night any  (frames)':22s} {d['night_any_frames']['G38']:>10d} "
        f"{d['night_any_frames'][TAG]:>10d}   {d['night_any_frames']['delta']:+d}",
        "",
        f"hits {res['hits_passed']}/{res['hits_need']} 필요 — {res['hits']}",
        f"guards {res['guards']}  tripped={res['guards_tripped']}",
        "",
        f"VERDICT = {res['verdict']}",
        "",
        f"★ {res['scope']}",
        f"★ n(real)={res['n']['real']}  n(night)={res['n']['night']}  seed 1개.",
    ]
    txt = "\n".join(lines)
    (HERE / f"RESULT_{TAG}.md").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt, flush=True)
    return txt


if __name__ == "__main__":
    log("START")
    if not DATA.exists():
        log(f"FAIL: {DATA} 없음 — lv_build_dataset.py 먼저")
        sys.exit(1)
    train()
    evaluate("G38", BASE)                       # 대조군 (기존 결과 있으면 재사용)
    evaluate(TAG, RUN / "weights/last.pt")
    report(verdict())
    log("DONE")
