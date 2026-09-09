#!/usr/bin/env python3
"""§11 corrected real-GT baseline — 학습부터 판정·알림까지 한 파일에서.

    STEP 1  R0 -> [FT 40ep] x {legacy, contract} x seed{0,1,2}   6 run
    STEP 2  R0 + 6 checkpoint 를 **같은 held-out 297장**으로 재채점
    STEP 3  사전등록 gate 로 판정
    STEP 4  판정과 핵심 수치를 알린다

완료 판정은 exit code 가 아니라 산출물(weights/best.pt + RESULT.json 의 최종 마크)로 한다.
gate 는 `data/pallet/results/next_accuracy_v2/METHOD_LOCK.json` 에 사전등록한 값을
여기 하드코딩한다 — 결과를 보고 고치지 못하게.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ONE = REPO / "challenge/yolo_pose_one_model"
RESULTS = REPO / "data/pallet/results/next_accuracy_v2"
NOTIFY = Path.home() / ".claude/hooks/discord-notify.sh"
PAD = 100

BASE = ONE / ("spatial_concat_scratch/runs/"
              "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
BASE_SHA = "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7"
ARMS = {"legacy": ONE / "datasets/live_gt_legacy_v2",
        "contract": ONE / "datasets/live_gt_contract_v2"}
SEEDS = [0, 1, 2]

RECIPE = dict(
    epochs=40, batch=32, imgsz=640,
    optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=True, warmup_epochs=3.0,
    momentum=0.937, weight_decay=0.0005,
    pose=12.0, kobj=1.0, box=7.5, cls=0.5, dfl=1.5,
    mosaic=0.3, close_mosaic=10, scale=0.25,
    fliplr=0.0, flipud=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    patience=0, deterministic=True, single_cls=True, workers=2,
    val=True, plots=False, verbose=False, exist_ok=True,
)

# --- 사전등록 gate (METHOD_LOCK.json) --------------------------------------
GATE_STRATUM = "<8"
GATE_RULE = "FT_CONTRACT - R0 의 <8 층 세션클러스터 95% CI 가 0 을 배제하고 FT 우세"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def notify(msg: str) -> None:
    print(msg, flush=True)
    if NOTIFY.is_file():
        try:
            subprocess.run(["bash", str(NOTIFY), msg], timeout=30, check=False)
        except Exception as e:      # 알림 실패가 실험을 죽이면 안 된다
            print(f"   (알림 실패: {e})", flush=True)


def drop_label_cache(ds: Path) -> None:
    """stale 라벨 캐시가 pose mAP 를 0.945 -> 0.0157 로 뒤집은 이력이 있다."""
    for c in (ds / "labels").glob("*.cache"):
        c.unlink()
        print(f"   캐시 제거: {c.relative_to(ONE)}")


# --- 평가 -------------------------------------------------------------------
def load_eval_index():
    """held-out 프레임: 적격 keypoint(manual_click) 와 앙각 층."""
    part = json.loads((RESULTS / "GT_PARTITION.json").read_text(encoding="utf-8"))
    gt_root = REPO / "challenge/data/01_real/live_capture_gt"
    idx = {}
    for r in part:
        if r["split_role"] != "HELD_OUT":
            continue
        folder = f"{r['session']}_manual_gt"
        stem = f"{folder}__{r['frame']}"
        doc = json.loads((gt_root / folder / f"{r['frame']}.json").read_text(encoding="utf-8"))
        ann = doc["objects"][0]["keypoint_annotations"][:9]
        xy = np.full((9, 2), np.nan)
        elig = np.zeros(9, bool)
        for i, e in enumerate(ann):
            if e.get("xy") is not None:
                xy[i] = e["xy"]
                elig[i] = (e.get("source") == "manual_click")
        band = ("<8" if r["elev_bin"] in ("0-3", "3-8")
                else ("8-15" if r["elev_bin"] == "8-15" else ">=15"))
        idx[stem] = {"xy": xy, "eligible": elig, "band": band,
                     "session": r["session"], "frame": r["frame"]}
    return idx


def evaluate(weights: Path, idx, images: Path):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    stems = sorted(idx)
    paths = [str(images / f"{s}.png") for s in stems]
    rows = []
    for i in range(0, len(paths), 32):
        chunk, cs = paths[i:i + 32], stems[i:i + 32]
        for s, res in zip(cs, model.predict(chunk, imgsz=640, verbose=False, device=0)):
            g = idx[s]
            rec = {"stem": s, "session": g["session"], "band": g["band"],
                   "detected": False, "errs": []}
            if res.keypoints is not None and len(res.boxes):
                k = int(np.argmax(res.boxes.conf.cpu().numpy()))
                kp = res.keypoints.xy.cpu().numpy()[k] - PAD   # 원본 픽셀로
                d = np.linalg.norm(kp - g["xy"], axis=1)
                rec["detected"] = True
                rec["errs"] = [float(d[j]) for j in range(9)
                               if g["eligible"][j] and np.isfinite(d[j])]
            rows.append(rec)
    return rows


def per_frame_median(rows):
    return {r["stem"]: (float(np.median(r["errs"])) if r["errs"] else np.nan)
            for r in rows}


def session_cluster_ci(pairs, n_boot=4000, seed=0):
    """pairs: [(session, delta)] — 세션 단위 재표집 95% CI."""
    rng = np.random.default_rng(seed)
    bys = {}
    for s, d in pairs:
        bys.setdefault(s, []).append(d)
    sess = sorted(bys)
    if len(sess) < 3:
        return None
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(sess), len(sess), replace=True)
        vals = [v for j in pick for v in bys[sess[j]]]
        if vals:
            boots.append(float(np.median(vals)))
    b = np.array(boots)
    return {"median": float(np.median([d for _, d in pairs])),
            "lo": float(np.percentile(b, 2.5)),
            "hi": float(np.percentile(b, 97.5)),
            "n_sessions": len(sess), "n_frames": len(pairs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    assert sha256(BASE) == BASE_SHA, "base checkpoint SHA 불일치 — 중단"
    for name, ds in ARMS.items():
        rep = json.loads((ds / "_contract_build.json").read_text(encoding="utf-8"))
        assert not rep["train_heldout_frame_overlap"], f"{name}: split 누수"
    print("전제 확인 OK", flush=True)

    from ultralytics import YOLO
    runs = {}
    for arm, ds in ARMS.items():
        for seed in SEEDS:
            tag = f"{arm}_s{seed}"
            out = HERE / tag / "weights/best.pt"
            runs[tag] = out
            if out.is_file() and args.skip_train:
                print(f"   건너뜀 {tag}", flush=True)
                continue
            drop_label_cache(ds)
            print(f"\n=== STEP 1 {tag} ===", flush=True)
            YOLO(str(BASE)).train(data=str(ds / "data.yaml"), project=str(HERE),
                                  name=tag, seed=seed, **RECIPE)
            assert out.is_file(), f"{tag}: best.pt 가 없다"

    print("\n=== STEP 2 재채점 ===", flush=True)
    idx = load_eval_index()
    images = ARMS["contract"] / "images/val"
    per_arm = {}
    all_rows = {}
    for tag, w in [("R0", BASE)] + sorted(runs.items()):
        rows = evaluate(Path(w), idx, images)
        all_rows[tag] = rows
        per_arm[tag] = per_frame_median(rows)
        det = sum(r["detected"] for r in rows)
        for band in ("<8", "8-15"):
            e = [v for r in rows if r["band"] == band for v in r["errs"]]
            print(f"   {tag:<14}{band:<6} kp {len(e):>5}  med "
                  f"{np.median(e) if e else float('nan'):7.2f} px", flush=True)
        print(f"   {tag:<14}검출 {det}/{len(rows)}", flush=True)

    print("\n=== STEP 3 판정 ===", flush=True)
    bands = {r["stem"]: r["band"] for r in all_rows["R0"]}
    sess = {r["stem"]: r["session"] for r in all_rows["R0"]}
    verdict = {}
    for arm in ("contract", "legacy"):
        seeds_med = {}
        for band in ("<8", "8-15"):
            pairs = []
            for stem in per_arm["R0"]:
                if bands[stem] != band:
                    continue
                d = [per_arm[f"{arm}_s{s}"][stem] for s in SEEDS]
                d = [x for x in d if np.isfinite(x)]
                r0 = per_arm["R0"][stem]
                if not d or not np.isfinite(r0):
                    continue
                pairs.append((sess[stem], r0 - float(np.mean(d))))  # +면 FT 우세
            seeds_med[band] = session_cluster_ci(pairs)
        verdict[f"{arm}_minus_R0"] = seeds_med

    g = verdict["contract_minus_R0"][GATE_STRATUM]
    passed = bool(g and g["lo"] > 0)
    result = {
        "schema_version": "next_accuracy_v2_stage1_result_v1",
        "method_lock": "data/pallet/results/next_accuracy_v2/METHOD_LOCK.json",
        "gate_rule": GATE_RULE, "gate_stratum": GATE_STRATUM,
        "base_sha256": BASE_SHA, "seeds": SEEDS, "recipe": RECIPE,
        "deltas_px_positive_means_FT_better": verdict,
        "raw": {tag: [{k: v for k, v in r.items() if k != "errs"} |
                      {"n_eligible_kp": len(r["errs"]),
                       "median_err_px": (float(np.median(r["errs"])) if r["errs"] else None)}
                      for r in rows] for tag, rows in all_rows.items()},
        "verdict": ("REAL_SUPERVISION_LEVER_REPRODUCED" if passed
                    else "REAL_SUPERVISION_LEVER_NOT_REPRODUCED"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "FINAL": "STEP4_DONE",
    }
    dst = HERE / "RESULT.json"
    dst.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"[next_accuracy_v2 §11] {result['verdict']}",
             f"gate: {GATE_RULE}"]
    for arm in ("contract", "legacy"):
        for band in ("<8", "8-15"):
            c = verdict[f"{arm}_minus_R0"][band]
            if c:
                lines.append(f"  {arm:<9}{band:<6} Δ{c['median']:+.2f} px "
                             f"[{c['lo']:+.2f}, {c['hi']:+.2f}]  "
                             f"세션 {c['n_sessions']} 프레임 {c['n_frames']}")
    lines.append(f"  {result['elapsed_min']} 분")
    notify("\n".join(lines))
    print(f"\nwrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
