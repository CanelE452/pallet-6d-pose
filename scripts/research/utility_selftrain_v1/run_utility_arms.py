"""utility_selftrain_v1 section 7 — equal-N micro fine-tuning utility arms.

One driver: build manifests -> build datasets -> train -> evaluate on POLICY_VAL ->
apply the pre-registered gate -> notify with the verdict.  It does not stop at training.

Matching semantics are UNIQUE-QUANTITY-MATCHED (the exposure lock's appendix A2 sense):
every arm gets the same number of unique pseudo-labels, and the exposure slots stay at
1440 per epoch for all arms, so optimizer updates are identical by construction.

Arm definitions and the utility gate come from METHOD_LOCK_UTILITY_ST.json and were
frozen before any arm was trained.  STUDENT_EVAL is never read here.
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/self_training_yolo"))
from build_pseudo_datasets import pseudo_label_line, sha256_file  # noqa: E402

RESULTS = ROOT / "data/pallet/results/utility_selftrain_v1"
DOCS = ROOT / "_docs/experiments/utility_selftrain_v1"
MANIFESTS = RESULTS / "pseudo_manifests"
DATASETS = ROOT / "challenge/yolo_pose_one_model/datasets/utility_selftrain_v1"
RUNS = ROOT / "challenge/yolo_pose_one_model/utility_selftrain_v1"
STATE = RESULTS / "DRIVER_STATE.json"
NOTIFY = Path.home() / ".claude/hooks/discord-notify.sh"

POOL_FEATURES = RESULTS / "FEATURES_POOL.csv"
CACHE = ROOT / "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json"
EXPOSURE = ROOT / "data/pallet/results/paper_selftrain_v1/SELFTRAIN_EXPOSURE_LOCK.json"
FILTER_LOCK = ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
REPLAY_LIST = ROOT / "data/pallet/results/paper_selftrain_v1/SYNTHETIC_REPLAY_SUBSET.txt"
R0_DATASET = ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
WS = ROOT / "data/evaluation/pallet_eval_v1"

TRAIN_ARGS = dict(
    imgsz=640, batch=32, optimizer="SGD", lr0=0.002, lrf=0.01, cos_lr=True,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=1.0, patience=0,
    box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=0.0, translate=0.1, scale=0.25,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
    mosaic=0.15, close_mosaic=3, mixup=0.0, copy_paste=0.0, erasing=0.4,
    seed=42, deterministic=True, val=False, plots=False, save_period=-1,
    workers=2, cache=False,
)
PAD, IMGSZ, CONF_FLOOR = 100, 640, 0.001
GROSS_PX = 20.0
SAMPLING_SEEDS = (42, 43, 44)


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ── arm membership (METHOD_LOCK binning) ────────────────────────────────────

def arm_members(rows, tau_box, tau_geom):
    def q(key, p):
        v = sorted(x for x in (num(r[key]) for r in rows) if x is not None)
        return v[int(p * (len(v) - 1))]

    g_q1, g_q3 = q("max_geom_f4", 0.25), q("max_geom_f4", 0.75)
    s_q1, s_q3 = q("bbox_diag_frac", 0.25), q("bbox_diag_frac", 0.75)
    e_q1, e_q3 = q("pred_elev_deg", 0.25), q("pred_elev_deg", 0.75)

    def f4(r):
        c, m, f = num(r["box_conf"]), num(r["s_remove"]), num(r["s_flip"])
        return (c is not None and c >= tau_box and m is not None and m <= tau_geom
                and f is not None and f <= tau_geom)

    def geom_min(r):
        g = num(r["max_geom_f4"])
        return g is not None and g <= tau_geom

    defs = {
        "U0_RANDOM_MATCHED": lambda r: True,
        "U1_CURRENT_F4": f4,
        "U2_SMALL_SCALE": lambda r: geom_min(r) and num(r["bbox_diag_frac"]) < s_q1,
        "U3_LARGE_SCALE": lambda r: geom_min(r) and num(r["bbox_diag_frac"]) >= s_q3,
        "U4_LOW_ELEV": lambda r: geom_min(r) and num(r["pred_elev_deg"]) < e_q1,
        "U5_HIGH_ELEV": lambda r: geom_min(r) and num(r["pred_elev_deg"]) >= e_q3,
        "U6_LOWCONF_STRONGGEOM": lambda r: (num(r["box_conf"]) < tau_box
                                            and num(r["max_geom_f4"]) is not None
                                            and num(r["max_geom_f4"]) <= g_q1),
        "U7_HIGHCONF_BORDERLINEGEOM": lambda r: (num(r["box_conf"]) >= tau_box
                                                 and num(r["max_geom_f4"]) is not None
                                                 and num(r["max_geom_f4"]) >= g_q3),
    }
    members = {k: [r for r in rows if v(r)] for k, v in defs.items()}
    edges = {"max_geom_f4_q1": g_q1, "max_geom_f4_q3": g_q3,
             "bbox_diag_frac_q1": s_q1, "bbox_diag_frac_q3": s_q3,
             "pred_elev_deg_q1": e_q1, "pred_elev_deg_q3": e_q3}
    return members, edges


# ── step 1: manifests ───────────────────────────────────────────────────────

def build_manifests(min_arm_size):
    rows = [r for r in csv.DictReader(POOL_FEATURES.open()) if r["candidate"] == "True"]
    lock = json.loads(FILTER_LOCK.read_text())
    members, edges = arm_members(rows, lock["TAU_BOX"],
                                 lock["geometry_thresholds"]["tau_remove"])

    viable = {k: v for k, v in members.items() if len(v) >= min_arm_size}
    dropped = {k: len(v) for k, v in members.items() if len(v) < min_arm_size}
    matched_n = min(len(v) for k, v in viable.items() if k != "U0_RANDOM_MATCHED")

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    cols = ["image_path", "image_sha256", "paper_condition", "capture_session",
            "box_conf", "kp_conf_median8", "valid_corners", "s_reproj", "s_remove", "s_flip"]
    cache = {e["image_sha256"]: e for e in json.loads(CACHE.read_text())["entries"]}

    census = {}
    for arm, rs in viable.items():
        census[arm] = {
            "pool_size": len(rs), "matched_n": matched_n,
            "daytime": sum(r["condition"] == "daytime" for r in rs),
            "nighttime": sum(r["condition"] == "nighttime" for r in rs),
            "sessions": sorted({r["session"] for r in rs}),
            "manifests": {},
        }
        for seed in SAMPLING_SEEDS:
            rng = random.Random(f"{seed}:{arm}:membership")
            picked = sorted(rs, key=lambda r: r["frame_id"])
            rng.shuffle(picked)
            picked = picked[:matched_n]
            path = MANIFESTS / f"{arm}_S{seed}.csv"
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in picked:
                    e = cache[r["frame_id"]]
                    w.writerow({
                        "image_path": e["image_path"], "image_sha256": e["image_sha256"],
                        "paper_condition": e["paper_condition"],
                        "capture_session": e["capture_session"],
                        "box_conf": e["top1"]["box_conf"],
                        "kp_conf_median8": e["top1"]["kp_conf_median8"],
                        "valid_corners": r["valid_corner_count"],
                        "s_reproj": r["s_reproj"], "s_remove": r["s_remove"],
                        "s_flip": r["s_flip"]})
            census[arm]["manifests"][str(seed)] = {
                "path": str(path.relative_to(ROOT)), "n": len(picked),
                "sha256": sha256_file(path)}
    return census, dropped, edges, matched_n, {k: len(v) for k, v in members.items()}


# ── step 2: datasets (exposure contract copied from build_pseudo_datasets) ──

def build_dataset(arm, seed, matched_n):
    exposure = json.loads(EXPOSURE.read_text())
    kp_conf = float(json.loads(FILTER_LOCK.read_text())["keypoint_validity"]["kp_conf_threshold"])
    pseudo_slots = int(exposure["pseudo_exposures_per_epoch"])
    synthetic_slots = int(exposure["synthetic_exposures_per_epoch"])
    cache = {e["image_sha256"]: e for e in json.loads(CACHE.read_text())["entries"]}
    replay = [n for n in REPLAY_LIST.read_text().split("\n") if n]

    name = f"{arm}_S{seed}"
    dataset = DATASETS / name
    images, labels = dataset / "images/train", dataset / "labels/train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    for n in replay:
        link = images / f"replay__{n}"
        if not link.exists():
            link.symlink_to(R0_DATASET / "images/train" / n)
        tgt = (labels / f"replay__{n}").with_suffix(".txt")
        if not tgt.exists():
            tgt.write_text((R0_DATASET / "labels/train" / n).with_suffix(".txt").read_text())

    rrng = random.Random(f"{seed}:{arm}:replay")
    pool = [f"replay__{n}" for n in replay]
    filled = []
    while len(filled) < synthetic_slots:
        rrng.shuffle(pool)
        filled += pool[: synthetic_slots - len(filled)]
    entries = list(filled)

    accepted = []
    for row in csv.DictReader((MANIFESTS / f"{name}.csv").open()):
        line = pseudo_label_line(cache[row["image_sha256"]], kp_conf)
        if line is None:
            continue
        fn = f"pl__{row['capture_session']}__{Path(row['image_path']).name}"
        link = images / fn
        if not link.exists():
            link.symlink_to(ROOT / row["image_path"])
        (labels / fn).with_suffix(".txt").write_text(line + "\n")
        accepted.append(fn)
    if not accepted:
        raise SystemExit(f"NO_PSEUDO_LABELS_FOR_ARM: {name}")

    prng = random.Random(f"{seed}:{arm}")
    order, pfilled = list(accepted), []
    while len(pfilled) < pseudo_slots:
        prng.shuffle(order)
        pfilled += order[: pseudo_slots - len(pfilled)]
    entries += pfilled

    random.Random(f"{seed}:{arm}:order").shuffle(entries)
    # Ultralytics does not invalidate its label cache when labels are rewritten in
    # place, and a stale cache has silently zeroed pose mAP in this repo before.
    for stale in (labels.with_suffix(".cache"), labels.parent / "train.cache"):
        if stale.exists():
            stale.unlink()
    listing = dataset / "train.txt"
    listing.write_text("\n".join(str(images / e) for e in entries) + "\n")
    (dataset / "data.yaml").write_text(
        f"path: {dataset}\ntrain: train.txt\nval: train.txt\nnc: 1\n"
        f"kpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
        f"names:\n  0: pallet\n")
    total = int(exposure["pseudo_exposures_per_epoch"]) + int(exposure["synthetic_exposures_per_epoch"])
    if len(entries) != total:
        raise SystemExit(f"EXPOSURE_MISMATCH: {name} {len(entries)} != {total}")
    return {"dataset": str(dataset.relative_to(ROOT)), "unique_pseudo": len(accepted),
            "entries_per_epoch": len(entries),
            "pseudo_exposures_per_unique": round(pseudo_slots / len(accepted), 2),
            "train_list_sha256": sha256_file(listing)}


# ── step 3: train ───────────────────────────────────────────────────────────

def train(arm, seed, epochs):
    from ultralytics import YOLO
    name = f"{arm}_S{seed}"
    run = RUNS / name
    results_csv = run / "results.csv"
    if results_csv.exists() and sum(1 for _ in results_csv.open()) - 1 >= epochs:
        return {"status": "CACHED", "run": str(run.relative_to(ROOT))}
    exposure = json.loads(EXPOSURE.read_text())
    init = ROOT / exposure["initialisation"]["checkpoint"]
    got = sha256_file(init)
    if got != exposure["initialisation"]["sha256"]:
        raise SystemExit(f"INIT_SHA_MISMATCH: {got}")
    t0 = time.time()
    YOLO(str(init)).train(data=str(DATASETS / name / "data.yaml"), epochs=epochs,
                          project=str(RUNS), name=name, exist_ok=True, **TRAIN_ARGS)
    rows = sum(1 for _ in results_csv.open()) - 1
    if rows < epochs:
        raise SystemExit(f"TRAIN_INCOMPLETE: {name} {rows}/{epochs}")
    return {"status": "OK", "run": str(run.relative_to(ROOT)),
            "elapsed_s": round(time.time() - t0, 1),
            "weights_sha256": sha256_file(run / "weights/last.pt")[:16]}


# ── step 4: evaluate on POLICY_VAL ──────────────────────────────────────────

def policy_val_frames():
    sys.path.insert(0, str(ROOT / "scripts/evaluation"))
    from eval_workspace import load_frames, evaluation_population_views
    split = json.loads((DOCS / "SPLIT_CONTRACT.json").read_text())
    want = set(split["roles"]["POLICY_VAL"]["sessions"])
    pos = evaluation_population_views(load_frames(WS))["PAPER_EVAL_POSITIVE"]
    return [f for f in pos if f["session_id"] in want]


def evaluate(weights, frames):
    """Pooled supervised keypoint error, the same statistic evaluate_arms.py reports."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(weights))
    errors, per_session, detected, n = [], {}, 0, 0
    for f in frames:
        img = cv2.imread(str(WS / f["image_path"]))
        if img is None:
            continue
        H, W = img.shape[:2]
        ann = json.loads((WS / f["annotation_path"]).read_text())
        gt = ann["objects"][0].get("keypoint_annotations")
        if not gt:
            continue
        padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        res = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR, verbose=False)[0]
        n += 1
        if res.boxes is None or len(res.boxes) == 0:
            continue
        i = int(np.argmax(res.boxes.conf.cpu().numpy()))
        kp = res.keypoints.xy.cpu().numpy()[i] - PAD
        box = res.boxes.xyxy.cpu().numpy()[i] - PAD

        gxy, sup = [], []
        for a in gt:
            xy = a.get("xy")
            gxy.append(xy if xy else [float("nan")] * 2)
            sup.append(bool(a.get("visibility", 0)) and a.get("in_frame", True) and xy is not None)
        g, m = np.asarray(gxy, float), np.asarray(sup, bool)
        if not m.any():
            continue
        e = np.linalg.norm(kp[: len(g)][m] - g[m], axis=1)
        errors += e.tolist()
        per_session.setdefault(f["session_id"], []).extend(e.tolist())

        gb = [np.nanmin(g[m, 0]), np.nanmin(g[m, 1]), np.nanmax(g[m, 0]), np.nanmax(g[m, 1])]
        ix0, iy0 = max(box[0], gb[0]), max(box[1], gb[1])
        ix1, iy1 = min(box[2], gb[2]), min(box[3], gb[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        ua = (box[2] - box[0]) * (box[3] - box[1])
        ub = (gb[2] - gb[0]) * (gb[3] - gb[1])
        if ua + ub - inter > 1e-9 and inter / (ua + ub - inter) >= 0.5:
            detected += 1

    if not errors:
        return {"n_frames": n, "n_keypoints": 0}
    a = np.asarray(errors)
    return {
        "n_frames": n, "n_keypoints": len(a),
        "detection_rate_iou50": detected / n if n else None,
        "corner_median_px": float(np.median(a)),
        "corner_p90_px": float(np.percentile(a, 90)),
        "gross_rate": float(np.mean(a > GROSS_PX)),
        "by_session": {s: {"n_keypoints": len(v),
                           "corner_median_px": float(np.median(v)),
                           "corner_p90_px": float(np.percentile(v, 90))}
                       for s, v in per_session.items()},
    }


# ── step 5: verdict ─────────────────────────────────────────────────────────

def verdict(results, gate):
    need = gate["required_improvement_vs_each_baseline"]["corner_median_px_pct"]
    allow = gate["max_allowed_regression"]["corner_p90_px_pct"]

    def mean_of(arm, key):
        v = [results[arm][str(s)]["eval"].get(key) for s in SAMPLING_SEEDS
             if str(s) in results[arm] and results[arm][str(s)].get("eval")]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    out = {}
    for arm in results:
        row = {"corner_median_px_mean": mean_of(arm, "corner_median_px"),
               "corner_p90_px_mean": mean_of(arm, "corner_p90_px"),
               "detection_rate_mean": mean_of(arm, "detection_rate_iou50")}
        out[arm] = row
    for arm, row in out.items():
        if arm in ("U0_RANDOM_MATCHED", "U1_CURRENT_F4"):
            row["verdict"] = "BASELINE"
            continue
        passes, detail = [], {}
        for base in ("U0_RANDOM_MATCHED", "U1_CURRENT_F4"):
            b = out[base]["corner_median_px_mean"]
            a = row["corner_median_px_mean"]
            if b is None or a is None:
                detail[base] = None
                passes.append(False)
                continue
            imp = (b - a) / b * 100.0
            detail[base] = round(imp, 2)
            passes.append(imp >= need)
        bp = out["U0_RANDOM_MATCHED"]["corner_p90_px_mean"]
        reg = ((row["corner_p90_px_mean"] - bp) / bp * 100.0) if bp else None
        row["corner_median_improvement_pct"] = detail
        row["corner_p90_regression_pct_vs_U0"] = round(reg, 2) if reg is not None else None
        row["verdict"] = ("UTILITY_POSITIVE" if all(passes)
                          and (reg is None or reg <= allow) else "NOT_UTILITY_POSITIVE")
    return out


def notify(text):
    if NOTIFY.exists():
        subprocess.run([str(NOTIFY), text], check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--min-arm-size", type=int, default=40)
    ap.add_argument("--stage", choices=["all", "build", "train", "eval"], default="all")
    args = ap.parse_args()

    exposure = json.loads(EXPOSURE.read_text())
    epochs = args.epochs or int(exposure["epochs"])
    lock = json.loads((DOCS / "METHOD_LOCK_UTILITY_ST.json").read_text())

    print("STEP 1  manifests", flush=True)
    census, dropped, edges, matched_n, all_sizes = build_manifests(args.min_arm_size)
    print(f"  matched N = {matched_n}   viable arms = {len(census)}")
    for a, n in sorted(all_sizes.items(), key=lambda kv: -kv[1]):
        mark = "  DROPPED (too small)" if a in dropped else ""
        print(f"    {a:30s} pool={n:4d}{mark}")

    state = {"schema_version": "utility_selftrain_v1_micro_ft",
             "RESULT_ROLE": lock["RESULT_ROLE"],
             "matching_semantics": "UNIQUE-QUANTITY-MATCHED (exposure lock appendix A2 sense)",
             "matched_n": matched_n, "min_arm_size": args.min_arm_size,
             "arm_pool_sizes": all_sizes, "dropped_arms": dropped,
             "bin_edges": edges, "census": census, "epochs": epochs,
             "sampling_seeds": list(SAMPLING_SEEDS), "arms": {}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    if args.stage == "build":
        return 0

    print("\nSTEP 2-3  datasets + training", flush=True)
    frames = policy_val_frames()
    print(f"  POLICY_VAL frames = {len(frames)}")
    total = len(census) * len(SAMPLING_SEEDS)
    done = 0
    for arm in census:
        state["arms"].setdefault(arm, {})
        for seed in SAMPLING_SEEDS:
            done += 1
            tag = f"{arm}_S{seed}"
            ds = build_dataset(arm, seed, matched_n)
            tr = train(arm, seed, epochs)
            print(f"  [{done}/{total}] {tag:34s} uniquePL={ds['unique_pseudo']:3d} "
                  f"rep={ds['pseudo_exposures_per_unique']:6.1f}  {tr['status']}", flush=True)
            ev = evaluate(RUNS / tag / "weights/last.pt", frames)
            show = lambda k, f="{:.3f}": ("--" if ev.get(k) is None else f.format(ev[k]))
            print(f"        corner med={show('corner_median_px')}px "
                  f"p90={show('corner_p90_px', '{:.2f}')}px "
                  f"det={show('detection_rate_iou50')}", flush=True)
            state["arms"][arm][str(seed)] = {"dataset": ds, "train": tr, "eval": ev}
            STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False))

    print("\nSTEP 4-5  verdict", flush=True)
    table = verdict(state["arms"], lock["utility_gate"])
    state["verdict_table"] = table
    state["gate"] = lock["utility_gate"]
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    (DOCS / "UTILITY_MICRO_FT.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    print(f"\n{'arm':30s}{'cornerMed':>11s}{'p90':>9s}{'det':>7s}"
          f"{'dU0%':>8s}{'dF4%':>8s}  verdict")
    for arm, row in table.items():
        d = row.get("corner_median_improvement_pct") or {}
        fmt = lambda v, f="{:.3f}": "--" if v is None else f.format(v)
        print(f"{arm:30s}{fmt(row['corner_median_px_mean']):>11s}"
              f"{fmt(row['corner_p90_px_mean'],'{:.2f}'):>9s}"
              f"{fmt(row['detection_rate_mean']):>7s}"
              f"{fmt(d.get('U0_RANDOM_MATCHED'),'{:+.1f}'):>8s}"
              f"{fmt(d.get('U1_CURRENT_F4'),'{:+.1f}'):>8s}  {row['verdict']}")

    positive = [a for a, r in table.items() if r["verdict"] == "UTILITY_POSITIVE"]
    final = "UTILITY_POSITIVE_SUBSET_FOUND" if positive else "NO_USEFUL_PSEUDOLABEL_REGIME_FOUND"
    state["stage_c_verdict"] = final
    state["utility_positive_arms"] = positive
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    (DOCS / "UTILITY_MICRO_FT.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
    print(f"\nSTAGE C VERDICT = {final}   positive arms = {positive or 'none'}")
    notify(f"utility_selftrain_v1 STAGE C: {final}. matched N={matched_n}, "
           f"arms={len(census)}x3 draws. positive={positive or 'none'}. "
           f"RESULT_ROLE={lock['RESULT_ROLE']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
