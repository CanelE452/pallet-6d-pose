"""SITE_A 전체 pool 의 pseudo-label 수급을 진단한다.  PREFLIGHT ONLY.

    python3 scripts/self_training_yolo/site_audit/preflight_site_pseudo_pool.py

출력  data/pallet/results/paper_selftrain_site_v1/preflight/
        SITE_A_PSEUDO_PREFLIGHT.json · .md · F0_NAIVE.csv · F1_CONF.csv · F4_PROPOSED.csv

질문은 quantity · diversity · exposure concentration 셋뿐이다.  purity 는 묻지
않는다 — 평가 GT 를 열지 않는다(`GT_USED_FOR_SELECTION = false`).

filter 는 새로 만들지 않고 `PSEUDOLABEL_FILTER_LOCK.json` 을 그대로 읽는다.
F4 Proposed = F0 후보 + box_conf >= TAU_BOX + s_remove <= tau_remove +
s_flip <= tau_flip.  reprojection 은 F4 조건이 아니다.

GO/STOP 을 자동 판정하지 않는다.  새 임계값을 만들지 않는다.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts/self_training_yolo"))

D = REPO_ROOT / "data/pallet/results/paper_selftrain_site_v1/preflight"
CACHE = D / "SITE_A_TEACHER_CACHE.json"
LOCK = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
A8_MANIFEST = (REPO_ROOT / "data/pallet/results/paper_selftrain_v1"
               / "pseudo_manifests/F4_DAY_ONLY.csv")
POOL_OBJECT_TYPE = "plastic_standard_110x130x11"
N_CORNERS = 8
PSEUDO_EXPOSURES_PER_EPOCH = 1440   # 기존 MAIN 계약값.  여기서 새로 정하지 않는다.


def registry_dimensions(object_type: str) -> dict:
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == object_type:
            return entry["physical_dimensions_m"]
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type}")


def describe(values, name):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()), "max": float(arr.max())}


def main() -> int:
    from pseudo_label_filters import geometry_scores

    lock = json.loads(LOCK.read_text())
    tau_box = float(lock["TAU_BOX"])
    kp_valid = float(lock["keypoint_validity"]["kp_conf_threshold"])
    min_corners = int(lock["keypoint_validity"]["min_valid_corners"])
    tau_remove = float(lock["geometry_thresholds"]["tau_remove"])
    tau_flip = float(lock["geometry_thresholds"]["tau_flip"])
    dimensions = registry_dimensions(POOL_OBJECT_TYPE)
    print(f"lock  TAU_BOX {tau_box}  kp_conf {kp_valid}  min_corners {min_corners}  "
          f"tau_remove {tau_remove}  tau_flip {tau_flip}")

    cache = json.loads(CACHE.read_text())
    records = []
    for entry in cache["entries"]:
        top = entry["top1"]
        record = {
            "image_path": entry["image_path"],
            "image_sha256": entry["image_sha256"],
            "capture_session": entry["capture_session"],
            "paper_condition": entry["paper_condition"],
            "detected": top is not None,
            "valid_corners": 0, "box_conf": None, "kp_conf_median8": None,
            "s_reproj": None, "s_remove": None, "s_flip": None,
            "bbox_area_frac": None, "center_x_frac": None, "center_y_frac": None,
            "bbox_aspect": None, "kp_spread_frac": None,
        }
        if top is not None:
            keypoints = np.asarray(top["keypoints_xy"], float)
            confidences = np.nan_to_num(np.asarray(top["keypoints_conf"], float), nan=0.0)
            valid = confidences >= kp_valid
            record["valid_corners"] = int(np.count_nonzero(valid[:N_CORNERS]))
            record["box_conf"] = float(top["box_conf"])
            record["kp_conf_median8"] = float(top["kp_conf_median8"])

            width, height = entry["image_width"], entry["image_height"]
            x1, y1, x2, y2 = top["box_xyxy"]
            box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
            record["bbox_area_frac"] = float(box_w * box_h / (width * height))
            record["center_x_frac"] = float((x1 + x2) / 2 / width)
            record["center_y_frac"] = float((y1 + y2) / 2 / height)
            record["bbox_aspect"] = float(box_w / box_h)
            usable = keypoints[valid[:len(keypoints)]]
            if len(usable) >= 2:
                span = usable.max(axis=0) - usable.min(axis=0)
                record["kp_spread_frac"] = float(np.hypot(*span) / np.hypot(width, height))

            camera = entry.get("camera_matrix")
            if camera is not None and record["valid_corners"] >= min_corners:
                flip = entry.get("flip_top1")
                flip_kp = flip_valid = None
                if flip is not None:
                    flip_kp = np.asarray(flip["keypoints_xy"], float)
                    flip_conf = np.nan_to_num(
                        np.asarray(flip["keypoints_conf"], float), nan=0.0)
                    flip_valid = flip_conf >= kp_valid
                scores = geometry_scores(keypoints, valid, np.asarray(camera, float),
                                         dimensions, flip_kp, flip_valid)
                record["s_reproj"] = scores["s_reproj"]
                record["s_remove"] = scores["s_remove"]
                record["s_flip"] = scores["s_flip"]
        records.append(record)

    def candidate(r):
        return r["detected"] and r["valid_corners"] >= min_corners

    def conf_ok(r):
        return candidate(r) and r["box_conf"] is not None and r["box_conf"] >= tau_box

    def proposed(r):
        return (conf_ok(r)
                and r["s_remove"] is not None and r["s_remove"] <= tau_remove
                and r["s_flip"] is not None and r["s_flip"] <= tau_flip)

    total = len(records)
    detected = [r for r in records if r["detected"]]
    cand = [r for r in records if candidate(r)]
    f1 = [r for r in records if conf_ok(r)]
    f4 = [r for r in records if proposed(r)]
    print(f"\nfunnel  total {total}  detected {len(detected)}  >=6corners {len(cand)}  "
          f"F1 {len(f1)}  F4 {len(f4)}")

    fields = ("image_path", "image_sha256", "paper_condition", "capture_session",
              "box_conf", "kp_conf_median8", "valid_corners",
              "s_reproj", "s_remove", "s_flip")
    for name, rows in (("F0_NAIVE", cand), ("F1_CONF", f1), ("F4_PROPOSED", f4)):
        with (D / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in rows)

    # ── recording 분해
    by_recording = {}
    for session in sorted({r["capture_session"] for r in records}):
        sub = lambda rows: [r for r in rows if r["capture_session"] == session]
        inp = len(sub(records))
        by_recording[session] = {
            "input": inp, "detected": len(sub(detected)), "candidate": len(sub(cand)),
            "confidence": len(sub(f1)), "proposed": len(sub(f4)),
            "retention_from_input": len(sub(f4)) / inp if inp else 0.0,
            "share_of_accepted": len(sub(f4)) / len(f4) if f4 else 0.0,
        }

    # ── A8 과의 membership 비교
    a8 = list(csv.DictReader(A8_MANIFEST.open()))
    a8_sha = {r["image_sha256"] for r in a8}
    full_sha = {r["image_sha256"] for r in f4}
    membership = {
        "a8_accepted": len(a8_sha),
        "full_accepted": len(full_sha),
        "a8_also_in_full": len(a8_sha & full_sha),
        "a8_lost_in_full": len(a8_sha - full_sha),
        "new_in_full": len(full_sha - a8_sha),
        "a8_by_recording": dict(sorted(Counter(r["capture_session"] for r in a8).items())),
    }

    # ── 노출 집중도 (학습을 뜻하지 않는다.  진단용)
    exposure = {
        "pseudo_exposures_per_epoch": PSEUDO_EXPOSURES_PER_EPOCH,
        "source": "existing MAIN contract; not chosen here",
        "a8_unique": len(a8_sha),
        "a8_repeat_per_epoch": PSEUDO_EXPOSURES_PER_EPOCH / len(a8_sha),
        "full_unique": len(full_sha),
        "full_repeat_per_epoch": (PSEUDO_EXPOSURES_PER_EPOCH / len(full_sha)
                                  if full_sha else None),
    }

    # ── coverage (예측만 사용, GT 미사용)
    a8_paths = {r["image_path"] for r in a8}
    a8_records = [r for r in records if r["image_path"] in a8_paths]
    coverage = {}
    for key in ("bbox_area_frac", "center_x_frac", "center_y_frac", "bbox_aspect",
                "kp_spread_frac", "box_conf", "s_remove", "s_flip", "valid_corners"):
        coverage[key] = {"a8": describe([r[key] for r in a8_records], key),
                         "full_f4": describe([r[key] for r in f4], key)}

    # ── temporal spacing (프레임 이름의 시퀀스 번호로만.  규칙을 새로 만들지 않는다)
    def sequence(path):
        stem = Path(path).stem
        return int(stem) if stem.isdigit() else None

    # 이 세션들의 파일명은 나노초 타임스탬프다.  간격을 프레임 번호로 다루면
    # 무의미한 숫자가 나오므로, pool 전체의 최빈 간격을 그 촬영의 프레임 주기로
    # 잡고 간격을 "몇 프레임 떨어졌나" 로 환산한다.
    temporal = {}
    for session in sorted({r["capture_session"] for r in f4}):
        pool_seq = sorted(s for s in (sequence(r["image_path"]) for r in records
                                      if r["capture_session"] == session)
                          if s is not None)
        seq = sorted(s for s in (sequence(r["image_path"]) for r in f4
                                 if r["capture_session"] == session) if s is not None)
        if len(seq) < 2 or len(pool_seq) < 2:
            temporal[session] = {"accepted": len(seq), "note": "too few to space"}
            continue
        period = float(np.median(np.diff(pool_seq)))          # 프레임 주기 (ns)
        gaps = np.diff(seq) / period                          # 프레임 단위 간격
        runs, run = [], 1
        for gap in gaps:
            if gap <= 1.5:      # 바로 다음 프레임이면 연속으로 본다
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)
        temporal[session] = {
            "accepted": len(seq),
            "pool_frames": len(pool_seq),
            "frame_period_ns": period,
            "median_gap_frames": float(np.median(gaps)),
            "p10_gap_frames": float(np.percentile(gaps, 10)),
            "longest_consecutive_run": int(max(runs)),
            "span_frames": float((seq[-1] - seq[0]) / period),
            "span_seconds": float((seq[-1] - seq[0]) / 1e9),
        }

    report = {
        "schema_version": "site_a_pseudo_preflight_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "PREFLIGHT_ONLY": True,
        "new_training": 0, "new_student": 0, "new_threshold": 0, "new_filter": 0,
        "new_teacher": 0,
        "GT_USED_FOR_SELECTION": False,
        "evaluation_gt_read": False,
        "existing_v1_artifact_modified": False,
        "teacher": {"checkpoint": cache["teacher_checkpoint"],
                    "sha256": cache["teacher_sha256"],
                    "recipe": cache["recipe"]},
        "filter_lock": {"path": str(LOCK.relative_to(REPO_ROOT)),
                        "TAU_BOX": tau_box, "kp_conf": kp_valid,
                        "min_valid_corners": min_corners,
                        "tau_remove": tau_remove, "tau_flip": tau_flip,
                        "reprojection_in_F4": False},
        "funnel": {"total": total, "detected": len(detected),
                   "candidate_ge6_corners": len(cand),
                   "confidence_F1": len(f1), "proposed_F4": len(f4)},
        "a8_reference": {"teacher_input": 500, "proposed_F4": len(a8_sha)},
        "by_recording": by_recording,
        "membership": membership,
        "exposure": exposure,
        "coverage": coverage,
        "temporal_spacing": temporal,
        "no_go_stop_decision_made_here": True,
    }
    (D / "SITE_A_PSEUDO_PREFLIGHT.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# SITE_A pseudo-label preflight", "",
             "Frozen R0 teacher, frozen filter lock, full SITE_A pool. No student was",
             "trained, no threshold was chosen here, and no evaluation ground truth was",
             "read — the question is quantity, diversity and exposure, not purity.", "",
             "```text", f"{'stage':26}{'A8 subset':>12}{'Full SITE_A':>14}",
             "─" * 52,
             f"{'Teacher input':26}{500:12d}{total:14d}",
             f"{'Detected':26}{'—':>12}{len(detected):14d}",
             f"{'>= 6 corners (F0)':26}{'—':>12}{len(cand):14d}",
             f"{'Confidence (F1)':26}{'—':>12}{len(f1):14d}",
             f"{'Proposed (F4)':26}{len(a8_sha):12d}{len(f4):14d}",
             "```", "", "## Per recording", "", "```text",
             f"{'Recording':20}{'Input':>8}{'Cand':>8}{'Conf':>8}{'Proposed':>10}"
             f"{'Retention':>11}{'Share':>8}", "─" * 73]
    for session, block in by_recording.items():
        lines.append(f"{session:20}{block['input']:8d}{block['candidate']:8d}"
                     f"{block['confidence']:8d}{block['proposed']:10d}"
                     f"{block['retention_from_input']:11.3f}"
                     f"{block['share_of_accepted']:8.3f}")
    lines += [f"{'ALL':20}{total:8d}{len(cand):8d}{len(f1):8d}{len(f4):10d}"
              f"{len(f4) / total:11.3f}{1.0:8.3f}", "```", "",
              "## Exposure concentration", "", "```text",
              f"pseudo exposures / epoch   {PSEUDO_EXPOSURES_PER_EPOCH}  "
              f"(existing contract, not chosen here)",
              f"A8   unique {exposure['a8_unique']:5d}   repeat "
              f"{exposure['a8_repeat_per_epoch']:6.2f} x / epoch",
              f"Full unique {exposure['full_unique']:5d}   repeat "
              f"{exposure['full_repeat_per_epoch']:6.2f} x / epoch",
              "```", "", "## Membership", "", "```text",
              f"A8 accepted                {membership['a8_accepted']}",
              f"Full accepted              {membership['full_accepted']}",
              f"A8 frames also in Full     {membership['a8_also_in_full']}",
              f"A8 frames lost in Full     {membership['a8_lost_in_full']}",
              f"new in Full only           {membership['new_in_full']}",
              "```", "", "## Coverage (predictions only, no GT)", "", "```text",
              f"{'quantity':20}{'A8 median':>12}{'A8 p10-p90':>22}"
              f"{'Full median':>13}{'Full p10-p90':>22}", "─" * 89]
    for key, block in coverage.items():
        a, f = block["a8"], block["full_f4"]
        if not a.get("n") or not f.get("n"):
            continue
        a_range = "[{:.4f}, {:.4f}]".format(a["p10"], a["p90"])
        f_range = "[{:.4f}, {:.4f}]".format(f["p10"], f["p90"])
        lines.append(f"{key:20}{a['median']:12.4f}{a_range:>22}"
                     f"{f['median']:13.4f}{f_range:>22}")
    lines += ["```", "", "## Temporal spacing of accepted frames", "", "```text",
              f"{'Recording':20}{'accepted':>10}{'pool':>8}{'med gap':>10}"
              f"{'p10 gap':>10}{'run':>7}{'span s':>10}", "─" * 75,
              "gaps are in frames, using each recording's own median frame period"]
    for session, block in temporal.items():
        if "median_gap_frames" not in block:
            lines.append(f"{session:20}{block['accepted']:10d}   {block['note']}")
            continue
        lines.append(f"{session:20}{block['accepted']:10d}{block['pool_frames']:8d}"
                     f"{block['median_gap_frames']:10.1f}{block['p10_gap_frames']:10.1f}"
                     f"{block['longest_consecutive_run']:7d}{block['span_seconds']:10.1f}")
    lines += ["```", "",
              "No GO/STOP decision is made in this document and no threshold was invented.",
              "It reports what the pool contains so a person can decide."]
    (D / "SITE_A_PSEUDO_PREFLIGHT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {(D / 'SITE_A_PSEUDO_PREFLIGHT.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
