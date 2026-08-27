"""PHASE 3 — DEV 161장 실패를 속성으로 귀속.

★ 인과를 분리했다고 주장하지 않는다.  target 세션은 **같은 물체**를 쓰고 night
세션도 그 물체다.  그래서 geometry 와 appearance 가 세션 수준에서 얽혀 있다.
교차표는 그 얽힘을 드러내려고 만드는 것이지 풀려고 만드는 게 아니다.
"""
from __future__ import annotations

import collections, csv, json, os, pathlib, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/model_compare", "scripts/stage0/real_eval",
            "scripts/stage0/stage_screens", "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import cv2                    # noqa: E402
import mc_geom as MG          # noqa: E402
import re_metrics as RM       # noqa: E402
from stage25_paperbase_eval import elev_from_pose  # noqa: E402

OUT = pathlib.Path(ROOT)/"challenge/yolo_pose_one_model/broad_family_v2"
PIPE = pathlib.Path(ROOT)/"challenge/yolo_pose_one_model/paper_generic_pipeline"
DUMP = pathlib.Path(ROOT)/"data/pallet/results/model_compare"
MODEL = "yolo26n_paper_generic_v1"
GROSS = 30.0


def proxies(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    b, g, r = cv2.split(image.astype(np.float64))
    return {"luma": float(np.median(grey)),
            "blur_proxy": float(lap.var()),
            "noise_proxy": float(np.median(np.abs(
                grey.astype(np.float64) - cv2.medianBlur(grey, 3)))),
            "wb_proxy": float(np.median(r)/max(np.median(b), 1e-6)),
            "contrast": float(np.percentile(grey, 95) - np.percentile(grey, 5))}


def main():
    man = {i["frame_id"]: i for i in
           json.load(open(PIPE/"eval_manifest.json"))["items"]}
    dump = json.load(open(DUMP/f"kps_{MODEL}.json"))
    rows = []
    for e in dump["frames"]:
        m = man[e["fid"]]
        image = cv2.imread(os.path.join(ROOT, m["image"]))
        h, w = image.shape[:2]
        gt8 = np.asarray(m["gt_corners_2d"], float)
        truth = {"R": np.asarray(m["R_gt"], float),
                 "t": np.asarray(m["t_gt"], float),
                 "K": np.asarray(m["K"], float),
                 "model": np.asarray(m["object_points"], float),
                 "gt8": gt8,
                 "extents": (m["dimensions_m"]["width"], m["dimensions_m"]["height"],
                             m["dimensions_m"]["depth"])}
        px = MG.points_of(e, MODEL)
        ok = np.isfinite(px).all(1)
        corner = float(np.median(np.linalg.norm(px[ok]-gt8[ok], axis=1))) \
            if ok.any() else np.nan
        pose = MG.solve(px, truth) if ok.sum() >= 4 else None
        R = t = np.nan; s5 = 0
        if pose is not None:
            R, t = RM.pose_error(pose[0], pose[1], truth["R"], truth["t"])
            s5 = int(RM.success_5cm5deg(pose[0], pose[1], truth["R"], truth["t"]))
        det = e["kps"] is not None
        cls = ("NO_BOX" if not det else
               "BOX_OK_KP_BAD" if (not np.isfinite(corner) or corner > GROSS) else
               "POSE_BAD_DESPITE_KP" if (pose is None or R > 10) else
               "BOX_OK_KP_GOOD")
        inside = ((gt8[:, 0] >= 0) & (gt8[:, 0] < w)
                  & (gt8[:, 1] >= 0) & (gt8[:, 1] < h))
        span = gt8.max(0) - gt8.min(0)
        pose4 = np.vstack([np.hstack([truth["R"], truth["t"][:, None]]),
                           [0, 0, 0, 1]]).tolist()
        row = {"fid": e["fid"], "set": e["set"], "population": m["population"],
               "failure_type": cls, "detected": int(det),
               "box_conf": e.get("box_conf") or np.nan,
               "corner_med": corner, "R": R, "t": t, "success_5cm5": s5,
               "elev": elev_from_pose(pose4),
               "n_inside_gt": int(inside.sum()),
               "truncated": int((~inside).any()),
               "bbox_area_frac": float(span[0]*span[1]/(w*h)),
               "obj_diag_frac": float(np.hypot(*span)/np.hypot(w, h)),
               "distance_m": float(np.linalg.norm(truth["t"])),
               **proxies(image)}
        for i in range(8):
            row[f"kp{i}_err"] = (float(np.linalg.norm(px[i]-gt8[i]))
                                 if np.isfinite(px[i]).all() else np.nan)
            kc = e.get("kp_conf")
            row[f"kp{i}_conf"] = (float(kc[i]) if kc and kc[i] is not None
                                  else np.nan)
        rows.append(row)

    with open(OUT/"REAL_DEV_FAILURE_ATTRIBUTE.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        wtr.writeheader(); wtr.writerows(rows)

    def cross(keyfn, order, label):
        tab = collections.defaultdict(collections.Counter)
        for r in rows:
            tab[keyfn(r)][r["failure_type"]] += 1
        lines = [f"### {label}", "", "```",
                 f"{'bucket':16}{'n':>5}{'NO_BOX':>9}{'KP_BAD':>9}"
                 f"{'POSE_BAD':>10}{'GOOD':>8}"]
        for b in order:
            c = tab.get(b)
            if not c:
                continue
            n = sum(c.values())
            lines.append(f"{b:16}{n:>5}"
                         f"{c['NO_BOX']/n:>9.2f}{c['BOX_OK_KP_BAD']/n:>9.2f}"
                         f"{c['POSE_BAD_DESPITE_KP']/n:>10.2f}"
                         f"{c['BOX_OK_KP_GOOD']/n:>8.2f}")
        lines += ["```", ""]
        return lines

    lb = lambda r: ("dark<60" if r["luma"] < 60 else
                    "dim60-100" if r["luma"] < 100 else
                    "mid100-140" if r["luma"] < 140 else "bright>=140")
    sb = lambda r: ("small<0.20" if r["obj_diag_frac"] < 0.20 else
                    "mid0.20-0.40" if r["obj_diag_frac"] < 0.40 else ">=0.40")
    eb = lambda r: ("<3" if r["elev"] is None or r["elev"] < 3 else
                    "3-8" if r["elev"] < 8 else
                    "8-15" if r["elev"] < 15 else "15+")
    tb = lambda r: "truncated" if r["truncated"] else "full"

    md = ["# REAL_DEV FAILURE ATTRIBUTE", "",
          f"model: {MODEL} (60 epoch, target-free BROAD 40K)", "",
          "> ★ **인과 분리 주장 금지.** target 세션과 night 세션은 **같은 물체**를",
          "> 쓴다. geometry 와 appearance 가 세션 수준에서 얽혀 있어, 아래 교차표는",
          "> 얽힘을 드러내는 것이지 푸는 것이 아니다.", ""]
    md += cross(lambda r: r["set"].replace("eval_", ""),
                ["outside", "noapril", "cad", "pallet07", "pallet09",
                 "night08", "night09"], "domain x failure_type")
    md += cross(lb, ["dark<60", "dim60-100", "mid100-140", "bright>=140"],
                "luma x failure_type")
    md += cross(sb, ["small<0.20", "mid0.20-0.40", ">=0.40"],
                "object size x failure_type")
    md += cross(eb, ["<3", "3-8", "8-15", "15+"], "elevation x failure_type")
    md += cross(tb, ["full", "truncated"], "truncation x failure_type")

    # per-keypoint
    md += ["### per-keypoint (CHALLENGE_105)", "", "```",
           f"{'kp':4}{'group':16}{'err med':>10}{'err p90':>10}"
           f"{'conf med':>10}{'missing':>9}"]
    ch = [r for r in rows if r["population"] == "REAL_CHALLENGE_DEV_105"]
    for i in range(8):
        e = np.array([r[f"kp{i}_err"] for r in ch], float)
        c = np.array([r[f"kp{i}_conf"] for r in ch], float)
        g = np.isfinite(e)
        grp = ("near " if i < 4 else "far ") + ("top" if i in (0, 1, 4, 5) else "bottom")
        md.append(f"{i:<4}{grp:16}"
                  f"{(np.median(e[g]) if g.any() else float('nan')):>10.1f}"
                  f"{(np.percentile(e[g], 90) if g.any() else float('nan')):>10.1f}"
                  f"{(np.nanmedian(c) if np.isfinite(c).any() else float('nan')):>10.3f}"
                  f"{1-g.mean():>9.2f}")
    md += ["```", ""]
    (OUT/"REAL_DEV_FAILURE_REPORT.md").write_text("\n".join(md)+"\n")
    print("\n".join(md[6:]))


if __name__ == "__main__":
    main()
