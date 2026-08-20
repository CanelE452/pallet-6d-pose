"""FINAL40K 의 corner gross-error 꼬리는 어디서 오는가.

실험 B 가 뒤집어 놓은 것: FINAL 의 corner **median** 은 9.30px 로 같은 데이터로
학습한 YOLO(15.43px)보다 오히려 좋다.  그런데 pose 성공률은 낮다 (0.250 vs 0.429).
p90 이 114.75px 이니 **꼬리가 pose 를 죽인다.**  격자 양자화는 median 을 설명하지
꼬리를 설명하지 못한다.  그래서 꼬리의 정체를 본다.

비교 대상은 `y_BROAD40K` 다 — **같은 데이터로 학습한** 모델이라, 여기서 갈리는
것은 데이터가 아니라 표현/아키텍처 쪽이다.

축은 전부 GT 와 이미지에서 직접 계산한다 (라벨에 없는 값을 지어내지 않는다):
  elevation      pose_transform  (정본 elev_from_pose)
  truncation     GT 8 코너 중 화면 밖 개수
  screen size    GT bbox 대각 / 이미지 대각
  luma           이미지 그레이 중앙값
단변량으로 보면 서로 얽혀 오귀인하므로 교차표도 같이 낸다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/stage_screens",
            "scripts/stage0/real_eval", "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                                  # noqa: E402
import mc_frames as MF                      # noqa: E402
import mc_geom as MG                        # noqa: E402
from stage25_paperbase_eval import elev_from_pose  # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
MODELS = ["FINAL40K_seed1", "yolo26n_broad40k_5ep"]
GROSS_PX = 30.0          # 이 위를 꼬리로 본다. median(9.3)의 3배 남짓


def frame_axes(label, image):
    obj = label["objects"][0]
    gt8 = np.asarray(obj["projected_cuboid"], float)[:8]
    h, w = image.shape[:2]
    inside = ((gt8[:, 0] >= 0) & (gt8[:, 0] < w)
              & (gt8[:, 1] >= 0) & (gt8[:, 1] < h))
    span = gt8.max(0) - gt8.min(0)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {"elev": elev_from_pose(obj["pose_transform"]),
            "n_outside": int((~inside).sum()),
            "truncated": bool((~inside).any()),
            "size_ratio": float(np.hypot(*span) / np.hypot(w, h)),
            "luma": float(np.median(grey))}


def main():
    axes, truths = {}, {}
    for key, sealed, jp, ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        image = cv2.imread(ip)
        axes[fid] = {"set": key, "sealed": sealed, **frame_axes(label, image)}
        truths[fid] = MG.gt_of(label)

    per_model = {}
    for name in MODELS:
        dump = json.load(open(os.path.join(OUT, f"kps_{name}.json")))
        rows = {}
        for e in dump["frames"]:
            px = MG.points_of(e, name)
            ok = np.isfinite(px).all(1)
            t = truths[e["fid"]]
            if not ok.any():
                rows[e["fid"]] = np.nan
                continue
            rows[e["fid"]] = float(np.median(
                np.linalg.norm(px[ok] - t["gt8"][ok], axis=1)))
        per_model[name] = rows

    report = {"gross_px": GROSS_PX, "axes": {}, "cross": {}, "compare": {}}
    fids = [f for f in axes if np.isfinite(per_model[MODELS[0]].get(f, np.nan))]

    def bucket_stats(name, keyfn, order):
        out = {}
        for b in order:
            sub = [f for f in fids if keyfn(axes[f]) == b]
            if not sub:
                continue
            err = np.array([per_model[name][f] for f in sub], float)
            err = err[np.isfinite(err)]
            out[b] = {"n": len(sub),
                      "corner_median": round(float(np.median(err)), 2),
                      "gross_rate": round(float(np.mean(err > GROSS_PX)), 3)}
        return out

    elev_b = lambda a: ("<3" if a["elev"] is None else
                        "<3" if a["elev"] < 3 else "3-8" if a["elev"] < 8
                        else "8-15" if a["elev"] < 15 else "15+")
    trunc_b = lambda a: "truncated" if a["truncated"] else "full"
    size_b = lambda a: ("<0.15" if a["size_ratio"] < 0.15 else
                        "0.15-0.30" if a["size_ratio"] < 0.30 else ">=0.30")
    luma_b = lambda a: ("dark<60" if a["luma"] < 60 else
                        "mid60-120" if a["luma"] < 120 else "bright>=120")

    for label, fn, order in (("elevation", elev_b, ["<3", "3-8", "8-15", "15+"]),
                             ("truncation", trunc_b, ["full", "truncated"]),
                             ("screen_size", size_b, ["<0.15", "0.15-0.30", ">=0.30"]),
                             ("luma", luma_b, ["dark<60", "mid60-120", "bright>=120"])):
        report["axes"][label] = {m: bucket_stats(m, fn, order) for m in MODELS}

    # 교차: 어느 축이 진짜인지 — 같은 버킷 안에서 다른 축이 갈리는지 본다
    for f in fids:
        a = axes[f]
        a["_elev"] = elev_b(a); a["_trunc"] = trunc_b(a)
        a["_size"] = size_b(a); a["_luma"] = luma_b(a)
    cross = {}
    for e in ["<3", "3-8", "8-15", "15+"]:
        for s in ["<0.15", "0.15-0.30", ">=0.30"]:
            sub = [f for f in fids if axes[f]["_elev"] == e and axes[f]["_size"] == s]
            if len(sub) < 5:
                continue
            row = {}
            for m in MODELS:
                err = np.array([per_model[m][f] for f in sub], float)
                err = err[np.isfinite(err)]
                row[m] = {"n": len(sub),
                          "median": round(float(np.median(err)), 1),
                          "gross": round(float(np.mean(err > GROSS_PX)), 2)}
            cross[f"elev {e} x size {s}"] = row
    report["cross"] = cross

    # 꼬리 프레임 목록 — 눈으로 볼 수 있게
    tail = sorted(((per_model["FINAL40K_seed1"][f], f) for f in fids
                   if per_model["FINAL40K_seed1"][f] > GROSS_PX), reverse=True)
    report["tail_frames"] = [
        {"fid": f, "final_corner_px": round(v, 1),
         "yolo_broad_corner_px": round(per_model["yolo26n_broad40k_5ep"][f], 1),
         **{k: axes[f][k] for k in ("set", "elev", "n_outside", "size_ratio", "luma")}}
        for v, f in tail[:25]]
    report["tail_summary"] = {
        "n_tail": len(tail), "n_total": len(fids),
        "rate": round(len(tail) / max(len(fids), 1), 3)}

    json.dump(report, open(os.path.join(OUT, "FINAL40K_TAIL_DIAG.json"), "w"),
              indent=1, default=str)

    for label in ("elevation", "truncation", "screen_size", "luma"):
        print(f"\n=== {label} ===")
        buckets = list(report["axes"][label][MODELS[0]])
        print(f"{'bucket':14}{'n':>5}" +
              "".join(f"{m.replace('_seed1','').replace('yolo26n_','y_'):>26}"
                      for m in MODELS))
        for b in buckets:
            row = f"{b:14}{report['axes'][label][MODELS[0]][b]['n']:>5}"
            for m in MODELS:
                e = report["axes"][label][m][b]
                row += f"   med {e['corner_median']:>7.1f}px gross {e['gross_rate']:>5.2f}"
            print(row)
    print(f"\n꼬리(>{GROSS_PX}px) 프레임 {report['tail_summary']['n_tail']}"
          f"/{report['tail_summary']['n_total']} "
          f"({report['tail_summary']['rate']:.1%})")
    print("-> FINAL40K_TAIL_DIAG.json")


if __name__ == "__main__":
    main()
