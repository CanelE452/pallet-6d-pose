"""PHASE 1~2 — top-K ranking oracle headroom + correct-box/bad-pose 분리.

질문: low-conf candidate list 안에 있는 correct candidate 를 top-1 confidence
ranking 이 놓치는 몫을, GT 없이 rerank 해서 **pose 까지** 회수할 여지가 있는가.

재추론 0 — `_cc_raw_dump.json`(conf=0.001 전수 후보) 을 그대로 쓴다.
★ oracle 은 deployment result 가 아니다. reranking 가치의 **상한**이다.

--- 결과 보기 전에 고정된 정의 -----------------------------------------------
  IOU_MATCH   0.5     correct box 판정
  USABLE      R<=10deg AND t<=0.10m   "쓸 만한 pose" (D3 의 GROSS_R=10 과 정합)
  oracle 선택 top-K 중 **R 오차 최소** candidate
  oracle_5cm5 top-K 중 **아무거나** 5cm5 성공 (5cm5 의 진짜 상한)
  STOP        Top5 oracle 대비 Top1 에서 recall gain <5pp AND 5cm5 gain <3pp
              -> RANKING_HEADROOM_TOO_SMALL, 여기서 종료
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/real_eval", "scripts/stage0/model_compare"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import cv2                    # noqa: E402
import re_metrics as RM       # noqa: E402

A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
IOU_MATCH = 0.5
USABLE_R, USABLE_T = 10.0, 0.10
KS = (1, 2, 3, 5)
STOP = {"recall_gain_min_pp": 5.0, "s5_gain_min_pp": 3.0}


def bbox_of(pts, w, h):
    p = np.asarray(pts, float)
    return [max(0.0, float(p[:, 0].min())), max(0.0, float(p[:, 1].min())),
            min(float(w), float(p[:, 0].max())), min(float(h), float(p[:, 1].max()))]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1-ix0) * max(0.0, iy1-iy0)
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + \
        max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def solve(kps, truth):
    px = np.asarray(kps, float)[:8]
    if not np.isfinite(px).all():
        return None
    ok, rv, tv = cv2.solvePnP(truth["model"][:8], px.reshape(-1, 1, 2),
                              truth["K"], None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(truth["model"][:8], px.reshape(-1, 1, 2),
                                  truth["K"], None, rv, tv)
    return cv2.Rodrigues(rv)[0], tv.reshape(3)


def build():
    """프레임마다 후보 전부에 GT 진단치 + GT-free feature 를 붙인다."""
    raw = json.load(open(os.path.join(A2, "_cc_raw_dump.json")))
    man = {i["frame_id"]: i for i in
           json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]}
    frames = []
    for e in raw["positive"]:
        m = man[e["fid"]]
        im = cv2.imread(os.path.join(ROOT, m["image"]))
        h, w = im.shape[:2]
        truth = {"R": np.asarray(m["R_gt"], float),
                 "t": np.asarray(m["t_gt"], float),
                 "K": np.asarray(m["K"], float),
                 "model": np.asarray(m["object_points"], float),
                 "gt8": np.asarray(m["gt_corners_2d"], float)}
        gtb = bbox_of(truth["gt8"], w, h)
        cands = []
        for rank, b in enumerate(e["boxes"]):          # 이미 conf 내림차순
            px = np.asarray(b["kps"], float)
            pb = [max(0.0, b["xyxy"][0]), max(0.0, b["xyxy"][1]),
                  min(float(w), b["xyxy"][2]), min(float(h), b["xyxy"][3])]
            p = solve(b["kps"], truth)
            if p is None:
                d = {"R": np.nan, "t": np.nan, "corner": np.nan, "s5": 0,
                     "reproj": np.nan, "depth_ok": 0, "cuboid_ok": 0}
            else:
                Rr, tt = RM.pose_error(p[0], p[1], truth["R"], truth["t"])
                cam = (p[0] @ truth["model"].T).T + p[1]
                z = cam[:, 2]
                zc = np.clip(z, 1e-6, None)
                proj = (truth["K"] @ (cam/zc[:, None]).T).T[:, :2]
                # GT-free feature
                reproj = float(np.median(np.linalg.norm(proj - px[:8], axis=1)))
                depth_ok = int(bool((z > 0).all()))
                # cuboid plausibility: 앞면 0123 이 볼록 사각형인가 + 면적 양수
                q = px[:4]
                cr = np.cross(np.roll(q, -1, 0) - q, np.roll(q, -2, 0)
                              - np.roll(q, -1, 0))
                cuboid_ok = int(bool(np.all(cr > 0) or np.all(cr < 0)))
                d = {"R": Rr, "t": tt, "s5": int(RM.success_5cm5deg(
                        p[0], p[1], truth["R"], truth["t"])),
                     "corner": float(np.median(np.linalg.norm(
                         px[:8] - truth["gt8"][:8], axis=1))),
                     "reproj": reproj, "depth_ok": depth_ok,
                     "cuboid_ok": cuboid_ok}
            kc = b["kp_conf"] or [0.0]*9
            kc = np.array([0.0 if v is None else v for v in kc], float)
            bw, bh = max(0.0, pb[2]-pb[0]), max(0.0, pb[3]-pb[1])
            cands.append({"rank": rank, "conf": b["conf"],
                          "iou": iou(pb, gtb), **d,
                          "kp_conf_mean": float(kc[:8].mean()),
                          "kp_conf_min": float(kc[:8].min()),
                          "box_area": bw*bh,
                          "box_diag": float(np.hypot(bw, bh))})
        frames.append({"fid": e["fid"], "set": e["set"],
                       "population": e["population"], "cands": cands})
    return frames


def main():
    frames = build()
    json.dump(frames, open(os.path.join(A2, "_rr_cands.json"), "w"), indent=1)

    def at_k(K):
        av = corr = 0
        R, T, S, ORS = [], [], [], []
        for f in frames:
            top = f["cands"][:K]
            if not top:
                continue
            av += 1
            if any(c["iou"] >= IOU_MATCH for c in top):
                corr += 1
            fin = [c for c in top if np.isfinite(c["R"])]
            pick = min(fin, key=lambda c: c["R"]) if fin else None
            if pick is not None:
                R.append(pick["R"]); T.append(pick["t"]); S.append(pick["s5"])
            else:
                S.append(0)
            ORS.append(int(any(c["s5"] for c in top)))   # 5cm5 진짜 상한
        n = len(frames)
        R, T = np.array(R, float), np.array(T, float)
        return {"K": K, "availability": av/n, "correct_recall": corr/n,
                "R_median": float(np.median(R)) if len(R) else None,
                "R_p90": float(np.percentile(R, 90)) if len(R) else None,
                "t_median": float(np.nanmedian(T)) if len(T) else None,
                "t_p90": float(np.nanpercentile(T, 90)) if len(T) else None,
                "success_5cm5_oracleR": float(np.mean(S)) if S else 0.0,
                "success_5cm5_any": float(np.mean(ORS)) if ORS else 0.0}

    ph1 = [at_k(k) for k in KS]
    k1, k5 = ph1[0], ph1[-1]
    rec_gain = (k5["correct_recall"] - k1["correct_recall"]) * 100
    s5_gain = (k5["success_5cm5_any"] - k1["success_5cm5_any"]) * 100
    stop = (rec_gain < STOP["recall_gain_min_pp"] and
            s5_gain < STOP["s5_gain_min_pp"])

    # ---- PHASE 2 (분류는 상한과 무관하게 진단 가치가 있어 항상 계산) ----
    cls = {"A_GOOD_CANDIDATE_MISRANKED": [], "B_CORRECT_BOX_BAD_KP": [],
           "C_NO_CORRECT_CANDIDATE": [], "TOP1_ALREADY_GOOD": []}
    detail = []
    for f in frames:
        top5 = f["cands"][:5]
        if not top5:
            cls["C_NO_CORRECT_CANDIDATE"].append(f["fid"]); continue
        corr = [c for c in top5 if c["iou"] >= IOU_MATCH]
        usable = [c for c in corr if np.isfinite(c["R"]) and
                  c["R"] <= USABLE_R and c["t"] <= USABLE_T]
        t1 = top5[0]
        t1_good = (t1["iou"] >= IOU_MATCH and np.isfinite(t1["R"]) and
                   t1["R"] <= USABLE_R and t1["t"] <= USABLE_T)
        if t1_good:
            k = "TOP1_ALREADY_GOOD"
        elif usable:
            k = "A_GOOD_CANDIDATE_MISRANKED"
        elif corr:
            k = "B_CORRECT_BOX_BAD_KP"
        else:
            k = "C_NO_CORRECT_CANDIDATE"
        cls[k].append(f["fid"])
        best = min(corr, key=lambda c: c["R"]) if corr and any(
            np.isfinite(c["R"]) for c in corr) else None
        detail.append({"fid": f["fid"], "set": f["set"],
                       "population": f["population"], "cls": k,
                       "n_cand": len(f["cands"]),
                       "top1_iou": t1["iou"], "top1_R": t1["R"],
                       "top1_t": t1["t"],
                       "best_correct_rank": None if best is None else best["rank"],
                       "best_correct_R": None if best is None else best["R"],
                       "best_correct_t": None if best is None else best["t"]})

    def blk(key):
        rows = [d for d in detail if d["cls"] == key]
        if not rows:
            return {"n": 0, "frac": 0.0}
        rk = [r["best_correct_rank"] for r in rows
              if r["best_correct_rank"] is not None]
        return {"n": len(rows), "frac": round(len(rows)/len(frames), 4),
                "best_correct_rank_median": (None if not rk
                                             else float(np.median(rk))),
                "by_population": {p: sum(1 for r in rows
                                         if r["population"] == p)
                                  for p in ("REAL_DEV_OPEN_56",
                                            "REAL_CHALLENGE_DEV_105")}}
    ph2 = {k: blk(k) for k in cls}

    out = {"note": "oracle 은 deployment result 가 아니다. reranking 가치의 상한.",
           "inference": "재추론 0 — _cc_raw_dump.json (conf=0.001 전수 후보)",
           "prelocked": {"iou_match": IOU_MATCH,
                         "usable_pose": f"R<={USABLE_R}deg AND t<={USABLE_T}m",
                         "oracle_pick": "top-K 중 R 오차 최소",
                         "success_5cm5_any": "top-K 중 아무거나 5cm5 성공 (진짜 상한)",
                         "stop_rule": STOP},
           "phase1_topk_oracle": ph1,
           "phase1_gains_top1_to_top5": {
               "correct_recall_gain_pp": round(rec_gain, 2),
               "success_5cm5_any_gain_pp": round(s5_gain, 2),
               "R_median_top1": k1["R_median"], "R_median_top5": k5["R_median"]},
           "phase2_classification": ph2,
           "phase1_verdict": ("RANKING_HEADROOM_TOO_SMALL" if stop
                              else "HEADROOM_SUFFICIENT_PROCEED")}
    json.dump(out, open(os.path.join(A2, "RERANK_ORACLE.json"), "w"),
              indent=1, ensure_ascii=False)
    json.dump(detail, open(os.path.join(A2, "_rr_detail.json"), "w"), indent=1)

    print(f"{'K':>3}{'avail':>8}{'corr_rec':>10}{'R med':>9}{'R p90':>9}"
          f"{'t med':>9}{'t p90':>9}{'5cm5(oR)':>10}{'5cm5(any)':>11}")
    print("─"*78)
    for s in ph1:
        print(f"{s['K']:>3}{s['availability']:>8.3f}{s['correct_recall']:>10.3f}"
              f"{s['R_median']:>9.2f}{s['R_p90']:>9.2f}{s['t_median']:>9.4f}"
              f"{s['t_p90']:>9.4f}{s['success_5cm5_oracleR']:>10.3f}"
              f"{s['success_5cm5_any']:>11.3f}")
    print(f"\nTop1->Top5  correct recall {rec_gain:+.2f}pp (기준 5pp)   "
          f"5cm5(any) {s5_gain:+.2f}pp (기준 3pp)")
    print(f"PHASE1 = {out['phase1_verdict']}\n")
    print("PHASE 2 분류 (top-5 기준, n=161)")
    for k, v in ph2.items():
        print(f"  {k:28} n={v['n']:>3} ({v['frac']:.3f})  "
              f"best_correct_rank_med={v.get('best_correct_rank_median')}  "
              f"{v.get('by_population')}")


if __name__ == "__main__":
    main()
