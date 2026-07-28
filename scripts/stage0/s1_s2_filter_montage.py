"""s1_s2_filter_montage.py — s1(paper_s1_maskaux) vs s2(Stage B) 추론 오버레이 +
배포필터 통과 여부를 여러 프레임 한 장 몽타주로 비교.

- s1 = pad100 (s1_cad_9filters canonical, s1은 squash 입력에 OOD → 제 방식으로 최적 추론)
- s2 = squash-parity (paper_s2 Stage B canonical)
- 배포필터 = f1&f2&f4&f5&f6&f7 (f3 제외=L-R broken, f8/f9=GT 필요). 각 모델 자기 decode로 계산.
- 패널: GT(초록)+pred(빨강) cuboid, 헤더에 model/corner_med/GOOD·BAD/필터PASS·FAIL.
  ★필터 통과 = 초록 테두리, 실패 = 빨강 테두리로 한눈에.
- 프레임 = filterval(outside+manual, base 품질 有) 검출분을 corner_med 순 균등 8개 선정
  (good→pass-but-bad→bad 스펙트럼 포함).

산출: data/pallet/eval_results/paper_s2_scratch_diffpnp/s1_s2_filter_montage.jpg

Usage: conda activate pallet-pose; python scripts/stage0/s1_s2_filter_montage.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_testset17_9filters as T   # noqa: E402  (squash infer, M, TAU)
import stage25_paperbase_eval as S        # noqa: E402  (frames_filterval)
import annotate_pnp as APNP               # noqa: E402
import cv2      # noqa: E402
import torch    # noqa: E402

M = T.M
TAU = T.TAU
E = T.E
THRESH = T.THRESH
N_DET_MIN = T.N_DET_MIN
GOOD_PX = T.GOOD_PX
DEPLOY = ["f1_peak", "f2_peak_ratio", "f4_tta_stab", "f5_rear_conf",
          "f6_frsep", "f7_posdepth"]
REAR = (4, 5, 6, 7)
ENV = {"size_lo": 0.0, "size_hi": 1.0, "asp_lo": 0.0, "asp_hi": 10.0}  # deploy 미사용
S1_W = os.path.join(ROOT, "weights", "paper_s1_maskaux", "net_epoch_0065.pth")
S2_W = os.path.join(ROOT, "weights", "paper_s2_stageB", "net_epoch_0057.pth")
S2_JSON = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp",
                       "orthogonal_filters_exp.json")
OUT = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp",
                   "s1_s2_filter_montage.jpg")
PAD = 100
PANEL_W = 520


def gt_from_json(jp):
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    gtc = np.array(o["projected_cuboid_centroid"], float)
    return gt8, gtc, E.K_from_json(d)


# 배포 6개 필터의 사람이 읽을 짧은 라벨 (opencv=ASCII만 렌더 → 영문)
FSHORT = {"f1_peak": "corner-conf", "f2_peak_ratio": "peak-sharp",
          "f4_tta_stab": "TTA-stable", "f5_rear_conf": "rear-conf",
          "f6_frsep": "depth-sep", "f7_posdepth": "pose-z>0"}


def deploy_verdict(scores, f7, n_det):
    """(passed, failed_labels). failed = 떨어진 필터의 읽을 라벨 list."""
    if n_det < N_DET_MIN:
        return False, ["under-det"]
    row = {"n_det": n_det, "scores": scores, "f7_posdepth": bool(f7),
           "pred_sr": None, "pred_asp": None}
    failed = [FSHORT[f] for f in DEPLOY if not M.apply_filter(f, row, TAU.get(f), ENV)]
    return (len(failed) == 0), failed


def infer_s2(model, img, K):
    """Stage B squash: pred8/pred_c + deploy filter."""
    belief, pred8, pred_c, peaks, ratios = T.infer_squash(model, img, DEV)
    n_det = int((~np.isnan(pred8[:, 0])).sum())
    scores, f7 = {}, False
    if n_det >= N_DET_MIN:
        det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
        scores["f1_peak"] = float(min(peaks[i] for i in det8))
        scores["f2_peak_ratio"] = float(min(ratios[i] for i in det8))
        scores["f4_tta_stab"] = T.tta_stab_squash(model, img, DEV)
        rear = [i for i in REAR if not np.isnan(pred8[i, 0])]
        scores["f5_rear_conf"] = float(min(peaks[i] for i in rear)) if len(rear) == 4 else None
        scores["f6_frsep"] = M.frsep_frac(pred8)
        f7, _ = M.posdepth_ok(pred8, pred_c, K, img.shape)
    passed, failed = deploy_verdict(scores, f7, n_det)
    return pred8, pred_c, n_det, passed, failed


def infer_s1(model, img, K):
    """paper_s1 pad100 (canonical): pred8/pred_c + deploy filter."""
    belief, geom, wh = M.infer_belief(model, img, DEV, PAD)
    pred8, pred_c, peaks, ratios = M.belief_to_pred(belief, geom, wh, PAD, THRESH)
    n_det = int((~np.isnan(pred8[:, 0])).sum())
    scores, f7 = {}, False
    if n_det >= N_DET_MIN:
        det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
        scores["f1_peak"] = float(min(peaks[i] for i in det8))
        scores["f2_peak_ratio"] = float(min(ratios[i] for i in det8))
        scores["f4_tta_stab"] = M.tta_stability(model, img, DEV, PAD, THRESH)
        rear = [i for i in REAR if not np.isnan(pred8[i, 0])]
        scores["f5_rear_conf"] = float(min(peaks[i] for i in rear)) if len(rear) == 4 else None
        scores["f6_frsep"] = M.frsep_frac(pred8)
        f7, _ = M.posdepth_ok(pred8, pred_c, K, img.shape)
    passed, failed = deploy_verdict(scores, f7, n_det)
    return pred8, pred_c, n_det, passed, failed


def corner_med(pred8, gt8):
    from eval_pvnet_heads import split_metrics
    return split_metrics(np.array(pred8, float), np.array(gt8, float))["overall"]


def panel(ip, gt8, gtc, pred8, pred_c, name, cm, passed, failed):
    img = cv2.imread(ip)
    M.draw_cuboid(img, gt8, gtc, M.GT_COL, 6, 2, 7, num=False)
    M.draw_cuboid(img, np.array(pred8, float), pred_c, M.PRED_COL, 4, -1, 6)
    H, W = img.shape[:2]
    scale = PANEL_W / W
    img = cv2.resize(img, (PANEL_W, int(H * scale)))
    # 필터 통과 테두리 (초록=PASS, 빨강=fail)
    bcol = (0, 200, 0) if passed else (0, 0, 230)
    img = cv2.copyMakeBorder(img, 56, 6, 6, 6, cv2.BORDER_CONSTANT, value=bcol)
    good = cm is not None and np.isfinite(cm) and cm < GOOD_PX
    cms = f"{cm:.1f}px" if (cm is not None and np.isfinite(cm)) else "n/a"
    gtag = "GOOD" if good else "BAD"
    ftag = "FILTER PASS" if passed else "filter FAIL"
    cv2.putText(img, f"{name}  cm={cms} [{gtag}]  {ftag}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    # 두 번째 줄: 떨어진 필터(왜 실패했나)
    sub = ("passes all 6" if passed else "fail: " + ", ".join(failed))
    cv2.putText(img, sub, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    return img


def hcat(a, b):
    h = max(a.shape[0], b.shape[0])
    def padh(x):
        return cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return np.hstack([padh(a), padh(b)])


def vcat(imgs):
    w = max(x.shape[1] for x in imgs)
    def padw(x):
        return cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return np.vstack([padw(x) for x in imgs])


MANIFEST = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/testset_full8_manifest.txt")
MAX_PER_IMG = 12   # 프레임/이미지 상한 (초과 시 자동 분할)
OUT_DIR = os.path.dirname(OUT)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sample_filterval(dom, n):
    """filterval 도메인 검출분을 corner_med 순 균등 n개 (good→bad 스펙트럼)."""
    recs = json.load(open(S2_JSON))
    d = [r for r in recs if r.get("n_det", 0) >= N_DET_MIN
         and r["dom"] == dom and r.get("corner_med") is not None]
    d.sort(key=lambda r: r["corner_med"])
    if not d:
        return []
    idx = np.linspace(0, len(d) - 1, min(n, len(d))).round().astype(int)
    return [(dom, str(d[i]["fid"])) for i in sorted(set(idx))]


def make_cell(m_s1, m_s2, dom, fid, jp, ip):
    gt8, gtc, K = gt_from_json(jp)
    img = cv2.imread(ip)
    p1, c1, nd1, pass1, fail1 = infer_s1(m_s1, img, K)
    p2, c2, nd2, pass2, fail2 = infer_s2(m_s2, img, K)
    cm1 = corner_med(p1, gt8) if nd1 >= N_DET_MIN else None
    cm2 = corner_med(p2, gt8) if nd2 >= N_DET_MIN else None
    pan1 = panel(ip, gt8, gtc, p1, c1, "s1(pad)", cm1, pass1, fail1)
    pan2 = panel(ip, gt8, gtc, p2, c2, "s2(squash)", cm2, pass2, fail2)
    row = hcat(pan1, pan2)
    cap = np.full((26, row.shape[1], 3), 40, np.uint8)
    cv2.putText(cap, f"{dom}/{fid[:16]}   GT=green pred=red", (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
    print(f"  {dom:<8}{fid[:14]:<15} s1: det{nd1} cm={cm1 and round(cm1,1)} "
          f"{'PASS' if pass1 else 'fail'} | s2: det{nd2} cm={cm2 and round(cm2,1)} "
          f"{'PASS' if pass2 else 'fail'}")
    return np.vstack([cap, row])


def montage(cells, title_txt):
    """cells (frame별 이미지) -> 2열 그리드 1장."""
    rows = []
    for r in range(0, len(cells), 2):
        pair = cells[r:r + 2]
        rows.append(hcat(pair[0], pair[1]) if len(pair) == 2 else pair[0])
    grid = vcat(rows)
    title = np.full((40, grid.shape[1], 3), 15, np.uint8)
    cv2.putText(title, title_txt, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([title, grid])


LEGEND = ("filter=corner-conf & peak-sharp & TTA-stable & rear-conf & depth-sep & pose-z>0 "
          "(flip EXCLUDED: broken by L-R asymmetry). green=PASS red=FAIL")


def build_group(m_s1, m_s2, frame_list, label, out_base):
    """frame_list=[(dom,fid,jp,ip)] -> MAX_PER_IMG 단위 자동 분할 저장. 경로 list 반환."""
    cells = [make_cell(m_s1, m_s2, dom, fid, jp, ip) for dom, fid, jp, ip in frame_list]
    chunks = [cells[i:i + MAX_PER_IMG] for i in range(0, len(cells), MAX_PER_IMG)]
    paths = []
    for k, ch in enumerate(chunks):
        part = f" (part {k+1}/{len(chunks)})" if len(chunks) > 1 else ""
        title = (f"s1(paper_s1 pad100) vs s2(Stage B squash){part} -- {label}.  {LEGEND}")
        img = montage(ch, title)
        suffix = f"_p{k+1}" if len(chunks) > 1 else ""
        p = out_base.replace(".jpg", f"{suffix}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[save] {p}  ({img.shape[1]}x{img.shape[0]})")
        paths.append(p)
    return paths


def domain_frames_sorted(m_s2, fv, dom, n=12):
    """도메인 검출분에서 cm-spread n개 뽑아, s2 필터 PASS 먼저 그다음 FAIL 순 정렬.
    (pass/fail 프레이밍 비교용)."""
    sel = sample_filterval(dom, n)          # cm-spread fids
    rows = []
    for _, fid in sel:
        jp, ip = fv[(dom, fid)]
        _, _, K = gt_from_json(jp)
        img = cv2.imread(ip)
        _, _, nd2, pass2, _ = infer_s2(m_s2, img, K)
        rows.append((0 if pass2 else 1, fid, jp, ip))   # pass(0) 먼저
    rows.sort(key=lambda x: x[0])
    return [(dom, fid, jp, ip) for _, fid, jp, ip in rows]


def sort_by_s2_pass(m_s2, frame_list):
    """명시적 (dom,fid,jp,ip) list 를 s2 필터 PASS 먼저 정렬 (검출 실패는 뒤)."""
    rows = []
    for dom, fid, jp, ip in frame_list:
        _, _, K = gt_from_json(jp)
        img = cv2.imread(ip)
        _, _, nd2, pass2, _ = infer_s2(m_s2, img, K)
        rank = 0 if pass2 else (1 if nd2 >= N_DET_MIN else 2)
        rows.append((rank, dom, fid, jp, ip))
    rows.sort(key=lambda x: x[0])
    return [(dom, fid, jp, ip) for _, dom, fid, jp, ip in rows]


def main():
    APNP.PALLET_DIMS = (1.1, 1.3, 0.12)
    m_s1 = E.load_model(S1_W, DEV)
    m_s2 = E.load_model(S2_W, DEV)

    # 지정 평가셋 testset17 (cad11 + noapril6) — 도메인별 s1 vs s2 배포필터 pass/fail
    ts17 = [(dom, str(fid), jp, ip) for dom, fid, jp, ip in T.read_manifest(MANIFEST)]
    paths = []
    for dom in ("cad", "noapril"):
        fl = [t for t in ts17 if t[0] == dom]
        print(f"[group] designated {dom} (N={len(fl)}) — s1 vs s2, deploy-filter pass/fail:")
        fl = sort_by_s2_pass(m_s2, fl)
        paths += build_group(m_s1, m_s2, fl,
                             f"designated eval {dom} (deploy-filter, pass-first)",
                             os.path.join(OUT_DIR, f"s1_s2_{dom}_passfail.jpg"))

    print(f"\n[done] {len(paths)} images:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
