"""stage0/extract_gt_candidates.py — 표적 GT 라벨 후보 추출 (실패모드별 stratify).

목적: filter-val GT가 작아 diag를 시험 못 함(구조-gross 0장). outside POOL에서
필터들이 *불일치*하는 프레임(=구분력 있는 후보)을 실패모드별로 골라, 수십 장만
표적 수동 GT 라벨하면 diag의 구조-파국 억제를 시험 + 데이터셋 기여.

stratify (outside, diag pass, n_det>=6):
  struct : diag pass ∧ flip FAIL(>tau_flip)          → flip이 거르는 구조 불일치 (diag-alone이면 통과)
  scale  : diag pass ∧ flip PASS ∧ size FAIL(a4>0)   → size가 거르는 scale 붕괴
  clean  : diag pass ∧ flip PASS ∧ size PASS         → 전부 통과(정상 후보, 대조)

기존 재사용: stage0 _records.json(diag/flip/n_det), PL json projected_cuboid,
four_arm a4_envelope. CPU only. final-test/cad는 pool에 없음(split-lock).
"""
import argparse, glob, json, os, cv2, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PL_DIR = os.path.join(ROOT, "data", "pallet", "pl", "stage0_base_v2_sigma2")
RAW = os.path.join(ROOT, "data", "pallet", "raw_data", "outside")
FA = os.path.join(ROOT, "data", "pallet", "eval_results", "stage0_four_arm", "four_arm_compare.json")
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def bbox_size_aspect(pts8, img_diag):
    a = np.asarray([p for p in pts8 if p is not None and p[0] > -50], float)
    if len(a) < 6:
        return None
    w = a[:,0].max()-a[:,0].min(); h = a[:,1].max()-a[:,1].min()
    if w <= 1e-6 or h <= 1e-6:
        return None
    return float(np.hypot(w,h)/img_diag), float(w/h)


def excursion(v, lo, hi):
    half = (hi-lo)/2.0; mid = (lo+hi)/2.0
    return max(0.0, abs(v-mid)-half)/half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_diag", type=float, default=0.05)
    ap.add_argument("--tau_flip", type=float, default=10.0)
    ap.add_argument("--out_dir", default=os.path.join(ROOT,"data","pallet","eval_results","stage0_gt_candidates"))
    ap.add_argument("--n_overlay", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = json.load(open(FA))["a4_envelope"]
    recs = [r for r in json.load(open(os.path.join(PL_DIR,"_records.json"))) if r["domain"]=="outside"]
    # session map (raw outside)
    sm = {}
    for d in glob.glob(os.path.join(RAW,"capture*")):
        for p in glob.glob(os.path.join(d,"rgb","*.png")):
            sm[os.path.splitext(os.path.basename(p))[0]] = p

    strata = {"struct": [], "scale": [], "clean": []}
    for r in recs:
        if r["n_detected"] < 6 or not r["diag_pass"]:
            continue
        if (r["diag_score"] or 9) >= args.tau_diag:
            continue
        fid = str(r["frame"])
        pj = os.path.join(PL_DIR, fid+".json")
        if not os.path.exists(pj):
            continue
        cub = json.load(open(pj))["objects"][0]["projected_cuboid"][:8]
        cam = json.load(open(pj)).get("camera_data",{})
        W = cam.get("width",640); H = cam.get("height",480)
        sa = bbox_size_aspect(cub, float(np.hypot(W,H)))
        a4 = 0.0 if sa is None else max(excursion(sa[0],env["size_lo"],env["size_hi"]),
                                        excursion(sa[1],env["asp_lo"],env["asp_hi"]))
        flip = r["flip_score"] if r["flip_score"] is not None else 1e9
        rec = {"frame": fid, "diag": r["diag_score"], "flip": flip, "a4": round(a4,3),
               "n_det": r["n_detected"], "img": sm.get(fid)}
        if flip > args.tau_flip:
            strata["struct"].append(rec)
        elif a4 > 0:
            strata["scale"].append(rec)
        else:
            strata["clean"].append(rec)

    print(f"=== outside 표적 GT 후보 (diag pass ∧ n_det>=6) ===")
    for k in ("struct","scale","clean"):
        print(f"  {k:<7}: {len(strata[k]):4d}   "
              + {"struct":"diag pass ∧ flip FAIL  (flip이 거르는 구조 불일치 = diag-alone이면 통과)",
                 "scale":"diag∧flip PASS ∧ size FAIL (size가 거르는 scale 붕괴)",
                 "clean":"diag∧flip∧size PASS  (정상 후보, 대조)"}[k])

    # 실패모드별 overlay 그리드 (struct/scale 우선 — diag 구분력 표적)
    for k in ("struct","scale"):
        items = sorted(strata[k], key=lambda x:-x["flip"] if k=="struct" else -x["a4"])[:args.n_overlay]
        if not items: continue
        cols = 4; rows = (len(items)+cols-1)//cols; cell=(320,240)
        grid = np.full((rows*cell[1],cols*cell[0],3),30,np.uint8)
        for i,it in enumerate(items):
            if not it["img"]: continue
            im = cv2.imread(it["img"])
            pj = json.load(open(os.path.join(PL_DIR,it["frame"]+".json")))
            pts = [(int(round(p[0])),int(round(p[1]))) for p in pj["objects"][0]["projected_cuboid"][:8]]
            for a,b in EDGES:
                if pts[a][0]>-50 and pts[b][0]>-50: cv2.line(im,pts[a],pts[b],(0,255,0),2)
            for p in pts:
                if p[0]>-50: cv2.circle(im,p,3,(0,0,255),-1)
            im = cv2.resize(im,cell)
            cv2.putText(im,f"{k} flip{it['flip']:.0f} a4{it['a4']:.2f}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,255),1)
            r_,c_ = divmod(i,cols); grid[r_*cell[1]:(r_+1)*cell[1],c_*cell[0]:(c_+1)*cell[0]]=im
        outp = os.path.join(args.out_dir,f"_grid_{k}.png"); cv2.imwrite(outp,grid)
        print(f"  overlay: {outp}")

    json.dump({"tau_diag":args.tau_diag,"tau_flip":args.tau_flip,"env":env,
               "counts":{k:len(v) for k,v in strata.items()},"strata":strata},
              open(os.path.join(args.out_dir,"gt_candidates.json"),"w"), indent=2)
    print(f"  saved: {os.path.join(args.out_dir,'gt_candidates.json')}")


if __name__ == "__main__":
    main()
