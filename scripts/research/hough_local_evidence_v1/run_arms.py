"""H0 / H1 — 국소 이미지 증거가 Hough 일반화의 병목인가.

정본 학습 루프(direct_hough_role_heatmap.train_network)와 같은 구성:
  lattice · hypothesis_features · target_distribution · cross_entropy ·
  batch_rows · decode · measure · summarise · AdamW(LR 1e-3, WD 1e-4) · BATCH 12
데이터만 살아 있는 셋으로 바꾼다 (원 학습셋 pallet6d_v2_10k 은 삭제됨).

H0  role @ f(theta,rho)                        현행 구조
H1  H0 + raster head + CoarseRadon 누적 점수    국소 증거 경로 (단일 변수 아님 — README §2)

학습에서 끝내지 않는다: 학습 -> 교차셋 평가 -> 사전등록 게이트 판정 -> 마크 기록.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time, glob, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/pallet/results/hough_local_evidence_v1"
TD = ROOT / "data/pallet/training_data"
STEPS = 3000
MARKS = (500, 1000, 2000, 3000)
TRAIN_SET = "mixed_v8_train"
CROSS = ["v4_split_base", "aug_squash_v2", "paper_4pallet_mask_v1"]
N_DEV, N_CROSS = 512, 384
GRID = 50
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

# 사전등록 게이트 (결과 보기 전 고정)
GATE = {"h0_dev_angle_lo": 3.0, "h0_dev_angle_hi": 6.0,
        "cross_improve_strong": 0.30, "cross_improve_weak": 0.10}


def _load(n, r):
    sp = importlib.util.spec_from_file_location(n, ROOT/r); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT/"PROGRESS.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ data
def frames_of(name, limit=None):
    files = sorted(glob.glob(str(TD/name/"*.json")))
    out = []
    for f in files:
        png = f[:-5] + ".png"
        if Path(png).is_file():
            out.append(Path(f).stem)
        if limit and len(out) >= limit:
            break
    return out


def load_batch(name, stems):
    ims, grids = [], []
    for s in stems:
        d = json.load(open(TD/name/f"{s}.json")); cam = d["camera_data"]
        pc = np.asarray(d["objects"][0].get("projected_cuboid", []), float)[:8]
        if pc.shape != (8,2) or not np.isfinite(pc).all():
            continue
        img = cv2.imread(str(TD/name/f"{s}.png"))
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (400,400)), cv2.COLOR_BGR2RGB)
        ims.append(((rgb.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1))
        W,H = float(cam["width"]), float(cam["height"])
        grids.append(np.stack([pc[:,0]*GRID/W, pc[:,1]*GRID/H], 1))
    if not ims:
        return None
    return {"images": torch.from_numpy(np.stack(ims)), "grid": np.stack(grids)}


# ------------------------------------------------------------------ H1
class LocalEvidenceModel(nn.Module):
    """H0 그대로 + raster head + CoarseRadon 누적 점수를 더한다."""

    def __init__(self, DH, SD, n_hyp):
        super().__init__()
        self.base = DH.DirectHoughModel()
        self.raster = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
                                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                                    nn.Conv2d(32, 12, 1))
        self.radon = SD.CoarseRadon()
        self.gain = nn.Parameter(torch.zeros(1))      # 0 에서 시작 -> step0 은 H0 와 동일
        self.n_hyp = n_hyp

    def forward(self, f50, features, valid_idx):
        base = self.base(f50, features)                       # (B, 12, Hyp)
        up = F.interpolate(f50, size=(100,100), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(self.raster(up))                 # (B, 12, 100, 100)
        B = prob.shape[0]
        acc = []
        for b in range(B):
            maps = prob[b].reshape(12, -1).transpose(0,1)     # (P, 12)
            sc = self.radon.scores(maps)["H2_ZERO_MEAN_NCC"]  # (T, Rho, 12)
            flat = sc.reshape(-1, 12).transpose(0,1)          # (12, T*Rho)
            acc.append(flat[:, valid_idx])
        return base + self.gain * torch.stack(acc)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["H0", "H1"], required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    DH = _load("DH", "scripts/stage0/line/direct_hough_role_heatmap.py")
    SD = _load("SD", "scripts/stage0/line/structural_line_hough_decoder.py")
    dev = DH.DEV
    torch.manual_seed(1); np.random.seed(1)

    gt_, gr_, valid = DH.lattice()
    feats = DH.hypothesis_features(gt_[valid], gr_[valid])
    valid_idx = torch.nonzero(valid).flatten()
    a1 = DH.V2.load_a1()

    all_stems = frames_of(TRAIN_SET)
    rng = np.random.default_rng(1); rng.shuffle(all_stems)
    dev_stems = all_stems[:N_DEV]; train_stems = all_stems[N_DEV:]
    log(f"{args.arm}: train {len(train_stems)}  dev {len(dev_stems)}  from {TRAIN_SET}")

    model = (DH.DirectHoughModel() if args.arm == "H0"
             else LocalEvidenceModel(DH, SD, int(valid.sum()))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def fwd(f50):
        return model(f50, feats) if args.arm == "H0" else model(f50, feats, valid_idx)

    def evaluate(name, stems, tag):
        model.eval(); A, O = [], []
        for i in range(0, len(stems), 12):
            pack = load_batch(name, stems[i:i+12])
            if pack is None: continue
            tc, rc, sup = DH.batch_rows(pack, EDGES)
            with torch.no_grad():
                f50, _, _ = a1(pack["images"].to(dev))
                sc = fwd(f50)
            for b in range(sc.shape[0]):
                live = torch.nonzero(sup[b]).flatten()
                if live.numel() == 0: continue
                th, rh = DH.decode(sc[b][live], gt_[valid], gr_[valid],
                                   torch.ones_like(gt_[valid], dtype=torch.bool))
                a, o = DH.measure(th, rh, tc[b][live], rc[b][live])
                A.append(a); O.append(o)
        A = np.concatenate(A); O = np.concatenate(O)
        r = DH.summarise(A, O); r["population"] = tag
        log(f"  {args.arm} {tag:24s} angle med {r['angle_median']:7.3f} "
            f"p90 {r['angle_p90']:7.2f}  offset med {r['offset_median']:6.3f}")
        return r

    history, losses, done = {}, [], 0
    t0 = time.time()
    while done < args.steps:
        rng.shuffle(train_stems)
        for i in range(0, len(train_stems), 12):
            if done >= args.steps: break
            pack = load_batch(TRAIN_SET, train_stems[i:i+12])
            if pack is None: continue
            model.train()
            tc, rc, sup = DH.batch_rows(pack, EDGES)
            target = DH.target_distribution(tc.reshape(-1), rc.reshape(-1),
                                            gt_[valid], gr_[valid],
                                            torch.ones_like(gt_[valid], dtype=torch.bool)
                                            ).reshape(*tc.shape, -1)
            with torch.no_grad():
                f50, _, _ = a1(pack["images"].to(dev))
            loss = DH.cross_entropy(fwd(f50), target, sup,
                                    torch.ones_like(gt_[valid], dtype=torch.bool))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach())); done += 1
            if done % 250 == 0:
                log(f"  {args.arm} step {done:5d}/{args.steps}  loss "
                    f"{np.mean(losses[-250:]):.4f}  {(time.time()-t0)/60:.1f} min")
            if done in MARKS:
                e = {"step": done, "loss_last250": float(np.mean(losses[-250:])),
                     "OWN_DEV": evaluate(TRAIN_SET, dev_stems, "OWN_DEV")}
                if done == args.steps:
                    for cs in CROSS:
                        e[cs] = evaluate(cs, frames_of(cs, N_CROSS), cs)
                history[str(done)] = e
                torch.save({"arm": args.arm, "step": done, "model": model.state_dict()},
                           OUT/f"{args.arm}_step{done:05d}.pth")

    (OUT/f"{args.arm}_HISTORY.json").write_text(
        json.dumps({"arm": args.arm, "steps": args.steps, "train_set": TRAIN_SET,
                    "n_train": len(train_stems), "n_dev": len(dev_stems),
                    "gate": GATE, "history": history}, indent=2) + "\n")
    log(f"{args.arm} DONE  {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
