"""TAKL / NRL / ASC+TAKL 후보 테스트. 15/15 PASS 전 학습 금지.

사용:  python test_candidate.py --candidate TAKL|NRL|ASC_TAKL
기존 ASC 하네스 재사용 — 새 프레임워크 없음.
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np, torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R = f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss"
OUT = f"{R}/overnight_20260823"
DATA = f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml"
INIT = f"{ROOT}/challenge/yolo_pose_one_model/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", required=True, choices=["TAKL", "NRL", "ASC_TAKL"])
A = ap.parse_args()
CFG = {"TAKL": "TAKL_LOSS_CONFIG.json", "NRL": "NRL_LOSS_CONFIG.json",
       "ASC_TAKL": "ASC_TAKL_LOSS_CONFIG.json"}[A.candidate]

res, ok_all = {}, True


def rec(n, ok, d=""):
    global ok_all
    res[n] = {"pass": bool(ok), "detail": d}
    ok_all &= bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {n:42} {d}", flush=True)


from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
import pallet_yolo_loss.symmetry as SY
from pallet_yolo_loss.symmetry import (A1SymmetryPoseLoss, takl_tail, nrl_coord,
                                       normalized_residual, asc_beta)
from pallet_yolo_loss.loss import PSPCPoseLoss26

torch.manual_seed(0)
DEV = "cuda"
C = json.load(open(f"{OUT}/{CFG}"))
TAU = C.get("takl_tau", 0.0)
ZC = C.get("takl_z_clip", 0.0)

y = YOLO(INIT, task="pose")
model = y.model.to(DEV).float().train()
for q in model.parameters():
    q.requires_grad_(True)
data = check_det_dataset(DATA)
args = get_cfg(DEFAULT_CFG, overrides=dict(
    task="pose", mode="train", data=DATA, imgsz=640, batch=8, epochs=1, workers=2,
    device=0, seed=0, single_cls=True, mosaic=0.3, scale=0.25, hsv_s=0.5, hsv_v=0.35,
    fliplr=0.0, flipud=0.0, cos_lr=True, close_mosaic=10))
model.args = args
model.nc = data["nc"]
model.names = data["names"]
ds = build_yolo_dataset(args, data["train"], 8, data, mode="train", rect=False, stride=32)
batch = next(iter(build_dataloader(ds, 8, 2, shuffle=False, rank=-1)))
batch["img"] = batch["img"].to(DEV).float() / 255
for k in ("keypoints", "bboxes", "cls", "batch_idx"):
    batch[k] = batch[k].to(DEV)


def cf(name, **kw):
    p = f"{OUT}/_t_{name}.json"
    d = dict(C)
    d.update(kw)
    json.dump(d, open(p, "w"))
    return p


def make(kind, cfg=None):
    os.environ["A1_CONFIG"] = cfg or ""
    os.environ["PSPC_CONFIG"] = ""
    torch.manual_seed(0)
    return E2ELoss(model, kind) if getattr(model, "end2end", False) else kind(model)


def run(c, grad=False):
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    p = model(batch["img"])
    l, it = c(p, copy.deepcopy(batch))
    if grad:
        l.sum().backward()
        return float(l.sum()), it.detach().cpu().numpy(), \
            [q.grad.detach().clone() for q in model.parameters() if q.grad is not None]
    return float(l.sum()), it.detach().cpu().numpy(), None


OFFKW = dict(takl_enabled=False, nrl_enabled=False, asc_enabled=False,
             enabled=False, sym_assets=[])
OFF = cf("off", **OFFKW)
ON = cf("on")
std = make(PoseLoss26)
off = make(A1SymmetryPoseLoss, OFF)
on = make(A1SymmetryPoseLoss, ON)
inner = getattr(on, "one2many", on)
SY.CURRENT_EPOCH["e"] = 45 if A.candidate != "ASC_TAKL" else 45

# ---- T1 / T2  disabled parity ---------------------------------------------
ls, is_, gs = run(std, True)
lc, ic, gc = run(off, True)
rec("T1 disabled == A0 total-loss parity",
    abs(ls - lc) < 1e-6 and np.abs(is_ - ic).max() < 1e-6,
    f"total {abs(ls-lc):.3e} items {np.abs(is_-ic).max():.3e}")
noise = 0.0
for _ in range(3):
    _, _, gn = run(std, True)
    noise = max(noise, max((a - b).abs().max().item() for a, b in zip(gs, gn)))
dg = max((a - b).abs().max().item() for a, b in zip(gs, gc))
rec("T2 disabled gradient parity", dg <= max(noise, 1e-6) * 1.5,
    f"dgrad {dg:.3e} noise {noise:.3e}")

# ---- 합성 텐서 ---------------------------------------------------------------
torch.manual_seed(3)
N, K = 24, 9
GT = torch.rand(N, K, 3, device=DEV)
GT[..., 2] = 1.0
AR = (torch.rand(N, 1, device=DEV) + 0.5)
M = GT[..., 2] != 0


def mkpred(scale):
    return GT.clone() + scale * torch.randn(N, K, 3, device=DEV) * AR.sqrt().unsqueeze(-1)


if A.candidate in ("TAKL", "ASC_TAKL"):
    tiny = GT.clone()
    tiny[..., :2] += 0.001 * AR.sqrt().unsqueeze(-1)      # r << tau
    r_t = normalized_residual(tiny, GT, AR)
    rec("T3 r<=tau 이면 tail 기여 0",
        float(r_t.max()) < TAU and float(takl_tail(tiny, GT, M, AR, TAU, ZC)) == 0.0,
        f"r_max {float(r_t.max()):.4f} < tau {TAU:.4f}, tail=0")
    big = mkpred(0.5)
    rec("T4 r>tau 이면 tail 양수",
        float(takl_tail(big, GT, M, AR, TAU, ZC)) > 0,
        f"tail {float(takl_tail(big,GT,M,AR,TAU,ZC)):.5f}")
    vals = [float(takl_tail(mkpred(s), GT, M, AR, TAU, 0.0)) for s in (0.05, 0.2, 0.5, 1.0)]
    rec("T5 residual 커질수록 단조 증가", all(b > a for a, b in zip(vals, vals[1:])),
        " < ".join(f"{v:.4f}" for v in vals))
    Mh = M.clone()
    Mh[:, 4:] = False
    a_all = float(takl_tail(big, GT, M, AR, TAU, ZC))
    a_half = float(takl_tail(big, GT, Mh, AR, TAU, ZC))
    rec("T6 visible mask 반영", abs(a_all - a_half) > 1e-6, f"all {a_all:.5f} vs half {a_half:.5f}")
    P = mkpred(0.5).requires_grad_(True)
    takl_tail(P, GT, Mh, AR, TAU, ZC).backward()
    rec("T7 invisible 점의 gradient 0",
        float(P.grad[:, 4:, :2].abs().sum()) == 0.0 and float(P.grad[:, :4, :2].abs().sum()) > 0,
        f"invis {float(P.grad[:,4:,:2].abs().sum()):.2e} vis {float(P.grad[:,:4,:2].abs().sum()):.3f}")
    Z = torch.zeros_like(M)
    rec("T8 all-mask-off 에서 finite",
        torch.isfinite(takl_tail(big, GT, Z, AR, TAU, ZC)).item(),
        f"{float(takl_tail(big,GT,Z,AR,TAU,ZC)):.3e}")
else:
    B = C["nrl_beta"]
    tiny = GT.clone()
    tiny[..., :2] += 0.0005 * AR.sqrt().unsqueeze(-1)
    v0 = float(nrl_coord(GT.clone(), GT, M, AR, B))
    rec("T3 pred==GT 이면 0", v0 == 0.0, f"{v0:.3e}")
    rec("T4 오차 있으면 양수", float(nrl_coord(mkpred(0.3), GT, M, AR, B)) > 0,
        f"{float(nrl_coord(mkpred(0.3),GT,M,AR,B)):.5f}")
    vals = [float(nrl_coord(mkpred(s), GT, M, AR, B)) for s in (0.05, 0.2, 0.5, 1.0)]
    rec("T5 residual 커질수록 단조 증가", all(b > a for a, b in zip(vals, vals[1:])),
        " < ".join(f"{v:.4f}" for v in vals))
    Mh = M.clone()
    Mh[:, 4:] = False
    big = mkpred(0.5)
    rec("T6 visible mask 반영",
        abs(float(nrl_coord(big, GT, M, AR, B)) - float(nrl_coord(big, GT, Mh, AR, B))) > 1e-6,
        "다름")
    P = big.clone().requires_grad_(True)
    nrl_coord(P, GT, Mh, AR, B).backward()
    rec("T7 invisible 점의 gradient 0",
        float(P.grad[:, 4:, :2].abs().sum()) == 0.0 and float(P.grad[:, :4, :2].abs().sum()) > 0,
        f"invis 0 vis {float(P.grad[:,:4,:2].abs().sum()):.3f}")
    # catastrophic residual 에서 gradient 가 finite & nonzero (포화 아님)
    cat = GT.clone()
    cat[..., :2] += 50.0
    P2 = cat.requires_grad_(True)
    nrl_coord(P2, GT, M, AR, B).backward()
    gnz = float(P2.grad[..., :2].abs().min())
    rec("T8 catastrophic residual gradient finite&nonzero",
        torch.isfinite(P2.grad).all().item() and gnz > 0, f"min|g| {gnz:.4e}")

# ---- T9..T15 -----------------------------------------------------------------
e2e = getattr(model, "end2end", False)
rec("T9 one2many 적용", (not e2e) or isinstance(on.one2many, A1SymmetryPoseLoss),
    type(getattr(on, "one2many", on)).__name__)
rec("T10 one2one 적용", (not e2e) or isinstance(on.one2one, A1SymmetryPoseLoss),
    type(getattr(on, "one2one", on)).__name__)
calls = {"n": 0}
_pc = PSPCPoseLoss26.projective_loss


def spy(self, *a, **k):
    calls["n"] += 1
    return _pc(self, *a, **k)


PSPCPoseLoss26.projective_loss = spy
SY.ROLE_CALLS["n"] = 0
for e in (0, 25, 45):
    SY.CURRENT_EPOCH["e"] = e
    run(on)
PSPCPoseLoss26.projective_loss = _pc
rec("T11 PC loss 호출 0", calls["n"] == 0 and inner.pspc.enabled is False, f"calls {calls['n']}")
rec("T12 symmetry 미적용(비ASC 후보)" if A.candidate != "ASC_TAKL" else "T12 symmetry 적용(ASC_TAKL)",
    (len(inner.a1.sym_assets) == 0) if A.candidate != "ASC_TAKL" else (len(inner.a1.sym_assets) == 2),
    f"sym_assets {list(inner.a1.sym_assets)}")
rec("T13 role-margin 호출 0", SY.ROLE_CALLS["n"] == 0 and inner.a1.lambda_role == 0.0,
    f"role {SY.ROLE_CALLS['n']}")
fin, det = True, []
for e in (0, 19, 25, 30, 59):
    SY.CURRENT_EPOCH["e"] = e
    l, _, gg = run(on, True)
    fin &= np.isfinite(l) and all(torch.isfinite(x).all().item() for x in gg)
    det.append(f"{e}:{l:.1f}")
rec("T14 forward/backward finite", fin, " ".join(det))
tmp = f"{OUT}/_ckpt_test.pt"
torch.save({"model": model}, tmp)
try:
    rl = torch.load(tmp, weights_only=False)["model"]
    okr = sum(p.numel() for p in rl.parameters()) == sum(p.numel() for p in model.parameters())
except Exception:
    okr = False
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
rec("T15 checkpoint reload", okr, "params 일치")

# ---- ASC_TAKL 전용 추가 검정 ---------------------------------------------------
if A.candidate == "ASC_TAKL":
    asc_only = make(A1SymmetryPoseLoss, cf("asconly", takl_enabled=False))
    takl_only = make(A1SymmetryPoseLoss, cf("taklonly", asc_enabled=False, sym_assets=[]))
    SY.CURRENT_EPOCH["e"] = 45      # beta=0 -> ASC 항 0
    a, _, _ = run(takl_only)
    b, _, _ = run(on)
    rec("T16 beta=0 이면 TAKL-only 와 동일", abs(a - b) < 1e-4, f"|diff| {abs(a-b):.3e}")
    SY.CURRENT_EPOCH["e"] = 5
    rec("T17 epoch>=30 에서 symmetry 정확히 0", asc_beta(30, 20, 30) == 0.0
        and asc_beta(59, 20, 30) == 0.0, "beta(30)=beta(59)=0")
    cfg_notail = json.load(open(f"{OUT}/{CFG}"))
    rec("T18 tail 은 beta 와 무관하게 활성",
        cfg_notail["takl_enabled"] and cfg_notail["takl_lambda"] > 0,
        f"lambda_tail {cfg_notail['takl_lambda']:.5f}")

json.dump({"candidate": A.candidate, "config": CFG, "all_pass": bool(ok_all),
           "n_pass": sum(v["pass"] for v in res.values()), "n_total": len(res),
           "tests": res}, open(f"{OUT}/TEST_RESULTS_{A.candidate}.json", "w"), indent=2)
n = sum(v["pass"] for v in res.values())
print(f"\n  {A.candidate}  {n}/{len(res)} PASS" + ("" if ok_all else "  ★ 학습 금지"))
sys.exit(0 if ok_all else 1)
