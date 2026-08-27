"""ASC — 15 테스트. 15/15 PASS 전 학습 금지. A1 하네스 재사용, 새 프레임워크 없음."""
from __future__ import annotations
import json, os, sys, copy
import numpy as np, torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R = f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss"
DATA = f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml"
INIT = f"{ROOT}/challenge/yolo_pose_one_model/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"

res = {}
ok_all = True


def rec(name, ok, detail=""):
    global ok_all
    res[name] = {"pass": bool(ok), "detail": detail}
    ok_all &= bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44} {detail}", flush=True)


from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
import pallet_yolo_loss.symmetry as SY
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss, asc_beta
from pallet_yolo_loss.loss import PSPCPoseLoss26

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
P180 = tuple(json.load(open(f"{R}/FIXED_P180_PERMUTATIONS.json"))["P180"])
SMAP = f"{R}/STEM_ASSET_MAP.json"
smap = json.load(open(SMAP))
SYM = ["scene.usd", "scene_1.usd"]
GLB = ["woodpallet_block_jtoastie_ccby.glb", "eur_pallet_bk_cc0.glb"]

# ---- T8 / T9  순열 성질 ----------------------------------------------------
rec("T8 P180 bijection+involution",
    sorted(P180) == list(range(8)) and all(P180[P180[i]] == i for i in range(8)), f"{P180}")
g = torch.arange(27.).view(1, 9, 3)
perm = list(P180) + [8]
rec("T9 centroid index 8 fixed",
    perm[8] == 8 and torch.equal(g[:, perm, :][:, 8], g[:, 8]), "centroid 불변")

# ---- T4 / T5  schedule -----------------------------------------------------
want = {0: 1.0, 10: 1.0, 19: 1.0, 20: 1.0, 25: 0.5, 29: 0.1, 30: 0.0, 45: 0.0, 59: 0.0}
got = {e: round(asc_beta(e, 20, 30), 6) for e in want}
rec("T4 beta schedule 값", all(abs(got[e] - v) < 1e-9 for e, v in want.items()), f"{got}")
rec("T5 boundary 19/20/29/30",
    got[19] == 1.0 and got[20] == 1.0 and abs(got[29] - 0.1) < 1e-9 and got[30] == 0.0,
    "19=1.0 20=1.0 29=0.1 30=0.0")

# ---- 배치 -------------------------------------------------------------------
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
dl = build_dataloader(ds, 8, 2, shuffle=False, rank=-1)
batch = next(iter(dl))
batch["img"] = batch["img"].to(DEV).float() / 255
for k in ("keypoints", "bboxes", "cls", "batch_idx"):
    batch[k] = batch[k].to(DEV)


def fresh():
    model.zero_grad(set_to_none=True)
    return model(batch["img"])


def cfgfile(name, **kw):
    p = f"{R}/_asc_{name}.json"
    base = dict(enabled=True, mode="exact_min", lambda_role=0.0, margin=0.0,
                p180=list(P180), centroid_index=8, sym_assets=SYM, asym_assets=[],
                stem_asset_map=SMAP, role_ramp=[5, 20],
                asc_enabled=True, asc_full_end=20, asc_ramp_end=30)
    base.update(kw)
    json.dump(base, open(p, "w"))
    return p


def make(kind, cfg=None):
    os.environ["A1_CONFIG"] = cfg or ""
    os.environ["PSPC_CONFIG"] = ""
    torch.manual_seed(0)
    return E2ELoss(model, kind) if getattr(model, "end2end", False) else kind(model)


def run(crit, grad=False):
    torch.manual_seed(0)
    p = fresh()
    b = copy.deepcopy(batch)
    l, items = crit(p, b)
    if grad:
        l.sum().backward()
        return float(l.sum()), items.detach().cpu().numpy(), \
            [q.grad.detach().clone() for q in model.parameters() if q.grad is not None]
    return float(l.sum()), items.detach().cpu().numpy(), None


ASC = cfgfile("asc")
A1 = cfgfile("a1_equiv", asc_enabled=False)

std = make(PoseLoss26)
asc = make(A1SymmetryPoseLoss, ASC)
a1 = make(A1SymmetryPoseLoss, A1)

# ---- T1 / T2  beta=0 -> A0 parity ------------------------------------------
SY.CURRENT_EPOCH["e"] = 45           # beta = 0
ls, is_, gs = run(std, True)
lc, ic, gc = run(asc, True)
rec("T1 beta=0 == A0 total-loss parity", abs(ls - lc) < 1e-6 and np.abs(is_ - ic).max() < 1e-6,
    f"total {abs(ls-lc):.3e}  items {np.abs(is_-ic).max():.3e}")
# 잡음바닥을 1회로 재면 불안정하다 (실측 3.8e-5 ~ 1.9e-4). 3회 반복의 최대를 쓴다.
noise = 0.0
for _ in range(3):
    _, _, gn = run(std, True)
    noise = max(noise, max((a - b).abs().max().item() for a, b in zip(gs, gn)))
dg = max((a - b).abs().max().item() for a, b in zip(gs, gc))
rec("T2 beta=0 gradient parity", dg <= max(noise, 1e-6) * 1.5,
    f"dgrad {dg:.3e}  noise {noise:.3e}")

# ---- T3  beta=1 -> A1_USD_ONLY parity --------------------------------------
SY.CURRENT_EPOCH["e"] = 5            # beta = 1
la, ia, _ = run(a1)
lb, ib, _ = run(asc)
# A1 은 pos.mean() (2단계), ASC 는 base+보정 (부모 단일 + 보정).  수학적으로 같고
# fp32 반올림만 다르다.  A0 parity(T1/T2)를 비트단위로 지키기 위한 의도적 선택.
rec("T3 beta=1 == A1_USD_ONLY parity", abs(la - lb) < 1e-4 and np.abs(ia - ib).max() < 1e-4,
    f"total {abs(la-lb):.3e}  items {np.abs(ia-ib).max():.3e}  (fp32 반올림 허용 1e-4)")

# ---- T6 / T7  asset 별 적용 --------------------------------------------------
inner = getattr(asc, "one2many", asc)
inner._batch = batch
stems = [os.path.splitext(os.path.basename(f))[0] for f in batch["im_file"]]
mk = torch.zeros(len(stems), 3, dtype=torch.bool, device=DEV)
mk[:, 0] = True
cls = inner._instance_class(mk).tolist()
rec("T6 USD asset 만 symmetry(class=1)",
    all((c == 1) == (smap[s] in SYM) for c, s in zip(cls, stems)),
    f"USD {sum(1 for s in stems if smap[s] in SYM)} / {len(stems)}")
rec("T7 GLB asset identity-only(class=0)",
    all((c == 0) == (smap[s] in GLB) for c, s in zip(cls, stems)),
    f"GLB {sum(1 for s in stems if smap[s] in GLB)} / {len(stems)}")

# ---- T10 / T11  E2E 두 경로 --------------------------------------------------
e2e = getattr(model, "end2end", False)
rec("T10 one2many 적용", (not e2e) or isinstance(asc.one2many, A1SymmetryPoseLoss),
    type(getattr(asc, "one2many", asc)).__name__)
rec("T11 one2one 적용", (not e2e) or isinstance(asc.one2one, A1SymmetryPoseLoss),
    type(getattr(asc, "one2one", asc)).__name__)

# ---- T12 / T13  PC / role 호출 0 ---------------------------------------------
calls = {"n": 0}
_pc = PSPCPoseLoss26.projective_loss


def spy(self, *a, **k):
    calls["n"] += 1
    return _pc(self, *a, **k)


PSPCPoseLoss26.projective_loss = spy
SY.ROLE_CALLS["n"] = 0
for e in (0, 25, 45):
    SY.CURRENT_EPOCH["e"] = e
    run(asc)
PSPCPoseLoss26.projective_loss = _pc
rec("T12 PC loss 호출 0", calls["n"] == 0 and inner.pspc.enabled is False,
    f"calls {calls['n']}  pspc.enabled {inner.pspc.enabled}")
rec("T13 role-margin 호출 0", SY.ROLE_CALLS["n"] == 0 and inner.a1.lambda_role == 0.0,
    f"role calls {SY.ROLE_CALLS['n']}  lambda_role {inner.a1.lambda_role}")

# ---- T14  forward/backward finite (전 구간) ----------------------------------
fin = True
det = []
for e in (0, 19, 20, 25, 29, 30, 59):
    SY.CURRENT_EPOCH["e"] = e
    l, _, gg = run(asc, True)
    gf = all(torch.isfinite(x).all().item() for x in gg)
    fin &= np.isfinite(l) and gf
    det.append(f"{e}:{l:.1f}")
rec("T14 forward/backward finite", fin, " ".join(det))

# ---- T15  checkpoint reload --------------------------------------------------
tmp = f"{R}/_asc_ckpt_test.pt"
torch.save({"model": model}, tmp)
try:
    rl = torch.load(tmp, weights_only=False)["model"]
    okr = sum(p.numel() for p in rl.parameters()) == sum(p.numel() for p in model.parameters())
except Exception as ex:
    okr = False
    det = str(ex)[:60]
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
rec("T15 checkpoint reload", okr, "params 일치" if okr else "reload 실패")

json.dump({"all_pass": bool(ok_all), "n_pass": sum(v["pass"] for v in res.values()),
           "n_total": len(res), "tests": res},
          open(f"{R}/ASC_TEST_RESULTS.json", "w"), indent=2)
print(f"\n  {sum(v['pass'] for v in res.values())}/{len(res)} PASS"
      + ("" if ok_all else "  ★ 학습 금지"))
sys.exit(0 if ok_all else 1)
