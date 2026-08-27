"""smoke dataset + gradient routing + zero-weight parity.  HARD BLOCK 전 단계."""
from __future__ import annotations
import hashlib, json, os, sys, copy

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/runs_posecls_g38"
DS = f"{Y}/datasets/g38_generic_only"
SMOKE = f"{Y}/datasets/g38_smoke256"
INIT = f"{ROOT}/challenge/weights/pretrained_yolo/yolo26n-pose.pt"
os.makedirs(f"{NS}/tests", exist_ok=True)

from pallet_yolo_loss.posecls import (PoseAwareClsLoss26,                # noqa: E402
                                      PoseAwareClsLoss26_Lambda0, CALLS)
from ultralytics.utils.loss import PoseLoss26, E2ELoss                   # noqa: E402
from ultralytics.nn.tasks import PoseModel                               # noqa: E402


# ------------------------------------------------------------- smoke dataset
def build_smoke():
    if os.path.exists(f"{SMOKE}/data.yaml"):
        return json.load(open(f"{SMOKE}/_build.json"))
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{DS}/images/train"))
    key = lambda s: hashlib.sha1(f"G38_SMOKE256_V1|{s}".encode()).hexdigest()
    pick = sorted(stems, key=key)[:256]                 # 결정론적, random 미사용
    for s in ("images/train", "labels/train", "images/val", "labels/val"):
        os.makedirs(f"{SMOKE}/{s}", exist_ok=True)
    for split in ("train", "val"):
        for st in pick:
            for a, b in ((f"{DS}/images/train/{st}.png", f"{SMOKE}/images/{split}/{st}.png"),
                         (f"{DS}/labels/train/{st}.txt", f"{SMOKE}/labels/{split}/{st}.txt")):
                if not os.path.lexists(b):
                    os.symlink(a, b)
    open(f"{SMOKE}/data.yaml", "w").write(
        f"path: {SMOKE}\ntrain: images/train\nval: images/val\nnc: 1\n"
        "kpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\nnames:\n  0: pallet\n")
    info = {"n": len(pick), "rule": "sha1('G38_SMOKE256_V1|'+stem) 오름차순 앞 256 개",
            "sha256": hashlib.sha256("\n".join(sorted(pick)).encode()).hexdigest(),
            "source": os.path.relpath(DS, ROOT), "random_choice": False}
    json.dump(info, open(f"{SMOKE}/_build.json", "w"), indent=2, ensure_ascii=False)
    return info


def make_trainer(data, loss_cls):
    from ultralytics.models.yolo.pose import PoseTrainer
    orig = PoseModel.init_criterion
    if loss_cls is not None:
        PoseModel.init_criterion = lambda self: E2ELoss(self, loss_cls)
    tr = PoseTrainer(overrides=dict(
        task="pose", mode="train", model=INIT, data=data, epochs=1, batch=8, imgsz=640,
        optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=True, close_mosaic=10,
        warmup_epochs=3.0, patience=0, single_cls=True, mosaic=0.3, scale=0.25,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, fliplr=0.0, flipud=0.0, erasing=0.4,
        seed=42, deterministic=True, save_period=10, device=0, workers=0,
        project=f"{NS}/tests", name="_probe", exist_ok=True, resume=False,
        val=False, plots=False))
    tr._setup_train()
    return tr, orig


def one_batch(tr):
    for b in tr.train_loader:
        return tr.preprocess_batch(b)


def group(model):
    g = {"cls_head": [], "kpt_head": [], "sigma_head": [], "other": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = n.lower()
        if ".cv3." in low or "cv3." in low:
            g["cls_head"].append((n, p))
        elif "flow" in low or "sigma" in low:
            g["sigma_head"].append((n, p))
        elif ".cv4." in low or "kpt" in low or "pose" in low:
            g["kpt_head"].append((n, p))
        else:
            g["other"].append((n, p))
    return g


def gnorm(params):
    t = 0.0
    for _, p in params:
        if p.grad is not None:
            t += float((p.grad.detach() ** 2).sum())
    return t ** 0.5


def main():
    info = build_smoke()
    print("smoke dataset", info["n"], info["sha256"][:16], flush=True)

    # ---------------- GRADIENT ROUTING -------------------------------------
    CALLS["align"] = CALLS["qpose"] = 0
    tr, orig = make_trainer(f"{SMOKE}/data.yaml", PoseAwareClsLoss26)
    batch = one_batch(tr)
    model = tr.model
    crit = model.criterion if getattr(model, "criterion", None) else model.init_criterion()
    model.criterion = crit
    preds = model(batch["img"])
    # posealign 만 backward
    for c in (crit.one2many, crit.one2one):
        c._reset()
    parsed = crit.one2many.parse_output(preds)
    align_terms, per_branch = [], {}
    for name, c, key in (("one2many", crit.one2many, "one2many"),
                         ("one2one", crit.one2one, "one2one")):
        c._reset()
        c.loss(parsed[key], batch)            # 내부 텐서 채우기
        a = c._posealign()
        per_branch[name] = {"reached": a is not None,
                            "value": None if a is None else float(a.detach())}
        if a is not None:
            align_terms.append(a)
    model.zero_grad(set_to_none=True)
    if align_terms:
        sum(align_terms).backward()
    g = group(model)
    routing = {"branches": per_branch,
               "grad_norm": {k: gnorm(v) for k, v in g.items()},
               "param_counts": {k: len(v) for k, v in g.items()},
               "q_pose_detached": True,
               "CALLS": dict(CALLS)}
    routing["PASS"] = bool(routing["grad_norm"]["cls_head"] > 0
                           and routing["grad_norm"]["kpt_head"] == 0.0
                           and routing["grad_norm"]["sigma_head"] == 0.0
                           and per_branch["one2many"]["reached"]
                           and per_branch["one2one"]["reached"])
    json.dump(routing, open(f"{NS}/tests/GRADIENT_ROUTING.json", "w"),
              indent=2, ensure_ascii=False)
    print("GRADIENT_ROUTING PASS =", routing["PASS"], routing["grad_norm"], flush=True)
    PoseModel.init_criterion = orig
    del tr, model
    torch.cuda.empty_cache()

    # ---------------- ZERO WEIGHT PARITY ------------------------------------
    states = {}
    for tag, loss_cls in (("Y0", None), ("Y1_lambda0", PoseAwareClsLoss26_Lambda0)):
        torch.manual_seed(42); np.random.seed(42)
        tr, orig = make_trainer(f"{SMOKE}/data.yaml", loss_cls)
        model = tr.model
        crit = model.init_criterion(); model.criterion = crit
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=0.001, momentum=0.9)
        it = iter(tr.train_loader)
        batches = []
        for _ in range(4):
            batches.append(tr.preprocess_batch(next(it)))
        for step in range(20):
            b = batches[step % 4]
            loss, _ = crit(model(b["img"]), b)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
        states[tag] = {n: p.detach().float().cpu().clone()
                       for n, p in model.named_parameters()}
        PoseModel.init_criterion = orig
        del tr, model, crit
        torch.cuda.empty_cache()
    diffs = {n: float((states["Y0"][n] - states["Y1_lambda0"][n]).abs().max())
             for n in states["Y0"]}
    mx = max(diffs.values())
    parity = {"steps": 20, "batches_replayed": 4, "max_param_diff": mx,
              "n_params": len(diffs), "exact_zero": mx == 0.0,
              "tolerance": 1e-6, "PASS": bool(mx <= 1e-6),
              "worst": sorted(diffs.items(), key=lambda kv: -kv[1])[:5]}
    json.dump(parity, open(f"{NS}/tests/ZERO_WEIGHT_PARITY.json", "w"),
              indent=2, ensure_ascii=False)
    print("ZERO_WEIGHT_PARITY PASS =", parity["PASS"], "max diff", mx, flush=True)

    json.dump({"smoke_dataset": info,
               "GRADIENT_ROUTING_PASS": routing["PASS"],
               "ZERO_WEIGHT_PARITY_PASS": parity["PASS"]},
              open(f"{NS}/tests/TESTS_SUMMARY.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
