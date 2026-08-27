"""FP vs FH — 같은 lambda 에서 전 grid 비교 (선택이 아니라 민감도 분석).

synthetic GATE_DEV pose 평가가 GT 축배정 문제로 막혔으므로 lambda 를 고르지 않는다.
대신 모든 lambda 에서 FP 와 FH 를 나란히 놓고, FH 가 FP 를 이기는 lambda 가 있는지 본다.
real 에서 하나를 골라 보고하지 않는다 — 전 grid 를 그대로 싣는다.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ah_control as AC                                    # noqa: E402

ROOT, NS = AC.ROOT, AC.NS
OUT = {"lambda_grid": AC.LAMBDA_GRID,
       "why_no_selection": (
           "synthetic GATE_DEV pose 평가가 GT per-frame 축배정(perm_v4) 없이는 성립하지 "
           "않는다 — model corner i <-> projected_cuboid[7-perm_v4[i]] 로 실측 확인(잔차 "
           "0.00 px). perm_v4 는 GT 전용이라 추론에서 쓸 수 없다. 따라서 lambda 를 고르지 "
           "않고 전 grid 에서 FP/FH 를 matched 비교한다. real 튜닝 아님."),
       "seeds": {}}
features = AC.MS.lattice()[3]
leak = set(json.load(open(f"{AC.YQ}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
items = [it for it in json.load(open(AC.MANI))["items"] if it["frame_id"] not in leak]

for seed in (1, 2):
    model, ckpt = AC.load_model(seed)
    AC.log(f"seed{seed} predict")
    preds, meta = [], []
    for it in items:
        jp, ip = os.path.join(ROOT, it["label"]), os.path.join(ROOT, it["image"])
        if not (os.path.exists(jp) and os.path.exists(ip)):
            continue
        p = AC.predict(model, features, ip, json.load(open(jp)))
        if p is None:
            continue
        st = it.get("set", "?")
        preds.append(p)
        meta.append({"fid": it["frame_id"], "set": st,
                     "group": "OPEN40" if st in AC.OPEN_SETS else "NEW88"})
    AC.log(f"seed{seed} n={len(preds)}  lambda sweep (matched)")
    base = [AC.metrics(AC.pose_from(p, None, None, None)[0], p) for p in preds]
    ent = {"checkpoint": ckpt, "n": len(preds), "grid": {}}
    for g, idx in (("ALL128", range(len(preds))),
                   ("OPEN40", [i for i, m in enumerate(meta) if m["group"] == "OPEN40"]),
                   ("NEW88", [i for i, m in enumerate(meta) if m["group"] == "NEW88"])):
        idx = list(idx)
        ent["grid"][g] = {"F0": AC.agg([{f"F0_{k}": v for k, v in base[i].items()}
                                        for i in idx], "F0", len(idx))}
    for lam in AC.LAMBDA_GRID:
        arms = {}
        for tag, key in (("FP", "P"), ("FH", "H")):
            ms = [AC.metrics(AC.pose_from(p, p[f"th_{key}"], p[f"rh_{key}"], lam)[1], p)
                  for p in preds]
            arms[tag] = ms
        for g, idx in (("ALL128", range(len(preds))),
                       ("OPEN40", [i for i, m in enumerate(meta) if m["group"] == "OPEN40"]),
                       ("NEW88", [i for i, m in enumerate(meta) if m["group"] == "NEW88"])):
            idx = list(idx)
            cell = {}
            for tag in ("FP", "FH"):
                rows = [{f"{tag}_{k}": v for k, v in arms[tag][i].items()} for i in idx]
                cell[tag] = AC.agg(rows, tag, len(idx))
            dR = np.array([arms["FP"][i]["R"] - arms["FH"][i]["R"] for i in idx], float)
            dR = dR[np.isfinite(dR)]
            cell["FH_minus_FP"] = {
                "R_median_delta": float(np.median(dR)) if dR.size else None,
                "frac_FH_better": float((dR > 0).mean()) if dR.size else None,
                "d_R_median": cell["FP"]["R_median"] - cell["FH"]["R_median"],
                "d_5cm5": cell["FH"]["success_5cm5deg"] - cell["FP"]["success_5cm5deg"],
                "t_degrade": (cell["FH"]["t_median"] - cell["FP"]["t_median"])
                / max(cell["FP"]["t_median"], 1e-12)}
            ent["grid"][g][str(lam)] = cell
        AC.log(f"  lam {lam} done")
    OUT["seeds"][f"seed{seed}"] = ent
    del model
    torch.cuda.empty_cache()

# ---- verdict: 어떤 lambda 에서든 FH 가 FP 를 두 seed 일관되게 이기는가
wins = {}
for lam in AC.LAMBDA_GRID:
    ok = []
    for s in ("seed1", "seed2"):
        c = OUT["seeds"][s]["grid"]["ALL128"][str(lam)]["FH_minus_FP"]
        ok.append(bool(c["d_R_median"] > 0 and c["t_degrade"] <= 0.03 and c["d_5cm5"] >= 0.0))
    wins[str(lam)] = {"seed1": ok[0], "seed2": ok[1], "BOTH": all(ok)}
any_lambda = any(v["BOTH"] for v in wins.values())
OUT["matched_lambda_wins"] = wins
OUT["ANY_LAMBDA_WHERE_FH_BEATS_FP"] = any_lambda
OUT["VERDICT"] = ("HOUGH_ADDS_INFORMATION_BEYOND_REWEIGHTING" if any_lambda
                  else "HOUGH_INCREMENTAL_VALUE_NOT_ESTABLISHED")
json.dump(OUT, open(f"{NS}/LAMBDA_MATCHED_FP_VS_FH.json", "w"), indent=2, ensure_ascii=False)
AC.log(f"ANY_LAMBDA_WHERE_FH_BEATS_FP = {any_lambda}  -> {OUT['VERDICT']}")
