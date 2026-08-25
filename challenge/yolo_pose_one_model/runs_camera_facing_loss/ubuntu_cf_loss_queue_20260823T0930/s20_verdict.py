"""S20 support probe — B10 대비 판정. gate 는 학습 전 freeze 된 것 그대로."""
import collections, json, os, subprocess, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
NOTIFY = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
LEAK = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])

GATE = {"night_top1_cbox_frames": 3, "night_median_rel": 0.15, "night_p90_rel": 0.15,
        "night_cand_rel": 0.20, "night_wrong_pp": 0.15, "night_margin_abs": 0.15,
        "need_hits": 2,
        "guards": {"all_cbox_pp": -0.03, "all_median_rel": -0.10, "night_any_cbox_frames": -2}}
json.dump(GATE, open(f"{Q}/S20_GATE_PREREG.json", "w"), indent=2)


def real(tag, w):
    o = f"{Q}/REAL_{tag}.json"
    if not os.path.exists(o) and os.path.exists(w):
        subprocess.run([sys.executable, f"{Q}/cf_real_eval.py", "--weights", w, "--tag", tag],
                       capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


def nightc(tag, w):
    o = f"{Q}/NIGHT_CAND_{tag}.json"
    if not os.path.exists(o) and os.path.exists(w):
        subprocess.run([sys.executable, f"{Q}/night_cand_one.py", "--weights", w, "--tag", tag],
                       capture_output=True, text=True)
    return json.load(open(o)) if os.path.exists(o) else None


ARMS = [("B10", f"{CFR}/ROOTFAST_B10_30EP_SEED42/weights/last.pt"),
        ("S20", f"{CFR}/ROOTFAST_S20_30EP_SEED42/weights/last.pt"),
        ("T10", f"{CFR}/ROOTFAST_T10_30EP_SEED42/weights/last.pt")]
R, NC = {}, {}
for a, w in ARMS:
    R[a] = real(f"RF_{a}", w)
    NC[a] = nightc(f"RF_{a}", w)


def agg(rs):
    cb = [r for r in rs if r.get("correct_box")]
    e = np.concatenate([r["err"] for r in cb]) if cb else np.array([])
    return {"n": len(rs), "cbox": len(cb) / max(len(rs), 1), "n_cbox": len(cb),
            "med": float(np.median(e)) if e.size else None,
            "p90": float(np.percentile(e, 90)) if e.size else None,
            "gross": float((e > 20).mean()) if e.size else None}


TAB = {}
for a, d in R.items():
    if not d:
        continue
    pf = [r for r in d["per_frame"] if r["frame"] not in LEAK]
    TAB[a] = {"ALL": agg(pf), "DAY": agg([r for r in pf if r["domain"] == "DAY"]),
              "NIGHT": agg([r for r in pf if r["domain"] == "NIGHT"])}

b, s = NC.get("B10"), NC.get("S20")
tb, ts = TAB.get("B10"), TAB.get("S20")
hits, guards = {}, {}
if b and s and tb and ts:
    N = b["n"]
    rel = lambda x, y: (x - y) / max(abs(x), 1e-12)
    hits = {
        "top1_cbox_+3f": (s["top1_cbox"] - b["top1_cbox"]) * N >= GATE["night_top1_cbox_frames"],
        "median_-15%": (rel(tb["NIGHT"]["med"], ts["NIGHT"]["med"]) >= GATE["night_median_rel"]
                        if (tb["NIGHT"]["med"] and ts["NIGHT"]["med"]) else False),
        "p90_-15%": (rel(tb["NIGHT"]["p90"], ts["NIGHT"]["p90"]) >= GATE["night_p90_rel"]
                     if (tb["NIGHT"]["p90"] and ts["NIGHT"]["p90"]) else False),
        "cand_-20%": rel(b["cand_per_frame"], s["cand_per_frame"]) >= GATE["night_cand_rel"],
        "wrong_-15pp": (b["wrong_present_frac"] - s["wrong_present_frac"]) >= GATE["night_wrong_pp"],
        "margin_+0.15": (s["margin_median"] - b["margin_median"]) >= GATE["night_margin_abs"]}
    guards = {
        "all_cbox": (ts["ALL"]["cbox"] - tb["ALL"]["cbox"]) >= GATE["guards"]["all_cbox_pp"],
        "all_median": (rel(tb["ALL"]["med"], ts["ALL"]["med"]) >= GATE["guards"]["all_median_rel"]
                       if (tb["ALL"]["med"] and ts["ALL"]["med"]) else True),
        "night_any_cbox": (s["any_cbox"] - b["any_cbox"]) * N >= GATE["guards"]["night_any_cbox_frames"]}
nhit = sum(1 for v in hits.values() if v)
SIG = "POSITIVE" if (nhit >= GATE["need_hits"] and all(guards.values())) else \
      ("HARM" if guards and not all(guards.values()) else "NULL")
RA = "EXPAND_TARGETED_SUPPORT_ONLY" if SIG == "POSITIVE" else "CLOSE_RENDER_MOVE_TO_ADAPTATION"

IMP = json.load(open(f"{Q}/UBUNTU_SUPPORT_PROBE_IMPORT_AUDIT.json"))
MAN = json.load(open(f"{Q}/S20_SUPPORT20_MANIFEST.json"))
out = {"gate": GATE, "hits": hits, "n_hits": nhit, "guards": guards,
       "SUPPORT_PROBE_SIGNAL": SIG,
       "GENERIC_SUPPORT_CAUSAL_SIGNAL": SIG == "POSITIVE",
       "RENDER_ACTION": RA, "table": TAB, "night_candidate": NC,
       "import": IMP, "manifest": MAN,
       "★caveat": ("T10 은 target-specific diagnostic — paper winner 로 쓰지 않는다. "
                   "NIGHT n=28, seed 1개, 30ep screen. support 는 "
                   "THIN_ENRICHED_GENERIC_SUPPORT (strict thin 아님, 초과 12.85%).")}
json.dump(out, open(f"{Q}/S20_SUPPORT_PROBE_RESULT.json", "w"), indent=2, ensure_ascii=False)

L = ["# S20 SUPPORT PROBE — Render 최종 판정 (30ep, effective 9,704)", "", "```",
     f"{'':16} {'B10':>10} {'S20':>10} {'T10':>10}", "-" * 50]
for sc in ("ALL", "DAY", "NIGHT"):
    for k, lab in (("cbox", "cbox"), ("med", "median"), ("p90", "p90"), ("gross", "gross20")):
        row = []
        for a in ("B10", "S20", "T10"):
            v = TAB.get(a, {}).get(sc, {}).get(k)
            row.append("       n/a" if v is None else
                       (f"{v:10.3f}" if k in ("cbox", "gross") else f"{v:10.2f}"))
        L.append(f"{sc+' '+lab:16} " + " ".join(row))
    L.append("")
L += ["```", "", "## NIGHT candidate", "```",
      f"{'':16} {'B10':>10} {'S20':>10} {'T10':>10}", "-" * 50]
for k, lab in (("any_cbox", "any-cbox"), ("top1_cbox", "top1-cbox"),
               ("cand_per_frame", "cand/frame"), ("wrong_present_frac", "wrong%"),
               ("margin_median", "margin")):
    row = []
    for a in ("B10", "S20", "T10"):
        v = (NC.get(a) or {}).get(k)
        row.append("       n/a" if v is None else f"{v:10.3f}")
    L.append(f"{lab:16} " + " ".join(row))
L += ["```", "", "## correct candidate rank (night)", "```"]
for a in ("B10", "S20", "T10"):
    if NC.get(a):
        L.append(f"{a:6} {NC[a].get('rank_hist')}")
L += ["```", "", f"hits {nhit}/6 → {hits}", f"guards {guards}", "",
      f"**SUPPORT_PROBE_SIGNAL = {SIG}**",
      f"**GENERIC_SUPPORT_CAUSAL_SIGNAL = {SIG == 'POSITIVE'}**",
      f"**RENDER_ACTION = {RA}**", "",
      f"support {MAN['support_N']} + broad {MAN['broad_N']} = {MAN['total_declared']}, target 0",
      f"r3d_rendered median {IMP.get('r3d_rendered',{}).get('median')}, "
      f"strict-thin(<= {IMP.get('frozen_thin_threshold')}) "
      f"{100*(1-IMP.get('over_thin_threshold',{}).get('frac',0)):.1f}%", "",
      "★ T10 은 target-specific diagnostic reference — paper winner 아님. NIGHT n=28, seed 1."]
txt = "\n".join(L)
open(f"{Q}/S20_SUPPORT_PROBE_RESULT.md", "w").write(txt)
print(txt)
try:
    subprocess.run([NOTIFY, txt[:1800]], timeout=60)
except Exception:
    pass
