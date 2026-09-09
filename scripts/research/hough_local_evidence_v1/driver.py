"""H0 -> H1 -> 판정.  학습에서 끝내지 않는다.  판정 기준은 여기 하드코딩돼 있다."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/hough_local_evidence_v1"
PY_ENV = "/home/minjae/anaconda3/envs/pallet-pose/bin/python"
RUN = ROOT / "scripts/research/hough_local_evidence_v1/run_arms.py"
CROSS = ["v4_split_base", "aug_squash_v2", "paper_4pallet_mask_v1"]
GATE = {"h0_dev_lo": 3.0, "h0_dev_hi": 6.0, "strong": 0.30, "weak": 0.10}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT / "DRIVER.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    for arm in ("H0", "H1"):
        log(f"=== TRAIN {arm} ===")
        r = subprocess.run([PY_ENV, str(RUN), "--arm", arm], cwd=ROOT)
        if r.returncode != 0:
            log(f"{arm} FAILED rc={r.returncode}"); return 1
        log(f"{arm} done")

    h = {a: json.loads((OUT / f"{a}_HISTORY.json").read_text())["history"] for a in ("H0", "H1")}
    last = str(max(int(k) for k in h["H0"]))
    h0, h1 = h["H0"][last], h["H1"][last]

    dev0 = h0["OWN_DEV"]["angle_median"]
    gate1 = GATE["h0_dev_lo"] <= dev0 <= GATE["h0_dev_hi"]
    log(f"1차 관문  H0 own-dev angle median {dev0:.3f}  "
        f"(기대 {GATE['h0_dev_lo']}~{GATE['h0_dev_hi']})  -> {'PASS' if gate1 else 'FAIL'}")

    imp = {}
    for cs in CROSS:
        a, b = h0.get(cs, {}).get("angle_median"), h1.get(cs, {}).get("angle_median")
        if a and b:
            imp[cs] = (a - b) / a
            log(f"  {cs:24s} H0 {a:7.3f} -> H1 {b:7.3f}   {imp[cs]*100:+6.1f}%")
    mean_imp = sum(imp.values()) / len(imp) if imp else 0.0

    if not gate1:
        verdict = "PIPELINE_NOT_REPRODUCED"          # 가설 검증 아님
    elif mean_imp >= GATE["strong"]:
        verdict = "LOCAL_EVIDENCE_HELPS"
    elif mean_imp >= GATE["weak"]:
        verdict = "WEAK"
    else:
        verdict = "LOCAL_EVIDENCE_NOT_THE_CAUSE"

    res = {"schema_version": "hough_local_evidence_v1_result", "gate": GATE,
           "h0_own_dev_angle_median": dev0, "gate1_pass": gate1,
           "cross_improvement": imp, "cross_improvement_mean": mean_imp,
           "verdict": verdict,
           "scope_note": ("H1 은 raster head + CoarseRadon 을 함께 넣으므로 단일 변수가 "
                          "아니다 — '국소 증거 경로 통째로' 의 효과다."),
           "H0_last": h0, "H1_last": h1}
    (OUT / "RESULT.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    log("=" * 56)
    log(f"VERDICT = {verdict}   cross mean {mean_imp*100:+.1f}%")
    log("=" * 56)
    (OUT / "DONE.mark").write_text(verdict + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
