"""PHASE 10 — negative FT 에 들어갈 데이터가 target 을 포함하는지 감사.

positive 쪽 감사(TARGET_ASSET_EXCLUSION_AUDIT)와 같은 원칙: 파일명이 아니라
라벨의 asset 식별자와 기하로 본다.  negative 는 팔레트가 없어야 하므로
`object_present` 계약도 함께 검사한다.
"""
from __future__ import annotations
import json, pathlib, sys, collections
ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
TARGET_TOKENS = ("pallet_full", "palletobj", "scan_cleanup", "v1", "v2")


def audit(folder):
    folder = pathlib.Path(folder)
    labels = sorted(folder.rglob("*_label.json")) or sorted(folder.rglob("*.json"))
    bad, assets = [], collections.Counter()
    n_present = 0
    for p in labels:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("object_present"):
            n_present += 1
            bad.append(f"{p.name}: object_present=True (negative 여야 한다)")
        for o in (d.get("objects") or []):
            a = str(o.get("source_asset"))
            assets[a] += 1
            if any(t in a.lower() for t in TARGET_TOKENS):
                bad.append(f"{p.name}: target 계열 asset {a}")
    return {"folder": str(folder), "n_labels": len(labels),
            "n_object_present": n_present, "assets": dict(assets),
            "violations": bad[:50], "n_violations": len(bad),
            "VERDICT": "PASS" if not bad else "HARD_BLOCK"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: audit_negative_target_leakage.py <negative folder>")
    print(json.dumps(audit(sys.argv[1]), indent=1, ensure_ascii=False))
