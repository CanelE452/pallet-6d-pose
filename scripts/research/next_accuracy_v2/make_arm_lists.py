"""§13 arm 프레임 목록을 만든다 — 앙각만 다르고 나머지는 같게.

METHOD_LOCK.json stage_2 대로 handheld 2 세션 안에서만 뽑는다.
세션별 min(L, M) 이라 두 arm 의 **세션 구성이 글자 그대로 같다**.

seed 는 dataloader 에 도달하지 않아 복제가 되지 않는다(텐서 비트 동일 확인).
그래서 복제는 **membership 이 실제로 다른 draw** 로 만든다.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/pallet/results/next_accuracy_v2/arms"
SESSIONS = {"capture_20260902_kimjihoon": None, "capture_20260902": None}
N_DRAWS = 3


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((REPO / "data/pallet/results/next_accuracy_v2/GT_PARTITION.json")
                      .read_text(encoding="utf-8"))
    pool = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["split_role"] != "TRAIN" or r["session"] not in SESSIONS:
            continue
        band = ("L" if r["elev_bin"] in ("0-3", "3-8")
                else ("M" if r["elev_bin"] == "8-15" else None))
        if band:
            pool[r["session"]][band].append(r["frame_id"])

    quota = {s: min(len(pool[s]["L"]), len(pool[s]["M"])) for s in pool}
    total = sum(quota.values())
    print("세션별 할당 (min(L, M)):")
    for s, q in sorted(quota.items()):
        print(f"  {s:<32} L {len(pool[s]['L']):>4}  M {len(pool[s]['M']):>4}  -> {q}")
    print(f"  arm 크기 = {total}")

    meta = {"arm_size": total, "quota": quota, "n_draws": N_DRAWS,
            "sessions": sorted(pool), "files": []}
    for d in range(N_DRAWS):
        rng = np.random.default_rng(1000 + d)
        for band in ("L", "M"):
            picked = []
            for s in sorted(pool):
                ids = sorted(pool[s][band])
                picked += [ids[i] for i in rng.choice(len(ids), quota[s], replace=False)]
            f = OUT / f"arm_{band}_d{d}.txt"
            f.write_text("\n".join(sorted(picked)) + "\n", encoding="utf-8")
            meta["files"].append({"arm": band, "draw": d,
                                  "file": str(f.relative_to(REPO)), "n": len(picked)})
            print(f"  arm_{band}_d{d}: {len(picked)}")

    # draw 끼리 membership 이 실제로 다른가 — seed 함정의 재발 방지
    for band in ("L", "M"):
        sets = [set((OUT / f"arm_{band}_d{d}.txt").read_text().split())
                for d in range(N_DRAWS)]
        jac = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
               for i in range(N_DRAWS) for j in range(i + 1, N_DRAWS)]
        meta[f"{band}_draw_jaccard"] = [round(x, 3) for x in jac]
        print(f"  {band} draw 간 Jaccard {[round(x,3) for x in jac]}"
              f"  {'★동일 — 복제가 아니다' if max(jac) > 0.999 else ''}")

    (OUT / "ARMS.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"\nwrote {(OUT / 'ARMS.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
