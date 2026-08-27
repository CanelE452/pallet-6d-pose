"""평가 프레임 정본 — manual GT 161장 전수 (봉인 포함).

사용자가 2026-08-20 에 봉인 해제를 명시 승인했다. FINAL_TEST 105 장은 이 비교
이후 소진되며 재봉인할 수 없다. 그 사실을 결과 JSON 에 박아 둔다.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "challenge"))
import data_paths as DP  # noqa: E402

SEALED = set(DP.FINAL_TEST)


def frames():
    """[(set_key, sealed, json_path, image_path, label)] — split=='eval' 만."""
    out = []
    for key, rel in DP.EVAL_CANONICAL.items():
        folder = os.path.join(ROOT, rel)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            jp = os.path.join(folder, name)
            payload = json.load(open(jp))
            objects = payload.get("objects") or []
            if not objects or objects[0].get("split") != "eval":
                continue
            for ext in (".png", ".jpg", ".jpeg"):
                ip = jp[:-5] + ext
                if os.path.exists(ip):
                    out.append((key, key in SEALED, jp, ip, payload))
                    break
    return out


if __name__ == "__main__":
    rows = frames()
    from collections import Counter
    counts = Counter(r[0] for r in rows)
    print(f"총 {len(rows)}장")
    for key, n in counts.items():
        print(f"  {key:16} {n:3d}  {'SEALED' if key in SEALED else 'open'}")
