"""real GT 의 keypoint 필드 계약을 전수 집계한다 (§3, §4).

읽기 전용.  새 학습·추론 없음.

세 가지를 센다.
  1. keypoint_annotations 를 쓰는 프레임 vs projected_cuboid 로 fallback 하는 프레임
  2. xy=None (= 좌표를 모르는 점) 이 몇 개나 있고
  3. 그 점이 prepare_yolo_pose 의 실제 변환을 거치면 최종 라벨에서 v 가 무엇이 되는가
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "challenge/yolo_pose_one_model/scripts"))
import prepare_yolo_pose as pyp  # noqa: E402

ROOTS = [
    "challenge/data/01_real/live_capture_gt",
    "challenge/data/01_real/eval_canonical",
    "challenge/data/01_real/manual_gt",
    "challenge/data/01_real/gt_v2_canonical",
    "challenge/data/01_real/augmented",
    "challenge/data/01_real/pseudo_gt",
]


def main() -> int:
    per_root = {}
    fallback_frames = []
    none_frames = []
    state_counts = Counter()
    entry_key_counts = Counter()

    for root_rel in ROOTS:
        root = REPO / root_rel
        if not root.exists():
            per_root[root_rel] = {"exists": False}
            continue
        n_total = n_ka = n_fallback = n_unusable = 0
        n_none_kp = 0
        n_frames_with_none = 0
        for jp in sorted(root.rglob("*.json")):
            if jp.name.startswith("_") or not jp.stem.isdigit():
                continue
            n_total += 1
            try:
                obj = json.load(open(jp, encoding="utf-8"))["objects"][0]
            except Exception:
                n_unusable += 1
                continue
            ann = obj.get("keypoint_annotations")
            if isinstance(ann, list) and len(ann) >= 9:
                n_ka += 1
                k_none = 0
                for e in ann[:9]:
                    if not isinstance(e, dict):
                        continue
                    entry_key_counts.update(e.keys())
                    state_counts[(e.get("visibility"), e.get("in_frame"),
                                  e.get("source"), e.get("reason"),
                                  e.get("xy") is None)] += 1
                    if e.get("xy") is None:
                        k_none += 1
                if k_none:
                    n_frames_with_none += 1
                    n_none_kp += k_none
                    none_frames.append((str(jp.relative_to(REPO)), k_none))
            else:
                proj = obj.get("projected_cuboid")
                if proj and len(proj) >= 8:
                    n_fallback += 1
                    fallback_frames.append(str(jp.relative_to(REPO)))
                else:
                    n_unusable += 1
        per_root[root_rel] = {
            "exists": True,
            "N_total_real_json": n_total,
            "N_with_keypoint_annotations": n_ka,
            "N_fallback_to_projected_cuboid": n_fallback,
            "N_unusable": n_unusable,
            "N_frames_with_xy_none": n_frames_with_none,
            "N_keypoints_xy_none": n_none_kp,
        }

    out = {
        "per_root": per_root,
        "totals": {
            k: sum(v.get(k, 0) for v in per_root.values() if v.get("exists"))
            for k in ("N_total_real_json", "N_with_keypoint_annotations",
                      "N_fallback_to_projected_cuboid", "N_unusable",
                      "N_frames_with_xy_none", "N_keypoints_xy_none")
        },
        "entry_key_frequency": dict(entry_key_counts.most_common()),
        "state_combinations": [
            {"visibility": k[0], "in_frame": k[1], "source": k[2],
             "reason": k[3], "xy_is_none": k[4], "count": c}
            for k, c in state_counts.most_common()
        ],
        "fallback_frames": fallback_frames,
        "frames_with_xy_none": none_frames,
        "constants": {"PAD": pyp.PAD, "SENTINEL": pyp.SENTINEL},
    }
    dst = REPO / "data/pallet/results/next_accuracy_v2/KEYPOINT_CONTRACT_CENSUS.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    t = out["totals"]
    print(f"N_total_real_json                 {t['N_total_real_json']}")
    print(f"N_with_keypoint_annotations       {t['N_with_keypoint_annotations']}")
    print(f"N_fallback_to_projected_cuboid    {t['N_fallback_to_projected_cuboid']}")
    print(f"N_unusable                        {t['N_unusable']}")
    print(f"N_frames_with_xy_none             {t['N_frames_with_xy_none']}")
    print(f"N_keypoints_xy_none               {t['N_keypoints_xy_none']}")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
