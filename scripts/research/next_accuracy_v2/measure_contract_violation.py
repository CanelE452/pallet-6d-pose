"""현재 load_kps 가 만드는 라벨과 스키마 계약이 요구하는 라벨을 프레임 단위로 비교한다 (§4).

계약 정본은 ``scripts/annotate/real_gt_v2_schema.keypoint_annotations_to_ultralytics``:
visibility==0 또는 xy is None 이면 학습 타깃은 [0,0,0] 이어야 한다.
현재 ``prepare_yolo_pose.load_kps`` 는 visibility 를 읽지 않는다.

읽기 전용.  새 학습·추론 없음.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "challenge/yolo_pose_one_model/scripts"))
sys.path.insert(0, str(REPO / "scripts/annotate"))
import prepare_yolo_pose as pyp  # noqa: E402

ROOTS = ["challenge/data/01_real/live_capture_gt",
         "challenge/data/01_real/gt_v2_canonical"]


def main() -> int:
    tot = Counter()
    per_source_supervised = Counter()
    bad_frames = []
    bbox_frames = []

    for root_rel in ROOTS:
        root = REPO / root_rel
        if not root.exists():
            continue
        for jp in sorted(root.rglob("*.json")):
            if jp.name.startswith("_") or not jp.stem.isdigit():
                continue
            doc = json.load(open(jp, encoding="utf-8"))
            obj = doc["objects"][0]
            ann = obj.get("keypoint_annotations")
            if not (isinstance(ann, list) and len(ann) >= 9):
                continue
            cam = doc.get("camera_data") or {}
            iw, ih = int(cam.get("width", 640)), int(cam.get("height", 480))
            pw, ph = iw + 2 * pyp.PAD, ih + 2 * pyp.PAD
            tot["frames"] += 1

            kps = pyp.load_kps(jp)
            padded = [(x + pyp.PAD, y + pyp.PAD) for x, y in kps]
            v_now = [2 if (0 <= x < pw and 0 <= y < ph) else 0 for x, y in padded]
            v_contract, wrong_idx = [], []
            for i, e in enumerate(ann[:9]):
                vis = int(e.get("visibility", 0))
                unknown = (vis == 0) or (e.get("xy") is None)
                vc = 0 if unknown else v_now[i]
                v_contract.append(vc)
                tot["keypoints"] += 1
                if unknown:
                    tot["kp_unknown_or_vis0"] += 1
                if vc != v_now[i]:
                    wrong_idx.append(i)
                    tot["kp_wrongly_supervised"] += 1
                if v_now[i] == 2:
                    per_source_supervised[e.get("source")] += 1

            if wrong_idx:
                # bbox 가 실제로 달라지는가?
                now = [p for p, v in zip(padded, v_now) if v == 2]
                con = [p for p, v in zip(padded, v_contract) if v == 2]
                d = 0.0
                if now and con:
                    bn = (min(p[0] for p in now), min(p[1] for p in now),
                          max(p[0] for p in now), max(p[1] for p in now))
                    bc = (min(p[0] for p in con), min(p[1] for p in con),
                          max(p[0] for p in con), max(p[1] for p in con))
                    d = max(abs(a - b) for a, b in zip(bn, bc))
                tot["frames_with_wrong_kp"] += 1
                if d > 1.0:
                    tot["frames_with_shifted_bbox"] += 1
                    bbox_frames.append({"frame": str(jp.relative_to(REPO)),
                                        "bbox_shift_px": round(d, 1)})
                bad_frames.append({"frame": str(jp.relative_to(REPO)),
                                   "wrong_indices": wrong_idx,
                                   "bbox_shift_px": round(d, 1)})
                if not con:
                    tot["frames_that_would_become_empty"] += 1

    out = {"totals": dict(tot),
           "supervised_keypoints_by_source_now": dict(per_source_supervised),
           "frames_with_shifted_bbox": sorted(
               bbox_frames, key=lambda r: -r["bbox_shift_px"])[:50],
           "n_bad_frames": len(bad_frames),
           "bad_frames": bad_frames[:200]}
    dst = REPO / "data/pallet/results/next_accuracy_v2/CONTRACT_VIOLATION_MEASURE.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("현재 라벨 vs 스키마 계약")
    for k, v in tot.items():
        print(f"  {k:<34}{v:>8}")
    print("\n지금 v=2 로 감독되는 keypoint 의 source 분포")
    for k, v in per_source_supervised.most_common():
        print(f"  {str(k):<24}{v:>8}")
    print(f"\nbbox 가 1px 넘게 달라지는 프레임 상위 5")
    for r in out["frames_with_shifted_bbox"][:5]:
        print(f"  {r['bbox_shift_px']:>8.1f} px  {r['frame']}")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
