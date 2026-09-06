"""GT 를 지표별 적격성으로 나눈다 (§8 · §9 · §32).

프레임을 통째로 버리지 않는다.  대신 **지표마다 따로** 적격성을 매긴다.

  2D keypoint   해당 점이 사람이 찍은 것(source == manual_click)일 때만 1급 증거.
                pnp_projected / centroid_auto 는 저장된 pose 를 재투영한 값이라
                모델을 그 값에 맞추는 것은 순환이다.  버리지 않고 층을 나눠 보고한다.
  axis / yaw    canonical_pose_candidates 가 1개로 좁혀졌을 때만 적격.
  full 6D       위 둘이 모두 적격일 때만.

읽기 전용.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SEAL = REPO / "data/pallet/results/accuracy_root_cause_v1/next_experiment"
TABLE = REPO / "data/pallet/results/next_accuracy_v2/LIVE_GT_FRAME_TABLE.json"
GT_ROOT = REPO / "challenge/data/01_real/live_capture_gt"


def main() -> int:
    rows = json.load(open(TABLE, encoding="utf-8"))
    train = {l.strip() for l in open(SEAL / "HOLDOUT_TRAIN_FRAMES.txt", encoding="utf-8") if l.strip()}
    held = {l.strip() for l in open(SEAL / "HOLDOUT_HELD_OUT_FRAMES.txt", encoding="utf-8") if l.strip()}

    out, agg = [], Counter()
    for r in rows:
        fid = f"{r['session']}_manual_gt/{r['frame']}"
        role = "TRAIN" if fid in train else ("HELD_OUT" if fid in held else "UNUSED")

        # 2D — 프레임 등급은 클릭 증거 수로만 매긴다
        n_click = r["n_manual_click"]
        if r["n_xy_none"]:
            kp2d = "GT_SUSPECT"
        elif n_click >= 6:
            kp2d = "GT_STRONG"
        elif n_click >= 4:
            kp2d = "GT_PARTIAL"
        else:
            kp2d = "GT_SUSPECT"

        axis = ("GT_STRONG" if r["n_pose_candidates"] == 1
                else "GT_AMBIGUOUS")
        six = "GT_STRONG" if (kp2d == "GT_STRONG" and axis == "GT_STRONG") else "GT_AMBIGUOUS"
        if kp2d == "GT_SUSPECT":
            six = "GT_SUSPECT"

        out.append({**{k: r[k] for k in
                       ("session", "group", "frame", "elev_bin", "elev_deg",
                        "n_manual_click", "n_pnp_projected", "n_centroid_auto",
                        "n_xy_none", "reproj_error_px", "n_pose_candidates")},
                    "frame_id": fid, "split_role": role,
                    "eligibility_2d": kp2d, "eligibility_axis": axis,
                    "eligibility_6d": six})
        agg[(role, kp2d)] += 1
        agg[("_axis", axis)] += 1
        agg[("_6d", six)] += 1
        agg[("_kp", "manual_click")] += r["n_manual_click"]
        agg[("_kp", "pnp_projected")] += r["n_pnp_projected"]
        agg[("_kp", "centroid_auto")] += r["n_centroid_auto"]
        agg[("_kp", "unknown_or_none")] += r["n_unknown"] if "n_unknown" in r else 0

    dst = REPO / "data/pallet/results/next_accuracy_v2/GT_PARTITION.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print("=== 2D keypoint 적격성 (프레임) ===")
    print(f"{'split':<10}{'GT_STRONG':>11}{'GT_PARTIAL':>12}{'GT_SUSPECT':>12}{'계':>7}")
    for role in ("TRAIN", "HELD_OUT", "UNUSED"):
        s = [agg[(role, k)] for k in ("GT_STRONG", "GT_PARTIAL", "GT_SUSPECT")]
        if sum(s):
            print(f"{role:<10}{s[0]:>11}{s[1]:>12}{s[2]:>12}{sum(s):>7}")
    print("\n=== axis / 6D 적격성 (전체 851) ===")
    for tag in ("_axis", "_6d"):
        d = {k[1]: v for k, v in agg.items() if k[0] == tag}
        print(f"  {tag[1:]:<6}{d}")
    print("\n=== keypoint 출처 (8,919점) ===")
    tot = sum(v for k, v in agg.items() if k[0] == "_kp")
    for k in ("manual_click", "pnp_projected", "centroid_auto"):
        v = agg[("_kp", k)]
        print(f"  {k:<16}{v:>6}  {v/tot*100:5.1f}%")
    print("\n=== held-out 앙각층 x 2D 적격성 ===")
    c = Counter((r["elev_bin"], r["eligibility_2d"])
                for r in out if r["split_role"] == "HELD_OUT")
    print(f"{'층':<8}{'GT_STRONG':>11}{'GT_PARTIAL':>12}{'GT_SUSPECT':>12}")
    for b in ("0-3", "3-8", "8-15", "15-30"):
        s = [c[(b, k)] for k in ("GT_STRONG", "GT_PARTIAL", "GT_SUSPECT")]
        if sum(s):
            print(f"{b:<8}{s[0]:>11}{s[1]:>12}{s[2]:>12}")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
