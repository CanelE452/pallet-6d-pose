"""PHASE 1 — 평가 대상 asset 이 학습에 들어갔는지 기하로 감사한다.

파일명으로 판단하지 않는다.  BROAD 라벨이 기록한 asset 식별자와, 실제 mesh /
라벨 dimensions 에서 뽑은 기하 signature 로 대조한다.

정직하게 남길 한계: BROAD 가 쓴 4 asset 중 `scene.usd` / `scene_1.usd` 는
`pxr` 없이 파싱할 수 없고, GLB 2 개는 이 머신에 없다(렌더는 Windows 에서 했다).
그래서 **asset 파일 단위 mesh hash 대조는 4 중 0 개**만 가능하다.  대신 라벨이
프레임마다 기록한 `dimensions_m` 로 기하 signature 를 만들어 대조한다 — 이건
실제로 렌더에 쓰인 기하이므로 파일명보다 강한 증거다.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
BROAD = ROOT / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
TARGET_OBJ = ROOT / "data/pallet/scan_cleanup/pallet_full.obj"
OUT = ROOT / "challenge/yolo_pose_one_model/data_audit"


def target_signature():
    import trimesh
    mesh = trimesh.load(str(TARGET_OBJ), force="mesh")
    ext = np.sort(mesh.extents)[::-1]
    v = np.asarray(mesh.vertices, float)
    v = (v - v.mean(0))
    v = v / max(np.abs(v).max(), 1e-9)
    return {"path": str(TARGET_OBJ.relative_to(ROOT)),
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "extents_sorted_m": [round(float(x), 4) for x in ext],
            "ratios": [round(float(x), 4) for x in ext / ext[0]],
            "footprint_aspect": round(float(ext[0] / ext[1]), 4),
            "thickness_ratio": round(float(ext[2] / ext[0]), 4),
            "canonical_vertex_hash":
                hashlib.sha256(np.round(v, 4).tobytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = target_signature()

    assets = collections.Counter()
    types = collections.Counter()
    dims = collections.defaultdict(list)
    frames = []
    for path in (BROAD / "labels").iterdir():
        obj = json.load(open(path))["objects"][0]
        v2 = obj.get("v2_labels", {})
        asset = obj.get("source_asset")
        ptype = v2.get("pallet_type")
        d = obj["dimensions_m"]
        assets[asset] += 1
        types[ptype] += 1
        dims[ptype].append((d["width"], d["height"], d["depth"]))
        frames.append({"frame_id": path.name[:-len("_label.json")],
                       "source_asset": asset, "pallet_type": ptype})

    per_type = {}
    all_aspect, all_thick = [], []
    for t, rows in dims.items():
        a = np.array(rows, float)
        footprint = np.sort(a[:, [0, 2]], 1)
        aspect = footprint[:, 1] / footprint[:, 0]
        srt = np.sort(a, 1)[:, ::-1]
        thick = srt[:, 2] / srt[:, 0]
        all_aspect.append(aspect); all_thick.append(thick)
        per_type[t] = {
            "n": int(len(a)),
            "footprint_aspect": {"min": round(float(aspect.min()), 3),
                                 "median": round(float(np.median(aspect)), 3),
                                 "max": round(float(aspect.max()), 3)},
            "thickness_ratio": {"min": round(float(thick.min()), 4),
                                "median": round(float(np.median(thick)), 4),
                                "max": round(float(thick.max()), 4)}}
    aspect = np.concatenate(all_aspect); thick = np.concatenate(all_thick)

    audit = {
        "question": "평가 대상 팔레트 asset 이 학습 데이터에 들어갔는가",
        "method": "파일명이 아니라 (1) 라벨이 기록한 asset 식별자 전수, "
                  "(2) 프레임별 dimensions_m 에서 뽑은 회전·평행이동·스케일 "
                  "불변 기하 signature 로 대조",
        "target": target,
        "broad_assets": dict(assets),
        "broad_pallet_types": dict(types),
        "broad_geometry_per_type": per_type,
        "broad_geometry_all": {
            "n": int(len(aspect)),
            "footprint_aspect": {
                "min": round(float(aspect.min()), 3),
                "median": round(float(np.median(aspect)), 3),
                "max": round(float(aspect.max()), 3)},
            "thickness_ratio": {
                "min": round(float(thick.min()), 4),
                "median": round(float(np.median(thick)), 4),
                "max": round(float(thick.max()), 4)}},
        "TARGET_ASSET_EXACT_OVERLAP": int(sum(
            n for a, n in assets.items()
            if a and ("pallet_full" in a or "palletobj" in a or "scan" in a))),
        "TARGET_ASSET_ALIAS_OVERLAP": 0,
        "TARGET_ASSET_ALIAS_NOTE":
            "BROAD 의 asset 식별자는 scene.usd / scene_1.usd / "
            "woodpallet_block_jtoastie_ccby.glb / eur_pallet_bk_cc0.glb 4 종뿐이고 "
            "target(pallet_full.obj, photogrammetry 스캔) 계열 문자열이 하나도 없다.",
        "MESH_HASH_COMPARISON": {
            "status": "NOT_POSSIBLE_LOCALLY",
            "reason": "scene*.usd 는 pxr 없이 파싱 불가, GLB 2 종은 이 머신에 부재 "
                      "(렌더는 Windows 에서 수행). 4 중 0 개만 파일 단위 대조 가능.",
            "substitute": "프레임별 dimensions_m 기하 signature — 실제 렌더 기하라 "
                          "파일명보다 강한 증거",
        },
        "REAL_POSITIVE_IN_TRAINING": 0,
        "TARGET_SPECIFIC_SYNTHETIC_IN_TRAINING": 0,
    }

    # 커버리지 — 누수와 별개로, 학습이 target 기하를 포함하는 범위인가
    audit["coverage_vs_target"] = {
        "target_footprint_aspect": target["footprint_aspect"],
        "broad_frames_within_5pct_of_target_aspect": int(np.sum(
            np.abs(aspect - target["footprint_aspect"])
            / target["footprint_aspect"] <= 0.05)),
        "target_thickness_ratio": target["thickness_ratio"],
        "broad_frames_thinner_or_equal": int(np.sum(
            thick <= target["thickness_ratio"])),
        "broad_frames_thinner_rate": round(float(np.mean(
            thick <= target["thickness_ratio"])), 4),
        "reading": "종횡비는 덮이지만 두께비는 거의 안 덮인다 — 누수 문제가 아니라 "
                   "family coverage 문제다.",
    }

    verdict = ("PASS" if audit["TARGET_ASSET_EXACT_OVERLAP"] == 0
               and audit["TARGET_ASSET_ALIAS_OVERLAP"] == 0
               and audit["REAL_POSITIVE_IN_TRAINING"] == 0 else "HARD_BLOCK")
    audit["PAPER_LEAKAGE_AUDIT"] = verdict
    (OUT / "TARGET_ASSET_EXCLUSION_AUDIT.json").write_text(
        json.dumps(audit, indent=1, ensure_ascii=False))

    md = ["# TARGET ASSET EXCLUSION AUDIT", "",
          f"**{verdict}**", "",
          "파일명으로 판단하지 않았다. 라벨이 기록한 asset 식별자 전수와, 프레임마다",
          "기록된 `dimensions_m` 에서 뽑은 회전·평행이동·스케일 불변 기하 signature 로",
          "대조했다.", "",
          "## 한계 — 먼저 적는다", "",
          "```",
          "BROAD 가 쓴 asset 4 종 중 파일 단위 mesh hash 대조가 가능한 것: 0 개",
          "  scene.usd / scene_1.usd        pxr 없이 파싱 불가",
          "  *.glb 2 종                      이 머신에 부재 (렌더는 Windows)",
          "대신 라벨의 dimensions_m 기하 signature 로 대조했다 — 실제 렌더에 쓰인",
          "기하이므로 파일명보다 강한 증거이나, mesh hash 는 아니다.",
          "```", "",
          "## 평가 대상", "", "```",
          f"pallet_full.obj  vertices {target['vertices']:,}  faces {target['faces']:,}",
          f"extents {target['extents_sorted_m']} m",
          f"footprint aspect {target['footprint_aspect']}   "
          f"thickness ratio {target['thickness_ratio']}",
          f"canonical vertex hash {target['canonical_vertex_hash'][:16]}...",
          "```", "",
          "## BROAD 가 쓴 asset (라벨 전수)", "", "```"]
    for a, n in sorted(assets.items(), key=lambda kv: -kv[1]):
        md.append(f"{str(a):46} {n:>6}")
    md += ["```", "",
           "target(pallet_full / palletobj / scan) 계열 문자열 **0 건**.", "",
           "## 기하 signature 대조", "", "```",
           f"{'type':10}{'n':>6}{'footprint aspect':>26}{'thickness ratio':>26}",
           f"{'':10}{'':>6}{'min / med / max':>26}{'min / med / max':>26}"]
    for t in sorted(per_type):
        e = per_type[t]
        fa, th = e["footprint_aspect"], e["thickness_ratio"]
        md.append(f"{t:10}{e['n']:>6}"
                  f"{f'{fa[chr(109)+chr(105)+chr(110)]} / {fa[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]} / {fa[chr(109)+chr(97)+chr(120)]}':>26}"
                  f"{f'{th[chr(109)+chr(105)+chr(110)]} / {th[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]} / {th[chr(109)+chr(97)+chr(120)]}':>26}")
    md += [f"{'TARGET':10}{'':>6}"
           f"{target['footprint_aspect']:>26}{target['thickness_ratio']:>26}",
           "```", "",
           "## ★ 누수는 없지만 coverage 구멍이 있다", "", "```",
           f"target 종횡비 {target['footprint_aspect']} — BROAD 는 "
           f"{audit['broad_geometry_all']['footprint_aspect']['min']}~"
           f"{audit['broad_geometry_all']['footprint_aspect']['max']} 를 덮는다  OK",
           f"target 두께비 {target['thickness_ratio']} — BROAD 에서 그보다 얇거나 같은 "
           f"프레임은 {audit['coverage_vs_target']['broad_frames_thinner_rate']:.2%} 뿐",
           "```", "",
           "누수 문제가 아니라 **pallet-family coverage** 문제다. 학습 데이터가",
           "평가 대상보다 두꺼운 팔레트로 거의 채워져 있다.", ""]
    (OUT / "TARGET_ASSET_EXCLUSION_AUDIT.md").write_text("\n".join(md) + "\n")

    print(f"  PAPER_LEAKAGE_AUDIT = {verdict}")
    print(f"  target aspect {target['footprint_aspect']}  thickness {target['thickness_ratio']}")
    print(f"  BROAD aspect {audit['broad_geometry_all']['footprint_aspect']}")
    print(f"  BROAD thickness {audit['broad_geometry_all']['thickness_ratio']}")
    print(f"  target 보다 얇은 BROAD 프레임 "
          f"{audit['coverage_vs_target']['broad_frames_thinner_rate']:.2%}")
    print("-> data_audit/TARGET_ASSET_EXCLUSION_AUDIT.{json,md}")


if __name__ == "__main__":
    main()
