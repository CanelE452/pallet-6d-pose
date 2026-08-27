"""PHASE 2·3 — 이 머신에서 실제로 쓸 수 있는 generic pallet mesh 은행.

"20 개 파일" 이 목표가 아니라 **독립 topology** 가 목표다.  그래서 near-duplicate
클러스터링을 먼저 하고, 같은 클러스터는 unique instance 로 세지 않는다.

★ 스케일 변형은 새 instance 가 아니다.  회전·평행이동·균일스케일 불변 서명으로
클러스터링하므로 스케일만 다른 mesh 는 같은 클러스터로 묶인다.
"""
from __future__ import annotations

import csv, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from audit_v2 import load_usd, signature  # noqa: E402

ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
OUT = ROOT/"challenge/yolo_pose_one_model/broad_family_v2"
ISAAC = ROOT/("data/isaac/isaac_assets/Assets/Isaac/4.5/Isaac/Environments/"
              "Simple_Warehouse/Props")
CLUSTER_L1 = 0.15          # 회전불변 히스토그램 L1 이 이 아래면 같은 topology 로 본다

CANDIDATES = [
    # (asset_id, path, origin, license, 현재 BROAD 에서 쓰는가)
    ("scene.usd", ROOT/"data/pallet/raw_data/models_usd/scene.usd",
     "project (Isaac 시절 자산)", "UNKNOWN — 확인 필요", True),
    ("scene_1.usd", ROOT/"data/pallet/raw_data/models_usd/scene_1.usd",
     "project", "UNKNOWN — 확인 필요", True),
    ("scene_2.usd", ROOT/"data/pallet/raw_data/models_usd/scene_2.usd",
     "project (BROAD 미사용)", "UNKNOWN — 확인 필요", False),
    ("scene_3.usd", ROOT/"data/pallet/raw_data/models_usd/scene_3.usd",
     "project (BROAD 미사용)", "UNKNOWN — 확인 필요", False),
    ("SM_PaletteA_01.usd", ISAAC/"SM_PaletteA_01.usd",
     "NVIDIA Isaac Sim 4.5 Simple_Warehouse", "NVIDIA Omniverse asset EULA — 확인 필요", False),
    ("SM_PaletteA_02.usd", ISAAC/"SM_PaletteA_02.usd",
     "NVIDIA Isaac Sim 4.5 Simple_Warehouse", "NVIDIA Omniverse asset EULA — 확인 필요", False),
]

ASPECT_BINS = [("LOW", 1.00, 1.15), ("MID", 1.15, 1.40), ("HIGH", 1.40, 9.99)]
THICK_BINS = [("THIN", 0.00, 0.10), ("MID", 0.10, 0.15), ("THICK", 0.15, 9.99)]


def binof(v, bins):
    for name, lo, hi in bins:
        if lo <= v < hi:
            return name
    return bins[-1][0]


def main():
    import trimesh
    rows = []
    for aid, path, origin, lic, used in CANDIDATES:
        if not path.exists():
            rows.append({"asset_id": aid, "path": str(path), "origin": origin,
                         "license": lic, "used_in_broad": used,
                         "status": "FILE_MISSING"})
            continue
        if str(path).endswith((".usd", ".usda", ".usdc")):
            v, f = load_usd(path)
        else:
            m = trimesh.load(str(path), force="mesh")
            v, f = np.asarray(m.vertices, float), len(m.faces)
        if v is None:
            rows.append({"asset_id": aid, "path": str(path), "origin": origin,
                         "license": lic, "used_in_broad": used,
                         "status": "NO_MESH"})
            continue
        s = signature(aid, v, f)
        rows.append({"asset_id": aid, "path": str(path.relative_to(ROOT)),
                     "origin": origin, "license": lic, "used_in_broad": used,
                     "status": "OK", "vertices": s["vertices"], "faces": s["faces"],
                     "aspect": s["footprint_aspect"],
                     "thickness": s["thickness_ratio"],
                     "aspect_bin": binof(s["footprint_aspect"], ASPECT_BINS),
                     "thickness_bin": binof(s["thickness_ratio"], THICK_BINS),
                     "rot_inv_sha": s["rotation_invariant"]["sha256"][:16],
                     "_hist": s["rotation_invariant"]["hist"]})

    # near-duplicate 클러스터링 — 스케일 변형을 unique 로 세지 않기 위해
    ok = [r for r in rows if r["status"] == "OK"]
    clusters, assigned = [], {}
    for r in ok:
        h = np.array(r["_hist"])
        placed = False
        for ci, c in enumerate(clusters):
            if np.abs(h - np.array(c["rep"])).sum() < CLUSTER_L1:
                c["members"].append(r["asset_id"]); assigned[r["asset_id"]] = ci
                placed = True
                break
        if not placed:
            clusters.append({"rep": r["_hist"], "members": [r["asset_id"]]})
            assigned[r["asset_id"]] = len(clusters) - 1
    for r in ok:
        r["topology_cluster"] = assigned[r["asset_id"]]
    for r in rows:
        r.pop("_hist", None)

    target = json.load(open(OUT/"TARGET_ASSET_EXCLUSION_AUDIT_V2.json"))["target"]
    cover_before, cover_after = set(), set()
    for r in ok:
        cell = (r["aspect_bin"], r["thickness_bin"])
        cover_after.add(cell)
        if r["used_in_broad"]:
            cover_before.add(cell)

    all_cells = [(a[0], t[0]) for a in ASPECT_BINS for t in THICK_BINS]
    with open(OUT/"GENERIC_MESH_BANK.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    with open(OUT/"GEOMETRY_COVERAGE_BEFORE_AFTER.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["aspect_bin", "thickness_bin", "before_broad4",
                    "after_local_bank", "target_cell"])
        tcell = (binof(target["footprint_aspect"], ASPECT_BINS),
                 binof(target["thickness_ratio"], THICK_BINS))
        for c in all_cells:
            w.writerow([c[0], c[1], int(c in cover_before),
                        int(c in cover_after), int(c == tcell)])

    n_unique = len(clusters)
    n_new = len({r["topology_cluster"] for r in ok if not r["used_in_broad"]}
                - {r["topology_cluster"] for r in ok if r["used_in_broad"]})
    summary = {
        "raw_mesh_files_examined": len(CANDIDATES),
        "mesh_readable": len(ok),
        "independent_topology_clusters": n_unique,
        "clusters": [{"id": i, "members": c["members"]}
                     for i, c in enumerate(clusters)],
        "new_topology_not_in_broad": n_new,
        "license_safe_confirmed": 0,
        "license_note": "확인된 것 0 건. Isaac asset 은 NVIDIA Omniverse EULA, "
                        "project USD 는 출처 미상. 논문 배포 전 반드시 확인해야 한다.",
        "coverage_cells_total": len(all_cells),
        "coverage_before": sorted(f"{a}/{t}" for a, t in cover_before),
        "coverage_after_local": sorted(f"{a}/{t}" for a, t in cover_after),
        "target_cell": f"{binof(target['footprint_aspect'], ASPECT_BINS)}/"
                       f"{binof(target['thickness_ratio'], THICK_BINS)}",
        "THIN_stratum_available_locally": any(
            r["thickness_bin"] == "THIN" for r in ok),
        "BLOCKER": None,
    }
    if n_unique < 8:
        summary["BLOCKER"] = (
            f"독립 topology 가 {n_unique} 개뿐이다. G_CONSERVATIVE(+8) 조차 "
            f"로컬 자산으로 도달할 수 없다. 외부 mesh 확보가 렌더의 선결 조건이다.")
    (OUT/"GENERIC_MESH_BANK_AUDIT.md").write_text("\n".join([
        "# GENERIC MESH BANK — 로컬 실측", "",
        "목표는 파일 수가 아니라 **독립 topology** 다. 회전·평행이동·균일스케일",
        "불변 서명으로 near-duplicate 클러스터링을 먼저 했다 — 스케일 변형은 같은",
        "클러스터로 묶이므로 unique instance 를 부풀리지 않는다.", "", "```",
        json.dumps(summary, indent=1, ensure_ascii=False), "```", "",
        "## 후보별", "", "```",
        f"{'asset_id':22}{'status':14}{'verts':>8}{'aspect':>8}{'thick':>8}"
        f"{'cell':>14}{'cluster':>9}{'in BROAD':>10}",
    ] + [
        f"{r['asset_id']:22}{r['status']:14}"
        f"{r.get('vertices', 0):>8,}{r.get('aspect', 0):>8.3f}"
        f"{r.get('thickness', 0):>8.4f}"
        f"{(r.get('aspect_bin', '-') + '/' + r.get('thickness_bin', '-')):>14}"
        f"{r.get('topology_cluster', -1):>9}{str(r['used_in_broad']):>10}"
        for r in rows
    ] + ["```", "",
         f"평가 대상 cell = **{summary['target_cell']}** "
         f"(aspect {target['footprint_aspect']}, thickness {target['thickness_ratio']})",
         "",
         "★ target cell 을 겨냥해 asset 을 만들지 않는다. 위 표는 현재 support 가",
         "어느 cell 에 몰려 있는지를 보이기 위한 것이다.", ""]) + "\n")

    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
