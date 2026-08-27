"""PHASE 1 V2 — mesh 수준 target 배제 감사.

V1 은 4 asset 중 **0 개**만 파일 단위 대조가 가능했다. `usd-core` 를 설치해
USD 2 개를 실제로 파싱했으므로 이제 **2/4** 다. 나머지 2 개(GLB)는 이 머신에
없어 UNRESOLVED 로 남긴다 — 이름이 CC 인터넷 모델처럼 보인다는 건 근거가 아니다.

canonicalization 규칙
    병합     stage 안 모든 UsdGeom.Mesh 정점을 하나로
    평행이동 centroid 를 원점으로
    스케일   최대 extent 로 1 정규화
    회전     정점 좌표는 회전 불변이 아니므로 centroid 거리 히스토그램(64bin)을
             회전·정점순서 불변 서명으로 별도 계산. 정렬 extent 비율은 축 치환 불변
"""
from __future__ import annotations

import hashlib, json, pathlib
import numpy as np

ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
OUT = ROOT/"challenge/yolo_pose_one_model/broad_family_v2"
USED = {"scene.usd": "Pallet_0", "scene_1.usd": "Pallet_1",
        "woodpallet_block_jtoastie_ccby.glb": "Pallet_2",
        "eur_pallet_bk_cc0.glb": "Pallet_3"}
BINS = 64


def rotation_invariant(v):
    v = v - v.mean(0)
    s = np.linalg.norm(v, axis=1)
    s = s / max(s.max(), 1e-12)
    hist, _ = np.histogram(s, bins=BINS, range=(0.0, 1.0))
    hist = hist / max(hist.sum(), 1)
    return {"hist": [round(float(x), 6) for x in hist],
            "sha256": hashlib.sha256(np.round(hist, 6).tobytes()).hexdigest()}


def signature(name, verts, faces):
    ext = np.sort(verts.max(0) - verts.min(0))[::-1]
    canon = verts - verts.mean(0)
    canon = canon / max(np.abs(canon).max(), 1e-12)
    return {"asset": name, "vertices": int(len(verts)), "faces": int(faces),
            "extents_sorted": [round(float(x), 6) for x in ext],
            "dimension_ratios": [round(float(x), 6) for x in ext/ext[0]],
            "footprint_aspect": round(float(ext[0]/ext[1]), 4),
            "thickness_ratio": round(float(ext[2]/ext[0]), 4),
            "canonical_vertex_sha256":
                hashlib.sha256(np.round(canon, 4).tobytes()).hexdigest(),
            "rotation_invariant": rotation_invariant(verts)}


def load_usd(path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    pts, faces = [], 0
    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        a = mesh.GetPointsAttr().Get()
        if a:
            pts.append(np.array(a, dtype=float))
        c = mesh.GetFaceVertexCountsAttr().Get()
        if c:
            faces += len(c)
    return (np.vstack(pts), faces) if pts else (None, 0)


def main():
    import trimesh
    tm = trimesh.load(str(ROOT/"data/pallet/scan_cleanup/pallet_full.obj"),
                      force="mesh")
    target = signature("pallet_full.obj", np.asarray(tm.vertices, float),
                       len(tm.faces))

    resolved, unresolved = {}, {}
    for name, ptype in USED.items():
        hit = [h for h in ROOT.rglob(name) if ".git" not in str(h)]
        if not hit:
            unresolved[name] = {"pallet_type": ptype,
                                "reason": "파일이 이 머신에 없다 (렌더는 Windows)"}
            continue
        if name.endswith(".usd"):
            v, f = load_usd(hit[0])
        else:
            m = trimesh.load(str(hit[0]), force="mesh")
            v, f = np.asarray(m.vertices, float), len(m.faces)
        if v is None:
            unresolved[name] = {"pallet_type": ptype, "reason": "mesh 0"}
            continue
        s = signature(name, v, f)
        s["pallet_type"] = ptype
        s["path"] = str(hit[0].relative_to(ROOT))
        resolved[name] = s

    def compare(a, b):
        h1 = np.array(a["rotation_invariant"]["hist"])
        h2 = np.array(b["rotation_invariant"]["hist"])
        return {"exact_vertex_hash":
                    a["canonical_vertex_sha256"] == b["canonical_vertex_sha256"],
                "vertex_count_equal": a["vertices"] == b["vertices"],
                "face_count_equal": a["faces"] == b["faces"],
                "rotation_invariant_hash_equal":
                    a["rotation_invariant"]["sha256"] == b["rotation_invariant"]["sha256"],
                "shape_hist_L1": round(float(np.abs(h1-h2).sum()), 4),
                "ratio_L1": round(float(np.abs(
                    np.array(a["dimension_ratios"]) -
                    np.array(b["dimension_ratios"])).sum()), 4)}

    comp = {n: compare(target, s) for n, s in resolved.items()}
    exact = sum(1 for c in comp.values() if c["exact_vertex_hash"])
    rot = sum(1 for c in comp.values() if c["rotation_invariant_hash_equal"])

    audit = {"question": "BROAD 가 쓴 mesh 가 평가 대상 mesh 와 같은가",
             "canonicalization": {
                 "merge": "stage 내 모든 UsdGeom.Mesh 정점 병합",
                 "translation": "centroid 원점화", "scale": "최대 extent 정규화",
                 "rotation": "centroid 거리 히스토그램(64bin) = 회전·정점순서 불변. "
                             "정렬 extent 비율 = 축 치환 불변"},
             "target": target, "resolved": resolved, "unresolved": unresolved,
             "comparisons": comp,
             "coverage": {"assets_used_by_broad": len(USED),
                          "mesh_hash_comparable": len(resolved),
                          "note_v1": "V1 은 0/4. usd-core 설치로 2/4."},
             "EXACT_OVERLAP": exact, "ALIAS_OVERLAP": 0,
             "NORMALIZED_GEOMETRY_OVERLAP": rot,
             "MESH_EXCLUSION_EXACT":
                 "RESOLVED" if len(resolved) == len(USED) else "PARTIAL",
             "unresolved_note": "GLB 2 종은 파일 부재로 mesh 단위 대조 불가. "
                                "이름이 CC 인터넷 모델처럼 보인다는 것은 근거가 "
                                "아니다. 렌더 머신에서 회수하면 닫힌다."}
    (OUT/"TARGET_ASSET_EXCLUSION_AUDIT_V2.json").write_text(
        json.dumps(audit, indent=1, ensure_ascii=False))

    md = ["# TARGET ASSET EXCLUSION AUDIT — V2 (mesh 수준)", "",
          f"**EXACT_OVERLAP {exact} / NORMALIZED_GEOMETRY_OVERLAP {rot} / "
          f"MESH_EXCLUSION_EXACT = {audit['MESH_EXCLUSION_EXACT']}**", "",
          "V1 은 4 asset 중 **0 개**만 파일 단위 대조가 가능했다. `usd-core` 로",
          "USD 2 개를 실제 파싱해 이제 **2/4**.", "", "## 서명", "", "```",
          f"{'asset':36}{'verts':>10}{'faces':>10}{'aspect':>9}{'thick':>9}",
          f"{'TARGET pallet_full.obj':36}{target['vertices']:>10,}"
          f"{target['faces']:>10,}{target['footprint_aspect']:>9.3f}"
          f"{target['thickness_ratio']:>9.4f}"]
    for n, s in resolved.items():
        md.append(f"{n + ' (' + s['pallet_type'] + ')':36}{s['vertices']:>10,}"
                  f"{s['faces']:>10,}{s['footprint_aspect']:>9.3f}"
                  f"{s['thickness_ratio']:>9.4f}")
    md += ["```", "", "## 대조", "", "```"]
    for n, c in comp.items():
        md += [f"{n}",
               f"  exact vertex hash        {c['exact_vertex_hash']}",
               f"  회전불변 형상 hash        {c['rotation_invariant_hash_equal']}",
               f"  vertex/face 수 일치      {c['vertex_count_equal']} / {c['face_count_equal']}",
               f"  형상 히스토그램 L1       {c['shape_hist_L1']}   (0 이면 동일)",
               f"  치수비 L1                {c['ratio_L1']}"]
    md += ["```", "", "## 아직 닫히지 않은 것", "", "```"]
    for n, u in unresolved.items():
        md.append(f"{n:44} {u['pallet_type']}  {u['reason']}")
    md += ["```", "",
           "이름이 CC 라이선스 인터넷 모델처럼 보인다는 것은 **근거가 아니다.**",
           "렌더 머신에서 두 GLB 를 회수하면 이 항목이 닫힌다.", ""]
    (OUT/"TARGET_ASSET_EXCLUSION_AUDIT_V2.md").write_text("\n".join(md)+"\n")

    print(f"  mesh 대조 가능 {len(resolved)}/{len(USED)}  (V1 은 0/4)")
    for n, c in comp.items():
        print(f"  {n:36} exact {c['exact_vertex_hash']}  "
              f"rot-inv {c['rotation_invariant_hash_equal']}  "
              f"histL1 {c['shape_hist_L1']}  ratioL1 {c['ratio_L1']}")
    print(f"  MESH_EXCLUSION_EXACT = {audit['MESH_EXCLUSION_EXACT']}")


if __name__ == "__main__":
    main()
