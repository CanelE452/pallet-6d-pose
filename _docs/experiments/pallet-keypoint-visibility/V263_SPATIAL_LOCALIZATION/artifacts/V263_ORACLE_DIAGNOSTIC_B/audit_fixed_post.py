"""Closed SYN1003 fixed-post replay; unavailable physical oracles stay unavailable."""
import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np

ORACLES = ("GT_VISIBILITY_ONLY", "DIRECT_OBSERVABLE_GT_LOCATION_ONLY", "BOTH")


def guard(path):
    p = Path(path)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("absolute bounded path required")
    if any(x in str(p).lower() for x in ("v112", "cache", ".pyc")):
        raise ValueError("forbidden lexical path")
    for a in (p, *p.parents):
        if a.is_symlink():
            raise ValueError("symlink artifact/ancestor")
    return p


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def checked(binding):
    p = guard(binding["path"])
    b = p.read_bytes()
    if len(b) != binding["size_bytes"] or sha_bytes(b) != binding["sha256"]:
        raise ValueError("binding mismatch: " + str(p))
    return b


def dump(path, value):
    with guard(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def csvsave(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with guard(path).open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def module(name, path, source, injected=None, remove_import=None, functions_only=None):
    tree = ast.parse(source)
    if remove_import:
        removed = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == remove_import]
        if len(removed) != 1:
            raise ValueError("exact inherited import not found")
        tree.body = [n for n in tree.body if n not in removed]
    if functions_only:
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions_only]
        if {n.name for n in tree.body} != set(functions_only):
            raise ValueError("missing pure functions")
    m = types.ModuleType(name)
    m.__file__ = str(path)
    m.__dict__.update(injected or {})
    sys.modules[name] = m
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), m.__dict__)
    return m


def load_backend(bindings, raw):
    geom = module("v263_geometry", bindings["geometry"]["path"], raw["geometry"])
    selector = module("v263_selector", bindings["selector"]["path"], raw["selector"],
                      {"geometry": geom}, remove_import="scripts.annotate")
    masked = module("v263_masked", bindings["masked_backend"]["path"], raw["masked_backend"],
                    {"S": selector}, remove_import="challenge.evaluation_v2")
    # masked_backend's two original file reads are bounded by the pins already
    # verified above: selector.__file__ and the fixed V54 cuboid/solve source.
    evaluator = module("v263_metrics", bindings["evaluator"]["path"], raw["evaluator"],
                       {"np": np, "math": __import__("math")},
                       functions_only={"angle", "rotations", "metric", "summarize"})
    old_transform = json.loads(raw["v58_transform"])
    if masked.TRANSFORM_PROVENANCE != old_transform:
        raise ValueError("V58 AST transform provenance changed")
    # Meaningful no-six contract check: no solver operation may be called.
    original_solve = cv2.solvePnP
    def forbidden(*args, **kwargs):
        raise AssertionError("solver called below original six-point minimum")
    cv2.solvePnP = forbidden
    try:
        v = np.r_[np.ones(5, bool), np.zeros(4, bool)]
        assert masked.infer_masked(np.zeros((9, 2)), np.eye(3), np.ones(3), v, source=True) == {"available": False}
    finally:
        cv2.solvePnP = original_solve
    return masked, evaluator


def pose_gap(a, b):
    if a["available"] != b["available"]:
        return None, False
    if not a["available"]:
        return 0.0, True
    fields = ("R_cf", "R_physical", "centroid", "cf_extents")
    d = max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in fields)
    return d, d <= 1e-9 and a.get("selected_hypothesis") == b.get("selected_hypothesis")


def run(entry):
    if entry.get("execution_authorized") is not True or entry.get("scope") != "SYN1003_FIXED_POST_ONLY":
        raise ValueError("Root-authorized SYN-only entry required")
    if checked(entry["code"]) != Path(__file__).read_bytes():
        raise ValueError("running code is not entry-bound")
    approval = json.loads(checked(entry["root_scope"]))
    if approval.get("approved") is not True or approval["entry_contract"] != {k: v for k, v in entry.items() if k != "root_scope"}:
        raise ValueError("scope contract mismatch")
    out = guard(entry["output_dir"])
    if out.exists():
        raise ValueError("fresh output directory required")
    b = entry["inputs"]
    raw = {k: checked(v) for k, v in b.items()}
    # These admissions are read from the original policy, never inferred from
    # ray fractions or from a projected cuboid landing inside a silhouette.
    policy = json.loads(raw["v54_policy"])
    assert "BLOCKED" in policy["oracle_matrix"]["visibility_only"]
    assert "BLOCKED" in policy["oracle_matrix"]["direct_position_only"]
    assert policy["physical_label_policy"].startswith("no DIRECT/EXTERNAL")
    assert json.loads(raw["v57_policy"])["not_physical_DIRECT"] is True
    assert all(a["surface_supported_n"] == 0 for a in json.loads(raw["v59_summary"])["assets"])
    up = [json.loads(x) for x in gzip.decompress(raw["upstream"]).splitlines()]
    gt = [json.loads(x) for x in gzip.decompress(raw["GT"]).splitlines()]
    assert len(up) == len(gt) == 1003
    assert len({x["id"] for x in up}) == len({x["id"] for x in gt}) == 1003
    by_gt = {x["id"]: x for x in gt}
    assert set(by_gt) == {x["id"] for x in up}
    assert {s: sum(x["split"] == s for x in up) for s in ("train", "validation")} == {"train": 768, "validation": 235}
    cv2.setNumThreads(1)
    masked, ev = load_backend(b, raw)
    out.mkdir()
    dump(out / "START.json", {"scope": entry["scope"], "entry": entry,
                               "oracle_admission": "three physical cells unavailable; no proxy substitution"})
    start = time.monotonic()
    parity, frame, coverage, predictions = [], [], [], []
    for i, u in enumerate(up):
        g = by_gt[u["id"]]
        q, K, xyz = np.asarray(u["initial_points"], float), np.asarray(u["K"], float), np.asarray(u["known_physical_dimensions_xyz_m"], float)
        X, RG, tG = np.asarray(g["native_GT_xyz"], float), np.asarray(g["R_fixed"], float), np.asarray(g["t"], float)
        assert u["split"] == g["split"] and np.array_equal(K, np.asarray(g["K"]))
        assert np.array_equal(xyz, np.asarray(g["physical_dimensions_xyz_m"]))
        assert q.shape == (9, 2) and np.isfinite(q).all() and X.shape == (8, 3)
        camera = X @ RG.T + tG
        homogeneous = camera @ K.T
        projected = homogeneous[:, :2] / homogeneous[:, 2:]
        projection_gap = float(np.max(np.abs(projected - np.asarray(g["projected_GT_native_xy"]))))
        assert projection_gap <= .01
        valid = np.ones(9, bool)
        p = masked.infer_masked(q, K, xyz, valid, source=True)
        delta, ok = pose_gap(p, u["initial_pose"])
        parity.append({"id": u["id"], "split": u["split"], "max_pose_component_difference": delta, "PASS": ok})
        if not ok:
            csvsave(out / "PARITY_FAILED.csv", parity)
            dump(out / "FAILURE.json", {"stage": "fixed all-valid baseline parity", "id": u["id"], "difference": delta, "automatic_retry": False})
            raise RuntimeError("original fixed-post parity failed")
        m = ev.metric(p, {"R": g["R_fixed"], "t": g["t"], "order": g["symmetry_order"]})
        frame.append({"id": u["id"], "split": u["split"], "asset": g["source_asset"], "arm": "ORIGINAL",
                      "symmetry_order": g["symmetry_order"], "dimensions_xyz_m": json.dumps(xyz.tolist()),
                      "fallback": False, "retained_native_points": 8, **m})
        predictions.append({"id": u["id"], "arm": "ORIGINAL", "valid9": valid.tolist(), "pose": p,
                            "q9_F64_sha256": sha_bytes(q.astype("<f8").tobytes()), "native3D_F64_sha256": sha_bytes(np.asarray(u["native_xyz"], "<f8").tobytes())})
        for arm in ORACLES:
            coverage.append({"id": u["id"], "split": u["split"], "asset": g["source_asset"], "arm": arm,
                             "configuration_executed": False, "native_queries": 8, "certified_direct_points": 0,
                             "certified_physical_visibility_labels": 0, "oracle_pose_available": None,
                             "reason": "UNAVAILABLE_PHYSICAL_NATIVE_SUPPORT_AND_ORIGINAL_RENDER_DEPTH_OWNER_BINDING",
                             "proxy_substituted": False, "projection_rounding_max_px": projection_gap})
        if (i + 1) % 250 == 0 or i + 1 == len(up):
            print("ORIGINAL_FIXED_POST", i + 1, "/1003", flush=True)
    summaries = []
    strata = [("ALL1003", frame)]
    strata += [("split:" + s, [x for x in frame if x["split"] == s]) for s in ("train", "validation")]
    strata += [("asset:" + a, [x for x in frame if x["asset"] == a]) for a in sorted({x["asset"] for x in frame})]
    strata += [("split:" + s + "|asset:" + a, [x for x in frame if x["split"] == s and x["asset"] == a])
               for s in ("train", "validation") for a in sorted({x["asset"] for x in frame})]
    for name, rs in strata:
        summaries.append({"stratum": name, "arm": "ORIGINAL", **ev.summarize(rs)})
    cells = [{"configuration": "ORIGINAL", "requested_frames": 1003, "executed_frames": 1003, "status": "EXECUTED_FIXED_POST_PARITY"}]
    cells += [{"configuration": a, "requested_frames": 1003, "executed_frames": 0, "status": "UNAVAILABLE_CONTRACT",
               "missing": "native physical support plus same-render pointwise DIRECT/EXTERNAL/depth-owner evidence",
               "virtual_projection_not_direct_observation": True} for a in ORACLES]
    csvsave(out / "BASELINE_PARITY1003.csv", parity)
    csvsave(out / "BASELINE_PER_FRAME.csv", frame)
    csvsave(out / "BASELINE_SUMMARY.csv", summaries)
    csvsave(out / "ORACLE_COVERAGE_ALL1003.csv", coverage)
    dump(out / "FOUR_CONFIGURATION_STATUS.json", cells)
    dump(out / "ORIGINAL_PREDICTIONS.json", predictions)
    inventory = list(csv.DictReader(raw["CAL_inventory"].decode().splitlines()))
    dump(out / "REAL_CAL_DEPENDENCY_INVENTORY.json", {"historical_metadata_only": True, "rows": inventory,
        "no_CAL_artifact_or_weights_loaded": True, "new_fit_or_calibration": 0})
    report = """# V263 고정 후단 SYN 진단

원 공통 1,003장(원 TRAIN768/반복 노출 VAL235)을 전수 재생했다. 원 finite9×2, native 첫8 대응 index, K/치수, W/D 선택, SQPnP와 기본 LM, source basis를 유지했다. 전체 결과의 저장 N0 일치를 검사했으며 설정·문턱·초기값을 최적화하지 않았다.

GT visibility-only / 직접 관측 GT location-only / both는 실행 가능한 물리 oracle 계약이 없다. 기존 native8은 원 camera_dynamic_0123_v4의 cuboid 투영 채널이고, 원 render의 점별 물리 표면 지원·depth/instance-owner가 인증되지 않았다. V57의 convex-self/virtual-ray 결과를 DIRECT로 승격하지 않았다. 모든 1,003장×3 미지원 cell을 원 분모 그대로 보존했다. 이것은 oracle 실패 성능 0점이 아니라 oracle 입력의 부재다.

V59의 별도 로컬 mesh fixture에서 native32점 표면 지원0/32였다는 기존 metadata는 경계 근거다. 그 fixture를 과거 원 render 전체의 인증으로 확대하지 않는다. 필요한 것은 원 source asset/import transform과 native3D order를 결합한 same-render 표면/depth/owner 증거이며, 최근접 표면 snap이나 GT-best permutation으로 대체하면 목표가 바뀐다.

ORIGINAL 통계는 asset 및 split별 평균·중앙값·P90/P95·최대 T/R, canonical near90와 symmetry-aware R>45를 별도 기록한다. asset 이름을 목재/정사각형으로 임의 매핑하지 않았다. 세 oracle의 correct→wrong/회복·damage는 비교 자체가 미지원이므로 계산하지 않았다. near90를 width/depth 또는 native correspondence의 확정 원인으로 부르지 않는다.

기존 real-CAL 의존성은 G0 metadata inventory를 그대로 보존했다. state classifier·feature statistics, residual covariance/reliability와 pose prior·bias, O/T empirical domains, H1의 transitive bank/cost 의존성이 있다. 이 실행은 해당 artifact/모델을 불러오거나 적합하지 않았다. raw RGB·annotation archive·mesh·실사/current/held case·모델/encoder/학습·PnP 변경은0이다.
"""
    (out / "REPORT_KO.md").write_text(report, encoding="utf-8")
    outputs = {}
    for name in ("START.json", "BASELINE_PARITY1003.csv", "BASELINE_PER_FRAME.csv", "BASELINE_SUMMARY.csv", "ORACLE_COVERAGE_ALL1003.csv", "FOUR_CONFIGURATION_STATUS.json", "ORIGINAL_PREDICTIONS.json", "REAL_CAL_DEPENDENCY_INVENTORY.json", "REPORT_KO.md"):
        p = out / name
        data = p.read_bytes()
        outputs[name] = {"path": str(p), "sha256": sha_bytes(data), "size_bytes": len(data)}
    complete = {"completed": True, "PASS": True, "objective": "baseline executed; physical oracle gaps bounded, not solved",
        "fixed_post_baseline_frames": 1003, "all_valid_N0_parity_frames": sum(x["PASS"] for x in parity),
        "max_N0_pose_gap": max(x["max_pose_component_difference"] for x in parity),
        "physical_oracle_executed_frames": {a: 0 for a in ORACLES}, "oracle_unknown_frame_cells": 3009,
        "physical_native_label_coverage": "UNAVAILABLE; no virtual-to-DIRECT upgrade", "all1003_no_cherry_pick": True,
        "same_2D_3D_mask_ast_exact_V58": True, "min6_no_solver_check_PASS": True,
        "fallback_policy": "same unavailable result; raw-incumbent sensitivity not invoked because no partial physical oracle is defined",
        "runtime": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__, "cv2_threads": cv2.getNumThreads()},
        "new_training": 0, "new_calibration": 0, "new_model_forward": 0, "new_render": 0,
        "raw_RGB_depth_mask_mesh_reads": 0, "private_current_or_held_case_payloads": 0,
        "SOURCE_derived_pose_and_point_metadata_read": True,
        "source_inputs": b, "code": entry["code"], "outputs": outputs, "seconds": time.monotonic() - start}
    dump(out / "COMPLETE.json", complete)
    print(json.dumps({k: complete[k] for k in ("completed", "PASS", "fixed_post_baseline_frames", "max_N0_pose_gap", "physical_oracle_executed_frames", "seconds")}), flush=True)
    return complete


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    args = ap.parse_args()
    p = guard(args.entry)
    run(json.loads(p.read_text()))
