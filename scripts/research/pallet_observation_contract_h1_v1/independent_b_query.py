"""Independent linear-system and query-construction audit of saved B diagnostics.

No producer geometry helper is imported. This is not a physical-edge ownership
certification and does not alter any original strict numerical assertion.
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import time
import numpy as np
from scripts.research.pallet_risk_information_v1.common import read, rows, sha, write

VARIANTS = ("Q00_ORIGINAL", "Q10_GT_ANCHOR", "Q01_GT_DIRECTION", "Q11_GT_BOTH")


def run(output, factorial, original_rows, cache):
    output, factorial, original_rows, cache = map(Path, (output, factorial, original_rows, cache))
    start = time.monotonic()
    seal = read(factorial / "B_QUERY_FACTORIAL_SEAL.json")
    assert seal["status"] == "COMPLETE"
    assert sha(original_rows) == seal["original_diagnostic_rows_sha256"]
    for name, digest in seal["artifacts"].items():
        assert sha(factorial / name) == digest
    index = {}
    for path in (cache / "frames").glob("*/FRAME_SEAL.json"):
        fid = read(path)["id"]
        assert fid not in index
        index[fid] = path.parent
    checks, failures = Counter(), []
    maxima = dict(lambda_absolute=0., normal_offset_px_absolute=0., substituted_center_px_absolute=0.)
    numeric_ambiguities = []
    def check(key, value, detail=None):
        checks[key] += 1
        if not bool(value):
            failures.append(dict(check=key, detail=detail))
    nframes = nb = nprofiles = 0
    for new, old in zip_longest(rows(factorial / "B_QUERY_FACTORIAL_ROWS.jsonl.gz"), rows(original_rows)):
        assert new is not None and old is not None
        check("original_order_identity", new["id"] == old["id"] and new["split"] == old["split"], new["id"])
        nframes += 1
        nb += bool(new["original_B_input_present"])
        folder = index[new["id"]]
        for name in ("BOUNDARY_BUNDLE.jsonl.gz", "E_RUN.jsonl.gz"):
            assert sha(folder / name) == seal["original_frame_bindings"][new["id"]][name]
        bundle = next(rows(folder / "BOUNDARY_BUNDLE.jsonl.gz"))
        e = next(rows(folder / "E_RUN.jsonl.gz"))
        queries = {q["support_id"]: q for g in (bundle or {}).get("groups", []) for q in g["queries"]}
        anchors = {int(m["native_corner"]): np.asarray(m["xy"], float) for m in e["packet"]["solver_input"]["visible"]}
        phases = {p["phase_index"]: p for p in old["registered_phases"]}
        for family in new["families"]:
            original_family = next(f for f in phases[family["phase_index"]]["families"] if f["family"] == family["family"])
            uv = np.asarray(original_family["raw_physical_projection_native"], float)
            initial = np.asarray(bundle["initial_projection_xy"], float) if bundle else None
            for q in family["profiles"]:
                nprofiles += 1
                k, m = q["native_corner"], q["native_neighbor"]
                source = queries[q["support_id"]]
                variant = q["query_variant"]
                check("fixed_profile_native_along", (k, m, q["along_base_px"]) ==
                      (source["native_corner"], source["native_neighbor"], source["along_base_px"]), new["id"])
                check("original_state_not_reclassified", q["original_corner_state"] == bundle["original_states"][k], new["id"])
                gt_direction = uv[m] - uv[k]
                gt_length = np.linalg.norm(gt_direction)
                init_direction = initial[m] - initial[k]
                tangent = gt_direction / gt_length if variant in VARIANTS[2:] else init_direction / np.linalg.norm(init_direction)
                normal = np.array([-tangent[1], tangent[0]])
                if variant in VARIANTS[2:]:
                    midpoint_outward = .5 * (uv[k] + uv[m]) - np.mean(uv, axis=0)
                    if normal @ midpoint_outward < 0:
                        normal = -normal
                else:
                    normal = np.asarray(source["outward_normal_xy"], float)
                anchor = uv[k] if variant in (VARIANTS[1], VARIANTS[3]) else anchors[k]
                center = np.asarray(source["center_xy"], float) if variant == VARIANTS[0] else anchor + q["along_raw_px"] * tangent
                difference = float(np.max(np.abs(center - q["center_xy"])))
                maxima["substituted_center_px_absolute"] = max(maxima["substituted_center_px_absolute"], difference)
                check("anchor_direction_factorial_center", difference <= 1e-9, new["id"])
                check("independent_tangent", np.max(np.abs(tangent - q["tangent_xy"])) <= 1e-12, new["id"])
                check("consistent_outward_normal", np.max(np.abs(normal - q["outward_normal_xy"])) <= 1e-12, new["id"])
                check("unclamped_along_overrun", q["query_domain_overrun"] == (q["along_raw_px"] > gt_length), new["id"])
                check("no_GT_candidate_or_stale_optical_cost", q["production_candidate"] is False and q["new_optical_energy_evaluated"] is False and q["original_RGB_distribution_reused_as_new_cost"] is False, new["id"])
                matrix = np.column_stack((uv[m] - uv[k], -normal))
                if q["geometry_status"] not in ("NONFINITE_PROJECTION", "DEGENERATE_PROJECTED_EDGE", "PARALLEL_NORMAL_AND_EDGE"):
                    solved = np.linalg.solve(matrix, center - uv[k])
                    condition = float(np.linalg.cond(matrix))
                    # A priori double-precision, condition-scaled comparison;
                    # original finite classification is never loosened by it.
                    bounds = 64 * np.finfo(float).eps * condition * np.maximum(1., np.abs(solved))
                    bounds = np.maximum(bounds, [1e-12, 1e-9])
                    errors = np.abs(solved - [q["lambda_segment"], q["normal_offset_px"]])
                    maxima["lambda_absolute"] = max(maxima["lambda_absolute"], float(errors[0]))
                    maxima["normal_offset_px_absolute"] = max(maxima["normal_offset_px_absolute"], float(errors[1]))
                    check("independent_intersection_lambda", errors[0] <= bounds[0], new["id"])
                    check("independent_intersection_offset_px", errors[1] <= bounds[1], new["id"])
                    eps = 8 * np.sqrt(np.finfo(float).eps)
                    finite = -eps <= solved[0] <= 1 + eps and family["minimum_raw_depth_m"] > 0
                    if bool(finite) != q["finite_support"]:
                        numeric_ambiguities.append(dict(id=new["id"], phase=family["phase_index"], family=family["family"], profile=q["support_id"], variant=variant,
                            original_lambda=q["lambda_segment"], independent_lambda=float(solved[0]), original_support=q["finite_support"], independent_support=bool(finite)))
                    check("original_finite_classification_unchanged", bool(finite) == q["finite_support"], new["id"])
            for variant in VARIANTS:
                pp = [p for p in family["profiles"] if p["query_variant"] == variant]
                s = family["summaries"][variant]
                check("profile_to_frame_joint", s["all_profiles_finite"] == (all(p["finite_support"] for p in pp) if pp else None), new["id"])
                for corner, summary in s["corner_groups"].items():
                    cp = [p for p in pp if p["native_corner"] == int(corner)]
                    check("profile_to_corner_joint", summary["all_profiles_finite"] == all(p["finite_support"] for p in cp), new["id"])
    check("all1003_frames", nframes == 1003, nframes)
    check("original673_B_frames", nb == 673, nb)
    check("all139976_profile_variants", nprofiles == 139976, nprofiles)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="INDEPENDENT_B_QUERY_GEOMETRY_V1", status="PASS" if not failures else "FAIL",
                  checks=sum(checks.values()), checks_by_kind=dict(checks), failures=failures,
                  failure_count=len(failures), frames=nframes, B_input_frames=nb,
                  profile_variant_rows=nprofiles, maximum_differences=maxima,
                  finite_classification_expression_disagreements=numeric_ambiguities,
                  source_sha256=sha(__file__), factorial_seal_sha256=sha(factorial / "B_QUERY_FACTORIAL_SEAL.json"),
                  diagnostic_only=True, independent_physical_owner="UNVERIFIED",
                  original_strict_roundtrip_assertions_not_modified=True,
                  wall_seconds=time.monotonic() - start)
    write(output / "VERIFICATION.json", result)
    print(result["status"], result["checks"], "checks", result["failure_count"], "failures", flush=True)
    if failures:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    for n in ("output", "factorial", "original-rows", "cache"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    run(a.output, a.factorial, a.original_rows, a.cache)


if __name__ == "__main__":
    main()
