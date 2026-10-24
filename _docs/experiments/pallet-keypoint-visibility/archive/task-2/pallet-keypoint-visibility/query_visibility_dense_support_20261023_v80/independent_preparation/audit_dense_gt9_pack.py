"""GT-only saved-array audit against V79, without source image/mask decoding."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
A = R.parent / "query_visibility_natural_occlusion_audit_20261023_v79"


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(record):
    assert bind(record["path"]) == record


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fingerprint(valid, full, visible):
    payload = np.stack([valid, full & valid, visible & valid])
    return hashlib.sha256(b"V79_MAP_V1|grid9" + np.packbits(payload, bitorder="little").tobytes()).hexdigest()


def generated():
    producer = module(A / "coverage/coverage_math.py", "source_gt_math")
    scalar = module(A / "independent_preparation/independent_grid_math.py", "independent_grid_math")
    full = np.zeros((9, 11), bool)
    full[1:8, 2:10] = True
    visible = full.copy()
    visible[3:7, 5:8] = False
    queries = [[4.5, 4.5], [0., 0.], [-1., -1.], [-.25, 3.],
               [10.49, 8.49], [10.5, 8.5], [11., 9.], [float("nan"), 2.]]
    cases = 0
    for side in [9., 16., 1.e-6]:
        for query in queries:
            q8 = np.tile(query, (8, 1))
            grid = producer.coordinates(q8, side, 9)
            cached = producer.cached_formula(grid, full.shape, 9, q8)
            counts, _, pix, native = producer.count_grid(full, visible, q8, side, 9, cached)
            valid = native[0] & cached[0]
            own_grid = np.asarray(scalar.grid_scalar(query, side, 9))
            assert np.array_equal(grid[0], own_grid, equal_nan=True)
            replay = scalar.replay_masks(full, visible, query, side, 9, cached[0], "native9")
            expected = replay["counts"]["intersection"]
            assert counts["class_counts"][0, 2].tolist() == [expected[k] for k in ["V", "C", "O"]]
            f = np.zeros((9, 9), bool)
            v = f.copy()
            f[valid] = full[pix[0, ..., 1][valid], pix[0, ..., 0][valid]]
            v[valid] = visible[pix[0, ..., 1][valid], pix[0, ..., 0][valid]]
            assert fingerprint(valid, f, v) == counts["gt_map_intersection_sha256"][0].decode()
            if np.array_equal(query, [-1., -1.]) or not np.isfinite(query).all():
                assert not valid.any() and not f.any() and not v.any()
            cases += 1
    write(R / "independent_preparation/GENERATED_DENSE_GT9_INDEPENDENT_TESTS.json", {
        "PASS": True, "cases": cases, "boundary_half_tie_missing_nonfinite_cases": True,
        "literal_FP64_grid_scalar_and_producer_exact": True,
        "class_counts_and_little_endian_fingerprints_exact": True,
        "all_inputs_generated": True, "source_mask_RGB_model_current_fit_PnP_reads": 0,
        "audit_code": bind(__file__),
        "independent_scalar_math": bind(A / "independent_preparation/independent_grid_math.py"),
        "producer_math": bind(A / "coverage/coverage_math.py")})


def freeze():
    generated_path = R / "independent_preparation/GENERATED_DENSE_GT9_INDEPENDENT_TESTS.json"
    generated_report = json.loads(generated_path.read_text())
    assert generated_report["PASS"] and generated_report["audit_code"] == bind(__file__)
    write(R / "independent_preparation/AUDIT_POLICY_BEFORE_DENSE_OUTPUT_READS.json", {
        "frozen_before_dense_output_reads": True, "audit_code": bind(__file__),
        "generated_tests": bind(generated_path),
        "GT_protocol": bind(R / "gt_preparation/GT_BUILD_PROTOCOL.json"),
        "V79_fixed_scalar_audit": bind(A / "independent_audits/COVERAGE_AGGREGATE_AND_SCALAR_AUDIT.json"),
        "plan": "All80016 nominal9 valid/full/visible bit-packed fingerprints and V/C/O counts exact against frozen V79 intersection; same IDs/source/split/domain; subset/validity/paired monotonicity; bind existing independently decoded fixed8 rows rather than decode again.",
        "numerics": "Literal FP64 coordinates, floor+.5 NN and original missing guard inherited unchanged. Discrete bool maps/fingerprints/counts exact; no new float32 equivalence or numeric tolerance gate.",
        "semantics": "M0 negative only expected-support auxiliary; visible-given-M0 loss ignores outsideM0. Invalid9 ignored, never target absence or semanticE/S. Original primary unchanged.",
        "no_new_source_decode_model_forward_fit_current_PnP": True})


def run():
    policy_path = R / "independent_preparation/AUDIT_POLICY_BEFORE_DENSE_OUTPUT_READS.json"
    policy = json.loads(policy_path.read_text())
    check(policy["audit_code"])
    check(policy["generated_tests"])
    check(policy["GT_protocol"])
    check(policy["V79_fixed_scalar_audit"])
    protocol = json.loads(Path(policy["GT_protocol"]["path"]).read_text())
    for binding in protocol["bindings"]:
        check(binding)
    seal_path = R / "dense_gt/SEAL.json"
    seal = json.loads(seal_path.read_text())
    assert seal["completed"]
    for key in ["protocol", "targets", "results"]:
        check(seal[key])
    assert seal["protocol"] == policy["GT_protocol"]
    with np.load(seal["targets"]["path"], allow_pickle=False) as data:
        actual = {k: data[k] for k in data.files}
    assert set(actual) == {"valid9", "expected_support_M0", "visible_target_M4", "id", "source", "split", "domain"}
    with np.load(A / "coverage/results/COVERAGE_PER_QUERY.npz", allow_pickle=False) as data:
        reference = {k: data[k] for k in ["id", "source", "split", "domain", "original_row_index", "query_missing",
            "class_counts", "valid_counts", "gt_map_intersection_sha256"]}
    for key in ["id", "source", "split", "domain"]:
        assert np.array_equal(actual[key], reference[key])
    valid = actual["valid9"]
    full = actual["expected_support_M0"]
    visible = actual["visible_target_M4"]
    for array in [valid, full, visible]:
        assert array.shape == (10002, 8, 9, 9) and array.dtype == np.bool_
    assert not (full & ~valid).any()
    assert not (visible & ~full).any()
    assert not valid[reference["query_missing"]].any()
    counts = np.stack([visible.sum((2, 3)), (full & ~visible).sum((2, 3)),
                       (valid & ~full).sum((2, 3))], axis=-1)
    assert np.array_equal(counts, reference["class_counts"][:, :, 0, 2])
    assert np.array_equal(valid.sum((2, 3)), reference["valid_counts"][:, :, 0, 6])
    fingerprint_mismatches = 0
    for i in range(10002):
        for k in range(8):
            target = reference["gt_map_intersection_sha256"][i, k, 0].decode()
            fingerprint_mismatches += int(fingerprint(valid[i, k], full[i, k], visible[i, k]) != target)
    assert fingerprint_mismatches == 0
    for i in range(5203, 10002):
        parent = int(reference["original_row_index"][i])
        assert 0 <= parent < 5203
        assert np.array_equal(valid[i], valid[parent])
        assert np.array_equal(full[i], full[parent])
        assert not (visible[i] & ~visible[parent]).any()
    summary = json.loads(Path(seal["results"]["path"]).read_text())
    totals = {"grid_positions": int(valid.sum()), "visible": int(visible.sum()),
              "covered": int((full & ~visible).sum()), "outside": int((valid & ~full).sum())}
    for key, value in totals.items():
        assert summary[key] == value
    assert summary["PASS"] and summary["rows"] == 10002 and summary["queries"] == 80016
    previous = json.loads(Path(policy["V79_fixed_scalar_audit"]["path"]).read_text())
    assert previous["PASS"] and previous["scalar_cells_in_both_grids"] == 595008
    fixed_rows = previous["fixed_scalar_rows"]
    assert len(fixed_rows) == 8
    for row in fixed_rows:
        i = row["frame_index"]
        assert actual["id"][i] == row["id"]
        assert row["all8queries_9_and96_count_unique_fingerprint_qcenter_exact"]
        assert row["actual_cached_canvas_and_literal_formula_exact"]
    strata = []
    for domain in [0, 1]:
        for source in ["SYNTHETIC", "YCB_PUBLIC_REAL"]:
            for split in ["train", "val"]:
                filt = (actual["domain"] == domain) & (actual["source"] == source) & (actual["split"] == split)
                eligible = valid[filt]
                f = full[filt]
                v = visible[filt]
                strata.append({"domain": domain, "source": source, "split": split,
                    "target_rows": int(filt.sum()), "queries": int(filt.sum()) * 8,
                    "A_eligible_cells": int(eligible.sum()), "A_positive_cells": int(f.sum()),
                    "A_negative_cells": int((eligible & ~f).sum()),
                    "visible_given_M0_eligible_cells": int(f.sum()), "visible_positive_cells": int(v.sum()),
                    "visible_negative_covered_cells": int((f & ~v).sum()),
                    "queries_no_eligible_cells": int(np.count_nonzero(eligible.sum((2, 3)) == 0))})
    report = {"PASS": True, "completed": True, "audit_code": bind(__file__), "frozen_policy": bind(policy_path),
        "GT_seal": bind(seal_path), "GT_pack": seal["targets"],
        "all_ID_source_split_domain_exact": True, "all80016_fingerprints_exact": True,
        "fingerprint_mismatches": fingerprint_mismatches, "all80016_class_and_valid_counts_exact": True,
        "all4799_pairs_valid_M0_unchanged_and_M4_after_subset": True,
        "all_M4_subset_M0_subset_valid": True, "missing_query_valid_cell_count": int(valid[reference["query_missing"]].sum()),
        "totals": totals, "strata": strata,
        "independent_source_scalar_evidence": {"bound_prior_audit": policy["V79_fixed_scalar_audit"],
            "fixed_rows": fixed_rows, "native9_independently_replayed_cells": 8 * 8 * 81,
            "prior_both_grid_scalar_cells": 595008, "new_source_mask_decodes": 0},
        "auxiliary_GT_only_semantics": policy["semantics"],
        "limitations": ["Checks discrete nominal9 GT anchors; does not establish exact RGB branch receptive-field pixel phase or human semantic D/E/S gold.",
            "Independent scalar masks/cached-valid replay is inherited from the sealed fixed8 V79 rows. This audit avoids repeating all source decodes as requested.",
            "No source/performance admission gate changed and no primary labels/inputs altered."],
        "RGB_feature_model_forward_fit_current_PnP": 0}
    out = R / "independent_gt_audit/INDEPENDENT_DENSE_GT9_AUDIT.json"
    out.parent.mkdir(exist_ok=True)
    write(out, report)
    print(json.dumps({"PASS": True, "report": bind(out), "totals": totals}))


if __name__ == "__main__":
    {"generated": generated, "freeze": freeze, "run": run}[sys.argv[1]]()
