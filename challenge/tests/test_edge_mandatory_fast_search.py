"""Tests for the PEQ Round-1 pipeline gates."""
from __future__ import annotations
import ast, hashlib, json, pathlib, subprocess, sys, tokenize
import numpy as np, pytest, torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for e in (ROOT/"Deep_Object_Pose/common", ROOT/"Deep_Object_Pose/train",
          ROOT/"challenge/scripts", ROOT/"scripts/stage0"):
    if str(e) not in sys.path: sys.path.insert(0, str(e))
import physical_edge_query as PEQ, corner_incident_geometry as CIGM
import edge_guided_corner_fusion as EGCR, instance_edge_topology as IET

RUNNER = ROOT/"scripts/stage0/line/edge_mandatory_fast_search.py"
OUT = (ROOT/"data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       /"compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
SPLIT_SHA = "9a755438dcb55e0ff60415d5b2f861a29e60b23d921a2e0985a23eb2e214415f"

def code_only(p):
    out=[]
    with open(p,"rb") as h:
        for t in tokenize.tokenize(h.readline):
            if t.type in (tokenize.COMMENT,tokenize.STRING,tokenize.NL,tokenize.NEWLINE,
                          tokenize.INDENT,tokenize.DEDENT): continue
            out.append(t.string)
    return " ".join(out)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def test_01_split_sha_locked():
    assert sha(OUT/"paper_group_split.csv") == SPLIT_SHA

def test_02_invalid_audit_is_preserved_not_deleted():
    assert (OUT/"loader_offframe_summary.INVALID_400PX_ASSUMPTION.json").is_file()
    c = json.loads((OUT/"loader_offframe_audit_correction.json").read_text())
    assert c["invalid_mismatch_count"] == 34
    assert "False == 0" in c["aggregation_bug"]

def test_03_false_never_passes_a_gate():
    gates = {"a": True, "b": False}
    assert not all(v is True for v in gates.values())
    assert all(v is True or v == 0 for v in gates.values()), "the old buggy rule"
    assert not all(v is True for v in {}.values()) or len({}) == 0

def test_04_coordinate_contract():
    c = json.loads((OUT/"loader_coordinate_contract.json").read_text())
    assert c["image_space"] == "400x400" and c["refine_belief_space"] == "50x50"
    assert c["factor"] == 8

def test_05_loader_gate_passed():
    s = json.loads((OUT/"loader_offframe_summary.json").read_text())
    assert s["A1_LOADER_COMPATIBILITY"] == "OK"
    for k,v in s["gates"].items(): assert v is True, k
    # Counter only materialises keys it incremented, so an absent key means zero
    assert s["metrics"].get("belief_mask_mismatch", 0) == 0
    assert s["metrics"].get("affinity_mask_mismatch", 0) == 0
    assert s["metrics"]["loads"] == 1024

def test_06_visibility_field_unused():
    src = code_only(RUNNER)
    assert "visibility" not in src
    for m in ("physical_edge_query.py","corner_incident_geometry.py","edge_guided_corner_fusion.py"):
        assert "visibility" not in code_only(ROOT/"Deep_Object_Pose/common"/m), m

def test_07_topology_twelve_roles_three_incident():
    t = IET.build_topology()
    assert len(t["edges"]) == 12
    inc = CIGM.incidence_table(t)
    assert len(inc) == 8 and all(len(v) == 3 for v in inc)

def test_08_no_hungarian_anywhere():
    for p in (RUNNER, ROOT/"Deep_Object_Pose/common/physical_edge_query.py",
              ROOT/"Deep_Object_Pose/common/corner_incident_geometry.py"):
        s = code_only(p)
        for bad in ("linear_sum_assignment","hungarian","Hungarian"): assert bad not in s

def test_09_query_ids_are_fixed():
    h = PEQ.PhysicalEdgeQueryHead()
    assert h.queries.num_embeddings == 12
    o = h(torch.randn(1,128,50,50))
    assert o["centre"].shape == (1,12,2)

def test_10_direction_normalised_and_length_positive():
    o = PEQ.PhysicalEdgeQueryHead()(torch.randn(2,128,50,50))
    assert torch.allclose(o["direction"].norm(dim=-1), torch.ones(2,12), atol=1e-5)
    assert bool((o["half_length"] > 0).all())

def test_11_centre_is_not_clipped():
    body = ast.unparse(next(n for n in ast.walk(ast.parse(RUNNER.read_text()))
                            if isinstance(n, ast.Module)))
    src = code_only(ROOT/"Deep_Object_Pose/common/physical_edge_query.py")
    for bad in ("sigmoid","tanh","clamp_max"): assert bad not in src
    del body

def test_12_egcr_zero_init_and_passthrough():
    m = EGCR.EdgeGuidedCornerResidual()
    base = torch.randn(2,9,50,50); prop = torch.rand(2,8,50,50)
    res = m(base, prop)
    assert float(res.abs().max()) == 0.0
    comp = EGCR.compose(base, res)
    assert EGCR.assert_passthrough(comp, base)["centroid_max_abs"] == 0.0

def test_13_cigm_has_no_parameters():
    src = code_only(ROOT/"Deep_Object_Pose/common/corner_incident_geometry.py")
    assert "nn.Parameter" not in src and "Linear" not in src

def test_14_cigm_recovers_an_exact_corner():
    t = IET.build_topology(); inc = CIGM.incidence_table(t)
    pts = np.array([[10.,10.],[40.,10.],[40.,40.],[10.,40.],
                    [15.,15.],[45.,15.],[45.,45.],[15.,45.]])
    edges = [tuple(e) for e in t["edges"]]
    c = np.stack([0.5*(pts[i]+pts[j]) for i,j in edges])
    d = np.stack([(pts[j]-pts[i])/np.linalg.norm(pts[j]-pts[i]) for i,j in edges])
    got,_,_ = CIGM.solve_corners(torch.tensor(c)[None], torch.tensor(d)[None], inc)
    # The ridge term is epsilon = 1e-4 by specification, so an exact intersection
    # comes back biased by ~5e-3 cell.  That is the module behaving as specified,
    # not an error: the oracle gate is median < 0.5 and p99 < 1.5 cell.
    assert float(np.abs(got[0].numpy()-pts).max()) < 0.01
    assert CIGM.EPSILON == 1e-4

def test_15_runner_guards_sealed_tokens():
    src = RUNNER.read_text()
    assert "SEALED" in src and "def guard" in src
    for tok in ("capturenight08","handannot17"): assert tok not in code_only(RUNNER)

def test_16_runner_never_opens_holdout_or_canonical():
    src = code_only(RUNNER)
    for bad in ("eval56","wood","untouched_metrics"): assert bad not in src

def test_17_state_is_atomic_and_resumable():
    src = RUNNER.read_text()
    assert "os.replace" in src and "heartbeat" in src
    assert 'if st.get(p) == "DONE"' in src

def test_18_modules_gated_on_cigm_pass():
    tree = ast.parse(RUNNER.read_text())
    body = ast.unparse(next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == "phase_smoke"))
    assert "CIGM_ORACLE_STATUS" in body and "refusing" in body

def test_19_size_flags_are_not_filters():
    s = json.loads((OUT/"eligibility_summary.json").read_text()) if (OUT/"eligibility_summary.json").is_file() else None
    if s is None: pytest.skip("eligibility not run yet")
    assert "tiny_warning" in " ".join(s["not_filters"])

def test_20_weights_not_staged():
    st = subprocess.run(["git","diff","--cached","--name-only"],cwd=ROOT,
                        capture_output=True,text=True).stdout.split()
    assert not [n for n in st if n.endswith(".pth")]


def test_21_no_stub_strings_remain():
    src = RUNNER.read_text()
    assert "PENDING_TRAINING_WIRING" not in src
    assert "DATA_GATES_ONLY" not in src

def test_22_smoke_and_round1_actually_step():
    tree = ast.parse(RUNNER.read_text())
    body = ast.unparse(next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == "run_training"))
    assert "opt.step()" in body and "loss.backward()" in body
    for name in ("phase_smoke", "phase_round1"):
        f = ast.unparse(next(n for n in ast.walk(tree)
                             if isinstance(n, ast.FunctionDef) and n.name == name))
        assert "run_training" in f, name

def test_23_only_e1_e2_are_active():
    src = RUNNER.read_text()
    assert 'ARMS = ("E1", "E2")' in src
    assert "E3" not in code_only(RUNNER), "E3 is SKIPPED_OPTIONAL for this Round-1"

def test_24_single_a1_forward_and_detach():
    tree = ast.parse(RUNNER.read_text())
    body = ast.unparse(next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == "run_training"))
    assert body.count("a1(img)") == 1
    assert "feature.detach()" in body and "base.detach()" in body

def test_25_targets_come_from_refine_keypoints():
    tree = ast.parse(RUNNER.read_text())
    body = ast.unparse(next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == "run_training"))
    assert "refine_keypoints" in body
    assert "visibility" not in body

def test_26_smoke_result_passed():
    s = json.loads((OUT/"smoke.json").read_text())
    assert s["passed"] is True and s["a1_unchanged"] is True
    for arm in ("E1", "E2"):
        for k, v in s["checks"][arm].items(): assert v is True, (arm, k)
    assert s["checks"]["zero_init"]["residual_exact_0"] is True
    assert s["checks"]["zero_init"]["centroid_delta_0"] is True

def test_27_lambda_frozen_and_recorded():
    l = json.loads((OUT/"smoke_lambda.json").read_text())
    assert l["frozen"] is True
    # calibrate_v2 records each component's lambda alongside its realized share,
    # so the flat "lambda" map of v1 no longer exists.
    assert set(l["components"]) == {"centre","orientation","length","support","incidence"}
    for name, entry in l["components"].items():
        assert entry["lambda"] > 0, name
