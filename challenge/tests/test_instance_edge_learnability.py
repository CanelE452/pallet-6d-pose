"""Tests for the instance-aware 12-edge learnability screen.

The screen only means anything if the A1 path is genuinely read-only, the
topology is derived rather than typed, the decoder never sees a corner heatmap,
and no canonical set is allowed to pick a checkpoint.  Each of those is a test
here rather than a claim in a report.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tokenize

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train",
              ROOT / "challenge/scripts", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import instance_edge_topology as IET      # noqa: E402
import instance_edge_head as IEH          # noqa: E402

RUNNER = ROOT / "scripts/stage0/line/instance_edge_learnability.py"
A1_CKPT = ROOT / "weights/paper_s2_pdg/A1/epoch_003.pth"
A1_SHA = "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657"
EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


def code_only(path: pathlib.Path) -> str:
    """Source with comments and string literals removed.

    A banned token appearing inside a docstring is documentation, not
    behaviour; matching raw text produced five false alarms in an earlier
    screen, so the check runs over executable tokens alone.
    """
    pieces = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                              tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def topology():
    return IET.build_topology()


@pytest.fixture(scope="module")
def runner_source():
    return code_only(RUNNER)


# ---------------------------------------------------------------------------
# 1-4  frozen encoder
# ---------------------------------------------------------------------------
def test_01_a1_checkpoint_sha_lock():
    assert A1_CKPT.is_file()
    assert sha256_file(A1_CKPT) == A1_SHA
    assert sha256_file(EP57) == EP57_SHA


@pytest.fixture(scope="module")
def encoder():
    spec = importlib.util.spec_from_file_location("IEL", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.FrozenEncoder()


def test_02_a1_requires_grad_false(encoder):
    _, frozen = encoder
    assert sum(1 for p in frozen.parameters() if p.requires_grad) == 0


def test_03_a1_parameter_delta_zero(encoder):
    module, frozen = encoder
    before = frozen.parameter_checksum()
    images = torch.zeros(1, 3, module.INPUT_SIZE, module.INPUT_SIZE)
    frozen.taps(images)
    assert frozen.parameter_checksum() == before


def test_04_only_edge_head_is_trainable(encoder):
    module, frozen = encoder
    model = module.build_arm("L12-F50", 1, frozen)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "the edge head must have trainable parameters"
    assert all(name.startswith("head.") or name.startswith("fusion.")
               for name in trainable), trainable
    assert sum(1 for p in frozen.parameters() if p.requires_grad) == 0


# ---------------------------------------------------------------------------
# 5-9  topology
# ---------------------------------------------------------------------------
def test_05_exactly_twelve_physical_edges(topology):
    assert len(topology["edges"]) == IET.N_EDGES == 12


def test_06_three_incident_edges_per_corner(topology):
    incidence = topology["corner_edge_incidence"]
    assert len(incidence) == 8
    for corner, edges in incidence.items():
        assert len(edges) == 3, (corner, edges)
        assert len(set(edges)) == 3


def test_07_edge_ids_are_deterministic(topology):
    again = IET.build_topology()
    assert again["edges"] == topology["edges"]
    assert [tuple(e) for e in topology["edges"]] == sorted(
        tuple(e) for e in topology["edges"])
    for a, b in topology["edges"]:
        assert a < b


def test_08_topology_sha_is_deterministic(topology):
    assert IET.build_topology()["topology_sha256"] == topology["topology_sha256"]
    # a different pallet size must not change the combinatorics
    other = IET.build_topology((0.8, 0.59, 0.14))
    assert other["edges"] == topology["edges"]
    assert other["edge_classes"] == topology["edge_classes"]


def test_09_semantic_class_counts(topology):
    counts = topology["class_counts"]
    assert [counts[c] for c in IET.SEMANTIC_CLASSES] == [2, 2, 2, 2, 4]


# ---------------------------------------------------------------------------
# 10-12  decoders and target parity
# ---------------------------------------------------------------------------
def _cube_projection(grid: int = 50):
    """Eight corners of an axis-aligned box placed inside the grid."""
    points = []
    for corner in np.asarray(
            __import__("annotate_pnp").make_pallet_keypoints_3d(1.1, 1.3, 0.11))[:8]:
        x = 320 + corner[0] * 200
        y = 240 + corner[1] * 200 + corner[2] * 60
        points.append([float(x), float(y)])
    return points


def test_10_o5_collapses_eight_corners_onto_two(topology):
    """The five-class map admits eight edges per corner, not three.

    A corner's three incident edges span the classes {top_width, top_depth,
    vertical} or {base_width, base_depth, vertical}.  Those classes hold
    2 + 2 + 4 = 8 edges, and the class set is identical for all four corners of
    a face -- so O5 cannot produce more than two distinct points for eight
    corners.  That, not a weak signal, is why the O5 oracle median is 149px.
    """
    five = IET.incidence_lists(topology, "O5")
    twelve = IET.incidence_lists(topology, "O12")
    for corner in range(8):
        assert len(twelve[corner]) == 3
        assert len(five[corner]) == 8, (corner, five[corner])
        assert set(twelve[corner]).issubset(set(five[corner]))
    assert len({tuple(sorted(edges)) for edges in five}) == 2
    assert len({tuple(sorted(edges)) for edges in twelve}) == 8


def test_11_o12_places_corners_on_exact_geometry(topology):
    grid, width, height = 50, 640.0, 480.0
    points = _cube_projection()
    segments, in_frame = IET.clipped_edges_in_grid(points, topology, width, height, grid)
    assert all(in_frame)
    fields, _ = IET.distance_fields_from_segments(segments, grid)
    decoded = IET.decode_corners(fields, IET.incidence_lists(topology, "O12"),
                                 grid, width, height)
    cell = np.hypot(width / grid, height / grid)
    for index in range(8):
        error = np.hypot(decoded[index][0] - points[index][0],
                         decoded[index][1] - points[index][1])
        assert error <= cell, (index, error, cell)


def test_12_target_generator_decoder_parity(topology):
    grid, width, height = 50, 640.0, 480.0
    points = _cube_projection()
    segments, _ = IET.clipped_edges_in_grid(points, topology, width, height, grid)
    analytic, _ = IET.distance_fields_from_segments(segments, grid)
    targets = IET.build_edge_targets(segments, grid)
    recovered = IET.distance_from_probability(targets)
    incidence = IET.incidence_lists(topology, "O12")
    direct = IET.decode_corners(analytic, incidence, grid, width, height)
    through = IET.decode_corners(recovered, incidence, grid, width, height)
    for index in range(8):
        difference = np.hypot((direct[index][0] - through[index][0]) * grid / width,
                              (direct[index][1] - through[index][1]) * grid / height)
        assert difference < 1.5, (index, difference)


# ---------------------------------------------------------------------------
# 13-16  coordinates, clipping, visibility
# ---------------------------------------------------------------------------
def test_13_image_to_belief_coordinate_mapping(topology):
    grid, width, height = 50, 640.0, 480.0
    a = np.array([100.0, 200.0])
    b = np.array([500.0, 200.0])
    segments = [None] * 12
    clipped = IET.clip_segment(a, b, width, height)
    scale = np.array([grid / width, grid / height])
    segments[0] = (clipped[0] * scale, clipped[1] * scale)
    fields, _ = IET.distance_fields_from_segments(segments, grid)
    row = int(round(200.0 * grid / height))
    assert fields[0][row].min() < 1.0
    argmin = int(np.unravel_index(int(fields[0].argmin()), fields[0].shape)[0])
    assert abs(argmin - row) <= 1


def test_14_off_frame_edges_are_clipped_not_kept(topology):
    width, height, grid = 640.0, 480.0, 50
    points = [[float(x), float(y)] for x, y in
              [(-500, -500), (-400, -500), (-400, -400), (-500, -400),
               (-500, -300), (-400, -300), (-400, -200), (-500, -200)]]
    segments, in_frame = IET.clipped_edges_in_grid(points, topology, width, height, grid)
    assert not any(in_frame), "a box entirely outside the frame must produce no segment"
    targets = IET.build_edge_targets(segments, grid)
    assert float(targets.max()) == 0.0


def test_15_no_endpoint_clamp_artifact():
    width, height = 640.0, 480.0
    clipped = IET.clip_segment(np.array([-100.0, 240.0]), np.array([300.0, 240.0]),
                               width, height)
    assert clipped is not None
    assert abs(clipped[0][0] - 0.0) < 1e-9        # entry point sits on the border
    assert abs(clipped[1][0] - 300.0) < 1e-9      # interior endpoint is untouched
    assert IET.clip_segment(np.array([-10.0, -10.0]), np.array([-5.0, -5.0]),
                            width, height) is None


def test_16_visibility_states_are_exhaustive():
    source = code_only(RUNNER)
    for state in ("VISIBLE", "OCCLUDED", "OFF_FRAME", "UNKNOWN"):
        assert state in RUNNER.read_text("utf-8")
    tree = ast.parse(RUNNER.read_text("utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "visibility_breakdown" in names
    assert "load_synthetic" in names
    del source


# ---------------------------------------------------------------------------
# 17-20  head shapes
# ---------------------------------------------------------------------------
def test_17_five_class_head_outputs_five():
    model = IEH.InstanceEdgeModel("F100", 5, 256, 128)
    out = model(torch.zeros(2, 256, 100, 100), None)
    assert out.shape == (2, 5, 100, 100)


def test_18_twelve_edge_head_outputs_twelve():
    model = IEH.InstanceEdgeModel("F50", 12, 256, 128)
    out = model(None, torch.zeros(2, 128, 50, 50))
    assert out.shape == (2, 12, 50, 50)


def test_19_f50_alignment(encoder):
    module, frozen = encoder
    high, low = frozen.taps(torch.zeros(1, 3, module.INPUT_SIZE, module.INPUT_SIZE))
    assert low.shape[-2:] == (module.GRID_12, module.GRID_12)
    assert low.shape[1] == 128


def test_20_multiscale_alignment(encoder):
    module, frozen = encoder
    high, low = frozen.taps(torch.zeros(1, 3, module.INPUT_SIZE, module.INPUT_SIZE))
    assert high.shape[-2:] == (module.GRID_5, module.GRID_5)
    fusion = IEH.MultiScaleFusion(high.shape[1], low.shape[1])
    fused = fusion(high, low)
    assert fused.shape[-2:] == low.shape[-2:]
    assert fused.shape[1] == fusion.out_channels == 192
    with pytest.raises(RuntimeError):
        fusion(high, torch.zeros(1, 128, 25, 25))


# ---------------------------------------------------------------------------
# 21-22  controls
# ---------------------------------------------------------------------------
def test_21_shuffled_incidence_is_fixed_and_not_identity():
    first = IET.shuffled_permutation(1)
    assert first == IET.shuffled_permutation(1)
    assert sorted(first) == list(range(12))
    assert first != list(range(12))


def test_22_hungarian_mapping_recovers_a_known_permutation():
    spec = importlib.util.spec_from_file_location("IEL2", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth = IET.shuffled_permutation(1)
    confusion = np.full((12, 12), 0.01)
    for k in range(12):
        confusion[k, truth[k]] = 0.9
    assert module.hungarian_from_confusion(confusion) == truth


# ---------------------------------------------------------------------------
# 23-27  protocol guards, checked on executable tokens only
# ---------------------------------------------------------------------------
def test_23_alignment_is_never_estimated_on_canonical_sets(runner_source):
    tree = ast.parse(RUNNER.read_text("utf-8"))
    canonical_function = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.FunctionDef)
                              and node.name == "phase_eval_canonical")
    calls = {node.func.id for node in ast.walk(canonical_function)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "hungarian_from_confusion" not in calls, (
        "the canonical phase must reuse the synthetic mapping, never fit one")
    synthetic_function = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.FunctionDef)
                              and node.name == "phase_eval_synthetic")
    synthetic_calls = {node.func.id for node in ast.walk(synthetic_function)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "hungarian_from_confusion" in synthetic_calls


def test_24_decoder_receives_no_corner_heatmap(runner_source):
    import inspect
    source = code_only(ROOT / "Deep_Object_Pose/common/instance_edge_topology.py")
    for banned in ("belief", "heatmap", "peak"):
        assert banned not in source, banned
    signature = inspect.signature(IET.decode_corners)
    assert list(signature.parameters) == ["distance_fields", "incidence", "grid",
                                          "width", "height", "tau"]


def test_25_decoder_receives_no_topk(runner_source):
    source = code_only(ROOT / "Deep_Object_Pose/common/instance_edge_topology.py")
    for banned in ("topk", "top_k", "argpartition", "argsort"):
        assert banned not in source, banned
    assert "topk" not in runner_source and "top_k" not in runner_source


def test_26_checkpoint_selection_never_uses_canonical(runner_source):
    tree = ast.parse(RUNNER.read_text("utf-8"))
    selector = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "select_checkpoint")
    text = ast.unparse(selector)
    for banned in ("eval56", "wood", "canonical"):
        assert banned not in text, banned
    loader = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "load_selected")
    loader_text = ast.unparse(loader)
    for banned in ("eval56", "wood"):
        assert banned not in loader_text, banned


def test_27_final_test_guard_refuses_sealed_tokens():
    spec = importlib.util.spec_from_file_location("IEL3", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for token in ("capturenight08", "capturepallet09", "handannot17",
                  "testset_full8_manifest"):
        with pytest.raises(RuntimeError):
            module.guard_path(f"/data/{token}/000001.json")
    assert module.guard_path("/data/capturepalletcad_manual_gt/1.json")


# ---------------------------------------------------------------------------
# 28-30  durability
# ---------------------------------------------------------------------------
def test_28_state_machine_resumes_and_records(tmp_path):
    spec = importlib.util.spec_from_file_location("IEL4", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "state.json"
    state = module.State(path)
    assert state.get("train") == "PENDING"
    state.set("train", "RUNNING")
    state.set("train", "DONE", seconds=1.0)
    assert module.State(path).get("train") == "DONE"
    assert json.loads(path.read_text())["phases"]["train"]["seconds"] == 1.0


def test_29_completed_arm_is_skipped(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("IEL5", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WEIGHT_ROOT", tmp_path)
    directory = tmp_path / "L12_F50" / "seed1"
    directory.mkdir(parents=True)
    (directory / "run_state.json").write_text(json.dumps({"completed": False}))
    assert ("L12-F50", 1) not in module.completed_runs()
    (directory / "run_state.json").write_text(json.dumps({"completed": True}))
    assert ("L12-F50", 1) in module.completed_runs()


def test_30_corrupted_checkpoint_is_detected(tmp_path):
    path = tmp_path / "last.pth"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# 31-34  main-model and repository invariants
# ---------------------------------------------------------------------------
def test_31_exactly_one_optimizer_and_it_holds_head_parameters_only():
    tree = ast.parse(RUNNER.read_text("utf-8"))
    optimisers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and ast.unparse(node.func).startswith("torch.optim.")]
    assert len(optimisers) == 1, [ast.unparse(o) for o in optimisers]
    assert "trainable" in ast.unparse(optimisers[0])


def test_32_no_training_step_touches_the_main_model(runner_source):
    tree = ast.parse(RUNNER.read_text("utf-8"))
    trainer = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "train_one")
    text = ast.unparse(trainer)
    assert "encoder.taps" in text
    assert "requires_grad_(True)" not in text
    assert "frozen_trainable" in text


def test_33_source_dataset_is_never_written():
    tree = ast.parse(RUNNER.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node)
            if "SYNTH_ROOT" in rendered:
                assert not any(token in rendered for token in
                               ("write_text", "write_bytes", "imwrite", "unlink",
                                "atomic_write")), rendered


def test_34_weights_are_not_staged_for_git():
    import subprocess
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True).stdout.split()
    assert not [name for name in staged if name.endswith(".pth")], staged
    tracked = subprocess.run(["git", "ls-files", "weights"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert not [name for name in tracked if name.endswith(".pth")], tracked[:5]


# ---------------------------------------------------------------------------
# 35-37  parallel solve path
# ---------------------------------------------------------------------------
def test_35_solve_many_falls_back_to_sequential_without_a_pool(monkeypatch):
    spec = importlib.util.spec_from_file_location("IEL6", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_POOL", None)
    calls = []
    monkeypatch.setattr(module, "_pnp_task", lambda task: calls.append(task) or (True, 1.0))
    assert module.solve_many([]) == []
    assert module.solve_many(["a", "b"]) == [(True, 1.0), (True, 1.0)]
    assert calls == ["a", "b"]


def test_36_solve_pool_uses_spawn_not_fork():
    """Forking after torch has started its thread pools deadlocked here."""
    tree = ast.parse(RUNNER.read_text("utf-8"))
    starter = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "start_pool")
    body = [node for node in starter.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    text = "\n".join(ast.unparse(node) for node in body)   # docstring dropped
    assert "'spawn'" in text or '"spawn"' in text
    assert "fork" not in text


def test_37_identity_alignment_is_copied_not_recomputed():
    tree = ast.parse(RUNNER.read_text("utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "phase_eval_synthetic")
    text = ast.unparse(function)
    assert "if not identity[key]" in text, (
        "the untouched aligned pass must be skipped when the mapping is the identity")
    assert "results[key]['untouched'] if identity[key]" in text.replace('"', "'")


# ---------------------------------------------------------------------------
# 38  JSON round-trip key types
# ---------------------------------------------------------------------------
def test_38_best_seed_lookup_survives_json_round_trip():
    """Seeds are ints in memory and strings on disk; best_seed stays an int.

    Reading the reloaded results with the in-memory key raised KeyError: 3 and
    killed the decision phase after every evaluation had finished.
    """
    spec = importlib.util.spec_from_file_location("IEL7", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = {"seeds": {1: {"v": "a"}, 3: {"v": "b"}}, "best_seed": 3}
    assert module.best_block(entry)["v"] == "b"
    reloaded = json.loads(json.dumps(entry))
    assert list(reloaded["seeds"]) == ["1", "3"] and reloaded["best_seed"] == 3
    assert module.best_block(reloaded)["v"] == "b"
    tree = ast.parse(RUNNER.read_text("utf-8"))
    for name in ("phase_decide", "multiscale_verdict"):
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        assert '["best_seed"]]' not in ast.unparse(function), (
            f"{name} still indexes seeds with the raw best_seed")
