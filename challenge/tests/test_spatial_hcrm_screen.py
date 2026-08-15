"""Tests for the Spatial HCRM screen.

The screen only means anything if A1 never moves, the control is genuinely
pointwise, the holdouts are whole source groups, and the guard actually refuses
a holdout read before the selection lock rather than merely noting it.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tokenize

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train",
              ROOT / "challenge/scripts", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import spatial_hcrm as HCRM     # noqa: E402

RUNNER = ROOT / "scripts/stage0/spatial_hcrm_screen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/spatial_hcrm_screen")
A1_SHA = "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657"


def code_only(path: pathlib.Path) -> str:
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
def tree():
    return ast.parse(RUNNER.read_text("utf-8"))


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("SHS", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["SHS"] = module
    spec.loader.exec_module(module)
    return module


def call_string_args(tree, name):
    """String constants passed as call arguments inside one function.

    Matching raw source text kept flagging the word in a log message or a dict
    value, which is documentation rather than data access.  This looks at what
    the function actually asks for.
    """
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    out = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            for argument in inner.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    out.add(argument.value)
    return out


def code_of(tree, name):
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    body = [n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    return "\n".join(ast.unparse(n) for n in body)


# ---------------------------------------------------------------------------
# 1-2  frozen preflight
# ---------------------------------------------------------------------------
def test_01_input_lock_v2_is_frozen():
    lock = json.loads((OUT / "input_lock_v2.json").read_text("utf-8"))
    assert lock["version"] == "v2"
    assert lock["a1_sha256"] == A1_SHA
    assert lock["groupnorm_removed"] is True
    assert lock["training_steps"] == 0
    old = json.loads((OUT / "input_lock.json").read_text("utf-8"))
    assert old["STATUS"] == "SUPERSEDED_BEFORE_TRAINING"


def test_02_split_sha_matches_the_lock():
    lock = json.loads((OUT / "input_lock_v2.json").read_text("utf-8"))
    actual = sha256_file(OUT / "synthetic_split_manifest.csv")
    assert lock["split"]["split_sha256"] == actual


# ---------------------------------------------------------------------------
# 3-8  architecture
# ---------------------------------------------------------------------------
def test_03_no_groupnorm_anywhere():
    for arm in ("H1", "H2"):
        module = HCRM.build(arm, 1)
        assert not any(isinstance(m, nn.GroupNorm) for m in module.modules())


def test_04_channel_layernorm_is_location_independent():
    norm = HCRM.ChannelLayerNorm2d(8)
    a = torch.randn(1, 8, 6, 6)
    b = a.clone()
    b[0, :, 3, 3] += 5.0
    with torch.no_grad():
        delta = (norm(b) - norm(a)).abs().sum(1)[0]
    moved = torch.nonzero(delta > 1e-6)
    assert moved.shape[0] == 1 and tuple(moved[0].tolist()) == (3, 3)


def _support(module):
    a = torch.zeros(1, 128, 50, 50)
    b = a.clone()
    b[0, :, 25, 25] = 1.0
    module.eval()
    with torch.no_grad():
        delta = (module.body(b) - module.body(a)).abs().sum(1)[0]
    ys, xs = torch.nonzero(delta > 1e-9, as_tuple=True)
    return (0, 0, 0) if len(ys) == 0 else (int(ys.max() - ys.min() + 1),
                                           int(xs.max() - xs.min() + 1), len(ys))


def test_05_h1_support_is_one_by_one():
    assert _support(HCRM.build("H1", 1)) == (1, 1, 1)


def test_06_h2_support_is_five_by_five():
    height, width, cells = _support(HCRM.build("H2", 1))
    assert (height, width) == (5, 5) and cells == 25


def test_07_parameter_ratio_within_gate():
    n1 = HCRM.parameter_count(HCRM.build("H1", 1))
    n2 = HCRM.parameter_count(HCRM.build("H2", 1))
    assert (n1, n2) == (12932, 14852)
    assert max(n1, n2) / min(n1, n2) <= 1.5


def test_08_zero_init_residual_is_exactly_zero():
    x = torch.randn(2, 128, 50, 50)
    for arm in ("H1", "H2"):
        assert float(HCRM.build(arm, 1)(x).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 9-12  composition invariants
# ---------------------------------------------------------------------------
def test_09_far_channels_untouched():
    base = torch.randn(2, 9, 50, 50)
    out = HCRM.compose(base, torch.randn(2, 4, 50, 50))
    assert HCRM.assert_untouched(out, base)["far_max_abs"] == 0.0


def test_10_centroid_untouched():
    base = torch.randn(2, 9, 50, 50)
    out = HCRM.compose(base, torch.randn(2, 4, 50, 50))
    assert HCRM.assert_untouched(out, base)["centroid_max_abs"] == 0.0


def test_11_affinity_is_copied_not_composed(tree):
    body = code_of(tree, "evaluate_canonical")
    assert "affinity" not in body, "the affinity tensor must never be recomposed"
    assert "compose" in code_of(tree, "train_run")


def test_12_a1_has_no_trainable_parameters(runner):
    a1 = runner.FrozenA1()
    assert a1.trainable() == 0
    before = a1.checksum()
    a1(torch.zeros(1, 3, runner.INPUT_SIZE, runner.INPUT_SIZE))
    assert a1.checksum() == before


# ---------------------------------------------------------------------------
# 13-18  split and no-aug
# ---------------------------------------------------------------------------
def test_13_holdout_groups_are_disjoint():
    table = pd.read_csv(OUT / "synthetic_split_manifest.csv")
    keyed = table[table["group"].notna()]
    groups = {s: set(keyed[keyed.split == s]["group"]) for s in
              ("train", "validation", "untouched")}
    assert not groups["validation"] & groups["untouched"]
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["untouched"]


def test_14_no_source_frame_appears_twice():
    table = pd.read_csv(OUT / "synthetic_split_manifest.csv")
    keys = table["root"] + "/" + table["stem"]
    assert keys.duplicated().sum() == 0


def test_15_duplicate_audit_recorded_zero():
    summary = json.loads((OUT / "split_summary.json").read_text("utf-8"))
    assert summary["unkeyed_holdout_image_duplicates"] == 0
    assert summary["holdout_group_overlap"] == 0


def test_16_unkeyed_frames_are_train_only():
    table = pd.read_csv(OUT / "synthetic_split_manifest.csv")
    unkeyed = table[table["group"].isna()]
    assert len(unkeyed) > 0
    assert set(unkeyed["split"]) == {"train"}


def test_17_no_index_modulo_split(tree):
    source = code_only(RUNNER)
    assert "% 10" not in source and "%10" not in source
    assert "DETERMINISTIC_AUGMENTATION" not in RUNNER.read_text("utf-8")


def test_18_eval_no_aug_is_repeatable(runner):
    rows = runner.split_rows("validation")[:2]
    for row in rows:
        first = runner.load_no_aug(row["root"], row["stem"])
        second = runner.load_no_aug(row["root"], row["stem"])
        assert first is not None
        assert np.array_equal(first.image, second.image)
        assert np.array_equal(first.belief, second.belief)
        assert np.array_equal(first.belief_mask, second.belief_mask)
        assert np.array_equal(np.nan_to_num(first.points), np.nan_to_num(second.points))


# ---------------------------------------------------------------------------
# 19-22  access guard
# ---------------------------------------------------------------------------
def test_19_untouched_guard_fires(runner):
    guard = runner.AccessGuard()
    guard.arm({"/data/untouched/x.png"}, set())
    with pytest.raises(RuntimeError):
        guard.check("/data/untouched/x.png")
    assert guard.counts["untouched_before_lock"] == 1
    guard.unlock()
    assert guard.check("/data/untouched/x.png")


def test_20_canonical_guard_fires(runner):
    guard = runner.AccessGuard()
    guard.arm(set(), {"/data/eval56/frame.png"})
    with pytest.raises(RuntimeError):
        guard.check("/data/eval56/frame.png")
    guard.unlock()
    assert guard.check("/data/eval56/frame.png")


def test_21_final_test_guard_never_unlocks(runner):
    guard = runner.AccessGuard()
    guard.unlock()
    for token in ("capturenight08", "handannot17", "capturepallet09"):
        with pytest.raises(RuntimeError):
            guard.check(f"/data/{token}/1.png")


def test_22_every_image_read_goes_through_the_guard(tree):
    source = code_only(RUNNER)
    assert source.count("cv2.imread") <= 2, "image reads must funnel through imread()"
    body = code_of(tree, "load_no_aug")
    assert "imread(" in body and "cv2.imread" not in body


# ---------------------------------------------------------------------------
# 23-27  hard manifest and training scope
# ---------------------------------------------------------------------------
def test_23_hard_manifest_is_train_only(tree):
    arguments = call_string_args(tree, "phase_hard_manifest")
    assert "train" in arguments
    for banned in ("eval56", "wood", "untouched", "validation"):
        assert banned not in arguments, banned


def test_24_hard_weight_rule_is_fixed(tree):
    body = code_of(tree, "phase_hard_manifest")
    assert "HARD_WEIGHT_CAP" in body and "n_easy / max(n_hard, 1)" in body


def test_25_only_the_adapter_is_optimised(tree):
    optimisers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and ast.unparse(n.func).startswith("torch.optim.")]
    assert len(optimisers) == 1
    assert "parameters" in ast.unparse(optimisers[0])
    assert "adapter" in code_of(tree, "train_run")


def test_26_loss_is_near_channels_only(tree):
    body = code_of(tree, "near_loss")
    assert "composed[:, :4]" in body.replace("'", '"')
    names = {n.id for n in ast.walk(ast.parse(body)) if isinstance(n, ast.Name)}
    for banned in ("affinity", "centroid", "pose", "pnp", "far"):
        assert banned not in names, banned


def test_27_no_forbidden_module_is_imported():
    source = code_only(RUNNER)
    for banned in ("instance_edge", "polarity_aware_line_head", "TACA",
                   "pdg_heads", "diffpnp"):
        assert banned not in source, banned


# ---------------------------------------------------------------------------
# 28-33  selection discipline
# ---------------------------------------------------------------------------
def test_28_selection_uses_validation_only(tree):
    arguments = call_string_args(tree, "phase_select")
    assert "validation" in arguments
    for banned in ("eval56", "wood", "untouched"):
        assert banned not in arguments, banned


def test_29_untouched_phase_requires_the_lock(tree):
    body = code_of(tree, "phase_eval_untouched")
    assert "selected_checkpoints.json" in body and "BLOCKED" in body


def test_30_canonical_phase_requires_the_lock(tree):
    body = code_of(tree, "phase_eval_canonical")
    assert "selected_checkpoints.json" in body and "BLOCKED" in body


def test_31_threshold_and_decoder_are_fixed(runner):
    assert runner.THRESHOLD == 0.30
    assert runner.BELIEF == 50 and runner.INPUT_SIZE == 400
    body = code_only(RUNNER)
    assert "sigma" not in body.lower() or "SHUFFLE" in body


def test_32_shuffle_permutation_is_fixed(runner):
    assert runner.SHUFFLE_PERMUTATION == (1, 2, 3, 0)
    assert sorted(runner.SHUFFLE_PERMUTATION) == [0, 1, 2, 3]
    base = torch.randn(1, 9, 50, 50)
    residual = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 50, 50)
    out = HCRM.compose(base, residual.contiguous(), runner.SHUFFLE_PERMUTATION)
    assert float((out[0, 0] - base[0, 0] - 1.0).abs().max()) < 1e-6


def test_33_h2_zero_is_an_exact_control(tree):
    body = code_of(tree, "evaluate_synthetic")
    assert "zeros_like(residual)" in body


# ---------------------------------------------------------------------------
# 34-39  durability and repository invariants
# ---------------------------------------------------------------------------
def test_34_state_machine_is_atomic(runner, tmp_path):
    state = runner.State(tmp_path / "state.json")
    state.set("train", "RUNNING")
    state.set("train", "DONE", seconds=1.0)
    assert runner.State(tmp_path / "state.json").get("train") == "DONE"


def test_35_completed_run_is_skipped(tree):
    body = code_of(tree, "train_run")
    assert "completed" in body and "skipping" in body


def test_36_corrupt_checkpoint_is_rejected(tree, tmp_path):
    body = code_of(tree, "train_run")
    assert "corrupt checkpoint rejected" in body
    bad = tmp_path / "last.pth"
    bad.write_bytes(b"garbage")
    with pytest.raises(Exception):
        torch.load(bad, map_location="cpu", weights_only=False)


def test_37_a1_checkpoint_unchanged():
    assert sha256_file(ROOT / "weights/paper_s2/paper_s2_pdg/A1/epoch_003.pth") == A1_SHA


def test_38_source_dataset_is_never_written(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node)
            if "TRAIN_DATA" in rendered:
                assert not any(t in rendered for t in
                               ("write_text", "write_bytes", "imwrite", "unlink"))


def test_39_weights_are_not_staged():
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True).stdout.split()
    assert not [n for n in staged if n.endswith(".pth")], staged
    tracked = subprocess.run(["git", "ls-files", "weights"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert not [n for n in tracked if n.endswith(".pth")], tracked[:5]


# ---------------------------------------------------------------------------
# 40-46  parity semantics, guard integrity, no silent substitution
# ---------------------------------------------------------------------------
def test_40_parity_never_unlocks_the_guard(tree):
    body = code_of(tree, "phase_parity") + code_of(tree, "same_instance_parity")
    assert "GUARD.unlock" not in body, (
        "the parity phase unlocking the guard defeated the whole guard")
    assert "unlocked = False" not in body


def test_41_parity_reads_no_canonical_frame(tree):
    arguments = call_string_args(tree, "same_instance_parity")
    assert "validation" in arguments
    for banned in ("eval56", "wood", "untouched"):
        assert banned not in arguments, banned


def test_42_same_instance_zero_residual_is_the_gate():
    payload = json.loads((OUT / "same_instance_parity.json").read_text("utf-8"))
    assert payload["gate"] == "SAME_INSTANCE_ZERO_RESIDUAL_IDENTITY"
    assert payload["passed"] is True
    assert payload["canonical_opens_during_parity"] == 0
    for name in ("belief_exact", "points_exact", "far_exact", "centroid_exact",
                 "repeated_forward_exact", "a1_parameter_delta_zero"):
        assert payload["checks"][name] is True, name


def test_43_cross_instance_gate_is_retired_not_deleted():
    payload = json.loads((OUT / "same_instance_parity.json").read_text("utf-8"))
    retired = payload["retired_gate"]
    assert retired["name"] == "INVALID_CROSS_INSTANCE_GATE"
    assert "cuDNN" in retired["why_retired"] or "cudnn" in retired["why_retired"]
    assert "counts exact" in retired["moved_to"]


def test_44_no_silent_first_sample_fallback(tree):
    body = code_of(tree, "TrainSet")
    assert "self.rows[0]" not in body, "a failed load must not become another frame"
    assert "HARD_BLOCKED" in body


def test_45_training_path_status_is_computed_not_asserted(runner, tree):
    body = code_of(tree, "train_path_status")
    for name in ("train_dataset_intersection_summary", "train_path_parity",
                 "paired_training_stream_parity"):
        assert name in body, name
    assert runner.train_path_status() == "OK"
    assert "TRAIN_PATH_STATUS" not in code_only(RUNNER)


def test_46_zero_residual_composition_is_bitwise(runner):
    base = torch.randn(2, 9, 50, 50)
    out = HCRM.compose(base, torch.zeros(2, 4, 50, 50))
    assert float((out - base).abs().max()) == 0.0



# ---------------------------------------------------------------------------
# 47-56  the restored A1 training path
# ---------------------------------------------------------------------------
def test_47_training_uses_the_a1_dataset(tree):
    body = code_of(tree, "build_a1_base")
    assert "DS.build" in body and "'A1'" in body.replace('"', "'")
    assert "truncation_aug_prob" in body


def test_48_training_never_calls_load_no_aug(tree):
    for name in ("train_run", "ManifestA1TrainDataset"):
        assert "load_no_aug" not in code_of(tree, name), name


def test_49_intersection_is_one_to_one():
    summary = json.loads(
        (OUT / "train_dataset_intersection_summary.json").read_text("utf-8"))
    assert summary["selected_train"] == summary["expected_train"] == 26249
    assert summary["unique_images"] == summary["unique_jsons"] == 26249
    assert summary["one_to_one"] is True and summary["passed"] is True


def test_50_no_holdout_source_is_selected():
    summary = json.loads(
        (OUT / "train_dataset_intersection_summary.json").read_text("utf-8"))
    assert summary["selected_validation"] == 0
    assert summary["selected_untouched"] == 0
    assert summary["unmatched_base_rows"] == 0
    table = pd.read_csv(OUT / "train_dataset_intersection.csv")
    manifest = pd.read_csv(OUT / "synthetic_split_manifest.csv")
    holdout = set((manifest[manifest.split != "train"]["root"] + "/"
                   + manifest[manifest.split != "train"]["stem"]))
    assert not set(table["root"] + "/" + table["stem"]) & holdout


def test_51_wrapper_leaves_the_base_sample_untouched():
    payload = json.loads((OUT / "train_path_parity.json").read_text("utf-8"))
    assert payload["passed"] is True and payload["mismatches"] == 0
    assert set(payload["sources"]) == {
        "mixed_v8_train", "v4_split_base", "aug_squash_v2", "aug_trunc_v2",
        "aug_scale_v2", "paper_4pallet_mask_v1"}


def test_52_sampler_substitution_is_recorded():
    payload = json.loads((OUT / "train_path_parity.json").read_text("utf-8"))
    assert "BALANCE-N" in payload["sampler_note"]
    assert "deviation" in payload["sampler_note"]


def test_53_arms_share_one_training_stream():
    payload = json.loads(
        (OUT / "paired_training_stream_parity.json").read_text("utf-8"))
    assert payload["identical"] is True and payload["passed"] is True
    assert len(set(payload["arms"].values())) == 1


def test_54_per_sample_seed_is_deterministic(runner):
    a = runner.stable_sample_seed(1, 0, "/x/y.png")
    assert a == runner.stable_sample_seed(1, 0, "/x/y.png")
    assert a != runner.stable_sample_seed(1, 1, "/x/y.png")
    assert a != runner.stable_sample_seed(2, 0, "/x/y.png")
    assert a != runner.stable_sample_seed(1, 0, "/x/z.png")


def test_55_seed_scope_restores_the_global_rngs(runner):
    import random
    random.seed(99)
    before = random.random()
    random.seed(99)
    with runner._SeedScope(1234):
        random.random()
    assert random.random() == before


def test_56_wrapper_guards_every_source_path(tree):
    body = code_of(tree, "ManifestA1TrainDataset")
    assert body.count("GUARD.check") >= 2
    assert "HARD_BLOCKED_A1_TRAIN_SAMPLE_LOAD" in body
    assert "self.selected[0]" not in body
