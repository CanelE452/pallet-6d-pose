"""Instance-aware 12-edge learnability screen.

The O12 oracle places corners at 98.7% / 96.1% within 20px using no corner
heatmap and no top-K, while the five-class O5 oracle reaches 2.7% / 0.6%.  That
is a statement about representation capacity from ground-truth geometry, not
about learnability.  This screen asks whether the twelve physical-edge fields
can be predicted from images by a head on a frozen A1 encoder, and whether
line-only corner generation survives the transfer to the canonical sets.

Nothing here fuses the line branch with the corner branch, adds a corner
heatmap loss, or touches a PnP objective.  The A1 local path is read-only
throughout; the only trainable parameters are the edge head.

    python scripts/stage0/instance_edge_learnability.py all
    python scripts/stage0/instance_edge_learnability.py status
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "scripts/stage0"
DOPE = ROOT / "Deep_Object_Pose"
for _extra in (STAGE0, DOPE / "common", DOPE / "train", ROOT / "challenge/scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

RESULT_ROOT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
               / "compatibility_calibration/canonical_corner_audit"
               / "instance_edge_learnability")
REPORT_ROOT = (ROOT / "_docs/audits/eval56_summary/canonical_corner_audit"
               / "instance_edge_learnability")
WEIGHT_ROOT = ROOT / "weights/paper_s2_instance_edge"
PPD_ROOT = ROOT / "data/pallet/results/paper_s2_palletgraph_line_screen"
PPD_WEIGHTS = ROOT / "weights/paper_s2_ppd_t2_screen"
SYNTH_ROOT = ROOT / "data/pallet/training_data/paper_4pallet_mask_v1"

EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
A1_CKPT = ROOT / "weights/paper_s2_pdg/A1/epoch_003.pth"

# Sealed tokens.  Any path carrying one of these is refused before it is opened.
SEALED = ("capturenight08", "capturenight09", "capturepallet07", "capturepallet09",
          "testset_full8_manifest", "handannot17")

# ---------------------------------------------------------------------------
# Frozen protocol constants.  Every one of these was fixed before the first
# training step and none may be revised in response to a result.
# ---------------------------------------------------------------------------
SEED_LIST = (1, 2, 3)
EPOCHS = 20                # PPD exact
BATCH = 8                  # PPD exact
LR = 3e-4                  # PPD exact
WD = 1e-4                  # PPD exact
GRID_12 = 50               # O12 oracle grid and the F50 tap resolution
GRID_5 = 100               # PPD grid and the F100 tap resolution
INPUT_SIZE = 400
CORNER_OK_PX = 20.0        # a corner counts as placed within 20px
R4_MIN, R6_MIN = 4, 6
SMOKE_STEPS = 100
CALIBRATION_BATCHES = 20   # PPD: "train split only, 20 batches, no update"
TOLERANCE_CELLS = 2        # PPD line_metrics tolerance
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Target-parity population, locked before any run: a corner takes part in the
# parity check only when all three of its incident edges have an in-frame
# clipped segment.  A corner whose edges leave the frame cannot be represented
# in a rasterised field at all, so including it would measure clipping rather
# than the generator.
PARITY_MEDIAN_CELLS = 0.5
PARITY_MAX_CELLS = 1.5

ORACLE_REFERENCE = {
    "eval56": {"O12": {"le20": 0.987, "pnp": 56}, "O5": {"le20": 0.027, "pnp": 0}},
    "wood": {"O12": {"le20": 0.961, "pnp": 45}, "O5": {"le20": 0.006, "pnp": 0}},
}
ORACLE_TOLERANCE = 0.001   # 0.1 percentage point

ARMS = ("L5-CTRL", "L12-F50", "L12-MS")
ARM_SEEDS = {"L5-CTRL": (1,), "L12-F50": SEED_LIST, "L12-MS": SEED_LIST}

PHASES = ("prepare", "oracle-parity", "recipe-parity", "targets", "smoke", "train",
          "eval-synthetic", "eval-canonical", "ppd", "decide", "test", "report")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_path(path: Any) -> Any:
    text = str(path)
    for token in SEALED:
        if token in text:
            raise RuntimeError(f"BLOCKED: sealed token {token!r} in {text}")
    return path


def atomic_write(path: pathlib.Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(obj: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(str(path) + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# state machine
# ============================================================================
class State:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"phases": {}, "created": time.time()}
        if path.is_file():
            self.data = json.loads(path.read_text("utf-8"))
        self.data.setdefault("phases", {})

    def get(self, phase: str) -> str:
        return self.data["phases"].get(phase, {}).get("status", "PENDING")

    def set(self, phase: str, status: str, **extra: Any) -> None:
        entry = self.data["phases"].setdefault(phase, {})
        entry["status"] = status
        entry["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry.update(extra)
        self.flush()

    def flush(self) -> None:
        atomic_write(self.path, json.dumps(self.data, indent=1))


# ============================================================================
# lazy heavy imports
# ============================================================================
_MODULES: dict[str, Any] = {}


def modules() -> dict[str, Any]:
    if _MODULES:
        return _MODULES
    spec = importlib.util.spec_from_file_location("E56", STAGE0 / "paper_s2_eval56.py")
    e56 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e56)
    import instance_edge_topology as IET
    import instance_edge_head as IEH
    import pallet_graph_geometry as PG
    import polarity_aware_line_head as PLH
    import pdg_stage1_model as PSM
    _MODULES.update({"E56": e56, "IET": IET, "IEH": IEH, "PG": PG, "PLH": PLH,
                     "PSM": PSM, "FZ": e56.FZ, "MD": e56.MD})
    return _MODULES


# ============================================================================
# encoder
# ============================================================================
class FrozenEncoder(torch.nn.Module):
    """A1's VGG with the two taps the arms use.

    A1 was trained with its VGG frozen, so these taps are bit-identical to
    ep57's.  That is asserted rather than assumed, because it is what lets the
    five-class control and the twelve-edge arms share one feature source
    instead of confounding representation with backbone.
    """

    def __init__(self) -> None:
        super().__init__()
        M = modules()
        self.model = M["PSM"].PDGStage1Model("A1")
        state = torch.load(str(A1_CKPT), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.high_index, self.high_channels = M["PLH"].find_high_resolution_feature(
            self.model.net.vgg, torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE), GRID_5)
        self._high: Optional[torch.Tensor] = None
        self.model.net.vgg[self.high_index].register_forward_hook(
            lambda module, inputs, output: setattr(self, "_high", output))

    @torch.no_grad()
    def taps(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.model.net.vgg(images)
        high = self._high
        assert high is not None and high.shape[-2:] == (GRID_5, GRID_5), high.shape
        assert low.shape[-2:] == (GRID_12, GRID_12), low.shape
        return high.detach(), low.detach()

    def vgg_checksum(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.net.vgg.named_parameters()):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def parameter_checksum(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


def ep57_vgg_checksum() -> str:
    M = modules()
    from models import DopeNetwork
    net = DopeNetwork(numSeg=1)
    net.load_state_dict(torch.load(str(EP57), map_location="cpu", weights_only=True),
                        strict=True)
    digest = hashlib.sha256()
    for name, parameter in sorted(net.vgg.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    del M
    return digest.hexdigest()


# ============================================================================
# Phase A -- input lock, topology, oracle parity
# ============================================================================
def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:      # pragma: no cover
        return ""


def phase_prepare(state: State) -> dict[str, Any]:
    M = modules()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    purpose = RESULT_ROOT / "PURPOSE.md"
    if not purpose.is_file():
        atomic_write(purpose,
                     "[소비처] 논문 architecture 결정 — instance-aware line branch 채택 여부\n"
                     "[문장] 12개 physical-edge instance field 는 frozen A1 feature 에서\n"
                     "학습 가능하고, line-only corner 생성이 정본으로 전이된다 (또는 안 된다).\n")

    topology = M["IET"].build_topology()
    atomic_write(RESULT_ROOT / "physical_edges.json",
                 json.dumps({"edges": topology["edges"],
                             "edge_classes": topology["edge_classes"]}, indent=1))
    atomic_write(RESULT_ROOT / "corner_edge_incidence.json",
                 json.dumps(topology["corner_edge_incidence"], indent=1))
    atomic_write(RESULT_ROOT / "semantic_edge_classes.json",
                 json.dumps({"classes": list(M["IET"].SEMANTIC_CLASSES),
                             "counts": topology["class_counts"]}, indent=1))
    atomic_write(RESULT_ROOT / "topology_sha.json",
                 json.dumps({"topology_sha256": topology["topology_sha256"]}, indent=1))
    atomic_write(RESULT_ROOT / "topology.json", json.dumps(topology, indent=1))

    encoder = FrozenEncoder()
    vgg_a1 = encoder.vgg_checksum()
    vgg_ep57 = ep57_vgg_checksum()
    if vgg_a1 != vgg_ep57:
        raise RuntimeError("BLOCKED: A1 VGG differs from ep57 VGG; the shared-feature "
                           "premise of this screen does not hold")

    eval56 = json.loads((M["E56"].OUT / "eval56_manifest.json").read_text("utf-8"))
    wood = json.loads((M["E56"].OUT / "wood_manifest.json").read_text("utf-8"))
    for manifest in (eval56, wood):
        for frame in manifest["frames"]:
            guard_path(frame["json_path"])
    splits = sorted({f.get("split", "<none>") for f in eval56["frames"]})
    if splits != ['eval']:
        raise RuntimeError(f"BLOCKED: eval56 carries non-eval splits {splits}")

    ppd_state = {}
    for arm in ("L0", "M0", "M1"):
        path = PPD_WEIGHTS / arm / "run_state.json"
        if path.is_file():
            ppd_state[arm] = json.loads(path.read_text("utf-8"))

    lock = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "origin_main": git("rev-parse", "origin/main"),
        "git_status": git("status", "--porcelain"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "opencv": cv2.__version__,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "a1_checkpoint": str(A1_CKPT.relative_to(ROOT)),
        "a1_sha256": sha256_file(A1_CKPT),
        "a1_run_state": json.loads((A1_CKPT.parent / "run_state.json").read_text("utf-8")),
        "a1_vgg_checksum": vgg_a1,
        "ep57_vgg_checksum": vgg_ep57,
        "a1_vgg_equals_ep57": vgg_a1 == vgg_ep57,
        "a1_parameter_checksum_before": encoder.parameter_checksum(),
        "ep57_sha256": sha256_file(EP57),
        "ep57_sha256_expected": EP57_SHA,
        "feature_taps": {"F100": f"vgg[{encoder.high_index}] ch={encoder.high_channels}",
                         "F50": "net.vgg(x) ch=128"},
        "eval56": {"n": eval56["eval_frame_count"],
                   "membership_sha256": eval56["membership_sha256"],
                   "splits": splits, "per_domain": eval56["per_domain"]},
        "wood": {"n": wood["eval_frame_count"],
                 "membership_sha256": wood["membership_sha256"],
                 "per_domain": wood["per_domain"]},
        "synthetic_root": str(SYNTH_ROOT.relative_to(ROOT)),
        "synthetic_manifests": {
            name: {"n": json.loads((PPD_ROOT / f"ppd_{name}_manifest.json").read_text())["n"],
                   "sha256": sha256_file(PPD_ROOT / f"ppd_{name}_manifest.json")}
            for name in ("train", "val", "untouched")},
        "ppd_checkpoints": {arm: {"path": str((PPD_WEIGHTS / arm / "last.pth").relative_to(ROOT)),
                                  "sha256": sha256_file(PPD_WEIGHTS / arm / "last.pth")}
                            for arm in ("L0", "M0", "M1")
                            if (PPD_WEIGHTS / arm / "last.pth").is_file()},
        "ppd_run_state": ppd_state,
        "final_test_prohibited_tokens": list(SEALED),
        "final_test_open_count": 0,
        "a1_optimizer_creation_count": 0,
        "a1_training_step_count": 0,
        "checkpoint_mtimes": {str(p.relative_to(ROOT)): p.stat().st_mtime
                              for p in [A1_CKPT, EP57]},
        "topology_sha256": topology["topology_sha256"],
    }
    atomic_write(RESULT_ROOT / "input_lock.json", json.dumps(lock, indent=1))
    log(f"[prepare] head {lock['head'][:8]}  eval56 {lock['eval56']['n']}  "
        f"wood {lock['wood']['n']}  A1 vgg == ep57 vgg: {lock['a1_vgg_equals_ep57']}")
    return lock


# ---------------------------------------------------------------------------
# canonical frames
# ---------------------------------------------------------------------------
def canonical_frames(label: str) -> list[dict[str, Any]]:
    M = modules()
    manifest = json.loads((M["E56"].OUT / f"{label}_manifest.json").read_text("utf-8"))
    return [guard_path(spec) or spec for spec in manifest["frames"]]


def oracle_run(label: str, mode: str, topology: dict[str, Any],
               unclipped: bool = True) -> dict[str, Any]:
    """The c190284 oracle: ground-truth edge geometry straight into the decoder."""
    M = modules()
    IET = M["IET"]
    incidence = IET.incidence_lists(topology, mode)
    errors: list[float] = []
    solved = 0
    used = 0
    for spec in canonical_frames(label):
        frame = M["E56"].EvalFrame(spec)
        width, height = spec["image_width"], spec["image_height"]
        gt = frame.gt_points[:8]
        if any(point is None for point in gt):
            continue
        used += 1
        grid_points = [np.array([p[0] * GRID_12 / width, p[1] * GRID_12 / height])
                       for p in gt]
        if unclipped:
            fields = np.stack([IET.segment_distance_field(
                GRID_12, grid_points[i], grid_points[j])
                for i, j in [tuple(e) for e in topology["edges"]]])
        else:
            segments, _ = IET.clipped_edges_in_grid(gt, topology, width, height, GRID_12)
            fields, _ = IET.distance_fields_from_segments(segments, GRID_12)
        if mode == "O5C":
            fields = IET.class_distance_fields(fields, topology)
        points = IET.decode_corners(fields, incidence, GRID_12, width, height)
        for index in range(8):
            errors.append(float(np.hypot(points[index][0] - gt[index][0],
                                         points[index][1] - gt[index][1])))
        solved += int(frame.solve(points + [None]) is not None)
    array = np.array(errors)
    return {"set": label, "mode": mode, "n_frames": used,
            "le20": float((array <= 20).mean()), "le50": float((array <= 50).mean()),
            "gt100": float((array > 100).mean()), "median": float(np.median(array)),
            "pnp": solved}


def phase_oracle_parity(state: State) -> dict[str, Any]:
    M = modules()
    topology = json.loads((RESULT_ROOT / "topology.json").read_text("utf-8"))
    rows = []
    failures = []
    for label in ("eval56", "wood"):
        for mode in ("O12", "O5", "O5C"):
            row = oracle_run(label, mode, topology)
            rows.append(row)
            if mode == "O5C":
                # The five-channel oracle a five-class head can actually be
                # scored against.  It has no c190284 reference number, so it is
                # reported rather than gated.
                row["parity"] = None
                log(f"[oracle] O5C  {label:7s} n={row['n_frames']:3d} "
                    f"<=20px {100*row['le20']:5.1f}% med {row['median']:6.2f} "
                    f"PnP {row['pnp']:3d}  (reference for L5-CTRL)")
                continue
            reference = ORACLE_REFERENCE[label][mode]
            row["reference_le20"] = reference["le20"]
            row["reference_pnp"] = reference["pnp"]
            row["le20_delta"] = row["le20"] - reference["le20"]
            row["parity"] = (abs(row["le20_delta"]) <= ORACLE_TOLERANCE
                             and row["pnp"] == reference["pnp"])
            if not row["parity"]:
                failures.append(f"{label}/{mode}: le20 {row['le20']:.4f} vs "
                                f"{reference['le20']:.4f}, PnP {row['pnp']} vs "
                                f"{reference['pnp']}")
            log(f"[oracle] {mode:4s} {label:7s} n={row['n_frames']:3d} "
                f"<=20px {100*row['le20']:5.1f}% med {row['median']:6.2f} "
                f"PnP {row['pnp']:3d}  parity={row['parity']}")
    # The clipped policy the learned path must use, reported alongside.
    clipped = [oracle_run(label, "O12", topology, unclipped=False)
               for label in ("eval56", "wood")]
    payload = {"rows": rows, "clipped_o12": clipped, "failures": failures,
               "passed": not failures}
    atomic_write(RESULT_ROOT / "oracle_parity.json", json.dumps(payload, indent=1))
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "oracle_parity.csv", index=False)
    if failures:
        state.set("oracle-parity", "HARD_BLOCKED", reason="HARD_BLOCKED_ORACLE_PARITY",
                  failures=failures)
        raise RuntimeError("HARD_BLOCKED_ORACLE_PARITY: " + "; ".join(failures))
    del M
    return payload


# ============================================================================
# Phase B -- PPD recipe trace
# ============================================================================
def phase_recipe_parity(state: State) -> dict[str, Any]:
    calibration = json.loads((PPD_ROOT / "ppd_t2_loss_calibration.json").read_text("utf-8"))
    history_path = PPD_WEIGHTS / "L0" / "metrics_by_epoch.json"
    history = json.loads(history_path.read_text("utf-8")) if history_path.is_file() else []
    recipe = {
        "source": "scripts/stage0/paper_s2_ppd_long_run.py",
        "feature_source": "ep57 VGG tap found at runtime for a 100x100 output",
        "output_resolution": GRID_5,
        "head": "PolarityLineHead: 1x1 stem over [f, gate, gate*f], 2 residual blocks, 1x1 out",
        "target": "exp(-d^2/2s^2) soft distance field, sigma 1.5 cells, "
                  "observed_fragment mode",
        "loss": "BCEWithLogits(pos_weight) + lambda_pol * polarity_contrast",
        "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "scheduler": None,
        "epochs": EPOCHS, "batch": BATCH, "seed": 1,
        "split": {"train": 3039, "val": 1045, "untouched": 5916,
                  "group_key": "hdri|background|floor"},
        "augmentation": "none",
        "best_checkpoint_policy": "polarity_acc, then inversion_rate, then indexed "
                                  "reprojection, then macro F1, then earliest epoch",
        "calibration": calibration,
        "calibration_rule": "lambda_x = c_x * median(L_line)/median(L_x) over 20 train "
                            "batches with no update; c = 0.1 (pol), 0.5 (mask), "
                            "0.05 (outside)",
        "ppd_l0_epochs_recorded": len(history),
        "ppd_l0_best_macro_f1": (max(h["macro_f1"] for h in history) if history else None),
        "changed_for_12_edge": ["output channels 5 -> 12",
                                "target: semantic class -> physical edge instance",
                                "decoder: O5 -> O12 incidence",
                                "pos_weight recomputed by the same rule for 12 channels"],
        "unchanged": ["optimizer", "lr", "weight_decay", "scheduler", "epochs",
                      "batch", "seed policy", "target field scale", "loss form",
                      "trunk", "split"],
        "deviations": [
            "L12-MS uses the same PPD trunk as L12-F50 rather than the plain conv "
            "stack named in the instruction, so Phase H varies feature scale alone.",
            "L12 arms decode at 50x50 (the O12 oracle grid and the F50 tap); "
            "L5-CTRL stays at PPD's 100x100.  Each arm matches its own reference.",
        ],
    }
    # Verify the lambda rule reproduces the stored calibration.
    checks = {}
    for name, coefficient in (("pol", 0.1), ("mask", 0.5), ("out", 0.05)):
        expected = coefficient * calibration["L_line_median"] / calibration[f"L_{name}_median"]
        checks[f"lambda_{name}"] = {
            "stored": calibration[f"lambda_{name}"], "reconstructed": expected,
            "match": abs(expected - calibration[f"lambda_{name}"]) < 1e-9}
    recipe["lambda_rule_checks"] = checks
    recipe["lambda_rule_reproduced"] = all(v["match"] for v in checks.values())
    atomic_write(RESULT_ROOT / "ppd_recipe.json", json.dumps(recipe, indent=1))
    log(f"[recipe] lambda rule reproduced: {recipe['lambda_rule_reproduced']}")
    return recipe


# ============================================================================
# synthetic frames
# ============================================================================
class SyntheticFrame:
    __slots__ = ("file", "image", "target12", "chmask12", "target5", "corners",
                 "K", "dims", "R", "t", "size", "visibility", "asset", "group")


def load_synthetic(file_name: str, topology: dict[str, Any], want5: bool
                   ) -> Optional[SyntheticFrame]:
    M = modules()
    IET, PG, PLH = M["IET"], M["PG"], M["PLH"]
    json_path = SYNTH_ROOT / file_name
    guard_path(json_path)
    payload = json.loads(json_path.read_text("utf-8"))
    obj = payload["objects"][0]
    camera = payload["camera_data"]
    intrinsics = camera["intrinsics"]
    K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                  [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1.0]])
    dims = (obj["dimensions_m"]["width"], obj["dimensions_m"]["depth"],
            obj["dimensions_m"]["height"])
    transform = np.asarray(obj["pose_transform"], float)
    R, t = transform[:3, :3], transform[:3, 3]
    width, height = camera["width"], camera["height"]
    image = cv2.imread(str(SYNTH_ROOT / file_name.replace(".json", ".png")))
    if image is None:
        return None

    corners3d = PG.make_corners(*dims)[:8]
    projected, depth = PG.project_points(corners3d, R, t, K)
    points: list[Optional[list[float]]] = [
        None if depth[i] <= 1e-6 else [float(projected[i][0]), float(projected[i][1])]
        for i in range(8)]

    segments, in_frame = IET.clipped_edges_in_grid(points, topology, width, height, GRID_12)
    target12 = IET.build_edge_targets(segments, GRID_12)
    chmask12 = np.array([1.0 if flag else 0.0 for flag in in_frame], np.float32)

    # visibility state per physical edge
    self_visible = PG.visible_edges(R, t, dims)
    mask = None
    if "mask_rle" in obj:
        try:
            mask = PLH.decode_mask_rle(obj["mask_rle"], (height, width))
        except Exception:
            mask = None
    states = []
    for k, (i, j) in enumerate([tuple(e) for e in topology["edges"]]):
        if not in_frame[k]:
            states.append("OFF_FRAME")
            continue
        if not self_visible.get((min(i, j), max(i, j)), self_visible.get((i, j), False)):
            states.append("OCCLUDED")
            continue
        if mask is None:
            states.append("UNKNOWN")
            continue
        a, b = segments[k]
        steps = max(int(np.hypot(*(b - a))) * 2, 2)
        ts = np.linspace(0, 1, steps)
        xs = np.clip(((a[0] + ts * (b[0] - a[0])) * width / GRID_12).astype(int), 0, width - 1)
        ys = np.clip(((a[1] + ts * (b[1] - a[1])) * height / GRID_12).astype(int), 0, height - 1)
        inside = float(mask[ys, xs].mean())
        states.append("VISIBLE" if inside >= 0.5 else "OCCLUDED")

    frame = SyntheticFrame()
    frame.file = file_name
    frame.image = cv2.cvtColor(cv2.resize(image, (INPUT_SIZE, INPUT_SIZE)),
                               cv2.COLOR_BGR2RGB)
    frame.target12 = target12
    frame.chmask12 = chmask12
    frame.target5 = None
    if want5:
        target5, _ = PLH.build_polarity_targets_v2(
            R, t, K, dims, (width, height), "observed_fragment", image_bgr=image)
        frame.target5 = target5
    frame.corners = points
    frame.K, frame.dims, frame.R, frame.t = K, dims, R, t
    frame.size = (width, height)
    frame.visibility = states
    frame.asset = obj.get("class", obj.get("name", "?"))
    return frame


def normalise(images_uint8: np.ndarray) -> torch.Tensor:
    array = images_uint8.astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(array.transpose(0, 3, 1, 2))


def load_split(name: str, topology: dict[str, Any], want5: bool,
               limit: int = 0) -> list[SyntheticFrame]:
    manifest = json.loads((PPD_ROOT / f"ppd_{name}_manifest.json").read_text("utf-8"))
    files = [f["file"] for f in manifest["frames"]]
    groups = {f["file"]: f.get("group", "?") for f in manifest["frames"]}
    assets = {f["file"]: f.get("asset", "?") for f in manifest["frames"]}
    if limit:
        files = files[:limit]
    start = time.time()
    frames = []
    for index, file_name in enumerate(files):
        frame = load_synthetic(file_name, topology, want5)
        if frame is None:
            continue
        frame.group = groups.get(file_name, "?")
        frame.asset = assets.get(file_name, frame.asset)
        frames.append(frame)
        if (index + 1) % 500 == 0:
            log(f"[data] {name} {index+1}/{len(files)} ({time.time()-start:.0f}s)")
    log(f"[data] {name} loaded {len(frames)} frames in {time.time()-start:.0f}s")
    return frames


# ============================================================================
# Phase C -- target parity
# ============================================================================
def phase_targets(state: State, limit: int = 200) -> dict[str, Any]:
    M = modules()
    IET = M["IET"]
    topology = json.loads((RESULT_ROOT / "topology.json").read_text("utf-8"))
    incidence = IET.incidence_lists(topology, "O12")
    manifest = json.loads((PPD_ROOT / "ppd_train_manifest.json").read_text("utf-8"))
    files = [f["file"] for f in manifest["frames"]][:limit]

    differences: list[float] = []
    off_frame_corners = 0
    parity_corners = 0
    clamp_violations = 0
    for file_name in files:
        frame = load_synthetic(file_name, topology, want5=False)
        if frame is None:
            continue
        width, height = frame.size
        segments, in_frame = IET.clipped_edges_in_grid(
            frame.corners, topology, width, height, GRID_12)
        analytic, _ = IET.distance_fields_from_segments(segments, GRID_12)
        recovered = IET.distance_from_probability(frame.target12)
        direct = IET.decode_corners(analytic, incidence, GRID_12, width, height)
        through = IET.decode_corners(recovered, incidence, GRID_12, width, height)
        for corner in range(8):
            usable = all(in_frame[k] for k in incidence[corner])
            point = frame.corners[corner]
            inside = (point is not None and 0 <= point[0] < width and 0 <= point[1] < height)
            if not (usable and inside):
                off_frame_corners += 1
                continue
            parity_corners += 1
            differences.append(float(np.hypot(
                (direct[corner][0] - through[corner][0]) * GRID_12 / width,
                (direct[corner][1] - through[corner][1]) * GRID_12 / height)))
        # a clipped endpoint must never sit exactly on the border unless the
        # unclipped endpoint did, which would be a clamp artefact
        for k, segment in enumerate(segments):
            if segment is None:
                continue
            for end in segment:
                on_border = (abs(end[0]) < 1e-9 or abs(end[1]) < 1e-9
                             or abs(end[0] - GRID_12) < 1e-9 or abs(end[1] - GRID_12) < 1e-9)
                if on_border:
                    i, j = topology["edges"][k]
                    a, b = frame.corners[i], frame.corners[j]
                    if a is not None and b is not None:
                        inside_a = 0 <= a[0] <= width and 0 <= a[1] <= height
                        inside_b = 0 <= b[0] <= width and 0 <= b[1] <= height
                        if inside_a and inside_b:
                            clamp_violations += 1

    array = np.array(differences) if differences else np.zeros(1)
    payload = {
        "frames": len(files), "parity_corners": parity_corners,
        "excluded_off_frame_corners": off_frame_corners,
        "median_cells": float(np.median(array)), "max_cells": float(array.max()),
        "p99_cells": float(np.percentile(array, 99)),
        "clamp_violations": clamp_violations,
        "gate_median": PARITY_MEDIAN_CELLS, "gate_max": PARITY_MAX_CELLS,
        "passed": bool(np.median(array) < PARITY_MEDIAN_CELLS
                       and array.max() < PARITY_MAX_CELLS and clamp_violations == 0),
    }
    atomic_write(RESULT_ROOT / "target_parity.json", json.dumps(payload, indent=1))
    log(f"[targets] parity median {payload['median_cells']:.4f} max "
        f"{payload['max_cells']:.4f} cells over {parity_corners} corners "
        f"(excluded {off_frame_corners}) -> {payload['passed']}")
    if not payload["passed"]:
        state.set("targets", "HARD_BLOCKED", reason="HARD_BLOCKED_TARGET_PARITY")
        raise RuntimeError("HARD_BLOCKED_TARGET_PARITY")
    return payload


# ============================================================================
# arms
# ============================================================================
def build_arm(arm: str, seed: int, encoder: FrozenEncoder):
    M = modules()
    seed_all(seed)
    if arm == "L5-CTRL":
        model = M["IEH"].InstanceEdgeModel("F100", 5, encoder.high_channels, 128)
    elif arm == "L12-F50":
        model = M["IEH"].InstanceEdgeModel("F50", 12, encoder.high_channels, 128)
    elif arm == "L12-MS":
        model = M["IEH"].InstanceEdgeModel("MS", 12, encoder.high_channels, 128)
    else:
        raise ValueError(arm)
    return model.to(device)


def arm_grid(arm: str) -> int:
    return GRID_5 if arm == "L5-CTRL" else GRID_12


def arm_channels(arm: str) -> int:
    return 5 if arm == "L5-CTRL" else 12


def stack_targets(frames: list[SyntheticFrame], arm: str) -> torch.Tensor:
    if arm == "L5-CTRL":
        return torch.from_numpy(np.stack([f.target5 for f in frames]))
    return torch.from_numpy(np.stack([f.target12 for f in frames]))


def stack_masks(frames: list[SyntheticFrame], arm: str) -> torch.Tensor:
    if arm == "L5-CTRL":
        return torch.ones(len(frames), 5)
    return torch.from_numpy(np.stack([f.chmask12 for f in frames]))


def calibrate(arm: str, frames: list[SyntheticFrame], encoder: FrozenEncoder,
              topology: dict[str, Any]) -> dict[str, Any]:
    """PPD's calibration procedure: train split only, 20 batches, no update."""
    M = modules()
    if arm == "L5-CTRL":
        stored = json.loads((PPD_ROOT / "ppd_t2_loss_calibration.json").read_text("utf-8"))
        return {"pos_weight": stored["pos_weight"], "lambda_pol": stored["lambda_pol"],
                "source": "ppd_t2_loss_calibration.json (exact reuse)"}
    channels = arm_channels(arm)
    positives = np.zeros(channels)
    negatives = np.zeros(channels)
    model = build_arm(arm, 1, encoder)
    line_values, polarity_values = [], []
    pairs = [tuple(p) for p in topology["polarity_pairs"]]
    for b in range(CALIBRATION_BATCHES):
        chunk = frames[b * BATCH:(b + 1) * BATCH]
        if not chunk:
            break
        targets = stack_targets(chunk, arm).to(device)
        masks = stack_masks(chunk, arm).to(device)
        binary = (targets >= M["PLH"].POSITIVE_THRESHOLD).float()
        positives += binary.sum(dim=(0, 2, 3)).cpu().numpy()
        negatives += (1 - binary).sum(dim=(0, 2, 3)).cpu().numpy()
        images = normalise(np.stack([f.image for f in chunk])).to(device)
        high, low = encoder.taps(images)
        with torch.no_grad():
            logits = model(high, low)
            unit = torch.ones(channels, device=device)
            line_values.append(float(M["IEH"].masked_field_loss(logits, targets, unit, masks)))
            polarity_values.append(float(M["IEH"].polarity_contrast_loss(logits, targets, pairs)))
    ratio = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 200.0)
    line_median = float(np.median(line_values))
    polarity_median = float(np.median(polarity_values))
    return {"pos_weight": [float(v) for v in ratio],
            "L_line_median": line_median, "L_pol_median": polarity_median,
            "lambda_pol": 0.1 * line_median / max(polarity_median, 1e-9),
            "source": "recomputed by the PPD rule on the train split, 20 batches, no update"}


# ============================================================================
# Phase E -- smoke and training
# ============================================================================
def train_one(arm: str, seed: int, train_frames: list[SyntheticFrame],
              val_frames: list[SyntheticFrame], encoder: FrozenEncoder,
              topology: dict[str, Any], calibration: dict[str, Any],
              smoke: bool = False) -> dict[str, Any]:
    M = modules()
    out = WEIGHT_ROOT / arm.replace("-", "_") / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "run_state.json"
    if not smoke and state_path.is_file():
        recorded = json.loads(state_path.read_text("utf-8"))
        if recorded.get("completed"):
            log(f"[{arm} s{seed}] already completed -- skipping")
            return json.loads((out / "metrics_by_epoch.json").read_text("utf-8"))

    model = build_arm(arm, seed, encoder)
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_trainable = sum(1 for p in encoder.parameters() if p.requires_grad)
    if frozen_trainable != 0:
        raise RuntimeError(f"BLOCKED: {frozen_trainable} trainable A1 parameters")
    optimiser = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    pairs = [tuple(p) for p in topology["polarity_pairs"]]
    positive_weight = torch.tensor(calibration["pos_weight"], dtype=torch.float32)
    lambda_pol = float(calibration["lambda_pol"])

    history: list[dict[str, Any]] = []
    start_epoch = 0
    if not smoke and state_path.is_file():
        recorded = json.loads(state_path.read_text("utf-8"))
        checkpoint = out / "last.pth"
        if checkpoint.is_file():
            try:
                blob = torch.load(checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(blob["model"])
                optimiser.load_state_dict(torch.load(out / "optimizer_last.pth",
                                                     map_location=device))
                torch.set_rng_state(torch.load(out / "rng_state.pt")["cpu"])
                start_epoch = recorded["epoch"]
                history = json.loads((out / "metrics_by_epoch.json").read_text("utf-8"))
                log(f"[{arm} s{seed}] resumed at epoch {start_epoch}")
            except Exception as error:
                log(f"[{arm} s{seed}] corrupted checkpoint rejected: {error}")
                start_epoch = 0
                history = []

    epochs = 1 if smoke else EPOCHS
    steps_done = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(train_frames), generator=generator)
        losses = []
        began = time.time()
        for b in range(0, len(order), BATCH):
            if smoke and steps_done >= SMOKE_STEPS:
                break
            index = order[b:b + BATCH].tolist()
            chunk = [train_frames[i] for i in index]
            images = normalise(np.stack([f.image for f in chunk])).to(device)
            targets = stack_targets(chunk, arm).to(device)
            masks = stack_masks(chunk, arm).to(device)
            high, low = encoder.taps(images)
            logits = model(high, low)
            loss = M["IEH"].masked_field_loss(logits, targets, positive_weight, masks)
            if arm != "L5-CTRL":
                loss = loss + lambda_pol * M["IEH"].polarity_contrast_loss(
                    logits, targets, pairs)
            else:
                loss = loss + lambda_pol * M["PLH"].polarity_contrast_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"BLOCKED: non-finite loss in {arm} seed {seed}")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
            steps_done += 1
        if smoke:
            gradient = sum(float(p.grad.abs().sum()) for p in trainable if p.grad is not None)
            return {"arm": arm, "seed": seed, "steps": steps_done,
                    "loss_finite": bool(np.isfinite(np.mean(losses))),
                    "mean_loss": float(np.mean(losses)),
                    "head_gradient_l1": gradient,
                    "trainable_parameters": int(sum(p.numel() for p in trainable))}
        metrics = evaluate_synthetic(arm, model, encoder, val_frames, topology,
                                     label="val", quick=True)
        metrics.update({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "seconds": time.time() - began})
        history.append(metrics)
        atomic_torch_save({"model": model.state_dict()}, out / "last.pth")
        atomic_torch_save({"model": model.state_dict()}, out / f"epoch_{epoch+1:03d}.pth")
        atomic_torch_save(optimiser.state_dict(), out / "optimizer_last.pth")
        atomic_torch_save({"cpu": torch.get_rng_state()}, out / "rng_state.pt")
        atomic_write(out / "metrics_by_epoch.json", json.dumps(history, indent=1))
        atomic_write(state_path, json.dumps(
            {"arm": arm, "seed": seed, "epoch": epoch + 1,
             "completed": epoch + 1 >= EPOCHS, "head": git("rev-parse", "HEAD"),
             "timestamp": time.time()}, indent=1))
        log(f"[{arm} s{seed}] epoch {epoch+1}/{EPOCHS} loss {np.mean(losses):.4f} "
            f"corner<=20px {100*metrics['corner_le20']:.1f}% R4 {100*metrics['r4']:.1f}% "
            f"({metrics['seconds']:.0f}s)")
    return history


def select_checkpoint(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Synthetic validation only: <=20px rate, then median error, then earliest."""
    return min(history, key=lambda h: (-h["corner_le20"], h["corner_median"], h["epoch"]))


# ============================================================================
# evaluation
# ============================================================================
_POOL = None


def _pnp_init() -> None:
    cv2.setNumThreads(1)
    modules()


def _pnp_task(task):
    """One canonical solve plus its fixed-observation reprojection.

    The solver is deterministic -- annotate_pnp reaches cv2.solvePnP and
    cv2.solvePnPGeneric, never a RANSAC variant -- so splitting the work across
    processes cannot change a number.  It is 98% of the evaluation cost and the
    only part that does not fit on the GPU: OpenCV's calib3d has no CUDA
    solvePnP, and the solver itself is the project's canonical one, so
    replacing it would break comparability with every earlier verdict.
    """
    points, intrinsics, dims, shape, ground_truth = task
    M = modules()
    solver = M["FZ"].CurrentSolveCache(intrinsics, dims, shape, auto_swap_dims=True)
    pose, _, _, _ = solver.solve(points)
    if pose is None:
        return False, None
    value, _ = M["FZ"].fixed_observation_reprojection(pose, ground_truth,
                                                     intrinsics, dims)
    return True, value


def start_pool(workers: int = 0):
    """Start the solve pool.

    Spawn, not fork: torch has already started thread pools by the time this
    runs, and forking a multi-threaded process deadlocked reproducibly here.
    Spawn costs a one-off worker import and was measured at 3.6x with results
    identical to the sequential path, checked value by value rather than
    approximately.
    """
    global _POOL
    if _POOL is not None:
        return _POOL
    import multiprocessing
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 2)
    context = multiprocessing.get_context("spawn")
    _POOL = context.Pool(workers, initializer=_pnp_init)
    log(f"[pool] {workers} solve workers (spawn)")
    return _POOL


def solve_many(tasks: list) -> list:
    if not tasks:
        return []
    if _POOL is None:
        return [_pnp_task(task) for task in tasks]
    return _POOL.map(_pnp_task, tasks, chunksize=max(1, len(tasks) // 64))


def decode_batch(probability: np.ndarray, incidence: list[list[int]], grid: int,
                 width: float, height: float) -> list[list[float]]:
    M = modules()
    distances = M["IET"].distance_from_probability(probability)
    return M["IET"].decode_corners(distances, incidence, grid, width, height)


def corner_rows(points: list[list[float]], gt: list[Optional[list[float]]]
                ) -> list[dict[str, Any]]:
    rows = []
    for index in range(8):
        truth = gt[index]
        if truth is None:
            continue
        rows.append({"corner": index,
                     "err": float(np.hypot(points[index][0] - truth[0],
                                           points[index][1] - truth[1])),
                     "near": index < 4})
    return rows


def summarise_corners(rows: list[dict[str, Any]], frames: int, r4: int, r6: int,
                      pnp: int, reproj: list[float]) -> dict[str, Any]:
    errors = np.array([r["err"] for r in rows]) if rows else np.zeros(1)
    near = np.array([r["err"] for r in rows if r["near"]]) if rows else np.zeros(1)
    far = np.array([r["err"] for r in rows if not r["near"]]) if rows else np.zeros(1)
    out = {
        "n_frames": frames, "n_corners": len(rows),
        "corner_le10": float((errors <= 10).mean()),
        "corner_le20": float((errors <= CORNER_OK_PX).mean()),
        "corner_le50": float((errors <= 50).mean()),
        "corner_gt100": float((errors > 100).mean()),
        "corner_median": float(np.median(errors)),
        "corner_p90": float(np.percentile(errors, 90)),
        "near_median": float(np.median(near)), "far_median": float(np.median(far)),
        "near_le20": float((near <= CORNER_OK_PX).mean()),
        "far_le20": float((far <= CORNER_OK_PX).mean()),
        "r4": r4 / max(frames, 1), "r6": r6 / max(frames, 1),
        "pnp": pnp, "pnp_rate": pnp / max(frames, 1),
        "reproj_median": float(np.median(reproj)) if reproj else None,
    }
    for index in range(8):
        subset = [r["err"] for r in rows if r["corner"] == index]
        out[f"corner{index}_le20"] = (float(np.mean(np.array(subset) <= CORNER_OK_PX))
                                      if subset else None)
    return out


def map_metrics(probabilities: np.ndarray, targets: np.ndarray,
                channel_masks: np.ndarray) -> dict[str, Any]:
    """Recall/precision/F1 at the PPD tolerance plus channel activity."""
    M = modules()
    channels = probabilities.shape[1]
    predicted = torch.from_numpy((probabilities >= 0.5).astype(np.float32))
    truth = torch.from_numpy((targets >= 0.5).astype(np.float32))
    kernel = 2 * TOLERANCE_CELLS + 1
    dilated_truth = F.max_pool2d(truth, kernel, 1, TOLERANCE_CELLS)
    dilated_pred = F.max_pool2d(predicted, kernel, 1, TOLERANCE_CELLS)
    recall = ((truth * dilated_pred).sum(dim=(0, 2, 3))
              / truth.sum(dim=(0, 2, 3)).clamp_min(1)).numpy()
    precision = ((predicted * dilated_truth).sum(dim=(0, 2, 3))
                 / predicted.sum(dim=(0, 2, 3)).clamp_min(1)).numpy()
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    activation = (probabilities.reshape(probabilities.shape[0], channels, -1).max(axis=2)
                  >= 0.5).mean(axis=0)
    foreground = truth.numpy().mean()
    # confusion: mean predicted probability of channel m over the ground-truth
    # support of channel k, row-normalised
    confusion = np.zeros((channels, channels))
    for k in range(channels):
        support = targets[:, k] >= 0.5
        if not support.any():
            continue
        for m in range(channels):
            confusion[k, m] = probabilities[:, m][support].mean()
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalised = confusion / np.maximum(row_sum, 1e-9)
    del M
    return {"per_channel_recall": [float(v) for v in recall],
            "per_channel_precision": [float(v) for v in precision],
            "macro_f1": float(np.mean(f1)),
            "channel_activation": [float(v) for v in activation],
            "active_channels": int((activation >= 0.05).sum()),
            "min_channel_recall": float(recall.min()),
            "foreground_fraction": float(foreground),
            "channel_mask_mean": [float(v) for v in channel_masks.mean(axis=0)],
            "confusion": normalised.tolist()}


def hungarian_from_confusion(confusion: np.ndarray) -> list[int]:
    from scipy.optimize import linear_sum_assignment
    rows, columns = linear_sum_assignment(-np.asarray(confusion))
    mapping = list(range(len(confusion)))
    for r, c in zip(rows, columns):
        mapping[int(r)] = int(c)
    return mapping


@torch.no_grad()
def predict(model, encoder: FrozenEncoder, frames: list[SyntheticFrame],
            batch: int = 16) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch]
        images = normalise(np.stack([f.image for f in chunk])).to(device)
        high, low = encoder.taps(images)
        outputs.append(torch.sigmoid(model(high, low)).float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0,))


def evaluate_synthetic(arm: str, model, encoder: FrozenEncoder,
                       frames: list[SyntheticFrame], topology: dict[str, Any],
                       label: str, quick: bool = False,
                       incidence_override: Optional[list[list[int]]] = None,
                       permutation: Optional[list[int]] = None) -> dict[str, Any]:
    M = modules()
    grid = arm_grid(arm)
    mode = "O5C" if arm == "L5-CTRL" else "O12"
    incidence = incidence_override or M["IET"].incidence_lists(topology, mode)
    probabilities = predict(model, encoder, frames)
    if permutation is not None:
        probabilities = probabilities[:, permutation]
    rows: list[dict[str, Any]] = []
    r4 = r6 = pnp = 0
    reproj: list[float] = []
    visibility_rows: list[dict[str, Any]] = []
    decoded = [decode_batch(probabilities[index], incidence, grid, *frame.size)
               for index, frame in enumerate(frames)]
    solved = solve_many([] if quick else [
        (points + [None], frame.K, frame.dims, (frame.size[1], frame.size[0], 3),
         list(frame.corners) + [None])
        for points, frame in zip(decoded, frames)])
    for index, frame in enumerate(frames):
        points = decoded[index]
        frame_rows = corner_rows(points, frame.corners)
        rows.extend(frame_rows)
        good = sum(1 for r in frame_rows if r["err"] <= CORNER_OK_PX)
        r4 += int(good >= R4_MIN)
        r6 += int(good >= R6_MIN)
        if not quick:
            success, value = solved[index]
            pnp += int(success)
            if value is not None:
                reproj.append(value)
            for corner in range(8):
                states = [frame.visibility[k] for k in incidence[corner]]
                truth = frame.corners[corner]
                if truth is None:
                    continue
                error = float(np.hypot(points[corner][0] - truth[0],
                                       points[corner][1] - truth[1]))
                visibility_rows.append({
                    "corner": corner, "err": error,
                    "all_visible": all(s == "VISIBLE" for s in states),
                    "any_occluded": any(s == "OCCLUDED" for s in states),
                    "any_off_frame": any(s == "OFF_FRAME" for s in states)})
    summary = summarise_corners(rows, len(frames), r4, r6, pnp, reproj)
    summary.update({"arm": arm, "set": label})
    if not quick:
        targets = np.stack([f.target5 if arm == "L5-CTRL" else f.target12 for f in frames])
        masks = np.stack([np.ones(5, np.float32) if arm == "L5-CTRL" else f.chmask12
                          for f in frames])
        summary.update(map_metrics(probabilities, targets, masks))
        summary["visibility"] = visibility_breakdown(visibility_rows)
        summary["edge_state_counts"] = {
            state: int(sum(f.visibility.count(state) for f in frames))
            for state in ("VISIBLE", "OCCLUDED", "OFF_FRAME", "UNKNOWN")}
    return summary


def visibility_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(subset):
        if not subset:
            return {"n": 0, "le20": None, "median": None}
        errors = np.array([r["err"] for r in subset])
        return {"n": len(subset), "le20": float((errors <= CORNER_OK_PX).mean()),
                "median": float(np.median(errors))}
    return {
        "all_incident_visible": block([r for r in rows if r["all_visible"]]),
        "any_incident_occluded": block([r for r in rows if r["any_occluded"]]),
        "any_incident_off_frame": block([r for r in rows if r["any_off_frame"]]),
    }


# ============================================================================
# canonical evaluation
# ============================================================================
class CanonicalFrame:
    __slots__ = ("spec", "eval_frame", "image", "size", "corners", "visibility",
                 "target12", "chmask12")


def load_canonical(topology: dict[str, Any], label: str) -> list[CanonicalFrame]:
    M = modules()
    IET = M["IET"]
    out = []
    for spec in canonical_frames(label):
        guard_path(spec["image_path"])
        raw = cv2.imread(spec["image_path"])
        if raw is None:
            continue
        evaluation = M["E56"].EvalFrame(spec)
        width, height = spec["image_width"], spec["image_height"]
        frame = CanonicalFrame()
        frame.spec = spec
        frame.eval_frame = evaluation
        frame.image = cv2.cvtColor(cv2.resize(raw, (INPUT_SIZE, INPUT_SIZE)),
                                   cv2.COLOR_BGR2RGB)
        frame.size = (width, height)
        frame.corners = evaluation.gt_points[:8]
        segments, in_frame = IET.clipped_edges_in_grid(
            frame.corners, topology, width, height, GRID_12)
        frame.target12 = IET.build_edge_targets(segments, GRID_12)
        frame.chmask12 = np.array([1.0 if f else 0.0 for f in in_frame], np.float32)
        states = []
        pose = evaluation.gt_pose
        self_visible = (M["PG"].visible_edges(pose["R"], pose["t"], evaluation.dims)
                        if pose is not None else None)
        for k, (i, j) in enumerate([tuple(e) for e in topology["edges"]]):
            if not in_frame[k]:
                states.append("OFF_FRAME")
            elif self_visible is None:
                states.append("UNKNOWN")
            else:
                visible = self_visible.get((min(i, j), max(i, j)),
                                           self_visible.get((i, j), None))
                states.append("UNKNOWN" if visible is None
                              else ("VISIBLE" if visible else "OCCLUDED"))
        frame.visibility = states
        out.append(frame)
    return out


def evaluate_canonical(arm: str, model, encoder: FrozenEncoder,
                       frames: list[CanonicalFrame], topology: dict[str, Any],
                       label: str, incidence_override=None,
                       permutation=None) -> dict[str, Any]:
    M = modules()
    grid = arm_grid(arm)
    mode = "O5C" if arm == "L5-CTRL" else "O12"
    incidence = incidence_override or M["IET"].incidence_lists(topology, mode)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(frames), 16):
            chunk = frames[start:start + 16]
            images = normalise(np.stack([f.image for f in chunk])).to(device)
            high, low = encoder.taps(images)
            probabilities.append(torch.sigmoid(model(high, low)).float().cpu().numpy())
    probabilities = np.concatenate(probabilities)
    if permutation is not None:
        probabilities = probabilities[:, permutation]

    rows: list[dict[str, Any]] = []
    visibility_rows: list[dict[str, Any]] = []
    r4 = r6 = pnp = 0
    reproj: list[float] = []
    for index, frame in enumerate(frames):
        width, height = frame.size
        points = decode_batch(probabilities[index], incidence, grid, width, height)
        frame_rows = corner_rows(points, frame.corners)
        rows.extend(frame_rows)
        good = sum(1 for r in frame_rows if r["err"] <= CORNER_OK_PX)
        r4 += int(good >= R4_MIN)
        r6 += int(good >= R6_MIN)
        pose = frame.eval_frame.solve(points + [None])
        pnp += int(pose is not None)
        metrics = frame.eval_frame.metrics(pose)
        if metrics["reproj_fixed_gt_px"] is not None:
            reproj.append(metrics["reproj_fixed_gt_px"])
        for corner in range(8):
            truth = frame.corners[corner]
            if truth is None:
                continue
            states = [frame.visibility[k] for k in incidence[corner]]
            visibility_rows.append({
                "corner": corner,
                "err": float(np.hypot(points[corner][0] - truth[0],
                                      points[corner][1] - truth[1])),
                "all_visible": all(s == "VISIBLE" for s in states),
                "any_occluded": any(s == "OCCLUDED" for s in states),
                "any_off_frame": any(s == "OFF_FRAME" for s in states)})
    summary = summarise_corners(rows, len(frames), r4, r6, pnp, reproj)
    summary.update({"arm": arm, "set": label})
    if arm != "L5-CTRL":
        targets = np.stack([f.target12 for f in frames])
        masks = np.stack([f.chmask12 for f in frames])
        summary.update(map_metrics(probabilities, targets, masks))
    summary["visibility"] = visibility_breakdown(visibility_rows)
    return summary


# ============================================================================
# streamed untouched evaluation
# ============================================================================
class Accumulator:
    """Aggregates corner, PnP, map and visibility statistics over chunks."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.rows: list[dict[str, Any]] = []
        self.visibility: list[dict[str, Any]] = []
        self.frames = self.r4 = self.r6 = self.pnp = 0
        self.reproj: list[float] = []
        self.truth_positive = np.zeros(channels)
        self.truth_hit = np.zeros(channels)
        self.pred_positive = np.zeros(channels)
        self.pred_hit = np.zeros(channels)
        self.activation = np.zeros(channels)
        self.confusion = np.zeros((channels, channels))
        self.confusion_rows = np.zeros(channels)
        self.state_counts = {s: 0 for s in ("VISIBLE", "OCCLUDED", "OFF_FRAME", "UNKNOWN")}

    def add_maps(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        predicted = torch.from_numpy((probabilities >= 0.5).astype(np.float32))
        truth = torch.from_numpy((targets >= 0.5).astype(np.float32))
        kernel = 2 * TOLERANCE_CELLS + 1
        dilated_truth = F.max_pool2d(truth, kernel, 1, TOLERANCE_CELLS)
        dilated_pred = F.max_pool2d(predicted, kernel, 1, TOLERANCE_CELLS)
        self.truth_positive += truth.sum(dim=(0, 2, 3)).numpy()
        self.truth_hit += (truth * dilated_pred).sum(dim=(0, 2, 3)).numpy()
        self.pred_positive += predicted.sum(dim=(0, 2, 3)).numpy()
        self.pred_hit += (predicted * dilated_truth).sum(dim=(0, 2, 3)).numpy()
        flat = probabilities.reshape(probabilities.shape[0], self.channels, -1)
        self.activation += (flat.max(axis=2) >= 0.5).sum(axis=0)
        for k in range(self.channels):
            support = targets[:, k] >= 0.5
            if not support.any():
                continue
            self.confusion_rows[k] += 1
            for m in range(self.channels):
                self.confusion[k, m] += probabilities[:, m][support].mean()

    def summary(self) -> dict[str, Any]:
        recall = self.truth_hit / np.maximum(self.truth_positive, 1)
        precision = self.pred_hit / np.maximum(self.pred_positive, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        activation = self.activation / max(self.frames, 1)
        confusion = self.confusion / np.maximum(self.confusion_rows[:, None], 1)
        confusion = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1e-9)
        out = summarise_corners(self.rows, self.frames, self.r4, self.r6,
                                self.pnp, self.reproj)
        out.update({
            "per_channel_recall": [float(v) for v in recall],
            "per_channel_precision": [float(v) for v in precision],
            "macro_f1": float(np.mean(f1)),
            "channel_activation": [float(v) for v in activation],
            "active_channels": int((activation >= 0.05).sum()),
            "min_channel_recall": float(recall.min()),
            "confusion": confusion.tolist(),
            "visibility": visibility_breakdown(self.visibility),
            "edge_state_counts": dict(self.state_counts),
        })
        return out


def accumulate_chunk(accumulator: Accumulator, arm: str, probabilities: np.ndarray,
                     frames: list[SyntheticFrame], incidence: list[list[int]],
                     topology: dict[str, Any]) -> None:
    M = modules()
    grid = arm_grid(arm)
    targets = np.stack([f.target5 if arm == "L5-CTRL" else f.target12 for f in frames])
    accumulator.add_maps(probabilities, targets)
    decoded = [decode_batch(probabilities[index], incidence, grid, *frame.size)
               for index, frame in enumerate(frames)]
    solved = solve_many([
        (points + [None], frame.K, frame.dims, (frame.size[1], frame.size[0], 3),
         list(frame.corners) + [None])
        for points, frame in zip(decoded, frames)])
    for index, frame in enumerate(frames):
        points = decoded[index]
        rows = corner_rows(points, frame.corners)
        accumulator.rows.extend(rows)
        good = sum(1 for r in rows if r["err"] <= CORNER_OK_PX)
        accumulator.frames += 1
        accumulator.r4 += int(good >= R4_MIN)
        accumulator.r6 += int(good >= R6_MIN)
        success, value = solved[index]
        accumulator.pnp += int(success)
        if value is not None:
            accumulator.reproj.append(value)
        for state in frame.visibility:
            accumulator.state_counts[state] = accumulator.state_counts.get(state, 0) + 1
        for corner in range(8):
            truth = frame.corners[corner]
            if truth is None:
                continue
            states = [frame.visibility[k] for k in incidence[corner]]
            accumulator.visibility.append({
                "corner": corner,
                "err": float(np.hypot(points[corner][0] - truth[0],
                                      points[corner][1] - truth[1])),
                "all_visible": all(s == "VISIBLE" for s in states),
                "any_occluded": any(s == "OCCLUDED" for s in states),
                "any_off_frame": any(s == "OFF_FRAME" for s in states)})


# ============================================================================
# gates
# ============================================================================
SYNTHETIC_GATE = {
    "val_le20": 0.70, "untouched_le20": 0.60,
    "val_r4": 0.85, "untouched_r4": 0.75,
    "val_pnp": 0.75, "untouched_pnp": 0.60,
    "shuffle_margin": 0.30, "alignment_gap": 0.15,
    "active_channels": 12, "min_channel_recall": 0.30, "seed_range": 0.15,
}
CANONICAL_GATE = {
    "oracle_fraction": 0.50, "shuffle_margin": 0.20, "alignment_gap": 0.20,
    "occluded_fraction": 0.40,
}


def synthetic_gate(arm_results: dict[str, Any]) -> dict[str, Any]:
    """The nine synthetic conditions, all fixed before the first run."""
    seeds = arm_results["seeds"]
    best = arm_results["best_seed"]
    val = seeds[best]["val"]
    untouched = seeds[best]["untouched"]
    checks = {
        "1_val_le20>=0.70": val["corner_le20"] >= SYNTHETIC_GATE["val_le20"],
        "2_untouched_le20>=0.60": untouched["corner_le20"] >= SYNTHETIC_GATE["untouched_le20"],
        "3_r4": (val["r4"] >= SYNTHETIC_GATE["val_r4"]
                 and untouched["r4"] >= SYNTHETIC_GATE["untouched_r4"]),
        "4_pnp": (val["pnp_rate"] >= SYNTHETIC_GATE["val_pnp"]
                  and untouched["pnp_rate"] >= SYNTHETIC_GATE["untouched_pnp"]),
        "5_fixed_beats_shuffled>=30pp": (
            val["corner_le20"] - seeds[best]["val_shuffled"]["corner_le20"]
            >= SYNTHETIC_GATE["shuffle_margin"]),
        "6_fixed_vs_aligned<=15pp": (
            abs(seeds[best]["val_aligned"]["corner_le20"] - val["corner_le20"])
            <= SYNTHETIC_GATE["alignment_gap"]),
        "7_active_channels==12": (untouched.get("active_channels", 0)
                                  >= SYNTHETIC_GATE["active_channels"]),
        "8_min_channel_recall>=0.30": (untouched.get("min_channel_recall", 0.0)
                                       >= SYNTHETIC_GATE["min_channel_recall"]),
        "9_seed_range<=15pp": arm_results["seed_range_le20"] <= SYNTHETIC_GATE["seed_range"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def synthetic_taxonomy(arm_results: dict[str, Any]) -> list[str]:
    seeds = arm_results["seeds"]
    best = arm_results["best_seed"]
    val = seeds[best]["val"]
    untouched = seeds[best]["untouched"]
    aligned = seeds[best]["val_aligned"]
    labels = []
    if (aligned["corner_le20"] - val["corner_le20"]) > SYNTHETIC_GATE["alignment_gap"]:
        labels.append("PERMUTATION_FAILURE")
    if (untouched.get("active_channels", 0) < 10
            or untouched.get("min_channel_recall", 1.0) < 0.10):
        labels.append("CHANNEL_COLLAPSE")
    visible = untouched.get("visibility", {}).get("all_incident_visible", {}).get("le20")
    occluded = untouched.get("visibility", {}).get("any_incident_occluded", {}).get("le20")
    if visible is not None and occluded is not None and (visible - occluded) > 0.30:
        labels.append("VISIBLE_ONLY_FAILURE")
    if untouched["corner_median"] > 20 and untouched.get("macro_f1", 0.0) >= 0.5:
        labels.append("LOCALIZATION_FAILURE")
    if (untouched["corner_le20"] < 0.40
            and aligned.get("corner_le20", 0.0) < 0.40):
        labels.append("DIRECT_HEAD_FAILURE")
    return labels


def canonical_gate(arm_results: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    seeds = arm_results["seeds"]
    best = arm_results["best_seed"]
    fraction = {}
    for label in ("eval56", "wood"):
        reference = oracle[label]
        learned = seeds[best][label]
        fraction[label] = {
            "le20_fraction": (learned["corner_le20"] / reference["le20"]
                              if reference["le20"] > 0 else None),
            "pnp_fraction": (learned["pnp"] / reference["pnp"]
                             if reference["pnp"] > 0 else None),
            "r4_fraction": learned["r4"],
        }
    directions = sum(
        1 for seed in seeds
        if seeds[seed]["eval56"]["corner_le20"] / max(oracle["eval56"]["le20"], 1e-9)
        >= CANONICAL_GATE["oracle_fraction"])
    visible = seeds[best]["eval56"].get("visibility", {}).get(
        "all_incident_visible", {}).get("le20")
    occluded = seeds[best]["eval56"].get("visibility", {}).get(
        "any_incident_occluded", {}).get("le20")
    checks = {
        "1_eval56_le20>=50%_oracle": (fraction["eval56"]["le20_fraction"] or 0)
        >= CANONICAL_GATE["oracle_fraction"],
        "2_wood_le20>=50%_oracle": (fraction["wood"]["le20_fraction"] or 0)
        >= CANONICAL_GATE["oracle_fraction"],
        "3_eval56_pnp>=50%_oracle": (fraction["eval56"]["pnp_fraction"] or 0)
        >= CANONICAL_GATE["oracle_fraction"],
        "4_wood_pnp>=50%_oracle": (fraction["wood"]["pnp_fraction"] or 0)
        >= CANONICAL_GATE["oracle_fraction"],
        "5_fixed_beats_shuffled>=20pp": (
            seeds[best]["eval56"]["corner_le20"]
            - seeds[best]["eval56_shuffled"]["corner_le20"]
            >= CANONICAL_GATE["shuffle_margin"]),
        "6_active_channels==12": seeds[best]["eval56"].get("active_channels", 0) >= 12,
        "7_fixed_vs_aligned<=20pp": abs(
            seeds[best]["eval56_aligned"]["corner_le20"]
            - seeds[best]["eval56"]["corner_le20"]) <= CANONICAL_GATE["alignment_gap"],
        "8_no_collapse": bool(np.isfinite(seeds[best]["eval56"]["corner_median"])
                              and np.isfinite(seeds[best]["wood"]["corner_median"])),
        "9_occluded>=40%_of_visible": (
            occluded is not None and visible is not None and visible > 0
            and occluded / visible >= CANONICAL_GATE["occluded_fraction"]),
        "10_two_of_three_seeds": directions >= 2,
    }
    return {"checks": checks, "passed": all(checks.values()), "fraction": fraction,
            "seeds_meeting_direction": directions}


# ============================================================================
# phase drivers
# ============================================================================
def load_topology() -> dict[str, Any]:
    return json.loads((RESULT_ROOT / "topology.json").read_text("utf-8"))


def arm_dir(arm: str, seed: int) -> pathlib.Path:
    return WEIGHT_ROOT / arm.replace("-", "_") / f"seed{seed}"


def completed_runs() -> list[tuple[str, int]]:
    out = []
    for arm in ARMS:
        for seed in ARM_SEEDS[arm]:
            path = arm_dir(arm, seed) / "run_state.json"
            if path.is_file() and json.loads(path.read_text("utf-8")).get("completed"):
                out.append((arm, seed))
    return out


def load_selected(arm: str, seed: int, encoder: FrozenEncoder):
    """Checkpoint chosen on synthetic validation only."""
    directory = arm_dir(arm, seed)
    history = json.loads((directory / "metrics_by_epoch.json").read_text("utf-8"))
    best = select_checkpoint(history)
    path = directory / f"epoch_{best['epoch']:03d}.pth"
    model = build_arm(arm, seed, encoder)
    model.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=False)["model"])
    model.eval()
    return model, best, path


def phase_smoke(state: State) -> dict[str, Any]:
    topology = load_topology()
    encoder = FrozenEncoder().to(device)
    before = encoder.parameter_checksum()
    frames = load_split("train", topology, want5=True, limit=BATCH * (SMOKE_STEPS + 2))
    results = []
    for arm in ARMS:
        calibration = calibrate(arm, frames, encoder, topology)
        result = train_one(arm, 1, frames, frames[:32], encoder, topology,
                           calibration, smoke=True)
        model = build_arm(arm, 1, encoder)
        probabilities = predict(model, encoder, frames[:4])
        incidence = modules()["IET"].incidence_lists(
            topology, "O5C" if arm == "L5-CTRL" else "O12")
        points = decode_batch(probabilities[0], incidence, arm_grid(arm), 640, 480)
        result.update({
            "output_channels": int(probabilities.shape[1]),
            "output_grid": int(probabilities.shape[-1]),
            "all_channels_active": bool((probabilities.reshape(
                probabilities.shape[0], probabilities.shape[1], -1).max(axis=2) > 0).all()),
            "decoder_finite": bool(np.isfinite(np.asarray(points)).all()),
            "calibration": calibration,
        })
        results.append(result)
        log(f"[smoke] {arm}: {result['steps']} steps, loss {result['mean_loss']:.4f}, "
            f"head grad L1 {result['head_gradient_l1']:.3e}, "
            f"{result['trainable_parameters']} trainable")
    after = encoder.parameter_checksum()
    payload = {"arms": results, "a1_checksum_before": before,
               "a1_checksum_after": after, "a1_unchanged": before == after,
               "passed": all(r["loss_finite"] and r["head_gradient_l1"] > 0
                             and r["decoder_finite"] for r in results)
               and before == after}
    atomic_write(RESULT_ROOT / "smoke.json", json.dumps(payload, indent=1))
    if not payload["passed"]:
        state.set("smoke", "FAILED")
        raise RuntimeError("BLOCKED: smoke failed")
    return payload


def phase_train(state: State) -> dict[str, Any]:
    topology = load_topology()
    encoder = FrozenEncoder().to(device)
    before = encoder.parameter_checksum()
    train_frames = load_split("train", topology, want5=True)
    val_frames = load_split("val", topology, want5=True)
    rows = []
    calibrations = {}
    for arm in ARMS:
        calibrations[arm] = calibrate(arm, train_frames, encoder, topology)
        atomic_write(RESULT_ROOT / f"calibration_{arm}.json",
                     json.dumps(calibrations[arm], indent=1))
        for seed in ARM_SEEDS[arm]:
            history = train_one(arm, seed, train_frames, val_frames, encoder,
                                topology, calibrations[arm])
            best = select_checkpoint(history)
            rows.append({"arm": arm, "seed": seed, "epochs": len(history),
                         "selected_epoch": best["epoch"],
                         "selected_val_le20": best["corner_le20"],
                         "final_train_loss": history[-1]["train_loss"]})
            after = encoder.parameter_checksum()
            if after != before:
                raise RuntimeError("BLOCKED: A1 parameters changed during training")
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "training_runs.csv", index=False)
    payload = {"runs": rows, "a1_unchanged": True,
               "calibrations": calibrations}
    atomic_write(RESULT_ROOT / "training_summary.json", json.dumps(payload, indent=1))
    return payload


def phase_eval_synthetic(state: State, untouched_chunk: int = 400) -> dict[str, Any]:
    M = modules()
    topology = load_topology()
    start_pool()
    encoder = FrozenEncoder().to(device)
    val_frames = load_split("val", topology, want5=True)
    runs = completed_runs()
    if not runs:
        raise RuntimeError("BLOCKED: no completed training run")
    shuffled = M["IET"].shuffled_permutation(1)

    results: dict[str, Any] = {}
    permutations: dict[str, list[int]] = {}
    for arm, seed in runs:
        model, best, path = load_selected(arm, seed, encoder)
        mode = "O5C" if arm == "L5-CTRL" else "O12"
        fixed = M["IET"].incidence_lists(topology, mode)
        shuffle_incidence = M["IET"].incidence_lists(topology, mode, permutation=shuffled)
        val = evaluate_synthetic(arm, model, encoder, val_frames, topology, "val")
        val_shuffled = evaluate_synthetic(arm, model, encoder, val_frames, topology,
                                          "val_shuffled",
                                          incidence_override=shuffle_incidence)
        mapping = hungarian_from_confusion(np.array(val["confusion"]))
        permutations[f"{arm}|{seed}"] = mapping
        val_aligned = evaluate_synthetic(arm, model, encoder, val_frames, topology,
                                         "val_aligned", permutation=mapping)
        results[f"{arm}|{seed}"] = {
            "arm": arm, "seed": seed, "selected_epoch": best["epoch"],
            "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(path),
            "val": val, "val_shuffled": val_shuffled, "val_aligned": val_aligned}
        log(f"[synthetic] {arm} s{seed} val <=20px {100*val['corner_le20']:.1f}% "
            f"(shuffled {100*val_shuffled['corner_le20']:.1f}%, "
            f"aligned {100*val_aligned['corner_le20']:.1f}%)")
    del val_frames

    manifest = json.loads((PPD_ROOT / "ppd_untouched_manifest.json").read_text("utf-8"))
    files = [f["file"] for f in manifest["frames"]]
    accumulators = {key: {"fixed": Accumulator(arm_channels(results[key]["arm"])),
                          "aligned": Accumulator(arm_channels(results[key]["arm"]))}
                    for key in results}
    models = {key: load_selected(results[key]["arm"], results[key]["seed"], encoder)[0]
              for key in results}
    # A Hungarian mapping that comes back as the identity means the predicted
    # channels already sit on their physical edges.  Re-running the whole
    # untouched pass through an identity permutation would repeat the same
    # solves for the same answer, so it is copied instead of recomputed.
    identity = {key: permutations[key] == list(range(arm_channels(results[key]["arm"])))
                for key in results}
    log(f"[synthetic] identity alignment: "
        f"{sum(identity.values())}/{len(identity)} arms (their aligned pass is copied)")
    log(f"[synthetic] streaming untouched: {len(files)} frames, full coverage")
    for start in range(0, len(files), untouched_chunk):
        chunk_files = files[start:start + untouched_chunk]
        chunk = [f for f in (load_synthetic(name, topology, want5=True)
                             for name in chunk_files) if f is not None]
        for key in results:
            arm = results[key]["arm"]
            mode = "O5C" if arm == "L5-CTRL" else "O12"
            fixed = M["IET"].incidence_lists(topology, mode)
            probabilities = predict(models[key], encoder, chunk)
            accumulate_chunk(accumulators[key]["fixed"], arm, probabilities, chunk,
                             fixed, topology)
            if not identity[key]:
                accumulate_chunk(accumulators[key]["aligned"], arm,
                                 probabilities[:, permutations[key]], chunk,
                                 fixed, topology)
        log(f"[synthetic] untouched {min(start+untouched_chunk, len(files))}/{len(files)}")
    for key in results:
        results[key]["untouched"] = accumulators[key]["fixed"].summary()
        results[key]["untouched_aligned"] = (
            results[key]["untouched"] if identity[key]
            else accumulators[key]["aligned"].summary())

    assets_train = {f.get("asset") for f in json.loads(
        (PPD_ROOT / "ppd_train_manifest.json").read_text("utf-8"))["frames"]}
    assets_untouched = {f.get("asset") for f in manifest["frames"]}
    held_out = sorted(assets_untouched - assets_train)

    grouped: dict[str, Any] = {}
    for arm in ARMS:
        seeds = {seed: results[f"{arm}|{seed}"] for seed in ARM_SEEDS[arm]
                 if f"{arm}|{seed}" in results}
        if not seeds:
            continue
        values = [v["val"]["corner_le20"] for v in seeds.values()]
        best_seed = max(seeds, key=lambda s: seeds[s]["val"]["corner_le20"])
        grouped[arm] = {"seeds": seeds, "best_seed": best_seed,
                        "seed_range_le20": float(max(values) - min(values)),
                        "permutations": {str(s): permutations[f"{arm}|{s}"]
                                         for s in seeds}}
        grouped[arm]["gate"] = synthetic_gate(grouped[arm])
        grouped[arm]["taxonomy"] = synthetic_taxonomy(grouped[arm])
        log(f"[synthetic] {arm} gate {'PASS' if grouped[arm]['gate']['passed'] else 'FAIL'}"
            f"  taxonomy {grouped[arm]['taxonomy']}")

    payload = {"arms": grouped,
               "asset_held_out_available": bool(held_out),
               "held_out_assets": held_out,
               "untouched_frames": len(files),
               "note_coverage": "untouched evaluated in full; nothing sampled or capped"}
    atomic_write(RESULT_ROOT / "synthetic_results.json", json.dumps(payload, indent=1))
    flatten_to_csv(grouped, RESULT_ROOT / "synthetic_corner_metrics.csv",
                   ("val", "val_shuffled", "val_aligned", "untouched",
                    "untouched_aligned"))
    return payload


def flatten_to_csv(grouped: dict[str, Any], path: pathlib.Path,
                   keys: tuple[str, ...]) -> None:
    rows = []
    for arm, entry in grouped.items():
        for seed, seed_entry in entry["seeds"].items():
            for key in keys:
                if key not in seed_entry:
                    continue
                block = seed_entry[key]
                rows.append({"arm": arm, "seed": seed, "eval": key,
                             **{k: v for k, v in block.items()
                                if not isinstance(v, (dict, list))}})
    pd.DataFrame(rows).to_csv(path, index=False)


def phase_eval_canonical(state: State) -> dict[str, Any]:
    M = modules()
    topology = load_topology()
    synthetic = json.loads((RESULT_ROOT / "synthetic_results.json").read_text("utf-8"))
    encoder = FrozenEncoder().to(device)
    shuffled = M["IET"].shuffled_permutation(1)
    frames = {label: load_canonical(topology, label) for label in ("eval56", "wood")}
    log(f"[canonical] eval56 {len(frames['eval56'])}  wood {len(frames['wood'])}")

    oracle_rows = json.loads((RESULT_ROOT / "oracle_parity.json").read_text("utf-8"))["rows"]
    oracle = {row["set"]: row for row in oracle_rows if row["mode"] == "O12"}
    oracle_o5 = {row["set"]: row for row in oracle_rows if row["mode"] == "O5"}

    grouped: dict[str, Any] = {}
    for arm, entry in synthetic["arms"].items():
        seeds: dict[str, Any] = {}
        for seed_key, seed_entry in entry["seeds"].items():
            seed = int(seed_key)
            model, best, path = load_selected(arm, seed, encoder)
            mode = "O5C" if arm == "L5-CTRL" else "O12"
            fixed = M["IET"].incidence_lists(topology, mode)
            shuffle_incidence = M["IET"].incidence_lists(topology, mode,
                                                         permutation=shuffled)
            mapping = entry["permutations"][str(seed)]
            block: dict[str, Any] = {"selected_epoch": best["epoch"],
                                     "checkpoint": str(path.relative_to(ROOT))}
            for label in ("eval56", "wood"):
                block[label] = evaluate_canonical(arm, model, encoder, frames[label],
                                                  topology, label)
                block[f"{label}_shuffled"] = evaluate_canonical(
                    arm, model, encoder, frames[label], topology, f"{label}_shuffled",
                    incidence_override=shuffle_incidence)
                block[f"{label}_aligned"] = evaluate_canonical(
                    arm, model, encoder, frames[label], topology, f"{label}_aligned",
                    permutation=mapping)
                log(f"[canonical] {arm} s{seed} {label}: <=20px "
                    f"{100*block[label]['corner_le20']:.1f}% "
                    f"PnP {block[label]['pnp']}  R4 {100*block[label]['r4']:.1f}%")
            seeds[seed] = block
        grouped[arm] = {"seeds": seeds,
                        "best_seed": int(entry["best_seed"])}
        grouped[arm]["gate"] = canonical_gate(
            grouped[arm], {label: {"le20": oracle[label]["le20"],
                                   "pnp": oracle[label]["pnp"]}
                           for label in ("eval56", "wood")})
        log(f"[canonical] {arm} gate "
            f"{'PASS' if grouped[arm]['gate']['passed'] else 'FAIL'}")

    local = local_reference()
    payload = {"arms": grouped, "oracle_o12": oracle, "oracle_o5": oracle_o5,
               "local_corner_reference": local}
    atomic_write(RESULT_ROOT / "canonical_results.json", json.dumps(payload, indent=1))
    flatten_to_csv(grouped, RESULT_ROOT / "canonical_corner_metrics.csv",
                   ("eval56", "eval56_shuffled", "eval56_aligned",
                    "wood", "wood_shuffled", "wood_aligned"))
    return payload


def local_reference() -> dict[str, Any]:
    """A0/A1 local corner performance, read from the Stage 1 canonical re-eval."""
    path = (RESULT_ROOT.parent.parent / "pdg_unified_program/stage1_failure_audit"
            / "canonical_reeval.csv")
    if not path.is_file():
        return {"available": False, "source": str(path)}
    table = pd.read_csv(path)
    out: dict[str, Any] = {"available": True, "source": str(path.relative_to(ROOT))}
    for (arm, subset), block in table.groupby(["arm", "set"]):
        out[f"{arm}|{subset}"] = {
            "n": int(len(block)), "pnp": int(block["pnp"].sum()),
            "r4": float(block["R4"].mean()),
            "corner_median_px": float(block["cmed"].median()),
            "reproj_median_px": float(block["reproj"].median())}
    return out


def phase_ppd(state: State) -> dict[str, Any]:
    """Close the five-class history on the canonical sets, read-only."""
    M = modules()
    topology = load_topology()
    encoder = FrozenEncoder().to(device)
    incidence = M["IET"].incidence_lists(topology, "O5C")
    frames = {label: load_canonical(topology, label) for label in ("eval56", "wood")}
    rows = []
    for arm in ("L0", "M1"):
        checkpoint = PPD_WEIGHTS / arm / "last.pth"
        if not checkpoint.is_file():
            continue
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        head = M["PLH"].PolarityLineHead(encoder.high_channels).to(device)
        head.load_state_dict(blob["line"])
        head.eval()
        mask_head = M["PLH"].FreshMaskHead(encoder.high_channels).to(device)
        mask_head.load_state_dict(blob["mask"])
        mask_head.eval()
        for label, block in frames.items():
            errors: list[float] = []
            pnp = 0
            activations = []
            with torch.no_grad():
                for start in range(0, len(block), 16):
                    chunk = block[start:start + 16]
                    images = normalise(np.stack([f.image for f in chunk])).to(device)
                    high, _ = encoder.taps(images)
                    gate = (M["PLH"].soft_gate(mask_head(high)) if arm == "M1"
                            else torch.ones_like(high[:, :1]))
                    probability = torch.sigmoid(head(high, gate)).cpu().numpy()
                    activations.append(probability.reshape(
                        probability.shape[0], 5, -1).max(axis=2))
                    for index, frame in enumerate(chunk):
                        width, height = frame.size
                        points = decode_batch(probability[index], incidence, GRID_5,
                                              width, height)
                        for corner in range(8):
                            truth = frame.corners[corner]
                            if truth is None:
                                continue
                            errors.append(float(np.hypot(points[corner][0] - truth[0],
                                                         points[corner][1] - truth[1])))
                        pnp += int(frame.eval_frame.solve(points + [None]) is not None)
            array = np.array(errors)
            peak = np.concatenate(activations)
            rows.append({"arm": f"PPD-{arm}", "set": label, "n_frames": len(block),
                         "corner_le20": float((array <= CORNER_OK_PX).mean()),
                         "corner_median": float(np.median(array)),
                         "pnp": pnp,
                         "channel_activation": [float(v) for v in (peak >= 0.5).mean(axis=0)],
                         "mean_peak": float(peak.mean())})
            log(f"[ppd] {arm} {label}: <=20px {100*rows[-1]['corner_le20']:.1f}% "
                f"median {rows[-1]['corner_median']:.1f}px PnP {pnp}")
    payload = {"rows": rows,
               "note": "O5 is structurally unable to place a corner even from ground "
                       "truth, so these numbers close the history; they do not gate "
                       "the twelve-edge decision."}
    atomic_write(RESULT_ROOT / "ppd_canonical.json", json.dumps(payload, indent=1))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)}
                  for r in rows]).to_csv(RESULT_ROOT / "ppd_canonical.csv", index=False)
    return payload


# ============================================================================
# Phase H and J -- decision
# ============================================================================
def best_block(entry: dict[str, Any]) -> dict[str, Any]:
    """The best seed's block, whichever way JSON left the key.

    Seeds are integers in memory and strings after a JSON round-trip, while
    ``best_seed`` stays an integer because it is a value rather than a key.
    Reading the reloaded file with the in-memory key raised KeyError.
    """
    seeds = entry["seeds"]
    key = entry["best_seed"]
    if key in seeds:
        return seeds[key]
    return seeds[str(key)]


def multiscale_verdict(synthetic: dict[str, Any], canonical: dict[str, Any]
                       ) -> dict[str, Any]:
    if "L12-F50" not in synthetic["arms"] or "L12-MS" not in synthetic["arms"]:
        return {"verdict": "NOT_COMPARABLE", "reason": "an arm is missing"}
    f50 = synthetic["arms"]["L12-F50"]
    ms = synthetic["arms"]["L12-MS"]
    f50_best = best_block(f50)
    ms_best = best_block(ms)
    untouched_gain = (ms_best["untouched"]["corner_le20"]
                      - f50_best["untouched"]["corner_le20"])
    canonical_gain = {}
    pnp_drop = False
    for label in ("eval56", "wood"):
        a = best_block(canonical["arms"]["L12-F50"])
        b = best_block(canonical["arms"]["L12-MS"])
        canonical_gain[label] = b[label]["corner_le20"] - a[label]["corner_le20"]
        pnp_drop = pnp_drop or (b[label]["pnp"] < a[label]["pnp"])
    directions = sum(
        1 for seed in ms["seeds"]
        if seed in f50["seeds"]
        and ms["seeds"][seed]["untouched"]["corner_le20"]
        > f50["seeds"][seed]["untouched"]["corner_le20"])
    checks = {
        "1_untouched>=10pp": untouched_gain >= 0.10,
        "2_canonical>=10pp": max(canonical_gain.values()) >= 0.10,
        "3_no_pnp_drop": not pnp_drop,
        "4_two_of_three_seeds": directions >= 2,
    }
    if all(checks.values()):
        verdict = "MULTISCALE_REQUIRED"
    elif untouched_gain <= 0.0:
        verdict = "MULTISCALE_NO_VALUE"
    else:
        verdict = "F50_SUFFICIENT"
    return {"verdict": verdict, "checks": checks,
            "untouched_gain_pp": 100 * untouched_gain,
            "canonical_gain_pp": {k: 100 * v for k, v in canonical_gain.items()},
            "seeds_same_direction": directions}


def phase_decide(state: State) -> dict[str, Any]:
    synthetic = json.loads((RESULT_ROOT / "synthetic_results.json").read_text("utf-8"))
    canonical = json.loads((RESULT_ROOT / "canonical_results.json").read_text("utf-8"))
    oracle = canonical["oracle_o12"]

    twelve = [arm for arm in ("L12-F50", "L12-MS") if arm in synthetic["arms"]]
    if not twelve:
        decision = "BLOCKED"
        primary = None
        detail: dict[str, Any] = {"reason": "no twelve-edge arm completed"}
    else:
        passing = [arm for arm in twelve if synthetic["arms"][arm]["gate"]["passed"]]
        primary = max(twelve, key=lambda a: best_block(
            synthetic["arms"][a])["untouched"]["corner_le20"])
        detail = {}
        if not passing:
            taxonomy = set()
            for arm in twelve:
                taxonomy.update(synthetic["arms"][arm]["taxonomy"])
            if "PERMUTATION_FAILURE" in taxonomy:
                decision = "INSTANCE_ID_PERMUTATION_FAILURE"
            elif "VISIBLE_ONLY_FAILURE" in taxonomy:
                decision = "OCCLUDED_EDGE_INFERENCE_FAILURE"
            else:
                decision = "DIRECT_12EDGE_HEAD_NOT_LEARNABLE"
            detail["taxonomy"] = sorted(taxonomy)
        else:
            primary = max(passing, key=lambda a: best_block(
                synthetic["arms"][a])["untouched"]["corner_le20"])
            gate = canonical["arms"][primary]["gate"]
            fractions = gate["fraction"]
            le20 = [fractions[s]["le20_fraction"] or 0 for s in ("eval56", "wood")]
            pnp = [fractions[s]["pnp_fraction"] or 0 for s in ("eval56", "wood")]
            if gate["passed"]:
                decision = "INSTANCE_LINE_LEARNABLE"
            elif max(le20 + pnp) < 0.25:
                decision = "REPRESENTATION_VALID_PREDICTOR_TRANSFER_FAIL"
            elif any(0.25 <= v < 0.50 for v in le20 + pnp):
                decision = "INSTANCE_LINE_PARTIAL_TRANSFER"
            else:
                decision = "INSTANCE_LINE_PARTIAL_TRANSFER"
            detail["canonical_gate"] = gate
            if passing == ["L12-MS"]:
                decision = ("MULTISCALE_REQUIRED" if decision == "INSTANCE_LINE_LEARNABLE"
                            else decision)
                detail["note"] = "only the multi-scale arm passed the synthetic gate"

    scale = multiscale_verdict(synthetic, canonical)
    architecture = {
        "IAEH": "GO" if decision == "INSTANCE_LINE_LEARNABLE" else "STOP",
        "CIGM": "GO" if decision == "INSTANCE_LINE_LEARNABLE" else "STOP",
        "fusion": "GO" if decision == "INSTANCE_LINE_LEARNABLE" else "STOP",
        "next_case": {
            "INSTANCE_LINE_LEARNABLE": "IAEH_FIRST_GO",
            "INSTANCE_LINE_PARTIAL_TRANSFER": "IAEH_TRANSFER_STABILIZATION_FIRST",
            "REPRESENTATION_VALID_PREDICTOR_TRANSFER_FAIL":
                "INSTANCE_EDGE_DOMAIN_TRANSFER_FIRST",
            "INSTANCE_ID_PERMUTATION_FAILURE": "INSTANCE_ID_STABILIZATION_REQUIRED",
            "OCCLUDED_EDGE_INFERENCE_FAILURE": "AMODAL_EDGE_INFERENCE_REQUIRED",
            "DIRECT_12EDGE_HEAD_NOT_LEARNABLE": "DIRECT_INSTANCE_CHANNEL_STOP",
            "MULTISCALE_REQUIRED": "MULTISCALE_INSTANCE_EDGE_HEAD_REQUIRED",
            "BLOCKED": "BLOCKED",
        }.get(decision, "BLOCKED"),
    }
    payload = {"decision": decision, "primary_arm": primary, "detail": detail,
               "multiscale": scale, "architecture": architecture,
               "oracle_reference": oracle,
               "note": "O12 is a representation-capacity oracle using ground-truth "
                       "geometry; the learned arms are a separate claim."}
    atomic_write(RESULT_ROOT / "final_decision.json", json.dumps(payload, indent=1))
    log(f"[decide] {decision}  primary {primary}  multiscale {scale['verdict']}")
    return payload


# ============================================================================
# reports
# ============================================================================
def table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    """Space-aligned block: pipe tables break when Korean text is mixed in."""
    header = "  ".join(f"{title:>{width}}" if index else f"{title:<{width}}"
                       for index, (key, title, width) in
                       enumerate((k, t, int(w)) for k, t, w in columns))
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for index, (key, title, width) in enumerate(
                (k, t, int(w)) for k, t, w in columns):
            value = row.get(key)
            if isinstance(value, float):
                text = f"{value:.4g}"
            elif value is None:
                text = "-"
            else:
                text = str(value)
            cells.append(f"{text:>{width}}" if index else f"{text:<{width}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def phase_report(state: State) -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((RESULT_ROOT / "input_lock.json").read_text("utf-8"))
    topology = load_topology()
    oracle = json.loads((RESULT_ROOT / "oracle_parity.json").read_text("utf-8"))
    parity = json.loads((RESULT_ROOT / "target_parity.json").read_text("utf-8"))
    recipe = json.loads((RESULT_ROOT / "ppd_recipe.json").read_text("utf-8"))
    written = []

    def write(name: str, body: str) -> None:
        atomic_write(REPORT_ROOT / name, body)
        written.append(name)

    write("INSTANCE_EDGE_INPUT_LOCK.md",
          "# Input lock\n\n```\n" + json.dumps(
              {k: v for k, v in lock.items() if k != "a1_run_state"}, indent=1)
          + "\n```\n")
    write("INSTANCE_EDGE_TOPOLOGY.md",
          "# Twelve-edge topology\n\nDerived from "
          "`annotate_pnp.make_pallet_keypoints_3d` by the rule that a physical edge is "
          "a corner pair differing along exactly one local axis.  No index is written "
          "by hand.\n\n```\n"
          + table([{"id": k, "edge": tuple(e), "class": c,
                    "incident_corners": tuple(e)}
                   for k, (e, c) in enumerate(zip(topology["edges"],
                                                  topology["edge_classes"]))],
                  [("id", "id", 3), ("edge", "corners", 10), ("class", "class", 12)])
          + "\n\nclass counts  " + json.dumps(topology["class_counts"])
          + "\npolarity pairs " + json.dumps(topology["polarity_pairs"])
          + "\nsha256        " + topology["topology_sha256"] + "\n```\n")
    write("INSTANCE_EDGE_TARGET_PARITY.md",
          "# Target generator parity\n\nGround-truth endpoints through the target "
          "generator and back out of the fixed O12 decoder, against the same decoder "
          "run on analytic geometry.\n\n```\n" + json.dumps(parity, indent=1)
          + "\n```\n\nThe parity population was fixed before the run: a corner counts "
            "only when all three incident edges have an in-frame clipped segment and "
            "the corner itself projects inside the image.  A corner outside the frame "
            "cannot be represented in a rasterised field, so including it would "
            "measure clipping rather than the generator.\n")
    write("PPD_RECIPE_PARITY.md",
          "# PPD recipe trace\n\n```\n" + json.dumps(recipe, indent=1) + "\n```\n")
    write("INSTANCE_EDGE_ORACLE_PARITY.md",
          "# Oracle parity\n\n```\n"
          + table(oracle["rows"],
                  [("mode", "mode", 5), ("set", "set", 8), ("n_frames", "n", 4),
                   ("le20", "<=20px", 8), ("median", "median", 8), ("pnp", "PnP", 5),
                   ("reference_le20", "ref<=20", 8), ("parity", "parity", 7)])
          + "\n```\n")

    smoke_path = RESULT_ROOT / "smoke.json"
    if smoke_path.is_file():
        write("INSTANCE_EDGE_SMOKE.md", "# Smoke\n\n```\n"
              + json.dumps(json.loads(smoke_path.read_text("utf-8")), indent=1) + "\n```\n")

    synthetic_path = RESULT_ROOT / "synthetic_results.json"
    canonical_path = RESULT_ROOT / "canonical_results.json"
    decision_path = RESULT_ROOT / "final_decision.json"
    if synthetic_path.is_file():
        synthetic = json.loads(synthetic_path.read_text("utf-8"))
        rows = []
        for arm, entry in synthetic["arms"].items():
            for seed, block in entry["seeds"].items():
                for key in ("val", "val_shuffled", "val_aligned", "untouched"):
                    if key in block:
                        rows.append({"arm": arm, "seed": seed, "eval": key,
                                     **block[key]})
        write("INSTANCE_EDGE_SYNTHETIC_LEARNABILITY.md",
              "# Synthetic learnability\n\n```\n"
              + table(rows, [("arm", "arm", 9), ("seed", "s", 2), ("eval", "eval", 16),
                             ("corner_le20", "<=20px", 8), ("corner_median", "median", 8),
                             ("r4", "R4", 7), ("pnp_rate", "PnP", 7),
                             ("macro_f1", "F1", 7), ("active_channels", "act", 4)])
              + "\n```\n\n```\n"
              + json.dumps({arm: {"gate": entry["gate"], "taxonomy": entry["taxonomy"],
                                  "seed_range_le20": entry["seed_range_le20"]}
                            for arm, entry in synthetic["arms"].items()}, indent=1)
              + "\n```\n")
        write("INSTANCE_EDGE_CHANNEL_STABILITY.md",
              "# Channel identity stability\n\n```\n"
              + json.dumps({arm: {"permutations": entry["permutations"],
                                  "fixed_vs_aligned": {
                                      seed: {"fixed": block["val"]["corner_le20"],
                                             "aligned": block["val_aligned"]["corner_le20"]}
                                      for seed, block in entry["seeds"].items()}}
                            for arm, entry in synthetic["arms"].items()}, indent=1)
              + "\n```\n")
        write("INSTANCE_EDGE_VISIBILITY_ANALYSIS.md",
              "# Visibility decomposition\n\n```\n"
              + json.dumps({arm: {seed: {"untouched": block["untouched"].get("visibility"),
                                         "edge_states": block["untouched"].get(
                                             "edge_state_counts")}
                                  for seed, block in entry["seeds"].items()}
                            for arm, entry in synthetic["arms"].items()}, indent=1)
              + "\n```\n")
    if canonical_path.is_file():
        canonical = json.loads(canonical_path.read_text("utf-8"))
        rows = []
        for arm, entry in canonical["arms"].items():
            for seed, block in entry["seeds"].items():
                for key in ("eval56", "eval56_shuffled", "eval56_aligned",
                            "wood", "wood_shuffled", "wood_aligned"):
                    if key in block:
                        rows.append({"arm": arm, "seed": seed, "eval": key, **block[key]})
        write("INSTANCE_EDGE_CANONICAL_TRANSFER.md",
              "# Canonical transfer\n\nLine-only corner generation: no corner heatmap "
              "and no top-K enter the decoder.\n\n```\n"
              + table(rows, [("arm", "arm", 9), ("seed", "s", 2), ("eval", "eval", 18),
                             ("corner_le20", "<=20px", 8), ("corner_median", "median", 8),
                             ("r4", "R4", 7), ("pnp", "PnP", 5),
                             ("reproj_median", "reproj", 8)])
              + "\n```\n\n```\n"
              + json.dumps({"oracle_o12": canonical["oracle_o12"],
                            "oracle_o5": canonical["oracle_o5"],
                            "local_corner_reference": canonical["local_corner_reference"],
                            "gates": {arm: entry["gate"]
                                      for arm, entry in canonical["arms"].items()}},
                           indent=1)
              + "\n```\n")
    ppd_path = RESULT_ROOT / "ppd_canonical.json"
    if ppd_path.is_file():
        write("PPD_CANONICAL_REFERENCE.md", "# Existing PPD on the canonical sets\n\n```\n"
              + json.dumps(json.loads(ppd_path.read_text("utf-8")), indent=1) + "\n```\n")
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text("utf-8"))
        write("F50_VS_MULTISCALE.md", "# F50 against multi-scale\n\n```\n"
              + json.dumps(decision["multiscale"], indent=1) + "\n```\n")
        write("INSTANCE_EDGE_FINAL_DECISION.md",
              f"# Final decision: {decision['decision']}\n\n```\n"
              + json.dumps(decision, indent=1) + "\n```\n")
        write("INSTANCE_EDGE_NEXT_ARCHITECTURE.md", next_architecture(decision))
    write("INSTANCE_EDGE_PROVENANCE.md",
          "# Provenance\n\n```\n" + json.dumps({
              "result_root": str(RESULT_ROOT.relative_to(ROOT)),
              "weights_root": str(WEIGHT_ROOT.relative_to(ROOT)),
              "head": lock["head"], "a1_sha256": lock["a1_sha256"],
              "topology_sha256": lock["topology_sha256"],
              "final_test_open_count": 0,
              "runner": "scripts/stage0/instance_edge_learnability.py",
              "tests": "challenge/tests/test_instance_edge_learnability.py",
          }, indent=1) + "\n```\n")
    return {"written": written}


def next_architecture(decision: dict[str, Any]) -> str:
    case = decision["architecture"]["next_case"]
    body = [f"# Next architecture protocol: {case}", "",
            "Nothing below is trained in this run.  These are protocols only.", ""]
    if case == "IAEH_FIRST_GO":
        body += [
            "## IAEH -- Instance-Aware Edge Head",
            "Input: the tap the multi-scale gate selected.  Output: twelve physical-edge",
            "fields.  No corner fusion inside the head.", "",
            "## CIGM -- Corner-Incident Geometry Module",
            "Fixed topology: corner i is decoded from its three incident physical edges.",
            "The incidence is never learned.", "",
            "## Fusion, next macro only",
            "Local branch is A1 belief/affinity; structural branch is IAEH + CIGM.",
            "Candidates: confidence-gated coordinate selection, zero-initialised",
            "feature-level fusion, line-generated auxiliary belief, local/line",
            "consistency loss.  Compare line-only, local-only and fused."]
    elif case == "IAEH_TRANSFER_STABILIZATION_FIRST":
        body += ["Transfer is partial.  Stabilise before any fusion:",
                 "object-conditioned edge prediction, synthetic appearance and domain",
                 "augmentation, edge-instance consistency, a small real pseudo-label",
                 "or self-training pass.  Fusion stays closed."]
    elif case == "INSTANCE_EDGE_DOMAIN_TRANSFER_FIRST":
        body += ["The representation survives; the predictor does not cross the domain.",
                 "Keep the twelve-edge target and redesign predictor and domain only."]
    elif case == "INSTANCE_ID_STABILIZATION_REQUIRED":
        body += ["Channel identity is unstable.  Candidates: semantic class plus a",
                 "within-class binary index, endpoint-conditioned identity, a",
                 "permutation-consistency loss, corner-incidence consistency."]
    elif case == "AMODAL_EDGE_INFERENCE_REQUIRED":
        body += ["Visible edges are learned and occluded ones are not.  A visible/",
                 "occluded branch or a geometry completion module comes first."]
    elif case == "DIRECT_INSTANCE_CHANNEL_STOP":
        body += ["Twelve independent instance channels are not learnable here.",
                 "Protocol candidates only: factorised five-class plus instance index,",
                 "twelve edge endpoints, corner-conditioned edge triplets,",
                 "graph-structured edge decoding.  No further training in this run."]
    elif case == "MULTISCALE_INSTANCE_EDGE_HEAD_REQUIRED":
        body += ["Only the multi-scale arm survives.  The edge head needs the",
                 "high-resolution tap; carry F100+F50 into every later design."]
    else:
        body += ["Blocked; nothing to propose."]
    return "\n".join(body) + "\n"


def phase_test(state: State) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q",
               "challenge/tests/test_instance_edge_learnability.py"]
    new = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    full = subprocess.run([sys.executable, "-m", "pytest", "-q", "challenge/tests"],
                          cwd=ROOT, capture_output=True, text=True)
    payload = {"new_tests_returncode": new.returncode,
               "new_tests_tail": new.stdout.strip().splitlines()[-6:],
               "full_returncode": full.returncode,
               "full_tail": full.stdout.strip().splitlines()[-6:]}
    atomic_write(RESULT_ROOT / "tests.json", json.dumps(payload, indent=1))
    log(f"[test] new={new.returncode} full={full.returncode}")
    return payload


# ============================================================================
# driver
# ============================================================================
DRIVERS = {
    "prepare": phase_prepare,
    "oracle-parity": phase_oracle_parity,
    "recipe-parity": phase_recipe_parity,
    "targets": phase_targets,
    "smoke": phase_smoke,
    "train": phase_train,
    "eval-synthetic": phase_eval_synthetic,
    "eval-canonical": phase_eval_canonical,
    "ppd": phase_ppd,
    "decide": phase_decide,
    "test": phase_test,
    "report": phase_report,
}


def run_phase(state: State, phase: str, force: bool = False) -> None:
    if state.get(phase) == "DONE" and not force:
        log(f"[{phase}] already DONE -- skipping")
        return
    if state.get(phase) == "HARD_BLOCKED" and not force:
        raise RuntimeError(f"{phase} is HARD_BLOCKED; refusing to continue")
    state.set(phase, "RUNNING")
    began = time.time()
    try:
        DRIVERS[phase](state)
    except Exception as error:
        state.set(phase, "FAILED", error=repr(error))
        raise
    state.set(phase, "DONE", seconds=round(time.time() - began, 1))


def complete_marker(state: State) -> None:
    decision_path = RESULT_ROOT / "final_decision.json"
    decision = (json.loads(decision_path.read_text("utf-8"))
                if decision_path.is_file() else {"decision": "BLOCKED"})
    selected = {}
    for arm, seed in completed_runs():
        directory = arm_dir(arm, seed)
        history = json.loads((directory / "metrics_by_epoch.json").read_text("utf-8"))
        best = select_checkpoint(history)
        path = directory / f"epoch_{best['epoch']:03d}.pth"
        selected[f"{arm}|{seed}"] = {"epoch": best["epoch"],
                                     "path": str(path.relative_to(ROOT)),
                                     "sha256": sha256_file(path)}
    tests = RESULT_ROOT / "tests.json"
    payload = {
        "final_decision": decision["decision"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_checkpoints": selected,
        "main_model_training_steps": 0,
        "edge_head_training_steps": sum(
            EPOCHS * (3039 // BATCH + 1) for _ in completed_runs()),
        "final_test_open_count": 0,
        "tests": (json.loads(tests.read_text("utf-8")) if tests.is_file() else None),
        "phases": {phase: state.get(phase) for phase in PHASES},
    }
    atomic_write(RESULT_ROOT / "COMPLETE", json.dumps(payload, indent=1))
    log(f"[COMPLETE] {payload['final_decision']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(DRIVERS) + ["all", "resume", "status"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-phase", default=None)
    arguments = parser.parse_args()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    state = State(RESULT_ROOT / "state.json")

    if arguments.command == "status":
        for phase in PHASES:
            print(f"{phase:16s} {state.get(phase)}")
        return
    if arguments.command in ("all", "resume"):
        phases = list(PHASES)
        if arguments.from_phase:
            phases = phases[phases.index(arguments.from_phase):]
        for phase in phases:
            run_phase(state, phase, force=arguments.force and phase == arguments.from_phase)
        complete_marker(state)
        return
    run_phase(state, arguments.command, force=arguments.force)


if __name__ == "__main__":
    main()
