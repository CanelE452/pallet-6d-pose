"""Spatial HCRM screen: does the F50 spatial arrangement become a near-corner
response, and does it survive on the canonical sets.

A1 is frozen throughout; the only trainable parameters are the adapter, whose
residual is added to the four near belief channels the decoder reads.  Far
corners, the centroid and every affinity map are copied through and asserted
bit-identical.

Two things the preflight fixed and this runner must not undo.  The pointwise
control uses ChannelLayerNorm2d rather than GroupNorm, so H1 really is 1x1 and
the arms differ only in the 5x5 support.  And the holdouts are whole source
groups, never sample indices: 13,069 frames carry no scene metadata and stay in
train, which makes the holdouts a KEYED_SOURCE_GROUP_HOLDOUT and not an
unbiased sample of the synthetic distribution.

    python scripts/stage0/spatial_hcrm_screen.py all
    python scripts/stage0/spatial_hcrm_screen.py status
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "scripts/stage0"
for _extra in (STAGE0, ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train",
               ROOT / "challenge/scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/spatial_hcrm_screen")
REPORTS = (ROOT / "_docs/audits/eval56_summary/canonical_corner_audit"
           / "spatial_hcrm_screen")
WEIGHTS = ROOT / "weights/paper_s2_spatial_hcrm"
TRAIN_DATA = ROOT / "data/pallet/training_data"
A1_CKPT = ROOT / "weights/paper_s2_pdg/A1/epoch_003.pth"
A1_SHA = "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657"

SEALED = ("capturenight08", "capturenight09", "capturepallet07", "capturepallet09",
          "testset_full8_manifest", "handannot17")

# Frozen protocol constants.
SEEDS = (1, 2, 3)
ARMS = ("H1", "H2", "H3")
EPOCHS = 3
BATCH = 12
LR = 1e-3
WD = 1e-4
SMOKE_STEPS = 100
THRESHOLD = 0.30
INPUT_SIZE = 400
BELIEF = 50
NEAR = (0, 1, 2, 3)
FAR = (4, 5, 6, 7)
ID12 = (1, 2)
OK_PX = 20.0
HARD_WEIGHT_CAP = 4.0
SHUFFLE_PERMUTATION = (1, 2, 3, 0)
A1_PARITY = {"eval56": (55, 50, 52, 10.1608), "wood": (45, 42, 44, 8.9569)}
PHASES = ("prepare", "parity", "eval-no-aug", "hard-manifest", "smoke", "train",
          "select", "eval-untouched", "eval-canonical", "decide", "report", "test")

# The A1 parity gate compares a reprojection median to 1e-4 px.  Two separately
# constructed instances of the same frozen network came back 5.7e-4 apart in the
# belief map because cuDNN is free to pick a different convolution algorithm,
# which moved the median 0.03px.  Pinning the algorithm choice makes the forward
# reproducible instead of loosening the gate after seeing it fail.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
# access guard -- enforced at the reader, not by convention
# ============================================================================
class AccessGuard:
    """Refuses holdout and canonical reads until the selection lock is written.

    A guard that only records intent would not have caught anything; this one
    raises, and a test deliberately opens an untouched path to prove it fires.
    """

    def __init__(self) -> None:
        self.untouched: set[str] = set()
        self.canonical: set[str] = set()
        self.unlocked = False
        self.counts = {"untouched_before_lock": 0, "canonical_before_lock": 0,
                       "final_test": 0}

    def arm(self, untouched: set[str], canonical: set[str]) -> None:
        self.untouched = {str(p) for p in untouched}
        self.canonical = {str(p) for p in canonical}

    def unlock(self) -> None:
        self.unlocked = True

    def check(self, path: Any) -> Any:
        text = str(path)
        for token in SEALED:
            if token in text:
                self.counts["final_test"] += 1
                raise RuntimeError(f"BLOCKED: sealed token {token!r} in {text}")
        if not self.unlocked:
            if text in self.untouched:
                self.counts["untouched_before_lock"] += 1
                raise RuntimeError(f"BLOCKED: untouched read before selection lock: {text}")
            if text in self.canonical:
                self.counts["canonical_before_lock"] += 1
                raise RuntimeError(f"BLOCKED: canonical read before selection lock: {text}")
        return path


GUARD = AccessGuard()


def imread(path: Any) -> np.ndarray:
    return cv2.imread(str(GUARD.check(path)))


# ============================================================================
# lazy modules
# ============================================================================
_M: dict[str, Any] = {}


def modules() -> dict[str, Any]:
    if _M:
        return _M
    spec = importlib.util.spec_from_file_location("E56", STAGE0 / "paper_s2_eval56.py")
    e56 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e56)
    import pdg_targets as TARGETS
    import pdg_stage1_model as PSM
    import spatial_hcrm as HCRM
    _M.update({"E56": e56, "MD": e56.MD, "FZ": e56.FZ, "TARGETS": TARGETS,
               "PSM": PSM, "HCRM": HCRM})
    return _M


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalise(images: np.ndarray) -> torch.Tensor:
    array = images.astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(array.transpose(0, 3, 1, 2))


# ============================================================================
# Phase D -- EVAL_NO_AUG
# ============================================================================
class NoAugSample:
    __slots__ = ("root", "stem", "group", "split", "image", "points", "belief",
                 "belief_mask", "grid", "size")


def load_no_aug(root: str, stem: str) -> Optional[NoAugSample]:
    """The exact source frame, canonically squashed, with no random transform.

    Repeated calls return identical arrays: there is no sampling anywhere in
    this path, which is why the holdouts can be called no-aug at all.
    """
    M = modules()
    json_path = TRAIN_DATA / root / f"{stem}.json"
    image_path = TRAIN_DATA / root / f"{stem}.png"
    if not json_path.is_file() or not image_path.is_file():
        return None
    payload = json.loads(json_path.read_text("utf-8"))
    obj = (payload.get("objects") or [{}])[0]
    camera = payload.get("camera_data", {})
    width = int(camera.get("width", 640))
    height = int(camera.get("height", 480))
    cuboid = obj.get("projected_cuboid") or []
    centroid = obj.get("projected_cuboid_centroid")
    points = np.full((9, 2), np.nan)
    valid = np.zeros(9, bool)
    for index, entry in enumerate(cuboid[:8]):
        if entry is not None:
            points[index] = [float(entry[0]), float(entry[1])]
            valid[index] = True
    if centroid is not None:
        points[8] = [float(centroid[0]), float(centroid[1])]
        valid[8] = True
    if not valid[:8].any():
        return None
    image = imread(image_path)
    if image is None:
        return None
    filled = np.nan_to_num(points, nan=-1e6)
    targets = M["TARGETS"].build_targets(filled, width, height, source_valid=valid)
    sample = NoAugSample()
    sample.root, sample.stem = root, stem
    sample.image = cv2.cvtColor(cv2.resize(image, (INPUT_SIZE, INPUT_SIZE)),
                                cv2.COLOR_BGR2RGB)
    sample.points = points
    sample.belief = targets["belief"]
    sample.belief_mask = targets["belief_mask"]
    sample.grid = targets["grid"]
    sample.size = (width, height)
    return sample


def split_rows(which: Optional[str] = None) -> list[dict[str, str]]:
    table = pd.read_csv(OUT / "synthetic_split_manifest.csv")
    if which:
        table = table[table["split"] == which]
    return table.to_dict("records")


def split_paths(which: str) -> set[str]:
    return {str(TRAIN_DATA / row["root"] / f"{row['stem']}.png")
            for row in split_rows(which)}


# ============================================================================
# model wrapper
# ============================================================================
class FrozenA1(torch.nn.Module):
    """A1 with everything frozen and the F50 tap exposed."""

    def __init__(self) -> None:
        super().__init__()
        M = modules()
        self.model = M["PSM"].PDGStage1Model("A1")
        self.model.load_state_dict(
            torch.load(str(A1_CKPT), map_location="cpu", weights_only=True), strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        # One forward, not two.  Running net.vgg() separately and then net()
        # recomputes the trunk, and the second pass came back different in the
        # last bits: the corner counts stayed exact but the reprojection median
        # moved 0.03px, which the 1e-4 parity gate caught.  A hook takes the
        # tap out of the single pass instead.
        self._feature: Optional[torch.Tensor] = None
        self.model.net.vgg.register_forward_hook(
            lambda module, inputs, output: setattr(self, "_feature", output))

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        outputs = self.model.net(images)
        feature = self._feature
        belief = outputs[0][-1][:, :9].detach().float()
        affinity = outputs[1][-1].detach().float()
        assert feature is not None and feature.shape[-2:] == (BELIEF, BELIEF)
        return feature.detach(), belief, affinity

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def trainable(self) -> int:
        return sum(1 for p in self.model.parameters() if p.requires_grad)


def build_adapter(arm: str, seed: int):
    M = modules()
    return M["HCRM"].build("H1" if arm == "H1" else "H2", seed).to(device)


def compose(base: torch.Tensor, residual: torch.Tensor,
            permutation: Optional[tuple[int, ...]] = None) -> torch.Tensor:
    M = modules()
    return M["HCRM"].compose(base, residual, permutation)


# ============================================================================
# decoding and metrics
# ============================================================================
def decode(belief: np.ndarray, width: float, height: float, gt_points) -> list:
    M = modules()
    return M["MD"].decode_all(belief, width / BELIEF, height / BELIEF, gt_points)["D0"]


def corner_stats(points, gt, size) -> list[dict[str, Any]]:
    rows = []
    for channel in range(8):
        truth = gt[channel]
        if truth is None or (isinstance(truth, float) and np.isnan(truth)):
            continue
        predicted = points[channel]
        error = (None if predicted is None else
                 float(np.hypot(predicted[0] - truth[0], predicted[1] - truth[1])))
        rows.append({"channel": channel, "near": channel < 4,
                     "detected": predicted is not None, "err": error})
    return rows


def gt_list(sample: NoAugSample) -> list:
    out = []
    for index in range(9):
        point = sample.points[index]
        out.append(None if not np.isfinite(point).all() else [float(point[0]), float(point[1])])
    return out


def aggregate(rows: list[dict[str, Any]], frames: int) -> dict[str, Any]:
    if not rows:
        return {"n_frames": frames}
    frame_ = pd.DataFrame(rows)
    near = frame_[frame_.near]
    far = frame_[~frame_.near]
    id12 = frame_[frame_.channel.isin(ID12)]

    def block(sub, prefix):
        if not len(sub):
            return {}
        detected = sub[sub.detected]
        errors = detected["err"].dropna()
        return {
            f"{prefix}_recall": float(sub.detected.mean()),
            f"{prefix}_le10": float((errors <= 10).sum() / max(len(sub), 1)),
            f"{prefix}_le20": float((errors <= OK_PX).sum() / max(len(sub), 1)),
            f"{prefix}_gt50": float((errors > 50).sum() / max(len(sub), 1)),
            f"{prefix}_gt100": float((errors > 100).sum() / max(len(sub), 1)),
            f"{prefix}_median": float(errors.median()) if len(errors) else None,
            f"{prefix}_n": int(len(sub)),
        }

    out = {"n_frames": frames}
    out.update(block(near, "near"))
    out.update(block(far, "far"))
    out.update(block(id12, "id12"))
    for channel in range(4):
        sub = frame_[frame_.channel == channel]
        out[f"id{channel}_recall"] = float(sub.detected.mean()) if len(sub) else None
    return out


# ============================================================================
# phases
# ============================================================================
class State:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.data = json.loads(path.read_text()) if path.is_file() else {"phases": {}}
        self.data.setdefault("phases", {})

    def get(self, phase: str) -> str:
        return self.data["phases"].get(phase, {}).get("status", "PENDING")

    def set(self, phase: str, status: str, **extra) -> None:
        entry = self.data["phases"].setdefault(phase, {})
        entry.update({"status": status, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                 time.gmtime())})
        entry.update(extra)
        atomic_write(self.path, json.dumps(self.data, indent=1))


def phase_prepare(state) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((OUT / "input_lock_v2.json").read_text("utf-8"))
    if sha256_file(A1_CKPT) != A1_SHA:
        raise RuntimeError("HARD_BLOCKED: A1 checkpoint changed")
    manifest_sha = sha256_file(OUT / "synthetic_split_manifest.csv")
    if not lock["split"]["split_sha256"].startswith(manifest_sha[:16]):
        raise RuntimeError("HARD_BLOCKED: split manifest changed since the lock")
    GUARD.arm(split_paths("untouched"), canonical_image_paths())
    payload = {"head": git("rev-parse", "HEAD"), "a1_sha256": A1_SHA,
               "split_sha256": manifest_sha,
               "holdout_name": "KEYED_SOURCE_GROUP_HOLDOUT",
               "unkeyed_train_only_frames": lock["split"]["unkeyed_frames"],
               "guard_armed": {"untouched": len(GUARD.untouched),
                               "canonical": len(GUARD.canonical)}}
    atomic_write(OUT / "resume_lock.json", json.dumps(payload, indent=1))
    log(f"[prepare] split sha {manifest_sha[:16]}  guard armed "
        f"untouched={len(GUARD.untouched)} canonical={len(GUARD.canonical)}")
    return payload


def canonical_image_paths() -> set[str]:
    M = modules()
    paths = set()
    for label in ("eval56", "wood"):
        manifest = json.loads((M["E56"].OUT / f"{label}_manifest.json").read_text("utf-8"))
        paths.update(entry["image_path"] for entry in manifest["frames"])
    return paths


def phase_parity(state) -> dict[str, Any]:
    """A1 canonical parity, and the composed model's step-0 identity to it."""
    M = modules()
    GUARD.unlock()                      # parity is the one sanctioned canonical read
    a1 = FrozenA1().to(device)
    results = {}
    for label, reference in A1_PARITY.items():
        manifest = json.loads((M["E56"].OUT / f"{label}_manifest.json").read_text("utf-8"))
        centroid = r4 = pnp = 0
        reproj = []
        for entry in manifest["frames"]:
            frame = M["E56"].EvalFrame(entry)
            image = cv2.imread(entry["image_path"])
            height, width = image.shape[:2]
            batch = normalise(cv2.cvtColor(cv2.resize(image, (INPUT_SIZE, INPUT_SIZE)),
                                           cv2.COLOR_BGR2RGB)[None]).to(device)
            _, belief, _ = a1(batch)
            points = decode(belief[0].cpu().numpy(), width, height, frame.gt_points)
            centroid += points[8] is not None
            r4 += sum(1 for k in range(8) if points[k] is not None) >= 4
            pose = frame.solve(points)
            pnp += pose is not None
            value = frame.metrics(pose)["reproj_fixed_gt_px"]
            if value is not None:
                reproj.append(value)
        median = float(np.median(reproj))
        ok = ((centroid, r4, pnp) == reference[:3]
              and abs(median - reference[3]) <= 1e-4)
        results[label] = {"centroid": centroid, "R4": r4, "PnP": pnp,
                          "reproj": median, "reference": list(reference), "parity": ok}
        log(f"[parity] {label}: centroid {centroid} R4 {r4} PnP {pnp} "
            f"reproj {median:.4f}  parity={ok}")
    GUARD.unlocked = False
    if not all(v["parity"] for v in results.values()):
        state.set("parity", "HARD_BLOCKED", reason="HARD_BLOCKED_A1_PARITY")
        raise RuntimeError("HARD_BLOCKED_A1_PARITY")
    atomic_write(OUT / "a1_parity.json", json.dumps(results, indent=1))
    return results


def phase_eval_no_aug(state) -> dict[str, Any]:
    rows = split_rows("validation")[:8]
    report = {"samples": [], "repeatable": True}
    for row in rows:
        first = load_no_aug(row["root"], row["stem"])
        second = load_no_aug(row["root"], row["stem"])
        same = (np.array_equal(first.image, second.image)
                and np.array_equal(np.nan_to_num(first.points), np.nan_to_num(second.points))
                and np.array_equal(first.belief, second.belief)
                and np.array_equal(first.belief_mask, second.belief_mask))
        report["samples"].append({"root": row["root"], "stem": row["stem"],
                                  "identical_on_reload": bool(same)})
        report["repeatable"] &= bool(same)
    report["mode"] = "EVAL_NO_AUG"
    report["holdout_name"] = "KEYED_SOURCE_GROUP_HOLDOUT"
    atomic_write(OUT / "eval_no_aug_parity.json", json.dumps(report, indent=1))
    log(f"[eval-no-aug] repeatable on {len(rows)} samples: {report['repeatable']}")
    if not report["repeatable"]:
        raise RuntimeError("HARD_BLOCKED: EVAL_NO_AUG is not deterministic")
    return report


def phase_hard_manifest(state) -> dict[str, Any]:
    """Hard/easy labels from a frozen A1 forward over the train split only."""
    a1 = FrozenA1().to(device)
    rows = []
    train = split_rows("train")
    log(f"[hard] frozen A1 over {len(train)} train frames (train split only)")
    for index, row in enumerate(train):
        sample = load_no_aug(row["root"], row["stem"])
        if sample is None:
            continue
        batch = normalise(sample.image[None]).to(device)
        _, belief, _ = a1(batch)
        belief = belief[0].cpu().numpy()
        width, height = sample.size
        points = decode(belief, width, height, gt_list(sample))
        for channel in NEAR:
            if sample.belief_mask[channel] <= 0:
                continue
            peak = float(belief[channel].max())
            predicted = points[channel]
            truth = sample.points[channel]
            error = (None if predicted is None else
                     float(np.hypot(predicted[0] - truth[0], predicted[1] - truth[1])))
            hard = (peak <= THRESHOLD or predicted is None
                    or error is None or error > OK_PX)
            rows.append({"root": row["root"], "stem": row["stem"],
                         "group": row.get("group") or "UNKEYED_TRAIN_ONLY",
                         "channel": channel, "peak": peak,
                         "err": error, "hard": bool(hard)})
        if (index + 1) % 2000 == 0:
            log(f"[hard] {index+1}/{len(train)}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "hard_manifest.csv", index=False)
    weights = {}
    for channel in NEAR:
        sub = table[table.channel == channel]
        n_hard = int(sub.hard.sum())
        n_easy = int(len(sub) - n_hard)
        weights[str(channel)] = {
            "n_easy": n_easy, "n_hard": n_hard,
            "hard_weight": float(min(HARD_WEIGHT_CAP, n_easy / max(n_hard, 1))),
            "easy_weight": 1.0}
    summary = {"rows": len(table), "weights": weights,
               "source": "train split only, EVAL_NO_AUG, frozen A1",
               "excluded": "validation, untouched, eval56, wood"}
    atomic_write(OUT / "hard_manifest_summary.json", json.dumps(summary, indent=1))
    atomic_write(OUT / "hard_manifest_sha.json", json.dumps(
        {"sha256": sha256_file(OUT / "hard_manifest.csv")}, indent=1))
    log(f"[hard] {len(table)} near-corner rows; weights "
        + json.dumps({k: round(v['hard_weight'], 3) for k, v in weights.items()}))
    return summary


# ============================================================================
# training
# ============================================================================
# The adapter trains on unaugmented source frames.  Declared, not hidden: A1's
# own loader augments, so this is a departure, and its cost is that the adapter
# never sees an augmented frame and its robustness is untested.  The benefit is
# that H1, H2 and H3 see byte-identical inputs in an identical order, which is
# what makes the arm comparison a single-variable one.  This is a screen for
# whether the spatial signal converts, not a deployment recipe.
TRAIN_INPUT_MODE = "NO_AUG_SOURCE"


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, rows, weights) -> None:
        self.rows = rows
        self.weights = weights

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        sample = load_no_aug(row["root"], row["stem"])
        if sample is None:
            sample = load_no_aug(self.rows[0]["root"], self.rows[0]["stem"])
        weight = np.ones(len(NEAR), np.float32)
        key = (row["root"], row["stem"])
        for position, channel in enumerate(NEAR):
            weight[position] = self.weights.get((key, channel), 1.0)
        return {"image": normalise(sample.image[None])[0],
                "belief": torch.from_numpy(sample.belief[:4]),
                "mask": torch.from_numpy(sample.belief_mask[:4]),
                "weight": torch.from_numpy(weight)}


def hard_weight_table(arm: str) -> dict:
    if arm != "H3":
        return {}
    table = pd.read_csv(OUT / "hard_manifest.csv")
    summary = json.loads((OUT / "hard_manifest_summary.json").read_text("utf-8"))
    weights = {}
    for row in table.itertuples():
        value = (summary["weights"][str(row.channel)]["hard_weight"]
                 if row.hard else 1.0)
        weights[((row.root, row.stem), int(row.channel))] = float(value)
    return weights


def near_loss(composed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A1's masked belief loss restricted to the four near channels."""
    scale = mask[:, :, None, None]
    if weight is not None:
        scale = scale * weight[:, :, None, None]
    return (((composed[:, :4] - target) ** 2) * scale).mean()


def run_key(arm: str, seed: int) -> str:
    return f"{arm}_seed{seed}"


def run_dir(arm: str, seed: int) -> pathlib.Path:
    return WEIGHTS / arm / f"seed{seed}"


def train_run(arm: str, seed: int, smoke: bool = False) -> dict[str, Any]:
    directory = run_dir(arm, seed)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "run_state.json"
    if not smoke and state_path.is_file():
        recorded = json.loads(state_path.read_text("utf-8"))
        if recorded.get("completed"):
            log(f"[train] {run_key(arm, seed)} already completed -- skipping")
            return recorded

    a1 = FrozenA1().to(device)
    before = a1.checksum()
    if a1.trainable() != 0:
        raise RuntimeError("BLOCKED: A1 has trainable parameters")
    seed_all(seed)
    adapter = build_adapter(arm, seed)
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)

    rows = split_rows("train")
    dataset = TrainSet(rows, hard_weight_table(arm))
    start_epoch, history = 0, []
    if not smoke and state_path.is_file() and (directory / "last.pth").is_file():
        try:
            blob = torch.load(directory / "last.pth", map_location=device,
                              weights_only=False)
            adapter.load_state_dict(blob["adapter"])
            optimiser.load_state_dict(torch.load(directory / "optimizer.pth",
                                                  map_location=device))
            recorded = json.loads(state_path.read_text("utf-8"))
            start_epoch = int(recorded["epoch"])
            history = json.loads((directory / "metrics.json").read_text("utf-8"))
            log(f"[train] {run_key(arm, seed)} resumed at epoch {start_epoch}")
        except Exception as error:
            log(f"[train] {run_key(arm, seed)} corrupt checkpoint rejected: {error}")
            start_epoch, history = 0, []

    epochs = 1 if smoke else EPOCHS
    gradient_trace = []
    for epoch in range(start_epoch, epochs):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH, shuffle=True, num_workers=6,
            generator=generator, drop_last=True, persistent_workers=False)
        adapter.train()
        losses = []
        began = time.time()
        for step, batch in enumerate(loader):
            if smoke and step >= SMOKE_STEPS:
                break
            images = batch["image"].to(device)
            feature, base, _ = a1(images)
            residual = adapter(feature)
            composed = compose(base, residual)
            loss = near_loss(composed, batch["belief"].to(device),
                             batch["mask"].to(device),
                             batch["weight"].to(device) if arm == "H3" else None)
            if not torch.isfinite(loss):
                raise RuntimeError(f"BLOCKED: non-finite loss in {run_key(arm, seed)}")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if smoke and step < 2:
                gradient_trace.append({
                    "step": step + 1,
                    "output_grad": float(adapter.out.weight.grad.abs().sum()),
                    "trunk_grad": float(sum(
                        p.grad.abs().sum() for n, p in adapter.named_parameters()
                        if p.grad is not None and not n.startswith("out."))),
                })
            optimiser.step()
            losses.append(float(loss))
        if smoke:
            with torch.no_grad():
                images = normalise(load_no_aug(rows[0]["root"], rows[0]["stem"]).image[None]).to(device)
                feature, base, affinity = a1(images)
                composed = compose(base, adapter(feature))
                M = modules()
                untouched = M["HCRM"].assert_untouched(composed, base)
            return {"arm": arm, "seed": seed, "steps": len(losses),
                    "mean_loss": float(np.mean(losses)),
                    "finite": bool(np.isfinite(np.mean(losses))),
                    "gradient_trace": gradient_trace,
                    "residual_max_abs": float((composed[:, :4] - base[:, :4]).abs().max()),
                    "untouched": untouched,
                    "a1_unchanged": a1.checksum() == before,
                    "trainable_parameters": int(sum(p.numel() for p in parameters))}
        entry = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                 "steps": len(losses), "seconds": round(time.time() - began, 1)}
        history.append(entry)
        atomic_torch_save({"adapter": adapter.state_dict()}, directory / "last.pth")
        atomic_torch_save({"adapter": adapter.state_dict()},
                          directory / f"epoch_{epoch+1:03d}.pth")
        atomic_torch_save(optimiser.state_dict(), directory / "optimizer.pth")
        atomic_write(directory / "metrics.json", json.dumps(history, indent=1))
        atomic_write(state_path, json.dumps(
            {"arm": arm, "seed": seed, "epoch": epoch + 1,
             "completed": epoch + 1 >= EPOCHS, "a1_sha256": A1_SHA,
             "a1_checksum_unchanged": a1.checksum() == before,
             "train_input_mode": TRAIN_INPUT_MODE,
             "timestamp": time.time()}, indent=1))
        log(f"[train] {run_key(arm, seed)} epoch {epoch+1}/{EPOCHS} "
            f"loss {entry['loss']:.6f} ({entry['seconds']}s)")
    if a1.checksum() != before:
        raise RuntimeError("BLOCKED: A1 parameters changed during training")
    return json.loads(state_path.read_text("utf-8"))


def phase_smoke(state) -> dict[str, Any]:
    results = []
    for arm in ARMS:
        result = train_run(arm, 1, smoke=True)
        results.append(result)
        log(f"[smoke] {arm}: {result['steps']} steps loss {result['mean_loss']:.6f} "
            f"residual {result['residual_max_abs']:.3e} "
            f"grad {result['gradient_trace']} A1 unchanged {result['a1_unchanged']}")
    payload = {"arms": results, "guard_counts": GUARD.counts,
               "passed": all(r["finite"] and r["a1_unchanged"]
                             and r["untouched"]["far_max_abs"] == 0.0
                             and r["untouched"]["centroid_max_abs"] == 0.0
                             and r["gradient_trace"][0]["output_grad"] > 0
                             for r in results)}
    atomic_write(OUT / "smoke.json", json.dumps(payload, indent=1))
    if not payload["passed"]:
        raise RuntimeError("BLOCKED: smoke failed")
    return payload


def phase_train(state) -> dict[str, Any]:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(train_run(arm, seed))
    pd.DataFrame(rows).to_csv(OUT / "training_runs.csv", index=False)
    return {"runs": len(rows)}


# ============================================================================
# evaluation
# ============================================================================
def evaluate_synthetic(adapter, a1, rows, permutation=None, zero=False) -> dict:
    stats, frames = [], 0
    for row in rows:
        sample = load_no_aug(row["root"], row["stem"])
        if sample is None:
            continue
        frames += 1
        images = normalise(sample.image[None]).to(device)
        with torch.no_grad():
            feature, base, _ = a1(images)
            if adapter is None:
                composed = base
            else:
                residual = adapter(feature)
                if zero:
                    residual = torch.zeros_like(residual)
                composed = compose(base, residual, permutation)
        width, height = sample.size
        points = decode(composed[0].cpu().numpy(), width, height, gt_list(sample))
        stats.extend(corner_stats(points, gt_list(sample), sample.size))
    return aggregate(stats, frames)


def evaluate_canonical(adapter, a1, label, permutation=None, zero=False) -> dict:
    M = modules()
    manifest = json.loads((M["E56"].OUT / f"{label}_manifest.json").read_text("utf-8"))
    stats = []
    centroid = r4 = r6 = pnp = 0
    reproj = []
    for entry in manifest["frames"]:
        frame = M["E56"].EvalFrame(entry)
        image = imread(entry["image_path"])
        height, width = image.shape[:2]
        images = normalise(cv2.cvtColor(cv2.resize(image, (INPUT_SIZE, INPUT_SIZE)),
                                        cv2.COLOR_BGR2RGB)[None]).to(device)
        with torch.no_grad():
            feature, base, _ = a1(images)
            if adapter is None:
                composed = base
            else:
                residual = adapter(feature)
                if zero:
                    residual = torch.zeros_like(residual)
                composed = compose(base, residual, permutation)
        points = decode(composed[0].cpu().numpy(), width, height, frame.gt_points)
        stats.extend(corner_stats(points, frame.gt_points, (width, height)))
        centroid += points[8] is not None
        good = sum(1 for k in range(8) if points[k] is not None)
        r4 += good >= 4
        r6 += good >= 6
        pose = frame.solve(points)
        pnp += pose is not None
        value = frame.metrics(pose)["reproj_fixed_gt_px"]
        if value is not None:
            reproj.append(value)
    out = aggregate(stats, len(manifest["frames"]))
    out.update({"set": label, "centroid": centroid, "R4": r4, "R6": r6, "PnP": pnp,
                "reproj_median": float(np.median(reproj)) if reproj else None})
    return out


def phase_select(state) -> dict[str, Any]:
    """Validation only.  Untouched and canonical stay closed until this locks."""
    a1 = FrozenA1().to(device)
    rows = split_rows("validation")
    selected = {}
    for arm in ARMS:
        for seed in SEEDS:
            directory = run_dir(arm, seed)
            best = None
            for epoch in range(1, EPOCHS + 1):
                path = directory / f"epoch_{epoch:03d}.pth"
                if not path.is_file():
                    continue
                adapter = build_adapter(arm, seed)
                adapter.load_state_dict(torch.load(path, map_location=device,
                                                   weights_only=False)["adapter"])
                adapter.eval()
                metrics = evaluate_synthetic(adapter, a1, rows)
                key = (-metrics.get("id12_recall", 0.0),
                       -metrics.get("near_le20", 0.0),
                       metrics.get("near_gt50", 1.0), epoch)
                if best is None or key < best[0]:
                    best = (key, epoch, metrics, path)
            if best is None:
                continue
            selected[run_key(arm, seed)] = {
                "arm": arm, "seed": seed, "epoch": best[1],
                "checkpoint": str(best[3].relative_to(ROOT)),
                "sha256": sha256_file(best[3]),
                "validation": best[2],
                "selection_set": "KEYED_SOURCE_GROUP_VALIDATION"}
            log(f"[select] {run_key(arm, seed)} epoch {best[1]} "
                f"id12 {100*best[2].get('id12_recall',0):.1f}% "
                f"near<=20 {100*best[2].get('near_le20',0):.1f}%")
    atomic_write(OUT / "selected_checkpoints.json", json.dumps(selected, indent=1))
    atomic_write(OUT / "selected_checkpoints_sha.json", json.dumps(
        {"sha256": sha256_file(OUT / "selected_checkpoints.json"),
         "untouched_open_before_lock": GUARD.counts["untouched_before_lock"],
         "canonical_open_before_lock": GUARD.counts["canonical_before_lock"],
         "final_test_open": GUARD.counts["final_test"]}, indent=1))
    GUARD.unlock()
    log(f"[select] locked {len(selected)} checkpoints; holdout access unlocked "
        f"(before-lock opens: {GUARD.counts})")
    return selected


def arm_states(a1):
    """H0, every selected run, plus the ZERO and SHUFFLE controls."""
    selected = json.loads((OUT / "selected_checkpoints.json").read_text("utf-8"))
    yield "H0", None, None, False
    for key, entry in selected.items():
        adapter = build_adapter(entry["arm"], entry["seed"])
        adapter.load_state_dict(torch.load(ROOT / entry["checkpoint"],
                                           map_location=device,
                                           weights_only=False)["adapter"])
        adapter.eval()
        yield key, adapter, None, False
        if entry["arm"] == "H2" and entry["seed"] == 1:
            yield "H2-ZERO", adapter, None, True
            yield "H2-SHUFFLE", adapter, SHUFFLE_PERMUTATION, False


def phase_eval_untouched(state) -> dict[str, Any]:
    if not (OUT / "selected_checkpoints.json").is_file():
        raise RuntimeError("BLOCKED: untouched before the selection lock")
    GUARD.unlock()
    a1 = FrozenA1().to(device)
    rows = split_rows("untouched")
    results = {}
    for name, adapter, permutation, zero in arm_states(a1):
        results[name] = evaluate_synthetic(adapter, a1, rows, permutation, zero)
        log(f"[untouched] {name}: id12 {100*results[name].get('id12_recall',0):.1f}% "
            f"near<=20 {100*results[name].get('near_le20',0):.1f}% "
            f"far>50 {100*results[name].get('far_gt50',0):.1f}%")
    atomic_write(OUT / "untouched_metrics.json", json.dumps(results, indent=1))
    pd.DataFrame(results).T.to_csv(OUT / "synthetic_corner_metrics.csv")
    return results


def phase_eval_canonical(state) -> dict[str, Any]:
    if not (OUT / "selected_checkpoints.json").is_file():
        raise RuntimeError("BLOCKED: canonical before the selection lock")
    GUARD.unlock()
    a1 = FrozenA1().to(device)
    results = {}
    for name, adapter, permutation, zero in arm_states(a1):
        results[name] = {}
        for label in ("eval56", "wood"):
            results[name][label] = evaluate_canonical(adapter, a1, label,
                                                      permutation, zero)
        e = results[name]["eval56"]
        w = results[name]["wood"]
        log(f"[canonical] {name}: eval56 id12 {100*e.get('id12_recall',0):.1f}% "
            f"R4 {e['R4']} PnP {e['PnP']} | wood id12 "
            f"{100*w.get('id12_recall',0):.1f}% R4 {w['R4']} PnP {w['PnP']}")
    atomic_write(OUT / "canonical_metrics.json", json.dumps(results, indent=1))
    rows = [{"arm": n, **{f"{s}_{k}": v for s, b in r.items()
                          for k, v in b.items() if not isinstance(v, (dict, list))}}
            for n, r in results.items()]
    pd.DataFrame(rows).to_csv(OUT / "canonical_corner_metrics.csv", index=False)
    return results


# ============================================================================
# gates and decision
# ============================================================================
def phase_decide(state) -> dict[str, Any]:
    untouched = json.loads((OUT / "untouched_metrics.json").read_text("utf-8"))
    canonical = json.loads((OUT / "canonical_metrics.json").read_text("utf-8"))
    base = untouched["H0"]

    def pick(prefix):
        keys = [k for k in untouched if k.startswith(prefix + "_seed")]
        return max(keys, key=lambda k: untouched[k].get("id12_recall", 0.0)) if keys else None

    best = {arm: pick(arm) for arm in ARMS}
    spatial = max([k for k in (best["H2"], best["H3"]) if k],
                  key=lambda k: untouched[k].get("id12_recall", 0.0), default=None)
    if spatial is None:
        payload = {"decision": "BLOCKED", "reason": "no spatial arm completed"}
        atomic_write(OUT / "final_decision.json", json.dumps(payload, indent=1))
        return payload
    s = untouched[spatial]

    def delta(metric, arm=spatial, reference=None):
        reference = reference or base
        return s.get(metric, 0.0) - reference.get(metric, 0.0) if arm == spatial \
            else untouched[arm].get(metric, 0.0) - reference.get(metric, 0.0)

    directions = sum(1 for seed in SEEDS
                     if f"{spatial.split('_')[0]}_seed{seed}" in untouched
                     and untouched[f"{spatial.split('_')[0]}_seed{seed}"].get("id12_recall", 0)
                     > base.get("id12_recall", 0))
    zero_parity = (abs(untouched.get("H2-ZERO", {}).get("id12_recall", -1)
                       - base.get("id12_recall", 0)) < 1e-12)
    synthetic = {
        "1_id12+8pp": delta("id12_recall") >= 0.08,
        "2_near_le20+5pp": delta("near_le20") >= 0.05,
        "3_near_gt50<=+5pp": delta("near_gt50") <= 0.05,
        "4_far_gt50<=+5pp": delta("far_gt50") <= 0.05,
        "5_two_of_three_seeds": directions >= 2,
        "6_H2ZERO_parity": zero_parity,
    }
    pointwise = untouched.get(best["H1"] or "", {})
    module_value = {
        "id12_or_le20_+3pp": (s.get("id12_recall", 0) - pointwise.get("id12_recall", 0) >= 0.03
                              or s.get("near_le20", 0) - pointwise.get("near_le20", 0) >= 0.03),
        "better_than_shuffle": (s.get("id12_recall", 0)
                                > untouched.get("H2-SHUFFLE", {}).get("id12_recall", 1.0)),
    }
    ce = canonical[spatial]["eval56"]
    cw = canonical[spatial]["wood"]
    be = canonical["H0"]["eval56"]
    bw = canonical["H0"]["wood"]
    near_gain = {
        "eval56_id12+8pp": ce.get("id12_recall", 0) - be.get("id12_recall", 0) >= 0.08,
        "eval56_R4+2": ce["R4"] - be["R4"] >= 2,
        "wood_id12_no_drop": cw.get("id12_recall", 0) >= bw.get("id12_recall", 0),
        "wood_R4_drop<=1": bw["R4"] - cw["R4"] <= 1,
    }
    pose_safety = {
        "eval56_PnP_no_drop": ce["PnP"] >= be["PnP"],
        "wood_PnP_no_drop": cw["PnP"] >= bw["PnP"],
        "reproj_degradation<=5%": all(
            (c.get("reproj_median") or 0) <= 1.05 * (b.get("reproj_median") or 1e9)
            for c, b in ((ce, be), (cw, bw))),
    }
    far_safety = {
        "far_gt50<=+10%": all(c.get("far_gt50", 0) <= b.get("far_gt50", 0) + 0.10
                              for c, b in ((ce, be), (cw, bw))),
        "far_recall_drop<=5pp": all(c.get("far_recall", 0) >= b.get("far_recall", 0) - 0.05
                                    for c, b in ((ce, be), (cw, bw))),
    }
    recalls = [untouched[f"{spatial.split('_')[0]}_seed{x}"].get("id12_recall", 0)
               for x in SEEDS if f"{spatial.split('_')[0]}_seed{x}" in untouched]
    stability = {"two_of_three_seeds": directions >= 2,
                 "id12_range<=10pp": (max(recalls) - min(recalls)) <= 0.10 if recalls else False}

    gates = {"SYNTHETIC_HCRM_SIGNAL": synthetic, "SPATIAL_MODULE_VALUE": module_value,
             "CANONICAL_NEAR_GAIN": near_gain, "POSE_SAFETY": pose_safety,
             "FAR_SAFETY": far_safety, "STABILITY": stability}
    passed = {k: all(v.values()) for k, v in gates.items()}
    h1_best = untouched.get(best["H1"] or "", {})
    if all(passed.values()):
        decision = "SPATIAL_HCRM_GO"
    elif not passed["POSE_SAFETY"] or not passed["FAR_SAFETY"]:
        decision = "HCRM_HEALTHY_FRAME_REGRESSION" if passed["SYNTHETIC_HCRM_SIGNAL"] \
            else "SPATIAL_SIGNAL_NOT_CONVERTED"
    elif not passed["STABILITY"]:
        decision = "HCRM_UNSTABLE"
    elif not passed["SYNTHETIC_HCRM_SIGNAL"]:
        improved = (h1_best.get("id12_recall", 0) > base.get("id12_recall", 0)
                    and s.get("id12_recall", 0) > base.get("id12_recall", 0))
        decision = "POINTWISE_ADAPTER_SUFFICIENT" if (
            improved and not passed["SPATIAL_MODULE_VALUE"]) else (
            "SPATIAL_SIGNAL_NOT_CONVERTED" if not improved else "A1_ONLY")
    elif not passed["CANONICAL_NEAR_GAIN"]:
        decision = "SYNTHETIC_ONLY_HCRM_GAIN"
    else:
        decision = "SPATIAL_HCRM_PARTIAL_GO"
    payload = {"decision": decision, "spatial_arm": spatial, "pointwise_arm": best["H1"],
               "gates": gates, "gates_passed": passed,
               "module_status": {
                   "A1_BALCH": "KEEP",
                   "Spatial_HCRM": {"SPATIAL_HCRM_GO": "GO",
                                    "SPATIAL_HCRM_PARTIAL_GO": "PARTIAL"}.get(decision, "STOP"),
                   "dense_line": "STOP", "twelve_edge": "VALID_BUT_DEFERRED",
                   "CIGM": "VALID_BUT_BLOCKED", "fusion": "STOP", "RCIM": "DEFERRED"},
               "holdout_caveat": ("validation and untouched are KEYED_SOURCE_GROUP_HOLDOUT; "
                                  "13,069 unkeyed frames are train-only, so they are not an "
                                  "unbiased sample of the synthetic distribution"),
               "train_input_mode": TRAIN_INPUT_MODE}
    atomic_write(OUT / "final_decision.json", json.dumps(payload, indent=1, default=str))
    log(f"[decide] {decision}  spatial {spatial}  gates "
        + json.dumps({k: bool(v) for k, v in passed.items()}))
    return payload


def phase_report(state) -> dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("input_lock_v2.json", "split_summary.json", "a1_parity.json",
                 "eval_no_aug_parity.json", "hard_manifest_summary.json",
                 "smoke.json", "selected_checkpoints.json",
                 "selected_checkpoints_sha.json", "untouched_metrics.json",
                 "canonical_metrics.json", "final_decision.json",
                 "synthetic_corner_metrics.csv", "canonical_corner_metrics.csv",
                 "training_runs.csv"):
        source = OUT / name
        if source.is_file():
            atomic_write(REPORTS / name, source.read_text("utf-8"))
            written.append(name)
    decision = json.loads((OUT / "final_decision.json").read_text("utf-8")) \
        if (OUT / "final_decision.json").is_file() else {"decision": "INCOMPLETE"}
    atomic_write(REPORTS / "SPATIAL_HCRM_FINAL_DECISION.md",
                 f"# Spatial HCRM screen: {decision['decision']}\n\n"
                 "Holdouts are whole source groups from the keyed roots; the 13,069\n"
                 "frames with no scene metadata are train-only, so validation and\n"
                 "untouched are a KEYED_SOURCE_GROUP_HOLDOUT and not an unbiased sample\n"
                 "of the synthetic distribution.  The adapter trained on unaugmented\n"
                 "source frames, which keeps the arm comparison single-variable and\n"
                 "leaves its robustness untested.\n\n```\n"
                 + json.dumps(decision, indent=1, default=str) + "\n```\n")
    written.append("SPATIAL_HCRM_FINAL_DECISION.md")
    return {"written": written}


def phase_test(state) -> dict[str, Any]:
    new = subprocess.run([sys.executable, "-m", "pytest", "-q",
                          "challenge/tests/test_spatial_hcrm_screen.py"],
                         cwd=ROOT, capture_output=True, text=True)
    full = subprocess.run([sys.executable, "-m", "pytest", "-q", "challenge/tests"],
                          cwd=ROOT, capture_output=True, text=True)
    payload = {"new_tail": new.stdout.strip().splitlines()[-1:],
               "full_tail": full.stdout.strip().splitlines()[-1:],
               "new_returncode": new.returncode, "full_returncode": full.returncode}
    atomic_write(OUT / "tests.json", json.dumps(payload, indent=1))
    log(f"[test] {payload['new_tail']} {payload['full_tail']}")
    return payload


DRIVERS = {"prepare": phase_prepare, "parity": phase_parity,
           "eval-no-aug": phase_eval_no_aug, "hard-manifest": phase_hard_manifest,
           "smoke": phase_smoke, "train": phase_train, "select": phase_select,
           "eval-untouched": phase_eval_untouched,
           "eval-canonical": phase_eval_canonical, "decide": phase_decide,
           "report": phase_report, "test": phase_test}


def complete_marker(state) -> None:
    decision = json.loads((OUT / "final_decision.json").read_text("utf-8")) \
        if (OUT / "final_decision.json").is_file() else {"decision": "INCOMPLETE"}
    lock = json.loads((OUT / "selected_checkpoints_sha.json").read_text("utf-8")) \
        if (OUT / "selected_checkpoints_sha.json").is_file() else {}
    atomic_write(OUT / "COMPLETE", json.dumps({
        "final_decision": decision["decision"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "a1_sha256": sha256_file(A1_CKPT), "a1_unchanged": sha256_file(A1_CKPT) == A1_SHA,
        "main_model_training_steps": 0,
        "selection_lock": lock,
        "guard_counts": GUARD.counts,
        "train_input_mode": TRAIN_INPUT_MODE,
        "phases": {p: state.get(p) for p in PHASES}}, indent=1))
    log(f"[COMPLETE] {decision['decision']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(DRIVERS) + ["all", "resume", "status"])
    parser.add_argument("--from-phase", default=None)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = State(OUT / "state.json")
    if arguments.command == "status":
        for phase in PHASES:
            print(f"{phase:16s} {state.get(phase)}")
        return
    phases = list(PHASES)
    if arguments.command not in ("all", "resume"):
        phases = [arguments.command]
    elif arguments.from_phase:
        phases = phases[phases.index(arguments.from_phase):]
    for phase in phases:
        if state.get(phase) == "DONE" and not arguments.force:
            log(f"[{phase}] already DONE -- skipping")
            continue
        state.set(phase, "RUNNING")
        began = time.time()
        try:
            DRIVERS[phase](state)
        except Exception as error:
            state.set(phase, "FAILED", error=repr(error))
            raise
        state.set(phase, "DONE", seconds=round(time.time() - began, 1))
    if arguments.command in ("all", "resume"):
        complete_marker(state)


if __name__ == "__main__":
    main()
