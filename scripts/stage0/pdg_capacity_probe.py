"""Frozen-feature capacity probe.

Asks only what information is present in the frozen features, never whether a
model could learn to use it.  The main network is read-only; the only fitting is
a fixed logistic probe that is never saved.

Feature taps confirmed at runtime, not hardcoded:
    F100 = vgg[17] ReLU output, 256 x 100 x 100
    F50  = vgg[26], the shared DOPE feature, 128 x 50 x 50
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts/stage0", ROOT / "Deep_Object_Pose/common",
           ROOT / "Deep_Object_Pose/train", ROOT / "challenge/scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/capacity_audit")
F100_INDEX, F50_INDEX = 17, 26
RING_RADII = (20.0, 40.0, 60.0)
RING_ANGLES = tuple(np.deg2rad(a) for a in (0, 45, 90, 135, 180, 225, 270, 315))
SEED = 1


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "eval56", ROOT / "scripts/stage0/paper_s2/paper_s2_eval56.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_patch(feature: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """Mean and max over a size x size window, clamped at the border."""
    channels, height, width = feature.shape
    radius = size // 2
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(channels * 2, dtype=np.float32)
    window = feature[:, y0:y1, x0:x1]
    return np.concatenate([window.mean(axis=(1, 2)), window.max(axis=(1, 2))]
                          ).astype(np.float32)


def descriptors(f100: np.ndarray, f50: np.ndarray, gap: np.ndarray,
                x_img: float, y_img: float, width: int, height: int) -> dict:
    """C0..C3.  No absolute coordinate ever enters a descriptor."""
    c0 = sample_patch(f50, x_img * 50.0 / width, y_img * 50.0 / height, 3)
    c1 = sample_patch(f100, x_img * 100.0 / width, y_img * 100.0 / height, 5)
    c2 = np.concatenate([c0, c1])
    return {"C0": c0, "C1": c1, "C2": c2, "C3": np.concatenate([c2, gap])}


def topk_peaks(belief: np.ndarray, k: int = 5, radius: int = 3):
    work = np.asarray(belief, dtype=np.float64).copy()
    found = []
    for _ in range(k):
        index = int(work.argmax())
        y, x = np.unravel_index(index, work.shape)
        found.append((float(x), float(y), float(work[y, x])))
        y0, y1 = max(0, y - radius), min(work.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(work.shape[1], x + radius + 1)
        work[y0:y1, x0:x1] = -1e9
    return found


@torch.no_grad()
def extract(runner, arm_path=None):
    """One forward per canonical frame; sample descriptors at fixed locations."""
    from models import DopeNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DopeNetwork(numSeg=1)
    net.load_state_dict(torch.load(str(arm_path or (ROOT / "weights/paper_s2_stageB"
                                                    / "net_epoch_0057.pth")),
                                   map_location="cpu", weights_only=True), strict=True)
    net = net.to(device).eval()
    layers = list(net.vgg)
    rng = np.random.default_rng(SEED)
    rows = []
    for label in ("eval56", "wood"):
        manifest = json.loads(
            (runner.OUT / f"{label}_manifest.json").read_text("utf-8"))
        for spec in manifest["frames"]:
            frame = runner.EvalFrame(spec)
            width, height = spec["image_width"], spec["image_height"]
            tensor = runner.FZ.preprocess_squash(
                cv2.imread(spec["image_path"])).to(device)
            hidden = tensor
            f100 = None
            for index, layer in enumerate(layers):
                hidden = layer(hidden)
                if index == F100_INDEX:
                    f100 = hidden[0].float().cpu().numpy()
            f50 = hidden[0].float().cpu().numpy()
            belief = net(tensor)[0][5][0, :9].float().cpu().numpy()
            gap = f50.mean(axis=(1, 2))

            for corner in range(8):
                gt = frame.gt_points[corner]
                if gt is None or not (0 <= gt[0] < width and 0 <= gt[1] < height):
                    continue
                peaks = topk_peaks(belief[corner])
                sx, sy = width / 50.0, height / 50.0
                top1 = (peaks[0][0] * sx, peaks[0][1] * sy)
                top1_error = float(np.hypot(top1[0] - gt[0], top1[1] - gt[1]))
                base = {"set": label, "fid": str(spec["frame_id"]),
                        "session": spec.get("domain", label), "corner": corner,
                        "role": "near" if corner < 4 else "far",
                        "peak": float(belief[corner].max()),
                        "top1_error": top1_error}
                rows.append({**base, "kind": "positive", "label": 1,
                             **{k: v for k, v in descriptors(
                                 f100, f50, gap, gt[0], gt[1], width, height).items()}})
                if top1_error > 20.0:
                    rows.append({**base, "kind": "top1_wrong", "label": 0,
                                 **descriptors(f100, f50, gap, top1[0], top1[1],
                                               width, height)})
                for rank in range(1, len(peaks)):
                    px, py = peaks[rank][0] * sx, peaks[rank][1] * sy
                    if np.hypot(px - gt[0], py - gt[1]) <= 20.0:
                        continue
                    rows.append({**base, "kind": f"topk{rank}", "label": 0,
                                 **descriptors(f100, f50, gap, px, py, width, height)})
                for radius in RING_RADII:
                    for angle in RING_ANGLES:
                        px = gt[0] + radius * np.cos(angle)
                        py = gt[1] + radius * np.sin(angle)
                        if not (0 <= px < width and 0 <= py < height):
                            continue
                        rows.append({**base, "kind": "ring", "label": 0,
                                     **descriptors(f100, f50, gap, px, py,
                                                   width, height)})
                for other in range(8):
                    if other == corner:
                        continue
                    point = frame.gt_points[other]
                    if point is None or not (0 <= point[0] < width
                                             and 0 <= point[1] < height):
                        continue
                    rows.append({**base, "kind": "other_corner", "label": 0,
                                 **descriptors(f100, f50, gap, point[0], point[1],
                                               width, height)})
    del net
    torch.cuda.empty_cache()
    return rows


def fit_probe(features, labels, groups, folds: int = 5):
    """Fixed pipeline, grouped by frame.  Scaler is fit on the train fold only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique = len(set(groups))
    if unique < 2 or len(set(labels)) < 2:
        return {"auc": np.nan, "n": int(len(labels)), "folds": 0}
    splitter = GroupKFold(n_splits=min(folds, unique))
    scores, sizes = [], []
    for train_index, test_index in splitter.split(features, labels, groups):
        if len(set(labels[train_index])) < 2 or len(set(labels[test_index])) < 2:
            continue
        # leakage guard: a frame may never appear on both sides
        assert not (set(groups[train_index]) & set(groups[test_index]))
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, class_weight="balanced",
                                                 max_iter=2000, random_state=SEED))
        model.fit(features[train_index], labels[train_index])
        probability = model.predict_proba(features[test_index])[:, 1]
        scores.append(roc_auc_score(labels[test_index], probability))
        sizes.append(len(test_index))
    if not scores:
        return {"auc": np.nan, "n": int(len(labels)), "folds": 0}
    return {"auc": float(np.mean(scores)), "auc_median": float(np.median(scores)),
            "n": int(len(labels)), "folds": len(scores),
            "positives": int((labels == 1).sum())}


def transfer_probe(train_features, train_labels, test_features, test_labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
        return np.nan
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced",
                                             max_iter=2000, random_state=SEED))
    model.fit(np.asarray(train_features, dtype=np.float64), np.asarray(train_labels))
    probability = model.predict_proba(np.asarray(test_features,
                                                 dtype=np.float64))[:, 1]
    return float(roc_auc_score(np.asarray(test_labels), probability))


def identity_probe(features, labels, groups, folds: int = 5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    scores = []
    for train_index, test_index in splitter.split(features, labels, groups):
        assert not (set(groups[train_index]) & set(groups[test_index]))
        if len(set(labels[train_index])) < 2:
            continue
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, class_weight="balanced",
                                                 max_iter=2000, random_state=SEED))
        model.fit(features[train_index], labels[train_index])
        scores.append(f1_score(labels[test_index],
                               model.predict(features[test_index]),
                               average="macro"))
    return {"macro_f1": float(np.mean(scores)) if scores else np.nan,
            "folds": len(scores), "n": int(len(labels))}
