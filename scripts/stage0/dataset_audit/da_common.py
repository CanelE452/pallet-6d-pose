"""Dataset registry and per-frame feature extraction for the composition audit.

Roles are not inferred from directory names.  Every dataset here is described by
what its own labels say -- gates, visibility counts, loss cause -- and the role
column in the master table is derived from those fields plus the experimental
record, never from the string in the path.

Three counts are kept apart on purpose, because conflating them is what made the
old `V` column mean the wrong thing:

    n_inframe      corners whose projection lands inside the image
    V_vis_actual   corners the renderer reports as actually visible
    n_supervised   channels the corner loss actually trains on -- which is the
                   in-frame test over 9 channels (8 corners + centroid), taken
                   from mh_data.belief_target, not a guess
"""
from __future__ import annotations

import io
import json
import pathlib
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RELEASE = ROOT / "data/pallet/training_data/paper_release"
OUT = ROOT / "data/pallet/results/paper_s2_multihead"
AUDIT = OUT / "data_audit"
EDA = OUT / "eda"
RELEASE_OUT = OUT / "dataset_release"

GRID = 50          # mh_data.GRID -- the in-frame test happens on this grid


# --------------------------------------------------------------------------
# registry


class Source:
    """A dataset that can enumerate (stem, label dict) without being unpacked."""

    def __init__(self, dataset_id, path, kind, subdir=""):
        self.dataset_id = dataset_id
        self.path = pathlib.Path(path)
        self.kind = kind                # "dir" | "zip"
        self.subdir = subdir

    @property
    def exists(self):
        return self.path.exists()

    def stems(self):
        if not self.exists:
            return []
        if self.kind == "dir":
            labels = self.path / "labels"
            return sorted(p.name[: -len("_label.json")]
                          for p in labels.iterdir()
                          if p.name.endswith("_label.json"))
        with zipfile.ZipFile(self.path) as zf:
            return sorted(pathlib.PurePosixPath(n).name[: -len("_label.json")]
                          for n in zf.namelist()
                          if n.endswith("_label.json"))

    def labels(self):
        """Yield (stem, payload).  Zips are streamed, never extracted."""
        if not self.exists:
            return
        if self.kind == "dir":
            for stem in self.stems():
                path = self.path / "labels" / f"{stem}_label.json"
                yield stem, json.loads(path.read_text("utf-8"))
            return
        with zipfile.ZipFile(self.path) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith("_label.json"))
            for name in names:
                stem = pathlib.PurePosixPath(name).name[: -len("_label.json")]
                with zf.open(name) as fh:
                    yield stem, json.loads(io.TextIOWrapper(fh, "utf-8").read())

    def rgb_names(self):
        if not self.exists:
            return []
        if self.kind == "dir":
            rgb = self.path / "rgb"
            return sorted(p.name for p in rgb.iterdir()) if rgb.exists() else []
        with zipfile.ZipFile(self.path) as zf:
            return sorted(n for n in zf.namelist()
                          if "/rgb/" in n or n.startswith("rgb/"))


POSITIVE_SOURCES = [
    Source("BROAD_40K", RELEASE / "v2_prod40k_clean_merged", "dir"),
    Source("CORNER_LA_Y15_30",
           RELEASE / "oblique/extracted/corner_la_oblique_v1_y15_30", "dir"),
    Source("CORNER_LA_Y30_PLUS",
           RELEASE / "oblique/extracted/corner_la_oblique_v1_y30_plus", "dir"),
    Source("CORNER_LA_FRONTAL", RELEASE / "frontal", "dir"),
    Source("EDGE_HARD_TRUNC_TRAIN",
           RELEASE / "edge/edge_complement_v1_trunc_train.zip", "zip"),
    Source("EDGE_HARD_TRUNC_DEV",
           RELEASE / "edge/edge_complement_v1_trunc_dev.zip", "zip"),
    Source("EDGE_HARD_TRUNC_UNTOUCHED",
           RELEASE / "edge/edge_complement_v1_trunc_untouched.zip", "zip"),
    Source("EDGE_HARD_CLEAN_UNTOUCHED",
           RELEASE / "edge/edge_complement_v1_clean_untouched.zip", "zip"),
]

NEGATIVE_SOURCES = [
    Source("NEGATIVE_SYNTH_V1_TRAIN",
           RELEASE / "negative/extracted/negative_synth_v1_train", "dir"),
    Source("NEGATIVE_SYNTH_V1_DEV",
           RELEASE / "negative/extracted/negative_synth_v1_dev", "dir"),
]


# --------------------------------------------------------------------------
# feature extraction


def _get(node, *keys, default=None):
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _inframe_mask(points, width, height):
    """The loss's own test, on the 50-grid, so it matches what is trained.

    mh_data.belief_target maps x -> x * GRID / width and keeps a channel when
    0 <= x < GRID and 0 <= y < GRID.  Doing the same here means n_supervised is
    the trained count rather than a plausible-looking proxy.
    """
    pts = np.asarray(points, float)
    if pts.size == 0:
        return np.zeros(0, bool)
    gx = pts[:, 0] * GRID / max(width, 1)
    gy = pts[:, 1] * GRID / max(height, 1)
    finite = np.isfinite(gx) & np.isfinite(gy)
    return finite & (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID)


def positive_row(dataset_id, stem, payload):
    """One canonical row.

    Flags are DERIVED from geometry for every dataset, because the datasets do
    not agree on metadata: BROAD carries `pnp_conditioning.loss_cause`, while
    EDGE_HARD has no `pnp_conditioning` at all (its 72.7% truncation figure
    lives in a README, not in the frames).  A label-declared cause is kept in a
    separate column so the two are never confused.
    """
    camera = payload.get("camera_data", {}) or {}
    objects = payload.get("objects") or [{}]
    obj = objects[0] if objects else {}
    v2 = _get(obj, "v2_labels", default={}) or {}
    place = _get(obj, "scene_placement_v2", default={}) or {}
    gates = _get(obj, "safety_gates", default={}) or {}
    pnp = _get(obj, "pnp_conditioning", default={}) or {}
    seeds = _get(place, "stage_seeds", default={}) or {}

    width = camera.get("width") or 0
    height = camera.get("height") or 0
    corners = obj.get("projected_cuboid") or []
    centroid = obj.get("projected_cuboid_centroid")

    inframe_corner = _inframe_mask(corners, width, height)
    channels = list(corners) + ([centroid] if centroid else [])
    inframe_channel = _inframe_mask(channels, width, height)
    n_inframe = int(inframe_corner.sum())

    pts = np.asarray(corners, float) if corners else np.zeros((0, 2))
    if len(pts):
        span = pts.max(0) - pts.min(0)
        bw = float(span[0]) / max(width, 1)
        bh = float(span[1]) / max(height, 1)
    else:
        bw = bh = float("nan")

    facing = obj.get("facing_margin")
    canonical_yaw = (45.0 - float(facing)) if facing is not None else float("nan")

    v_vis = v2.get("V_vis_actual")
    ext_occ = v2.get("ext_occ_corners_actual") or 0
    occluded = (int(v_vis) < n_inframe) if v_vis is not None else None

    occ_frac = v2.get("occlusion_fraction") or []
    return {
        "dataset_id": dataset_id,
        "frame_id": stem,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "pallet_type": v2.get("pallet_type") or obj.get("name"),
        "source_asset": obj.get("source_asset"),
        "background_asset": camera.get("background_asset"),
        "scene_preset": camera.get("scene_preset"),
        "floor_mode": camera.get("floor_mode"),
        "keypoint_convention": obj.get("keypoint_convention"),

        "elevation_deg_actual": v2.get("elevation_deg_actual"),
        "azimuth_deg_target": v2.get("azimuth_deg_target"),
        "facing_margin": facing,
        "canonical_frontal_yaw_deg": canonical_yaw,
        "camera_distance_actual_m": v2.get("camera_distance_actual_m"),
        "projected_size_actual": v2.get("projected_size_actual"),

        "bbox_width_norm": bw,
        "bbox_height_norm": bh,
        "bbox_diag_norm": float(np.hypot(bw, bh)),
        "bbox_area_fraction": float(bw * bh),

        "n_inframe": n_inframe,
        "n_supervised": int(inframe_channel.sum()),
        "V_actual": v2.get("V_actual"),
        "V_vis_actual": v_vis,
        "visible_kp_count": pnp.get("visible_kp_count"),
        "ext_occ_corners_actual": ext_occ,

        # derived, identical rule for every dataset
        "truncation": bool(n_inframe < len(corners)) if corners else None,
        "any_occlusion": occluded,
        "external_occlusion": bool(ext_occ) and int(ext_occ) > 0,
        "self_occlusion": bool(occluded) and int(ext_occ) == 0,
        "occluded_corner_count": int(sum(1 for f in occ_frac[:8] if f)),
        # label-declared, present only where the generator wrote it
        "declared_loss_cause": pnp.get("loss_cause"),
        "declared_degeneracy": pnp.get("degeneracy"),
        "has_pnp_conditioning": bool(pnp),

        "occluder_placed": bool(v2.get("occluder_placed")),
        "f_total": v2.get("f_total"),

        "luma_frame": v2.get("luma_actual"),
        "luma_pallet": v2.get("luma_pallet_actual"),
        "luma_frame_final": v2.get("luma_frame_final"),
        "exposure_ev": v2.get("exposure_ev", camera.get("exposure_ev")),
        "noise_tier": v2.get("noise_tier"),

        "gate_all_pass": gates.get("all_pass"),
        "gate_G1_vvis_ge4": gates.get("G1_Vvis>=4"),
        "gate_G2_extocc": gates.get("G2_extocc_1to4"),
        "gate_G3_visible_half": gates.get("G3_visible>=0.5unocc"),
        "gate_G4_center_inframe": gates.get("G4_center_inframe"),
        "gate_G5_luma_floor": gates.get("G5_luma_floor"),

        "seed_pallet": seeds.get("pallet"),
        "seed_background": seeds.get("background"),
        "seed_anchor": seeds.get("anchor"),
        "seed_cargo": seeds.get("cargo"),
        "seed_context": seeds.get("context"),
        "seed_occluder": seeds.get("occluder"),
    }


def negative_row(dataset_id, stem, payload):
    """Negatives carry their own contract; positive geometry is not forced on them.

    Fields come from the two blocks the generator actually writes -- `negative`
    (family_intent / impostor_type / subtype / neg_id / source_mode) and
    `release` (negative_type / split / sample_id / origin) -- rather than from a
    guessed schema.
    """
    camera = payload.get("camera_data", {}) or {}
    neg = payload.get("negative", {}) or {}
    rel = payload.get("release", {}) or {}
    width = camera.get("width") or 0
    height = camera.get("height") or 0
    return {
        "dataset_id": dataset_id,
        "frame_id": stem,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "background_asset": camera.get("background_asset"),
        "scene_preset": camera.get("scene_preset"),
        "floor_mode": camera.get("floor_mode"),

        "negative_type": rel.get("negative_type"),
        "family_intent": neg.get("family_intent"),
        "impostor_type": neg.get("impostor_type"),
        "subtype": neg.get("subtype"),
        "source_mode": neg.get("source_mode"),
        "neg_id": neg.get("neg_id"),
        "has_object": neg.get("has_object"),
        "target_present": neg.get("target_present"),

        "declared_split": rel.get("split"),
        "sample_id": rel.get("sample_id"),
        "origin": rel.get("origin"),
        "release_index": rel.get("index"),

        "object_present": payload.get("object_present"),
        "pose_valid": payload.get("pose_valid"),
        "n_objects": len(payload.get("objects") or []),
        "n_keypoints": len(payload.get("keypoints") or []),
        "n_structural_lines": len(payload.get("structural_lines") or []),
    }
