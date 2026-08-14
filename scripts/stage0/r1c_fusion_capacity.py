"""R1C: teacher-forced GT-edge fusion capacity, and the direct CIGM->PnP interface.

Post-validation diagnostic.  Round-1 stands at NO_EDGE_ARCHITECTURE_PASS and
nothing here edits it.

Round-1's ORACLE row fed clean edges at inference into an EGCR that had been
fitted against a noisy PEQ, which measures that pipeline's response to clean
input rather than the fusion's own capacity.  Here EGCR is fitted on ground-truth
edges from the first step, on the same subset, the same frozen A1 and the same
166 steps, so the two differ in one thing.

Both arms consume ground-truth edges at inference.  They are capacity oracles,
never candidates.
"""
from __future__ import annotations

import hashlib, importlib.util, json, pathlib, sys
import numpy as np, torch, cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "Deep_Object_Pose/train",
           "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))

OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
DATA = ROOT / "data/pallet/training_data/pallet6d_v2_10k"
GRID, STEPS, SEED, LR, WD = 50, 166, 1, 1e-3, 1e-4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modules():
    spec = importlib.util.spec_from_file_location(
        "EFS", ROOT / "scripts/stage0/line/edge_mandatory_fast_search.py")
    efs = importlib.util.module_from_spec(spec)
    sys.modules["EFS"] = efs
    spec.loader.exec_module(efs)
    import instance_edge_topology as IET, corner_incident_geometry as CG
    import edge_guided_corner_fusion as EG, spatial_hcrm as HC
    import annotate_pnp as APNP, pallet_graph_geometry as PG
    return efs, IET, CG, EG, HC, APNP, PG


def load_frame(index):
    payload = json.loads((DATA / "all" / f"{index}.json").read_text("utf-8"))
    camera = payload["camera_data"]
    image = cv2.imread(str(DATA / "all" / f"{index}.png"))
    height, width = image.shape[:2]
    array = cv2.cvtColor(cv2.resize(image, (400, 400)), cv2.COLOR_BGR2RGB)
    array = (array.astype(np.float32) / 255.0 - MEAN) / STD
    cuboid = np.asarray(payload["objects"][0]["projected_cuboid"], float)
    intrinsics = np.array([[camera["intrinsics"]["fx"], 0, camera["intrinsics"]["cx"]],
                           [0, camera["intrinsics"]["fy"], camera["intrinsics"]["cy"]],
                           [0, 0, 1.0]])
    dims = payload["objects"][0]["dimensions_m"]
    grid = np.stack([cuboid[:, 0] * GRID / width, cuboid[:, 1] * GRID / height], 1)
    return (torch.from_numpy(array.transpose(2, 0, 1)), grid, (width, height), intrinsics,
            (dims["width"], dims["depth"], dims["height"]), cuboid)


def decode(belief):
    """The canonical DOPE readout: raw peak >= 0.30, then argmax."""
    points = []
    for channel in range(8):
        heat = belief[channel]
        if float(heat.max()) < 0.30:
            points.append(None)
            continue
        y, x = np.unravel_index(heat.argmax(), heat.shape)
        points.append((float(x), float(y)))
    return points


def canonical_line(centre, direction):
    """Infinite supporting line as (unit normal, rho), sign disambiguated.

    CIGM consumes lines, not segments, so the noise study perturbs this form
    rather than a centre and a half-length.
    """
    normal = np.stack([-direction[..., 1], direction[..., 0]], -1)
    normal = normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9, None)
    rho = (normal * centre).sum(-1)
    lead = np.where(np.abs(normal[..., 0]) > 1e-9, normal[..., 0], normal[..., 1])
    flip = lead < 0
    normal = np.where(flip[..., None], -normal, normal)
    rho = np.where(flip, -rho, rho)
    return normal, rho


def line_to_segment(normal, rho, half_length):
    """Back to the (centre, direction) pair solve_corners expects."""
    direction = np.stack([-normal[..., 1], normal[..., 0]], -1)
    centre = normal * rho[..., None]
    return centre, direction, half_length
