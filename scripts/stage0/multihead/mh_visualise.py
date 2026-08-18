"""Section 22: the same frame through every stage of both branches.

Six panels per frame, in the order the brief lists them, so a reader can follow
one image from pixels to pose without switching figures:

    RGB | corner heatmaps + decoded corners | 12 predicted structural lines
    line -> CIGM corners | mask GT vs prediction | final PnP cuboid

Attention maps are written to a separate file when asked for.  They are never
used as evidence for a performance claim -- an attention map shows where a query
read, not whether the answer was right.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                            # noqa: E402
import mh_cigm as CG                                            # noqa: E402
import mh_data as MD                                            # noqa: E402
import mh_screen as MS                                          # noqa: E402
from mh_arms import CAP, DH                                     # noqa: E402

OUT = MD.OUT / "figures"
PANEL = 400
EDGE_COLOUR = (60, 220, 255)
CIGM_COLOUR = (80, 255, 120)
DIRECT_COLOUR = (255, 140, 80)
GT_COLOUR = (200, 200, 200)


def _canvas(image, title):
    panel = cv2.resize(image, (PANEL, PANEL))
    cv2.rectangle(panel, (0, 0), (PANEL, 22), (0, 0, 0), -1)
    cv2.putText(panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _to_panel(grid_points):
    """50-grid coordinates -> pixel coordinates on a PANEL-sized square."""
    return grid_points * (PANEL / MD.GRID)


def _draw_points(panel, points, colour, label=True):
    for index, (x, y) in enumerate(points):
        if not np.isfinite([x, y]).all():
            continue
        centre = (int(round(x)), int(round(y)))
        cv2.circle(panel, centre, 4, colour, -1, cv2.LINE_AA)
        if label:
            cv2.putText(panel, str(index), (centre[0] + 5, centre[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


def _heatmap_panel(belief, decoded, truth):
    """All nine channels collapsed by max, with decoded and GT overlaid."""
    collapsed = belief.max(0)
    collapsed = collapsed / max(float(collapsed.max()), 1e-6)
    coloured = cv2.applyColorMap((collapsed * 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)
    coloured = cv2.resize(coloured, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    _draw_points(coloured, _to_panel(truth[:8]), GT_COLOUR, label=False)
    _draw_points(coloured, _to_panel(decoded[:8]), DIRECT_COLOUR)
    return _canvas(coloured, "corner heatmap + decoded (grey = GT)")


def _lines_panel(rgb, theta, rho):
    """Each predicted line drawn across the frame, in canonical 50-grid."""
    panel = cv2.resize(rgb, (PANEL, PANEL)).copy()
    scale = PANEL / MD.GRID
    for role in range(theta.shape[0]):
        normal = np.array([np.cos(theta[role]), np.sin(theta[role])])
        point = normal * rho[role]
        direction = np.array([-normal[1], normal[0]])
        p0 = (point - direction * 2 * MD.GRID) * scale
        p1 = (point + direction * 2 * MD.GRID) * scale
        cv2.line(panel, tuple(np.int32(p0)), tuple(np.int32(p1)),
                 EDGE_COLOUR, 1, cv2.LINE_AA)
    return _canvas(panel, "12 predicted structural lines")


def _cuboid_panel(rgb, model, pose, camera, width, height, title):
    panel = cv2.resize(rgb, (PANEL, PANEL)).copy()
    if pose is None:
        return _canvas(panel, title + " (no solution)")
    rotation, translation = pose
    points = (rotation @ model.T).T + translation
    projected = (camera @ points.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    projected = projected * np.array([PANEL / width, PANEL / height])
    for a, b in CG.EDGES:
        cv2.line(panel, tuple(np.int32(projected[a])), tuple(np.int32(projected[b])),
                 CIGM_COLOUR, 2, cv2.LINE_AA)
    return _canvas(panel, title)


def _mask_panel(truth, predicted):
    stack = np.zeros((MD.GRID, MD.GRID, 3), np.uint8)
    stack[..., 1] = (truth * 255).astype(np.uint8)          # GT green
    stack[..., 2] = (predicted * 255).astype(np.uint8)      # prediction red
    return _canvas(cv2.resize(stack, (PANEL, PANEL),
                              interpolation=cv2.INTER_NEAREST),
                   "mask  green = GT, red = predicted")


@torch.no_grad()
def render(model, stem, features, grid_theta, grid_rho, valid):
    pack = MD.load_pack([stem])
    out = model(pack["images"], features)
    rgb = cv2.cvtColor(pack["rgb"][0], cv2.COLOR_RGB2BGR)
    truth = pack["grid"][0]
    width, height = pack["resolution"][0]
    label = MD.read_label(stem)
    camera = CG.intrinsics(label)
    object_points = CG.object_points(label)

    theta_hat, rho_hat = DH.decode(out["line_scores"], grid_theta, grid_rho, valid)
    theta_can, rho_can = DH.canonical_from_centred(theta_hat, rho_hat)
    corners_l, _, _ = CG.cigm_corners(theta_hat, rho_hat)
    corners_l = corners_l[0].cpu().numpy()

    panels = [_canvas(rgb, f"RGB  {stem}")]
    if "beliefs" in out:
        belief = out["beliefs"][-1][0, :9].float().cpu().numpy()
        decoded = MS._decode_peaks(out["beliefs"][-1][:, :9])[0]
        panels.append(_heatmap_panel(belief, decoded, truth))
    else:
        panels.append(_canvas(np.zeros_like(rgb), "no corner head in this arm"))
    panels.append(_lines_panel(rgb, theta_can[0].cpu().numpy(),
                               rho_can[0].cpu().numpy()))

    cigm_panel = cv2.resize(rgb, (PANEL, PANEL)).copy()
    _draw_points(cigm_panel, _to_panel(truth[:8]), GT_COLOUR, label=False)
    _draw_points(cigm_panel, _to_panel(corners_l), CIGM_COLOUR)
    panels.append(_canvas(cigm_panel, "line -> CIGM corners (grey = GT)"))

    if "segments" in out:
        predicted = (torch.sigmoid(out["segments"][-1])[0, 0] > 0.5).cpu().numpy()
        panels.append(_mask_panel(pack["mask"][0, 0].cpu().numpy(), predicted))
    else:
        panels.append(_canvas(np.zeros_like(rgb), "no mask head in this arm"))

    pose = CG.solve(object_points,
                    CG.grid_to_pixels(corners_l, width, height), camera)
    panels.append(_cuboid_panel(rgb, object_points, pose, camera, width, height,
                                "PnP cuboid from PATH-L"))
    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])
    return np.vstack([top, bottom])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="A2_CORNER_LINE_MASK")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--frames", type=int, default=6)
    arguments = parser.parse_args()

    MS.deterministic()
    checkpoint = (MS.CKPT / f"screen_{arguments.arm}"
                  / f"step_{arguments.step:05d}.pth")
    state = torch.load(checkpoint, map_location=MH.DEV, weights_only=False)
    torch.manual_seed(CAP.SEED)
    model = MH.MultiHeadModel(arguments.arm)
    model.load_state_dict(state["model"])
    model.eval()

    grid_theta, grid_rho, valid, features = MS.lattice()
    _, populations = MD.pools()
    stems = populations["D2_MH_DEV512"]
    meta = {row["stem"]: row for row in MD.load_split()}
    # one frame per stratum, so the figure is not six easy frames
    chosen, seen = [], set()
    for stem in stems:
        stratum = meta[stem]["stratum"]
        if stratum not in seen:
            seen.add(stratum)
            chosen.append(stem)
        if len(chosen) >= arguments.frames:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    for stem in chosen:
        image = render(model, stem, features, grid_theta, grid_rho, valid)
        path = OUT / f"{arguments.arm}_step{arguments.step:05d}_{stem}.jpg"
        cv2.imwrite(str(path), image)
        print("->", path)


if __name__ == "__main__":
    main()
