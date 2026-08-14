"""wood_pose_figure.py — presentation figure: raw frame vs inference-only 6D pose.

The evaluation overlays under `wood_gt_overlays/` draw ground truth in green next
to the prediction, which is what a *diagnostic* wants and the wrong thing for a
slide: it shows the answer alongside the guess.  This renders the same frame as
the deployed system would see it -- no ground truth anywhere -- plus the pose
angles, which the diagnostic overlay never printed.

Inference path is `wood_infer_filter` unchanged (Stage B `net_epoch_0057`,
squash-parity preprocessing, `solve_pose` with the user-given wood dimensions),
so the drawn cuboid is the same pose the evaluation scored.

Both the drawn triad and the printed angles use a **display frame** with Y up and
Z toward the viewer, which is the orientation people read off a slide.  The
repository's own convention is OpenCV-style (X right, Y **down**, Z **away**), so
the display frame is that one conjugated by ``diag(1, -1, -1)`` -- applied to the
object and the camera alike, and still right-handed.  Angles are the object's
orientation relative to the camera, Tait-Bryan Y-X-Z:

    R_display = Ry(yaw) . Rx(pitch) . Rz(roll)

The sequence is intrinsic -- yaw about Y, then pitch about the new X, then roll
about the new Z -- so the arrows are labelled the way body axes are labelled in
aerospace: Y is the yaw axis, X the pitch axis, Z the roll axis.

They are not gravity-referenced: the camera itself is tilted, so `pitch` mostly
reports that tilt rather than any tilt of the pallet.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wf = _load("wf_fig", os.path.join(ROOT, "scripts", "stage0", "wood_infer_filter.py"))
APNP = wf.APNP
K_for_resolution = wf.K_for_resolution

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
# Y up, Z toward the viewer.  Its own inverse, and right-handed: X x (-Y) = -Z.
DISPLAY_FLIP = np.diag([1.0, -1.0, -1.0])
YAW_COLOUR, PITCH_COLOUR, ROLL_COLOUR = (60, 220, 60), (60, 60, 255), (255, 160, 60)
FRAME_ROOT = os.path.join(ROOT, "data", "pallet", "raw_data", "wood", "selected")
FIG_DIR = os.path.join(ROOT, "_docs", "figures")


def tait_bryan_yxz(R):
    """(yaw, pitch, roll) in degrees for R = Ry(yaw) . Rx(pitch) . Rz(roll)."""
    pitch = -math.asin(float(np.clip(R[1, 2], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-6:                      # gimbal lock
        yaw = math.atan2(-R[2, 0], R[0, 0])
        roll = 0.0
    else:
        yaw = math.atan2(R[0, 2], R[2, 2])
        roll = math.atan2(R[1, 0], R[1, 1])
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def rotation_from_yxz(yaw, pitch, roll):
    """Inverse of :func:`tait_bryan_yxz`; used to check the decomposition."""
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                   [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)],
                   [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0],
                   [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return ry @ rx @ rz


def draw_pose(image, corners, keypoints, axes, angles, distance, thick=6,
              panel=False):
    """Slide rendering: heavy strokes, and by default no text block.

    ``panel=True`` restores the numeric corner block, which is useful when the
    figure has to stand on its own without a caption.
    """
    out = image.copy()
    for a, b in EDGES:
        pa, pb = corners[a], corners[b]
        if min(pa[0], pb[0]) < -1e5:
            continue
        # dark casing first, so the cuboid stays legible over pale concrete
        cv2.line(out, tuple(np.int32(pa)), tuple(np.int32(pb)), (0, 0, 0), thick + 4)
        cv2.line(out, tuple(np.int32(pa)), tuple(np.int32(pb)), (0, 215, 255), thick)
    for point in keypoints:
        if point is None or np.isnan(point[0]):
            continue
        cv2.circle(out, tuple(np.int32(point)), 9, (0, 0, 0), -1)
        cv2.circle(out, tuple(np.int32(point)), 7, (255, 255, 255), -1)
    origin = tuple(np.int32(axes[0]))
    yaw, pitch, roll = angles
    # Body-axis naming, as in aerospace: the intrinsic sequence is yaw about Y,
    # then pitch about the new X, then roll about the new Z, so each arrow is
    # the axis of the angle printed in its own colour.
    for end, colour, label in ((axes[2], YAW_COLOUR, "yaw"),
                               (axes[1], PITCH_COLOUR, "pitch"),
                               (axes[3], ROLL_COLOUR, "roll")):
        tip = tuple(np.int32(end))
        cv2.arrowedLine(out, origin, tip, (0, 0, 0), thick + 4, tipLength=0.20)
        cv2.arrowedLine(out, origin, tip, colour, thick, tipLength=0.20)
        anchor = tuple(np.int32(end) + np.array([10, -10]))
        cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    colour, 3, cv2.LINE_AA)
    if not panel:
        return out

    lines = [(f"yaw   {yaw:+7.1f} deg", YAW_COLOUR),
             (f"pitch {pitch:+7.1f} deg", PITCH_COLOUR),
             (f"roll  {roll:+7.1f} deg", ROLL_COLOUR),
             (f"dist  {distance:6.2f} m", (255, 255, 255))]
    height = 34 * len(lines) + 22
    panel = out[10:10 + height, 10:10 + 300].copy()
    out[10:10 + height, 10:10 + 300] = cv2.addWeighted(
        panel, 0.25, np.zeros_like(panel), 0.75, 0)
    for i, (line, colour) in enumerate(lines):
        cv2.putText(out, line, (26, 46 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, colour, 2, cv2.LINE_AA)
    return out


def render(session, frame, axis_length=0.25, thick=6, panel=False,
           device="cuda"):
    path = os.path.join(FRAME_ROOT, session, f"{frame}.jpg")
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"frame not found: {path}")
    height, width = image.shape[:2]
    K, k_source = K_for_resolution(width, height)
    if K is None:
        raise SystemExit(k_source)
    K = np.asarray(K, float)

    import torch
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = wf.s1.E.load_model(wf.WEIGHTS, device)
    pred8, pred_c, _, _ = wf.decode_squash(model, image, device)
    pose, posdepth, reproj = wf.solve_wood(pred8, pred_c, K, image.shape)
    if pose is None:
        raise SystemExit("no pose recovered for this frame")

    R = np.array(pose["R"], float)
    t = np.array(pose["t"], float).reshape(3)
    # Same flip on the object and on the camera, so this is a change of display
    # basis rather than a different pose.
    R_display = DISPLAY_FLIP @ R @ DISPLAY_FLIP
    angles = tait_bryan_yxz(R_display)
    residual = float(np.abs(rotation_from_yxz(*angles) - R_display).max())
    if residual > 1e-6:
        raise SystemExit(f"angle decomposition does not reproduce R: {residual:.2e}")

    rvec = cv2.Rodrigues(R)[0]
    triad = np.array([[0, 0, 0], [axis_length, 0, 0],
                      [0, axis_length, 0], [0, 0, axis_length]], float) @ DISPLAY_FLIP
    axes = cv2.projectPoints(triad, rvec, t, K, None)[0].reshape(-1, 2)
    corners = np.array(pose["projected_all"], float)[:8]   # index 8 is the centroid
    keypoints = [None if np.isnan(pred8[i, 0]) else pred8[i] for i in range(8)]

    overlay = draw_pose(image, corners, keypoints, axes, angles,
                        float(np.linalg.norm(t)), thick=thick, panel=panel)
    os.makedirs(FIG_DIR, exist_ok=True)
    stem = f"wood_{session.split('_')[-1]}_{frame}"
    raw_path = os.path.join(FIG_DIR, f"{stem}_raw.png")
    pose_path = os.path.join(FIG_DIR, f"{stem}_pose.png")
    pair_path = os.path.join(FIG_DIR, f"{stem}_pair.png")
    cv2.imwrite(raw_path, image)
    cv2.imwrite(pose_path, overlay)
    gap = np.full((height, 10, 3), 255, np.uint8)
    cv2.imwrite(pair_path, np.hstack([image, gap, overlay]))

    print(f"frame      {session}/{frame}  {width}x{height}")
    print(f"K          {k_source}")
    print(f"detected   {int(np.sum(~np.isnan(pred8[:, 0])))}/8 corners"
          f"   centroid {'yes' if pred_c else 'no'}")
    print(f"yaw/pitch/roll  {angles[0]:+.1f} / {angles[1]:+.1f} / {angles[2]:+.1f} deg"
          "   (display frame: Y up, Z toward viewer)")
    print(f"t          [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m   |t| {np.linalg.norm(t):.3f} m")
    print(f"reproj self-consistency  {reproj:.2f} px   positive depth {posdepth}")
    print(f"wrote      {raw_path}\n           {pose_path}\n           {pair_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="pallet_20260618_184309")
    parser.add_argument("--frame", default="000132")
    parser.add_argument("--axis-length", type=float, default=0.25)
    parser.add_argument("--thick", type=int, default=6)
    parser.add_argument("--panel", action="store_true",
                        help="restore the numeric text block")
    arguments = parser.parse_args()
    render(arguments.session, arguments.frame, arguments.axis_length,
           arguments.thick, arguments.panel)
