"""Draw what the PPD failure looks like on the actual images.

The numbers say the learned 5-class polarity map inverts on real (0.023) while an
oracle map on the same candidates scores 86/86.  This renders that claim so it can
be checked by eye: predicted class map, the candidate the scorer picked, and the
candidate it should have picked.

    python scripts/stage0/paper_s2/paper_s2_ppd_overlay.py --arm L0 --n 6
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib

import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2_palletgraph_line_screen/figures/overlays"

_spec = importlib.util.spec_from_file_location(
    "LR", ROOT / "scripts/stage0/paper_s2/paper_s2_ppd_long_run.py")
LR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LR)
PG, PPD, PLH = LR.PG, LR.PPD, LR.PLH

# BGR, one per CLASS_ORDER entry.  top warm, base cool, vertical grey.
COLOR = {"top_width": (60, 60, 235), "top_depth": (60, 170, 245),
         "base_width": (235, 140, 60), "base_depth": (200, 220, 90),
         "vertical": (170, 170, 170)}


def class_map(prob, size):
    """argmax colour image at full resolution, dark where nothing fires."""
    up = np.stack([cv2.resize(c, size, interpolation=cv2.INTER_LINEAR) for c in prob])
    idx, peak = up.argmax(0), up.max(0)
    canvas = np.zeros((size[1], size[0], 3), np.uint8)
    for k, name in enumerate(PLH.CLASS_ORDER):
        canvas[(idx == k) & (peak > 0.5)] = COLOR[name]
    return canvas, peak


def draw_edges(canvas, R, t, K, dims, thickness=2, colour=None):
    W, H = canvas.shape[1], canvas.shape[0]
    proj, dep = PG.project_points(PG.make_corners(*dims)[:8], R, t, K)
    for (i, j), cls in PPD.polarity_edge_classes(dims):
        if dep[i] <= 1e-6 or dep[j] <= 1e-6:
            continue
        clipped = PG.clip_segment_to_image(proj[i], proj[j], W, H)
        if clipped is None:
            continue
        a, b = np.round(clipped[0]).astype(int), np.round(clipped[1]).astype(int)
        cv2.line(canvas, tuple(a), tuple(b), colour or COLOR[cls], thickness, cv2.LINE_AA)
    return canvas


def label(img, text, colour=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
    return img


def upright_candidate(frame):
    """The candidate closest to the reference pose — what a correct map would pick."""
    best, err = None, None
    for cand in frame["cands"]:
        e = PPD.vertical_polarity_error_deg(cand, frame["R"], frame["dims"])
        if err is None or e < err:
            best, err = cand, e
    return best, err


def panel(frame, prob, picked, arm):
    img = LR_IMAGES[frame["file"]]
    size = (img.shape[1], img.shape[0])
    cmap, peak = class_map(prob, size)

    a = label(img.copy(), "1. image + reference pose (GT-solved)")
    draw_edges(a, frame["R"], frame["t"], frame["K"], frame["dims"])

    b = label(cv2.addWeighted(img, 0.35, cmap, 0.65, 0), f"2. predicted 5-class map ({arm})")

    upright, _ = upright_candidate(frame)
    c = img.copy()
    # picked first, upright thinner on top: a 180-degree flip of a flat pallet
    # projects to almost the same place, which is exactly why the map has to decide.
    draw_edges(c, picked, frame["t"], frame["K"], frame["dims"], 4, (60, 60, 235))
    draw_edges(c, upright, frame["t"], frame["K"], frame["dims"], 1, (90, 230, 90))
    label(c, "3. green = upright candidate   red = the one the map picked")

    top = np.maximum(prob[PLH.CLASS_ORDER.index("top_width")],
                     prob[PLH.CLASS_ORDER.index("top_depth")])
    base = np.maximum(prob[PLH.CLASS_ORDER.index("base_width")],
                      prob[PLH.CLASS_ORDER.index("base_depth")])
    diff = cv2.resize(base - top, size, interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8((np.clip(diff, -1, 1) + 1) * 127.5), cv2.COLORMAP_JET)
    d = label(cv2.addWeighted(img, 0.4, heat, 0.6, 0), "4. base minus top  (red = says BASE)")

    row1 = np.hstack([a, b])
    row2 = np.hstack([c, d])
    return np.vstack([row1, row2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="L0")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="also render this many synthetic validation frames for contrast")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = LR.Base().to(device).eval()
    frames = LR.real_frames()
    _, channels = base.discover(frames[0]["x"][None].to(device))
    line, _, epoch = LR.load_best(args.arm, channels)
    line = line.to(device).eval()
    print(f"[overlay] {args.arm} epoch {epoch}, {len(frames)} real frames")

    global LR_IMAGES
    spec = importlib.util.spec_from_file_location(
        "LS", ROOT / "scripts/stage0/paper_s2_palletgraph_line_screen.py")
    ls = importlib.util.module_from_spec(spec); spec.loader.exec_module(ls)
    LR_IMAGES = ls.LineScreenEvaluator().images

    written = []
    with torch.no_grad():
        for frame in frames:
            feature = base(frame["x"][None].to(device))
            gate = torch.ones(1, 1, LR.GRID, LR.GRID, device=device)
            prob = torch.sigmoid(line(feature, gate))[0].cpu().numpy()
            picked = LR.polarity_select(prob, frame)
            if picked is None:
                continue
            err = PPD.vertical_polarity_error_deg(picked["R"], frame["R"], frame["dims"])
            _, best_err = upright_candidate(frame)
            inverted = err > 45.0
            written.append((frame, prob, picked["R"], inverted, err, best_err))

    inverted = [w for w in written if w[3]]
    correct = [w for w in written if not w[3]]
    print(f"[overlay] inverted {len(inverted)} / kept {len(written)}  correct {len(correct)}")

    chosen = correct[: max(1, args.n // 3)] + inverted[: args.n - max(1, args.n // 3)]
    for frame, prob, picked, is_inv, err, best_err in chosen:
        tag = "INVERTED" if is_inv else "CORRECT"
        sheet = panel(frame, prob, picked, args.arm)
        header = np.zeros((44, sheet.shape[1], 3), np.uint8)
        cv2.putText(header, f"{tag}  {frame['domain']}  vertical-polarity err {err:.1f} deg "
                            f"(best candidate available {best_err:.1f} deg)",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (80, 80, 255) if is_inv else (120, 240, 120), 2, cv2.LINE_AA)
        sheet = np.vstack([header, sheet])
        scale = min(1.0, 1800 / sheet.shape[1])
        if scale < 1.0:
            sheet = cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = f"{tag.lower()}_{frame['domain']}_{str(frame['file']).rsplit(':', 1)[-1]}.jpg"
        cv2.imwrite(str(OUT / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  wrote {OUT / name}")

    if args.synthetic:
        import json
        val = json.loads((LR.D / "ppd_val_manifest.json").read_text())
        files = [f["file"] for f in val["frames"]]
        done = 0
        with torch.no_grad():
            for fn in files:
                if done >= args.synthetic:
                    break
                frame = LR.load_frame(fn, with_candidates=True)
                if not frame["cands"]:
                    continue
                feature = base(frame["x"][None].to(device))
                gate = torch.ones(1, 1, LR.GRID, LR.GRID, device=device)
                prob = torch.sigmoid(line(feature, gate))[0].cpu().numpy()
                picked = LR.polarity_select(prob, frame)
                if picked is None:
                    continue
                err = PPD.vertical_polarity_error_deg(picked["R"], frame["R"], frame["dims"])
                LR_IMAGES[frame["file"]] = cv2.imread(
                    str(LR.DATA / fn.replace(".json", ".png")))
                frame["domain"] = "synthetic"
                sheet = panel(frame, prob, picked["R"], args.arm)
                header = np.zeros((44, sheet.shape[1], 3), np.uint8)
                cv2.putText(header, f"SYNTHETIC (same checkpoint)  vertical-polarity err "
                                    f"{err:.1f} deg", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 220, 120), 2, cv2.LINE_AA)
                sheet = np.vstack([header, sheet])
                scale = min(1.0, 1800 / sheet.shape[1])
                if scale < 1.0:
                    sheet = cv2.resize(sheet, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                name = f"synthetic_{pathlib.Path(fn).stem}.jpg"
                cv2.imwrite(str(OUT / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"  wrote {OUT / name}")
                done += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
