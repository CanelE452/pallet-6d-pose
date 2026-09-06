"""R0 오검출(false positive) 육안 검수용 contact sheet + 전체 프레임 오버레이.

목적 : 사람이 이미지만 보고 "이 오검출이 무엇에 붙었는가" 를 스스로 분류할 수 있게 깔아 준다.
지표 : 크롭에 주변 맥락이 남아 있는가 · 캡션에 해석이 하나도 없는가.

★이 스크립트는 오검출에 이름표를 붙이지 않는다. 카테고리는 이미지를 본 사람이 만든다.
  캡션에 적는 것은 conf 와 프레임 stem 두 개뿐이다.

산출:
    _docs/audits/.../false_positive_review/sheet_NN.png   5x4 = 20 셀, conf 내림차순
    _docs/audits/.../false_positive_review/full/<stem>.png 원본 전체 + 박스 (원해상도)
    data/pallet/results/.../FALSE_POSITIVE_MANIFEST.json   시트별 셀 순서 <-> stem 매핑
"""
from __future__ import annotations

import csv
import datetime
import json
import pathlib
import subprocess
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

ROOT = pathlib.Path(__file__).resolve().parents[3]

ASSETS = pathlib.Path.home() / ".claude/agents/viz-expert/assets"
plt.style.use(ASSETS / "analysis.mplstyle")
sys.path.insert(0, str(ASSETS))
from palette import CV  # noqa: E402

RESULTS = ROOT / "data/pallet/results/accuracy_root_cause_v1"
OUT = ROOT / "_docs/audits/accuracy_root_cause_v1/false_positive_review"
FULL = OUT / "full"
SRC_CSV = RESULTS / "R0_FALSE_POSITIVES.csv"

EXPAND = 1.6            # 박스 확장 배율 — 주변 맥락이 보여야 무엇에 붙었는지 안다
CELL_ASPECT = 4 / 3     # 셀 종횡비 (letterbox 기준)
NCOL, NROW = 5, 4
PER_SHEET = NCOL * NROW
DPI = 100               # full/ 원해상도 저장용 (figsize x DPI = 원본 픽셀)
SHEET_DPI = 150         # contact sheet

C_BOX = CV["bbox_pred"]
C_TEXT = CV["text"]
C_TEXT_BG = CV["text_bg"]
PAD_RGB = (200, 200, 200)   # letterbox 여백 — 야간/주간 사진 어느 쪽과도 안 헷갈리게


def stroked(ax, x, y, s, size, **kw):
    """야간 프레임에서도 읽히도록 외곽선을 준다 (강조가 아니라 대비 장치)."""
    return ax.text(x, y, s, color=C_TEXT, fontsize=size,
                   path_effects=[pe.withStroke(linewidth=2.2, foreground=C_TEXT_BG)],
                   **kw)


def crop_region(box, img_w, img_h):
    """박스를 EXPAND 배 확장하고 셀 종횡비에 맞춘 crop 사각형.

    확장된 사각형이 이미지를 넘으면 (가) 이미지 안에 들어가도록 줄이고 (나) 안쪽으로 민다.
    바깥은 어차피 볼 것이 없어서, 여백으로 채우면 셀 면적만 버리게 된다.
    줄여도 박스보다 작아지면 그때만 여백을 허용한다 (박스가 잘리는 편이 더 나쁘다).
    """
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cw, ch = bw * EXPAND, bh * EXPAND
    if cw / ch < CELL_ASPECT:
        cw = ch * CELL_ASPECT
    else:
        ch = cw / CELL_ASPECT

    s = min(1.0, img_w / cw, img_h / ch)
    if s < 1.0 and cw * s >= bw and ch * s >= bh:
        cw, ch = cw * s, ch * s

    rx, ry = cx - cw / 2, cy - ch / 2
    if cw <= img_w:
        rx = min(max(rx, 0.0), img_w - cw)
    if ch <= img_h:
        ry = min(max(ry, 0.0), img_h - ch)
    return rx, ry, cw, ch


def letterboxed_crop(img, rect):
    """crop 사각형을 이미지에서 잘라내고 밖으로 나간 만큼은 여백으로 채운다.

    잘라낸 뒤 종횡비를 다시 맞추지 않는다 — 늘어나면 사람이 형태를 오판한다.
    """
    rx, ry, rw, rh = rect
    x0, y0 = int(np.floor(rx)), int(np.floor(ry))
    W, H = int(np.ceil(rw)), int(np.ceil(rh))
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = PAD_RGB
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + W, img.shape[1]), min(y0 + H, img.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return canvas, x0, y0


def draw_full(img, row, out_path):
    """원본 전체 + 박스. 원해상도 그대로 나가야 하므로 Bbox 를 명시한다."""
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_axis_off()
    x1, y1, x2, y2 = row["box"]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                           edgecolor=C_BOX, linewidth=2.0,
                           path_effects=[pe.withStroke(linewidth=3.6,
                                                       foreground=C_TEXT_BG)]))
    stroked(ax, 8, 8, f"conf {row['conf']:.3f} | {row['stem']}", 13,
            ha="left", va="top")
    fig.savefig(out_path, dpi=DPI, facecolor="white",
                bbox_inches=Bbox.from_bounds(0, 0, w / DPI, h / DPI), pad_inches=0)
    plt.close(fig)


def draw_sheet(cells, sheet_idx, n_sheet, n_total, out_path):
    """5x4 contact sheet. 셀 축은 그림 좌표로 직접 잡아 종횡비를 정확히 고정한다."""
    fw = 16.0
    ml, mr, mt, mb = 0.35, 0.35, 0.78, 0.42
    gx, gy = 0.12, 0.16
    cw = (fw - ml - mr - gx * (NCOL - 1)) / NCOL
    ch = cw / CELL_ASPECT
    fh = mb + NROW * ch + gy * (NROW - 1) + mt

    fig = plt.figure(figsize=(fw, fh))
    fig.set_layout_engine("none")

    for k, c in enumerate(cells):
        r, col = divmod(k, NCOL)
        left = (ml + col * (cw + gx)) / fw
        bottom = (mb + (NROW - 1 - r) * (ch + gy)) / fh
        ax = fig.add_axes([left, bottom, cw / fw, ch / fh])
        ax.imshow(c["crop"], aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("0.55")
            sp.set_linewidth(0.8)
        bx1, by1, bx2, by2 = c["box_in_crop"]
        ax.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False,
                               edgecolor=C_BOX, linewidth=1.6,
                               path_effects=[pe.withStroke(linewidth=3.0,
                                                           foreground=C_TEXT_BG)]))
        stroked(ax, 0.02, 0.97, f"conf {c['conf']:.3f} | {c['stem']}", 8.5,
                ha="left", va="top", transform=ax.transAxes)

    lo = (sheet_idx - 1) * PER_SHEET + 1
    hi = lo + len(cells) - 1
    fig.text(0.5, 1 - 0.30 / fh,
             "R0 detections on real negatives (no pallet present) "
             f"— all are false positives | sheet {sheet_idx}/{n_sheet} "
             f"| cells {lo}-{hi} of {n_total} | crop = detection box x1.6, "
             "sorted by confidence (descending)",
             ha="center", va="top", fontsize=11)
    fig.text(0.01, 0.10 / fh, f"src: {SRC_CSV.relative_to(ROOT)}",
             ha="left", va="bottom", fontsize=8, color="gray")
    fig.text(0.99, 0.10 / fh,
             f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
             ha="right", va="bottom", fontsize=8, color="gray")

    fig.savefig(out_path, dpi=SHEET_DPI, facecolor="white",
                bbox_inches=Bbox.from_bounds(0, 0, fw, fh), pad_inches=0)
    plt.close(fig)


def main():
    rows = []
    for r in csv.DictReader(SRC_CSV.open()):
        ip = ROOT / r["image"]
        rows.append(dict(
            frame_id=r["frame_id"],
            stem=pathlib.Path(r["image"]).stem,
            image=r["image"],
            image_path=ip,
            exists=r["exists"] == "True",
            conf=float(r["conf"]),
            box=[float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
            n_cand=int(r["n_cand"]),
        ))
    missing = [r["image"] for r in rows if not r["image_path"].exists()]
    if missing:
        raise SystemExit(f"missing images: {missing[:5]} ({len(missing)})")

    rows.sort(key=lambda r: (-r["conf"], r["stem"]))
    OUT.mkdir(parents=True, exist_ok=True)
    FULL.mkdir(parents=True, exist_ok=True)

    n_total = len(rows)
    n_sheet = (n_total + PER_SHEET - 1) // PER_SHEET
    sheets, sizes = [], set()

    for si in range(n_sheet):
        chunk = rows[si * PER_SHEET:(si + 1) * PER_SHEET]
        cells, meta = [], []
        for j, r in enumerate(chunk):
            bgr = cv2.imread(str(r["image_path"]))
            if bgr is None:
                raise SystemExit(f"unreadable: {r['image']}")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            sizes.add((W, H))
            rect = crop_region(r["box"], W, H)
            crop, x0, y0 = letterboxed_crop(img, rect)
            x1, y1, x2, y2 = r["box"]
            cells.append(dict(crop=crop, conf=r["conf"], stem=r["stem"],
                              box_in_crop=(x1 - x0, y1 - y0, x2 - x0, y2 - y0)))
            full_name = f"{r['stem']}.png"
            draw_full(img, r, FULL / full_name)
            gcell = si * PER_SHEET + j + 1
            row_i, col_i = divmod(j, NCOL)
            meta.append(dict(
                cell=gcell, cell_in_sheet=j + 1, row=row_i, col=col_i,
                stem=r["stem"], frame_id=r["frame_id"], conf=r["conf"],
                box_xyxy=[round(v, 1) for v in r["box"]],
                n_cand=r["n_cand"],
                image_size=[W, H],
                source_image=r["image"],
                full_view=str((FULL / full_name).relative_to(ROOT)),
                crop_rect_xywh=[round(v, 1) for v in rect],
                label="",          # 사람이 이미지를 보고 채운다 (여기서 정하지 않는다)
            ))
        name = f"sheet_{si + 1:02d}.png"
        draw_sheet(cells, si + 1, n_sheet, n_total, OUT / name)
        sheets.append(dict(sheet=si + 1, file=str((OUT / name).relative_to(ROOT)),
                           grid=[NROW, NCOL], n_cells=len(chunk), cells=meta))
        print(f"sheet {si + 1}/{n_sheet}: {len(chunk)} cells -> {name}")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    manifest = dict(
        schema="false_positive_review_manifest_v1",
        generated_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        head=head,
        source_csv=str(SRC_CSV.relative_to(ROOT)),
        review_dir=str(OUT.relative_to(ROOT)),
        full_view_dir=str(FULL.relative_to(ROOT)),
        n_false_positives=n_total,
        order="confidence descending, tie broken by stem",
        crop_rule=f"detection box expanded x{EXPAND}, then padded to {CELL_ASPECT:.4f} "
                  "aspect, clipped to image, out-of-image area filled with flat gray",
        caption_rule="confidence and frame stem only; no interpretation is written "
                     "on the images. Categories are decided by the human reviewer.",
        label_field="each cell's 'label' is intentionally empty "
                    "— to be filled in after viewing",
        distinct_image_sizes=sorted([list(s) for s in sizes]),
        sheets=sheets,
    )
    (RESULTS / "FALSE_POSITIVE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(dict(n=n_total, sheets=n_sheet,
                          sizes=sorted(str(s) for s in sizes)), indent=2))


if __name__ == "__main__":
    main()
