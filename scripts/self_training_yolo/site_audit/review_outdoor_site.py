"""outdoor capturepallet 계열 recording 에 사람이 직접 site 를 부여한다.

    python3 scripts/self_training_yolo/site_audit/review_outdoor_site.py \
        --output-dir data/pallet/results/site_environment_audit_v1

왼쪽에 recording 썸네일 격자, 오른쪽에 선택된 recording 의 contact sheet 전체.

키
    ↑ ↓ ← →   선택 이동
    A B C D   SITE_A / SITE_B / SITE_C / SITE_D 부여
    U         UNCLEAR
    0         라벨 지우기
    Q         저장하고 종료

쌍을 45 번 누르게 하지 않는다(§3) — recording 마다 site 를 한 번 부여한다.
capturepallet02 를 기준으로 "02 와 같은가" 를 묻지 않는다.  물리 장소 자체를
판단하고, 같다고 보이는 것끼리 같은 글자를 준다.

모델 결과·keypoint 오차·pose 오차·self-training 결과는 표시하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"
OUTDOOR_ROOT = REPO_ROOT / "data/pallet/raw_data/outside"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

REVIEW_ORDER = ["capturepallet01", "capturepallet02", "capturepallet03",
                "capturepallet04", "capturepallet05", "capturepallet06",
                "capturepallet07", "capturepallet08", "capturepallet09",
                "capturepallet10", "capturepallet11", "capturepalletcad"]

CHOICES = {"a": "SITE_A", "b": "SITE_B", "c": "SITE_C", "d": "SITE_D",
           "u": "UNCLEAR"}
N_FRAMES = 16


def build_sheet(image_dir: Path, caption: list[str], out_path: Path) -> None:
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        return
    positions = (np.linspace(0, len(paths) - 1, min(N_FRAMES, len(paths)))
                 .round().astype(int))
    chosen = [paths[i] for i in positions]
    cell_w, cell_h, columns = 240, 180, 4
    rows = (len(chosen) + columns - 1) // columns
    header = 20 * len(caption) + 10
    sheet = np.full((header + rows * cell_h, columns * cell_w, 3), 24, np.uint8)
    for index, path in enumerate(chosen):
        image = cv2.imread(str(path))
        if image is None:
            continue
        y = header + (index // columns) * cell_h
        x = (index % columns) * cell_w
        sheet[y:y + cell_h, x:x + cell_w] = cv2.resize(image, (cell_w, cell_h))
    for index, line in enumerate(caption):
        cv2.putText(sheet, line, (8, 18 + index * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


class Review:
    def __init__(self, root, units, labels_path):
        self.root = root
        self.units = units
        self.labels_path = labels_path
        self.labels = {}
        if labels_path.exists():
            self.labels = json.loads(labels_path.read_text()).get("labels", {})
        self.index = 0

        root.title("Which physical site is this recording?")
        root.configure(bg="#111")
        self.header = tk.Label(root, bg="#111", fg="#eee", justify="left",
                               font=("TkFixedFont", 12))
        self.header.pack(anchor="w", padx=10, pady=(8, 4))
        body = tk.Frame(root, bg="#111")
        body.pack()
        self.grid_frame = tk.Frame(body, bg="#111")
        self.grid_frame.pack(side="left", padx=6, anchor="n")
        self.detail = tk.Label(body, bg="#111")
        self.detail.pack(side="left", padx=6)
        tk.Label(root, bg="#111", fg="#9ad", justify="left", font=("TkFixedFont", 11),
                 text="arrows move    A B C D assign site    U unclear    "
                      "0 clear    Q save and quit").pack(anchor="w", padx=10, pady=(4, 8))

        self.cells = []
        for position, unit in enumerate(units):
            cell = tk.Label(self.grid_frame, bg="#111", bd=3, relief="flat")
            cell.grid(row=position // 3, column=position % 3, padx=3, pady=3)
            self.cells.append(cell)
            if unit["sheet"] is not None:
                image = Image.open(unit["sheet"])
                image.thumbnail((190, 150))
                photo = ImageTk.PhotoImage(image)
                cell.config(image=photo)
                cell.image = photo
            else:
                cell.config(text=f"{unit['name']}\nno raw frames",
                            fg="#f66", width=24, height=8)
        root.bind("<Key>", self.on_key)
        self.show()

    def show(self):
        unit = self.units[self.index]
        assigned = sum(1 for u in self.units if u["name"] in self.labels)
        summary = "  ".join(
            f"{u['name'][-3:]}={self.labels.get(u['name'], {}).get('site', '·')}"
            for u in self.units)
        self.header.config(text=(
            f"[{self.index + 1}/{len(self.units)}]  assigned {assigned}/{len(self.units)}\n"
            f"{unit['name']}   frames {unit['frames']}   "
            f"current label {self.labels.get(unit['name'], {}).get('site', '(none)')}\n"
            f"{summary}"))
        for position, cell in enumerate(self.cells):
            colour = "#4af" if position == self.index else "#111"
            cell.config(bd=3, relief="solid", highlightbackground=colour,
                        bg=colour if position == self.index else "#111")
        if unit["sheet"] is not None:
            image = Image.open(unit["sheet"])
            scale = min(820 / image.width, 720 / image.height)
            image = image.resize((int(image.width * scale), int(image.height * scale)))
            photo = ImageTk.PhotoImage(image)
            self.detail.config(image=photo, text="")
            self.detail.image = photo
        else:
            self.detail.config(image="", text=f"{unit['name']}\nno raw frames — skip",
                               fg="#f66", font=("TkFixedFont", 14))

    def on_key(self, event):
        char = event.keysym.lower()
        if char == "q":
            self.save()
            self.root.destroy()
            return
        if char in ("right", "down"):
            self.index = (self.index + 1) % len(self.units)
        elif char in ("left", "up"):
            self.index = (self.index - 1) % len(self.units)
        elif char in CHOICES:
            unit = self.units[self.index]
            if unit["sheet"] is None:
                return
            self.labels[unit["name"]] = {
                "site": CHOICES[char],
                "recording_id": unit["recording_id"],
                "frames": unit["frames"],
                "decided_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.save()
            self.index = min(self.index + 1, len(self.units) - 1)
        elif char == "0":
            self.labels.pop(self.units[self.index]["name"], None)
            self.save()
        self.show()

    def save(self):
        self.labels_path.write_text(json.dumps({
            "schema_version": "outdoor_site_review_v1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "question": "which physical site is this recording",
            "reference_free": "the user judges the place itself, not similarity to capturepallet02",
            "model_results_shown": False,
            "n_recordings": len(self.units),
            "n_labelled": len(self.labels),
            "labels": self.labels,
        }, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    sheets_dir = out_dir / "contact_sheets_outdoor"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    groups = json.loads((out_dir / "SOURCE_RECORDING_GROUPS.json").read_text())["groups"]
    recording_of = {}
    for group in groups:
        for session in group["sessions"]:
            recording_of[session["session_key"]] = group["recording_id"]

    units = []
    for name in REVIEW_ORDER:
        image_dir = OUTDOOR_ROOT / name / "rgb"
        frames = (len([p for p in image_dir.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES])
                  if image_dir.is_dir() else 0)
        sheet = sheets_dir / f"{name}.jpg"
        if frames and not sheet.exists():
            build_sheet(image_dir, [f"{name}   frames {frames}"], sheet)
        units.append({
            "name": name,
            "frames": frames,
            "recording_id": recording_of.get(f"data/pallet/raw_data/outside/{name}"),
            "sheet": sheet if frames and sheet.exists() else None,
        })

    missing = [u["name"] for u in units if u["sheet"] is None]
    print(f"검토 대상 {sum(1 for u in units if u['sheet'])} / {len(units)}")
    if missing:
        print(f"  raw 없음 — 표시만 하고 건너뜀: {', '.join(missing)}")

    root = tk.Tk()
    Review(root, units, out_dir / "OUTDOOR_SITE_REVIEW.json")
    root.mainloop()
    print(f"저장: {out_dir / 'OUTDOOR_SITE_REVIEW.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
