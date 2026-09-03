"""두 세션의 contact sheet 를 나란히 놓고 같은 장소인지만 사람이 판정한다.

    python3 scripts/self_training_yolo/site_audit/review_site_groups.py \
        --output-dir data/pallet/results/site_environment_audit_v1

    --smoke N   앞의 N 쌍만 (도구 점검용, 별도 파일에 저장)

키
    1  같은 site, 비슷한 viewpoint
    2  같은 site, 다른 viewpoint
    3  다른 site
    U  모르겠음
    ←  이전 쌍          Q  저장하고 종료

검토 단위는 이미지가 아니라 **세션**이다(§13).  자동 유사도가 높은 쌍부터 보여준다.

화면에 모델 성능 · keypoint 오차 · pose 오차 · self-training 결과를 표시하지
않는다(§7).  장소 판정에 필요한 것만 보여준다.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"

CHOICES = {
    "1": "SAME_SITE_SIMILAR_VIEWPOINT",
    "2": "SAME_SITE_DIFFERENT_VIEWPOINT",
    "3": "DIFFERENT_SITE",
    "u": "UNCLEAR",
}


class Review:
    def __init__(self, root, pairs, sheets_dir, labels_path):
        self.root = root
        self.pairs = pairs
        self.sheets_dir = sheets_dir
        self.labels_path = labels_path
        self.labels = {}
        if labels_path.exists():
            self.labels = json.loads(labels_path.read_text()).get("labels", {})
        self.index = 0
        while self.index < len(pairs) and self.key(self.index) in self.labels:
            self.index += 1

        root.title("Same physical site?")
        root.configure(bg="#111")
        self.header = tk.Label(root, bg="#111", fg="#eee", justify="left",
                               font=("TkFixedFont", 11))
        self.header.pack(anchor="w", padx=10, pady=(8, 4))
        strip = tk.Frame(root, bg="#111")
        strip.pack()
        self.left = tk.Label(strip, bg="#111")
        self.left.pack(side="left", padx=4)
        self.right = tk.Label(strip, bg="#111")
        self.right.pack(side="left", padx=4)
        self.footer = tk.Label(
            root, bg="#111", fg="#9ad", justify="left", font=("TkFixedFont", 11),
            text="1 same site / similar viewpoint    2 same site / different viewpoint"
                 "    3 different site    U unclear        <- back    Q save and quit")
        self.footer.pack(anchor="w", padx=10, pady=(4, 8))

        root.bind("<Key>", self.on_key)
        self.show()

    def key(self, index):
        pair = self.pairs[index]
        return f"{pair['recording_a']}|{pair['recording_b']}"

    def sheet_for(self, recording_id):
        matches = sorted(self.sheets_dir.glob(f"{recording_id}__*.jpg"))
        return matches[0] if matches else None

    def show(self):
        if self.index >= len(self.pairs):
            self.save()
            self.header.config(text="검토 완료. Q 로 종료.")
            return
        pair = self.pairs[self.index]
        done = len(self.labels)
        self.header.config(text=(
            f"[{self.index + 1}/{len(self.pairs)}]  labelled {done}\n"
            f"A  {pair['recording_a']}  {pair['session_a']}\n"
            f"B  {pair['recording_b']}  {pair['session_b']}\n"
            f"background match inliers {pair['geometric_match_inliers']}   "
            f"lighting {pair['lighting_a']} / {pair['lighting_b']}"))
        for widget, rid in ((self.left, pair["recording_a"]),
                            (self.right, pair["recording_b"])):
            path = self.sheet_for(rid)
            if path is None:
                widget.config(image="", text=f"{rid} sheet 없음", fg="#f66")
                continue
            image = Image.open(path)
            scale = min(660 / image.width, 760 / image.height)
            image = image.resize((int(image.width * scale), int(image.height * scale)))
            photo = ImageTk.PhotoImage(image)
            widget.config(image=photo, text="")
            widget.image = photo

    def on_key(self, event):
        char = event.keysym.lower()
        if char == "q":
            self.save()
            self.root.destroy()
            return
        if char == "left":
            self.index = max(0, self.index - 1)
            self.show()
            return
        if char in CHOICES and self.index < len(self.pairs):
            pair = self.pairs[self.index]
            self.labels[self.key(self.index)] = {
                "verdict": CHOICES[char],
                "recording_a": pair["recording_a"],
                "recording_b": pair["recording_b"],
                "session_a": pair["session_a"],
                "session_b": pair["session_b"],
                "decided_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.index += 1
            self.save()
            self.show()

    def save(self):
        self.labels_path.write_text(json.dumps({
            "schema_version": "site_review_labels_v1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "n_pairs_offered": len(self.pairs),
            "n_labelled": len(self.labels),
            "model_results_shown": False,
            "labels": self.labels,
        }, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    proposals = json.loads((out_dir / "PROPOSED_SITE_GROUPS.json").read_text())
    pairs = (proposals["likely_same_site_pairs"]
             + proposals["possible_same_site_pairs"])
    pairs.sort(key=lambda p: -p["geometric_match_inliers"])
    labels_name = "SITE_REVIEW_LABELS.json"
    if args.smoke:
        pairs = pairs[:args.smoke]
        labels_name = "SITE_REVIEW_LABELS_SMOKE.json"

    if not pairs:
        print("검토할 쌍이 없다")
        return 1
    print(f"쌍 {len(pairs)} 개 — 유사도 높은 순")

    root = tk.Tk()
    Review(root, pairs, out_dir / "contact_sheets", out_dir / labels_name)
    root.mainloop()
    print(f"저장: {(out_dir / labels_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
