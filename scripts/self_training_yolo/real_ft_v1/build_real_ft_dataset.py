"""REAL_FT_V1 데이터셋 빌드 — pseudo 슬롯을 real manual GT 로 바꾼다.

V1~V5 와 바뀌는 것은 **1440 개 비합성 슬롯의 라벨 출처 하나뿐**이다.
합성 replay 1440 장은 기존 V3 데이터셋에서 그대로 가져와 멤버십 동일성을 보장한다.

라벨 변환 규칙은 `REAL_FT_V1_METHOD_LOCK.json` 의 `label_conversion` 에 사전등록돼
있고 여기서 그대로 구현한다 — 결과를 보고 고치지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK = REPO_ROOT / "data/pallet/results/paper_real_ft_v1/REAL_FT_V1_METHOD_LOCK.json"
LIVE_GT = REPO_ROOT / "challenge/data/01_real/live_capture_gt"
V3_REFERENCE = (REPO_ROOT / "challenge/yolo_pose_one_model/datasets"
                / "paper_selftrain_v3/V3B_TRUE_IGNORE_AMBIG")
R0_DATASET = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
OUT_ROOT = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_real_ft_v1"

# 이미지가 어노테이션 폴더에 같이 있지 않은 세션
EXTERNAL_IMAGES = {
    "capture_20260902_kimjihoon_manual_gt":
        REPO_ROOT / "challenge/data/01_real/_live_captures/handheld_20260902"
                    "/sessions/capture_20260902_kimjihoon/rgb",
}

SLOTS = 1440


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_image(session_dir: Path, stem: str) -> Path | None:
    for parent in (session_dir, EXTERNAL_IMAGES.get(session_dir.name)):
        if parent is None:
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def to_yolo_label(payload: dict) -> str | None:
    """사전등록된 label_conversion 규칙 그대로."""
    obj = payload["objects"][0]
    width = float(payload["camera_data"]["width"])
    height = float(payload["camera_data"]["height"])
    points = obj["keypoint_annotations"]
    if len(points) != 9:
        return None

    coords = np.zeros((9, 2), dtype=float)
    vis = np.zeros(9, dtype=int)
    for i, point in enumerate(points):
        xy = point.get("xy")
        if xy is None:
            continue                                   # v=0, 좌표 0
        x, y = float(xy[0]), float(xy[1])
        coords[i] = (x / width, y / height)
        # 화면 밖 코너는 v=0 — 합성 규약에 [0,1] 밖 keypoint 가 0 개이고
        # belief head 가 격자 밖에 peak 를 못 찍기 때문이다.
        inside = 0.0 <= x < width and 0.0 <= y < height
        vis[i] = 2 if inside else 0

    visible = vis[:8] > 0
    if visible.sum() < 4:
        return None
    box_pts = np.clip(coords[:8][visible], 0.0, 1.0)
    x0, y0 = box_pts.min(axis=0)
    x1, y1 = box_pts.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None

    tokens = ["0", f"{(x0 + x1) / 2:.6f}", f"{(y0 + y1) / 2:.6f}",
              f"{bw:.6f}", f"{bh:.6f}"]
    for i in range(9):
        cx, cy = (coords[i] if vis[i] > 0 else (0.0, 0.0))
        tokens += [f"{cx:.6f}", f"{cy:.6f}", str(int(vis[i]))]
    return " ".join(tokens)


def collect_real_frames(exclude: set[str]) -> list[dict]:
    frames = []
    for session in sorted(d for d in LIVE_GT.iterdir() if d.is_dir()):
        if session.name.startswith("_"):
            continue
        for annotation in sorted(session.glob("*.json")):
            key = f"{session.name}/{annotation.stem}"
            if key in exclude:
                continue
            image = find_image(session, annotation.stem)
            if image is None:
                continue
            payload = json.loads(annotation.read_text())
            line = to_yolo_label(payload)
            if line is None:
                continue
            frames.append({"key": key, "session": session.name,
                           "stem": annotation.stem, "image": image, "label": line})
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-file", type=Path, default=None,
                        help="한 줄에 'session/stem' — 3d-expert 판정으로 배제할 프레임")
    parser.add_argument("--arm", default="REAL_FT")
    args = parser.parse_args()

    exclude: set[str] = set()
    if args.exclude_file and args.exclude_file.exists():
        exclude = {ln.strip() for ln in args.exclude_file.read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")}

    frames = collect_real_frames(exclude)
    if not frames:
        print("ABORT: 사용 가능한 real 프레임이 0 개다", file=sys.stderr)
        return 1

    # P2 — replay 1440 장을 V3 데이터셋에서 그대로 가져온다
    reference_images = sorted(
        p for p in (V3_REFERENCE / "images" / "train").iterdir()
        if p.name.startswith("replay__"))
    if len(reference_images) != SLOTS:
        print(f"ABORT P2: replay {len(reference_images)} != {SLOTS}", file=sys.stderr)
        return 1

    dataset = OUT_ROOT / args.arm
    images_dir, labels_dir = dataset / "images" / "train", dataset / "labels" / "train"
    for directory in (images_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    replay_names = []
    for source in reference_images:
        name = source.name
        link = images_dir / name
        if not link.exists():
            link.symlink_to(source.resolve())
        label = (V3_REFERENCE / "labels" / "train" / name).with_suffix(".txt")
        (labels_dir / name).with_suffix(".txt").write_text(label.read_text())
        replay_names.append(name)

    real_names = []
    for frame in frames:
        name = f"real__{frame['session']}__{frame['stem']}{frame['image'].suffix}"
        link = images_dir / name
        if not link.exists():
            link.symlink_to(frame["image"].resolve())
        (labels_dir / name).with_suffix(".txt").write_text(frame["label"] + "\n")
        real_names.append(name)

    # 노출 슬롯: real 을 1440 까지 결정적 round-robin 으로 채운다.
    # ⚠️ resolve() 하지 않는다.  ultralytics 는 im_file 의 `/images/` 를 `/labels/` 로
    # 바꿔 라벨을 찾으므로, 심링크를 풀면 원본 촬영 폴더를 보게 되고 라벨을 못 찾아
    # 전 프레임이 조용히 background 로 들어간다 (실제로 1440/1440 이 그렇게 됐다).
    exposures = [images_dir / replay_names[i] for i in range(SLOTS)]
    exposures += [images_dir / real_names[i % len(real_names)] for i in range(SLOTS)]
    (dataset / "train.txt").write_text(
        "\n".join(str(p) for p in exposures) + "\n")

    (dataset / "data.yaml").write_text(
        f"path: {dataset}\n"
        "train: train.txt\n"
        "val: train.txt\n"
        "kpt_shape: [9, 3]\n"
        "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
        "names:\n  0: item\n")

    membership = sha256_bytes("\n".join(sorted(f["key"] for f in frames)).encode())
    replay_membership = sha256_bytes("\n".join(sorted(replay_names)).encode())
    report = {
        "arm": args.arm,
        "real_frames_accepted": len(frames),
        "real_frames_excluded": len(exclude),
        "real_membership_sha256": membership,
        "replay_membership_sha256": replay_membership,
        "exposures_total": len(exposures),
        "exposures_real": SLOTS,
        "exposures_replay": SLOTS,
        "repeats_per_real_frame": round(SLOTS / len(frames), 3),
        "sessions": {s: sum(1 for f in frames if f["session"] == s)
                     for s in sorted({f["session"] for f in frames})},
    }
    (dataset / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
