"""세션 대표 프레임 · contact sheet · 배경 중심 시각 유사도 · site 후보를 만든다.

    python3 scripts/self_training_yolo/site_audit/build_visual_site_candidates.py \
        --output-dir data/pallet/results/site_environment_audit_v1

출력
    contact_sheets/<RECORDING>__<session>.jpg
    SESSION_VISUAL_SIMILARITY.csv
    PROPOSED_SITE_GROUPS.json

자동 단계는 **후보 생성까지**다(§6).  site_id 를 확정하지 않는다.

§9 팔레트 때문에 묶이는 것을 막는다 — 화면 중앙(대개 팔레트가 있는 곳)을 가리고
테두리 · 위쪽 배경 · 좌우 구조물에서만 특징을 뽑는다.  같은 팔레트를 여러 장소에서
찍었으므로 물체 외형으로 장소를 묶으면 안 된다.

§16 모델 예측 · GT 정확도 · 성능 지표는 읽지 않는다.  이 파일은 그런 경로를
열지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

N_REPRESENTATIVE = 16      # §7 균등 위치 12~20 장
MIN_IMAGES = 100           # 이보다 작은 recording 은 site 판정 단위로 삼지 않는다
MATCH_FRAMES = 4           # 쌍 비교에 쓸 프레임 수 (4x4 = 16 조합)
RATIO_TEST = 0.75
RANSAC_PX = 4.0
MIN_MATCHES_FOR_GEOMETRY = 12

# 등급 경계 — 결과를 보기 전에 고정한다.  자동 확정용이 아니라 사람에게 보여줄 순서용.
LIKELY_SAME_INLIERS = 30
POSSIBLE_SAME_INLIERS = 8

# 오버레이 산출물이 이름 규칙을 빠져나간 경우
OVERLAY_HINTS = ("_overlay", "overlay/", "gt_final_overlay", "gt_overlay",
                 "gt_final_isaac_overlay")


def background_mask(shape) -> np.ndarray:
    """중앙을 가리고 테두리·위쪽 배경만 남긴다 (§9)."""

    height, width = shape[:2]
    mask = np.full((height, width), 255, np.uint8)
    y0, y1 = int(height * 0.35), int(height * 0.95)
    x0, x1 = int(width * 0.18), int(width * 0.82)
    mask[y0:y1, x0:x1] = 0
    return mask


def representative_paths(image_dir: Path, count: int) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) <= count:
        return paths
    positions = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[i] for i in positions]


def describe(path: Path, sift):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    if max(image.shape) > 640:
        scale = 640.0 / max(image.shape)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return sift.detectAndCompute(image, background_mask(image.shape))


def colour_histogram(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    mask = background_mask(image.shape)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [24, 24], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def geometric_inliers(desc_a, kp_a, desc_b, kp_b, matcher) -> tuple[int, float]:
    """SIFT 매칭 뒤 RANSAC homography 로 기하 검증.  (inlier 수, inlier 비율)"""

    if desc_a is None or desc_b is None:
        return 0, 0.0
    if len(desc_a) < 2 or len(desc_b) < 2:
        return 0, 0.0
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO_TEST * n.distance]
    if len(good) < MIN_MATCHES_FOR_GEOMETRY:
        return 0, 0.0
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if inlier_mask is None:
        return 0, 0.0
    inliers = int(inlier_mask.sum())
    return inliers, inliers / len(good)


def contact_sheet(paths: list[Path], caption: list[str], out_path: Path) -> None:
    cell_w, cell_h, columns = 240, 180, 4
    rows = (len(paths) + columns - 1) // columns
    header = 20 * len(caption) + 10
    sheet = np.full((header + rows * cell_h, columns * cell_w, 3), 24, np.uint8)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        cell = cv2.resize(image, (cell_w, cell_h))
        y = header + (index // columns) * cell_h
        x = (index % columns) * cell_w
        sheet[y:y + cell_h, x:x + cell_w] = cell
        cv2.putText(sheet, path.stem[-10:], (x + 4, y + cell_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(sheet, path.stem[-10:], (x + 4, y + cell_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    for index, line in enumerate(caption):
        cv2.putText(sheet, line, (8, 18 + index * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def lighting_of(session_key: str) -> str:
    lowered = session_key.lower()
    if "night" in lowered:
        return "night"
    if "day" in lowered or "outside" in lowered:
        return "day"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--min-images", type=int, default=MIN_IMAGES)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    sheets_dir = out_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    groups = json.loads((out_dir / "SOURCE_RECORDING_GROUPS.json").read_text())
    inventory = {s["session_key"]: s for s in
                 json.loads((out_dir / "SESSION_INVENTORY.json").read_text())["sessions"]}

    units = []
    for group in groups["groups"]:
        if group["is_collection"] or group["n_unique_images"] < args.min_images:
            continue
        representative = group["representative"]
        if any(hint in representative for hint in OVERLAY_HINTS):
            continue          # 오버레이 산출물은 촬영본이 아니다
        session = inventory[representative]
        units.append({
            "recording_id": group["recording_id"],
            "session_key": representative,
            "image_dir": session["image_dir"],
            "n_images": group["n_unique_images"],
            "resolution": session["resolution"],
            "lighting": lighting_of(representative),
            "n_sessions_in_recording": group["n_sessions"],
        })
    print(f"site 판정 단위 {len(units)} 개 (>= {args.min_images} 장, collection·overlay 제외)")

    sift = cv2.SIFT_create(nfeatures=1200)
    matcher = cv2.BFMatcher()

    # ── 대표 프레임 · contact sheet · 기술자
    for unit in units:
        image_dir = REPO_ROOT / unit["image_dir"]
        paths = representative_paths(image_dir, N_REPRESENTATIVE)
        unit["representative_frames"] = [p.name for p in paths]
        contact_sheet(paths, [
            f"{unit['recording_id']}   {unit['session_key']}",
            f"recording sessions {unit['n_sessions_in_recording']}   "
            f"frames {unit['n_images']}   lighting {unit['lighting']}   "
            f"{unit['resolution']}",
        ], sheets_dir / f"{unit['recording_id']}__{Path(unit['session_key']).name}.jpg")

        chosen = paths[:: max(1, len(paths) // MATCH_FRAMES)][:MATCH_FRAMES]
        unit["_features"] = [describe(p, sift) for p in chosen]
        unit["_hists"] = [h for h in (colour_histogram(p) for p in chosen)
                          if h is not None]
        # §11 세션 내부 장면 전환 — 대표 프레임을 순서대로 이웃끼리 비교
        walk = [describe(p, sift) for p in paths]
        neighbour = []
        for i in range(len(walk) - 1):
            (kp_a, da), (kp_b, db) = walk[i], walk[i + 1]
            neighbour.append(geometric_inliers(da, kp_a, db, kp_b, matcher)[0])
        unit["internal_neighbour_inliers"] = neighbour
        unit["internal_min_inliers"] = min(neighbour) if neighbour else None
        unit["multi_site_suspected"] = bool(neighbour) and min(neighbour) == 0
        print(f"  {unit['recording_id']:8} {Path(unit['session_key']).name[:34]:34} "
              f"frames {unit['n_images']:6d}  "
              f"내부 최소 inlier {unit['internal_min_inliers']}")

    # ── 쌍별 유사도
    rows = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            a, b = units[i], units[j]
            best_inliers, best_ratio = 0, 0.0
            for kp_a, da in a["_features"]:
                for kp_b, db in b["_features"]:
                    inliers, ratio = geometric_inliers(da, kp_a, db, kp_b, matcher)
                    if inliers > best_inliers:
                        best_inliers, best_ratio = inliers, ratio
            hist = 0.0
            if a["_hists"] and b["_hists"]:
                hist = max(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
                           for ha in a["_hists"] for hb in b["_hists"])
            grade = ("LIKELY_SAME_SITE" if best_inliers >= LIKELY_SAME_INLIERS
                     else "POSSIBLE_SAME_SITE" if best_inliers >= POSSIBLE_SAME_INLIERS
                     else "LIKELY_DIFFERENT_SITE")
            rows.append({
                "recording_a": a["recording_id"],
                "recording_b": b["recording_id"],
                "session_a": a["session_key"],
                "session_b": b["session_key"],
                "geometric_match_inliers": best_inliers,
                "geometric_match_inlier_ratio": round(best_ratio, 4),
                "background_colour_similarity": round(float(hist), 4),
                "camera_intrinsics_same": (
                    inventory[a["session_key"]]["intrinsics_hash"] is not None
                    and inventory[a["session_key"]]["intrinsics_hash"]
                    == inventory[b["session_key"]]["intrinsics_hash"]),
                "resolution_same": a["resolution"] == b["resolution"],
                "lighting_a": a["lighting"],
                "lighting_b": b["lighting"],
                "provenance_relation": "SAME_SOURCE_RECORDING"
                if a["recording_id"] == b["recording_id"] else "DISTINCT_RECORDING",
                "automatic_grade": grade,
            })
        print(f"  pairwise {i + 1}/{len(units)}", flush=True)

    rows.sort(key=lambda r: -r["geometric_match_inliers"])
    with (out_dir / "SESSION_VISUAL_SIMILARITY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    proposals = {
        "schema_version": "proposed_site_groups_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "AUTO_GROUPING_IS_FINAL": False,
        "HUMAN_CONFIRMATION_REQUIRED": True,
        "background_masking": "central region removed so pallet appearance cannot drive grouping",
        "primary_signal": "SIFT + RANSAC homography inliers on background regions",
        "secondary_signal": "HSV histogram correlation, never primary because day/night shifts colour",
        "local_scene_encoder": "not used — no DINO/CLIP weights present locally and no download was made",
        "grade_thresholds": {"likely_same_inliers": LIKELY_SAME_INLIERS,
                             "possible_same_inliers": POSSIBLE_SAME_INLIERS,
                             "declared_before_results": True},
        "units": [{k: v for k, v in unit.items() if not k.startswith("_")}
                  for unit in units],
        "likely_same_site_pairs": [r for r in rows
                                   if r["automatic_grade"] == "LIKELY_SAME_SITE"],
        "possible_same_site_pairs": [r for r in rows
                                     if r["automatic_grade"] == "POSSIBLE_SAME_SITE"],
        "multi_site_suspected": [u["recording_id"] for u in units
                                 if u["multi_site_suspected"]],
    }
    (out_dir / "PROPOSED_SITE_GROUPS.json").write_text(
        json.dumps(proposals, indent=2) + "\n")

    print(f"\nlikely same site  {len(proposals['likely_same_site_pairs'])}")
    print(f"possible same site {len(proposals['possible_same_site_pairs'])}")
    print(f"multi-site 의심     {proposals['multi_site_suspected']}")
    print(f"contact sheets     {sheets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
