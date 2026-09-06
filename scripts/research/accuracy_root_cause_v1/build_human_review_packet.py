"""GT semantics 사람 리뷰 패킷을 생성한다 (읽기 전용 — GT JSON 미수정).

목적 : 리뷰어가 현재 모델을 모르는 상태에서 두 가지를 판정할 수 있게 한다.
       (가) W/D 축 배정 두 가설 중 어느 쪽이 사진과 맞는가
       (나) 저장된 코너 좌표가 물리 코너인가 추측인가
지표 : anchoring 단서가 이미지에 하나도 남지 않는가 (모델 예측 0 · 재투영 수치 0 ·
       채택 가설 표시 0).

프레임마다 아래를 만든다.

    01_raw.png        원본 그대로 (바이트 복사)
    02_gt_only.png    저장된 keypoint 9개. 클릭=채운 원 / 외삽=빈 사각 (모양으로 구분)
    03_geometry_A.png 두 W/D 가설 중 하나의 cuboid (A/B 배정은 프레임별 무작위)
    03_geometry_B.png 나머지 하나

cuboid pose 는 **클릭 코너만으로** PnP 를 다시 푼다. 외삽 코너는 채택 pose 의
투영이라 그걸 쓰면 채택 가설이 항상 이긴다(순환).

phase 2 (`--phase 2`) 는 같은 규약으로 실패농축 표본을 **덧붙인다**. 기존 42장의
폴더·정답키·리뷰 시트를 지우지 않는다 (정답키와 시트는 append).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import shutil
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import matplotlib.patheffects as pe

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from challenge.data_paths import EVAL_CANONICAL

sys.path.insert(0, str(ROOT / "scripts" / "annotate"))
from annotate_pnp import make_pallet_keypoints_3d_diagram as kp3d

ASSETS = pathlib.Path.home() / ".claude/agents/viz-expert/assets"
plt.style.use(ASSETS / "analysis.mplstyle")
sys.path.insert(0, str(ASSETS))
from palette import color_for, CV  # noqa: E402

RESULTS = ROOT / "data/pallet/results/accuracy_root_cause_v1"
REVIEW = ROOT / "_docs/audits/accuracy_root_cause_v1/human_review"
FRAME_LIST = RESULTS / "GT_REVIEW_FRAME_LIST.csv"

SEED = 20260906          # A/B 무작위 배정 + repeat 표본 추출 seed
N_REPEAT = 20            # 2회 리뷰 대상 프레임 수 (plan §repeatability)
DPI = 100                # figsize x DPI = 원본 픽셀 (해상도 유지)

# phase 별 설정. phase 1 은 기존 42장 (기본값 — 인자 없이 돌리면 그대로 재현된다).
PHASES = {
    1: dict(frame_list="GT_REVIEW_FRAME_LIST.csv",
            manifest="HUMAN_REVIEW_MANIFEST.json",
            n_repeat=N_REPEAT, append=False),
    2: dict(frame_list="GT_REVIEW_FRAME_LIST_PHASE2.csv",
            manifest="HUMAN_REVIEW_MANIFEST_PHASE2.json",
            n_repeat=0, append=True),
}

# 정답키 열 순서는 phase 1 에서 굳었다. append 할 때 헤더와 대조해 어긋나면 멈춘다.
ANSWER_FIELDS = ["frame_id", "folder", "frame", "role", "n_click",
                 "hypothesis_A", "hypothesis_B", "stored_width", "stored_depth",
                 "height", "clickonly_rms_stored", "clickonly_rms_swapped"]

C_CLICK = CV["keypoint"]                 # #E69F00
C_EXTRAP = color_for("extrapolated")     # #009E73
C_CUBOID = CV["skeleton"]                # #56B4E9 — A/B 동일 (색이 단서가 되면 안 된다)
C_TEXT = CV["text"]
C_TEXT_BG = CV["text_bg"]

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def is_click(p) -> bool:
    """마우스 클릭은 정수 픽셀로 저장된다. PnP 투영은 거의 확실히 실수다."""
    return (float(p[0]).is_integer() and float(p[1]).is_integer()
            and not (p[0] == -1.0 and p[1] == -1.0))


def solve_from_clicks(pts2d, pts3d, K, idx):
    """클릭 부분집합만으로 PnP. (rvec, tvec, rms) 또는 None."""
    if len(idx) < 4:
        return None
    op = np.asarray([pts3d[i] for i in idx], np.float64)
    ip = np.asarray([pts2d[i] for i in idx], np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(op, ip, K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok:
        return None
    proj, _ = cv2.projectPoints(op, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip) ** 2, axis=1))))
    return rvec, tvec, rms


def new_canvas(img):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_axis_off()
    return fig, ax, w, h


def stroked(ax, x, y, s, size, ha="left", va="top"):
    """야간 프레임에서도 읽히도록 외곽선을 준다 (강조가 아니라 가독성 장치)."""
    ax.text(x, y, s, color=C_TEXT, fontsize=size, ha=ha, va=va,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=C_TEXT_BG)])


def save_exact(fig, path, w, h):
    fig.savefig(path, dpi=DPI, facecolor="white",
                bbox_inches=Bbox.from_bounds(0, 0, w / DPI, h / DPI), pad_inches=0)
    plt.close(fig)


# 위 코너(0,1,4,5)는 라벨을 위로, 아래 코너(2,3,6,7)는 아래로 뺀다.
# 같은 오프셋을 쓰면 위/아래 코너 쌍이 화면에서 겹쳐 번호를 못 읽는다.
_LABEL_ABOVE = {0, 1, 4, 5, 8}


def draw_gt_only(img, kps, clicks, frame_id, out):
    fig, ax, w, h = new_canvas(img)
    for i, p in enumerate(kps):
        if p is None or (p[0] == -1 and p[1] == -1):
            continue
        if i in clicks:
            ax.plot(p[0], p[1], marker="o", ms=8, mfc=C_CLICK, mec=C_TEXT_BG,
                    mew=1.0, ls="none")
        else:
            ax.plot(p[0], p[1], marker="s", ms=9, mfc="none", mec=C_EXTRAP,
                    mew=2.0, ls="none")
        above = i in _LABEL_ABOVE
        tx = min(max(p[0] + 8, 10), w - 10)
        ty = min(max(p[1] + (-10 if above else 10), 12), h - 12)
        stroked(ax, tx, ty, str(i), 13, ha="left",
                va="bottom" if above else "top")
    handles = [
        Line2D([], [], marker="o", ms=8, mfc=C_CLICK, mec=C_TEXT_BG, ls="none",
               label="clicked (integer coords)"),
        Line2D([], [], marker="s", ms=9, mfc="none", mec=C_EXTRAP, mew=2.0, ls="none",
               label="extrapolated (non-integer)"),
    ]
    # loc="best" 로 두면 matplotlib 이 실제 점 분포를 보고 겹침이 가장 적은 모서리를 고른다.
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.85)
    stroked(ax, 6, 6, f"02 GT keypoints as stored | {frame_id}", 9)
    save_exact(fig, out, w, h)


def draw_geometry(img, corners2d, label, frame_id, out):
    fig, ax, w, h = new_canvas(img)
    for a, b in EDGES:
        ax.plot([corners2d[a][0], corners2d[b][0]],
                [corners2d[a][1], corners2d[b][1]],
                color=C_CUBOID, lw=1.8, solid_capstyle="round",
                path_effects=[pe.withStroke(linewidth=3.4, foreground=C_TEXT_BG)])
    stroked(ax, 6, 6, f"03 Hypothesis {label} | {frame_id}", 9)
    save_exact(fig, out, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=sorted(PHASES))
    ap.add_argument("--force", action="store_true",
                    help="덮어쓰기 모드에서 기존 CSV 를 지우는 것을 명시적으로 허용")
    args = ap.parse_args()
    cfg = PHASES[args.phase]

    frame_list = RESULTS / cfg["frame_list"]
    rows = list(csv.DictReader(frame_list.open()))
    REVIEW.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "gt_human_review_manifest_v1",
        "phase": args.phase,
        "generated_utc": None,
        "head": None,
        "source_frame_list": str(frame_list.relative_to(ROOT)),
        "ab_assignment_seed": SEED,
        "ab_assignment_rule": "per-frame RNG seeded with md5(f'{SEED}:{frame_id}')",
        "answer_key": "data/pallet/results/accuracy_root_cause_v1/_ANSWER_KEY.csv",
        "review_dir": str(REVIEW.relative_to(ROOT)),
        "repeat_seed": SEED,
        "repeat_frames": [],
        "frames": [],
    }
    answer_rows = []

    for r in rows:
        folder, frame = r["folder"], r["frame"]
        frame_id = f"{folder}__{frame}"
        jp = ROOT / EVAL_CANONICAL[folder] / f"{frame}.json"
        ip = jp.with_suffix(".png")
        o = json.loads(jp.read_text())
        ob = o["objects"][0]
        ci = o["camera_data"]["intrinsics"]
        K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]],
                     np.float64)
        mk = ob["manual_kps"]
        kps = [None if p is None else [float(p[0]), float(p[1])] for p in mk]
        clicks = [i for i in range(8)
                  if kps[i] is not None and is_click(kps[i])]
        dm = ob["dimensions_m"]
        wd, dp, ht = float(dm["width"]), float(dm["depth"]), float(dm["height"])

        d = REVIEW / frame_id
        d.mkdir(parents=True, exist_ok=True)
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        shutil.copyfile(ip, d / "01_raw.png")
        draw_gt_only(img, kps, clicks, frame_id, d / "02_gt_only.png")
        files = ["01_raw.png", "02_gt_only.png"]

        # 두 가설을 클릭 코너만으로 각각 다시 푼다.
        variants = {
            "stored": kp3d(width=wd, depth=dp, height=ht)[:8],
            "swapped": kp3d(width=dp, depth=wd, height=ht)[:8],
        }
        pts2d = np.asarray([[np.nan, np.nan] if p is None else p for p in kps[:8]],
                           np.float64)
        solved, skip = {}, None
        for tag, m3d in variants.items():
            s = solve_from_clicks(pts2d, m3d, K, clicks)
            if s is None:
                skip = (f"PnP unsolvable for hypothesis '{tag}' "
                        f"(n_click={len(clicks)})")
                break
            solved[tag] = s

        geometry_generated = skip is None
        if geometry_generated:
            rng = random.Random(
                int(hashlib.md5(f"{SEED}:{frame_id}".encode()).hexdigest()[:8], 16))
            order = ["stored", "swapped"]
            rng.shuffle(order)
            ab = {"A": order[0], "B": order[1]}
            for lb in ("A", "B"):
                tag = ab[lb]
                rvec, tvec, _ = solved[tag]
                proj, _ = cv2.projectPoints(variants[tag], rvec, tvec, K, None)
                draw_geometry(img, proj.reshape(-1, 2), lb, frame_id,
                              d / f"03_geometry_{lb}.png")
                files.append(f"03_geometry_{lb}.png")
            answer_rows.append(dict(
                frame_id=frame_id, folder=folder, frame=frame, role=r["role"],
                n_click=len(clicks),
                hypothesis_A=ab["A"], hypothesis_B=ab["B"],
                stored_width=wd, stored_depth=dp, height=ht,
                clickonly_rms_stored=round(solved["stored"][2], 4),
                clickonly_rms_swapped=round(solved["swapped"][2], 4),
            ))

        # phase 1 은 wd_margin_px, phase 2 는 note 를 실어 온다.
        extra = {}
        if r.get("wd_margin_px"):
            extra["wd_margin_px"] = float(r["wd_margin_px"])
        if r.get("note"):
            extra["note"] = r["note"]

        manifest["frames"].append(dict(
            frame_id=frame_id, folder=folder, frame=frame, role=r["role"],
            **extra,
            n_click=len(clicks), click_indices=clicks,
            image_size=[W, H],
            dir=str((d).relative_to(ROOT)),
            files=files,
            geometry_generated=geometry_generated,
            geometry_skip_reason=skip,
        ))

    # ── repeatability 표본 (plan §repeatability, 20장) ────────────────────
    ids = [f["frame_id"] for f in manifest["frames"]]
    rep = sorted(random.Random(SEED).sample(ids, cfg["n_repeat"]))
    manifest["repeat_frames"] = rep

    # ── 리뷰 시트 (빈 양식) ────────────────────────────────────────────────
    corner_fields = ["frame_id", "corner_id", "directly_visible",
                     "occluded_but_geometrically_inferable", "outside_image",
                     "physical_surface_corner", "virtual_cuboid_corner", "ambiguous",
                     "reviewer_xy_u", "reviewer_xy_v", "semantic_role_confident", "note"]
    frame_fields = ["frame_id", "hypothesis_A_better", "hypothesis_B_better",
                    "cannot_tell", "confidence_1to5", "note"]
    sheet_ids = ids + [f"{i}_rep2" for i in rep]

    # append 모드에서는 기존 행을 절대 지우지 않는다. 헤더가 어긋나면 쓰지 않고 멈춘다.
    # 덮어쓰기 모드(phase 1)로 다시 돌리면 뒤에 append 된 phase 2 행이 날아가므로,
    # 파일이 이미 있으면 --force 없이는 멈춘다.
    def write_rows(path, fields, records):
        appending = cfg["append"] and path.exists()
        if not appending and path.exists() and not args.force:
            raise SystemExit(
                f"{path} already exists — overwriting would drop rows appended by a "
                f"later phase. Re-run with --force only if that is what you want.")
        if appending:
            have = next(csv.reader(path.open()))
            if have != fields:
                raise SystemExit(f"header mismatch in {path}: {have} != {fields}")
        with path.open("a" if appending else "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=fields)
            if not appending:
                w_.writeheader()
            w_.writerows(records)
        return appending

    write_rows(REVIEW / "REVIEW_FORM_corner.csv", corner_fields,
               [{"frame_id": fid, "corner_id": c} for fid in sheet_ids for c in range(9)])
    write_rows(REVIEW / "REVIEW_FORM_frame.csv", frame_fields,
               [{"frame_id": fid} for fid in sheet_ids])
    appended_key = write_rows(RESULTS / "_ANSWER_KEY.csv", ANSWER_FIELDS, answer_rows)
    manifest["answer_key_appended"] = appended_key

    import datetime
    import subprocess
    manifest["generated_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    manifest["head"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT, capture_output=True,
                                      text=True).stdout.strip()
    manifest["counts"] = {
        "frames": len(manifest["frames"]),
        "geometry_generated": sum(1 for f in manifest["frames"] if f["geometry_generated"]),
        "geometry_skipped": sum(1 for f in manifest["frames"] if not f["geometry_generated"]),
        "corner_form_rows": len(sheet_ids) * 9,
        "frame_form_rows": len(sheet_ids),
        "repeat_frames": len(rep),
    }
    (RESULTS / cfg["manifest"]).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))

    print(json.dumps(manifest["counts"], indent=2))
    print("answer key rows added:", len(answer_rows))


if __name__ == "__main__":
    main()
