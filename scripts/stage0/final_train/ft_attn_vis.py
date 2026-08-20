"""모델이 어디를 보고 엣지와 코너를 찍는지 -- 프레임마다 한 장에 담는다.

두 가지는 성질이 다르므로 섞어 부르지 않는다.

    line branch    진짜 attention.  12 개 role query 가 f50 의 2500 개 위치를
                   가중합한다.  가중치 (12, 50, 50) 이 곧 "엣지 k 를 찍을 때
                   본 곳"이다.  `DirectHoughModel.descriptors()` 와 같은 경로를
                   재현하되 need_weights 만 켜서 꺼낸다 -- 모델 파일은 손대지
                   않는다.
    corner head    attention 이 아니다.  순수 conv DOPE stage 의 belief map
                   9 채널 (50, 50).  "어디를 보는가"가 아니라 "어디라고
                   답하는가"이므로 라벨을 그렇게 붙인다.

role k 가 어느 엣지인지는 `mh_cigm.EDGES` 로만 정한다.  그리기 편의로 쓰이는
`ft_overlays.EDGES` 는 같은 12 엣지의 **다른 순서**라, 그걸 쓰면 12 개 라벨이
전부 어긋난다.

선 좌표계는 canonical 50-grid: normal = (cos t, sin t), normal . x = rho.
이미지로 되돌릴 때 x 와 y 의 배율이 다르므로(squash), 선을 픽셀에서 직접 세우지
않고 50-grid 에서 두 점을 잡아 각각 변환한다.

그림만 뽑고 끝내지 않는다.  프레임마다 role 별 attention 질량이 GT 엣지 위에
얼마나 놓였는지를 재서, 마지막에 집계 그림과 판정 JSON 까지 같은 실행에서
쓴다.  PURPOSE 의 물음("국소 증거인가, 붕괴인가")에 그림이 아니라 수로 답한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate", "scripts/stage0/final_train"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2                                        # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
# 한글 라벨이 두부(□)로 나오면 그림이 읽히지 않는다.  이 머신에 실재하는 CJK
# 폰트로 고정한다.  숫자를 세로로 맞춰 읽는 타일 라벨은 고정폭이어야 하므로,
# monospace 쪽에는 한글이 있는 고정폭 글꼴을 따로 등록한다.
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
_MONO = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"
if os.path.exists(_MONO):
    import matplotlib.font_manager as _fm                   # noqa: E402
    _fm.fontManager.addfont(_MONO)
    matplotlib.rcParams["font.monospace"] = ["NanumGothicCoding", "DejaVu Sans Mono"]
import matplotlib.pyplot as plt                   # noqa: E402
import matplotlib.patheffects as pe               # noqa: E402
from matplotlib.gridspec import GridSpec          # noqa: E402
from matplotlib.lines import Line2D               # noqa: E402

import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mh_cigm as CG                              # noqa: E402
import mh_fusion as FU                            # noqa: E402
import line_feature_capacity_v2 as V2             # noqa: E402
import re_metrics as RM                           # noqa: E402
import annotate_pnp as APNP                       # noqa: E402
import ft_f0f3_eval as EV                         # noqa: E402
from mh_arms import DH                            # noqa: E402
from ft_attn_smoke import attention_weights       # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402

GRID, ROLES, TOKENS = 50, 12, 2500
BAND_CELL = 3.0            # "엣지 위" 로 셀 판정: GT 선분에서 3 cell 이내
CORNER_NAMES = ["0 near-TL", "1 near-TR", "2 near-BR", "3 near-BL",
                "4 far-TL", "5 far-TR", "6 far-BR", "7 far-BL", "8 centroid"]
DRAW_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def edge_family(edge):
    """near / far / depth.  depth 엣지는 저앙각에서 짧아지는 바로 그 축이다."""
    near = sum(1 for corner in edge if corner < 4)
    return "near" if near == 2 else ("far" if near == 0 else "depth")


FAMILY = [edge_family(e) for e in CG.EDGES]
FAMILY_COLOUR = {"near": "#4aa8ff", "far": "#c77dff", "depth": "#ff9a4d"}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ geometry
def segment_distance(points, a, b):
    """각 점에서 선분 ab 까지의 거리.  선분 밖 연장선은 증거가 아니므로 제외."""
    delta = b - a
    length2 = float(delta @ delta)
    if length2 < 1e-12:
        return np.linalg.norm(points - a, axis=-1)
    t = np.clip(((points - a) @ delta) / length2, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * delta), axis=-1)


def line_through_grid(theta_rad, rho, span=200.0):
    """canonical 50-grid 의 (theta, rho) -> 그 선 위의 먼 두 점."""
    normal = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    direction = np.array([-normal[1], normal[0]])
    base = rho * normal
    return base - span * direction, base + span * direction


def grid_to_pixel(points, width, height):
    return np.stack([points[..., 0] * width / GRID,
                     points[..., 1] * height / GRID], -1)


def cell_centres():
    axis = np.arange(GRID) + 0.5
    xx, yy = np.meshgrid(axis, axis)          # (row, col) -> x=col, y=row
    return np.stack([xx, yy], -1).reshape(-1, 2)


CELLS = cell_centres()


# ------------------------------------------------------------------- drawing
def backdrop(axis, image):
    grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2RGB)
    axis.imshow(grey, interpolation="nearest")
    axis.set_xticks([]); axis.set_yticks([])


def heat(axis, field, image, cmap):
    """(50,50) 을 원본 해상도로 펴서 덮는다.  squash 였으므로 되돌리기도 비등방.

    불투명도를 값에 비례시킨다.  일정 alpha 로 덮으면 히트맵의 바닥값이 사진을
    전부 가려, 모델이 본 곳이 배경의 어느 물체인지 알 수 없게 된다.
    """
    height, width = image.shape[:2]
    big = cv2.resize(field.astype(np.float32), (width, height),
                     interpolation=cv2.INTER_LINEAR)
    ceiling = float(big.max()) or 1.0
    axis.imshow(big, cmap=cmap, alpha=np.clip(big / ceiling, 0.0, 1.0) ** 0.7 * 0.88,
                interpolation="bilinear", vmin=0.0, vmax=ceiling)


def draw_cuboid(axis, pts, colour, style="-", width=1.6):
    ok = np.isfinite(pts).all(1)
    for a, b in DRAW_EDGES:
        if ok[a] and ok[b]:
            axis.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                      style, color=colour, linewidth=width)


def tile_label(axis, text, colour="w"):
    axis.text(0.5, -0.09, text, transform=axis.transAxes, ha="center",
              va="top", fontsize=8, color=colour, family="monospace")


# ------------------------------------------------------------------ one frame
@torch.no_grad()
def analyse(model, features, weight, jp, ip, label, seed):
    image = cv2.imread(ip)
    if image is None:
        return None
    height, width = image.shape[:2]
    obj = label["objects"][0]
    dims = obj["dimensions_m"]
    model_pts = APNP.make_pallet_keypoints_3d_diagram(
        width=dims["width"], depth=dims["depth"], height=dims["height"])[:8]
    extents = (dims["width"], dims["height"], dims["depth"])
    K = CG.intrinsics(label)
    R_gt, t_gt = CG.gt_pose(label)
    cuboid = np.asarray(obj["projected_cuboid"], float)
    gt8 = cuboid[:8]

    out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
    belief = out["beliefs"][-1][0, :9].float().cpu().numpy()
    peaks = MS._decode_peaks(out["beliefs"][-1][:, :9])[0]
    thresholded = extract_keypoints_from_belief(
        out["beliefs"][-1][0].float().cpu().numpy(), EV.THRESH)
    n_det = int(sum(1 for k in thresholded[:8] if k[0] >= 0))
    score_4kp = float(np.sort(np.max(belief[:8].reshape(8, -1), axis=1))[::-1][3])

    grid_theta, grid_rho, valid = DH.lattice()
    theta_c, rho_c = DH.decode(out["line_scores"], grid_theta, grid_rho, valid)
    theta_can, rho_can = DH.canonical_from_centred(theta_c, rho_c)
    theta_can = theta_can[0].cpu().numpy()
    rho_can = rho_can[0].cpu().numpy()
    attention = attention_weights(model, out["f50"])[0].float().cpu().numpy()

    # GT lines on the same 50-grid, and which roles the loss ever supervised.
    gt_grid = EV.pixels_to_grid(gt8, width, height)[None]
    gt_theta, gt_rho, p0, p1, length = V2.gt_lines(gt_grid, CG.EDGES)
    support = V2.visible_segments(p0, p1, length)["hit"][0]
    # `gt_lines` 의 atan2 는 (-180, 180] 을 준다.  학습은 이것을 `batch_rows` 에서
    # [0, 180) 으로 접고 rho 부호를 따라 뒤집은 뒤에야 격자에 올린다.  같은 접기를
    # 하지 않으면 -176 도가 175 도 오차로 찍혀, 그림에서는 거의 같은 방향인 선이
    # 라벨에서만 정반대로 보인다.
    centred_theta, centred_rho = DH.centred_from_canonical(
        torch.tensor(gt_theta[0]).float(), torch.tensor(gt_rho[0]).float())
    folded_theta = centred_theta % 180.0
    folded_rho = torch.where(((centred_theta // 180.0) % 2) == 1,
                             -centred_rho, centred_rho)
    # line_distance 는 가설 x GT 를 모두 재는 (12,12) 행렬을 준다.  여기서 필요한
    # 것은 role k 대 엣지 k 뿐이므로 대각선만 취한다 -- wrap 처리는 그대로 받는다.
    angle, offset, _ = DH.line_distance(theta_c[0].cpu(), rho_c[0].cpu(),
                                        folded_theta, folded_rho)
    theta_err = angle.diagonal().numpy(); rho_err = offset.diagonal().numpy()

    # pose through the canonical F0/F3 route -- no second solver is written here
    data = {"resolution": np.array([[width, height]]),
            "model": np.array([model_pts]), "K": np.array([K]),
            "pred_corner": np.array([peaks]),
            "pred_theta": theta_can[None], "pred_rho": rho_can[None],
            "support": np.array([EV.support_from_grid(peaks, width, height)])}
    arms, _, _, _ = FU.solve_arms(data, 0, weight)

    # 코너를 두 경로로 각각 찍어 둔다.  belief 가 못 잡은 코너를 line 이 대신
    # 찾아주는지는 이 둘을 같은 프레임에서 나란히 재야만 말할 수 있다.
    # CIGM = 그 코너에 붙은 엣지 3 개의 최소자승 교점.
    cigm, residual, condition = CG.cigm_corners(theta_c, rho_c)
    cigm_px = grid_to_pixel(cigm[0].cpu().numpy(), width, height)
    direct_px = grid_to_pixel(np.asarray(peaks, float)[:8, :2], width, height)
    in_frame = ((gt8[:, 0] >= 0) & (gt8[:, 0] < width)
                & (gt8[:, 1] >= 0) & (gt8[:, 1] < height))
    detected = np.array([thresholded[c][0] >= 0 for c in range(8)])
    corner_peak = belief[:8].reshape(8, -1).max(1)
    direct_err = np.linalg.norm(direct_px - gt8, axis=1)
    cigm_err = np.linalg.norm(cigm_px - gt8, axis=1)

    # role 별: attention 질량이 GT 엣지 밴드 안에 얼마나 놓였나
    on_edge, centroid_gap, entropy = [], [], []
    for r in range(ROLES):
        flat = attention[r].reshape(-1).astype(np.float64)
        entropy.append(float(-(flat * np.log(np.clip(flat, 1e-12, None))).sum()))
        distance = segment_distance(CELLS, p0[0, r], p1[0, r])
        on_edge.append(float(flat[distance <= BAND_CELL].sum()))
        mass_centre = (CELLS * flat[:, None]).sum(0)
        centroid_gap.append(float(segment_distance(mass_centre[None],
                                                   p0[0, r], p1[0, r])[0]))
    band_prior = float((segment_distance(CELLS, p0[0, 0], p1[0, 0])
                        <= BAND_CELL).mean())

    record = {
        "frame": os.path.splitext(os.path.basename(ip))[0], "seed": seed,
        "image": image, "gt8": gt8, "centroid_gt": cuboid[8] if len(cuboid) > 8 else None,
        "belief": belief, "peaks": peaks, "thresholded": thresholded,
        "attention": attention, "n_det": n_det, "score_4kp": score_4kp,
        "theta_can": theta_can, "rho_can": rho_can, "gt_p0": p0[0], "gt_p1": p1[0],
        "support": support, "theta_err": theta_err, "rho_err": rho_err,
        "on_edge": np.array(on_edge), "centroid_gap": np.array(centroid_gap),
        "entropy": np.array(entropy), "band_prior": band_prior,
        "model_pts": model_pts, "K": K, "extents": extents,
        "R_gt": R_gt, "t_gt": t_gt, "arms": arms,
        "cigm_px": cigm_px, "direct_px": direct_px, "in_frame": in_frame,
        "detected": detected, "corner_peak": corner_peak,
        "direct_err": direct_err, "cigm_err": cigm_err,
        "cigm_condition": condition[0].cpu().numpy()}
    return record


def project(R, t, model_pts, K):
    camera = (R @ model_pts.T).T + t
    depth = np.clip(camera[:, 2], 1e-6, None)
    return (K @ (camera / depth[:, None]).T).T[:, :2]


# ------------------------------------------------------------------- figure
def render(record, key, path):
    image, height, width = record["image"], *record["image"].shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    figure = plt.figure(figsize=(24.0, 16.4), facecolor="#101014")
    grid = GridSpec(4, 36, figure=figure, height_ratios=[6.0, 2.4, 3.0, 3.0],
                    hspace=0.28, wspace=0.16,
                    left=0.012, right=0.988, top=0.925, bottom=0.02)

    errors = {}
    for arm in ("F0", "F3"):
        pose = record["arms"].get(arm)
        errors[arm] = None if pose is None else RM.pose_error(
            pose[0], pose[1], record["R_gt"], record["t_gt"])
    head = (f"{record['frame']}   {key}   seed{record['seed']}   "
            f"{width}x{height}   detected {record['n_det']}/8 @0.3   "
            f"score_4kp {record['score_4kp']:.3f}")
    for arm in ("F0", "F3"):
        head += (f"   |  {arm} " + ("PnP FAILED" if errors[arm] is None else
                 f"R {errors[arm][0]:.2f}deg t {errors[arm][1]:.3f}m"))
    figure.text(0.012, 0.973, head, color="w", fontsize=13, family="monospace")
    figure.text(0.012, 0.949,
                "line = 진짜 attention (role query -> f50 2500 위치)   |   "
                "corner = belief map (attention 아님, conv 응답)   |   "
                "초록 GT · 빨강 예측 · 노랑 검출@0.3",
                color="#9aa0aa", fontsize=10.5, family="monospace")

    # --- row 0: 개요 3장 ---------------------------------------------------
    axis = figure.add_subplot(grid[0, 0:12])
    axis.imshow(rgb); axis.set_xticks([]); axis.set_yticks([])
    draw_cuboid(axis, record["gt8"], "#3ecf5a", "-", 2.0)
    for arm, colour in (("F0", "#4aa8ff"), ("F3", "#ff4d4d")):
        pose = record["arms"].get(arm)
        if pose is not None:
            draw_cuboid(axis, project(*pose, record["model_pts"], record["K"]),
                        colour, "--", 1.8)
    for k in record["thresholded"][:8]:
        if k[0] >= 0:
            axis.plot(k[0] * width / GRID, k[1] * height / GRID, "x",
                      color="#ffd400", markersize=9, markeredgewidth=2)
    axis.set_title("입력 + GT(초록) / F0(파랑) / F3(빨강) / 검출(노랑)",
                   color="w", fontsize=11)

    axis = figure.add_subplot(grid[0, 12:24])
    backdrop(axis, image)
    heat(axis, record["belief"][:9].max(0), image, "inferno")
    for k in range(8):
        axis.plot(*record["gt8"][k], "o", mfc="none", mec="#3ecf5a", ms=8, mew=1.6)
    axis.set_title("corner belief 9채널 최대 — 모델이 코너라고 답한 곳",
                   color="w", fontsize=11)

    axis = figure.add_subplot(grid[0, 24:36])
    backdrop(axis, image)
    heat(axis, record["attention"].max(0), image, "viridis")
    # 12 개 선을 겹쳐 그으면 화면이 선으로 덮여 정작 히트맵이 안 보인다.
    # 대신 role 마다 가장 크게 본 한 점에 번호를 찍는다.
    for r in range(ROLES):
        row, column = np.unravel_index(record["attention"][r].argmax(), (GRID, GRID))
        axis.text((column + 0.5) * width / GRID, (row + 0.5) * height / GRID,
                  str(r), color="#ffd400", fontsize=11, ha="center", va="center",
                  fontweight="bold",
                  path_effects=[pe.withStroke(linewidth=2.4, foreground="#101014")])
    axis.set_title("line attention 12 role 최대 + role별 최대점 — 모델이 본 곳",
                   color="w", fontsize=11)

    # --- row 1: corner belief 9 채널 ---------------------------------------
    for c in range(9):
        axis = figure.add_subplot(grid[1, c * 4:(c + 1) * 4])
        backdrop(axis, image)
        heat(axis, record["belief"][c], image, "inferno")
        peak = record["peaks"][c]
        axis.plot(peak[0] * width / GRID, peak[1] * height / GRID, "x",
                  color="#ff4d4d", ms=9, mew=2)
        truth = (record["gt8"][c] if c < 8 else record["centroid_gt"])
        note = ""
        if c < 8:
            # 같은 코너를 line 교점으로 찍은 자리.  belief 가 비어 있는 코너에서
            # 이 삼각형이 GT 에 붙어 있으면, 그때만 "엣지가 대신 찾았다"고 말할 수 있다.
            axis.plot(*record["cigm_px"][c], "^", mfc="none", mec="#4aa8ff",
                      ms=10, mew=1.8)
        if truth is not None:
            axis.plot(*truth, "o", mfc="none", mec="#3ecf5a", ms=9, mew=1.7)
            note = f" err{record['direct_err'][c]:5.1f}" if c < 8 else ""
        if c < 8:
            note += f"|cigm{record['cigm_err'][c]:5.1f}px"
            if not record["detected"][c]:
                note += "  [미검출]"
        tile_label(axis, f"{CORNER_NAMES[c]}  pk{record['belief'][c].max():.2f}{note}",
                   colour="#ff9a4d" if c < 8 and not record["detected"][c] else "w")

    # --- row 2-3: line attention 12 role -----------------------------------
    for r in range(ROLES):
        row, column = 2 + r // 6, (r % 6) * 6
        axis = figure.add_subplot(grid[row, column:column + 6])
        backdrop(axis, image)
        heat(axis, record["attention"][r], image, "viridis")
        gt = grid_to_pixel(np.stack([record["gt_p0"][r], record["gt_p1"][r]]),
                           width, height)
        axis.plot(gt[:, 0], gt[:, 1], "-", color="#3ecf5a", linewidth=2.2)
        a, b = line_through_grid(record["theta_can"][r], record["rho_can"][r])
        segment = grid_to_pixel(np.stack([a, b]), width, height)
        axis.plot(segment[:, 0], segment[:, 1], "--", color="#ff4d4d", linewidth=1.8)
        axis.set_xlim(0, width); axis.set_ylim(height, 0)
        edge = CG.EDGES[r]
        mark = "" if record["support"][r] else "  [화면밖·미감독]"
        tile_label(axis,
                   f"role {r:2d}  edge {edge} {FAMILY[r]}{mark}\n"
                   f"on-edge {record['on_edge'][r]:.2f}  ent {record['entropy'][r]:.2f}\n"
                   f"th {record['theta_err'][r]:5.2f}deg  rho {record['rho_err'][r]:6.2f}px",
                   colour=FAMILY_COLOUR[FAMILY[r]] if record["support"][r] else "#ff5555")

    figure.legend(handles=[
        Line2D([], [], color="#3ecf5a", lw=2.2, label="GT"),
        Line2D([], [], color="#ff4d4d", ls="--", lw=1.8, label="예측(corner/line)"),
        Line2D([], [], color="#4aa8ff", marker="^", mfc="none", ls="",
               label="line 교점 corner (CIGM)"),
        Line2D([], [], color="#ffd400", marker="x", ls="", label="검출 @0.3")],
        loc="upper right", ncol=4, frameon=False, fontsize=10,
        labelcolor="w", bbox_to_anchor=(0.988, 0.998))
    figure.savefig(path, dpi=95, facecolor=figure.get_facecolor())
    plt.close(figure)


# ------------------------------------------------------------------ summary
def summarise(records, folder, seed):
    supported = np.concatenate([r["on_edge"][r["support"]] for r in records])
    unsupported = np.concatenate([r["on_edge"][~r["support"]] for r in records
                                  if (~r["support"]).any()])
    prior = float(np.mean([r["band_prior"] for r in records]))
    rho = np.concatenate([r["rho_err"][r["support"]] for r in records])
    theta = np.concatenate([r["theta_err"][r["support"]] for r in records])
    gap = np.concatenate([r["centroid_gap"][r["support"]] for r in records])

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.2), facecolor="#101014")
    for axis in axes:
        axis.set_facecolor("#17171d")
        axis.tick_params(colors="#c8ccd4")
        for spine in axis.spines.values():
            spine.set_color("#3a3a46")

    axes[0].hist(supported, bins=40, range=(0, 1), color="#4aa8ff",
                 label=f"supervised n={supported.size}")
    if unsupported.size:
        axes[0].hist(unsupported, bins=40, range=(0, 1), color="#ff9a4d",
                     alpha=0.65, label=f"unsupervised n={unsupported.size}")
    axes[0].axvline(prior, color="#ff4d4d", ls="--",
                    label=f"균일 기대 {prior:.3f}")
    axes[0].set_title(f"GT 엣지 ±{BAND_CELL:.0f}cell 안의 attention 질량",
                      color="w")
    axes[0].set_xlabel("on-edge mass", color="#c8ccd4")
    axes[0].legend(labelcolor="w", facecolor="#17171d", edgecolor="#3a3a46")

    by_role = [np.concatenate([r["on_edge"][k:k + 1] for r in records
                               if r["support"][k]]) if
               any(r["support"][k] for r in records) else np.array([np.nan])
               for k in range(ROLES)]
    axes[1].boxplot(by_role, labels=[f"{k}\n{CG.EDGES[k]}" for k in range(ROLES)])
    axes[1].axhline(prior, color="#ff4d4d", ls="--")
    axes[1].set_title("role 별 on-edge mass", color="w")
    axes[1].tick_params(axis="x", labelsize=7)

    family = np.concatenate([np.array(FAMILY)[r["support"]] for r in records])
    for name, colour in FAMILY_COLOUR.items():
        pick = family == name
        axes[2].scatter(supported[pick], rho[pick], s=10, alpha=0.45,
                        color=colour, label=f"{name} n={int(pick.sum())}")
    axes[2].set_xlabel("on-edge mass", color="#c8ccd4")
    axes[2].set_ylabel("rho 오차 (MAP100 px)", color="#c8ccd4")
    correlation = float(np.corrcoef(supported, rho)[0, 1])
    axes[2].set_title(f"attention 국소성 vs rho 오차   r={correlation:+.3f}",
                      color="w")
    axes[2].legend(labelcolor="w", facecolor="#17171d", edgecolor="#3a3a46",
                   fontsize=9)
    for axis in axes:
        axis.title.set_fontsize(11)
    figure.tight_layout()
    path = os.path.join(folder, f"SUMMARY_seed{seed}.png")
    figure.savefig(path, dpi=110, facecolor=figure.get_facecolor())
    plt.close(figure)

    def by_family(values):
        return {name: {"n": int((family == name).sum()),
                       "median": float(np.median(values[family == name]))}
                for name in FAMILY_COLOUR if (family == name).any()}

    verdict = {
        "seed": seed, "frames": len(records),
        "band_cell": BAND_CELL, "uniform_band_mass_prior": prior,
        "on_edge_supervised": {"n": int(supported.size),
                               "mean": float(supported.mean()),
                               "median": float(np.median(supported)),
                               "p10": float(np.percentile(supported, 10))},
        "on_edge_unsupervised": ({"n": int(unsupported.size),
                                  "mean": float(unsupported.mean())}
                                 if unsupported.size else None),
        "centroid_gap_cell_median": float(np.median(gap)),
        "theta_err_deg_median": float(np.median(theta)),
        "rho_err_px_median": float(np.median(rho)),
        "corr_on_edge_vs_rho_err": correlation,
        "by_edge_family": {"on_edge": by_family(supported),
                           "theta_err_deg": by_family(theta),
                           "rho_err_px": by_family(rho)},
        "ATTENTION_IS_LOCAL": bool(np.median(supported) > 3.0 * prior),
        "summary_figure": os.path.relpath(path, ROOT)}
    open(os.path.join(folder, f"ATTENTION_VERDICT_seed{seed}.json"), "w").write(
        json.dumps(verdict, indent=2))
    return verdict


# ------------------------------------------- belief 가 비었을 때 line 이 찾는가
# 코너를 어느 구간에서 볼 것인지.  구간 정의는 결과를 보기 전에 고정한다.
BANDS = {
    "ALL": lambda r: np.ones(8, bool),
    "DETECTED": lambda r: r["detected"],
    "MISSED": lambda r: ~r["detected"],
    "IN_FRAME_MISSED": lambda r: (~r["detected"]) & r["in_frame"],
    "OFF_FRAME": lambda r: ~r["in_frame"],
    "FAR_FACE": lambda r: np.array([c >= 4 for c in range(8)]),
}


def cluster_bootstrap(per_frame, draws=4000, seed=20260907):
    """프레임을 단위로 리샘플한다.  한 프레임의 8 코너는 서로 독립이 아니다."""
    frames = [np.asarray(v) for v in per_frame if len(v)]
    if len(frames) < 2:
        return None
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(frames), size=(draws, len(frames)))
    medians = np.array([np.median(np.concatenate([frames[j] for j in row]))
                        for row in picks])
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def corner_rescue(records, folder, seed):
    """belief 가 못 잡은 코너를 line 교점(CIGM)이 대신 찾아주는가.

    같은 프레임에서 두 경로가 같은 코너를 찍으므로 paired 로 비교한다.
    차이 = cigm_err - direct_err 이고, 음수여야 line 이 이긴 것이다.
    """
    bands, table = {}, []
    for name, select in BANDS.items():
        direct, cigm, paired = [], [], []
        for record in records:
            pick = select(record)
            if not pick.any():
                continue
            direct.append(record["direct_err"][pick])
            cigm.append(record["cigm_err"][pick])
            paired.append(record["cigm_err"][pick] - record["direct_err"][pick])
        if not direct:
            bands[name] = None
            continue
        flat_direct = np.concatenate(direct)
        flat_cigm = np.concatenate(cigm)
        flat_paired = np.concatenate(paired)
        interval = cluster_bootstrap(paired)
        wins = float((flat_paired < 0).mean())
        bands[name] = {
            "n_corners": int(flat_direct.size), "n_frames": len(direct),
            "direct_err_px_median": float(np.median(flat_direct)),
            "cigm_err_px_median": float(np.median(flat_cigm)),
            "paired_diff_px_median": float(np.median(flat_paired)),
            "paired_diff_ci95": interval,
            "cigm_wins_fraction": wins,
            "LINE_BEATS_CORNER": bool(interval is not None and interval[1] < 0.0)}
        table.append((name, bands[name]))

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.4), facecolor="#101014")
    for axis in axes:
        axis.set_facecolor("#17171d"); axis.tick_params(colors="#c8ccd4")
        for spine in axis.spines.values():
            spine.set_color("#3a3a46")

    names = [n for n, _ in table]
    spots = np.arange(len(names))
    axes[0].bar(spots - 0.2, [b["direct_err_px_median"] for _, b in table],
                0.4, color="#ff4d4d", label="corner belief (direct)")
    axes[0].bar(spots + 0.2, [b["cigm_err_px_median"] for _, b in table],
                0.4, color="#4aa8ff", label="line 교점 (CIGM)")
    axes[0].set_xticks(spots)
    axes[0].set_xticklabels([f"{n}\nn={b['n_corners']}" for n, b in table],
                            fontsize=8)
    axes[0].set_ylabel("코너 오차 중앙값 (px)", color="#c8ccd4")
    axes[0].set_title("구간별 코너 오차 — 낮을수록 좋다", color="w")
    axes[0].legend(labelcolor="w", facecolor="#17171d", edgecolor="#3a3a46")

    missed = [r for r in records if (~r["detected"]).any()]
    if missed:
        x = np.concatenate([r["direct_err"][~r["detected"]] for r in missed])
        y = np.concatenate([r["cigm_err"][~r["detected"]] for r in missed])
        limit = float(max(x.max(), y.max())) * 1.05
        axes[1].scatter(x, y, s=14, alpha=0.5, color="#ff9a4d")
        axes[1].plot([0, limit], [0, limit], "--", color="#c8ccd4", lw=1)
        axes[1].set_xlim(0, limit); axes[1].set_ylim(0, limit)
        axes[1].set_xlabel("direct 오차 (px)", color="#c8ccd4")
        axes[1].set_ylabel("CIGM 오차 (px)", color="#c8ccd4")
        axes[1].set_title(f"미검출 코너 n={x.size} — 대각선 아래면 line 이 이긴 것",
                          color="w")

    for index, (name, band) in enumerate(table):
        interval = band["paired_diff_ci95"]
        if interval is None:
            continue
        colour = "#3ecf5a" if band["LINE_BEATS_CORNER"] else "#c8ccd4"
        axes[2].plot([interval[0], interval[1]], [index, index], "-",
                     color=colour, lw=3)
        axes[2].plot(band["paired_diff_px_median"], index, "o", color=colour, ms=7)
    axes[2].axvline(0, color="#ff4d4d", ls="--")
    axes[2].set_yticks(spots); axes[2].set_yticklabels(names, fontsize=9)
    axes[2].set_xlabel("CIGM - direct (px), 음수면 line 이 낫다", color="#c8ccd4")
    axes[2].set_title("paired 차이 95% CI (프레임 클러스터 부트스트랩)", color="w")
    for axis in axes:
        axis.title.set_fontsize(11)
    figure.tight_layout()
    path = os.path.join(folder, f"CORNER_RESCUE_seed{seed}.png")
    figure.savefig(path, dpi=110, facecolor=figure.get_facecolor())
    plt.close(figure)

    return {"bands": bands, "figure": os.path.relpath(path, ROOT),
            "EDGE_RESCUES_MISSED_CORNERS":
                bool(bands.get("MISSED") and bands["MISSED"]["LINE_BEATS_CORNER"])}


# --------------------------------------------------------------------- main
def run_seed(seed, output, features, weight):
    model, ckpt = EV.load(seed)
    log(f"seed{seed}  {ckpt}")
    folder = os.path.join(output, f"seed{seed}")
    records, written = [], 0
    for key in EV.OPEN_SETS:
        os.makedirs(os.path.join(folder, key), exist_ok=True)
        for jp, ip, label in EV.frames(key):
            record = analyse(model, features, weight, jp, ip, label, seed)
            if record is None:
                continue
            path = os.path.join(folder, key, f"{record['frame']}.png")
            render(record, key, path)
            written += 1
            record.pop("image")
            records.append(record)
            if written % 10 == 0:
                log(f"  seed{seed}  {written}장")
    verdict = summarise(records, folder, seed)
    verdict["corner_rescue"] = corner_rescue(records, folder, seed)
    del model
    torch.cuda.empty_cache()
    return written, verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead/final_train/attention"))
    parser.add_argument("--seeds", default="1,2")
    arguments = parser.parse_args()

    MS.deterministic()
    _, _, _, features = MS.lattice()
    weight = json.loads(open(os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead",
        "theta_posealigned_d0.json")).read())["seeds"]["seed1"]["selected_lambda_theta"]
    os.makedirs(arguments.output, exist_ok=True)

    total, verdicts = 0, {}
    for seed in [int(s) for s in arguments.seeds.split(",")]:
        written, verdict = run_seed(seed, arguments.output, features, weight)
        total += written
        verdicts[f"seed{seed}"] = verdict
        log(f"seed{seed} 판정  on-edge median "
            f"{verdict['on_edge_supervised']['median']:.3f} "
            f"(균일 {verdict['uniform_band_mass_prior']:.3f})  "
            f"ATTENTION_IS_LOCAL={verdict['ATTENTION_IS_LOCAL']}  "
            f"corr(on-edge, rho err) {verdict['corr_on_edge_vs_rho_err']:+.3f}")
        for name, band in verdict["corner_rescue"]["bands"].items():
            if band is None:
                continue
            interval = band["paired_diff_ci95"]
            span = "-" if interval is None else f"[{interval[0]:+.2f},{interval[1]:+.2f}]"
            log(f"  {name:<16} n={band['n_corners']:4d}  direct "
                f"{band['direct_err_px_median']:6.2f}px  cigm "
                f"{band['cigm_err_px_median']:6.2f}px  diff "
                f"{band['paired_diff_px_median']:+6.2f} CI95 {span}  "
                f"LINE_WINS={band['LINE_BEATS_CORNER']}")

    # 선언이 아니라 파일로 확인한다.
    on_disk = sum(len([f for f in files if f.endswith(".png")])
                  for _, _, files in os.walk(arguments.output))
    open(os.path.join(arguments.output, "ATTENTION_VERDICT.json"), "w").write(
        json.dumps({"seeds": verdicts, "frames_rendered": total,
                    "png_on_disk": on_disk}, indent=2))
    log(f"완료: {total}장 렌더, 디스크 png {on_disk}개 -> "
        f"{os.path.relpath(arguments.output, ROOT)}")


if __name__ == "__main__":
    main()
