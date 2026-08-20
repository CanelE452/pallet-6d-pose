"""어노테이션이 없는 근접 프레임에서 모델이 어디를 보는가.

대상은 `capturepalletcad` 원본 1,179 장이다.  이 세션은 전략 문서가 근접이라고
직접 적어 둔 유일한 세션이고(`paper_strategy_master.md:132` "근접 캘리브성 촬영,
덱 꽉참/코너 잘림 → 분포 이질"), 그래서 GT 37 장만 붙은 채 정본에서 빠져 있다.
바꿔 말해 **평가에서 빠진 레짐**이라 눈으로 볼 값어치가 있다.

GT 가 없으므로 못 하는 것을 먼저 못 박는다.

    못 함   theta/rho 오차, 코너 오차, on-edge mass, direct 대 CIGM 우열
            -- 전부 GT 엣지·코너를 기준으로 정의된 값이다.
    함      attention 과 belief 히트맵, 예측 선/코너, 그리고 GT 없이도 성립하는
            자기일관성 하나: 같은 코너를 belief 로 찍은 자리와 그 코너에 붙은
            엣지 3 개의 교점(CIGM)으로 찍은 자리가 **얼마나 어긋나는가**.
            둘이 붙어 있으면 서로 확증한 것이고, 벌어지면 최소한 하나는 틀렸다.

"가까이"는 사람이 고르지 않는다.  검출된 코너가 만드는 상자의 대각을 이미지
대각으로 나눈 값으로 전 프레임을 재고 그 상위만 그린다.  이 값은 모델 예측에서
나오므로, 팔레트가 아예 안 잡힌 프레임이 큰 값을 받지 않도록 score_4kp 로 먼저
거른다.

전처리는 정본 평가와 같은 squash 를 쓴다.  다만 memory `dope-inference-needs-
reflect-padding` 이 "plain squash 는 근접·truncation 에서 체계적으로 과소검출
한다"고 기록해 두었으므로, 여기서 검출이 낮게 나오면 그것은 모델 실패가 아니라
전처리 탓일 수 있다 -- 그림을 읽을 때 이 점을 잊지 말 것.
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
sys.path.insert(0, os.path.join(ROOT, "scripts/stage0/final_train"))

import cv2                                        # noqa: E402
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.gridspec import GridSpec          # noqa: E402
from matplotlib.lines import Line2D               # noqa: E402
import matplotlib.patheffects as pe               # noqa: E402

import ft_attn_vis as AV                          # noqa: E402
import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mh_cigm as CG                              # noqa: E402
import ft_f0f3_eval as EV                         # noqa: E402
from mh_arms import DH                            # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402

GRID, ROLES = AV.GRID, AV.ROLES
SOURCE = os.path.join(ROOT, "data/pallet/raw_data/outside/capturepalletcad/rgb")
MIN_SCORE = 0.3          # 팔레트가 실제로 잡힌 프레임만 "근접" 후보로 본다


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@torch.no_grad()
def scan_frame(model, features, path):
    """forward 한 번.  그림은 아직 그리지 않고 크기와 신뢰도만 잰다."""
    image = cv2.imread(path)
    if image is None:
        return None
    height, width = image.shape[:2]
    out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
    belief = out["beliefs"][-1][0, :9].float().cpu().numpy()
    peaks = np.asarray(MS._decode_peaks(out["beliefs"][-1][:, :9])[0], float)
    thresholded = extract_keypoints_from_belief(
        out["beliefs"][-1][0].float().cpu().numpy(), EV.THRESH)
    seen = np.array([k[0] >= 0 for k in thresholded[:8]])
    score_4kp = float(np.sort(np.max(belief[:8].reshape(8, -1), axis=1))[::-1][3])

    span = 0.0
    if seen.sum() >= 2:
        pixels = np.array([[k[0] * width / GRID, k[1] * height / GRID]
                           for k in np.asarray(thresholded, object)[:8][seen]],
                          dtype=float)
        box = pixels.max(0) - pixels.min(0)
        span = float(np.hypot(*box) / np.hypot(width, height))
    return {"path": path, "frame": os.path.splitext(os.path.basename(path))[0],
            "span": span, "score_4kp": score_4kp, "n_det": int(seen.sum())}


@torch.no_grad()
def analyse(model, features, path):
    """그릴 프레임 하나를 끝까지 계산한다.  GT 는 쓰지 않는다."""
    image = cv2.imread(path)
    height, width = image.shape[:2]
    out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
    belief = out["beliefs"][-1][0, :9].float().cpu().numpy()
    peaks = np.asarray(MS._decode_peaks(out["beliefs"][-1][:, :9])[0], float)
    thresholded = extract_keypoints_from_belief(
        out["beliefs"][-1][0].float().cpu().numpy(), EV.THRESH)
    detected = np.array([thresholded[c][0] >= 0 for c in range(8)])
    score_4kp = float(np.sort(np.max(belief[:8].reshape(8, -1), axis=1))[::-1][3])

    theta_c, rho_c = DH.decode(out["line_scores"], *DH.lattice())
    theta_can, rho_can = DH.canonical_from_centred(theta_c, rho_c)
    attention = AV.attention_weights(model, out["f50"])[0].float().cpu().numpy()

    cigm, _, condition = CG.cigm_corners(theta_c, rho_c)
    cigm_px = AV.grid_to_pixel(cigm[0].cpu().numpy(), width, height)
    direct_px = AV.grid_to_pixel(peaks[:8, :2], width, height)
    disagree = np.linalg.norm(direct_px - cigm_px, axis=1)

    pixels = direct_px[detected] if detected.sum() >= 2 else direct_px
    box = pixels.max(0) - pixels.min(0)
    span = float(np.hypot(*box) / np.hypot(width, height))
    flat = attention.reshape(ROLES, -1).astype(np.float64)
    entropy = -(flat * np.log(np.clip(flat, 1e-12, None))).sum(1)

    return {"frame": os.path.splitext(os.path.basename(path))[0],
            "image": image, "belief": belief, "peaks": peaks,
            "attention": attention, "theta_can": theta_can[0].cpu().numpy(),
            "rho_can": rho_can[0].cpu().numpy(), "entropy": entropy,
            "direct_px": direct_px, "cigm_px": cigm_px, "disagree": disagree,
            "detected": detected, "score_4kp": score_4kp, "span": span,
            "n_det": int(detected.sum()),
            "cigm_condition": condition[0].cpu().numpy()}


def render(record, path, seed):
    image = record["image"]
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    figure = plt.figure(figsize=(24.0, 16.4), facecolor="#101014")
    grid = GridSpec(4, 36, figure=figure, height_ratios=[6.0, 2.4, 3.0, 3.0],
                    hspace=0.28, wspace=0.16,
                    left=0.012, right=0.988, top=0.925, bottom=0.02)

    shown = record["disagree"][record["detected"]]
    figure.text(0.012, 0.973,
                f"{record['frame']}   capturepalletcad (GT 없음)   seed{seed}   "
                f"{width}x{height}   화면점유 {record['span']:.3f}   "
                f"detected {record['n_det']}/8 @0.3   "
                f"score_4kp {record['score_4kp']:.3f}   "
                f"corner-line 불일치 중앙값 "
                f"{np.median(shown) if shown.size else float('nan'):.1f}px",
                color="w", fontsize=13, family="monospace")
    figure.text(0.012, 0.949,
                "GT 가 없다 — 오차가 아니라 두 경로의 합치만 잰다.  "
                "line = 진짜 attention (role query -> f50 2500 위치)   |   "
                "corner = belief map (attention 아님)   |   "
                "빨강 belief 최대점 · 파랑 line 교점(CIGM) · 노랑 검출@0.3",
                color="#9aa0aa", fontsize=10.5)

    axis = figure.add_subplot(grid[0, 0:12])
    axis.imshow(rgb); axis.set_xticks([]); axis.set_yticks([])
    for c in range(8):
        axis.plot(*record["direct_px"][c], "x", color="#ff4d4d", ms=9, mew=2)
        axis.plot(*record["cigm_px"][c], "^", mfc="none", mec="#4aa8ff",
                  ms=10, mew=1.8)
        axis.plot([record["direct_px"][c, 0], record["cigm_px"][c, 0]],
                  [record["direct_px"][c, 1], record["cigm_px"][c, 1]],
                  "-", color="#7a7a8a", lw=0.9)
    axis.set_xlim(0, width); axis.set_ylim(height, 0)
    axis.set_title("입력 + belief 최대점(빨강 x) / line 교점(파랑 △) — 회색선이 불일치",
                   color="w", fontsize=11)

    axis = figure.add_subplot(grid[0, 12:24])
    AV.backdrop(axis, image)
    AV.heat(axis, record["belief"][:9].max(0), image, "inferno")
    axis.set_title("corner belief 9채널 최대 — 모델이 코너라고 답한 곳",
                   color="w", fontsize=11)

    axis = figure.add_subplot(grid[0, 24:36])
    AV.backdrop(axis, image)
    AV.heat(axis, record["attention"].max(0), image, "viridis")
    for r in range(ROLES):
        row, column = np.unravel_index(record["attention"][r].argmax(),
                                       (GRID, GRID))
        axis.text((column + 0.5) * width / GRID, (row + 0.5) * height / GRID,
                  str(r), color="#ffd400", fontsize=11, ha="center", va="center",
                  fontweight="bold",
                  path_effects=[pe.withStroke(linewidth=2.4, foreground="#101014")])
    axis.set_title("line attention 12 role 최대 + role별 최대점 — 모델이 본 곳",
                   color="w", fontsize=11)

    for c in range(9):
        axis = figure.add_subplot(grid[1, c * 4:(c + 1) * 4])
        AV.backdrop(axis, image)
        AV.heat(axis, record["belief"][c], image, "inferno")
        axis.plot(*record["direct_px"][c] if c < 8 else
                  (record["peaks"][c][0] * width / GRID,
                   record["peaks"][c][1] * height / GRID),
                  "x", color="#ff4d4d", ms=9, mew=2)
        note = ""
        if c < 8:
            axis.plot(*record["cigm_px"][c], "^", mfc="none", mec="#4aa8ff",
                      ms=10, mew=1.8)
            note = f"  불일치 {record['disagree'][c]:5.1f}px"
            if not record["detected"][c]:
                note += " [미검출]"
        AV.tile_label(axis,
                      f"{AV.CORNER_NAMES[c]}  pk{record['belief'][c].max():.2f}{note}",
                      colour="#ff9a4d" if c < 8 and not record["detected"][c] else "w")

    for r in range(ROLES):
        row, column = 2 + r // 6, (r % 6) * 6
        axis = figure.add_subplot(grid[row, column:column + 6])
        AV.backdrop(axis, image)
        AV.heat(axis, record["attention"][r], image, "viridis")
        a, b = AV.line_through_grid(record["theta_can"][r], record["rho_can"][r])
        segment = AV.grid_to_pixel(np.stack([a, b]), width, height)
        axis.plot(segment[:, 0], segment[:, 1], "--", color="#ff4d4d", lw=1.8)
        axis.set_xlim(0, width); axis.set_ylim(height, 0)
        AV.tile_label(axis,
                      f"role {r:2d}  edge {CG.EDGES[r]} {AV.FAMILY[r]}\n"
                      f"entropy {record['entropy'][r]:.2f}",
                      colour=AV.FAMILY_COLOUR[AV.FAMILY[r]])

    figure.legend(handles=[
        Line2D([], [], color="#ff4d4d", marker="x", ls="", label="belief 최대점"),
        Line2D([], [], color="#4aa8ff", marker="^", mfc="none", ls="",
               label="line 교점 (CIGM)"),
        Line2D([], [], color="#ff4d4d", ls="--", lw=1.8, label="예측 선")],
        loc="upper right", ncol=3, frameon=False, fontsize=10,
        labelcolor="w", bbox_to_anchor=(0.988, 0.998))
    figure.savefig(path, dpi=95, facecolor=figure.get_facecolor())
    plt.close(figure)


def summarise(scanned, drawn, folder, seed):
    """가까울수록 두 경로가 더 갈라지는가 -- GT 없이 물을 수 있는 형태로."""
    span = np.array([r["span"] for r in scanned])
    score = np.array([r["score_4kp"] for r in scanned])
    detected = np.array([r["n_det"] for r in scanned])

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.4), facecolor="#101014")
    for axis in axes:
        axis.set_facecolor("#17171d"); axis.tick_params(colors="#c8ccd4")
        for spine in axis.spines.values():
            spine.set_color("#3a3a46")

    axes[0].scatter(span, score, s=7, alpha=0.35, color="#4aa8ff")
    axes[0].axhline(MIN_SCORE, color="#ff4d4d", ls="--", label=f"score {MIN_SCORE}")
    axes[0].set_xlabel("화면 점유 (검출 상자 대각 / 이미지 대각)", color="#c8ccd4")
    axes[0].set_ylabel("score_4kp", color="#c8ccd4")
    axes[0].set_title(f"스캔 {span.size}장 — 크기 대 신뢰도", color="w")
    axes[0].legend(labelcolor="w", facecolor="#17171d", edgecolor="#3a3a46")

    axes[1].hist(detected, bins=np.arange(10) - 0.5, color="#c77dff")
    axes[1].set_xlabel("검출 코너 수 @0.3", color="#c8ccd4")
    axes[1].set_title("근접 세션의 검출 분포", color="w")

    drawn_span = np.array([r["span"] for r in drawn])
    gaps = np.array([float(np.median(r["disagree"][r["detected"]]))
                     if r["detected"].any() else np.nan for r in drawn])
    good = np.isfinite(gaps)
    axes[2].scatter(drawn_span[good], gaps[good], s=26, alpha=0.7, color="#ff9a4d")
    axes[2].set_xlabel("화면 점유", color="#c8ccd4")
    axes[2].set_ylabel("corner-line 불일치 중앙값 (px)", color="#c8ccd4")
    trend = (float(np.corrcoef(drawn_span[good], gaps[good])[0, 1])
             if good.sum() > 2 else float("nan"))
    axes[2].set_title(f"그린 {int(good.sum())}장 — 클수록 갈라지는가  r={trend:+.3f}",
                      color="w")
    for axis in axes:
        axis.title.set_fontsize(11)
    figure.tight_layout()
    path = os.path.join(folder, f"UNLABELED_SUMMARY_seed{seed}.png")
    figure.savefig(path, dpi=110, facecolor=figure.get_facecolor())
    plt.close(figure)

    return {"scanned": int(span.size), "drawn": len(drawn),
            "source": os.path.relpath(SOURCE, ROOT),
            "GT": "none -- 오차 계산 불가, 자기일관성만",
            "span_p50": float(np.median(span)), "span_p95": float(np.percentile(span, 95)),
            "score_ge_0.3_fraction": float((score >= MIN_SCORE).mean()),
            "detected_8of8_fraction": float((detected == 8).mean()),
            "drawn_span_min": float(drawn_span.min()),
            "corner_line_gap_px_median": float(np.nanmedian(gaps)),
            "corr_span_vs_gap": trend,
            "summary_figure": os.path.relpath(path, ROOT)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead/final_train/attention",
        "unlabeled_cad"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top", type=int, default=24)
    arguments = parser.parse_args()

    MS.deterministic()
    _, _, _, features = MS.lattice()
    model, ckpt = EV.load(arguments.seed)
    os.makedirs(arguments.output, exist_ok=True)
    log(f"seed{arguments.seed}  {ckpt}")

    paths = sorted(os.path.join(SOURCE, n) for n in os.listdir(SOURCE)
                   if n.endswith(".png"))
    log(f"스캔 시작 {len(paths)}장  {os.path.relpath(SOURCE, ROOT)}")
    scanned = []
    for index, path in enumerate(paths, 1):
        row = scan_frame(model, features, path)
        if row is not None:
            scanned.append(row)
        if index % 200 == 0:
            log(f"  스캔 {index}/{len(paths)}")

    usable = [r for r in scanned if r["score_4kp"] >= MIN_SCORE and r["n_det"] >= 2]
    usable.sort(key=lambda r: -r["span"])
    picked = usable[:arguments.top]
    log(f"스캔 완료 {len(scanned)}장, score>={MIN_SCORE} 이고 2점 이상 {len(usable)}장, "
        f"상위 {len(picked)}장 렌더")
    if not picked:
        raise SystemExit("근접 후보가 없다 — 임계나 소스를 다시 볼 것")

    drawn = []
    for rank, row in enumerate(picked):
        record = analyse(model, features, row["path"])
        render(record, os.path.join(
            arguments.output,
            f"{rank:02d}_span{record['span']:.3f}_det{record['n_det']}"
            f"_{record['frame']}.png"), arguments.seed)
        record.pop("image")
        drawn.append(record)
        if (rank + 1) % 8 == 0:
            log(f"  렌더 {rank + 1}/{len(picked)}")

    verdict = summarise(scanned, drawn, arguments.output, arguments.seed)
    verdict["checkpoint"] = ckpt
    verdict["scan_index"] = [{k: r[k] for k in ("frame", "span", "score_4kp", "n_det")}
                             for r in usable[:200]]
    open(os.path.join(arguments.output,
                      f"UNLABELED_VERDICT_seed{arguments.seed}.json"), "w").write(
        json.dumps(verdict, indent=2))
    on_disk = len([f for f in os.listdir(arguments.output) if f.endswith(".png")])
    log(f"완료: 렌더 {len(drawn)}장, 디스크 png {on_disk}개  "
        f"점유 {verdict['drawn_span_min']:.3f}~  "
        f"불일치 중앙 {verdict['corner_line_gap_px_median']:.1f}px  "
        f"corr(점유,불일치) {verdict['corr_span_vs_gap']:+.3f}  -> "
        f"{os.path.relpath(arguments.output, ROOT)}")


if __name__ == "__main__":
    main()
