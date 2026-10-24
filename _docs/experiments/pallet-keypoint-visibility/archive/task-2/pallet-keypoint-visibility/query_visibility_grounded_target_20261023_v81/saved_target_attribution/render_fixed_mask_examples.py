"""Fixed min-SHA mask-only display. Source RGB is never opened."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from analyze_saved_target_ownership import bind, lines, write

R = Path(__file__).resolve().parent
V = R.parent


def boundary(mask):
    inside = mask.copy()
    inside[0] = inside[-1] = False
    inside[:, 0] = inside[:, -1] = False
    inside[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return mask & ~inside


def read_mask(binding, threshold=0):
    assert bind(binding["path"])["sha256"] == binding["sha256"]
    return np.asarray(Image.open(binding["path"])) > threshold


def local(mask, q, side, n=96):
    j = np.arange(n, dtype=np.float64)
    x = q[0] + (((j + .5) / n) - .5) * side
    y = q[1] + (((j + .5) / n) - .5) * side
    ix, iy = np.floor(x + .5).astype(int), np.floor(y + .5).astype(int)
    valid = (ix[None, :] >= 0) & (ix[None, :] < mask.shape[1]) & (iy[:, None] >= 0) & (iy[:, None] < mask.shape[0])
    out = np.zeros((n, n), bool)
    xx, yy = np.broadcast_arrays(ix[None, :], iy[:, None])
    out[valid] = mask[yy[valid], xx[valid]]
    return out


def main():
    selection_path = R / "FIXED_MIN_SHA_EXAMPLES_BEFORE_VISUALIZATION.json"
    examples = json.loads(selection_path.read_text())["examples"]
    gt = {row["input_id"]: row for row in lines(V / "source_evaluation_preparation/GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl")}
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(examples), 4, figsize=(14, 4.2 * len(examples)), squeeze=False)
    reviews = []
    for r, example in enumerate(examples):
        lineage = gt[example["input_id"]]
        full = read_mask(lineage["M0"])
        visible = read_mask(lineage["M4"])
        q, side = example["query_xy"], example["L"]
        other = read_mask(example["best_other_mask_binding"]) if example.get("best_other_mask_binding") else None
        predicted = None
        if example.get("prediction_mask_binding"):
            binding = example["prediction_mask_binding"]
            assert bind(binding["path"])["sha256"] == binding["sha256"]
            with np.load(binding["path"], allow_pickle=False) as data:
                predicted = data["visible_mask"]
        canvas = np.full((*full.shape, 3), 248, np.uint8)
        canvas[full] = [215, 215, 215]
        canvas[visible] = [50, 170, 90]
        views = [canvas, np.full_like(canvas, 248), np.full_like(canvas, 248)]
        if other is not None:
            views[1][other] = [245, 160, 20]
        if predicted is not None:
            views[2][predicted] = [190, 80, 180]
        titles = ["의도한 대상 GT: M0 회색 / M4 초록", "가장 잘 겹친 다른 instance GT M4", "이미 선택된 예측 mask (재선택 없음)"]
        for column in range(3):
            ax = axes[r, column]
            ax.imshow(views[column]); ax.plot(q[0], q[1], "+", color="red", markersize=12, markeredgewidth=1.5)
            ax.add_patch(plt.Rectangle((q[0] - side / 2, q[1] - side / 2), side, side, fill=False, edgecolor="blue", linewidth=1.2))
            ax.set_title(titles[column], fontsize=10); ax.axis("off")
        if other is None:
            axes[r, 1].text(.5, .5, "비교 마스크 미정\n보류 U", ha="center", va="center", transform=axes[r, 1].transAxes)
        if predicted is None:
            axes[r, 2].text(.5, .5, "U: 선택 마스크 없음\n0 mask로 만들지 않음", ha="center", va="center", transform=axes[r, 2].transAxes)
        composed = np.full((96, 96, 3), 248, np.uint8)
        composed[local(full, q, side)] = [215, 215, 215]
        composed[local(visible, q, side)] = [50, 170, 90]
        if other is not None:
            composed[local(other, q, side)] = [245, 160, 20]
        if predicted is not None:
            composed[boundary(local(predicted, q, side))] = [190, 50, 180]
        axes[r, 3].imshow(composed, interpolation="nearest")
        axes[r, 3].plot(47.5, 47.5, "+", color="red", markersize=13, markeredgewidth=1.5)
        axes[r, 3].set_title("원래 q/L crop: 예측 경계 자홍", fontsize=10); axes[r, 3].axis("off")
        text = f"{example['original_id']} / k{example['native_k']} / q=({q[0]:.2f},{q[1]:.2f}) / L={side:.2f}"
        if predicted is not None:
            text += f"\n의도 IoU={example['intended_overlap']['IoU']:.5f}, 다른 instance IoU={example['best_other_overlap']['IoU']:.5f} / phrase={example['selected_phrase']}"
        else:
            text += "\n최대 초기 box IoU < 0.1에 따른 보류 (고정 5개 중 최소 SHA)"
        axes[r, 0].text(0., -.10, text, transform=axes[r, 0].transAxes, fontsize=9, va="top")
        reviews.append({"input_id": example["input_id"], "original_id": example["original_id"], "native_k": example["native_k"],
            "attribution_bin": example["attribution_bin"], "mask_only_no_RGB": True,
            "설명": "다른 instance M4와 예측 mask의 일치를 보여주는 고정 진단도. 실제 RGB 없이 detector 단독 실패로 확정하지 않음." if predicted is not None else "보류를 원래 U로 표시하며 예측 mask를 꾸며 만들지 않음."})
    fig.suptitle("고정 최소 SHA 사례: 실제 RGB가 아닌 저장 native mask 비교\n빨간 십자=q, 파란 사각형=원래 q 중심 crop. GT 기반 재선택·좌표이동 없음.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.subplots_adjust(hspace=.55)
    png = R / "FIXED_MIN_SHA_MASK_ONLY_CONTACT_SHEET.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    write(R / "FIXED_EXAMPLE_MASK_REVIEW_KO.json", {"completed": True, "selection": bind(selection_path), "code": bind(__file__),
        "figure": bind(png), "examples": reviews, "source_RGB_pixels_model_newforward_fit_current_PnP": 0})
    print(json.dumps(bind(png)))


if __name__ == "__main__":
    main()
