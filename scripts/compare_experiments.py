"""실험 간 메트릭 비교.

weights/ 하위 디렉토리의 eval_summary.json을 스캔하여 테이블로 출력.

사용법:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --dirs weights/misc/pallet_category weights/blender_v1
    python scripts/compare_experiments.py --sort pck@3px
"""

import argparse
import glob
import json
import os
import sys


def find_experiments(base_dir="weights"):
    """eval_summary.json이 있는 실험 디렉토리 탐색."""
    experiments = []
    pattern = os.path.join(base_dir, "*", "eval_results", "eval_summary.json")
    for path in sorted(glob.glob(pattern)):
        exp_name = path.split(os.sep)[-3]
        with open(path) as f:
            summary = json.load(f)
        summary["_name"] = exp_name
        summary["_path"] = path
        experiments.append(summary)
    return experiments


def load_specific(dirs):
    """지정된 디렉토리들에서 eval_summary.json 로드."""
    experiments = []
    for d in dirs:
        summary_path = os.path.join(d, "eval_results", "eval_summary.json")
        if not os.path.exists(summary_path):
            print(f"[WARN] {summary_path} not found, skipping")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        summary["_name"] = os.path.basename(d)
        summary["_path"] = summary_path
        experiments.append(summary)
    return experiments


def print_table(experiments, sort_key=None):
    """메트릭 비교 테이블 출력."""
    if not experiments:
        print("No experiments found with eval_summary.json")
        return

    # 정렬
    if sort_key:
        key = sort_key.lower()
        def sort_fn(e):
            if key.startswith("pck"):
                return e.get("pck", {}).get(key.replace("pck", ""), 0)
            elif key == "reproj":
                v = e.get("reproj_mean_px")
                return v if v is not None else 999
            elif key == "pnp":
                return e.get("pnp_success_rate", 0)
            return 0
        experiments.sort(key=sort_fn, reverse=(not key.startswith("reproj")))

    # 헤더
    name_w = max(len(e["_name"]) for e in experiments)
    name_w = max(name_w, 10)

    header = (
        f"{'Experiment':<{name_w}}  "
        f"{'PCK@3px':>8}  {'PCK@5px':>8}  {'PCK@10px':>9}  "
        f"{'PnP Rate':>9}  {'Reproj(px)':>11}  "
        f"{'Vol Ratio':>10}  {'Vol<20%':>8}  {'Frames':>6}"
    )
    print(f"\n{'=' * len(header)}")
    print(" Experiment Comparison")
    print(f"{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")

    for e in experiments:
        pck = e.get("pck", {})
        pck3 = pck.get("@3px", 0)
        pck5 = pck.get("@5px", 0)
        pck10 = pck.get("@10px", 0)
        pnp = e.get("pnp_success_rate", 0)
        reproj = e.get("reproj_mean_px")
        reproj_str = f"{reproj:.2f}" if reproj is not None else "N/A"
        vol_ratio = e.get("volume_ratio_median")
        vol_str = f"{vol_ratio:.3f}" if vol_ratio is not None else "N/A"
        vol_20 = e.get("volume_within_20pct")
        vol_20_str = f"{vol_20*100:.1f}%" if vol_20 is not None else "N/A"
        frames = e.get("num_frames", 0)

        print(
            f"{e['_name']:<{name_w}}  "
            f"{pck3:>8.4f}  {pck5:>8.4f}  {pck10:>9.4f}  "
            f"{pnp:>8.1%}  {reproj_str:>11}  "
            f"{vol_str:>10}  {vol_20_str:>8}  {frames:>6}"
        )

    print(f"{'=' * len(header)}")

    # 최고 성능 하이라이트
    if len(experiments) > 1:
        best_pck3 = max(experiments, key=lambda e: e.get("pck", {}).get("@3px", 0))
        reproj_valid = [e for e in experiments if e.get("reproj_mean_px") is not None]
        best_reproj = min(reproj_valid, key=lambda e: e["reproj_mean_px"]) if reproj_valid else None

        print(f"\n  Best PCK@3px:  {best_pck3['_name']} ({best_pck3['pck'].get('@3px', 0):.4f})")
        if best_reproj:
            print(f"  Best Reproj:   {best_reproj['_name']} ({best_reproj['reproj_mean_px']:.2f}px)")


def main():
    parser = argparse.ArgumentParser(description="실험 메트릭 비교")
    parser.add_argument("--dirs", nargs="+", help="비교할 weight 디렉토리 (미지정 시 weights/ 전체 스캔)")
    parser.add_argument("--sort", default=None, help="정렬 기준 (pck@3px, pck@5px, reproj, pnp)")
    args = parser.parse_args()

    if args.dirs:
        experiments = load_specific(args.dirs)
    else:
        experiments = find_experiments()

    print_table(experiments, sort_key=args.sort)


if __name__ == "__main__":
    main()
