#!/usr/bin/env python3
"""TensorBoard 런처 — config/default.yaml 기반.

Usage:
    python scripts/launch_tensorboard.py                          # 전체 실험 (weights/)
    python scripts/launch_tensorboard.py --logdir weights/misc/pallet_v11/runs/
    python scripts/launch_tensorboard.py --compare pretrain finetune_v11
    python scripts/launch_tensorboard.py --summary                # loss 요약만
"""
import argparse
import sys
from glob import glob
from pathlib import Path

import yaml
from tensorboard import program
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# config/default.yaml의 train 섹션에서 실험 목록 자동 생성
EXPERIMENT_MAP = {
    "pretrain": lambda cfg: cfg["train"]["pretrain"]["output_dir"] + "/" + cfg["logging"]["log_dir_suffix"],
    "finetune": lambda cfg: cfg["train"]["finetune"]["output_dir"] + "/" + cfg["logging"]["log_dir_suffix"],
}


def print_summary(logdir):
    """logdir 내 이벤트 파일에서 최종 loss 요약 출력."""
    event_files = glob(str(Path(logdir) / "**" / "events.out.tfevents*"), recursive=True)
    if not event_files:
        print(f"  (이벤트 파일 없음: {logdir})")
        return

    for ef in sorted(event_files)[-1:]:
        try:
            ea = EventAccumulator(ef)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if not tags:
                continue
            print(f"  Latest event: {Path(ef).name}")
            for tag in tags:
                events = ea.Scalars(tag)
                if events:
                    last = events[-1]
                    print(f"    {tag}: {last.value:.6f} (epoch {last.step})")
        except Exception as e:
            print(f"  (읽기 실패: {e})")


def main():
    config = load_config()
    server_cfg = config.get("logging", {}).get("server", {})

    parser = argparse.ArgumentParser(description="TensorBoard launcher (config/default.yaml)")
    parser.add_argument("--logdir", type=str, default=None, help="직접 logdir 지정")
    parser.add_argument("--experiment", "-e", type=str, default=None,
                        choices=list(EXPERIMENT_MAP.keys()),
                        help="실험 이름 (pretrain / finetune)")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="여러 실험 비교 (이름 나열)")
    parser.add_argument("--port", type=int, default=server_cfg.get("port", 6006))
    parser.add_argument("--summary", action="store_true", help="loss 요약만 출력 (서버 실행 안 함)")
    args = parser.parse_args()

    # logdir 결정
    if args.logdir:
        logdir = args.logdir
    elif args.compare:
        parts = []
        for name in args.compare:
            if name not in EXPERIMENT_MAP:
                print(f"[ERROR] Unknown experiment: {name}")
                print(f"  Available: {list(EXPERIMENT_MAP.keys())}")
                sys.exit(1)
            parts.append(f"{name}:{EXPERIMENT_MAP[name](config)}")
        logdir = ",".join(parts)
    elif args.experiment:
        logdir = EXPERIMENT_MAP[args.experiment](config)
        print(f"Experiment: {args.experiment}")
    else:
        logdir = "weights/"

    print(f"Log directory: {logdir}")
    print()

    # 요약 출력
    if "," in logdir:
        for part in logdir.split(","):
            name, path = part.split(":", 1) if ":" in part else ("", part)
            print(f"[{name or path}]")
            print_summary(path)
    else:
        print_summary(logdir)
    print()

    if args.summary:
        return

    # TensorBoard 실행
    print(f"Starting TensorBoard on port {args.port}...")
    print(f"  http://localhost:{args.port}")
    print()

    tb = program.TensorBoard()
    tb.configure(argv=[
        None,
        "--logdir", logdir,
        "--port", str(args.port),
        "--reload_interval", str(server_cfg.get("reload_interval", 30)),
        "--bind_all",
    ])
    url = tb.launch()
    print(f"TensorBoard started: {url}")

    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")


if __name__ == "__main__":
    main()
