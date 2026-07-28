"""Stage 3: FixMatch-based Self-Training with Geometric Filter.

DOPE 의 unsupervised domain adaptation (synthetic → real). Round 마다:
  1. unlabeled real 에 weak aug + DOPE inference + peak extract + PnP + geometric filter
     → pseudo-label 생성
  2. synthetic (GT) + pseudo-labeled real (strong aug) 혼합 학습

분리된 모듈:
  self_train_data.py    SyntheticDataset / RealUnlabeledDataset / PseudoLabeledDataset
  self_train_pseudo.py  extract_peaks / generate_belief_maps / _apply_filter / generate_pseudo_labels
  self_train_step.py    train_one_epoch (synthetic + pseudo 혼합 학습 step)

사용:
    python scripts/self_training/self_train.py \\
        --config config/stage3_selftrain.yaml \\
        --pretrained weights/pallet_category/net_pallet_best.pth \\
        --synthetic_dir data/pallet/training_data/train \\
        --real_dir data/pallet/real_unlabeled \\
        --output_dir output/stage3_selftrain
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import yaml

# DOPE / self_training 모듈 path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_project_root, "Deep_Object_Pose", "train"))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_project_root, "scripts", "data_prep"))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix
from geometric_filter import GeometricFilter
from augmentations import WeakAugmentation, StrongAugmentation

from self_train_data import (
    SyntheticDataset, RealUnlabeledDataset, PseudoLabeledDataset,
)
from self_train_pseudo import generate_pseudo_labels
from self_train_step import train_one_epoch


def self_training_loop(config, args):
    """Main self-training loop — N round × M epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    st_cfg = config["self_training"]
    output_dir = args.output_dir or config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # --- 모델 로드 ---
    weights_path = args.pretrained or config["paths"]["pretrained_weights"]
    print(f"Loading pre-trained model: {weights_path}")
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)

    # --- PnP solver + geometric filter ---
    cam_cfg = config["camera"]
    camera_matrix = make_camera_matrix(
        cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"])
    pallet_cfg = config["pallet"]
    pnp_solver = PalletPnPSolver(
        camera_matrix,
        pallet_dims=(pallet_cfg["width"], pallet_cfg["depth"], pallet_cfg["height"]),
        use_ransac=config["pnp"]["use_ransac"],
        ransac_reproj_threshold=config["pnp"]["ransac_reproj_threshold"],
        ransac_iterations=config["pnp"]["ransac_iterations"],
    )
    geo_filter = GeometricFilter(pnp_solver, config["geometric_filter"])

    # LOO filter 면 default-ordering PnP 별도 필요
    filter_type = config["geometric_filter"].get("filter_type", "ransac")
    if filter_type in ("ransac_loo", "ransac_loo_flip"):
        loo_solver = PalletPnPSolver(
            camera_matrix,
            pallet_dims=(pallet_cfg["width"], pallet_cfg["depth"], pallet_cfg["height"]))
        config["geometric_filter"]["_loo_solver"] = loo_solver

    # --- Augmentation ---
    weak_aug = WeakAugmentation(
        brightness=config["augmentation"]["weak"]["brightness"],
        contrast=config["augmentation"]["weak"]["contrast"],
        noise_std=config["augmentation"]["weak"]["gaussian_noise_std"],
    )
    strong_aug = StrongAugmentation(config["augmentation"]["strong"])

    # --- Dataset / Loader ---
    syn_dir = args.synthetic_dir or config["paths"]["synthetic_data"]
    real_dir = args.real_dir or config["paths"]["real_unlabeled"]

    syn_dataset = SyntheticDataset(
        syn_dir,
        image_size=st_cfg["image_size"],
        output_size=st_cfg["belief_map_size"],
        sigma=st_cfg["sigma"],
    )
    syn_loader = data.DataLoader(
        syn_dataset, batch_size=st_cfg["batch_size"],
        shuffle=True, num_workers=st_cfg["num_workers"],
        pin_memory=True, drop_last=True,
    )

    real_dataset = RealUnlabeledDataset(real_dir, image_size=st_cfg["image_size"])
    real_loader = data.DataLoader(
        real_dataset, batch_size=st_cfg["batch_size"],
        shuffle=False, num_workers=st_cfg["num_workers"], pin_memory=True,
    )

    # --- Optimizer ---
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=st_cfg["learning_rate"],
        weight_decay=st_cfg.get("weight_decay", 0),
    )

    num_rounds = st_cfg["num_rounds"]
    epochs_per_round = st_cfg["epochs_per_round"]
    convergence_threshold = st_cfg.get("convergence_threshold", 0.01)
    convergence_patience = st_cfg.get("convergence_patience", 3)

    history = []
    acceptance_rates = []
    converged_count = 0

    print(f"\nStarting self-training: {num_rounds} rounds, {epochs_per_round} epochs/round")
    print(f"Synthetic: {len(syn_dataset)} samples, Real unlabeled: {len(real_dataset)} images")
    print("=" * 60)

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")

        # Step 1: pseudo-label 생성
        print(f"Generating pseudo-labels (filter_type={filter_type})...")
        accepted, pl_stats = generate_pseudo_labels(
            model, real_loader, pnp_solver, geo_filter,
            weak_aug, config, device,
        )
        print(f"  Accepted: {pl_stats['accepted']}/{pl_stats['total']} "
              f"({pl_stats['acceptance_rate']:.1%})")
        print(f"  PnP failures: {pl_stats['pnp_fail']}, "
              f"Filter rejections: {pl_stats['filter_fail']}")
        if pl_stats['reproj_error_mean'] > 0:
            print(f"  Mean reproj error: {pl_stats['reproj_error_mean']:.2f} px")

        # Convergence
        acceptance_rates.append(pl_stats['acceptance_rate'])
        if len(acceptance_rates) >= 2:
            rate_change = abs(acceptance_rates[-1] - acceptance_rates[-2])
            if rate_change < convergence_threshold:
                converged_count += 1
                if converged_count >= convergence_patience:
                    print(f"\nConverged: acceptance rate stable for "
                          f"{convergence_patience} rounds.")
                    break
            else:
                converged_count = 0

        # Step 2: 혼합 학습
        pseudo_dataset = PseudoLabeledDataset(
            accepted, image_size=st_cfg["image_size"], strong_aug=strong_aug)
        print(f"Training ({epochs_per_round} epochs, "
              f"{len(syn_dataset)} syn + {len(pseudo_dataset)} pseudo-real)...")

        round_losses = []
        for epoch in range(epochs_per_round):
            epoch_stats = train_one_epoch(
                model, optimizer, syn_loader, pseudo_dataset,
                config, device, epoch, round_idx + 1,
            )
            round_losses.append(epoch_stats)

        round_time = time.time() - round_start
        avg_loss = np.mean([s["loss_total"] for s in round_losses])
        round_result = {
            "round": round_idx + 1,
            "pseudo_labels": pl_stats,
            "avg_loss": float(avg_loss),
            "time_seconds": round_time,
        }
        history.append(round_result)
        print(f"  Avg loss: {avg_loss:.6f}, Time: {round_time:.1f}s")

        # Checkpoint
        if config["logging"].get("save_every_round", True):
            ckpt_path = os.path.join(output_dir, f"round_{round_idx + 1:02d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # 최종 저장
    final_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Self-Training Summary")
    print("=" * 60)
    for r in history:
        print(f"  Round {r['round']:2d}: "
              f"accepted={r['pseudo_labels']['accepted']:4d}/"
              f"{r['pseudo_labels']['total']:4d} "
              f"({r['pseudo_labels']['acceptance_rate']:.1%}), "
              f"loss={r['avg_loss']:.6f}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: FixMatch-based Self-Training with Geometric Filter")
    parser.add_argument("--config", type=str,
                        default="config/stage3_selftrain.yaml")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--synthetic_dir", type=str, default=None)
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--epochs_per_round", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--filter_type", type=str, default=None,
                        choices=["ransac", "bc", "conf", "none", "ransac_loo", "ransac_loo_flip"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading config: {args.config}")
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI override
    if args.num_rounds is not None:
        config["self_training"]["num_rounds"] = args.num_rounds
    if args.epochs_per_round is not None:
        config["self_training"]["epochs_per_round"] = args.epochs_per_round
    if args.lr is not None:
        config["self_training"]["learning_rate"] = args.lr
    if args.filter_type is not None:
        config["geometric_filter"]["filter_type"] = args.filter_type

    self_training_loop(config, args)


if __name__ == "__main__":
    main()
