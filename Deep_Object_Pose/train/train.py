#!/usr/bin/python3

"""
Example usage:

 python -m torch.distributed.launch --nproc_per_node=1 train.py --data ../sample_data/ --object cracker
"""


import argparse
import copy
import datetime
import json
import os
from queue import Queue
import random
import warnings
warnings.filterwarnings("ignore")

try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import sys
sys.path.insert(1, '../common')
from models import *
from utils import *
from geo_loss import GeometricLoss, StructuralLoss, ReliabilityLoss, VisibilityCoordLoss, SpatialSoftArgmax2D
from heatmap_refinement import (
    CornerUncertaintyLoss,
    MaskExtentLoss,
    ProjectedSpanLoss,
    SignedFootprintLoss,
    channel_masked_mse,
    unpack_dope_output,
)
from teacher_constraints import (
    final_belief_distillation_loss,
    mask_extent_per_frame,
    teacher_peak_retention_loss,
    top_fraction_cvar,
)



def _runnetwork(net, optimizer, local_rank, epoch, train_loader, writer=None,
                geo_loss_module=None, geo_lambda=0.0, geo_warmup=5,
                struct_loss_module=None, struct_lambda=1.0, struct_warmup=10,
                rel_loss_module=None, rel_lambda=1.0, rel_warmup=0,
                vis_loss_module=None, vis_lambda=0.005, vis_warmup=0,
                mask_aux=False, mask_weight=0.0, mask_warmup=0,
                corner_quality_loss=None, quality_weight=0.0,
                mask_extent_loss=None, extent_weight=0.0,
                projected_span_loss=None, span_weight=0.0,
                signed_footprint_loss=None, signed_weight=0.0,
                teacher_net=None, teacher_distill_weight=0.0,
                teacher_peak_weight=0.0, teacher_peak_threshold=0.3,
                teacher_peak_margin=0.05, extent_cvar_fraction=1.0,
                refinement_warmup=0, refinement_ramp=500,
                diffpnp_sa=None, diffpnp_loss_module=None, diffpnp_lambda=0.0,
                diffpnp_warmup=0, diffpnp_ramp=500,
                encoder_freeze_steps=0, global_step=0):
    loss_avg_to_log = {}
    loss_avg_to_log["loss"] = []
    loss_avg_to_log["loss_affinities"] = []
    loss_avg_to_log["loss_belief"] = []
    if geo_loss_module is not None:
        loss_avg_to_log["loss_geo"] = []
    if struct_loss_module is not None:
        loss_avg_to_log["loss_struct"] = []
    if rel_loss_module is not None:
        loss_avg_to_log["loss_rel"] = []
    if vis_loss_module is not None:
        loss_avg_to_log["loss_vis"] = []
    if mask_aux:
        loss_avg_to_log["loss_mask"] = []
        loss_avg_to_log["mask_valid_frac"] = []
    if corner_quality_loss is not None:
        loss_avg_to_log["loss_quality"] = []
        loss_avg_to_log["quality_valid_frac"] = []
        loss_avg_to_log["quality_mean_error"] = []
        loss_avg_to_log["quality_mean_sigma"] = []
    if mask_extent_loss is not None:
        loss_avg_to_log["loss_extent"] = []
        loss_avg_to_log["extent_valid_frac"] = []
        loss_avg_to_log["extent_gap_cells"] = []
        loss_avg_to_log["extent_cvar_selected_frac"] = []
    if projected_span_loss is not None:
        loss_avg_to_log["loss_projected_span"] = []
        loss_avg_to_log["span_valid_frac"] = []
        loss_avg_to_log["span_min_ratio"] = []
        loss_avg_to_log["span_under_log"] = []
        loss_avg_to_log["span_coord_cells"] = []
        loss_avg_to_log["span_min_edge_ratio"] = []
        loss_avg_to_log["span_hard_fraction"] = []
    if signed_footprint_loss is not None:
        loss_avg_to_log["loss_signed_footprint"] = []
        loss_avg_to_log["signed_valid_frac"] = []
        loss_avg_to_log["signed_undercoverage"] = []
        loss_avg_to_log["signed_min_edge_ratio"] = []
        loss_avg_to_log["signed_min_radial_ratio"] = []
    if teacher_net is not None:
        loss_avg_to_log["loss_teacher_distill"] = []
        loss_avg_to_log["loss_teacher_peak"] = []
    if diffpnp_loss_module is not None:
        loss_avg_to_log["loss_diffpnp"] = []
        loss_avg_to_log["diffpnp_valid_frac"] = []
        loss_avg_to_log["diffpnp_raw_L"] = []
        loss_avg_to_log["diffpnp_geometry_L"] = []
        loss_avg_to_log["diffpnp_undercoverage_L"] = []
        loss_avg_to_log["diffpnp_min_span_ratio"] = []
        loss_avg_to_log["diffpnp_tz_ratio"] = []
        loss_avg_to_log["diffpnp_hard_fraction"] = []
        loss_avg_to_log["diffpnp_fit_coverage_L"] = []
        loss_avg_to_log["diffpnp_fit_min_span_ratio"] = []
        loss_avg_to_log["diffpnp_fit_hard_fraction"] = []
    _base_net = net.module if hasattr(net, "module") else net
    for batch_idx, targets in enumerate(train_loader):
        # Encoder(VGG) freeze for the first N global steps, then unfreeze once.
        if encoder_freeze_steps > 0:
            should_freeze = global_step < encoder_freeze_steps
            cur = next(_base_net.vgg.parameters()).requires_grad
            if cur == should_freeze:  # state needs toggling
                for p in _base_net.vgg.parameters():
                    p.requires_grad = (not should_freeze)
                if not should_freeze:
                    print(f"[FREEZE] encoder unfrozen at global_step={global_step}")
                elif global_step == 0:
                    print(f"[FREEZE] encoder frozen for first {encoder_freeze_steps} steps")
        optimizer.zero_grad()

        data = Variable(targets["img"].cuda())
        target_belief = Variable(targets["beliefs"].cuda())
        target_affinities = Variable(targets["affinities"].cuda())
        belief_channel_mask = targets["belief_channel_mask"].cuda()
        affinity_channel_mask = targets["affinity_channel_mask"].cuda()

        output_belief, output_aff, _vec, output_seg, output_refinement = \
            unpack_dope_output(net(data))

        teacher_belief = None
        if teacher_net is not None:
            with torch.no_grad():
                teacher_output = teacher_net(data)
                teacher_beliefs, _ta, _tv, _ts, _tr = unpack_dope_output(
                    teacher_output)
                teacher_belief = teacher_beliefs[-1][:, :9].detach()

        # Shape check (first batch of first epoch only)
        if batch_idx == 0 and epoch == 0:
            print("target_belief:", tuple(target_belief.shape))
            print("target_aff   :", tuple(target_affinities.shape))
            print("output_belief:", tuple(output_belief[-1].shape))
            print("output_aff   :", tuple(output_aff[-1].shape))
            assert tuple(output_belief[-1].shape) == tuple(target_belief.shape), \
                f"Belief shape mismatch: output {tuple(output_belief[-1].shape)} vs target {tuple(target_belief.shape)}"
            assert tuple(output_aff[-1].shape) == tuple(target_affinities.shape), \
                f"Affinity shape mismatch: output {tuple(output_aff[-1].shape)} vs target {tuple(target_affinities.shape)}"

        loss = None

        loss_belief = torch.tensor(0).float().cuda()
        loss_affinities = torch.tensor(0).float().cuda()

        for stage in range(len(output_aff)):  # output, each belief map layers.
            loss_affinities += channel_masked_mse(
                output_aff[stage], target_affinities,
                affinity_channel_mask)

            if opt.symmetric_loss:
                # 180° Y-rotation swap: 0↔5, 1↔4, 2↔7, 3↔6, centroid(8) unchanged
                swap_idx = [5, 4, 7, 6, 1, 0, 3, 2, 8]
                target_swapped = target_belief[:, swap_idx]
                mask_swapped = belief_channel_mask[:, swap_idx]
                loss_bel_orig = channel_masked_mse(
                    output_belief[stage], target_belief,
                    belief_channel_mask)
                loss_bel_swap = channel_masked_mse(
                    output_belief[stage], target_swapped, mask_swapped)
                loss_belief += torch.min(loss_bel_orig, loss_bel_swap)
            else:
                loss_belief += channel_masked_mse(
                    output_belief[stage], target_belief,
                    belief_channel_mask)

        loss = loss_affinities + loss_belief

        # Keep the scoped student close to the accepted ep57 teacher while
        # allowing a separate one-sided hard-tail objective to move only the
        # deficient cases.  The teacher and all peak selections are detached.
        loss_teacher_distill = torch.tensor(0.0, device=data.device)
        loss_teacher_peak = torch.tensor(0.0, device=data.device)
        if teacher_belief is not None:
            if teacher_distill_weight > 0:
                raw_teacher_distill = final_belief_distillation_loss(
                    output_belief[-1][:, :9], teacher_belief,
                    channel_mask=belief_channel_mask)
                loss_teacher_distill = (
                    teacher_distill_weight * raw_teacher_distill)
                loss = loss + loss_teacher_distill
            if teacher_peak_weight > 0:
                raw_teacher_peak = teacher_peak_retention_loss(
                    output_belief[-1][:, :9], teacher_belief,
                    channel_validity=belief_channel_mask,
                    teacher_peak_threshold=teacher_peak_threshold,
                    margin=teacher_peak_margin)
                loss_teacher_peak = teacher_peak_weight * raw_teacher_peak
                loss = loss + loss_teacher_peak

        # Geometric loss (soft-argmax + BPnP)
        loss_geo = torch.tensor(0.0).cuda()
        if geo_loss_module is not None and geo_lambda > 0:
            geo_total, geo_dict = geo_loss_module(
                output_belief[-1][:, :9], target_belief[:, :9],
                epoch=epoch, warmup=geo_warmup
            )
            loss_geo = geo_lambda * geo_total
            loss = loss + loss_geo

        # Visibility-aware coordinate loss
        loss_vis = torch.tensor(0.0).cuda()
        if vis_loss_module is not None and vis_lambda > 0:
            vis_weight = targets.get("visibility")
            if vis_weight is not None:
                vis_weight = vis_weight.cuda()
                vis_total, vis_dict = vis_loss_module(
                    output_belief[-1][:, :9], target_belief[:, :9],
                    vis_weight, epoch=epoch, warmup=vis_warmup
                )
                # linear λ ramp over first ~800 steps (avoid early soft-argmax
                # instability jerking belief peaks). No new CLI param.
                _VIS_RAMP = 800.0
                vis_ramp = min(1.0, global_step / _VIS_RAMP)
                loss_vis = vis_lambda * vis_ramp * vis_total
                loss = loss + loss_vis
                if global_step <= 1:
                    _rb = float(loss_belief.item())
                    _rv = float(vis_total.item())
                    _ratio = (vis_lambda * _rv) / max(_rb, 1e-9)
                    print(f"[r_coord] raw vis_total={_rv:.6f} loss_belief={_rb:.6f} "
                          f"lambda={vis_lambda} -> vis/bel contrib ratio={_ratio:.4f} "
                          f"(target 0.05-0.10; lambda for 7.5% = {0.075*_rb/max(_rv,1e-9):.4f})",
                          flush=True)

        # Reliability-aware coordinate loss
        loss_rel = torch.tensor(0.0).cuda()
        if rel_loss_module is not None and rel_lambda > 0:
            rel_total, rel_dict = rel_loss_module(
                output_belief[-1][:, :9], target_belief[:, :9],
                epoch=epoch, warmup=rel_warmup
            )
            loss_rel = rel_lambda * rel_total
            loss = loss + loss_rel

        # Structural loss (flip equivariance + sparse edge + coord Huber)
        loss_struct = torch.tensor(0.0).cuda()
        if struct_loss_module is not None and struct_lambda > 0:
            # Create horizontally flipped input for flip equivariance
            data_flip = torch.flip(data, dims=[-1])  # flip width
            struct_total, struct_dict = struct_loss_module(
                output_belief[-1][:, :9], target_belief[:, :9],
                net=net, data_flip=data_flip,
                epoch=epoch, warmup=struct_warmup
            )
            loss_struct = struct_lambda * struct_total
            loss = loss + loss_struct

        # Mask auxiliary (seg head, BCE). TRAINING-ONLY aux feature; main output
        # stays the belief heatmap (inference unchanged, no hard-gate). Loss is
        # applied per-frame ONLY where a real mask exists (pvnet_mask_valid==1);
        # old base data (no mask_rle) has valid==0 and contributes 0 mask loss.
        loss_mask = torch.tensor(0.0).cuda()
        mask_valid_frac = 0.0
        if mask_aux and output_seg is not None and mask_weight > 0 and epoch >= mask_warmup:
            mask_gt = targets.get("pvnet_mask")
            mask_valid = targets.get("pvnet_mask_valid")
            if mask_gt is not None and mask_valid is not None:
                mask_gt = mask_gt.cuda()                       # (B,1,H,W) {0,1}
                mask_valid = mask_valid.cuda().view(-1)        # (B,)
                mask_valid_frac = float(mask_valid.mean().item())
                vsum = mask_valid.sum()
                if vsum > 0:
                    seg_bce = torch.tensor(0.0).cuda()
                    for seg_stage in output_seg:  # [seg1, seg2] logits (B,1,H,W)
                        per_px = torch.nn.functional.binary_cross_entropy_with_logits(
                            seg_stage, mask_gt, reduction="none")
                        # mean over pixels per frame, then weight by valid, then
                        # normalize by #valid frames (so weight is per-valid-frame).
                        per_frame = per_px.mean(dim=(1, 2, 3))   # (B,)
                        seg_bce = seg_bce + (per_frame * mask_valid).sum() / vsum
                    loss_mask = mask_weight * seg_bce
                    loss = loss + loss_mask

        # Fixed-50 refinement losses. All are independently flag-gated and use
        # a shared step ramp after the optional epoch warmup. The corner sigma
        # target is detached observed decoder error, while mask extent only
        # penalizes under-coverage.
        refinement_scale = 0.0
        if epoch >= refinement_warmup:
            refinement_scale = min(
                1.0, global_step / max(1.0, float(refinement_ramp)))

        loss_quality = torch.tensor(0.0, device=data.device)
        raw_quality = torch.tensor(0.0, device=data.device)
        quality_info = {"valid_frac": 0.0, "mean_error": 0.0, "mean_sigma": 0.0}
        if corner_quality_loss is not None and quality_weight > 0:
            log_sigma = output_refinement.get("corner_log_sigma")
            gt_xy = targets.get("refine_keypoints")
            gt_valid = targets.get("refine_keypoints_valid")
            if log_sigma is not None and gt_xy is not None and gt_valid is not None:
                raw_quality, quality_info = corner_quality_loss(
                    output_belief[-1][:, :9], log_sigma,
                    gt_xy.cuda(), gt_valid.cuda())
                loss_quality = quality_weight * refinement_scale * raw_quality
                loss = loss + loss_quality

        loss_extent = torch.tensor(0.0, device=data.device)
        raw_extent = torch.tensor(0.0, device=data.device)
        extent_info = {
            "valid_frac": 0.0,
            "mean_gap_cells": 0.0,
            "cvar_selected_fraction": 0.0,
        }
        if mask_extent_loss is not None and extent_weight > 0:
            mask_gt = targets.get("pvnet_mask")
            mask_valid = targets.get("pvnet_mask_valid")
            if mask_gt is not None and mask_valid is not None:
                mask_gt_cuda = mask_gt.cuda()
                mask_valid_cuda = mask_valid.cuda()
                if extent_cvar_fraction < 1.0:
                    if teacher_belief is None:
                        raise RuntimeError(
                            "extent CVaR requires a frozen teacher")
                    student_per_frame, student_valid, student_gap = \
                        mask_extent_per_frame(
                            output_belief[-1][:, :8], mask_gt_cuda,
                            mask_valid_cuda, radius=mask_extent_loss.radius,
                            temperature=mask_extent_loss.temperature,
                            tolerance=mask_extent_loss.tolerance)
                    with torch.no_grad():
                        teacher_per_frame, teacher_valid, _teacher_gap = \
                            mask_extent_per_frame(
                                teacher_belief[:, :8], mask_gt_cuda,
                                mask_valid_cuda,
                                radius=mask_extent_loss.radius,
                                temperature=mask_extent_loss.temperature,
                                tolerance=mask_extent_loss.tolerance)
                    cvar_valid = student_valid & teacher_valid
                    raw_extent, cvar_info = top_fraction_cvar(
                        student_per_frame, cvar_valid,
                        extent_cvar_fraction, rank_by=teacher_per_frame)
                    extent_info = {
                        "valid_frac": float(cvar_valid.float().mean().item()),
                        "mean_gap_cells": float(
                            student_gap[cvar_valid].mean().detach().item())
                        if bool(cvar_valid.any()) else 0.0,
                        "cvar_selected_fraction": cvar_info[
                            "selected_fraction"],
                    }
                else:
                    raw_extent, extent_info = mask_extent_loss(
                        output_belief[-1][:, :8],
                        mask_gt_cuda, mask_valid_cuda)
                    extent_info["cvar_selected_fraction"] = 1.0
                loss_extent = extent_weight * refinement_scale * raw_extent
                loss = loss + loss_extent

        loss_projected_span = torch.tensor(0.0, device=data.device)
        raw_projected_span = torch.tensor(0.0, device=data.device)
        span_info = {
            "valid_frac": 0.0,
            "mean_min_span_ratio": 0.0,
            "mean_under_log": 0.0,
            "mean_coord_cells": 0.0,
            "mean_min_edge_ratio": 0.0,
            "hard_fraction": 0.0,
        }
        if projected_span_loss is not None and span_weight > 0:
            gt_xy = targets.get("refine_keypoints")
            gt_valid = targets.get("refine_keypoints_valid")
            if gt_xy is not None and gt_valid is not None:
                raw_projected_span, span_info = projected_span_loss(
                    output_belief[-1][:, :8],
                    gt_xy.cuda(), gt_valid.cuda())
                loss_projected_span = (
                    span_weight * refinement_scale * raw_projected_span)
                loss = loss + loss_projected_span

        loss_signed_footprint = torch.tensor(0.0, device=data.device)
        raw_signed_footprint = torch.tensor(0.0, device=data.device)
        signed_info = {
            "valid_frac": 0.0,
            "mean_undercoverage": 0.0,
            "mean_min_edge_signed_ratio": 0.0,
            "mean_min_radial_signed_ratio": 0.0,
        }
        if signed_footprint_loss is not None and signed_weight > 0:
            gt_xy = targets.get("refine_keypoints")
            gt_valid = targets.get("refine_keypoints_valid")
            if gt_xy is not None and gt_valid is not None:
                raw_signed_footprint, signed_info = signed_footprint_loss(
                    output_belief[-1][:, :8],
                    gt_xy.cuda(), gt_valid.cuda())
                loss_signed_footprint = (
                    signed_weight * refinement_scale * raw_signed_footprint)
                loss = loss + loss_signed_footprint

        if global_step <= 1:
            base = max(float(loss_belief.detach().item()), 1.0e-9)
            for name, raw_value, weight, weighted in (
                    ("quality", raw_quality, quality_weight, loss_quality),
                    ("extent", raw_extent, extent_weight, loss_extent),
                    ("projected_span", raw_projected_span, span_weight,
                     loss_projected_span),
                    ("signed_footprint", raw_signed_footprint, signed_weight,
                     loss_signed_footprint)):
                if ((name == "quality" and corner_quality_loss is not None)
                        or (name == "extent" and mask_extent_loss is not None)
                        or (name == "projected_span"
                            and projected_span_loss is not None)
                        or (name == "signed_footprint"
                            and signed_footprint_loss is not None)):
                    value = float(weighted.detach().item())
                    nominal = float((raw_value.detach() * weight).item())
                    print(f"[REFINE] {name}/belief contribution ratio="
                          f"{value/base:.6f}, nominal_full_ramp={nominal/base:.6f} "
                          f"(weighted={value:.6f}, "
                          f"raw={float(raw_value.detach().item()):.6f}, "
                          f"belief={base:.6f}, ramp={refinement_scale:.4f})",
                          flush=True)

        # DiffPnP3D geometry regularizer (PAPER_S2). local soft-argmax on the 8
        # corner belief channels -> pred_xy(orig px) -> unrolled GN PnP ->
        # 3D-corner Huber(/diag). Applied ONLY to pnp_valid_3d & V8 frames (mask);
        # gradient flows into the belief head. Off/no-valid => contributes 0.
        loss_diffpnp = torch.tensor(0.0).cuda()
        diffpnp_valid_frac = 0.0
        diffpnp_raw_L = 0.0
        diffpnp_geometry_L = 0.0
        diffpnp_undercoverage_L = 0.0
        diffpnp_min_span_ratio = 0.0
        diffpnp_tz_ratio = 0.0
        diffpnp_hard_fraction = 0.0
        diffpnp_fit_coverage_L = 0.0
        diffpnp_fit_min_span_ratio = 0.0
        diffpnp_fit_hard_fraction = 0.0
        if (diffpnp_loss_module is not None and diffpnp_lambda > 0
                and epoch >= diffpnp_warmup):
            dvalid = targets.get("diffpnp_valid")
            if dvalid is not None and float(dvalid.sum().item()) > 0:
                pred_xy, _conf = diffpnp_sa(output_belief[-1][:, :8])   # (B,8,2) orig px
                L_pnp3d, dinfo = diffpnp_loss_module(
                    pred_xy,
                    targets["diffpnp_X"].cuda(),
                    targets["diffpnp_K"].cuda(),
                    targets["diffpnp_R"].cuda(),
                    targets["diffpnp_t"].cuda(),
                    targets["diffpnp_diag"].cuda(),
                    dvalid.cuda().bool(),
                )
                ramp = min(1.0, global_step / max(1.0, float(diffpnp_ramp)))
                loss_diffpnp = diffpnp_lambda * ramp * L_pnp3d
                loss = loss + loss_diffpnp
                diffpnp_valid_frac = dinfo["valid_frac"]
                diffpnp_raw_L = float(L_pnp3d.item())
                diffpnp_geometry_L = dinfo["mean_geometry_L"]
                diffpnp_undercoverage_L = dinfo["mean_undercoverage_L"]
                diffpnp_min_span_ratio = dinfo["mean_min_span_ratio"]
                diffpnp_tz_ratio = dinfo["mean_tz_ratio"]
                diffpnp_hard_fraction = dinfo["hard_fraction"]
                diffpnp_fit_coverage_L = dinfo["mean_fit_coverage_L"]
                diffpnp_fit_min_span_ratio = dinfo[
                    "mean_fit_min_span_ratio"]
                diffpnp_fit_hard_fraction = dinfo["fit_hard_fraction"]

        if batch_idx == 0:
            post = "train"

            if writer is not None and local_rank == 0:
                for i_output in range(1):

                    # input images
                    writer.add_image(
                        f"{post}_input_{i_output}",
                        targets["img_original"][i_output],
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps gt
                    imgs = VisualizeBeliefMap(target_belief[i_output])
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps_gt.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_ground_truth_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps guess
                    imgs = VisualizeBeliefMap(output_belief[-1][i_output])
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_guess_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )


        if not torch.isfinite(loss):
            raise RuntimeError(
                f"[NAN-GUARD] non-finite loss at epoch {epoch} "
                f"batch {batch_idx} global_step {global_step}: {loss.item()}")

        loss.backward()

        optimizer.step()
        global_step += 1

        # log the loss
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())
        if geo_loss_module is not None:
            loss_avg_to_log["loss_geo"].append(loss_geo.item())
        if struct_loss_module is not None:
            loss_avg_to_log["loss_struct"].append(loss_struct.item())
        if rel_loss_module is not None:
            loss_avg_to_log["loss_rel"].append(loss_rel.item())
        if vis_loss_module is not None:
            loss_avg_to_log["loss_vis"].append(loss_vis.item())
        if mask_aux:
            loss_avg_to_log["loss_mask"].append(loss_mask.item())
            loss_avg_to_log["mask_valid_frac"].append(mask_valid_frac)
        if corner_quality_loss is not None:
            loss_avg_to_log["loss_quality"].append(loss_quality.item())
            loss_avg_to_log["quality_valid_frac"].append(quality_info["valid_frac"])
            loss_avg_to_log["quality_mean_error"].append(quality_info["mean_error"])
            loss_avg_to_log["quality_mean_sigma"].append(quality_info["mean_sigma"])
        if mask_extent_loss is not None:
            loss_avg_to_log["loss_extent"].append(loss_extent.item())
            loss_avg_to_log["extent_valid_frac"].append(extent_info["valid_frac"])
            loss_avg_to_log["extent_gap_cells"].append(extent_info["mean_gap_cells"])
            loss_avg_to_log["extent_cvar_selected_frac"].append(
                extent_info["cvar_selected_fraction"])
        if projected_span_loss is not None:
            loss_avg_to_log["loss_projected_span"].append(
                loss_projected_span.item())
            loss_avg_to_log["span_valid_frac"].append(span_info["valid_frac"])
            loss_avg_to_log["span_min_ratio"].append(
                span_info["mean_min_span_ratio"])
            loss_avg_to_log["span_under_log"].append(
                span_info["mean_under_log"])
            loss_avg_to_log["span_coord_cells"].append(
                span_info["mean_coord_cells"])
            loss_avg_to_log["span_min_edge_ratio"].append(
                span_info["mean_min_edge_ratio"])
            loss_avg_to_log["span_hard_fraction"].append(
                span_info["hard_fraction"])
        if signed_footprint_loss is not None:
            loss_avg_to_log["loss_signed_footprint"].append(
                loss_signed_footprint.item())
            loss_avg_to_log["signed_valid_frac"].append(
                signed_info["valid_frac"])
            loss_avg_to_log["signed_undercoverage"].append(
                signed_info["mean_undercoverage"])
            loss_avg_to_log["signed_min_edge_ratio"].append(
                signed_info["mean_min_edge_signed_ratio"])
            loss_avg_to_log["signed_min_radial_ratio"].append(
                signed_info["mean_min_radial_signed_ratio"])
        if teacher_net is not None:
            loss_avg_to_log["loss_teacher_distill"].append(
                loss_teacher_distill.item())
            loss_avg_to_log["loss_teacher_peak"].append(
                loss_teacher_peak.item())
        if diffpnp_loss_module is not None:
            loss_avg_to_log["loss_diffpnp"].append(loss_diffpnp.item())
            loss_avg_to_log["diffpnp_valid_frac"].append(diffpnp_valid_frac)
            loss_avg_to_log["diffpnp_raw_L"].append(diffpnp_raw_L)
            loss_avg_to_log["diffpnp_geometry_L"].append(diffpnp_geometry_L)
            loss_avg_to_log["diffpnp_undercoverage_L"].append(
                diffpnp_undercoverage_L)
            loss_avg_to_log["diffpnp_min_span_ratio"].append(
                diffpnp_min_span_ratio)
            loss_avg_to_log["diffpnp_tz_ratio"].append(diffpnp_tz_ratio)
            loss_avg_to_log["diffpnp_hard_fraction"].append(
                diffpnp_hard_fraction)
            loss_avg_to_log["diffpnp_fit_coverage_L"].append(
                diffpnp_fit_coverage_L)
            loss_avg_to_log["diffpnp_fit_min_span_ratio"].append(
                diffpnp_fit_min_span_ratio)
            loss_avg_to_log["diffpnp_fit_hard_fraction"].append(
                diffpnp_fit_hard_fraction)

        # Belief peak health (every batch)
        with torch.no_grad():
            peak_vals = output_belief[-1][:, :9].view(output_belief[-1].shape[0], 9, -1).max(dim=-1).values
            if "belief_peak" not in loss_avg_to_log:
                loss_avg_to_log["belief_peak"] = []
            loss_avg_to_log["belief_peak"].append(peak_vals.mean().item())

        if batch_idx % opt.loginterval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.15f} \tLocal Rank: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    local_rank,
                )
            )

    # log the loss values
    if writer is not None and local_rank == 0:
        mean_bel = np.mean(loss_avg_to_log["loss_belief"])
        writer.add_scalar(
            "loss/train_loss", np.mean(loss_avg_to_log["loss"]), epoch
        )
        writer.add_scalar(
            "loss/train_aff", np.mean(loss_avg_to_log["loss_affinities"]), epoch
        )
        writer.add_scalar(
            "loss/train_bel", mean_bel, epoch
        )
        if "loss_geo" in loss_avg_to_log and loss_avg_to_log["loss_geo"]:
            mean_geo = np.mean(loss_avg_to_log["loss_geo"])
            writer.add_scalar("loss/train_geo", mean_geo, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_geo_bel", mean_geo / mean_bel, epoch)
        if "loss_struct" in loss_avg_to_log and loss_avg_to_log["loss_struct"]:
            mean_struct = np.mean(loss_avg_to_log["loss_struct"])
            writer.add_scalar("loss/train_struct", mean_struct, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_struct_bel", mean_struct / mean_bel, epoch)
        if "loss_rel" in loss_avg_to_log and loss_avg_to_log["loss_rel"]:
            mean_rel = np.mean(loss_avg_to_log["loss_rel"])
            writer.add_scalar("loss/train_rel", mean_rel, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_rel_bel", mean_rel / mean_bel, epoch)
        if "loss_vis" in loss_avg_to_log and loss_avg_to_log["loss_vis"]:
            mean_vis = np.mean(loss_avg_to_log["loss_vis"])
            writer.add_scalar("loss/train_vis", mean_vis, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_vis_bel", mean_vis / mean_bel, epoch)

        if "loss_mask" in loss_avg_to_log and loss_avg_to_log["loss_mask"]:
            mean_mask = np.mean(loss_avg_to_log["loss_mask"])
            writer.add_scalar("loss/train_mask", mean_mask, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_mask_bel", mean_mask / mean_bel, epoch)
            writer.add_scalar("health/mask_valid_frac",
                              np.mean(loss_avg_to_log["mask_valid_frac"]), epoch)

        if "loss_quality" in loss_avg_to_log and loss_avg_to_log["loss_quality"]:
            mean_quality = np.mean(loss_avg_to_log["loss_quality"])
            writer.add_scalar("loss/train_corner_quality", mean_quality, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_quality_bel",
                                  mean_quality / mean_bel, epoch)
            writer.add_scalar("health/quality_valid_frac",
                              np.mean(loss_avg_to_log["quality_valid_frac"]), epoch)
            writer.add_scalar("health/quality_error_cells",
                              np.mean(loss_avg_to_log["quality_mean_error"]), epoch)
            writer.add_scalar("health/quality_sigma_cells",
                              np.mean(loss_avg_to_log["quality_mean_sigma"]), epoch)

        if "loss_extent" in loss_avg_to_log and loss_avg_to_log["loss_extent"]:
            mean_extent = np.mean(loss_avg_to_log["loss_extent"])
            writer.add_scalar("loss/train_mask_extent", mean_extent, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_extent_bel",
                                  mean_extent / mean_bel, epoch)
            writer.add_scalar("health/extent_valid_frac",
                              np.mean(loss_avg_to_log["extent_valid_frac"]), epoch)
            writer.add_scalar("health/extent_gap_cells",
                              np.mean(loss_avg_to_log["extent_gap_cells"]), epoch)
            writer.add_scalar(
                "health/extent_cvar_selected_frac",
                np.mean(loss_avg_to_log["extent_cvar_selected_frac"]), epoch)

        if ("loss_projected_span" in loss_avg_to_log
                and loss_avg_to_log["loss_projected_span"]):
            mean_span = np.mean(loss_avg_to_log["loss_projected_span"])
            writer.add_scalar("loss/train_projected_span", mean_span, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_projected_span_bel",
                                  mean_span / mean_bel, epoch)
            writer.add_scalar("health/span_valid_frac",
                              np.mean(loss_avg_to_log["span_valid_frac"]), epoch)
            writer.add_scalar("health/span_min_ratio",
                              np.mean(loss_avg_to_log["span_min_ratio"]), epoch)
            writer.add_scalar("health/span_under_log",
                              np.mean(loss_avg_to_log["span_under_log"]), epoch)
            writer.add_scalar("health/span_coord_cells",
                              np.mean(loss_avg_to_log["span_coord_cells"]), epoch)
            writer.add_scalar("health/span_min_edge_ratio",
                              np.mean(loss_avg_to_log["span_min_edge_ratio"]), epoch)
            writer.add_scalar("health/span_hard_fraction",
                              np.mean(loss_avg_to_log["span_hard_fraction"]), epoch)

        if ("loss_signed_footprint" in loss_avg_to_log
                and loss_avg_to_log["loss_signed_footprint"]):
            mean_signed = np.mean(loss_avg_to_log["loss_signed_footprint"])
            writer.add_scalar(
                "loss/train_signed_footprint", mean_signed, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar(
                    "loss/ratio_signed_footprint_bel",
                    mean_signed / mean_bel, epoch)
            writer.add_scalar(
                "health/signed_valid_frac",
                np.mean(loss_avg_to_log["signed_valid_frac"]), epoch)
            writer.add_scalar(
                "health/signed_undercoverage",
                np.mean(loss_avg_to_log["signed_undercoverage"]), epoch)
            writer.add_scalar(
                "health/signed_min_edge_ratio",
                np.mean(loss_avg_to_log["signed_min_edge_ratio"]), epoch)
            writer.add_scalar(
                "health/signed_min_radial_ratio",
                np.mean(loss_avg_to_log["signed_min_radial_ratio"]), epoch)

        if "loss_teacher_distill" in loss_avg_to_log:
            mean_distill = np.mean(loss_avg_to_log["loss_teacher_distill"])
            mean_peak = np.mean(loss_avg_to_log["loss_teacher_peak"])
            writer.add_scalar("loss/train_teacher_distill", mean_distill, epoch)
            writer.add_scalar("loss/train_teacher_peak", mean_peak, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar(
                    "loss/ratio_teacher_distill_bel",
                    mean_distill / mean_bel, epoch)
                writer.add_scalar(
                    "loss/ratio_teacher_peak_bel",
                    mean_peak / mean_bel, epoch)

        if "loss_diffpnp" in loss_avg_to_log and loss_avg_to_log["loss_diffpnp"]:
            mean_dp = np.mean(loss_avg_to_log["loss_diffpnp"])
            writer.add_scalar("loss/train_diffpnp", mean_dp, epoch)
            writer.add_scalar("loss/train_diffpnp_raw",
                              np.mean(loss_avg_to_log["diffpnp_raw_L"]), epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_diffpnp_bel", mean_dp / mean_bel, epoch)
            writer.add_scalar("health/diffpnp_valid_frac",
                              np.mean(loss_avg_to_log["diffpnp_valid_frac"]), epoch)
            writer.add_scalar("health/diffpnp_geometry_L",
                              np.mean(loss_avg_to_log["diffpnp_geometry_L"]), epoch)
            writer.add_scalar(
                "health/diffpnp_undercoverage_L",
                np.mean(loss_avg_to_log["diffpnp_undercoverage_L"]), epoch)
            writer.add_scalar(
                "health/diffpnp_min_span_ratio",
                np.mean(loss_avg_to_log["diffpnp_min_span_ratio"]), epoch)
            writer.add_scalar("health/diffpnp_tz_ratio",
                              np.mean(loss_avg_to_log["diffpnp_tz_ratio"]), epoch)
            writer.add_scalar(
                "health/diffpnp_hard_fraction",
                np.mean(loss_avg_to_log["diffpnp_hard_fraction"]), epoch)
            writer.add_scalar(
                "health/diffpnp_fit_coverage_L",
                np.mean(loss_avg_to_log["diffpnp_fit_coverage_L"]), epoch)
            writer.add_scalar(
                "health/diffpnp_fit_min_span_ratio",
                np.mean(loss_avg_to_log[
                    "diffpnp_fit_min_span_ratio"]), epoch)
            writer.add_scalar(
                "health/diffpnp_fit_hard_fraction",
                np.mean(loss_avg_to_log[
                    "diffpnp_fit_hard_fraction"]), epoch)

        # Belief peak health monitoring
        if "belief_peak" in loss_avg_to_log and loss_avg_to_log["belief_peak"]:
            writer.add_scalar("health/belief_peak_mean",
                              np.mean(loss_avg_to_log["belief_peak"]), epoch)

    return global_step


def main(opt):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.gradcheck = False
    torch.backends.cudnn.benchmark = True

    local_rank = opt.local_rank

    # Validate Arguments
    if opt.use_s3 and (opt.train_buckets is None or opt.endpoint is None):
        raise ValueError(
            "--train_buckets and --endpoint must be specified if training with data from s3 bucket."
        )

    if not opt.use_s3 and opt.data is None:
        raise ValueError("--data field must be specified.")

    os.makedirs(opt.outf, exist_ok=True)

    random_seed = random.randint(1, 10000)
    if opt.manualseed is not None:
        random_seed = opt.manualseed

    # Save run parameters in a file
    with open(opt.outf + "/header.txt", "w") as file:
        file.write(str(opt) + "\n")
        file.write("seed: " + str(random_seed) + "\n")

    writer = None
    if local_rank == 0:
        writer = SummaryWriter(opt.outf + "/runs/")

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.cuda.set_device(local_rank)
    # Windows 단일 GPU: distributed 우회
    if os.name == 'nt' or int(os.environ.get('WORLD_SIZE', '1')) <= 1:
        pass  # skip distributed init
    else:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")


    # Data Augmentation
    transform = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.ToTensor()
    ])

    # Fixed-400/50 enhancement switches. The bundle is convenience only; each
    # component remains independently ablatable. Mask fusion/extent imply the
    # existing real-mask auxiliary path but never introduce a hard inference
    # gate.
    enhance_bundle = bool(getattr(opt, "heatmap_pnp_enhance", False))
    use_clip_belief_border = bool(
        getattr(opt, "clip_belief_border", False) or enhance_bundle)
    use_mask_belief_fusion = bool(
        getattr(opt, "mask_belief_fusion", False) or enhance_bundle)
    use_extent_loss = bool(getattr(opt, "extent_loss", False) or enhance_bundle)
    use_corner_quality = bool(
        getattr(opt, "corner_quality", False) or enhance_bundle)
    use_projected_span_loss = bool(
        getattr(opt, "projected_span_loss", False))
    use_signed_footprint_loss = bool(
        getattr(opt, "signed_footprint_loss", False))
    use_diffpnp_tail_loss = bool(
        getattr(opt, "diffpnp", False)
        and getattr(opt, "diffpnp_lambda", 0.0) > 0
        and (getattr(opt, "diffpnp_geometry_weight", 1.0) > 0
             or getattr(opt, "diffpnp_undercoverage_weight", 0.0) > 0
             or getattr(opt, "diffpnp_fit_coverage_weight", 0.0) > 0))
    use_teacher_constraints = bool(
        getattr(opt, "teacher_distill_weight", 0.0) > 0
        or getattr(opt, "teacher_peak_weight", 0.0) > 0
        or getattr(opt, "extent_cvar_fraction", 1.0) < 1.0)
    trainable_scope = getattr(opt, "trainable_scope", "all")
    train_only_mask_fusion = trainable_scope == "mask_fusion"
    train_only_belief_tail = trainable_scope == "belief_tail"
    scoped_training = trainable_scope != "all"
    use_mask_aux = bool(
        getattr(opt, "mask_aux", False)
        or use_mask_belief_fusion
        or use_extent_loss)
    if (enhance_bundle or use_clip_belief_border or use_mask_belief_fusion
            or use_extent_loss or use_corner_quality
            or use_projected_span_loss or use_signed_footprint_loss) \
            and opt.imagesize != 400:
        raise ValueError(
            "fixed-grid enhancements require --imagesize 400 (belief stays 50x50)")
    if scoped_training and opt.net_path is None:
        raise ValueError(
            f"--trainable_scope {trainable_scope} requires a frozen base "
            "checkpoint via --net_path")
    if train_only_mask_fusion:
        if not use_mask_belief_fusion:
            raise ValueError(
                "--trainable_scope mask_fusion requires --mask_belief_fusion")
        if use_corner_quality or use_clip_belief_border:
            raise ValueError(
                "--trainable_scope mask_fusion is restricted to isolated "
                "mask-fusion refinement losses")
    if train_only_belief_tail:
        if not (use_extent_loss or use_projected_span_loss
                or use_signed_footprint_loss or use_diffpnp_tail_loss):
            raise ValueError(
                "--trainable_scope belief_tail requires "
                "an extent/signed-span or enabled DiffPnP loss")
        if (use_mask_belief_fusion or use_corner_quality
                or use_clip_belief_border):
            raise ValueError(
                "--trainable_scope belief_tail is restricted to direct "
                "extent/footprint/DiffPnP refinement")
    if scoped_training:
        if getattr(opt, "encoder_freeze_steps", 0):
            raise ValueError(
                f"--trainable_scope {trainable_scope} cannot be combined with "
                "--encoder_freeze_steps")
        enc_scale = getattr(opt, "encoder_lr_scale", None)
        if enc_scale is not None and enc_scale != 1.0:
            raise ValueError(
                f"--trainable_scope {trainable_scope} cannot be combined with "
                "--encoder_lr_scale")
    if use_teacher_constraints:
        if opt.net_path is None:
            raise ValueError("teacher constraints require --net_path")
        if not getattr(opt, "teacher_checkpoint", None):
            raise ValueError(
                "teacher constraints require --teacher_checkpoint")
        if (os.path.realpath(opt.teacher_checkpoint)
                != os.path.realpath(opt.net_path)):
            raise ValueError(
                "teacher checkpoint must equal the locked initialization "
                "checkpoint for this anti-forgetting experiment")
    if not 0.0 < getattr(opt, "extent_cvar_fraction", 1.0) <= 1.0:
        raise ValueError("--extent_cvar_fraction must be in (0,1]")
    if opt.extent_cvar_fraction < 1.0 and not use_extent_loss:
        raise ValueError("--extent_cvar_fraction <1 requires --extent_loss")
    if min(opt.teacher_distill_weight, opt.teacher_peak_weight) < 0:
        raise ValueError("teacher constraint weights must be non-negative")
    if opt.teacher_peak_margin < 0:
        raise ValueError("--teacher_peak_margin must be non-negative")

    # Load Model
    # B2 mask auxiliary (STEP13): add a flag-gated seg head (numSeg=1, BCE). The
    # seg head shares VGG features; it is a TRAINING-ONLY auxiliary feature. The
    # belief(heatmap) head stays the main output and inference decodes the belief
    # peak unchanged (NO mask hard-gate). numSeg=0 default => byte-identical model.
    net = DopeNetwork(
        numSeg=1 if use_mask_aux else 0,
        maskBeliefFusion=use_mask_belief_fusion,
        cornerQuality=use_corner_quality,
    )
    if scoped_training:
        # Preserve the deployed ep57 representation exactly.  Only the small
        # zero-init residual fusion branch may change; the VGG, belief,
        # affinity, and segmentation paths remain immutable.  Configure this
        # before DDP/optimizer creation so distributed runs see the same
        # trainable parameter set.
        for parameter in net.parameters():
            parameter.requires_grad = False
        if train_only_mask_fusion:
            for parameter in net.m_mask_belief_fusion.parameters():
                parameter.requires_grad = True
            expected_prefixes = ("m_mask_belief_fusion.",)
            expected_tensors = 4
            expected_parameters = 74953
        else:
            # The final two 1x1 convolutions can recalibrate channel-ordered
            # corner peaks without changing the encoder or the first five
            # belief/affinity stages.
            for layer_index in (10, 12):
                for parameter in net.m6_2[layer_index].parameters():
                    parameter.requires_grad = True
            expected_prefixes = ("m6_2.10.", "m6_2.12.")
            expected_tensors = 4
            expected_parameters = 17673
        trainable = [
            (name, parameter) for name, parameter in net.named_parameters()
            if parameter.requires_grad
        ]
        trainable_names = [name for name, _ in trainable]
        trainable_count = sum(parameter.numel() for _, parameter in trainable)
        assert trainable_names and all(
            name.startswith(expected_prefixes) for name in trainable_names)
        assert len(trainable_names) == expected_tensors, trainable_names
        assert trainable_count == expected_parameters, trainable_count
        print(
            f"[SCOPED:{trainable_scope}] frozen core; trainable tensors="
            f"{len(trainable_names)} params={trainable_count} "
            f"names={trainable_names}")
    output_size = 50
    # sigma is controlled via --sigma CLI argument (default: 4.0)

    # Convert object names to lower-case for comparison later
    for idx in range(len(opt.object)):
        opt.object[idx] = opt.object[idx].lower()

    # DiffPnP3D (PAPER_S2): load the per-frame audit index (abs json path -> entry)
    # and enable aspect-resize so belief<->orig is a fixed anisotropic scale.
    diffpnp_index = None
    if getattr(opt, "diffpnp", False):
        import glob as _glob
        diffpnp_index = {}
        for jf in _glob.glob(os.path.join(opt.diffpnp_index_dir, "*.json")):
            idx = json.load(open(jf))
            for rel, entry in idx.items():
                diffpnp_index[os.path.abspath(os.path.join(opt.diffpnp_root, rel))] = entry
        n_valid = sum(1 for e in diffpnp_index.values()
                      if e.get("pnp_valid_3d") and e.get("V8"))
        print(f"[DIFFPNP] index loaded: {len(diffpnp_index)} frames, "
              f"{n_valid} pnp_valid_3d&V8 (aspect_resize ON, rotate skipped on "
              f"eligible frames)")

    training_dataset = CleanVisiiDopeLoader(
        opt.data,
        sigma=opt.sigma,
        output_size=output_size,
        objects=opt.object,
        use_s3=opt.use_s3,
        buckets=opt.train_buckets,
        endpoint_url=opt.endpoint,
        truncation_aug_prob=opt.truncation_aug_prob,
        mask_aux=use_mask_aux,
        clip_belief_border=use_clip_belief_border,
        refinement_targets=(use_corner_quality or use_projected_span_loss
                            or use_signed_footprint_loss),
        # Keep deterministic squash preprocessing available independently of
        # DiffPnP.  Self-training can therefore preserve pseudo-label geometry
        # without loading any PnP targets or adding a PnP loss.
        aspect_resize=(getattr(opt, "aspect_resize", False)
                       or getattr(opt, "diffpnp", False)),
        diffpnp_index=diffpnp_index,
    )
    # Optional group-balanced sampling: give equal total draw-weight to samples
    # whose image path contains opt.balance_substr vs the rest. Used to keep a
    # smaller add-on set (e.g. addon_v1) from being drowned out by a larger base
    # set (e.g. v3). Default None => original shuffle behaviour (unchanged).
    train_sampler = None
    # Keep the sampled frame order and worker-side augmentation RNG identical
    # across ablation arms. Optional heads consume different amounts of the
    # global torch RNG during construction, so the sampler/loader must not
    # inherit that mutable state.
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(random_seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(random_seed)
    _epoch_n = getattr(opt, "epoch_size", None)
    if not _epoch_n or _epoch_n <= 0:
        _epoch_n = len(training_dataset)   # full dataset per epoch (default)
    if getattr(opt, "balance_groups", None):
        # N-way ratio sampler. Spec: "g1sub1|g1sub2:r1,g2sub:r2,...".
        # Each comma-separated group = one or more '|'-joined path substrings and
        # a ratio. A group's total draw-weight == its ratio, distributed
        # uniformly per image inside the group (so larger sub-sources within a
        # group are sampled proportionally more often). A sample belongs to the
        # first group whose any substring matches. Unmatched samples are excluded
        # (weight 0). Default None => unchanged.
        import numpy as _np
        paths = [t[0] for t in training_dataset.imgs]
        specs = []  # list of (label, [substrs], ratio)
        for tok in opt.balance_groups.split(","):
            tok = tok.strip()
            if not tok:
                continue
            subs_str, _, wstr = tok.rpartition(":")
            subs = [s for s in subs_str.split("|") if s]
            specs.append((subs_str, subs, float(wstr)))
        w = _np.zeros(len(paths), dtype=_np.float64)
        assigned = _np.zeros(len(paths), dtype=bool)
        counts = {}
        for label, subs, ratio in specs:
            grp = _np.array(
                [(not assigned[i]) and any(s in p for s in subs)
                 for i, p in enumerate(paths)], dtype=bool)
            n_g = int(grp.sum())
            counts[label] = n_g
            if n_g > 0 and ratio > 0:
                w[grp] = ratio / n_g
            assigned |= grp
        n_unmatched = int((~assigned).sum())
        if w.sum() <= 0:
            print(f"[BALANCE-N] WARN no group matched ({counts}); falling back to shuffle")
        else:
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(w, dtype=torch.double),
                num_samples=_epoch_n,
                replacement=True,
                generator=sampler_generator,
            )
            ratios = {l: r for l, _, r in specs}
            print(f"[BALANCE-N] ratio sampler ratios={ratios} counts={counts} "
                  f"unmatched={n_unmatched} epoch_size={_epoch_n}")
    elif getattr(opt, "balance_substr", None):
        import numpy as _np
        paths = [t[0] for t in training_dataset.imgs]
        in_grp = _np.array([opt.balance_substr in p for p in paths], dtype=bool)
        n_in, n_out = int(in_grp.sum()), int((~in_grp).sum())
        if n_in == 0 or n_out == 0:
            print(f"[BALANCE] WARN substr '{opt.balance_substr}' -> in={n_in} "
                  f"out={n_out}; falling back to shuffle")
        else:
            w = _np.where(in_grp, 1.0 / n_in, 1.0 / n_out)
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(w, dtype=torch.double),
                num_samples=_epoch_n,
                replacement=True,
                generator=sampler_generator,
            )
            print(f"[BALANCE] 1:1 group sampling on substr '{opt.balance_substr}': "
                  f"in={n_in} out={n_out}, epoch_size={_epoch_n}")
    training_data = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt.batchsize,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.workers,
        pin_memory=True,
        generator=loader_generator,
    )
    print(f"[REPRO] sampler/loader generators fixed at seed={random_seed}")

    if not training_data is None:
        print("training data: {} batches".format(len(training_data)))

        print("Loading Model...")
        if os.name == 'nt' or int(os.environ.get('WORLD_SIZE', '1')) <= 1:
            net = net.cuda()
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[local_rank],
                output_device=local_rank
            )

    # Load any previous checkpoint (i.e. current job is a follow-up job)
    if opt.net_path is not None:
        # A base/ep57 checkpoint legitimately lacks only the newly enabled
        # heads. Audit every missing prefix so strict=False cannot hide a typo or
        # wrong backbone. DDP and plain checkpoints are both accepted.
        state = torch.load(opt.net_path, map_location="cpu")
        if any(k.startswith("module.") for k in state):
            state = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state.items()
            }
        _bn = net.module if hasattr(net, "module") else net
        missing, unexpected = _bn.load_state_dict(state, strict=False)
        allowed_missing_prefixes = []
        if use_mask_aux:
            allowed_missing_prefixes += ["m_seg1.", "m_seg2."]
        if use_mask_belief_fusion:
            allowed_missing_prefixes += ["m_mask_belief_fusion."]
        if use_corner_quality:
            allowed_missing_prefixes += ["m_corner_quality."]
        bad_missing = [
            key for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        assert not bad_missing, f"unexpected checkpoint missing keys: {bad_missing}"
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
        print(f"[CKPT] strict=False audited: missing={len(missing)} "
              f"(allowed new heads only), unexpected=0")

    teacher_net = None
    if use_teacher_constraints:
        # Copy only after the locked ep57 checkpoint has been loaded.  The
        # independent module is excluded from the optimizer and remains eval;
        # no teacher activation graph is retained in _runnetwork.
        _student_base = net.module if hasattr(net, "module") else net
        teacher_net = copy.deepcopy(_student_base).cuda().eval()
        for parameter in teacher_net.parameters():
            parameter.requires_grad = False
        print(
            "[TEACHER] frozen initialization checkpoint copied for belief "
            f"constraints: {opt.teacher_checkpoint}")

    enc_scale = getattr(opt, "encoder_lr_scale", None)
    if enc_scale is not None and enc_scale != 1.0:
        # Discriminative LR: encoder(VGG) at lr*scale, heads at lr.
        base_net = net.module if hasattr(net, "module") else net
        enc_params = list(base_net.vgg.parameters())
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in base_net.parameters() if id(p) not in enc_ids]
        optimizer = optim.Adam([
            {"params": [p for p in enc_params if p.requires_grad], "lr": opt.lr * enc_scale},
            {"params": [p for p in head_params if p.requires_grad], "lr": opt.lr},
        ])
        print(f"[LR] discriminative: encoder lr={opt.lr * enc_scale:g} "
              f"(scale={enc_scale}), head lr={opt.lr:g}")
    else:
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.Adam(parameters, lr=opt.lr)

    print("ready to train!")
    start_time = datetime.datetime.now()
    print("start:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

    ckpt_q = None
    if opt.nb_checkpoints > 0:
        ckpt_q = Queue(maxsize=opt.nb_checkpoints)

    start_epoch = 0
    if opt.net_path is not None:
        # We started with a saved checkpoint, we start numbering checkpoints
        # after the loaded one
        try:
            start_epoch = int(os.path.splitext(os.path.basename(opt.net_path).split('_')[-1])[0]) + 1
        except:
            start_epoch = 1
        print(f"Starting at epoch {start_epoch}")

    # Geometric loss setup
    geo_loss_module = None
    if opt.geo_loss:
        from geo_loss import GeometricLoss
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'self_training'))
        from pnp_solver import make_pallet_keypoints_3d, make_camera_matrix
        kp3d = make_pallet_keypoints_3d(1.1, 1.1, 0.15)
        K = make_camera_matrix(opt.geo_fx, opt.geo_fy, opt.geo_cx, opt.geo_cy)
        geo_loss_module = GeometricLoss(
            kp3d, K,
            belief_size=output_size,
            input_size=opt.imagesize,
            orig_size=(opt.geo_img_w, opt.geo_img_h),
            temperature=opt.geo_temperature,
        ).cuda()
        print(f"[GEO] Geometric loss enabled (lambda={opt.geo_lambda}, warmup={opt.geo_warmup})")

    # Visibility-aware coordinate loss setup
    vis_loss_module = None
    if opt.vis_coord_loss:
        from geo_loss import VisibilityCoordLoss
        vis_loss_module = VisibilityCoordLoss(
            temperature=opt.geo_temperature,
            delta=0.03,
        ).cuda()
        print(f"[VIS] Visibility coord loss enabled (lambda={opt.vis_lambda}, warmup={opt.vis_warmup})")

    # Reliability loss setup
    rel_loss_module = None
    if opt.rel_loss:
        from geo_loss import ReliabilityLoss
        rel_loss_module = ReliabilityLoss(
            temperature=opt.geo_temperature,
            delta=opt.rel_delta,
            lambda_log=opt.rel_lambda_log,
        ).cuda()
        print(f"[REL] Reliability loss enabled (lambda={opt.rel_lambda}, "
              f"warmup={opt.rel_warmup}, delta={opt.rel_delta}, "
              f"lambda_log={opt.rel_lambda_log})")

    # Structural loss setup
    struct_loss_module = None
    if opt.struct_loss:
        from geo_loss import StructuralLoss, SpatialSoftArgmax2D
        soft_argmax = SpatialSoftArgmax2D(temperature=opt.geo_temperature)
        struct_lambdas = {
            'flip': opt.struct_flip,
            'edge': opt.struct_edge,
            'coord': opt.struct_coord,
            'vp': opt.struct_vp,
        }
        struct_loss_module = StructuralLoss(
            soft_argmax, lambdas=struct_lambdas, delta=opt.struct_delta
        ).cuda()
        print(f"[STRUCT] Structural loss enabled (lambda={opt.struct_lambda}, "
              f"warmup={opt.struct_warmup}, flip={opt.struct_flip}, "
              f"edge={opt.struct_edge}, coord={opt.struct_coord}, "
              f"vp={opt.struct_vp})")

    corner_quality_loss = None
    if use_corner_quality:
        corner_quality_loss = CornerUncertaintyLoss(
            min_sigma=0.25, max_sigma=20.0, window=11).cuda()
        print(f"[QUALITY] corner log-sigma calibration enabled "
              f"(weight={opt.quality_weight}, units=50-grid cells)")

    mask_extent_loss = None
    if use_extent_loss:
        mask_extent_loss = MaskExtentLoss(
            radius=opt.extent_radius,
            temperature=opt.extent_temperature,
            tolerance=opt.extent_tolerance,
        ).cuda()
        print(f"[EXTENT] local-softargmax mask under-coverage enabled "
              f"(weight={opt.extent_weight}, radius={opt.extent_radius}, "
              f"temp={opt.extent_temperature}, tol={opt.extent_tolerance})")

    projected_span_loss = None
    if use_projected_span_loss:
        projected_span_loss = ProjectedSpanLoss(
            window=opt.span_window,
            offset=opt.span_decoder_offset,
            smooth_sigma=opt.span_smooth_sigma,
            interior_margin=opt.span_interior_margin,
            min_span=opt.span_min_size,
            coord_weight=opt.span_coord_weight,
            overshoot_weight=opt.span_overshoot_weight,
            overshoot_ratio=opt.span_overshoot_ratio,
            huber_delta=opt.span_huber_delta,
            footprint_edge_weight=opt.span_footprint_edge_weight,
            hard_edge_threshold=opt.span_hard_edge_threshold,
            hard_example_gain=opt.span_hard_example_gain,
            min_edge_length=opt.span_min_edge_length,
        ).cuda()
        print(
            "[PROJECTED-SPAN] deployed-centroid 8-corner GT footprint "
            f"enabled (weight={opt.span_weight}, window={opt.span_window}, "
            f"coord={opt.span_coord_weight}, "
            f"edge={opt.span_footprint_edge_weight}, "
            f"hard={opt.span_hard_example_gain}x<"
            f"{opt.span_hard_edge_threshold}, "
            f"over={opt.span_overshoot_weight}@{opt.span_overshoot_ratio}, "
            f"margin={opt.span_interior_margin}, "
            f"delta={opt.span_huber_delta})")

    signed_footprint_loss = None
    if use_signed_footprint_loss:
        signed_footprint_loss = SignedFootprintLoss(
            window=opt.span_window,
            offset=opt.span_decoder_offset,
            smooth_sigma=opt.span_smooth_sigma,
            interior_margin=opt.span_interior_margin,
            min_edge_length=opt.span_min_edge_length,
            min_radial_length=opt.signed_min_radial_length,
            width_weight=opt.signed_width_weight,
            depth_weight=opt.signed_depth_weight,
            radial_weight=opt.signed_radial_weight,
            overshoot_weight=opt.signed_overshoot_weight,
            overshoot_ratio=opt.signed_overshoot_ratio,
            huber_delta=opt.signed_huber_delta,
            grid_size=50,
        ).cuda()
        print(
            "[SIGNED-FOOTPRINT] GT-direction edge/radial under-coverage "
            f"enabled (weight={opt.signed_weight}, "
            f"components=W{opt.signed_width_weight}/D"
            f"{opt.signed_depth_weight}/R{opt.signed_radial_weight}, "
            f"over={opt.signed_overshoot_weight}@"
            f"{opt.signed_overshoot_ratio})")

    # DiffPnP3D loss modules (PAPER_S2)
    diffpnp_sa = None
    diffpnp_loss_module = None
    if getattr(opt, "diffpnp", False):
        from diffpnp3d_loss import LocalSoftArgmax2D, DiffPnP3DLoss
        diffpnp_sa = LocalSoftArgmax2D(
            window=7, temperature=opt.diffpnp_temp,
            orig_size=(640, 480), belief_size=(output_size, output_size)).cuda()
        diffpnp_loss_module = DiffPnP3DLoss(
            n_gn=4,
            huber_delta=0.05,
            geometry_weight=opt.diffpnp_geometry_weight,
            undercoverage_weight=opt.diffpnp_undercoverage_weight,
            span_under_weight=opt.diffpnp_span_under_weight,
            depth_under_weight=opt.diffpnp_depth_under_weight,
            span_margin=opt.diffpnp_span_margin,
            depth_margin=opt.diffpnp_depth_margin,
            hard_span_threshold=opt.diffpnp_hard_span_threshold,
            hard_depth_threshold=opt.diffpnp_hard_depth_threshold,
            hard_example_gain=opt.diffpnp_hard_example_gain,
            fit_coverage_weight=opt.diffpnp_fit_coverage_weight,
            fit_span_margin=opt.diffpnp_fit_span_margin,
            fit_hard_span_threshold=opt.diffpnp_fit_hard_span_threshold,
            fit_min_observed_span=opt.diffpnp_fit_min_observed_span,
        ).cuda()
        print(f"[DIFFPNP] DiffPnP3D loss enabled (lambda={opt.diffpnp_lambda}, "
              f"warmup={opt.diffpnp_warmup}, ramp={opt.diffpnp_ramp} steps, "
              f"temp={opt.diffpnp_temp}, geom={opt.diffpnp_geometry_weight}, "
              f"under={opt.diffpnp_undercoverage_weight}, "
              f"components=span{opt.diffpnp_span_under_weight}/"
              f"depth{opt.diffpnp_depth_under_weight}, "
              f"margins=span{opt.diffpnp_span_margin}/"
              f"depth{opt.diffpnp_depth_margin}, "
              f"hard={opt.diffpnp_hard_example_gain}x@"
              f"span<{opt.diffpnp_hard_span_threshold}|"
              f"tz>{opt.diffpnp_hard_depth_threshold}, "
              f"fit={opt.diffpnp_fit_coverage_weight}@"
              f"{opt.diffpnp_fit_span_margin})")

    if use_mask_aux:
        print(f"[MASK-AUX] seg-head BCE aux enabled "
              f"(weight={opt.mask_weight}, warmup={opt.mask_warmup}); "
              f"TRAINING-ONLY, no inference hard-gate. "
              f"mask GT = mask_rle decode (valid only on v3/addon).")

    if use_clip_belief_border:
        print("[BORDER-GT] centre-inside Gaussians use clipped border tails")
    if use_mask_belief_fusion:
        print("[MASK-FUSION] zero-init mask-guided residual on final belief stage")

    net.train()
    if scoped_training:
        # Keep any present/future BatchNorm or Dropout in the frozen core from
        # changing behavior or buffers.  The residual fusion branch alone is
        # placed in training mode.
        base_net = net.module if hasattr(net, "module") else net
        for _child_name, child in base_net.named_children():
            child.eval()
        if train_only_mask_fusion:
            base_net.m_mask_belief_fusion.train()
        else:
            base_net.m6_2.train()
        print(
            f"[SCOPED:{trainable_scope}] frozen modules eval; selected branch train")
    global_step = 0
    for epoch in range(start_epoch, opt.epochs + 1):
        global_step = _runnetwork(net, optimizer, local_rank, epoch, training_data, writer,
                    encoder_freeze_steps=getattr(opt, "encoder_freeze_steps", 0),
                    global_step=global_step,
                    geo_loss_module=geo_loss_module,
                    geo_lambda=opt.geo_lambda,
                    geo_warmup=opt.geo_warmup,
                    struct_loss_module=struct_loss_module,
                    struct_lambda=opt.struct_lambda,
                    struct_warmup=opt.struct_warmup,
                    rel_loss_module=rel_loss_module,
                    rel_lambda=opt.rel_lambda,
                    rel_warmup=opt.rel_warmup,
                    vis_loss_module=vis_loss_module,
                    vis_lambda=opt.vis_lambda,
                    vis_warmup=opt.vis_warmup,
                    mask_aux=use_mask_aux,
                    mask_weight=getattr(opt, "mask_weight", 0.0),
                    mask_warmup=getattr(opt, "mask_warmup", 0),
                    corner_quality_loss=corner_quality_loss,
                    quality_weight=opt.quality_weight,
                    mask_extent_loss=mask_extent_loss,
                    extent_weight=opt.extent_weight,
                    projected_span_loss=projected_span_loss,
                    span_weight=opt.span_weight,
                    signed_footprint_loss=signed_footprint_loss,
                    signed_weight=opt.signed_weight,
                    teacher_net=teacher_net,
                    teacher_distill_weight=opt.teacher_distill_weight,
                    teacher_peak_weight=opt.teacher_peak_weight,
                    teacher_peak_threshold=opt.teacher_peak_threshold,
                    teacher_peak_margin=opt.teacher_peak_margin,
                    extent_cvar_fraction=opt.extent_cvar_fraction,
                    refinement_warmup=opt.refinement_warmup,
                    refinement_ramp=opt.refinement_ramp,
                    diffpnp_sa=diffpnp_sa,
                    diffpnp_loss_module=diffpnp_loss_module,
                    diffpnp_lambda=getattr(opt, "diffpnp_lambda", 0.0),
                    diffpnp_warmup=getattr(opt, "diffpnp_warmup", 0),
                    diffpnp_ramp=getattr(opt, "diffpnp_ramp", 500))

        try:
            if local_rank == 0 and epoch > 0 and epoch % opt.save_every == 0:
                out_fn = f"{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
                torch.save(net.state_dict(), out_fn)

                # Clean up old checkpoints if we're limiting the number saved
                if ckpt_q is not None:
                    if ckpt_q.full():
                        to_del = ckpt_q.get()
                        os.remove(to_del)
                    ckpt_q.put(out_fn)

        except Exception as e:
            print(f"Encountered Exception: {e}")

    if local_rank == 0:
        torch.save(
            net.state_dict(),
            f"{opt.outf}/final_net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
        )

    print("end:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    print("Total time taken: ", str(datetime.datetime.now() - start_time).split(".")[0])
    return


if __name__ == "__main__":
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False,
    )
    conf_parser.add_argument(
        "-c", "--config",
        help="Specify config file",
        metavar="FILE"
    )
    # Read the config but do not overwrite the args written
    args, remaining_argv = conf_parser.parse_known_args()


    parser = argparse.ArgumentParser()
    # Specify Training Data
    parser.add_argument(
        "--data",
        nargs="+",
        help="Path to training data"
    )
    parser.add_argument(
        "--use_s3",
        action="store_true",
        help="Use s3 buckets for training data"
    )
    parser.add_argument(
        "--train_buckets",
        nargs="+",
        default=[],
        help="s3 buckets containing training data. Can list multiple buckets separated by a space.",
    )
    parser.add_argument(
        "--endpoint",
        "--endpoint_url",
        type=str,
        default=None
    )

    # Specify Training Object
    parser.add_argument(
        "--object",
        nargs="+",
        required=True,
        default=[],
        help='Object to train network for. Must match "class" field in groundtruth .json file.'
        ' For best performance, only put one object of interest.',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers"
    )
    parser.add_argument(
        "--batchsize", "--batch_size",
        type=int,
        default=32,
        help="input batch size"
    )
    parser.add_argument(
        "--imagesize",
        type=int,
        default=448,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate, default=0.0001"
    )
    parser.add_argument(
        "--net_path",
        default=None, help="path to net (to continue training)"
    )
    parser.add_argument(
        "--namefile",
        default="epoch",
        help="name to put on the file of the save weights"
    )
    parser.add_argument(
        "--manualseed",
        type=int,
        help="manual random number seed"
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        "-e",
        type=int,
        default=60,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=0,
        help="If >0, cap #samples drawn per epoch (WeightedRandomSampler "
             "num_samples). Keeps balance_groups ratio. Used for pilot runs.",
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=100
    )
    parser.add_argument(
        "--outf",
        default="output/weights",
        help="folder to output images and model checkpoints",
    )
    parser.add_argument(
        "--nb_checkpoints",
        type=int,
        default=0,
        help="Number of checkpoints (.pth files) to save. Older ones will be "
        "deleted as new ones are saved. A value of 0 means an unlimited "
        "number will be saved"
    )
    parser.add_argument(
        '--save_every',
        type=int, default=1,
        help='How often (in epochs) to save a snapshot'
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="keypoint creation sigma (Gaussian std for belief map)")
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0
    )

    parser.add_argument("--save", action="store_true", help="save a batch and quit")

    # On-the-fly truncation augmentation
    parser.add_argument("--truncation_aug_prob", type=float, default=0.0,
                        help="Probability per sample of applying on-the-fly "
                             "truncation crop+pad augmentation (0.0=off, "
                             "challenge pretrain uses 0.6)")

    # Group-balanced sampling (1:1 in-group vs rest by image-path substring)
    parser.add_argument("--balance_substr", type=str, default=None,
                        help="If set, use a WeightedRandomSampler giving equal "
                             "total weight to samples whose image path contains "
                             "this substring vs the rest (1:1 group balance). "
                             "None=off (original shuffle).")

    # N-way group-balanced sampling by image-path substrings with ratios.
    # Format: "substr1:w1,substr2:w2,...[,*:wrest]". Each named group receives
    # total draw-weight proportional to its w; samples matching none of the
    # named substrings form an implicit "rest" group (use *:w to weight it,
    # default rest weight = 0 i.e. excluded). None=off (uses balance_substr or
    # plain shuffle). Independent of --balance_substr; if both set this wins.
    parser.add_argument("--balance_groups", type=str, default=None,
                        help="N-way ratio sampler: 'substrA:2,substrB:1,substrC:1'. "
                             "None=off (original behaviour unchanged).")

    # Discriminative LR: scale the encoder (VGG feature extractor) LR by this
    # factor relative to --lr (heads keep --lr). 1.0 or None = single LR (default
    # unchanged). Use e.g. 0.1 to slow encoder during finetune (anti-forgetting).
    parser.add_argument("--encoder_lr_scale", type=float, default=None,
                        help="Encoder(VGG) LR = lr*scale; heads use lr. "
                             "None/1.0 = single-LR Adam (unchanged).")

    # Freeze the encoder (VGG) for the first N optimizer steps, then unfreeze.
    # 0 = no freeze (default unchanged).
    parser.add_argument("--encoder_freeze_steps", type=int, default=0,
                        help="Freeze encoder(VGG) for first N steps then unfreeze "
                             "(0=off, default unchanged).")

    parser.add_argument(
        "--aspect_resize", action="store_true",
        help="Use deterministic square Resize instead of RandomCrop, independent "
             "of DiffPnP. Keeps pseudo-label geometry aligned; off=legacy crop.")

    # Mask auxiliary (STEP13 B2): flag-gated seg head (numSeg=1, BCE) trained as a
    # TRAINING-ONLY auxiliary feature alongside heatmap+affinity. GT = JSON
    # mask_rle decode only; old data without mask_rle has valid=0 (no mask loss).
    # Inference is unchanged (belief-peak decode, NO mask hard-gate). Off=default.
    parser.add_argument("--mask_aux", action="store_true",
                        help="Enable seg-head mask auxiliary (BCE on mask_rle). "
                             "Training-only feature; no inference hard-gate. "
                             "off=default (model byte-identical).")
    parser.add_argument("--mask_weight", type=float, default=0.05,
                        help="Weight for mask BCE aux loss (default: 0.05).")
    parser.add_argument("--mask_warmup", type=int, default=0,
                        help="Epochs before enabling mask aux loss (default: 0).")

    # Fixed-400/50 heatmap + defensive-PnP training enhancements. Every switch
    # is independently ablatable; --heatmap_pnp_enhance enables all four.
    parser.add_argument("--heatmap_pnp_enhance", action="store_true",
                        help="Enable clipped border GT, mask-belief fusion, mask "
                             "extent loss, and corner uncertainty.")
    parser.add_argument("--clip_belief_border", action="store_true",
                        help="Draw centre-inside GT Gaussians and clip only their "
                             "off-grid tails. Default keeps legacy drop policy.")
    parser.add_argument("--mask_belief_fusion", action="store_true",
                        help="Enable zero-init mask-guided residual on final belief "
                             "stage (implies mask_aux).")
    parser.add_argument("--trainable_scope",
                        choices=("all", "mask_fusion", "belief_tail"),
                        default="all",
                        help="Parameter scope to optimize. mask_fusion freezes "
                             "the base model and trains its residual branch; "
                             "belief_tail trains only m6_2 layers 10 and 12.")
    parser.add_argument("--extent_loss", action="store_true",
                        help="Penalize final 8-corner extent lying inside aligned "
                             "real mask bbox (implies mask_aux).")
    parser.add_argument("--corner_quality", action="store_true",
                        help="Predict per-corner log localization sigma for PnP "
                             "weighting/rejection.")
    parser.add_argument("--quality_weight", type=float, default=0.01,
                        help="Corner log-sigma calibration loss weight.")
    parser.add_argument("--extent_weight", type=float, default=0.05,
                        help="Mask under-extent loss weight.")
    parser.add_argument("--refinement_warmup", type=int, default=0,
                        help="Epochs before fixed-grid refinement losses.")
    parser.add_argument("--refinement_ramp", type=int, default=500,
                        help="Steps to linearly ramp refinement losses 0->1.")
    parser.add_argument("--extent_radius", type=int, default=5,
                        help="Local soft-argmax radius for extent loss.")
    parser.add_argument("--extent_temperature", type=float, default=0.10,
                        help="Local soft-argmax temperature for extent loss.")
    parser.add_argument("--extent_tolerance", type=float, default=1.0,
                        help="Allowed under-extent in 50-grid cells.")
    parser.add_argument(
        "--extent_cvar_fraction", type=float, default=1.0,
        help="Teacher-ranked worst valid mask-extent fraction to optimize. "
             "1.0 preserves the mean-loss behavior.")
    parser.add_argument("--projected_span_loss", action="store_true",
                        help="Directly supervise the deployed-decoder 8-corner "
                             "GT footprint and penalize projected shrinkage.")
    parser.add_argument("--span_weight", type=float, default=0.01,
                        help="Overall projected-span loss weight.")
    parser.add_argument("--span_window", type=int, default=11,
                        help="Odd deployed centroid window size.")
    parser.add_argument("--span_decoder_offset", type=float, default=0.4395,
                        help="Deployed weighted-centroid coordinate offset.")
    parser.add_argument("--span_smooth_sigma", type=float, default=2.0,
                        help="Gaussian sigma used for deployed NMS selection.")
    parser.add_argument("--span_interior_margin", type=float, default=4.0,
                        help="Required GT distance from every 50-grid border.")
    parser.add_argument("--span_min_size", type=float, default=2.0,
                        help="Minimum valid GT span in a PCA axis, grid cells.")
    parser.add_argument("--span_coord_weight", type=float, default=1.0,
                        help="Ordered coordinate anchor inside span loss.")
    parser.add_argument("--span_overshoot_weight", type=float, default=0.25,
                        help="Relative soft penalty for excessive footprint.")
    parser.add_argument("--span_overshoot_ratio", type=float, default=1.10,
                        help="Pred/GT span ratio allowed without overshoot loss.")
    parser.add_argument("--span_huber_delta", type=float, default=0.05,
                        help="Huber transition for normalized/log residuals.")
    parser.add_argument("--span_footprint_edge_weight", type=float, default=0.0,
                        help="Weight for channel-ordered W/depth edge ratios.")
    parser.add_argument("--span_hard_edge_threshold", type=float, default=0.85,
                        help="Mean W/depth ratio below which a frame is hard.")
    parser.add_argument("--span_hard_example_gain", type=float, default=0.0,
                        help="Extra multiplier applied to hard footprint frames.")
    parser.add_argument("--span_min_edge_length", type=float, default=1.0,
                        help="Minimum valid projected GT edge in grid cells.")
    parser.add_argument(
        "--signed_footprint_loss", action="store_true",
        help="Use GT-direction signed W/depth edges and radial coverage on "
             "the unchanged deployed 50-grid decoder.")
    parser.add_argument("--signed_weight", type=float, default=0.01,
                        help="Overall signed-footprint loss weight.")
    parser.add_argument("--signed_width_weight", type=float, default=1.0)
    parser.add_argument("--signed_depth_weight", type=float, default=1.0)
    parser.add_argument("--signed_radial_weight", type=float, default=1.0)
    parser.add_argument("--signed_min_radial_length", type=float, default=1.0)
    parser.add_argument("--signed_overshoot_weight", type=float, default=0.20)
    parser.add_argument("--signed_overshoot_ratio", type=float, default=1.10)
    parser.add_argument("--signed_huber_delta", type=float, default=0.05)
    parser.add_argument(
        "--teacher_checkpoint", type=str, default=None,
        help="Frozen accepted checkpoint used for anti-forgetting constraints; "
             "must equal --net_path.")
    parser.add_argument("--teacher_distill_weight", type=float, default=0.0,
                        help="Final belief-map MSE distillation weight.")
    parser.add_argument("--teacher_peak_weight", type=float, default=0.0,
                        help="Per-channel teacher peak-retention hinge weight.")
    parser.add_argument("--teacher_peak_threshold", type=float, default=0.3)
    parser.add_argument("--teacher_peak_margin", type=float, default=0.05)

    # DiffPnP3D geometry regularizer (PAPER_S2). Flag-gated; default off =>
    # training is byte-identical (loader aspect_resize off, no targets, no loss).
    parser.add_argument("--diffpnp", action="store_true",
                        help="Enable DiffPnP3D 3D-corner geometry loss on "
                             "pnp_valid_3d&V8 frames (aspect_resize ON). off=default.")
    parser.add_argument("--diffpnp_lambda", type=float, default=0.003,
                        help="Weight for DiffPnP3D loss (default: 0.003).")
    parser.add_argument("--diffpnp_geometry_weight", type=float, default=1.0,
                        help="Weight of the symmetric 3D-corner term inside "
                             "DiffPnP (default preserves legacy behavior).")
    parser.add_argument("--diffpnp_undercoverage_weight", type=float, default=0.0,
                        help="Weight of one-sided PnP footprint/depth loss.")
    parser.add_argument("--diffpnp_span_under_weight", type=float, default=1.0,
                        help="Projected-span component inside undercoverage loss.")
    parser.add_argument("--diffpnp_depth_under_weight", type=float, default=1.0,
                        help="Positive t_z-bias component inside undercoverage loss.")
    parser.add_argument("--diffpnp_span_margin", type=float, default=1.0,
                        help="No-penalty projected-span ratio in (0,1].")
    parser.add_argument("--diffpnp_depth_margin", type=float, default=1.0,
                        help="No-penalty predicted/GT t_z ratio (>=1).")
    parser.add_argument("--diffpnp_hard_span_threshold", type=float, default=0.90,
                        help="Projected-span ratio used for hard-frame diagnostics.")
    parser.add_argument("--diffpnp_hard_depth_threshold", type=float, default=1.10,
                        help="Predicted/GT t_z ratio used for hard diagnostics.")
    parser.add_argument("--diffpnp_hard_example_gain", type=float, default=0.0,
                        help="Extra multiplier for PnP hard frames (default off).")
    parser.add_argument(
        "--diffpnp_fit_coverage_weight", type=float, default=0.0,
        help="Weight inside DiffPnP for one-sided rigid-projection coverage "
             "of the detached heatmap observation footprint.")
    parser.add_argument("--diffpnp_fit_span_margin", type=float, default=1.0)
    parser.add_argument(
        "--diffpnp_fit_hard_span_threshold", type=float, default=0.90)
    parser.add_argument(
        "--diffpnp_fit_min_observed_span", type=float, default=1.0)
    parser.add_argument("--diffpnp_warmup", type=int, default=0,
                        help="Epochs before enabling DiffPnP3D loss (default: 0).")
    parser.add_argument("--diffpnp_ramp", type=int, default=500,
                        help="Steps to linearly ramp DiffPnP3D lambda 0->1 (default: 500).")
    parser.add_argument("--diffpnp_temp", type=float, default=0.1,
                        help="Local soft-argmax temperature (default: 0.1).")
    parser.add_argument("--diffpnp_index_dir", type=str,
                        default="data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index",
                        help="Dir with per-dataset pnp_valid_3d index JSONs.")
    parser.add_argument("--diffpnp_root", type=str,
                        default="data/pallet/training_data",
                        help="Root the index relpaths are relative to.")

    # Symmetric loss (180° front-back swap)
    parser.add_argument("--symmetric_loss", action="store_true",
                        help="Use min(orig, 180°-swapped) belief loss for symmetric objects")

    # Geometric loss arguments
    parser.add_argument("--geo_loss", action="store_true",
                        help="Enable geometric loss (soft-argmax + BPnP)")
    parser.add_argument("--geo_lambda", type=float, default=0.1,
                        help="Weight for geometric loss (default: 0.1)")
    parser.add_argument("--geo_warmup", type=int, default=5,
                        help="Epochs before enabling PnP-based losses (default: 5)")
    parser.add_argument("--geo_temperature", type=float, default=1.0,
                        help="Soft-argmax temperature (default: 1.0)")
    parser.add_argument("--geo_fx", type=float, default=614.18)
    parser.add_argument("--geo_fy", type=float, default=614.31)
    parser.add_argument("--geo_cx", type=float, default=329.28)
    parser.add_argument("--geo_cy", type=float, default=234.53)
    parser.add_argument("--geo_img_w", type=int, default=640)
    parser.add_argument("--geo_img_h", type=int, default=480)

    # Visibility-aware coordinate loss arguments
    parser.add_argument("--vis_coord_loss", action="store_true",
                        help="Enable visibility-aware coordinate loss")
    parser.add_argument("--vis_lambda", type=float, default=0.005,
                        help="Weight for visibility coord loss (default: 0.005)")
    parser.add_argument("--vis_warmup", type=int, default=0,
                        help="Epochs before enabling vis coord loss (default: 0)")

    # Reliability loss arguments
    parser.add_argument("--rel_loss", action="store_true",
                        help="Enable reliability-aware coordinate loss")
    parser.add_argument("--rel_lambda", type=float, default=0.005,
                        help="Weight for reliability loss (default: 0.005)")
    parser.add_argument("--rel_warmup", type=int, default=0,
                        help="Epochs before enabling reliability loss (default: 0)")
    parser.add_argument("--rel_delta", type=float, default=0.03,
                        help="Huber delta for reliability loss (default: 0.03)")
    parser.add_argument("--rel_lambda_log", type=float, default=0.5,
                        help="Log-regularizer weight (default: 0.5)")

    # Structural loss arguments
    parser.add_argument("--struct_loss", action="store_true",
                        help="Enable structural losses (flip + edge + coord)")
    parser.add_argument("--struct_lambda", type=float, default=1.0,
                        help="Overall weight for structural loss (default: 1.0)")
    parser.add_argument("--struct_warmup", type=int, default=10,
                        help="Epochs before enabling structural losses (default: 10)")
    parser.add_argument("--struct_flip", type=float, default=0.02,
                        help="Flip equivariance loss weight (default: 0.02)")
    parser.add_argument("--struct_edge", type=float, default=0.05,
                        help="Sparse edge loss weight (default: 0.05)")
    parser.add_argument("--struct_coord", type=float, default=0.10,
                        help="Coordinate Huber loss weight (default: 0.10)")
    parser.add_argument("--struct_vp", type=float, default=0.0,
                        help="Vanishing-point concurrency loss weight (default: 0.0)")
    parser.add_argument("--struct_delta", type=float, default=0.03,
                        help="Huber delta for structural losses (default: 0.03)")

    opt = parser.parse_args(remaining_argv)

    main(opt)
