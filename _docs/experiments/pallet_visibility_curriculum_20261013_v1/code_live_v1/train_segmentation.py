"""Fixed512-update tiny RGB segmentation pilot on actual source object masks.

Train768 and CAL128 retain their existing family membership. Source TEST128,
all real images, and FINAL are never opened here. CAL chooses a checkpoint
from four prespecified validations; the fixed update count never changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from . import common as C
from .segmentation import TinyUNet, DEFAULT_CONFIG
from .source_benchmark import FAMILY, CACHE, roi_mask, segmentation_metrics


SEED = 20261013
SIZE = 256
STEPS = 512
BATCH = 8


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=C.sha(path))


def make_cache(families, cached, output):
    selected = [r for r in families if r["partition"] in ("train", "calibration")]
    if Counter(r["partition"] for r in selected) != {"train": 768, "calibration": 128}:
        raise ValueError("Existing768/128 family membership is required")
    images = np.empty((896, 3, SIZE, SIZE), np.uint8)
    masks = np.empty((896, SIZE * SIZE // 8), np.uint8)
    cal_native = {}
    metadata, foreground_pixels = [], 0
    start = time.monotonic()
    for index, row in enumerate(selected):
        rgb_path = C.ROOT / row["raw"]["rgb"]
        mask_path = C.ROOT / row["raw"]["visible"]
        source = cached[row["id"]]
        if source["partition"] != row["partition"] or source["family"] != row["family"] or source["variant"] != 0:
            raise ValueError("Original source prediction/family contract differs")
        rgb_sha = C.sha(rgb_path)
        if rgb_sha != source["original_rgb"]["sha256"]:
            raise ValueError("Original source RGB hash differs")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        actual = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or actual is None or image.shape[:2] != actual.shape:
            raise ValueError("Source RGB/mask mismatch")
        images[index] = cv2.cvtColor(cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        target = cv2.resize(actual, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127
        masks[index] = np.packbits(target.reshape(-1))
        if row["partition"] == "train":
            foreground_pixels += int(target.sum())
        else:
            cal_native[index] = (np.packbits((actual > 127).reshape(-1)), actual.shape)
        metadata.append(dict(index=index, id=row["id"], family=row["family"], partition=row["partition"],
                             rgb_sha256=rgb_sha, mask_sha256=C.sha(mask_path), raw_hw=list(actual.shape),
                             frozen_selected_points=source["selected_points"]))
        if (index + 1) % 128 == 0:
            print("SOURCE_CACHE", index + 1, "seconds", round(time.monotonic() - start, 2), flush=True)
    cache_bytes = images.nbytes + masks.nbytes + sum(a.nbytes for a, hw in cal_native.values())
    if cache_bytes > 200 * 1024 * 1024:
        raise ValueError("Source RAM cache exceeds200MiB")
    foreground_fraction = foreground_pixels / (768 * SIZE * SIZE)
    C.write(output / "CACHE_RECEIPT.json", dict(complete=True, frames=896, metadata=metadata,
        uint8_RGB_bytes=images.nbytes, packed256_mask_bytes=masks.nbytes,
        packed_native_CAL_mask_bytes=sum(a.nbytes for a, hw in cal_native.values()),
        total_array_cache_bytes=cache_bytes, source_train_foreground_fraction=foreground_fraction,
        original_RGB_or_mask_edits=0, real_image_reads=0, source_TEST_reads=0,
        prepare_seconds=time.monotonic() - start))
    return images, masks, cal_native, metadata, foreground_fraction


def augmentation(image, generator):
    """Global photometric changes only; labels do not select recoloured pixels."""
    n = len(image)
    brightness = torch.rand((n, 1, 1, 1), generator=generator, device=image.device) * .24 - .12
    contrast = torch.rand((n, 1, 1, 1), generator=generator, device=image.device) * .5 + .75
    channel_gain = torch.rand((n, 3, 1, 1), generator=generator, device=image.device) * .5 + .75
    saturation = torch.rand((n, 1, 1, 1), generator=generator, device=image.device) * .8 + .6
    # Different gain per RGB channel changes hue globally without any mask input.
    value = ((image - .5) * contrast + .5 + brightness) * channel_gain
    gray = (value * image.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    return (gray + saturation * (value - gray)).clamp(0., 1.)


def loss_value(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    total = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (total + 1)).mean()
    return bce + dice, bce, dice


def distribution(values):
    a = np.asarray([v for v in values if v is not None], float)
    if not len(a):
        return dict(n=0, mean=None, median=None, std=None, P10=None, P90=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std()), P10=float(np.quantile(a, .1)), P90=float(np.quantile(a, .9)))


def validate(model, images, cal_native, metadata, device):
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(768, 896, BATCH):
            ids = list(range(start, start + BATCH))
            image = torch.from_numpy(np.ascontiguousarray(images[ids])).to(device, dtype=torch.float32).div_(255)
            probs = model(image).sigmoid()[:, 0].cpu().numpy()
            for offset, index in enumerate(ids):
                packed, hw = cal_native[index]
                actual = np.unpackbits(packed, count=int(np.prod(hw))).reshape(hw).astype(np.uint8)
                prob = cv2.resize(probs[offset].astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
                predicted = prob >= .5
                roi, bbox = roi_mask(hw, metadata[index]["frozen_selected_points"])
                full = segmentation_metrics(predicted, actual, np.ones(hw, bool))
                full["target"] = "actual_source_visible_mask_full_native_image"
                records.append(dict(id=metadata[index]["id"], family=metadata[index]["family"],
                    full_image=full,
                    frozen_prediction_ROI=segmentation_metrics(predicted, actual, roi),
                    prediction_bbox_plus32px=bbox))
    summary = {}
    for scope in ("full_image", "frozen_prediction_ROI"):
        cells = [r[scope] for r in records]
        tp, fp, fn = [sum(r[k] for r in cells) for k in ("tp", "fp", "fn")]
        summary[scope] = dict(iou=distribution([r["iou"] for r in cells]),
            precision=distribution([r["precision"] for r in cells]), recall=distribution([r["recall"] for r in cells]),
            pooled_iou=tp / (tp + fp + fn) if tp + fp + fn else None,
            pooled_precision=tp / (tp + fp) if tp + fp else None,
            pooled_recall=tp / (tp + fn) if tp + fn else None,
            tp=tp, fp=fp, fn=fn)
    model.train()
    return summary, records


def save_checkpoint(path, model, step, protocol_sha, summary):
    torch.save(dict(schema="tiny_source_RGB_mask_v1", state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    config=DEFAULT_CONFIG, step=step, seed=SEED, protocol_sha256=protocol_sha,
                    source_CAL=summary, training_real_images=0, source_TEST_seen=False), path)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new segmentation output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fixed training pilot")
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    families = C.read(args.family_manifest)["records"]
    cached = {r["id"]: r for r in C.read(args.cache_manifest)["records"]}
    train_ids = [r["id"] for r in families if r["partition"] == "train"]
    cal_ids = [r["id"] for r in families if r["partition"] == "calibration"]
    model = TinyUNet().cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count * 4 > 1024 * 1024:
        raise ValueError("Tiny segmentation weight budget exceeds1MiB")
    protocol = dict(schema="source_RGB_segmentation_fixed512_v1", client_date="2026-10-13",
        train_families=768, calibration_families=128, train_ids=train_ids, calibration_ids=cal_ids,
        source_TEST_families_used=0, real_CAL_images_used=0, real_TEST_images_used=0, FINAL_images_used=0,
        dataset_family_membership_changed=False, model="TinyUNet8/16/32_GroupNorm", parameter_count=parameter_count,
        input_size=SIZE, input_preprocessing="full BGR->RGB, resize256square AREA, /255; no GT crop or mask in inference",
        target="actual visible object mask; nearest resize for loss, native masks for CAL evaluation",
        seed=SEED, deterministic_algorithms=True, AMP=False, TF32=False,
        steps=STEPS, batch=BATCH, optimizer=dict(name="AdamW", lr=.001, weight_decay=.0001),
        loss="fixed TRAIN pixel-ratio balanced BCE(pos_weight clipped1..20) + soft Dice",
        photometric_augmentation=dict(global_brightness=[-.12, .12], global_contrast=[.75, 1.25],
            independent_global_RGB_channel_gain=[.75, 1.25], global_saturation=[.6, 1.4],
            mask_selected_pixel_recolouring=False, geometric_augmentation=False),
        validation_steps=[128, 256, 384, 512], checkpoint_selection="best mean fullimage source CAL IoU at fixed0.5; last also retained",
        total_updates_cannot_change_after_CAL_scores=True, maximum_RAM_array_cache_MiB=200,
        original_RGB_and_masks_mutations=0, source_material_domain="Existing source1024 all scene.usd/Pallet_0/graphite; wood/green transfer not guaranteed",
        inputs=[binding(args.family_manifest), binding(args.cache_manifest), binding(Path(__file__).with_name("segmentation.py")), binding(__file__)])
    C.write(output / "PROTOCOL.json", protocol)
    protocol_sha = C.sha(output / "PROTOCOL.json")
    images, masks, cal_native, metadata, fraction = make_cache(families, cached, output)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    order = np.random.default_rng(SEED).integers(0, 768, size=(STEPS, BATCH))
    pos_weight = torch.tensor(min(20., max(1., (1 - fraction) / fraction)), device="cuda")
    history, validations = [], []
    best_score, best_step = -1., None
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step, ids in enumerate(order, 1):
        image = torch.from_numpy(np.ascontiguousarray(images[ids])).to("cuda", dtype=torch.float32).div_(255)
        target = np.unpackbits(masks[ids], axis=1).reshape(BATCH, 1, SIZE, SIZE)
        target = torch.from_numpy(target).to("cuda", dtype=torch.float32)
        image = augmentation(image, generator)
        optimizer.zero_grad(set_to_none=True)
        loss, bce, dice = loss_value(model(image), target, pos_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite source segmentation loss")
        loss.backward()
        optimizer.step()
        if step % 16 == 0:
            history.append(dict(step=step, loss=float(loss.detach()), bce=float(bce.detach()), dice=float(dice.detach())))
        if step % 128 == 0:
            summary, records = validate(model, images, cal_native, metadata, "cuda")
            score = summary["full_image"]["iou"]["mean"]
            better = score > best_score
            if better:
                best_score, best_step = score, step
                save_checkpoint(output / "best_source_CAL.pt", model, step, protocol_sha, summary)
            validations.append(dict(step=step, selected_best=better, summary=summary))
            C.save_rows(output / f"SOURCE_CAL_STEP{step:03d}.jsonl.gz", records)
            C.write(output / "PROGRESS.json", dict(step=step, steps_total=STEPS, best_step=best_step,
                source_CAL_mean_full_IoU=score, source_CAL_predROI_pooled_IoU=summary["frozen_prediction_ROI"]["pooled_iou"],
                elapsed_seconds=time.monotonic() - began))
            print("SEGMENTATION_TRAIN", step, "loss", round(float(loss.detach()), 4),
                  "CALmeanIoU", round(score, 4), "CALpredROIIoU", round(summary["frozen_prediction_ROI"]["pooled_iou"], 4),
                  "seconds", round(time.monotonic() - began, 2), flush=True)
    torch.cuda.synchronize()
    save_checkpoint(output / "last.pt", model, STEPS, protocol_sha, validations[-1]["summary"])
    baseline = C.read(C.SCRATCH / "source_v1/METRICS.json")["groups"]["NATURAL:FROZEN_PREDICTED_XY_SELF"]
    best_validation = next(r for r in validations if r["step"] == best_step)
    completion = dict(schema=protocol["schema"], complete=True, training_updates=STEPS,
        training_image_exposures=STEPS * BATCH, unique_train_families_seen=int(len(np.unique(order))),
        source_CAL_passes=4, source_CAL_inference_images=512, source_TEST_reads=0,
        real_CAL_reads=0, real_TEST_reads=0, FINAL_reads=0, source_original_mutations=0,
        best_source_CAL_step=best_step, checkpoint_selection_metric=best_score,
        selected_source_CAL=best_validation["summary"], last_source_CAL=validations[-1]["summary"],
        baseline_same_native_prediction_ROI_material_mask=dict(iou=baseline["pooled_material_mask_iou"],
            precision=baseline["pooled_material_mask_precision"], recall=baseline["pooled_material_mask_recall"]),
        comparison_Calibration_partition_reused_for_selection=True,
        loss_history=history, validation_history=validations,
        elapsed_training_validation_seconds=time.monotonic() - began,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        optimizer=dict(pos_weight=float(pos_weight), lr=.001),
        source_batch_order_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
        outputs=[binding(output / "best_source_CAL.pt"), binding(output / "last.pt")],
        protocol=binding(output / "PROTOCOL.json"),
        limitations=["CAL scores chose a checkpoint; CAL improvement is development evidence, not independent heldout confirmation.",
                     "Source all-graphite Pallet_0 masks cannot certify real wood/green generalization.",
                     "A learned object mask cannot establish individual3D keypoint visibility or same-colour occluder ownership.",
                     "No real TEST or FINAL evaluation was performed."])
    C.write(output / "COMPLETION.json", completion)
    print("SEGMENTATION_COMPLETE", str(output), "best_step", best_step, "seconds", round(time.monotonic() - began, 2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-manifest", type=Path, default=FAMILY)
    parser.add_argument("--cache-manifest", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=C.SCRATCH / "segmentation_v1")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
