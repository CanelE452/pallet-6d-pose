"""Fixed source-only local opacity pilot, followed by isolated CAL diagnostics."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from . import common as C
from .local_occlusion import LocalSupportCNN, SIZE, tensor_inputs, infer_local_support
from .source_benchmark import FAMILY, CACHE, geometry, initial_pose, disk_stats
from . import appearance as A


SEED = 20261013
STEPS = 1024
BATCH = 32
KINDS = ("FLAT_RANDOM", "BACKGROUND_TEXTURE", "MATCH_LOCAL_MATERIAL", "OTHER_FOREGROUND_TEXTURE")


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=C.sha(path))


def make_dataset(families, cached, out):
    data = {part: dict(patches=[], expected=[], background=[], keys=[]) for part in ("train", "calibration")}
    reasons = Counter()
    metadata = []
    for index, source in enumerate(r for r in families if r["partition"] in data):
        part = source["partition"]
        image_path, mask_path, label_path = [C.ROOT / source["raw"][k] for k in ("rgb", "visible", "label")]
        frozen = cached[source["id"]]
        if C.sha(image_path) != frozen["original_rgb"]["sha256"] or frozen["partition"] != part:
            raise ValueError("Frozen RGB/family source contract differs")
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), 0)
        g = geometry(C.read(label_path))
        q = np.asarray(frozen["selected_points"], float)[:8]
        _, H, _ = initial_pose(q, g["xyz"], g["K"])
        hull = np.zeros(mask.shape, np.uint8)
        if np.isfinite(q).all():
            A._fill_polygon(hull, q)
        h, w = mask.shape
        backgrounds = []
        for center in ((32., 32.), (w - 33., 32.), (32., h - 33.), (w - 33., h - 33.)):
            tile = cv2.getRectSubPix(image, (SIZE, SIZE), center)
            ownership = cv2.getRectSubPix(mask, (SIZE, SIZE), center)
            backgrounds.append((float(np.mean(ownership > 127)), tile))
        data[part]["background"].append(min(backgrounds, key=lambda r: r[0])[1])
        eligible = []
        for k in range(8):
            if H[k] or g["self_hidden"][k]:
                reasons[part + ":self_hidden"] += 1
                continue
            local = disk_stats(mask, q[k], radius=3)
            reference = disk_stats(mask, g["gt_xy"][k], radius=3)
            distance = float(np.linalg.norm(q[k] - g["gt_xy"][k]))
            # This is deliberately a conservative LOCAL_BODY_SUPPORT proxy.
            # Actual mask adjacency at a virtual corner is not turned into
            # a verified physical3D corner visibility or natural occlusion label.
            if (not local["in_frame"] or not reference["in_frame"] or distance > 8
                    or not local["foreground"] or not reference["foreground"]
                    or (local["fraction"] or 0) < .10):
                reasons[part + ":ambiguous_or_no_local_body_support"] += 1
                continue
            center = tuple(float(v) for v in q[k])
            data[part]["patches"].append(cv2.getRectSubPix(image, (SIZE, SIZE), center))
            data[part]["expected"].append(cv2.getRectSubPix(hull, (SIZE, SIZE), center))
            data[part]["keys"].append([source["id"], k])
            eligible.append(dict(corner=k, query_to_virtual_reference_px=distance,
                query_local_foreground=local, reference_local_foreground=reference,
                label_scope="LOCAL_BODY_SUPPORT_PROXY_ONLY"))
        metadata.append(dict(id=source["id"], family=source["family"], partition=part,
            eligible=eligible, fixed_predicted_self_hidden=H.tolist(), rgb_sha256=frozen["original_rgb"]["sha256"],
            visible_mask_sha256=C.sha(mask_path), source_label_sha256=C.sha(label_path)))
        if (index + 1) % 128 == 0:
            print("LOCAL_SOURCE_PREP", index + 1, flush=True)
    for part in data:
        for key in ("patches", "expected", "background"):
            data[part][key] = np.asarray(data[part][key], np.uint8)
    C.write(out / "SOURCE_ELIGIBILITY.json", dict(metadata=metadata, excluded=dict(reasons),
        eligible_counts={part: len(d["patches"]) for part, d in data.items()},
        source_TEST_reads=0, physical_semantic_corner_labels_available=False,
        safe_fallback="Only synthetic opacity on verified local rendered-object support; natural source no-match is never a negative label"))
    return data


def opaque_variant(original, expected, background, foreground, rng, kind):
    shape = np.zeros((SIZE, SIZE), np.uint8)
    half = rng.integers(8, 43, size=2)
    center = np.array([31.5, 31.5]) + rng.uniform(-3, 3, size=2)
    box = cv2.boxPoints((tuple(center), tuple((2 * half).astype(float)), float(rng.uniform(-45, 45))))
    cv2.fillConvexPoly(shape, np.rint(box).astype(np.int32), 255)
    if not shape[29:35, 29:35].all():
        raise AssertionError("Opaque intervention must cover the query footprint")
    if kind == "IDENTICAL_PIXEL_OWNERSHIP_CONTROL":
        filling = original
    elif kind == "FLAT_RANDOM":
        filling = np.broadcast_to(rng.integers(0, 256, size=3, dtype=np.uint8), original.shape)
    elif kind == "MATCH_LOCAL_MATERIAL":
        body = original[expected > 127]
        color = np.rint(np.median(body if len(body) else original.reshape(-1, 3), axis=0)).astype(np.uint8)
        filling = np.broadcast_to(color, original.shape)
    elif kind == "BACKGROUND_TEXTURE":
        filling = background
    elif kind == "OTHER_FOREGROUND_TEXTURE":
        filling = foreground
    else:
        raise ValueError(kind)
    value = original.copy()
    covered = shape > 0
    value[covered] = filling[covered]
    delta = float(np.mean(np.abs(value[covered].astype(float) - original[covered].astype(float))) / 255.)
    return value, dict(kind=kind, known_opaque_body_loss=True, mean_RGB_change=delta,
                       visual_identity_ambiguous=delta < 2 / 255., gradient_label_eligible=delta >= 2 / 255.)


def augmentation(value, generator):
    rgb, geom = value[:, :3], value[:, 3:]
    n = len(rgb)
    gain = .5 + torch.rand((n, 3, 1, 1), device=rgb.device, generator=generator) * 1.2
    exposure = torch.rand((n, 1, 1, 1), device=rgb.device, generator=generator) * .5 - .25
    saturation = torch.rand((n, 1, 1, 1), device=rgb.device, generator=generator) * 1.6 + .2
    image = rgb * gain + exposure
    gray = image.mean(1, keepdim=True)
    image = (gray + saturation * (image - gray)).clamp(0, 1)
    noise = torch.randn(image.shape, device=image.device, generator=generator) * (3 / 255.)
    return torch.cat(((image + noise).clamp(0, 1), geom), 1)


def fixed_evaluation_data(cal):
    rng = np.random.default_rng(SEED + 128)
    sets = {name: [] for name in ("NATURAL_LOCAL_BODY_PROXY", *KINDS, "IDENTICAL_PIXEL_OWNERSHIP_CONTROL")}
    provenance = {name: [] for name in sets}
    for i, original in enumerate(cal["patches"]):
        sets["NATURAL_LOCAL_BODY_PROXY"].append(original)
        provenance["NATURAL_LOCAL_BODY_PROXY"].append(dict(known_opaque_body_loss=False, visual_identity_ambiguous=False))
        for name in (*KINDS, "IDENTICAL_PIXEL_OWNERSHIP_CONTROL"):
            value, info = opaque_variant(original, cal["expected"][i], cal["background"][i % len(cal["background"])],
                cal["patches"][(i + 17) % len(cal["patches"])], rng, name)
            sets[name].append(value)
            provenance[name].append(info)
    return {name: np.asarray(values) for name, values in sets.items()}, provenance


def evaluate_source(model, cal, sets, provenance):
    from sklearn.metrics import roc_auc_score
    model.eval()
    predictions, result = {}, {}
    with torch.inference_mode():
        for name, patches in sets.items():
            pieces = []
            for start in range(0, len(patches), BATCH):
                batch = tensor_inputs(patches[start:start+BATCH], cal["expected"][start:start+BATCH], "cuda")
                pieces.append(model(batch).sigmoid().cpu().numpy())
            p = np.concatenate(pieces).astype(float)
            predictions[name] = p
            clipped = np.clip(p, 1e-8, 1 - 1e-8)
            entropy = -(clipped*np.log(clipped)+(1-clipped)*np.log(1-clipped))/np.log(2.)
            result[name] = dict(n=len(p), mean_support_score=float(p.mean()),
                direct_recall_at0p5=float(np.mean(p >= .5)) if name == "NATURAL_LOCAL_BODY_PROXY" else None,
                false_visible_at0p5=float(np.mean(p >= .5)) if name != "NATURAL_LOCAL_BODY_PROXY" else None,
                unknown_high_entropy_fraction=float(np.mean(entropy >= .8)),
                visually_ambiguous_cases=sum(r["visual_identity_ambiguous"] for r in provenance[name]),
                probabilities=p.tolist())
    natural = predictions["NATURAL_LOCAL_BODY_PROXY"]
    negative = np.concatenate([predictions[name][[not r["visual_identity_ambiguous"] for r in provenance[name]]] for name in KINDS])
    y = np.r_[np.ones(len(natural)), np.zeros(len(negative))]
    scores = np.r_[natural, negative]
    auc = float(roc_auc_score(y, scores))
    paired_balanced = .5 * (float(np.mean(natural >= .5)) + float(np.mean(negative < .5)))
    model.train()
    return dict(groups=result, pooled_identifiable_AUC=auc, pooled_identifiable_balanced_accuracy_at0p5=paired_balanced,
                identical_input_probability_max_difference=float(np.max(np.abs(natural-predictions["IDENTICAL_PIXEL_OWNERSHIP_CONTROL"]))))


def save_weights(path, model, step, protocol_sha):
    torch.save(dict(schema="local_RGB_support_synthetic_opacity_v1", seed=SEED, step=step,
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}, protocol_sha256=protocol_sha,
        inputs="RGB64+predictedhull+queryheatmap", target="localbodyproxy under syntheticopacity, not physical3D corner visibility"), path)


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for fixed source local CNN pilot")
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    files = [Path(__file__), Path(__file__).with_name("local_occlusion.py"), FAMILY, CACHE]
    protocol = dict(schema="source_local_RGB_opacity_fixed1024_v1", steps=STEPS, batch=BATCH, seed=SEED,
        optimizer=dict(name="AdamW", lr=.001, weight_decay=.0001), AMP=False, TF32=False, deterministic=True,
        source_train=768, source_CAL=128, source_TEST_reads=0, real_CAL_training_reads=0, real_TEST_reads=0, FINAL_reads=0,
        target="LOCAL_BODY_SUPPORT_PROXY plus known memory-only opaque coverage",
        semantic_label_blocker="Cuboid virtual corners need not be mesh vertices; masks alone cannot certify3D point observability",
        conservative_proxy_guard="nonself source/predictedH, q/gtinframe, q-versusgt<=8px, actual object pixels within3px at both and query foreground fraction>=.10",
        no_match_not_a_training_label=True, synthetic_kinds=KINDS, opacity=1., rotated_rectangles=True,
        identity_control="Same pixels, different ownership: loss-ignore/heldout diagnostic, not a learnable external label",
        visual_delta_gradient_gate=2/255., model="RGB64+predhull+centerheatmap Conv8/16/32+FC64",
        augmentation="wholepatch RGBchannelgain.5..1.7, exposure±.25, saturation.2..1.8, RGBnoiseσ3/255; geom unchanged",
        validation_steps=[256,512,768,1024], selection="best identifiable sourceCAL AUC; fixed updates unchanged",
        observation_points_changed=False, predicted_selfH_frozen=True, original_source_RGB_changes=0,
        entropy_unknown_rule="normalized binaryentropy>=.8; not an error or ownership certificate",
        bindings=[binding(p) for p in files])
    C.write(out/"PROTOCOL.json",protocol);protocol_sha=C.sha(out/"PROTOCOL.json")
    families=C.read(FAMILY)["records"];cached={r["id"]:r for r in C.read(CACHE)["records"]}
    dataset=make_dataset(families,cached,out);tr=dataset["train"];cal=dataset["calibration"]
    sets,provenance=fixed_evaluation_data(cal)
    model=LocalSupportCNN().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(SEED);generator=torch.Generator(device="cuda").manual_seed(SEED)
    validations=[];history=[];best=-1.;beststep=None;inactive=0;types=Counter();start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,STEPS+1):
        ids=rng.integers(0,len(tr["patches"]),size=BATCH)
        rgb=tr["patches"][ids].copy();expected=tr["expected"][ids];label=np.ones(BATCH,np.float32);active=np.ones(BATCH,bool)
        for i in range(BATCH//2,BATCH):
            kind=KINDS[int(rng.integers(len(KINDS)))];types[kind]+=1
            rgb[i],info=opaque_variant(rgb[i],expected[i],tr["background"][int(rng.integers(len(tr["background"])))],
                tr["patches"][int(rng.integers(len(tr["patches"])))],rng,kind)
            label[i]=0;active[i]=info["gradient_label_eligible"];inactive+=int(not active[i])
        values=augmentation(tensor_inputs(rgb,expected,"cuda"),generator)
        labels=torch.from_numpy(label).cuda();selection=torch.from_numpy(active).cuda()
        optimizer.zero_grad(set_to_none=True);logits=model(values);loss=F.binary_cross_entropy_with_logits(logits[selection],labels[selection])
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite local support loss")
        loss.backward();optimizer.step()
        if step%32==0:history.append(dict(step=step,loss=float(loss.detach())))
        if step%256==0:
            report=evaluate_source(model,cal,sets,provenance);score=report["pooled_identifiable_AUC"]
            if score>best:best=score;beststep=step;save_weights(out/"best_source_CAL.pt",model,step,protocol_sha)
            validations.append(dict(step=step,metrics=report))
            C.write(out/f"SOURCE_CAL_STEP{step}.json",report)
            print("LOCAL_OPACITY_TRAIN",step,"loss",round(float(loss.detach()),4),"sourceCAL_AUC",round(score,4),"seconds",round(time.monotonic()-start,2),flush=True)
    save_weights(out/"last.pt",model,STEPS,protocol_sha);torch.cuda.synchronize()
    C.write(out/"COMPLETION.json",dict(complete=True,steps=STEPS,batch=BATCH,source_patch_exposures=STEPS*BATCH,
        best_source_CAL_step=beststep,best_source_CAL_AUC=best,inactive_ambiguous_negative_exposures=inactive,
        negative_kind_exposures=dict(types),loss_history=history,validation_history=validations,
        eligible_train_queries=len(tr["patches"]),eligible_CAL_queries=len(cal["patches"]),
        parameter_count=sum(p.numel() for p in model.parameters()),CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        elapsed_training_validation_seconds=time.monotonic()-start,source_TEST_reads=0,real_TEST_reads=0,FINAL_reads=0,
        weights=[binding(out/"best_source_CAL.pt"),binding(out/"last.pt")],protocol=binding(out/"PROTOCOL.json"),
        labels_are_local_support_proxy_not_physical_visibility=True))
    print("LOCAL_OPACITY_COMPLETE",beststep,round(best,4),str(out),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=C.SCRATCH/"local_occlusion_v1")
    train(parser.parse_args())


if __name__=="__main__":main()
