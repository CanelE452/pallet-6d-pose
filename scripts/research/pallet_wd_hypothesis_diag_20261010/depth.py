"""Pinned official Metric3D-v2 Small adapter; private RGB/depth only, no GT.

``prepare`` verifies the downloaded files and imports the factory without creating
a model. ``cache`` is the Stage-3 entry point, to be invoked after the Stage-2
report. Neither function changes the existing Python environment or source tree.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import urllib.request

import cv2
import numpy as np

SOURCE_COMMIT = "eb5b6fac0dc155e4e52f576e304fbf11655ff339"
SOURCE_ARCHIVE_SHA256 = "6859f6024544916afe44adcbbcf7e5837032db732e406a5129df44240c2dc223"
HF_REVISION = "80d2d1410afb4b23cd9d18c6be9144483d4b70b6"
WEIGHT_NAME = "metric_depth_vit_small_800k.pth"
WEIGHT_BYTES = 150120967
WEIGHT_SHA256 = "b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6"
INPUT_HW = (616, 1064)
RGB_MEAN = (123.675, 116.28, 103.53)
RGB_STD = (58.395, 57.12, 57.375)
CANONICAL_FOCAL = 1000.0
FRONT_SHRINK_FACTOR = 0.85


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def fetch(private_root):
    """Fetch only the prechosen official source/Small weight into a fresh folder."""
    root=Path(private_root).resolve()
    if root.exists():
        raise FileExistsError("Pinned fetch requires a fresh isolated private directory")
    root.mkdir(parents=True)
    specifications=[("official-source.tar.gz",
        f"https://codeload.github.com/YvanYin/Metric3D/tar.gz/{SOURCE_COMMIT}",
        SOURCE_ARCHIVE_SHA256,49543259),(WEIGHT_NAME,
        f"https://huggingface.co/JUGGHM/Metric3D/resolve/{HF_REVISION}/{WEIGHT_NAME}",
        WEIGHT_SHA256,WEIGHT_BYTES)]
    files=[]
    for basename,url,expected,size in specifications:
        started=time.perf_counter();path=root/basename
        with urllib.request.urlopen(url,timeout=60) as response, path.open("xb") as out:
            while block:=response.read(1048576):out.write(block)
        actual=sha256(path)
        if actual != expected or path.stat().st_size != size:
            raise RuntimeError("Official pinned artifact SHA/size mismatch; inference forbidden")
        files.append(dict(basename=basename,url=url,sha256=actual,bytes=size,seconds=time.perf_counter()-started))
    source=root/"Metric3D";source.mkdir()
    with tarfile.open(root/"official-source.tar.gz","r:gz") as tar:
        for member in tar.getmembers():
            relative=Path(*Path(member.name).parts[1:])
            if not member.isfile() or not relative.parts:continue
            selected=relative.parts[0]=="mono" or relative.name in ("hubconf.py","requirements_v2.txt","LICENSE","README.md","test_vit.sh")
            if selected:
                path=source/relative
                if not path.resolve().is_relative_to(source.resolve()):raise ValueError("Unsafe archive member")
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(member).read())
    receipt=dict(status="SOURCE_AND_FIXED_V2_SMALL_WEIGHT_FETCHED",source_commit=SOURCE_COMMIT,
        checkpoint_repo_revision=HF_REVISION,files=files,models_constructed=0,model_forwards=0,
        training_updates=0,private_only=True)
    write_json(root/"FETCH_RECEIPT.json",receipt)
    return receipt


def prepare(private_root):
    """Validate artifacts and import source; do not load weights/create a model."""
    root = Path(private_root).resolve()
    source = root / "Metric3D"
    weight = root / WEIGHT_NAME
    if weight.stat().st_size != WEIGHT_BYTES or sha256(weight) != WEIGHT_SHA256:
        raise RuntimeError("Pinned official ViT-Small weight size/SHA mismatch")
    receipt = json.loads((root / "FETCH_RECEIPT.json").read_text())
    # The archived source and extracted files are both bound to the fetch receipt.
    archive = root / "official-source.tar.gz"
    if not archive.is_file() or sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Pinned official source archive is missing")
    if receipt["source_commit"] != SOURCE_COMMIT or receipt["checkpoint_repo_revision"] != HF_REVISION:
        raise RuntimeError("Official source/checkpoint revision mismatch")
    source_files = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            relative = Path(*Path(member.name).parts[1:])
            selected = relative.suffix == ".py" and (relative.parts[0] == "mono" or relative.name == "hubconf.py")
            if member.isfile() and selected:
                expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                if sha256(source / relative) != expected:
                    raise RuntimeError("Extracted official inference source was modified")
                source_files += 1
    for p in (root / "deps", source):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("pallet_pinned_metric3d_hub", source / "hubconf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dict(status="PREPARED_NO_MODEL_RUN", model="Metric3D-v2 ViT-Small RAFT-4",
        source_commit=SOURCE_COMMIT, source_archive_sha256=sha256(archive),
        source_hubconf_sha256=sha256(source / "hubconf.py"), hf_revision=HF_REVISION,
        verified_extracted_source_files=source_files,
        checkpoint=WEIGHT_NAME, checkpoint_bytes=WEIGHT_BYTES, checkpoint_sha256=WEIGHT_SHA256,
        input_hw=list(INPUT_HW), models_constructed=0, model_forwards=0, training_updates=0,
        official_attention_fallback="Torch Attention when xformers is absent; official source unchanged")


def preprocess(rgb, K):
    """Official hubconf RGB resize/pad/normalize, using matching native intrinsics."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Finite 3x3 matching native K with positive fx/fy is required")
    if not np.array_equal(K[2], [0., 0., 1.]) or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError("Official fx/fy/cx/cy adapter does not support skew/projective K")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Native uint8 RGB HxWx3 image required")
    h, w = rgb.shape[:2]
    scale = min(INPUT_HW[0] / h, INPUT_HW[1] / w)
    resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    rh, rw = resized.shape[:2]
    ph, pw = INPUT_HW[0] - rh, INPUT_HW[1] - rw
    pads = [ph // 2, ph - ph // 2, pw // 2, pw - pw // 2]
    padded = cv2.copyMakeBorder(resized, *pads, cv2.BORDER_CONSTANT, value=RGB_MEAN)
    # Keep the official float32 arithmetic and scalar fx de-canonicalization.
    array = padded.transpose(2, 0, 1).astype(np.float32)
    array = (array - np.asarray(RGB_MEAN, np.float32)[:, None, None]) / np.asarray(RGB_STD, np.float32)[:, None, None]
    return array, dict(native_hw=[h, w], resize_scale=float(scale), resized_hw=[rh, rw],
        pads_tblr=pads, scaled_intrinsic=[float(K[0, 0]*scale), float(K[1, 1]*scale),
                                       float(K[0, 2]*scale), float(K[1, 2]*scale)],
        canonical_to_real_scale=float(K[0, 0]*scale/CANONICAL_FOCAL))


def front_depth_median(depth, q_final):
    """Median in the predicted front quad (0..3), shrunk 15% about its centroid.

    Native pixel centers are rasterized by cv2.fillPoly after np.rint. Positive
    finite depth samples only; no new minimum-count threshold or GT is used.
    """
    depth = np.asarray(depth, np.float32)
    q = np.asarray(q_final, np.float64)
    if (depth.ndim != 2 or q.shape != (9, 2) or not np.isfinite(q[:4]).all()
            or np.any(np.all(q[:4] == -1, axis=1))):
        return dict(available=False, reason="invalid_depth_or_predicted_quad", pixels=0, median_m=None)
    quad = q[:4].mean(0) + FRONT_SHRINK_FACTOR * (q[:4] - q[:4].mean(0))
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(quad).astype(np.int32)], 1)
    values = depth[(mask != 0) & np.isfinite(depth) & (depth > 0)]
    return dict(available=bool(len(values)), reason=None if len(values) else "no_positive_finite_depth_in_quad",
        pixels=int(len(values)), median_m=float(np.median(values)) if len(values) else None,
        shrink_factor=FRONT_SHRINK_FACTOR, quad_native_px=quad.tolist())


class MetricDepthRunner:
    """Create only after Stage-2 reporting; no training, tuning, or GT input."""
    def __init__(self, private_root):
        module, self.provenance = prepare(private_root)
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("Fixed official GPU inference requires available CUDA")
        self.model = module.metric3d_vit_small(pretrain=False)
        checkpoint = torch.load(Path(private_root) / WEIGHT_NAME, map_location="cpu")
        loaded = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError("Official Small checkpoint/model keys do not exactly match")
        self.model.cuda().eval()
        self.provenance.update(models_constructed=1, device=str(torch.cuda.get_device_name(0)),
                               checkpoint_missing_keys=[], checkpoint_unexpected_keys=[])
        self.forwards = 0

    def infer(self, rgb, K):
        array, metadata = preprocess(rgb, K)
        t = self.torch
        tensor = t.from_numpy(array)[None].cuda()
        t.cuda.synchronize()
        started = time.perf_counter()
        with t.no_grad():
            predicted, _, _ = self.model.inference({"input": tensor})
        self.forwards += 1
        depth = predicted.squeeze()
        top, bottom, left, right = metadata["pads_tblr"]
        depth = depth[top:depth.shape[0]-bottom if bottom else depth.shape[0],
                      left:depth.shape[1]-right if right else depth.shape[1]]
        depth = t.nn.functional.interpolate(depth[None, None], metadata["native_hw"], mode="bilinear").squeeze()
        depth = t.clamp(depth * metadata["canonical_to_real_scale"], 0, 300)
        t.cuda.synchronize()
        metadata.update(inference_seconds=float(time.perf_counter()-started),
                        depth_units="m", official_clip_m=[0., 300.], model_forward=self.forwards)
        return depth.detach().cpu().numpy().astype(np.float32), metadata


def cache(records, private_root, output):
    """Full fixed input list; exclusive private output, stop on any input failure.

    Records contain id/population/image_path/K/raw_hw and optional crop_lrtb.
    K is already expressed in the cropped native frame. No GT, dimensions, pose,
    hypothesis error, or final-test membership enters model inference.
    """
    output = Path(output).resolve()
    if not output.is_relative_to(Path(private_root).resolve().parent):
        raise ValueError("Depth maps must remain under the isolated private parent directory")
    keys = [(r["population"], r["id"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Each fixed RGB is inferred once; duplicate population/id inputs are forbidden")
    if output.exists():
        raise FileExistsError("Use a fresh private depth-cache directory")
    output.mkdir(parents=True)
    _, pinned = prepare(private_root)
    rgb_hashes = {}
    inputs = []
    for r in records:
        image_sha = sha256(r["image_path"])
        if r.get("rgb_sha256") is not None and image_sha != r["rgb_sha256"]:
            raise RuntimeError("Pre-pinned RGB input SHA mismatch")
        rgb_hashes[(r["population"], r["id"])] = image_sha
        inputs.append(dict(id=r["id"], population=r["population"], rgb_sha256=image_sha,
                           K=r["K"], raw_hw=r["raw_hw"], crop_lrtb=r.get("crop_lrtb", [0,0,0,0])))
    write_json(output / "DEPTH_INFERENCE_LOCK.json", dict(status="PINNED_BEFORE_MODEL_CREATION_AND_FORWARD",
        model=pinned, adapter_source_sha256=sha256(__file__), inputs=inputs, frames=len(inputs),
        GT_used_for_inference=False, models_constructed=0, model_forwards=0))
    runner = MetricDepthRunner(private_root)
    rows = []
    for record in records:
        image = Path(record["image_path"])
        payload = image.read_bytes()
        if hashlib.sha256(payload).hexdigest() != rgb_hashes[(record["population"],record["id"])]:
            raise RuntimeError("RGB input changed after inference lock")
        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Input RGB failed to decode")
        left, top, right, bottom = map(int, record.get("crop_lrtb", [0, 0, 0, 0]))
        h, w = bgr.shape[:2]
        if min(left, top, right, bottom) < 0 or left+right >= w or top+bottom >= h:
            raise ValueError("Invalid fixed native crop")
        bgr = bgr[top:h-bottom if bottom else h, left:w-right if right else w]
        if list(bgr.shape[:2]) != list(record["raw_hw"]):
            raise RuntimeError("Native RGB size/K frame contract mismatch")
        depth, metadata = runner.infer(bgr[:, :, ::-1].copy(), record["K"])
        identity = str(record["population"]) + ":" + str(record["id"])
        name = hashlib.sha256(identity.encode()).hexdigest() + ".npz"
        path = output / name
        np.savez_compressed(path, depth_m=depth)
        rows.append(dict(id=record["id"], population=record["population"], cache=name,
            cache_sha256=sha256(path), rgb_sha256=hashlib.sha256(payload).hexdigest(),
            K=record["K"], crop_lrtb=[left,top,right,bottom], **metadata))
        if len(rows) % 100 == 0:
            print(json.dumps({"depth_cached":len(rows),"model_forwards":runner.forwards}), flush=True)
    manifest = dict(status="DEPTH_CACHE_COMPLETE", private_only=True, GT_used_for_inference=False,
        checkpoint=runner.provenance, rows=rows, frames=len(rows), model_forwards=runner.forwards,
        training_updates=0, fresh_model_selection=False)
    write_json(output / "DEPTH_CACHE_MANIFEST.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "prepare", "cache"))
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-after-stage2", action="store_true",
                        help="Stage-3 entry point, invoked only after Stage-2 report publication")
    args = parser.parse_args()
    if args.action == "fetch":
        receipt=fetch(args.private_root)
        print(json.dumps({"status":receipt["status"],"files":receipt["files"],"model_forwards":0}))
        return
    if args.action == "prepare":
        _, receipt = prepare(args.private_root)
        print(json.dumps(receipt, ensure_ascii=False))
        return
    if not args.run_after_stage2 or args.records is None or args.output is None:
        parser.error("cache requires completed Stage-2 reporting, --run-after-stage2, --records and fresh --output")
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    receipt = cache(records, args.private_root, args.output)
    print(json.dumps({"status":receipt["status"],"frames":receipt["frames"],"model_forwards":receipt["model_forwards"]}))


if __name__ == "__main__":
    main()
