"""Isolated, frozen MobileSAM supplier. Inputs contain RGB and assisted prompts only.

Run as a separate process after the parent protocol and prompts have been sealed.
This module never imports the pose, target-mask, audit, or evaluation pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

CATEGORY = "visible traffic cone"
IDS = (
    "plastic_night_01:038630", "wood_day_01:005286", "wood_night_01:030698",
    "plastic_day_01:020955", "plastic_day_01:014704", "wood_night_01:030898",
)
CHECKPOINT_SHA = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
PROMPT_KEYS = {
    "image_id", "ordinal", "rgb_relative_path", "rgb_sha256", "canvas_wh", "category",
    "status", "box_xyxy", "positive_point_xy", "coordinate_convention", "author_kind",
    "prompt_mode", "selection_provenance", "actual_tools_viewed", "frozen_at_utc", "explanation",
}
RGB_KEYS = {"schema", "image_id", "rgb_path", "rgb_file_sha256", "rgb_pixel_sha256", "image_hw"}
CONFIG_KEYS = {"schema", "vendor_root", "checkpoint_path", "checkpoint_sha256", "runtime_files_sha256",
               "model_name", "device", "cpu_threads", "walltime_limit_seconds"}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_keys(row, allowed, label):
    require(isinstance(row, dict) and set(row) == allowed, f"{label}: unexpected/missing fields")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_inputs(rgb_rows, prompts):
    require(tuple(r.get("image_id") for r in rgb_rows) == IDS, "RGB rows must contain fixed six IDs in order")
    require(tuple(r.get("image_id") for r in prompts) == IDS, "Prompt rows must contain fixed six IDs in order")
    for ordinal, (rgb, prompt) in enumerate(zip(rgb_rows, prompts), 1):
        exact_keys(rgb, RGB_KEYS, "neutral RGB")
        exact_keys(prompt, PROMPT_KEYS, "prompt")
        require(rgb["schema"] == "NON_TARGET_RGB_NEUTRAL_V1", "Neutral schema mismatch")
        require(prompt["ordinal"] == ordinal, "Prompt ordinal mismatch")
        require(prompt["category"] == CATEGORY and prompt["prompt_mode"] == "ASSISTED", "Category/mode mismatch")
        require(prompt["author_kind"] in ("human", "model"), "Prompt author must be explicit")
        require(prompt["coordinate_convention"] == "original_image_pixels_xyxy_exclusive_max", "Coordinate convention mismatch")
        h, w = rgb["image_hw"]
        require(isinstance(h, int) and isinstance(w, int) and h > 1 and w > 1, "Invalid canvas")
        require(prompt["canvas_wh"] == [w, h], "Prompt canvas mismatch")
        require(prompt["rgb_sha256"] == rgb["rgb_file_sha256"], "Prompt/RGB byte binding mismatch")
        require(Path(rgb["rgb_path"]).is_absolute(), "Absolute RGB path required")
        for key in ("rgb_file_sha256", "rgb_pixel_sha256"):
            require(isinstance(rgb[key], str) and len(rgb[key]) == 64 and all(c in "0123456789abcdef" for c in rgb[key]), "Invalid SHA")
        require(Path(rgb["rgb_path"]).name == Path(prompt["rgb_relative_path"]).name, "RGB basename mismatch")
        status = prompt["status"]
        require(status in ("ADMISSIBLE_NON_TARGET_EVIDENCE", "NO_ADMISSIBLE_NON_TARGET_EVIDENCE"), "Invalid prompt status")
        box, point = prompt["box_xyxy"], prompt["positive_point_xy"]
        if status == "NO_ADMISSIBLE_NON_TARGET_EVIDENCE":
            require(box is None and point is None, "Absent evidence cannot carry prompts")
            continue
        require(isinstance(box, list) and len(box) == 4 and isinstance(point, list) and len(point) == 2, "One box and positive point required")
        require(all(type(v) in (int, float) and float("-inf") < v < float("inf") for v in box + point), "Finite coordinates required")
        x0, y0, x1, y1 = box
        x, y = point
        require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "Box outside original canvas")
        require(x0 <= x < x1 and y0 <= y < y1, "Positive point must lie inside box")


class ReadBoundary:
    """Python audit-hook allowlist, not a hostile-code/native-library sandbox.

    Scientific libraries and CUDA require runtime files. Dataset access is exact
    file based; neither the repository nor the private experiment root is allowed
    as a read tree. The source process has no evaluator/candidate module import.
    """
    def __init__(self, read_files, code_roots, output):
        self.read_files = {str(Path(p).resolve()) for p in read_files}
        self.code_roots = tuple(str(Path(p).resolve()) + os.sep for p in code_roots)
        self.output = str(Path(output).resolve()) + os.sep
        self.allowed = set()
        self.denied = []
        self.enforcing = False

    def check_open(self, raw_path, mode="r", flags=0):
        if isinstance(raw_path, int):
            return
        path = str(Path(os.fsdecode(raw_path)).resolve())
        write = any(c in (mode or "") for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if write:
            okay = path.startswith(self.output) or path == "/dev/null"
        else:
            okay = (path in self.read_files or path.startswith(self.output)
                    or path.startswith(self.code_roots)
                    or path.startswith(("/usr/lib/", "/usr/share/zoneinfo/", "/lib/", "/proc/", "/sys/"))
                    or path in ("/dev/null", "/dev/urandom", "/dev/random", "/etc/localtime", "/etc/ld.so.cache"))
        if not okay:
            self.denied.append({"path": path, "write": write})
            raise PermissionError("Supplier read/write boundary rejected: " + path)
        self.allowed.add(path)

    def audit(self, event, args):
        if event == "open":
            self.check_open(args[0], args[1], args[2])
        elif event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind"):
            self.denied.append({"event": event})
            raise PermissionError("Supplier forbids network and child processes: " + event)

    def install(self):
        sys.addaudithook(self.audit)
        self.enforcing = True

    def report(self):
        return {"kind": "PYTHON_AUDIT_OPEN_ALLOWLIST", "enforcing": self.enforcing,
                "security_limit": "Cooperative Python boundary; native C/CUDA file access is not universally intercepted.",
                "allowed_open_paths": sorted(self.allowed), "denied_attempts": self.denied,
                "dataset_read_files": sorted(self.read_files), "code_roots": list(self.code_roots)}


def choose_index(quality):
    import numpy as np
    quality = np.asarray(quality)
    require(quality.shape == (3,) and np.isfinite(quality).all(), "Exactly three finite quality scores required")
    return int(np.argmax(quality))  # First original index wins an exact tie.


def infer_once(predictor, rgb, prompt, synchronize=lambda: None):
    """One image encoder and one combined box+positive-point decode, no retries."""
    import numpy as np
    synchronize()
    started = time.perf_counter()
    predictor.set_image(rgb, image_format="RGB")
    synchronize()
    encoder_seconds = time.perf_counter() - started
    started = time.perf_counter()
    masks, quality, logits = predictor.predict(
        point_coords=np.asarray([prompt["positive_point_xy"]], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        box=np.asarray(prompt["box_xyxy"], dtype=np.float32),
        mask_input=None, multimask_output=True, return_logits=False)
    synchronize()
    decoder_seconds = time.perf_counter() - started
    require(masks.dtype == np.bool_ and masks.shape == (3,) + rgb.shape[:2], "Binary masks must have original canvas")
    require(logits.shape == (3, 256, 256) and np.isfinite(logits).all(), "Unexpected low-resolution logits")
    return masks, quality, logits, choose_index(quality), encoder_seconds, decoder_seconds


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def final_receipt(counters, frame_timings, model_load_seconds, walltime_seconds, complete, denied):
    """Persist measurements even if imports, inference, or packaging later fails."""
    return {"schema": "NON_TARGET_SUPPLIER_FINAL_RECEIPT_V1", "clean_exit": complete,
            "counters": dict(counters), "per_frame_timings": list(frame_timings),
            "model_load_seconds": model_load_seconds, "walltime_seconds": walltime_seconds,
            "blocked_source_file_attempts": [r for r in denied if "path" in r],
            "blocked_runtime_operations": [r for r in denied if "event" in r],
            "network_and_child_process_boundary_relaxed": False}


def self_test():
    import numpy as np
    class FakePredictor:
        encoders = decoders = 0
        def set_image(self, rgb, image_format):
            require(image_format == "RGB", "RGB convention")
            self.encoders += 1
        def predict(self, **kwargs):
            self.decoders += 1
            require(kwargs["multimask_output"] is True and kwargs["return_logits"] is False and kwargs["mask_input"] is None, "No iterative prompts")
            require(kwargs["point_labels"].tolist() == [1] and kwargs["point_coords"].shape == (1, 2), "Exactly one positive point")
            return np.zeros((3, 8, 9), bool), np.asarray([.5, .8, .8]), np.zeros((3, 256, 256), np.float32)
    fake = FakePredictor()
    result = infer_once(fake, np.zeros((8, 9, 3), np.uint8), {"box_xyxy": [1, 1, 5, 5], "positive_point_xy": [3, 3]})
    require(result[3] == 1 and fake.encoders == 1 and fake.decoders == 1, "Single-call first tie contract")
    guard = ReadBoundary(["/neutral/rgb.png"], [sys.prefix], "/tmp/supplier-fixture-output")
    sentinels = ["/forbidden/SUPPORT_DECISIONS.json", "/forbidden/GT.json", "/forbidden/target_mask.npz", "/forbidden/candidate_bank.json", "/etc/passwd"]
    for sentinel in sentinels:
        try:
            guard.check_open(sentinel)
        except PermissionError:
            continue
        raise AssertionError("Forbidden source admitted")
    guard.check_open("/neutral/rgb.png")
    for event in ("socket.connect", "subprocess.Popen"):
        try:
            guard.audit(event, ())
        except PermissionError:
            continue
        raise AssertionError("Network/subprocess admitted")
    receipt = final_receipt({"image_encoder_forwards": fake.encoders, "mask_decoder_forwards": fake.decoders},
                            [{"image_id": "mock", "encoder_seconds": result[4], "decoder_seconds": result[5]}],
                            0.1, 0.2, False, guard.denied)
    require(receipt["counters"]["image_encoder_forwards"] == 1 and len(receipt["per_frame_timings"]) == 1
            and len(receipt["blocked_source_file_attempts"]) == 5 and len(receipt["blocked_runtime_operations"]) == 2,
            "Failed-run receipt must preserve counters, timing and denied categories")
    with tempfile.TemporaryDirectory(prefix="non-target-supplier-test-") as temporary:
        receipt_path = Path(temporary) / "FINAL_RECEIPT.json"
        write_json(receipt_path, receipt)
        require(json.loads(receipt_path.read_text()) == receipt, "Failure receipt serialization")
    print(json.dumps({"schema": "NON_TARGET_SUPPLIER_PURE_TEST_V1", "passed": True,
                      "fake_encoder_calls": fake.encoders, "fake_decoder_calls": fake.decoders,
                      "first_tie_index": result[3], "forbidden_source_sentinels": len(sentinels),
                      "failure_receipt_serialization": True,
                      "new_NN_calls": 0, "actual_RGB_reads": 0}, indent=2))


def run(args):
    import numpy as np
    from PIL import Image
    # Environment limits are also set by launcher before scientific imports.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    started = time.perf_counter()
    output = Path(args.output).resolve()
    require(not output.exists(), "Preserve previous attempts: output directory must be new")
    output.mkdir(parents=True)
    runtime_tmp = output / "runtime_tmp"
    runtime_tmp.mkdir()
    tempfile.tempdir = str(runtime_tmp)
    for key in ("TMPDIR", "TEMP", "TMP", "WANDB_DIR", "WANDB_CONFIG_DIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
        os.environ[key] = str(runtime_tmp)
    os.environ["WANDB_MODE"] = "disabled"
    for path, expected in ((args.neutral_rgb, args.rgb_sha256), (args.prompts, args.prompt_sha256), (args.model_config, args.config_sha256)):
        require(sha_file(path) == expected, "Sealed manifest SHA mismatch: " + path)
    rgb_rows, prompts = load_jsonl(args.neutral_rgb), load_jsonl(args.prompts)
    validate_inputs(rgb_rows, prompts)
    config = json.loads(Path(args.model_config).read_text())
    exact_keys(config, CONFIG_KEYS, "model config")
    require(config["schema"] == "NON_TARGET_MOBILESAM_CONFIG_V1" and config["model_name"] == "vit_t", "Fixed model contract")
    require(config["checkpoint_sha256"] == CHECKPOINT_SHA and 1 <= config["cpu_threads"] <= 2, "Fixed checkpoint/worker bound")
    require(0 < config["walltime_limit_seconds"] <= 1800, "GPU walltime exceeds authorized ceiling")
    require(config["device"] in ("cuda:0", "cpu"), "One local CUDA device or CPU required")
    vendor = Path(config["vendor_root"]).resolve()
    checkpoint = Path(config["checkpoint_path"]).resolve()
    runtime = {str((vendor / rel).resolve()): expected for rel, expected in config["runtime_files_sha256"].items()}
    require(str(vendor / "mobile_sam/predictor.py") in runtime and str(vendor / "mobile_sam/build_sam.py") in runtime, "Runtime API hashes required")
    allowed_files = [args.neutral_rgb, args.prompts, args.model_config, __file__, checkpoint] + list(runtime) + [r["rgb_path"] for r in rgb_rows]
    guard = ReadBoundary(allowed_files, [sys.prefix, vendor / "mobile_sam"], output)
    guard.install()
    # Actual denied reads, not only path-name tests, prove the audit hook is active.
    sentinel_checks = []
    for name in ("SUPPORT_DECISIONS.json", "GT.json", "target_mask.npz", "candidate_bank.json"):
        sentinel = str(output.parent / ("forbidden_" + name))
        try:
            with open(sentinel, "rb"):
                pass
        except PermissionError:
            sentinel_checks.append({"path": sentinel, "blocked_before_read": True})
        else:
            raise AssertionError("Runtime sentinel admitted")
    # These expected sentinel attempts are identified separately from violations.
    guard.denied.clear()
    write_json(output / "BOUNDARY_SENTINELS.json", sentinel_checks)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Supplier walltime limit")))
    signal.alarm(int(config["walltime_limit_seconds"]))
    records = []
    counters = {"image_encoder_forwards": 0, "prompt_decoder_calls": 0, "prompt_encoder_forwards": 0, "mask_decoder_forwards": 0}
    frame_timings = []
    model_load_seconds = None
    complete = False
    try:
        for path, expected in runtime.items():
            require(sha_file(path) == expected, "Vendor runtime changed: " + path)
        require(sha_file(checkpoint) == CHECKPOINT_SHA, "Checkpoint bytes changed")
        sys.path.insert(0, str(vendor))
        import torch
        from mobile_sam import SamPredictor, sam_model_registry
        torch.set_num_threads(config["cpu_threads"])
        torch.set_num_interop_threads(1)
        require(not config["device"].startswith("cuda") or torch.cuda.is_available(), "Frozen CUDA runtime unavailable; no silent fallback")
        model_start = time.perf_counter()
        model = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).to(device=config["device"]).eval()
        require(float(model.mask_threshold) == 0.0, "Expected vendor logit threshold zero")
        def count(key):
            def hook(*_):
                counters[key] += 1
            return hook
        handles = [model.image_encoder.register_forward_hook(count("image_encoder_forwards")),
                   model.prompt_encoder.register_forward_hook(count("prompt_encoder_forwards")),
                   model.mask_decoder.register_forward_hook(count("mask_decoder_forwards"))]
        predictor = SamPredictor(model)
        synchronize = (lambda: torch.cuda.synchronize(config["device"])) if config["device"].startswith("cuda") else (lambda: None)
        synchronize()
        model_load_seconds = time.perf_counter() - model_start
        for rgb_row, prompt in zip(rgb_rows, prompts):
            row_start = time.perf_counter()
            image_id = rgb_row["image_id"]
            base = {"schema": "NON_TARGET_INPUT_V1", "image_id": image_id, "supplier_mode": "ASSISTED",
                    "category": CATEGORY, "prompt_author_kind": prompt["author_kind"],
                    "prompt_manifest_sha256": args.prompt_sha256, "rgb_file_sha256": rgb_row["rgb_file_sha256"],
                    "rgb_pixel_sha256": rgb_row["rgb_pixel_sha256"], "image_hw": rgb_row["image_hw"],
                    "checkpoint_sha256": CHECKPOINT_SHA, "model_name": "MobileSAM vit_t",
                    "binary_dtype": "bool", "mask_threshold": "vendor original-resolution logits > 0.0",
                    "resize_policy": "vendor ResizeLongestSide then native postprocess_masks to original H,W",
                    "pixel_sha_convention": "SHA256 of C-contiguous uint8 mask (0/1), row-major H,W; no encoding header",
                    "nearest_pixel_convention": "floor(x + 0.5), floor(y + 0.5), in-range original pixel only",
                    "quality_selection_rule": "maximum uncalibrated quality; exact tie first original index",
                    "reviewer_kind": "pending", "admitted_for_rerank": False, "physical_depth_order_asserted": False,
                    "selected_mask_path": None, "all_masks_path": None, "selected_index": None,
                    "alternative_pixel_sha256": [], "quality": [], "encoder_calls": 0, "decoder_calls": 0,
                    "encoder_seconds": 0.0, "decoder_seconds": 0.0}
            data = Path(rgb_row["rgb_path"]).read_bytes()
            require(sha_bytes(data) == rgb_row["rgb_file_sha256"], "Actual RGB byte SHA mismatch")
            with Image.open(io.BytesIO(data)) as image:
                require(image.mode == "RGB", "Original RGB required; no implicit channel conversion")
                rgb = np.asarray(image).copy()
            require(rgb.shape == tuple(rgb_row["image_hw"]) + (3,) and rgb.dtype == np.uint8, "Actual RGB canvas mismatch")
            require(sha_bytes(rgb.tobytes(order="C")) == rgb_row["rgb_pixel_sha256"], "Decoded RGB pixel SHA mismatch")
            if prompt["status"] == "NO_ADMISSIBLE_NON_TARGET_EVIDENCE":
                base.update(status="NO_ADMISSIBLE_NON_TARGET_EVIDENCE", supplier_seconds=time.perf_counter()-row_start)
                records.append(base)
                with (output / "NON_TARGET_INPUT_MANIFEST.jsonl").open("a") as ledger:
                    ledger.write(json.dumps(base, ensure_ascii=False, allow_nan=False) + "\n")
                continue
            before = dict(counters)
            with torch.inference_mode():
                masks, quality, logits, index, enc_s, dec_s = infer_once(predictor, rgb, prompt, synchronize)
            counters["prompt_decoder_calls"] += 1
            frame_timings.append({"image_id": image_id, "encoder_seconds": enc_s, "decoder_seconds": dec_s})
            require(counters["image_encoder_forwards"] - before["image_encoder_forwards"] == 1 and counters["mask_decoder_forwards"] - before["mask_decoder_forwards"] == 1 and counters["prompt_encoder_forwards"] - before["prompt_encoder_forwards"] == 1, "Actual forward-call count contract violated")
            prefix = image_id.replace(":", "__")
            packet = output / (prefix + "__alternatives.npz")
            np.savez_compressed(packet, masks=masks, quality=quality, low_resolution_logits=logits,
                                original_indices=np.arange(3, dtype=np.int32), selected_index=np.asarray(index, np.int32))
            mask_paths, mask_sha = [], []
            for alternative_index, mask in enumerate(masks):
                binary = np.ascontiguousarray(mask, dtype=np.uint8)
                path = output / (prefix + f"__mask_{alternative_index}.png")
                Image.fromarray(binary * 255, mode="L").save(path)
                mask_paths.append(str(path))
                mask_sha.append(sha_bytes(binary.tobytes(order="C")))
            base.update(status="PREDICTED_PENDING_FIXED_MASK_REVIEW", all_masks_path=str(packet),
                        all_masks_file_sha256=sha_file(packet), alternative_mask_paths=mask_paths,
                        alternative_mask_file_sha256=[sha_file(p) for p in mask_paths],
                        selected_mask_path=mask_paths[index], selected_mask_file_sha256=sha_file(mask_paths[index]),
                        selected_pixel_sha256=mask_sha[index], alternative_pixel_sha256=mask_sha,
                        quality=[float(v) for v in quality], selected_index=index,
                        original_alternative_indices=[0, 1, 2], encoder_calls=1, decoder_calls=1,
                        encoder_seconds=enc_s, decoder_seconds=dec_s, supplier_seconds=time.perf_counter()-row_start,
                        foreground_pixels=[int(m.sum()) for m in masks], prompt_box_xyxy=prompt["box_xyxy"],
                        positive_point_xy=prompt["positive_point_xy"])
            records.append(base)
            with (output / "NON_TARGET_INPUT_MANIFEST.jsonl").open("a") as ledger:
                ledger.write(json.dumps(base, ensure_ascii=False, allow_nan=False) + "\n")
            predictor.reset_image()
        for handle in handles:
            handle.remove()
        require(not any("path" in attempt for attempt in guard.denied), "Unexpected forbidden source-file access attempt")
        (output / "NON_TARGET_INPUT_MANIFEST.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
        write_json(output / "SUPPLIER_RUN.json", {"schema": "NON_TARGET_SUPPLIER_RUN_V1", "complete": True,
            "input_rows": len(records), "admissible_prompts": sum(r["encoder_calls"] for r in records),
            "counters": counters, "model_load_seconds": model_load_seconds, "walltime_seconds": time.perf_counter()-started,
            "device": config["device"], "torch_version": torch.__version__, "python_executable": sys.executable,
            "api_source": str(vendor / "mobile_sam/predictor.py"), "checkpoint_sha256": CHECKPOINT_SHA,
            "prompt_manifest_sha256": args.prompt_sha256, "neutral_rgb_manifest_sha256": args.rgb_sha256,
            "model_config_sha256": args.config_sha256, "supplier_source_sha256": sha_file(__file__),
            "quality_is_calibrated_probability": False, "selection_gate": "AWAITING_FIXED_SELECTED_MASK_RGB_REVIEW",
            "GPU_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(config["device"]) if config["device"].startswith("cuda") else 0})
        complete = True
    except BaseException as error:
        write_json(output / "SUPPLIER_FAILURE.json", {"type": type(error).__name__, "message": str(error),
                   "rows_finished": len(records), "counters": counters, "walltime_seconds": time.perf_counter()-started})
        raise
    finally:
        signal.alarm(0)
        write_json(output / "SUPPLIER_FINAL_RECEIPT.json",
                   final_receipt(counters, frame_timings, model_load_seconds,
                                 time.perf_counter() - started, complete, guard.denied))
        write_json(output / "SOURCE_ACCESS_AUDIT.json", guard.report())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--neutral-rgb")
    parser.add_argument("--prompts")
    parser.add_argument("--model-config")
    parser.add_argument("--output")
    parser.add_argument("--rgb-sha256")
    parser.add_argument("--prompt-sha256")
    parser.add_argument("--config-sha256")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(all(getattr(args, name) for name in ("neutral_rgb", "prompts", "model_config", "output", "rgb_sha256", "prompt_sha256", "config_sha256")), "All sealed input arguments required")
    run(args)


if __name__ == "__main__":
    main()
