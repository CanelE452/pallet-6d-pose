"""Independently verify the completed real HTML, coordinates and line evidence.

Reads final saved outputs only. No model forward, GPU, source edit, browser
desktop opening, or notification. A missing/incomplete new12-plus-reused9 experiment
fails; generated fixtures cannot satisfy the actual-artifact contract.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html.parser
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from scripts.research.pallet_dht_joint_v1.line_targets import EDGES, ROLE_NAMES
from scripts.research.pallet_dht_coupling_v2.aggregate import NEW_ARMS, REFERENCE_ARMS

ARMS = REFERENCE_ARMS + NEW_ARMS
SEEDS = (1, 2, 3)
TITLE = "Pallet DHT Coupling · 점·선 결합 학습 검증"
CASE = "eval_pallet07:1778652166837872128"
POSE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DataScript(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside, self.chunks, self.external = False, [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == "report-data":
            self.inside = True
        if tag in ("script", "img", "link"):
            value = attrs.get("src", attrs.get("href", ""))
            if value and not value.startswith("data:"):
                self.external.append(value)

    def handle_endtag(self, tag):
        if tag == "script":
            self.inside = False

    def handle_data(self, data):
        if self.inside:
            self.chunks.append(data)


def exact(actual, expected, name):
    require(actual == expected, f"HTML/saved artifact mismatch: {name}")


def close(actual, expected, name, tolerance=1e-10):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), f"Malformed {name}")
    delta = float(np.max(np.abs(a - b))) if a.size else 0.
    require(delta <= tolerance, f"{name} differs by {delta} (limit {tolerance})")
    return delta


def independent_backprojection(probability, height, width):
    """Explicit (K_rho A)^T H / (K_rho A)^T1, without the sparse mm helper.

    The frozen model applies a symmetric [1/4,1/2,1/4] rho kernel before
    transpose voting. Boundary normalization uses that same operator.
    """
    probability = np.asarray(probability, dtype=np.float64)
    require(probability.shape == (12, 90, 113), "Expected twelve frozen 90×113 Hough channels")
    theta = np.arange(90, dtype=np.float64) * (math.pi / 90)
    cosine, sine = np.cos(theta), np.sin(theta)
    cosine[np.abs(cosine) < 1e-12] = 0.
    sine[np.abs(sine) < 1e-12] = 0.
    yy, xx = np.meshgrid(np.arange(height) - (height - 1) / 2,
                         np.arange(width) - (width - 1) / 2, indexing="ij")
    rho_index = (cosine[:, None] * xx.ravel()[None] + sine[:, None] * yy.ravel()[None] + 28.) / .5
    nearest = np.rint(rho_index)
    rho_index = np.where(np.abs(rho_index - nearest) < 1e-10, nearest, rho_index)
    lo = np.floor(rho_index).astype(int)
    alpha = rho_index - lo
    require(lo.min() >= 0 and (lo + 1).max() < 113, "Unexpected voting support outside frozen lattice")
    angles = np.arange(90)[:, None]
    padded = np.pad(probability, ((0, 0), (0, 0), (1, 1)))
    smoothed = .25 * padded[..., :-2] + .5 * padded[..., 1:-1] + .25 * padded[..., 2:]
    sampled = smoothed[:, angles, lo] * (1. - alpha) + smoothed[:, angles, lo + 1] * alpha
    mass = np.ones(113, dtype=float)
    mass[[0, -1]] = .75
    denominator = (mass[lo] * (1. - alpha) + mass[lo + 1] * alpha).mean(axis=0)
    return (sampled.mean(axis=1) / denominator[None]).reshape(12, height, width)


def independent_affine(raw_shape, input_shape):
    raw_h, raw_w = map(int, raw_shape)
    padded_h, padded_w = raw_h + 200, raw_w + 200
    ratio = min(640. / padded_h, 640. / padded_w)
    resized_w, resized_h = round(padded_w * ratio), round(padded_h * ratio)
    dw, dh = (640 - resized_w) % 32, (640 - resized_h) % 32
    expected_shape = [resized_h + dh, resized_w + dw]
    exact(list(map(int, input_shape)), expected_shape, "rect640 actual input shape")
    left, top = round(dw / 2 - .1), round(dh / 2 - .1)
    return np.array([[ratio, 0., 100 * ratio + left], [0., ratio, 100 * ratio + top]], dtype=float)


def verify_display_map(display, exact_values, label):
    values = np.asarray(exact_values, dtype=np.float32)
    exact(display["shape"], list(values.shape), label + " shape")
    lo, hi = values.min(axis=(-2, -1)), values.max(axis=(-2, -1))
    close(display["minimum"], lo, label + " lower scale", 0.)
    close(display["maximum"], hi, label + " upper scale", 0.)
    actual = np.frombuffer(base64.b64decode(display["bytes"], validate=True), dtype=np.uint8).reshape(values.shape)
    expected = np.rint((values - lo[:, None, None]) / np.maximum(hi - lo, 1e-12)[:, None, None] * 255).astype(np.uint8)
    require(np.array_equal(actual, expected), f"HTML byte map differs from saved {label}")


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        if expected is not None:
            require(value == expected, f"Bound input changed: {path}")
        self.hashes[str(path)] = value
        return path

    def read(self, path, expected=None):
        return read(self.bind(path, expected))


def csv_errors(inputs, path):
    path = inputs.bind(path)
    rows = {row["frame_id"]: row for row in csv.DictReader(path.open()) if row["kind"] == "POSITIVE"}
    require(len(rows) == 319, "All 319 positive CSV rows are required")
    result = {}
    for key, row in rows.items():
        values = [float(v) for v in row["top_keypoint_supervised_errors_px"].split(";") if v]
        result[key] = float(np.mean(values)) if values and row["top_iou50_match"] == "True" else None
    return result


def audit_saved_data(run_dir, inputs):
    render = inputs.read(run_dir / "REPORT_RENDER.json")
    require(render.get("complete") and render.get("PASS") and render.get("experiment_complete")
            and render.get("n_completed_evaluations") == 12 and render.get("n_verified_reference_evaluations") == 9 and render.get("n_gallery_model_runs") == 21 and render.get("n_gallery_frames") == 319,
            "Actual visual QA requires completed new12/reused9/319-frame report")
    require(not any(Path(path).name in ("VISUAL_QA.json", "ACTUAL_VISUAL_QA.json") for path in render["input_sha256"]),
            "Report inputs must not create a circular visual-QA binding")
    for path, digest in render["input_sha256"].items():
        inputs.bind(path, digest)
    page = inputs.bind(run_dir / "index.html", render["html_sha256"])
    parser = DataScript()
    parser.feed(page.read_text())
    require(not parser.external, "Offline report contains an external rendering dependency")
    data = json.loads(''.join(parser.chunks))
    exact(data["arms"], list(ARMS), "seven registered arms")
    exact(data["edges"], [list(edge) for edge in EDGES], "twelve line endpoints")
    exact(data["roles"], list(ROLE_NAMES), "twelve line roles")
    exact(data["required_case"], CASE, "required problem case")
    lut = np.asarray(data["color_lut"])
    require(lut.shape == (256, 3) and np.isfinite(lut).all() and lut.min() >= 0 and lut.max() <= 255,
            "Malformed native palette lookup")
    exact(data["overlay_alpha"], .5, "skill analysis overlay alpha")
    summary = inputs.read(run_dir / "SUMMARY.json")
    verdict = inputs.read(run_dir / "VERDICT.json")
    require(summary.get("complete") and summary.get("PASS") and verdict.get("complete") and verdict.get("PASS"), "Final statistics/verdict incomplete")
    require(len(summary["runs"]) == 21, "Summary must contain12 new and9 reused runs")
    manifest = inputs.read(ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json")
    require(manifest["role"] == "DEV" and manifest["expected_count"] == 319, "Only canonical DEV319 is allowed")
    frames = {row["id"]: row for row in data["frames"]}
    items = {row["frame_id"]: row for row in manifest["items"]}
    exact(sorted(frames), sorted(items), "all 319 gallery IDs")
    require(CASE in frames, "Required problem case absent")
    errors_r0 = csv_errors(inputs, ROOT / "data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv")
    r0 = inputs.read(POSE / "predictions/R0.json")["frames"]
    r0_by_image = {str((ROOT / row["image"]).resolve()): r0[row["frame_id"]] for row in inputs.read(POSE / "AXIS_REVIEW_MANIFEST.json")["frames_list"]}
    image_hashes, encoded_hashes, frame_by_path = {}, {}, {}
    for key, frame in frames.items():
        exact(sorted(frame["runs"]), sorted(f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS),
              key + " exact21 model labels")
        item = items[key]
        image_path = inputs.bind(ROOT / item["image_path"])
        image_hashes[str(image_path)] = inputs.hashes[str(image_path)]
        frame_by_path[str(image_path)] = frame
        annotation = inputs.read(ROOT / item["gt_v2_path"])["objects"][0]["keypoint_annotations"]
        expected_gt = [a["xy"] if a["visibility"] > 0 else None for a in annotation]
        exact(frame["gt"], expected_gt, key + " GT supervision/raw coordinates")
        exact(frame["r0"], r0_by_image[str(image_path)].get("keypoints_xy"), key + " historical R0 points")
        exact(frame["r0_error"], errors_r0[key], key + " historical R0 frame error")
        difficulty = "missing" if errors_r0[key] is None else "easy_le10" if errors_r0[key] <= 10 else "moderate_10_20" if errors_r0[key] <= 20 else "hard_gt20"
        exact(frame["difficulty"], difficulty, key + " posthoc difficulty")
        exact(frame["session"], item["session_id"], key + " session")
        image = cv2.imread(str(image_path))
        require(image is not None, "Cannot decode a canonical positive source image")
        exact([frame["height"], frame["width"]], list(image.shape[:2]), key + " original shape")
        # Recreate only JPEG encoding, never a model output. This independently
        # checks that each embedded image belongs to its original-coordinate data.
        h, w = image.shape[:2]
        if max(h, w) > 1000:
            image = cv2.resize(image, (round(w * 1000 / max(h, w)), round(h * 1000 / max(h, w))), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        require(ok and frame["image"].startswith("data:image/jpeg;base64,"), "Embedded image encoding differs")
        raw_encoded = base64.b64decode(frame["image"].split(',', 1)[1], validate=True)
        require(raw_encoded == jpg.tobytes(), f"Embedded image differs from original source: {key}")
        encoded_hashes[key] = hashlib.sha256(raw_encoded).hexdigest()
    controls = inputs.read(run_dir / "REUSED_CONTROLS.json")
    require(controls.get("complete") and controls.get("PASS") and controls["n_control_cells"] == 9, "Control reuse receipt differs")
    for path, digest in controls["input_sha256"].items():
        inputs.bind(path, digest)
    control_root = Path(controls["control_run_dir"])
    n_predictions, n_evidence, max_backprojection, max_affine = 0, 0, 0., 0.
    evidence_counts = {}
    for arm in ARMS:
        for seed in SEEDS:
            label = f"{arm}_seed{seed}"
            directory = (control_root if arm in REFERENCE_ARMS else run_dir) / "evaluation" / label
            completion = inputs.read(directory / "COMPLETION.json")
            require(completion.get("complete") and completion.get("PASS") and completion.get("actual_positive_forwards") == 319
                    and completion.get("actual_negative_forwards") == 2689 and completion.get("baseline_candidate_copying") is False,
                    f"{label} is not an actual completed full-model evaluation")
            predictions = inputs.read(directory / "PREDICTIONS.json", completion["output_sha256"]["PREDICTIONS.json"])
            errors = csv_errors(inputs, directory / "PAPER_2D_per_frame.csv")
            evidence = inputs.read(directory / "LINE_EVIDENCE.json", completion["output_sha256"]["LINE_EVIDENCE.json"])
            saved_evidence = {row["image_key"]: row for row in evidence["samples"]}
            require(len(saved_evidence) == len(evidence["samples"]), "Duplicate evidence image key")
            expected_ids = set(evidence["selected_frame_ids"])
            if arm == "point_only":
                require(not saved_evidence and evidence["expected_absent"], "Point-only must not have invented Hough evidence")
            else:
                exact(sorted(row["frame_id"] for row in saved_evidence.values()), sorted(expected_ids), label + " fixed evidence selection")
                require(CASE in expected_ids, "Problem case has no saved actual Hough evidence")
            evidence_counts[label] = len(saved_evidence)
            seen = set()
            for image_key, metadata in predictions["frame_metadata"].items():
                if metadata["kind"] != "positive":
                    continue
                path = str((ROOT / image_key).resolve())
                require(path in frame_by_path, "Noncanonical image in saved positive predictions")
                frame = frame_by_path[path]
                key = frame["id"]
                require(key not in seen, "Duplicate actual positive prediction")
                seen.add(key)
                exact(metadata["image_sha256"], image_hashes[path], label + '/' + key + " image SHA")
                row = frame["runs"][label]
                candidates = predictions["frames"][image_key]
                top = max(candidates, key=lambda c: c["score"]) if candidates else {}
                for field, saved in [("points", "keypoints_xy"), ("box", "box_xyxy"), ("score", "score")]:
                    exact(row[field], top.get(saved), label + '/' + key + '/' + field)
                exact(row["error"], errors[key], label + '/' + key + "/frame mean error")
                exact(row["n_candidates"], len(candidates), label + '/' + key + "/candidate count")
                n_predictions += 1
                if image_key not in saved_evidence:
                    require(row["evidence"] is None, "HTML invented evidence for an uncaptured frame")
                    continue
                reference = saved_evidence[image_key]
                npz = inputs.bind(reference["path"], reference["sha256"])
                with np.load(npz, allow_pickle=False) as a:
                    e = row["evidence"]
                    require(e is not None, "Captured actual evidence absent from HTML")
                    height, width = map(int, a["feature_shape_hw"])
                    close(a["input_shape_hw"], [height * 16, width * 16], "P4 stride16", 0.)
                    close(a["original_shape_hw"], [frame["height"], frame["width"]], "evidence original size", 0.)
                    close(a["theta_radians"], np.arange(90) * math.pi / 90, "theta lattice", 2e-7)
                    close(a["rho_values"], np.arange(-56, 57) * .5, "rho lattice", 0.)
                    # Stable float64 sigmoid is independent of the saved Torch path.
                    logits = np.asarray(a["logits"], dtype=float)
                    sigmoid = np.exp(-np.logaddexp(0., -logits))
                    close(a["sigmoid_probability"], sigmoid, "saved logit sigmoid", 1e-7)
                    spatial = independent_backprojection(a["sigmoid_probability"], height, width)
                    max_backprojection = max(max_backprojection, close(a["normalized_backprojection"], spatial, "independent smoothed-rho normalized transpose", 2e-6))
                    affine = independent_affine([frame["height"], frame["width"]], a["input_shape_hw"])
                    max_affine = max(max_affine, close(a["raw_to_input_affine"], affine, "raw-to-letterbox affine", 1e-12))
                    close(e["affine"], affine, "HTML raw affine", 0.)
                    close(e["input_shape_hw"], a["input_shape_hw"], "HTML input size", 0.)
                    close(e["theta_degrees"], np.rad2deg(a["theta_radians"]), "HTML normal-angle axis", 0.)
                    close(e["rho_values"], a["rho_values"], "HTML signed-rho axis", 0.)
                    verify_display_map(e["hough"], a["sigmoid_probability"], "Hough sigmoid")
                    verify_display_map(e["spatial"], a["normalized_backprojection"], "normalized backprojection")
                    points = np.array([[0., 0.], [frame["width"] - 1., frame["height"] - 1.], [frame["width"] / 2, frame["height"] / 2]])
                    mapped = points * np.diag(affine[:, :2]) + affine[:, 2]
                    close((mapped - affine[:, 2]) / np.diag(affine[:, :2]), points, "raw/input coordinate round-trip", 1e-10)
                n_evidence += 1
            exact(sorted(seen), sorted(frames), label + " all positive IDs")
    exact(n_predictions, 319 * 21, "actual gallery prediction pairs")
    return data, dict(PASS=True, n_original_images=319, n_new_runs=12, n_reused_runs=9, n_total_displayed_runs=21, n_prediction_overlays=n_predictions,
        gt_visibility_and_ids_exact=True, all_top1_points_boxes_scores_exact=True,
        all_embedded_original_images_exact=True, all_frame_metric_and_posthoc_bins_exact=True,
        n_actual_evidence_samples=n_evidence, evidence_by_run=evidence_counts,
        independent_backprojection_max_abs_delta=max_backprojection, backprojection_tolerance=2e-6,
        raw_affine_max_abs_delta=max_affine, original_letterbox_roundtrip_PASS=True,
        all_twelve_role_display_maps_and_axes_exact=True, embedded_jpeg_sha256=encoded_hashes)


class Browser:
    """Small CDP client using an isolated headless Chrome profile."""
    def __init__(self, directory):
        import websocket
        self.websocket = websocket
        self.directory = directory
        self.profile = tempfile.TemporaryDirectory(prefix="pallet-dht-actual-qa-")
        launcher = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
        require(launcher is not None, "Installed Chrome/Chromium is required for actual browser QA")
        self.log = (directory / "chrome.log").open("w")
        self.process = subprocess.Popen([launcher, "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
            "--remote-debugging-port=0", "--remote-allow-origins=*", f"--user-data-dir={self.profile.name}", "about:blank"],
            stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
        self.serial, self.errors, self.console_errors, self.external_requests = 0, [], [], []
        port_file = Path(self.profile.name) / "DevToolsActivePort"
        for _ in range(150):
            if port_file.is_file():
                break
            require(self.process.poll() is None, "Headless Chrome exited before debugging port opened")
            time.sleep(.1)
        require(port_file.is_file(), "Chrome debugging port did not open")
        port = int(port_file.read_text().splitlines()[0])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=10) as response:
            targets = json.load(response)
        target = next(row for row in targets if row["type"] == "page")
        self.ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=60)
        self.call("Runtime.enable")
        self.call("Page.enable")
        self.call("Network.enable")
        self.call("Emulation.setDeviceMetricsOverride", dict(width=1560, height=1100, deviceScaleFactor=1, mobile=False))

    def call(self, method, params=None):
        self.serial += 1
        request_id = self.serial
        self.ws.send(json.dumps(dict(id=request_id, method=method, params=params or {})))
        while True:
            value = json.loads(self.ws.recv())
            event, payload = value.get("method"), value.get("params", {})
            if event == "Runtime.exceptionThrown":
                self.errors.append(payload.get("exceptionDetails"))
            if event == "Runtime.consoleAPICalled" and payload.get("type") == "error":
                self.console_errors.append(payload.get("args"))
            if event == "Network.requestWillBeSent":
                url = payload["request"]["url"]
                if url.startswith(("http://", "https://")):
                    self.external_requests.append(url)
            if value.get("id") == request_id:
                require("error" not in value, f"CDP {method} failed: {value.get('error')}")
                return value.get("result", {})

    def js(self, expression):
        result = self.call("Runtime.evaluate", dict(expression=expression, returnByValue=True, awaitPromise=True))
        require("exceptionDetails" not in result, f"Browser JavaScript evaluation failed: {result.get('exceptionDetails')}")
        return result.get("result", {}).get("value")

    def ready(self):
        require(self.js("new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{if(window.REPORT_READY){clearInterval(t);resolve(true)}else if(n++>1100){clearInterval(t);reject('report render timeout')}},40)})"), "Report did not become ready")

    def select(self, field, value):
        self.js(f"document.getElementById({json.dumps(field)}).value={json.dumps(str(value))};document.getElementById({json.dumps(field)}).dispatchEvent(new Event('change'));true")
        self.ready()

    def screenshot(self, path, clip=None):
        params = dict(format="png", captureBeyondViewport=True)
        if clip:
            params["clip"] = clip
        raw = self.call("Page.captureScreenshot", params)["data"]
        Path(path).write_bytes(base64.b64decode(raw))
        require(cv2.imread(str(path)) is not None, "Browser screenshot PNG is unreadable")

    def close(self):
        if hasattr(self, "ws"):
            self.ws.close()
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if hasattr(self, "log"):
            self.log.close()
        if hasattr(self, "profile"):
            self.profile.cleanup()


def audit_browser(run_dir, data):
    directory = run_dir / "actual_visual_qa"
    directory.mkdir(exist_ok=True)
    browser = Browser.__new__(Browser)
    checks = []
    try:
        browser.__init__(directory)
        browser.call("Page.navigate", dict(url=(run_dir / "index.html").as_uri()))
        browser.ready()
        state = browser.js("({title:document.title,frames:REPORT_FRAME_COUNT,current:REPORT_CURRENT_FRAME,arm:document.getElementById('arm').value,seed:document.getElementById('seed').value,role_count:document.getElementById('role').options.length,frame_options:document.getElementById('frame').options.length})")
        exact(state, dict(title=TITLE, frames=319, current=CASE, arm="balanced", seed="1", role_count=12, frame_options=319), "actual browser default")
        checks.append("Default opens the required actual case with balanced seed1 and all319 frames")
        # Decode every embedded image in bounded batches; no images or sources
        # outside the completed HTML are requested by this browser check.
        decoded = browser.js("(async()=>{let n=0;for(let start=0;start<DATA.frames.length;start+=12){await Promise.all(DATA.frames.slice(start,start+12).map(f=>new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>{if(i.naturalWidth<=0||i.naturalHeight<=0)reject('empty image');else{n++;resolve(true)}};i.onerror=()=>reject('image decode: '+f.id);i.src=f.image})));}return n})()")
        exact(decoded, 319, "all actual browser images decode")
        browser.screenshot(directory / "home.png")
        browser.js("window.QA_TRANSFORMS=[];window.QA_ORIGINAL_TRANSFORM=CanvasRenderingContext2D.prototype.setTransform;CanvasRenderingContext2D.prototype.setTransform=function(...a){if(this.canvas.id==='evidence')QA_TRANSFORMS.push(a);return QA_ORIGINAL_TRANSFORM.apply(this,a)};true")
        for arm in ARMS:
            browser.select("arm", arm)
            for seed in SEEDS:
                browser.select("seed", seed)
                exact(browser.js("document.getElementById('frame').options.length"), 319, f"{arm}/{seed} browser coverage")
                exact(browser.js("REPORT_CURRENT_FRAME"), CASE, "problem case retained across arms/seeds")
                if arm == "point_only":
                    exact(browser.js("document.getElementById('hough').height"), 1, "point-only must not show Hough map")
        checks.append("All seven registered arms and three seeds retain319 images; point-only has no invented Hough evidence")
        browser.select("arm", "balanced")
        browser.select("seed", 1)
        case = next(row for row in data["frames"] if row["id"] == CASE)
        affine = case["runs"]["balanced_seed1"]["evidence"]["affine"]
        expected_transform = [16 / affine[0][0], 0., 0., 16 / affine[1][1], -affine[0][2] / affine[0][0], -affine[1][2] / affine[1][1]]
        for role in range(12):
            browser.select("role", role)
            text = browser.js("document.getElementById('evidenceinfo').textContent")
            require(f"role {role} " in text, f"Role {role} did not render actual evidence")
            close(browser.js("QA_TRANSFORMS.at(-1)"), expected_transform, "actual Canvas feature-to-raw transform", 1e-10)
        checks.append("All12 actual roles render; intercepted Canvas transform equals independent inverse LetterBox mapping")
        for rank in ("best", "worst", "neutral", "evidence"):
            browser.select("rank", rank)
            require(browser.js("document.getElementById('frame').options.length") > 0, f"Actual {rank} gallery is empty")
        browser.select("reference", "R0")
        for difficulty in ("easy_le10", "moderate_10_20", "hard_gt20", "missing"):
            browser.select("difficulty", difficulty)
            count = browser.js("document.getElementById('frame').options.length")
            require(count >= 0, "Invalid filtered frame count")
        checks.append("Best/worst/neutral/evidence, R0 reference and all GT-posthoc difficulty filters render without errors")
        browser.select("difficulty", "all")
        browser.select("reference", "hough_joint")
        browser.select("rank", "case")
        browser.select("session", case["session"])
        exact(browser.js("REPORT_CURRENT_FRAME"), CASE, "session filter retains required case")
        browser.select("session", "all")
        browser.select("role", 7)  # rear_left_height GT4–7, fixed before new outcomes.
        for field in ("showgt", "showpoints", "showheat"):
            browser.js(f"document.getElementById('{field}').click();true")
            browser.ready()
            browser.js(f"document.getElementById('{field}').click();true")
            browser.ready()
        browser.js("document.getElementById('zoom').click();true")
        require(browser.js("document.getElementById('gallery').classList.contains('zoom')"), "Zoom toggle did not work")
        browser.js("document.getElementById('zoom').click();true")
        checks.append("Session/GT/points/heatmap/zoom controls work and restore original-coordinate overlays")
        runtime = read(run_dir / "RUNTIME.json")
        require(runtime.get("complete") and runtime.get("timing_collection_complete"), "Actual runtime collection incomplete")
        exact(sum(row["n"] for row in runtime["runs"]), 21*26*3, "actual runtime observations")
        body = browser.js("document.body.textContent")
        if runtime.get("parity_PASS") is False:
            require(runtime.get("PASS") is False and runtime["strict_failures"], "Invalid strict failure state")
            maximum = max(o["max_abs_delta_by_field"]["keypoints_xy"] for r in runtime["runs"] for o in r["observations"])
            require("PASS=false / parity_PASS=false" in body and f"{maximum:.9f}px" in body
                    and "참고 실측 시간" in body and "Accuracy 예측과의 parity를 검증했습니다." not in body,
                    "Actual report hides or misstates strict runtime parity failure")
        else:
            require(runtime.get("PASS") is True and "Accuracy 예측과의 parity를 검증했습니다." in body,
                    "Successful runtime verification text differs")
        checks.append("Runtime completion and strict parity are displayed separately, including any actual failure and maximum drift")
        # Capture the entire actual case panel, including raw points, left-height
        # heatmap and theta/rho plot, without changing the stored HTML.
        browser.js("document.getElementById('gallery').scrollIntoView();true")
        box = browser.js("(()=>{const r=document.getElementById('gallery').getBoundingClientRect();return{x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()")
        browser.screenshot(directory / "problem_case_left_height.png", box)
        require(not browser.errors and not browser.console_errors, "Actual report emitted JavaScript/console errors")
        require(not browser.external_requests, "Offline actual report requested external URLs")
        return dict(PASS=True, default=state, decoded_embedded_images=decoded, checks=checks,
            JS_exceptions=browser.errors, console_errors=browser.console_errors, external_http_requests=browser.external_requests,
            required_case=CASE, captured_role=7, captured_role_name="rear_left_height",
            runtime_strict_parity_PASS=runtime["parity_PASS"],
            actual_canvas_feature_to_raw_transform=expected_transform,
            screenshots={str(path.resolve()): sha(path) for path in (directory / "home.png", directory / "problem_case_left_height.png")})
    finally:
        browser.close()


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    require((run_dir / "PURPOSE.md").is_file(), "Purpose-declared output root required")
    destination = run_dir / "ACTUAL_VISUAL_QA.json"
    if destination.is_file():
        old = read(destination)
        require(old.get("complete") and old.get("PASS") and old["html_sha256"] == sha(run_dir / "index.html")
                and old["source_sha256"][str(Path(__file__).resolve())] == sha(__file__), "Existing actual QA differs; preserve and explicitly supersede it before rerunning")
        for key in ("input_sha256", "output_sha256", "source_sha256"):
            for path, digest in old[key].items():
                require(sha(path) == digest, f"Existing actual QA artifact changed: {path}")
        return old
    inputs = Inputs()
    data, numerical = audit_saved_data(run_dir, inputs)
    print(f"Actual saved-data QA PASS: {numerical['n_prediction_overlays']} overlays, {numerical['n_actual_evidence_samples']} independently checked Hough evidence samples", flush=True)
    browser = audit_browser(run_dir, data)
    # Source/output hashes are checked again after browser interaction.
    for path, digest in inputs.hashes.items():
        require(sha(path) == digest, f"Input changed while visual QA ran: {path}")
    result = dict(schema="pallet_dht_coupling_actual_visual_qa_v2", complete=True, PASS=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(), html_sha256=sha(run_dir / "index.html"),
        experiment_complete=True, n_completed_evaluations=12, n_verified_reference_evaluations=9, n_gallery_model_runs=21, n_gallery_frames=319, broken_local_images=0,
        scope="Automated independent actual-artifact, original-coordinate and headless-browser checks. Screenshots require a separate human visual review; this receipt does not claim one.",
        numerical=numerical, browser=browser, input_sha256=inputs.hashes,
        output_sha256=browser["screenshots"], source_sha256={str(Path(__file__).resolve()): sha(__file__), str(ROOT / "scripts/research/pallet_dht_joint_v1/line_targets.py"): sha(ROOT / "scripts/research/pallet_dht_joint_v1/line_targets.py")},
        actual_model_forwards=0, GPU_used=False, desktop_browser_opened=False, notification_sent=False)
    write(destination, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.run_dir)
    except Exception as exc:
        page = args.run_dir / "index.html"
        write(args.run_dir / "ACTUAL_VISUAL_QA_FAILURE.json", dict(complete=False, PASS=False,
            error_type=type(exc).__name__, error=str(exc), html_sha256=sha(page) if page.is_file() else None,
            source_sha256={str(Path(__file__).resolve()): sha(__file__)}, actual_model_forwards=0, GPU_used=False))
        raise
    print(json.dumps(dict(complete=result["complete"], PASS=result["PASS"], html_sha256=result["html_sha256"],
                         receipt=str(args.run_dir.resolve() / "ACTUAL_VISUAL_QA.json")), ensure_ascii=False), flush=True)
