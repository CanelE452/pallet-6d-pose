"""Frozen native-2D caches -> exact native SubPix/current F -> D1 input schema.

Preparation hashes/join checks only. Production is a separate ROOT-GO command.
No models, labels, references, historical backbone poses or scoring are loaded.
All238 identities survive, including unavailable F rows for the outer wrapper.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

SCHEMA = "BOUNDARY_ACTION_D1_UPSTREAM_INPUT_V1"
BACKBONES = ("dope", "resnet18")
STAGES = ("RAW_F", "N3_F", "N3_SUBPIX_F")
META_KEYS = frozenset(("schema", "id", "partition", "split", "session", "cluster_id", "image", "image_sha256", "raw_hw", "K", "native_xyz", "known_physical_dimensions_xyz_m", "prediction_object_type", "crop_transform", "upstream_provenance"))
SOURCES = ("scripts/research/pallet_d1_three_backbones_20261020_v1/upstream.py",
    "scripts/research/pallet_dim_conditioned_p_v1/pose.py",
    "scripts/research/pallet_training_free_compare_20261007_v1/methods.py",
    "scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py",
    "challenge/evaluation_v2/pnp_selector.py", "scripts/annotate/pallet_geometry.py")
BASE_MODELS = dict(dope=("weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth", "0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f"),
    resnet18=("data/pallet/results/pallet_resnet18_dimension_20261002_v1/DSNT_FULL_CONSTANT_epoch10.pt", "a94e55f028b704057c77fd045d96dd2dbb2ae6be2a49e945b23c821a30d8e1c4"))
SUBPIX = dict(winSize=[5, 5], zeroZone=[-1, -1], minimum_image_extent=15,
    criteria=dict(maxCount=40, epsilon=.001))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""): digest.update(block)
    return digest.hexdigest()


def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream: stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
            for value in values: stream.write((canonical(value) + "\n").encode())


def rows(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def _skip_value(text, index):
    """Skip an unneeded canonical packet value without numeric pose decoding."""
    while text[index].isspace(): index += 1
    if text[index] == '"':
        return json.decoder.scanstring(text, index + 1)[1]
    if text[index] not in "[{":
        while index < len(text) and text[index] not in ",}": index += 1
        return index
    stack = [text[index]]; index += 1
    while stack:
        char = text[index]
        if char == '"': index = json.decoder.scanstring(text, index + 1)[1]; continue
        if char in "[{": stack.append(char)
        elif char in "]}": stack.pop()
        index += 1
    return index


def metadata_only(line):
    """Read current camera/registered geometry/identity, skip saved pose/points."""
    decoder = json.JSONDecoder(); index = 1; output = {}
    while index < len(line):
        while index < len(line) and line[index].isspace(): index += 1
        if line[index] == "}": break
        key, index = decoder.raw_decode(line, index)
        while line[index].isspace(): index += 1
        if line[index] != ":": raise ValueError("Malformed canonical upstream packet")
        index += 1
        while line[index].isspace(): index += 1
        if key in META_KEYS: output[key], index = decoder.raw_decode(line, index)
        else: index = _skip_value(line, index)
        while index < len(line) and line[index].isspace(): index += 1
        if line[index] == ",": index += 1
        elif line[index] != "}": raise ValueError("Malformed canonical packet separator")
    if set(output) != META_KEYS: raise ValueError("Missing canonical metadata fields")
    if output["schema"] != SCHEMA: raise ValueError("Canonical upstream schema drift")
    return output


def _bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def _verify(binding):
    path = Path(binding["path"])
    if not path.is_file() or path.stat().st_size != binding["bytes"] or sha(path) != binding["sha256"]:
        raise ValueError("Bound input changed: " + str(path))
    return path


def _cached_points(block):
    """Finite native9 storage; explicit missingness retained separately."""
    valid = block["valid"]
    if len(valid) != 9 or any(type(value) is not bool for value in valid): raise ValueError("Explicit nine-slot valid mask required")
    if len(block["points"]) != 9: raise ValueError("Native nine-slot topology required")
    values = []
    for point, supported in zip(block["points"], valid):
        if supported:
            if len(point) != 2 or any(value is None or not math.isfinite(value) for value in point) or point == [-1, -1]:
                raise ValueError("Supported native slot is invalid")
            values.append([float(value) for value in point])
        else: values.append([-1.0, -1.0])
    return values


def cache_index(cache, backbone):
    if not (cache.get("schema") == "pallet_n3_completion_v3_prediction_v1" and cache.get("dataset") == "DEV319"
            and cache.get("complete") is True and cache.get("GT_inputs") is False and cache.get("camera_inputs") is False
            and cache.get("all_image_bytes_bound") is True and cache.get("backbone") == backbone
            and cache.get("coordinate_system") == "original_unpadded_pixels"):
        raise ValueError("Frozen prediction-only original-pixel DEV319 cache required")
    forbidden = {"gt", "truth", "pose", "reference", "target"}
    if any(forbidden & set(frame) for frame in cache["frames"]): raise ValueError("Cache contains forbidden scientific answers")
    index = {frame["id"]: frame for frame in cache["frames"]}
    if len(index) != 319 or len(cache["frames"]) != 319: raise ValueError("DEV319 cache identity drift")
    return index


def prepare(repo, canonical_upstream, asset_root, output, backbones=BACKBONES, seed=1):
    """Read-only native2D/RGB/hash join; no image decode, SubPix, pose or model."""
    if seed != 1: raise ValueError("Current YOLO path uses predeclared N3 seed1; no seed choice")
    repo, canonical_upstream, asset_root, out = map(lambda value: Path(value).resolve(), (repo, canonical_upstream, asset_root, output))
    if out.exists(): raise FileExistsError("Fresh preparation directory required")
    sources = {str(repo / relative): sha(repo / relative) for relative in SOURCES}
    input_bindings = [_bind(canonical_upstream), _bind(canonical_upstream.with_name("UPSTREAM_SEAL.json"))]
    old_seal = read(canonical_upstream.with_name("UPSTREAM_SEAL.json"))
    if old_seal.get("complete") is not True or old_seal["artifacts"][canonical_upstream.name] != sha(canonical_upstream):
        raise ValueError("Canonical upstream seal mismatch")
    with gzip.open(canonical_upstream, "rt") as stream: frames = [metadata_only(line) for line in stream if line.strip()]
    if len(frames) != 238 or len({frame["id"] for frame in frames}) != 238 or Counter(frame["partition"] for frame in frames) != Counter(TEST=49, REMAINDER=143, CAL=46):
        raise ValueError("Exact current238 identities/partitions required")
    prepared = {}; audit = {}; docs = repo / "_docs/experiments/pallet_n3_completion_v3"
    for backbone in backbones:
        if backbone not in BACKBONES: raise ValueError(backbone)
        cache_path = asset_root / f"data/pallet/results/pallet_n3_completion_v3/predictions/{backbone}_DEV319.json"
        cache = read(cache_path); input_bindings.append(_bind(cache_path))
        index = cache_index(cache, backbone)
        completion_path = docs / f"INFERENCE_{backbone.upper()}_DEV319_COMPLETE.json"
        completion = read(completion_path); input_bindings.append(_bind(completion_path))
        declared_cache = completion["predictions"]
        if not (completion.get("complete") is True and completion.get("backbone") == backbone and completion.get("dataset") == "DEV319"
                and completion.get("frames") == 319 and completion.get("GT_inputs") is False
                and declared_cache["sha256"] == sha(cache_path) and declared_cache["bytes"] == cache_path.stat().st_size):
            raise ValueError("Authoritative frozen inference completion/cache binding mismatch")
        for role in ("protocol", "selection", "training", "registry", "code"):
            declared = cache[role]; relative = Path(declared["path"])
            path = (repo if str(relative).startswith(("_docs/", "scripts/")) else asset_root) / relative
            actual = _bind(path)
            if actual["sha256"] != declared["sha256"] or actual["bytes"] != declared["bytes"]: raise ValueError("Cache parent mismatch: " + role)
            input_bindings.append(actual)
        base_rel, base_sha = BASE_MODELS[backbone]; base_path = asset_root / base_rel
        if sha(base_path) != base_sha: raise ValueError("Frozen base checkpoint mismatch")
        head_path = asset_root / f"data/pallet/results/pallet_n3_completion_v3/runs/{backbone}/seed1/last.pt"
        head_sha = sha(head_path); input_bindings.extend((_bind(base_path), _bind(head_path)))
        values = []; counts = Counter()
        for frame in frames:
            cached = index[frame["id"]]; raw = cached["predictions"]["base"]; n3 = cached["predictions"]["N3_seed1"]
            path = Path(frame["image"])
            if sha(path) != frame["image_sha256"] or cached["image_sha256"] != frame["image_sha256"] or cached["raw_shape_hw"] != frame["raw_hw"]:
                raise ValueError("Current/cached RGB canvas or byte identity differs: " + frame["id"])
            if sha(asset_root / cached["image_key"]) != frame["image_sha256"]: raise ValueError("Cache RGB path alias mismatch")
            if sorted(cached["dimensions_wdh_m"]) != sorted(frame["known_physical_dimensions_xyz_m"]):
                if not all(math.isclose(a, b, abs_tol=1e-10, rel_tol=0) for a, b in zip(sorted(cached["dimensions_wdh_m"]), sorted(frame["known_physical_dimensions_xyz_m"]))):
                    raise ValueError("Registered physical extents mismatch")
            for field in ("valid", "detected", "bbox", "score", "base_checkpoint_sha256"):
                if raw[field] != n3[field]: raise ValueError("N3 changed inherited base evidence: " + field)
            if n3["temperature"] != 1.0 or raw["base_checkpoint_sha256"] != base_sha or n3["n3_checkpoint_sha256"] != head_sha:
                raise ValueError("Frozen N3 seed1 checkpoint/temperature mismatch")
            if raw["points"][8] != n3["points"][8]: raise ValueError("N3 changed centroid slot")
            raw_points, n3_points = _cached_points(raw), _cached_points(n3)
            support = list(raw["valid"] if raw["detected"] else [False] * 9)
            counts["frames"] += 1; counts["RGB_exact"] += 1; counts["detected"] += bool(raw["detected"])
            counts["at_least6_supported_corners"] += sum(support[:8]) >= 6
            values.append(dict(template=frame, raw_points=raw_points, n3_points=n3_points, prediction_support=support,
                cached_valid9=raw["valid"], detected=raw["detected"], cached_status=raw["status"],
                cache_row_sha256=digest(cached), base_checkpoint_sha256=base_sha, n3_checkpoint_sha256=head_sha,
                corner_mapping="unchanged channels0..7;centroid8;original unpadded pixels;no permutation"))
        prepared[backbone] = values; audit[backbone] = dict(counts)
    out.mkdir(parents=True)
    artifacts = {}
    for backbone, values in prepared.items():
        path = out / f"{backbone}_PREPARED238.jsonl.gz"; save(path, values); artifacts[path.name] = sha(path)
    plan = dict(schema="D1_THREE_BACKBONE_UPSTREAM_PREPARATION_V1", complete=True, rows=238,
        canonical_yolo_N3_seed=1, upstream_seed=1, repo=str(repo), asset_root=str(asset_root), backbones=list(backbones),
        bindings=input_bindings, source_sha256=sources, artifacts=artifacts, audit=audit,
        missing_policy="Store missing2D as[-1,-1] with inherited mask. F input copy uses NaN for maskFalse; no invented corners. No detected object gives unavailable F. All238 rows retained.",
        geometry_policy="CurrentK/knownphysicalXYZ/RGB shared. Every available new F supplies its own cf_extents/nativeXYZ and physical Q; YOLO nativeXYZ is not reused for a valid different W/D result. Unavailable context is archived but not consumed.",
        method="frozen N3_seed1 -> native cornerSubPix(5,5),40,.001 -> total raw-base-anchored1% diagonalcap -> unchanged currentF",
        scientific_calls=dict(model_forwards=0, SubPix=0, pose=0, fits=0, GT_or_reference_reads=0, old_pose_decodes=0))
    write(out / "UPSTREAM_PREPARATION.json", plan)
    return plan


def _primitive(path, names, namespace):
    tree = ast.parse(Path(path).read_text()); nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names): raise ValueError("Frozen primitive missing")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def load_primitives(repo):
    import cv2
    import numpy as np
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    repo = Path(repo)
    solved = _primitive(repo / SOURCES[3], ["cuboid", "solve"], dict(np=np, cv2=cv2))
    pose = _primitive(repo / SOURCES[1], ["rotations", "infer"], dict(np=np, cv2=cv2, cuboid=solved["cuboid"], solve=solved["solve"], select_pnp_hypotheses=select_pnp_hypotheses))
    sub = _primitive(repo / SOURCES[2], ["_inputs", "correct", "cap_points"], dict(np=np, cv2=cv2, Counter=Counter, METHODS=("SUBPIX", "CVRANK"), _configuration=lambda: {"SUBPIX": SUBPIX}))
    return pose, sub


def _finite(value):
    if isinstance(value, dict): return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_finite(item) for item in value]
    if hasattr(value, "tolist"): return _finite(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def build_frame(prepared, pose, sub, image, stage_poses=True):
    """Actual SubPix/F only when invoked by separately authorized production."""
    import cv2
    import numpy as np
    frame = deepcopy(prepared["template"]); support = np.asarray(prepared["prediction_support"], bool)
    q0, qn = np.asarray(prepared["raw_points"], float), np.asarray(prepared["n3_points"], float)
    native, diagnostics = sub["correct"](cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), qn, support, "SUBPIX")
    final = sub["cap_points"](q0, native, image.shape[1], image.shape[0], support)
    stage_points = dict(RAW_F=q0, N3_F=qn, N3_SUBPIX_F=final); outcomes = {}; calls = 0
    for stage in STAGES if stage_poses else ("N3_SUBPIX_F",):
        masked = stage_points[stage].copy(); masked[~support] = np.nan
        if prepared["detected"]:
            actual = pose["infer"](masked, np.asarray(frame["K"], float), np.asarray(frame["known_physical_dimensions_xyz_m"], float), source=False); calls += 1
        else: actual = dict(available=False)
        outcomes[stage] = _finite(actual)
    initial = outcomes["N3_SUBPIX_F"]
    if initial["available"]:
        extent = np.asarray(initial["cf_extents"], float); a, b, c = extent / 2
        native_xyz = [[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]]
        R = np.asarray(initial["R_cf"], float); physical = np.asarray(initial["R_physical"], float)
        Q = R.T @ physical
        if not (np.isfinite(R).all() and np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) and np.linalg.det(R) > 0 and np.linalg.det(Q) > 0 and initial["centroid"][2] > 0):
            raise ValueError("Current F returned invalid physical registration")
        frame["native_xyz"] = _finite(native_xyz); reason = "AVAILABLE_CURRENT_F"
    else:
        Q = None
        reason = "NO_DETECTION" if not prepared["detected"] else "INSUFFICIENT_SUPPORTED_CORNERS" if sum(support[:8]) < 6 else "CURRENT_F_UNAVAILABLE"
    frame["initial_pose"] = initial; frame["initial_points"] = final.tolist()
    if not np.isfinite(np.asarray(frame["initial_points"])).all(): raise ValueError("D1 finite9 missing-sentinel contract violated")
    frame["upstream_provenance"] = dict(source="FROZEN_BACKBONE_N3_SEED1_NATIVE_SUBPIX_CURRENT_F", base_checkpoint_sha256=prepared["base_checkpoint_sha256"],
        n3_checkpoint_sha256=prepared["n3_checkpoint_sha256"], cache_row_sha256=prepared["cache_row_sha256"],
        cached_valid9=prepared["cached_valid9"], prediction_support=support.tolist(), no_reference_inputs=True,
        original_YOLO_N0_not_reused=True, missing_sentinel_is_not_a_measured_point=True, unavailable_geometry_not_consumed=not initial["available"])
    journal = dict(id=frame["id"], partition=frame["partition"], status=reason, pose_available=initial["available"],
        prediction_support=support.tolist(), cached_valid9=prepared["cached_valid9"], detected=prepared["detected"],
        raw_points=q0.tolist(), n3_points=qn.tolist(), native_SubPix_points=native.tolist(), final_points=final.tolist(),
        stage_poses=outcomes, physical_registration_Q=_finite(Q), SubPix=diagnostics,
        F_primitive_calls=calls, internal_F_PnP_calls="NOT_INSTRUMENTED", new_model_forwards=0, new_fits=0, GT_inputs=False)
    return frame, journal


def produce(preparation, backbone, output, root_go, stage_poses=True):
    start = time.monotonic(); cpu = time.process_time(); prep_path = Path(preparation).resolve(); plan = read(prep_path); go = read(root_go)
    if not (go.get("schema") == "D1_THREE_BACKBONE_UPSTREAM_ROOT_GO_V1" and go.get("approved") is True
            and go.get("preparation_sha256") == sha(prep_path) and go.get("source_sha256") == sha(__file__)
            and backbone in go.get("allowed_backbones", []) and go.get("stage_poses") is stage_poses):
        raise PermissionError("Exact ROOT source/preparation/stage approval required")
    if backbone not in plan["backbones"] or not plan["complete"]: raise ValueError("Unprepared backbone")
    for binding in plan["bindings"]: _verify(binding)
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected: raise ValueError("Frozen scientific source changed")
    path = prep_path.parent / f"{backbone}_PREPARED238.jsonl.gz"
    if sha(path) != plan["artifacts"][path.name]: raise ValueError("Prepared inputs changed")
    out = Path(output).resolve()
    if out.exists(): raise FileExistsError("Fresh production output required; partial outputs are never overwritten")
    if not out.is_relative_to(Path(go["output_root"]).resolve()): raise PermissionError("Output outside ROOT-approved run")
    import cv2
    cv2.setNumThreads(1); pose, sub = load_primitives(plan["repo"]); prepared_rows = list(rows(path)); outputs = []; journals = []; counts = Counter()
    out.mkdir(parents=True)
    for prepared in prepared_rows:
        template = prepared["template"]
        if sha(template["image"]) != template["image_sha256"]: raise ValueError("Bound RGB changed")
        image = cv2.imread(template["image"])
        if image is None or list(image.shape[:2]) != template["raw_hw"]: raise ValueError("Actual native RGB canvas mismatch")
        frame, journal = build_frame(prepared, pose, sub, image, stage_poses)
        outputs.append(frame); journals.append(journal); counts[journal["status"]] += 1
        counts["F_primitive_calls"] += journal["F_primitive_calls"]; counts["SubPix_corner_calls"] += journal["SubPix"]["algorithm_corner_calls"]
    save(out / "UPSTREAM238.jsonl.gz", outputs); save(out / "UPSTREAM_JOURNAL.jsonl.gz", journals)
    seal = dict(schema="D1_THREE_BACKBONE_UPSTREAM_SEAL_V1", complete=True, rows=len(outputs), backbone=backbone, N3_seed=1,
        before_reference_scoring=True, preparation_sha256=sha(prep_path), ROOT_GO_sha256=sha(root_go), source_sha256=sha(__file__),
        source_closure=plan["source_sha256"], input_bindings=plan["bindings"],
        artifacts={name: sha(out / name) for name in ("UPSTREAM238.jsonl.gz", "UPSTREAM_JOURNAL.jsonl.gz")},
        all238_including_failures_retained=len(outputs) == 238, partitions=dict(Counter(frame["partition"] for frame in outputs)),
        available_rows=sum(frame["initial_pose"]["available"] for frame in outputs), counts=dict(counts),
        stage_poses=stage_poses, new_model_forwards=0, new_training=0, GT_or_reference_reads=0, old_pose_reuse=0,
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start, CPU_threads=1,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if len(outputs) != 238: raise ValueError("Never drop an upstream failure")
    write(out / "UPSTREAM_SEAL.json", seal)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare"); p.add_argument("--repo", required=True); p.add_argument("--canonical-upstream", required=True); p.add_argument("--asset-root", required=True); p.add_argument("--output", required=True)
    p = commands.add_parser("produce"); p.add_argument("--preparation", required=True); p.add_argument("--backbone", choices=BACKBONES, required=True); p.add_argument("--output", required=True); p.add_argument("--root-go", required=True); p.add_argument("--final-stage-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.repo, args.canonical_upstream, args.asset_root, args.output)
    else: result = produce(args.preparation, args.backbone, args.output, args.root_go, not args.final_stage_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
