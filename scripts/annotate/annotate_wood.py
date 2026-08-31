"""Registry-backed wood-pallet annotation launcher.

The launcher prepares the historical JPEG sequence for the shared annotation
UI, but object geometry is selected explicitly from
``OBJECT_GEOMETRY_REGISTRY.json``.  It never mutates ``annotate_pnp.PALLET_DIMS``
and never writes to the legacy manual-GT directory.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def find_repo_root(start: str | Path) -> Path:
    """Find a checkout root from a .git directory or worktree marker file."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"cannot find repository .git marker above {start}")


_HERE = Path(__file__).resolve().parent
_REPO_PATH = find_repo_root(_HERE)
_REPO = str(_REPO_PATH)  # historical import compatibility
sys.path.insert(0, str(_HERE))

from object_geometry_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)


_DEFAULT_REGISTRY = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
_WOOD_SPEC = _DEFAULT_REGISTRY.resolve(WOOD_OBJECT_TYPE)
WOOD_DIMS = _WOOD_SPEC.legacy_wdh_tuple  # explicit compatibility order W,D,H
WOOD_ROOT = str(_REPO_PATH / "data" / "pallet" / "raw_data" / "wood" / "selected")

# RealSense D435I profiles queried in the historical wood annotation process.
# Values are (fx, fy, cx, cy); the existing 1280x720 labels use the 1080p
# profile scaled by 2/3 and are therefore SENSOR_PROFILE_SCALED, not calibrated.
RS_INTRINSICS = {
    (640, 480): (605.9065, 605.9698, 317.5962, 256.2923),
    (1920, 1080): (1363.2896, 1363.4321, 954.5915, 576.6577),
}
_SCALED_PROFILE_SOURCE = (
    "RealSense D435I 1920x1080 sensor profile scaled linearly to the frame "
    "resolution; camera serial and distortion model unavailable"
)


def K_for_resolution(w, h):
    """Return ``(K, source)`` while preserving the historical public API."""
    if (w, h) in RS_INTRINSICS:
        fx, fy, cx, cy = RS_INTRINSICS[(w, h)]
        source = f"RealSense D435I sensor profile {w}x{h}"
    elif h > 0 and abs(w / h - 16 / 9) < 0.02:
        fx0, fy0, cx0, cy0 = RS_INTRINSICS[(1920, 1080)]
        sx, sy = w / 1920.0, h / 1080.0
        fx, fy, cx, cy = fx0 * sx, fy0 * sy, cx0 * sx, cy0 * sy
        source = _SCALED_PROFILE_SOURCE
    else:
        return None, (
            f"{w}x{h} is neither a registered RealSense profile nor 16:9 "
            "(--K/--hfov required)")
    matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matrix, source


def estimate_K(w, h, hfov_deg):
    """Estimate intrinsics from resolution and horizontal field of view."""
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array(
        [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _validated_video_name(video: str) -> str:
    if (not video or video in {".", ".."}
            or Path(video).name != video or "/" in video or "\\" in video):
        raise ValueError("--video must be one folder name under wood/selected")
    return video


def default_output_dir(video: str, repo: str | Path = _REPO_PATH) -> Path:
    video = _validated_video_name(video)
    return (Path(repo) / "challenge" / "data" / "01_real" /
            "gt_v2_canonical" / "manual_gt" /
            f"wood_{video}_manual_gt")


def legacy_read_dir(video: str, repo: str | Path = _REPO_PATH) -> Path:
    video = _validated_video_name(video)
    return (Path(repo) / "challenge" / "data" / "01_real" / "manual_gt" /
            f"wood_{video}_manual_gt")


def wood_session_id(video: str) -> str:
    """Return the stable namespaced session used by the wood audit."""
    prefix = "pallet_20260618_"
    return "wood_" + (video[len(prefix):] if video.startswith(prefix) else video)


def prep_seq(video, K, *, repo: str | Path = _REPO_PATH,
             wood_root: str | Path | None = None):
    """Create a staging ``rgb`` view without changing source JPEGs or GT."""
    video = _validated_video_name(video)
    repo = Path(repo).resolve()
    source_root = (Path(wood_root).resolve() if wood_root is not None
                   else repo / "data" / "pallet" / "raw_data" / "wood" / "selected")
    source = source_root / video
    jpgs = sorted(source.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"no jpg in {source}")
    sequence = repo / "data" / "pallet" / "raw_data" / "wood" / f"_annotate_{video}"
    rgb = sequence / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)
    for source_image in jpgs:
        link = rgb / f"{source_image.stem}.png"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(source_image.resolve())
    np.savetxt(sequence / "cam_K.txt", np.asarray(K, dtype=np.float64).reshape(3, 3))
    return str(sequence), len(jpgs)


def build_annotate_argv(*, video, sequence, stride, start, out_dir,
                        population_role, registry_path, intrinsics_quality,
                        intrinsics_source, repo: str | Path = _REPO_PATH):
    """Build the explicit, testable dispatch contract for ``annotate.main``."""
    argv = [
        "--seq", str(sequence),
        "--stride", str(stride),
        "--start", str(start),
        "--out_dir", str(out_dir),
        "--population-role", str(population_role),
        "--object-type", WOOD_OBJECT_TYPE,
        "--geometry-registry", str(registry_path),
        "--intrinsics-quality", str(intrinsics_quality),
        "--intrinsics-source", str(intrinsics_source),
        "--capture-session-id", wood_session_id(video),
    ]
    legacy = legacy_read_dir(video, repo)
    if legacy.is_dir():
        argv.extend(["--legacy-read-dir", str(legacy)])
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", default="pallet_20260618_183705",
        help="one folder name under data/pallet/raw_data/wood/selected",
    )
    parser.add_argument("--stride", type=int, default=5, help="annotate every Nth frame")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--K", default=None, help="explicit 3x3 intrinsics file")
    parser.add_argument("--hfov", type=float, default=None,
                        help="estimate K from horizontal field of view in degrees")
    parser.add_argument(
        "--out_dir", default=None,
        help="default: challenge/data/01_real/gt_v2_canonical/manual_gt/wood_*",
    )
    parser.add_argument("--population-role", "--population_role", required=True,
                        choices=["DEV", "FINAL"])
    parser.add_argument(
        "--intrinsics-quality", default=None,
        choices=["CALIBRATED", "SENSOR_PROFILE_SCALED", "ESTIMATED_HFOV", "UNKNOWN"],
        help="classification override for an explicit --K when known",
    )
    parser.add_argument("--intrinsics-source", default=None)
    parser.add_argument("--geometry-registry", default=str(DEFAULT_REGISTRY_PATH))
    args = parser.parse_args(argv)

    try:
        video = _validated_video_name(args.video)
    except ValueError as exc:
        parser.error(str(exc))
    registry_path = (Path(args.geometry_registry).expanduser()
                     if Path(args.geometry_registry).is_absolute()
                     else _REPO_PATH / args.geometry_registry)
    try:
        registry = load_object_geometry_registry(registry_path)
        wood_spec = registry.resolve(WOOD_OBJECT_TYPE)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"invalid wood geometry registry: {exc}")

    source = Path(WOOD_ROOT) / video
    jpgs = sorted(source.glob("*.jpg"))
    if not jpgs:
        available = (sorted(path.name for path in Path(WOOD_ROOT).iterdir())
                     if Path(WOOD_ROOT).is_dir() else [])
        parser.error(f"no jpg in {source}; available={available}")
    with Image.open(jpgs[0]) as image:
        w, h = image.size

    if args.K:
        try:
            K = np.loadtxt(args.K).reshape(3, 3)
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read --K as a 3x3 matrix: {exc}")
        quality = args.intrinsics_quality or "UNKNOWN"
        source_description = args.intrinsics_source or f"file:{Path(args.K).resolve()}"
    elif args.hfov is not None:
        if not 0.0 < args.hfov < 180.0:
            parser.error("--hfov must be in (0, 180) degrees")
        if args.intrinsics_quality not in (None, "ESTIMATED_HFOV"):
            parser.error("--hfov requires intrinsics quality ESTIMATED_HFOV")
        K = estimate_K(w, h, args.hfov)
        quality = "ESTIMATED_HFOV"
        source_description = (args.intrinsics_source
                              or f"estimated from horizontal FOV {args.hfov:g} deg")
    else:
        K, automatic_source = K_for_resolution(w, h)
        if K is None:
            parser.error(f"automatic K selection failed: {automatic_source}")
        inferred_quality = ("UNKNOWN" if (w, h) in RS_INTRINSICS
                            else "SENSOR_PROFILE_SCALED")
        quality = args.intrinsics_quality or inferred_quality
        source_description = args.intrinsics_source or automatic_source

    output = (Path(args.out_dir).expanduser() if args.out_dir
              else default_output_dir(video))
    if not output.is_absolute():
        output = _REPO_PATH / output

    try:
        sequence, count = prep_seq(video, K)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print("=" * 64)
    print(f"[annotate_wood] video={video} frames={count} res={w}x{h}")
    print(f"  object_type = {wood_spec.object_type}")
    print(f"  canonical XYZ = {wood_spec.physical_dimensions_m} m")
    print(f"  compatibility W,D,H = {wood_spec.legacy_wdh_tuple} m")
    print(f"  K quality/source = {quality} / {source_description}")
    print(f"  legacy source = {legacy_read_dir(video)} (read-only, if present)")
    print(f"  canonical output = {output}")
    print("=" * 64)

    import annotate

    annotate.main(build_annotate_argv(
        video=video,
        sequence=sequence,
        stride=args.stride,
        start=args.start,
        out_dir=output,
        population_role=args.population_role,
        registry_path=registry.source_path,
        intrinsics_quality=quality,
        intrinsics_source=source_description,
    ))


if __name__ == "__main__":
    main()
