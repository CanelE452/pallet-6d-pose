"""Make deterministic lossless publication copies without changing source files."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
DOC = Path("_docs/experiments/pallet_paper_review_20261006_v1/portable_evidence")
RESULT = Path("_docs/experiments/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/RESULT.json")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    (ROOT / DOC / "compressed").mkdir(parents=True, exist_ok=True)
    (ROOT / DOC / "reference_images").mkdir(parents=True, exist_ok=True)
    result = json.loads((ROOT / RESULT).read_text())
    reference = json.loads(Path(result["sources"]["assisted_reference"]["path"]).read_text())
    entries = []
    for index, item in enumerate(selection["compress_originals_for_evidence"], 1):
        original_rel = Path(item["original_path"])
        original = ROOT / original_rel
        before, size = sha(original), original.stat().st_size
        if size != item["bytes"]:
            raise ValueError(f"Publication inventory size changed: {original_rel}")
        artifact_rel = DOC / "compressed" / f"{index:02d}_{original.name}.gz"
        artifact = ROOT / artifact_rel
        with original.open("rb") as source, artifact.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output,
                               compresslevel=9, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
        digest, decompressed_size = hashlib.sha256(), 0
        with gzip.open(artifact, "rb") as copied:
            for block in iter(lambda: copied.read(1024 * 1024), b""):
                digest.update(block)
                decompressed_size += len(block)
        if sha(original) != before or digest.hexdigest() != before or decompressed_size != size:
            raise ValueError(f"Source preservation or lossless copy failed: {original_rel}")
        entries.append(dict(original_path=original_rel.as_posix(), original_bytes=size,
                            original_sha256=before, artifact_path=artifact_rel.as_posix(),
                            artifact_bytes=artifact.stat().st_size, artifact_sha256=sha(artifact),
                            decompressed_sha256=digest.hexdigest(), decompressed_bytes=size,
                            encoding="gzip", gzip_mtime=0, gzip_original_filename="",
                            source_preserved=True))
    images = []
    manifest_parent = Path(result["sources"]["manifest"]["path"]).parent
    for record in reference["records"]:
        original = manifest_parent / record["image_path"]
        artifact_rel = DOC / "reference_images" / original.name
        artifact = ROOT / artifact_rel
        before = sha(original)
        if before != record["image_sha256"]:
            raise ValueError(f"Reference image changed: {record['frame_id']}")
        shutil.copyfile(original, artifact)
        if sha(original) != before or sha(artifact) != before:
            raise ValueError("Reference image copy is not byte identical")
        images.append(dict(frame_id=record["frame_id"],
                           original_path=original.relative_to(ROOT).as_posix(),
                           original_bytes=original.stat().st_size, original_sha256=before,
                           artifact_path=artifact_rel.as_posix(), artifact_bytes=artifact.stat().st_size,
                           artifact_sha256=sha(artifact), encoding="identity", source_preserved=True))
    manifest = dict(schema_version="pallet_portable_publication_evidence_v1",
                    source_repository_prefix=str(ROOT),
                    source_local_head=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_raw_prediction_commit=json.loads(Path(
                        result["sources"]["run_identity"]["path"]).read_text())["source_commit"],
                    result_path=RESULT.as_posix(), result_sha256=sha(ROOT / RESULT),
                    compressed_sources=entries, reference_images=images,
                    original_paths_not_rewritten=True, all_source_files_preserved=True,
                    training_runs=0, new_model_forward_frames=0, optimizer_updates=0,
                    hardware_control_calls=0,
                    total_original_compressed_source_bytes=sum(e["original_bytes"] for e in entries),
                    total_compressed_artifact_bytes=sum(e["artifact_bytes"] for e in entries),
                    reference_image_bytes=sum(e["artifact_bytes"] for e in images),
                    scope="Completed evidence packaging; no new evaluation population")
    (ROOT / DOC / "PORTABLE_EVIDENCE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="LOSSLESS_PORTABLE_EVIDENCE_CREATED",
                          gzip_artifact_count=len(entries), reference_image_count=len(images),
                          original_bytes=manifest["total_original_compressed_source_bytes"],
                          compressed_bytes=manifest["total_compressed_artifact_bytes"],
                          source_files_preserved=True, training_runs=0,
                          new_model_forward_frames=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
