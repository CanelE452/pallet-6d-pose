"""Serve and validate the prediction-blind static review on localhost."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .audit import DOC, RAW, sha256


SCHEMA = "pallet_static_registry_review_response_v1"
APP = Path(__file__).parent / "static_review_app"
SEVERITIES = {"clean", "moderate", "severe", "unknown"}
VISIBILITY = {"DIRECT_VISIBLE", "EXTERNAL_OCCLUDED", "SELF_OCCLUDED",
              "OUT_OF_FRAME", "OBJECT_ABSENT", "UNKNOWN"}


class ReviewError(ValueError):
    """The review payload does not match the frozen manifest."""


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _case_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ReviewError("manifest cases missing")
    indexed = {case["case_id"]: case for case in cases}
    if len(indexed) != len(cases):
        raise ReviewError("duplicate manifest case IDs")
    return indexed


def progress(state: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict:
    responses = state.get("responses", {})
    if not isinstance(responses, Mapping):
        raise ReviewError("responses must be an object")
    square_total = square_done = 0
    visibility_total = visibility_done = 0
    optional_done = 0
    for case in manifest["cases"]:
        response = responses.get(case["case_id"], {})
        if not isinstance(response, Mapping):
            raise ReviewError("each response must be an object")
        if case["population"] == "GREEN0918":
            square_total += 1
            if response.get("frame_severity") in SEVERITIES:
                square_done += 1
        point_responses = response.get("corners", {})
        if not isinstance(point_responses, Mapping):
            raise ReviewError("corner responses must be an object")
        for point in case["corners"]:
            if point["locked"]:
                continue
            value = point_responses.get(str(point["corner_id"]))
            if point["metric_reference"]:
                visibility_total += 1
                if value in VISIBILITY:
                    visibility_done += 1
            elif value in VISIBILITY:
                optional_done += 1
    return {
        "square_frame_severity": {"done": square_done, "total": square_total,
                                  "status": "COMPLETE" if square_done == square_total else "PARTIAL"},
        "metric_corner_visibility": {"done": visibility_done, "total": visibility_total,
                                     "status": "COMPLETE" if visibility_done == visibility_total else "PARTIAL"},
        "optional_nonmetric_corner_statuses": optional_done,
    }


def validate_state(state: Mapping[str, Any], manifest: Mapping[str, Any],
                   manifest_hash: str, *, submit: bool = False) -> dict:
    if state.get("schema") != SCHEMA:
        raise ReviewError("response schema mismatch")
    if state.get("manifest_sha256") != manifest_hash:
        raise ReviewError("response is bound to another manifest")
    cases = _case_index(manifest)
    responses = state.get("responses")
    if not isinstance(responses, Mapping) or not set(responses).issubset(cases):
        raise ReviewError("unknown response case ID")
    for case_id, response in responses.items():
        if not isinstance(response, Mapping):
            raise ReviewError(f"{case_id}: response must be an object")
        case = cases[case_id]
        severity = response.get("frame_severity")
        if severity is not None:
            if case["frame_severity"]["locked"]:
                raise ReviewError(f"{case_id}: locked completed severity cannot be overwritten")
            if severity not in SEVERITIES:
                raise ReviewError(f"{case_id}: invalid severity")
            if severity == "unknown" and not str(response.get("frame_unknown_reason", "")).strip():
                raise ReviewError(f"{case_id}: unknown severity needs a reason")
        corners = response.get("corners", {})
        if not isinstance(corners, Mapping):
            raise ReviewError(f"{case_id}: corners must be an object")
        by_id = {str(point["corner_id"]): point for point in case["corners"]}
        if not set(corners).issubset(by_id):
            raise ReviewError(f"{case_id}: unknown corner ID")
        for corner_id, value in corners.items():
            if by_id[corner_id]["locked"]:
                raise ReviewError(f"{case_id}/{corner_id}: approved legacy status is locked")
            if value not in VISIBILITY:
                raise ReviewError(f"{case_id}/{corner_id}: invalid visibility state")
        if "overlay_exposed" in response and type(response["overlay_exposed"]) is not bool:
            raise ReviewError(f"{case_id}: overlay_exposed must be boolean")
    reviewer = state.get("reviewer", "")
    if reviewer is not None and not isinstance(reviewer, str):
        raise ReviewError("reviewer must be text")
    counts = progress(state, manifest)
    if submit:
        if not str(reviewer).strip():
            raise ReviewError("explicit reviewer identity is required for submission")
        if counts["square_frame_severity"]["status"] != "COMPLETE":
            raise ReviewError("all 119 square frame severities are required before submission")
    return counts


class Context:
    def __init__(self, manifest_path: Path, store_path: Path, source_root: Path):
        self.manifest_path = manifest_path.resolve()
        self.store_path = store_path.resolve()
        self.source_root = source_root.resolve()
        self.manifest = read(self.manifest_path)
        self.manifest_hash = sha256(self.manifest_path)
        if Path(self.manifest.get("source_root", "")).resolve() != self.source_root:
            raise ReviewError("source root differs from manifest")
        _case_index(self.manifest)

    def empty(self) -> dict:
        return {
            "schema": SCHEMA,
            "manifest_sha256": self.manifest_hash,
            "reviewer": "",
            "review_status": "DRAFT_NOT_HUMAN_REVIEWED",
            "created_at": None,
            "updated_at": None,
            "responses": {},
        }

    def load(self) -> dict:
        if not self.store_path.is_file():
            return self.empty()
        state = read(self.store_path)
        validate_state(state, self.manifest, self.manifest_hash)
        return state

    def save(self, payload: Mapping[str, Any]) -> dict:
        state = dict(payload)
        counts = validate_state(state, self.manifest, self.manifest_hash)
        now = datetime.now(timezone.utc).isoformat()
        previous = self.load()
        state["created_at"] = previous.get("created_at") or now
        state["updated_at"] = now
        state["review_status"] = "DRAFT_HUMAN_ENTERED"
        state["progress"] = counts
        atomic_write(self.store_path, state)
        return state

    def submit(self, payload: Mapping[str, Any]) -> tuple[dict, Path]:
        state = dict(payload)
        counts = validate_state(state, self.manifest, self.manifest_hash, submit=True)
        now = datetime.now(timezone.utc).isoformat()
        state["created_at"] = self.load().get("created_at") or now
        state["updated_at"] = now
        state["submitted_at"] = now
        state["source_kind"] = "human_reviewed"
        state["frame_severity_status"] = "HUMAN_REVIEW_COMPLETE"
        state["corner_visibility_status"] = (
            "HUMAN_REVIEW_COMPLETE" if counts["metric_corner_visibility"]["status"] == "COMPLETE"
            else "PARTIAL_HUMAN_REVIEW")
        state["review_status"] = "SUBMITTED_HUMAN_REVIEW"
        state["progress"] = counts
        signature = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")).encode()).hexdigest()[:12]
        output = self.store_path.parent / f"STATIC_REVIEW_SUBMITTED_{signature}.json"
        if not output.exists():
            atomic_write(output, state)
        atomic_write(self.store_path, state)
        return state, output

    def asset(self, index: int, kind: str) -> Path:
        if kind not in {"image", "overlay"}:
            raise ReviewError("unknown asset kind")
        cases = self.manifest["cases"]
        if index < 0 or index >= len(cases):
            raise ReviewError("case index out of range")
        binding = cases[index][kind]
        path = self.source_root / binding["path"]
        if not path.is_file() or sha256(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
            raise ReviewError("asset binding changed")
        return path


def handler(context: Context):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = 200,
                  disposition: str | None = None) -> None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(data)

        def _file(self, path: Path, content_type: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            try:
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    return self._file(APP / "index.html", "text/html; charset=utf-8")
                if path == "/app.js":
                    return self._file(APP / "app.js", "text/javascript; charset=utf-8")
                if path == "/style.css":
                    return self._file(APP / "style.css", "text/css; charset=utf-8")
                if path == "/manifest":
                    return self._json(context.manifest)
                if path == "/state":
                    return self._json(context.load())
                if path == "/export":
                    return self._json(context.load(), disposition=(
                        'attachment; filename="STATIC_REVIEW_IN_PROGRESS.json"'))
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "asset":
                    asset = context.asset(int(parts[1]), parts[2])
                    content = "image/png" if asset.suffix.lower() == ".png" else "image/jpeg"
                    return self._file(asset, content)
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ReviewError, ValueError) as error:
                self._json({"status": "ERROR", "reason": str(error)}, 400)

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20_000_000:
                    raise ReviewError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path in {"/save", "/import"}:
                    state = context.save(payload)
                    return self._json({"status": "SAVED_DRAFT", "progress": state["progress"]})
                if path == "/submit":
                    state, output = context.submit(payload)
                    return self._json({"status": state["review_status"],
                                       "output": str(output), "progress": state["progress"]})
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ReviewError, ValueError, json.JSONDecodeError) as error:
                self._json({"status": "ERROR", "reason": str(error)}, 400)

        def log_message(self, template, *args):
            print("static-review", self.address_string(), template % args)

    return Handler


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DOC / "review/STATIC_REVIEW_MANIFEST.json")
    parser.add_argument("--store", type=Path,
                        default=RAW / "STATIC_REVIEW_IN_PROGRESS.json")
    parser.add_argument("--source-root", type=Path,
                        default=Path("/home/minjae/Documents/github/pallet-pose"))
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args(argv)
    context = Context(args.manifest, args.store, args.source_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(context))
    print(f"Static review: http://127.0.0.1:{args.port}")
    print(f"Store: {context.store_path}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
