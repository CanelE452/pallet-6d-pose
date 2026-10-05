"""Local-only human review of frozen selected-object correspondence.

The page exposes the raw image and one already frozen selected box.  It never
shows keypoints, errors, method names, or performance, and never chooses a
decision on the reviewer's behalf.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from bridge import (GateError, QUEUE_SCHEMA, SIDECAR_SCHEMA, read_json, sha256,
                    validate_sidecar, write_json)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_reviewer(value):
    if not isinstance(value, dict):
        raise GateError("reviewer must be entered")
    if (not isinstance(value.get("id"), str) or not value["id"].strip()
            or value.get("entered_by") != "human" or value.get("confirmation") is not True):
        raise GateError("explicit human reviewer confirmation required")
    # Seeing the fixed box is machine assistance and must be disclosed as such.
    if value.get("machine_assistance") is not True:
        raise GateError("selected-box exposure must be recorded as machine assistance")
    if value.get("previous_prediction_exposure") is not True:
        raise GateError("current selected-box prediction exposure must be recorded")
    if not isinstance(value.get("previous_annotation_exposure"), bool):
        raise GateError("previous annotation exposure answer required")
    if not isinstance(value.get("exposure_notes", ""), str):
        raise GateError("exposure notes must be text")


class Context:
    def __init__(self, queue_path, manifest_path, predictions_path,
                 reviewed_path, store_path):
        self.queue_path = Path(queue_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.predictions_path = Path(predictions_path).resolve()
        self.reviewed_path = Path(reviewed_path).resolve()
        self.store_path = Path(store_path).resolve()
        for path in (self.queue_path, self.manifest_path, self.predictions_path,
                     self.reviewed_path):
            if not path.is_file():
                raise GateError("missing bound object-review input: " + str(path))
        self.queue = read_json(self.queue_path)
        if self.queue.get("schema_version") != QUEUE_SCHEMA:
            raise GateError("bad object-review queue schema")
        observed = {"predictions_sha256": sha256(self.predictions_path),
                    "reviewed_reference_sha256": sha256(self.reviewed_path),
                    "manifest_sha256": sha256(self.manifest_path)}
        for key, value in observed.items():
            if self.queue.get("bindings", {}).get(key) != value:
                raise GateError("object-review queue binding mismatch: " + key)
        identity = self.predictions_path.parent / "RUN_IDENTITY.json"
        if (not identity.is_file() or self.queue["bindings"].get(
                "prediction_run_identity_sha256") != sha256(identity)):
            raise GateError("prediction run identity changed")
        manifest = read_json(self.manifest_path)
        frames = {frame["frame_id"]: frame for frame in manifest.get("frames", [])}
        self.records = {}
        self.images = {}
        for record in self.queue.get("records", []):
            frame_id = record.get("frame_id")
            if frame_id in self.records or frame_id not in frames:
                raise GateError("unknown/duplicate object-review frame")
            frame = frames[frame_id]
            if record.get("image_sha256") != frame.get("image_sha256"):
                raise GateError("object-review image binding changed")
            image = (self.manifest_path.parent / frame["image_path"]).resolve()
            if not image.is_file() or sha256(image) != frame["image_sha256"]:
                raise GateError("object-review source image changed")
            self.records[frame_id] = record
            self.images[frame_id] = image
        self.sessions = {}
        self.lock = threading.Lock()
        self.store = {"schema_version": SIDECAR_SCHEMA,
                      "source_kind": "human_in_progress",
                      "queue_sha256": sha256(self.queue_path), "records": []}
        if self.store_path.is_file():
            candidate = read_json(self.store_path)
            if candidate.get("queue_sha256") != sha256(self.queue_path):
                raise GateError("stored object decisions belong to a different queue")
            # A partial store has human-reviewed records but remains in-progress.
            check = copy.deepcopy(candidate)
            check["source_kind"] = "human_reviewed"
            validate_sidecar(self.queue, check)
            self.store = candidate

    def _validate_record(self, record):
        frozen = self.records.get(record.get("frame_id"))
        if frozen is None:
            raise GateError("unknown object-review frame")
        for key in ("target_object_id", "selected_index", "selected_box_xyxy"):
            if record.get(key) != frozen.get(key):
                raise GateError("frozen object selection changed: " + key)
        if record.get("decision") not in ("same", "different", "undetermined"):
            raise GateError("human must select same/different/undetermined")
        if record.get("source_kind") != "human_reviewed":
            raise GateError("submitted decision must be human-reviewed")
        validate_reviewer(record.get("reviewer"))
        timing = record.get("review_time", {})
        if (timing.get("clock_source") != "server_wall_and_monotonic"
                or not isinstance(timing.get("duration_seconds"), (int, float))
                or timing["duration_seconds"] < 0):
            raise GateError("actual object-review timing required")
        if not isinstance(record.get("edit_reason", ""), str):
            raise GateError("edit reason must be text")
        return record

    def counts(self):
        counts = {name: 0 for name in ("same", "different", "undetermined")}
        for record in self.store["records"]:
            counts[record["decision"]] += 1
        counts["reviewed"] = len(self.store["records"])
        counts["required"] = len(self.records)
        counts["pending"] = len(self.records) - len(self.store["records"])
        return counts

    def catalog(self):
        reviewed = {record["frame_id"]: record["decision"]
                    for record in self.store["records"]}
        safe = [{key: value for key, value in record.items()
                 if key != "image_path"} for record in self.records.values()]
        return {"status": "WAITING_HUMAN", "policy": self.queue["policy"],
                "records": safe, "reviewed": reviewed, "counts": self.counts(),
                "store_path": str(self.store_path)}

    def start(self, payload):
        frame_id = payload.get("frame_id")
        if frame_id not in self.records:
            raise GateError("unknown object-review frame")
        validate_reviewer(payload.get("reviewer"))
        token = uuid.uuid4().hex
        self.sessions[token] = {"frame_id": frame_id,
                                "reviewer": copy.deepcopy(payload["reviewer"]),
                                "started_at": utc_now(), "start_mono": time.monotonic()}
        previous = next((copy.deepcopy(record) for record in self.store["records"]
                         if record["frame_id"] == frame_id), None)
        return {"token": token, "record": previous,
                "started_at": self.sessions[token]["started_at"]}

    def save(self, payload):
        with self.lock:
            session = self.sessions.get(payload.get("token"))
            if session is None:
                raise GateError("start object review before saving")
            supplied = payload.get("record", {})
            if supplied.get("frame_id") != session["frame_id"]:
                raise GateError("review session/frame mismatch")
            frozen = self.records[session["frame_id"]]
            record = {"frame_id": frozen["frame_id"],
                      "target_object_id": frozen["target_object_id"],
                      "selected_index": frozen["selected_index"],
                      "selected_box_xyxy": frozen["selected_box_xyxy"],
                      "decision": supplied.get("decision"),
                      "decision_reason": supplied.get("decision_reason", ""),
                      "edit_reason": supplied.get("edit_reason", ""),
                      "source_kind": "human_reviewed",
                      "reviewer": copy.deepcopy(session["reviewer"]),
                      "review_time": {"started_at": session["started_at"],
                                      "finished_at": utc_now(),
                                      "duration_seconds": round(time.monotonic() - session["start_mono"], 6),
                                      "clock_source": "server_wall_and_monotonic"}}
            self._validate_record(record)
            previous = next((item for item in self.store["records"]
                             if item["frame_id"] == record["frame_id"]), None)
            if previous and not record["edit_reason"].strip():
                raise GateError("editing a submitted decision requires a reason")
            if previous:
                history = self.store_path.with_suffix(self.store_path.suffix + ".history.jsonl")
                history.parent.mkdir(parents=True, exist_ok=True)
                with history.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps({"superseded_at": utc_now(), "record": previous},
                                            ensure_ascii=False, allow_nan=False) + "\n")
            self.store["records"] = [item for item in self.store["records"]
                                     if item["frame_id"] != record["frame_id"]] + [record]
            self.store["source_kind"] = "human_in_progress"
            write_json(self.store_path, self.store)
            return {"saved": True, "record": record, "counts": self.counts()}

    def export(self):
        result = copy.deepcopy(self.store)
        result["source_kind"] = "human_reviewed" if result["records"] else "human_in_progress"
        result["status"] = ("HUMAN_REVIEW_SUBMITTED" if self.counts()["pending"] == 0
                            else "WAITING_HUMAN")
        if result["records"]:
            validate_sidecar(self.queue, result)
        return result

    def import_sidecar(self, incoming):
        if incoming.get("queue_sha256") != sha256(self.queue_path):
            raise GateError("import belongs to a different queue")
        validate_sidecar(self.queue, incoming)
        for record in incoming["records"]:
            old = next((item for item in self.store["records"]
                        if item["frame_id"] == record["frame_id"]), None)
            if old is not None and old != record:
                raise GateError("conflicting object-match decision; edit through UI")
            if old is None:
                self.store["records"].append(copy.deepcopy(record))
        write_json(self.store_path, self.store)
        return self.counts()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, context):
        self.context = context
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, value, status=200, content_type="application/json; charset=utf-8",
             attachment=None):
        data = value if isinstance(value, bytes) else json.dumps(
            value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(data)

    def check_origin(self, post=False):
        port = self.server.server_address[1]
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        if self.headers.get("Host", "") not in hosts:
            raise GateError("localhost Host required")
        if post and self.headers.get("Origin") not in tuple("http://" + host for host in hosts):
            raise GateError("localhost Origin required")

    def do_GET(self):
        try:
            self.check_origin()
            request = urlsplit(self.path)
            if request.path == "/api/catalog":
                self.send(self.server.context.catalog())
            elif request.path == "/api/export":
                self.send(self.server.context.export(), attachment="LIFTER_OBJECT_MATCH_REVIEWED.json")
            elif request.path == "/api/image":
                frame_id = parse_qs(request.query).get("frame_id", [""])[0]
                image = self.server.context.images.get(frame_id)
                if image is None:
                    raise GateError("unknown image")
                self.send(image.read_bytes(), content_type=mimetypes.guess_type(image)[0] or "application/octet-stream")
            elif request.path in ("/", "/index.html", "/app.js", "/style.css"):
                name = "index.html" if request.path == "/" else request.path[1:]
                path = Path(__file__).with_name(name)
                self.send(path.read_bytes(), content_type=mimetypes.guess_type(path)[0] or "text/plain")
            else:
                self.send({"error": "not found"}, 404)
        except (GateError, OSError, ValueError) as error:
            self.send({"error": str(error)}, 400)

    def do_POST(self):
        try:
            self.check_origin(post=True)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                raise GateError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            route = urlsplit(self.path).path
            if route == "/api/start":
                self.send(self.server.context.start(payload))
            elif route == "/api/save":
                self.send(self.server.context.save(payload))
            else:
                self.send({"error": "not found"}, 404)
        except (GateError, OSError, ValueError, KeyError, TypeError) as error:
            self.send({"error": str(error)}, 400)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("serve", "validate", "import", "export"):
        command = sub.add_parser(name)
        command.add_argument("--queue", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--predictions", type=Path, required=True)
        command.add_argument("--reviewed", type=Path, required=True)
        command.add_argument("--store", type=Path, required=True)
        if name in ("validate", "import"):
            command.add_argument("--input", type=Path, required=True)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        if name == "serve":
            command.add_argument("--port", type=int, default=8766)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        context = Context(args.queue, args.manifest, args.predictions,
                          args.reviewed, args.store)
        if args.command == "serve":
            server = Server(("127.0.0.1", args.port), context)
            print(json.dumps({"status": "WAITING_HUMAN",
                              "url": f"http://127.0.0.1:{server.server_address[1]}",
                              "counts": context.counts()}, ensure_ascii=False), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        elif args.command in ("validate", "import"):
            incoming = read_json(args.input)
            if incoming.get("queue_sha256") != sha256(args.queue):
                raise GateError("sidecar queue hash mismatch")
            validate_sidecar(context.queue, incoming)
            counts = context.import_sidecar(incoming) if args.command == "import" else context.counts()
            print(json.dumps({"valid": True, "records": len(incoming["records"]),
                              "counts": counts}, ensure_ascii=False))
        else:
            exported = context.export()
            if not exported["records"]:
                raise GateError("WAITING_HUMAN: no object-match decisions")
            if args.output.exists():
                raise GateError("refusing to overwrite object-match export")
            write_json(args.output, exported)
            print(json.dumps({"output": str(args.output),
                              "counts": context.counts()}, ensure_ascii=False))
        return 0
    except (GateError, OSError, ValueError, KeyError, TypeError,
            json.JSONDecodeError) as error:
        print(json.dumps({"status": "WAITING_HUMAN" if "WAITING_HUMAN" in str(error)
                          else "BLOCKED_CONTRACT", "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
