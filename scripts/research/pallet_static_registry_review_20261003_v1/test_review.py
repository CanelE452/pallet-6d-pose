from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .audit import DOC
from .serve_review import Context, ReviewError, handler, progress, validate_state


SOURCE = Path("/home/minjae/Documents/github/pallet-pose")
MANIFEST = DOC / "review/STATIC_REVIEW_MANIFEST.json"


def context(tmp_path: Path) -> Context:
    return Context(MANIFEST, tmp_path / "state.json", SOURCE)


def _http_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    request = Request(url, data=(json.dumps(payload).encode() if payload is not None else None),
                      headers={"Content-Type": "application/json"},
                      method="POST" if payload is not None else "GET")
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_draft_roundtrip_and_locked_provenance(self) -> None:
        ctx = context(self.tmp_path)
        state = ctx.empty()
        self.assertEqual(progress(state, ctx.manifest)["square_frame_severity"], {
            "done": 0, "total": 119, "status": "PARTIAL"})
        self.assertEqual(progress(state, ctx.manifest)["metric_corner_visibility"]["total"], 3030)

        square = next(case for case in ctx.manifest["cases"] if case["population"] == "GREEN0918")
        state["responses"] = {square["case_id"]: {
            "frame_severity": "clean", "corners": {"0": "DIRECT_VISIBLE"},
            "overlay_exposed": True}}
        saved = ctx.save(state)
        self.assertEqual(saved["review_status"], "DRAFT_HUMAN_ENTERED")
        self.assertEqual(ctx.load(), saved)
        self.assertTrue(ctx.asset(0, "image").is_file())
        self.assertTrue(ctx.asset(0, "overlay").is_file())

        rectangular = next(case for case in ctx.manifest["cases"] if case["population"] == "DEV319")
        bad = ctx.empty()
        bad["responses"] = {rectangular["case_id"]: {"frame_severity": "severe", "corners": {}}}
        with self.assertRaisesRegex(ReviewError, "locked completed severity"):
            validate_state(bad, ctx.manifest, ctx.manifest_hash)

        locked = next((case, corner) for case in ctx.manifest["cases"]
                      for corner in case["corners"] if corner["locked"])
        bad = ctx.empty()
        bad["responses"] = {locked[0]["case_id"]: {
            "corners": {str(locked[1]["corner_id"]): "UNKNOWN"}}}
        with self.assertRaisesRegex(ReviewError, "approved legacy status is locked"):
            validate_state(bad, ctx.manifest, ctx.manifest_hash)

    def test_submit_requires_human_and_all_square_severity(self) -> None:
        ctx = context(self.tmp_path)
        state = ctx.empty()
        with self.assertRaisesRegex(ReviewError, "reviewer identity"):
            ctx.submit(state)
        state["reviewer"] = "fixture-reviewer"
        with self.assertRaisesRegex(ReviewError, "all 119 square"):
            ctx.submit(state)

        state["responses"] = {
            case["case_id"]: {"frame_severity": "clean", "corners": {}}
            for case in ctx.manifest["cases"] if case["population"] == "GREEN0918"
        }
        submitted, output = ctx.submit(state)
        self.assertEqual(submitted["source_kind"], "human_reviewed")
        self.assertEqual(submitted["frame_severity_status"], "HUMAN_REVIEW_COMPLETE")
        self.assertEqual(submitted["corner_visibility_status"], "PARTIAL_HUMAN_REVIEW")
        self.assertTrue(output.is_file())
        self.assertEqual(json.loads(output.read_text()), submitted)

    def test_localhost_save_import_export_roundtrip(self) -> None:
        ctx = context(self.tmp_path)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(ctx))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            status, manifest = _http_json(base + "/manifest")
            self.assertEqual(status, 200)
            self.assertEqual(len(manifest["cases"]), 438)
            status, draft = _http_json(base + "/state")
            self.assertEqual(status, 200)
            self.assertEqual(draft["responses"], {})
            first = manifest["cases"][0]
            draft["responses"] = {first["case_id"]: {
                "frame_severity": "unknown", "frame_unknown_reason": "fixture ambiguity",
                "corners": {}, "overlay_exposed": False}}
            status, saved = _http_json(base + "/save", draft)
            self.assertEqual((status, saved["status"]), (200, "SAVED_DRAFT"))
            status, exported = _http_json(base + "/export")
            self.assertEqual(status, 200)
            self.assertEqual(exported["responses"], draft["responses"])
            exported["reviewer"] = "fixture-importer"
            status, imported = _http_json(base + "/import", exported)
            self.assertEqual((status, imported["status"]), (200, "SAVED_DRAFT"))
            self.assertEqual(ctx.load()["reviewer"], "fixture-importer")
            invalid = ctx.empty()
            invalid["responses"] = {first["case_id"]: {
                "frame_severity": "unknown", "corners": {}}}
            status, error = _http_json(base + "/import", invalid)
            self.assertEqual(status, 400)
            self.assertIn("needs a reason", error["reason"])
        finally:
            server.shutdown()
            thread.join(timeout=10)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
