"""리뷰 패킷을 서빙하고, 브라우저 localStorage 를 POST 로 받아 파일로 떨군다.

목적 : 사용자가 「내려받기」를 누르는 대신 자동으로 회수한다. 지난번 두 번째 CSV 가
       브라우저의 연속 다운로드 차단으로 유실됐고, 그 답이 localStorage 에만 남아 있다.
지표 : REVIEW_LOCALSTORAGE.json 이 디스크에 생기고 프레임 수가 맞는가.

127.0.0.1 에만 바인딩한다. POST 는 /save 하나뿐이고 지정한 파일에만 쓴다.
"""
import http.server, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
PACK = ROOT / "_docs/audits/accuracy_root_cause_v1/human_review"
OUT = ROOT / "data/pallet/results/accuracy_root_cause_v1/REVIEW_LOCALSTORAGE.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(PACK), **kw)

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self.send_error(400, str(exc))
            return
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        msg = json.dumps({"ok": True, "keys": list(data.keys()),
                          "bytes": len(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)
        print(f"[save] {len(body)} bytes -> {OUT}", flush=True)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
