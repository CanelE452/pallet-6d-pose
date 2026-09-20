"""Isolated, CPU-only Chrome QA for the already generated offline review page."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

import websocket

from . import protocol as P


class CDP:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=10, suppress_origin=True)
        self.sequence = 0
        self.events = []

    def call(self, method, params=None):
        self.sequence += 1
        ident = self.sequence
        self.socket.send(json.dumps({"id": ident, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == ident:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})
            if "method" in message:
                self.events.append(message)

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", dict(expression=expression, awaitPromise=True, returnByValue=True))
        assert "exceptionDetails" not in result, result
        return result["result"].get("value")


def main():
    page = P.OUT / "RESULTS_REVIEW.html"
    assert page.exists()
    profile = Path(tempfile.mkdtemp(prefix="pallet-corner-viewer-qa-", dir="/tmp"))
    log_path = profile / "chrome.log"
    argv = ["/usr/bin/google-chrome", "--headless=new", "--disable-gpu",
            "--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check",
            "--disable-background-networking", "--disable-component-update",
            "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
            "--user-data-dir=" + str(profile), "about:blank"]
    controller = None
    result = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=log)
        try:
            active = profile / "DevToolsActivePort"
            deadline = time.monotonic() + 20
            while not active.exists():
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text(errors="replace"))
                assert time.monotonic() < deadline, "Chrome debugging endpoint did not start"
                time.sleep(.1)
            port = int(active.read_text().splitlines()[0])
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as response:
                targets = json.load(response)
            target = next(t for t in targets if t["type"] == "page")
            controller = CDP(target["webSocketDebuggerUrl"])
            for method in ("Page.enable", "Runtime.enable", "Log.enable", "Network.enable"):
                controller.call(method)
            controller.call("Network.setBlockedURLs", {"urls": ["http://*", "https://*"]})
            views = {}
            for label, width, height, mobile in (("desktop", 1440, 1000, False), ("mobile", 390, 844, True)):
                controller.call("Emulation.setDeviceMetricsOverride", dict(width=width, height=height, deviceScaleFactor=1, mobile=mobile))
                controller.call("Page.navigate", {"url": page.resolve().as_uri()})
                deadline = time.monotonic() + 20
                while controller.evaluate("document.readyState") != "complete":
                    assert time.monotonic() < deadline, "Page did not load"
                    time.sleep(.1)
                # Decode every embedded lazy image, including off-screen cases.
                loaded = controller.evaluate("""Promise.all(Array.from(document.images).map(async image=>{
                    image.loading='eager'; await image.decode();
                    return {width:image.naturalWidth,height:image.naturalHeight,embedded:image.src.startsWith('data:image/png;base64,')};
                }))""")
                assert len(loaded) == 6 and all(i["width"] > 0 and i["height"] > 0 and i["embedded"] for i in loaded)
                anchors = controller.evaluate("""Array.from(document.querySelectorAll('nav a')).map(a=>({href:a.getAttribute('href'), exists:!!document.querySelector(a.getAttribute('href'))}))""")
                assert len(anchors) == 6 and all(a["exists"] for a in anchors)
                for anchor in anchors:
                    selector = json.dumps(anchor["href"])
                    check = controller.evaluate(f"""(()=>{{const target=document.querySelector({selector});target.scrollIntoView({{behavior:'instant'}});return {{id:target.id,visible:target.getBoundingClientRect().bottom>0 && target.getBoundingClientRect().top<innerHeight}};}})()""")
                    assert check["visible"], check
                controller.evaluate("scrollTo({top:0,behavior:'instant'})")
                layout = controller.evaluate("""({title:document.title,innerWidth:innerWidth,bodyWidth:document.body.scrollWidth,
                    documentWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth,
                    tables:document.querySelectorAll('table').length,externalResources:performance.getEntriesByType('resource').filter(r=>/^https?:/.test(r.name)).map(r=>r.name)})""")
                assert layout["bodyWidth"] <= layout["innerWidth"] + 1, layout
                assert layout["documentWidth"] <= layout["innerWidth"] + 1, layout
                assert layout["tables"] == 10 and not layout["externalResources"], layout
                shot = controller.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                import base64
                screenshot = P.OUT / f"VIEWER_QA_{label}.png"
                screenshot.write_bytes(base64.b64decode(shot["data"]))
                views[label] = dict(viewport=[width, height], layout=layout, images=loaded, anchors=anchors, screenshot=P.bound(screenshot))
            errors = [event for event in controller.events if event["method"] == "Runtime.exceptionThrown"
                      or (event["method"] == "Log.entryAdded" and event["params"]["entry"].get("level") == "error")]
            assert not errors, errors
            result = dict(passed=True, browser="Chrome headless isolated temporary profile", browser_argv=argv,
                CPU_only_requested=True, gpu_disabled=True, existing_browser_modified=False,
                external_page_network_blocked=True, page=P.bound(page), source=P.bound(Path(__file__)),
                views=views, javascript_errors=errors, process_pid=process.pid, profile=str(profile))
        finally:
            if controller is not None:
                controller.socket.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    result["own_headless_process_terminated"] = process.poll() is not None
    result["process_exit_code"] = process.returncode
    output = P.OUT / "VIEWER_QA.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=result["passed"], report=str(output), views=list(views),
                         own_process_terminated=result["own_headless_process_terminated"]), ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
