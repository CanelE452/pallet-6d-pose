"""Isolated CPU Chrome QA; no model inference or prior artifact edits."""

import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

from . import core as P
from scripts.research.pallet_posefix_corner_gate_v1.viewer_qa import CDP


def main():
    page = P.OUT / "LEARNED_SELECTOR_REVIEW.html"
    assert page.exists()
    profile = Path(tempfile.mkdtemp(prefix="pallet-selector-viewer-qa-", dir="/tmp"))
    argv = ["/usr/bin/google-chrome", "--headless=new", "--disable-gpu", "--disable-dev-shm-usage",
            "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
            "--disable-component-update", "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
            "--user-data-dir=" + str(profile), "about:blank"]
    controller = None
    with (profile / "chrome.log").open("wb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=log)
        try:
            active = profile / "DevToolsActivePort"
            deadline = time.monotonic() + 20
            while not active.exists():
                assert process.poll() is None, (profile / "chrome.log").read_text(errors="replace")
                assert time.monotonic() < deadline, "Chrome endpoint timeout"
                time.sleep(.1)
            port = int(active.read_text().splitlines()[0])
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as response:
                targets = json.load(response)
            controller = CDP(next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page"))
            for method in ("Page.enable", "Runtime.enable", "Log.enable", "Network.enable"):
                controller.call(method)
            controller.call("Network.setBlockedURLs", {"urls": ["http://*", "https://*"]})
            views = {}
            for label, width, height, mobile in (("desktop", 1440, 1000, False), ("mobile", 390, 844, True)):
                controller.call("Emulation.setDeviceMetricsOverride", dict(width=width, height=height, deviceScaleFactor=1, mobile=mobile))
                controller.call("Page.navigate", {"url": page.resolve().as_uri()})
                deadline = time.monotonic() + 20
                while controller.evaluate("document.readyState") != "complete":
                    assert time.monotonic() < deadline
                    time.sleep(.1)
                images = controller.evaluate("""Promise.all(Array.from(document.images).map(async i=>{i.loading='eager';await i.decode();return {width:i.naturalWidth,height:i.naturalHeight,embedded:i.src.startsWith('data:image/png;base64,')};}))""")
                assert len(images) == 6 and all(i["width"] > 0 and i["height"] > 0 and i["embedded"] for i in images)
                anchors = controller.evaluate("Array.from(document.querySelectorAll('nav a')).map(a=>a.getAttribute('href'))")
                assert len(anchors) == 6
                for anchor in anchors:
                    visible = controller.evaluate(f"""(()=>{{let t=document.querySelector({json.dumps(anchor)});if(!t)return false;t.scrollIntoView({{behavior:'instant'}});return t.getBoundingClientRect().bottom>0&&t.getBoundingClientRect().top<innerHeight;}})()""")
                    assert visible, anchor
                controller.evaluate("scrollTo({top:0,behavior:'instant'})")
                layout = controller.evaluate("""({title:document.title,innerWidth:innerWidth,bodyWidth:document.body.scrollWidth,documentWidth:document.documentElement.scrollWidth,tables:document.querySelectorAll('table').length,externalResources:performance.getEntriesByType('resource').filter(r=>/^https?:/.test(r.name)).map(r=>r.name)})""")
                assert layout["title"] == "팔레트 · 학습된 보정 선택기 결과", layout
                assert layout["bodyWidth"] <= layout["innerWidth"] + 1 and layout["documentWidth"] <= layout["innerWidth"] + 1, layout
                assert layout["tables"] == 18 and not layout["externalResources"], layout
                shot = controller.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                screenshot = P.OUT / f"VIEWER_QA_{label}.png"
                screenshot.write_bytes(base64.b64decode(shot["data"]))
                views[label] = dict(viewport=[width, height], layout=layout, images=images, anchors=anchors, screenshot=P.bound(screenshot))
            errors = [e for e in controller.events if e["method"] == "Runtime.exceptionThrown" or
                      (e["method"] == "Log.entryAdded" and e["params"]["entry"].get("level") == "error")]
            assert not errors, errors
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
    result = dict(passed=True, page=P.bound(page), source=P.bound(Path(__file__)), views=views,
        javascript_errors=errors, gpu_disabled=True, external_page_network_blocked=True,
        own_process_terminated=process.poll() is not None, existing_browser_modified=False,
        temporary_profile=str(profile), argv=argv)
    (P.OUT / "VIEWER_QA.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=True, page=str(page), views=list(views), own_process_terminated=result["own_process_terminated"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
