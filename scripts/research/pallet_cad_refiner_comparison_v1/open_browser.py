"""Open a visible isolated Chrome window and verify this gallery has 18 images.

No access to the user's browser history/profile; leave the requested window open.
"""
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/pallet_cad_refiner_comparison_v1'


def main():
    profile=Path(tempfile.mkdtemp(prefix='pallet-cad-visible-',dir='/tmp'))
    with (profile/'chrome.log').open('w') as log:
        proc=subprocess.Popen(['/usr/bin/google-chrome','--no-first-run','--no-default-browser-check',
            '--disable-background-networking','--disable-sync','--remote-debugging-address=127.0.0.1',
            '--remote-debugging-port=0',f'--user-data-dir={profile}','--window-size=1480,1080',
            '--new-window','about:blank'],stdout=log,stderr=log,start_new_session=True)
    for _ in range(200):
        if (profile/'DevToolsActivePort').exists():break
        assert proc.poll() is None, f'Chrome exited; see {profile}/chrome.log'
        time.sleep(.1)
    port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
    pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=5))
    page=next(p for p in pages if p['type']=='page')
    ws=websocket.create_connection(page['webSocketDebuggerUrl'],timeout=15,suppress_origin=True)
    seq=0
    def call(method,params=None):
        nonlocal seq
        seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
        while True:
            result=json.loads(ws.recv())
            if result.get('id')==seq:
                assert 'error' not in result,result
                return result['result']
    def js(expression):
        r=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
        assert 'exceptionDetails' not in r,r
        return r['result'].get('value')
    try:
        call('Page.enable');call('Page.navigate',dict(url=(OUT/'index.html').as_uri()))
        for _ in range(200):
            if js('document.querySelectorAll(".card img").length === 18 && [...document.querySelectorAll(".card img")].every(x => x.complete && x.naturalWidth === 1280)'):break
            time.sleep(.1)
        assert js('document.querySelectorAll(".card").length')==18
        assert js('[...document.querySelectorAll(".card img")].every(x => x.complete && x.naturalWidth===1280 && x.naturalHeight===1088)')
        assert js('document.querySelectorAll("table").length')==2
        call('Page.bringToFront')
        js('document.getElementById("f1").scrollIntoView(); window.scrollBy(0,-55)')
        time.sleep(.3)
        shot=call('Page.captureScreenshot',dict(format='png'))
        (OUT/'browser_preview.png').write_bytes(base64.b64decode(shot['data']))
        report=dict(status='PASS',visible_window=True,frames=18,loaded_comparison_images=18,
            metric_tables=2,title=js('document.title'),window_left_open=True,chrome_pid=proc.pid,
            isolated_profile=str(profile),url=str(OUT/'index.html'))
        (OUT/'BROWSER_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        ws.close()


if __name__=='__main__':main()
