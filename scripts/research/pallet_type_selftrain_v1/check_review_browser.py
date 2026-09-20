"""Isolated Chrome smoke test; never reads/writes the user's browser review state."""
import json
import subprocess
import tempfile
import time
import urllib.request
import base64
from pathlib import Path
import websocket
from . import common as C


def main():
    out=C.OUT/'stability_review'
    profile=Path(tempfile.mkdtemp(prefix='pallet249-ui-',dir='/tmp'))
    process=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-gpu',
        '--no-first-run','--remote-debugging-port=0','--remote-allow-origins=*',f'--user-data-dir={profile}',
        '--window-size=1600,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ws=None
    try:
        for _ in range(150):
            if (profile/'DevToolsActivePort').exists():break
            assert process.poll() is None,'Chrome exited';time.sleep(.1)
        port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
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
            result=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            assert 'exceptionDetails' not in result,result
            return result['result'].get('value')
        call('Page.enable');call('Page.navigate',dict(url=(out/'index.html').as_uri()))
        for _ in range(150):
            if js('document.querySelectorAll(".card").length')==249:break
            time.sleep(.1)
        assert js('document.querySelectorAll(".card").length')==249
        js('el("filter").value="priority";render()')
        assert js('document.querySelectorAll(".card").length')==50
        js('document.querySelector(".card button.keep").click();document.querySelector(".card .corner input").click()')
        assert js('choice(DATA[0].id).decision')=='keep'
        assert js('choice(DATA[0].id).excluded_corners.join(",")')=='0'
        assert js('JSON.parse(localStorage.getItem(KEY))[DATA[0].id].decision')=='keep'
        js('window.savedBlob=null;URL.createObjectURL=b=>{window.savedBlob=b;return "blob:test"};HTMLAnchorElement.prototype.click=function(){};el("export").click()')
        exported=js('savedBlob.text().then(x=>JSON.parse(x))')
        assert len(exported['records'])==249 and exported['coordinates_changed'] is False
        assert exported['records'][0]['excluded_corners']==[0]
        # Remove test decisions from this isolated profile before screenshot.
        js('reviews={};save();el("filter").value="all";render()')
        for _ in range(50):
            if js('document.querySelector(".card img").complete && document.querySelector(".card img").naturalWidth>0'):break
            time.sleep(.1)
        assert js('document.querySelector(".card img").naturalWidth')==2560
        screenshot=call('Page.captureScreenshot',dict(format='png'))
        (profile/'screenshot.png').write_bytes(base64.b64decode(screenshot['data']))
        report=dict(status='PASS',checks=['249 cards','priority50 filter','keep decision','corner exclusion',
             'browser persistence','249-record JSON export','unchanged-coordinate flag','image loads'],
             gallery=C.bound(out/'index.html'),test_source=C.bound(__file__),screenshot=str(profile/'screenshot.png'))
        C.freeze(C.DOC/'stability_review/BROWSER_CHECK.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        if ws:ws.close()
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
