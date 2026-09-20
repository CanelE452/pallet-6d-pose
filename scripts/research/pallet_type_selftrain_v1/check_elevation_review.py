"""Isolated browser test of proposal-only exclusion review."""
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import websocket
from . import elevation_review as V


def main():
    profile=Path(tempfile.mkdtemp(prefix='pallet-elevation-ui-',dir='/tmp'))
    proc=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-gpu',
        '--no-first-run','--remote-debugging-port=0','--remote-allow-origins=*',f'--user-data-dir={profile}',
        '--window-size=1500,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ws=None
    try:
        for _ in range(150):
            if (profile/'DevToolsActivePort').exists():break
            assert proc.poll() is None;time.sleep(.1)
        port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
        ws=websocket.create_connection(next(p for p in pages if p['type']=='page')['webSocketDebuggerUrl'],timeout=20,suppress_origin=True);seq=0
        def call(method,params=None):
            nonlocal seq
            seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
            while True:
                r=json.loads(ws.recv())
                if r.get('id')==seq:
                    assert 'error' not in r,r
                    return r['result']
        def js(expression):
            r=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            assert 'exceptionDetails' not in r,r
            return r['result'].get('value')
        call('Page.enable');call('Page.navigate',dict(url=(V.OUT/'index.html').as_uri()))
        for _ in range(100):
            if js('document.querySelectorAll(".card").length')==104:break
            time.sleep(.1)
        assert js('document.querySelectorAll(".card").length')==104
        assert js('exported().records.every(r=>r.decision==="unreviewed")')
        first=js('document.querySelector(".card").dataset.id')
        js('document.querySelector(".card .exclude").click()')
        assert js(f'JSON.parse(localStorage.getItem(KEY))[{json.dumps(first)}]')=='exclude'
        js('el("filter").value="exclude";render()');assert js('document.querySelectorAll(".card").length')==1
        js('window.savedBlob=null;URL.createObjectURL=b=>{window.savedBlob=b;return "blob:test"};HTMLAnchorElement.prototype.click=function(){};el("export").click()')
        exported=js('savedBlob.text().then(x=>JSON.parse(x))')
        assert len(exported['records'])==194 and exported['active_dataset_changed'] is False and exported['criterion_confirmed'] is False
        expected=V.export_records(V.C.read(V.OUT/'FRAMES.json'),{first:'exclude'})
        assert exported['records']==expected
        js('decisions={};save();el("filter").value="all";render()')
        assert js('document.querySelectorAll(".card").length')==194
        assert js('Promise.all(Array.from(document.images).map(im=>{im.loading="eager";return im.decode().then(()=>im.naturalWidth>0)})).then(x=>x.every(Boolean))')
        js('el("filter").value="candidate";render()')
        assert js('Promise.all(Array.from(document.images).slice(0,2).map(im=>im.decode())).then(()=>true)')
        shot=profile/'screenshot.png';shot.write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        report=dict(status='PASS',default_cards=104,all_cards=194,images_loaded=194,
            proposal_export_matches_python=True,default_exclusions=0,actual_evaluation_untouched=True,
            screenshot=str(shot),page=V.C.bound(V.OUT/'index.html'),test=V.C.bound(__file__))
        V.C.freeze(V.DOC/'BROWSER_CHECK.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        if ws:ws.close()
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':main()
