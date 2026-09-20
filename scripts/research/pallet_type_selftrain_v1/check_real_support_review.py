"""Read-only gallery QA in a separate temporary browser profile."""
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import websocket
from . import real_support_review_v2 as V


def main():
    cards=V.C.read(V.OUT/'CANDIDATES.json');tiles=[]
    for i,r in enumerate(cards):
        V.C.verify(r['preview_binding']);im=cv2.imread(str(V.OUT/r['preview']))
        tile=cv2.resize(im[:,:im.shape[1]//2],(320,240));banner=np.zeros((28,320,3),np.uint8)
        cv2.putText(banner,f"{i+1:02d} {Path(r['session']).name}",(5,20),0,.5,(255,255,255),1)
        tiles.append(np.concatenate([banner,tile],axis=0))
    contact=np.concatenate([np.concatenate(tiles[i:i+5],axis=1) for i in range(0,30,5)],axis=0)
    contact_path=V.OUT/'contact.jpg'
    if not contact_path.exists():assert cv2.imwrite(str(contact_path),contact)
    profile=Path(tempfile.mkdtemp(prefix='pallet-support-ui-',dir='/tmp'))
    proc=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-gpu','--no-first-run',
        '--remote-debugging-port=0','--remote-allow-origins=*',f'--user-data-dir={profile}','--window-size=1500,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ws=None
    try:
        for _ in range(150):
            if (profile/'DevToolsActivePort').exists():break
            assert proc.poll() is None;time.sleep(.1)
        port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
        page=next(p for p in pages if p['type']=='page')
        ws=websocket.create_connection(page['webSocketDebuggerUrl'],timeout=15,suppress_origin=True);seq=0
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
            if js('document.querySelectorAll("article").length')==30:break
            time.sleep(.1)
        assert js('document.querySelectorAll("article").length')==30
        assert js('document.querySelectorAll("option").length')==3
        for role,count in [('proposed_support',20),('proposed_validation',10),('all',30)]:
            js(f'(()=>{{const s=document.querySelector("#role");s.value="{role}";s.dispatchEvent(new Event("change"));}})()')
            assert js('document.querySelectorAll("article:not([hidden])").length')==count
        assert js('Promise.all(Array.from(document.images).map(im=>{im.loading="eager";return im.decode().then(()=>im.naturalWidth>0)})).then(x=>x.every(Boolean))')
        snap=call('Page.captureScreenshot',dict(format='png'))
        screenshot=profile/'screenshot.png';screenshot.write_bytes(base64.b64decode(snap['data']))
        report=dict(status='PASS',cards=30,filters=[20,10,30],images_loaded=30,isolated_browser=True,
            source=V.C.bound(__file__),page=V.C.bound(V.OUT/'index.html'),screenshot=str(screenshot),contact=V.C.bound(contact_path))
        V.C.freeze(V.DOC/'BROWSER_CHECK.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        if ws:ws.close()
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':main()
