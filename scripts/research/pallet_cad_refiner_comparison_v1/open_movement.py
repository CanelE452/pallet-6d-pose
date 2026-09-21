"""Open and verify the motion gallery in our already-owned visible Chrome."""
import base64
import json
from pathlib import Path
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'outputs/pallet_cad_refiner_comparison_v1'
OUT=BASE/'movement'


def main():
    previous=json.loads((BASE/'BROWSER_CHECK.json').read_text())
    profile=Path(previous['isolated_profile'])
    port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
    version=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version',timeout=5))
    ws=websocket.create_connection(version['webSocketDebuggerUrl'],suppress_origin=True,timeout=15)
    seq=0
    def call(method,params=None):
        nonlocal seq
        seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
        while True:
            r=json.loads(ws.recv())
            if r.get('id')==seq:
                assert 'error' not in r,r
                return r['result']
    target=call('Target.createTarget',dict(url=(OUT/'index.html').as_uri()))['targetId']
    ws.close()
    pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=5))
    page=next(p for p in pages if p['id']==target)
    ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=15)
    def js(expression):
        r=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
        assert 'exceptionDetails' not in r,r
        return r['result'].get('value')
    try:
        for _ in range(100):
            if js('document.querySelectorAll(".card").length')==18:break
            time.sleep(.1)
        assert js('document.querySelectorAll(".card").length')==18
        assert js('document.querySelectorAll(".full svg").length')==36
        assert js('document.querySelectorAll(".zoom svg").length')==36
        assert js('Promise.all(DATA.map(r=>new Promise(resolve=>{const im=new Image();im.onload=()=>resolve(im.naturalWidth===640&&im.naturalHeight===480);im.onerror=()=>resolve(false);im.src=r.image}))).then(a=>a.every(Boolean))')
        exact=js('''(()=>{
          for(let i=0;i<DATA.length;i++) for(let k=0;k<2;k++) {
            const r=DATA[i],c=r.changes[k],root=document.querySelectorAll('.card')[i].querySelectorAll('.full')[k];
            for(const l of root.querySelectorAll('.motion-arrow')) {
              const j=Number(l.dataset.corner),a=['x1','y1','x2','y2'].map(x=>Number(l.getAttribute(x)));
              if(a.some((v,n)=>Math.abs(v-[...r.before[j],...c.after[j]][n])>1e-10)) return false;
            }
          }
          return true;
        })()''')
        assert exact
        js('document.querySelector("select").value="0";document.querySelector("select").dispatchEvent(new Event("change"))')
        assert js('document.getElementById("p-0-0-detail").textContent.includes("P0")')
        js('update(0,0,defaultCorner(DATA[0],DATA[0].changes[0]));document.querySelector("select").value=String(defaultCorner(DATA[0],DATA[0].changes[0]));document.getElementById("frame-1").scrollIntoView();window.scrollBy(0,-65)')
        call('Page.bringToFront');time.sleep(.3)
        screen=call('Page.captureScreenshot',dict(format='png'))
        (OUT/'browser_preview.png').write_bytes(base64.b64decode(screen['data']))
        result=dict(status='PASS',visible_browser=True,frames=18,full_panels=36,zoom_panels=36,
            all_RGB_loaded=True,arrow_endpoints_exact=True,corner_selection_works=True,source_browser_profile=str(profile),
            url=str(OUT/'index.html'),window_left_open=True)
        (OUT/'BROWSER_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:ws.close()


if __name__=='__main__':main()
