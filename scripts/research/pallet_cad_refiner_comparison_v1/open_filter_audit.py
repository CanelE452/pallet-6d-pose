"""Open audit in our isolated visible browser and verify the 18 image cards."""
import json
import time
from pathlib import Path
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'outputs/pallet_cad_refiner_comparison_v1'
OUT=BASE/'filter_audit'


def main():
    profile=Path(json.loads((BASE/'BROWSER_CHECK.json').read_text())['isolated_profile'])
    port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
    origin=f'http://127.0.0.1:{port}'
    version=json.load(urllib.request.urlopen(origin+'/json/version',timeout=5))
    ws=websocket.create_connection(version['webSocketDebuggerUrl'],suppress_origin=True,timeout=20)
    seq=0
    def call(method,params=None):
        nonlocal seq
        seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
        while True:
            r=json.loads(ws.recv())
            if r.get('id')==seq:
                assert 'error' not in r,r
                return r['result']
    target=call('Target.createTarget',dict(url=(OUT/'index.html').as_uri()+'#frame-5'))['targetId']
    ws.close()
    pages=json.load(urllib.request.urlopen(origin+'/json',timeout=5))
    page=next(p for p in pages if p['id']==target)
    ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=20)
    try:
        for _ in range(100):
            result=call('Runtime.evaluate',dict(expression='({cards:document.querySelectorAll(".card").length,images:[...document.images].filter(i=>i.complete&&i.naturalWidth>0).length})',returnByValue=True))['result'].get('value',{})
            if result==dict(cards=18,images=18):break
            time.sleep(.1)
        assert result==dict(cards=18,images=18),result
        call('Page.bringToFront')
        receipt=dict(status='PASS',**result,url=page['url'],window_left_open=True)
        (OUT/'BROWSER_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))
    finally:ws.close()


if __name__=='__main__':main()
