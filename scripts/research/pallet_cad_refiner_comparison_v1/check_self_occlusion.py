"""Independent preservation checks and visible HTML verification."""
import argparse
import copy
import json
import time
import urllib.request
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad_refiner_comparison_v1 import self_occlusion as S


def verify():
    payload=S.G.read(S.OUT/'PREDICTIONS.json')
    old=S.G.read(S.BASE/'CACHED_PREDICTIONS.json');old.update(S.G.read(S.BASE/'REPLAY_PREDICTIONS.json')['predictions'])
    for r in payload['decisions']:
        before=old[r['arm']][r['id']];after=payload['predictions'][r['arm']][r['id']]
        idx=before['selected_index'];q0=np.array(before['candidates'][idx]['keypoints_xy']);q1=np.array(after['candidates'][idx]['keypoints_xy'])
        assert not set(r['used'])&set(r['hidden'])
        changed=np.flatnonzero(np.any(q0!=q1,axis=1)).tolist()
        assert changed==(r['hidden'] if r['applied'] else [])
        restored=copy.deepcopy(after);restored['candidates'][idx]['keypoints_xy']=before['candidates'][idx]['keypoints_xy']
        assert restored==before
    # Independent ray/slab check of all conservative hidden decisions.
    for r in payload['decisions']:
        p=r.get('initial_pose')
        if not p:continue
        R=np.array(p['R_cf']);t=np.array(p['centroid']);C=-R.T@t
        x=S.cuboid(*p['cf_extents']);lo=x.min(0);hi=x.max(0)
        for j in r['hidden']:
            d=x[j]-C
            a=(lo-C)/d;b=(hi-C)/d
            enter=np.minimum(a,b).max();leave=np.maximum(a,b).min()
            assert enter < 1-1e-7 and leave>=1-1e-7,(r['id'],j,enter,leave)
    for b in S.G.read(S.OUT/'PROTOCOL.json')['sources']:S.G.verify(b)
    receipt=dict(variants_checked=len(payload['decisions']),all_nonhidden_coordinates_exact=True,
                 all_detector_fields_center_and_other_instances_exact=True,hidden_excluded_from_refit=True,
                 independent_ray_box_check=True,source_hashes_unchanged=True)
    S.G.write(S.OUT/'CHECKS.json',receipt);print(json.dumps(receipt))


def open_browser(page_path=None,expected_frames=18,artifact_prefix=''):
    import websocket
    profile=Path(S.G.read(S.BASE/'BROWSER_CHECK.json')['isolated_profile'])
    port=(profile/'DevToolsActivePort').read_text().splitlines()[0];origin=f'http://127.0.0.1:{port}'
    version=json.load(urllib.request.urlopen(origin+'/json/version',timeout=5))
    ws=websocket.create_connection(version['webSocketDebuggerUrl'],suppress_origin=True,timeout=20);seq=0
    def call(method,params=None):
        nonlocal seq
        seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
        while True:
            r=json.loads(ws.recv())
            if r.get('id')==seq:
                assert 'error' not in r,r
                return r['result']
    target=call('Target.createTarget',dict(url=(page_path or S.OUT/'index.html').as_uri()))['targetId'];ws.close()
    pages=json.load(urllib.request.urlopen(origin+'/json',timeout=5));page=next(p for p in pages if p['id']==target)
    ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=20)
    try:
        for _ in range(100):
            n=call('Runtime.evaluate',dict(expression='document.querySelectorAll(".card").length',returnByValue=True))['result'].get('value')
            if n==expected_frames:break
            time.sleep(.1)
        assert n==expected_frames
        expression='Promise.all([...new Set([...document.querySelectorAll("svg image")].map(x=>x.getAttribute("href")))].map(src=>new Promise(resolve=>{const im=new Image();im.onload=()=>resolve(im.naturalWidth===640&&im.naturalHeight===480);im.onerror=()=>resolve(false);im.src=src}))).then(a=>({images:a.length,loaded:a.every(Boolean)}))'
        loaded=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))['result']['value']
        assert loaded==dict(images=expected_frames,loaded=True),loaded
        call('Page.bringToFront')
        import base64
        shot=call('Page.captureScreenshot',dict(format='png'))
        (S.OUT/f'{artifact_prefix}browser_preview.png').write_bytes(base64.b64decode(shot['data']))
        result=dict(status='PASS',cards=n,**loaded,url=page['url'],window_left_open=True)
        S.G.write(S.OUT/f'{artifact_prefix}BROWSER_CHECK.json',result);print(json.dumps(result))
    finally:ws.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--open',action='store_true');a=p.parse_args()
    verify()
    if a.open:open_browser()
