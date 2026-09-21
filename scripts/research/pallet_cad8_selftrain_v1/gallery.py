"""Visualize frozen standalone student predictions on all primary OCC96 frames."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad8_selftrain_v1 import run as R

OUT=ROOT/'outputs/pallet_cad8_selftrain_v1'
ARMS=['R0','N3_PNP','REPLAY_PNP']


def build():
    p=R.check();OUT.mkdir(exist_ok=True)
    metrics=R.read(R.RAW/'METRICS.json');mm={a:{r['id']:r for r in rows} for a,rows in metrics.items()}
    paths={'R0':R.C.RAW/'EVAL_PREDICTIONS_R0.json',**{a:R.RAW/f'PREDICTIONS_{a}.json' for a in ARMS[1:]}}
    predictions={a:{r['id']:r['prediction'] for r in R.read(path)['records']} for a,path in paths.items()}
    records=[r for r in p['eval_records'] if r['id'] in p['primary_occlusion_ids']]
    assert len(records)==96 and all(r['session']!='eval_cad' for r in records)
    from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
    pe,pop=O.population_metadata();targets={}
    for item,_ in pop:
        if item.frame_id in p['primary_occlusion_ids']:
            t=pe.E._legacy_forbidden_target(item);targets[item.frame_id]=(np.asarray(t.keypoints_xy),np.asarray(t.keypoint_supervision_mask),np.asarray(t.box_xyxy))
    groups=R.read(R.C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    perms=next(g['permutations'] for g in groups if g['object_type']==R.C.TYPES['PLASTIC'])
    rows=[]
    for i,r in enumerate(records,1):
        R.G.verify(r['image']);R.G.verify(r['annotation']);fid=r['id'];gt,valid,box=targets[fid];arms=[]
        base=R.top(predictions['R0'][fid])
        for arm in ARMS:
            c=R.top(predictions[arm][fid]);m=mm[arm][fid]
            q=None if c is None else np.asarray(c['keypoints_xy'])
            branch=perms[m['branch']];g=gt[branch]
            e=None if q is None else np.linalg.norm(q-g,axis=1).tolist()
            # No coordinates are re-ordered for display; symmetry only labels GT error.
            arms.append(dict(arm=arm,points=None if q is None else q.tolist(),score=None if c is None else c['score'],
                box=None if c is None else c['box_xyxy'],iou=None if c is None else O.iou(c['box_xyxy'],box),
                matched=m['matched'],detected=m['detected'],mean=m['frame_mean_px'],
                median=float(np.median(m['observed_errors'])) if m['observed_errors'] else None,
                pck20=100*float(np.mean(np.array(m['errors'])<=20)),branch=m['branch'],native_errors=e))
        q0=None if base is None else np.array(base['keypoints_xy']);moves=[]
        for j in range(8):
            possible=q0 is not None and valid[j] and 0<=q0[j,0]<640 and 0<=q0[j,1]<480
            distances=[]
            for a in arms[1:]:
                if a['points'] is not None and possible:
                    q=np.array(a['points'][j]);
                    if 0<=q[0]<640 and 0<=q[1]<480:distances.append(float(np.linalg.norm(q-q0[j])))
            moves.append(max(distances) if distances else -1.)
        rows.append(dict(number=i,id=fid,session=r['session'],image=os.path.relpath(ROOT/r['image']['path'],OUT),
            gt=gt.tolist(),valid=valid.tolist(),arms=arms,default_corner=int(np.argmax(moves)),
            n3_delta=arms[1]['mean']-arms[0]['mean'],replay_delta=arms[2]['mean']-arms[0]['mean']))
    template=Path(__file__).with_name('gallery.html').read_text()
    page=template.replace('__DATA__',json.dumps(rows,ensure_ascii=False).replace('</','<\\/'))
    R.write(OUT/'index.html',page);R.write(OUT/'DATA.json',rows)
    R.write(OUT/'MANIFEST.json',dict(frames=96,arms=ARMS,standalone_students=True,GT_used_for_display_only=True,
        coordinates_unchanged=True,training_images_excluded=True,
        display='Native prediction channels; green GT wireframe; optional actual R0→student arrows. Branch labels indicate whole-object C2 evaluation matching only.',
        sources=[R.G.binding(path) for path in [*paths.values(),R.RAW/'METRICS.json',R.DOC/'PROTOCOL.json',Path(__file__),Path(__file__).with_name('gallery.html')]]))
    print(OUT/'index.html')


def open_page():
    import base64
    import time
    import urllib.request
    import websocket
    profile=Path(R.read(R.BASE/'BROWSER_CHECK.json')['isolated_profile'])
    port=(profile/'DevToolsActivePort').read_text().splitlines()[0];origin=f'http://127.0.0.1:{port}'
    version=json.load(urllib.request.urlopen(origin+'/json/version',timeout=5))
    ws=websocket.create_connection(version['webSocketDebuggerUrl'],suppress_origin=True,timeout=30);seq=0
    def call(method,params=None):
        nonlocal seq
        seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
        while True:
            r=json.loads(ws.recv())
            if r.get('id')==seq:
                assert 'error' not in r,r
                return r['result']
    target=call('Target.createTarget',dict(url=(OUT/'index.html').as_uri()))['targetId'];ws.close()
    pages=json.load(urllib.request.urlopen(origin+'/json',timeout=5));page=next(p for p in pages if p['id']==target)
    ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=30)
    def js(expression):
        r=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True));assert 'exceptionDetails' not in r,r
        return r['result'].get('value')
    try:
        for _ in range(100):
            if js('document.querySelectorAll(".card").length')==96:break
            time.sleep(.1)
        assert js('document.querySelectorAll(".card").length')==96
        assert js('document.querySelectorAll(".full svg").length')==288
        loaded=js('Promise.all(DATA.map(r=>new Promise(resolve=>{const im=new Image();im.onload=()=>resolve(im.naturalWidth===640&&im.naturalHeight===480);im.onerror=()=>resolve(false);im.src=r.image}))).then(a=>a.every(Boolean))')
        assert loaded
        js('document.querySelector(".corner").value="3";document.querySelector(".corner").dispatchEvent(new Event("change"))')
        assert js('document.querySelector(".detail").textContent.includes("P3")')
        js('drawCards();document.getElementById("mode").value="lost";applyFilter()')
        lost=js('[...document.querySelectorAll(".card")].filter(c=>!c.hidden).length')
        expected=js('DATA.filter(r=>r.arms[0].matched&&(!r.arms[1].matched||!r.arms[2].matched)).length')
        assert lost==expected
        js('document.getElementById("mode").value="all";applyFilter();window.scrollTo(0,0)')
        call('Page.bringToFront')
        shot=call('Page.captureScreenshot',dict(format='png'));(OUT/'browser_preview.png').write_bytes(base64.b64decode(shot['data']))
        R.write(OUT/'BROWSER_CHECK.json',dict(status='PASS',cards=96,full_panels=288,all_RGB_loaded=True,corner_dropdown_works=True,
            detection_loss_filter_count=lost,window_left_open=True,url=page['url']))
        print('BROWSER_PASS 96 frames /288 panels')
    finally:ws.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--open',action='store_true');a=p.parse_args();build()
    if a.open:open_page()
