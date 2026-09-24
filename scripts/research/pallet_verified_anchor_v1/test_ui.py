"""Exercise actual Tk interactions with all writes redirected to a temporary directory."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from . import common as C
from .label_anchor import App

def main():
    original_raw=C.RAW
    before={p.name:C.sha(p) for p in original_raw.iterdir() if p.is_file()}
    root=tk.Tk();app=App(root);root.update()
    try:
        with tempfile.TemporaryDirectory(prefix='pallet_anchor_ui_') as temp:
            C.RAW=Path(temp)
            app.labels=C.read(original_raw/'LABEL_TEMPLATE.json')
            app.index=0;app.corner=0;app.load()
            app.name.set('')
            x,y=120.,140.
            app.click(SimpleNamespace(x=x*app.scale+app.offset[0],y=y*app.scale+app.offset[1]))
            assert app.corner==1 and app.record()['corners'][0]['xy']==[x,y]
            assert app.record()['annotator']=='anonymous_local'
            assert app.record()['corners'][0]['status']=='DIRECT_VISIBLE'
            app.status(2);assert app.corner==2 and app.record()['corners'][1]['xy'] is None
            app.undo();assert app.corner==1 and app.item()['status'] is None
            app.choose(0)
            app.zoom(SimpleNamespace(x=240,y=240),1.2)
            app.pan_start(SimpleNamespace(x=20,y=20));app.pan_move(SimpleNamespace(x=40,y=50))
            x,y=100.,110.
            app.click(SimpleNamespace(x=x*app.scale+app.offset[0],y=y*app.scale+app.offset[1]))
            assert app.corner==1 and app.record()['corners'][0]['xy']==[x,y]
            saved=C.read(C.RAW/'LABELS.json');assert saved['frames'][0]['corners'][0]['xy']==[x,y]
            assert C.read(C.RAW/'ANNOTATION_PROGRESS.json')['completed_statuses']==1
            app.navigate(1);assert app.index==1
            app.navigate(-1);assert app.item()['xy']==[x,y]
            assert len((C.RAW/'HUMAN_HISTORY.jsonl').read_text().splitlines())==4
            app.choose(7);app.status(5);assert app.corner==7 and app.index==0
            app.choose(1);app.status(1)
            assert app.corner==2
            app.choose(1)
            app.click(SimpleNamespace(x=100*app.scale+app.offset[0],y=100*app.scale+app.offset[1]))
            assert app.corner==2 and app.record()['corners'][1]['status']=='VIRTUAL_INFERABLE'
            app.undo();assert app.corner==1 and app.item()['xy'] is None
            app.choose(0)
            for i,k in enumerate(('d','v','s','e','o','u')):
                app.key(SimpleNamespace(keysym=k))
                assert app.corner==i+1
                assert app.record()['corners'][i]['status']==C.STATUSES[i]
            app.undo();assert app.corner==5
    finally:
        C.RAW=original_raw;root.destroy()
    assert before=={p.name:C.sha(p) for p in original_raw.iterdir() if p.is_file()}, 'Production labels changed during test'
    print('PASS: actual Tk click/native coordinates, zoom/pan, status clearing, undo, autosave, navigation; production labels untouched')

if __name__=='__main__':main()
