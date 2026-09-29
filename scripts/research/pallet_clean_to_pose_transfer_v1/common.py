import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_clean_to_pose_transfer_v1'
DOC = ROOT/'_docs/experiments'/NAME
RAW = ROOT/'data/pallet/results'/NAME
OUT = ROOT/'outputs'/NAME

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def bind(path):
    p=Path(path).resolve()
    return dict(path=str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p),sha256=sha(p),bytes=p.stat().st_size)
def verify(b): assert sha(ROOT/b['path'])==b['sha256'],b['path']
def save(path,value,freeze=False):
    p=Path(path).resolve();assert any(p.is_relative_to(r) for r in (DOC,RAW,OUT))
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if freeze and p.exists():assert p.read_text()==text,p;return
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(text);os.replace(tmp,p)

def resource(event,seconds=0.,fits=0,updates=0,selector_fits=0,details=None):
    path=DOC/'RESOURCE_LEDGER.json';x=read(path)
    assert event not in [r['event'] for r in x['events']]
    x['events'].append(dict(event=event,at=now(),GPU_training_seconds=seconds,
        student_fits=fits,optimizer_updates=updates,selector_fits=selector_fits,details=details))
    x['totals']={k:sum(r[k] for r in x['events']) for k in
                 ('GPU_training_seconds','student_fits','optimizer_updates','selector_fits')}
    assert x['totals']['GPU_training_seconds']<=21600 and x['totals']['student_fits']<=10
    assert x['totals']['selector_fits']<=1
    save(path,x)
