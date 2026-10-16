"""Registered camera/observability diagnosis; no operating-model mutation."""
from pathlib import Path
import hashlib,json,time
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
RUN=ROOT/'data/pallet/results/pallet_camera_trust_v26'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_camera_trust_v26.md'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending.json');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V26 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')

