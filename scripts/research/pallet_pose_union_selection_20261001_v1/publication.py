"""Validate and copy this source screen into the clean publication checkout."""
from . import common as C
import ast
import csv
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from PIL import Image

CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')
BASE='647f09952ead26a5b6b06b07a25504593a7bb2af'


def bindings(x):
    if isinstance(x,dict):
        if 'path' in x and 'sha256' in x:yield x
        else:
            for v in x.values():yield from bindings(v)
    elif isinstance(x,list):
        for v in x:yield from bindings(v)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True).strip()
    assert not (C.DOC/'PUBLICATION_MANIFEST.json').exists()
    report=C.read(C.DOC/'REPORT_DATA.json');assert report['complete'] and not report['stable_joint_improvement_achieved']
    for name in ['REPORT_KO.md','REPORT_REVIEW_KO.md','TRAIN_CODE_REVIEW_KO.md','SOURCE_VAL_CODE_REVIEW_KO.md']:
        assert (C.DOC/name).exists(),name
    # Preserve and expose the exact initial implementation that produced the
    # logged failure. Its historical bindings must not be tested against the fix.
    archive={}
    for name in ['common.py','source_infer.py']:
        old=C.RAW/'code_before_runtime_amendment_01'/name
        archive[(str((C.HERE/name).relative_to(C.ROOT)),C.sha(old))]=old
        target=C.DOC/'runtime_initial'/name
        if not target.exists():C.save(target,old.read_text())
        assert target.read_bytes()==old.read_bytes()
    unique={}
    for path in C.DOC.rglob('*.json'):
        for b in bindings(C.read(path)):unique[(b['path'],b['sha256'])]=b
    checked=[]
    for key,b in unique.items():
        p=archive.get(key,C.ROOT/b['path'])
        assert p.is_file() and C.sha(p)==b['sha256'],(b['path'],str(p))
        if 'bytes' in b:assert p.stat().st_size==b['bytes']
        checked.append(dict(expected=b,verified_at=str(p.relative_to(C.ROOT)),historical_archive=key in archive))
    old=C.PREV/'RESULTS.json'
    assert old.read_bytes()==(CHECKOUT/old.relative_to(C.ROOT)).read_bytes()
    rows=list(csv.DictReader((C.DOC/'SOURCE_TRAIN_CANDIDATES.csv').open()))
    assert len(rows)==2598*4*2 and len({(r['id'],r['model'],r['hypothesis']) for r in rows})==len(rows)
    numeric=0
    with np.load(C.RAW/'SOURCE_TRAIN_LABELS.npz') as z:
        indices={fid:j for j,fid in enumerate(z['ids'].tolist())};hyp=z['hypothesis_names'].tolist()
        for r in rows:
            j=indices[r['id']];k=hyp.index(r['hypothesis'])
            for col,suffix in [('T_cm','_T_cm'),('R_deg','_R_deg')]:
                assert float(r[col])==z[r['model']+suffix][j,k];numeric+=1
    counts={'SOURCE_TRAIN_CANDIDATES.csv':len(rows)}
    for name in report['csv_rows']:
        a=list(csv.DictReader((C.DOC/name).open()));assert len(a)==report['csv_rows'][name];counts[name]=len(a)
    if report['VAL_gate_pass'] is not None:
        assert report['new_fits']==6 and report['new_optimizer_updates']==1980 and report['VAL_gate_pass'] is False
        gate=C.read(C.DOC/'SOURCE_VAL_GATE.json');assert not gate['real_routing_authorized']
        with np.load(C.RAW/'SOURCE_VAL_METRICS.npz') as z:
            ix={fid:j for j,fid in enumerate(z['ids'].tolist())}
            for r in csv.DictReader((C.DOC/'SOURCE_VAL_FRAME_RESULTS.csv').open()):
                for col,k in [('T_cm',0),('R_deg',1)]:assert float(r[col])==z[r['model']][ix[r['id']],k];numeric+=1
        checks=list(csv.DictReader((C.DOC/'SOURCE_VAL_CHECKS.csv').open()))
        assert len(checks)==45 and sum(r['PASS']=='False' for r in checks)==len(gate['failed_checks'])
    pictures=[]
    for p in C.DOC.rglob('*.png'):
        with Image.open(p) as im:im.load();pictures.append(dict(file=C.bind(p),size=list(im.size)))
    assert len(pictures)==(3 if report['VAL_gate_pass'] is not None else 2)
    codes=[]
    for p in C.HERE.glob('*.py'):ast.parse(p.read_text());codes.append(C.bind(p))
    links=[]
    for p in C.DOC.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http://','https://','mailto:','#')):continue
            q=(p.parent/target.split('#',1)[0]).resolve()
            if q==C.DOC/'PUBLICATION_MANIFEST.json':continue
            if q.is_relative_to(C.DOC) or q.is_relative_to(C.HERE):assert q.exists(),(p.name,target)
            else:assert (CHECKOUT/q.relative_to(C.ROOT)).exists(),('UNPUBLISHED_LINK',p.name,target)
            links.append(dict(document=p.name,target=target))
    validation=dict(status='PASS',created_at=C.now(),stable_joint_improvement_achieved=False,real_result_unchanged=True,
        report=C.bind(C.DOC/'REPORT_DATA.json'),csv_rows=counts,csv_numeric_checks=numeric,
        verified_bindings=checked,verified_binding_count=len(checked),pictures=pictures,python_AST_checked=codes,
        markdown_links=links,publication_base=BASE,
        visual_review='Root visually inspected actual fixed input gallery and generated numerical plots before publication.')
    C.save(C.DOC/'VALIDATION.json',validation)
    files=sorted([C.ROOT/'.gitignore',C.ROOT/'readme.md',*[p for p in C.DOC.rglob('*') if p.is_file()],*C.HERE.glob('*.py')])
    manifest=dict(base_commit=BASE,goal_achieved=False,performance_verdict='STABLE_JOINT_IMPROVEMENT_NOT_ESTABLISHED',
        public_files_including_this_manifest=len(files)+1,images=len(pictures),files=[C.bind(p) for p in files],
        excluded='Full prediction JSONs, feature/label/pose arrays, original YOLO/PoseFix weights and original datasets remain local with SHA bindings. Six small learned selector parameter sets are published losslessly as JSON. Initial failed inference implementation is published as an explicit historical snapshot.',
        verified_before_copy=C.bind(C.DOC/'VALIDATION.json'))
    C.save(C.DOC/'PUBLICATION_MANIFEST.json',manifest);files.append(C.DOC/'PUBLICATION_MANIFEST.json')
    for p in files:
        q=CHECKOUT/p.relative_to(C.ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        assert q.read_bytes()==p.read_bytes()
    C.save(C.RAW/'PUBLICATION_COPY.json',dict(base=BASE,files=[C.bind(p) for p in files],checkout=str(CHECKOUT)))
    print('READY_FOR_COMMIT',len(files),'images',len(pictures),'numeric_checks',numeric,'bindings',len(checked),'links',len(links),flush=True)


if __name__=='__main__':main()
