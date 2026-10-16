"""Move registered flat notes into topic folders without overwriting existing notes.

Past experiment scripts remain frozen. If they regenerate old flat files, --apply
moves only absent destinations or removes byte-identical duplicates; conflicts stop.
"""
from pathlib import Path
import argparse,hashlib,json,os,re
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[2]
NOTES=ROOT/'_docs/notes'
MAP=NOTES/'FILE_MAP.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def destination(name):
    if name in ('c4-rotation-symmetry.md','diffpnp-yolo.md','pnp-solver-swap.md'):return 'pose_geometry/'+name
    if name in ('corrected-real-supervision.md','multiteacher-corner-distill.md','utility-aware-selftraining.md'):return 'supervision_and_distillation/'+name
    match=re.search(r'_v(\d+)',name)
    folder=('02_local_and_identity' if int(match[1])<=12 else '03_correction_and_stability' if int(match[1])<=19 else '04_failure_analysis') if match else '01_hough_and_geometry'
    assert name.startswith(('pallet_','dht_','hough_')),name
    return 'point_line_fusion/'+folder+'/'+name
LINK=re.compile(r'(!?\[[^\]\n]*\]\()([^\n)]+)(\))')
def targets(text):
    fenced=False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith('```'):fenced=not fenced
        if not fenced:
            for m in LINK.finditer(line):yield m[2]
def split_target(raw):
    bracket=raw.startswith('<')
    if bracket:
        end=raw.index('>');return raw[1:end],raw[end+1:],True
    m=re.match(r'(\S+)(.*)',raw);return (m[1],m[2],False) if m else ('','',False)
def plausible(value,parent):
    return value.startswith(('.', '/')) or '/' in value or bool(re.search(r'\.(md|json|html|png|jpg|jpeg|svg|py|npz|pt|csv|txt)(?:[#?]|$)',value)) or (parent/value).exists()
def local_target(raw,parent):
    value,_,_=split_target(raw);u=urlsplit(value)
    if not value or u.scheme or value.startswith(('#','//')) or not plausible(value,parent):return None
    return (parent/unquote(u.path)).resolve()
def rewrite(text,old,new,mapping):
    def replace(m):
        value,suffix,bracket=split_target(m[2]);u=urlsplit(value)
        if not value or u.scheme or value.startswith(('#','//')) or not plausible(value,old.parent):return m[0]
        target=(old.parent/unquote(u.path)).resolve();target=mapping.get(target,target)
        if old==new and target==(old.parent/unquote(u.path)).resolve():return m[0]
        dest=os.path.relpath(target,new.parent) if not value.startswith('/') else str(target)
        if u.query:dest+='?'+u.query
        if u.fragment:dest+='#'+u.fragment
        return m[1]+('<'+dest+'>' if bracket or ' ' in dest else dest)+suffix+m[3]
    lines=[];fenced=False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith('```'):fenced=not fenced
        lines.append(line if fenced else LINK.sub(replace,line))
    out=''.join(lines)
    # Repository-root references in prose/backticks, not experiment source files.
    for oldpath,newpath in mapping.items():out=out.replace(str(oldpath.relative_to(ROOT)),str(newpath.relative_to(ROOT)))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    registered=json.loads(MAP.read_text())['moves'] if MAP.exists() else [{'old':'_docs/notes/'+p.name,'new':'_docs/notes/'+destination(p.name)} for p in sorted(NOTES.glob('*.md')) if p.name!='README.md']
    mapping={(ROOT/r['old']).resolve():(ROOT/r['new']).resolve() for r in registered};pending=[]
    for old,new in mapping.items():
        if old.exists():
            transformed=rewrite(old.read_text(),old,new,mapping)
            if new.exists() and new.read_text()!=transformed:raise RuntimeError(f'Conflicting note; not overwritten: {new}')
            pending.append((old,new,transformed))
    if not a.apply:
        print(json.dumps(dict(registered=len(registered),pending=len(pending),destinations_missing=[r['new'] for r in registered if not (ROOT/r['new']).exists() and not (ROOT/r['old']).exists()]),ensure_ascii=False,indent=2));return
    before={p:sha(p) for p,_,_ in pending};broken_before=[];rewritten=[]
    documents=list((ROOT/'_docs').rglob('*.md'))
    for old in documents:
        for raw in targets(old.read_text()):
            target=local_target(raw,old.parent)
            if target and not target.exists():broken_before.append((str(old.relative_to(ROOT)),raw))
    # Move existing files, keeping every document and only rebasing paths.
    for old,new,content in pending:
        new.parent.mkdir(parents=True,exist_ok=True)
        if new.exists():old.unlink() # verified identical transformed content above
        else:old.rename(new);new.write_text(content)
    for old in documents:
        if old in mapping:continue
        content=old.read_text();updated=rewrite(content,old,old,mapping)
        if updated!=content:old.write_text(updated);rewritten.append(str(old.relative_to(ROOT)))
    if not MAP.exists():MAP.write_text(json.dumps(dict(version=1,moves=registered),ensure_ascii=False,indent=2)+'\n')
    report=dict(moved=len(pending),before_sha256={str(p.relative_to(ROOT)):h for p,h in before.items()},after_sha256={str(n.relative_to(ROOT)):sha(n) for _,n,_ in pending},reference_documents_updated=rewritten,preexisting_unresolved_markdown_links=[dict(file=p,target=t) for p,t in broken_before],frozen_experiment_sources_modified=False)
    out=ROOT/'_docs/assets/point_line_fusion/ORGANIZATION_RECEIPT.json';out.parent.mkdir(parents=True,exist_ok=True)
    if pending:out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(moved=len(pending),reference_documents_updated=len(rewritten),preexisting_unresolved_links=len(broken_before)),ensure_ascii=False))
if __name__=='__main__':main()
