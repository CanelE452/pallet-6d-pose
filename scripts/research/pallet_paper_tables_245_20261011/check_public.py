"""Read-only public-file/RGB boundary and local Markdown link checks."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
from PIL import Image
from . import common as C

CODE=C.ROOT/'scripts/research/pallet_paper_tables_245_20261011'
PREFIXES=('_docs/experiments/pallet_paper_tables_245_20261011/','scripts/research/pallet_paper_tables_245_20261011/')
ALLOWED_SUFFIXES={'.json','.jsonl','.gz','.csv','.md','.py','.png','.txt'}

def _paths():
    return sorted(p for root in (C.DOC,CODE) for p in root.rglob('*') if p.is_file())

def _git_old_changes():
    command=['git','status','--porcelain=v1','--untracked-files=all','-z']
    output=subprocess.check_output(command,cwd=C.ROOT)
    entries=output.split(b'\0');changes=[]
    i=0
    while i<len(entries):
        raw=entries[i];i+=1
        if not raw:continue
        status=raw[:2].decode();path=raw[3:].decode()
        changes.append(dict(status=status,path=path))
        assert any(path.startswith(prefix) for prefix in PREFIXES),('Unexpected old-file modification',status,path)
        if 'R' in status or 'C' in status:
            original=entries[i].decode();i+=1
            assert any(original.startswith(prefix) for prefix in PREFIXES),('Old-file rename',original,path)
    return changes

def content(private_dir):
    assert not (C.DOC/'PUBLIC_CONTENT_CHECK.json').exists()
    index=C.read(C.DOC/'FIGURE_INDEX.json');receipt=C.read(C.DOC/'PRIVATE_FIGURES_RECEIPT.json')
    assert index['status']=='COMPLETE' and index['numeric_and_coordinate_only'] and index['raw_RGB_published'] is False
    assert len(index['public_figures'])==4 and len(receipt['outputs'])==11
    public=[];private=[]
    for rel in index['public_figures']:
        p=C.DOC/rel;assert p.is_file() and p.suffix=='.png'
        with Image.open(p) as im:
            assert im.format=='PNG';size=list(im.size);im.verify()
        public.append(dict(path=rel,sha256=C.sha(p),bytes=p.stat().st_size,dimensions_px=size))
    for r in receipt['outputs']:
        p=(private_dir/r['filename']).resolve()
        assert not p.is_relative_to(C.ROOT.resolve()) and p.is_file(),('Private RGB boundary',r['filename'])
        assert C.sha(p)==r['sha256']
        with Image.open(p) as im:assert im.format=='PNG';im.verify()
        private.append(dict(filename=r['filename'],sha256=r['sha256'],bytes=p.stat().st_size))
    public_hashes={r['sha256'] for r in public};private_hashes={r['sha256'] for r in private}
    diagnostic=[]
    amendment=C.read(C.DOC/'FIGURE_REPORT_SOURCE_AMENDMENT.json')
    for artifact in amendment['preserved_initial_artifacts']:
        if Path(artifact['path']).suffix=='.png':
            p=C.DOC/artifact['path'];assert p.is_file() and C.sha(p)==artifact['sha256']
            with Image.open(p) as im:assert im.format=='PNG';im.verify()
            diagnostic.append(dict(path=artifact['path'],sha256=artifact['sha256'],bytes=p.stat().st_size))
    assert C.read(C.DOC/'FIGURE_LAYOUT_VERIFICATION.json')['all_A_numeric_groups_identical'] is True
    all_numeric_hashes=public_hashes|{r['sha256'] for r in diagnostic}
    assert not public_hashes&private_hashes
    input_RGB_hashes={r['image']['sha256'] for r in C.read(C.DOC/'INPUT_AUDIT.json')['images']}
    input_RGB_hashes.update(r['image']['sha256'] for r in C.read(C.ROOT/'_docs/experiments/pallet_square6d_manualpnp_20261010/INPUT_AUDIT.json')['inputs'])
    assert not all_numeric_hashes&input_RGB_hashes and not all_numeric_hashes&private_hashes
    files=[]
    for p in _paths():
        assert not p.is_symlink(),('Public symlink',p.name)
        assert p.suffix in ALLOWED_SUFFIXES,('Unexpected binary/filetype',p.name)
        assert p.stat().st_size<100_000_000,('GitHub100MB limit',p.name,p.stat().st_size)
        assert not p.name.endswith('.pending') and '__pycache__' not in p.parts
        if p.suffix=='.png':assert C.sha(p) in all_numeric_hashes,('Unexpected public PNG',p.name)
        files.append(dict(path=str(p.relative_to(C.ROOT)),bytes=p.stat().st_size,sha256=C.sha(p)))
    # Source-level review: public overlays omit rgb; RGB overlays write only private_dir.
    source=CODE/'figures.py';tree=ast.parse(source.read_text());rgb_writes=[];public_overlay=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='overlay':
            keywords={k.arg:ast.unparse(k.value) for k in node.keywords}
            if 'rgb' in keywords:rgb_writes.append(dict(line=node.lineno,rgb=keywords['rgb']))
            else:public_overlay.append(node.lineno)
    assert len(rgb_writes)==2 and len(public_overlay)==2
    source_text=source.read_text();assert 'pp=private_dir/name' in source_text and 'save(fig,figdir/"FIGURE_C_coordinates.png")' in source_text and 'save(fig,figdir/"FIGURE_D_coordinates.png")' in source_text
    changes=_git_old_changes()
    C.write('PUBLIC_CONTENT_CHECK.json',dict(status='PASS',file_count=len(files),files=files,
        public_numeric_coordinate_figures=public,archived_numeric_diagnostic_figures=diagnostic,
        private_RGB_figures=private,private_RGB_saved_outside_worktree=True,
        public_private_PNG_SHA_disjoint=True,public_input_RGB_SHA_disjoint=True,raw_RGB_published=False,model_weights_or_unapproved_binary_files=0,
        maximum_new_file_bytes=max(r['bytes'] for r in files),all_files_below100MB=True,
        figure_source_sha256=C.sha(source),source_review=dict(public_overlay_call_lines=public_overlay,private_RGB_overlay_calls=rgb_writes),
        publication_worktree_old_tracked_modifications=0,new_namespace_git_changes=len(changes),
        original_workspace_preservation_receipt='WORKSPACE_PRESERVATION.json',
        scope='Pre-report public content boundary check; final Markdown links are checked separately in LOCAL_LINK_CHECK.json.',
        additional_model_forwards=0,additional_F_calls=0))
    print('PUBLIC_CONTENT_CHECK_PASS',len(files),len(public),len(private),flush=True)

def links():
    assert (C.DOC/'REPORT_KO.md').exists() and not (C.DOC/'LOCAL_LINK_CHECK.json').exists()
    checked=[];external=[]
    pattern=re.compile(r'(!?\[[^\]]*\])\(([^)]+)\)')
    for p in _paths():
        if p.suffix!='.md':continue
        text=p.read_text()
        # Do not interpret example links inside fenced code as active hyperlinks.
        text=re.sub(r'```.*?```','',text,flags=re.S)
        for label,target in pattern.findall(text):
            target=target.strip()
            if target.startswith('<'):
                target=target[1:target.index('>')]
            else:target=target.split(' "',1)[0].split(" '",1)[0]
            parsed=urlsplit(target)
            if parsed.scheme in ('http','https','mailto'):
                external.append(dict(document=str(p.relative_to(C.ROOT)),target=target));continue
            if not parsed.path:continue
            targetpath=Path(unquote(parsed.path));targetpath=targetpath if targetpath.is_absolute() else p.parent/targetpath
            targetpath=targetpath.resolve()
            assert targetpath.exists(),('Broken Markdown link',str(p.relative_to(C.ROOT)),target)
            if label.startswith('!'):
                assert targetpath.is_relative_to(C.DOC.resolve()) and targetpath.suffix=='.png',('Private/non-PNG embedded image',target)
                assert str(targetpath.relative_to(C.DOC)) in C.read(C.DOC/'FIGURE_INDEX.json')['public_figures']
            checked.append(dict(document=str(p.relative_to(C.ROOT)),target=target,exists=True))
    C.write('LOCAL_LINK_CHECK.json',dict(status='PASS',local_links=len(checked),checked=checked,external_links_count=len(external),
        external_links=external,all_embedded_images_are_declared_public_numeric_figures=True,
        source_sha256=C.sha(Path(__file__)),report_sha256=C.sha(C.DOC/'REPORT_KO.md')))
    print('LOCAL_MARKDOWN_LINK_CHECK_PASS',len(checked),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['content','links']);parser.add_argument('--private-dir',type=Path)
    args=parser.parse_args()
    if args.stage=='content':assert args.private_dir is not None;content(args.private_dir)
    else:links()

if __name__=='__main__':main()
