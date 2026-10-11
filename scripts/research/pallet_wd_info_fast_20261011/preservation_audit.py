"""Verify the user's original changes and bound historical inputs remain intact."""
import ast
import hashlib
import json
import subprocess
from . import common as C


def run():
    def git(*args):
        return subprocess.check_output(['git','-C',str(C.SOURCE),*args])
    before=C.read(C.PRIVATE/'START.json')
    status=git('status','--porcelain=v1','-z')
    diff=git('diff','--binary','HEAD')
    assert hashlib.sha256(status).hexdigest()==before['status_sha256']
    assert hashlib.sha256(diff).hexdigest()==before['tracked_diff_sha256']
    assert status==(C.PRIVATE/'ORIGINAL_STATUS.bin').read_bytes()
    assert diff==(C.PRIVATE/'ORIGINAL_TRACKED.diff').read_bytes()
    bound={};unowned=[]
    def visit(v):
        if isinstance(v,dict):
            if v.get('owner') in ('historical_source','published_source') and 'path' in v and 'sha256' in v:
                root=C.SOURCE if v['owner']=='historical_source' else C.PUBLIC
                bound[v['owner'],v['path']]=v['sha256']
            for x in v.values():visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    for f in C.DOC.glob('*.json'):visit(C.read(f))
    # D source bindings are explicitly relative to the original repository.
    for v in C.read(C.DOC/'D_CONTRACT.json')['source_bindings']:
        bound['historical_source',v['path']]=v['sha256']
    results=[]
    for (owner,path),expected in sorted(bound.items()):
        root=C.SOURCE if owner=='historical_source' else C.PUBLIC
        target=root/path
        # Existing huge memmaps are opened read-only and tied to manifests;
        # do not invent before/after full-file hashes that were never captured.
        if target.stat().st_size>128*1024*1024:
            results.append(dict(owner=owner,path=path,status='MANIFEST_BOUND_READ_ONLY',bytes=target.stat().st_size));continue
        actual=C.sha(target);assert actual==expected,(owner,path)
        results.append(dict(owner=owner,path=path,status='SHA256_UNCHANGED',sha256=actual))
    for v in C.read(C.DOC/'PLAN_LOCK.json')['new_code_bindings']:
        assert C.sha(C.ROOT/v['path'])==v['sha256'],v['path']
    checked=0
    for f in (C.ROOT/'scripts/research/pallet_wd_info_fast_20261011').rglob('*.py'):
        ast.parse(f.read_text(),filename=str(f.relative_to(C.ROOT)));checked+=1
    disallowed=[]
    for f in list(C.DOC.rglob('*'))+list((C.ROOT/'scripts/research/pallet_wd_info_fast_20261011').rglob('*')):
        if not f.is_file():continue
        if f.suffix in ('.pt','.pth','.npz','.jpg','.jpeg','.tif','.tiff','.pdf','.tex') or f.stat().st_size>16*1024*1024:
            disallowed.append(str(f.relative_to(C.ROOT)))
        if f.suffix in ('.md','.json','.csv','.py','.mplstyle'):
            text=f.read_text()
            prefixes=['/'+'home'+'/', '/'+'dev'+'/'+'shm'+'/', '/'+'tmp'+'/']
            assert not any(token in text for token in prefixes),f.name
    assert not disallowed,disallowed
    C.write(C.DOC/'SOURCE_PRESERVATION.json',dict(status='PASS',
        original_status_count=before['original_status_count'],original_status_bytes_identical=True,
        original_tracked_diff_bytes_identical=True,
        status_sha256=before['status_sha256'],tracked_diff_sha256=before['tracked_diff_sha256'],
        historical_bound_files=results,frozen_plan_core_hashes_unchanged=True,
        Python_AST_files_checked=checked,private_absolute_paths_found=0,
        disallowed_public_artifacts=disallowed,
        old_manuscript_data_checkpoint_and_results_modified=False,
        limitation='Huge cache arrays are read-only/manifest-bound; full pre-run cache hashes were not fabricated'))
    print('PRESERVED',before['original_status_count'],'status entries;',len(results),'historical bindings;',checked,'Python files')


if __name__=='__main__':run()
