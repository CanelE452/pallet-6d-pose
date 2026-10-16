"""Prepare an explicitly bound reader copy; ROOT executes freeze/run.

Scientific gzip is neither decoded nor recompressed. Existing parent bytes are
verified and retained. Raw hardlinks are reused originals, never fresh restores.
The historical publication physical-index mismatch remains in its original JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PRIVATE=Path(__file__).resolve().parent
FAMILY='pallet_finite_support_orthogonal_audit_20261011_v19'
DOC=Path('_docs/experiments')/FAMILY
LATEST=Path('_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md')
FILES=('COPY_PROTOCOL.json','COPY_STARTED.json','COPY_CHECKS.json')

def require(v,s):
    if not v: raise ValueError(s)

def read(p): return json.loads(Path(p).read_bytes())

def binding(p):
    p=Path(p).absolute(); require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)), 'regular nonsymlink '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def write(p,v):
    with p.open('x') as f: json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())

def snapshot(root):
    def git(*a): return subprocess.check_output(['git','--no-optional-locks','-C',str(root),*a])
    return dict(HEAD=git('rev-parse','HEAD').decode().strip(),index_sha256=hashlib.sha256(git('ls-files','--stage','-z')).hexdigest(),tracked_diff_sha256=hashlib.sha256(git('diff','--binary')).hexdigest(),status_sha256=hashlib.sha256(git('status','--porcelain=v1','-uall')).hexdigest())

def plan(a):
    proto=a.stage/DOC/'publication/PUBLICATION_PROTOCOL.json'; checks=a.stage/DOC/'publication/PUBLICATION_CHECKS.json'
    p,c=read(proto),read(checks)
    require(c['complete'] is c['passed'] is True and c['error'] is None,'completed original byte preparation')
    require(binding(proto)=={k:c['protocol'][k] for k in ('bytes','sha256')},'original publication protocol')
    require(c['original_and_source_Git_state_unchanged'] is False,'preserve actual physical-index warning')
    fields=('HEAD','staged','tracked','status')
    require(all(all(p['protected'][r][k]==c['source_protection_after'][r][k] for k in fields) for r in p['protected']),'unchanged original publication semantic fields')
    selected={}
    def add(row):
        path=Path(row['path']); require(not path.is_absolute() and '..' not in path.parts,'public relative path')
        value={k:row[k] for k in ('bytes','sha256')}
        require(path not in selected or selected[path]==value,'single exact public byte origin'); selected[path]=value
    for row in c['actual_direct_files']: add(row)
    for row in c['generated_files']: add(row)
    add(dict(path=str(DOC/'publication/PUBLICATION_CHECKS.json'),**binding(checks)))
    for archive in c['actual_ordered_archives']:
        add(archive['original'])
        for part in archive['parts']: add(part)
    records=[]
    for path,expected in sorted(selected.items()):
        require(binding(a.stage/path)==expected,'staged exact bytes '+str(path))
        target=a.copy/path; before=binding(target) if target.exists() else None
        require(before is None or before==expected or path==LATEST,'preserve existing copied parent '+str(path))
        records.append(dict(path=str(path),expected=expected,before=before))
    semantic={str(r):snapshot(r) for r in p['protected']}
    g=read(PRIVATE/'geometry_ready/GEOMETRY_RECEIPT.json')['protection_after']
    require(semantic[str(a.source)]==g['research'] and semantic[str(a.original)]==g['original'],'current canonical index/source/user changes preserved since actual geometry')
    return dict(records=records,original_protocol=binding(proto),original_checks=binding(checks),source=binding(__file__),protection=semantic,physical_index_mismatch_preserved=True,scientific_calls=0,raw_original_storage='reused immutable hardlink; not ordered restoration',copied_root=str(a.copy),stage_root=str(a.stage))

def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('freeze','run'))
    for k in ('stage','copy','source','original'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    for k in ('stage','copy','source','original'):
        v=getattr(a,k).absolute(); require(v.is_dir() and not any(q.is_symlink() for q in (v,*v.parents)),'existing nonsymlink root'); setattr(a,k,v)
    require(a.copy!=a.stage and not a.copy.is_relative_to(a.stage) and not a.stage.is_relative_to(a.copy),'disjoint byte copy')
    out=PRIVATE/'copied_reader_preparation'; out.mkdir(exist_ok=True)
    for f in FILES if a.phase=='freeze' else FILES[1:]: require(not(out/f).exists(),'preserve '+f)
    current=plan(a)
    if a.phase=='freeze': write(out/FILES[0],dict(schema='observed_frame_reader_copy_protocol_v19',plan=current,frozen_before_copy=True)); return
    own=read(out/FILES[0]); require(own['plan']==current and own['frozen_before_copy'] is True,'exact own frozen copy plan')
    write(out/FILES[1],dict(protocol=binding(out/FILES[0]),plan=current)); actions=[]
    for row in current['records']:
        path=Path(row['path']); source=a.stage/path; target=a.copy/path; target.parent.mkdir(parents=True,exist_ok=True)
        require(binding(source)==row['expected'],'current source bytes')
        if row['before']==row['expected']: storage='retained_exact_existing_copy'
        elif row['before'] is not None:
            require(path==LATEST and binding(target)==row['before'],'sole frozen LATEST replacement')
            temp=target.with_name('LATEST_V19_COPY_PENDING.md'); require(not temp.exists(),'preserve copy prefix'); os.link(source,temp); os.replace(temp,target); storage='explicit_LATEST_only_replacement'
        else: require(not target.exists(),'exclusive missing copy'); os.link(source,target); storage='immutable_public_stage_hardlink'
        require(binding(target)==row['expected'],'copied exact bytes'); actions.append(dict(row,storage=storage))
    protection={str(r):snapshot(r) for r in (a.source,a.original)}; require(protection==current['protection'],'canonical source/user changes retained')
    write(out/FILES[2],dict(schema='observed_frame_reader_copy_checks_v19',complete=True,passed=True,files=len(actions),actions=actions,protection=protection,physical_index_mismatch_preserved=True,portable_reader_execution_completed=False,ordered_restore_performed=False,scientific_calls=0,protocol=binding(out/FILES[0]),source=binding(__file__)))
    print('COPY_BYTES_PASS',len(actions))

if __name__=='__main__': main()
