"""Byte-only preparation of an existing reader copy; ROOT alone executes.

This checks the selected publication receipt, not a new clone or science run.
No nested historical declarations are traversed and no existing file is changed.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import traceback

FILES=('COPIED_PREPARE_PROTOCOL.json','COPIED_PREPARE_STARTED.json','COPIED_PREPARE_CHECKS.json')
LIMITS=dict(existing_copied_parent_root=True,new_Git_clone_or_reader_execution=False,
    nested_historical_dependency_traversal=False,source_or_Git_writes=False,
    scientific_RGB_model_GT_metric_timing_calls=0,gzip_decode_or_recompression_calls=0)

def require(value,message):
    if not value:raise ValueError(message)

def safe(value):
    p=Path(value).absolute()
    require('..' not in p.parts and not any(q.is_symlink() for q in (p,*p.parents)),'unsafe/symlink path '+str(p))
    return p

def relative(value):
    require(isinstance(value,str) and '\0' not in value and '\n' not in value,'invalid path declaration')
    p=Path(value)
    require(not p.is_absolute() and '..' not in p.parts and len(p.parts)>2
        and p.parts[:2] in (('scripts','research'),('_docs','experiments')),'outside selected public trees')
    return p.as_posix()

def digest(value):
    p=safe(value);require(p.is_file(),'missing regular file '+str(p));h=hashlib.sha256();n=0
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block);n+=len(block)
    require(n==p.stat().st_size,'file changed while hashing '+str(p))
    return dict(bytes=n,sha256=h.hexdigest())

def binding(value):return dict(path=str(safe(value)),**digest(value))
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def read(value):return json.loads(safe(value).read_bytes())

def write(value,record):
    p=safe(value);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as stream:
        stream.write((json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())

def plan(a,receipt):
    require(receipt['schema']=='explicit_private_publication_checks_v21'
        and receipt['complete'] is True and receipt['passed'] is True and receipt['error'] is None
        and receipt['original_and_source_Git_state_unchanged'] is True,'exact successful source publication receipt')
    records={};kinds={}
    def add(name,item,kind):
        name=relative(name);require(type(item['bytes']) is int and item['bytes']>=0
            and isinstance(item['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['sha256']),'invalid byte declaration')
        expected={k:item[k] for k in ('bytes','sha256')}
        require(name not in records or records[name]==expected,'conflicting declarations '+name)
        records[name]=expected;kinds.setdefault(name,set()).add(kind)
    for item in receipt['actual_direct_files']:
        require(isinstance(item['source'],str),'direct source provenance');add(item['target'],item,'direct')
    for item in receipt['generated_files']:add(item['path'],item,'generated')
    for archive in receipt['actual_ordered_archives']:
        add(archive['original']['path'],archive['original'],'original_gzip')
        require(archive['original']['path'].endswith('.gz') and archive['storage']=='ordered_parts','ordered gzip declaration')
        for part in archive['parts']:add(part['path'],part,'ordered_part')
    require(records,'empty copied-reader plan');result=[]
    for name,expected in sorted(records.items()):
        source=safe(a.stage/name);target=safe(a.copied_root/name)
        require(same(digest(source),expected),'stage bytes differ '+name)
        if target.exists():require(same(digest(target),expected),'existing copied bytes differ '+name)
        result.append(dict(path=name,source=str(source),target=str(target),**expected,
            declaration_kinds=sorted(kinds[name]),existing_byte_identical=target.exists()))
    return result

def link(item):
    source=safe(item['source']);target=safe(item['target'])
    require(same(digest(source),item),'stage changed before copy '+item['path'])
    if target.exists():require(same(digest(target),item),'existing copied conflict');return 'reused'
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target);kind='hardlinked'
    except OSError as exc:
        if exc.errno!=errno.EXDEV:raise
        with source.open('rb') as src,target.open('xb') as out:
            shutil.copyfileobj(src,out,1048576);out.flush();os.fsync(out.fileno())
        kind='exclusive_copied'
    require(same(digest(target),item),'prepared bytes differ');return kind

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','run'))
    for name in ('stage','copied-root','publication-checks','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    for name in ('stage','copied_root','publication_checks','output'):setattr(a,name,safe(getattr(a,name)))
    require(a.stage.is_dir() and a.copied_root.is_dir(),'both roots must already exist')
    roots=(a.stage,a.copied_root,a.output)
    require(all(not x.is_relative_to(y) and not y.is_relative_to(x) for i,x in enumerate(roots) for y in roots[i+1:]),'disjoint roots/output')
    require(not a.publication_checks.is_relative_to(a.output) and not safe(__file__).is_relative_to(a.output),'own output would contain an input')
    for name in FILES if a.mode=='freeze' else FILES[1:]:require(not(a.output/name).exists(),'preserve own receipt prefix '+name)
    checks=binding(a.publication_checks);receipt=read(a.publication_checks);entries=plan(a,receipt)
    require(binding(a.publication_checks)==checks,'source checks changed during planning')
    config={name:str(getattr(a,name)) for name in ('stage','copied_root','publication_checks','output')}
    frozen=dict(schema='copied_selected_publication_protocol_v21',configuration=config,
        helper=binding(__file__),source_publication_checks=checks,files=entries,frozen_before_copy=True,limits=LIMITS)
    proto=a.output/FILES[0]
    if a.mode=='freeze':write(proto,frozen);return
    require(read(proto)==frozen,'frozen copied-reader source/plan differs');protocol=binding(proto)
    write(a.output/FILES[1],dict(protocol=protocol,helper=frozen['helper'],source_publication_checks=checks,limits=LIMITS))
    actions=[];error=None
    try:
        for item in entries:actions.append(dict(path=item['path'],storage=link(item),bytes=item['bytes'],sha256=item['sha256']))
        require(binding(a.publication_checks)==checks and binding(__file__)==frozen['helper'],'source receipt/helper changed')
        for item in entries:
            require(same(digest(item['source']),item) and same(digest(item['target']),item),'final selected byte proof differs')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    counts={kind:sum(item['storage']==kind for item in actions) for kind in ('reused','hardlinked','exclusive_copied')}
    result=dict(schema='copied_selected_publication_checks_v21',complete=error is None,passed=error is None,error=error,
        protocol=protocol,helper=frozen['helper'],source_publication_checks=checks,actions=actions,
        actual_counts=dict(selected_files=len(entries),completed_files=len(actions),copied=counts['hardlinked']+counts['exclusive_copied'],
            reused=counts['reused'],hash_verified=len(actions),**counts),all_selected_source_and_target_bytes_verified=error is None,
        partial_copied_prefix_preserved_on_failure=True,limits=LIMITS)
    write(a.output/FILES[2],result);print(json.dumps(dict(complete=result['complete'],passed=result['passed'],error=error)),flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()
