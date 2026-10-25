"""V250 explicit public-repository replay and path-only protocol relocation.

This wrapper has no numeric decoder or selector math. It runs only the pinned
sibling runner after all ten public input artifacts and policy bytes verify.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

ORIGINAL_PROTOCOL_SHA='af9ddfbce63992575eac86f073e2418be08021df72cddcdd6716fda0720b921b'
ORIGINAL_PROTOCOL_BYTES=9431
MAP_SHA='1d59f481139c3f14607bb7e484e63c9fc07bb53787e5c993258dc959a2c7a9c3'
MAP_BYTES=6725
# Exact immutable B FINAL; preparation never executes this runner.
RUNNER_DIRECTORY='V250_INPUT_REFERENCE_SELECTOR_B'
RUNNER_FILENAME='select_input_reference.py'
RUNNER_SHA='0e08c0b8d1ba82795aa56be9326ff57f9bdcdb29a300b1de5d74f22d412b421a'
RUNNER_BYTES=22416
IMPLEMENTATION_PROTOCOL_SHA='142eed16be5d342ad2d6d2a4be43b2282a26f563e2e6f2bd21ee1f55a0bc75bc'
IMPLEMENTATION_PROTOCOL_BYTES=1412


def lexical(value):
    text=str(value)
    if any(t in text.casefold() for t in ('v112','cache','.pyc','current223','publicheld','public_held','oldval','oldhold','/hold/','v177','v178','source_holdout','source_val','sourceval','v182','private_evaluation')) or '\x00' in text:
        raise ValueError('Forbidden path before filesystem access')
    if '..' in Path(text).parts: raise ValueError('Parent path component forbidden')
    return text


def guarded(value,existing=False,regular=False):
    path=Path(os.path.abspath(lexical(value)))
    for parent in reversed((path,*path.parents)):
        try: info=os.lstat(parent)
        except FileNotFoundError:
            if existing: raise
            continue
        if stat.S_ISLNK(info.st_mode): raise ValueError('Symlink ancestor/artifact rejected')
    if existing and regular and not stat.S_ISREG(os.lstat(path).st_mode): raise ValueError('Regular file required')
    return path


def read_bytes(path):
    with os.fdopen(os.open(guarded(path,True,True),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def binding(path):
    path=guarded(path,True,True);sha=hashlib.sha256();size=0
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):sha.update(b);size+=len(b)
    return {'path':str(path),'sha256':sha.hexdigest(),'size_bytes':size}


def verify(path,sha,size):
    b=binding(path)
    if b['sha256']!=sha or b['size_bytes']!=size: raise ValueError('Pinned SHA/size mismatch: '+str(path))
    return b


def write_bytes(path,data):
    with os.fdopen(os.open(guarded(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data)
    return binding(path)


def write_json(path,value):
    return write_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())


def repo_path(repo,record):
    value=lexical(record['repository_relative_path']);p=PurePosixPath(value)
    if p.is_absolute() or not p.parts or '\\' in value or value!=p.as_posix() or any(x in ('.','..') for x in p.parts):raise ValueError('Canonical repository-relative POSIX path only')
    target=guarded(repo/Path(*p.parts),True,True)
    if not target.is_relative_to(repo):raise ValueError('Outside explicit --public-repo')
    return target


def relocate(original,mapping,repo):
    expected={(s,k) for s in ('inputs','metadata_provenance') for k in original[s]}
    records=mapping['dependencies']
    if len(records)!=10 or {(x['section'],x['key']) for x in records}!=expected:raise ValueError('Exactly eight payload and two provenance bindings')
    relocated=copy.deepcopy(original);verified=[]
    for record in records:
        section,key=record['section'],record['key'];old=original[section][key]
        # This compares inert provenance strings; original_local_path is NEVER opened.
        if old!={'path':record['original_local_path'],'sha256':record['sha256'],'size_bytes':record['size_bytes']}:raise ValueError('Map differs from original frozen input binding')
        target=repo_path(repo,record);checked=verify(target,record['sha256'],record['size_bytes'])
        relocated[section][key]['path']=str(target)
        verified.append({'section':section,'key':key,'repository_relative_path':record['repository_relative_path'],'original_binding':old,'relocated_binding':checked})
    restored=copy.deepcopy(relocated)
    for record in records:restored[record['section']][record['key']]['path']=record['original_local_path']
    if restored!=original:raise ValueError('Only binding paths may be relocated; scientific policy changed')
    return relocated,verified


def runner_command(runner,source_protocol,relocated_protocol,results):
    implementation=guarded(runner.parent/'IMPLEMENTATION_PROTOCOL.json',True,True)
    implementation_binding=verify(implementation,IMPLEMENTATION_PROTOCOL_SHA,IMPLEMENTATION_PROTOCOL_BYTES)
    relocated=json.loads(read_bytes(relocated_protocol))
    entry={'approved_SOURCE_input_reference_selector':True,'automatic_retry':False,
           'code':verify(runner,RUNNER_SHA,RUNNER_BYTES),
           'implementation_protocol':implementation_binding,
           'original_protocol':verify(source_protocol,ORIGINAL_PROTOCOL_SHA,ORIGINAL_PROTOCOL_BYTES),
           'root_protocol':binding(relocated_protocol),
           'inputs':relocated['inputs'],'metadata_provenance':relocated['metadata_provenance'],
           'output_directory':str(guarded(results))}
    scope=write_json(results.parent/'PUBLIC_REPLAY_SCOPE.json',
                     {'approved':True,'entry_contract':copy.deepcopy(entry)})
    entry['root_scope']=scope
    actual=write_json(results.parent/'PUBLIC_REPLAY_ENTRY.json',entry)
    return [sys.executable,'-B',str(runner),'--protocol',str(relocated_protocol),
            '--entry',actual['path'],'--output-dir',str(results)]


def replay(public_repo,output_directory):
    if not Path(lexical(public_repo)).is_absolute():raise ValueError('--public-repo must be ABS')
    repo=guarded(public_repo,True)
    if not repo.is_dir():raise ValueError('--public-repo must be a directory')
    if not Path(lexical(output_directory)).is_absolute():raise ValueError('--output-dir must be FRESHABS')
    out=guarded(output_directory)
    if out.exists():raise ValueError('Fresh output directory required; overwrite/retry forbidden')
    here=guarded(__file__,True,True).parent
    source=guarded(here/'ORIGINAL_PROTOCOL.json',True,True)
    source_binding=verify(source,ORIGINAL_PROTOCOL_SHA,ORIGINAL_PROTOCOL_BYTES)
    source_raw=read_bytes(source)
    if hashlib.sha256(source_raw).hexdigest()!=ORIGINAL_PROTOCOL_SHA:raise ValueError('Original protocol changed')
    original=json.loads(source_raw)
    map_path=guarded(here/'public_dependencies.json',True,True);map_binding=verify(map_path,MAP_SHA,MAP_BYTES)
    mapping=json.loads(read_bytes(map_path))
    if mapping.get('completed') is not True or mapping.get('PASS') is not True or mapping['original_protocol_sha256']!=ORIGINAL_PROTOCOL_SHA:raise ValueError('Exact relocation map required')
    runner=guarded(here.parent/RUNNER_DIRECTORY/RUNNER_FILENAME,True,True)
    code_binding=verify(runner,RUNNER_SHA,RUNNER_BYTES)
    verify(runner.parent/'IMPLEMENTATION_PROTOCOL.json',IMPLEMENTATION_PROTOCOL_SHA,IMPLEMENTATION_PROTOCOL_BYTES)
    relocated,verified=relocate(original,mapping,repo)
    # Input byte verification completes before any child numeric/label decoder.
    out.mkdir(parents=True)
    original_copy=write_bytes(out/'ORIGINAL_PROTOCOL.json',source_raw)
    relocated_binding=write_json(out/'RELOCATED_PROTOCOL.json',relocated)
    receipt=write_json(out/'PATH_ONLY_RELOCATION_RECEIPT.json',{'completed':True,'PASS':True,'original_protocol':source_binding,'original_protocol_copy':original_copy,'relocated_protocol':relocated_binding,'dependency_map':map_binding,'runner_code':code_binding,'public_repo':str(repo),'inputs':verified,'only_binding_path_fields_changed':True,'scientific_policy_or_SHA_or_size_changed':False,'original_absolute_input_paths_followed':0,'wrapper_numeric_or_label_decoders':0,'input_reference_selector_query_GTID':False,'frozen_reference_bank_is_SOURCEFIT_supervised':True,'late_teacher_metadata_evaluation_only':True,'new_training_or_model_forward':0,'current_or_held_input':0})
    results=guarded(out/'results')
    command=runner_command(runner,out/'ORIGINAL_PROTOCOL.json',Path(relocated_binding['path']),results)
    stdout=out/'RUNNER_STDOUT.txt';stderr=out/'RUNNER_STDERR.txt'
    with guarded(stdout).open('xb') as so,guarded(stderr).open('xb') as se:
        proc=subprocess.run(command,stdout=so,stderr=se,check=False)
    execution=write_json(out/'RUNNER_EXECUTION_RECEIPT.json',{'completed':proc.returncode==0,'PASS':proc.returncode==0,'exit_code':proc.returncode,'command':command,'runner_code':code_binding,'path_relocation':receipt,'stdout':binding(stdout),'stderr':binding(stderr),'automatic_retry':False})
    if proc.returncode:raise RuntimeError('Pinned runner failed; partial outputs preserved/no automatic retry')
    child_complete=guarded(results/'COMPLETE.json',True,True)
    child=json.loads(read_bytes(child_complete))
    if child.get('completed') is not True or child.get('PASS') is not True:raise ValueError('Pinned runner completion not PASS')
    complete=write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,'schema':'V250_PUBLIC_PATH_ONLY_PROTOCOL_REPLAY_V1','wrapper_code':binding(__file__),'original_protocol':source_binding,'relocated_protocol':relocated_binding,'path_relocation':receipt,'execution':execution,'runner_complete':binding(child_complete),'scientific_outputs_unmodified_by_wrapper':True,'optimizer_updates':0,'model_forward':0,'current_admission':False})
    print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-repo',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();replay(args.public_repo,args.output_dir)
