"""V249 complete CPU SOURCE replay from an explicit pinned public dependency map.

No Root absolute-path fallback, model, optimizer, RGB, GT source or network reader.
Installed NumPy is the only non-stdlib dependency. The diagnostic is teacher-oracle.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import types

MAP_SHA='95a379fb591e8ee25c356d47bc21e9dd2a99d49106cc8de1abc812d0c4ed51f5'
MAP_BYTES=6857
DIAGNOSTIC_FILENAME='diagnose_source_rgb_references_v2.py'
DIAGNOSTIC_SHA='01f0f81d1add4cada7791103b40fa8b0b3619ad3ce74ab825d3dfe0298a7b91e'
DIAGNOSTIC_BYTES=20380
INPUT_KEYS=('original_features','samples','YCB_metadata')
SAMPLE_KEYS=('target','family','fit','available','row_id')
COLS=('row_id','sample_index','role','scene','frame_id','object_id','aux_slot','associated_r_state','q_x','q_y','r_x_GT_diagnostic_only','r_y_GT_diagnostic_only','q_reference_distance_px','L')


def lexical(value):
    text=str(value)
    if any(t in text.casefold() for t in ('v112','cache','.pyc','current223','publicheld','public_held','oldval','oldhold','/hold/','v177','v178','source_holdout','source_val','sourceval','v182','private_evaluation')) or '\x00' in text:
        raise ValueError('Forbidden path before any filesystem access')
    if '..' in Path(text).parts:
        raise ValueError('Parent path component forbidden')
    return text


def guarded(value,existing=False,regular=False):
    text=lexical(value)
    path=Path(os.path.abspath(text))
    for part in reversed((path,*path.parents)):
        try: info=os.lstat(part)
        except FileNotFoundError:
            if existing: raise
            continue
        if stat.S_ISLNK(info.st_mode): raise ValueError('Symlink ancestor/artifact rejected')
    if existing:
        info=os.lstat(path)
        if regular and not stat.S_ISREG(info.st_mode): raise ValueError('Regular file required')
    return path


def read_bytes(path):
    path=guarded(path,existing=True,regular=True)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        return f.read()


def digest_file(path):
    path=guarded(path,existing=True,regular=True);sha=hashlib.sha256();size=0
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            sha.update(block);size+=len(block)
    return {'path':str(path),'sha256':sha.hexdigest(),'size_bytes':size}


def verify(path,sha,size):
    result=digest_file(path)
    if result['sha256']!=sha or result['size_bytes']!=size:
        raise ValueError('Exact pinned SHA/size mismatch: '+str(path))
    return result


def safe_repo_input(repo,record):
    relative=lexical(record['repository_relative_path'])
    if '\\' in relative:
        raise ValueError('POSIX repository-relative input path required')
    posix=PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or any(x in ('..','.') for x in posix.parts) or relative!=posix.as_posix():
        raise ValueError('Canonical safe relative path required')
    path=guarded(repo/Path(*posix.parts),existing=True,regular=True)
    if not path.is_relative_to(repo): raise ValueError('Input outside explicit public repo')
    # original_local_path is provenance only and is never followed.
    return path


def load_pinned_code(path):
    code=verify(path,DIAGNOSTIC_SHA,DIAGNOSTIC_BYTES)
    raw=read_bytes(path)
    if hashlib.sha256(raw).hexdigest()!=DIAGNOSTIC_SHA: raise ValueError('Diagnostic CODE changed after verification')
    module=types.ModuleType('v249_public_fixed_pure_diagnostic')
    module.__file__=str(path);sys.modules[module.__name__]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    if tuple(module.COLS)!=COLS or tuple(module.SAMPLE_KEYS)!=SAMPLE_KEYS:
        raise ValueError('Pinned pure API schema mismatch')
    if not callable(module.run_diagnostic) or not callable(module.write_results): raise ValueError('Required pure API missing')
    return module,code


def selected_metadata(path):
    """CSV transport is tokenized; only fourteen declared raw fields are exposed."""
    path=guarded(path,existing=True,regular=True)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'r',newline='',encoding='utf-8') as f:
        reader=csv.reader(f);header=next(reader)
        if len(header)!=len(set(header)) or any(k not in header for k in COLS): raise ValueError('Unique header and selected14 required')
        indices=[header.index(k) for k in COLS];rows=[]
        for raw in reader:
            if len(raw)!=len(header): raise ValueError('CSV field count mismatch')
            rows.append({k:raw[j] for k,j in zip(COLS,indices)})
    return rows


def write_json(path,obj):
    path=guarded(path)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return digest_file(path)


def replay(public_repo,output_directory):
    repo=guarded(public_repo,existing=True)
    if not repo.is_dir(): raise ValueError('--public-repo must be an existing directory')
    if not Path(lexical(output_directory)).is_absolute(): raise ValueError('--output-dir must be absolute')
    out=guarded(output_directory)
    if out.exists(): raise ValueError('Fresh output directory required; no automatic retry/overwrite')
    here=guarded(__file__,existing=True,regular=True).parent
    map_path=guarded(here/'dependency_map.json',existing=True,regular=True)
    map_binding=verify(map_path,MAP_SHA,MAP_BYTES);mapping=json.loads(read_bytes(map_path))
    if mapping.get('completed') is not True or mapping.get('PASS') is not True or set(mapping['by_key'])!=set(INPUT_KEYS):
        raise ValueError('Only fixed verified3SOURCE dependency map permitted')
    code_path=guarded(here.parent/'V249_RGB_REFERENCE_DIAGNOSTIC_B'/DIAGNOSTIC_FILENAME,existing=True,regular=True)
    # Verify CODE and ALL input bytes before any NPZ/NPY/CSV decoder or statistic.
    code_binding=verify(code_path,DIAGNOSTIC_SHA,DIAGNOSTIC_BYTES)
    paths={};verified={}
    for key in INPUT_KEYS:
        record=mapping['by_key'][key]
        path=safe_repo_input(repo,record)
        paths[key]=path;verified[key]=verify(path,record['sha256'],int(record['size_bytes']))
    module,code_binding=load_pinned_code(code_path)
    import numpy as np
    features=np.load(guarded(paths['original_features'],existing=True,regular=True),mmap_mode='r',allow_pickle=False)
    with np.load(guarded(paths['samples'],existing=True,regular=True),allow_pickle=False) as z:
        if any(k not in z.files for k in SAMPLE_KEYS): raise ValueError('Required5 nonpixel NPZ members absent')
        samples={k:z[k] for k in SAMPLE_KEYS}
    metadata=selected_metadata(paths['YCB_metadata'])
    result=module.run_diagnostic(metadata,features,samples)
    outputs=module.write_results(str(out),result)
    provenance=write_json(out/'PUBLIC_REPLAY_PROVENANCE.json',{
        'completed':True,'PASS':True,'mode':'PUBLIC_REPO_FIXED_SOURCE_CPU_ORACLE_REFERENCE_DIAGNOSTIC',
        'public_repo':str(repo),'input_source_commit':mapping['source_commit'],'dependency_map':map_binding,
        'diagnostic_code':code_binding,'wrapper_code':digest_file(__file__),'inputs':verified,
        'numpy_version':np.__version__,'selected_CSV_columns':list(COLS),'decoded_NPZ_members':list(SAMPLE_KEYS),
        'CSV_extra_fields':'CSV transport tokenization only; discarded before exposure/semantic interpretation',
        'statistic_provenance':'Exact pinned run_diagnostic API: equal-weight FIT-D descriptor means/FP64 sums+unitL2; deterministic wrong-object rule; fixed3 views/strata/directions. No thresholds or best view selected.',
        'reference_ID_is_teacher_oracle':True,'reference_retrieval_is_deployable':False,
        'SOURCE_DEV_exposed_reused_not_blind':True,'native_mapping':'UNKNOWN',
        'original_absolute_input_paths_followed':0,'original_RGB_raw_GT_weights_or_model_reads':0,'teacher_reference_scalar_metadata_selected_CSV':True,
        'optimizer_updates':0,'encoder_or_head_forward':0,'current_or_held_payloads':0,
        'actual_training':0,'automatic_retry':False,'outputs':outputs})
    complete=write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,'schema':'V249_PUBLIC_FULL_CPU_REPLAY_V1','dependency_map':map_binding,'diagnostic_code':code_binding,'wrapper_code':digest_file(__file__),'inputs':verified,'outputs':outputs,'provenance':provenance,'reference_identity_is_oracle':True,'optimizer_updates':0,'model_forward':0,'current_admission':False})
    print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-repo',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    replay(args.public_repo,args.output_dir)
