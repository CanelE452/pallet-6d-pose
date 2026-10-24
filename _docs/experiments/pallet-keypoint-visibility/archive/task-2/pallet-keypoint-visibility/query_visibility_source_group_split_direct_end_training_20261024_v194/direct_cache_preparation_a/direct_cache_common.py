"""Explicit source-only I/O and exact pure inherited helpers; no work on import."""
from pathlib import Path
import ast, hashlib, importlib.util, json, math, os, stat, sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
V189=P/'query_visibility_sampled_domain_direct_end_20261024_v189'
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_COMPONENT_SPLIT_BEFORE_METADATA_CALCULATIONS.json'
SAM_ADDENDUM=V189/'ROOT_CHANGED_RGB_SAM_METADATA_ONLY_READINESS_ADDENDUM.json'
TRAIN_SCOPE=P/'query_visibility_direct_end_paired_source_training_20261024_v190/ROOT_PROSPECTIVE_DIRECT_END_PAIRED_SOURCE_SCOPE_BEFORE_CACHE_AND_FIT.json'
V181=P/'query_visibility_local_support_raw_head_20261024_v181'
V183=P/'query_visibility_query_jitter_paired_source_training_20261024_v183'
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
OLD_TRAINER=V183/'source_fit_preparation/train_jitter_paired_source.py'
OLD_TRAIN_PROTOCOL=OLD_TRAINER.with_name('PROTOCOL_BEFORE_AUGMENTED_CACHE_OR_FIT_BODIES.json')
HEAD=V181/'head_preparation/local_support_head.py'
CANDIDATE=V189/'direct_end_training_preparation_c/synthetic_direct_end_candidate.py'
PIXEL_HELPER=V181/'raw_cache_preparation/build_source_raw96_cache.py'
SAM_HELPER=V180/'runner_preparation/run_frozen_sam_source.py'
SAM_PROTOCOL=SAM_HELPER.with_name('PROTOCOL_BEFORE_WEIGHTS_OR_SOURCE_RGB_BODIES.json')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
INPUTS=('rgb96','valid96','sam_mask96','sam_available96')+META

def need(v,m):
    if not bool(v):raise AssertionError(m)

def safe(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and all('v112' not in q.casefold() for q in p.parts),'Explicit safe source path before access')
    need(p.is_relative_to(P),'Only exact source project paths')
    need(not any(x in str(p).casefold() for x in ('current223','_v169/','_v177/','_v178/','_v184/','_v191/','_v182/','synthetic_val')),'No real evaluation paths')
    return p

def fresh_output(path):
    p=safe(path);first=p.relative_to(P).parts[0]
    need(first==V.name,'New V194 output only; old phases immutable')
    need(p not in (P,P/first),'Use fresh child output directory')
    cur=Path(p.anchor)
    for part in p.parts[1:-1]:
        cur=cur/part
        try:mode=cur.lstat().st_mode
        except FileNotFoundError:cur.mkdir();mode=cur.lstat().st_mode
        need(stat.S_ISDIR(mode),'Ordinary output ancestors only')
    p.mkdir(exist_ok=False)
    return p

def ordinary_parent(p):
    cur=Path(p.anchor)
    for part in p.parts[1:-1]:
        cur=cur/part;need(stat.S_ISDIR(cur.lstat().st_mode),'No destination symlink ancestors')

def opened(path):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for q in p.parts[1:-1]:
            nxt=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular source file')
    return os.fdopen(last,'rb')

def sha(path):
    h=hashlib.sha256()
    with opened(path) as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(safe(p)),sha256=sha(p))
def checked(b):need(sha(b['path'])==b['sha256'],'Bound bytes unchanged');return safe(b['path'])
def read(p):
    with opened(p) as f:return json.load(f)
def read_bound(b):return read(checked(b))
def lines(b):
    with opened(checked(b)) as f:return [json.loads(s) for s in f if s.strip()]
def write(p,o):
    p=safe(p);ordinary_parent(p)
    with Path(p).open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rows):
    p=safe(p);ordinary_parent(p)
    with p.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def arrays(b,keys):
    # Root-entry only. Hash/open the same nofollow descriptor; deserialize only
    # explicitly named members, never an implicit full NPZ dictionary.
    with opened(b['path']) as f:
        h=hashlib.sha256()
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
        need(h.hexdigest()==b['sha256'],'Bound source NPZ SHA');f.seek(0)
        with np.load(f,allow_pickle=False) as z:return {k:z[k] for k in keys}
def save_arrays(p,a):
    p=safe(p);ordinary_parent(p)
    with p.open('xb') as f:np.savez_compressed(f,**a)
    return bind(p)
def functions(path,names,ns):
    with opened(path) as f:tree=ast.parse(f.read().decode())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    need({n.name for n in nodes}==set(names),'Exact pure AST helper names')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::BOUND_FUNCTIONS_ONLY','exec'),ns)
    return ns
def module(path,name):
    safe(path);s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sampled_helpers():return functions(PIXEL_HELPER,('qarray','same','validq','grid96','sample_sam'),dict(np=np,require=need))
def sam_helpers():
    import subprocess
    p=read(SAM_PROTOCOL);pin=read_bound(p['source_pin'])
    ns=dict(np=np,Path=Path,sys=sys,math=math,hashlib=hashlib,subprocess=subprocess,arrsha=ash,VENDOR=V180/'vendor/segment-anything',COMMIT=pin['exact_commit'],RUNTIME=p['runtime'])
    functions(SAM_HELPER,('prompt_box','predict_warn_only_call','actual_cuda_cumsum_guard','state_sha','import_runtime','git'),ns)
    return ns,p,pin
def validate_input10(x,n):
    need(set(x)==set(INPUTS),'Target-free exact INPUTS10')
    for k in ['rgb96','valid96','sam_mask96','sam_available96']:
        need(x[k].shape==((n,96,96,3) if k=='rgb96' else (n,96,96)),'Pixel shape '+k)
        need(x[k].dtype==(np.uint8 if k=='rgb96' else bool),'Pixel dtype '+k)
    for k,dtype,shape in [('original_global_index',np.int64,(n,)),('native_slot',np.int8,(n,)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('original_query_available',bool,(n,))]:need(x[k].dtype==dtype and x[k].shape==shape,'META dtype/shape '+k)
    need(x['row_id'].dtype.kind=='U' and x['row_id'].shape==(n,) and len(set(x['row_id'].tolist()))==n,'Unique occurrence IDs; repeated source global IDs allowed')
    av=x['original_query_available'];need(np.isfinite(x['query_xy'][av]).all() and np.isfinite(x['L'][av]).all() and (x['L'][av]>0).all(),'Available q/L finite')
    need(((x['native_slot']>=0)&(x['native_slot']<8)).all(),'All original native0..7')
    return x
