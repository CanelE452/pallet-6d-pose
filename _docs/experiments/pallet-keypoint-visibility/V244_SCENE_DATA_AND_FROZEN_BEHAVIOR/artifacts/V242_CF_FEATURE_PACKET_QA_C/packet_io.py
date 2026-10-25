"""정확 entry의 nonpixel packet만 읽는 I/O. 모델/영상 decoder 없음."""
import hashlib,json,os,stat
from pathlib import Path
import numpy as np

def need(ok,message):
    if not ok:raise ValueError(message)

def lexical(value):
    s=str(value);p=Path(s)
    need(p.is_absolute() and '..' not in p.parts and not any(x in s.casefold() for x in
        ('v112','__pycache__','.pyc','current223','publicheld','oldval','source_holdout',
         'private_evaluation','v177','v178','v182')),'forbidden lexical path before FS')
    return p

def safe(value):
    p=lexical(value)
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'parent symlink/non-directory')
    try:need(stat.S_ISREG(os.lstat(p).st_mode) or stat.S_ISDIR(os.lstat(p).st_mode),'leaf symlink/nonregular')
    except FileNotFoundError:pass
    return p

def identity(b):
    need(isinstance(b,dict) and {'path','sha256'}<=set(b),'binding fields')
    sha=b['sha256'];need(isinstance(sha,str) and len(sha)==64 and set(sha)<=set('0123456789abcdef'),'SHA format')
    return str(lexical(b['path'])),sha

def raw(path):
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def binding(path):
    h=hashlib.sha256();n=0
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x);n+=len(x)
    return dict(path=str(path),sha256=h.hexdigest(),size_bytes=n)

def verify(b,white):
    need(identity(b) in {identity(x) for x in white},'closed whitelist before FS')
    a=binding(b['path']);need(identity(a)==identity(b) and ('size_bytes' not in b or b['size_bytes']==a['size_bytes']),'bound bytes changed')
    return safe(b['path'])

def load_json(b,white):return json.loads(raw(verify(b,white)))
def load_rows(b,white):return [json.loads(x) for x in raw(verify(b,white)).decode().splitlines() if x.strip()]

def map_npy(b,white,dtype,shape):
    with os.fdopen(os.open(verify(b,white),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        version=np.lib.format.read_magic(f);need(version in ((1,0),(2,0)),'NPY version1/2')
        read_header=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        sh,fortran,dt=read_header(f)
        need(sh==shape and not fortran and dt==np.dtype(dtype) and not dt.hasobject,'exact C-order numeric NPY header')
        offset=f.tell();need(offset+int(np.prod(shape))*dt.itemsize==os.fstat(f.fileno()).st_size,'NPY exact length')
        return np.memmap(f,dtype=dt,mode='r',offset=offset,shape=shape,order='C')

def load_npz(b,white,keys):
    with os.fdopen(os.open(verify(b,white),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f,np.load(f,allow_pickle=False) as z:
        need(set(z.files)==set(keys),'exact nonpixel NPZ members')
        return {k:z[k] for k in keys}

def write_json(path,value):
    with os.fdopen(os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return binding(path)
