"""Exact Root entry I/O。payloadはseparateactual承認後のみ。"""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import types
import zipfile
import numpy as np

def need(ok,message):
    if not ok:raise ValueError(message)

def lexical(value):
    s=str(value);p=Path(s)
    need(p.is_absolute() and '..' not in p.parts and not any(x in s.casefold() for x in
        ('v112','__pycache__','.pyc','current223','publicheld','oldval','source_holdout',
         'private_evaluation','v177','v178','v182')), 'forbidden lexical path before FS')
    return p

def safe(value):
    p=lexical(value)
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'parent symlink/non-directory')
    if p.exists() or p.is_symlink():need(not stat.S_ISLNK(os.lstat(p).st_mode),'leaf symlink')
    return p

def identity(b):return str(lexical(b['path'])),b['sha256']

def raw(path):
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def verify(b,white):
    need(identity(b) in {identity(x) for x in white},'outside exact whitelist')
    h=hashlib.sha256();n=0
    with os.fdopen(os.open(safe(b['path']),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x);n+=len(x)
    need(h.hexdigest()==b['sha256'] and ('size_bytes' not in b or n==b['size_bytes']),'bound bytes changed')
    return safe(b['path'])

def binding(path):
    b=raw(path);return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def read_json(b,white):return json.loads(raw(verify(b,white)))
def read_jsonl(b,white):return [json.loads(x) for x in raw(verify(b,white)).decode().splitlines()]

def iter_jsonl(b,white):
    p=verify(b,white)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'r') as f:
        for line in f:yield json.loads(line)

def map_npy(b,white,dtype,shape):
    p=verify(b,white)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        version=np.lib.format.read_magic(f)
        need(version in ((1,0),(2,0)),'numeric NPY v1/v2 only')
        header_reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        actual,fortran,dt=header_reader(f)
        need(actual==shape and not fortran and dt==np.dtype(dtype) and not dt.hasobject,'exact NPY header')
        offset=f.tell();need(offset+int(np.prod(shape))*dt.itemsize==os.fstat(f.fileno()).st_size,'NPY exact bytes')
        return np.memmap(f,dtype=dt,mode='r',offset=offset,shape=shape,order='C')

def write_json(path,obj):
    with os.fdopen(os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return binding(path)

def pinned_math(b,white,names,pin):
    need(b['sha256']==pin,'pure sampler SHA pin')
    source=raw(verify(b,white));functions=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name in names]
    need({x.name for x in functions}==set(names),'exact pure sampler functions')
    module=types.SimpleNamespace();env={'np':np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),b['path'],'exec'),env)
    for name in names:setattr(module,name,env[name])
    return module

class LRU:
    def __init__(self,maxbytes):
        from collections import OrderedDict
        self.items=OrderedDict();self.nbytes=0;self.cap=maxbytes;self.peak=0
    def size(self,value):
        if isinstance(value,np.ndarray):return value.nbytes
        if isinstance(value,dict):return sum(self.size(x) for x in value.values())
        return 0
    def get(self,key,load):
        if key in self.items:
            value=self.items.pop(key);self.items[key]=value;return value
        value=load();size=self.size(value)
        if size>self.cap:return value
        while self.items and self.nbytes+size>self.cap:
            _,old=self.items.popitem(last=False);self.nbytes-=self.size(old)
        self.items[key]=value;self.nbytes+=size;self.peak=max(self.peak,self.nbytes)
        return value

def decode_base(b,white):
    need(identity(b) in {identity(x) for x in white},'base804 whitelist before FS')
    encoded=raw(b['path']);need(hashlib.sha256(encoded).hexdigest()==b['sha256'],'base encoded SHA before decode')
    if b['format']=='npz':
        need(b.get('member')=='RGB_uint8','only RGB_uint8 member; no member enumeration')
        with zipfile.ZipFile(io.BytesIO(encoded)) as z:
            with z.open('RGB_uint8.npy') as f:member=f.read()
        rgb=np.load(io.BytesIO(member),allow_pickle=False)
    elif b['format']=='png':
        need(b.get('member') is None,'PNG no member')
        from PIL import Image
        with Image.open(io.BytesIO(encoded)) as image:rgb=np.asarray(image.convert('RGB')).copy()
    else:raise ValueError('Only existing base PNG/NPZ RGB_uint8')
    need(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3,'base native RGB U8')
    return rgb
