#!/usr/bin/env python3
"""Stdlib-only saved V235 alpha/teacher metadata verifier; no RGB/model replay."""
import argparse
import ast
import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

EXPECTED={'attempts':5406,'success':5367,'failed':39,'success_by_family':{0:4537,1:830}}
BIT_BYTES=tuple(bytes((value>>bit)&1 for bit in range(7,-1,-1)) for value in range(256))


class VerificationError(ValueError):
    pass


def require(condition,message):
    if not condition:raise VerificationError(message)


def lexical(path):
    value=os.fspath(path);fold=value.casefold()
    require(value and '\x00' not in value and '\\' not in value,'Valid POSIX argument path required')
    require('v112' not in fold and not any(p=='__pycache__' or p.endswith('.pyc') for p in fold.split('/')),'Forbidden path before filesystem access')
    require('..' not in value.split('/'),'Parent traversal rejected before filesystem access')
    absolute=os.path.abspath(value)
    require('v112' not in absolute.casefold(),'Forbidden absolute argument path before filesystem access')
    return absolute


def open_input(path):
    absolute=lexical(path)
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW;directory=os.open('/',flags)
    parts=absolute.split('/')[1:]
    try:
        for part in parts[:-1]:
            nxt=os.open(part,flags,dir_fd=directory);os.close(directory);directory=nxt
        fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode):os.close(fd);raise VerificationError('Regular argument file required')
        return os.fdopen(fd,'rb')
    finally:os.close(directory)


@contextlib.contextmanager
def text_input(path):
    lower=os.fspath(path).casefold()
    with open_input(path) as raw:
        with contextlib.ExitStack() as stack:
            stream=stack.enter_context(gzip.GzipFile(fileobj=raw,mode='rb')) if lower.endswith('.gz') else raw
            text=io.TextIOWrapper(stream,encoding='utf-8')
            try:yield text
            finally:text.detach()


def json_rows(path):
    require(os.fspath(path).casefold().endswith(('.jsonl','.gz')),'JSONL or JSONL.gz argument required')
    with text_input(path) as stream:
        for line in stream:
            if line.strip():
                row=json.loads(line);require(isinstance(row,dict),'JSONL objects required');yield row


def input_binding(path):
    digest=hashlib.sha256();size=0
    with open_input(path) as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);size+=len(block)
    return {'path':os.fspath(path),'sha256':digest.hexdigest(),'bytes':size}


def read_exact(stream,count):
    result=stream.read(count);require(len(result)==count,'Truncated NPY bytes');return result


def npy_header(stream,count):
    require(read_exact(stream,6)==b'\x93NUMPY','NPY magic')
    major,minor=read_exact(stream,2);require((major,minor) in [(1,0),(2,0),(3,0)],'Supported NPY v1/v2/v3')
    length=struct.unpack('<H' if major==1 else '<I',read_exact(stream,2 if major==1 else 4))[0]
    require(0<length<=1024*1024,'Bounded NPY header length')
    header=ast.literal_eval(read_exact(stream,length).decode('utf-8' if major==3 else 'latin1'))
    require(isinstance(header,dict) and set(header)=={'descr','fortran_order','shape'},'Plain NPY header dictionary')
    require(header['descr'] in ['|u1','<u1','>u1','u1'] and header['fortran_order'] is False,'C-order uint8 packed alpha')
    require(type(header['shape']) is tuple and all(type(x) is int for x in header['shape']) and header['shape']==(count,2,1152),'Exact [attempts,2,1152] NPY shape')
    offset=stream.tell();require(os.fstat(stream.fileno()).st_size==offset+count*2*1152,'Exact NPY payload bytecount/no trailing data')
    return {'version':[major,minor],'shape':list(header['shape']),'dtype':header['descr'],'data_offset':offset,'payload_bytes':count*2304}


def alpha_at(packed,x,y):
    require(type(x) is int and type(y) is int and 0<=x<96 and 0<=y<96,'Required teacher tap inside96')
    index=y*96+x
    return bool((packed[index//8]>>(7-index%8))&1)


def taps(q,L,r,nearest3):
    require(isinstance(q,list) and isinstance(r,list) and len(q)==len(r)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in q+r),'Finite two-coordinate q/ref metadata')
    require(type(L) in (int,float) and math.isfinite(L) and L>0,'Finite positive original L')
    if nearest3:
        rx,ry=(math.floor(float(v)+.5) for v in r)
        points=[(rx+dx,ry+dy) for dy in (-1,0,1) for dx in (-1,0,1)]
    else:points=[tuple(r)]
    result=[]
    for x,y in points:
        a=math.floor((x-q[0])/L*96.+47.5);b=math.floor((y-q[1])/L*96.+47.5)
        result.extend([(a,b),(a+1,b),(a,b+1),(a+1,b+1)])
    require(all(0<=x<96 and 0<=y<96 for x,y in result),'Continuous/reference-nearest3x3 support inside96')
    return result


def verify(metadata,alpha,parent_diagnostics,ycb_reference=None,*,expected=None):
    counts=EXPECTED if expected is None else expected
    for path in [metadata,alpha,parent_diagnostics]+([] if ycb_reference is None else [ycb_reference]):lexical(path)
    parents={}
    for row in json_rows(parent_diagnostics):
        index=row['sample_index'];require(type(index) is int and index not in parents,'Unique parent diagnostic sample_index');parents[index]=row
    eligible={i for i,d in parents.items() if d['fit'] is True and d['available'] is True and d['target']==0}
    require(len(eligible)==counts['attempts'],'Complete original FIT-D parent population')
    ycb={}
    if ycb_reference is not None:
        with text_input(ycb_reference) as stream:
            for row in csv.DictReader(stream):
                index=int(row['sample_index']);require(index not in ycb,'Unique YCB reference sample_index');ycb[index]=row
    seen=set();success_count=0;failed=0;family_success={0:0,1:0};family_attempt={0:0,1:0};coverage_checks=0;external_ycb=0;bool_hash_checks=0;row_count=0
    with open_input(alpha) as stream:
        header=npy_header(stream,counts['attempts'])
        for j,row in enumerate(json_rows(metadata)):
            require(j<counts['attempts'] and row['pair_index']==j,'Every pair_index exactly0..N-1 in alpha row order')
            pair=read_exact(stream,2304);packed=[pair[:1152],pair[1152:]]
            index=row['parent_sample_index'];require(type(index) is int and index in eligible and index not in seen,'Unique exact original FIT-D parent');seen.add(index);d=parents[index]
            family=row['family'];require(type(family) is int and family in (0,1) and family==d['family_code'],'Original family identity')
            require(row['row_id']==d['row_id'] and row['original_role']=='FIT' and row['base_original_state']=='D' and d['original_state']=='D','Original parent identity/FIT/D authority')
            require(row['original_query_xy']==d['query_xy'] and row['original_L']==d['L'],'Original q/L unchanged')
            require(row['original_pixel_sha256']==d['pixel_sha256'] and row['original_valid_sha256']==d['valid_sha256'],'Original pixel/valid SHA strings unchanged (not pixel replay)')
            q=row['original_query_xy'];L=row['original_L'];r=row['teacher_reference_xy'];family_attempt[family]+=1
            if family==0:require(r==d['reference_xy_GT_diagnostic_only'],'Saved SYN reference exact parent diagnostic match')
            elif ycb_reference is not None:
                require(index in ycb,'YCB reference row present');y=ycb[index]
                require(y['row_id']==d['row_id'] and y['role']=='FIT' and y['associated_r_state']=='D','YCB external reference identity/authority')
                require([float(y['q_x']),float(y['q_y'])]==q and float(y['L'])==L and [float(y['r_x_GT_diagnostic_only']),float(y['r_y_GT_diagnostic_only'])]==r,'YCB exact original q/L/associated-r match');external_ycb+=1
            require(type(row['success']) is bool and type(row['masked']) is bool,'Boolean construction status')
            if row['success']:
                require(row['masked'] is False and row['proposed_states']=={'COVER':'E','SHAM':'D'},'Successful conditional associated-point E/D states')
                for nearest in (False,True):
                    support=taps(q,L,r,nearest)
                    require(all(alpha_at(packed[0],x,y) for x,y in support),'COVER continuous/ref-nearest3x3 taps all opaque')
                    require(all(not alpha_at(packed[1],x,y) for x,y in support),'SHAM continuous/ref-nearest3x3 taps all clear');coverage_checks+=1
                require(sum(v.bit_count() for v in packed[1])>=64,'SHAM visible alpha at least64')
                for k,role in enumerate(['COVER','SHAM']):
                    require(sum(v.bit_count() for v in packed[k])==row['teacherchecks']['roles'][role]['alpha_visible_pixels'],'Saved alpha pixel count exact packed-bit match')
                    booleans=b''.join(BIT_BYTES[value] for value in packed[k])
                    require(hashlib.sha256(booleans).hexdigest()==row['pair_alpha_sha256_bool'][k],'Packed MSB alpha decodes to original bool-byte SHA');bool_hash_checks+=1
                success_count+=1;family_success[family]+=1
            else:
                require(row['masked'] is True and row['proposed_states'] is None and bool(row['failure_reason']),'Failed pair remains masked/no E fallback')
                require(not any(pair),'Failed pair alpha allzero; original D retained');failed+=1
            row_count+=1
        require(row_count==counts['attempts'] and seen==eligible and stream.read(1)==b'','Full metadata/alpha/parent conservation')
    require(success_count==counts['success'] and failed==counts['failed'] and family_success==counts['success_by_family'],'Exact fixed construction counts')
    return {'completed':True,'PASS':True,'metadata_rows':row_count,'successful':success_count,'failed_masked_original_D_retained':failed,'success_by_family':family_success,'attempts_by_family':family_attempt,'NPY':header,'continuous_and_half_up_nearest3x3_coverage_checks':coverage_checks,'bool_alpha_SHA_checks':bool_hash_checks,'YCB_external_reference_rows_matched':external_ycb,'YCB_external_reference_supplied':ycb_reference is not None,'inputs':{k:input_binding(p) for k,p in [('metadata',metadata),('alpha',alpha),('parent_diagnostics',parent_diagnostics)]+([] if ycb_reference is None else [('ycb_reference',ycb_reference)])},'limits':['Only saved packed layer support and teacher/input metadata are checked, not raw RGB, original validity bitmap, donor pixels/masks/depth, model or natural occlusion.','Original pixel/valid SHA fields are compared as metadata strings; their underlying source bytes and donor SHA are not rehashed.','YCB reference is checked against the supplied CSV only when provided; otherwise its saved pair reference remains the teacher-side authority.','No native physical gold, target relabeling, model/data/threshold choice or new training.','Metadata absolute pointer strings are never followed; only explicit CLI argument files are opened.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',required=True);p.add_argument('--alpha',required=True);p.add_argument('--parent-diagnostics',required=True);p.add_argument('--ycb-reference')
    a=p.parse_args()
    try:print(json.dumps(verify(a.metadata,a.alpha,a.parent_diagnostics,a.ycb_reference),ensure_ascii=False));return 0
    except (ValueError,OSError,KeyError,TypeError,OverflowError,EOFError,SyntaxError,struct.error) as error:print(json.dumps({'completed':False,'PASS':False,'error':str(error)},ensure_ascii=False),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
