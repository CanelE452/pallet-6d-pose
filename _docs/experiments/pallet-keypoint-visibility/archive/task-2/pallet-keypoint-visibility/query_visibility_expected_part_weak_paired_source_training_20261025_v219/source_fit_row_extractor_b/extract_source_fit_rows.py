"""Root-entry-only SOURCE FIT row extraction; unselected row bytes stay opaque."""
import argparse
import ast
import hashlib
import io
import json
import os
import stat
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

H = Path(__file__).parent
PHASE = H.parent
OUT = PHASE/'source_fit_rows_v1'
PROTOCOL = H/'PROTOCOL_BEFORE_MIXED_SOURCE_NPZ_OR_SELECTED_ROW_VALUES.json'
CONTRACT = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V219_TRAINING_PREPARATION_C/source_training_contract.py'
CONTRACT_SHA = 'fc0a161e8a4c1a413d528ce1e66eed1b2addf2d4593c632506280f10732451d7'
DOMAINS = ('SYNTH_AUG65536', 'PUBLIC_TRAIN1480', 'CF3816')
META6 = ('row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available')
INPUT8 = ('rgb96', 'valid96')+META6
TARGET5 = ('support_target', 'owner_soft', 'owner_target', 'owner_ambiguous_mass', 'owner_invalid_mass')

def need(v, message):
    if not bool(v):
        raise ValueError(message)

def safe_open(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower(), 'literal absolute non-V112 path before filesystem')
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        leaf = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    f = os.fdopen(leaf, 'rb')
    need(stat.S_ISREG(os.fstat(f.fileno()).st_mode), 'regular file only')
    return f

def binding(path):
    with safe_open(path) as f:
        b = f.read()
    return dict(path=str(path), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))

def validate_binding(b):
    need(isinstance(b, dict) and set(b) in ({'path','sha256'}, {'path','sha256','size_bytes'}), 'exact binding keys')
    need(isinstance(b['path'],str) and isinstance(b['sha256'],str) and len(b['sha256']) == 64 and all(c in '0123456789abcdef' for c in b['sha256']), 'path/SHA binding')
    need('size_bytes' not in b or (type(b['size_bytes']) is int and b['size_bytes'] >= 0), 'optional nonnegative integer size')

def same_binding(a, b):
    validate_binding(a); validate_binding(b)
    return a['path'] == b['path'] and a['sha256'] == b['sha256']

def allowed(b, whitelist):
    validate_binding(b)
    need(any(same_binding(b, q) for q in whitelist), 'exact whitelist before source filesystem access')

def checked(b, whitelist):
    allowed(b, whitelist)
    with safe_open(b['path']) as f:
        data = f.read()
    need(hashlib.sha256(data).hexdigest() == b['sha256'] and ('size_bytes' not in b or len(data) == b['size_bytes']), 'bound metadata/code bytes')
    return data

def metadata(b, whitelist):
    need(Path(b['path']).suffix in ('.json','.jsonl'), 'text metadata only')
    data = checked(b, whitelist).decode('utf-8')
    return [json.loads(line) for line in data.splitlines() if line.strip()] if b['path'].endswith('.jsonl') else json.loads(data)

def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=2); f.write('\n')
    return binding(path)

def fresh_directory(path):
    p=Path(path)
    need(p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower(),'literal fresh directory before filesystem')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        os.mkdir(p.name,dir_fd=fd)
    finally:os.close(fd)

def opaque_verified_open(b, whitelist):
    allowed(b, whitelist)
    need(Path(b['path']).suffix == '.npz', 'only the six root-declared SOURCE NPZ containers')
    f = safe_open(b['path'])
    try:
        before = os.fstat(f.fileno())
        h = hashlib.sha256(); count = 0
        while True:
            block = f.read(1024*1024)
            if not block: break
            h.update(block); count += len(block)
        need(h.hexdigest() == b['sha256'] and ('size_bytes' not in b or count == b['size_bytes']), 'opaque source SHA/size before ZIP header or member access')
        f.seek(0)
        return f, (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns), dict(binding=b,opaque_bytes_hashed=count)
    except BaseException:
        f.close(); raise

def read_exact(stream, size):
    parts=[]; left=size
    while left:
        block=stream.read(min(left,1024*1024))
        need(bool(block), 'truncated selected row bytes')
        parts.append(block); left-=len(block)
    return b''.join(parts)

def discard_exact(stream, size):
    left=size
    while left:
        block=stream.read(min(left,1024*1024))
        need(bool(block), 'truncated opaque skipped bytes')
        left-=len(block)
    return size

def row_schema(key, shape, dtype, n):
    need(len(shape)>=1 and shape[0]==n and not dtype.hasobject and dtype.fields is None, 'non-object nonstructured full row schema '+key)
    tails={'rgb96':(96,96,3),'valid96':(96,96),'row_id':(),'original_global_index':(),'native_slot':(),'query_xy':(2,),'L':(),'original_query_available':(),
           'support_target':(2,),'owner_soft':(3,),'owner_target':(),'owner_ambiguous_mass':(),'owner_invalid_mass':()}
    need(tuple(shape[1:])==tails[key], 'exact trailing shape '+key)
    if key=='rgb96': ok=dtype==np.dtype('uint8')
    elif key in ('valid96','original_query_available'): ok=dtype==np.dtype('bool')
    elif key in ('support_target','owner_target'): ok=dtype==np.dtype('int8')
    elif key in ('query_xy','L','owner_soft','owner_ambiguous_mass','owner_invalid_mass'): ok=dtype==np.dtype('float64')
    elif key=='row_id': ok=dtype.kind=='U'
    else: ok=dtype.kind in 'iu'
    need(ok, 'exact source dtype '+key)

def selected_member(z, key, n, indices):
    """Only selected row bytes reach frombuffer; skipped decompressed bytes stay opaque."""
    need(all(type(i) is int for i in indices) and indices==sorted(set(indices)) and all(0<=i<n for i in indices), 'sorted unique source indices')
    infos=[i for i in z.infolist() if i.filename==key+'.npy']
    need(len(infos)==1, 'exact unique requested NPY member '+key)
    info=infos[0]
    need(not(info.flag_bits & 1) and not stat.S_ISLNK(info.external_attr>>16), 'unencrypted nonsymlink requested member')
    with z.open(info,'r') as stream:
        version=np.lib.format.read_magic(stream)
        need(version in ((1,0),(2,0),(3,0)), 'supported NPY header version')
        if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else: shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
        dtype=np.dtype(dtype)
        need(not fortran and not dtype.hasobject, 'C-contiguous non-object header before row value interpretation')
        row_schema(key,shape,dtype,n)
        row_bytes=int(np.prod(shape[1:],dtype=np.int64))*dtype.itemsize
        need(row_bytes>0 and info.file_size==stream.tell()+n*row_bytes, 'exact NPY header and row-byte extent')
        result=np.empty((len(indices),)+tuple(shape[1:]),dtype=dtype,order='C')
        cursor=0; pos=0; skipped=0; ranges=[]
        while pos<len(indices):
            lo=indices[pos]; end=pos+1
            while end<len(indices) and end-pos<64 and indices[end]==indices[end-1]+1: end+=1
            hi=indices[end-1]+1
            skipped+=discard_exact(stream,(lo-cursor)*row_bytes)
            data=read_exact(stream,(hi-lo)*row_bytes)
            # This is the sole numerical decoding site. data contains selected rows only.
            values=np.frombuffer(data,dtype=dtype).reshape((hi-lo,)+tuple(shape[1:]))
            result[pos:end]=values
            ranges.append([lo,hi]); cursor=hi; pos=end
        skipped+=discard_exact(stream,(n-cursor)*row_bytes)
        need(stream.read(1)==b'', 'exact member end and ZIP CRC completion')
    return result,dict(member=key+'.npy',full_rows=n,selected_rows=len(indices),numerically_decoded_source_ranges=ranges,
                       selected_value_bytes=len(indices)*row_bytes,opaque_unselected_decompressed_bytes_discarded=skipped,unselected_numeric_rows_decoded=0,
                       dtype=dtype.str,shape=list(shape),fortran_order=False)

def meta_equal(key, original, extracted):
    if key in ('query_xy','L'):
        missing=[np.nan,np.nan] if key=='query_xy' else np.nan
        return np.array_equal(np.asarray(missing if original is None else original,np.float64),np.asarray(extracted,np.float64),equal_nan=True)
    return np.array_equal(np.asarray(original),np.asarray(extracted))

def run(entry_path):
    with safe_open(entry_path) as f: e=json.load(f)
    need(e['approved_source_fit_row_extraction'] is True and e['source_only'] is True and e['automatic_retry'] is False, 'explicit actual root extraction entry first')
    need(e['output_directory']==str(OUT) and 'v112' not in str(OUT).lower(), 'exact fresh output')
    need(e['code']==binding(__file__) and e['protocol']==binding(PROTOCOL), 'frozen extractor/protocol')
    text_allow=e['metadata_whitelist']; binary=e['source_npz_whitelist']
    scope=metadata(e['root_scope'],text_allow)
    need(scope['approved_source_fit_row_extraction'] is True and scope['mixed_container_opaque_hash_headers_decompression_authorized'] is True and scope['unselected_numeric_decoding_authorized'] is False, 'exact mixed-container/selected-row authority')
    need(same_binding(e['training_contract'],dict(path=CONTRACT,sha256=CONTRACT_SHA)), 'frozen original fit-mask function')
    contract=checked(e['training_contract'],e['code_whitelist']).decode()
    nodes=[x for x in ast.parse(contract).body if isinstance(x,ast.FunctionDef) and x.name=='fit_mask']
    need(len(nodes)==1,'one exact pure fit-mask function')
    pure=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),'BOUND_FIT_MASK','exec'),pure)
    qa=metadata(e['lineage_assembly_QA'],text_allow)
    need(qa['completed'] is True and qa['PASS'] is True and qa['all_old_public_CF_descendants_of_new_DEV_excluded'] is True and qa['q8_input_provenance_verified_no_target_geometry_features'] is True,'A lineage/context QA')
    selection=metadata(e['source_selection'],text_allow)
    dev=set(e['new_DEV_component_ids']);need(len(dev)==4,'same fixed DEV4')
    records={};source_bindings=[]
    for d in DOMAINS:
        r=e['old_caches'][d];rr=metadata(r['metadata'],text_allow)
        need(isinstance(rr,list) and rr,'full source metadata in original cache order')
        indices=selection['domains'][d]['fit_indices']
        need(all(type(i) is int for i in indices),'explicit source-row index integers')
        expected=np.flatnonzero(pure['fit_mask'](rr,dev)).tolist()
        need(indices==expected and len(indices)>0,'exact A selection equals original TRAIN & nominal & lineage & nonDEV fit mask')
        records[d]=(r,rr,indices)
        source_bindings.extend((r['inputs'],r['targets']))
    need(len(binary)==6 and len({b['path'] for b in binary})==6 and len({b['path'] for b in source_bindings})==6 and all(any(same_binding(b,q) for q in binary) for b in source_bindings),'closed exact six source NPZ files')
    point=metadata(e['point_metadata7040'],text_allow)
    need(isinstance(point,list) and len(point)==7040 and len({r['row_id'] for r in point})==7040,'entire original point7040 unique metadata')
    sr,smeta,sindices=records[DOMAINS[0]];sselected=[smeta[i] for i in sindices]
    sb={r['row_id']:r for r in sselected}
    need(len(sb)==7040 and set(sb)=={r['row_id'] for r in point},'all point7040 equals filtered synthetic7040, including original U/UNKNOWN')
    for r in point:
        m=sb[r['row_id']]
        for k in META6:need(meta_equal(k,m[k],r[k]),'point7040 original META6 '+k)
        need(r['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']] and r['source_image_original_global_index']==m['source_image_original_global_index'] and r['anchor_original_global_index']==m['anchor_original_global_index'],'same own-image teacher/anchor lineage')
    # Close all six opaque SHA checks before opening any NPY member header.
    opened=[];proof=[]
    try:
        for b in source_bindings:
            f,identity,record=opaque_verified_open(b,binary);opened.append((b,f,identity));proof.append(record)
        fresh_directory(OUT)
        hashproof=write_json(OUT/'BEFORE_ANY_NPY_HEADER_OPAQUE_SIX_SOURCE_SHA_QA.json',dict(completed=True,PASS=True,entry=binding(entry_path),sources=proof,all_six_before_any_header=True))
        outputs={};census={};members=[]
        for di,d in enumerate(DOMAINS):
            original,rr,indices=records[d];loaded={}
            for off,(kind,keys) in enumerate((('inputs',INPUT8),('targets',TARGET5))):
                b,f,_=opened[2*di+off];need(same_binding(b,original[kind]),'source order')
                with zipfile.ZipFile(f,'r') as z:
                    arrays={}
                    for key in keys:
                        arrays[key],audit=selected_member(z,key,len(rr),indices);audit.update(domain=d,kind=kind);members.append(audit)
                loaded[kind]=arrays
            xx=loaded['inputs'];yy=loaded['targets'];selected=[]
            for j,i in enumerate(indices):
                m=rr[i]
                for k in META6:need(meta_equal(k,m[k],xx[k][j]),'selected array/metadata META6 '+d+'/'+k)
                need(type(m['native_slot']) is int and 0<=m['native_slot']<8 and type(m['original_query_available']) is bool,'original native/U types')
                m=dict(m);need('fit_extraction_source_row_index' not in m,'new provenance key cannot overwrite original metadata');m['fit_extraction_source_row_index']=i;selected.append(m)
            unavailable=~xx['original_query_available']
            need((yy['support_target'][unavailable]==-1).all() and (yy['owner_target'][unavailable]==-1).all(),'selected original U stays unknown')
            paths={}
            for kind in ('inputs','targets'):
                p=OUT/(d+'_'+kind.upper()+'.npz')
                with p.open('xb') as f:np.savez_compressed(f,**loaded[kind])
                paths[kind]=binding(p)
            mp=OUT/(d+'_FILTERED_METADATA.jsonl')
            with mp.open('x') as f:
                for r in selected:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            paths['metadata']=binding(mp);outputs[d]=paths
            census[d]=dict(original_rows=len(rr),selected_rows=len(indices),unselected_numeric_rows_decoded=0,selected_U=int(unavailable.sum()),selected_native_counts=dict(sorted(Counter(int(v) for v in xx['native_slot']).items())),original_roles_preserved=True,all_selected_fit_masks_true=True,original_input_binding=original['inputs'],original_target_binding=original['targets'],original_metadata_binding=original['metadata'])
            del loaded
        for b,f,identity in opened:
            s=os.fstat(f.fileno());need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==identity,'original open source inode/size/mtime unchanged')
        cp=write_json(OUT/'SELECTED_ROW_EXTRACTION_CENSUS.json',dict(completed=True,PASS=True,domains=census,members=members,unselected_rows_only_opaque_discarded=True,zero_unselected_byte_access_claim=False,selected_numpy_decode_only=True,source_GT_or_PNG_or_models_or_protected_access=0))
        complete=write_json(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,code=binding(__file__),protocol=binding(PROTOCOL),entry=binding(entry_path),root_scope=e['root_scope'],lineage_assembly_QA=e['lineage_assembly_QA'],source_selection=e['source_selection'],old_caches=outputs,point_metadata7040=e['point_metadata7040'],synthetic_selected7040_exact_point_identity=True,all_selected_masks_true=True,opaque_sources=hashproof,census=cp,unselected_numeric_rows_decoded=0,original_container_bytes_hashed_and_excluded_decompressed_bytes_may_be_traversed=True,protected_dataset_access=0,no_fit_or_forward=True))
        print(json.dumps(dict(completed=True,PASS=True,COMPLETE=complete,selected_rows={k:v['selected_rows'] for k,v in census.items()})))
    finally:
        for _,f,_ in opened:f.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
