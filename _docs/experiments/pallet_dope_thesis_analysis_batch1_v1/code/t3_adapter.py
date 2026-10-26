"""Shared, lazy T3/T4 metadata/geometry helpers. Import opens no data/model files.

Source functions are imported unchanged (geometry), or AST-extracted unchanged
(registry hypotheses and hypothesis ranking). No training/finalizer main is run.
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path
import numpy as np
from common import OUT, FLIP, NOFLIP, REPO, R0, KINDS, read, sha, kind_of, load_module

MODEL_SOURCE = REPO/'Deep_Object_Pose/common/models.py'
ADAPTER_SOURCE = REPO/'scripts/research/pallet_dope_refiner_20261001_v1/dope_adapter.py'
GEOMETRY_INPUTS = OUT/'cache/DEV_GEOMETRY_INPUTS.json'
EXPECTED_COUNTS = {'PLASTIC':128, 'WOOD':45, 'GREEN':119}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def execution_gates():
    before=OUT/'hashes/SNAPSHOT_BEFORE.json'
    initial=OUT/'preflight/INITIAL_GATES.json'
    require(before.is_file(),'Root before-snapshot is required before execution')
    require(initial.is_file(),'Root initial gates are required before execution')
    require(read(initial).get('initial_file_count_gate_pass') is True,'Initial file/count gate has not passed')
    require(GEOMETRY_INPUTS.is_file(),'Root native-intrinsics metadata cache is required')
    return {str(p):sha(p) for p in (before,initial,GEOMETRY_INPUTS)}

def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    return value

def ast_functions(path, names):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names), f'Missing original functions: {path}/{names}')
    space={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
    return {name:space[name] for name in names}

def geometry_modules():
    result={}
    for mode,folder in (('main',FLIP),('noflip',NOFLIP)):
        protocol=read(folder/'GREEN/PROTOCOL.json')
        for source in (folder/'geometry.py',folder/'run_type.py',MODEL_SOURCE,ADAPTER_SOURCE):
            require(sha(source)==protocol['sources'][str(source)], f'Frozen source changed: {source}')
        for kind in KINDS:
            other=read(folder/kind/'PROTOCOL.json')
            require(other['filter']==protocol['filter'], f'Filter differs across kinds: {mode}/{kind}')
        module=load_module(folder/'geometry.py','batch1_original_geometry_'+mode)
        functions=ast_functions(folder/'run_type.py',('cuboid','hypotheses','choose'))
        result[mode]=(module,module.GeometryConfig(**protocol['filter']),functions)
    require(result['noflip'][1].use_flip is False,'Original ablation must remain no-flip')
    return result

def load_geometry_inputs():
    require(GEOMETRY_INPUTS.is_file(),f'Await root-created intrinsics/registry metadata cache: {GEOMETRY_INPUTS}')
    payload=read(GEOMETRY_INPUTS)
    frames=read(FLIP/'MANIFEST.json')['records']
    require(len(frames)==292 and len({r['id'] for r in frames})==292,'DEV population differs')
    records=payload['records']
    require(len(records)==292 and len({r['id'] for r in records})==292,'Geometry cache population differs')
    mapping={r['id']:r for r in records}
    require(set(mapping)=={r['id'] for r in frames},'Geometry cache IDs differ from DEV292')
    counts={k:0 for k in KINDS}
    registry=ast_functions(FLIP/'run_type.py',('cuboid','hypotheses'))
    out=[]
    for frame in frames:
        item=dict(mapping[frame['id']]);kind=kind_of(frame);counts[kind]+=1
        require(kind_of(item)==kind,'Geometry cache kind differs')
        for name in ('width','height'):
            require(item[name]==frame[name],f'Geometry cache {name} differs')
        if 'image_sha256' in item:
            require(item['image_sha256']==frame['image_sha256'],'Geometry cache image binding differs')
        K=np.asarray(item['K'],float)
        require(K.shape==(3,3) and np.isfinite(K).all(),'Invalid native K')
        if 'hypotheses' in item:
            hs=[(h['name'],np.asarray(h['xyz'],float)) for h in item['hypotheses']]
        else:
            require('dimensions_xyz_m' in item,'Geometry cache requires hypotheses or registered dimensions_xyz_m')
            hs=registry['hypotheses'](item)
        require(hs and len({name for name,_ in hs})==len(hs),'Invalid registry hypothesis names')
        require(all(xyz.shape==(8,3) and np.isfinite(xyz).all() for _,xyz in hs),'Invalid fixed cuboid geometry')
        if 'dimensions_xyz_m' in item:
            expected=registry['hypotheses'](item)
            require(len(hs)==len(expected) and all(a[0]==b[0] and np.array_equal(a[1],b[1]) for a,b in zip(hs,expected)),
                    'Geometry hypotheses/order differ from original registry function')
        out.append(dict(frame=frame,kind=kind,K=K,hypotheses=hs))
    require(counts==EXPECTED_COUNTS,'DEV per-kind population differs')
    return out,payload

def teacher_jobs(geometry_records):
    jobs=[]
    base=read(FLIP/'GREEN/PROTOCOL.json')['base_checkpoint']
    require(Path(base['path'])==R0,'Unexpected R0 path')
    jobs.append(dict(model='R0_before_selftraining',role='R0',checkpoint=base,
                     records=geometry_records,protocol_path=str(FLIP/'GREEN/PROTOCOL.json')))
    for kind in KINDS:
        own=[r for r in geometry_records if r['kind']==kind]
        for role,folder,stem,tag in (('flip_R1',FLIP,'corrected_LOO_flip','LOO_flip'),
                                      ('loo_R1',NOFLIP,'corrected_LOO_only','LOO_only')):
            path=folder/kind/f'{stem}_R1_TRAIN_COMPLETE.json';receipt=read(path)
            cp=dict(path=receipt['checkpoint_path'],sha256=receipt['checkpoint_sha256'])
            require(Path(cp['path'])==folder/kind/f'{stem}_R1.pth','Teacher checkpoint path differs')
            require(receipt['complete'] is True and receipt['round']==1 and receipt['kind']==kind,'Incomplete/wrong R1 teacher')
            require(receipt['teacher_checkpoint_sha256']==base['sha256'],'R1 teacher did not start at R0')
            jobs.append(dict(model=f'{kind}_{tag}_R1',role=role,checkpoint=cp,records=own,
                             protocol_path=str(folder/kind/'PROTOCOL.json'),receipt_path=str(path)))
    require(sum(len(j['records']) for j in jobs)==876,'Expected exactly 876 teacher-frame pairs')
    return jobs

def source_bindings():
    paths=[MODEL_SOURCE,ADAPTER_SOURCE,FLIP/'geometry.py',NOFLIP/'geometry.py',
           FLIP/'run_type.py',NOFLIP/'run_type.py',Path(__file__),OUT/'code/common.py',
           OUT/'code/t3_filter_diagnostics.py']
    return {str(p):sha(p) for p in paths}

def assert_bindings(bindings):
    for path,expected in bindings.items():
        require(sha(path)==expected,f'Changed binding: {path}')

def native_predictions(row, frame):
    p=np.asarray(row['points'],float);v=np.asarray(row['valid'],bool);c=np.asarray(row['confidence'],float)
    require(p.shape==(9,2) and v.shape==c.shape==(9,),'Expected nine prediction slots')
    v=v & np.isfinite(p).all(1) & (p[:,0]>=0) & (p[:,0]<frame['width']) & (p[:,1]>=0) & (p[:,1]<frame['height'])
    return p,v,c
