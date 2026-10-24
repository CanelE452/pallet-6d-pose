"""Three bounded invented tests. No operational source payload is opened."""
import ast
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

H=Path(__file__).parent
spec=importlib.util.spec_from_file_location('v219_row_extractor',H/'extract_source_fit_rows.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

def rejects(fn):
    try:fn()
    except (ValueError,OSError):return
    raise AssertionError('Expected rejection')

def archive(arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);b.seek(0);return b

def main():
    # Original roles, new DEV and missing lineage determine indices; values do not rank rows.
    contract=E.checked(dict(path=E.CONTRACT,sha256=E.CONTRACT_SHA),[dict(path=E.CONTRACT,sha256=E.CONTRACT_SHA)]).decode()
    node=next(n for n in ast.parse(contract).body if isinstance(n,ast.FunctionDef) and n.name=='fit_mask')
    pure={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'BOUND_FIT_MASK','exec'),pure)
    rr=[dict(component_role=role,nominal_fit_eligible=True,lineage_bound=True,component_id='a'*64) for role in ('HOLDOUT','TRAIN','TRAIN','TRAIN','TRAIN')]
    rr[2]['component_id']='d'*64;rr[3]['lineage_bound']=False
    indices=np.flatnonzero(pure['fit_mask'](rr,{'d'*64})).tolist()
    assert indices==[1,4]
    n=5
    rgb=np.stack([np.full((96,96,3),v,np.uint8) for v in (251,17,252,253,41)])
    arrays=dict(rgb96=rgb,valid96=np.ones((n,96,96),bool),row_id=np.asarray(['HOLD_SENTINEL','FIT_A','DEV_SENTINEL','NO_LINEAGE_SENTINEL','FIT_B']),original_global_index=np.arange(n,dtype=np.int64)+100,native_slot=np.arange(n,dtype=np.int64),query_xy=np.arange(n*2,dtype=np.float64).reshape(n,2),L=np.full(n,96.,np.float64),original_query_available=np.asarray([True,False,True,True,True]),support_target=np.full((n,2),-1,np.int8),owner_soft=np.full((n,3),np.nan,np.float64),owner_target=np.full(n,-1,np.int8),owner_ambiguous_mass=np.full(n,np.nan,np.float64),owner_invalid_mass=np.full(n,np.nan,np.float64))
    audits=[]
    with zipfile.ZipFile(archive(arrays),'r') as z:
        for key in E.INPUT8+E.TARGET5:
            selected,a=E.selected_member(z,key,n,indices)
            assert np.array_equal(selected,arrays[key][indices],equal_nan=True) if selected.dtype.kind in 'fc' else np.array_equal(selected,arrays[key][indices])
            assert a['numerically_decoded_source_ranges']==[[1,2],[4,5]] and a['unselected_numeric_rows_decoded']==0
            assert a['opaque_unselected_decompressed_bytes_discarded']==3*(a['selected_value_bytes']//2)
            audits.append(a)
    # Unselected sentinels never reached the only decoding site; q/L/U stay exact.
    assert np.array_equal(rgb[indices,0,0,0],[17,41])
    assert arrays['original_query_available'][indices].tolist()==[False,True]
    for key in E.META6:assert E.meta_equal(key,None if key in ('query_xy','L') else arrays[key][1],np.full(2,np.nan) if key=='query_xy' else np.nan if key=='L' else arrays[key][1])

    # Header-only rejection of object or Fortran members; no numerical conversion fallback.
    with zipfile.ZipFile(archive({'row_id':np.asarray([object()]*n,object)}),'r') as z:
        rejects(lambda:E.selected_member(z,'row_id',n,indices))
    with zipfile.ZipFile(archive({'query_xy':np.asfortranarray(np.zeros((n,2),np.float64))}),'r') as z:
        rejects(lambda:E.selected_member(z,'query_xy',n,indices))
    with zipfile.ZipFile(archive({'query_xy':np.zeros((n,2),np.float64)}),'r') as z:
        rejects(lambda:E.selected_member(z,'query_xy',n,[4,1]))
    src=(H/'extract_source_fit_rows.py').read_text()
    tree=ast.parse(src)
    assert not any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='np' and n.attr=='load' for n in ast.walk(tree))
    assert len([n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='np' and n.attr=='frombuffer'])==1

    # False entry fails before source whitelist access; lexical rejection precedes open.
    with tempfile.TemporaryDirectory(prefix='invented_gate_',dir=H) as tmp:
        p=Path(tmp)/'FALSE.json';p.write_text(json.dumps(dict(approved_source_fit_row_extraction=False,source_only=True,automatic_retry=False)))
        rejects(lambda:E.run(p))
        f=Path(tmp)/'ordinary.txt';f.write_text('invented');link=Path(tmp)/'link.txt';link.symlink_to(f)
        rejects(lambda:E.safe_open(link))
    original=E.os.open
    def forbidden_open(*a,**k):raise AssertionError('Filesystem touched before lexical rejection')
    E.os.open=forbidden_open
    try:rejects(lambda:E.safe_open('/tmp/forbidden_v112_case/file.npz'))
    finally:E.os.open=original
    result=dict(completed=True,PASS=True,groups3=['actual fit_mask and selected-only row decoding across INPUT8+TARGET5 with HOLD/DEV/missing-lineage sentinels','C-order/nonobject header rejection and single selected-only frombuffer site; no np.load','false-entry and nofollow/lexical-before-filesystem guards'],selected_original_indices=indices,kept_original_U=True,member_audits=audits,operational_NPZ_stat_hash_header_or_decode=0,unselected_operational_numeric_decode=0,model_GT_protected_access=0)
    out=H/'GENERATED_INVENTED_SELECTED_ROW_STREAM_CHECKS.json'
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,groups=3,checks=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))

if __name__=='__main__':main()
