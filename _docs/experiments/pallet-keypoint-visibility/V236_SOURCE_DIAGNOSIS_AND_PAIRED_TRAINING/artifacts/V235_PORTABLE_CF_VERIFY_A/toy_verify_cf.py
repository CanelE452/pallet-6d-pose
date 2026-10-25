"""Stdlib invented fixtures only, including manually encoded NPY headers."""
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import verify_cf_teacher as checker


def packed(points):
    a=bytearray(1152)
    for x,y in points:
        index=y*96+x;a[index//8]|=1<<(7-index%8)
    return bytes(a)


def npy(path,payload,count=2,version=1,fortran=False):
    header=repr({'descr':'|u1','fortran_order':fortran,'shape':(count,2,1152)}).encode('ascii')
    prefix_size=10 if version==1 else 12
    header+=b' '*((-prefix_size-len(header)-1)%16)+b'\n'
    path.write_bytes(b'\x93NUMPY'+bytes([version,0])+struct.pack('<H' if version==1 else '<I',len(header))+header+payload)


def run():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='v235_cf_verify_toy_') as temporary:
        root=Path(temporary);q=[100.,100.];r=[100.,100.];L=96.
        cover=packed({(x,y) for y in range(46,50) for x in range(46,50)})
        sham=packed({(x,y) for y in range(8) for x in range(8)});zeros=bytes(1152);payload=cover+sham+zeros+zeros
        parents=[];rows=[]
        for index,family in [(0,0),(1,1)]:
            d={'sample_index':index,'fit':True,'available':True,'target':0,'family_code':family,'row_id':'invented'+str(index),'original_state':'D','query_xy':q,'L':L,'pixel_sha256':str(index)*64,'valid_sha256':'a'*64,'reference_xy_GT_diagnostic_only':r if family==0 else None}
            parents.append(d)
            row={'pair_index':index,'parent_sample_index':index,'family':family,'row_id':d['row_id'],'original_role':'FIT','base_original_state':'D','original_query_xy':q,'original_L':L,'original_pixel_sha256':d['pixel_sha256'],'original_valid_sha256':d['valid_sha256'],'teacher_reference_xy':r,'success':index==0,'masked':index!=0,'proposed_states':{'COVER':'E','SHAM':'D'} if index==0 else None,'failure_reason':None if index==0 else 'INVENTED_CONSTRUCTION_FAILURE','donor_RGB':{'path':'/never/v112/never-open.png','sha256':'b'*64}}
            if index==0:
                row['teacherchecks']={'roles':{'COVER':{'alpha_visible_pixels':16},'SHAM':{'alpha_visible_pixels':64}}}
                row['pair_alpha_sha256_bool']=[hashlib.sha256(b''.join(checker.BIT_BYTES[v] for v in a)).hexdigest() for a in [cover,sham]]
            rows.append(row)
        metadata=root/'metadata.jsonl.gz';diagnostics=root/'parents.jsonl';alpha=root/'alpha.npy';reference=root/'reference.csv'
        def write_rows(changed=rows):
            with gzip.open(metadata,'wt') as out:
                for row in changed:out.write(json.dumps(row)+'\n')
        write_rows();diagnostics.write_text(''.join(json.dumps(d)+'\n' for d in parents))
        with reference.open('w',newline='') as out:
            keys=['sample_index','row_id','role','associated_r_state','q_x','q_y','L','r_x_GT_diagnostic_only','r_y_GT_diagnostic_only'];w=csv.DictWriter(out,keys);w.writeheader();w.writerow(dict(zip(keys,[1,'invented1','FIT','D',100.,100.,96.,100.,100.])))
        expected={'attempts':2,'success':1,'failed':1,'success_by_family':{0:1,1:0}}
        for version in [1,2,3]:
            npy(alpha,payload,version=version);result=checker.verify(metadata,alpha,diagnostics,reference,expected=expected)
            assert result['PASS'] and result['NPY']['version']==[version,0] and result['YCB_external_reference_rows_matched']==1
        checks.append('NPY_v1_v2_v3_and_full_metadata_coverage_and_optional_YCB_match')
        assert checker.alpha_at(bytes([128])+bytes(1151),0,0) and not checker.alpha_at(bytes([128])+bytes(1151),7,0)
        assert checker.alpha_at(bytes([1])+bytes(1151),7,0);checks.append('MSB_y96plusx_bit_order')
        def reject(name,action):
            try:action()
            except (checker.VerificationError,OSError):checks.append(name)
            else:raise AssertionError(name)
        wrong=bytearray(cover);point=47*96+47;wrong[point//8]&=255^(1<<(7-point%8));npy(alpha,bytes(wrong)+sham+zeros+zeros)
        reject('independent_bad_COVER_tap_rejected',lambda:checker.verify(metadata,alpha,diagnostics,reference,expected=expected))
        npy(alpha,payload[:-1]);reject('NPY_bytecount_truncation_rejected',lambda:checker.verify(metadata,alpha,diagnostics,reference,expected=expected))
        npy(alpha,payload);bad=copy.deepcopy(rows);bad[0]['original_query_xy']=[101.,100.];write_rows(bad)
        reject('parent_q_identity_mismatch_rejected',lambda:checker.verify(metadata,alpha,diagnostics,reference,expected=expected))
        bad=copy.deepcopy(rows);bad[1]['proposed_states']={'COVER':'E','SHAM':'D'};write_rows(bad)
        reject('failed_pair_E_fallback_rejected',lambda:checker.verify(metadata,alpha,diagnostics,reference,expected=expected));write_rows()
        for value in ['/never/V112/not-open','../not-open','/never/__pycache__/x','/never/x.PYC']:
            reject('lexical_before_FS:'+value,lambda value=value:checker.open_input(value))
        (root/'linked.jsonl.gz').symlink_to(metadata)
        reject('argument_symlink_rejected',lambda:checker.open_input(root/'linked.jsonl.gz'))
    return {'completed':True,'PASS':True,'checks':checks,'fixture_groups':8,'actual_SOURCE_pair_alpha_diag_YCB_reference_or_pixel_reads':0,'numpy_or_torch_imported':False,'manual_struct_NPY_fixture_only':True,'inert_forbidden_donor_pointer_never_followed':True}


if __name__=='__main__':
    result=run()
    with (Path(__file__).parent/'TOY_RESULT.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result))
