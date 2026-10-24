"""Exercise actual payload decoder/gate functions with invented bytes only."""
from pathlib import Path
import hashlib,io,json
import numpy as np
from PIL import Image
import build_raw96_witness_controls as B

def run():
    names=[]
    def ok(n,v):assert v,n;names.append(n)
    def bad(n,f):
        try:f()
        except (AssertionError,ValueError,TypeError):names.append(n);return
        raise AssertionError(n)
    a=np.zeros((96,96,3),np.uint8);a[7,8]=[13,24,35];buf=io.BytesIO();Image.fromarray(a).save(buf,format='PNG')
    ok('Actual_CLEAN_decoder_preserves_bytes',np.array_equal(B.decode_png(buf.getvalue(),'CLEAN96'),a))
    vv=np.ones((96,96),np.uint8)*255;vv[80:]=0;buf=io.BytesIO();Image.fromarray(vv).save(buf,format='PNG')
    ok('Actual_VALID_decoder_preserves_padding',np.array_equal(B.decode_png(buf.getvalue(),'VALID96'),vv>0))
    invalid=vv.copy();invalid[0,0]=128;buf=io.BytesIO();Image.fromarray(invalid).save(buf,format='PNG')
    bad('Non_binary_valid_rejected',lambda:B.decode_png(buf.getvalue(),'VALID96'))
    rgb=np.arange(5*6*3,dtype=np.uint8).reshape(5,6,3);alpha=np.ones((5,6),bool);alpha[2,3]=False
    proof={n:dict(shape=list(arr.shape),dtype=str(arr.dtype),array_sha256=hashlib.sha256(arr.tobytes()).hexdigest())for n,arr in [('rgb',rgb),('alpha',alpha)]}
    buf=io.BytesIO();np.savez_compressed(buf,rgb=rgb,alpha=alpha);data=buf.getvalue()
    r,a=B.decode_cutout(data,dict(recorded_array_proof=proof))
    ok('Actual_NPZ_decoder_checks_recorded_shapes_dtypes_hashes',np.array_equal(r,rgb)and np.array_equal(a,alpha))
    drift={k:dict(v)for k,v in proof.items()};drift['alpha']['array_sha256']='0'*64
    bad('Recorded_array_hash_mismatch_rejected',lambda:B.decode_cutout(data,dict(recorded_array_proof=drift)))
    bad('Unauthorized_reader_cannot_be_created',lambda:B.PayloadReader([],False))
    fake=[dict(path=f'/not_read_{i}.npz',sha256='a'*64,kind='COMPACT_RGBA')for i in range(26)]
    reader=B.PayloadReader(fake,True)
    bad('Non_whitelisted_binding_rejected_before_file_access',lambda:reader.read(dict(path='/not_whitelisted.png',sha256='a'*64,kind='CLEAN96')))
    ok('Rejected_reader_open_count_stays_zero',reader.opened==[])
    return dict(completed=True,PASS=True,invented_only=True,count=len(names),checks=names,production_assets_stat_hash_header_decode=0)

if __name__=='__main__':
    result=run();p=Path(__file__).parent/'INVENTED_LOADER_CHECKS.json'
    assert 'v112'not in str(p).casefold()
    with p.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,count=result['count'],checks_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
