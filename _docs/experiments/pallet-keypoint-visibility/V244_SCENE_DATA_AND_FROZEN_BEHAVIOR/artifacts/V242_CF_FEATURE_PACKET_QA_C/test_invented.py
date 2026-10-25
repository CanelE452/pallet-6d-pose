"""작은 가짜 packet만 사용. SOURCE 파일/모델/encoder 실행 없음."""
import json,tempfile
from pathlib import Path
import numpy as np
from packet_io import binding,map_npy,lexical
from verify_features import relative,check_role,same,norm_error,run

def rejects(f):
    try:f()
    except (ValueError,PermissionError):return
    raise AssertionError('negative check failed')

def tests():
    q=np.array([[[10.,20.],[12.,20.],[999.,999.],[11.,19.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]]])
    avail=np.array([[1,1,0,1,0,0,0,0]],bool)
    g={'q8':q,'L8':np.full((1,8),2.,np.float64),'original_available8':avail,
       'node_available':avail,'context_fraction':np.array([1.],np.float32)}
    r=relative(g,np.array([0]),np.array([0]));assert np.array_equal(r[0,1],[1,0]) and not r[0,0].any() and not r[0,2].any()
    clipped={**g,'L8':np.full((1,8),.01,np.float64)};assert relative(clipped,np.array([0]),np.array([0]))[0,1,0]==8
    n=3;cons=np.array([1,1,0],bool);indices=np.arange(n,dtype=np.int64);cache=np.zeros(n,np.int64);slot=np.zeros(n,np.int64)
    d={'condition':np.arange(n*32,dtype=np.float32).reshape(n,32),'fractions':np.ones((n,2),np.float32)}
    base=np.zeros((n,1186),np.float32);node=np.zeros((n,8,384),np.float32);context=np.zeros((n,384),np.float32)
    node[:2,avail[0],0]=1;context[:2,0]=1;base[:2,0]=1;base[:2,768]=1;base[:2,1152:1154]=1;base[:2,1154:]=d['condition'][:2]
    got,err,_=check_role(base,node,context,g,cache,slot,cons,d,indices)
    assert got['constructed_rows']==2 and got['supported_nodes_processed']==6 and max(err.values())==0
    # 두 번째 constructed 행을 proof UNKNOWN이라 가정해도 동일하게 unit feature가 필요함.
    b=base.copy();nn=node.copy();cc=context.copy();b[1]=0;nn[1]=0;cc[1]=0
    rejects(lambda:check_role(b,nn,cc,g,cache,slot,cons,d,indices))
    b=base.copy();b[0,0]=np.nextafter(np.float32(1),np.float32(2));rejects(lambda:check_role(b,node,context,g,cache,slot,cons,d,indices))
    nn=node.copy();nn[0,2,0]=1;rejects(lambda:check_role(base,nn,context,g,cache,slot,cons,d,indices))
    b=base.copy();b[2,0]=1;rejects(lambda:check_role(b,node,context,g,cache,slot,cons,d,indices))
    assert norm_error(np.array([[1.,0.]],np.float32))==0;rejects(lambda:norm_error(np.array([[0.,0.]],np.float32)))
    with tempfile.TemporaryDirectory(prefix='v242-feature-toy-') as td:
        root=Path(td);a=np.array([[1.,2.]],np.float32)
        for version in ((1,0),(2,0)):
            p=root/('header'+str(version[0])+'.npy')
            with p.open('wb') as f:np.lib.format.write_array(f,a,version=version,allow_pickle=False)
            b=binding(p);assert same(map_npy(b,[b],'float32',(1,2)),a)
        p=root/'entry.json';p.write_text('{"approved_independent_SOURCE_CF_feature_packet_QA":false}')
        rejects(lambda:run(str(p)))
    rejects(lambda:lexical('/tmp/V112/absent'));rejects(lambda:lexical('/tmp/__pycache__/unread.pyc'))
    return {'completed':True,'PASS':True,'invented_groups':5,'checks':['F64 relative/self/U/clip','constructed+UNKNOWN not feature-masked; unsupported and unconstructed zero','single-ULP sharing drift rejected','CLS unit norm and zero descriptor rejected','public NPY1/2 header, false actual gate, lexical pre-FS rejection'],'actual_SOURCE_packet_model_or_RGB_reads':0}

if __name__=='__main__':print(json.dumps(tests(),ensure_ascii=False))
