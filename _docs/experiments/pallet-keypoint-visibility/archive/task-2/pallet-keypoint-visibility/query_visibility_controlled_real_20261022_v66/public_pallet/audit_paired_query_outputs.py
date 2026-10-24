#!/usr/bin/env python3
"""Additive read-only reconstruction of the frozen public paired query outputs."""
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
V66=ROOT.parent
OUT=ROOT/'PAIRED_QUERIES'
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def bind(b):assert sha(b['path'])==b['sha256'],b['path']
def erode(m):return cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0

def replay(pair):
    ident=pair['pair_id']
    rng={}
    for stage in ['box','anchor','jitter']:
        raw=hashlib.sha256(f'20261022:{ident}:{stage}'.encode()).digest()
        rng[stage]=np.random.default_rng(int.from_bytes(raw[:8],byteorder='little'))
    masks=[cv2.imread(pair['files'][v]['mask_teacher']['path'],-1)>0 for v in ['A','B']]
    yy,xx=np.nonzero(masks[0]|masks[1])
    original=np.array([min(xx),min(yy),max(xx)+1,max(yy)+1],dtype=np.float64)
    size=original[2:]-original[:2];center=(original[:2]+original[2:])/2
    dx,dy=rng['box'].uniform(-.05,.05),rng['box'].uniform(-.05,.05)
    sx,sy=rng['box'].uniform(.9,1.1),rng['box'].uniform(.9,1.1)
    c=center+size*np.array([dx,dy]);wh=size*np.array([sx,sy])
    box=np.r_[c-wh/2,c+wh/2];box=np.minimum(np.maximum(box,0),[639,479,639,479])
    corners=np.floor(box.reshape(2,2)+.5).astype(np.int64)
    with np.load(pair['evidence']['path'])as z:
        va,vb,ea,eb=[z[k]for k in ['visible_A','visible_B','external_A','external_B']]
    union=va|vb
    pools={'E_core_union':erode(ea)|erode(eb),'E_union':ea|eb,'V_union':union,
           'commonV_core':erode(va&vb),'V_boundary':union&~erode(union)}
    chain=[['E_core_union','E_union','V_union']]*2+[['commonV_core','V_union']]*2+[['V_boundary','V_union']]*2
    used=set();anchors=[];selected=[]
    for k in range(8):
        anchor=None
        if k<6:
            for name in chain[k]:
                points=np.argwhere(pools[name])[:,::-1] # row-major y/x -> x/y
                eligible=points[np.array([tuple(p)not in used for p in points],dtype=bool)]
                if len(eligible):
                    anchor=eligible[int(rng['anchor'].choice(np.arange(len(eligible))))];pool=name;break
        if anchor is None:
            anchor=corners[k%2].copy();pool='roughbox_TL'if k%2==0 else'roughbox_BR'
        anchors.append(anchor);used.add(tuple(anchor));selected.append(pool)
    anchors=np.asarray(anchors,np.int64)
    jitter=rng['jitter'].uniform(-3,3,(8,2));query=anchors+jitter
    side=float(np.clip(.25*max(np.max(query[:,0])-np.min(query[:,0]),np.max(query[:,1])-np.min(query[:,1])),16,640))
    return box,anchors,jitter,query,side,selected

def independent_sample(grid,xy):
    pixel=np.floor(xy+.5).astype(np.int64);h,w=grid['exact_state'].shape
    x,y=pixel[...,0],pixel[...,1]
    valid=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(x>=0)&(x<w)&(y>=0)&(y<h)
    x,y=x.clip(0,w-1),y.clip(0,h-1)
    result={'valid':valid,'allowed':np.where(valid,grid['joint_compatible_mask'][y,x],63).astype(np.uint8)}
    for k in ['exact_state','radius2_state','joint_state']:result[k]=np.where(valid,grid[k][y,x],-1).astype(np.int8)
    result['joint_known']=valid&grid['joint_known'][y,x]
    return result

def main():
    output=OUT/'QUERY_PARITY_AUDIT.json';assert not output.exists()
    seal=json.loads((OUT/'SEAL.json').read_text());assert seal['completed']
    for b in seal['outputs']+[seal['input_lock'],seal['recipe'],seal['interpretation']]:bind(b)
    lock=json.loads(Path(seal['input_lock']['path']).read_text())
    for key in ['recipe','interpretation','source_evidence_manifest','label_protocol','source_split_lock','conditional_admission','denylist']:bind(lock[key])
    for b in lock['code']+lock['source_files']:bind(b)
    path=V66/'contract/partial_truth_grid.py';spec=importlib.util.spec_from_file_location('parity_partial_grid',path);partial=importlib.util.module_from_spec(spec);spec.loader.exec_module(partial)
    inputs=rows(OUT/'INPUTS_RGB_ONLY.jsonl');queries=rows(OUT/'QUERIES_LABELS.jsonl');anchors=rows(OUT/'ANCHOR_SELECTION.jsonl');source=rows(ROOT/'SYMMETRIC_ALL50_V2/PAIR_EVIDENCE_MANIFEST.jsonl')
    assert len(inputs)==100 and len(queries)==800 and len(anchors)==50 and len(source)==50
    input_map={r['id']:r for r in inputs};query_map={r['sample_id']:r for r in queries};anchor_map={r['pair_id']:r for r in anchors}
    allowed_fields={'id','global_id','group_id','split','source_split','source','image','image_sha256','raw_hw','query','query8','side','box'}
    for r in inputs:
        assert set(r)==allowed_fields and r['source']=='PALLOX'
        assert r['id']==r['global_id']and r['query']==r['query8']
        assert Path(r['image']).is_relative_to(ROOT/'data')and sha(r['image'])==r['image_sha256']
    caches={s:np.load(OUT/('PALLOX_TRAIN.npz'if s=='TRAIN'else'PALLOX_VAL.npz'))for s in ['TRAIN','VALIDATION']}
    ti={s:{str(x):i for i,x in enumerate(z['global_id'])}for s,z in caches.items()}
    checks=Counter();all63=Counter();rng_mismatches=[];label_mismatches=[]
    for pair in source:
        box,an,jit,q,side,pools=replay(pair);stored=anchor_map[pair['pair_id']]
        for name,a,b in [('box',box,stored['box']),('anchors',an,stored['anchors']),('jitter',jit,stored['jitter']),('query',q,stored['query'])]:
            diff=int(np.count_nonzero(a!=np.asarray(b)));checks['rng_'+name+'_values']+=int(a.size)
            if diff:rng_mismatches.append(dict(pair=pair['pair_id'],field=name,N=diff,max_abs=float(np.max(np.abs(a-np.asarray(b))))))
        assert side==stored['side']and pools==[r['selected_pool']for r in stored['selection']]
        # Rebuild both native coordinate systems independently of grid helper.
        dx=(np.arange(9,dtype=np.float64)-4)*side/9;lx,ly=np.meshgrid(dx,dx)
        localxy=q[:,None,None,:]+np.stack([lx,ly],axis=-1)
        center=(box[:2]+box[2:])/2;step=1.25*(box[2:]-box[:2])/64
        gx,gy=np.meshgrid(np.arange(64,dtype=np.float64)+.5-32,np.arange(64,dtype=np.float64)+.5-32)
        globalxy=center+np.stack([gx,gy],axis=-1)*step
        with np.load(pair['evidence']['path'])as e:
            for view in ['A','B']:
                ident='PALLOXPAIR:'+pair['pair_id']+':'+view;r=input_map[ident];split=r['split'];z=caches[split];target=ti[split][ident];ix=np.flatnonzero(z['target_index']==target)
                assert len(ix)==8 and np.array_equal(z['native_corner'][ix],np.arange(8))
                assert np.array_equal(z['query'][ix],q)and np.array_equal(z['side'][ix],np.repeat(side,8))
                assert np.array_equal(z['global_box'][target],box)
                assert r['query']==q.tolist()and r['box']==box.tolist()and r['side']==side
                assert str(z['global_image_sha256'][target])==r['image_sha256']and np.all(z['image_sha256'][ix]==r['image_sha256'])
                cert={k:e[k+'_'+view]for k in ['visible_positive','visible_negative','full_positive','full_negative']};assert not cert['full_negative'].any()
                grid=partial.partial_truth_grid(cert)
                for prefix,xy,index in [('dense',localxy,ix),('global',globalxy,target)]:
                    expected=independent_sample(grid,xy)
                    for key in ['allowed','valid','exact_state','radius2_state']:
                        actual=z[prefix+'_'+key][index];d=int(np.count_nonzero(actual!=expected[key]));checks[prefix+'_'+key+'_cells']+=int(actual.size)
                        if d:label_mismatches.append(dict(target=ident,field=prefix+'_'+key,N=d))
                    valid=expected['valid'];all63[(split,prefix,'valid_informative')]+=int((valid&(expected['allowed']!=63)).sum())
                    all63[(split,prefix,'valid_uninformative63')]+=int((valid&(expected['allowed']==63)).sum())
                    all63[(split,prefix,'invalid63')]+=int((~valid).sum())
                    assert not((expected['exact_state']==2)&valid).any()and not((expected['radius2_state']==2)&valid).any()
                for k in range(8):
                    qr=query_map[ident+':'+str(k)];assert str(z['sample_id'][ix[k]])==qr['sample_id']
                    for key in ['allowed','valid','joint_known','joint_state','exact_state','radius2_state']:
                        actual=z['center_'+key][ix[k]];name='joint_compatible_mask'if key=='allowed'else key
                        assert actual==qr[name]
                    checks['query_JSON_center_fields']+=6
                checks['targets']+=1
    assert not rng_mismatches and not label_mismatches
    for b in seal['outputs']+[seal['input_lock'],seal['recipe'],seal['interpretation']]:bind(b)
    report=dict(PASS=True,schema='PAIRED_PUBLIC_QUERY_RNG_AND_PARTIAL_GRID_PARITY_V1',
        seal=ref(OUT/'SEAL.json'),lock=ref(OUT/'QUERY_INPUT_LOCK.json'),audit_code=ref(__file__),
        RNG_reconstruction='Independent row-major np.argwhere pool and used-coordinate tuple exclusion; one frozen anchor RNG; center/scale draws scalar-ordered; no model output use',
        label_reconstruction='Native certainty-grid helper proof plus independently reconstructed float64 local/global XY, bounds and nearest-floor+.5 sampler',
        pairs=50,targets=100,queries=800,rng_mismatches=rng_mismatches,label_mismatches=label_mismatches,
        checks=dict(checks),validity_denominators=[dict(split=k[0],grid=k[1],kind=k[2],N=n)for k,n in sorted(all63.items())],
        input_only_manifest_whitelist_PASS=True,source_feature_inputs_contain_GT_depth_pose_or_evidence=False,
        all63_native_valid_preserved=True,hard_O_labels=0,semantic_self_GT=False,
        commonV_definition='erode2(Va&Vb), as query recipe; stable20mm is a separate descriptive statistic and is not an added sampler gate',
        source_conditioning_gap_disclosed=True,source_only=True,private_RGB_GT_scores_read=False,
        models_features_GPU_training_PnP_calls=0,runtime=dict(python=sys.executable,numpy=np.__version__,cv2=cv2.__version__))
    output.write_text(json.dumps(report,indent=2)+'\n');print('QUERY_PARITY_PASS',sha(output),dict(checks),flush=True)

if __name__=='__main__':main()
