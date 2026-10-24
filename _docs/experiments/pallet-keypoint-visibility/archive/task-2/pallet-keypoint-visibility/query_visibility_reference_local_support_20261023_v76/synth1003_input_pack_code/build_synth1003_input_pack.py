"""Materialize only original N3 model inputs, without GT/model imports.

Uses original cached ROI and q/L exactly. Native rawRGB is already unpadded.
Sampler must be explicitly frozen by the bound root protocol before build.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys,time
from collections import Counter
import numpy as np
from PIL import Image
import PIL

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'synth1003_input_pack'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def bound_protocol(args):
    p=Path(args.protocol).resolve();assert sha(p)==args.protocol_sha
    assert args.protocol_sha=='603aeea407f14caa76665cef08219b0c0c8f5bb7ef5ad70a6597aacb6bcec33f' and args.sampler=='bilinear'
    d=load(p);assert len(d['bindings'])==5 and d['schema']=='V76_ORIGINAL_N3_1003_INPUT_PACK_BUILD_PROTOCOL_V2'
    for b in d['bindings']:check(b)
    return p,d
def pure_context(path):
    node=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='input_context_features')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['input_context_features']

def pure_crop(path):
    node=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['crop']

def initial_query_box(q,hw):
    h,w=hw;good=np.isfinite(q).all(1)&~(q==-1).all(1)
    if good.sum()<2:return None
    lo=np.maximum(q[good].min(0),[0.,0.]);hi=np.minimum(q[good].max(0),[w-1.,h-1.])
    return np.r_[lo,hi]if(hi-lo>=2).all()else None

def crop_nearest(rgb,q,L,n):
    h,w=rgb.shape[:2];u=(np.arange(n,dtype=np.float64)+.5)/n-.5
    xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L)
    px=np.floor(xx+.5).astype(np.int64);py=np.floor(yy+.5).astype(np.int64)
    valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(px>=0)&(px<w)&(py>=0)&(py<h)
    out=np.zeros((n,n,3),np.uint8);out[valid]=rgb[py[valid],px[valid]]
    return out.transpose(2,0,1).copy(),valid[None].copy()

def crop_bilinear(rgb,q,L,n):
    h,w=rgb.shape[:2];u=(np.arange(n,dtype=np.float64)+.5)/n-.5
    xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L)
    ix=np.floor(xx).astype(np.int64);iy=np.floor(yy).astype(np.int64);dx=xx-ix;dy=yy-iy
    acc=np.zeros((n,n,3),np.float64)
    for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        x,y=ix+ax,iy+ay;valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        acc[valid]+=rgb[y[valid],x[valid]].astype(np.float64)*weight[valid,None]
    valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    out=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);out[~valid]=0
    return out.transpose(2,0,1).copy(),valid[None].copy()

def scalar_pixel(rgb,q,L,n,iy,ix,mode):
    x=float(q[0])+((ix+.5)/n-.5)*L;y=float(q[1])+((iy+.5)/n-.5)*L;h,w=rgb.shape[:2]
    if mode=='nearest':
        px=int(np.floor(x+.5));py=int(np.floor(y+.5));v=0<=x<w and 0<=y<h and 0<=px<w and 0<=py<h
        return rgb[py,px]if v else np.zeros(3,np.uint8),bool(v)
    v=-.5<=x<w-.5 and -.5<=y<h-.5
    if not v:return np.zeros(3,np.uint8),False
    x0=int(np.floor(x));y0=int(np.floor(y));acc=np.zeros(3,float)
    for dx in (0,1):
        for dy in (0,1):
            px=x0+dx;py=y0+dy
            if 0<=px<w and 0<=py<h:acc+=rgb[py,px]*((1-(x-x0))if dx==0 else x-x0)*((1-(y-y0))if dy==0 else y-y0)
    return np.floor(np.clip(acc,0,255)+.5).astype(np.uint8),True

def compile_frame(rgb,q9,L,roi,cached,context,crop):
    q9=np.asarray(q9,np.float64);q=q9[:8];h,w=rgb.shape[:2]
    # Exact producer excludes explicit(-1,-1); actual1003has none. Preserve raw.
    good=np.isfinite(q).all(1)&~(q==-1).all(1);internal=q.copy();internal[~good]=np.nan
    safe=np.where(good[:,None],internal,0.).astype(np.float32)
    pixel=np.floor(np.where(good[:,None],q,-1e8)+.5)
    continuous=good&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    nearest=good&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    local=np.zeros((8,3,96,96),np.uint8);lv=np.zeros((8,1,96,96),bool)
    for k in np.flatnonzero(good):local[k],lv[k]=crop(rgb,q[k],L,96)
    qb=initial_query_box(q,[h,w]);hasbox=qb is not None;box=np.zeros(4,np.float64)if qb is None else qb
    target=np.zeros((3,128,128),np.uint8);tv=np.zeros((1,128,128),bool);center=np.zeros(2,np.float64);side=0.
    if hasbox:center=(box[:2]+box[2:])/2;side=float(1.25*np.max(box[2:]-box[:2]));target,tv=crop(rgb,center,side,128)
    norm=np.zeros((8,2),np.float32)
    if hasbox:norm[good]=((q[good]-center)/side).astype(np.float32)
    ctx=np.stack([context(internal,k,box,[h,w])for k in range(8)])
    assert ctx.shape==(8,42)and np.isfinite(ctx).all()
    return dict(p3_roi=np.asarray(roi).copy(),rgb96=local,crop_valid96=lv,query9_xy_original=q9.copy(),query8_xy=q.copy(),query8_xy_safe=safe,
        query_valid_cached=np.asarray(cached,bool).copy(),pure_continuous_query_valid=continuous,nearest_pixel_valid=nearest,
        query_center_valid=continuous&nearest,ROI_center_valid=np.asarray(roi[:,64,4,4],bool),ROI_anyvalid=np.asarray(roi[:,64],bool).reshape(8,-1).any(1),query_missing=~good,
        L=np.asarray(L,np.float64),query_box_xyxy=box,target_rgb128=target,target_valid128=tv,target_crop_center_xy=center,target_crop_side=np.asarray(side,np.float64),
        target_context_valid=np.asarray(hasbox,bool),native_index=np.arange(8,dtype=np.int8),initial8_norm=norm,initial8_norm_valid=good,context42=ctx)

def generated(args):
    protocol,d=bound_protocol(args);p=HERE/'GENERATED_INPUT_PACKER_TESTS.json';assert not p.exists()
    mode=args.sampler;crop=pure_crop(d['bindings'][4]['path']);context=pure_context(d['bindings'][3]['path'])
    rng=np.random.default_rng(20261023);tested=0
    for hw in [(17,29),(480,640),(540,960)]:
        rgb=rng.integers(0,256,(*hw,3),np.uint8)
        for q in [[.0,.0],[12.3,15.7],[hw[1]-.1,hw[0]-.4],[-4.,8.]]:
            for L in (16.,63.7):
                for n in (96,128):
                    a,v=crop(rgb,q,L,n)
                    for iy,ix in [(0,0),(n-1,n-1),(n//2,n//2),(13,17),(23,37)]:
                        px,pv=scalar_pixel(rgb,q,L,n,iy,ix,mode);assert np.array_equal(a[:,iy,ix],px)and bool(v[0,iy,ix])==pv;tested+=1
    q=np.array([[10+k,20+k]for k in range(9)],float);q[1]=[-1,-1];q[2]=[-2.,20.];q[3]=[1000,2000]
    roi=np.zeros((8,65,9,9),np.float16);cached=np.ones(8,bool)
    pack=compile_frame(np.zeros((48,64,3),np.uint8),q,16.,roi,cached,context,crop)
    assert np.array_equal(pack['query9_xy_original'],q)and np.array_equal(pack['query_valid_cached'],cached)and not pack['query_missing'][2]
    assert pack['query_missing'][1]and not pack['crop_valid96'][1].any()and np.isfinite(pack['context42']).all()
    assert not pack['pure_continuous_query_valid'][2]and not pack['pure_continuous_query_valid'][3]
    write(p,dict(PASS=True,code=bind(Path(__file__)),protocol=bind(protocol),sampler=mode,generated_scalar_RGB_pixels_tested=tested,
        original_q9_and_cachedvalid_unchanged=True,finite_outsidepoints_not_recentered=True,missing_sentinel_raw_preserved_numeric_safe_only=True,
        context_shape=[8,42],GT_masks_labels_refs_weights_read=0,source_RGB_or_ROI_arrays_read=0,new_model_forward_fit_current_PnP=0))
    print(json.dumps(bind(p)),flush=True)

def freeze(args):
    protocol,d=bound_protocol(args);test=load(HERE/'GENERATED_INPUT_PACKER_TESTS.json')
    assert test['code']==bind(Path(__file__))and test['protocol']==bind(protocol)and test['sampler']==args.sampler
    assert args.sampler=='bilinear'and PIL.__version__=='12.0.0'
    write(HERE/'INPUT_PACK_BUILD_ENTRYLOCK.json',dict(approved=True,protocol=bind(protocol),code=bind(Path(__file__)),generated=bind(HERE/'GENERATED_INPUT_PACKER_TESTS.json'),
        bindings=d['bindings'],sampler=args.sampler,sampler_reuse='AST-only exact boundV74crop function',chunk_frames=128,output=str(OUT),runtime=dict(executable=sys.executable,python=sys.version,Pillow=PIL.__version__,numpy=np.__version__),
        allowed_reads=['boundmodelinputlinks','boundoriginalSYNTH_ROI_CACHE.x,pixel_valid','listedrawRGB','boundpureinput_context_featurescode'],
        forbidden_reads=['GTteacherindex','GTannotation','GTreference','mask','label','weight','currentRGB/cache/annotation/scores'],new_model_forward_fit_current_PnP=0))
    print(json.dumps(bind(HERE/'INPUT_PACK_BUILD_ENTRYLOCK.json')),flush=True)

def run(args):
    protocol,d=bound_protocol(args);entry=load(HERE/'INPUT_PACK_BUILD_ENTRYLOCK.json')
    assert entry['code']==bind(Path(__file__))and entry['protocol']==bind(protocol)and entry['sampler']==args.sampler
    check(entry['generated']);assert PIL.__version__=='12.0.0'
    rows=[json.loads(x)for x in Path(d['bindings'][0]['path']).read_text().splitlines()]
    assert len(rows)==1003 and len({r['id']for r in rows})==1003 and Counter(r['split']for r in rows)=={'train':768,'validation':235}
    context=pure_context(d['bindings'][3]['path']);crop=pure_crop(d['bindings'][4]['path'])
    with np.load(d['bindings'][2]['path'],allow_pickle=False)as z:
        assert set(z.files)=={'x','pixel_valid'};roi=z['x'];cached=z['pixel_valid']
    assert roi.shape==(1003,8,65,9,9)and roi.dtype==np.float16 and cached.shape==(1003,8)
    OUT.mkdir(exist_ok=False);(OUT/'chunks').mkdir();start=time.monotonic();batch=[];lineage=[];chunks=[];counts=Counter();uncompressed=0;disk=0
    for i,r in enumerate(rows):
        assert r['frozen_ROI']['frame_index']==i and r['frozen_ROI']['path']==d['bindings'][2]['path']
        image=Path(r['raw_RGB']['path']);assert sha(image)==r['raw_RGB']['sha256'];rgb=np.array(Image.open(image).convert('RGB'));assert list(rgb.shape[:2])==r['raw_RGB']['raw_hw']
        q9=np.asarray(r['unchanged_initial_query9'],np.float64);L=float(r['input_L'])
        one=compile_frame(rgb,q9,L,roi[i],cached[i],context,crop);batch.append(one)
        counts['cached_ROI_anyvalid']+=int(cached[i].sum());counts['continuous_valid']+=int(one['pure_continuous_query_valid'].sum());counts['nearest_valid']+=int(one['nearest_pixel_valid'].sum());counts['query_center_valid']+=int(one['query_center_valid'].sum());counts['ROI_center_valid']+=int(one['ROI_center_valid'].sum());counts['missing_sentinel']+=int(one['query_missing'].sum())
        counts['cached_vs_query_center_different']+=int(np.sum(one['query_valid_cached']!=one['query_center_valid']))
        assert np.array_equal(one['ROI_anyvalid'],cached[i]) and not one['query_missing'].any()
        assert np.array_equal(one['p3_roi'],roi[i])and np.array_equal(one['query9_xy_original'],q9)and float(one['L'])==L
        lineage.append(dict(frame_index=i,id=r['id'],split=r['split'],raw_RGB=r['raw_RGB'],raw_RGB_decoded_pixel_SHA256=hashlib.sha256(rgb.tobytes()).hexdigest(),source_ROI_frame=i,
            original_query9_sha256=hashlib.sha256(q9.tobytes()).hexdigest(),original_ROI_row_sha256=hashlib.sha256(roi[i].tobytes()).hexdigest(),input_L=L,
            producer=r['producer'],query_cached8=one['query_valid_cached'].tolist(),query_continuous8=one['pure_continuous_query_valid'].tolist(),query_nearest8=one['nearest_pixel_valid'].tolist(),query_center8=one['query_center_valid'].tolist(),GT_reads=False))
        if len(batch)==128 or i==1002:
            begin=i+1-len(batch);p=OUT/'chunks'/f'frames_{begin:06d}_{i+1:06d}.npz';arrays={k:np.stack([a[k]for a in batch])for k in batch[0]}
            bytes_=sum(a.nbytes for a in arrays.values());uncompressed+=bytes_;np.savez_compressed(p,**arrays);disk+=p.stat().st_size
            assert uncompressed<500_000_000
            chunks.append(dict(start=begin,end=i+1,frames=len(batch),file=bind(p),uncompressed_bytes=bytes_,schema={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in arrays.items()}));batch.clear()
            print(json.dumps(dict(frames=i+1,total=1003,disk_bytes=disk,uncompressed_bytes=uncompressed,seconds=time.monotonic()-start)),flush=True)
    line=OUT/'INDEX_LINEAGE_INPUT_ONLY.jsonl';line.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'for r in lineage))
    write(OUT/'PACK_MANIFEST.json',dict(completed=True,protocol=bind(protocol),entry=bind(HERE/'INPUT_PACK_BUILD_ENTRYLOCK.json'),frame_count=1003,query_count=8024,original_query9_count=9027,
        partition_counts={'train':768,'validation':235},chunks=chunks,lineage=bind(line),query_counts=dict(counts),uncompressed_bytes=uncompressed,NPZ_disk_bytes=disk,
        query_valid_definition='query_valid_cached preserves originalROIanyvalid; query_center_valid=finite&continuous&nearestbounds; pure_continuous/nearest/ROIcenter/ROIany storedseparately. No downstreamlossgate defined.',
        sampler=args.sampler,raw_RGB_already_unpadded=True,extra100border_removed=0,original_ROI_q9_L_values_copied_exact=True,GT_masks_labels_reference_weights_read=0,new_model_forward_fit_current_PnP=0,
        runtime=entry['runtime'],seconds=time.monotonic()-start))
    write(OUT/'PACK_SEAL.json',dict(completed=True,PASS=True,protocol=bind(protocol),entry=bind(HERE/'INPUT_PACK_BUILD_ENTRYLOCK.json'),code=bind(Path(__file__)),manifest=bind(OUT/'PACK_MANIFEST.json'),lineage=bind(line),all_chunk_SHA=[c['file']for c in chunks],frames=1003,queries=8024,
        original_N3_q_ROI_L_unchanged=True,R0_ROI_or_query_used=False,GT_masks_labels_reference_weights_read=0,new_model_forward_fit_current_PnP=0))
    print(json.dumps(dict(completed=True,seal=bind(OUT/'PACK_SEAL.json'),seconds=time.monotonic()-start)),flush=True)

def qa(args):
    protocol,d=bound_protocol(args);entry=load(HERE/'INPUT_PACK_BUILD_ENTRYLOCK.json');assert entry['code']==bind(Path(__file__))and entry['protocol']==bind(protocol)
    manifest=load(OUT/'PACK_MANIFEST.json');seal=load(OUT/'PACK_SEAL.json');check(seal['manifest']);check(seal['lineage'])
    rows=[json.loads(x)for x in Path(d['bindings'][0]['path']).read_text().splitlines()];lineage=[json.loads(x)for x in Path(seal['lineage']['path']).read_text().splitlines()]
    with np.load(d['bindings'][2]['path'],allow_pickle=False)as z:roi=z['x'];cached=z['pixel_valid']
    differences=0;scalar_checks=0;context=pure_context(d['bindings'][3]['path']);fixed=[]
    for c in manifest['chunks']:
        check(c['file']);start,end=c['start'],c['end']
        with np.load(c['file']['path'],allow_pickle=False)as z:
            assert np.array_equal(z['p3_roi'],roi[start:end])and np.array_equal(z['query_valid_cached'],cached[start:end])and np.array_equal(z['ROI_anyvalid'],cached[start:end])
            for j in range(end-start):
                i=start+j;r=rows[i];q=np.asarray(r['unchanged_initial_query9'],np.float64)
                assert lineage[i]['id']==r['id']and lineage[i]['frame_index']==i and lineage[i]['source_ROI_frame']==i
                assert np.array_equal(z['query9_xy_original'][j],q)and np.array_equal(z['query8_xy'][j],q[:8])and float(z['L'][j])==r['input_L']
                assert np.array_equal(z['native_index'][j],np.arange(8))
                ref=np.stack([context(q[:8],k,z['query_box_xyxy'][j],r['raw_RGB']['raw_hw'])for k in range(8)])
                assert np.array_equal(z['context42'][j],ref)
            # Fixed first frame of every chunk, all8nativequeries, scalar pixels.
            r=rows[start];image=Path(r['raw_RGB']['path']);assert sha(image)==r['raw_RGB']['sha256'];rgb=np.array(Image.open(image).convert('RGB'))
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==lineage[start]['raw_RGB_decoded_pixel_SHA256']
            for k in range(8):
                for iy,ix in [(0,0),(95,95),(48,48),(13,17),(23,37)]:
                    p,v=scalar_pixel(rgb,z['query8_xy'][0,k],float(z['L'][0]),96,iy,ix,args.sampler)
                    differences+=int(not np.array_equal(z['rgb96'][0,k,:,iy,ix],p));differences+=int(bool(z['crop_valid96'][0,k,0,iy,ix])!=v);scalar_checks+=1
            for iy,ix in [(0,0),(127,127),(64,64),(13,17),(23,37)]:
                p,v=scalar_pixel(rgb,z['target_crop_center_xy'][0],float(z['target_crop_side'][0]),128,iy,ix,args.sampler)
                differences+=int(not np.array_equal(z['target_rgb128'][0,:,iy,ix],p));differences+=int(bool(z['target_valid128'][0,0,iy,ix])!=v);scalar_checks+=1
            fixed.append(dict(frame_index=start,id=r['id'],image=r['raw_RGB']))
    assert differences==0
    p=OUT/'INPUT_PACK_QA.json';write(p,dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=bind(protocol),pack_seal=bind(OUT/'PACK_SEAL.json'),
        all1003_original_q9_ROI_L_cachedflags_ID_order_exact=True,all_context42_exact=True,fixed_first_frame_each_chunk=fixed,
        independent_scalar_RGB_pixels_checked=scalar_checks,RGB_or_valid_differences=differences,sampler=args.sampler,GT_masks_labels_reference_weights_read=0,new_model_forward_fit_current_PnP=0))
    write(OUT/'INPUT_PACK_QA_SEAL.json',dict(completed=True,PASS=True,QA=bind(p),pack_seal=bind(OUT/'PACK_SEAL.json'),new_model_forward_fit_current_PnP=0))
    print(json.dumps(bind(p)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','run','qa']);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--sampler',choices=['nearest','bilinear'],required=True)
    a=p.parse_args();{'generated':generated,'freeze':freeze,'run':run,'qa':qa}[a.stage](a)
