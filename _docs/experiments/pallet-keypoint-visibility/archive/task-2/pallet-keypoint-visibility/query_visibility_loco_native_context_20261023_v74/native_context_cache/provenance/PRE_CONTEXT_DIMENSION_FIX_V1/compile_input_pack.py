"""Compile selected frozen source inputs. No GT arrays/model/backbone/PnP.

Reads selected P3 memmap rows only, native-pixel crops from already-padded source
PNGs, and GT-free initial query/box arrays. Never reads matched or gt_* files.
"""
from pathlib import Path
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import time
import sys
import numpy as np
from PIL import Image
import PIL
import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]
CACHE_ROOT=Path('/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_line_pose_v1')
CACHE=CACHE_ROOT/'cache'
SELECTION=HERE/'SOURCE_ROW_SELECTION_LOCK.json'
SELECTION_SHA='0102d3cce2cab9f2356e4b3039f4fd6ecca9f98e61e30076119a014c106ba594'
ROI_CODE=PARENT.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
CONTRACT=HERE.parent/'source_contract/native_context_contract.py'
SAMPLER=PARENT/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
SMALL_KEYS=['points','boxes','point_valid','input_shape','gain','detected','record_index','selected_candidate_index']
PACK_KEYS={'p3_roi','rgb96','crop_valid96','query_valid','saved_point_valid','roi_valid','query8_xy','query8_xy_safe','L','query_box_xyxy','target_rgb128','target_valid128','target_crop_center_xy','target_crop_side','target_context_valid','native_index','initial8_norm','initial8_norm_valid','context44','detected'}


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']


def module(p):
    spec=importlib.util.spec_from_file_location('_bound_'+sha(p)[:16],p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def pure(p,name):
    node=next(n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    d={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),d);return d[name]


def crop(rgb,q,L,n):
    h,w=rgb.shape[:2];u=(np.arange(n,dtype=np.float64)+.5)/n-.5
    xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);dx=xx-ix;dy=yy-iy
    acc=np.zeros((n,n,3),np.float64)
    for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        x,y=ix+ax,iy+ay;inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        acc[inside]+=rgb[y[inside],x[inside]].astype(np.float64)*weight[inside,None]
    valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    out=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);out[~valid]=0
    return out.transpose(2,0,1).copy(),valid[None].copy()


def compile_frame(rgb,q9,box_input,prepared_hw,input_hw,stored_gain,p3,point_valid,detected,roi,contract,boxrule):
    q9raw,rawbox,gain,offset=contract.source_canvas_to_raw(q9,box_input,prepared_hw,input_hw,pad=100)
    assert stored_gain==gain
    q=q9raw[:8];h,w=rgb.shape[:2];finite=np.isfinite(q).all(1)
    safe=np.where(finite[:,None],q,-1.)
    # Only nonfinite samples receive the sentinel for numerical interpolation;
    # preserved raw q remains NaN. Finite q is never moved or clipped.
    _,_,L=roi.native_grid(safe,[h,w])
    patch,roival,_=roi.patches(dict(p3=torch.from_numpy(np.array(p3,copy=True)),added_border=100),safe,[h,w],gain,offset)
    query_valid=finite & np.asarray(point_valid[:8],bool)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    pixel=np.floor(np.where(finite[:,None],q,-1e8)+.5)
    query_valid&=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    local=np.zeros((8,3,96,96),np.uint8);valid=np.zeros((8,1,96,96),bool)
    for k in np.flatnonzero(finite):local[k],valid[k]=crop(rgb,q[k],L,96)
    qb=boxrule(q,[h,w]);target=np.zeros((3,128,128),np.uint8);targetv=np.zeros((1,128,128),bool)
    center=np.zeros(2,np.float64);side=0.;hasbox=qb is not None
    if hasbox:
        center=(qb[:2]+qb[2:])/2;side=float(1.25*np.max(qb[2:]-qb[:2]));target,targetv=crop(rgb,center,side,128)
    box=np.zeros(4,np.float64) if qb is None else qb
    norm=np.zeros((8,2),np.float32)
    if hasbox:norm[finite]=((q[finite]-center)/side).astype(np.float32)
    # Pure public slot/query-set identity; no renderer/native8 GT association.
    ctx=np.stack([contract.input_context_features(q,k,box,[h,w]) for k in range(8)])
    arrays=dict(p3_roi=patch.numpy(),rgb96=local,crop_valid96=valid,query_valid=query_valid,saved_point_valid=np.asarray(point_valid[:8],bool),roi_valid=np.asarray(roival,bool),
        query8_xy=q.astype(np.float64),query8_xy_safe=np.where(finite[:,None],q,0).astype(np.float32),L=np.asarray(L,np.float64),query_box_xyxy=box,
        target_rgb128=target,target_valid128=targetv,target_crop_center_xy=center,target_crop_side=np.asarray(side,np.float64),target_context_valid=np.asarray(hasbox,bool),
        native_index=np.arange(8,dtype=np.int8),initial8_norm=norm,initial8_norm_valid=finite,context44=ctx,detected=np.asarray(detected,bool))
    assert set(arrays)==PACK_KEYS and patch.shape==(8,65,9,9) and ctx.shape==(8,44)
    return arrays,dict(native_q9=q9raw,producer_box_native=rawbox,gain=gain,offset=offset,L=L,box_available=hasbox)


def mock():
    out=HERE/'GENERATED_PACKER_TESTS.json';assert not out.exists()
    torch.set_num_threads(1);roi=module(ROI_CODE);contract=module(CONTRACT);oldcrop=pure(SAMPLER,'crop96');boxrule=pure(SAMPLER,'initial_query_box')
    rng=np.random.default_rng(20261023);pix=0
    for hw in [(31,47),(540,960),(480,720)]:
        rgb=rng.integers(0,256,(*hw,3),np.uint8)
        for q in [[0.,0.],[13.3,20.2],[hw[1]-.2,hw[0]-.4],[-3,12]]:
            for L in [16.,31.6,100.]:
                a,b=crop(rgb,q,L,96);c,d=oldcrop(rgb,q,L);assert np.array_equal(a,c) and np.array_equal(b,d);pix+=96*96
    rgb=np.zeros((48,64,3),np.uint8);raw=np.array([[10+k*3,15+k]for k in range(9)],float)
    prepared=np.array([248,264]);gain=min(640/prepared);ih=np.array([608,640]);offset=np.array([round((ih[1]-round(prepared[1]*gain))/2-.1),round((ih[0]-round(prepared[0]*gain))/2-.1)])
    points=((raw+100)*gain+offset).astype(np.float32);box=((np.array([8,10,42,30]).reshape(2,2)+100)*gain+offset).reshape(4).astype(np.float32)
    p3=rng.normal(size=(64,80,80)).astype(np.float16)
    pack,info=compile_frame(rgb,points,box,prepared,ih,gain,p3,np.ones(9,bool),True,roi,contract,boxrule)
    assert np.max(np.abs(info['native_q9']-raw))<1e-5 and np.array_equal(pack['native_index'],np.arange(8))
    nanpack,naninfo=compile_frame(rgb,np.full((9,2),np.nan,np.float32),np.full(4,np.nan,np.float32),prepared,ih,gain,p3,np.zeros(9,bool),False,roi,contract,boxrule)
    assert not nanpack['query_valid'].any() and not nanpack['p3_roi'].any() and not nanpack['target_context_valid'] and np.isnan(nanpack['query8_xy']).all()
    assert np.isfinite(nanpack['query8_xy_safe']).all() and np.isfinite(nanpack['context44']).all()
    write(out,dict(PASS=True,code=bind(__file__),RGB96_sampler_differences=0,generated_RGB96_pixels=pix,V60_roi_preflight=roi.roi_preflight(),
        affine_roundtrip_max_px=float(np.max(np.abs(info['native_q9']-raw))),missing_detection_retained=True,missing_detection_zerosafe_features=True,
        raw9_to_native_indices0through7_unchanged=True,actual_source_cache_RGB_or_GT_reads=0,new_model_forward_training_PnP=0))
    print(json.dumps(bind(out)),flush=True)


def freeze():
    p=HERE/'INPUT_PACK_PROTOCOL.json';assert not p.exists();assert sha(SELECTION)==SELECTION_SHA
    selection=load(SELECTION);assert len(selection['selected'])==5120 and selection['train']==4096 and selection['heldout']==1024
    deps=[SELECTION,CACHE_ROOT/'SOURCE_MANIFEST.json',CACHE/'CACHE_MANIFEST.json',CACHE/'CACHE_COMPLETE.json',ROI_CODE,CONTRACT,SAMPLER,HERE/'GENERATED_PACKER_TESTS.json']+[CACHE/(k+'.npy')for k in SMALL_KEYS]
    st=(CACHE/'p3.npy').stat()
    write(p,dict(schema='V74_FIXED_GT_FREE_NATIVE_CONTEXT_INPUT_PACK_V1',frozen_before_actual_selected_RGB_P3_sampling=True,code=bind(__file__),bindings=[bind(x)for x in deps],
        selection_sha256=SELECTION_SHA,frames=5120,queries=40960,TRAIN4096=True,HELDOUT1024=True,chunk_frames=128,output_directory=str(HERE/'SYNTH5120_INPUT_PACK'),output_disk_budget_bytes=2000000000,
        query='Saved highest-confidence prediction before GTmatch. Invert original integer LetterBox affine then subtract100once. No GT/matched/score filters; retain all9 raw coordinates and use original0..7 slots.',
        prepared_RGB='PIL12.0RGBdecode boundPNG then removeexact100pixels eachside once; no resize. RawHW must equal selected rawHW.',
        local_L='Exact V60:clip(.25*max finite initialq8 span,16,max(rawHW));16fallback. Native9float64 calculations then gridfloat32 into unchangedP3FP16->float32 grid_sample align_cornersFalse; returnFP16.',
        local_RGB96='V71/V72 exact pixel-center bilinear from continuous rawq, constant0 outside-native footprint, floor(value+.5). No q rounding/recentering.',
        target_RGB128='V65initialq8 minmaxclippedbox, center fixed; square side1.25*max(boxwidth,boxheight). Same native pixel-center sampler128 and valid; no box=>zero target/flagFalse.',
        query_context='Public initializer nativeindex0..7; fullsame8query relative-to-targetcropcenter dividedbytargetside, invalidzero+validflag; pure44context from boundinput_context_features. Native8dictionarycertificationnotprovidedbythisinputpack.',
        numeric_nulls='Raw NaN queries preserved inquery8_xy/lineage. Only sampler/forward safe representations receive sentinel/zero withvalidFalse. Invalid/outside/missingproposal retained; never teacherE/O by absence.',
        forbidden_inputs=['gt_*','matched','matched_gt_index','matched_iou','reference','mask','cueID','semanticGT','PnP'],model_tensor_keys=sorted(PACK_KEYS),
        P3_memmap=dict(path=str(CACHE/'p3.npy'),bytes=st.st_size,mtime_ns=st.st_mtime_ns,shape=[60000,64,80,80],dtype='float16',integrity='Existing cache COMPLETE/shard bindings preserved; selected rowSHA recomputed. Whole49GB file is not copied or rehashed.'),
        runtime=dict(executable=sys.executable,python=sys.version,Pillow=PIL.__version__,torch=str(torch.__version__),numpy=np.__version__,CPU_threads=1),new_backbone_model_forward_training_current_PnP=0))
    print(json.dumps(bind(p)),flush=True)


def run():
    prot=load(HERE/'INPUT_PACK_PROTOCOL.json');assert prot['code']==bind(__file__) and PIL.__version__=='12.0.0';assert sha(SELECTION)==SELECTION_SHA
    for b in prot['bindings']:check(b)
    st=(CACHE/'p3.npy').stat();assert st.st_size==prot['P3_memmap']['bytes'] and st.st_mtime_ns==prot['P3_memmap']['mtime_ns']
    out=Path(prot['output_directory']);out.mkdir(parents=True,exist_ok=False);(out/'chunks').mkdir()
    torch.set_num_threads(1);roi=module(ROI_CODE);contract=module(CONTRACT);boxrule=pure(SAMPLER,'initial_query_box')
    arrays={k:np.load(CACHE/(k+'.npy'),mmap_mode='r',allow_pickle=False)for k in SMALL_KEYS};p3=np.load(CACHE/'p3.npy',mmap_mode='r',allow_pickle=False)
    selection=load(SELECTION)['selected'];start=time.monotonic();chunk=[];metadata=[];chunks=[];missing=0;valid_queries=0;input_bytes=0;disk_bytes=0;cachemeta=load(CACHE/'CACHE_MANIFEST.json')
    complete=load(CACHE/'CACHE_COMPLETE.json');bound_shards={}
    for i,r in enumerate(selection):
        row=int(r['cache_row']);assert int(arrays['record_index'][row])==row and cachemeta['record_ids'][row]==r['id']
        shard_name=f'{row//128*128:06d}_{min((row//128+1)*128,60000):06d}.json'
        if shard_name not in bound_shards:
            sp=CACHE/'shards'/shard_name;assert sha(sp)==complete['shard_marker_sha256'][shard_name];bound_shards[shard_name]=bind(sp)
        assert sha(r['image'])==r['image_sha256'];prepared=np.array(Image.open(r['image']).convert('RGB'));assert list(prepared.shape[:2])==r['prepared_shape_hw']
        pad=r['reflect_pad_px'];assert pad==100;rgb=prepared[pad:-pad,pad:-pad].copy();assert list(rgb.shape[:2])==r['raw_shape_hw']
        one,info=compile_frame(rgb,arrays['points'][row],arrays['boxes'][row],r['prepared_shape_hw'],arrays['input_shape'][row],float(arrays['gain'][row]),p3[row],arrays['point_valid'][row],bool(arrays['detected'][row]),roi,contract,boxrule)
        chunk.append(one);missing+=not bool(arrays['detected'][row]);valid_queries+=int(one['query_valid'].sum())
        # Runtime provenance only; neither GT nor labels are opened or stored.
        metadata.append(dict(frame_index=i,cache_row=row,id=r['id'],source=r['source'],partition=r['partition'],prepared_image=r['image'],prepared_image_sha256=r['image_sha256'],raw_HW=r['raw_shape_hw'],raw_RGB_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            P3_selected_row_sha256=hashlib.sha256(np.ascontiguousarray(p3[row]).tobytes()).hexdigest(),original_input_points_sha256=hashlib.sha256(np.ascontiguousarray(arrays['points'][row]).tobytes()).hexdigest(),
            detected=bool(arrays['detected'][row]),selected_candidate_index=int(arrays['selected_candidate_index'][row]),query_valid8=one['query_valid'].tolist(),target_box_available=bool(one['target_context_valid']),
            raw_L=info['L'],raw_q9_audit_only=[[float(x) if np.isfinite(x) else None for x in xy]for xy in info['native_q9']],
            original_producer_box_native_audit_only=[float(x) if np.isfinite(x) else None for x in info['producer_box_native']],
            gain=info['gain'],integer_letterbox_offset=info['offset'].tolist(),GT_used=False))
        if len(chunk)==128 or i==len(selection)-1:
            begin=i+1-len(chunk);path=out/'chunks'/f'frames_{begin:06d}_{i+1:06d}.npz'
            combined={k:np.stack([x[k]for x in chunk])for k in PACK_KEYS};assert set(combined)==PACK_KEYS
            input_bytes+=sum(x.nbytes for x in combined.values());np.savez_compressed(path,**combined);disk_bytes+=path.stat().st_size
            assert disk_bytes<prot['output_disk_budget_bytes'],'Diskbudget reached; preserve partial run; do not silently reduce fields/data.'
            chunks.append(dict(start=begin,end=i+1,frames=len(chunk),file=bind(path),uncompressed_bytes=sum(x.nbytes for x in combined.values()),keys=sorted(PACK_KEYS)))
            chunk.clear();print(json.dumps(dict(frames=i+1,total=5120,disk_bytes=disk_bytes,seconds=time.monotonic()-start)),flush=True)
    (out/'INDEX_LINEAGE_INPUT_ONLY.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n'for x in metadata))
    write(out/'PACK_MANIFEST.json',dict(schema='V74_NATIVE_CONTEXT_INPUT_PACK_CHUNKS',completed=True,protocol=bind(HERE/'INPUT_PACK_PROTOCOL.json'),selection=bind(SELECTION),model_tensor_keys=sorted(PACK_KEYS),frame_count=5120,query_count=40960,chunks=chunks,lineage=bind(out/'INDEX_LINEAGE_INPUT_ONLY.jsonl'),source_partition_counts={f'{s}|{p}':sum(x['source']==s and x['partition']==p for x in selection) for s in ['G38','P0','TEX'] for p in ['train','heldout']},missing_detections_retained=int(missing),query_valid_count=valid_queries,uncompressed_bytes=input_bytes,NPZ_disk_bytes=disk_bytes,source_shard_proofs=list(bound_shards.values()),GT_files_or_arrays_read=[],new_backbone_model_forward_training_current_PnP=0,runtime=prot['runtime'],seconds=time.monotonic()-start))
    write(out/'PACK_SEAL.json',dict(completed=True,PASS=True,protocol=bind(HERE/'INPUT_PACK_PROTOCOL.json'),code=bind(__file__),manifest=bind(out/'PACK_MANIFEST.json'),lineage=bind(out/'INDEX_LINEAGE_INPUT_ONLY.jsonl'),all_chunk_SHA=[c['file']for c in chunks],frames=5120,queries=40960,all_saved_queries_unfiltered=True,GT_or_reference_or_mask_forward_inputs=False,new_backbone_model_forward_training_current_PnP=0))
    print(json.dumps(dict(completed=True,seal=bind(out/'PACK_SEAL.json'),disk_bytes=disk_bytes,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mock','freeze','run']);a=p.parse_args()
    {'mock':mock,'freeze':freeze,'run':run}[a.stage]()
