"""Input-only independent affine/validity/scalar-crop audit of finished chunks."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
PACK=HERE/'SYNTH5120_INPUT_PACK'
SOURCE=Path('/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_line_pose_v1/cache')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def scalar_pixel(rgb,q,L,n,row,col):
    h,w=rgb.shape[:2];x=float(q[0])+((col+.5)/n-.5)*float(L);y=float(q[1])+((row+.5)/n-.5)*float(L)
    if not(-.5<=x<w-.5 and -.5<=y<h-.5):return np.zeros(3,np.uint8),False
    ix,iy=int(np.floor(x)),int(np.floor(y));dx,dy=x-ix,y-iy;value=np.zeros(3,float)
    for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        if 0<=ix+ax<w and 0<=iy+ay<h:value+=rgb[iy+ay,ix+ax]*weight
    return np.floor(np.clip(value,0,255)+.5).astype(np.uint8),True


def main():
    result=HERE/'INPUT_PACK_QA.json';assert not result.exists()
    seal=json.loads((PACK/'PACK_SEAL.json').read_text());assert seal['completed'] and seal['PASS'] and seal['frames']==5120
    for b in [seal['protocol'],seal['code'],seal['manifest'],seal['lineage'],*seal['all_chunk_SHA']]:assert sha(b['path'])==b['sha256']
    manifest=json.loads((PACK/'PACK_MANIFEST.json').read_text());selected=json.loads((HERE/'SOURCE_ROW_SELECTION_LOCK.json').read_text())['selected'];assert len(selected)==5120
    points=np.load(SOURCE/'points.npy',mmap_mode='r');valid0=np.load(SOURCE/'point_valid.npy',mmap_mode='r');shapes=np.load(SOURCE/'input_shape.npy',mmap_mode='r');gains=np.load(SOURCE/'gain.npy',mmap_mode='r');detected=np.load(SOURCE/'detected.npy',mmap_mode='r')
    frames=0;scalar_channels=0;target_channels=0;validqueries=0;missing=0;sample_frames=[]
    for desc in manifest['chunks']:
        with np.load(desc['file']['path'],allow_pickle=False) as z:part={k:z[k]for k in z.files}
        f=desc['end']-desc['start'];assert part['p3_roi'].shape==(f,8,65,9,9) and part['rgb96'].dtype==np.uint8
        assert np.isfinite(part['p3_roi']).all() and np.isin(part['p3_roi'][:,:,64],(0,1)).all()
        assert not np.any(part['p3_roi'][:,:,:64]*(1-part['p3_roi'][:,:,64:65]))
        for j in range(f):
            i=desc['start']+j;r=selected[i];row=r['cache_row'];ph,pw=r['prepared_shape_hw'];ih,iw=shapes[row];h,w=r['raw_shape_hw']
            gain=min(640/ph,640/pw);off=np.array([round((int(iw)-round(pw*gain))/2-.1),round((int(ih)-round(ph*gain))/2-.1)])
            q=(np.asarray(points[row],np.float64)-off)/gain-100;q=q[:8]
            assert gain==gains[row] and np.array_equal(part['query8_xy'][j],q,equal_nan=True)
            finite=np.isfinite(q).all(1);good=finite&~(q==-1).all(1)
            L=float(np.clip(.25*np.ptp(q[good],axis=0).max(),16,max(h,w))) if good.sum()>=2 else 16.
            assert part['L'][j]==L and np.array_equal(part['native_index'][j],np.arange(8))
            pixel=np.floor(np.where(finite[:,None],q,-1e8)+.5)
            qvalid=finite&valid0[row,:8]&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
            assert np.array_equal(qvalid,part['query_valid'][j]);validqueries+=int(qvalid.sum());missing+=not bool(detected[row])
            if good.sum()>=2:
                lo=np.maximum(q[good].min(0),[0,0]);hi=np.minimum(q[good].max(0),[w-1,h-1]);hasbox=bool((hi-lo>=2).all())
            else:hasbox=False
            assert bool(part['target_context_valid'][j])==hasbox
            if hasbox:
                box=np.r_[lo,hi];center=(lo+hi)/2;side=float(1.25*np.max(hi-lo))
                assert np.array_equal(part['query_box_xyxy'][j],box) and np.array_equal(part['target_crop_center_xy'][j],center) and part['target_crop_side'][j]==side
                expect=np.zeros((8,2),np.float32);expect[finite]=((q[finite]-center)/side).astype(np.float32);assert np.array_equal(part['initial8_norm'][j],expect)
            else:assert not part['target_rgb128'][j].any() and not part['target_valid128'][j].any()
            assert np.array_equal(part['initial8_norm_valid'][j],finite) and np.isfinite(part['context42'][j]).all()
            assert bool(part['detected'][j])==bool(detected[row])
            if not detected[row]:assert not part['query_valid'][j].any() and not part['p3_roi'][j].any()
            frames+=1
        # First frame of every fixed chunk, independent native scalar sampling.
        i=desc['start'];r=selected[i];rgb=np.array(Image.open(r['image']).convert('RGB'))[100:-100,100:-100].copy();assert list(rgb.shape[:2])==r['raw_shape_hw'];sample_frames.append(i)
        for k,q in enumerate(part['query8_xy'][0]):
            if not np.isfinite(q).all():continue
            for yy,xx in [(0,0),(17,61),(47,48),(73,19),(95,95)]:
                expected,valid=scalar_pixel(rgb,q,part['L'][0],96,yy,xx)
                assert np.array_equal(expected,part['rgb96'][0,k,:,yy,xx]) and valid==part['crop_valid96'][0,k,0,yy,xx];scalar_channels+=3
        if part['target_context_valid'][0]:
            for yy,xx in [(0,0),(23,82),(63,64),(91,27),(127,127)]:
                expected,valid=scalar_pixel(rgb,part['target_crop_center_xy'][0],part['target_crop_side'][0],128,yy,xx)
                assert np.array_equal(expected,part['target_rgb128'][0,:,yy,xx]) and valid==part['target_valid128'][0,0,yy,xx];target_channels+=3
    assert frames==5120 and validqueries==manifest['query_valid_count'] and missing==manifest['missing_detections_retained']
    write(result,dict(completed=True,PASS=True,code=bind(__file__),pack_seal=bind(PACK/'PACK_SEAL.json'),all5120_affines_q8_L_native_indices_target_box_square_norm_and_validity_checked=True,
        full_chunk_SHA_verified=len(manifest['chunks']),frames=frames,queries=40960,query_valid=validqueries,missing_detection_retained=int(missing),
        actual_scalar_sample_frames=sample_frames,local_RGB96_scalar_channels=scalar_channels,target_RGB128_scalar_channels=target_channels,scalar_pixel_differences=0,prepared_pad_removed_once=True,
        P3_ROI_all_finite_and_native_invalid_cells_zero=True,GT_matched_label_or_P4_reads=0,new_backbone_model_forward_training_current_PnP=0))
    integration=dict(completed=True,pack_root=str(PACK),frame_axis=5120,query_axis_per_frame=8,flatten_query_order='frame_index*8+native_index',
        frame_inputs=['target_rgb128 F3x128x128 uint8','target_valid128 F1x128x128 bool','initial8_norm F8x2 float32','initial8_norm_valid F8 bool','query_box_xyxy F4 float64','L F float64'],
        query_inputs=['p3_roi F8x65x9x9 float16','rgb96 F8x3x96x96 uint8','crop_valid96 F8x1x96x96 bool','native_index F8 int8','context42 F8x42 float32','query8_xy_safe F8x2 float32','query_valid F8 bool'],
        raw_query_audit='query8_xy is original inverse-affine raw native q, can be NaN for missing detection. Never forward NaN blindly; use safe coordinates+validity or fixed context42/normalized set.',
        lineage='INDEX_LINEAGE_INPUT_ONLY.jsonl frame_index/cache_row/id/source/partition are loader/sampler metadata, not model ID features. GT is a future separate sameID join.',
        loader_recipe='Read PACK_MANIFEST chunks in start/end order. Concatenate requested fields along frame axis. Querybatch target_inverse=floor(flatqueryindex/8); encode unique frame targetRGB once, keep native_index=flatqueryindex%8. No GT-based query selection.',
        source_partition_counts=manifest['source_partition_counts'],NPZ_disk_bytes=manifest['NPZ_disk_bytes'],uncompressed_bytes=manifest['uncompressed_bytes'],missing_detection_U_policy='Keep rows and mask unknown labels separately; no deletion or automaticE/O label.',
        model_teacher_training_or_current_inference_performed=False)
    write(HERE/'INPUT_PACK_INTEGRATION.json',integration)
    write(HERE/'INPUT_PACK_QA_SEAL.json',dict(completed=True,PASS=True,pack_seal=bind(PACK/'PACK_SEAL.json'),QA=bind(result),integration=bind(HERE/'INPUT_PACK_INTEGRATION.json'),code=bind(__file__)))
    print(json.dumps(dict(QA=bind(result),seal=bind(HERE/'INPUT_PACK_QA_SEAL.json'),local_scalar_channels=scalar_channels,target_scalar_channels=target_channels)))


if __name__=='__main__':main()
