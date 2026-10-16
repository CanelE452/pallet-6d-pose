"""Native RGB ROI and paired renderer silhouette distance; supervision separate."""
from context17 import *
import cv2

def main():
    setup();cv2.setNumThreads(1);RUN.mkdir(exist_ok=True)
    assert not (RUN/'PREPARATION_COMPLETE.json').exists()
    probe={r['merged_stem']:r for r in map(json.loads,(ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl').open())}
    records=[];mask_audit=[];verified=set()
    for split in ('heldout','synth_val','real'):
        for condition in (TRAIN_CONDITIONS if split=='heldout' else CONDITIONS):
            m=read(C.RUN/'cache/heldout'/condition/'manifest.json') if split=='heldout' else manifest(split,condition)
            rgbs=[];ds=[];has=[];ids=[]
            for r in m['rows']:
                if not r['detected']:continue
                ip=ROOT/r['image']
                if str(ip) not in verified:assert sha(ip)==r['image_sha256'];verified.add(str(ip))
                raw=PREP.raw_image(r);canvas=raw.copy()
                if r['mask'] is not None:
                    x,y,z,w=r['mask'];canvas[y:w,x:z]=r['fill_bgr']
                if condition.startswith('crop'):canvas=canvas[:,r['cut']:] if r['crop_side']=='left' else canvas[:,:r['cut']]
                assert list(canvas.shape[:2])==r['image_hw']
                bb=np.array(r['box']);xx,yy=np.meshgrid(np.linspace(bb[0],bb[2],96),np.linspace(bb[1],bb[3],96));xx=xx.astype(np.float32);yy=yy.astype(np.float32)
                rgb=cv2.remap(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB),xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128))
                rgbs.append(torch.from_numpy(rgb.transpose(2,0,1).copy()));ids.append(r['id'])
                if split=='real' or (split=='synth_val' and condition!='clean'):continue
                pr=probe[r['id']];mp=None
                if pr['source']=='G38':
                    label=ROOT/pr['renderer_label_locator'];mp=label.parent.parent/'mask_visible'/(label.name.replace('_label.json','.png'))
                available=mp is not None and mp.exists();has.append(available)
                if available:
                    mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE);assert mask.shape==raw.shape[:2];binary=(mask>0).astype(np.uint8)
                    contour=cv2.morphologyEx(binary,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));dist=cv2.distanceTransform(1-contour,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                    # Original object silhouette only: artificial crop/occluder borders are not positive labels.
                    sample=cv2.remap(dist,xx+r['origin'][0],yy+r['origin'][1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=10000)
                    ds.append(torch.from_numpy(sample.copy()))
                    if condition=='clean':mask_audit.append(dict(split=split,id=r['id'],path=str(mp.relative_to(ROOT)),sha256=sha(mp),shape=list(mask.shape),positive_pixels=int(binary.sum())))
                else:ds.append(torch.full((96,96),10000.))
            d=RUN/'rgb'/split;d.mkdir(parents=True,exist_ok=True);path=d/f'{condition}.pt';torch.save(dict(ids=ids,rgb=torch.stack(rgbs)),path)
            if ds:
                d=RUN/'supervision'/split;d.mkdir(parents=True,exist_ok=True);torch.save(dict(ids=ids,boundary=torch.stack(ds),has_mask=torch.tensor(has)),d/f'{condition}.pt')
            records.append(dict(split=split,condition=condition,detected=len(ids),mask_available=sum(has),rgb_sha256=sha(path)))
            print('PREPARED',split,condition,len(ids),'masks',sum(has),flush=True)
    write(RUN/'MASK_PROVENANCE.json',dict(rows=mask_audit,scope='Renderer visible object silhouette, not internal face/board edges. G38 only; other sources not silently assigned masks. Original image hashes and dimensions verified.'))
    write(RUN/'PREPARATION_COMPLETE.json',dict(complete=True,rows=records,unique_images_hashed=len(verified),rgb_no_gt=True))

if __name__=='__main__':main()
