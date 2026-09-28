"""Private image checks: fixed TRAIN only, never a new RGB publication."""
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
from . import common as C
from .occlusion import SharedOcclusion, reference_labels
from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine import make_dataset, protocol, seed

def main():
    dst=C.RAW/'INPUT_PIXEL_EXAMPLES.json'
    if dst.exists(): print('INPUT_EXAMPLES_REUSED'); return
    rows=C.read(C.RAW/'OCCLUSION_PREFLIGHT_PLASTIC_PRIVATE.json')
    selected=[r for r in rows if r['role']=='REAL' and r['arms']['REF']['plan']['applied']][:3]
    _,p=protocol('PLASTIC'); ds=make_dataset('PLASTIC','REF'); original=copy.deepcopy(ds.transforms)
    ds.transforms=SharedOcclusion(ds.transforms,reference_labels(p,C.ROOT)); out=[]
    for j,row in enumerate(selected):
        index=row['index']; seed(280901+index)
        before=original(copy.deepcopy(ds.get_image_and_label(index)))
        seed(280901+index); after=ds[index]
        assert torch.equal(before['keypoints'],after['keypoints'])
        l,t,w,h=after['occlusion_info']['rectangle']
        diff=(before['img']!=after['img']).any(0).numpy()
        outside=diff.copy(); outside[t:t+h,l:l+w]=False; assert not outside.any() and diff.any()
        images=[cv2.resize(cv2.imread(ds.im_files[index]),(640,640)),
                before['img'].permute(1,2,0).numpy()[:,:,::-1],after['img'].permute(1,2,0).numpy()[:,:,::-1]]
        canvas=np.concatenate(images,axis=1)
        for x,label in enumerate(('Stored padded TRAIN (display resized)','Original augmentation','Shared RGB-only occlusion')):
            cv2.putText(canvas,label,(640*x+10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        dest=C.RAW/'private_input_examples'/f'input_{j+1}.png'; dest.parent.mkdir(parents=True,exist_ok=True)
        assert cv2.imwrite(str(dest),canvas)
        out.append(dict(image_name=Path(ds.im_files[index]).name,example=C.bind(dest),
            changed_pixels=int(diff.sum()),only_rectangle_changed=True,targets_identical=True,
            plan=after['occlusion_info'],publication='PRIVATE_NOT_STAGED'))
    C.save(dst,dict(examples=out,source='fixed preflight TRAIN exposures',fits=0,updates=0),True)
    print('PIXEL_CHECKS_PASS',len(out))

if __name__=='__main__': main()
