"""Correct mask decoding, preserve invalid run, reuse identical non-GT RGB inputs."""
from context17 import *
import cv2
import shutil

def main():
    setup();cv2.setNumThreads(1);RUN.mkdir(exist_ok=True);old=ROOT/'data/pallet/results/pallet_detail_v17'
    assert not (RUN/'PREPARATION_COMPLETE.json').exists()
    prep=read(old/'PREPARATION_COMPLETE.json')
    if not (RUN/'rgb').exists():shutil.copytree(old/'rgb',RUN/'rgb',copy_function=shutil.copy2)
    shutil.copy2(old/'RGB_MASK_PAIR_AUDIT.json',RUN/'RGB_MASK_PAIR_AUDIT.json')
    for r in prep['rows']:assert sha(RUN/'rgb'/r['split']/f"{r['condition']}.pt")==r['rgb_sha256']
    provenance=read(old/'MASK_PROVENANCE.json');lookup={x['id']:x for x in provenance['rows']};area_rows=[];hist=np.zeros(256,np.int64)
    # One frame missed by clean detector reappears under stress; retain its mask.
    for c in TRAIN_CONDITIONS:
        for r in read(C.RUN/'cache/heldout'/c/'manifest.json')['rows']:
            if r['detected'] and r['id'].startswith('G38__') and r['id'] not in lookup:
                mp=ROOT/'data/pallet/training_data/paper_release/v2_prod40k_clean_merged/mask_visible'/(r['id'].split('__')[-1]+'.png')
                rec=dict(id=r['id'],split='heldout',path=str(mp.relative_to(ROOT)),sha256=sha(mp));lookup[r['id']]=rec;provenance['rows'].append(rec)
    for rec in provenance['rows']:
        mp=ROOT/rec['path'];assert sha(mp)==rec['sha256'];mask=cv2.imread(str(mp),0);binary=(mask>127).astype(np.uint8)
        lp=mp.parent.parent/'labels'/(mp.stem+'_label.json');reported=read(lp)['objects'][0]['v2_labels']['mask_area_visible'];area=int(binary.sum());assert area==reported,(rec['id'],area,reported)
        rec['positive_pixels']=area;area_rows.append(dict(id=rec['id'],split=rec['split'],mask_sha256=sha(mp),label_sha256=sha(lp),decoded_area=area,renderer_recorded_area=reported));hist+=np.bincount(mask.ravel(),minlength=256)
    for split in ('heldout','synth_val'):
        for c in (TRAIN_CONDITIONS if split=='heldout' else ('clean',)):
            m=read(C.RUN/'cache/heldout'/c/'manifest.json') if split=='heldout' else manifest(split,c);ids=[];distances=[];has=[]
            for r in m['rows']:
                if not r['detected']:continue
                ids.append(r['id']);available=r['id'] in lookup;has.append(available)
                if not available:distances.append(torch.full((96,96),10000.));continue
                mask=cv2.imread(str(ROOT/lookup[r['id']]['path']),0);binary=(mask>127).astype(np.uint8)
                contour=cv2.morphologyEx(binary,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));dist=cv2.distanceTransform(1-contour,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                bb=r['box'];xx,yy=np.meshgrid(np.linspace(bb[0],bb[2],96),np.linspace(bb[1],bb[3],96));xx=xx.astype(np.float32)+r['origin'][0];yy=yy.astype(np.float32)+r['origin'][1]
                sampled=cv2.remap(dist,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=10000);distances.append(torch.from_numpy(sampled.copy()))
            previous=torch.load(old/'supervision'/split/f'{c}.pt',weights_only=True);assert ids==previous['ids'] and torch.equal(torch.tensor(has),previous['has_mask'])
            d=RUN/'supervision'/split;d.mkdir(parents=True,exist_ok=True);torch.save(dict(ids=ids,boundary=torch.stack(distances),has_mask=torch.tensor(has)),d/f'{c}.pt');print('CORRECTED',split,c,sum(has),flush=True)
    provenance['scope']='Visible object silhouette decoded raw>127; exact renderer area and RGB pairing verified; excludes internal face creases.';write(RUN/'MASK_PROVENANCE.json',provenance)
    write(RUN/'MASK_DECODING_AUDIT.json',dict(status='PASS',threshold=127,source_convention='scripts/stage0/multihead/mh_data.py MASK_THRESHOLD=127',rows=area_rows,value_histogram={str(i):int(n) for i,n in enumerate(hist) if n},all_decoded_areas_exact=True))
    prep.update(correction='raw>127 instead of raw>0; prior result invalid; RGB inputs and mask availability unchanged',supersedes=str(old.relative_to(ROOT)),rgb_reused_exact=True);write(RUN/'PREPARATION_COMPLETE.json',prep)

if __name__=='__main__':main()
