# Observation extractor inherited unchanged except fresh test-only output.
"""Extract observation-only inputs; GT is stored separately for scoring/training."""
from common import *
import numpy as np,torch,cv2
import torch.nn.functional as F

def main():
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    torch.set_num_threads(3);cv2.setNumThreads(1)
    assert sha(CKPT)==register()['upstream_checkpoint_sha256']
    rows=read(OUT/'RENDER_COMPLETE.json')['rows'];m=YOLO(str(CKPT)).model.float().cuda().eval();head=m.model[-1]
    captured={};hooks=[mod.register_forward_hook(lambda module,args,out,i=i:captured.__setitem__(i,out.detach())) for i,mod in enumerate(head.one2one_cv4)]
    values=[];records=[];parity=[]
    def sample(plane,pixels):
        grid=(pixels/640*2-1)[None,:,None,:]
        return F.grid_sample(plane,grid,align_corners=False,padding_mode='zeros')[0,:,:,0].T
    for n,r in enumerate(rows):
        im=cv2.imread(str(ROOT/r['image']));h,w=im.shape[:2]
        # Same 100-pixel reflected border as this pretrained model's preparation.
        im=cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101)
        ih,iw=im.shape[:2];gain=min(640/ih,640/iw)
        offset=np.array([round((640-round(iw*gain))/2-.1),round((640-round(ih*gain))/2-.1)])
        x=LetterBox((640,640),auto=False)(image=im)
        x=torch.from_numpy(x[:,:,::-1].transpose(2,0,1).copy()).float().cuda()[None]/255
        with torch.no_grad():
            full=m(x);raw=full[1]['one2one'];decoded=head._inference(raw);allpoints=head.kpts_decode(raw['kpts'])
            conf,j=raw['scores'][0,0].sigmoid().max(0);j=int(j)
            box=decoded[0,:4,j];p=allpoints[0,:,j].reshape(9,3)
            sizes=[a.shape[-2]*a.shape[-1] for a in captured.values()];level=int(np.searchsorted(np.cumsum(sizes),j,side='right'))
            center=(box[:2]+box[2:])/2;size=(box[2:]-box[:2]).clamp_min(16)
            yy,xx=torch.meshgrid(torch.linspace(-.625,.625,7,device='cuda'),torch.linspace(-.625,.625,7,device='cuda'),indexing='ij')
            roi_xy=torch.stack([xx,yy],-1).reshape(-1,2)*size+center
            point_feat=sample(captured[level],p[:,:2]);roi_feat=sample(captured[level],roi_xy)
            point_coord=(p[:,:2]-center)/size
            # Verify selected outputs against the public end-to-end top detection.
            public=full[0][0,0]
            parity.append(float((public[6:].reshape(9,3)-p).abs().max()))
            stock=(p[:,:2].cpu().numpy()-offset)/gain-100
            values.append(dict(point_feat=point_feat.cpu().numpy(),roi_feat=roi_feat.cpu().numpy(),
                point_coord=point_coord.cpu().numpy(),roi_coord=((roi_xy-center)/size).cpu().numpy(),
                visibility=p[:,2].cpu().numpy(),stock=stock.astype('float32'),dims=np.array(r['dims_xyz'],dtype='float32')))
            records.append(dict(r,confidence=float(conf),pred_box=((box.cpu().numpy()-np.tile(offset,2))/gain-100).tolist()))
        if (n+1)%128==0:print('CACHE',n+1,len(rows),flush=True)
    for h in hooks:h.remove()
    assert max(parity)<.002,max(parity)
    for split in ['test']:
        ix=[i for i,r in enumerate(records) if r['split']==split]
        z={k:torch.tensor(np.stack([values[i][k] for i in ix])) for k in values[0]}
        torch.save(z,OUT/f'{split}.pt');write(OUT/f'{split}_records.json',[records[i] for i in ix])
    write(OUT/'CACHE_COMPLETE.json',dict(count=len(rows),public_decode_max_abs= max(parity),
        input_keys=list(values[0]),GT_model_inputs=False,source_sha256=sha(KIT/'cache.py'),
        padding='100px reflect101, then640letterbox; held fixed for all arms'))
if __name__=='__main__':main()
