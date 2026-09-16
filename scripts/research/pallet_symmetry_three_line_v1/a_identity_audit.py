"""Actual 16-image loader/head/stock identity-gradient audit; optimizer updates ZERO."""
import copy
import cv2,torch,numpy as np
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss,PoseLoss26,RLELoss
import env as E
from a_loss import GenericSymmetryPoseLoss

def batch(records):
    images=[];boxes=[];points=[];classes=[];batch_idx=[]
    for i,r in enumerate(records):
        im=cv2.imread(str(E.ROOT/r['image']));h,w=im.shape[:2];gain=min(640/h,640/w)
        nw,nh=round(w*gain),round(h*gain);left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
        im=cv2.resize(im,(nw,nh));im=cv2.copyMakeBorder(im,top,640-nh-top,left,640-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
        images.append(im[:,:,::-1].transpose(2,0,1).copy())
        a=np.array((E.ROOT/r['label']).read_text().split(),float).reshape(-1,32)
        for row in a:
            bb=row[1:5].copy();bb[:2]=(bb[:2]*[w,h]*gain+[left,top])/640;bb[2:]*=np.array([w,h])*gain/640
            kp=row[5:].reshape(9,3).copy();kp[:,:2]=(kp[:,:2]*[w,h]*gain+[left,top])/640
            assert np.isfinite(kp).all()
            boxes.append(bb);points.append(kp);classes.append([row[0]]);batch_idx.append(i)
    return dict(img=torch.tensor(np.stack(images),device='cuda').float()/255,
       bboxes=torch.tensor(np.array(boxes),device='cuda',dtype=torch.float32),keypoints=torch.tensor(np.array(points),device='cuda',dtype=torch.float32),
       cls=torch.tensor(classes,device='cuda',dtype=torch.float32),batch_idx=torch.tensor(batch_idx,device='cuda',dtype=torch.float32))
def main():
    E.gpu();torch.set_num_threads(4);cv2.setNumThreads(1);assert torch.cuda.is_available()
    model=YOLO(str(E.R0)).model.cuda().float();model.args=get_cfg();model.requires_grad_(True).train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    square=E.read(E.RAW/'A/square_membership.json')['train']['records'][:16]
    source=E.read(E.C.LINE/'SOURCE_MANIFEST.json')['records'];rect=[r for r in source if r['source_split']=='train'][:16]
    results=[]
    for cohort,records in [('RECT',rect),('SQUARE',square)]:
        b=batch(records)
        with torch.no_grad():preds=model(b['img'])
        # Test exact gradients at actual head outputs + flow-model parameters.
        def leaves(d):
            if isinstance(d,dict):return {k:leaves(v) for k,v in d.items()}
            if isinstance(d,list):return [leaves(v) for v in d]
            if torch.is_tensor(d):return d.detach().clone().requires_grad_(d.is_floating_point())
            return d
        p=leaves(preds)
        def tensors(d):
            if isinstance(d,dict):return sum([tensors(v) for v in d.values()],[])
            if isinstance(d,list):return sum([tensors(v) for v in d],[])
            return [d] if torch.is_tensor(d) and d.requires_grad else []
        parameters=tensors(p)+list(model.model[-1].flow_model.parameters())
        stock=E2ELoss(model,PoseLoss26);adapt=GenericSymmetryPoseLoss(model)
        old=stock(p,b)[0].sum();ga=torch.autograd.grad(old,parameters,allow_unused=True,retain_graph=True)
        # Deliberately scrambled GT rows. Stable adapter sorting restores assignment and masks.
        shuffled=dict(b);order=torch.arange(len(b['batch_idx'])-1,-1,-1,device='cuda')
        for key in ['batch_idx','cls','bboxes','keypoints']:shuffled[key]=b[key][order]
        new=adapt(p,shuffled)[0].sum();gb=torch.autograd.grad(new,parameters,allow_unused=True)
        errors=[float((a-z).abs().max()) for a,z in zip(ga,gb) if a is not None and z is not None]
        assert torch.equal(old,new) and max(errors,default=0)==0
        assert all((a is None)==(z is None) for a,z in zip(ga,gb))
        results.append(dict(cohort=cohort,images=16,GT_rows=len(b['batch_idx']),loss=float(old),
          loss_bit_exact=True,gradient_max_abs=max(errors,default=0),both_heads_present=set(preds)>=set(['one2many','one2one']),
          shuffled_GT_rows_verified=True,head_weights=[stock.o2m,stock.o2o]))
        print('A IDENTITY',results[-1],flush=True)
        del preds,p,parameters,ga,gb,b;torch.cuda.empty_cache()
    # Actual installed RLE reduction demonstrates cross-object clamp nonseparability.
    fn=RLELoss();sig=torch.full((2,2),.5);err=torch.zeros(2,2);phi=torch.tensor([1.,-3.]);weights=torch.ones(2)
    together=fn(sig,phi,err,weights).clamp(min=0)
    independently=sum(fn(sig[i:i+1],phi[i:i+1],err[i:i+1],weights[i:i+1]).clamp(min=0)/2 for i in range(2))
    assert together!=independently
    E.write(E.DOC/'A/IDENTITY_AND_RLE_AUDIT.json',dict(PASS=True,identity=results,actual_detector_forwards=32,
      optimizer_updates=0,RLE_global=float(together),RLE_per_object_clamped=float(independently),
      stock_RLE_not_object_separable=True,equivariant_loss_tests='NOT_RUN_PENDING_OBJECTIVE_DECISION',
      no_A_training_claim=True,source_loss_sha256=E.sha(__import__('ultralytics.utils.loss',fromlist=['']).__file__)))
if __name__=='__main__':main()
