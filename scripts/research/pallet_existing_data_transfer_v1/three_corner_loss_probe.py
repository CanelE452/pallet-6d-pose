import sys,json,types,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26

torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
root=Path(__file__).resolve().parents[3]
base=root/'data/pallet/results/pallet_existing_data_transfer_v1'
fit=json.load(open(root/'_docs/experiments/pallet_existing_data_transfer_v1/FIT_T2_HARD_MANUAL.json'))
checkpoint=root/fit['checkpoint']['path']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==fit['checkpoint']['sha256']
model=YOLO(str(checkpoint),task='pose').model.float().cuda().eval()
for p in model.parameters():p.requires_grad_(False)
model.args=get_cfg(overrides=model.args)
print('MODEL',type(model).__name__,'hyp',dict(pose=model.args.pose,rle=model.args.rle,epochs=model.args.epochs),'flow',model.model[-1].flow_model is not None,flush=True)
# Only head output mode changes, not any BN/training weights. No optimizer exists.
model.model[-1].training=True
all_occ=json.load(open(base/'OCCURRENCES.json'))
occ=[r for r in all_occ if r['role']=='REPLACEMENT' and r['pair']==0]
chosen=[occ[0],next(r for r in occ if r['epoch']==4 and not r['plans']['M']['applied'])]
results=[]
for r in chosen:
    b=r['cache']['M'];path=root/b['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==b['sha256']
    with np.load(path) as z:
        batch={k:torch.from_numpy(z[k].copy()).cuda() for k in ('img','keypoints','bboxes','cls','batch_idx')}
    batch['img']=batch['img'][None].float()/255
    with torch.no_grad():raw=model(batch['img'])
    for branch in ('one2many','one2one'):
        preds=raw[branch]
        preds['kpts']=preds['kpts'].detach().requires_grad_(True)
        preds['kpts_sigma']=preds['kpts_sigma'].detach().requires_grad_(True)
        criterion=TrueIgnorePoseLoss26(model,tal_topk=10 if branch=='one2many' else 7,tal_topk2=None if branch=='one2many' else 1)
        orig=criterion.calculate_keypoints_loss;record=dict(occ=r['occ'],epoch=r['epoch'],branch=branch)
        def capture(self,masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts):
            out=orig(masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts)
            selected=self._select_target_keypoints(keypoints,batch_idx,target_gt_idx,masks)
            selected[...,:2]/=stride_tensor.view(1,-1,1,1)
            gt=selected[masks];pred=pred_kpts[masks];area=((target_bboxes/stride_tensor)[masks][:,2:]-(target_bboxes/stride_tensor)[masks][:,:2]).prod(1)
            e=((pred[...,:2]-gt[...,:2])**2).sum(-1)/((2*self.keypoint_loss.sigmas)**2*area[:,None]*2)
            gradloc=torch.autograd.grad(out[0],pred_kpts,retain_graph=True)[0][masks][...,:2]
            gradrle=torch.autograd.grad(out[2],pred_kpts,retain_graph=True,allow_unused=True)[0]
            gradrle=gradrle[masks][...,:2] if gradrle is not None else torch.zeros_like(gradloc)
            stride=stride_tensor.view(1,-1,1).expand(masks.shape[0],-1,-1)[masks]
            rawrle=(pred[...,:2]-gt[...,:2])/(pred[...,-2:].sigmoid()+1e-9)
            record.update(positive_anchors=int(masks.sum()),location_loss=float(out[0].detach()),rle_loss=float(out[2].detach()),sigmas=self.keypoint_loss.sigmas.detach().cpu().tolist(),points={})
            for j in (0,3,4):
                record['points'][j]=dict(supervised=int((gt[:,j,2]==2).sum()),
                    cache_pixel_errors=((pred[:,j,:2]-gt[:,j,:2]).norm(dim=-1)*stride[:,0]).detach().cpu().tolist(),
                    oks_exponent=e[:,j].detach().cpu().tolist(),exp_minus_e=torch.exp(-e[:,j]).detach().cpu().tolist(),
                    location_gradient_norm=gradloc[:,j].norm(dim=-1).detach().cpu().tolist(),
                    rle_gradient_norm=gradrle[:,j].norm(dim=-1).detach().cpu().tolist(),
                    rle_abs_normalized_error=rawrle[:,j].abs().detach().cpu().tolist(),
                    rle_clamp_fraction=float((rawrle[:,j].abs()>100).float().mean().detach()))
            return out
        criterion.calculate_keypoints_loss=types.MethodType(capture,criterion)
        loss,items=criterion.loss(preds,batch)
        record['weighted_loss_items']=items.cpu().tolist()
        results.append(record)
        print(json.dumps(record),flush=True)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==fit['checkpoint']['sha256']
artifact=dict(protocol="Final checkpoint probe on two frozen training inputs; not historical training gradients. BN eval; head raw-output mode only. No optimizer or weight update.",checkpoint=fit['checkpoint'],probe_hyp=dict(pose=model.args.pose,rle=model.args.rle,epochs=model.args.epochs),records=results,checkpoint_unchanged=True)
(root/'_docs/experiments/pallet_existing_data_transfer_v1/THREE_CORNER_LOSS_PROBE.json').write_text(json.dumps(artifact,indent=2)+'\n')
print('NO_OPTIMIZER_NO_WEIGHT_UPDATE',flush=True)
