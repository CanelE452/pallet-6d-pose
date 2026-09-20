"""A fixed 300-update trainability sanity check, not a new final model."""
import copy
import time
import numpy as np
import cv2
import torch
from scripts.research.pallet_posefix_large_error_v1 import core as C
from scripts.research.pallet_sensors_submission_v1.prior_model import TFAdam, expectation


def items():
    result = []
    for r in C.E.read(C.R.DOC/'SPLIT.json')['train']:
        for key in ('image','annotation','cache'): C.F.verify(r[key])
        im = cv2.imread(str(C.ROOT/r['image']['path'])); assert im is not None
        cache = torch.load(C.ROOT/r['cache']['path'],map_location='cpu',weights_only=False)['captured']
        pred = dict(candidates=cache['candidates'],selected_index=cache['selected_index'])
        x = C.prepare_input(im,pred); assert x is not None
        gt,valid = C.R.manual_target(C.E.read(C.ROOT/r['annotation']['path']))
        assert valid.tolist() == r['manual_mask']
        target = C.transform_points(np.where(valid[:,None],gt,0),x['matrix']).astype(np.float32)
        support = valid & (target>=0).all(-1) & (target[:,0]<288) & (target[:,1]<384)
        support[8] = False
        x.update(id=r['id'],target=target,target_valid=support,original_gt=gt,manual_mask=valid)
        result.append(x)
    assert len(result)==9 and sum(int(x['manual_mask'].sum()) for x in result)==38
    C.freeze(C.DOC/'TRAIN_SUPPORT.json',dict(records=[dict(id=x['id'],manual=x['manual_mask'].tolist(),
        crop_supported=x['target_valid'].tolist(),matrix=x['matrix'].tolist()) for x in result],
        manual_corners=38,crop_supported_corners=sum(int(x['target_valid'].sum()) for x in result),
        no_eval_images=True, centroid_supervised=False))
    return result


def corrupted(item,rng,force=False):
    x = dict(item); x['points'] = item['points'].copy(); x['valid'] = item['valid'].copy()
    if force or rng.random() < .5:
        mask = item['target_valid']
        n = int(mask.sum()); angles=rng.uniform(0,2*np.pi,n)
        radius=rng.uniform(.05,.15,n)*item['bbox_diagonal']
        delta=np.stack((np.cos(angles),np.sin(angles)),-1)*radius[:,None]
        # isotropic axis-aligned crop maps a vector without translation.
        x['points'][mask]=item['target'][mask]+(delta @ item['matrix'][:2,:2].T).astype(np.float32)
        x['valid'][mask]=True
    return x


def batch(rows):
    return {k:torch.as_tensor(np.stack([r[k] for r in rows]),device='cuda')
            for k in ('rgb','points','valid','target','target_valid')}


def error_summary(inputs,outputs):
    a,b=np.array(inputs),np.array(outputs); hard=a>20
    return dict(corners=len(a),input_mean_px=float(a.mean()),output_mean_px=float(b.mean()),
        output_median_px=float(np.median(b)),PCK10=float((b<=10).mean()),hard=int(hard.sum()),
        recovered=int((hard&(b<=10)).sum()),recovery_rate=float((b[hard]<=10).mean()) if hard.any() else None)


@torch.no_grad()
def probe(model,source):
    model.eval(); rng=np.random.default_rng(6402)
    result={}
    for mode in ('original_R0','noisy_manual'):
        rows=source if mode=='original_R0' else [corrupted(x,rng,True) for x in source for _ in range(8)]
        inputs=[]; outputs=[]
        for j in range(0,len(rows),2):
            bb=rows[j:j+2]; b=batch(bb)
            q=expectation(model(b['rgb'],b['points'],b['valid'])).cpu().numpy()
            for x,p in zip(bb,q):
                mask=x['target_valid']; scale=x['matrix'][0,0]
                assert abs(scale-x['matrix'][1,1])<1e-10
                inputs.extend((np.linalg.norm(x['points'][mask]-x['target'][mask],axis=-1)/scale).tolist())
                outputs.extend((np.linalg.norm(p[mask]-x['target'][mask],axis=-1)/scale).tolist())
        result[mode]=error_summary(inputs,outputs)
    return result


def run():
    protocol=C.E.read(C.DOC/'PROTOCOL.json'); C.verify_bindings(protocol['sources'])
    for b in protocol['code'].values(): C.F.verify(b)
    assert not (C.DOC/'FIT.json').exists(), 'Completed experiment is immutable'
    C.E.gpu(); assert torch.cuda.is_available()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    source=items(); model=C.load_model(); before=probe(model,source)
    C.freeze(C.DOC/'TRAIN_PROBE_BEFORE.json',before)
    model.requires_grad_(True); optimizer=TFAdam(model.parameters(),lr=1e-4)
    rng=np.random.default_rng(6401); history=[]; start=time.monotonic()
    # Keep BN population statistics of the synthetic pretrained model fixed.
    model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.BatchNorm2d): m.eval()
    bn_before={k:v.detach().clone() for k,v in model.named_buffers()}
    for step in range(1,301):
        sampled=rng.integers(0,len(source),8)
        rows=[corrupted(source[int(i)],rng) for i in sampled]
        optimizer.zero_grad(set_to_none=True); parts=[]
        for j in range(0,8,2):
            b=batch(rows[j:j+2]); logits=model(b['rgb'],b['points'],b['valid'])
            loss,detail=model.losses(logits,b['target'],b['target_valid'])
            assert torch.isfinite(loss)
            (loss/4).backward(); parts.append({k:float(v.detach()) for k,v in detail.items()})
            del b,logits,loss,detail
        norm=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
        assert np.isfinite(norm)
        update=float(optimizer.step()); assert np.isfinite(update) and update>0
        rec=dict(step=step,gradient_norm=norm,update_norm=update,
                 loss={k:float(np.mean([p[k] for p in parts])) for k in parts[0]})
        history.append(rec)
        if step==1 or step%50==0:
            status=dict(stage='TRAINING',**rec,elapsed_seconds=time.monotonic()-start,gpu=C.E.gpu())
            C.write(C.DOC/'STATUS.json',status); print(status,flush=True)
    assert all(torch.equal(bn_before[k],v) for k,v in model.named_buffers())
    after=probe(model,source); noisy=after['noisy_manual']
    passed=bool(noisy['output_mean_px']<=10 and noisy['recovery_rate'] is not None and
                noisy['recovery_rate']>=.5 and noisy['output_mean_px']<=.5*noisy['input_mean_px'])
    C.RAW.mkdir(parents=True,exist_ok=True)
    path=C.RAW/'last300.pt'
    assert not path.exists()
    torch.save(dict(complete=True,step=300,protocol_sha256=C.E.sha(C.DOC/'PROTOCOL.json'),
                    model_state_dict=model.state_dict()),path)
    C.freeze(C.RAW/'TRAIN_STEPS.json',history)
    C.freeze(C.DOC/'FIT.json',dict(complete=True,step=300,checkpoint=C.E.bound(path),before=before,after=after,
        trainability_pass=passed,elapsed_seconds=time.monotonic()-start,BN_buffers_bit_exact=True,
        train_frames=9,manual_corners=38,trainability_only=True,independent_test=False,auto_promoted=False))
    C.write(C.DOC/'STATUS.json',dict(stage='SANITY_COMPLETE',trainability_pass=passed))
    print('TRAINABILITY_PASS',passed,'BEFORE',before,'AFTER',after,flush=True)


if __name__=='__main__': run()
