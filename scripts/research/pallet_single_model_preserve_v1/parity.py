import cv2
import numpy as np
import torch
from . import common as C
from .adapter import Model
from scripts.research.pallet_selector_recovery_v1 import features as F,models as M

def main():
    C.setup();gpu=C.gpu();m=Model();rows=C.read(C.RAW/'INFERENCE_INPUTS.json')['source'];ids=C.read(C.DOC/'SANITY32_LOCK.json')['source_ids'];by={r['id']:r for r in rows};ck=torch.load(C.ROOT/m.bindings['scorer']['path'],map_location='cpu',weights_only=False);out=[]
    assert all(torch.count_nonzero(a.net[-1].weight)==0 and torch.count_nonzero(a.net[-1].bias)==0 for a in m.adapter)
    for fid in ids:
        r=by[fid];im=cv2.imread(str(C.ROOT/r['image']['path']));p=m.predict(im,0,False);q=m.predict(im,0,True);assert p['selected_index']==q['selected_index'] and len(p['candidates'])==len(q['candidates']);maxdiff=0.
        for a,b in zip(p['candidates'],q['candidates']):
            assert a['score']==b['score'] and a['box_xyxy']==b['box_xyxy'] and a['keypoints_conf']==b['keypoints_conf'];maxdiff=max(maxdiff,float(np.max(np.abs(np.array(a['keypoints_xy'])-b['keypoints_xy']))))
        assert maxdiff<=1e-6
        ga=F.extract(p,r['K'],r['dims'],im.shape[:2]);gb=F.extract(q,r['K'],r['dims'],im.shape[:2]);assert ga==gb
        if ga['valid']:
            sa=M.scores(ck,np.array(ga['features'],np.float32)[None]);sb=M.scores(ck,np.array(gb['features'],np.float32)[None]);assert np.array_equal(sa,sb)
        out.append(dict(id=fid,xy_max_px=maxdiff,box_class_conf_same=True,D9_same=True,GEO_LINEAR_same=True,selected_detection_same=True))
    # Gradient audit uses only the first locked training batch; no optimizer step.
    from .train import batch_data,losses
    rr=C.read(C.RAW/'OCCURRENCES.json')[:16];x,t,mask,is_occ=batch_data(rr)
    # Inference caches are inference tensors; force fresh normal anchor tensors for autograd.
    m.net.model.model[-1].shape=None
    with torch.no_grad():
        m.enabled=False;stock=m.net.model(x)[0];idx=stock[:,:,4].argmax(1);stock=stock[torch.arange(len(stock),device=stock.device),idx]
        functional,_=m.tensor(x,False);assert torch.equal(stock,functional),'Functional decoder must be bit-exact to stock'
    lc,lp,res,counts=losses(m,x,t,mask,is_occ);lc.backward();grad=sum(float(p.grad.abs().sum()) for p in m.adapter.parameters() if p.grad is not None)
    assert grad>0 and all(p.grad is None for p in m.net.model.parameters());m.integrity()
    C.freeze(C.DOC/'ZERO_INIT_PARITY.json',dict(created_at=C.now(),passed=True,frames=32,rows=out,max_xy_px=max(r['xy_max_px'] for r in out),adapter_gradient_L1=grad,base_gradient_zero=True,
        base_state_unchanged=True,only_adapter_trainable=True,box_class_conf_same=True,zero_last_conv=True,gradient_probe_optimizer_steps=0,adapter_params=sum(p.numel() for p in m.adapter.parameters()),
        injection=[dict(in_channels=a.net[0].in_channels,out_channels=a.net[-1].out_channels,bottleneck=32) for a in m.adapter],gpu=gpu,
        training_decoder='Functional exact Pose26 (raw+anchor)*stride, sigmoid visibility; stock in-place eval decoder has invalid autograd versioning. Stock vs functional outputs bit-exact. Architecture/loss unchanged.',
        functional_decoder_bit_exact=True,initial_failed_attempts='Foreign GPU guard stopped before inference; next attempt passed32 inference parity but stock in-place decoder backward raised, optimizer steps0. Fixed functional decoder before any fit.'))
    print('ZERO_INIT_PARITY_PASS',grad,flush=True)
if __name__=='__main__':main()
