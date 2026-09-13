"""No-update GPU wiring audit; never counted as a trained student."""
import torch
import reweight as W

def main():
    W.check_gpu();torch.set_num_threads(4);torch.manual_seed(20260913)
    m=W.S.load_model().cuda();m.args=W.get_cfg(overrides=W.HYP);m.train()
    for mod in m.modules():
        if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
    before=W.S.tensor_sha(m.state_dict());criterion=m.init_criterion();params=[p for _,p in W.proxy_parameters(m)]
    batch=W.S.device_batch(next(iter(W.loader('real',1,0))))
    meta=W.S.device_batch(next(iter(W.loader('meta',1,0))))
    gm=W.flat_grad(criterion(m(meta['img']),meta)[0].sum()/4,params)
    pred=m(batch['img'])
    losses=torch.stack([criterion(W.slice_predictions(pred,i),W.single_target(batch,i))[0].sum() for i in range(8)])
    alignment=torch.stack([torch.dot(W.flat_grad(l,params,True),gm) for l in losses])
    values={kind:W.weights(kind,losses,alignment).cpu().tolist() for kind in W.METHODS}
    losses.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert W.S.tensor_sha(m.state_dict())==before
    W.write(W.DOC/'reweight/SMOKE.json',dict(status='PASS',optimizer_updates=0,state_unchanged=True,
        losses=losses.detach().cpu().tolist(),alignment=alignment.cpu().tolist(),weights=values,
        proxy_parameter_count=sum(p.numel() for p in params)))
    print('GPU loss/gradient wiring PASS;0 updates',flush=True)

if __name__=='__main__':main()
