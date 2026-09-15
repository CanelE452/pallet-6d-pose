import numpy as np
import torch
from env import *
from prior_model import TFAdam,target_distribution,expectation
def run():
    r=np.load(RAW/'tf_train_ops.npz');bn=torch.nn.BatchNorm2d(3,eps=1.001e-5,momentum=.01);bn.train();bn_errors=[]
    for i in range(3):
        inp=torch.from_numpy(r['bn_input']+i*.1).permute(0,3,1,2)
        out=bn(inp).detach().permute(0,2,3,1).numpy();np.testing.assert_allclose(out,r['bn_output'][i],atol=1e-5,rtol=1e-4)
        stats=np.stack([bn.running_mean.numpy(),bn.running_var.numpy()]);np.testing.assert_allclose(stats,r['bn_stats'][i],atol=1e-5,rtol=1e-4)
        bn_errors.append(float(np.abs(stats-r['bn_stats'][i]).max()))
    p=torch.nn.Parameter(torch.from_numpy(r['adam_initial']).clone());optimizer=TFAdam([p]);errs=[]
    for i,g in enumerate(r['adam_gradients']):
        p.grad=torch.from_numpy(g).clone();optimizer.step();np.testing.assert_allclose(p.detach().numpy(),r['adam_states'][i],atol=1e-5,rtol=1e-4);errs.append(float(np.abs(p.detach().numpy()-r['adam_states'][i]).max()))
    z=torch.tensor(r['logits'].transpose(0,3,1,2),requires_grad=True);target=torch.from_numpy(r['target']);valid=torch.from_numpy(r['target_valid'].astype(bool));q,m=target_distribution(target,valid)
    np.testing.assert_allclose(q[:,:8].numpy(),r['target_distribution'].transpose(0,3,1,2).reshape(1,9,-1)[:,:8],atol=1e-5,rtol=1e-4)
    ce=(-(q*z.flatten(2).log_softmax(-1)).sum(-1)*m).mean();co=((expectation(z)/4-target/4).abs()*m[...,None]).mean()
    (ce+co).backward();np.testing.assert_allclose([ce.item(),co.item()],[r['heatmap_loss'],r['coordinate_loss']],atol=1e-5,rtol=1e-4)
    np.testing.assert_allclose(z.grad.numpy().transpose(0,2,3,1),r['logit_grad'],atol=1e-5,rtol=1e-4)
    write(DOC/'TRAIN_OPERATOR_PARITY.json',dict(complete=True,BN_steps=3,BN_moving_stats_max_abs=bn_errors,BN_updates_per_forward=1,activation_checkpointing=False,Adam_steps=5,Adam_parameter_max_abs=errs,bilinear_interior_mass=True,CE_coordinate_reduction=True,logit_gradient=True,atol=1e-5,rtol=1e-4,L2='0.5*1e-5*sum(conv_weights^2); excluded BN/bias; official Trainer include_wd=True',reference=bound(RAW/'tf_train_ops.npz')))
