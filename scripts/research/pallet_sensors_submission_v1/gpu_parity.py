"""CPU reference versus actual GPU eval and train BN, before any main fit."""
import gc
import torch,numpy as np
from env import *
from prior_model import PoseFixPallet9,expectation
def run():
    if complete('GPU_PARITY'):return
    start=now();assert not gpu()['foreign_compute'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a=np.load(RAW/'actual_source8.npz');records={}
    for seed in (1,2,3):
        m=PoseFixPallet9();m.load_tf(RAW/f'tf_initial_seed{seed}.npz');m.cuda().eval()
        args=[torch.as_tensor(a[k],device='cuda') for k in ('rgb','points','valid')]
        with torch.no_grad():q=expectation(m(*args)).cpu().numpy()
        r=np.load(RAW/f'tf_output_seed{seed}.npy');np.testing.assert_allclose(q,r,atol=3e-4,rtol=3e-4)
        records[str(seed)]=float(abs(q-r).max());del m,args;gc.collect();torch.cuda.empty_cache()
    r=np.load(RAW/'tf_train_ops.npz');bn=torch.nn.BatchNorm2d(3,eps=1.001e-5,momentum=.01).cuda().train()
    for i in range(3):
        out=bn(torch.as_tensor(r['bn_input']+i*.1,device='cuda').permute(0,3,1,2)).detach().permute(0,2,3,1).cpu().numpy()
        np.testing.assert_allclose(out,r['bn_output'][i],atol=1e-5,rtol=1e-4)
        np.testing.assert_allclose(np.stack([bn.running_mean.cpu().numpy(),bn.running_var.cpu().numpy()]),r['bn_stats'][i],atol=1e-5,rtol=1e-4)
    epsilon_source=RAW/'env_tf1/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_impl.py'
    write(DOC/'OPERATOR_AMENDMENT.json',dict(official_graph_requested_BN_epsilon=1e-9,actual_TF1_fused_minimum=1.001e-5,torch_epsilon=1.001e-5,source=bound(epsilon_source),source_lines='1486-1489',tolerance_unchanged=True,prior_main_updates_before_correction=0,CPU_and_GPU_train_BN_stats_verified=True,coordinate_GPU_max_absolute_px=records,TF1_CPU_reference=True,CUDA_torch_port=True,not_official_TF1_GPU_execution=True))
    receipt('GPU_PARITY',[HERE/'prior_model.py',HERE/'gpu_parity.py',DOC/'IMPLEMENTATION_COMPLETE.json'],[DOC/'OPERATOR_AMENDMENT.json'],start)
    print('GPU_PARITY_PASS',records,flush=True)
if __name__=='__main__':run()
