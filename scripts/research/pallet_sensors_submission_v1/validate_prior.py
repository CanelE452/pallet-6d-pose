"""Actual TF1 reference, transferred weights and torch coordinate/operator gates."""
import tarfile, time
import numpy as np
import torch
from env import *
from prior_model import PoseFixPallet9,gaussian,expectation,target_distribution,TFAdam
from prior_data import prepare
import posefix_contract_math as mathref

def run():
    if complete('IMPLEMENTATION_COMPLETE'):verify();print('IMPLEMENTATION_REUSED');return
    verify();start=now();torch.set_num_threads(4)
    if not (RAW/'actual_source8.npz').exists():prepare()
    archive=RAW/'resnet_v1_152_2016_08_28.tar.gz'
    assert archive.exists(),'Download original TF-Slim ImageNet archive first'
    assert hashlib.md5(archive.read_bytes()).hexdigest()=='98081f80332293357fb22a626345fcc7','Google Storage object MD5 mismatch'
    if not (RAW/'resnet_v1_152.ckpt').exists():
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                assert m.isfile() and Path(m.name).name==m.name and m.name.startswith('resnet_v1_152.ckpt'),m.name
            t.extractall(RAW)
    write(DOC/'IMAGENET_SOURCE.json',dict(url='https://storage.googleapis.com/download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',official_link='https://github.com/tensorflow/models/tree/master/research/slim',archive=bound(archive),checkpoint=bound(RAW/'resnet_v1_152.ckpt'),human_pose_weights=False,additional_supervision='ImageNet classification',original_download_host_TLS_error='download.tensorflow.org certificate hostname mismatch; same Google Storage object downloaded with TLS verification'))
    for seed in (1,2,3):
        subprocess.run([str(RAW/'env_tf1/bin/python'),'-B',str(HERE/'tf1_reference.py'),'--seed',str(seed)],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4'})
    arr=np.load(RAW/'actual_source8.npz');result=mathref.self_test();model=PoseFixPallet9();mapping=model.load_tf(RAW/'tf_initial_seed1.npz');model.eval()
    args=[torch.from_numpy(arr[k]) for k in ('rgb','points','valid')]
    with torch.no_grad():
        logits=model(*args);coords=expectation(logits).numpy()
        _,layers=model(*(x[7:8] for x in args),return_layers=True)
    ref=np.load(RAW/'tf_output_seed1.npy');np.testing.assert_allclose(coords,ref,atol=3e-4,rtol=3e-4)
    references=np.load(RAW/'tf_layers.npz');errors={}
    for k,x in layers.items():
        q=x.numpy().transpose(0,2,3,1);r=references[k]
        relative=float(np.linalg.norm(q-r)/max(np.linalg.norm(r),1e-12))
        assert relative<1e-4,(k,relative)
        np.testing.assert_allclose(q,r,atol=3e-4,rtol=3e-4)
        errors[k]=dict(max_abs=float(np.max(np.abs(q-r))),relative_L2=relative)
    hm=gaussian(args[1],args[2]).numpy().transpose(0,2,3,1)
    np.testing.assert_allclose(hm,mathref.gaussian_input_maps(arr['points'],arr['valid']),atol=1e-5,rtol=1e-4)
    np.testing.assert_allclose(expectation(logits).numpy(),mathref.coordinate_expectation(logits.numpy().transpose(0,2,3,1)),atol=1e-5,rtol=1e-4)
    # Explicit boundary mass and nonfinite masking regression.
    p=torch.tensor([[[0.,0.],[287.,383.],[-1.,10.],[float('nan'),1.]]]);v=torch.ones(1,4,dtype=torch.bool)
    q,m=target_distribution(p,v);assert m.tolist()==[[True,True,False,False]];assert torch.equal(q.sum(-1),torch.tensor([[1.,1.,0.,0.]]))
    write(DOC/'WEIGHT_CONVERSION.json',dict(mapping=mapping,source=bound(RAW/'tf_initial_seed1.npz'),layout='conv HWIO to OIHW; deconv HWOI to IOHW; exact named parameters',no_torchvision=True))
    write(DOC/'NETWORK_PARITY.json',dict(complete=True,scope='official TF1 CPU eval graph vs torch CPU, same ImageNet+official-initialized new weights,8 source train images',layers=errors,coordinate_max_abs=float(np.abs(coords-ref).max()),math=result,target_boundary_test=True,training_BN_and_optimizer_parity='see TRAIN_OPERATOR_PARITY.json'))
    subprocess.run([str(RAW/'env_tf1/bin/python'),'-B',str(HERE/'tf1_train_ops.py')],check=True)
    from train_ops_test import run as train_test
    train_test()
    from resource_probe import run as resource
    resource()
    receipt('IMPLEMENTATION_COMPLETE',[HERE/'prior_model.py',HERE/'prior_data.py',HERE/'tf1_reference.py',HERE/'tf1_train_ops.py',HERE/'train_ops_test.py',HERE/'resource_probe.py',DOC/'PRIOR_PROTOCOL_LOCK.json'],[DOC/'NETWORK_PARITY.json',DOC/'TRAIN_OPERATOR_PARITY.json',DOC/'RESOURCE_AMENDMENT.json',DOC/'SMOKE_COMPLETE.json'],start)
    print('PRIOR_IMPLEMENTATION_VALIDATED',flush=True)
