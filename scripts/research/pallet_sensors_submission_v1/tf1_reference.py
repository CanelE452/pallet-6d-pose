"""Run the unmodified official TF1 model graph on CPU; isolate dataset config.

Python3.7/TF1.15 only. No imports from the current YOLO runtime.
Official config and trainer I/O are replaced with a no-write config/interface.
Official network functions are used unmodified, including BN and initializers.
"""
import os, sys, json, types, argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'data/pallet/results/pallet_sensors_submission_v1'
OFFICIAL=ROOT/'data/pallet/results/pallet_sensors_refinement_closeout_v1/external/PoseFix_RELEASE'
def load_source():
    config=types.ModuleType('config')
    config.cfg=types.SimpleNamespace(bn_train=True,weight_decay=1e-5,num_kps=9,input_shape=(384,288),output_shape=(96,72),input_sigma=9.,batch_size=1,backbone='resnet152')
    sys.modules['config']=config
    base=types.ModuleType('tfflat.base')
    class ModelDesc:
        def set_inputs(self,*x):self.inputs=x
        def set_outputs(self,*x):self.outputs=x
        def set_loss(self,x):self.loss=x
        def add_tower_summary(self,*x):pass
    base.ModelDesc=ModelDesc;sys.modules['tfflat.base']=base
    sys.path[:0]=[str(OFFICIAL/'main'),str(OFFICIAL/'lib')]
    from model import Model
    return Model
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    dest=RAW/('tf_initial_seed%d.npz'%a.seed)
    if dest.exists() and (RAW/('TF_REFERENCE_seed%d.json'%a.seed)).exists():print('TF_REFERENCE_REUSED',a.seed);return
    tf.set_random_seed(a.seed);np.random.seed(a.seed)
    Model=load_source();model=Model();model.make_network(False);print('TF_GRAPH_BUILT',a.seed,flush=True)
    inputs=np.load(str(RAW/'actual_source8.npz'))
    variables=tf.global_variables();reader=tf.train.NewCheckpointReader(str(RAW/'resnet_v1_152.ckpt'))
    cfg=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=4,inter_op_parallelism_threads=1)
    loaded=[];fresh=[];available=reader.get_variable_to_shape_map()
    with tf.Session(config=cfg) as s:
        s.run(tf.global_variables_initializer())
        assignments=[]
        for v in variables:
            key=v.name.split(':')[0]
            if key=='resnet_v1_152/conv1/weights':
                arr=s.run(v);arr[:,:,:3,:]=reader.get_tensor(key);assignments.append(v.assign(arr));loaded.append(key+' [RGB only; pose channels official random init]')
            elif key in available:assignments.append(v.assign(reader.get_tensor(key)));loaded.append(key)
            else:fresh.append(key)
        s.run(assignments)
        arrays=dict(zip([v.name.split(':')[0] for v in variables],s.run(variables)));np.savez(str(dest),**arrays);print('TF_WEIGHTS_EXPORTED',a.seed,flush=True)
        all_outputs=[]
        for i in range(8):
            feed={model.inputs[0]:inputs['rgb'][i:i+1].transpose(0,2,3,1),model.inputs[1]:inputs['points'][i:i+1],model.inputs[2]:inputs['valid'][i:i+1]}
            all_outputs.append(s.run(model.outputs[0],feed_dict=feed))
        np.save(str(RAW/('tf_output_seed%d.npy'%a.seed)),np.concatenate(all_outputs))
        if a.seed==1:
            # Every stage endpoint is taken from the actual official graph, not a reimplemented reference.
            endpoint_names=['resnet_v1_152/conv1/Relu:0','resnet_v1_152/block1/unit_3/bottleneck_v1/Relu:0','resnet_v1_152/block2/unit_8/bottleneck_v1/Relu:0','resnet_v1_152/block3/unit_36/bottleneck_v1/Relu:0','resnet_v1_152/block4/unit_3/bottleneck_v1/Relu:0','up1/Relu:0','up2/Relu:0','up3/Relu:0','out/BiasAdd:0']
            keys=['stem','block1','block2','block3','block4','up1','up2','up3','logits']
            tensors=[]
            for name in endpoint_names:
                suffix=name.split('/',1)[1] if name.startswith('resnet') else name
                matches=[op.outputs[0] for op in tf.get_default_graph().get_operations() if op.outputs and (op.outputs[0].name==name if name=='resnet_v1_152/conv1/Relu:0' else op.outputs[0].name.endswith(suffix))]
                assert len(matches)==1,(name,[x.name for x in matches])
                tensors.append(matches[0])
            np.savez(str(RAW/'tf_layers.npz'),**dict(zip(keys,s.run(tensors,feed_dict=feed))))
    info=dict(complete=True,seed=a.seed,tensorflow=tf.__version__,CPU_reference=True,official_network_executed=True,source_commit='5556364bb0f43b0743a5fcd820de48f34b3d4360',loaded_ImageNet=loaded,new_parameters=fresh,unused_checkpoint=sorted(k for k in available if k not in arrays),source_train_cases=8)
    (RAW/('TF_REFERENCE_seed%d.json'%a.seed)).write_text(json.dumps(info,indent=2)+'\n')
    print('OFFICIAL_TF1_REFERENCE_COMPLETE',a.seed,flush=True)
if __name__=='__main__':main()
