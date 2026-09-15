"""Small exact TF1 training-operator references, no main training updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import numpy as np
import tensorflow as tf
from tf1_reference import load_source,RAW
tf.logging.set_verbosity(tf.logging.ERROR)
def main():
    rng=np.random.RandomState(20260915);data=rng.randn(4,7,5,3).astype('float32');grad=rng.randn(5,7).astype('float32');init=rng.randn(7).astype('float32')
    x=tf.placeholder(tf.float32,data.shape)
    y=tf.contrib.layers.batch_norm(x,decay=.99,epsilon=1e-9,scale=True,is_training=True,updates_collections=tf.GraphKeys.UPDATE_OPS,scope='bn')
    ups=tf.get_collection(tf.GraphKeys.UPDATE_OPS);moving=[v for v in tf.global_variables() if 'moving_' in v.name]
    p=tf.Variable(init,name='p');g=tf.placeholder(tf.float32,[7]);op=tf.train.AdamOptimizer(.0005).apply_gradients([(g,p)])
    Model=load_source();model=Model()
    logits=rng.randn(1,96,72,9).astype('float32');target=rng.uniform([4,4],[280,376],(1,9,2)).astype('float32');valid=np.array([[1]*8+[0]],np.float32)
    z=tf.placeholder(tf.float32,logits.shape);t=tf.constant(target)
    q=model.render_onehot_heatmap(t,(96,72));ce=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(tf.transpose(q,[0,3,1,2]),[1,9,-1]),logits=tf.reshape(tf.transpose(z,[0,3,1,2]),[1,9,-1]))*valid)
    coord=tf.reduce_mean(tf.abs(model.extract_coordinate(z)/4-t/4)*valid[:,:,None]);dz=tf.gradients(ce+coord,z)[0]
    with tf.Session(config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=4,inter_op_parallelism_threads=1)) as s:
        s.run(tf.global_variables_initializer());bnout=[];bnstats=[]
        for i in range(3):
            out,_=s.run([y,ups],{x:data+i*.1});bnout.append(out);bnstats.append(s.run(moving))
        states=[]
        for row in grad:s.run(op,{g:row});states.append(s.run(p))
        qv,cv,lv,dv=s.run([q,ce,coord,dz],{z:logits})
    np.savez(str(RAW/'tf_train_ops.npz'),bn_input=data,bn_output=np.array(bnout),bn_stats=np.array(bnstats),adam_initial=init,adam_gradients=grad,adam_states=np.array(states),logits=logits,target=target,target_valid=valid,target_distribution=qv,heatmap_loss=cv,coordinate_loss=lv,logit_grad=dv)
    print('TF1_TRAIN_OPERATORS_EXPORTED')
if __name__=='__main__':main()
