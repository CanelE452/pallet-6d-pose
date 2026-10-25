"""Invented canonical feature packet and random heads; no actual data/weights."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent


def load(path):
    assert 'v112' not in str(path).casefold() and '__pycache__' not in str(path).casefold() and not str(path).endswith('.pyc')
    for parent in reversed((path,*path.parents)):assert not parent.is_symlink()
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:raw=stream.read()
    ns={'__name__':'toy_only','__file__':str(path)};exec(compile(raw,str(path),'exec'),ns)
    return ns,raw


def failure(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected identity/role failure')


def run():
    torch.set_num_threads(1)
    m,raw=load(HERE/'train_canonical_pair.py')
    headmodule,_=load(Path(m['MODULE_PIN'][0]))
    torch.manual_seed(17);control=headmodule['CropFamilyHead']()
    torch.manual_seed(17);treat=headmodule['RelationalQueryHead']()
    assert m['state_sha'](control.state_dict())==m['state_sha'](treat.base.state_dict())==m['INITIAL_BASE_SHA']
    assert sum(p.numel() for p in control.parameters())==76098 and sum(p.numel() for p in treat.parameters())==86054
    assert len(list(control.parameters()))==4 and len(list(treat.parameters()))==10
    rng=np.random.Generator(np.random.PCG64(239));g=5;n=32
    group=np.repeat(np.arange(4,dtype=np.int64),8);slot=np.tile(np.arange(8,dtype=np.int64),4)
    geometry={'q8':np.stack([np.array([[20.+j,30.+2*j] for j in range(8)],np.float64) for _ in range(g)]),
              'L8':np.full((g,8),50.,np.float64),'original_available8':np.ones((g,8),bool),
              'node_available':np.ones((g,8),bool),'node_fraction':np.ones((g,8),np.float32),
              'context_fraction':np.ones(g,np.float32),'box_xyxy':np.tile(np.array([20.,30.,27.,44.],np.float64),(g,1))}
    geometry['node_available'][4]=False;geometry['context_fraction'][4]=0
    relative=m['relative_xy'](geometry,group,slot)
    data={'full_view':np.arange(n,dtype=np.int64)*2,'central_view':np.arange(n,dtype=np.int64)*2+1,
          'fractions':np.ones((n,2),np.float32),'central4':np.ones(n,bool),'target':np.tile(np.array([0]*4+[1]*4,np.int8),4),
          'family':(group//2).astype(np.int8),'fit':group%2==0,'available':np.ones(n,bool),
          'row_id':np.array(['INVENTED:'+str(i) for i in range(n)]),'condition':np.zeros((n,32),np.float32)}
    for i in np.flatnonzero(data['family']==0):
        data['condition'][i,slot[i]]=1;data['condition'][i,8:16]=1;data['condition'][i,8+slot[i]]=0
        data['condition'][i,16:]=relative[i].reshape(-1)
    m['validate_samples'](data)
    node=F.normalize(torch.from_numpy(rng.standard_normal((g,8,384),dtype=np.float32)),dim=2).numpy();node[4]=0
    context=F.normalize(torch.from_numpy(rng.standard_normal((g,384),dtype=np.float32)),dim=1).numpy();context[4]=0
    x=rng.standard_normal((n,1186),dtype=np.float32);x[:,:384]=node[group,slot];x[:,1152:1154]=data['fractions'];x[:,-32:]=data['condition']
    features={'ORIGINAL_ALL_FEATURES.npy':x,'NODE_CLS.npy':node,'CONTEXT_CLS.npy':context,
              'SAMPLE_GROUP_INDEX.npy':group,'SAMPLE_NODE_INDEX.npy':slot,'RELATIVE_XY8.npy':relative}
    qa=m['global_feature_QA'](features,data,geometry);assert qa['checked']==n and qa['tolerance_used']==0
    bad=x.copy();bad[0,0]=np.nextafter(bad[0,0],np.float32(np.inf));wrong=dict(features);wrong['ORIGINAL_ALL_FEATURES.npy']=bad
    failure(lambda:m['global_feature_QA'](wrong,data,geometry))
    bad_data={k:v.copy() for k,v in data.items()};bad_data['fit'][8]=True
    failure(lambda:m['global_feature_QA'](features,bad_data,geometry))
    assert not m['byte_equal'](np.array([0.],np.float32),np.array([-0.],np.float32))
    draws,eligible=m['fixed_draws'](data);again,_=m['fixed_draws'](data)
    assert np.array_equal(draws,again) and draws.shape==(1000,32) and eligible[draws].all()
    for offset,family in ((0,0),(16,1)):
        assert (data['family'][draws[:,offset:offset+16]]==family).all()
        assert (data['target'][draws[:,offset:offset+8]]==0).all() and (data['target'][draws[:,offset+8:offset+16]]==1).all()
    indices=np.arange(n);tx=torch.from_numpy(x);args=m['context_args'](indices,features,geometry,'cpu')
    z0=control(tx);z1=treat(tx,*args)[0];assert torch.equal(z0,z1)
    for head,z in ((control,z0),(treat,z1)):
        loss,parts=m['family_loss'](z,torch.from_numpy(data['target']),torch.from_numpy(data['family']))
        assert torch.equal(loss,.5*(parts[0]+parts[1]));loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    # Actual production train_arm rejects a DEV index before any head or optimizer step.
    class CountHead(headmodule['CropFamilyHead']):
        calls=0
        def forward(self,x):self.calls+=1;return super().forward(x)
    guarded=CountHead();broken=draws.copy();broken[0,0]=8
    fit=np.flatnonzero(eligible);fit_map=np.full(n,-1,np.int64);fit_map[fit]=np.arange(len(fit))
    with tempfile.TemporaryDirectory(prefix='toy_guard_',dir=HERE) as temp:
        failure(lambda:m['train_arm'](m['CONTROL'],guarded,data,features,geometry,broken,eligible,
                torch.from_numpy(x[fit]),fit_map,'cpu',Path(temp),{'path':'toy'}, {'path':'toy'},'toy-draw','toy-base'))
    assert guarded.calls==0
    def summary(sd,se,yd,ye):return {'family0_DEV':{'D':{'recall':sd},'E':{'recall':se}},'family1_DEV':{'D':{'recall':yd},'E':{'recall':ye}}}
    old=summary(.8,.8,.8,.2)
    assert m['source_gate'](old,summary(.9,.9,.9,.5))['PASS']
    assert not m['source_gate'](old,summary(.899,.95,.95,.7))['PASS']
    assert not m['source_gate'](old,summary(.95,.95,.95,.499))['PASS']
    for path in ('/never/V112/x','/never/__pycache__/x','/never/x.pyc'):failure(lambda path=path:m['safe'](path))
    # Source text has no checkpoint/prefix/image loader; output checkpoint writes only.
    assert not any(isinstance(x,ast.Attribute) and x.attr=='load' and isinstance(x.value,ast.Name) and x.value.id=='torch' for x in ast.walk(ast.parse(raw)))
    result={'completed':True,'PASS':True,'fixture_groups':6,'code_sha256':hashlib.sha256(raw).hexdigest(),
            'same_seed17_base_sha256':m['INITIAL_BASE_SHA'],'head_parameters':[76098,86054],
            'all32_invented_active_bytes_exact_and_1ULP_rejected':True,'signed_zero_not_equal':True,
            'original_PCG17_draws_and_FIT_classquota':True,'zero_residual_shared_input_logits_identical':True,
            'random_head_backward_calls':2,'optimizer_steps':0,'production_DEV_optimizer_input_rejected_before_head':True,
            'SYN_D_E_floor_and_YCB_E_floor_checks':True,'actual_SOURCE_feature_geometry_weight_RGB_or_prefix_reads':0,
            'DINO_or_oldtrained_model_instantiation':0,'actual_training':0}
    with open(HERE/'TOY_RESULT.json','x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':run()
