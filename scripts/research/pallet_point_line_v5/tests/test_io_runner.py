import json,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import numpy as np
import pytest
import torch
from plpose_v5.io import *
from plpose_v5.fixtures import geometry_fixture
from plpose_v5.contracts import *
from plpose_v5.model import ModelConfig
from plpose_v5.runner import train,score
from plpose_v5.assessment import evaluate,compare,paired_session_interval
from plpose_v5.adapters import capture_module_inputs,feature_affine,permutation_to_new


def make_manifest(root,count=3,split='generated'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    d,K,R,t,p,m=geometry_fixture(count,dtype=torch.float32);records=[]
    for i in range(count):
        box=torch.cat((p[i,:8].amin(0)-20,p[i,:8].amax(0)+20))[None]
        obs=Observation(torch.randn(1,8,12,12),torch.ones(1,1,12,12,dtype=torch.bool),box,K[i:i+1],d[i:i+1],torch.tensor([[480.,640.]]))
        op=root/f'obs_{i}.pt';sp=root/f'gt_{i}.pt';torch.save(vars(obs),op)
        torch.save({'points':p[i:i+1],'valid':torch.ones(1,9,dtype=torch.bool),'R':R[i:i+1],'t':t[i:i+1]},sp)
        records.append({'id':f'fixture_{i}','session':f'fixture_session_{i%2}','observation':op.name,'supervision':sp.name,
                        'observation_sha256':sha256(op),'supervision_sha256':sha256(sp),
                        'source_image_sha256':hashlib.sha256(f'generated_{root.name}_{i}'.encode()).hexdigest(),
                        'symmetry':asdict(SymmetrySpec(2,True,'generated geometric test only'))})
    path=root/'manifest.json';write_json(path,{'schema':SCHEMA,'coordinate_contract':'object_x_width_y_up_z_depth_v1',
                      'split':split,'population_scope':'GENERATED_ONLY','records':records});return path

def test_inference_physically_without_GT_files_and_offline_eval(tmp_path):
    manifest=make_manifest(tmp_path/'data');cfg=ModelConfig(in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2,solver_iterations=1)
    cp=tmp_path/'model.json';write_json(cp,asdict(cfg));run=tmp_path/'run'
    train(SimpleNamespace(output=str(run),manifest=str(manifest),config=str(cp),steps=2,batch=2,seeds=[1],device='cpu',lr=1e-4))
    payloads={p:p.read_bytes() for p in (tmp_path/'data').glob('gt_*.pt')}
    for p in payloads:p.unlink()
    try:
        for arm in ('point','direct','hough'):
            dst=tmp_path/f'{arm}.json';score(SimpleNamespace(manifest=str(manifest),checkpoint=str(run/f'{arm}_seed1/checkpoint_final.pt'),device='cpu',output=str(dst)))
            result=read_json(dst);assert result['loader_target_reads']==0 and result['loader_observation_reads']==3
    finally:
        for p,data in payloads.items():p.write_bytes(data)
    for arm in ('point','direct','hough'):
        report=evaluate(manifest,tmp_path/f'{arm}.json',tmp_path/f'{arm}_eval.json');assert report['frames']==3 and report['gt_corners']==24
    comp=compare([tmp_path/'point_eval.json'],[tmp_path/'hough_eval.json'],tmp_path/'comparison.json');assert comp['scientific_success'] is None
    r=read_json(run/'COMPLETION.json');assert r['cells']==3 and r['steps']==6
    receipts=[read_json(run/f'{a}_seed1/COMPLETION.json') for a in ('point','direct','hough')]
    assert len({r['shared_initial_sha256'] for r in receipts})==1 and len({r['plan_sha256'] for r in receipts})==1
    assert all(r['changed_tensor_values']>0 for r in receipts)

def test_preflight_reports_unavailable_image_bytes(tmp_path):
    manifest=make_manifest(tmp_path/'data');result=audit_manifests([manifest],tmp_path/'audit.json')
    assert result['status']=='PARTIAL_INPUT_VERIFICATION' and not result['source_image_bytes_all_rehashed']

def test_GT_frame_mismatch_fails_not_rewritten(tmp_path):
    manifest=make_manifest(tmp_path/'data');raw=read_json(manifest);sp=tmp_path/'data'/'gt_0.pt';gt=torch.load(sp,weights_only=True);gt['points']+=10;torch.save(gt,sp)
    raw['records'][0]['supervision_sha256']=sha256(sp);write_json(manifest,raw)
    with pytest.raises(ValueError,match='mismatch'):audit_manifests([manifest],tmp_path/'audit.json')

def test_observation_extra_GT_field_rejected(tmp_path):
    path=make_manifest(tmp_path/'data');raw=read_json(path);r=raw['records'][0];op=tmp_path/'data'/r['observation'];data=torch.load(op,weights_only=True);data['gt_points']=torch.zeros(1,9,2)
    torch.save(data,op);r['observation_sha256']=sha256(op);write_json(path,raw)
    with pytest.raises(ValueError):InstanceDataset(path).observation(0)

def test_observation_hash_checked(tmp_path):
    path=make_manifest(tmp_path/'data');op=tmp_path/'data'/'obs_0.pt';data=torch.load(op,weights_only=True);data['features']+=1;torch.save(data,op)
    with pytest.raises(ValueError,match='hash'):InstanceDataset(path).observation(0)

def test_forbidden_final_split(tmp_path):
    path=make_manifest(tmp_path/'data');m=read_json(path);m['split']='FINAL';write_json(path,m)
    with pytest.raises(ValueError):InstanceDataset(path)

def test_same_image_across_splits_rejected(tmp_path):
    p1=make_manifest(tmp_path/'a');p2=make_manifest(tmp_path/'b');m=read_json(p2);old=read_json(p1)
    for i,r in enumerate(m['records']):r['id']='b_'+r['id'];r['source_image_sha256']=old['records'][i]['source_image_sha256']
    write_json(p2,m)
    with pytest.raises(ValueError,match='Image SHA'):audit_manifests([p1,p2],tmp_path/'audit.json')

def test_bootstrap_constant_difference():
    out=paired_session_interval([-1.,-1.,-1.],['a','a','b'],draws=100);assert out['lower']==out['upper']==-1.

def test_hook_preserves_gradient_and_removed_on_exit():
    class Reader(torch.nn.Module):
        def forward(self,x):return x[0]*2
    mod=Reader();x=torch.ones(1,requires_grad=True)
    with capture_module_inputs(mod,detach=False) as capture:y=mod([x])
    capture['features'][0].sum().backward();assert x.grad==1 and len(mod._forward_pre_hooks)==0
    with pytest.raises(ValueError):
        with capture_module_inputs(mod):pass
    assert len(mod._forward_pre_hooks)==0

def test_feature_affine_explicit_offset():
    A=feature_affine(torch.eye(3)[None],(8.,16.),(3.5,7.5));torch.testing.assert_close(A[0],torch.tensor([[.125,0,-3.5/8],[0,1/16,-7.5/16],[0,0,1]]))

def test_no_arbitrary_assignment_adapter():
    p=torch.randn(1,9,2);v=torch.ones(1,9,dtype=torch.bool);bad=torch.zeros(1,9,dtype=torch.long)
    with pytest.raises(ValueError):permutation_to_new(p,v,bad)

def test_no_overwrite_completed_run(tmp_path):
    root=tmp_path/'occupied';root.mkdir();(root/'keep.txt').write_text('original')
    with pytest.raises(ValueError,match='Nonempty'):train(SimpleNamespace(output=str(root)))
    assert (root/'keep.txt').read_text()=='original'

def test_explicit_axis_change_pose():
    from plpose_v5.adapters import reframe_pose
    from plpose_v5.geometry import rotation_y,project,cuboid
    d,K,R,t,p,m=geometry_fixture(dtype=torch.float64);A=rotation_y(torch.tensor([.5],dtype=torch.float64));newR,newt=reframe_pose(R,t,A);Xnew=cuboid(d);Xold=torch.einsum('bij,bnj->bni',A,Xnew)
    torch.testing.assert_close(project(Xold,R,t,K)[0],project(Xnew,newR,newt,K)[0])

def test_metrics_hand_computed_translation(tmp_path):
    path=make_manifest(tmp_path/'data',count=1);ds=InstanceDataset(path);gt=ds.target(0)
    predictions={'schema':'plpose_v5_predictions_1','manifest_sha256':sha256(path),'arm':'fixture','seed':1,'records':[{'id':ds.records[0]['id'],'session':'fixture_session_0','R':gt.R[0].tolist(),'t':gt.t[0].tolist(),'points':(gt.points[0]+torch.tensor([3.,4.])).tolist(),'pose_valid':True}]}
    pp=tmp_path/'pred.json';write_json(pp,predictions);report=evaluate(path,pp,tmp_path/'eval.json')
    assert abs(report['fixed_error8_px']['mean']-5)<1e-5 and abs(report['primary']-5/800)<1e-5 and report['translation_m']['mean']==0

def test_missing_pose_in_primary_denominator(tmp_path):
    path=make_manifest(tmp_path/'data',count=1);ds=InstanceDataset(path);gt=ds.target(0)
    predictions={'schema':'plpose_v5_predictions_1','manifest_sha256':sha256(path),'arm':'fixture','seed':1,'records':[{'id':ds.records[0]['id'],'session':'fixture_session_0','R':gt.R[0].tolist(),'t':gt.t[0].tolist(),'points':gt.points[0].tolist(),'pose_valid':False}]}
    pp=tmp_path/'pred.json';write_json(pp,predictions);report=evaluate(path,pp,tmp_path/'eval.json')
    assert report['primary']==1 and report['observed_corners']==0 and report['gt_corners']==8 and report['symmetric_error8_px']['mean'] is None

def test_nonworse_metric_missing_is_not_a_pass():
    from plpose_v5.assessment import metric_nonworse
    assert not metric_nonworse(None,None) and not metric_nonworse(None,1.) and metric_nonworse(.9,1.)

def test_counters_are_real_loader_events(tmp_path):
    ds=InstanceDataset(make_manifest(tmp_path/'data'));assert ds.target_reads==ds.observation_reads==0
    ds.observation(0);ds.observation(1);ds.target(0);assert ds.observation_reads==2 and ds.target_reads==1
