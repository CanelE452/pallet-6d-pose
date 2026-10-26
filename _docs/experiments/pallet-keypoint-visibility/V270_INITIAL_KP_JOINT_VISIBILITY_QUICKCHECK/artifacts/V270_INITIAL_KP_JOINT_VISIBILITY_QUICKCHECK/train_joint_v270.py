"""Three prescribed final300 actual R0 pose-stem fits; no real data."""
import os,sys,json,time,types,hashlib
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
W=Path(__file__).parent
p=W/'model_initial_joint_v270.py'
for a in [p,*p.parents]:assert not a.is_symlink()
b=p.read_bytes();m=types.ModuleType('v270_model');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(b,str(p),'exec'),m.__dict__)
def main():
    protocol=json.loads(m.safe(W/'PROTOCOL_BEFORE_FIT.json').read_text())
    # The protocol binds this exact source and the complete immutable feature packet.
    assert protocol.get('PASS') and protocol.get('completed')
    for value in protocol['code'].values():m.verify(value)
    for key in ('feature_complete','shared_assignment','training_inputs','shared_fit_draws','shared_initial_head_state'):m.verify(protocol[key])
    feature=m.safe(W/'syn_initial_features_v1');complete=json.loads((feature/'COMPLETE.json').read_text());assert complete['PASS']
    for v in complete['outputs'].values():m.verify(v)
    out=m.safe(W/'syn_training_v1');assert not out.exists();out.mkdir()
    inputs=np.load(feature/'TRAIN_INPUTS.npz',allow_pickle=False);ass=np.load(feature/'ASSIGNMENTS.npz',allow_pickle=False)
    draws=np.load(feature/'FIT_DRAWS300x8.npy',allow_pickle=False);assert draws.shape==(300,8) and inputs['fit'][draws].all()
    feats=[np.load(feature/f'pyramid_P{i}.npy',mmap_mode='r') for i in (3,4,5)]
    initial=torch.load(feature/'INITIAL_HEAD_STATE.pt',map_location='cpu',weights_only=False)
    lr=float(protocol.get('learning_rate',1e-4));wd=float(protocol.get('weight_decay',1e-4))
    assert lr==1e-4 and wd==1e-4
    logs={};cp={};start=time.time();initial_sha=None;frozen_sha=None
    for arm in m.ARMS:
        torch.manual_seed(17);np.random.seed(17);torch.use_deterministic_algorithms(True)
        model=m.InitialKPJoint(arm,device='cuda');model.load_head_state(initial);model.train()
        stsha=m.state_sha({f'{k}.{j}':v for k,d in model.head_state().items() for j,v in d.items()})
        bnsha=m.state_sha(dict(model.base.named_buffers()));frsha=model.frozen_sha()
        if initial_sha is None:initial_sha=stsha;frozen_sha=frsha
        assert stsha==initial_sha and frsha==frozen_sha
        params=[p for p in model.parameters() if p.requires_grad]
        groups=[dict(params=list(model.head.one2one_cv4.parameters()),lr=1e-4)]
        if arm!='XY_ONLY':groups.append(dict(params=list(model.e_heads.parameters()),lr=1e-3))
        opt=torch.optim.AdamW(groups,weight_decay=wd,betas=(.9,.999),eps=1e-8)
        history=[];last=time.time()
        for step,ids in enumerate(draws,1):
            x=[torch.from_numpy(np.array(f[ids])).cuda() for f in feats]
            targets={k:torch.from_numpy(np.array(inputs[k][ids])).cuda() for k in ('xy_px','loc_valid9','canvas_obj9','E9','Emask9')}
            anchor=torch.from_numpy(ass['assigned_anchor'][ids]).cuda();bbox=torch.from_numpy(ass['bbox_xyxy_px'][ids]).cuda()
            opt.zero_grad(set_to_none=True);packet=model.forward_pyramid(x,with_det=False);loss=m.losses(packet,targets,anchor,bbox,complete['E_class_weights'],arm)
            assert all(torch.isfinite(v) for v in loss.values());loss['total'].backward();torch.nn.utils.clip_grad_norm_(params,10);opt.step()
            history.append(dict(step=step,**{k:float(v.detach()) for k,v in loss.items()}))
            if step%50==0:print(arm,step,history[-1],f'elapsed {time.time()-last:.1f}s',flush=True)
        assert model.frozen_sha()==frsha and m.state_sha(dict(model.base.named_buffers()))==bnsha
        checkpoint=dict(arm=arm,steps=300,head_state=model.head_state(),initial_head_sha256=stsha,frozen_before_sha256=frsha,frozen_after_sha256=model.frozen_sha(),BN_before_sha256=bnsha,BN_after_sha256=m.state_sha(dict(model.base.named_buffers())),model_code=m.binding(W/'model_initial_joint_v270.py'),protocol=m.binding(W/'PROTOCOL_BEFORE_FIT.json'),draws=m.binding(feature/'FIT_DRAWS300x8.npy'),trainable_parameters=sum(p.numel() for p in params),semantic_E_trained=arm!='XY_ONLY',RLE_loss=0)
        path=out/f'{arm}_FINAL300.pt';torch.save(checkpoint,path);cp[arm]=m.binding(path);m.dump(out/f'{arm}_TRAIN_LOG.json',history);logs[arm]=m.binding(out/f'{arm}_TRAIN_LOG.json')
        del opt,model;torch.cuda.empty_cache()
    # Barrier: all final checkpoints exist before any DEV prediction.
    m.dump(out/'ALL_CHECKPOINTS_FROZEN_BEFORE_DEV.json',dict(PASS=True,checkpoints=cp,steps_per_arm=300,draws=m.binding(feature/'FIT_DRAWS300x8.npy')))
    dev=np.flatnonzero(~inputs['fit']);pred={}
    for arm in m.ARMS:
        model=m.InitialKPJoint(arm,device='cuda');checkpoint=torch.load(m.verify(cp[arm]),map_location='cpu',weights_only=False);model.load_head_state(checkpoint['head_state']);model.eval()
        xy=[];obj=[];elog=[]
        with torch.no_grad():
            for lo in range(0,len(dev),8):
                ids=dev[lo:lo+8];packet=model.forward_pyramid([torch.from_numpy(np.array(f[ids])).cuda() for f in feats],with_det=False)
                s=m.selected(packet,torch.from_numpy(ass['detection_anchor'][ids]).cuda());xy.append(s['xy_px'].cpu().numpy());obj.append(s['canvas_logit'].cpu().numpy());elog.append(s['E_logit'].cpu().numpy())
        np.savez_compressed(out/f'{arm}_SYNDEV235.npz',frame_index=dev,xy_px=np.concatenate(xy),canvas_logit=np.concatenate(obj),E_logit=np.concatenate(elog) if arm!='XY_ONLY' else np.full((235,9),np.nan,np.float32),detection_anchor=ass['detection_anchor'][dev],detection_confidence=ass['detection_confidence'][dev]);pred[arm]=m.binding(out/f'{arm}_SYNDEV235.npz')
        del model;torch.cuda.empty_cache()
    result=dict(PASS=True,completed=True,checkpoints=cp,logs=logs,SYNDEV_predictions=pred,code=m.binding(Path(__file__)),model_code=m.binding(W/'model_initial_joint_v270.py'),protocol=m.binding(W/'PROTOCOL_BEFORE_FIT.json'),source_feature_complete=m.binding(feature/'COMPLETE.json'),initial_head_sha256=initial_sha,total_optimizer_updates=900,current223_reads=0,elapsed_seconds=time.time()-start)
    m.dump(out/'COMPLETE.json',result)
    m.dump(W/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json',dict(PASS=True,completed=True,train_arms=list(m.ARMS),model_code=result['model_code'],checkpoints=cp,training_complete=m.binding(out/'COMPLETE.json'),protocol=result['protocol'],semantic_threshold=.5,detection_choice='highest frozen one2one class score anchor; no GT',detection_threshold=.001,raw_model_slot_order='original raw9 dynamic cuboid +centroid8; currentnative identity source-proven',XY_ONLY_semantic_output_available=False))
    print('COMPLETE all3 FINAL300, config frozen',flush=True)
if __name__=='__main__':main()
