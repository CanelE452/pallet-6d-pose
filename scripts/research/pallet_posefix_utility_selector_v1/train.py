"""Train only the small selector on locked synthetic TRAIN scenario groups."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import core as P
from .model import UtilitySelector,choose_pairs,apply_pairs

KEYS=('rgb','patches','numeric','global_numeric')


def batch(data,indices,device='cuda'):
    return {k:torch.as_tensor(np.array(data[k][indices],copy=True),device=device) for k in KEYS}


def loss(output,target,mask):
    assert mask.any(),'Batch has no supported synthetic corner'
    return F.smooth_l1_loss(output[mask],target[mask],beta=1)


@torch.no_grad()
def predict(model,data,indices):
    return np.concatenate([model(batch(data,indices[i:i+32])).cpu().numpy() for i in range(0,len(indices),32)])


def utility_report(data,indices,utility):
    target=np.asarray(data['target'][indices]);mask=np.asarray(data['mask'][indices],bool)
    assert np.isfinite(utility).all()
    truth=target[mask];score=utility[mask]
    signs=truth>0
    from scipy.stats import rankdata
    npos=int(signs.sum());nneg=int((~signs).sum())
    auc=None if not npos or not nneg else float((rankdata(score)[signs].sum()-npos*(npos+1)/2)/(npos*nneg))
    output=dict(samples=len(indices),supported_corners=int(mask.sum()),target_positive=npos,
        predicted_positive=int((score>0).sum()),sign_accuracy=float(((score>0)==signs).mean()),
        majority_sign_accuracy=float(max(npos,nneg)/len(score)),ROC_AUC=auc,
        utility_MAE=float(np.abs(score-truth).mean()),units='percent predicted-box diagonal')
    for variant_id,label in enumerate(('clean','stress')):
        initial=[];raw=[];selected=[];accepted=0;frames=0
        for j,index in enumerate(indices):
            if int(data['variant'][index])!=variant_id:continue
            n2,r=data['n2'][index],data['replay'][index]
            pairs=choose_pairs(utility[j],data['valid'][index])
            q=apply_pairs(n2,r,pairs);v=data['mask'][index].astype(bool);gt=data['gt_aligned'][index]
            a=np.linalg.norm(n2[:8]-gt[:8],axis=-1)[v]
            b=np.linalg.norm(r[:8]-gt[:8],axis=-1)[v]
            c=np.linalg.norm(q[:8]-gt[:8],axis=-1)[v]
            initial.extend(a);raw.extend(b);selected.extend(c);accepted+=int(pairs.sum());frames+=1
        a,b,c=map(np.asarray,(initial,raw,selected))
        metric=lambda e:dict(corners=len(e),mean_px=float(e.mean()),median_px=float(np.median(e)),P90_px=float(np.quantile(e,.9)),PCK10=float((e<=10).mean()))
        output[label]=dict(frames=frames,accepted_pairs=accepted,N2=metric(a),Replay=metric(b),Learned=metric(c),
            new_gains=int(((a>10)&(c<=10)).sum()),new_damages=int(((a<=10)&(c>10)).sum()))
    return output


def run():
    p=P.verify();P.setup();runtime=[P.gpu()]
    done=P.DOC/'FIT.json'
    if done.exists():
        fit=P.read(done);P.verify_binding(fit['checkpoint'])
        if (P.DOC/'SOURCE_RESULTS.json').exists():
            P.verify_binding(P.read(P.DOC/'SOURCE_RESULTS.json')['diagnostics'])
            print('FIT_ALREADY_COMPLETE',flush=True);return
        source=P.read(P.DOC/'SOURCE_COMPLETE.json');P.verify_binding(source['artifact'])
        data=dict(np.load(P.ROOT/source['artifact']['path']))
        ck=torch.load(P.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['protocol']==P.bound(P.DOC/'PROTOCOL.json') and ck['step']==1500
        model=UtilitySelector(26,64);model.load_state_dict(ck['model_state_dict']);model=model.cuda().eval()
        diagnostics(model,data,np.flatnonzero(data['train']),np.flatnonzero(~data['train']))
        print('FIT_REUSED_DIAGNOSTICS_COMPLETED',flush=True);return
    source=P.read(P.DOC/'SOURCE_COMPLETE.json');assert source['complete']
    P.verify_binding(source['artifact'])
    data=dict(np.load(P.ROOT/source['artifact']['path']))
    train=np.flatnonzero(data['train']);validation=np.flatnonzero(~data['train'])
    assert len(train) and len(validation)
    assert not set(data['row'][train])&set(data['row'][validation])
    torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    model=UtilitySelector(p['numeric_dim'],p['global_dim']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['optimizer']['lr'],weight_decay=p['optimizer']['weight_decay'])
    rng=np.random.default_rng(p['sampler_seed'])
    probe=train[:min(128,len(train))]
    model.eval();before=predict(model,data,probe)
    initial=float(np.mean(np.abs(before[data['mask'][probe]]-data['target'][probe][data['mask'][probe]])))
    start=time.monotonic();history=[];torch.cuda.reset_peak_memory_stats();model.train()
    initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for step in range(1,p['steps']+1):
        ii=rng.choice(train,p['batch'],replace=False)
        x=batch(data,ii);t=torch.as_tensor(data['target'][ii],device='cuda');m=torch.as_tensor(data['mask'][ii],device='cuda',dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True);y=model(x);value=loss(y,t,m)
        assert torch.isfinite(value);value.backward()
        gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),p['optimizer']['gradient_clip'],error_if_nonfinite=True))
        assert gradient>0;optimizer.step()
        if step==1 or step%100==0:
            runtime.append(P.gpu())
            record=dict(step=step,loss=float(value.detach()),gradient_norm=gradient,elapsed_seconds=time.monotonic()-start)
            history.append(record);print(dict(stage='SELECTOR_TRAIN',**record),flush=True)
    model.eval();after=predict(model,data,probe)
    final=float(np.mean(np.abs(after[data['mask'][probe]]-data['target'][probe][data['mask'][probe]])))
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    update=float(sum((v.detach().cpu()-initial_state[k]).square().sum() for k,v in model.state_dict().items()).sqrt())
    assert update>0
    checkpoint=P.RAW/'last1500.pt';assert not checkpoint.exists()
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    with checkpoint.open('xb') as f:torch.save(dict(complete=True,step=p['steps'],protocol=P.bound(P.DOC/'PROTOCOL.json'),
        model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},numeric_dim=p['numeric_dim'],global_dim=p['global_dim']),f)
    # Freeze checkpoint before reading validation supervision for diagnostics.
    P.freeze(P.DOC/'FIT.json',dict(complete=True,steps=p['steps'],checkpoint=P.bound(checkpoint),
        source_complete=P.bound(P.DOC/'SOURCE_COMPLETE.json'),protocol=P.bound(P.DOC/'PROTOCOL.json'),
        train_samples=len(train),validation_samples=len(validation),history=history,
        training_probe_MAE_before=initial,training_probe_MAE_after=final,parameter_update_L2=update,
        elapsed_seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        runtime_checks=runtime,checkpoint_selection=False,original_models_updated=False,real_GT_used=False))
    summaries=diagnostics(model,data,train,validation)
    P.verify();print(dict(stage='SELECTOR_FIT_COMPLETE',fit=str(done),source_summary=summaries),flush=True)


def diagnostics(model,data,train,validation):
    predictions={};summaries={}
    path=P.RAW/'SOURCE_DIAGNOSTICS.npz'
    cached=dict(np.load(path)) if path.exists() else None
    for name,ii in [('train',train),('validation',validation)]:
        if cached is not None:
            np.testing.assert_array_equal(cached[name+'_indices'],ii)
            utility=cached[name+'_utility']
        else:utility=predict(model,data,ii)
        predictions[name+'_indices']=ii;predictions[name+'_utility']=utility
        summaries[name]=utility_report(data,ii,utility)
    if cached is None:P.save_npz(path,**predictions)
    P.freeze(P.DOC/'SOURCE_RESULTS.json',dict(complete=True,summary=summaries,fit=P.bound(P.DOC/'FIT.json'),
        diagnostics=P.bound(P.RAW/'SOURCE_DIAGNOSTICS.npz'),validation_used_for_selection=False,
        reused_synthetic_development=True,fully_unseen_candidate_data=False,C4_training_images=0))
    return summaries


if __name__=='__main__':run()
