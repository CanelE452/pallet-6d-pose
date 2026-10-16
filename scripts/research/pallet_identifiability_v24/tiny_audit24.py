from context24 import *
from model24 import DirectionGate,oracle_alpha
from train24 import predict,stats

def main():
    setup();t=read(RUN/'TINY_COMPLETE.json');proto=read(RUN/'TINY_TRAIN_PROTOCOL.json');assert sha(KIT/'train24.py')==proto['source_sha256'];assert sha(KIT/'model24.py')==proto['model_sha256'];assert t['neural_updates']==36000 and t['privileged_lookup_updates']==6000;maximum=0.;invariance=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=new_training(seed);plan=read(RUN/f'TINY_GROUPS_seed{seed}.json');train_ids=plan['ids'];check_ids=plan['check_ids'];assert not {x.split('::')[0] for x in train_ids}&{x.split('::')[0] for x in check_ids};lookup=torch.load(RUN/f'LOOKUP_seed{seed}.pt',weights_only=True);assert lookup['ids']==train_ids;starts=[]
        for arm in TINY_ARMS:
            folder=RUN/'tiny_models'/f'{arm}_seed{seed}';starts.append(read(folder/'START.json'));trace=[json.loads(x) for x in (folder/'TRACE.jsonl').read_text().splitlines()];assert len(trace)==6000 and all(x['step']==i+1 and all(np.isfinite(x[k]) for k in ('loss','point_error','gradient_norm')) for i,x in enumerate(trace));assert sha(folder/'final.pt')==read(folder/'COMPLETE.json')['sha256'];model=DirectionGate(seed).cuda().eval();model.load_state_dict(torch.load(folder/'final.pt',weights_only=True)['state_dict']);curve=read(folder/'CURVE.json')['rows'][-1]
            for split,selected in [('train',train_ids),('check',check_ids)]:
                ii=torch.tensor([i for i,x in enumerate(ids) if x in set(selected)],device='cuda:0');assert [ids[i] for i in ii.tolist()]==selected;out=predict(model,subset(O,ii),subset(r,ii),arm);actual=stats(out,gt[ii],valid[ii],kind[ii])
                for group,rr in actual.items():
                    for k,v in rr.items():diff=abs(v-curve[split][group][k]);maximum=max(maximum,diff);assert diff<.001
                # Independent numpy point/oracle calculation.
                b=out['base'].cpu().numpy().astype(float);d=out['proposal_delta'].cpu().numpy().astype(float);g=gt[ii].cpu().numpy().astype(float);v=valid[ii].cpu().numpy();a=np.clip(((g-b)*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1);oracle=np.linalg.norm(b+a[...,None]*d-g,axis=-1)[v].mean();assert abs(oracle-actual['all']['oracle_px'])<.001
            del model
        assert len({x['initial_sha256'] for x in starts})==1 and len({x['plan_sha256'] for x in starts})==1
        # The trained parent-control output must be invariant to removed image/context values.
        model=DirectionGate(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'unit_sign_parent_seed{seed}'/'final.pt',weights_only=True)['state_dict']);oo=subset(O,slice(0,4));rr=subset(r,slice(0,4))
        with torch.no_grad():
            a=model(oo,rr,'unit_sign_parent')['points'];b=model(dict(oo,features=torch.randn_like(oo['features'])),dict(rr,context=torch.randn_like(rr['context'])),'unit_sign_parent')['points'];diff=float((a-b).abs().max());assert diff==0;invariance.append(dict(seed=seed,max_error_px=diff))
        del O,r,gt,valid,model
    # Known geometry: a positive infinitesimal direction can overshoot at a full step.
    base=np.array([0.,0.]);gt=np.array([1.,0.]);d=np.array([10.,0.]);assert (gt-base)@d>0 and np.linalg.norm(base+d-gt)>np.linalg.norm(base-gt);assert np.isclose(((gt-base)@d)/(d@d),.1)
    write(RUN/'TINY_AUDIT.json',dict(status='PASS',neural_finite_updates=36000,privileged_lookup_updates=6000,curve_replay_max=maximum,independent_oracle_verified=True,tiny_group_disjoint=True,parent_control_invariance=invariance,positive_direction_does_not_imply_safe_full_step=True));print('TINY AUDIT PASS',maximum,flush=True)
if __name__=='__main__':main()
