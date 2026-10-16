"""Direct-loss finite differences, context invariance, saturation and geometric regret."""
from context22 import *
from model22 import ObjectiveGate,objective22
from model20 import oracle_alpha
from diagnose20 import statistics
from eval22 import population,prediction_path

def main():
    setup();training={};checks=[];geometry=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=proposal_training(seed)
        for arm in ARMS:
            model=ObjectiveGate(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);parts=[]
            with torch.no_grad():
                for first in range(0,len(ids),32):
                    sl=slice(first,first+32);out=model(subset(O,sl),subset(r,sl),arm);parts.append({k:out[k] for k in ('base','proposal_delta','alpha','points')})
                out={k:torch.cat([x[k] for x in parts]) for k in parts[0]};stats=statistics(out,gt,valid,kind)
                for typ,k in [('all',None),('hidden',1),('outside',2)]:
                    mask=valid if k is None else valid&(kind==k);alpha=out['alpha'][mask];stats[typ].update(alpha_above99=float((alpha>.99).float().mean()),alpha_below01=float((alpha<.01).float().mean()))
                training[f'{arm}_seed{seed}']=stats
                sl=slice(0,18);o=subset(O,sl);rep=subset(r,sl);a=model(o,rep,arm);b=model(o,dict(rep,context=rep['context'].roll(1,0)+3),arm);changed=float((a['points']-b['points']).abs().max());sign=model(o,dict(rep,lines=-rep['lines']),arm);assert torch.allclose(a['points'],sign['points'],atol=1e-5,rtol=0)
                if 'nocontext' in arm:assert changed==0
            out=model(o,rep,arm);loss,_,_=objective22(out,gt[sl],valid[sl],arm);model.zero_grad(set_to_none=True);loss.backward();grad=model.evidence_project.weight.grad;norm=0 if grad is None else float(grad.norm());assert np.isfinite(norm)
            if arm.endswith('_local'):assert norm==0
            else:assert norm>0
            checks.append(dict(seed=seed,arm=arm,parent_context_intervention_max_px=changed,extra_line_gradient_norm=norm,sign_invariance=True));del model
        del O,r,gt,valid
    # Independent numerical derivative for actual distance loss, including perpendicular errors.
    rng=np.random.default_rng(22000);b=torch.tensor(rng.normal(size=(64,2)),dtype=torch.float64);d=torch.tensor(rng.normal(size=(64,2)),dtype=torch.float64);g=torch.tensor(rng.normal(size=(64,2)),dtype=torch.float64);z=torch.tensor(rng.normal(size=64),dtype=torch.float64,requires_grad=True)
    def distance(x):return (b+x.sigmoid()[:,None]*d-g).norm(dim=-1).sum()
    grad=torch.autograd.grad(distance(z),z)[0].numpy();zn=z.detach().numpy();fd=[]
    for i in range(64):
        plus=zn.copy();minus=zn.copy();plus[i]+=1e-5;minus[i]-=1e-5;fd.append(float((distance(torch.tensor(plus))-distance(torch.tensor(minus)))/2e-5))
    maximum=float(np.abs(grad-fd).max());assert maximum<1e-7
    for c,t in [('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        m,ts=population('real',c);gt=np.array([x['points'] for x in ts]);v=np.array([x['valid'] for x in ts])&np.array([x[t] for x in ts]);det=np.array([x['detected'] for x in m['rows']]);v&=det[:,None]
        for seed in SEEDS:
            p=np.load(prediction_path('real',c,f'unit_bce_seed{seed}'));base=p['base'].astype(float);delta=p['proposal_delta'].astype(float);res=gt-base;projection=(res*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-12);optimal=np.clip(projection,0,1);oracle=np.linalg.norm(base+optimal[...,None]*delta-gt,axis=-1);before=np.linalg.norm(res,axis=-1);actual=np.linalg.norm(p['points']-gt,axis=-1);norm=np.linalg.norm(delta,axis=-1);wrong=v&(projection<=0)&(norm>.1)
            geometry.append(dict(seed=seed,condition=c,points=int(v.sum()),wrong_direction_points=int(wrong.sum()),beyond_full_points=int((v&(projection>1)).sum()),base_px=float(before[v].mean()),actual_px=float(actual[v].mean()),oracle_px=float(oracle[v].mean()),learned_scale_regret_px=float((actual-oracle)[v].mean()),wrong_direction_applied_movement_px=float((norm*p['alpha'])[wrong].mean()),scope='GT optimal retained amount is unavailable during inference; residual floor includes perpendicular error and fixed-vector limits.'))
    write(RUN/'DIAGNOSIS.json',dict(status='PASS',training=training,paths=checks,distance_gradient_finite_difference_max=maximum,geometry=geometry));print('DIAGNOSIS PASS',flush=True)

if __name__=='__main__':main()
