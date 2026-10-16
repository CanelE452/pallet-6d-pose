"""Training/DEV retention discrimination and true clipped-line/gradient checks."""
from context20 import *
ARMS=("unit_weighted_long","unit_weighted_cf","unit_weighted_cf_local")
from model20 import CauseGate,objective,observed_samples

def statistics(out,gt,valid,kind):
    base=out['base'];delta=out['proposal_delta'];alpha=out['alpha'];before=(base-gt).norm(dim=-1);full=(base+delta-gt).norm(dim=-1);after=(out['points']-gt).norm(dim=-1);rows={}
    for typ,k in [('all',None),('hidden',1),('outside',2)]:
        mask=valid if k is None else valid&(kind==k)
        if not mask.any():continue
        bad=mask&(full>before+.1);good=mask&(full<before-.1)
        rows[typ]=dict(points=int(mask.sum()),base_px=float(before[mask].mean()),full_px=float(full[mask].mean()),after_px=float(after[mask].mean()),mean_alpha=float(alpha[mask].mean()),mean_alpha_bad=float(alpha[bad].mean()) if bad.any() else None,mean_alpha_good=float(alpha[good].mean()) if good.any() else None,remaining_harms=int((mask&(after>before+.1)).sum()))
    return rows

def main():
    setup();training_rows={};paths=[];clips=[];encoded_scales={}
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed)
        for arm in ARMS:
            model=CauseGate(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);parts=[]
            with torch.inference_mode():
                for first in range(0,len(gt),32):
                    sl=slice(first,first+32);out=model(subset(O,sl),subset(rep,sl),arm);parts.append({k:v for k,v in out.items() if k in ('base','proposal_delta','alpha','points')})
            output={k:torch.cat([p[k] for p in parts]) for k in parts[0]};training_rows[f'{arm}_seed{seed}']=statistics(output,gt,valid,kind);sl=slice(0,18);o=subset(O,sl);r=subset(rep,sl);captured={};hook=model.point_encode.register_forward_pre_hook(lambda module,args:captured.update(x=args[0].detach()));out=model(o,r,arm);hook.remove();encoded_scales[f'{arm}_seed{seed}']={name:float(captured['x'][...,lo:hi].square().mean().sqrt()) for name,lo,hi in [('patch',0,144),('parent_context',144,208),('amodal_delta',223,225),('proposal_direction',225,227),('proposal_length',227,228)]};loss=objective(out,gt[sl],valid[sl],"unit_weighted")[0];model.zero_grad(set_to_none=True);loss.backward();grad=model.evidence_project.weight.grad;norm=0 if grad is None else float(grad.norm());assert np.isfinite(norm)
            if arm.endswith('_local'):assert norm==0
            else:assert norm>0
            with torch.no_grad():
                flipped=model(o,dict(r,lines=-r['lines']),arm);assert torch.allclose(out['points'],flipped['points'],atol=1e-5,rtol=0);off=model(o,r,arm,evidence_off=True)
            paths.append(dict(seed=seed,arm=arm,context_gradient_norm=norm,off_max_change_px=float((out['points']-off['points']).abs().max()),sign_invariance=True));del model
        # Recompute geometry on evaluation samples, verify incidence in independent float64.
        for c in ('clean','corner20','crop20','crop40'):
            m,o,r=eval_inputs('real',c,seed);sample=observed_samples(o,r,'line');pos=sample['pos'].numpy().astype(float).reshape(-1,8,3,12,2);valid_s=sample['valid'].numpy().reshape(-1,8,3,12);idx=np.argsort(-r['line_weights'][...,:-1].numpy(),axis=-1,kind='stable')[...,:3];lines=r['lines'].numpy();chosen=np.take_along_axis(np.broadcast_to(lines[:,None],(len(lines),8,*lines.shape[1:])),idx[...,None],axis=2).astype(float);distance=np.abs((pos*chosen[...,None,:2]).sum(-1)+chosen[...,None,2])/np.maximum(np.linalg.norm(chosen[...,:2],axis=-1)[...,None],1e-9);max_distance=float(distance[valid_s].max()) if valid_s.any() else 0;assert max_distance<.002
            t=targets('real',c);det=np.array([x['detected'] for x in m['rows']]);v=np.array([x['valid'] for x in t])[det]
            if c=='corner20':v&=np.array([x['hidden'] for x in t])[det]
            elif c.startswith('crop'):v&=np.array([x['outside'] for x in t])[det]
            frac=valid_s.mean((-1,-2));clips.append(dict(seed=seed,condition=c,points=int(v.sum()),no_valid_line_samples=int(((frac==0)&v).sum()),mean_coverage=float(frac[v].mean()),max_line_incidence_error_px=max_distance))
    write(RUN/'DIAGNOSIS_CF.json',dict(status='PASS',training=training_rows,encoded_scales=encoded_scales,paths=paths,clipped_line_geometry=clips,scope='Training residuals were seen by frozen V18. DEV generalization is separate. Geometric sample availability does not establish physical line semantics or causal usefulness.'));print('DIAGNOSIS PASS',flush=True)

if __name__=='__main__':main()
