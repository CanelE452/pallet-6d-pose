"""Training versus real vector direction, magnitude, and line-gradient path."""
from context18 import *
from model18 import VectorRefiner,objective

def summarize(base,points,gt,valid,kind):
    before=(base-gt).norm(dim=-1);after=(points-gt).norm(dim=-1);step=points-base;target=gt-base;length=step.norm(dim=-1);dot=(step*target).sum(-1);rows={}
    for typ,k in [('all',None),('hidden',1),('outside',2)]:
        mask=valid if k is None else valid&(kind==k)
        if not mask.any():continue
        move=mask&(length>.1);bad=move&(after>before+1e-4)
        rows[typ]=dict(points=int(mask.sum()),base_error_px=float(before[mask].mean()),error_px=float(after[mask].mean()),movement_px=float(length[mask].mean()),moved=int(move.sum()),helpful=int((move&(after<before-1e-4)).sum()),harmful=int(bad.sum()),wrong_direction=int((bad&(dot<=0)).sum()),overshoot=int((bad&(dot>0)).sum()))
    return rows

def main():
    setup();training_rows={};path_checks=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=training(seed)
        for arm in ARMS:
            model=VectorRefiner(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);parts=[]
            with torch.inference_mode():
                for i in range(0,len(gt),32):
                    sl=slice(i,i+32);parts.append(model(subset(O,sl),subset(rep,sl),arm)['points'])
            points=torch.cat(parts);training_rows[f'{arm}_seed{seed}']=summarize(rep['base'],points,gt,valid,kind)
            sl=slice(0,16);o=subset(O,sl);r=subset(rep,sl);out=model(o,r,arm);loss=objective(out,gt[sl],valid[sl],arm)[0];model.zero_grad(set_to_none=True);loss.backward()
            lp=model.line_project.weight.grad;line_grad=0 if lp is None else float(lp.norm());assert np.isfinite(line_grad)
            if arm.startswith('line_'):assert line_grad>0
            else:assert line_grad==0
            with torch.no_grad():
                off=model(o,r,arm,line_off=True);flipped=model(o,dict(r,lines=-r['lines']),arm);assert torch.allclose(out['points'],flipped['points'],atol=1e-5,rtol=0)
            path_checks.append(dict(seed=seed,arm=arm,line_gradient_norm=line_grad,line_off_max_change_px=float((out['points']-off['points']).abs().max()),line_sign_invariance=True));del model
    s=read(RUN/'SUMMARY.json');sessions={};ceilings={}
    from eval18 import prediction_path
    for condition,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        m,O=load('real',condition);t=targets('real',condition);gt=np.array([x['points'] for x in t]);valid=np.array([x['valid'] for x in t]);det=np.array([r['detected'] for r in m['rows']]);mask=valid&det[:,None]
        if typ:mask&=np.array([x[typ] for x in t])
        group=np.array([x['session'] for x in t]);session={};cap=[]
        for seed in SEEDS:
            rep=repcache('real',condition,seed);err=(candidates(O,rep)-torch.tensor(gt[det],dtype=torch.float32)[:,:,None]).norm(dim=-1).numpy();before=np.linalg.norm(rep['base'].numpy()-gt[det],axis=-1);radius=RADIUS*np.maximum(O['box'][:,2:].numpy()-O['box'][:,:2].numpy(),1).astype(float);radius=np.linalg.norm(radius,axis=-1);v=mask[det]
            cap.append(dict(seed=seed,old8_oracle_px=float(err.min(-1)[v].mean()),bounded_vector_infimum_px=float(np.maximum(before-radius[:,None],0)[v].mean()),scope='Posthoc GT-privileged geometric potential; not model output.'))
        ceilings[condition]=cap
        for arm in ARMS:
            errors=[]
            for seed in SEEDS:
                p=np.load(prediction_path('real',condition,f'{arm}_seed{seed}'));errors.append(np.linalg.norm(p['points']-gt,axis=-1)-np.linalg.norm(p['base']-gt,axis=-1))
            delta=np.mean(errors,axis=0);session[arm]={g:dict(points=int((mask&(group==g)[:,None]).sum()),difference_px=float(delta[mask&(group==g)[:,None]].mean())) for g in np.unique(group) if (mask&(group==g)[:,None]).any()}
        sessions[condition]=session
    write(RUN/'DIAGNOSIS.json',dict(status='PASS',training=training_rows,line_path=path_checks,real_session_vs_base=sessions,oracle=ceilings,
        definitions='For step d and GT residual t, squared error change=norm(d)^2-2*d dot t. Harm with d dot t<=0 is wrong direction; harm with positive dot is overshoot. Movement>.1px for descriptive counts only. Manual GT noise can affect tiny changes.',scope='Synthetic training versus reused real/artificial stress. Oracle is posthoc diagnostic. No extra fitting or model selection.'))
    print('DIAGNOSIS PASS',flush=True)

if __name__=='__main__':main()
