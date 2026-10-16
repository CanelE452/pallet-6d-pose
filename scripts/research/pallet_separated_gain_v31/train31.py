from context31 import *
from data31 import training,plan
from model31 import PointSelector,SeparatedGain,selector_loss,correction_loss
from model30 import DirectionalAdaptive,objective
from cli_train import state_sha

def precheck():
    torch.set_num_threads(2);B=2
    O=dict(features=torch.randn(B,192,24,24),content=torch.ones(B,1,24,24,dtype=torch.bool),edge=torch.randn(B,3,96,96),box=torch.tensor([[0.,0.,100.,100.]]).repeat(B,1),image_hw=torch.tensor([[100.,100.]]).repeat(B,1))
    base=torch.tensor([12.123,27.234])[None,None].repeat(B,8,1)
    S=dict(base=base,strong=base+torch.tensor([5.,-3.]),lines=torch.tensor([1.,0.,-14.567])[None,None,None,None].repeat(B,2,12,3,1),mass=torch.ones(B,2,12,3)/3,expert_points=torch.stack((base+6,base-4),2))
    gt=base+torch.tensor([1.,4.]);valid=torch.ones(B,8,dtype=torch.bool)
    sel=PointSelector(1);so=sel(O,S);loss,_=selector_loss(so,S,gt,valid);loss.backward();assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in sel.parameters() if p.grad is not None)
    changed=dict(S,lines=S['lines']+100,mass=S['mass']*100,expert_points=S['expert_points']+1000)
    assert torch.equal(sel(O,S)['parent_gain'],sel(O,changed)['parent_gain'])
    for arm in ('split','generic'):
        sel=PointSelector(1);model=SeparatedGain(1,sel);out=model(O,S,arm);loss,_=correction_loss(out,gt,valid);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.selector.parameters())
        assert (out['gain'].abs()<=out['move']+1e-5).all() and (out['lower']<=out['gain']+1e-5).all() and (out['lower']>=-out['move']-1e-5).all()
        assert torch.equal(model(O,S,arm,line_off=True)['points'],out['parent_points'])
        if arm=='split':
            delta=out['candidate_points'][:,:,8:]-out['anchors'][:,:,8:];assert delta[...,1].abs().max()<1e-6
        else:
            other=model(O,changed,arm);assert torch.equal(out['points'],other['points']) and torch.equal(out['rank'],other['rank'])
    write(RUN/'PRECHECK.json',dict(status='PASS',finite_gradients=True,selector_frozen_during_correction=True,selector_line_invariant=True,generic_line_invariant=True,normal_tangent_preserved=True,bounded_mean_and_lower=True,line_off_exact_parent=True))
    print('PRECHECK31 PASS',flush=True)

def fit(model,kind,arm,O,S,g,v,steps,folder,seed):
    if (folder/'COMPLETE.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);assert not (folder/'START.json').exists(),folder
    trainable=[p for p in model.parameters() if p.requires_grad]
    write(folder/'START.json',dict(initial_sha256=state_sha(model.state_dict()),plan_sha256=digest(json.dumps(steps)),steps=len(steps),parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in trainable)))
    optimizer=torch.optim.AdamW(trainable,lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
    with (folder/'TRACE.jsonl').open('w') as f:
        for step,ix in enumerate(steps):
            ii=torch.tensor(ix,device='cuda');o=subset(O,ii);s=subset(S,ii)
            if kind=='selector':out=model(o,s);loss,terms=selector_loss(out,s,g[ii],v[ii])
            elif kind=='joint':out=model(o,s,'normal_gain');loss,terms=objective(out,g[ii],v[ii])
            else:out=model(o,s,arm);loss,terms=correction_loss(out,g[ii],v[ii])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(trainable,5,error_if_nonfinite=True));optimizer.step()
            f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=norm,**{k:float(x.detach()) for k,x in terms.items()}))+'\n')
            if (step+1)%250==0:f.flush();print('TRAIN31',folder.name,step+1,round(float(terms['point_error'].detach()),3),round(time.monotonic()-tick),flush=True)
    path=folder/'final.pt';torch.save(dict(state_dict={k:p.detach().cpu() for k,p in model.state_dict().items()},kind=kind,arm=arm,seed=seed,steps=len(steps)),path)
    write(folder/'COMPLETE.json',dict(complete=True,seed=seed,kind=kind,arm=arm,steps=len(steps),checkpoint_sha256=sha(path),seconds=time.monotonic()-tick))

def load_selector(dataset,seed,device='cuda'):
    model=PointSelector(seed).to(device);model.load_state_dict(torch.load(RUN/'models'/f'selector_{dataset}_seed{seed}'/'final.pt',weights_only=True,map_location=device)['state_dict']);return model.eval()

def load_model(arm,seed,device='cuda'):
    if arm.startswith('selector'):return load_selector(arm.split('_')[-1],seed,device)
    model=DirectionalAdaptive(seed) if arm.startswith('joint') else SeparatedGain(seed,PointSelector(seed))
    model.to(device);model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location=device)['state_dict']);return model.eval()

def main():
    setup();assert read(RUN/'DATA_COMPLETE.json')['complete'];precheck()
    code={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py') if p.name in ('context31.py','data31.py','model31.py','backbone31.py','train31.py','prepare31.py')}
    write(RUN/'TRAIN_PROTOCOL.json',dict(source_hashes=code,protocol_sha256=sha(RUN/'PROTOCOL.json')))
    for seed in SEEDS:
        for dataset in ('old','new'):
            O,S,g,v,ids,kind=training(seed,dataset);steps=plan(ids,seed)
            write(RUN/f'PLAN_{dataset}_seed{seed}.json',dict(ids=ids,plan=steps,originals=len({i.split('::')[0] for i in ids}),valid_points=int(v.sum()),sha256=digest(json.dumps(steps))))
            O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()};g=g.cuda();v=v.cuda()
            model=DirectionalAdaptive(seed).cuda();fit(model,'joint','normal_gain',O,S,g,v,steps,RUN/'models'/f'joint_{dataset}_seed{seed}',seed);del model
            model=PointSelector(seed).cuda();fit(model,'selector',dataset,O,S,g,v,steps[:1000],RUN/'models'/f'selector_{dataset}_seed{seed}',seed);del model
            for family in (('split','generic') if dataset=='new' else ('split',)):
                model=SeparatedGain(seed,load_selector(dataset,seed)).cuda();fit(model,'correction',family,O,S,g,v,steps[1000:],RUN/'models'/f'{family}_{dataset}_seed{seed}',seed);del model
            del O,S,g,v
    runs=[read(p) for p in (RUN/'models').glob('*/COMPLETE.json')];assert len(runs)==21 and sum(r['steps'] for r in runs)==27000
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=27000,checkpoints=21,runs=runs))
    print('TRAINING31 COMPLETE',flush=True)

if __name__=='__main__':main()
