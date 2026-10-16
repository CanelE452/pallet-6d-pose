from context32 import *
from model32 import VisibleGeometry,objective,Hough
from data32 import training,plan
from cli_train import state_sha

def precheck():
    torch.set_num_threads(2);B=2;O=dict(features=torch.randn(B,192,24,24),content=torch.ones(B,1,24,24,dtype=torch.bool),edge=torch.randn(B,3,96,96),box=torch.tensor([[0.,0.,100.,100.]]).repeat(B,1),image_hw=torch.tensor([[100.,100.]]).repeat(B,1));base=torch.rand(B,8,2)*70+15;S=dict(base=base,strong=base+torch.randn_like(base)*10);gt=base+4;valid=torch.ones(B,8,dtype=torch.bool);target=torch.zeros(B,24,48,48);target[:,:,12,4:40]=1;sup=torch.ones(B,dtype=torch.bool)
    h=Hough();assert torch.allclose(h.basis.sum(1),torch.full((576,),24.),atol=1e-5)
    x=torch.randn(B,12,24,24,requires_grad=True);out=h(x);out.square().mean().backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    count={}
    for arm in ARMS:
        model=VisibleGeometry(1);out=model(O,S,arm);assert torch.equal(out['points'],base);count[arm]=sum(p.numel() for p in model.parameters());loss,_=objective(out,gt,valid,target,sup,arm);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if arm=='point':assert torch.equal(model(O,S,arm,intervention='donor')['points'],out['points'])
    model=VisibleGeometry(1)
    with torch.no_grad():model.head[-1].weight.normal_(0,.01)
    out=model(O,S,'latent_hough');loss,_=objective(out,gt,valid,target,sup,'latent_hough');loss.backward();gradient=float(model.roles.weight.grad.abs().sum());assert gradient>0
    assert len(set(count.values()))==1
    write(RUN/'PRECHECK.json',dict(status='PASS',parameters=count,hough_vote_mass_preserved=True,subpixel_rho_splat_gradients=True,initial_output_exact_base=True,all_arm_finite_gradients=True,point_loss_to_role_head_gradient=gradient))
    print('PRECHECK32 PASS',count,flush=True)

def main():
    setup();assert read(RUN/'DATA_COMPLETE.json')['complete'];precheck();files={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py') if p.name in ('model32.py','context32.py','geometry32.py','render32.py','prepare32.py','data32.py','train32.py')}
    write(RUN/'TRAIN_PROTOCOL.json',dict(source_hashes=files,protocol_sha256=sha(RUN/'PROTOCOL.json')))
    for seed in SEEDS:
        O,S,g,v,phys,virt,sup,ids,n=training(seed);steps=plan(ids,n,seed);write(RUN/f'PLAN_seed{seed}.json',dict(ids=ids,legacy_observations=n,render_observations=len(ids)-n,plan=steps,plan_sha256=digest(json.dumps(steps))))
        O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()};g=g.cuda();v=v.cuda();phys=phys.cuda();virt=virt.cuda();sup=sup.cuda()
        for arm in ARMS:
            folder=RUN/'models'/f'{arm}_seed{seed}'
            if (folder/'COMPLETE.json').exists():continue
            folder.mkdir(parents=True,exist_ok=True);assert not (folder/'START.json').exists(),folder
            model=VisibleGeometry(seed).cuda();write(folder/'START.json',dict(initial_sha256=state_sha(model.state_dict()),plan_sha256=digest(json.dumps(steps)),parameters=sum(p.numel() for p in model.parameters()),steps=2000));optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,ix in enumerate(steps):
                    ii=torch.tensor(ix,device='cuda');out=model(subset(O,ii),subset(S,ii),arm);target=phys[ii] if arm=='visible_hough' else virt[ii];loss,terms=objective(out,g[ii],v[ii],target,sup[ii],arm);assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));optimizer.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=norm,**{k:float(x.detach()) for k,x in terms.items()}))+'\n')
                    if (step+1)%400==0:f.flush();print('TRAIN32',seed,arm,step+1,round(float(terms['point_error'].detach()),3),round(time.monotonic()-tick),flush=True)
            path=folder/'final.pt';torch.save(dict(state_dict={k:p.detach().cpu() for k,p in model.state_dict().items()},arm=arm,seed=seed,steps=2000),path);write(folder/'COMPLETE.json',dict(complete=True,arm=arm,seed=seed,steps=2000,checkpoint_sha256=sha(path),seconds=time.monotonic()-tick));del model,optimizer
        del O,S,g,v,phys,virt,sup
    runs=[read(p) for p in (RUN/'models').glob('*/COMPLETE.json')];assert len(runs)==12
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=24000,runs=runs));print('TRAINING32 COMPLETE',flush=True)

def load_model(arm,seed):
    model=VisibleGeometry(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);return model

if __name__=='__main__':main()
