from context33 import *
from model33 import Correspondence,objective
from model32 import VisibleGeometry
from data32 import training,plan
from cli_train import state_sha

def precheck():
    setup();B=2;O=dict(features=torch.randn(B,192,24,24,device='cuda'),content=torch.ones(B,1,24,24,dtype=torch.bool,device='cuda'),edge=torch.randn(B,3,96,96,device='cuda'),box=torch.tensor([[-20.,0.,100.,60.],[0.,0.,100.,100.]],device='cuda'),image_hw=torch.tensor([[100.,100.]]*B,device='cuda'));base=torch.rand(B,8,2,device='cuda')*80;S=dict(base=base,strong=base+5);gt=base+3;v=torch.ones(B,8,dtype=torch.bool,device='cuda');target=torch.zeros(B,24,48,48,device='cuda');target[:,:,20,5:40]=1;sup=torch.ones(B,dtype=torch.bool,device='cuda');rows=[]
    for arm in ARMS:
        model=Correspondence(1).cuda();out=model(O,S,arm);assert torch.equal(out['points'],base)
        with torch.no_grad():model.head[-1].weight.normal_(0,.01)
        out=model(O,S,arm);loss,terms=objective(out,gt,v,target,sup,'visible_hough');loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if 'line_attention' in out:
            for k in ('line_attention','end_attention'):assert torch.allclose(out[k].sum(-1),torch.ones(B,16,device='cuda'),atol=1e-6)
        rows.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),role_gradient=float(model.roles.weight.grad.abs().sum())))
    model=Correspondence(1).cuda();old=VisibleGeometry(1).cuda()
    with torch.no_grad():old.head[-1].weight.normal_(0,.01)
    model.load_state_dict(old.state_dict(),strict=False)
    for arm,oldarm in [('pooled','visible_hough'),('point_aux','point')]:assert torch.equal(model(O,S,arm)['points'],old(O,S,oldarm)['points'])
    with torch.no_grad():model.head[-1].weight.normal_(0,.01)
    model.zero_grad();out=model(O,S,'query');loss,_=objective(out,gt,v,target,sup,'latent_hough');loss.backward()
    gradients={k:float(getattr(model,k)[0].weight.grad.abs().sum()) for k in ('line_token','end_token','query_encode','line_relation','end_relation')};assert min(gradients.values())>0
    # Endpoint weighted centroids recover a single positive subcell pixel.
    maps=torch.zeros(B,24,48,48,device='cuda');maps[:,12,10,13]=1;e=model.evidence(torch.randn(B,32,48,48,device='cuda'),maps,O['box'],O['image_hw']);cell=(10//6)*8+13//6;wh=O['box'][:,2:]-O['box'][:,:2];expected=(torch.tensor([13,10],device='cuda')/47*2-1)*wh/2/wh.norm(dim=-1)[:,None]
    assert torch.allclose(e['end_loc'][:,cell],expected,atol=1e-6)
    start=time.monotonic()
    for _ in range(10):model.zero_grad();out=model(O,S,'query');out['points'].sum().backward()
    torch.cuda.synchronize();write(RUN/'PRECHECK.json',dict(status='PASS',arms=rows,point_loss_gradients=gradients,weighted_endpoint_centroid_verified=True,pooled_original_forward_identical=True,seconds_per_two_sample_step=(time.monotonic()-start)/10));print('PRECHECK33 PASS',flush=True)

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=2000,actual_updates=36000,data='Exact V32 cached inputs, physical targets and plans; 8 legacy +8 renderer per batch. Existing reused real297 DEV and prospective/fresh/render check.',architecture='Same V32 trunk/roles/point head/physical loss. Query each base/restored point separately. Incident roles:4 top Hough bins, weighted visible support center/spread;64 spatial endpoint cells per role with weighted centroids.32d dot attention with relative geometry, score prior and zero null token; concatenate both parent queries into64d per evidence type.',controls='pooled exact V32 global feature path; query joint; no_geometry zeros explicit positional token channels/relation bias but retains role and support-conditioned appearance; line_only/end_only remove opposite tokens DURING training; point_aux zeros both tokens. Same total parameters but active capacity differs.',selection='Fixed final2000 steps, no real GT fitting or best-seed selection.',optimizer='V32 FP32 AdamW3e-4 wd1e-4 clip5; point normalized distance+0.05balanced physical BCE.',primary='query vs pooled AND end_only: crop20 outside >=1% improvement EACH seed and real9session paired95% upper<0. Allseed clean/crop10/20/30 within1%V12; corner20/crop40 retain90%V18 gain. Additional strong-parent preservation reported separately.',limits='Top4 bins may duplicate modes; bin locations discrete (gradients through votes/support only). CAD chain roles approximate physical-to-cuboid relation. Same2assets train/check. No sealed-final or unconditional improvement claim.'))
    previous=read(V32.RUN/'PRIOR_BINDINGS.json')['files'];previous.update(read(V32.RUN/'COMPLETION.json')['files']);write(RUN/'PRIOR_BINDINGS.json',dict(files=previous));(RUN/'PURPOSE.md').write_text('# V33\n\nTest point-specific line-support/endpoint correspondence against global pooling under identical V32 supervision.\n')
    record('V33 점별 선 support/끝점 대응6대조×3시드 사전등록. 물리 라벨/원본/배치/예산은 V32 유지, 최종2000step 및 전체 안정성+Hough 추가 이득 기준 고정.')

def main():
    register();precheck();setup();files=[KIT/'model33.py',KIT/'train33.py',KIT/'context33.py',V32.KIT/'data32.py',V32.KIT/'model32.py'];hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    if (RUN/'TRAIN_PROTOCOL.json').exists():assert read(RUN/'TRAIN_PROTOCOL.json')['source_hashes']==hashes
    else:write(RUN/'TRAIN_PROTOCOL.json',dict(source_hashes=hashes,protocol_sha256=sha(RUN/'PROTOCOL.json')))
    for seed in SEEDS:
        O,S,g,v,phys,virt,sup,ids,n=training(seed);steps=plan(ids,n,seed);assert steps==read(V32.RUN/f'PLAN_seed{seed}.json')['plan'];write(RUN/f'PLAN_seed{seed}.json',dict(plan_sha256=digest(json.dumps(steps)),source=str((V32.RUN/f'PLAN_seed{seed}.json').relative_to(ROOT)),source_sha256=sha(V32.RUN/f'PLAN_seed{seed}.json')))
        O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()};g=g.cuda();v=v.cuda();phys=phys.cuda();sup=sup.cuda()
        for arm in ARMS:
            folder=RUN/'models'/f'{arm}_seed{seed}'
            if (folder/'COMPLETE.json').exists():continue
            assert not (folder/'START.json').exists(),folder;folder.mkdir(parents=True,exist_ok=True);model=Correspondence(seed).cuda();write(folder/'START.json',dict(initial_sha256=state_sha(model.state_dict()),plan_sha256=digest(json.dumps(steps)),parameters=sum(p.numel() for p in model.parameters()),steps=2000));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,ix in enumerate(steps):
                    ii=torch.tensor(ix,device='cuda');out=model(subset(O,ii),subset(S,ii),arm);loss,terms=objective(out,g[ii],v[ii],phys[ii],sup[ii],'visible_hough');assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=norm,**{k:float(x.detach()) for k,x in terms.items()}))+'\n')
                    if (step+1)%400==0:f.flush();print('TRAIN33',seed,arm,step+1,round(time.monotonic()-tick),flush=True)
            path=folder/'final.pt';torch.save(dict(state_dict={k:x.detach().cpu() for k,x in model.state_dict().items()},arm=arm,seed=seed,steps=2000),path);write(folder/'COMPLETE.json',dict(complete=True,steps=2000,seed=seed,arm=arm,checkpoint_sha256=sha(path),seconds=time.monotonic()-tick));del model,opt
        del O,S,g,v,phys,sup
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=36000,models=18));print('TRAINING33 COMPLETE',flush=True)

def load_model(arm,seed):
    model=Correspondence(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);return model

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--precheck-only',action='store_true');a=p.parse_args()
    if a.precheck_only:RUN.mkdir(parents=True,exist_ok=True);precheck()
    else:main()
