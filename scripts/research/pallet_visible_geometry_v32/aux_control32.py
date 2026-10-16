"""Post-result bounded control: physical supervision without line-token fusion."""
from context32 import *
from model32 import VisibleGeometry,objective
from data32 import training,plan
from cli_train import state_sha
import eval32 as EV

def train():
    setup();write(RUN/'AUX_CONTROL_PROTOCOL.json',dict(exploratory=True,reason='Initial visible_hough beats point/virtual on crop20, but donor nearly unchanged and Hough-only ablation tiny. Isolate shared-trunk physical supervision from fusion.',arm='visible_aux_only',seeds=SEEDS,steps=2000,extra_updates=6000,forward='point: same model and physical role head, line/endpoint tokens zero',loss='visible_hough: same physical auxiliary targets/loss',same_data_plan_initialization=True,no_change_to_first12_models=True,not_a_preregistered_primary_comparison=True))
    for seed in SEEDS:
        folder=RUN/'models'/f'visible_aux_only_seed{seed}'
        if (folder/'COMPLETE.json').exists():continue
        O,S,g,v,phys,virt,sup,ids,n=training(seed);steps=plan(ids,n,seed);assert steps==read(RUN/f'PLAN_seed{seed}.json')['plan'];O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()};g=g.cuda();v=v.cuda();phys=phys.cuda();sup=sup.cuda()
        model=VisibleGeometry(seed).cuda();folder.mkdir(parents=True,exist_ok=True);initial=state_sha(model.state_dict());assert initial==read(RUN/'models'/f'visible_hough_seed{seed}'/'START.json')['initial_sha256'];write(folder/'START.json',dict(initial_sha256=initial,plan_sha256=digest(json.dumps(steps)),parameters=sum(p.numel() for p in model.parameters()),steps=2000))
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
        with (folder/'TRACE.jsonl').open('w') as f:
            for step,ix in enumerate(steps):
                ii=torch.tensor(ix,device='cuda');out=model(subset(O,ii),subset(S,ii),'point');loss,terms=objective(out,g[ii],v[ii],phys[ii],sup[ii],'visible_hough');assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=norm,**{k:float(x.detach()) for k,x in terms.items()}))+'\n')
                if (step+1)%500==0:f.flush();print('AUX_TRAIN32',seed,step+1,flush=True)
        path=folder/'final.pt';torch.save(dict(state_dict={k:x.detach().cpu() for k,x in model.state_dict().items()},arm='visible_aux_only',seed=seed,steps=2000),path);write(folder/'COMPLETE.json',dict(complete=True,steps=2000,seed=seed,arm='visible_aux_only',checkpoint_sha256=sha(path),seconds=time.monotonic()-tick));del model,opt,O,S,g,v,phys,sup
    write(RUN/'AUX_TRAINING_COMPLETE.json',dict(complete=True,updates=6000,models=3))

def infer():
    setup()
    for seed in SEEDS:
        model=VisibleGeometry(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'visible_aux_only_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict'])
        for split in EV.SPLITS:
            for c in EV.conditions(split):
                path=RUN/'predictions'/split/c/f'visible_aux_only_seed{seed}.npz'
                if path.exists():continue
                m,O,S=EV.inputs(split,c,seed);original=O.get('points')
                if original is None:original=EV.render_data(seed,'check',c)[1]['points']
                O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()};parts=[]
                with torch.no_grad():
                    for i in range(0,len(O['box']),32):
                        sl=slice(i,i+32);out=model(subset(O,sl),subset(S,sl),'point');out['lines']=EV.raw_lines(out['hough'],O['box'][sl]);keys=['points','base','strong','mix','residual','lines']+(['role_maps'] if split=='render_check' else []);parts.append({k:out[k].cpu() for k in keys})
                EV.save(split,c,f'visible_aux_only_seed{seed}',m,dict(original_points=original,**{k:torch.cat([p[k] for p in parts]) for k in parts[0]}));del O,S
        del model
    write(RUN/'AUX_INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False))

def score():
    summary=read(RUN/'SUMMARY.json');roles=read(RUN/'ROLE_METRICS.json');bindings=read(RUN/'EVALUATION_BINDINGS.json');comp={}
    for split in EV.SPLITS:
        for c in EV.conditions(split):
            m,ts=EV.population(split,c);arrays={};rows={}
            for seed in SEEDS:
                for a in ('visible_hough','visible_aux_only','point','base'):
                    name=f'{a}_seed{seed}';path=RUN/'predictions'/split/c/f'{name}.npz';row,arr=E.score_file(path,m,ts);summary['results'][split+'/'+c][name]=row;rows[name]=row;arrays[name]=arr;bindings['files'][str(path.relative_to(ROOT))]=sha(path)
                if split=='render_check':
                    p=np.load(RUN/'predictions'/split/c/f'visible_aux_only_seed{seed}.npz');_,obs,_,_=EV.render_data(seed,'check',c);roles['rows'][f'{split}/{c}/visible_aux_only_seed{seed}']=EV.role_metrics(p['role_maps'][p['detected']].astype(float),obs['physical_target'].numpy())
            if split=='real':
                arrays['stock']=arrays['base_seed1'];rows['stock']=rows['base_seed1'];valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();dd={}
                for a,b in [('visible_hough','visible_aux_only'),('visible_aux_only','point')]:
                    v=COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split);v['subgroups']={typ:COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split,mask=valid&(kind==k)) for typ,k in [('hidden',1),('outside',2)]};dd[a+'_vs_'+b]=v
                comp[split+'/'+c]=dd
    write(RUN/'SUMMARY.json',summary);write(RUN/'ROLE_METRICS.json',roles);write(RUN/'EVALUATION_BINDINGS.json',bindings);write(RUN/'AUX_COMPARISONS.json',dict(exploratory=True,comparisons=comp));print('AUX_CONTROL32 COMPLETE',flush=True)

if __name__=='__main__':train();infer();score()
