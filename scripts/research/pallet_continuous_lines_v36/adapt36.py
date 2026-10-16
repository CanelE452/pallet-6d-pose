"""Exploratory downstream adaptation with fixed newly trained line predictors."""
from context36 import *
from train36 import load_model
from model36 import Model
from cli_train import state_sha
from eval36 import predict,save
from data32 import population,conditions
from eval32 import SPLITS
import subprocess
ADAPT=('grid','refine','both')

def make(arm,seed):
    m=load_model(arm,seed);m.requires_grad_(False);m.head.load_state_dict(m.core.head.state_dict());m.head.requires_grad_(True);m.core.line_token.requires_grad_(True);m.core.line_relation.requires_grad_(True);return m

def worker(arm,seed):
    setup36();folder=RUN/'adapt/models'/f'{arm}_seed{seed}'
    if (folder/'COMPLETE.json').exists():return
    assert not (folder/'START.json').exists();m=make(arm,seed);d=torch.load(RUN/'cache'/f'training_seed{seed}.pt',weights_only=True,map_location='cpu');x={k:v.cuda() for k,v in d['x'].items()};gt=d['gt'].cuda();valid=d['valid'].cuda();folder.mkdir(parents=True,exist_ok=True);params=[p for p in m.parameters() if p.requires_grad];init={k:v for k,v in m.state_dict().items() if k.startswith(('head.','core.line_token.','core.line_relation.'))};write(folder/'START.json',dict(source_checkpoint_sha256=sha(RUN/'models'/f'{arm}_seed{seed}'/'final.pt'),point_path_initial_sha256=state_sha(init),fixed_refiner_sha256=state_sha(m.refiner.state_dict()),plan_sha256=digest(json.dumps(d['plan'])),steps=2000,trainable_parameters=sum(p.numel() for p in params)));opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4)
    with (folder/'TRACE.jsonl').open('w') as f:
        for step,ix in enumerate(d['plan']):
            ii=torch.tensor(ix,device='cuda');out=m(subset(x,ii),arm);error=(out['points']-gt[ii]).norm(dim=-1);loss=(error/out['diag'][:,None])[valid[ii]].mean();opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True));opt.step();assert torch.isfinite(loss);f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_px=float(error[valid[ii]].mean().detach()),gradient_norm=norm))+'\n')
            if (step+1)%500==0:f.flush();print('ADAPT36',arm,seed,step+1,flush=True)
    state={k:v.detach().cpu() for k,v in m.state_dict().items()};torch.save(dict(state_dict=state,arm=arm,seed=seed,steps=2000),folder/'final.pt');assert state_sha(m.refiner.state_dict())==read(folder/'START.json')['fixed_refiner_sha256'];write(folder/'COMPLETE.json',dict(complete=True,steps=2000,checkpoint_sha256=sha(folder/'final.pt'),trace_sha256=sha(folder/'TRACE.jsonl')))

def load(arm,seed):
    m=Model(seed).cuda().eval();m.load_state_dict(torch.load(RUN/'adapt/models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);return m

def train():
    write(RUN/'ADAPT_PROTOCOL.json',dict(exploratory=True,reason='Primary refined coordinates improve line metrics but final geometry reset barely affects points. Original line token/relation encoders were frozen. Adapt these encoders with identical point heads while keeping new line predictor frozen.',arms=ADAPT,seeds=SEEDS,steps=2000,additional_models=9,additional_updates=18000,total_models=21,total_updates=42000,initialization='All point heads and line token/relation encoders reset to same V34 seed state. Line predictors from each primary final checkpoint remain fixed. Same cached features/plans and pointloss/AdamW settings.',scope='Post-primary downstream adaptation; does not replace primary4arm result or retune prior gates. No new real GT fitting.',source_sha256=sha(KIT/'adapt36.py')))
    for seed in SEEDS:
        for arm in ADAPT:subprocess.run([sys.executable,str(KIT/'adapt36.py'),'--arm',arm,'--seed',str(seed)],cwd=ROOT,check=True)
        assert len({read(RUN/'adapt/models'/f'{a}_seed{seed}'/'START.json')['point_path_initial_sha256'] for a in ADAPT})==1
    write(RUN/'ADAPT_TRAINING_COMPLETE.json',dict(complete=True,models=9,updates=18000,total_models=21,total_updates=42000))

def infer_score():
    setup36();assert read(RUN/'ADAPT_TRAINING_COMPLETE.json')['complete'];results={};comparisons={};bindings={}
    for seed in SEEDS:
        models={a:load(a,seed) for a in ADAPT}
        for split in SPLITS:
          for c in conditions(split):
            d=torch.load(RUN/'cache'/split/c/f'seed{seed}.pt',weights_only=True,map_location='cpu');x={k:v.cuda() for k,v in d['x'].items()};m,ts=population(split,c);old=np.load(V34.RUN/'predictions'/split/c/f'line_only_r0_seed{seed}.npz')
            for arm,model in models.items():
                out=predict(model,x,arm);path=RUN/'adapt/predictions'/split/c/f'{arm}_seed{seed}.npz';save(out,path,m,old);bindings[str(path.relative_to(ROOT))]=sha(path)
                if split=='real' and c=='crop20' and arm!='grid':
                    rows=[r for r in m['rows'] if r['detected']];n=len(rows);ix=[next((i+n//2+j)%n for j in range(n) if rows[(i+n//2+j)%n]['session']!=r['session']) for i,r in enumerate(rows)];donor={k:out[k][ix].cuda() for k in ('normal','offset','loc','feature')}
                    for mode in ('geometry_donor','line_off'):
                        p=RUN/'adapt/predictions'/split/c/f'{arm}_{mode}_seed{seed}.npz';save(predict(model,x,arm,mode,donor if mode=='geometry_donor' else None),p,m,old);bindings[str(p.relative_to(ROOT))]=sha(p)
            print('ADAPT_INFER36',seed,split,c,flush=True)
    old=read(RUN/'POINT_SUMMARY.json')['results']
    for split in SPLITS:
      for c in conditions(split):
        m,ts=population(split,c);rows={};arrays={}
        for p in (RUN/'adapt/predictions'/split/c).glob('*.npz'):rows[p.stem],arrays[p.stem]=E.score_file(p,m,ts)
        for seed in SEEDS:
            for a in ('base','strong','v18','v30','v34'):
                if f'{a}_seed{seed}' in old[split+'/'+c]:rows[f'{a}_seed{seed}']=old[split+'/'+c][f'{a}_seed{seed}']
        results[split+'/'+c]=rows
        if split=='real':
            arrays['stock']=arrays['grid_seed1'];rows['stock']=rows['grid_seed1'];mask=np.array([t['valid'] for t in ts])&np.array([t['outside'] for t in ts]);comp={}
            for arm in ('refine','both'):
                comp[arm]=COMPARE(arrays,rows,arm,'grid',[t['session'] for t in ts],split,mask=mask)
                if c=='crop20':
                    for mode in ('geometry_donor','line_off'):comp[arm+'_'+mode]=COMPARE(arrays,rows,arm+'_'+mode,arm,[t['session'] for t in ts],split,mask=mask)
            comparisons[split+'/'+c]=comp
    gates={}
    for arm in ADAPT:
        per={}
        for seed in SEEDS:
            a=f'{arm}_seed{seed}';b=f'base_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[a]['frame_mean_px']<=1.01*rr[b]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[a]['strata']['outside']['mean_px']<=1.01*rr[b]['strata']['outside']['mean_px']
            for c,t in [('corner20','hidden'),('crop40','outside')]:
                rr=results['real/'+c];base=rr[b]['strata'][t]['mean_px'];g[c+'_retain90']=base-rr[a]['strata'][t]['mean_px']>=.9*(base-rr[f'v18_seed{seed}']['strata'][t]['mean_px'])
            per[str(seed)]=g
        comp=comparisons['real/crop20'].get(arm);added=bool(comp and min(comp['per_seed_gain_percent'])>=1 and comp['ci95'][1]<0);gates[arm]=dict(per_seed=per,stable=all(all(r.values()) for r in per.values()),added=added)
    write(RUN/'ADAPT_SUMMARY.json',dict(complete=True,exploratory=True,results=results,comparisons=comparisons,gates=gates,accepted=any(v['stable'] and v['added'] for v in gates.values())));write(RUN/'ADAPT_INFERENCE_COMPLETE.json',dict(complete=True,files=bindings));print('ADAPT_RESULT36',gates,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ADAPT);p.add_argument('--seed',type=int);p.add_argument('--phase',choices=['train','eval']);a=p.parse_args()
    if a.arm:worker(a.arm,a.seed)
    elif a.phase=='eval':infer_score()
    else:train();infer_score()
