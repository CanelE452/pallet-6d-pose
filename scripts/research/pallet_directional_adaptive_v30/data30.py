"""Frozen strong point parents and V29 line sources; GT kept outside forward."""
from context30 import *
from eval27 import inputs,population,conditions
from train27 import point_proposal
from train25 import dataset
from model29 import LineExpert,LineDecoder
from plpose_local.heads import roi_lines_to_raw

def line_models(seed):
    lines={};decoders={}
    for a in ('guided','independent'):
        m=LineExpert(V29.V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt',seed,a).cuda().eval()
        m.load_state_dict(torch.load(V29.RUN/'models'/f'{a}_dht_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);lines[a]=m
        d=LineDecoder(seed).cuda().eval();d.load_state_dict(torch.load(V29.RUN/'models'/f'{a}_decoder_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);decoders[a]=d
    return lines,decoders

def live_sources(O,r,seed):
    strong=r['base']+point_proposal(O,r,seed);models,decoders=line_models(seed);parts=[]
    with torch.no_grad():
        for i in range(0,len(O['box']),32):
            sl=slice(i,i+32);obs=subset(O,sl);ll=[];mm=[];pp=[]
            for a in ('guided','independent'):
                out=models[a](obs,True);ll.append(out['raw_lines']);mm.append(out['mass']);pp.append(decoders[a](out['lines'],out['mass'],obs['box'])['points'])
            parts.append(dict(lines=torch.stack(ll,1),mass=torch.stack(mm,1),expert_points=torch.stack(pp,2)))
    return dict(base=r['base'],strong=strong,**{k:torch.cat([p[k] for p in parts]) for k in parts[0]})

def training_data(seed):
    O,r,gt,valid,ids,kind,_=dataset(seed);old=read(V29.RUN/f'PLAN_seed{seed}.json');assert old['ids']==ids
    allowed=set(old['router_sources']);ix=torch.tensor([i for i,x in enumerate(ids) if x.split('::')[0] in allowed]);sourceids=[ids[i] for i in ix.tolist()]
    folder=RUN/'training_cache';folder.mkdir(exist_ok=True);path=folder/f'sources_seed{seed}.pt'
    OO={k:v[ix] for k,v in O.items()};rr={k:v[ix].cuda() for k,v in r.items()}
    if path.exists():S=torch.load(path,weights_only=True,map_location='cpu')
    else:
        S=live_sources({k:v.cuda() for k,v in OO.items()},rr,seed);S={k:v.cpu() for k,v in S.items()};torch.save(S,path)
    return OO,S,gt[ix],valid[ix],sourceids,kind[ix]

def evaluation_data(split,c,seed):
    m,O,r=inputs(split,c,seed);path=V29.RUN/'predictions'/split/c/f'bundle_seed{seed}.npz';b=np.load(path);assert b['ids'].tolist()==[x['id'] for x in m['rows']]
    det=b['detected'];p=np.load(OLD27.RUN/'predictions'/split/c/f'v25_point_seed{seed}.npz');assert np.array_equal(det,p['detected']) and np.array_equal(b['ids'],p['ids'])
    S=dict(base=r['base'],strong=torch.tensor(p['points'][det]),lines=torch.tensor(np.stack((b['guided_lines'][det],b['independent_lines'][det]),1)),mass=torch.tensor(np.stack((b['guided_mass'][det],b['independent_mass'][det]),1)),expert_points=torch.tensor(b['candidates'][det,:,2:]))
    return m,O,S,r

def plan(ids,seed):
    groups={}
    for i,fid in enumerate(ids):groups.setdefault(fid.split('::')[0],[]).append(i)
    names=sorted(groups);rng=np.random.default_rng(30100+seed);rows=[]
    for _ in range(STEPS):
        take=rng.choice(len(names),BATCH,replace=False);rows.append([int(rng.choice(groups[names[i]])) for i in take])
    return rows
