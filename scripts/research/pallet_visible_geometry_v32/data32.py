from context32 import *
from prepare32 import load_render
from data31 import fresh as fresh31
import eval31
KEYS32=('features','content','edge','box','image_hw')

def render_data(seed,split=None,c=None):
    m,O=load_render();S=torch.load(RUN/'parents'/f'seed{seed}.pt',weights_only=True,map_location='cpu');ts=read(RUN/'TARGETS.json')['rows'];take=[i for i,r in enumerate(m['rows']) if (split is None or r['split']==split) and (c is None or r['condition']==c)];rows=[m['rows'][i] for i in take];targets=[ts[i] for i in take];ix=torch.tensor([r['observation_index'] for r in rows if r['detected']]);return dict(rows=rows),{k:O[k][ix] for k in O},subset(S,ix),targets

def training(seed):
    oldO,oldS,g,v,ids,kind=fresh31(seed);n=len(ids)
    m,obs,source,ts=render_data(seed,'train');tt=[t for t,r in zip(ts,m['rows']) if r['detected']];newids=[r['id'] for r in m['rows'] if r['detected']]
    O={k:torch.cat((oldO[k],obs[k])) for k in KEYS32};S={k:torch.cat((oldS[k],source[k])) for k in ('base','strong')};gt=torch.cat((g,torch.tensor([t['points'] for t in tt],dtype=torch.float32)));valid=torch.cat((v,torch.tensor([t['valid'] for t in tt])))
    zeros=torch.zeros(n,24,48,48);physical=torch.cat((zeros,obs['physical_target']));virtual=torch.cat((zeros,obs['virtual_target']));sup=torch.cat((torch.zeros(n,dtype=torch.bool),torch.ones(len(tt),dtype=torch.bool)))
    return O,S,gt,valid,physical,virtual,sup,ids+newids,n

def plan(ids,n,seed):
    old={};new={}
    for i,x in enumerate(ids):
        group=x.split('::')[0];(old if i<n else new).setdefault(group,[]).append(i)
    rng=np.random.default_rng(32100+seed);out=[]
    for step in range(STEPS):
        row=[]
        for groups in (old,new):
            names=sorted(groups)
            for j in rng.choice(len(names),8,replace=False):row.append(int(rng.choice(groups[names[j]])))
        out.append(row)
    return out

def inputs(split,c,seed):
    if split=='render_check':
        m,O,S,ts=render_data(seed,'check',c);return m,{k:O[k] for k in KEYS32},S
    return eval31.inputs(split,c,seed)
def population(split,c):
    if split=='render_check':
        m=read(RUN/'cache/manifest.json');ts=read(RUN/'TARGETS.json')['rows'];take=[i for i,r in enumerate(m['rows']) if r['split']=='check' and r['condition']==c];return dict(rows=[m['rows'][i] for i in take]),[ts[i] for i in take]
    return eval31.population(split,c)
def conditions(split):return ('clean','crop20','crop40','corner20','physical_panel') if split=='render_check' else eval31.conditions(split)
