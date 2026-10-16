from context31 import *
from prepare31 import load_fresh
from data30 import training_data as old_training,evaluation_data,population,conditions
import features25

def fresh(seed,split='new_train',part=None):
    OO=[];SS=[];targets_all=[];ids=[]
    for c in TRAIN_CONDITIONS:
        m,O=load_fresh(split,c);features25.RUN=RUN
        O['edge']=features25.edges_for_manifest(m,split+'_'+c)
        S=torch.load(DATA/'sources'/split/c/f'seed{seed}.pt',weights_only=True,map_location='cpu')
        ts=read(DATA/'targets'/f'{split}_{c}.json')['rows']
        if part:
            keepids={r['id'] for r in read(DATA/'POPULATIONS.json')[split][slice(0,64) if part=='calibration' else slice(64,None)]}
        else:keepids={r['id'] for r in m['rows']}
        detected=[r for r in m['rows'] if r['detected']];ix=torch.tensor([i for i,r in enumerate(detected) if r['id'] in keepids])
        OO.append(subset(O,ix));SS.append(subset(S,ix))
        targets_all.extend(t for t,r in zip(ts,m['rows']) if r['detected'] and r['id'] in keepids)
        ids.extend(r['id']+'::'+c for r in detected if r['id'] in keepids)
    return ({k:torch.cat([p[k] for p in OO]) for k in OO[0]},{k:torch.cat([p[k] for p in SS]) for k in SS[0]},torch.tensor([t['points'] for t in targets_all],dtype=torch.float32),torch.tensor([t['valid'] for t in targets_all]),ids,C.P.target_type(targets_all))

def training(seed,dataset):
    if dataset=='new':return fresh(seed)
    previous=features25.RUN
    try:
        features25.RUN=PREV25.RUN
        O,S,g,v,ids,k=old_training(seed)
    finally:features25.RUN=previous
    ix=torch.tensor([i for i,x in enumerate(ids) if x.split('::')[-1] in TRAIN_CONDITIONS])
    return subset(O,ix),subset(S,ix),g[ix],v[ix],[ids[i] for i in ix.tolist()],k[ix]

def plan(ids,seed,n=STEPS):
    groups={}
    for i,x in enumerate(ids):groups.setdefault(x.split('::')[0],[]).append(i)
    names=sorted(groups);rng=np.random.default_rng(31100+seed)
    return [[int(rng.choice(groups[names[j]])) for j in rng.choice(len(names),BATCH,replace=False)] for _ in range(n)]
