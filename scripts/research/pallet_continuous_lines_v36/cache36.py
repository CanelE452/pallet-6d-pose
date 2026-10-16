from context36 import *
from model36 import Frozen
from data32 import training,plan,inputs,population,conditions
from eval32 import SPLITS

def capture(model,O,S):
    O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()};parts=[]
    with torch.no_grad():
        for i in range(0,len(O['box']),32):parts.append({k:v.cpu() for k,v in model.capture(subset(O,slice(i,i+32)),subset(S,slice(i,i+32))).items()})
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

def main():
    register();setup36();(RUN/'cache').mkdir(exist_ok=True);write(RUN/'CACHE_PROTOCOL.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in [KIT/'model36.py',KIT/'cache36.py',KIT/'context36.py']},GT_model_forward=False));meta=read(V35.RUN/'POPULATION.json')['rows'];physical=np.load(V35.RUN/'geometry.npz');lookup={r['id']:i for i,r in enumerate(meta)}
    for seed in SEEDS:
        model=Frozen(seed).cuda();path=RUN/'cache'/f'training_seed{seed}.pt'
        if not path.exists():
            O,S,g,v,phys,virt,sup,ids,n=training(seed);x=capture(model,O,S);support=torch.zeros(len(ids),12,2,2);hasline=torch.zeros(len(ids),12,dtype=torch.bool)
            for i in range(n,len(ids)):
                j=lookup[ids[i]];support[i]=torch.tensor(physical['support'][j],dtype=torch.float32);hasline[i]=torch.tensor(physical['length'][j]>0)
            x['supervised']=sup;steps=plan(ids,n,seed);assert steps==read(V32.RUN/f'PLAN_seed{seed}.json')['plan'];torch.save(dict(x=x,gt=g,valid=v,support=support,hasline=hasline,ids=ids,plan=steps),path);print('CACHE36 train',seed,len(ids),flush=True)
        for split in SPLITS:
            for c in conditions(split):
                p=RUN/'cache'/split/c/f'seed{seed}.pt';p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists():continue
                m,O,S=inputs(split,c,seed);x=capture(model,O,S);torch.save(dict(x=x,ids=[r['id'] for r in m['rows'] if r['detected']]),p);print('CACHE36',seed,split,c,flush=True)
        del model
    files=list((RUN/'cache').rglob('*.pt'));write(RUN/'CACHE_COMPLETE.json',dict(complete=True,files={str(p.relative_to(ROOT)):sha(p) for p in files},GT_model_forward=False))
if __name__=='__main__':main()
