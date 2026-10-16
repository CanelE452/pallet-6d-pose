"""Post-hoc cross-session role-map intervention; no retraining or GT selection."""
from context32 import *
import eval32 as EV
from train32 import load_model

def main():
    setup();rows=[];summary=read(RUN/'SUMMARY.json');bindings=read(RUN/'EVALUATION_BINDINGS.json')
    for seed in SEEDS:
        model=load_model('visible_hough',seed)
        for c in EV.conditions('real'):
            m,O,S=EV.inputs('real',c,seed);original=O['points'];detrows=[r for r in m['rows'] if r['detected']];sessions=[r['session'] for r in detrows];n=len(sessions)
            donor=[next((i+n//2+k)%n for k in range(n) if sessions[(i+n//2+k)%n]!=sessions[i]) for i in range(n)]
            assert all(sessions[i]!=sessions[j] for i,j in enumerate(donor))
            O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()};logits=[]
            with torch.no_grad():
                for i in range(0,n,32):logits.append(model(subset(O,slice(i,i+32)),subset(S,slice(i,i+32)))['role_logits'])
                logits=torch.cat(logits);parts=[]
                for i in range(0,n,32):
                    sl=slice(i,i+32);replacement=logits[donor[i:i+32]]
                    hook=model.roles.register_forward_hook(lambda module,args,out:replacement)
                    try:out=model(subset(O,sl),subset(S,sl))
                    finally:hook.remove()
                    parts.append({k:out[k].cpu() for k in ('points','base','strong','mix','residual')})
            name=f'cross_session_donor_seed{seed}';EV.save('real',c,name,m,dict(original_points=original,**{k:torch.cat([p[k] for p in parts]) for k in parts[0]}))
            path=RUN/'predictions/real'/c/f'{name}.npz';_,ts=EV.population('real',c);summary['results']['real/'+c][name]=E.score_file(path,m,ts)[0];bindings['files'][str(path.relative_to(ROOT))]=sha(path)
            rows.append(dict(seed=seed,condition=c,donor_indices=donor,ids=[r['id'] for r in detrows],all_different_session=True));print('CROSS_DONOR32',seed,c,flush=True)
    write(RUN/'SUMMARY.json',summary);write(RUN/'EVALUATION_BINDINGS.json',bindings);write(RUN/'CROSS_DONOR.json',dict(complete=True,exploratory=True,reason='Adjacent batch donors can be temporally similar. Use another session at same condition; keep recipient image trunk, box and point parents. Replace both role-map branches only.',GT_used=False,rows=rows))

if __name__=='__main__':main()
