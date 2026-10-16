"""Explicitly posthoc mechanism check of strip's observed crop40 improvement."""
from context16 import *
from model16 import EvidenceEvaluator
from eval16 import prediction_path

def main():
    setup();lock=RUN/'POSTHOC_INTERVENTION_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing overwrite')
    write(lock,dict(summary_sha256=sha(RUN/'SUMMARY.json'),posthoc=True,
        trigger='strip improved crop40 vs control but role donor was similar to actual. Test dependence on image-specific evidence.',
        variants='strip_off zero branch embedding; strip_donor roll source features by one in detected population; candidate set, point patches, geometry unchanged.',
        no_training_or_selection=True,conditions=['corner20','crop20','crop30','crop40'],splits=['real','synth_val']))
    results={};comps={};bindings={}
    for split in ('synth_val','real'):
        for condition in ('corner20','crop20','crop30','crop40'):
            m,O=load(split,condition,'cuda:0');donor=O['features'].roll(1,0)
            for seed in SEEDS:
                rep=repcache(split,condition,seed,device='cuda:0');model=EvidenceEvaluator(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'strip_seed{seed}'/'final.pt',weights_only=True)['state_dict'])
                for variant in ('off','donor'):
                    parts=[]
                    with torch.inference_mode():
                        for first in range(0,len(O['points']),32):
                            sl=slice(first,first+32);out=model(subset(O,sl),subset(rep,sl),'strip',evidence_off=variant=='off',evidence_features=donor[sl] if variant=='donor' else None)
                            parts.append({k:v.cpu() for k,v in out.items() if k in ('points','gain','choice','eligible','candidates')})
                    out={k:torch.cat([p[k] for p in parts]) for k in parts[0]};points=out.pop('points');save(split,condition,f'strip_{variant}_seed{seed}',m,O,points,out)
                del model
            t=targets(split,condition);valid=np.array([x['valid'] for x in t]);typ='hidden' if condition=='corner20' else 'outside';mask=valid&np.array([x[typ] for x in t]);rows={};arrays={}
            for a in ('stock','strip','strip_off','strip_donor'):
                for seed in ([1] if a=='stock' else SEEDS):
                    name='stock' if a=='stock' else f'{a}_seed{seed}';p=prediction_path(split,condition,name);rows[name],arrays[name]=E.score_file(p,m,t);bindings[str(p.relative_to(ROOT))]=sha(p)
                    if a.startswith('strip_'):
                        f=np.load(p);det=f['detected'];q=f['choice'][det].astype(int);cand=f['candidates'][det];assert np.array_equal(cand[np.arange(len(q))[:,None],np.arange(8)[None],q],f['points'][det])
                        original=np.load(prediction_path(split,condition,f'strip_seed{seed}'));assert np.array_equal(cand,original['candidates'][det])
            results[split+'/'+condition]=rows;comps[split+'/'+condition]={a:COMPARE(arrays,rows,'strip',a,[x['session'] for x in t],split,mask=mask) for a in ('strip_off','strip_donor')}
            print('POSTHOC',split,condition,comps[split+'/'+condition],flush=True)
    write(RUN/'POSTHOC_INTERVENTIONS.json',dict(results=results,comparisons=comps,bindings=bindings,posthoc=True,exact_candidate_and_selection_audit=True))

if __name__=='__main__':main()
