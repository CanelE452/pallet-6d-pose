from context42 import *
from experiment42 import load_inputs

def main():
    write(RUN/'PROBE_PROTOCOL.json',dict(exploratory=True,primary_sha256=sha(RUN/'SUMMARY.json'),question='Do valid constraints cover hard-point improvement opportunities, and how much is lost by classification versus residual ranking?',definitions='Postprimary nofits. GTbest-of3+fallback within GTvisible/physical/constraint or actualacceptedconstraint slots; this removesrankingerror but usesGTpoint. Opportunity=bestcandidate improvesfallbackEuclideanerror>1px. Selectedpoint easy/hard strata useoriginalV34>=10px, notfallback. Perseed masks preserved. CandidateGToracle improvements outsidevalidlabel are not reliablephysical information.'))
    p=population();part=partition(p);g=p['g'];rows=[];predictions={k:[] for k in ('visible','physical','constraint','accepted')};reference=[];hard=[]
    for seed in SEEDS:
        a,v,X=load_inputs(p,seed);q=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));base=np.linalg.norm(q['fallback']-g['gt'],axis=-1);ce=np.linalg.norm(a['candidate']-g['gt'][:,:,None],axis=-1);key='constraint_visual_pre_all';sel=q['selected_'+key];take=part['check'][:,None]&(sel>=0);switch=np.linalg.norm(q['points_'+key]-q['fallback'],axis=-1)>1e-8;categories={}
        for kind,mask in [('all',take),('hard',take&q['hard']),('easy',take&~q['hard'])]:
            categories[kind]=dict(selected=int(mask.sum()),switches=int((mask&switch).sum()),fallback_error_mean_px=float(base[mask].mean()) if mask.any() else None,fallback_error_median_px=float(np.median(base[mask])) if mask.any() else None)
        pools={}
        for pool in predictions:
            allow=q['accepted_constraint_visual'] if pool=='accepted' else q['label_'+pool]&a['valid'];err=np.where(allow,ce,np.inf);k=err.argmin(-1);use=err.min(-1)<base;point=np.where(use[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],k],q['fallback']);predictions[pool].append(point);gain=base-np.linalg.norm(point-g['gt'],axis=-1);check=part['check'][:,None];pools[pool]=dict(opportunities_over1px=int(((gain>1)&check).sum()),hard_opportunities_over1px=int(((gain>1)&check&q['hard']).sum()))
            np.savez_compressed(RUN/'arrays'/f'oracle_{pool}_seed{seed}.npz',points=point)
        unmasked=np.where(a['valid'],ce,np.inf).min(-1);pools['all_candidates']=dict(opportunities_over1px=int(((base-unmasked>1)&part['check'][:,None]).sum()),hard_opportunities_over1px=int(((base-unmasked>1)&part['check'][:,None]&q['hard']).sum()));rows.append(dict(selected_strata=categories,pools=pools));reference.append(base);hard.append(q['hard'])
    results={};R=np.stack(reference)
    for pool,pts in predictions.items():
        E=np.linalg.norm(np.stack(pts)-g['gt'][None],axis=-1);results[pool]={}
        for key,M in masks(p,np.stack(hard)):
            if key.startswith('check/'):results[pool][key]=metric(E,R,M,p['groups'])
    write(RUN/'PROBE.json',dict(complete=True,exploratory=True,rows=rows,oracle_results=results));print('PROBE42',rows,{k:v['check/all/all_detected']['mean_px'] for k,v in results.items()},flush=True)
if __name__=='__main__':main()
