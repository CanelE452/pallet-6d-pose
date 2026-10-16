"""Residual population shift, training discrimination, and actual projected gradient check."""
from context21 import *
from model20 import CauseGate,objective,oracle_alpha
from population20 import summarize
from diagnose20 import statistics
from train21 import flat_grad,project_pair
import torch.nn.functional as F

def main():
    setup();population_rows=[];training={};gradient=[]
    for seed in SEEDS:
        groups_by_c={c:{} for c in ('crop20','crop40')}
        for pop in ('old','new'):
            O,r,gt,valid,ids,kind=proposal_training(seed) if pop=='old' else new_training(seed);condition=np.array([x.rsplit('::',1)[1] for x in ids]);base=r['base'].cpu().numpy();delta=r['proposal_delta'].cpu().numpy();truth=gt.cpu().numpy();outside=valid.cpu().numpy()&(kind.cpu().numpy()==2)
            for c in groups_by_c:groups_by_c[c][pop+'_train']=summarize(base,delta,truth,outside&(condition==c)[:,None])
            for arm in [a for a in ARMS if a.startswith('unit_'+pop+'_')]:
                model=CauseGate(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);parts=[]
                with torch.no_grad():
                    for first in range(0,len(ids),32):
                        sl=slice(first,first+32);out=model(subset(O,sl),subset(r,sl),arm);parts.append({k:out[k] for k in ('base','proposal_delta','alpha','points')})
                training[f'{arm}_seed{seed}']=statistics({k:torch.cat([x[k] for x in parts]) for k in parts[0]},gt,valid,kind)
                # Fresh graph: sum of the two raw group gradients must equal the original objective gradient.
                sl=slice(0,32);out=model(subset(O,sl),subset(r,sl),arm);loss,_,target=objective(out,gt[sl],valid[sl],'unit_weighted');v=valid[sl];stop=v&(target<=.05);terms=F.binary_cross_entropy_with_logits(out['logits'],target,reduction='none')*out['proposal_delta'].norm(dim=-1).detach();params=list(model.parameters());g0=flat_grad(terms[stop].sum()/v.sum(),params);g1=flat_grad(terms[v&~stop].sum()/v.sum(),params);full=flat_grad(loss,params);summed_error=float((g0+g1-full).abs().max());assert summed_error<1e-4
                merged,cos,conflict=project_pair(g0,g1);a=g0.detach().cpu().numpy().astype(float);b=g1.detach().cpu().numpy().astype(float);dot=a@b;expected=a+b if dot>=0 else (a-dot/(b@b)*b)+(b-dot/(a@a)*a);project_error=float(np.abs(merged.detach().cpu().numpy()-expected).max());assert project_error<1e-4
                gradient.append(dict(arm=arm,seed=seed,sum_gradient_max_error=summed_error,projection_float64_max_error=project_error,cosine=cos,conflict=conflict));del model
            del O,r,gt,valid
        for c,groups in groups_by_c.items():
            for split in ('new_check','real'):
                if split=='new_check':m,_=new_load(split,c);r=new_rep(split,c,seed);ts=new_targets(split,c)
                else:m,_,r=eval_inputs(split,c,seed);ts=targets(split,c)
                det=np.array([x['detected'] for x in m['rows']]);truth=np.array([x['points'] for x in ts])[det];mask=(np.array([x['valid'] for x in ts])&np.array([x['outside'] for x in ts]))[det];groups[split]=summarize(r['base'].numpy(),r['proposal_delta'].numpy(),truth,mask)
            weights=np.array([x['n'] for x in groups['real']],float);weights/=weights.sum();matched={}
            for split,rs in groups.items():
                assert all(x['n']>0 for w,x in zip(weights,rs) if w>0);matched[split]={k:float(sum(w*x[k] for w,x in zip(weights,rs) if w>0)) for k in ('alpha','stop_fraction','base_error_px')}
            population_rows.append(dict(seed=seed,condition=c,point_id_matched_means=matched,real_point_id_weights=weights.tolist(),point_rows=groups))
    write(RUN/'DIAGNOSIS.json',dict(status='PASS',training=training,gradient=gradient,population=population_rows,scope='GT-based posthoc diagnosis; point-ID weights not training weights. Matching does not eliminate pose/texture/scale confounding. Gradient conflicts alone do not establish detrimental interference.'));print('DIAGNOSIS PASS',flush=True)

if __name__=='__main__':main()
