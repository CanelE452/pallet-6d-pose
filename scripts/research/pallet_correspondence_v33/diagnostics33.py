"""Post-result read-only comparisons and a backward determinism capability probe."""
from context33 import *
from data32 import population
from model33 import Correspondence
from model32 import VisibleGeometry
from cli_train import state_sha
import torch.nn.functional as F

def main():
    setup();repeat=[]
    for seed in SEEDS:
        model=Correspondence(seed);old=VisibleGeometry(seed);common={k:v for k,v in model.state_dict().items() if k in old.state_dict()};assert state_sha(common)==read(V32.RUN/'models'/f'visible_hough_seed{seed}'/'START.json')['initial_sha256']
        for a,b in [('pooled','visible_hough'),('point_aux','visible_aux_only')]:
            oldtrace=[json.loads(x) for x in (V32.RUN/'models'/f'{b}_seed{seed}'/'TRACE.jsonl').read_text().splitlines()];newtrace=[json.loads(x) for x in (RUN/'models'/f'{a}_seed{seed}'/'TRACE.jsonl').read_text().splitlines()];first=next((dict(step=i+1,old_loss=x['loss'],new_loss=y['loss'],difference=y['loss']-x['loss']) for i,(x,y) in enumerate(zip(oldtrace,newtrace)) if x['loss']!=y['loss']),None);repeat.append(dict(seed=seed,arm=a,common_initialization_exact=True,first_trace=oldtrace[0],first_loss_difference=first))
    original=torch.are_deterministic_algorithms_enabled();probe={}
    try:
        torch.use_deterministic_algorithms(True);x=torch.randn(1,1,4,4,device='cuda',requires_grad=True);grid=torch.rand(1,2,2,2,device='cuda')*2-1;F.grid_sample(x,grid,align_corners=True).sum().backward();probe=dict(deterministic_backward_available=True)
    except RuntimeError as e:probe=dict(deterministic_backward_available=False,error=str(e))
    finally:torch.use_deterministic_algorithms(original)
    comparisons={};s=read(RUN/'SUMMARY.json')
    for c in ('clean','corner20','crop20','crop40'):
        m,ts=population('real',c);arrays={};rows={}
        for a in ('line_only','end_only','pooled','point_aux'):
            for seed in SEEDS:
                name=f'{a}_seed{seed}';rows[name],arrays[name]=E.score_file(RUN/'predictions/real'/c/f'{name}.npz',m,ts)
        arrays['stock']=arrays['pooled_seed1'];rows['stock']=rows['pooled_seed1'];valid=np.array([t['valid'] for t in ts]);mask=valid&np.array([t['hidden' if c=='corner20' else 'outside'] for t in ts]) if c!='clean' else None
        comparisons[c]={b:COMPARE(arrays,rows,'line_only',b,[t['session'] for t in ts],'real',mask=mask) for b in ('pooled','end_only','point_aux')}
    template=[]
    for c in ('clean','corner20','crop20'):
        p=np.load(RUN/'predictions/real'/c/'query_seed1.npz');m,ts=population('real',c);rr=[r for r in m['rows'] if r['detected']];sessions=[r['session'] for r in rr];n=len(rr);donor=[next((i+n//2+k)%n for k in range(n) if sessions[(i+n//2+k)%n]!=sessions[i]) for i in range(n)];maps=p['role_maps'][p['detected']].astype(float);bb=np.array([r['box'] for r in rr]);half=np.maximum(bb[:,2:]-bb[:,:2],1)/2;diag=np.linalg.norm(2*half,axis=-1);normal=p['line_normal'][p['detected']];angle=np.arctan2(normal[...,1]*half[:,None,1],normal[...,0]*half[:,None,0]);ti=np.rint(angle*24/np.pi).astype(int)%24;length=np.linalg.norm(np.stack((np.cos(angle),np.sin(angle)),-1)/half[:,None],axis=-1);rho=p['line_offset'][p['detected']]*diag[:,None]*length;ri=np.rint((rho+2**.5)*47/(2*2**.5)).astype(int);bins=(ti*48+ri).reshape(n,12,4);agreement=np.mean([len(set(bins[i,j])&set(bins[d,j]))/4 for i,d in enumerate(donor) for j in range(12)])
        template.append(dict(condition=c,cross_session_map_L1=float(np.abs(maps-maps[donor]).mean()),cross_session_top4_bin_set_overlap=float(agreement),limits='seed1, matching role and same condition; appearance in tokens remains recipient-specific. Overlap is descriptive, not chance-corrected.'))
    write(RUN/'POSTHOC_DIAGNOSTICS.json',dict(exploratory=True,repeated_controls=repeat,training_deterministic_algorithms_enabled=original,grid_sample_capability=probe,line_only_comparisons=comparisons,role_template_checks=template,limits='Same initialization/plans but FP32 CUDA backward not guaranteed bitwise deterministic. Observed tiny early numerical divergence followed by material final variation; not an isolated proof of its sole cause. No checkpoints/targets changed, no new optimizer updates.'))
    print('POSTHOC33 COMPLETE',probe,flush=True)

if __name__=='__main__':main()
