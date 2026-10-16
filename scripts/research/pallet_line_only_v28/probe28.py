"""Case-specific line-only interventions; no new fitting or GT-based deployed routing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_direction_utility_v27'))
from context27 import *
from eval27 import inputs,population
from context14 import P,V11
from optional_model import predict_lines
from plpose_local.heads import decode_modes,roi_lines_to_raw
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_line_only_v28'
PRIOR=ROOT/'data/pallet/results/pallet_direction_utility_v27'
CASE=next(c for c in read(ROOT/'data/pallet/results/pallet_direction_utility_v27_cases/CASES.json')['cases'] if c['key']=='missed')
METHODS=('base','actual_v27','line_mix','line_keep','line_mix_full_amount','top_line_full','line_oracle_GT')

def calculate(p,gt):
    b=p['base'].astype(float);d=p['candidates'].astype(float);a=p['alpha'].astype(float);w=p['weights'].astype(float);valid=p['candidate_mask']==1
    def mix(first,include_keep=False,full=False):
        allowed=valid.copy();allowed[:,:,:first]=False
        if include_keep:allowed[:,:,0]=True
        score=np.where(allowed,w,0);mass=score.sum(-1,keepdims=True);ww=score/np.maximum(mass,1e-300);delta=(ww[...,None]*(1 if full else a[...,None])*d).sum(2)
        return b+delta,ww
    lm,lw=mix(4);lk,kw=mix(4,True);full,fw=mix(4,full=True);res=gt-b
    with np.errstate(invalid='ignore'):
        opt=np.clip((res[:,:,None]*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1);cand=b[:,:,None]+opt[...,None]*d;err=np.linalg.norm(cand-gt[:,:,None],axis=-1)
    allow=valid.copy();allow[:,:,:4]=False;allow[:,:,0]=True;err=np.where(allow,err,np.inf);best=err.argmin(-1);oracle=np.take_along_axis(cand,best[...,None,None],2)[:,:,0]
    return dict(base=b,actual_v27=p['points'].astype(float),line_mix=lm,line_keep=lk,line_mix_full_amount=full,top_line_full=b+d[:,:,4],line_oracle_GT=oracle),dict(line_weights=lw,keep_weights=kw,oracle_choice=best,oracle_amount=opt)

def score():
    RUN.mkdir(parents=True,exist_ok=True);rows=[];caseout=None;sources={}
    write(RUN/'PROTOCOL.json',dict(scope='Exploratory post-V27 inference interventions requested on displayed case3; not preregistered new training or stable-improvement claim.',target=CASE['id'],point=CASE['point'],methods=METHODS,definitions='line_mix: remove keep+point/x/y candidates, renormalize existing learned line/pair weights, retain predicted amounts. line_keep: retain keep and line candidates. Full amount removes learned amount. Point location/association still used in these corrections. Separate DHT gate bypass removes explicit point hints from line generation.',new_training=False,GT_used_for_oracle_only=True))
    for c in ('clean','corner20','crop10','crop20','crop30','crop40'):
        m,t=population('real',c);gt=np.array([x['points'] for x in t]);valid=np.array([x['valid'] for x in t]);typ='all' if c=='clean' else 'hidden' if c=='corner20' else 'outside';mask=valid if typ=='all' else valid&np.array([x[typ] for x in t])
        for seed in SEEDS:
            path=PRIOR/'predictions/real'/c/f'line_role_utility_seed{seed}.npz';p=np.load(path);assert p['ids'].tolist()==[x['id'] for x in t];out,extra=calculate(p,gt);keep=mask&p['detected'][:,None];sources[str(path.relative_to(ROOT))]=sha(path)
            for arm,points in out.items():
                error=np.linalg.norm(points-gt,axis=-1);perframe=np.where(keep,error,0).sum(-1)/np.maximum(keep.sum(-1),1);score0=perframe[keep.any(-1)].mean() if c=='clean' else error[keep].mean();move=np.linalg.norm(points-out['base'],axis=-1)[keep];rows.append(dict(condition=c,subset=typ,seed=seed,method=arm,requested_frames=len(t),detected_frames=int(p['detected'].sum()),total_points=int(mask.sum()),detected_points=int(keep.sum()),mean_px=float(score0),mean_move_px=float(move.mean()),median_move_px=float(np.median(move)),near_zero_move_fraction=float((move<.1).mean())))
            if c=='crop20' and seed==1:
                i=p['ids'].tolist().index(CASE['id']);j=CASE['point'];caseout=dict(id=CASE['id'],point=j,row=m['rows'][i],gt=gt[i,j].tolist(),points={k:v[i,j].tolist() for k,v in out.items()},errors={k:float(np.linalg.norm(v[i,j]-gt[i,j])) for k,v in out.items()},line_conditional_weights=extra['line_weights'][i,j].tolist(),original_weights=p['weights'][i,j].tolist(),predicted_amounts=p['alpha'][i,j].tolist(),oracle_choice=int(extra['oracle_choice'][i,j]),oracle_amount=float(extra['oracle_amount'][i,j,extra['oracle_choice'][i,j]]),lines=CASE['top3_lines'],top3_line_indices=CASE['top3_line_indices']);np.savez_compressed(RUN/'case_intervention.npz',**{k:v[i] for k,v in out.items()})
    write(RUN/'SUMMARY.json',dict(complete=True,rows=rows,case=caseout,sources=sources));print('LINE_ONLY_CASE',caseout['errors'],flush=True)

def dht_only():
    setup();m,O,r=inputs('real','crop20',1);i=next(i for i,row in enumerate(m['rows']) if row['id']==CASE['id']);j=m['rows'][i]['observation_index'];O={k:v[j:j+1].cuda() for k,v in O.items()};model=P.C.dht(1,'cuda:0');state=torch.load(V11/'models/lines_joint_seed1/final.pt',weights_only=False,map_location='cpu');model.load_state_dict(state['dht_state_dict']);model.eval();results={}
    from model import point_maps
    with torch.inference_mode():
        for mode in ('guided','zero_point_maps','gate_bypass'):
            if mode=='gate_bypass':
                x=model.trunk(O['features'],O['content'],O['K'],O['dims'],O['box'],O['image_hw']);logits=model.line(x);ll,mass=decode_modes(logits,model.lattice,model.thetas,model.rhos,model.modes);lines=roi_lines_to_raw(ll,O['box'])
            else:
                maps=point_maps(O['points'],O['box']);maps=torch.zeros_like(maps) if mode=='zero_point_maps' else maps;out=model(O,maps,decode=True);lines,mass=out['lines'],out['mass']
            results[mode]=dict(lines=lines.flatten(1,2)[0].cpu().numpy().tolist(),mass=mass.flatten(1,2)[0].cpu().numpy().tolist())
        # Ordinary forward replay must reproduce actual displayed DHT coordinates.
        cached=r['lines'][j].numpy();error=float(np.abs(np.array(results['guided']['lines'])-cached).max());assert error<.001
        O2=dict(O,points=O['points']+1000)
        def bare(o):return model.line(model.trunk(o['features'],o['content'],o['K'],o['dims'],o['box'],o['image_hw']))
        assert torch.equal(bare(O),bare(O2))
    write(RUN/'DHT_ONLY.json',dict(complete=True,variants=results,guided_cache_replay_max=error,bypass_point_coordinate_invariance=True,checkpoint_sha256=sha(V11/'models/lines_joint_seed1/final.pt'),limits='Same DHT weights, explicit point-neighborhood gate ablated at inference. YOLO box/features/K/dims remain. This is not a separately trained image-only line detector, nor point-free8corner pose decoding. Top mass ranks within learned output channels need not indicate semantic correctness.'))
    print('DHT_ONLY',error,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['score','dht'],required=True);a=p.parse_args();score() if a.phase=='score' else dht_only()
