from context31 import *
from eval31 import inputs,population,TRAINED,forward
from train31 import load_model
from model31 import PointSelector
import data30

def main():
    setup();assert read(RUN/'SUMMARY.json')['complete'];bound=read(RUN/'PRIOR_BINDINGS.json')['files']
    for p,h in bound.items():assert sha(ROOT/p)==h,('prior changed',p)
    for p,h in read(RUN/'TRAIN_PROTOCOL.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
    bindings=read(RUN/'EVALUATION_BINDINGS.json')['files'];checked=0;max_bound=0.
    for p,h in bindings.items():
        assert sha(ROOT/p)==h;p=ROOT/p;z=np.load(p)
        if 'parent_points' not in z:continue
        split,c=p.parent.parent.name,p.parent.name;m,ts=population(split,c)
        assert z['ids'].tolist()==[r['id'] for r in m['rows']]
        gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts])&z['detected'][:,None]
        mask=np.broadcast_to(valid[:,:,None],z['gain'].shape)
        assert np.isfinite(z['points'][valid]).all()
        assert (np.abs(z['gain'][mask])<=z['move'][mask]+1e-3).all()
        assert (z['lower'][mask]<=z['gain'][mask]+1e-3).all() and (z['lower'][mask]>=-z['move'][mask]-1e-3).all()
        actual=np.linalg.norm(z['anchors']-gt[:,:,None],axis=-1)-np.linalg.norm(z['candidate_points']-gt[:,:,None],axis=-1)
        assert (np.abs(actual[mask])<=z['move'][mask]+1e-3).all()
        e=np.linalg.norm(z['points']-gt,axis=-1);row=read(RUN/'SUMMARY.json')['results'][split+'/'+c][p.stem]
        assert abs(e[valid].mean()-row['strata']['all']['mean_px'])<1e-5
        checked+=1
    replay=[]
    for seed in SEEDS:
        # Full frozen point/DHT sources replay on actual cached real observations.
        m,O,S=inputs('real','crop20',seed);_,_,r=data30.inputs('real','crop20',seed)
        o={k:v[:3].cuda() for k,v in O.items()};rr={k:v[:3].cuda() for k,v in r.items()}
        live=data30.live_sources(o,rr,seed)
        for k in S:assert torch.allclose(live[k].cpu(),S[k][:3],atol=.003,rtol=0),k
        model=load_model('split_new',seed)
        with torch.no_grad():out=model(o,live);off=model(o,live,line_off=True)
        p=np.load(RUN/'predictions/real/crop20'/f'split_new_seed{seed}.npz');det=p['detected']
        delta=float(np.max(np.abs(out['points'].cpu().numpy()-p['points'][det][:3])));assert delta<.003
        assert torch.equal(off['points'],out['parent_points'])
        changed=dict(live,lines=live['lines'].roll(1,0),mass=live['mass'].roll(1,0),expert_points=live['expert_points'].roll(1,0))
        generic=load_model('generic_new',seed)
        with torch.no_grad():a=generic(o,live,'generic');b=generic(o,changed,'generic')
        assert torch.equal(a['points'],b['points']) and torch.equal(a['rank'],b['rank'])
        for family in ('split_new','generic_new'):
            trained=torch.load(RUN/'models'/f'{family}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']
            selector=torch.load(RUN/'models'/f'selector_new_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']
            assert all(torch.equal(v,trained['selector.'+k]) for k,v in selector.items())
        replay.append(dict(seed=seed,full_parent_module_max_px=delta))
    write(RUN/'AUDIT.json',dict(status='PASS',prior_bound_files=len(bound),prediction_files=len(bindings),separated_metric_and_bound_files=checked,full_parent_replay=replay,trained_generic_line_invariant=True,trained_selectors_frozen_identical=True,GT_forward=False,manual_GT_edited=False))
    print('AUDIT31 PASS',checked,flush=True)

if __name__=='__main__':main()
