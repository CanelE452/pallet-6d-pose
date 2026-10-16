"""Common population comparisons; real labels are opened only after inference."""
import argparse
from context14 import *
from model14 import FusionEvaluator

BASELINES={'stock':(V11,'stock'),'v10_local':(V11,'v10_local'),'v10_directional':(V11,'v10_directional'),
    'v11_local_extra':(V11,'local_extra'),'v11_joint':(V11,'lines_joint'),'v12_gate':(V12,'learned_gate'),
    'v12_amodal':(V12,'gated_amodal'),'v13_scorer':(V13,'image')}

def path_for(split,condition,name,seed):
    if name in BASELINES:
        root,arm=BASELINES[name];return root/'predictions'/split/condition/('stock.npz' if arm=='stock' else f'{arm}_seed{seed}.npz')
    return RUN/'predictions'/split/condition/f'{name}_seed{seed}.npz'

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete'];lock=RUN/'EVAL_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing to overwrite evaluation')
    write(lock,dict(source_sha256=sha(Path(__file__)),training_sha256=sha(RUN/'TRAIN_PROTOCOL.json'),
        policy='Fixed final checkpoint, all predictions precede real/validation scoring; no GT, annotation flags, masks or condition names at inference.',
        controls='Final same-model line-off and affine wrong-image line donor, recompute line candidates/descriptor; point/amodal candidates identical.',baseline_sources={k:[str(v[0]),v[1]] for k,v in BASELINES.items()}))
    receipts=[]
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m,O=load(split,condition,'cuda:0');save(split,condition,'stock',m,O,O['points'])
            for seed in SEEDS:
                rep=repcache(split,condition,seed,device='cuda:0')
                for arm in ARMS:
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=FusionEvaluator(seed).cuda().eval()
                    model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict'])
                    variants=('actual','off','donor') if arm=='heldout_fusion' else ('actual',)
                    for variant in variants:
                        r=repcache(split,condition,seed,'donor','cuda:0') if variant=='donor' else rep;parts=[]
                        with torch.inference_mode():
                            for first in range(0,len(O['points']),32):
                                sl=slice(first,first+32);p=model(subset(O,sl),subset(r,sl),arm,line_off=variant=='off')
                                assert torch.isfinite(p['points']).all();parts.append({k:v.cpu() for k,v in p.items()})
                        out={k:torch.cat([p[k] for p in parts]) for k in parts[0]};points=out.pop('points')
                        name=arm if variant=='actual' else 'fusion_'+variant
                        save(split,condition,f'{name}_seed{seed}',m,O,points,out)
                        receipts.append(dict(split=split,condition=condition,name=name,seed=seed,frames=len(points),checkpoint_sha256=sha(ck)))
                    del model
            print('INFERRED',split,condition,len(O['points']),flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=receipts,targets_used=False))

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    methods=list(BASELINES)+list(ARMS)+['fusion_off','fusion_donor']
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m=read(V11/'cache'/split/condition/'manifest.json');t=target_rows(split,condition)
            gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);kind=P.target_type(t).numpy();rows={};arrays={};diag={}
            for arm in methods:
                for seed in ([1] if arm=='stock' else SEEDS):
                    path=path_for(split,condition,arm,seed);name='stock' if arm=='stock' else f'{arm}_seed{seed}'
                    rows[name],arrays[name]=E.score_file(path,m,t);p=np.load(path);bindings[str(path.relative_to(ROOT))]=sha(path)
                    assert np.array_equal(p['detected'],arrays['stock']['detected'])
                    stock=np.load(path_for(split,condition,'stock',1));det=p['detected']
                    assert np.allclose(p['original_points'][det],stock['points'][det],atol=.001,rtol=0)
                    if arm not in ARMS and not arm.startswith('fusion_'):continue
                    e=(p['candidates']-gt[:,:,None]);e=np.linalg.norm(e,axis=-1);allowed=p['eligible']>.5
                    oracle=np.where(allowed,e,np.inf).min(-1);actual=arrays[name]['error'];gain=e[...,:1]-e;d={}
                    for typ,k in [('all',None),('hidden',1),('outside',2)]:
                        mask=valid&det[:,None]
                        if k is not None:mask=mask&(kind==k)
                        if not mask.any():continue
                        applied=mask&(p['choice']>0);bad=applied&(actual>e[...,0]);cmask=mask[:,:,None]&allowed;cmask[...,0]=False
                        d[typ]=dict(points=int(mask.sum()),choice_counts=[int(((p['choice']==j)&mask).sum()) for j in range(5)],
                            candidate_oracle_mean_px=float(oracle[mask].mean()),actual_mean_px=float(actual[mask].mean()),
                            regret_px=float((actual-oracle)[mask].mean()),applied_points=int(applied.sum()),harmful_applied_points=int(bad.sum()),
                            gain_mae_px=float(np.abs(gain-p['gain'])[cmask].mean()),
                            catastrophes_base5_to20=int((mask&(e[...,0]<=5)&(actual>20)).sum()))
                    diag[name]=d
            pairs=[('heldout_fusion',b) for b in methods if b!='heldout_fusion']+[('heldout_points','v13_scorer')]
            comp={};sessions=[r['session'] for r in t]
            for a,b in pairs:
                c=COMPARE(arrays,rows,a,b,sessions,split)
                c['subgroups']={typ:COMPARE(arrays,rows,a,b,sessions,split,mask=valid&(kind==k)) for typ,k in [('hidden',1),('outside',2),('visible',0)]}
                comp[f'{a}_vs_{b}']=c
            key=split+'/'+condition;results[key]=rows;comparisons[key]=comp;diagnostics[key]=diag
            print('SCORED',key,{a:[round(rows[f'{a}_seed{s}']['frame_mean_px'],3) for s in SEEDS] for a in ('v12_gate','v12_amodal')+ARMS},flush=True)
    gates={};line_gates={}
    for seed in SEEDS:
        key=f'heldout_fusion_seed{seed}';b=f'v12_gate_seed{seed}';a=f'v12_amodal_seed{seed}'
        clean=results['real/clean'];mild=results['real/crop20'];g=dict(clean_within1=clean[key]['frame_mean_px']<=1.01*clean[b]['frame_mean_px'],mild_outside_within1=mild[key]['strata']['outside']['mean_px']<=1.01*mild[b]['strata']['outside']['mean_px'])
        for condition,typ in [('corner20','hidden'),('crop40','outside')]:
            r=results['real/'+condition];base=r[b]['strata'][typ]['mean_px'];full=r[a]['strata'][typ]['mean_px'];new=r[key]['strata'][typ]['mean_px'];g[condition+'_retain80']=base-new>=.8*(base-full)
        gates[str(seed)]=g
    for condition,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        c=comparisons['real/'+condition]['heldout_fusion_vs_heldout_points']
        gain=c['relative_gain_percent'] if typ is None else c['subgroups'][typ]['per_seed_gain_percent']
        ci=c['paired_px']['ci95'] if typ is None else c['subgroups'][typ]['ci95']
        line_gates[condition]=dict(all_seeds_gain1=all(x>=1 for x in gain),ci_upper_below0=ci[1]<0,pass_gate=all(x>=1 for x in gain) and ci[1]<0)
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_gates=line_gates,
        robustness_pass=all(all(g.values()) for g in gates.values()),scope='Reused DEV evaluation; former calibration253 now scorer training, disjoint from refinement parent training. No real tuning. No guaranteed natural occlusion or6D claim.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inference','score'],default='inference');args=ap.parse_args()
    inference() if args.phase=='inference' else scoring()
