"""Independent arithmetic, data splits, training parity and immutable parent audit."""
from context29 import *
from eval27 import population,conditions

def main():
    summary=read(RUN/'SUMMARY.json');train=read(RUN/'TRAINING_COMPLETE.json');assert train['updates']==25200 and len(train['runs'])==24
    protocol=read(RUN/'PROTOCOL.json')
    for p,h in protocol['sources'].items():assert sha(ROOT/p)==h,p
    for seed in SEEDS:
        plan=read(RUN/f'PLAN_seed{seed}.json');ex=set(plan['expert_sources']);rt=set(plan['router_sources']);assert not ex&rt
        for name,key,allowed in [('expert','expert_plan',ex),('router','router_plan',rt)]:
            for batch in plan[key]:assert all(plan['ids'][i].split('::')[0] in allowed for i in batch)
        for names in [('guided_dht','independent_dht'),('guided_decoder','independent_decoder')]:
            starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in names]
            for key in ('initial_state_sha256','parameters','plan_sha256','steps'):assert len({s[key] for s in starts})==1,(names,key)
    traces=0
    for r in train['runs']:
        folder=RUN/'models'/f'{r["name"]}_seed{r["seed"]}';assert sha(folder/'final.pt')==r['checkpoint_sha256'];rows=[json.loads(s) for s in (folder/'TRACE.jsonl').read_text().splitlines()]
        assert len(rows)==r['steps'];assert [s['step'] for s in rows]==list(range(1,r['steps']+1))
        assert all(np.isfinite(s['loss']) and np.isfinite(s['gradient_norm']) for s in rows);traces+=len(rows)
    files=0;maxerr=0.;counts={}
    for split in EVAL_SPLITS:
        for c in conditions(split):
            m,t=population(split,c);g=np.array([x['points'] for x in t]);v=np.array([x['valid'] for x in t]);det=np.array([x['detected'] for x in m['rows']]);kind=C.P.target_type(t).numpy()
            counts[split+'/'+c]=dict(frames=len(t),detected=int(det.sum()),valid_points=int(v.sum()),detected_valid_points=int((v&det[:,None]).sum()),hidden=int((v&(kind==1)).sum()),outside=int((v&(kind==2)).sum()))
            for seed in SEEDS:
                b=np.load(RUN/'predictions'/split/c/f'bundle_seed{seed}.npz');assert b['ids'].tolist()==[x['id'] for x in t]
                for name in ('stock','point','guided','independent','adaptive','no_state','point_router','predicted_state','oracle_expert_GT','oracle_state_GT'):
                    p=np.load(RUN/'predictions'/split/c/f'{name}_seed{seed}.npz');assert np.array_equal(p['detected'],det);assert np.isfinite(p['points'][det]).all();assert np.allclose(p['original_points'],b['candidates'][:,:,0],equal_nan=True)
                    e=np.linalg.norm(p['points']-g,axis=-1);frame=np.where(v[det],e[det],0).sum(-1)/v[det].sum(-1)
                    delta=abs(float(frame.mean())-summary['results'][split+'/'+c][f'{name}_seed{seed}']['frame_mean_px']);maxerr=max(maxerr,delta);assert delta<1e-6
                    if name in ('adaptive','no_state','point_router'):
                        ii=p['chosen'][det].astype(int);expected=np.take_along_axis(b['candidates'][det],ii[...,None,None],2)[:,:,0];assert np.array_equal(p['points'][det],expected)
                    if name=='oracle_expert_GT':
                        errors=np.linalg.norm(b['candidates']-g[:,:,None],axis=-1);assert np.allclose(e[v&det[:,None]],errors.min(-1)[v&det[:,None]])
                    files+=1
    bindings=read(RUN/'EVALUATION_BINDINGS.json')['files']
    for p,h in bindings.items():assert sha(ROOT/p)==h,p
    priorfiles=0
    for prior in ('pallet_observation_v25','pallet_direction_utility_v27','pallet_direction_utility_v27_cases','pallet_line_only_v28'):
        path=ROOT/'data/pallet/results'/prior/'COMPLETION.json'
        if not path.exists():continue
        for p,h in read(path).get('files',{}).items():assert sha(ROOT/p)==h,(prior,p);priorfiles+=1
    checks=read(RUN/'INFERENCE_COMPLETE.json')['checks'];assert len(checks)==24 and all(x['independent_point_shift_invariant'] for x in checks)
    write(RUN/'AUDIT.json',dict(status='PASS',training_steps=traces,models=24,expert_router_original_overlap=0,matched_DHT_decoder_initialization_and_plan=True,finite_predictions_and_gradients=True,point_free_trained_invariance_checks=len(checks),scored_prediction_files=files,max_recomputed_frame_error_difference=maxerr,GT_oracle_recomputed=True,router_output_exact_candidate=True,counts=counts,prior_completed_files_unchanged=priorfiles,evaluation_bindings=len(bindings),limitations='Structural cuboid labels and artificial visibility states only. Line-pair oracle is best discrete pair, not necessarily a bound on learned convex mixtures. Raw click GT unchanged, annotation uncertainty not newly measured.'))
    print('AUDIT29 PASS',traces,files,priorfiles,flush=True)

if __name__=='__main__':main()
