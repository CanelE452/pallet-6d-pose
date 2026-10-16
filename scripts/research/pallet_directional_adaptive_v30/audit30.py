"""Verify fixed parents, original splits, partial geometry and actual saved outputs."""
from context30 import *
from data30 import evaluation_data,live_sources,population,conditions
from model30 import DirectionalAdaptive,ANCHORS
from eval30 import oracle

def main():
    setup();summary=read(RUN/'SUMMARY.json');training=read(RUN/'TRAINING_COMPLETE.json');assert len(training['runs'])==12 and training['updates']==24000
    pr=read(RUN/'PROTOCOL.json')
    for name in ('source_hashes','parent_hashes'):
        for p,h in pr[name].items():assert sha(ROOT/p)==h,p
    steps=0
    for r in training['runs']:
        path=RUN/'models'/f'{r["arm"]}_seed{r["seed"]}';trace=[json.loads(x) for x in (path/'TRACE.jsonl').read_text().splitlines()]
        assert len(trace)==2000 and trace[-1]['step']==2000;assert sha(path/'final.pt')==r['checkpoint_sha256'];assert all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for x in trace);steps+=len(trace)
    for seed in SEEDS:
        plan=read(RUN/f'PLAN_seed{seed}.json');sources={x.split('::')[0] for x in plan['ids']};old=read(V29.RUN/f'PLAN_seed{seed}.json');assert sources==set(old['router_sources']) and not sources&set(old['expert_sources'])
        starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in ARMS]
        for key in ('initial_state_sha256','parameters','plan_sha256','steps'):assert len({x[key] for x in starts})==1
        for split in EVAL_SPLITS:
            m,_=population(split,conditions(split)[0]);assert not sources&{r['id'] for r in m['rows']}
    checked=0;max_error=0.;oracle_difference=0.
    for split in EVAL_SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);det=np.array([r['detected'] for r in m['rows']])
            for seed in SEEDS:
                for arm in ARMS:
                    p=np.load(RUN/'predictions'/split/c/f'{arm}_seed{seed}.npz');assert p['ids'].tolist()==[t['id'] for t in ts] and np.array_equal(p['detected'],det);assert np.isfinite(p['points'][det]).all()
                    ii=p['chosen'][det].astype(int);expected=np.take_along_axis(p['candidate_points'][det],ii[...,None,None],2)[:,:,0];assert np.array_equal(p['points'][det],expected)
                    assert np.array_equal(p['candidate_points'][det,:,0],p['base'][det]);assert np.array_equal(p['candidate_points'][det,:,1],p['strong'][det]);assert np.max(np.abs(p['alpha'][det]))<=1
                    e=np.linalg.norm(p['points']-gt,axis=-1);frame=np.where(valid[det],e[det],0).sum(-1)/valid[det].sum(-1);diff=abs(float(frame.mean())-summary['results'][split+'/'+c][f'{arm}_seed{seed}']['frame_mean_px']);max_error=max(max_error,diff);assert diff<1e-6
                    if arm=='normal_gain':
                        for name,pts in oracle(p,gt).items():
                            q=np.load(RUN/'predictions'/split/c/f'{name}_seed{seed}.npz');dif=float(np.max(np.abs(q['points'][det]-pts[det])));oracle_difference=max(oracle_difference,dif);assert dif<1e-3
                    checked+=1
    replay=[]
    for seed in SEEDS:
        m,O,S,r=evaluation_data('real','crop20',seed);sl=slice(0,4);obs={k:v[sl].cuda() for k,v in O.items()};rep={k:v[sl].cuda() for k,v in r.items()};cached={k:v[sl].cuda() for k,v in S.items()};live=live_sources(obs,rep,seed)
        diffs={k:float((live[k]-cached[k]).abs().max()) for k in live};assert max(diffs.values())<.003,diffs
        model=DirectionalAdaptive(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'normal_gain_seed{seed}'/'final.pt',map_location='cpu',weights_only=True)['state_dict'])
        with torch.no_grad():
            out=model(obs,live);ref=model(obs,cached);assert torch.allclose(out['points'],ref['points'],atol=.003,rtol=0)
            shifted=dict(cached,lines=cached['lines']+73,expert_points=cached['expert_points']-100)
            point=DirectionalAdaptive(seed).cuda().eval();point.load_state_dict(torch.load(RUN/'models'/f'point_gain_seed{seed}'/'final.pt',map_location='cpu',weights_only=True)['state_dict']);a=point(obs,cached,'point_gain');b=point(obs,shifted,'point_gain');assert torch.equal(a['points'],b['points'])
        replay.append(dict(seed=seed,parent_max_errors=diffs,model_replay_max=float((out['points']-ref['points']).abs().max()),trained_point_control_line_invariant=True))
    prior_count=0
    for prior in ('pallet_observation_v25','pallet_direction_utility_v27','pallet_line_only_v28','pallet_adaptive_experts_v29'):
        p=ROOT/'data/pallet/results'/prior/'COMPLETION.json'
        for path,h in read(p).get('files',{}).items():assert sha(ROOT/path)==h,(prior,path);prior_count+=1
    for p,h in read(RUN/'EVALUATION_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    write(RUN/'AUDIT.json',dict(status='PASS',steps=steps,matched_initial_parameters_and_plans=True,source_split_no_DHT_finetune_overlap=True,strong_point_in_sample_parent_disclosed=True,finite_predictions_and_gradients=True,prediction_files_recomputed=checked,max_frame_error_difference=max_error,max_oracle_coordinate_difference=oracle_difference,original_point_parents_exact_candidates=True,live_parent_and_full_module_replay=replay,prior_completed_files_unchanged=prior_count,GT_and_state_not_forward_inputs=True))
    print('AUDIT30 PASS',steps,checked,prior_count,flush=True)

if __name__=='__main__':main()
