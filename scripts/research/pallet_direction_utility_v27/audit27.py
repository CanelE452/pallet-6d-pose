"""Replay trained outputs and metrics; verify immutable lineage and all24k updates."""
from context27 import *
from eval27 import population,conditions,SPLITS,METHODS,inputs
from model27 import DirectionUtility27
from network27 import DirectionUtilityNetwork
from train27 import point_proposal

def main():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['updates']==24000;assert read(RUN/'PRECHECK.json')['status']=='PASS';bindings={};oldcount=0
    for old in ('pallet_observation_v25','pallet_camera_trust_v26'):
        p=ROOT/'data/pallet/results'/old/'COMPLETION.json'
        for name,h in read(p)['files'].items():assert sha(ROOT/name)==h,name;oldcount+=1
    for name,h in read(RUN/'TRAIN_PROTOCOL.json')['sources'].items():assert sha(ROOT/name)==h;bindings[name]=h
    for name,h in read(RUN/'TRAIN_PROTOCOL.json')['plan_bindings'].items():assert sha(ROOT/name)==h;bindings[name]=h
    traces=[]
    for seed in SEEDS:
        plan=read(RUN/f'GROUPS_seed{seed}.json');assert plan['plan']==read(PREV25.RUN/f'GROUPS_seed{seed}.json')['plans']['paired'];initial=[]
        for arm in ARMS:
            d=RUN/'models'/f'{arm}_seed{seed}';t=[json.loads(l) for l in (d/'TRACE.jsonl').read_text().splitlines()];assert len(t)==2000 and [x['step'] for x in t]==list(range(1,2001));assert all(np.isfinite(x[k]) for x in t for k in ('loss','point','gradient_norm'));p=d/'final.pt';assert sha(p)==read(d/'COMPLETE.json')['sha256'];ck=torch.load(p,weights_only=True,map_location='cpu');assert ck['steps']==2000;initial.append(read(d/'START.json')['initial_sha256']);traces.append(dict(arm=arm,seed=seed,updates=len(t),last200_train_batch_point_px=float(np.mean([x['point'] for x in t[-200:]]))));bindings[str(p.relative_to(ROOT))]=sha(p)
        assert len(set(initial))==1
    s=read(RUN/'SUMMARY.json');maxmetric=0.;predfiles=0
    for split in SPLITS:
        for c in conditions(split):
            m,t=population(split,c);gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);det=np.array([r['detected'] for r in m['rows']]);ids=[r['id'] for r in m['rows']];diag=np.linalg.norm(np.array([r['original_hw'] for r in m['rows']]),axis=-1)
            for arm in METHODS:
                for seed in ([1] if arm=='stock' else SEEDS):
                    name='stock' if arm=='stock' else f'{arm}_seed{seed}';p=np.load(RUN/'predictions'/split/c/(name+'.npz'));assert p['ids'].tolist()==ids and np.array_equal(p['detected'],det);assert np.isfinite(p['points'][det]).all() and np.isnan(p['points'][~det]).all();e=np.sqrt(((p['points'].astype(float)-gt)**2).sum(-1));frame=np.where(valid[det],e[det],0).sum(-1)/valid[det].sum(-1);r=s['results'][split+'/'+c][name];err=abs(float(frame.mean())-r['frame_mean_px']);maxmetric=max(maxmetric,err);norm=np.ones(len(det));norm[det]=np.where(valid[det],np.minimum(e[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1);assert abs(norm.mean()-r['full_normalized_mean'])<1e-10;predfiles+=1
    assert maxmetric<1e-10
    replays=[]
    for seed in SEEDS:
        for split,c in [('real','crop20'),('camera','clean')]:
            m,O,r=inputs(split,c,seed);ii=torch.tensor([0,min(7,len(O['points'])-1)]);o={k:v[ii].cuda() for k,v in O.items()};rr={k:v[ii].cuda() for k,v in r.items()};d=point_proposal(o,rr,seed);fullids=np.flatnonzero([x['detected'] for x in m['rows']])[ii.numpy()]
            for arm in ARMS:
                net=DirectionUtilityNetwork(seed,arm)
                with torch.inference_mode():out=net(o)['points'].cpu().numpy();expected=np.load(RUN/'predictions'/split/c/f'{arm}_seed{seed}.npz')['points'][fullids];error=float(np.abs(out-expected).max());assert error<5e-4,(seed,arm,split,error)
                replays.append(dict(seed=seed,arm=arm,split=split,max_px=error));del net
    write(RUN/'AUDIT.json',dict(status='PASS',training_updates=24000,trained_models=12,identical_initial_state_per_seed=True,same_V25_paired_plans=True,prior_completion_files_unchanged=oldcount,metric_replay_max_px=maxmetric,prediction_files=predfiles,full_network_replays=replays,traces=traces,bindings=bindings,limits='Synthetic role supervision validates annotated cuboid correspondence, not actual visible mesh boundaries. Manual real GT was reused; no independent reannotation or sealed real evaluation.'));record('V27 감사 완료:24,000업데이트·동일초기값/배치·기존 완료파일 불변·전체 지표 재계산·24개 전체 네트워크 재현 통과.');print('AUDIT27',predfiles,maxmetric,max(x['max_px'] for x in replays),flush=True)

if __name__=='__main__':main()
