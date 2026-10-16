from context34 import *
from model33 import Correspondence,objective
from data32 import training,plan
from cli_train import state_sha
import subprocess

def worker(arm,seed,repeat):
    flags=deterministic();folder=RUN/'models'/f'{arm}_r{repeat}_seed{seed}'
    if (folder/'COMPLETE.json').exists():return
    assert not (folder/'START.json').exists(),folder
    O,S,g,v,phys,virt,sup,ids,n=training(seed);steps=plan(ids,n,seed);assert steps==read(V32.RUN/f'PLAN_seed{seed}.json')['plan'];O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()};g=g.cuda();v=v.cuda();phys=phys.cuda();sup=sup.cuda()
    model=Correspondence(seed).cuda();initial=state_sha(model.state_dict());assert initial==read(V33.RUN/'models'/f'{arm}_seed{seed}'/'START.json')['initial_sha256'];folder.mkdir(parents=True,exist_ok=True);write(folder/'START.json',dict(arm=arm,seed=seed,repeat=repeat,initial_sha256=initial,plan_sha256=digest(json.dumps(steps)),flags=flags,process_id=os.getpid(),steps=2000))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
    with (folder/'TRACE.jsonl').open('w') as f:
        for step,ix in enumerate(steps):
            ii=torch.tensor(ix,device='cuda');out=model(subset(O,ii),subset(S,ii),arm);loss,terms=objective(out,g[ii],v[ii],phys[ii],sup[ii],'visible_hough');assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=norm,**{k:float(x.detach()) for k,x in terms.items()}))+'\n')
            if (step+1)%500==0:f.flush();print('TRAIN34',arm,seed,repeat,step+1,round(time.monotonic()-tick),flush=True)
    path=folder/'final.pt';state={k:x.detach().cpu() for k,x in model.state_dict().items()};torch.save(dict(state_dict=state,arm=arm,seed=seed,repeat=repeat,steps=2000),path);write(folder/'COMPLETE.json',dict(complete=True,steps=2000,checkpoint_sha256=sha(path),state_sha256=state_sha(state),trace_sha256=sha(folder/'TRACE.jsonl'),seconds=time.monotonic()-tick))

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,repeats=REPEATS,steps=2000,actual_updates=24000,models=12,repetition='Fresh subprocess for each arm/seed/repeat, identical V33 initial state and exact V32 plans. Compare full state tensors and every trace step, then saved predictions. Repeats are not independent seeds.',determinism='CUBLAS_WORKSPACE_CONFIG=:4096:8; deterministic_algorithms strict True; cudnn.deterministic=True, benchmark=False; TF32=False; FP32. Same architecture/loss/optimizer/budget as V33.',primary='Exact two-repeat state/trace/output equality on this environment. line_only_r0 vs pooled_r0 crop20outside >=1% eachseed +paired9session95%upper<0; allseed V33 stability gates unchanged. Replicate equality is not accuracy stability.',counterfactual='V33 line_only and V34 deterministic line_only repeat0, all3seeds. Real clean/corner20/crop20/crop40 plus fresh_check crop20 and render_check crop20. Freeze points/image features; shift top attended line token for each point by +/-4 and +/-8px along its own normal, preserving token appearance, role, score/mass/spread and all other tokens. Update line offset and support center coherently. Also swap only line geometry, only line appearance, or both with a different session/group; line-off reference.',counterfactual_gates='Primary real crop20: both4/8px, eachseed median signed response ratio >.02 and positive response fraction >.60 among preselected active queries (baseline top-token attention >=.01 and valid detected point). Geometry-donor must worsen paired pixel error with9session95%lower>0 to claim geometry usefulness. These are operational probe thresholds, not physical laws. All queries and active subset reported, no GT in intervention selection.',limits='Frozen feature interventions may be off-manifold; local sensitivity is not calibrated correctness. Physical CAD-chain roles vs virtual corners remain approximate. Reused real297 DEV, not canonical140/paper/sealed final; no manual GT edits or real fitting.'))
    previous=read(V33.RUN/'PRIOR_BINDINGS.json')['files'];previous.update(read(V33.RUN/'COMPLETION.json')['files']);write(RUN/'PRIOR_BINDINGS.json',dict(files=previous));files=[KIT/'context34.py',KIT/'train34.py',V33.KIT/'model33.py',V32.KIT/'model32.py',V32.KIT/'data32.py'];write(RUN/'TRAIN_PROTOCOL.json',dict(source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},protocol_sha256=sha(RUN/'PROTOCOL.json')));(RUN/'PURPOSE.md').write_text('# V34\n\nVerify deterministic replication and actual line-geometry use without changing the learned architecture.\n');record('V34 결정적 반복12모델·24,000업데이트와 선 위치/영상 특징 분리 개입 사전등록. V33/V32 완료 결과는 읽기 전용.')

def main():
    register()
    for seed in SEEDS:
        for repeat in REPEATS:
            for arm in ARMS:subprocess.run([sys.executable,str(KIT/'train34.py'),'--arm',arm,'--seed',str(seed),'--repeat',str(repeat)],cwd=ROOT,check=True,env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    comparisons=[]
    for arm in ARMS:
        for seed in SEEDS:
            folders=[RUN/'models'/f'{arm}_r{r}_seed{seed}' for r in REPEATS];a,b=[read(p/'COMPLETE.json') for p in folders];comparisons.append(dict(arm=arm,seed=seed,state_equal=a['state_sha256']==b['state_sha256'],trace_equal=a['trace_sha256']==b['trace_sha256']))
    write(RUN/'REPLICATION.json',dict(rows=comparisons,all_equal=all(r['state_equal'] and r['trace_equal'] for r in comparisons)));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=24000,models=12));print('TRAINING34 COMPLETE',comparisons,flush=True)

def load_model(arm,seed,repeat=0):
    model=Correspondence(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_r{repeat}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);return model

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);p.add_argument('--repeat',type=int);a=p.parse_args()
    if a.arm:worker(a.arm,a.seed,a.repeat)
    else:main()
