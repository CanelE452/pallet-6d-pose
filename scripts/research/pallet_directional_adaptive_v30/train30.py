"""Fixed-budget matched candidate-gain and signed-amount learning."""
from context30 import *
from data30 import training_data,plan
from model30 import DirectionalAdaptive,objective,paths
from cli_train import state_sha

def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/'PURPOSE.md').write_text('V30: 상태 고정 배정 없이 강한 점 부모를 보존하며, 점별 선 방향의 보정량·이득을 학습하고 검증한다.\n')
    protocol=dict(arms=ARMS,seeds=SEEDS,steps=STEPS,batch=BATCH,updates=24000,
        objective='Per point: signed optimal amount clipped[-1,1] along each candidate; predicted current-amount error gain vs V12 baseline; family-balanced normalized amount L1 +10*Huber(gain,beta.02)+.05*CE(gain/.02,best current candidate). GT only synthetic losses.',
        architecture='44 paths: keep V12, keep strong V25 point; full/x/y partial strong-point displacement from either anchor; normal-only displacements from either anchor for2DHT sources x3incident roles x3modes. Final hard max predicted gain; V12 keep gain fixed0. No state class, condition name or visibility GT in input/loss.',
        controls='point_gain masks all line paths and inputs. whole_line replaces line normals with full displacement toward V29 decoded corner, 4valid line paths. normal_no_evidence removes along-line token and coverage; normal_gain reads observed strip features and derivatives. Shared initialization, parameter count and sample plan; candidate count differs by intentionally masked paths.',
        geometry='Normal directions bounded15% boxdiag, signed amount[-1,1] may move toward or away from line. Normal branch preserves chosen anchor tangent exactly. It estimates usefulness of direction, not guaranteed geometric snapping or physical line correctness.',
        parents='Freeze V12 baseline and V25 vector_local_edge strong point parent, V29 guided+independent DHT and decoders. Strong point exact output retained as selectable path; final learned choice does not guarantee preserving every prior gain.',
        sources_split='Only153 V29 router originals from V25 synthetic dataset, all their stress variants. No overlap with V29 DHT fine-tuning originals. V25 strong point parent did train on these originals, so its residuals remain in-sample; do not claim all-parent out-of-fold training.',
        optimizer='AdamW3e-4 decay1e-4 clip5 FP32 noTF32, fixed final2000 only; no result-based tuning.',
        evaluation='Explicit continuation of real297/9session reusedDEV, prospective128 and synth_val505, all existing stress conditions. Not canonical140/paper/new real holdout. Original clicks unchanged; natural visibility and physical boundary/endpoint GT not newly provided.',
        gates='Each seed: clean/crop10/20/30 within1% V12; corner20/crop40 retain90% V18 gain. Primary normal_gain vs point_gain crop20 outside >=1% each seed and paired session95% upper<0, plus all stability gates. Whole-line and evidence ablations descriptive with paired CIs.',
        interventions='Point-control line swap invariance; normal tangent preservation and sign/subcell checks; trained normal_gain evidence off/donor, line path off; full live parent replay against V29/V27 cached source outputs; GT oracle candidate and amount diagnostics separated from actual.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'context30.py',KIT/'model30.py',KIT/'data30.py',KIT/'train30.py')},parent_hashes={str(p.relative_to(ROOT)):sha(p) for p in list((PREV25.RUN/'models').glob('vector_local_edge_seed*/final.pt'))+list((V29.RUN/'models').glob('*dht_seed*/final.pt'))+list((V29.RUN/'models').glob('*decoder_seed*/final.pt'))})
    if (RUN/'PROTOCOL.json').exists():assert read(RUN/'PROTOCOL.json')==json.loads(json.dumps(protocol))
    else:write(RUN/'PROTOCOL.json',protocol)
    torch.set_num_threads(2);B=2
    O=dict(features=torch.randn(B,192,24,24),content=torch.ones(B,1,24,24,dtype=torch.bool),edge=torch.randn(B,3,96,96),box=torch.tensor([[0.,0.,100.,100.]]).repeat(B,1),image_hw=torch.tensor([[100.,100.]]).repeat(B,1))
    base=torch.tensor([12.123,27.234])[None,None].repeat(B,8,1);ll=torch.tensor([1.,0.,-14.567])[None,None,None,None].repeat(B,2,12,3,1);mass=torch.ones(B,2,12,3)/3
    S=dict(base=base,strong=base+torch.tensor([5.,-3.]),lines=ll,mass=mass,expert_points=torch.stack((base+6,base-4),2));gt=base+torch.tensor([1.,4.]);valid=torch.ones(B,8,dtype=torch.bool);counts={}
    for arm in ARMS:
        model=DirectionalAdaptive(1);out=model(O,S,arm);loss,terms=objective(out,gt,valid);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        counts[arm]=sum(p.numel() for p in model.parameters())
        if arm=='point_gain':
            changed=dict(S,lines=S['lines']+100,mass=S['mass']*100,expert_points=S['expert_points']+1000);z=model(O,changed,arm);assert torch.equal(out['points'],z['points']) and torch.equal(out['gain'][:,:,:8],z['gain'][:,:,:8])
        if arm=='normal_gain':
            delta=out['candidate_points'][:,:,8:]-out['anchors'][:,:,8:];assert delta[...,1].abs().max()<1e-6
            assert torch.allclose(out['directions'][:,:,8:26,0],torch.full((B,8,18),14.567-12.123),atol=2e-6)
        assert torch.equal(out['candidate_points'][:,:,0],base) and torch.equal(out['candidate_points'][:,:,1],S['strong'])
    assert len(set(counts.values()))==1
    old=read(V29.RUN/'PLAN_seed1.json');assert not set(old['router_sources'])&set(old['expert_sources'])
    write(RUN/'PRECHECK.json',dict(status='PASS',parameters=counts,normal_tangent_preserved=True,subcell_line_sign_checked=True,point_control_line_swap_invariant=True,parents_exact_keep_candidates=True,all_arm_finite_gradients=True,DHT_finetune_original_overlap=0,strong_point_parent_training_overlap_disclosed=True))
    record('V30 학습 전 등록: 상태 분류 없이 점별 방향/이득 학습, 기존 강한 점 부모 고정, 동일4대조×3시드×2000. V29 DHT 미세학습 원본과 분리; V25 부모의 in-sample 잔차 한계 명시.')
    print('PRECHECK30 PASS',counts,flush=True)

def train():
    setup();assert read(RUN/'PRECHECK.json')['status']=='PASS';pr=read(RUN/'PROTOCOL.json')
    for key in ('source_hashes','parent_hashes'):
        for p,h in pr[key].items():assert sha(ROOT/p)==h,p
    for seed in SEEDS:
        O,S,gt,valid,ids,kind=training_data(seed);steps=plan(ids,seed);write(RUN/f'PLAN_seed{seed}.json',dict(ids=ids,plan=steps,originals=len({i.split('::')[0] for i in ids}),sha256=digest(json.dumps(steps)),valid_points=int(valid.sum())))
        O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()};gt=gt.cuda();valid=valid.cuda()
        for arm in ARMS:
            folder=RUN/'models'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'COMPLETE.json').exists():continue
            assert not (folder/'START.json').exists(),f'incomplete cell requires investigation: {folder}'
            model=DirectionalAdaptive(seed).cuda();write(folder/'START.json',dict(initial_state_sha256=state_sha(model.state_dict()),steps=STEPS,plan_sha256=digest(json.dumps(steps)),parameters=sum(p.numel() for p in model.parameters())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(steps):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(S,ii),arm);loss,terms=objective(out,gt[ii],valid[ii]);assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=gn,**{k:float(v.detach()) for k,v in terms.items()}))+'\n')
                    if (step+1)%400==0:f.flush();print('TRAIN30',seed,arm,step+1,round(float(terms['point_error'].detach()),3),round(time.monotonic()-tick),flush=True)
            path=folder/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,steps=STEPS),path);write(folder/'COMPLETE.json',dict(complete=True,seed=seed,arm=arm,steps=STEPS,checkpoint_sha256=sha(path),seconds=time.monotonic()-tick));del model,opt
        del O,S,gt,valid
    runs=[read(p) for p in sorted((RUN/'models').glob('*/COMPLETE.json'))];assert len(runs)==12
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,runs=runs,updates=sum(r['steps'] for r in runs)));record('V30 실제12모델·24,000업데이트 완료. 최종 체크포인트를 고정해 GT없는 추론과 전체 조건 비교로 진행.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('prepare','train'),required=True);a=p.parse_args();prepare() if a.phase=='prepare' else train()
