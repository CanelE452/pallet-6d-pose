"""Register before learning and check geometry/gradient/data contracts."""
from context27 import *
from model27 import DirectionUtility27,candidates,objective,role_targets

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    (RUN/'PURPOSE.md').write_text('V27: 사용자 요청에 따른 점·선 방향별 효용 모듈의 합성 학습·3시드 안정성 검증. 운영 모델 변경 없음.\n')
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=2000,optimizer='AdamW lr3e-4 decay1e-4 clip5 FP32',same_plan='V25 paired GROUPS per seed, all original observed groups retained',new_training=True,parent='Frozen YOLO/DHT/V12 + V25 vector_local_edge; no backbone update',candidate_design='10 candidates: keep,point vector,x/y; line arms add3normal and3nonparallel intersections; point control equal10directions.15%boxdiag radius. Cross determinant>=.1.',objective='All arms point distance; utility arms add10*Huber(beta.01, normalized signed gain)+.25*movement-weighted amount L1; role arm+.25*incident cuboid compatibility BCE. GT only from synthetic training.',roles='Incident cuboid endpoint RMS softtarget exp(-rms/max(.02diag,1)). This is geometric correspondence, not pixel-visible physical edge/endpoint supervision.',selection='Fixed final2000 only; no real GT learning/checkpoint/threshold tuning; no post-result sweep.',stability=dict(each_seed=True,clean_frame_ratio_max=1.01,mild_outside_ratio_max=1.01,mild_conditions=['crop10','crop20','crop30'],hard_gain_retention_min=.9,hard_conditions=['corner20','crop40']),line_benefit=dict(primary='real/crop20 outside vs point_utility',each_seed_gain_min_percent=1,session_bootstrap_ci95_upper_below=0,requires_stability=True),scope='Explicit continuation of V25/V26 real297 reusedDEV, not canonical140/paper track, no new sealed real holdout. Prior parents trained on same synthetic originals as this module: in-sample residual training; separate-original synthetic evaluations disclosed.',sources={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'model27.py',KIT/'context27.py',NOTE)},parents={str(p.relative_to(ROOT)):sha(p) for p in (PREV25.RUN/'models').glob('vector_local_edge_seed*/final.pt')}))
    record('V27 학습 전 등록:4대조×3시드×2000, 방향 후보·효용/이동량/역할 감독, 동일 V25 paired 배치. 전체 안정성+crop20 선 추가1%/세션CI 기준 고정. 새 실사 봉인평가 아님.')

def checks():
    torch.set_num_threads(2);torch.manual_seed(1);B=2
    O=dict(features=torch.randn(B,192,24,24),content=torch.ones(B,1,24,24),edge=torch.randn(B,3,96,96),box=torch.tensor([[0.,0.,100.,100.]]).repeat(B,1),image_hw=torch.tensor([[100.,100.]]).repeat(B,1))
    base=torch.tensor([12.,2.])[None,None].repeat(B,8,1);ll=torch.tensor([[0.,1.,0.],[1.,0.,0.],[0.,1.,-5.]])[None].repeat(B,1,1);weights=torch.tensor([.4,.3,.2,.1])[None,None].repeat(B,8,1);r=dict(base=base,context=torch.randn(B,8,64),lines=ll,line_weights=weights);d=torch.ones_like(base)
    c=candidates(O,r,d);assert torch.allclose(c['delta'][:,:,4],torch.tensor([0.,-2.]).expand(B,8,2));assert torch.allclose(c['delta'][:,:,7],-base);assert not c['mask'][:,:,8].any();assert torch.isfinite(c['delta']).all()
    gt=base+torch.randn_like(base);valid=torch.ones(B,8,dtype=torch.bool);counts={}
    for arm in ARMS:
        net=DirectionUtility27(1);out=net(O,r,d,arm);loss,terms=objective(out,gt,valid,arm);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None);assert sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)>0;counts[arm]=sum(p.numel() for p in net.parameters())
        if arm=='point_utility':
            r2=dict(r,lines=r['lines']+137,line_weights=r['line_weights'].flip(-1));assert torch.equal(out['points'],net(O,r2,d,arm)['points'])
    assert len(set(counts.values()))==1
    # Manual endpoint roles: x/y axes through GT corner should receive unit compatibility.
    corner=base.new_tensor([[0,0],[10,0],[10,10],[0,10],[0,-10],[10,-10],[10,-20],[0,-20]])[None].repeat(B,1,1);t,support=role_targets(c['lines'],corner,valid,c['diag']);assert t[0,0,0]==1 and t[0,0,1]==1
    from eval25 import population,conditions
    trainids=read(PREV25.RUN/'GROUPS_seed1.json')['ids'];trainset={x.split('::')[0] for x in trainids};splitchecks={}
    for split in ('prospective','paired_check','synth_val','real'):
        m,_=population(split,conditions(split)[0]);evalset={r['id'] for r in m['rows']};overlap=trainset&evalset;assert not overlap,(split,list(overlap)[:3]);splitchecks[split]=dict(originals=len(evalset),overlap=0)
    write(RUN/'PRECHECK.json',dict(status='PASS',parameters=counts,normal_sign_and_pair_intersection=True,parallel_pair_mask=True,finite_gradient_all_arms=True,point_control_line_swap_invariant=True,geometric_role_targets_checked=True,train_eval_original_id_disjoint=splitchecks,limits='ID disjoint does not establish independent new real sessions; reused DEV and prior in-sample parent residuals remain.'))
    print('PRECHECK27',counts,flush=True)

if __name__=='__main__':register();checks()
