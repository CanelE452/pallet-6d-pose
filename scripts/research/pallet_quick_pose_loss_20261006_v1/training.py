"""Two fixed 6000-update fits; no old entrypoints or output destinations."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .common import (ROOT,DOC,OUTPUT,BANKS,COST,OLD_DOC,HARD_DOC,Data,Banks,
    read,write,sha,hash_value,dcp_env,initialize,numeric_contract,learning_rate,
    forward_bank,action_scores,code_bindings,verify_derived)
from .losses import pose_loss


def empty_stats():
    return dict(exposures=0,eligible=0,excluded=0,oracle_matches=0,NoOp=0,
        selected_cost_finite=0,selected_cost_infinite=0,selected_cost_sum_m=0.,
        oracle_gap_sum_m=0.,expected_cost_finite_rows=0,expected_cost_infinite_rows=0,
        expected_exact_cost_sum_m=0.,expected_finite_conditioned_sum_m=0.,
        expected_penalty_approximation_sum_m=0.,failed_probability_sum=0.,
        loss_weighted_sum=0.,gradient_norm_sum=0.,clipped_updates=0,updates=0,
        gradient_clip_scale_sum=0.)


@torch.no_grad()
def observe(scores,rows,cost,oracle,eligible,r_effective,scale,loss,norm):
    probability=scores.softmax(-1).double().cpu().numpy()
    chosen=scores.argmax(-1).cpu().numpy()
    c=np.array(cost[rows],copy=True)
    ok=np.isfinite(c); eligible=np.asarray(eligible,dtype=bool)
    finite_cost=np.where(ok,c,0.)
    selected=c[np.arange(len(rows)),chosen]
    minimum=np.min(c,axis=1)
    chosen_ok=np.isfinite(selected)&eligible
    failed_mass=(probability*~ok).sum(-1)
    valid_mass=(probability*ok).sum(-1)
    expected=(probability*finite_cost).sum(-1)
    exact_finite=(failed_mass==0)&eligible
    effective=np.asarray(r_effective,dtype=np.float64)
    approximate=minimum+scale*(probability*effective).sum(-1)
    result=empty_stats()
    result.update(exposures=len(rows),eligible=int(eligible.sum()),excluded=int((~eligible).sum()),
        oracle_matches=int(((chosen==oracle[rows])&eligible).sum()),NoOp=int((chosen==0).sum()),
        selected_cost_finite=int(chosen_ok.sum()),selected_cost_infinite=int((eligible&~chosen_ok).sum()),
        selected_cost_sum_m=float(selected[chosen_ok].sum()),
        oracle_gap_sum_m=float((selected-minimum)[chosen_ok].sum()),
        expected_cost_finite_rows=int(exact_finite.sum()),expected_cost_infinite_rows=int((eligible&~exact_finite).sum()),
        expected_exact_cost_sum_m=float(expected[exact_finite].sum()),
        expected_finite_conditioned_sum_m=float((expected[eligible]/valid_mass[eligible]).sum()),
        expected_penalty_approximation_sum_m=float(approximate[eligible].sum()),
        failed_probability_sum=float(failed_mass[eligible].sum()),loss_weighted_sum=float(loss)*int(eligible.sum()),
        gradient_norm_sum=float(norm),clipped_updates=int(norm>5),updates=1,
        gradient_clip_scale_sum=min(1.,5/(float(norm)+1e-6)))
    return result


def add_stats(total,values):
    for key in total:total[key]+=values[key]


def summarize(stats):
    out=dict(stats)
    mappings={'mean_actual_loss':('loss_weighted_sum','eligible'),
        'exact_oracle_index_accuracy':('oracle_matches','eligible'),
        'NoOp_fraction':('NoOp','exposures'),
        'hard_selected_finite_cached_ADDsym_mean_m':('selected_cost_sum_m','selected_cost_finite'),
        'hard_selected_finite_oracle_gap_mean_m':('oracle_gap_sum_m','selected_cost_finite'),
        'expected_exact_cached_ADDsym_mean_over_finite_rows_m':('expected_exact_cost_sum_m','expected_cost_finite_rows'),
        'expected_finite_conditioned_cached_ADDsym_mean_m':('expected_finite_conditioned_sum_m','eligible'),
        'expected_penalty_approximation_mean_m':('expected_penalty_approximation_sum_m','eligible'),
        'mean_failed_action_probability':('failed_probability_sum','eligible'),
        'mean_gradient_norm':('gradient_norm_sum','updates'),
        'clipped_update_fraction':('clipped_updates','updates'),
        'mean_gradient_clip_scale':('gradient_clip_scale_sum','updates')}
    for key,(a,b) in mappings.items():out[key]=stats[a]/stats[b] if stats[b] else None
    out['expected_exact_cached_ADDsym_infinite']=stats['expected_cost_infinite_rows']>0
    out['scope']='existing pre-update forwards; repeated ordered exposures under changing checkpoints; not final whole-TRAIN accuracy'
    return out


def atomic_checkpoint(path,checkpoint):
    pending=path.with_suffix('.pending.pt')
    torch.save(checkpoint,pending)
    pending.replace(path)


def fit(method,source_root=ROOT,bank_cache=BANKS,cost_cache=COST,output_cache=OUTPUT):
    assert method in ('SOFT6D','EXPECT6D')
    numeric_contract()
    setup=read(DOC/'LOSS_SETUP.json'); protocol=read(DOC/'PROTOCOL.json')
    assert setup['status']=='PASS'
    for key,value in (('source_root',source_root),('candidate_bank_cache',bank_cache),
                      ('pose_cost_cache',cost_cache),('output_cache',output_cache)):
        assert Path(protocol[key]).resolve()==Path(value).resolve(), 'Locked input/output path changed: '+key
    assert protocol['code_bindings']==code_bindings(), 'Locked code changed before/during run'
    receipts_path=DOC/'TRAIN_RECEIPTS.json'
    receipts=read(receipts_path) if receipts_path.exists() else {'methods':{}}
    if method in receipts['methods']:
        prior=receipts['methods'][method]
        assert prior['complete'] and sha(Path(prior['checkpoint_path']))==prior['checkpoint_sha256']
        return prior
    data=Data(source_root); banks=Banks(data,bank_cache,setup)
    config=read(OLD_DOC/'A_protocol.json')['config']
    paths=setup['paths']
    target_path=paths['soft_targets'] if method=='SOFT6D' else paths['r_effective']
    targets=np.load(target_path,mmap_mode='r')
    effective=np.load(paths['r_effective'],mmap_mode='r')
    eligible=np.load(paths['eligible'],mmap_mode='r')
    cost=np.load(Path(cost_cache)/'cost_ADDsym_m.npy',mmap_mode='r')
    oracle=np.load(Path(cost_cache)/'oracle_index.npy',mmap_mode='r')
    scale=float(setup['scale_m'])
    order_path=Path(source_root)/'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed1.npy'
    order=np.load(order_path,mmap_mode='r'); old=read(OLD_DOC/'A_fits/GEO_seed1.json')
    assert order.shape==(6000,16) and np.isin(order,data.train_rows).all()
    assert sha(order_path)==old['order_sha256']
    # Reset both RNGs after all toy/setup work, immediately before constructing
    # the original scorer. No old checkpoint is loaded here.
    head=initialize(1,config,'cuda').train()
    initial_sha=dcp_env.state_sha(head.state_dict())
    assert initial_sha==old['first_step']['initial_state_sha256']
    assert sum(p.numel() for p in head.parameters())==old['params']==20259
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999))
    dest=Path(output_cache)/'fits'/f'{method}_seed1'; dest.mkdir(parents=True,exist_ok=True)
    checkpoint_path=dest/'last.pt'; state_path=dest/'ATTEMPT_STATE.json'
    binding=hash_value(dict(method=method,protocol=sha(DOC/'PROTOCOL.json'),loss_setup=sha(DOC/'LOSS_SETUP.json'),
        order=old['order_sha256'],initial_state=initial_sha))
    start=0; prior_seconds=0.; totals=empty_stats(); window=empty_stats(); history=[]; first=None
    if checkpoint_path.exists():
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        state=read(state_path)
        assert ck['binding']==binding and state['status']=='CHECKPOINTED' and state['completed_updates']==ck['step'], \
            'RUN_FAILED: interrupted uncheckpointed interval cannot be silently replayed'
        head.load_state_dict(ck['model_state_dict']); optimizer.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start,prior_seconds,totals,history,first=ck['step'],ck['seconds'],ck['totals'],ck['history'],ck['first_step']
    elif state_path.exists():
        raise RuntimeError('RUN_FAILED: prior attempted training has no checkpoint; no silent restart')
    began=time.monotonic(); final_window=None
    write(state_path,dict(status='RUNNING',binding=binding,completed_updates=start,interval_checkpoint_every=500))
    for step in range(start+1,6001):
        rows=order[step-1]
        batch=data.batch(rows,supervision=False)
        assert not any(k.startswith('gt') or k in ('permutations','group_valid','truth') for k in batch)
        candidates,valid=banks.tensor_batch(rows,batch)
        assert candidates.shape==(16,201,9,2) and valid.all()
        optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(step)
        for group in optimizer.param_groups:group['lr']=lr
        output=forward_bank(head,batch,candidates,valid)
        scores=action_scores(output)
        target=torch.from_numpy(np.array(targets[rows],copy=True)).cuda()
        e=torch.from_numpy(np.array(eligible[rows],copy=True)).cuda()
        value,audit=pose_loss(scores,target,e,method)
        assert torch.isfinite(value), 'Non-finite actual formal loss'
        value.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
        if step==1:
            from scripts.research.pallet_pose_target_6d_20261006_v1.training import inference_support_without_features,teacher_2d
            support,_,_=inference_support_without_features(head,batch,candidates,valid)
            assert torch.equal(support,output['point_support'])
            supervision={k:torch.from_numpy(np.array(data.arrays[k][rows],copy=True)).cuda() for k in ('gt_points','gt_valid')}
            supervision.update({k:torch.from_numpy(data.side[k][rows].copy()).cuda() for k in ('permutations','group_valid')})
            teacher,counts,_=teacher_2d(output,supervision)
            assert np.array_equal(teacher.cpu().numpy(),np.load(Path(cost_cache)/'teacher_2d_index.npy',mmap_mode='r')[rows])
            assert np.array_equal(counts.cpu().numpy(),np.load(Path(cost_cache)/'teacher_2d_count.npy',mmap_mode='r')[rows])
            gradients={n:float(p.grad.norm()) if p.grad is not None else None for n,p in head.named_parameters()}
            assert gradients['adapt3.0.weight']>0 and gradients['adapt4.0.weight']>0
            assert dcp_env.state_sha(head.state_dict())==initial_sha
            first=dict(source_rows=rows.tolist(),native_indices=list(range(201)),
                coordinate_NoOp_center_and_action_valid=True,source_row_mapping=True,GT_separated_from_forward=True,
                original_teacher_and_support_parity=True,initial_state_sha256=initial_sha,
                finite_loss=float(value.detach()),finite_gradient_norm=float(norm),gradients=gradients,
                eligible=audit['eligible_frames'],excluded=audit['excluded_frames'])
            write(dest/'FIRST_STEP.json',first)
        obs=observe(scores,rows,cost,oracle,eligible[rows],effective[rows],scale,value.detach(),norm)
        assert obs['eligible']==audit['eligible_frames']
        add_stats(totals,obs); add_stats(window,obs)
        optimizer.step()
        elapsed=prior_seconds+time.monotonic()-began
        if step==1 or step%100==0:
            record=dict(step=step,actual_loss=float(value.detach()),lr=float(lr),gradient_norm=float(norm),
                seconds=elapsed,cumulative=summarize(totals),window=summarize(window))
            history.append(record)
            write(DOC/f'TRAIN_PROGRESS_{method}.json',dict(method=method,**record))
            write(state_path,dict(status='RUNNING',binding=binding,completed_updates=step,last_durable_checkpoint=(step//500)*500))
            print('QUICK_FIT',method,step,6000,round(record['actual_loss'],6),round(elapsed,1),flush=True)
        if step%100==0:
            final_window=dict(first_step=step-99,last_step=step,**summarize(window))
            window=empty_stats()
        if step%500==0:
            atomic_checkpoint(checkpoint_path,dict(step=step,complete=step==6000,method=method,seed=1,binding=binding,
                config=config,model_state_dict=head.state_dict(),optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seconds=elapsed,
                totals=totals,history=history,first_step=first,order_sha256=old['order_sha256']))
            write(state_path,dict(status='CHECKPOINTED',binding=binding,completed_updates=step,completed_exposures=step*16))
    assert totals['exposures']==96000 and totals['updates']==6000
    result=dict(method=method,seed=1,status='DONE',complete=True,updates=6000,exposures=96000,
        eligible_exposures=totals['eligible'],excluded_exposures=totals['excluded'],config=config,params=20259,
        checkpoint_path=str(checkpoint_path),checkpoint_sha256=sha(checkpoint_path),checkpoint_bytes=checkpoint_path.stat().st_size,
        initial_state_sha256=initial_sha,order_sha256=old['order_sha256'],order_path=str(order_path),binding=binding,
        code_bindings=protocol['code_bindings'],seconds=prior_seconds+time.monotonic()-began,
        first_step=first,training_stats=summarize(totals),last100_updates=final_window,history=history,
        new_backbone_forwards=0,new_final_F_calls=0,training_refiner_batch_forwards=6000,
        training_refiner_example_forwards=96000,standalone_toy_optimizer_updates=0,
        checkpoint_every=500,final_checkpoint_only=True,numeric=protocol['numeric'],
        initialization_source='original seed1 random initialization; no warmstart',
        expected_failure_cost='max valid scaled regret + 1: finite design approximation; original cache remains +inf')
    receipts['methods'][method]=result
    write(receipts_path,receipts)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('SOFT6D','EXPECT6D'),required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--bank-cache',type=Path,default=BANKS)
    parser.add_argument('--cost-cache',type=Path,default=COST)
    parser.add_argument('--output-cache',type=Path,default=OUTPUT)
    args=parser.parse_args()
    result=fit(args.method,args.source_root,args.bank_cache,args.cost_cache,args.output_cache)
    print(result['method'],result['status'],result['updates'],flush=True)


if __name__=='__main__':main()
