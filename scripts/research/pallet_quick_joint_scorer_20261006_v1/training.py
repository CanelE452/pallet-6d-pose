"""Two matched fresh-init fits; additional diagnostics reuse each training forward."""
import time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .model import initialize
from scripts.research.pallet_quick_pose_loss_20261006_v1.training import (
    empty_stats,observe,add_stats,summarize,atomic_checkpoint)


def stats_empty():
    result=empty_stats()
    result.update(target_entropy_sum=0.,prediction_entropy_sum=0.,NoOp_probability_sum=0.)
    return result


def stats_summary(stats):
    # Keep original cost/gradient diagnostics and add the fixed-soft-target fit
    # quantities. They are repeated pre-update exposures, not unique frames.
    base={k:stats[k] for k in empty_stats()};result=summarize(base)
    for name,key in [('target_entropy','target_entropy_sum'),('prediction_entropy','prediction_entropy_sum'),('mean_NoOp_probability','NoOp_probability_sum')]:
        result[name]=stats[key]/stats['eligible'] if stats['eligible'] else None
    result['CE_minus_target_entropy']=result['mean_actual_loss']-result['target_entropy'] if stats['eligible'] else None
    return result


def fit(method):
    assert method in ('LOCAL_CAP','JOINT8')
    protocol=read(DOC/'PROTOCOL.json');receipts=fit_receipts()
    assert receipts['inputs']['status']==receipts['tests']['status']=='PASS'
    prior=receipts['methods'].get(method)
    if prior:
        assert prior['complete'] and sha(Path(prior['checkpoint_path']))==prior['checkpoint_sha256']
        return prior
    assert protocol['training_code_bindings']==code_bindings(), 'Sealed training code changed'
    setup=load_setup();verify_derived(setup);numeric_contract()
    data=Data(ROOT);banks=Banks(data,BANKS)
    targets=np.load(setup['paths']['soft_targets'],mmap_mode='r')
    eligible=np.load(setup['paths']['eligible'],mmap_mode='r')
    effective=np.load(setup['paths']['r_effective'],mmap_mode='r')
    cost=np.load(COST/'cost_ADDsym_m.npy',mmap_mode='r');oracle=np.load(COST/'oracle_index.npy',mmap_mode='r')
    config=protocol['config'];old=read(OLD_DOC/'A_fits/GEO_seed1.json')
    order_path=ROOT/'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed1.npy'
    order=np.load(order_path,mmap_mode='r')
    assert sha(order_path)==old['order_sha256'] and order.shape==(6000,16)
    head=initialize(method,config,'cuda').train()
    initial=dcp_env.state_sha(head.inherited_state_dict());common_initial=head.common_state_sha();phi_sha=dcp_env.state_sha(head.phi.state_dict())
    assert initial==old['first_step']['initial_state_sha256']
    assert sum(p.numel() for p in head.parameters())==26169
    assert sum(p.numel() for p in head.residual_head.parameters())==4680
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999))
    torch.cuda.reset_peak_memory_stats()
    # Extra initialization seeds must not change the original training RNG.
    torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    dest=OUTPUT/'fits'/f'{method}_seed1';dest.mkdir(parents=True,exist_ok=True)
    state_path=dest/'ATTEMPT_STATE.json';checkpoint_path=dest/'last.pt'
    assert not state_path.exists() and not checkpoint_path.exists(), 'Incomplete attempt cannot be silently restarted'
    binding=hash_value(dict(method=method,protocol=sha(DOC/'PROTOCOL.json'),code=code_bindings(),
        initial=initial,phi=phi_sha,order=old['order_sha256'],targets=sha(LOSS_SETUP)))
    totals=stats_empty();window=stats_empty();history=[];first=None;gradient_observations=[];front_gradient_seen=False
    started=time.monotonic();participating=set()
    for step in range(1,6001):
        # A RUNNING marker at the beginning of each 500-step interval closes
        # the stale-checkpoint resume window without per-update filesystem I/O.
        if (step-1)%500==0:
            write(state_path,dict(status='RUNNING',binding=binding,interval_first_step=step,last_checkpoint=step-1))
        rows=order[step-1];batch=data.batch(rows,supervision=False)
        assert set(batch)=={'p3','p4','points','boxes','point_valid','input_shape','context'}
        q,valid=banks.tensor_batch(rows,batch)
        optimizer.zero_grad(set_to_none=True);lr=learning_rate(step)
        for group in optimizer.param_groups:group['lr']=lr
        output=forward_bank(head,batch,q,valid);scores=action_scores(output)
        target=torch.from_numpy(np.array(targets[rows],copy=True)).cuda()
        e=torch.from_numpy(np.array(eligible[rows],copy=True)).cuda()
        loss,audit=pose_loss(scores,target,e,'SOFT6D');assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
        if step<=10:
            participating.update(name for name,p in head.named_parameters() if p.grad is not None)
            gradients={name:float(p.grad.norm()) if p.grad is not None else None for name,p in head.named_parameters()
                if name.startswith(('phi.','residual_head.'))}
            assert all(v is None or np.isfinite(v) for v in gradients.values())
            gradient_observations.append(dict(step=step,gradients=gradients))
            if step==1:
                assert gradients['residual_head.2.weight']>0
                assert gradients['phi.0.weight']==gradients['residual_head.0.weight']==0
                assert dcp_env.state_sha(head.inherited_state_dict())==initial and head.common_state_sha()==common_initial
                assert torch.equal(output['residual_scores'],torch.zeros_like(output['residual_scores']))
                first=dict(source_rows=rows.tolist(),initial_state_sha256=initial,phi_initial_state_sha256=phi_sha,
                    original_native_indices=list(range(201)),zero_residual=True,GT_inference_access=False,
                    gradients=gradients,finite_loss=float(loss.detach()),gradient_norm=float(norm),params=26169)
            elif gradients['phi.0.weight']>0 and gradients['residual_head.0.weight']>0:
                front_gradient_seen=True
        probabilities=scores.detach().softmax(-1)
        entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1)
        target_entropy=-(target*target.clamp_min(1e-30).log()).sum(-1)
        obs=observe(scores,rows,cost,oracle,eligible[rows],effective[rows],setup['scale_m'],loss.detach(),norm)
        obs.update(target_entropy_sum=float(target_entropy[e].sum()),prediction_entropy_sum=float(entropy[e].sum()),
            NoOp_probability_sum=float(probabilities[e,0].sum()))
        add_stats(totals,obs);add_stats(window,obs)
        optimizer.step();elapsed=time.monotonic()-started
        if step==1 or step%100==0:
            record=dict(step=step,loss=float(loss.detach()),lr=float(lr),gradient_norm=float(norm),seconds=elapsed,
                cumulative=stats_summary(totals),window=stats_summary(window))
            history.append(record)
            write(DOC/f'progress_{method}.json',record)
            print('JOINT_FIT',method,step,6000,round(record['loss'],6),round(record['window']['CE_minus_target_entropy'],6),
                round(record['window']['NoOp_fraction'],4),round(elapsed,1),flush=True)
            if step==100:
                print('JOINT_SPEED',method,'remaining_loop_seconds',round(elapsed/100*5900,1),flush=True)
        if step%100==0:
            last_window=dict(first_step=step-99,last_step=step,**stats_summary(window));window=stats_empty()
        if step%500==0:
            atomic_checkpoint(checkpoint_path,dict(step=step,complete=step==6000,method=method,seed=1,config=config,
                binding=binding,model_state_dict=head.state_dict(),optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seconds=elapsed,
                totals=totals,history=history,gradient_observations=gradient_observations))
            write(state_path,dict(status='CHECKPOINTED',binding=binding,completed_updates=step,completed_exposures=step*16))
    assert front_gradient_seen and totals['updates']==6000 and totals['exposures']==96000
    result=dict(method=method,status='DONE',complete=True,seed=1,updates=6000,exposures=96000,
        eligible_exposures=totals['eligible'],excluded_exposures=totals['excluded'],params=26169,head_params=4680,phi_params=1230,
        config=config,initial_state_sha256=initial,common_initial_state_sha256=common_initial,phi_initial_state_sha256=phi_sha,initialization=head.initialization_metadata,
        order_sha256=old['order_sha256'],checkpoint_path=str(checkpoint_path),checkpoint_sha256=sha(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,binding=binding,code_bindings=code_bindings(),
        first_step=first,gradient_observations=gradient_observations,front_new_layer_gradients_seen_after_first_step=front_gradient_seen,
        training_stats=stats_summary(totals),last100_updates=last_window,history=history,seconds=time.monotonic()-started,
        new_backbone_forwards=0,new_final_F_calls=0,training_refiner_batches=6000,training_refiner_examples=96000,
        standalone_optimizer_updates=0,final_checkpoint_only=True,checkpoint_every=500,
        actual_training_peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
        parameters_connected_to_loss_first10=sum(p.numel() for n,p in head.named_parameters() if n in participating),
        inherited_unused_parameters_first10={n:p.numel() for n,p in head.named_parameters() if n not in participating and not n.startswith(('phi.','residual_head.'))},
        new_head_activation_shape='B,M,8,260' if method=='LOCAL_CAP' else 'B,M,36',
        parameter_equality_is_not_compute_or_memory_equality=True)
    record_method(method,result)
    return result
