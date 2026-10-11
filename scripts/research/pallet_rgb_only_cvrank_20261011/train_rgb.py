"""Exactly two last-step RGB_ONLY fits, reusing the prior M3 TRAIN contract."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import rgb_model as R

spec=importlib.util.spec_from_file_location('rgb_only_prior_train',R.PRIOR_NAMESPACE/'train.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
M=R.prior_model
COORD_WEIGHT=0.15214983019356332
STEPS=3000


def read(path):return json.loads(Path(path).read_text())
sha=T.sha
state_sha=T.state_sha
write=T.write
configure=T.configure
lr_at=T.lr_at


def prior_doc(args):
    supplied=getattr(args,'prior_public_dir',None)
    return Path(supplied) if supplied else R.ROOT/'_docs/experiments/pallet_nonsevere_precision_20261011'


def setup(args,seed,device=None):
    device=device or getattr(args,'device','cuda')
    env,refiner,data,template,order,path,targets,decode,loss=T.setup(args,seed,device=device)
    old=read(prior_doc(args)/f'TRAIN_RECEIPT_seed{seed}.json')
    initial=state_sha(template.state_dict())
    assert initial==old['shared_initial_state_sha256'] and sha(path)==old['original_checkpoint_sha256']
    cpu_rng=torch.random.get_rng_state().clone()
    model=R.RGBOnlyEvidence.from_native(template)
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and state_sha(model.state_dict())==initial
    assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()==old['order_sha256']
    assert read(prior_doc(args)/'AUX_WEIGHT_LOCK.json')['weights']['COORD']==COORD_WEIGHT
    assert model.parameter_counts()['serialized_parameters']==24387
    model.initialization_evidence=dict(shared_initial_state_sha256=initial,
        prior_M3_initial_state_sha256=old['shared_initial_state_sha256'],common_parameter_values_exact=True,
        RNG_after_prior_setup_sha256=hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
        module_transfer_additional_RNG_consumption=0,initial_checkpoint_sha256=sha(path),
        prior_TRAIN_receipt_sha256=sha(prior_doc(args)/f'TRAIN_RECEIPT_seed{seed}.json'),
        order_sha256=old['order_sha256'],warm_start='same-seed OLD_P last6000; RGB/residual from exact prior train.setup creation order')
    return env,refiner,data,model,order,path,targets,decode,loss


def batch(data,rows,device='cuda',supervision=True):
    # Neither features nor dimension context is read by the fit forward loader.
    keys=('points','boxes','point_valid','input_shape')
    if supervision:keys+=('gt_points','gt_valid')
    result={k:torch.from_numpy(np.array(data.arrays[k][rows],copy=True)).to(device) for k in keys}
    if supervision:
        for key in ('permutations','group_valid'):
            result[key]=torch.from_numpy(np.array(data.side[key][rows],copy=True)).to(device)
    return result


feature_free_batch=batch


def native_tiles(data,rows,batch_input,displacements,device='cuda'):
    # Exact selection.evaluation_tiles allocation fallback, with explicit device.
    boxes=batch_input['boxes']
    valid=torch.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
    tile_batch=dict(batch_input)
    tile_batch['boxes']=torch.where(valid[:,None],boxes,boxes.new_tensor([0,0,1,1]))
    return M.native_tiles(data,rows,tile_batch,displacements,device=device)


def components(output,batch_input,data,rows,refiner,targets,decode,loss):
    gt,valid,branch,_=refiner.local_phase(output['points_raw'],output['point_valid'],batch_input['gt_points'],batch_input['gt_valid'],batch_input['permutations'],batch_input['group_valid'],output['box_diagonal'])
    gain,hw=T.batch_metadata(data,rows,device=output['logits'].device)
    eligible,_=M.aux_mask(output,gt,valid,gain,hw,targets)
    ce=loss(output,gt,valid)
    coord,error=M.coordinate_loss(output,gt,eligible,gain,decode)
    return ce,coord,dict(eligible=int(eligible.sum()),valid_corners=int(valid[:,:8].sum()),
                         eligible_frames=int((eligible.sum(-1)>0).sum()),geo_frames=0), (gt,eligible,gain)


def frozen_state(model):
    return {key:value for key,value in model.state_dict().items()
            if any(key.startswith('base.'+name+'.') for name in R.UNUSED_MODULES)}


def gradient_norm(parameters):
    gradients=[p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(gradients).sum())) if gradients else 0.


def feature_probe(args,seed,phase,data,model,order,targets,decode,loss,refiner):
    """Actual entrance/call captures on fixed TRAIN inputs, private coordinates."""
    start=time.monotonic();private=Path(args.private_dir);rows=np.asarray(order[0]);other_rows=np.asarray(order[1])
    filename=f'RGB_ONLY_FEATURE_PROBE_{phase}_seed{seed}.npz';capture_path=private/filename
    assert not capture_path.exists()
    device=args.device;original=data.batch(rows,'N3_DIM_SYM',device=device)
    other=data.batch(other_rows,'N3_DIM_SYM',device=device)
    tiles=native_tiles(data,rows,original,model.base.displacements,device=device)
    variants={'original':original,'zero':{**original,'p3':torch.zeros_like(original['p3']),'p4':torch.zeros_like(original['p4'])},
              'other':{**original,'p3':other['p3'],'p4':other['p4']},
              'missing':{k:v for k,v in original.items() if k not in ('p3','p4')},'changed_rgb':original}
    state_before=state_sha(model.state_dict());mode=model.training;model.eval()
    captures={'rows':rows.astype('<i8'),'other_feature_rows':other_rows.astype('<i8')}
    outputs={};records=[];feature_calls={name:0 for name in R.UNUSED_MODULES};handles=[];actual_inputs={}
    current=['original']
    for name in R.UNUSED_MODULES:
        def called(module,inputs,output,_name=name):feature_calls[_name]+=1
        handles.append(getattr(model.base,name).register_forward_hook(called))
    for name,layer in [('regular',model.base.scorer[0]),('null',model.base.null_scorer[0]),('residual',model.residual[0])]:
        def capture(module,inputs,_name=name):
            value=inputs[0].detach();assert torch.count_nonzero(value[...,:48])==0
            actual_inputs[_name]=value.clone()
            if current[0]=='original':
                captures[f'entrance_{_name}']=value.cpu().numpy()
        handles.append(layer.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for name,variant in variants.items():
                current[0]=name
                used_tiles=(-tiles[0],tiles[1]) if name=='changed_rgb' else tiles
                output=model(variant,True,used_tiles)
                snapshot={key:output[key].detach().cpu().numpy() for key in ('base_logits','residual_logits','logits')}
                snapshot['probabilities']=output['logits'].softmax(-1).cpu().numpy()
                snapshot['coordinates']=decode(output,1.,1.,None).cpu().numpy()
                outputs[name]=snapshot
                for key,value in snapshot.items():captures[f'{name}_{key}']=value
                records.append(dict(variant=name,slots48_max_abs={key:float(value[...,:48].abs().max()) for key,value in actual_inputs.items()},
                    entrance_widths={key:int(value.shape[-1]) for key,value in actual_inputs.items()},
                    arrays_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in snapshot.items()}))
                if name=='original':
                    for key in ('points_raw','candidate_displacements','point_support','box_diagonal','coverage'):
                        captures[key]=output[key].detach().cpu().numpy()
    finally:
        for handle in handles:handle.remove()
    assert all(count==0 for count in feature_calls.values())
    invariant={}
    for name in ('zero','other','missing'):
        invariant[name]={key:float(np.max(np.abs(value-outputs['original'][key]))) for key,value in outputs[name].items()}
        assert all(value==0 for value in invariant[name].values()),invariant[name]
    rgb_response={key:float(np.max(np.abs(value-outputs['original'][key]))) for key,value in outputs['changed_rgb'].items()}
    # Separate source diagnostic: keep the exact common weights and allow the
    # old full-feature forward only to record the input-removal change/context.
    nonimage={};control_hooks=[]
    for name,layer in [('regular',model.base.scorer[0]),('null',model.base.null_scorer[0])]:
        def control_capture(module,inputs,_name=name):nonimage[_name]=inputs[0].detach().cpu().numpy()
        control_hooks.append(layer.register_forward_pre_hook(control_capture))
    try:
        with torch.no_grad():control=M.NativeEvidence.forward(model,original,True,tiles)
    finally:
        for handle in control_hooks:handle.remove()
    for name in nonimage:assert np.array_equal(nonimage[name][...,48:],captures['entrance_'+name][...,48:])
    for key in ('point_support','box_diagonal','coverage'):assert np.array_equal(control[key].cpu().numpy(),captures[key])
    captures['common_weights_full_feature_control_logits']=control['logits'].cpu().numpy()
    input_removal_change=float(np.max(np.abs(captures['common_weights_full_feature_control_logits']-outputs['original']['logits'])))
    assert state_sha(model.state_dict())==state_before
    np.savez_compressed(capture_path,**captures)
    model.train(mode)
    if device.startswith('cuda'):torch.cuda.synchronize()
    return dict(seed=seed,phase=phase,status='PASS',probe_source='fixed first TRAIN batch; other actual features from fixed second TRAIN batch',
        rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),probe_frames=len(rows),variants=records,
        feature_replacement_max_abs=invariant,changed_RGB_max_abs=rgb_response,
        ablation_forward_feature_module_calls=feature_calls,ablation_scorer48_exact_zero=True,
        original_nonimage_context_coverage_support_exact=True,
        common_weights_full_feature_forward_max_logit_difference=input_removal_change,
        full_feature_control_forward_is_separate_diagnostic_only=True,
        state_unchanged_by_probe=True,model_updates=0,backbone_forwards=0,REAL_reads=0,geometry_cache_reads=0,
        private_capture=dict(relative_path=filename,sha256=sha(capture_path)),wall_seconds=time.monotonic()-start)


def preflight(args):
    start=time.monotonic();doc=Path(args.public_dir);assert not (doc/'RGB_ONLY_PREFLIGHT.json').exists()
    evidence=[]
    for seed in (1,2):
        env,refiner,data,model,order,path,targets,decode,loss=setup(args,seed)
        state_before=state_sha(model.state_dict());unused=state_sha(frozen_state(model))
        probe=feature_probe(args,seed,'PREFIT',data,model,order,targets,decode,loss,refiner)
        rows=order[0];inputs=batch(data,rows,args.device);tiles=native_tiles(data,rows,inputs,model.base.displacements,args.device)
        model.train();model.zero_grad(set_to_none=True);output=model(inputs,True,tiles)
        ce,coord,count,(gt,eligible,gain)=components(output,inputs,data,rows,refiner,targets,decode,loss)
        ce_gradient=torch.autograd.grad(ce,output['logits'],retain_graph=True)[0]
        coordinate_gradient=torch.autograd.grad(coord,output['logits'],retain_graph=True)[0]
        assert torch.isfinite(coordinate_gradient).all() and coordinate_gradient.norm()>0
        position=tuple(int(x) for x in np.unravel_index(int(coordinate_gradient.abs().argmax()),coordinate_gradient.shape))
        eps=1e-3;values=[]
        for sign in (-1,1):
            altered={key:(value.detach().double() if isinstance(value,torch.Tensor) and value.is_floating_point() else value) for key,value in output.items()}
            altered['logits']=altered['logits'].clone();altered['logits'][position]+=sign*eps
            value,_=M.coordinate_loss(altered,gt.double(),eligible,gain.double(),decode);values.append(float(value))
        fd=(values[1]-values[0])/(2*eps);automatic=float(coordinate_gradient[position])
        assert abs(fd-automatic)<=1e-6+1e-3*abs(fd),(fd,automatic)
        (ce+COORD_WEIGHT*coord).backward()
        assert all(p.grad is None for name in R.UNUSED_MODULES for p in getattr(model.base,name).parameters())
        gradient=dict(CE_logit_L2=float(ce_gradient.norm()),coordinate_logit_L2=float(coordinate_gradient.norm()),
                      coordinate_weighted_to_CE=float(COORD_WEIGHT*coordinate_gradient.norm()/ce_gradient.norm().clamp_min(1e-12)),
                      RGB_CNN_L2=gradient_norm(model.rgb.parameters()),residual_L2=gradient_norm(model.residual.parameters()),
                      coordinate_finite_difference=dict(index=list(position),epsilon=eps,automatic=automatic,finite_difference=fd),**count)
        model.zero_grad(set_to_none=True)
        assert state_sha(model.state_dict())==state_before and state_sha(frozen_state(model))==unused
        evidence.append(dict(seed=seed,initialization=model.initialization_evidence,parameters=model.parameter_counts(),
                             feature_evidence=probe,gradient_preflight=gradient,optimizer_updates=0,
                             unused_frozen_state_sha256=unused,source_checkpoint_sha256=sha(path)))
        del model,data
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    write(doc/'RGB_ONLY_PREFLIGHT.json',dict(status='PASS_BEFORE_FITS',seeds=[1,2],records=evidence,
          COORD_weight=COORD_WEIGHT,GEO_weight=0,weight_recalibration=0,optimizer_updates=0,new_geometry_cache=0,
          source_code_sha256={name:sha(Path(__file__).parent/name) for name in ('rgb_model.py','train_rgb.py')},
          wall_seconds=time.monotonic()-start,device=args.device))
    print('RGB_ONLY_PREFLIGHT_PASS',flush=True)


def train(args):
    doc=Path(args.public_dir);private=Path(args.private_dir)
    assert not (doc/'RGB_ONLY_TRAIN_RECEIPTS.json').exists()
    plan=read(doc/'PLAN_AND_CONTRACT.json');assert plan['status']=='LOCKED_BEFORE_FITS'
    for name,digest in plan['training_code_sha256'].items():assert sha(Path(__file__).parent/name)==digest
    assert set(plan['training_code_sha256'])=={'rgb_model.py','train_rgb.py'}
    pre=read(doc/'RGB_ONLY_PREFLIGHT.json');assert pre['status']=='PASS_BEFORE_FITS'
    assert pre['source_code_sha256']==plan['training_code_sha256']
    receipts=[];start_job=time.monotonic();device_seconds=0.
    for seed in (1,2):
        env,refiner,data,model,order,path,targets,decode,loss=setup(args,seed)
        assert model.initialization_evidence==next(r['initialization'] for r in pre['records'] if r['seed']==seed)
        unused_before=state_sha(frozen_state(model));initial=state_sha(model.state_dict())
        protocol=read(env.DOC/'TRAIN_PROTOCOL_LOCK.json');opt=protocol['optimizer']
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=opt['lr'],
            betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
        model.train();history=[];start=time.monotonic()
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        for step,rows in enumerate(order,1):
            inputs=batch(data,rows,args.device);tiles=native_tiles(data,rows,inputs,model.base.displacements,args.device)
            optimizer.zero_grad(set_to_none=True);lr=lr_at(step,opt)
            for group in optimizer.param_groups:group['lr']=lr
            output=model(inputs,True,tiles)
            ce,coord,count,_=components(output,inputs,data,rows,refiner,targets,decode,loss)
            value=ce+COORD_WEIGHT*coord;assert torch.isfinite(value),('NONFINITE',seed,step)
            gradients={}
            if step in (1,100,1000,STEPS):
                a=torch.autograd.grad(ce,output['logits'],retain_graph=True)[0].norm()
                b=torch.autograd.grad(coord,output['logits'],retain_graph=True)[0].norm()
                gradients=dict(CE_logit_L2=float(a),coordinate_logit_L2=float(b),coordinate_weighted_to_CE=float(COORD_WEIGHT*b/a.clamp_min(1e-12)))
            value.backward()
            assert all(p.grad is None for name in R.UNUSED_MODULES for p in getattr(model.base,name).parameters())
            if step in (1,100,1000,STEPS):
                gradients.update(RGB_CNN_L2=gradient_norm(model.rgb.parameters()),residual_L2=gradient_norm(model.residual.parameters()),
                                 base_scorer_L2=gradient_norm(model.base.scorer.parameters()),null_scorer_L2=gradient_norm(model.base.null_scorer.parameters()),
                                 role_embedding_L2=gradient_norm(model.base.role_embedding.parameters()),unused_feature_modules_gradient_None=True,
                                 zero_feature_scorer_columns_data_gradient_max=max(float(layer.weight.grad[:,:48].abs().max()) for layer in (model.base.scorer[0],model.base.null_scorer[0],model.residual[0])))
            norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),opt['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%100==0:
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                event=dict(seed=seed,step=step,loss=float(value.detach()),CE=float(ce.detach()),coord=float(coord.detach()),GEO=0,
                           lr=lr,gradient_norm=float(norm),gradients=gradients,**count,wall_seconds=time.monotonic()-start)
                history.append(event);print('RGB_ONLY_TRAIN',json.dumps(event),flush=True)
            if seed==1 and step==200:
                elapsed=time.monotonic()-start
                write(doc/'RGB_ONLY_THROUGHPUT.json',dict(status='MEASURED_FIRST200_UPDATES',seed=seed,actual_updates=200,batch=16,
                    wall_seconds=elapsed,seconds_per_update=elapsed/200,projected_two_seed3000_seconds=elapsed/200*6000,
                    planned_updates_unchanged=6000,performance_based_condition_changes=0,device_budget_seconds=2700))
            if device_seconds+time.monotonic()-start>2700:raise RuntimeError('BUDGET_LIMITED; no condition or seed substitution authorized')
        assert step==STEPS
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.monotonic()-start;device_seconds+=elapsed
        assert state_sha(frozen_state(model))==unused_before
        final_probe=feature_probe(args,seed,'POSTFIT',data,model,order,targets,decode,loss,refiner)
        assert final_probe['changed_RGB_max_abs']['logits']>0,'Trained RGB branch has no observed response on fixed TRAIN probe'
        assert state_sha(frozen_state(model))==unused_before
        checkpoint=private/f'checkpoints/RGB_ONLY_seed{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True);assert not checkpoint.exists()
        torch.save(dict(arm='RGB_ONLY',seed=seed,step=STEPS,state_dict=model.state_dict(),config=protocol['config'],
            initial_checkpoint_sha256=sha(path),initial_state_sha256=initial,order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            plan_sha256=sha(doc/'PLAN_AND_CONTRACT.json'),COORD_weight=COORD_WEIGHT,GEO_weight=0,parameter_counts=model.parameter_counts()),checkpoint)
        assert sha(path)==model.initialization_evidence['initial_checkpoint_sha256']
        receipt=dict(status='DONE',arm='RGB_ONLY',seed=seed,step=STEPS,updates=STEPS,batch=16,last_checkpoint_only=True,
            initialization=model.initialization_evidence,parameters=model.parameter_counts(),optimizer=opt,
            schedule='source prior train.lr_at, 100-step warmup and cosine horizon3000',
            order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),unique_TRAIN_rows=int(len(np.unique(order))),
            source_checkpoint_unchanged=True,COORD_weight=COORD_WEIGHT,GEO_weight=0,history=history,
            frozen_feature_modules_before_sha256=unused_before,frozen_feature_modules_after_sha256=state_sha(frozen_state(model)),
            evidence_refs=dict(prefit=next(r['feature_evidence'] for r in pre['records'] if r['seed']==seed),postfit=final_probe),
            private_checkpoint=dict(relative_path=str(checkpoint.relative_to(private)),sha256=sha(checkpoint)),
            fit_wall_seconds=elapsed,postfit_probe_wall_seconds=final_probe['wall_seconds'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None,
            backbone_updates=0,backbone_forwards=0,real_or_Severe_or_protected_training_reads=0,
            fit_p3_p4_array_reads=0,fit_feature_adapter_calls=0,new_geometry_cache=0,old_model_fits=0)
        write(doc/f'RGB_ONLY_TRAIN_RECEIPT_seed{seed}.json',receipt);receipts.append(receipt)
        del model,data,optimizer
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    write(doc/'RGB_ONLY_TRAIN_RECEIPTS.json',dict(status='DONE',model='RGB_ONLY',seeds=[1,2],fits=2,updates_per_seed=STEPS,
        actual_updates=6000,matched_last_step=True,COORD_weight=COORD_WEIGHT,GEO_weight=0,receipts=[f'RGB_ONLY_TRAIN_RECEIPT_seed{s}.json' for s in (1,2)],
        whole_job_wall_seconds=time.monotonic()-start_job,fit_wall_seconds=device_seconds,device_budget_seconds=2700,
        backbone_updates=0,existing_model_fits=0,new_geometry_cache=0,new_synthetic_images=0,REAL_selection=False,
        source_code_sha256={name:sha(Path(__file__).parent/name) for name in ('rgb_model.py','train_rgb.py')}))
    print('RGB_ONLY_TWO_FITS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','train'])
    parser.add_argument('--source-root',required=True);parser.add_argument('--private-dir',required=True);parser.add_argument('--public-dir',required=True)
    parser.add_argument('--prior-public-dir');parser.add_argument('--prior-private-dir');parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    args=parser.parse_args();configure();(preflight if args.stage=='preflight' else train)(args)
