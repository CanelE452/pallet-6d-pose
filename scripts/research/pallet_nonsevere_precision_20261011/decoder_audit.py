"""Audit actual source target/decode operators using existing synthetic residuals.

No new neural inference, pose solving, model fit, or raw RGB is performed.
All coordinates remain private; published frame rows contain derived scalars.
"""
import argparse,gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import geometry as G
import precision_model as M

def correlation(x,y):
    x=np.asarray(x);y=np.asarray(y);keep=np.isfinite(x)&np.isfinite(y)
    return float(np.corrcoef(x[keep],y[keep])[0,1]) if keep.sum()>2 and np.std(x[keep])>0 and np.std(y[keep])>0 else None

def half_pixel(source,Base):
    """Evaluate source feature centers and native integer pixel centers exactly."""
    module=G.load_module(Path(source)/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py','source_decoder_half_pixel')
    feature=torch.arange(8,dtype=torch.float64)[None,None,None,:].expand(1,1,4,8).clone()
    positions=torch.tensor([[[[12.,12.],[16.,12.],[20.,12.]]]],dtype=torch.float64)
    sampled=module.sample(feature,positions,torch.tensor([[32.,64.]],dtype=torch.float64),8).reshape(-1)
    expected=torch.tensor([1.,1.5,2.],dtype=torch.float64)
    native=torch.arange(8,dtype=torch.float64)[None,None,None,:].expand(1,1,4,8).clone()
    pos=torch.tensor([[[[1.,1.],[1.5,1.],[2.,1.]]]],dtype=torch.float64)
    grid=2*(pos+.5)/pos.new_tensor([8,4])-1
    got=F.grid_sample(native,grid,align_corners=False).reshape(-1)
    return dict(source_feature_center_max_error=float((sampled-expected).abs().max()),native_integer_center_max_error=float((got-expected).abs().max()),source_feature_formula='2*prepared_position/(stride*feature_size)-1; feature centers=(index+0.5)*stride',native_formula='2*(native_pixel_index+0.5)/tile_size-1',global_half_pixel_coordinate_shift=False,native_tiles_resized=False)

def logit_gradient(out,gt,gain,eligible,decode):
    logits=out['logits'].double().detach().requires_grad_(True)
    o={k:(v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k,v in out.items()};o['logits']=logits
    # Native differentiable auxiliary, independent of lam=0 source forward output.
    loss,_=M.coordinate_loss(o,gt.double(),eligible,gain.double(),decode)
    grad=torch.autograd.grad(loss,logits)[0];chosen=torch.nonzero(grad.abs()>1e-6)[:8];errors=[]
    for index in chosen.tolist():
        ix=tuple(index);lp=logits.detach().clone();lm=lp.clone();lp[ix]+=1e-5;lm[ix]-=1e-5
        op=dict(o,logits=lp);om=dict(o,logits=lm)
        fp=M.coordinate_loss(op,gt.double(),eligible,gain.double(),decode)[0]
        fm=M.coordinate_loss(om,gt.double(),eligible,gain.double(),decode)[0]
        numeric=float((fp-fm)/2e-5);errors.append(abs(numeric-float(grad[ix]))/max(abs(numeric),1e-12))
    assert errors and max(errors)<1e-4 and torch.isfinite(grad).all() and grad.norm()>0
    assert torch.equal(decode(o,1,0,None),o['points_raw'])
    return dict(status='PASS',finite_difference_entries=len(errors),relative_errors=errors,gradient_norm=float(grad.norm()),lam_zero_initial_points_exact=True,auxiliary_uses_expected_decode_temperature=1.,auxiliary_gain_lambda=1.,auxiliary_cap=None)

def run(args):
    start=time.monotonic();root=Path(args.source_root);doc=Path(args.public_dir);private=Path(args.private_dir)
    source,arrays,side,exact=G.inputs(root);_,refiner,_,Base,targets,decode,_=M.source_modules(root)
    config=json.loads((root/'_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/PARAMETER_BUDGET_LOCK.json').read_text())['config'];base=Base(**config)
    parts=np.array([source['records'][int(i)]['partition'] for i in arrays['record_index']]);n=len(parts)
    binding=['scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py','data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json','data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz','_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/P_SELECTION.json']
    logs={s:np.load(root/f'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/validation_P{s}.npz') for s in (1,2)}
    log_bindings={str(s):G.sha(root/f'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/validation_P{s}.npz') for s in (1,2)}
    lock=dict(status='LOCKED_BEFORE_DECODER_ARITHMETIC',code_sha256=G.sha(__file__),geometry_code_sha256=G.sha(G.__file__),bindings=[dict(path=p,sha256=G.sha(root/p)) for p in binding],network_logits_sha256=log_bindings,target_rows=n,temperatures=[1.,2.],gain_lambdas=[1.,.5],cap='fixed 1% native image diagonal, applied after expectation',classification='actual 13-spoke convex hull and native cap; valid/support must pass',no_new_network_forward=True)
    attempt=0
    while (doc/('DECODER_LOCK.json' if attempt==0 else f'DECODER_REPAIR_LOCK_{attempt}.json')).exists():attempt+=1
    G.write(doc/('DECODER_LOCK.json' if attempt==0 else f'DECODER_REPAIR_LOCK_{attempt}.json'),lock)
    geometryfile=np.load(private/'GEOMETRY_CACHE.npz') if (private/'GEOMETRY_CACHE.npz').exists() else None
    geometry={k:geometryfile[k] for k in geometryfile.files} if geometryfile is not None else None
    targetrows=[];probabilityrows=[];stats={};gradient=None
    rowsname='DECODER_FRAME_SCALARS.jsonl.gz' if attempt==0 else f'DECODER_FRAME_SCALARS_ATTEMPT{attempt}.jsonl.gz'
    with gzip.open(doc/rowsname,'xt',encoding='utf-8') as f:
        for lo in range(0,n,256):
            rows=np.arange(lo,min(n,lo+256));out=G.fixed_outputs(arrays,rows,base)
            gt=torch.tensor(np.array(arrays['gt_points'][rows]),dtype=torch.float32);gv=torch.tensor(np.array(arrays['gt_valid'][rows])).bool()
            gt,gv,branch,_=refiner.local_phase(out['points_raw'],out['point_valid'],gt,gv,torch.tensor(side['permutations'][rows]).long(),torch.tensor(side['group_valid'][rows]).bool(),out['box_diagonal'])
            gain,offset,pad,hw=G.metadata(source,arrays,rows);g=torch.tensor(gain,dtype=torch.float32);cap=torch.tensor(np.linalg.norm(hw,axis=-1)*.01,dtype=torch.float32)
            mask,target=M.aux_mask(out,gt,gv,g,torch.tensor(hw),targets)
            residual=target['residual']/g[:,None,None];wanted=residual.norm(dim=-1);support=target['support']
            radius=out['box_diagonal']*.08/g;theta=(torch.arange(13,dtype=torch.float32)+.5)*2*math.pi/13;normals=torch.stack([theta.cos(),theta.sin()],-1)
            hull=(torch.einsum('bkd,ad->bka',residual,normals)<=radius[:,None,None]*math.cos(math.pi/13)+1e-6).all(-1)
            targetout=dict(out,logits=target['distribution'].clamp_min(torch.finfo(torch.float32).tiny).log())
            native_deltas={};native_errors={}
            for T,lam,capped,label in [(1.,1.,False,'T1_gain1_uncapped'),(1.,1.,True,'T1_gain1_capped'),(2.,1.,False,'T2_gain1_uncapped'),(1.,.5,False,'T1_gain0_5_uncapped')]:
                q=decode(targetout,T,lam,cap*g if capped else None)
                delta=(q[:,:8]-out['points_raw'][:,:8])/g[:,None,None]
                native_deltas[label]=delta.norm(dim=-1).numpy();native_errors[label]=(delta-residual).norm(dim=-1).numpy()
                if label=='T1_gain1_uncapped':uncapped_error=(delta-residual).numpy()
            if gradient is None and mask.sum()>0:gradient=logit_gradient(targetout,gt,g,mask,decode)
            for j,row in enumerate(rows):
                sup=support[j].numpy();e=mask[j].numpy();h=hull[j].numpy();beyondcap=(wanted[j]>cap[j]+1e-6).numpy()
                sensitivity=None
                if geometry is not None and geometry['valid'][row]:
                    rv=np.where(e[:,None],uncapped_error[j],0.)
                    sensitivity=float(np.linalg.norm((geometry['A24'][row]*rv.reshape(1,16)).reshape(8,48),axis=-1).mean())
                d=dict(id=source['records'][int(arrays['record_index'][row])]['id'],partition=str(parts[row]),supported_corners=int(sup.sum()),eligible_corners=int(e.sum()),outside_candidate_hull_corners=int((sup&~h).sum()),beyond_native_cap_corners=int((sup&beyondcap).sum()),desired_native_displacement_mean_px=float(wanted[j].numpy()[sup].mean()) if sup.any() else None,box_diagonal_native_px=float(out['box_diagonal'][j]/g[j]),target_covariance_sensitivity_m=sensitivity)
                for label in native_deltas:
                    d[label+'_movement_mean_px']=float(native_deltas[label][j][sup].mean()) if sup.any() else None
                    d[label+'_error_mean_px']=float(native_errors[label][j][sup].mean()) if sup.any() else None
                f.write(json.dumps(d,allow_nan=False)+'\n');targetrows.append(d)
            if lo%16384==0:print('DECODER_TARGET_AUDIT',lo,n,flush=True)
        # Existing actual network outputs; no feature or neural forward is called.
        for seed,l in logs.items():
            for lo in range(0,len(l['rows']),256):
                rows=l['rows'][lo:lo+256];out=G.fixed_outputs(arrays,rows,base);out['logits']=torch.tensor(l['logits'][lo:lo+256]);out['point_support']=torch.tensor(l['support'][lo:lo+256]).bool()
                gt=torch.tensor(np.array(arrays['gt_points'][rows]),dtype=torch.float32);gv=torch.tensor(np.array(arrays['gt_valid'][rows])).bool()
                gt,gv,branch,_=refiner.local_phase(out['points_raw'],out['point_valid'],gt,gv,torch.tensor(side['permutations'][rows]).long(),torch.tensor(side['group_valid'][rows]).bool(),out['box_diagonal'])
                gain,_,_,hw=G.metadata(source,arrays,rows);g=torch.tensor(gain,dtype=torch.float32);cap=torch.tensor(np.linalg.norm(hw,axis=-1)*.01,dtype=torch.float32)
                target=targets(out,gt,gv);prob=out['logits'].softmax(-1);q=decode(out,1,1,None);qc=decode(out,1,1,cap*g)
                delta=(q[:,:8]-out['points_raw'][:,:8])/g[:,None,None];dc=(qc[:,:8]-out['points_raw'][:,:8])/g[:,None,None]
                for j,row in enumerate(rows):
                    sup=target['support'][j].numpy();movement=delta[j].norm(dim=-1).numpy();err=((q[j,:8]-gt[j,:8])/gain[j]).norm(dim=-1).numpy();old=((out['points_raw'][j,:8]-gt[j,:8])/gain[j]).norm(dim=-1).numpy()
                    d=dict(id=source['records'][int(arrays['record_index'][row])]['id'],partition=str(parts[row]),seed=seed,supported_corners=int(sup.sum()),mean_expected_move_native_px=float(movement[sup].mean()) if sup.any() else None,mean_capped_move_native_px=float(dc[j].norm(dim=-1).numpy()[sup].mean()) if sup.any() else None,mean_expected_error_native_px=float(err[sup].mean()) if sup.any() else None,mean_initial_error_native_px=float(old[sup].mean()) if sup.any() else None,mean_probability_entropy=float((-(prob[j]*prob[j].clamp_min(1e-30).log()).sum(-1)).numpy()[sup].mean()) if sup.any() else None,mean_null_probability=float(prob[j,:,-1].numpy()[sup].mean()) if sup.any() else None)
                    probabilityrows.append(d)
    with gzip.open(doc/'DECODER_PROBABILITY_SCALARS.jsonl.gz','xt',encoding='utf-8') as f:
        for d in probabilityrows:f.write(json.dumps(d,allow_nan=False)+'\n')
    def agg(records,key):return G.summary([r[key] for r in records if r[key] is not None])
    fields=[k for k in targetrows[0] if k.endswith('_px') or k.endswith('_m')]
    targetsummary={p:{k:agg([r for r in targetrows if r['partition']==p],k) for k in fields} for p in sorted(set(parts))}
    networksummary={f'seed{s}_{p}':{k:agg([r for r in probabilityrows if r['seed']==s and r['partition']==p],k) for k in ('mean_expected_move_native_px','mean_capped_move_native_px','mean_expected_error_native_px','mean_initial_error_native_px','mean_probability_entropy','mean_null_probability')} for s in (1,2) for p in ('calibration','selection','heldout')}
    half=half_pixel(root,Base);assert half['source_feature_center_max_error']<1e-12 and half['native_integer_center_max_error']<1e-12
    corrrows=[r for r in targetrows if r['target_covariance_sensitivity_m'] is not None and r['desired_native_displacement_mean_px'] is not None]
    report=dict(status='PASS',target_rows=n,actual_source_operator='GenericPointRefiner.targets and decode',target_summary=targetsummary,actual_network_summary=networksummary,actual_network_probability_rows=len(probabilityrows),network_source='existing OLD_P validation_P1/P2 logits; two seeds separately',training_network_probability_audit='ROOT calibration capture supplements this audit; no TRAIN inference is fabricated',corner_counts=dict(supported=sum(r['supported_corners'] for r in targetrows),auxiliary_eligible=sum(r['eligible_corners'] for r in targetrows),outside_hull=sum(r['outside_candidate_hull_corners'] for r in targetrows),beyond_native_cap=sum(r['beyond_native_cap_corners'] for r in targetrows)),half_pixel=half,logits_gradient=gradient,same_frame_displacement_vs_local_covariance_correlation=correlation([r['desired_native_displacement_mean_px'] for r in corrrows],[r['target_covariance_sensitivity_m'] for r in corrrows]),sensitivity_definition='fixed exact-TRAIN local covariance surrogate in metres, not an actual PnP pose error',global_displacement_rescaling_applied=False,mean_bias='actual irregular origin/null/radial candidate density makes expectation differ from Gaussian target center; no universal1.33 correction',loss_native_gain_division=True,training_temperature=1.,training_lambda=1.,training_expected_decode_uncapped=True,final_cap_native_image_diagonal_fraction=.01,centroid8_fixed=True,raw_rgb_reads=0,pose_solver_calls=0,new_network_inferences=0,optimizer_updates=0,real_reads=0,source_bindings_unchanged=all(G.sha(root/b['path'])==b['sha256'] for b in lock['bindings']),actual_cpu_wall_seconds=time.monotonic()-start,paper_covariance_source='https://arxiv.org/html/2303.11516',paper_scope='fixed pure covariance Eq11 normalization; not full Eq17 LC loss')
    G.write(doc/'DECODER_AUDIT.json',report);print('DECODER_AUDIT_DONE',report['corner_counts'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);run(a)
