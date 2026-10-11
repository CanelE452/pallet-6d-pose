"""Detached, exact-renderer synthetic covariance sensitivity; never fits a pose.

The source camera perturbation is R'=exp([dr]x)R, t'=t+dt.  Only the
eight corners enter.  A24=J3d (J2d.T W J2d)^-1 J2d.T W is computed once;
the loss is mean_i ||A24[3i:3i+3,:] diag(r)||_F, in metres.
"""
import argparse, hashlib, importlib.util, json, math, time
from pathlib import Path
import numpy as np
import torch
import precision_model as M

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def load_module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def inputs(source):
    root=Path(source);line=root/'data/pallet/results/pallet_line_pose_v1'
    source=json.loads((line/'SOURCE_MANIFEST.json').read_text())
    cache=line/'cache'
    arrays={k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ('points','point_valid','boxes','input_shape','gt_points','gt_valid','gain','record_index','matched')}
    side_file=np.load(root/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz')
    side={k:side_file[k] for k in side_file.files}
    original=np.load(root/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz')
    assert np.array_equal(side['record_index'],arrays['record_index'])
    assert len(source['records'])==len(original['stems'])==60000
    ids=[r['id'] for r in source['records']];stems=original['stems'].tolist()
    assert len(set(ids))==len(set(stems))==60000 and set(ids)==set(stems)
    lookup={s:i for i,s in enumerate(stems)};join=np.array([lookup[s] for s in ids])
    exact={k:original[k][join] for k in original.files}
    exact['source_join']=join
    assert all(source['records'][i]['id']==str(exact['stems'][i]) for i in range(60000))
    assert all(r['source_kind']=='synthetic' for r in source['records'])
    return source,arrays,side,exact

def metadata(source,arrays,rows):
    records=[source['records'][int(arrays['record_index'][r])] for r in rows]
    gain=np.asarray(arrays['gain'][rows],np.float64)
    offsets=np.array([[round((int(arrays['input_shape'][r,1])-round(x['prepared_shape_hw'][1]*g))/2-.1),round((int(arrays['input_shape'][r,0])-round(x['prepared_shape_hw'][0]*g))/2-.1)] for r,x,g in zip(rows,records,gain)],np.float64)
    pad=np.array([r['reflect_pad_px'] for r in records],np.float64)
    hw=np.array([r['raw_shape_hw'] for r in records],np.int64)
    return gain,offsets,pad,hw

def fixed_outputs(arrays,rows,base):
    """Exact source support, using separability of its Cartesian 4x8 stencil."""
    p=torch.tensor(np.array(arrays['points'][rows]),dtype=torch.float32)
    b=torch.tensor(np.array(arrays['boxes'][rows]),dtype=torch.float32)
    valid=torch.tensor(np.array(arrays['point_valid'][rows])).bool()&torch.isfinite(p).all(-1)&~(p==-1).all(-1)
    box_valid=torch.isfinite(b).all(-1)&(b[:,2:]>b[:,:2]).all(-1)
    safe=torch.where(valid[...,None],p,torch.zeros_like(p))
    bb=torch.where(box_valid[:,None],b,b.new_tensor([0,0,1,1]))
    diag=(bb[:,2:]-bb[:,:2]).norm(dim=-1).clamp_min(1)
    candidates=diag[:,None,None]*base.displacements[None]
    locations=safe[:,:8,None]+candidates[:,None,:-1]
    shape=torch.tensor(np.array(arrays['input_shape'][rows]),dtype=torch.float32)
    xs=locations[...,0,None]+diag[:,None,None,None]*base.stencil_fraction*base.stencil[:8,0]
    ys=locations[...,1,None]+diag[:,None,None,None]*base.stencil_fraction*base.stencil[::8,1]
    inside_x=((xs>=0)&(xs<shape[:,None,None,None,1])).any(-1)
    inside_y=((ys>=0)&(ys<shape[:,None,None,None,0])).any(-1)
    support=valid[:,:8]&box_valid[:,None]&(inside_x&inside_y).any(-1)
    return dict(points_raw=p,point_valid=valid,point_support=support,box_diagonal=diag,candidate_displacements=candidates,logits=p.new_zeros((len(rows),8,222)))

def native_reference(source,arrays,exact,row,permutation=None):
    """Renderer truth in native pixels, without recovering a signed runtime pose."""
    i=int(arrays['record_index'][row]);k=exact['K'][i].copy();k[2:]-=exact['pad'][i]
    K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]],np.float64)
    X=exact['Xcf'][i].copy()
    if permutation is not None:X=X[np.asarray(permutation)[:8]]
    return K,exact['R'][i].copy(),exact['t'][i].copy(),X,exact['dims'][i].copy()

def covariance_loss(residual,A24,valid):
    A24=A24.detach();weighted=(A24*residual.reshape(-1,1,16)).reshape(-1,8,48)
    per=torch.linalg.vector_norm(weighted,dim=-1).mean(-1)
    return (per*valid).sum()/valid.sum().clamp_min(1)

def geo_for(cache,rows,branch,mask,device=None):
    rows=np.asarray(rows);assert np.array_equal(cache['branch'][rows],branch.detach().cpu().numpy())
    assert np.array_equal(cache['eligible'][rows],mask.detach().cpu().numpy())
    device=mask.device if device is None else device
    return torch.as_tensor(np.array(cache['A24'][rows]),device=device).detach(),torch.as_tensor(np.array(cache['valid'][rows]),device=device).bool()

def summary(x):
    x=np.asarray(x,dtype=np.float64);x=x[np.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None,p99=float(np.quantile(x,.99)) if len(x) else None,max=float(x.max()) if len(x) else None)

def build(args):
    start=time.monotonic();root=Path(args.source_root);doc=Path(args.public_dir);private=Path(args.private_dir)
    assert not (private/'GEOMETRY_CACHE.npz').exists()
    bindings=['scripts/research/pallet_translation_loss_v1/pnp_jacobian.py','challenge/yolo_pose_one_model/pallet_translation_loss_v1/a2b_loss.py','challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz','data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json','data/pallet/results/pallet_line_pose_v1/cache/CACHE_MANIFEST.json','data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz','scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py','scripts/research/pallet_dim_conditioned_p_v1/refiner.py','_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/PARAMETER_BUDGET_LOCK.json']
    lock=dict(status='LOCKED_BEFORE_GEOMETRY_ARITHMETIC',code_sha256=sha(__file__),model_sha256=sha(Path(__file__).parent/'precision_model.py'),bindings=[dict(path=p,sha256=sha(root/p)) for p in bindings],parameters=dict(minimum_auxiliary_corners=4,rank_required=6,svd_relative_cutoff=1e-6,maximum_condition=1e6,damping=0.,projection_max_residual_gate_px=.5,finite_difference_eps=1e-6,finite_difference_frames=32,centroid_included=False,geometry_units='native pixels, camera-axis radians/metres, covariance metres',branch='source local_phase once from frozen initial points; whole object permutation',loss='pure E_cov mean eight 3D corner covariance standard deviations; not full LC',old_A2B_zero_diag_fallback='excluded: original helper returns 1 at exact zero; pure covariance norm returns 0'))
    suffix=0
    while (doc/('GEOMETRY_LOCK.json' if suffix==0 else f'GEOMETRY_REPAIR_LOCK_{suffix}.json')).exists():suffix+=1
    write(doc/('GEOMETRY_LOCK.json' if suffix==0 else f'GEOMETRY_REPAIR_LOCK_{suffix}.json'),lock)
    source,arrays,side,exact=inputs(root)
    _,refiner,_,Base,targets,decode,_=M.source_modules(root)
    config=json.loads((root/bindings[-1]).read_text())['config'];base=Base(**config);jac=load_module(root/bindings[0],'source_pnp_jacobian')
    n=len(arrays['record_index']);A=np.zeros((n,24,16),np.float32);eligible=np.zeros((n,8),bool);branch=np.zeros(n,np.int8);valid=np.zeros(n,bool)
    residual_max=np.full(n,np.nan);conditions=np.full(n,np.nan);ranks=np.zeros(n,np.int8);reason=np.full(n,'TOO_FEW_AUXILIARY_CORNERS',dtype='<U40')
    sample=sorted(range(n),key=lambda r:hashlib.sha256((source['records'][int(arrays['record_index'][r])]['id']+'geometry-fd').encode()).digest())[:32]
    fd=[];fd3=[];parity=[]
    for lo in range(0,n,256):
        rows=np.arange(lo,min(lo+256,n));out=fixed_outputs(arrays,rows,base)
        gt=torch.tensor(np.array(arrays['gt_points'][rows]),dtype=torch.float32);gvalid=torch.tensor(np.array(arrays['gt_valid'][rows])).bool()
        perm=torch.tensor(side['permutations'][rows]).long();groups=torch.tensor(side['group_valid'][rows]).bool()
        chosen,v,br,_=refiner.local_phase(out['points_raw'],out['point_valid'],gt,gvalid,perm,groups,out['box_diagonal'])
        gain,offset,pad,hw=metadata(source,arrays,rows)
        mask,target=M.aux_mask(out,chosen,v,torch.tensor(gain,dtype=torch.float32),torch.tensor(hw),targets)
        eligible[rows]=mask.numpy();branch[rows]=br.numpy();qnative=(chosen.numpy()-offset[:,None])/gain[:,None,None]-pad[:,None,None]
        for j,row in enumerate(rows):
            K,R,t,X,_=native_reference(source,arrays,exact,int(row),side['permutations'][row,branch[row]])
            predicted=jac.project(K,R,t,X)
            gv=v[j,:8].numpy();residual_max[row]=np.linalg.norm(predicted-qnative[j,:8],axis=-1)[gv].max() if gv.any() else np.nan
            if not np.isfinite(residual_max[row]) or residual_max[row]>.5:reason[row]='PROJECTION_RESIDUAL_GT_0_5PX';continue
            if np.sum(eligible[row])<4:continue
            if not np.isfinite(X).all() or not np.isfinite(R).all() or not np.isfinite(t).all() or ((X@R.T+t)[:,2]<=0).any():reason[row]='NONFINITE_OR_NONPOSITIVE_DEPTH';continue
            J=jac.pnp_jacobian(K,R,t,X);Jm=J*np.repeat(eligible[row],2)[:,None]
            sv=np.linalg.svd(Jm,compute_uv=False);ranks[row]=np.sum(sv>sv[0]*1e-6);conditions[row]=sv[0]/sv[-1] if sv[-1]>0 else np.inf
            if ranks[row]!=6 or conditions[row]>1e6:reason[row]='RANK_OR_CONDITION_GATE';continue
            RX=X@R.T;J3=np.concatenate([-np.array([jac.skew(x) for x in RX]),np.tile(np.eye(3),(8,1,1))],axis=-1).reshape(24,6)
            coefficient=J3@np.linalg.solve(Jm.T@Jm,Jm.T)
            if not np.isfinite(coefficient).all():reason[row]='NONFINITE_COEFFICIENT';continue
            A[row]=coefficient;valid[row]=True;reason[row]='VALID'
            if row in sample:
                f=jac.finite_difference_jacobian(K,R,t,X,1e-6)
                fd.append(np.linalg.norm(J-f,axis=0)/np.maximum(np.linalg.norm(f,axis=0),1e-12))
                f3=np.zeros((24,6))
                for c in range(6):
                    d=np.zeros(6);d[c]=1e-6
                    pp=X@(jac.expm_so3(d[:3])@R).T+t+d[3:]
                    pm=X@(jac.expm_so3(-d[:3])@R).T+t-d[3:]
                    f3[:,c]=((pp-pm)/2e-6).reshape(-1)
                fd3.append(np.linalg.norm(J3-f3,axis=0)/np.maximum(np.linalg.norm(f3,axis=0),1e-12))
                parity.append(float(np.max(np.abs(Jm@np.linalg.solve(Jm.T@Jm,Jm.T)@Jm-Jm))))
        if lo%8192==0:print('GEOMETRY_CACHE',lo,n,'valid',int(valid[:lo+len(rows)].sum()),flush=True)
    # FD gates cover eligible exact renderer rows; absence disables only geo arms.
    fd_stats=summary(np.array(fd).ravel());fd3_stats=summary(np.array(fd3).ravel())
    fd_pass=len(fd)>0 and fd_stats['median']<1e-4 and fd_stats['p99']<1e-2 and fd3_stats['p99']<1e-4
    assert fd_pass,('JACOBIAN_FINITE_DIFFERENCE_FAIL',fd_stats,fd3_stats)
    rng=np.random.default_rng(20261011);good=np.flatnonzero(valid);assert len(good)>0
    example=A[good[:8]].astype(np.float64);r=torch.tensor(rng.normal(size=(len(example),8,2)),dtype=torch.float64,requires_grad=True)
    a=torch.tensor(example);v=torch.ones(len(example),dtype=torch.bool)
    value=covariance_loss(r,a,v);grad=torch.autograd.grad(value,r)[0];errors=[]
    for idx in [(0,0,0),(0,1,1),(min(1,len(example)-1),2,0)]:
        rp=r.detach().clone();rm=rp.clone();rp[idx]+=1e-6;rm[idx]-=1e-6
        numeric=float((covariance_loss(rp,a,v)-covariance_loss(rm,a,v))/2e-6)
        errors.append(abs(numeric-float(grad[idx]))/max(abs(numeric),1e-12))
    zero=torch.zeros_like(r,requires_grad=True);zloss=covariance_loss(zero,a,v);zgrad=torch.autograd.grad(zloss,zero)[0]
    assert float(zloss)==0 and torch.isfinite(zgrad).all() and (zgrad==0).all() and max(errors)<1e-4
    # Physical proper groups are accepted only by the source object contract.
    group_counts={str(k):int((side['order']==k).sum()) for k in (1,2,4)}
    assert (side['permutations'][:,:,8]==8).all()
    for row in range(n):
        for p in side['permutations'][row,side['group_valid'][row]]:assert sorted(p.tolist())==list(range(9))
    np.savez(private/'GEOMETRY_CACHE.npz',A24=A,valid=valid,branch=branch,eligible=eligible,record_index=arrays['record_index'])
    np.savez(private/'GEOMETRY_DIAGNOSTICS.npz',residual_max=residual_max,condition=conditions,rank=ranks,reason=reason)
    parts=np.array([source['records'][int(i)]['partition'] for i in arrays['record_index']])
    report=dict(status='PASS_PARTIAL_ELIGIBILITY',rows=n,partition_results={p:dict(rows=int((parts==p).sum()),geo_eligible_frames=int((valid&(parts==p)).sum()),auxiliary_eligible_corners=int(eligible[parts==p].sum()),reason_counts={str(k):int(c) for k,c in zip(*np.unique(reason[parts==p],return_counts=True))}) for p in sorted(set(parts))},group_orders=group_counts,centroid_excluded=True,whole_object_branch_counts={str(k):int(c) for k,c in zip(*np.unique(branch,return_counts=True))},projection_residual_native_px=summary(residual_max),condition_number=summary(conditions),fd_J2d=fd_stats,fd_J3d24=fd3_stats,fd_actual_eligible_frames=len(fd),fd_requested_frames=32,masked_projection_identity_max_px=float(max(parity)) if parity else None,covariance_logit_chain_part='residual gradient; logits chain independently decoder-audited',covariance_residual_gradient_fd_relative_errors=errors,exact_zero_loss=float(zloss),exact_zero_gradient_finite=bool(torch.isfinite(zgrad).all()),exact_zero_gradient_norm=float(zgrad.norm()),loss_normalization='mean of eight sqrt(sum of the three 3D covariance diagonal components), no division by24, units metres',fixed_correspondence_weights='eligible 0/1 mask, no learned uncertainty weights',damping=0.,rank_gate=6,minimum_auxiliary_corners=4,geometry_detached=True,projection_arithmetic_calls=n,pose_solver_calls=0,new_model_inference_calls=0,optimizer_updates=0,raw_rgb_reads=0,real_reads=0,old_runtime_C1_signed_axis_gate='separate failed legacy runtime mapping is not repaired or waived; exact renderer TRAIN Jacobian uses original Xcf/R/t only',private_cache_artifact='GEOMETRY_CACHE.npz',private_cache_sha256=sha(private/'GEOMETRY_CACHE.npz'),source_bindings_unchanged=all(sha(root/b['path'])==b['sha256'] for b in lock['bindings']),actual_cpu_wall_seconds=time.monotonic()-start)
    write(doc/'GEOMETRY_AUDIT.json',report)
    print('GEOMETRY_DONE',len(good),'FD',fd_stats,'CPU',report['actual_cpu_wall_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True)
    args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);assert args.build;build(args)
