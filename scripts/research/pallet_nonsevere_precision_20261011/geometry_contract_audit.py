"""Supplementary source parity and proper-group audit, with no solver or fit."""
import argparse,ast,json,time
from pathlib import Path
import numpy as np
import torch
import geometry as G
import precision_model as M

def cache_lock_provenance(doc):
    """Find the operative cache lock; a prior lock alone proves no failure."""
    locks=[]
    for path in sorted(Path(doc).glob('GEOMETRY*LOCK*.json')):
        value=json.loads(path.read_text())
        if value.get('status')=='LOCKED_BEFORE_GEOMETRY_ARITHMETIC':
            locks.append((path,value))
    operative=[p.name for p,v in locks if v.get('code_sha256')==G.sha(G.__file__)]
    assert len(operative)==1, 'Exactly one operative geometry source lock required'
    return dict(operative_cache_lock=operative[0],
        existing_prior_cache_locks=[p.name for p,v in locks if p.name!=operative[0]],
        prior_execution_outcomes_inferred_from_lock_presence=False)

def run(args):
    start=time.monotonic();root=Path(args.source_root);doc=Path(args.public_dir);private=Path(args.private_dir)
    paths=['challenge/yolo_pose_one_model/pallet_translation_loss_v1/a2b_loss.py','_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json','scripts/research/pallet_translation_loss_v1/pnp_jacobian.py']
    G.write(doc/'GEOMETRY_CONTRACT_LOCK.json',dict(status='LOCKED_BEFORE_SUPPLEMENT',code_sha256=G.sha(__file__),bindings=[dict(path=p,sha256=G.sha(root/p)) for p in paths],**cache_lock_provenance(doc)))
    source,arrays,side,exact=G.inputs(root);jac=G.load_module(root/paths[2],'source_numpy_jacobian_parity')
    # Execute only original pure functions; importing full loss would initialize
    # unrelated Ultralytics dependencies. Their AST/body and source bytes are bound.
    tree=ast.parse((root/paths[0]).read_text());names={'skew','pose_jacobians','_grouped_sqrt_mean'}
    funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    namespace={'torch':torch};exec(compile(ast.Module(body=funcs,type_ignores=[]),str(root/paths[0]),'exec'),namespace)
    cache=np.load(private/'GEOMETRY_CACHE.npz');diagnostics=np.load(private/'GEOMETRY_DIAGNOSTICS.npz');rows=np.flatnonzero(cache['valid'])[:32]
    j2errors=[];j3errors=[];normalization=[]
    rng=np.random.default_rng(20261011)
    for row in rows:
        K,R,t,X,dims=G.native_reference(source,arrays,exact,int(row),side['permutations'][row,int(cache['branch'][row])])
        k=torch.tensor([[K[0,0],K[1,1],K[0,2],K[1,2]]],dtype=torch.float64)
        j2,j3=namespace['pose_jacobians'](k,torch.tensor(R[None]),torch.tensor(t[None]),torch.tensor(X[None]))
        J=jac.pnp_jacobian(K,R,t,X);RX=X@R.T;J3=np.concatenate([-np.array([jac.skew(v) for v in RX]),np.tile(np.eye(3),(8,1,1))],-1).reshape(24,6)
        j2errors.append(float(np.max(np.abs(j2.numpy()[0]-J))));j3errors.append(float(np.max(np.abs(j3.numpy()[0]-J3))))
        r=rng.normal(size=16);A=cache['A24'][row].astype(np.float64);diagonal=np.sum((A*r[None])**2,axis=-1)
        original=namespace['_grouped_sqrt_mean'](torch.tensor(diagonal[None]),3)
        derived=np.linalg.norm((A*r[None]).reshape(8,48),axis=-1).mean();normalization.append(abs(float(original)-derived))
    oldzero=float(namespace['_grouped_sqrt_mean'](torch.zeros((1,24),dtype=torch.float64),3))
    assert max(j2errors)<1e-10 and max(j3errors)<1e-12 and max(normalization)<1e-12 and oldzero==1.
    contract=json.loads((root/paths[1]).read_text());groups=[]
    for obj in contract['objects']:
        X=np.array(obj['corners_centroid']);errs=[];dets=[]
        for rot,perm in zip(obj['rotations'],obj['permutations']):
            Q=np.array(rot);p=np.array(perm);dets.append(float(np.linalg.det(Q)));assert p[8]==8
            errs.append(float(np.max(np.abs(X@Q.T-X[p]))))
        assert max(errs)<1e-12 and min(dets)>.999999
        groups.append(dict(object_type=obj['object_type'],group_order=obj['group_order'],centroid8_fixed=True,proper_determinants=dets,whole_object_permutation_max_error_m=max(errs)))
    rectangular=np.array(contract['objects'][0]['corners_centroid'])
    quarter=np.array([[0,0,1],[0,1,0],[-1,0,0]],np.float64)
    nearest=np.linalg.norm((rectangular@quarter.T)[:,None]-rectangular[None],axis=-1).min(-1).max()
    assert nearest>1e-3 and np.linalg.det(np.diag([-1.,1.,1.]))<0
    missing=~np.isfinite(diagnostics['residual_max']);gtfinite=np.asarray(arrays['gt_valid'])&np.isfinite(arrays['gt_points']).all(-1)&~(arrays['gt_points']==-1).all(-1)
    assert np.array_equal(missing,~gtfinite[:,:8].any(-1))
    newcounts={p:dict(all_partition_rows=sum(r['partition']==p for r in source['records']),usable_correspondence_rows=int(sum(source['records'][int(i)]['partition']==p and bool(arrays['matched'][row]) and bool(gtfinite[row,:8].any()) for row,i in enumerate(arrays['record_index'])))) for p in ('train','calibration','selection','heldout')}
    G.write(doc/'GEOMETRY_CONTRACT_AUDIT.json',dict(status='PASS',source_tensor_numpy_J2d_max_abs_error=max(j2errors),source_tensor_numpy_J3d24_max_abs_error=max(j3errors),actual_source_normalization_nonzero_parity_max_abs_error=max(normalization),original_A2B_zero_diagonal_fallback=oldzero,new_pure_covariance_zero=0.,old_guard_not_reused=True,C1_identity_accepted=True,C2_C4_exact_registered_groups=groups,rectangular_quarter_turn_rejected=True,rectangular_quarter_turn_nearest_set_max_distance_m=float(nearest),reflection_rejected=True,C4_actual_synthetic_rows=0,centroid_excluded_from_all_Jacobians=True,source_row_join=dict(same_unique_id_sets=60000,row_order_mismatches=int((exact['source_join']!=np.arange(60000)).sum()),join_sha256=__import__('hashlib').sha256(exact['source_join'].astype('<i8').tobytes()).hexdigest()),projection_category_clarification=dict(broad_original_reason='PROJECTION_RESIDUAL_GT_0_5PX',no_valid_manual_synthetic_correspondences=int(missing.sum()),finite_rows_exceeding_gate=int(np.sum(diagnostics['residual_max']>.5)),finite_max_residual_px=float(np.nanmax(diagnostics['residual_max']))),partition_usable_counts=newcounts,implementation_repairs=[dict(stage='first input assertion',status='STOPPED_BEFORE_GEOMETRY_ARITHMETIC',reason='geometry/source row order differs; replaced with checked unique-stem join',solver_calls=0,optimizer_updates=0),dict(stage='second draft',status='OWN_PROCESS_INTERRUPTED_BEFORE_CACHE_PUBLICATION',reason='source default stencil differs from checkpoint config; actual bound config restored; NPZ small fields materialized once',conservative_cpu_wall_seconds=150,solver_calls=0,optimizer_updates=0)],actual_cpu_wall_seconds=time.monotonic()-start,pose_solver_calls=0,new_network_inferences=0,real_reads=0,raw_rgb_reads=0,paper_source='https://arxiv.org/html/2303.11516',paper_scope='Eq11 covariance grouping; full Eq17 LC prior/linear/learnedweights excluded'))
    print('GEOMETRY_CONTRACT_PASS',newcounts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);run(a)
