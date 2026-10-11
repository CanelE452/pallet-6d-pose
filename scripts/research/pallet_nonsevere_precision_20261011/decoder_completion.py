"""Complete persisted decoder scalars and audit already captured TRAIN logits.

The original audit's full arithmetic rows are reused. Its diagnostic ramp used
an incorrect tensor rank; this supplement tests the actual required rank without
repeating inference, fitting, pose solving, or full-dataset target arithmetic.
"""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import decoder_audit as D
import geometry as G
import precision_model as M

def run(a):
    start=time.monotonic();root=Path(a.source_root);doc=Path(a.public_dir);private=Path(a.private_dir)
    candidates=list(doc.glob('DECODER_FRAME_SCALARS*.jsonl.gz'));complete=[]
    for p in candidates:
        with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
        if len(rows)==60000:complete.append((p,rows))
    assert len(complete)==1;path,records=complete[0]
    with gzip.open(doc/'DECODER_PROBABILITY_SCALARS.jsonl.gz','rt') as f:network=[json.loads(x) for x in f]
    capture=np.load(private/'TRAIN_DECODER_LOGITS.npz');c={k:capture[k] for k in capture.files}
    G.write(doc/'DECODER_COMPLETION_LOCK.json',dict(status='LOCKED_BEFORE_COMPLETION_AUDIT',code_sha256=G.sha(__file__),source_audit_sha256=G.sha(D.__file__),full_target_rows_file=path.name,full_target_rows_sha256=G.sha(path),actual_train_capture_sha256=G.sha(private/'TRAIN_DECODER_LOGITS.npz'),actual_train_rows=512,actual_train_updates=0,repair='source sample requires positions[B,K,C,N,2], ramp diagnostic rank restored',no_new_inference=True))
    source,arrays,side,exact=G.inputs(root);_,refiner,_,Base,targets,decode,_=M.source_modules(root)
    config=json.loads((root/'_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/PARAMETER_BUDGET_LOCK.json').read_text())['config'];base=Base(**config)
    sample=G.load_module(root/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py','actual_source_sampler').sample
    feature=torch.arange(8,dtype=torch.float64)[None,None,None,:].expand(1,1,4,8).clone()
    positions=torch.tensor([[[[[12.,12.],[16.,12.],[20.,12.]]]]],dtype=torch.float64)
    got=sample(feature,positions,torch.tensor([[32.,64.]],dtype=torch.float64),8).reshape(-1);expected=torch.tensor([1.,1.5,2.],dtype=torch.float64)
    nativepos=torch.tensor([[[[1.,1.],[1.5,1.],[2.,1.]]]],dtype=torch.float64);grid=2*(nativepos+.5)/nativepos.new_tensor([8,4])-1
    ngot=F.grid_sample(feature,grid,align_corners=False).reshape(-1)
    half=dict(source_feature_center_max_error=float((got-expected).abs().max()),native_integer_center_max_error=float((ngot-expected).abs().max()),source_feature_centers='(index+0.5)*stride',native_pixel_centers='index with +0.5 only when converted to align_corners=False normalized grid',global_coordinate_half_pixel_shift=False,native_tile_resize=False)
    assert half['source_feature_center_max_error']<1e-12 and half['native_integer_center_max_error']<1e-12
    results=[];gradient=None;geodata=np.load(private/'GEOMETRY_CACHE.npz');geo={k:geodata[k] for k in geodata.files}
    for lo in range(0,512,32):
        rows=c['rows'][lo:lo+32];out=G.fixed_outputs(arrays,rows,base);out['logits']=torch.tensor(c['logits'][lo:lo+32]);out['point_support']=torch.tensor(c['point_support'][lo:lo+32]).bool()
        gt=torch.tensor(np.array(arrays['gt_points'][rows]),dtype=torch.float32);gv=torch.tensor(np.array(arrays['gt_valid'][rows])).bool()
        gt,gv,branch,_=refiner.local_phase(out['points_raw'],out['point_valid'],gt,gv,torch.tensor(side['permutations'][rows]).long(),torch.tensor(side['group_valid'][rows]).bool(),out['box_diagonal'])
        gain,offset,pad,hw=G.metadata(source,arrays,rows);g=torch.tensor(gain,dtype=torch.float32);mask,target=M.aux_mask(out,gt,gv,g,torch.tensor(hw),targets)
        assert np.array_equal(geo['branch'][rows],branch.numpy()) and np.array_equal(geo['eligible'][rows],mask.numpy())
        if gradient is None:gradient=D.logit_gradient(out,gt,g,mask,decode)
        q=decode(out,1,1,None);cap=torch.tensor(np.linalg.norm(hw,axis=-1)*.01,dtype=torch.float32);qc=decode(out,1,1,cap*g);prob=out['logits'].softmax(-1)
        for j,row in enumerate(rows):
            m=target['support'][j].numpy();err=((q[j,:8]-gt[j,:8])/gain[j]).norm(dim=-1).numpy();initial=((out['points_raw'][j,:8]-gt[j,:8])/gain[j]).norm(dim=-1).numpy();move=((q[j,:8]-out['points_raw'][j,:8])/gain[j]).norm(dim=-1).numpy();cmove=((qc[j,:8]-out['points_raw'][j,:8])/gain[j]).norm(dim=-1).numpy()
            r=np.where(mask[j].numpy()[:,None],((q[j,:8]-gt[j,:8])/gain[j]).numpy(),0.)
            sensitivity=float(np.linalg.norm((geo['A24'][row]*r.reshape(1,16)).reshape(8,48),axis=-1).mean()) if geo['valid'][row] else None
            results.append(dict(id=source['records'][int(arrays['record_index'][row])]['id'],partition='train',seed=1,supported_corners=int(m.sum()),eligible_corners=int(mask[j].sum()),mean_expected_move_native_px=float(move[m].mean()) if m.any() else None,mean_capped_move_native_px=float(cmove[m].mean()) if m.any() else None,mean_expected_error_native_px=float(err[m].mean()) if m.any() else None,mean_initial_error_native_px=float(initial[m].mean()) if m.any() else None,mean_null_probability=float(prob[j,:,-1].numpy()[m].mean()) if m.any() else None,local_covariance_sensitivity_m=sensitivity))
    with gzip.open(doc/'DECODER_TRAIN_PROBABILITY_SCALARS.jsonl.gz','xt') as f:
        for r in results:f.write(json.dumps(r,allow_nan=False)+'\n')
    # Reproduce the motivating small-displacement check with actual source target.
    examples=[]
    for desired in (1.,1.7,3.):
        p=torch.full((1,9,2),100.,dtype=torch.float32);gt=p.clone();gt[:,:8,0]+=desired
        out=dict(points_raw=p,point_support=torch.ones((1,8),dtype=torch.bool),box_diagonal=torch.tensor([340.]),candidate_displacements=340*base.displacements[None],logits=torch.zeros((1,8,222)))
        target=targets(out,gt,torch.ones((1,9),dtype=torch.bool));out['logits']=target['distribution'].clamp_min(torch.finfo(torch.float32).tiny).log();q=decode(out,1,1,None)
        examples.append(dict(box_diagonal_native_px=340.,desired_dx_native_px=desired,expected_dx_native_px=float(q[0,0,0]-p[0,0,0]),expected_dy_native_px=float(q[0,0,1]-p[0,0,1]),temperature=1.,gain=1.,cap=None))
    def agg(rs,key):return G.summary([r[key] for r in rs if r.get(key) is not None])
    fields=[k for k in records[0] if k.endswith('_px') or k.endswith('_m')]
    summary={p:{k:agg([r for r in records if r['partition']==p],k) for k in fields} for p in ('train','calibration','selection','heldout')}
    nfields=['mean_expected_move_native_px','mean_capped_move_native_px','mean_expected_error_native_px','mean_initial_error_native_px','mean_probability_entropy','mean_null_probability']
    nsummary={f'seed{s}_{p}':{k:agg([r for r in network if r['seed']==s and r['partition']==p],k) for k in nfields} for s in (1,2) for p in ('calibration','selection','heldout')}
    nsummary['seed1_actual_train_first32_minibatches']={k:agg(results,k) for k in nfields if k!='mean_probability_entropy'}
    corr=[r for r in records if r['target_covariance_sensitivity_m'] is not None and r['desired_native_displacement_mean_px'] is not None]
    report=dict(status='PASS',actual_source_operator='GenericPointRefiner.targets/decode/sample',target_rows=60000,complete_target_rows_artifact=path.name,target_summary=summary,actual_network_summary=nsummary,actual_validation_probability_rows=len(network),actual_train_probability_rows=len(results),actual_train_source='ROOT first32 common seed1 TRAIN minibatches before any update, existing calibration forward capture',corner_counts=dict(supported=sum(r['supported_corners'] for r in records),auxiliary_eligible=sum(r['eligible_corners'] for r in records),outside_hull=sum(r['outside_candidate_hull_corners'] for r in records),beyond_native_cap=sum(r['beyond_native_cap_corners'] for r in records)),half_pixel=half,actual_train_logit_coordinate_gradient=gradient,source_target_small_displacement_examples=examples,same_frame_desired_displacement_vs_local_covariance_correlation=D.correlation([r['desired_native_displacement_mean_px'] for r in corr],[r['target_covariance_sensitivity_m'] for r in corr]),sensitivity_definition='exact renderer local covariance surrogate, not measured PnP pose error',global_1_33_scaling=False,loss_beta_native_px=1.,loss_temperature=1.,loss_gain_lambda=1.,auxiliary_expected_decode_cap=None,final_native_cap_fraction=.01,whole_object_branch_and_mask_cache_match=True,geometry_detached=True,centroid8_fixed=True,raw_rgb_reads=0,pose_solver_calls=0,new_network_inferences=0,optimizer_updates=0,real_reads=0,diagnostic_repairs='Original complete rows retained; original ramp tensor-rank error corrected by this supplement. Earlier nonfinite scalar and NPZ repeated-read drafts remain recorded as implementation failures, without any model/configuration sweep.',paper_source='https://arxiv.org/html/2303.11516',paper_scope='fixed pure E_cov Eq11 mean8 normalization; full Eq17 LC not claimed',actual_cpu_wall_seconds=time.monotonic()-start)
    G.write(doc/'DECODER_AUDIT.json',report);print('DECODER_FULL_PASS',report['corner_counts'],examples,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);run(a)
