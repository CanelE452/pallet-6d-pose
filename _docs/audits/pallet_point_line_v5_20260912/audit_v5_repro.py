"""Read-only source audit on generated fixtures. No pallet training or inference.

Usage: python audit_v5_repro.py --source /path/to/pallet_point_line_v5 --output /new/output
Original kit files are not patched. Results are counterexample/contract diagnostics,
not measured prevalence or claims about an unavailable actual v5 experiment.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import replace, asdict


def main(source: Path, output: Path) -> None:
    sys.path.insert(0, str(source.resolve()))
    import numpy as np
    import torch
    from plpose_v5.fixtures import geometry_fixture
    from plpose_v5.geometry import roi_affine, cuboid
    from plpose_v5.hough import Lattice, decode_modes
    from plpose_v5.solver import PointLineRefiner, SolverConfig
    from plpose_v5.contracts import Observation, Supervision, SymmetrySpec
    from plpose_v5.objective import compute_loss, LossConfig
    from plpose_v5.io import write_json, sha256, SCHEMA, audit_manifests
    from plpose_v5.assessment import compare, evaluate
    torch.set_num_threads(2)
    torch.manual_seed(20260912)
    if output.exists():
        raise ValueError('Use a fresh output directory; original results are never overwritten.')
    output.mkdir(parents=True)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'plpose_v5').glob('*.py')}
    d,K,R,t,p,m = geometry_fixture(dtype=torch.float64)
    box = torch.cat((p[:,:8].amin(1)-20,p[:,:8].amax(1)+20),-1)
    L = Lattice().double()
    content = torch.ones(1,1,24,24,dtype=torch.bool)
    _, mass = L(torch.ones(1,1,24,24,dtype=torch.float64),content)
    valid = (mass.flatten(2)>1e-6).expand(-1,12,-1)
    logits = torch.zeros(1,12,L.T*L.R,dtype=torch.float64)
    lines, lp, lv = decode_modes(logits,valid,L,roi_affine(box),3)
    mm = replace(m,lines=lines,line_logprob=lp,line_valid=lv,
                 line_sigma=torch.full((1,12,3),4.,dtype=torch.float64),
                 point_sigma=torch.full((1,9),4.,dtype=torch.float64))
    decoded = {}
    for name,weight in [('point_only',0.),('point_plus_flat_lines',.25)]:
        result=PointLineRefiner(SolverConfig(iterations=8,line_weight=weight))(mm,d,K,R[:,None],t[:,None])
        decoded[name]={'mean8_px':float(torch.linalg.vector_norm(result['points'][:,:8]-p[:,:8],dim=-1).mean()),
                       'translation_error_m':float(torch.linalg.vector_norm(result['t']-t)),
                       'pose_valid':bool(result['pose_valid'][0]),
                       'initial_energy':float(result['objective_trace'][0,0,0]),
                       'final_energy':float(result['objective_trace'][0,0,-1])}
    issue1={'type':'CONTROLLED_INFORMATION_FAILURE','scope':'one generated cuboid; exact initial pose and points',
            'description':'A task-uninformative uniform line score is reduced to 3 valid modes per edge and treated as usable evidence.',
            'valid_bins_per_edge':int(valid[0,0].sum()),'valid_retained_modes':int(lv.sum()),
            'mode_weights_first_edge':lp[0,0].exp().tolist(),'comparison':decoded,
            'reproduced':decoded['point_only']['mean8_px']<1e-9 and decoded['point_plus_flat_lines']['mean8_px']>1,
            'limit':'Does not prove this occurs frequently in a trained model. Not a real-image performance number.'}
    obs=Observation(torch.zeros(1,32,24,24,dtype=torch.float64),content,box,K,d,torch.tensor([[480.,640.]],dtype=torch.float64))
    gt=Supervision(p,torch.ones(1,9,dtype=torch.bool),R,t,[SymmetrySpec(1,False,'generated contract diagnostic')])
    all_R=R[:,None].repeat(1,2,1,1)
    all_t=t[:,None].repeat(1,2,1);all_t[:,0,0]+=.2
    energy=torch.tensor([[5.,0.]],dtype=torch.float64,requires_grad=True)
    out={'all_R':all_R,'all_t':all_t,'seed_R':all_R,'seed_t':all_t,
         'energy':energy,'measurements':m,'line_logits':logits,
         'line_bin_valid':torch.zeros_like(valid),'raw_to_grid':roi_affine(box),
         'candidates_valid':torch.tensor([[True,False]])}
    cfg=LossConfig(point=0.,line=0.,pose=1.,seed=0.,ranking=0.)
    loss_invalid,_=compute_loss(out,obs,gt,L.lines,cfg)
    out_valid=dict(out,candidates_valid=torch.tensor([[True,True]]))
    loss_valid,_=compute_loss(out_valid,obs,gt,L.lines,cfg)
    masked_index=int(torch.where(out['candidates_valid'],energy,torch.full_like(energy,1e9)).argmin())
    grad=torch.autograd.grad(loss_invalid,energy)[0]
    issue2={'type':'TRAIN_INFERENCE_CONTRACT_GAP','description':'compute_loss ignores candidates_valid while inference selects only valid candidates.',
            'controlled_hypotheses':'candidate0 offset by0.2m and valid; candidate1 exact pose but explicitly flagged ineligible',
            'training_probability_on_ineligible':float(torch.softmax(-energy,dim=-1)[0,1].detach()),
            'loss_with_ineligible_candidate':float(loss_invalid.detach()),
            'loss_with_both_eligible':float(loss_valid.detach()),'inference_selected':masked_index,
            'selected_translation_error_m':float(torch.linalg.vector_norm(all_t[0,masked_index]-t[0])),
            'energy_gradient':grad.tolist(),'reproduced':float(loss_invalid.detach())==float(loss_valid.detach()),
            'limit':'Direct output-contract fixture, not a claim these hypotheses came from a trained solver. Invalid seed rescue needs a separate, explicit objective.'}
    def evaluation_doc(seed,manifest_sha,arm,err):
        return {'seed':seed,'arm':arm,'manifest_sha256':manifest_sha,'gt_corners':8,
                'valid_pose_frames':1,'symmetric_error8_px':{'median':err,'p90':err},
                'records':[{'id':'same_printed_id','session':'session_a','fixed_gt_mask8':[True]*8,
                  'primary_capped_normalized_corner_mean':err/100.,'fixed_error8_px':[err]*8,
                  'symmetric_frame_mean_px':err}]}
    refs=[];methods=[]
    for seed in (1,2):
        for arm,err,paths in [('point',20.,refs),('hough',10.,methods)]:
            path=output/f'{arm}_seed{seed}_different_manifest.json'
            write_json(path,evaluation_doc(seed,str(seed)*64,arm,err));paths.append(path)
    result=compare(refs,methods,output/'mixed_manifest_comparison.json')
    issue3={'type':'EVALUATION_BINDING_GAP','description':'compare accepts different manifest hashes across seeds when pairs/printed IDs agree.',
            'per_seed_manifest_sha':['1'*64,'2'*64],'accepted':True,'reported_seed_count':len(result['seeds']),
            'reported_difference':result['paired_session_interval']['difference'],'reproduced':len(result['seeds'])==2,
            'limit':'This constructs inconsistent evaluation fixtures. It is not evidence any v4/v5 run mixed populations.'}
    data=output/'input_contract_fixture';data.mkdir();obsfile=data/'obs.pt';gtfile=data/'gt.pt'
    torch.save(vars(obs),obsfile)
    gtdata={'points':p,'valid':gt.valid,'R':R,'t':torch.full_like(t,float('nan'))};torch.save(gtdata,gtfile)
    record={'id':'nan_t_fixture','session':'generated_only','observation':'obs.pt','supervision':'gt.pt',
            'observation_sha256':sha256(obsfile),'supervision_sha256':sha256(gtfile),
            'source_image_sha256':'a'*64,'symmetry':asdict(gt.specs[0])}
    manifest=data/'manifest.json'
    write_json(manifest,{'schema':SCHEMA,'coordinate_contract':'object_x_width_y_up_z_depth_v1',
                         'split':'generated','population_scope':'GENERATED_CONTRACT_ONLY','records':[record]})
    audit=audit_manifests([manifest],output/'nan_translation_preflight.json')
    issue4={'type':'PREFLIGHT_FINITE_CHECK_GAP','description':'GT translation NaN is not rejected by audit_manifests and yields reported max projection error0.',
            'arithmetic_pass':audit['arithmetic_and_record_checks_pass'],
            'reported_max_projection_error_px':audit['records'][0]['max_projection_error_px'],'status':audit['status'],
            'reproduced':audit['arithmetic_and_record_checks_pass'],
            'limit':'compute_loss has a finite-GT-t guard and would reject subsequent training. This is a preflight guarantee defect, not proof of silent trained-model corruption.'}
    gtdata['t']=t;torch.save(gtdata,gtfile);record['supervision_sha256']=sha256(gtfile)
    write_json(manifest,{'schema':SCHEMA,'coordinate_contract':'object_x_width_y_up_z_depth_v1',
                         'split':'generated','population_scope':'GENERATED_CONTRACT_ONLY','records':[record]})
    pf=output/'malformed_pose_predictions.json'
    write_json(pf,{'manifest_sha256':sha256(manifest),'arm':'hough','seed':1,'records':[
        {'id':record['id'],'pose_valid':True,'points':p[0].tolist(),'R':(2*R[0]).tolist(),'t':t[0].tolist()}]})
    ev=evaluate(manifest,pf,output/'malformed_pose_evaluation.json')
    issue5={'type':'OUTPUT_CONSISTENCY_CHECK_GAP','description':'Evaluator accepts a non-rotation2R paired with copied exact2D points, and reports primary0/rotation0/translation0.',
            'determinant':float(torch.linalg.det(2*R[0])),'reported_primary':ev['primary'],
            'reported_rotation_deg':ev['rotation_sym_deg']['mean'],'reported_translation_m':ev['translation_m']['mean'],
            'reported_valid_frames':ev['valid_pose_frames'],'per_frame_cuboid_add_sym_m':ev['records'][0]['cuboid_add_sym_m'],
            'reproduced':ev['primary']==0 and ev['valid_pose_frames']==1,
            'limit':'Normal model forward validates rotations. This tests evaluator robustness to corrupted/mixed serialized outputs; no such corruption was observed in actual result files.'}
    report={'executed_at_utc':datetime.now(timezone.utc).isoformat(),'audit_date_kst':'2026-09-12','scope':'GENERATED_DIAGNOSTICS_NO_PALLET_RUN',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
            'source':str(source.resolve()),'source_sha256':source_hashes,'optimizer_updates':0,'real_image_forwards':0,
            'original_sources_modified':False,'checks':[issue1,issue2,issue3,issue4,issue5]}
    assert all(r['reproduced'] for r in report['checks'])
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'plpose_v5').glob('*.py')}
    assert after==source_hashes
    write_json(output/'V5_ADVERSARIAL_AUDIT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    v=a.parse_args();main(v.source,v.output)
