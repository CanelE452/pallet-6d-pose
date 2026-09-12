"""Lock sources, then test the fixed teacher mechanism before any student fit."""
import argparse
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import read_json, sha256, immutable_json, canonical_sha
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from .geometry import incidence_loss, line_weights, baseline_assignment, gt_point_loss

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'scripts/research/pallet_dht_pseudoline_selftrain_v1'
DOC=ROOT/'_docs/experiments/pallet_dht_pseudoline_selftrain_v1'
RAW=ROOT/'data/pallet/results/pallet_dht_pseudoline_selftrain_v1'
POINT=ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
DHT=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction/heads/hough_seed1/checkpoint_final.pt'
EXPORT=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v1/export'
SOURCES={'calibration':ROOT/'data/pallet/results/pallet_hough_gain_selector_v1/calibration_Q.json',
         'synth_val':ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction/predictions/hough_seed1_synth_val.json'}
POOL=ROOT/'data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv'
PSEUDO=ROOT/'challenge/yolo_pose_one_model/datasets/paper_selftrain_v3/V3A_TRUE_IGNORE'


def save(name,value):immutable_json(DOC/name,value)


def source_inventory():
    files=[POINT,DHT,POOL,ROOT/'data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json',*SOURCES.values()]
    for name in ('pallet_symmetry_dht_local_v1','pallet_symmetry_dht_local_v2','pallet_symmetry_dht_local_v2_wls_correction','pallet_hough_gain_selector_v1'):
        files+=list((ROOT/'_docs/experiments'/name).rglob('*'))
    files+=list((ROOT/'_docs/paper/final').rglob('*'))
    for version in (1,2,3,4,5):
        files+=list((ROOT/f'data/pallet/results/paper_selftrain_v{version}').rglob('*'))
        files+=list((ROOT/f'challenge/yolo_pose_one_model/paper_selftrain_v{version}').rglob('*'))
    files+=list((PSEUDO/'labels/train').glob('pseudo__*.txt'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files)) if p.is_file()}


def provenance():
    rows=list(csv.DictReader(POOL.open()));assert len(rows)==1000
    for r in rows:assert sha256(ROOT/r['image_path'])==r['image_sha256']
    poolsha={r['image_sha256'] for r in rows};poolname={Path(r['image_path']).name for r in rows}
    population=[];meta=[]
    for name in ('PAPER_EVAL_ALL_POS','DEV_NEG2689'):
        path=ROOT/f'challenge/real_gt_v2/manifests/{name}.json';m=read_json(path)
        # Membership metadata only. Never dereference gt_v2_path.
        members=[ROOT/r.get('image_path',r.get('image','')) for r in m['items']]
        hashes={sha256(p) for p in members};names={p.name for p in members}
        assert not poolsha&hashes and not poolname&names
        population.append(dict(name=name,count=len(members),manifest_sha256=sha256(path),
            image_hash_intersection=0,filename_intersection=0,GT_annotations_opened=0))
        meta.extend(m['items'])
    labels=sorted((PSEUDO/'labels/train').glob('pseudo__*.txt'))
    label_hashes={str(p.relative_to(ROOT)):sha256(p) for p in labels}
    pool_paths={str((ROOT/r['image_path']).resolve()) for r in rows}
    used=[]
    for p in labels:
        image=PSEUDO/'images/train'/p.with_suffix('.png').name
        assert str(image.resolve()) in pool_paths
        used.append(str(image.resolve().relative_to(ROOT)))
    assert len(labels)==273
    teacher_cache=read_json(ROOT/'data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json')
    assert teacher_cache['teacher_sha256']==sha256(POINT)
    assert all(not any('gt' in key.lower() or 'annotation' in key.lower() for key in r) for r in teacher_cache['entries'])
    save('DATA_PROVENANCE.json',dict(pool=str(POOL.relative_to(ROOT)),pool_sha256=sha256(POOL),pool_count=len(rows),
        sessions=dict(Counter(r['capture_session'] for r in rows)),conditions=dict(Counter(r['paper_condition'] for r in rows)),
        all_pool_image_hashes_verified=True,pool_members=rows,evaluation_overlap_audit=population,
        untouched_eval_provenance_established=False,evidence_level='POSTHOC_DEVELOPMENT_ONLY',real_GT_annotation_access_count=0))
    save('PSEUDO_POINT_CACHE_AUDIT.json',dict(source=str(PSEUDO.relative_to(ROOT)),unique_frame_count=len(labels),
        labels_sha256=label_hashes,canonical_cache_digest=canonical_sha(label_hashes),point_teacher_sha256=sha256(POINT),
        C1_source=str(PSEUDO.relative_to(ROOT)),C2_source=str(PSEUDO.relative_to(ROOT)),
        C1_C2_byte_identical=True,frame_paths=used,selection='existing V3A true-ignore, unchanged labels; no new filter',
        real_GT_annotation_access_count=0,status='EXISTING_CACHE_VERIFIED_NOT_YET_USED_FOR_TRAINING'))


def lock():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
    assert sha256(POINT)=='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor','6f5db05e2b4de249f21a6e3aad5262c4f094b585','HEAD'],cwd=ROOT,check=True)
    DOC.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    save('PROTOCOL_LOCK.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),start_main=head,
        start_origin_main=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip(),branch='main',initial_worktree_clean=True,
        required_ancestor='6f5db05e2b4de249f21a6e3aad5262c4f094b585',point_sha256=sha256(POINT),DHT_sha256=sha256(DHT),
        weight='clamp(1-ambiguity,0,1); no mode_mass; unavailable/nonfinite/degenerate ->0',line_beta_normalized=.01,
        gate_population='synth_val512',calibration_role='diagnostic only before potential gradient calibration',
        gate=dict(mean_output_gradient_cosine_gt=0.,DHT_weighted_mean_error_less_than_P=True,better_edge_bootstrap_lower_gt=.5,
                  bootstrap_unit='frame',bootstrap_draws=4096,bootstrap_seed=1701,CI=[.025,.975]),
        virtual_step_px=.001,seeds=[1,2,3],conditional_student_updates_per_fit=900,student_fits_if_gate_pass=9,
        lambda_target_ratio=.25,lambda_clamp=[0,1000],lambda_parameter_calibration='first8 synthetic calibration frames, only after Stage A PASS',
        forbidden=['GT in real training','Q pseudo corners','line intersections as corners','teacher training','weight/threshold/lambda sweep','FINAL','paper/final edits'],
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (PACKAGE/'geometry.py',PACKAGE/'audit_mechanism.py',DOC/'METHOD_LOCK.md',DOC/'PURPOSE_AND_SCOPE.md')},
        predictions_sha256={k:sha256(p) for k,p in SOURCES.items()},evidence_level='POSTHOC_DEVELOPMENT_ONLY'))
    save('PRESERVED_SOURCE_SHA.json',source_inventory())
    provenance()
    save('TEACHER_CHECKPOINTS.json',dict(Point=dict(path=str(POINT.relative_to(ROOT)),sha256=sha256(POINT),architecture='ultralytics.nn.tasks.PoseModel / Pose26',kpt_shape=[9,3]),
        DHT=dict(path=str(DHT.relative_to(ROOT)),sha256=sha256(DHT),seed=1,selection='first numeric seed, no seed performance comparison'),
        student_init=str(POINT.relative_to(ROOT)),student_inference='stock YOLO26n pose, no DHT',student_parameters=3043704,
        real_GT_annotation_access_count=0))
    print('PROTOCOL, TEACHERS, POOL AND PSEUDO CACHE LOCKED',flush=True)


def line_from_points(p):
    v=p[1]-p[0];norm=np.linalg.norm(v)
    if norm<=1e-8:return None
    n=np.array([-v[1],v[0]])/norm
    return np.r_[n,-n@p[0]]


def summary_edges(rows):
    if not rows:return {'N':0}
    w=np.array([r['weight'] for r in rows]);p=np.array([r['P_error_px'] for r in rows]);d=np.array([r['DHT_error_px'] for r in rows])
    summary=lambda a:dict(mean=float(np.mean(a)),median=float(np.median(a)),p90=float(np.percentile(a,90)))
    return dict(N=len(rows),weight_sum=float(w.sum()),P_line=summary(p),DHT_line=summary(d),
        weighted_P_mean_px=float(np.sum(w*p)/w.sum()),weighted_DHT_mean_px=float(np.sum(w*d)/w.sum()),
        better_edge_fraction=float((d<p).mean()),weighted_better_edge_fraction=float(np.sum(w*(d<p))/w.sum()),
        P_angle_deg=summary([r['P_angle_deg'] for r in rows]),DHT_angle_deg=summary([r['DHT_angle_deg'] for r in rows]),
        mean_DHT_minus_P_px=float(np.mean(d-p)))


def mechanism(device_name):
    protocol=read_json(DOC/'PROTOCOL_LOCK.json')
    correction=read_json(DOC/'IMPLEMENTATION_CORRECTION.json') if (DOC/'IMPLEMENTATION_CORRECTION.json').exists() else {'files':{}}
    for path,digest in protocol['source_hashes'].items():
        expected=digest
        if path in correction['files']:
            assert correction['files'][path]['before']==digest
            expected=correction['files'][path]['after']
        assert sha256(ROOT/path)==expected
    device=torch.device(device_name)
    if device.type=='cuda':assert torch.cuda.is_available(),'CUDA requested; no CPU fallback'
    outputs={};quality={};all_edge_rows={};all_frames={};line_cache={}
    for split,path in SOURCES.items():
        pred=read_json(path);assert pred['GT_opened'] is False and pred['checkpoint_sha256']==sha256(DHT)
        saved={r['frame_id']:r for r in pred['records']}
        data=ObservationDataset(EXPORT/f'{split}.json',targets=True);edges=[];frames=[];point_errors=[]
        weights_all=[];available_count=0;normal_good=normal_bad=0;line_records=[]
        for i,item in enumerate(data):
            row=saved[item['frame_id']];p=item['base_points'].double().to(device);D=item['image_hw'].double().norm().to(device)
            y,yv,choice=baseline_assignment(p,item['target_points'].double().to(device),item['target_valid'].to(device),
                    item['symmetry_permutations'].to(device),D,item['point_valid'].to(device))
            lines=torch.tensor(row['raw_line'],dtype=torch.float64,device=device);ambiguity=torch.tensor(row['ambiguity'],dtype=torch.float64,device=device)
            available=torch.tensor(row['utility'],device=device)>0
            weight=line_weights(lines,ambiguity,available);weights_all.extend(weight.cpu().tolist());available_count+=int(available.sum())
            variable=p.detach().clone().requires_grad_()
            ll=incidence_loss(variable[None],lines[None],weight[None],D[None]).sum()
            mask=yv[:8]&item['point_valid'][:8].to(device)
            err=(variable[:8]-y[:8]).norm(dim=-1)
            # Missing point penalty is constant, as in the baseline metric contract.
            full=torch.where(item['point_valid'][:8].to(device),err,torch.ones_like(err)*D)
            gl=gt_point_loss(variable,y,yv,item['point_valid'].to(device),D)
            grad_line=torch.autograd.grad(ll,variable)[0];grad_gt=torch.autograd.grad(gl,variable)[0]
            assert torch.isfinite(grad_line).all() and torch.isfinite(grad_gt).all()
            nl=grad_line.norm();ng=grad_gt.norm();cos=float((grad_line*grad_gt).sum()/(nl*ng)) if nl>0 and ng>0 else 0.
            direction=-grad_line/nl.clamp_min(1e-30);step=p+.001*direction
            derivative=float((grad_gt*direction).sum())
            new_error=(step[:8]-y[:8]).norm(dim=-1);new_full=torch.where(item['point_valid'][:8].to(device),new_error,torch.ones_like(new_error)*D)
            after=float(new_full[yv[:8]].mean()/D) if yv[:8].any() else 0.
            baseline=float(full[yv[:8]].mean()) if yv[:8].any() else float(D)
            point_errors.extend(full[yv[:8]].detach().cpu().tolist() if yv[:8].any() else [float(D)]*8)
            meta=data.records[i];fr=dict(frame_id=item['frame_id'],gradient_cosine=cos,gradient_line_norm=float(nl),gradient_GT_norm=float(ng),
                zero_gradient=bool(nl==0 or ng==0),directional_derivative=derivative,virtual_GT_loss_change=after-float(gl),
                P_frame_mean_px=baseline,P_normalized=baseline/float(D),source=meta['source'],asset=meta['asset'],symmetry=f'C{len(meta["symmetry_permutations"])}',
                shared_symmetry_choice=choice,line_loss=float(ll),line_weight_sum=float(weight.sum()))
            frames.append(fr)
            pn=p.cpu().numpy();yn=y.cpu().numpy();ln=lines.cpu().numpy();wn=weight.cpu().numpy();valid=yv.cpu().numpy();pv=item['point_valid'].numpy()
            for e,(a,b) in enumerate(EDGES):
                if not (valid[[a,b]].all() and pv[[a,b]].all() and wn[e]>0):continue
                pl=line_from_points(pn[[a,b]]);gt=line_from_points(yn[[a,b]])
                if pl is None or gt is None:continue
                dl=ln[e]/np.linalg.norm(ln[e,:2]);ge=yn[[a,b]]
                pe=float(np.abs(ge@pl[:2]+pl[2]).mean());de=float(np.abs(ge@dl[:2]+dl[2]).mean())
                angle=lambda line:float(np.degrees(np.arccos(np.clip(abs(line[:2]@gt[:2]),0,1))))
                residual=pn[[a,b]]@dl[:2]+dl[2];normal_error=(pn[[a,b]]-ge)@dl[:2]
                normal_good+=int((normal_error*residual>0).sum());normal_bad+=int((normal_error*residual<0).sum())
                edges.append(dict(frame_id=item['frame_id'],edge=e,endpoints=[a,b],weight=float(wn[e]),ambiguity=float(ambiguity[e]),
                    P_error_px=pe,DHT_error_px=de,P_angle_deg=angle(pl),DHT_angle_deg=angle(dl),
                    P_endpoint_to_GT_line_px=float(np.abs(pn[[a,b]]@gt[:2]+gt[2]).mean()),
                    normal_error_component_px=float(np.abs(normal_error).mean()),normal_descent_help_count=int((normal_error*residual>0).sum()),
                    normal_descent_harm_count=int((normal_error*residual<0).sum()),
                    source=meta['source'],asset=meta['asset'],symmetry=fr['symmetry'],P_frame_mean_px=baseline))
            line_records.append(dict(frame_id=item['frame_id'],raw_line=row['raw_line'],ambiguity=row['ambiguity'],weight=wn.tolist(),
                available=available.cpu().tolist(),roles=[list(e) for e in EDGES],source_checkpoint_sha256=sha256(DHT)))
        counts={r['frame_id']:[0,0] for r in frames}
        for r in edges:counts[r['frame_id']][0]+=r['DHT_error_px']<r['P_error_px'];counts[r['frame_id']][1]+=1
        arr=np.array(list(counts.values()));rng=np.random.default_rng(1701);draw=rng.integers(0,len(arr),(4096,len(arr)))
        summed=arr[draw].sum(1);boot=summed[:,0]/np.maximum(summed[:,1],1);ci=np.quantile(boot,[.025,.975])
        s=summary_edges(edges);s.update(frame_N=len(frames),mean_gradient_cosine=float(np.mean([r['gradient_cosine'] for r in frames])),
            positive_alignment_fraction=float(np.mean([r['gradient_cosine']>0 for r in frames])),zero_gradient_count=sum(r['zero_gradient'] for r in frames),
            mean_directional_derivative=float(np.mean([r['directional_derivative'] for r in frames])),
            virtual_step_GT_loss_decreased_fraction=float(np.mean([r['virtual_GT_loss_change']<0 for r in frames])),
            better_edge_fraction_frame_bootstrap_CI95=ci.tolist())
        checks=dict(positive_mean_alignment=s['mean_gradient_cosine']>0,weighted_line_error_lower=s['weighted_DHT_mean_px']<s['weighted_P_mean_px'],better_edge_CI_lower_gt_half=bool(ci[0]>.5))
        s['gate_checks']=checks;s['PASS']=all(checks.values());outputs[split]=s
        quality[split]=dict(Point=dict(N_frames=len(frames),N_keypoints=len(point_errors),median_px=float(np.median(point_errors)),p90_px=float(np.percentile(point_errors,90)),
                gross20_rate=float(np.mean(np.array(point_errors)>20)),frame_mean_normalized=float(np.mean([r['P_normalized'] for r in frames]))),
            DHT_line=s,available_edges=available_count,total_edges=len(frames)*12,line_weight=dict(mean=float(np.mean(weights_all)),median=float(np.median(weights_all)),p90=float(np.percentile(weights_all,90))),
            normal_help_fraction=normal_good/max(normal_good+normal_bad,1),normal_harm_fraction=normal_bad/max(normal_good+normal_bad,1),
            normal_endpoint_N=normal_good+normal_bad,physical_visibility_claim=False)
        all_edge_rows[split]=edges;all_frames[split]=frames
        immutable_json(RAW/f'{split}_pseudo_lines.json',dict(GT_opened=False,source_prediction_sha256=sha256(path),records=line_records))
        line_cache[split]=dict(N=len(line_records),sha256=sha256(RAW/f'{split}_pseudo_lines.json'),source_sha256=sha256(path),GT_in_cache=False)
        print(json.dumps({split:s},indent=2),flush=True)
    status='DHT_PSEUDOLINE_MECHANISM_PASS' if outputs['synth_val']['PASS'] else 'DHT_PSEUDOLINE_MECHANISM_FAIL'
    save('LINE_MECHANISM_AUDIT.json',dict(status=status,gate_population='synth_val512',device=str(device),optimizer_updates=0,
        derivative_space='raw student-output coordinates at cached R0, not student parameter space',splits=outputs,frame_records=all_frames,edge_records=all_edge_rows))
    save('PSEUDO_SUPERVISION_QUALITY.json',quality)
    save('PSEUDO_LINE_CACHE_AUDIT.json',dict(synthetic=line_cache,real_cache_created=False,real_GT_annotation_access_count=0,
        source_checkpoint_sha256=sha256(DHT),weight_definition=protocol['weight'],Q_used_as_pseudo_keypoints=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['lock','mechanism']);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    lock() if args.phase=='lock' else mechanism(args.device)
