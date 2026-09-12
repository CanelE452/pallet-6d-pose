"""Offline evaluation with ONE whole-object symmetry per metric family.

These are transparent development metrics, NOT a replacement for canonical
BOP/MAIN6D evaluation. The manifest's population must include intended failures;
if it only contains detected instances the report is detector-conditional.
"""
from __future__ import annotations
import argparse,math
from pathlib import Path
import numpy as np
import torch
from .io import InstanceDataset,read_json,write_json,sha256
from .geometry import cuboid
from .contracts import symmetry_permutations


def stats(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return {'n':int(a.size),'mean':float(a.mean()) if a.size else None,
            'median':float(np.median(a)) if a.size else None,'p90':float(np.quantile(a,.9)) if a.size else None}


def evaluate(manifest,predictions,output):
    ds=InstanceDataset(manifest);pred=read_json(predictions)
    if pred['manifest_sha256']!=sha256(ds.path):raise ValueError('Prediction/manifest mismatch')
    byid={r['id']:r for r in pred['records']}
    if len(byid)!=len(pred['records']) or set(byid)!={r['id'] for r in ds.records}:raise ValueError('Missing/duplicate predictions')
    rows=[];allfixed=[];allsym=[];all9=[];total=0;observed=0
    for i,record in enumerate(ds.records):
        p=byid[record['id']];obs=ds.observation(i);gt=ds.target(i)
        d=obs.dims[0];G,perms=symmetry_permutations(d,gt.specs[0]);mask=gt.valid[0]
        target=torch.where(mask[:,None],gt.points[0],torch.zeros_like(gt.points[0])).double()
        predicted=torch.tensor(p['points'],dtype=torch.float64)
        ok=bool(p['pose_valid']) and bool(torch.isfinite(predicted).all())
        n=int(mask[:8].sum());n9=int(mask.sum());total+=n
        diagonal=float(torch.linalg.vector_norm(obs.image_hw[0]))
        fixed=torch.linalg.vector_norm(predicted-target,dim=-1)
        errors=torch.linalg.vector_norm(predicted[None]-target[perms],dim=-1)
        masks=mask[perms]
        candidate=(torch.where(masks[:,:8],errors[:,:8],0.)).sum(-1)/masks[:,:8].sum(-1).clamp_min(1)
        index=int(candidate.argmin());chosen=errors[index];m=masks[index]
        if ok:
            observed+=n;allfixed+=fixed[:8][mask[:8]].tolist();allsym+=chosen[:8][m[:8]].tolist();all9+=chosen[m].tolist()
        capped=float(torch.minimum(chosen[:8][m[:8]]/diagonal,torch.ones(n)).mean()) if ok and n else (1. if n else None)
        fixed_list=[float(fixed[j]) if ok and bool(mask[j]) else None for j in range(8)]
        sym_list=[float(chosen[j]) if ok and bool(m[j]) else None for j in range(8)]
        R=torch.tensor(p['R'],dtype=torch.float64);t=torch.tensor(p['t'],dtype=torch.float64)
        X=cuboid(d.double())[:8]
        Rp=R@X.T+t[:,None]
        Rg=gt.R[0].double()[None]@G.double()
        gt3=Rg@X.T+gt.t[0].double()[None,:,None]
        add=torch.linalg.vector_norm(Rp[None]-gt3,dim=1).mean(-1)
        rrel=Rg.transpose(-1,-2)@R
        angle=torch.acos(((rrel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))*180/math.pi
        rows.append({'id':record['id'],'session':record['session'],'symmetry_order':gt.specs[0].order,
                     'gt_corners':n,'pose_valid':ok,'symmetry_choice_2d':index,
                     'fixed_error8_px':fixed_list,'symmetric_error8_px':sym_list,'fixed_gt_mask8':mask[:8].tolist(),
                     'primary_capped_normalized_corner_mean':capped,
                     'symmetric_frame_mean_px':float(chosen[:8][m[:8]].mean()) if ok and n else None,
                     'translation_m':float(torch.linalg.vector_norm(t-gt.t[0])) if ok else None,
                     'rotation_sym_deg':float(angle.min()) if ok else None,
                     'cuboid_add_sym_m':float(add.min()) if ok else None})
    values=[r['primary_capped_normalized_corner_mean'] for r in rows if r['primary_capped_normalized_corner_mean'] is not None]
    report={'schema':'plpose_v5_evaluation_1','manifest_sha256':sha256(ds.path),'prediction_sha256':sha256(predictions),
            'arm':pred['arm'],'seed':pred['seed'],'split':ds.meta['split'],
            'population_scope':ds.meta.get('population_scope','NOT_CERTIFIED_COMPLETE_DETECTION_POPULATION'),
            'frames':len(rows),'valid_pose_frames':sum(r['pose_valid'] for r in rows),'gt_corners':total,'observed_corners':observed,
            'fixed_error8_px':stats(allfixed),'symmetric_error8_px':stats(allsym),'symmetric_error9_px':stats(all9),
            'primary':float(np.mean(values)) if values else None,
            'primary_definition':'Frame mean of valid-GT 8-corner errors/RAW image diagonal capped at1; missing pose=1. Whole-object permitted symmetry only.',
            'translation_m':stats([r['translation_m'] for r in rows if r['translation_m'] is not None]),
            'rotation_sym_deg':stats([r['rotation_sym_deg'] for r in rows if r['rotation_sym_deg'] is not None]),
            'canonical_metric_parity_checked':False,'physical_GT_accuracy_certified':False,'records':rows}
    write_json(output,report);return report


def metric_nonworse(method,reference):
    return method is not None and reference is not None and method<=reference


def paired_session_interval(differences,sessions,draws=20000,seed=20260909,alpha=.05):
    d=np.asarray(differences,float);s=np.asarray(sessions);groups=np.unique(s)
    if len(d)!=len(s) or not len(d) or not np.isfinite(d).all():raise ValueError('Invalid paired differences')
    sums=np.array([d[s==g].sum() for g in groups]);counts=np.array([(s==g).sum() for g in groups])
    rng=np.random.default_rng(seed);choices=rng.integers(0,len(groups),size=(draws,len(groups)))
    means=sums[choices].sum(1)/counts[choices].sum(1)
    return {'difference':float(d.mean()),'lower':float(np.quantile(means,alpha/2)),
            'upper':float(np.quantile(means,1-alpha/2)),'sessions':len(groups),'draws':draws,
            'warning':'Conditional session uncertainty; does not estimate unseen training-seed variability. Reused DEV is not an independent test.'}


def compare(reference_paths,method_paths,output):
    if len(reference_paths)!=len(method_paths):raise ValueError('Paired seeds required')
    paired=[];base=[];method=[];sessions=None;ids=None;seedrows=[]
    for rpath,mpath in zip(reference_paths,method_paths):
        r,m=read_json(rpath),read_json(mpath)
        if r['seed']!=m['seed'] or r['manifest_sha256']!=m['manifest_sha256']:raise ValueError('Seed/population mismatch')
        if r['gt_corners']!=m['gt_corners']:raise ValueError('GT denominator mismatch')
        rr={x['id']:x for x in r['records']};mm={x['id']:x for x in m['records']}
        if set(rr)!=set(mm):raise ValueError('Frame mismatch')
        if any(rr[k]['fixed_gt_mask8']!=mm[k]['fixed_gt_mask8'] or rr[k]['session']!=mm[k]['session'] for k in rr):
            raise ValueError('Per-frame GT mask or session mismatch')
        order=sorted(rr)
        if ids is not None and ids!=order:raise ValueError('Across-seed frame mismatch')
        ids=order;sessions=[rr[k]['session'] for k in ids]
        use=[k for k in ids if rr[k]['primary_capped_normalized_corner_mean'] is not None]
        rv=np.array([rr[k]['primary_capped_normalized_corner_mean'] for k in use])
        mv=np.array([mm[k]['primary_capped_normalized_corner_mean'] for k in use])
        if not np.isfinite(mv).all():raise ValueError('Method supervision denominator changed')
        base.append(rv);method.append(mv);paired.append(mv-rv)
        good=harm=catastrophic=0
        for k in use:
            for x,y in zip(rr[k]['fixed_error8_px'],mm[k]['fixed_error8_px']):
                if x is not None and x<=10:
                    good+=1;harm+=int(y is None or y>10)
            if rr[k]['symmetric_frame_mean_px'] is not None and rr[k]['symmetric_frame_mean_px']<=10 and (mm[k]['symmetric_frame_mean_px'] is None or mm[k]['symmetric_frame_mean_px']>50):catastrophic+=1
        seedrows.append({'seed':r['seed'],'relative_gain':float((rv.mean()-mv.mean())/max(rv.mean(),1e-12)),
                         'reference_pose_coverage':r['valid_pose_frames'],'method_pose_coverage':m['valid_pose_frames'],
                         'symmetric_median_nonworse':metric_nonworse(m['symmetric_error8_px']['median'],r['symmetric_error8_px']['median']),
                         'symmetric_p90_nonworse':metric_nonworse(m['symmetric_error8_px']['p90'],r['symmetric_error8_px']['p90']),
                         'fixed_ID_good_point_harm_diagnostic':harm/max(1,good),
                         'new_catastrophic_frames_mean10_to50':catastrophic})
    if len({x['seed'] for x in seedrows})!=len(seedrows):raise ValueError('Duplicate seed')
    interval=paired_session_interval(np.mean(paired,axis=0),[rr[k]['session'] for k in use])
    report={'schema':'plpose_v5_comparison_1','seeds':seedrows,'paired_session_interval':interval,
            'scientific_success':None,'protocol_gate_applied':False,
            'next':'Apply the preregistered protocol including baseline reference, source coverage and runtime. Do not infer success from CI alone.'}
    write_json(output,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    e=s.add_parser('evaluate');e.add_argument('--manifest',required=True);e.add_argument('--predictions',required=True);e.add_argument('--output',required=True)
    c=s.add_parser('compare');c.add_argument('--reference',nargs='+',required=True);c.add_argument('--method',nargs='+',required=True);c.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='evaluate':evaluate(a.manifest,a.predictions,a.output)
    else:compare(a.reference,a.method,a.output)
if __name__=='__main__':main()
