"""Exactly three GT-free repeat extractions at the largest recorded peak delta."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import dht_lines as D
import evaluate as E
import uncertainty_geometry as U


def diagnose(live):
    live=Path(live).resolve();out=live/'uncertainty_fusion_v1'
    target=out/'NUMERICAL_STABILITY_DHT.json'
    if target.exists():
        raise ValueError('Three-repeat diagnostic already exists; do not repeat or replace it')
    names=('DHT_UNCERTAINTY.json','PEAK_REPRODUCIBILITY_DHT.json')
    source_sha={name:E.sha(out/name) for name in names}
    source_sha.update({name:E.sha(live/name) for name in ('CONFIG.json','manifest.json','BASELINE_YOLO.json','DHT_LINES.json')})
    code_sha={str(p):E.sha(p) for p in (Path(__file__),Path(U.__file__),Path(D.__file__))}
    peaks=E.read(out/'PEAK_REPRODUCIBILITY_DHT.json')['records']
    worst=max(peaks,key=lambda r:max(r['absolute_delta']))
    frame_id,seed=worst['id'],int(worst['seed'])
    records=E.read(live/'manifest.json')['records']
    index=next(i for i,r in enumerate(records) if r['id']==frame_id)
    record=records[index]
    saved=E.read(out/'DHT_UNCERTAINTY.json')['records'][index]['seeds'][str(seed)]
    points=np.asarray(E.read(live/'BASELINE_YOLO.json')['records'][index]['kps'],float)
    reference_m=np.asarray(saved['moment_about_mode'],float)
    reference_v=U.line_variance_at_points(points,reference_m)
    runtime=E.read(live/'DHT_LINES.json')['runtime']
    if torch.__version__!=runtime['torch'] or cv2.__version__!=runtime['opencv']:
        raise ValueError('Use exact original DHT runtime')
    torch.set_num_threads(runtime['torch_threads']);cv2.setNumThreads(runtime['opencv_threads'])
    torch.backends.cudnn.benchmark=runtime['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=runtime['matmul_tf32']
    torch.backends.cudnn.allow_tf32=runtime['cudnn_tf32']
    pipeline=D.ImageDHT(E.read(live/'CONFIG.json'),seed=seed,device='cuda')
    lattice=pipeline.lattice
    valid=lattice.valid.flatten().cpu().numpy()
    inverse=np.full(len(valid),-1,int);inverse[valid]=np.arange(valid.sum())
    theta,rho=np.meshgrid(lattice.theta_degrees.cpu().numpy(),lattice.rho_values.cpu().numpy(),indexing='ij')
    theta,rho=theta.ravel()[valid],rho.ravel()[valid]
    if E.sha(record['image'])!=record['image_sha256']:
        raise ValueError('Original image checksum differs')
    image=cv2.imread(record['image'])
    runs=[];probabilities=[];first_features=None
    with torch.no_grad():
        for repeat in range(3):
            features=pipeline.features(image)
            if first_features is None:first_features=features.clone()
            features_delta=float((features-first_features).abs().max())
            scores=pipeline.head(features)
            masked=scores.masked_fill(~lattice.valid.flatten()[None,None],-1e9)
            p=masked.softmax(-1)[0].cpu().numpy()
            best=masked.argmax(-1)[0].cpu().numpy()
            predicted_theta,predicted_rho=[v[0].cpu().numpy() for v in D.decode(scores,lattice)]
            m=U.posterior_moments(p[:,valid],theta,rho,inverse[best],record['width'],record['height'])
            matrix=m['second_moment_about_mode'];variance=U.line_variance_at_points(points,matrix)
            abs_m=np.linalg.norm(matrix-reference_m,axis=(1,2))
            rel_m=abs_m/np.maximum(np.linalg.norm(reference_m,axis=(1,2)),1e-30)
            abs_v=np.abs(variance-reference_v)
            rel_v=abs_v/np.maximum(np.abs(reference_v),1e-30)
            runs.append(dict(repeat=repeat+1,peak_probability=p.max(-1),
                peak_abs_difference_from_saved_new_posterior=np.abs(p.max(-1)-saved['reextracted_peak_probability']),
                peak_abs_difference_from_prior_live=np.abs(p.max(-1)-saved['peak_probability']),
                theta_rho_exact_to_frozen=bool(np.array_equal(predicted_theta,saved['theta_deg']) and np.array_equal(predicted_rho,saved['rho'])),
                feature_max_abs_difference_from_first=features_delta,
                moment_about_mode=matrix,moment_absolute_frobenius_delta_per_role=abs_m,
                moment_relative_frobenius_delta_per_role=rel_m,
                incident_raw_variance_px2=variance,incident_raw_variance_absolute_delta_px2=abs_v,
                incident_raw_variance_relative_delta=rel_v,
                incident_floor1_variance_relative_delta=np.abs(np.maximum(variance,1)-np.maximum(reference_v,1))/np.maximum(reference_v,1),
                entropy_normalized=m['entropy_normalized'],
                entropy_absolute_difference=np.abs(m['entropy_normalized']-saved['entropy_normalized'])))
            normalized=p[:,valid].astype(float);normalized/=normalized.sum(-1,keepdims=True)
            probabilities.append(normalized)
    pairs=[]
    for a,b in ((0,1),(0,2),(1,2)):
        pa,pb=probabilities[a],probabilities[b];mid=(pa+pb)/2
        js=.5*np.sum(pa*np.log(np.maximum(pa,1e-300)/np.maximum(mid,1e-300)),axis=-1)
        js+=.5*np.sum(pb*np.log(np.maximum(pb,1e-300)/np.maximum(mid,1e-300)),axis=-1)
        pairs.append(dict(repeats=[a+1,b+1],jensen_shannon_nats_per_role=js,
                          total_variation_per_role=.5*np.abs(pa-pb).sum(-1)))
    result=dict(schema='dht_numerical_stability_v1',complete=True,n_repeats=3,
        id=frame_id,population=record['population'],seed=seed,
        selection='Largest absolute peak discrepancy in already-completed GT-free all-image extraction audit; no GT/accuracy selection',
        source_sha256=source_sha,code_sha256=code_sha,image_sha256=record['image_sha256'],runtime=runtime,
        reference_moment_about_mode=reference_m,reference_incident_raw_variance_px2=reference_v,
        predicted_yolo_corners_xy=points,recorded_peak_parity=worst,
        repeats=runs,repeat_distribution_comparison=pairs,
        all_argmax_exact=all(r['theta_rho_exact_to_frozen'] for r in runs),
        maximum_moment_relative_frobenius_delta=float(max(np.max(r['moment_relative_frobenius_delta_per_role']) for r in runs)),
        maximum_incident_raw_variance_absolute_delta_px2=float(max(np.nanmax(r['incident_raw_variance_absolute_delta_px2']) for r in runs)),
        maximum_incident_raw_variance_relative_delta=float(max(np.nanmax(r['incident_raw_variance_relative_delta']) for r in runs)),
        gt_used=False,calibration_used=False,selection_rule_changed=False,
        interpretation='Descriptive three-repeat numerical stability at one peak-selected frame; no global stability guarantee. Matrix Frobenius comparison mixes homogeneous coefficient units; residual variance at the saved predicted pixels is the directly relevant measure. No uncertainty calibration or performance judgment.')
    for name,digest in source_sha.items():
        path=out/name if name in names else live/name
        if E.sha(path)!=digest:raise ValueError('Frozen input changed during diagnostic')
    assert all(E.sha(path)==digest for path,digest in code_sha.items())
    E.write(target,result)
    print({k:result[k] for k in ('id','seed','n_repeats','all_argmax_exact','maximum_moment_relative_frobenius_delta','maximum_incident_raw_variance_relative_delta')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    diagnose(parser.parse_args().run_dir)
