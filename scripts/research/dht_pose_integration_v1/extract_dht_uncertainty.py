"""Re-extract frozen actual-image DHT distributions and compact GT-free moments."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch

import dht_lines as D
import evaluate as E
import uncertainty_geometry as U

HERE = Path(__file__).resolve().parent
PEAK_ATOL = 1e-6
PEAK_RTOL = 1e-5


def source_hashes():
    return D.source_hashes() | {str(path): E.sha(path) for path in (
        Path(__file__), HERE/'uncertainty_geometry.py', HERE/'fusion.py', HERE/'live_lines.py')}


def extract(run_dir):
    root = Path(run_dir).resolve()
    output = root/'uncertainty_fusion_v1'
    if not (output/'PURPOSE.md').is_file() or not (output/'PROTOCOL.json').is_file():
        raise ValueError('Registered uncertainty PURPOSE/PROTOCOL are required')
    cfg, manifest, prior = [E.read(root/name) for name in ('CONFIG.json','manifest.json','DHT_LINES.json')]
    records, seeds = manifest['records'], cfg['dht']['seeds']
    if len(records) != 692 or seeds != [1,2,3] or not prior['complete']:
        raise ValueError('Expected frozen692 actual-image records and three DHT seeds')
    if [r['id'] for r in records] != [r['id'] for r in prior['records']]:
        raise ValueError('Frozen live DHT identity/order mismatch')
    identity = dict(config_sha256=E.sha(root/'CONFIG.json'), manifest_sha256=E.sha(root/'manifest.json'),
        prior_lines_sha256=E.sha(root/'DHT_LINES.json'), protocol_sha256=E.sha(output/'PROTOCOL.json'),
        checkpoint_sha256=cfg['dht']['checkpoint_sha256'], source_sha256=source_hashes())
    extraction_protocol = dict(schema='dht_uncertainty_extraction_protocol_v1', **identity,
        theta_rho_parity='exact array equality', line_endpoint_parity='exact array equality',
        peak_diagnostic_atol=PEAK_ATOL, peak_diagnostic_rtol=PEAK_RTOL,
        peak_parity_is_extraction_pass_condition=False,
        numeric_tolerance_reason='Fixed-input repeated calls localized peak variations to CUDA SparseDHT voting reduction. Peak differences are measured for every role separately; PASS requires unchanged argmax/coordinates, finite normalized posterior and verified sources. No GT or accuracy determined this distinction.',
        probe_sha256={p.name:E.sha(p) for p in (output/'DHT_PARITY_PROBE.json', output/'DHT_REDUCTION_PROBE.json',output/'DHT_PARITY_PROBE_008168.json',output/'DHT_EXTRACTION_FAILURE_HISTORY.json') if p.exists()},
        point_estimate='Preserve frozen live theta/rho/line endpoints and peak_probability; reextracted_peak_probability is stored separately for transparency',
        uncertainty='Full valid-bin distribution moments, no GT or score-based tuning')
    extraction_protocol_path=output/'EXTRACTION_PROTOCOL_DHT.json'
    if extraction_protocol_path.exists() and E.read(extraction_protocol_path)!=extraction_protocol:
        raise ValueError('Existing DHT extraction protocol differs')
    E.write(extraction_protocol_path,extraction_protocol)
    identity['extraction_protocol_sha256']=E.sha(extraction_protocol_path)
    destination = output/'DHT_UNCERTAINTY.json'
    audit_path = output/'EXTRACTION_AUDIT_DHT.json'
    if destination.exists() or audit_path.exists():
        if destination.exists() and audit_path.exists():
            saved, audit = E.read(destination), E.read(audit_path)
            if saved.get('complete') and audit.get('PASS') and all(saved.get(k) == v for k,v in identity.items()) and audit['output_sha256'] == E.sha(destination):
                print('Verified existing complete DHT uncertainty extraction', flush=True)
                return
        raise ValueError('Existing uncertainty extraction cannot be overwritten with different provenance')
    runtime = prior['runtime']
    if torch.__version__ != runtime['torch'] or cv2.__version__ != runtime['opencv']:
        raise ValueError(f'Use original DHT runtime torch{runtime["torch"]}/opencv{runtime["opencv"]}; '
                         f'current torch{torch.__version__}/opencv{cv2.__version__}')
    torch.set_num_threads(runtime['torch_threads'])
    cv2.setNumThreads(runtime['opencv_threads'])
    torch.backends.cudnn.benchmark = runtime['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = runtime['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = runtime['cudnn_tf32']
    begin = time.perf_counter()
    pipeline = D.ImageDHT(cfg, seed=1, device='cuda')
    heads = {1:(pipeline.head,pipeline.lattice)}
    for seed in (2,3):
        heads[seed] = D.load_head(cfg,seed,torch.device('cuda'))
    lattice = pipeline.lattice
    valid = lattice.valid.flatten().cpu().numpy()
    valid_indices = np.flatnonzero(valid)
    inverse_indices = np.full(len(valid),-1,int)
    inverse_indices[valid_indices] = np.arange(len(valid_indices))
    theta_grid, rho_grid = np.meshgrid(lattice.theta_degrees.cpu().numpy(), lattice.rho_values.cpu().numpy(), indexing='ij')
    theta_valid, rho_valid = theta_grid.ravel()[valid], rho_grid.ravel()[valid]
    count = len(valid_indices)
    output_rows=[]
    max_peak_error=max_line_error=max_h_line_residual=max_quadratic_error=0.
    max_probability_sum_error=0.
    peak_comparisons=[]
    minimum_eigenvalue=float('inf')
    checked_roles=0
    with torch.no_grad():
        for i,record in enumerate(records):
            if E.sha(record['image']) != record['image_sha256']:
                raise ValueError(f'Original image bytes changed: {record["id"]}')
            image=cv2.imread(record['image'])
            if image is None or image.shape[:2] != (record['height'],record['width']):
                raise ValueError('Image dimensions changed')
            features=pipeline.features(image)
            row={k:record[k] for k in ('id','population','group','width','height')}
            row['seeds']={}
            for seed,(head,current_lattice) in heads.items():
                scores=head(features)
                masked=scores.masked_fill(~current_lattice.valid.flatten()[None,None],-1e9)
                probability=masked.softmax(-1)[0].cpu().numpy()
                probability_sum_error=float(np.abs(probability.sum(-1)-1).max())
                max_probability_sum_error=max(max_probability_sum_error,probability_sum_error)
                if not np.isfinite(probability).all() or (probability<0).any() or probability_sum_error>1e-5:
                    raise ValueError('Posterior is not finite/nonnegative/normalized')
                best=masked.argmax(-1)[0].cpu().numpy()
                theta,rho=[v[0].cpu().numpy() for v in D.decode(scores,current_lattice)]
                lines=D.T.line_pixels(theta,rho,record['width'],record['height'])
                peaks=probability.max(-1)
                old=prior['records'][i]['seeds'][str(seed)]
                if not np.array_equal(theta,np.asarray(old['theta_deg'])) or not np.array_equal(rho,np.asarray(old['rho'])):
                    raise ValueError(f'Frozen live argmax mismatch: {record["id"]} seed{seed}')
                peak_error=float(np.abs(peaks-np.asarray(old['peak_probability'])).max())
                line_error=float(np.abs(lines-np.asarray(old['lines'])).max())
                max_peak_error,max_line_error=max(max_peak_error,peak_error),max(max_line_error,line_error)
                if not np.array_equal(lines,np.asarray(old['lines'])):
                    raise ValueError(f'Frozen live exact line mismatch: {record["id"]} seed{seed}: {line_error}')
                absolute=np.abs(peaks-np.asarray(old['peak_probability']))
                relative=absolute/np.maximum(np.asarray(old['peak_probability']),1e-30)
                peak_comparisons.append(dict(id=record['id'],seed=seed,absolute_delta=absolute,
                    relative_delta=relative,within_diagnostic_tolerance=np.isclose(peaks,np.asarray(old['peak_probability']),atol=PEAK_ATOL,rtol=PEAK_RTOL)))
                moment=U.posterior_moments(probability[:,valid],theta_valid,rho_valid,inverse_indices[best],
                                           record['width'],record['height'])
                matrix=moment['second_moment_about_mode']
                if not all(np.isfinite(value).all() for value in moment.values()):
                    raise ValueError('Nonfinite compact posterior moment')
                eig=float(np.linalg.eigvalsh(matrix).min())
                minimum_eigenvalue=min(minimum_eigenvalue,eig)
                if eig < -1e-7:
                    raise ValueError('Posterior second moment is not numerically positive semidefinite')
                z=np.concatenate([lines,np.ones((8,2,1))],-1)
                h_residual=float(np.abs(np.einsum('rpi,ri->rp',z,moment['mode_h'])).max())
                max_h_line_residual=max(max_h_line_residual,h_residual)
                if h_residual>1e-8:
                    raise ValueError('Original homogeneous mode line differs from frozen pixel endpoints')
                if i==0:
                    # Compact quadratic form vs the full distribution at three arbitrary pixels.
                    h=U.candidate_homogeneous(theta_valid,rho_valid,record['width'],record['height'])
                    mode=moment['mode_h']
                    polarity=np.where(mode[:,:2]@h[:,:2].T>=0,1.,-1.)
                    delta=polarity[...,None]*h[None]-mode[:,None]
                    p=probability[:,valid].astype(float);p/=p.sum(-1,keepdims=True)
                    for point in ([0,0,1],[231,117,1],[639,479,1]):
                        point=np.asarray(point,float)
                        explicit=np.sum(p*(delta@point)**2,-1)
                        compact=np.einsum('i,rij,j->r',point,matrix,point)
                        max_quadratic_error=max(max_quadratic_error,float(np.abs(explicit-compact).max()))
                # Preserve the already-frozen argmax endpoints bit for bit.
                row['seeds'][str(seed)]=dict(lines=old['lines'],theta_deg=old['theta_deg'],rho=old['rho'],
                    peak_probability=old['peak_probability'],reextracted_peak_probability=peaks,
                    h_mode=moment.pop('mode_h'),
                    moment_about_mode=moment.pop('second_moment_about_mode'),
                    mean_delta=moment.pop('mean_delta_h'),**moment)
                checked_roles+=8
            output_rows.append(row)
            if (i+1)%100==0 or i+1==len(records):
                print(f'DHT posterior extraction {i+1}/{len(records)} x3seeds; peak parity max{max_peak_error:.3g}',flush=True)
    if max_quadratic_error>1e-6:
        raise ValueError('Compact residual variance differs from full-posterior quadratic form')
    if source_hashes()!=identity['source_sha256'] or E.sha(root/'DHT_LINES.json')!=identity['prior_lines_sha256'] or E.sha(output/'PROTOCOL.json')!=identity['protocol_sha256']:
        raise ValueError('Inputs/source changed during extraction')
    result=dict(schema='dht_uncertainty_v1',complete=True,**identity,n_frames=len(records),seeds=seeds,
        records=output_rows,n_valid_hough_bins=count,n_total_hough_bins=len(valid),
        entropy_denominator_nats=float(np.log(count)),
        distribution='Softmax over original valid theta/rho lattice; allglobal modes retained, no temperature or GT fitting',
        moment_semantics='h=(nx,ny,c) has original-pixel unit normal; signs aligned by dot(n_candidate,n_mode)>=0. M=E[(h-h_mode)(h-h_mode)^T]; zMz is squared residual difference around the unchanged argmax line, not calibrated error variance.',
        local_chart=dict(theta_window_deg=5,rho_window_feature_pixels=2,
                         coordinates=['delta_theta_radians','delta_rho_feature_pixels'],
                         semantics='Conditional on local window, seam corrected; metadata only, can omit distant modes'),
        inference_path='Exact original live ImageDHT batch1 uint8reflect100/square400 feature extraction, FP16 then FP32 features, three frozen heads atbatch1',
        runtime=runtime,wall_seconds=time.perf_counter()-begin,
        no_gt_input=True,no_full_logits_persisted=True)
    E.write(destination,result)
    peak_absolute=np.concatenate([r['absolute_delta'] for r in peak_comparisons])
    peak_relative=np.concatenate([r['relative_delta'] for r in peak_comparisons])
    peak_within=np.concatenate([r['within_diagnostic_tolerance'] for r in peak_comparisons])
    peak_diagnosis=dict(schema='dht_peak_numerical_reproducibility_v1',complete=True,
        criterion='Separate numerical diagnosis; not a geometric point-estimate or extraction PASS condition',
        reference_tolerance=dict(atol=PEAK_ATOL,rtol=PEAK_RTOL),n_roles=len(peak_absolute),
        n_within_reference_tolerance=int(peak_within.sum()),n_outside_reference_tolerance=int((~peak_within).sum()),
        absolute_delta_p50_p90_p99_max=np.percentile(peak_absolute,[50,90,99,100]),
        relative_delta_p50_p90_p99_max=np.percentile(peak_relative,[50,90,99,100]),records=peak_comparisons,
        probe_sha256=extraction_protocol['probe_sha256'],gt_or_accuracy_consulted=False)
    E.write(output/'PEAK_REPRODUCIBILITY_DHT.json',peak_diagnosis)
    audit=dict(schema='dht_uncertainty_extraction_audit_v1',PASS=True,complete=True,
        **identity,output_sha256=E.sha(destination),n_frames=len(records),n_frame_seed_pairs=len(records)*3,
        n_role_parity_checks=checked_roles,theta_rho_argmax_exact_match=True,
        peak_probability_is_pass_condition=False,
        peak_reproducibility_sha256=E.sha(output/'PEAK_REPRODUCIBILITY_DHT.json'),
        peak_n_outside_reference_tolerance=int((~peak_within).sum()),
        posterior_finite_nonnegative_normalized=True,maximum_raw_probability_sum_error=max_probability_sum_error,
        maximum_peak_probability_difference=max_peak_error,maximum_line_endpoint_difference_px=max_line_error,
        maximum_h_mode_line_endpoint_residual_px=max_h_line_residual,
        minimum_second_moment_eigenvalue=minimum_eigenvalue,
        maximum_full_vs_compact_residual_quadratic_error_px2=max_quadratic_error,
        n_valid_hough_bins=count,entropy_denominator_nats=float(np.log(count)),
        gt_used=False,selection_used=False,calibration_used=False,
        all_original_images_sha256_verified=True,
        scope='Inference/distribution geometry parity only; no uncertainty calibration or accuracy claim')
    E.write(audit_path,audit)
    print(f'PASS DHT extraction: {checked_roles} matching role predictions; validbins{count}; output={destination}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Existing integration live/ directory')
    extract(parser.parse_args().run_dir)
