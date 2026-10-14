"""Observed-outline uncertainty and bounded fusion; no pose or label input.

Resampling estimates line measurement variation, not semantic correctness or
model bias. Adjacent vertices share a resampled line and remain correlated.
"""
from copy import deepcopy
import numpy as np


def fit_resampled_line(points,confidence,reference_normal):
    points=np.asarray(points,float);confidence=np.clip(confidence,.05,1.)
    if len(points)<3:return None
    weights=confidence.copy()
    for _ in range(4):
        center=np.average(points,axis=0,weights=weights);delta=points-center
        covariance=(delta*weights[:,None]).T@delta/weights.sum()
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        if eigenvalues[-1]<1e-6:return None
        normal=eigenvectors[:,0]
        if normal@reference_normal<0:normal=-normal
        offset=-float(normal@center)
        residual=np.abs(points@normal+offset)
        weights=confidence*np.minimum(1.,1.25/np.maximum(residual,1e-12))
    return normal,offset


def line_resamples(edge,repetitions=64,block_length=3,seed=20261013):
    if not edge.get('supported'):return None
    points=np.asarray(edge['observed_points'],float);confidence=np.asarray(edge['observation_confidence'],float)
    inlier=np.asarray(edge['inlier_mask'],bool)
    points=points[inlier];confidence=confidence[inlier]
    if len(points)<3:return None
    normal=np.asarray(edge['normal'],float);center=np.average(points,axis=0,weights=np.clip(confidence,.05,1.))
    tangent=np.array([-normal[1],normal[0]]);span=float(np.ptp(points@tangent))
    rng=np.random.default_rng(seed);samples=[];parameters=[]
    for _ in range(repetitions):
        # Moving blocks preserve local dependence among neighboring mask pixels.
        starts=rng.integers(0,len(points),size=int(np.ceil(len(points)/block_length)))
        ids=np.concatenate([(s+np.arange(block_length))%len(points) for s in starts])[:len(points)]
        fitted=fit_resampled_line(points[ids],confidence[ids],normal)
        if fitted is None:samples.append(None);continue
        n,b=fitted
        angle=float(np.arctan2(normal[0]*n[1]-normal[1]*n[0],normal@n))
        parameters.append([angle,float(n@center+b)]);samples.append((n,b))
    if len(parameters)<16:return None
    return dict(samples=samples,normal=normal,offset=float(edge['offset']),center=center,
        observed_points=points,inlier_count=len(points),span_px=span,
        bootstrap_parameters_covariance=np.cov(np.asarray(parameters),rowvar=False),
        parameter_units=['radians','pixels'],successful_resamples=len(parameters),block_length=block_length)


def intersection_uncertainty(correction,repetitions=64,block_length=3,normal_floor_px=1.,seed=20261013):
    """Return per-corner covariance in native pixel coordinates and line records."""
    lines={}
    for edge in correction['edge_records']:
        value=line_resamples(edge,repetitions,block_length,seed+int(edge['edge'])*1009)
        if value is not None:lines[int(edge['edge'])]=value
    result={}
    for corner in correction['corner_records']:
        if not corner.get('refined'):continue
        first,second=corner['incident_shared_edges']
        if first not in lines or second not in lines:continue
        a,b=lines[first],lines[second];matrix=np.stack([a['normal'],b['normal']]);inv=np.linalg.inv(matrix)
        samples=[]
        for l1,l2 in zip(a['samples'],b['samples']):
            if l1 is None or l2 is None:continue
            n1,c1=l1;n2,c2=l2;M=np.stack([n1,n2])
            if abs(np.linalg.det(M))<.25:continue
            samples.append(np.linalg.solve(M,-np.array([c1,c2])))
        if len(samples)<16:continue
        bootstrap=np.cov(np.asarray(samples),rowvar=False)
        # A one-native-pixel independent normal offset floor prevents exact
        # raster lines from implying zero mask/model boundary uncertainty.
        floor=normal_floor_px**2*(inv@inv.T)
        covariance=bootstrap+floor
        eig=np.linalg.eigvalsh(covariance)
        result[int(corner['corner'])]=dict(covariance_px2=covariance.tolist(),bootstrap_covariance_px2=bootstrap.tolist(),
            floor_covariance_px2=floor.tolist(),eigenvalues_px2=eig.tolist(),largest_std_px=float(np.sqrt(eig[-1])),
            successful_pair_resamples=len(samples),incident_shared_edges=[first,second],
            samples=np.asarray(samples).tolist(),candidate=corner['candidate'])
    diagnostics={i:{k:v for k,v in d.items() if k not in ('samples','observed_points')}
                 for i,d in lines.items()}
    return result,diagnostics


def fuse_observations(original,correction,uncertainty,detector_sigma_px,maximum_alpha=.5):
    q=np.asarray(original,float);output=q.copy();records=[]
    candidate=np.asarray(correction['points'],float)
    for corner,meta in uncertainty.items():
        variance=float(meta['eigenvalues_px2'][-1])
        alpha=float(min(maximum_alpha,detector_sigma_px**2/(detector_sigma_px**2+variance)))
        output[corner]=q[corner]+alpha*(candidate[corner]-q[corner])
        records.append(dict(corner=corner,alpha=alpha,detector_sigma_assumed_px=detector_sigma_px,
            largest_intersection_variance_px2=variance,applied_shift_px=float(np.linalg.norm(output[corner]-q[corner])),
            covariance_is_semantic_correctness_probability=False))
    return output,records


def uncertainty_accuracy(appearance,uncertainty,detector_sigma_px=2.):
    """One declared difficulty proxy; preserve the original visible pool.

    Only measured points receive an evidence reduction. Preserve the .10
    eligibility floor when an originally eligible observation was above it.
    This is not a calibrated detector accuracy score.
    """
    result=deepcopy(appearance)
    for corner,meta in uncertainty.items():
        old=float(result['local_accuracy_score'][corner])
        reliability=detector_sigma_px**2/(detector_sigma_px**2+meta['eigenvalues_px2'][-1])
        value=old*(.5+.5*reliability)
        result['local_accuracy_score'][corner]=max(.10,value) if old>=.10 else value
    result['diagnostics']['SAM_uncertainty_difficulty']=dict(calibrated_probability=False,
        update='existing local accuracy times (.5+.5*variance_reliability), preserving original .10 eligible floor')
    return result
