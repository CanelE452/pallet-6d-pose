"""Calibrated boundary observations; no detector, pose, or truth interface."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def load_calibration(path=None):
    return json.loads(Path(path or DOC / 'CALIBRATION.json').read_text())


def logits_statistics(logits):
    if hasattr(logits, 'detach'):
        logits = logits.detach().float().cpu().numpy()
    raw = np.asarray(logits, dtype=np.float32)
    if raw.shape != (84,66) or not np.isfinite(raw).all():
        raise ValueError('Expected finite 84 by 66 frozen head logits')
    z = raw.astype(np.float64)
    best = z[:,:65].argmax(1)
    margin = z[np.arange(84),best] - z[:,65]
    score = 1 / (1 + np.exp(-np.clip(margin,-700,700)))
    p = np.exp(z[:,:65] - z[:,:65].max(1,keepdims=True))
    p /= p.sum(1,keepdims=True)
    # Coordinates stay at one MAP bin; this second moment never averages positions.
    sigma = np.sqrt((p * (np.arange(65)[None,:]-best[:,None])**2).sum(1))
    return raw, best, score, np.maximum(.5,sigma)


def query_geometry(points):
    points = np.asarray(points,float)
    center=[]; normal=[]; identity=[]; valid=[]
    for e,(a,b) in enumerate(EDGES):
        delta=points[b]-points[a]; length=float(np.linalg.norm(delta))
        tangent=delta/max(length,1e-6)
        for u in np.arange(1,8)/8:
            center.append((1-u)*points[a]+u*points[b])
            normal.append([-tangent[1],tangent[0]])
            identity.append([e,a,b,float(u),length])
            valid.append(bool(np.isfinite(delta).all() and length>1e-6 and not np.any(np.all(points[[a,b]]==-1,axis=1))))
    center=np.asarray(center); normal=np.asarray(normal)
    return dict(center=center,normal=normal,identity=identity,valid=np.asarray(valid),
                candidate=center[:,None]+np.arange(-32,33)[None,:,None]*normal[:,None])


def fit_line(points, radii):
    """Weighted TLS and a conservative local parameter covariance surrogate."""
    points=np.asarray(points,float); radii=np.asarray(radii,float)
    weights=1/np.maximum(.5,radii)**2
    center=np.average(points,axis=0,weights=weights); dx=points-center
    eig,v=np.linalg.eigh((dx*weights[:,None]).T@dx/weights.sum())
    if eig[-1]<=1e-12:
        return None
    tangent=v[:,-1]
    if tangent[np.argmax(np.abs(tangent))]<0:
        tangent=-tangent
    normal=np.array([-tangent[1],tangent[0]]); offset=float(normal@center)
    s=dx@tangent; design=np.column_stack([s,np.ones(len(s))])
    information=(design*weights[:,None]).T@design
    if np.linalg.det(information)<=1e-12:
        return None
    residual=points@normal-offset
    scale=max(1.,float(np.mean((residual/np.maximum(.5,radii))**2)))
    covariance=np.linalg.inv(information)*scale
    return dict(normal=normal.tolist(),offset=offset,tangent=tangent.tolist(),center=center.tolist(),
                support_length_px=float(np.ptp(s)),support_interval_px=[float(s.min()),float(s.max())],
                support_points=points.tolist(),residual_rms_px=float(np.sqrt(np.mean(residual**2))),
                standardized_residual_rms=float(np.sqrt(np.mean((residual/np.maximum(.5,radii))**2))),
                parameter_covariance=covariance.tolist(),covariance_is_calibrated_surrogate=True)


def build_lines(query, records):
    lines=[]; rejected=[]; pair_count=0
    for edge,(a,b) in enumerate(EDGES):
        ids=[q['query'] for q in records if q['edge']==edge and q['selected_xy'] is not None]
        length=float(query['identity'][edge*7][4]); spacing=length/8
        if len(ids)<3:
            rejected.append(dict(edge=edge,reason='FEWER_THAN_THREE_QUERY_OBSERVATIONS',queries=ids)); continue
        xy=np.asarray([records[i]['selected_xy'] for i in ids]); radius=np.asarray([records[i]['radius_px'] for i in ids]); hypotheses=[]
        for aa,bb in combinations(range(len(ids)),2):
            pair_count+=1
            tangent=xy[bb]-xy[aa]; norm=np.linalg.norm(tangent)
            if norm<=1e-6: continue
            normal=np.array([-tangent[1],tangent[0]])/norm; offset=float(normal@xy[aa])
            residual=np.abs(xy@normal-offset); accepted=np.flatnonzero(residual<=radius)
            if len(accepted)<3 or max(ids[j]%7 for j in accepted)-min(ids[j]%7 for j in accepted)<2: continue
            span=float(np.ptp(xy[accepted]@(tangent/norm)))
            if span<2*spacing*(1-64*np.finfo(float).eps): continue
            hypotheses.append(((-len(accepted),float(np.mean((residual[accepted]/radius[accepted])**2)),-span,ids[aa],ids[bb]),accepted))
        if not hypotheses:
            rejected.append(dict(edge=edge,reason='NO_THREE_QUERY_SPANNING_CONSENSUS',queries=ids)); continue
        accepted=min(hypotheses,key=lambda x:x[0])[1]
        fitted=fit_line(xy[accepted],radius[accepted])
        if fitted is None:
            rejected.append(dict(edge=edge,reason='DEGENERATE_LINE_INFORMATION',queries=ids)); continue
        normal=np.array(fitted['normal']); offset=fitted['offset']
        # One predetermined recheck/refit, not an iterative threshold search.
        accepted=np.flatnonzero(np.abs(xy@normal-offset)<=radius)
        if len(accepted)<3 or max(ids[j]%7 for j in accepted)-min(ids[j]%7 for j in accepted)<2:
            rejected.append(dict(edge=edge,reason='REFIT_LOST_QUERY_CONSENSUS',queries=ids)); continue
        fitted=fit_line(xy[accepted],radius[accepted])
        if fitted is None or fitted['support_length_px']<2*spacing*(1-64*np.finfo(float).eps):
            rejected.append(dict(edge=edge,reason='INSUFFICIENT_LINE_SPAN',queries=ids)); continue
        final_ids=[ids[j] for j in accepted]
        fitted.update(edge=edge,endpoints=[a,b],queries=final_ids,query_radii_px=radius[accepted].tolist(),
                      nominal_query_spacing_px=spacing,reason='CALIBRATED_THREE_QUERY_SPANNING_CONSENSUS',
                      correlated_observation_source='one semantic edge; query samples are not independent corner IDs')
        lines.append(fitted)
    return lines,rejected,pair_count


def intersection(line_a,line_b):
    matrix=np.array([line_a['normal'],line_b['normal']]); det=float(np.linalg.det(matrix))
    if abs(det)<=1e-6: return None
    xy=np.linalg.solve(matrix,[line_a['offset'],line_b['offset']])
    if not np.isfinite(xy).all(): return None
    variances=[]; gaps=[]
    for line in [line_a,line_b]:
        s=float((xy-np.array(line['center']))@line['tangent']); lo,hi=line['support_interval_px']
        gap=max(lo-s,s-hi,0.)/max(line['support_length_px'],1e-6)
        h=np.array([s,1.]); variances.append(float(h@np.array(line['parameter_covariance'])@h)); gaps.append(gap)
    inv=np.linalg.inv(matrix); covariance=inv@np.diag(variances)@inv.T
    sigma=float(np.sqrt(max(0.,np.linalg.eigvalsh(covariance).max())))
    spacing=min(line_a['nominal_query_spacing_px'],line_b['nominal_query_spacing_px'])
    return dict(xy=xy.tolist(),edges=[line_a['edge'],line_b['edge']],absolute_normal_determinant=abs(det),
                covariance_px2=covariance.tolist(),sigma_px=sigma,nominal_query_spacing_px=spacing,
                extrapolation_ratios=gaps,max_extrapolation_ratio=max(gaps))


def decode(query,logits,calibration):
    raw,best,score,sigma=logits_statistics(logits)
    if 'raw_hw' not in query: raise ValueError('query.raw_hw is required to exclude off-image proposals')
    h,w=query['raw_hw']; supported=set(calibration['supported_edges']); records=[]
    enabled=calibration['confidence']['enabled']; cutoff=calibration['confidence']['threshold']
    for i,(edge,a,b,u,length) in enumerate(query['identity']):
        xy=np.asarray(query['candidate'][i,best[i]],float); valid=bool(query['valid'][i]); inside=bool(np.isfinite(xy).all() and 0<=xy[0]<w and 0<=xy[1]<h)
        reason='CALIBRATED_QUERY_OBSERVATION'
        if not valid: reason='INVALID_QUERY_GEOMETRY'
        elif edge not in supported: reason='MODEL_CALIBRATION_UNSUPPORTED'
        elif not inside: reason='CANDIDATE_OUTSIDE_IMAGE'
        elif not enabled: reason='CALIBRATION_PRECISION_NOT_ATTAINED'
        elif score[i]<cutoff: reason='CALIBRATED_NO_MATCH_OR_LOW_CONFIDENCE'
        accepted=reason=='CALIBRATED_QUERY_OBSERVATION'; radius=float(sigma[i]*calibration['uncertainty']['query_scale'])
        records.append(dict(query=i,edge=int(edge),endpoints=[int(a),int(b)],fraction=float(u),center=np.asarray(query['center'][i]).tolist(),
                            normal=np.asarray(query['normal'][i]).tolist(),candidate_logits=raw[i].tolist(),chosen_candidate=int(best[i]),
                            selected_xy=xy.tolist() if accepted else None,no_match=not accepted,reason=reason,
                            confidence=float(score[i]),sigma_mode_px=float(sigma[i]),radius_px=radius,
                            calibrated_model_coverage=edge in supported,physical_absence_inferred=False))
    lines,rejected,pair_count=build_lines(query,records); lookup={line['edge']:line for line in lines}; corners=[]; corner_diagnostics=[]
    for corner in range(8):
        incident=[e for e,pair in enumerate(EDGES) if corner in pair and e in lookup]; candidates=[]
        for ea,eb in combinations(incident,2):
            value=intersection(lookup[ea],lookup[eb])
            if value is None: continue
            xy=np.asarray(value['xy']); radius=value['sigma_px']*calibration['uncertainty']['corner_scale']
            normalized=radius/max(value['nominal_query_spacing_px'],1e-6)
            reason='CALIBRATED_INCIDENT_LINE_INTERSECTION'
            if not calibration['geometry']['enabled']: reason='SOURCE_CORNER_GEOMETRY_UNCALIBRATED'
            elif not(0<=xy[0]<w and 0<=xy[1]<h): reason='CORNER_OUTSIDE_IMAGE'
            elif value['max_extrapolation_ratio']>calibration['geometry']['max_extrapolation_ratio']+64*np.finfo(float).eps*max(1.,value['max_extrapolation_ratio'],calibration['geometry']['max_extrapolation_ratio']): reason='EXCESSIVE_SUPPORT_EXTRAPOLATION'
            elif normalized>1.: reason='CORNER_UNCERTAINTY_EXCEEDS_ONE_QUERY_SPACING'
            value.update(id=corner,radius_px=radius,normalized_uncertainty=normalized,reason=reason,accepted=reason=='CALIBRATED_INCIDENT_LINE_INTERSECTION')
            corner_diagnostics.append(value)
            if value['accepted']: candidates.append(value)
        if candidates:
            selected=min(candidates,key=lambda c:(c['radius_px'],c['max_extrapolation_ratio'],c['edges']))
            selected.update(source='two calibrated observed edge lines; N3 admission is a separate downstream gate',
                            correlated_edges=selected['edges'],extrapolation_possible=selected['max_extrapolation_ratio']>0)
            corners.append(selected)
    used_edges=set(e for c in corners for e in c['edges'])
    return dict(corners=corners,lines=lines,queries=records,partial_lines=[line for line in lines if line['edge'] not in used_edges],
                selected_queries=sum(q['selected_xy'] is not None for q in records),all_no_match=not any(q['selected_xy'] is not None for q in records),
                raw_logits_sha256=hashlib.sha256(raw.astype('<f4').tobytes()).hexdigest(),
                feature_initial_pose=query.get('initial_pose'),feature_hidden_initial=query.get('hidden_initial'),predicted_roles=query.get('role'),
                diagnostics=dict(line_rejections=rejected,corner_candidates=corner_diagnostics,line_pair_hypotheses=pair_count,
                                 native_corner_ids_unique=len({c['id'] for c in corners})==len(corners),
                                 unsupported_is_model_coverage_only=True,reprojected_points_used_as_observations=False))
