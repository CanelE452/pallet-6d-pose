"""Public CPU math audit from sealed CAL logits and existing READY labels.

Does not import torch, read RGB/weights/features/real scores, or run PnP/rays.
Default prints results; --output exclusively writes a new verification receipt.
"""
from pathlib import Path
from collections import Counter
import argparse
import base64
import gzip
import hashlib
import json
import math
import zlib
import numpy as np
from . import observations as O


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(line) for line in gzip.open(path,'rt')]


def verify():
    d=O.DOC; repair=d.parent/'pallet_kp_supervision_repair_20261010_v1'
    cal=O.load_calibration(); protocol=json.loads((d/'CALIBRATION_PROTOCOL.json').read_text())
    assert sha(d/'CALIBRATION_PROTOCOL.json')==cal['protocol_sha256']
    for key in ['raw_rows','threshold_scan','geometry_rows']:
        assert sha(O.REPO/cal[key]['path'])==cal[key]['sha256']
    rawrows=rows(d/'CALIBRATION_ROWS.jsonl.gz'); source=rows(repair/'READY_SOURCE_TARGET_ROWS.jsonl.gz')[768:896]
    a=dict(np.load(repair/'READY_PREPARED_TARGETS.npz'))
    assert [r['index'] for r in rawrows]==list(range(768,896))
    assert sha(repair/'READY_PREPARED_TARGETS.npz')==protocol['inputs']['ready_targets']['sha256']
    supported=sorted(set(q['edge'] for r in source for q in r['queries'] if q['proposed_target']=='POSITIVE'))
    assert supported==cal['supported_edges']
    scores=[]; correct=[]; accepted_errors=[]; ratios=[]; counts=Counter(); decodedrows=[]
    for sr,r in zip(source,rawrows):
        assert sr['index']==r['index'] and sr['id']==r['id'] and sr['partition']=='calibration'
        semantic=hashlib.sha256(json.dumps(sr,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert semantic==r['source_ready_row_semantic_sha256']
        rawbytes=zlib.decompress(base64.b64decode(r['logits_fp32_zlib_base64']))
        assert hashlib.sha256(rawbytes).hexdigest()==r['raw_logits_sha256']
        z=np.frombuffer(rawbytes,dtype='<f4').reshape(84,66).astype(np.float64)
        k=np.argmax(z[:,:65],axis=1); difference=z[np.arange(84),k]-z[:,65]
        s=1/(1+np.exp(-np.clip(difference,-700,700)))
        candidate_probability=np.exp(z[:,:65]-z[:,:65].max(1)[:,None]); candidate_probability/=candidate_probability.sum(1)[:,None]
        sig=np.maximum(.5,np.sqrt(np.sum(candidate_probability*(np.arange(65)[None,:]-k[:,None])**2,axis=1)))
        i=sr['index']; valid=a['valid'][i]; positive=valid&(a['lo'][i]<65)
        err=abs(k-(a['lo'][i]+a['weight'][i]))
        query=O.query_geometry(sr['frozen_selected_points']); query['raw_hw']=sr['raw_hw']; xy=query['candidate'][np.arange(84),k]; h,w=sr['raw_hw']
        inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        eligible=valid&query['valid']&inside&np.isin(np.arange(84)//7,supported)
        iscorrect=positive&(err<=2)
        scores.extend(s[eligible]); correct.extend(iscorrect[eligible])
        accepted=eligible&(s>=cal['confidence']['threshold']) if cal['confidence']['enabled'] else np.zeros(84,bool)
        ap=accepted&positive; accepted_errors.extend(err[ap]); ratios.extend(err[ap]/sig[ap])
        counts.update(known_eligible=int(eligible.sum()),known_accepted=int(accepted.sum()),known_correct=int((accepted&iscorrect).sum()),known_POSITIVE_accepted=int(ap.sum()),known_NONE_accepted=int((accepted&~positive).sum()))
        decoded=O.decode(query,z,cal); counts.update(source_frames=1,selected_queries=decoded['selected_queries'],lines=len(decoded['lines']),accepted_corners=len(decoded['corners']),frames_ge4=len(decoded['corners'])>=4,line_pair_hypotheses=decoded['diagnostics']['line_pair_hypotheses'])
        counts['unknown_IGNORE_accepted']+=sum(q['selected_xy'] is not None and not valid[q['query']] for q in decoded['queries'])
        lookup={line['edge']:line for line in decoded['lines']}
        for line in decoded['lines']:
            points=np.array(line['support_points']); radii=np.array(line['query_radii_px']); residual=points@line['normal']-line['offset']
            assert (abs(residual)<=radii+1e-10).all()
            assert len(set(line['queries']))>=3 and max(j%7 for j in line['queries'])-min(j%7 for j in line['queries'])>=2
            assert line['support_length_px']>=2*line['nominal_query_spacing_px']*(1-64*np.finfo(float).eps)
            weights=1/np.maximum(.5,radii)**2; center=np.average(points,axis=0,weights=weights); ss=(points-center)@line['tangent']; design=np.column_stack([ss,np.ones(len(ss))])
            cov=np.linalg.inv((design*weights[:,None]).T@design)*max(1.,float(np.mean((residual/np.maximum(.5,radii))**2)))
            np.testing.assert_allclose(cov,line['parameter_covariance'],rtol=1e-10,atol=1e-10)
            counts['independent_line_covariance_checks']+=1
        for candidate in decoded['diagnostics']['corner_candidates']:
            lines=[lookup[e] for e in candidate['edges']]; matrix=np.array([l['normal'] for l in lines]); inv=np.linalg.inv(matrix); vv=[]
            for line in lines:
                ss=float((np.array(candidate['xy'])-line['center'])@line['tangent']); v=np.array([ss,1.]); vv.append(float(v@np.array(line['parameter_covariance'])@v))
            cc=inv@np.diag(vv)@inv.T
            np.testing.assert_allclose(cc,candidate['covariance_px2'],rtol=1e-10,atol=1e-10)
            counts['independent_corner_covariance_checks']+=1
        assert len({c['id'] for c in decoded['corners']})==len(decoded['corners'])
        decodedrows.append(decoded)
    # Independent ascending-cutoff calculation; different from production descending scan.
    s=np.array(scores); good=np.array(correct,bool); passing=[]; z95=1.6448536269514722
    for threshold in np.unique(s[s>=.5]):
        selection=s>=threshold; n=int(selection.sum()); success=int(good[selection].sum()); p=success/n
        lower=(p+z95*z95/(2*n)-z95*math.sqrt((p*(1-p)+z95*z95/(4*n))/n))/(1+z95*z95/n)
        if lower>=.95: passing.append((float(threshold),n,success,lower))
    chosen=min(passing) if passing else None
    assert (chosen is not None)==cal['confidence']['enabled']
    if chosen:
        assert chosen[0]==cal['confidence']['threshold'] and chosen[1]==cal['confidence']['chosen']['accepted'] and chosen[2]==cal['confidence']['chosen']['correct']
        assert abs(chosen[3]-cal['confidence']['chosen']['Wilson_lower'])<1e-12
    queryscale=max(1.,float(np.quantile(ratios,.95,method='higher')))
    assert queryscale==cal['uncertainty']['query_scale']
    assert float(np.quantile(accepted_errors,.95,method='higher'))==cal['uncertainty']['query_error_q95_px']
    geometry=rows(d/'CALIBRATION_GEOMETRY_ROWS.jsonl.gz')
    assert len(geometry)==cal['geometry']['calibration_intersections']
    source_lookup={r['index']:r for r in source}
    for row in geometry:
        truth=[]
        for edge,ids in zip(row['edges'],row['supporting_query_ids']):
            qs=[source_lookup[row['index']]['queries'][j] for j in ids]; assert all(q['proposed_target']=='POSITIVE' and q['edge']==edge for q in qs)
            points=np.array([q['actual_target_uv'] for q in qs]); center=points.mean(0); _,_,v=np.linalg.svd(points-center,full_matrices=False); tangent=v[0]; normal=np.array([-tangent[1],tangent[0]])
            truth.append((normal,float(normal@center)))
        reference=np.linalg.solve([l[0] for l in truth],[l[1] for l in truth])
        np.testing.assert_allclose(reference,row['reference_xy'],rtol=1e-9,atol=1e-7)
        error=float(np.linalg.norm(np.array(row['predicted_xy'])-reference)); assert abs(error-row['error_px'])<1e-7
        counts['independent_actual_wire_reference_intersections']+=1
    cornerscale=max(1.,float(np.quantile([r['error_over_sigma'] for r in geometry],.95,method='higher')))
    extrap=max(1/6,float(np.quantile([r['true_corner_support_extrapolation_ratio'] for r in geometry],.95,method='higher')))
    assert cornerscale==cal['uncertainty']['corner_scale'] and extrap==cal['geometry']['max_extrapolation_ratio']
    return dict(complete=True,passed=True,calibration_sha256=sha(d/'CALIBRATION.json'),protocol_sha256=sha(d/'CALIBRATION_PROTOCOL.json'),
                verifier_sha256=sha(Path(__file__)),counts=dict(counts),known_CAL_conditional_precision=counts['known_correct']/counts['known_accepted'],
                least_restrictive_threshold=chosen[0] if chosen else None,unknown_acceptance_has_no_truth_claim=True,
                source_test_rows=0,real_rows=0,new_head_forwards=0,new_detector=0,new_features=0,new_training=0,new_PnP=0,new_rays=0,new_RGB=0,
                geometry_reference_limits='Actual-wire-supported line intersection, not independent physicalcorner/endpoints or real transfer certification',
                Wilson_limits='Correlated source queries and threshold selection; not a formal generalization guarantee')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output'); args=parser.parse_args(); result=verify()
    if args.output:
        with Path(args.output).open('x') as f: f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
