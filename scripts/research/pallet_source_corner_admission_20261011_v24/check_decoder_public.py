"""Independent stdlib check of saved decoder arithmetic and decisions."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def read(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def close(a,b):
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    else:assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8),(a,b)


def main():
    folder=Path(sys.argv[1]);frames=read(folder/'DECODED_FRAMES.jsonl.gz');decisions=read(folder/'DECISIONS.jsonl.gz')
    summary=json.loads((folder/'SUMMARY.json').read_text());checks=json.loads((folder/'CHECKS.json').read_text())
    assert len(frames)==384 and len(decisions)==573 and checks['passed']
    assert hashlib.sha256((folder/'DECODED_FRAMES.jsonl.gz').read_bytes()).hexdigest()==checks['frames_sha256']
    assert hashlib.sha256((folder/'DECISIONS.jsonl.gz').read_bytes()).hexdigest()==checks['decisions_sha256']
    lookup={};selected_keys=set();counts={a:collections.Counter() for a in summary['counts']};line_checks=0;candidate_checks=0
    cal={a:json.loads((folder/('CALIBRATION_'+a+'.json')).read_text()) for a in counts}
    raw_hw=json.loads((folder/'SOURCE_RAW_HW.json').read_text())['raw_hw_by_id']
    for r in frames:
        a=r['arm'];c=counts[a];lines={l['edge']:l for l in r['lines']}
        assert len(lines)==len(r['lines'])
        for l in lines.values():
            qids=l['queries'];assert len(qids)>=3 and max(x%7 for x in qids)-min(x%7 for x in qids)>=2
            assert all(q//7==l['edge'] for q in qids)
            assert l['support_length_px']>=2*l['nominal_query_spacing_px']*(1-64*sys.float_info.epsilon)
            points=l['support_points'];radii=l['query_radii_px'];weights=[1/max(.5,v)**2 for v in radii]
            center=[math.fsum(w*p[j] for w,p in zip(weights,points))/math.fsum(weights) for j in (0,1)]
            close(center,l['center'])
            ss=[dot([x-y for x,y in zip(p,center)],l['tangent']) for p in points]
            info=[[math.fsum(w*s*s for w,s in zip(weights,ss)),math.fsum(w*s for w,s in zip(weights,ss))],
                  [math.fsum(w*s for w,s in zip(weights,ss)),math.fsum(weights)]]
            det=info[0][0]*info[1][1]-info[0][1]*info[1][0];assert det>1e-12
            residual=[dot(p,l['normal'])-l['offset'] for p in points]
            scale=max(1.,math.fsum((v/max(.5,rr))**2 for v,rr in zip(residual,radii))/len(residual))
            covariance=[[scale*info[1][1]/det,-scale*info[0][1]/det],[-scale*info[1][0]/det,scale*info[0][0]/det]]
            close(covariance,l['parameter_covariance']);line_checks+=1
        accepted=[]
        for z in r['corner_candidates']:
            la,lb=[lines[e] for e in z['edges']];na,nb=la['normal'],lb['normal']
            det=na[0]*nb[1]-na[1]*nb[0];assert abs(det)>1e-6
            close(abs(det),z['absolute_normal_determinant'])
            inv=[[nb[1]/det,-na[1]/det],[-nb[0]/det,na[0]/det]]
            xy=[dot(v,[la['offset'],lb['offset']]) for v in inv];close(xy,z['xy'])
            variances=[];gaps=[]
            for l in (la,lb):
                s=dot([x-y for x,y in zip(xy,l['center'])],l['tangent']);lo,hi=l['support_interval_px']
                gaps.append(max(lo-s,s-hi,0)/max(l['support_length_px'],1e-6))
                h=[s,1.];C=l['parameter_covariance'];variances.append(dot(h,[dot(v,h) for v in C]))
            close(gaps,z['extrapolation_ratios'])
            cv=[[math.fsum(inv[i][k]*variances[k]*inv[j][k] for k in (0,1)) for j in (0,1)] for i in (0,1)]
            close(cv,z['covariance_px2'])
            eigen=(cv[0][0]+cv[1][1]+math.sqrt((cv[0][0]-cv[1][1])**2+4*cv[0][1]*cv[1][0]))/2
            sigma=math.sqrt(max(0,eigen));close(sigma,z['sigma_px'])
            radius=sigma*cal[a]['uncertainty']['corner_scale'];close(radius,z['radius_px'])
            spacing=min(la['nominal_query_spacing_px'],lb['nominal_query_spacing_px']);close(spacing,z['nominal_query_spacing_px'])
            normal=radius/max(spacing,1e-6);close(normal,z['normalized_uncertainty'])
            cap=cal[a]['geometry']['max_extrapolation_ratio'];h,w=raw_hw[r['id']]
            if not cal[a]['geometry']['enabled']:reason='SOURCE_CORNER_GEOMETRY_UNCALIBRATED'
            elif not(0<=xy[0]<w and 0<=xy[1]<h):reason='CORNER_OUTSIDE_IMAGE'
            elif max(gaps)>cap+64*sys.float_info.epsilon*max(1,max(gaps),cap):reason='EXCESSIVE_SUPPORT_EXTRAPOLATION'
            elif normal>1.:reason='CORNER_UNCERTAINTY_EXCEEDS_ONE_QUERY_SPACING'
            else:reason='CALIBRATED_INCIDENT_LINE_INTERSECTION'
            assert reason==z['reason'] and z['accepted']==(reason=='CALIBRATED_INCIDENT_LINE_INTERSECTION')
            if z['accepted']:accepted.append(z)
            lookup[(a,r['id'],z['id'],tuple(z['edges']))]=z
            candidate_checks+=1
        selected={}
        for z in accepted:
            key=lambda v:(v['radius_px'],v['max_extrapolation_ratio'],v['edges'])
            if z['id'] not in selected or key(z)<key(selected[z['id']]):selected[z['id']]=z
        assert sorted((z['id'],tuple(z['edges'])) for z in r['corners'])==sorted((k,tuple(z['edges'])) for k,z in selected.items())
        selected_keys.update((a,r['id'],k,tuple(z['edges'])) for k,z in selected.items())
        c['frames']+=1;c['selected_queries']+=len(r['selected_query_ids']);c['lines']+=len(lines)
        c['accepted_corners']+=len(r['corners']);c['frames_ge4']+=len(r['corners'])>=4
        c['all_candidate_pairs']+=len(r['corner_candidates']);c['line_pair_hypotheses']+=r['line_pair_hypotheses']
    for d in decisions:
        a=d['arm'];c=counts[a];z=lookup[(a,d['id'],d['corner'],tuple(d['edges']))]
        assert z==d['final_decoder_candidate'] and d['physical_corner_ownership']=='UNKNOWN'
        assert d['selected_decoder_corner']==((a,d['id'],d['corner'],tuple(d['edges'])) in selected_keys)
        c['conditional_rows']+=1;c['conditional_accepted_pair']+=z['accepted'];c['conditional_selected_corner']+=d['selected_decoder_corner']
        if d['saved_error_px']>8:
            c['error_gt8px:'+z['reason']]+=1;c['error_gt8px:selected_decoder_corner']+=d['selected_decoder_corner']
    for a,c in counts.items():
        assert all(summary['counts'][a][k]==v for k,v in c.items())
    four=summary['previously_unknown_four_rows'];assert len(four)==4 and all(d['selected_decoder_corner'] for d in four)
    result=dict(passed=True,frames=384,conditional_rows=573,independent_line_covariance_checks=line_checks,
        independent_intersection_covariance_and_gate_checks=candidate_checks,
        four_admitted_SOURCE_candidates_verified=True,new_head=0,new_PnP=0,new_pose_scoring=0,
        scope='Saved decoder arithmetic and decisions; no upstream supervision or physical ownership replay.')
    print(json.dumps(result))


if __name__=='__main__':
    main()
