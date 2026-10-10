"""Independent stdlib scalar replay of the two new source-CAL calibrations.

Reads published CAL logits/diagnostics, READY source labels/NPZ and protocols.
Never imports calibration.py, observations.py, NumPy, Torch or a pose solver;
never opens private features/checkpoints, RGB, real scores, mesh or ray data.
ROLE reuse is checked byte-exact, with historical eight versus current zero
calls kept separate. Disabled confidence and fewer-than-ten geometry branches
are valid abstention outcomes, not checker failures.

Scalar floating-point checks use the pre-existing v2 review tolerances. Recorded
FP64 confidence/sigma are used for exact cutoff membership only after each is
independently checked against FP32 logits. This avoids claiming bit-identical
libm/BLAS arithmetic. Source-wire virtual intersections, Wilson precision and
local covariance surrogates do not certify physical corners or real transfer.
Run once only after completed calibration; existing receipts are protected.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import struct
import sys
import time
import zipfile
import zlib

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_three_head_observation_20261010_v7'
PRIVATE = Path('/tmp/pallet-three-head-observation-private-20261010-v7')
V2_DOC = REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
REPAIR = REPO / '_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
CORRECTED = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
INDICES = list(range(768, 896))
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
ROLE_FILES = ('CALIBRATION_PROTOCOL.json', 'CALIBRATION_START.json', 'CALIBRATION_ROWS.jsonl.gz',
              'CALIBRATION_THRESHOLD_SCAN.jsonl.gz', 'CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz',
              'CALIBRATION_GEOMETRY_ROWS.jsonl.gz', 'CALIBRATION.json', 'CALIBRATION_EXECUTION.json',
              'CALIBRATION_CHECKS.json')
Z95, EPS = 1.6448536269514722, 2.220446049250313e-16
CHECKPOINT_SHAS = dict(GEOMETRY_ONLY='d188dcc68bd8795c88232d5bf1b85259d684695723b96809017abd47d6ac009b',
    IMAGE_NO_ROLE='9c52a2e2036ee8f65ba2e191bdcbebe93ac3ecde60a6aa31e71a4ffbadc0f835',
    IMAGE_ROLE='882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d')


def read(path):
    with Path(path).open(encoding='utf-8') as stream: return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream: return [json.loads(line) for line in stream]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve()
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,
                sha256=sha(p), bytes=p.stat().st_size)


class Audit:
    def __init__(self):
        self.checks = 0
        self.max_differences = Counter()
        self.counts = Counter()

    def require(self, ok, label):
        self.checks += 1
        if not ok: raise AssertionError(label)

    def close(self, a, b, label, atol=1e-10, rtol=1e-10):
        if a is None or b is None:
            self.require(a is b, label + ': nullable values differ'); return
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            self.require(isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a)==len(b), label + ': shape')
            for x,y in zip(a,b): self.close(x,y,label,atol,rtol)
            return
        x,y = float(a),float(b)
        self.require(math.isfinite(x) and math.isfinite(y), label + ': nonfinite')
        difference=abs(x-y)
        self.max_differences[label] = max(self.max_differences[label], difference)
        self.require(difference <= atol + rtol*abs(y), label + ': scalar difference')

    def bound(self, path, expected, label):
        actual = binding(path)
        self.require(all(actual[k]==expected[k] for k in ('sha256','bytes')), label + ': binding')


def target_CAL(path, audit):
    """Read NPY members with stdlib; decode only fixed CAL scalar positions."""
    arrays={}
    codes={'b1':'?', 'i1':'b', 'u1':'B', 'i2':'h', 'u2':'H', 'i4':'i', 'u4':'I',
           'i8':'q', 'u8':'Q', 'f2':'e', 'f4':'f', 'f8':'d'}
    with zipfile.ZipFile(path) as archive:
        for key in ('lo','hi','weight','valid'):
            audit.require(archive.getinfo(key+'.npy').file_size < 2*1024*1024, 'bounded target NPY member')
            data=archive.read(key+'.npy')
            audit.require(data[:6]==b'\x93NUMPY', 'target NPY magic')
            version=data[6:8]
            size=2 if version==b'\x01\x00' else 4
            audit.require(version in (b'\x01\x00',b'\x02\x00',b'\x03\x00'), 'target NPY version')
            length=int.from_bytes(data[8:8+size],'little'); start=8+size+length
            header=ast.literal_eval(data[8+size:start].decode('utf-8' if version==b'\x03\x00' else 'latin1'))
            audit.require(header['shape']==(1024,84) and header['fortran_order'] is False, 'target NPY shape/order')
            descr=header['descr']; audit.require(isinstance(descr,str) and descr[1:] in codes, 'target scalar dtype')
            audit.require(descr==dict(lo='<i2',hi='<i2',weight='<f4',valid='|b1')[key], 'original prepared target dtype')
            fmt=('>' if descr[0]=='>' else '<')+codes[descr[1:]]; item=struct.calcsize(fmt)
            audit.require(len(data)-start==1024*84*item, 'target NPY exact byte size')
            arrays[key]={i:[struct.unpack_from(fmt,data,start+(i*84+j)*item)[0] for j in range(84)] for i in INDICES}
    return arrays


def source_CAL(path,audit):
    source=[]; total=0
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for i,line in enumerate(stream):
            total+=1
            if 768<=i<896: source.append(json.loads(line))
    audit.require(total==1024 and [r['index'] for r in source]==INDICES, 'fixed READY CAL128')
    audit.require(len({r['id'] for r in source})==len({r['family'] for r in source})==128, 'CAL family/ID uniqueness')
    audit.require(all(r['partition']=='calibration' and len(r['queries'])==84 for r in source), 'CAL-only labels')
    return source


def q95(values):
    if not values: return None
    return sorted(values)[math.ceil(.95*(len(values)-1))]


def wilson(correct,n):
    p=correct/n; z2=Z95*Z95
    return (p+z2/(2*n)-Z95*math.sqrt((p*(1-p)+z2/(4*n))/n))/(1+z2/n)


def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))
def sub(a,b): return [a[0]-b[0],a[1]-b[1]]
def length(a): return math.hypot(a[0],a[1])
def f32(value): return struct.unpack('<f',struct.pack('<f',value))[0]


def geometry(points):
    p=[[float('nan') if x is None else float(x) for x in xy] for xy in points]
    result=[]
    for edge,(a,b) in enumerate(EDGES):
        delta=sub(p[b],p[a]); span=length(delta)
        normal=[-delta[1]/max(span,1e-6),delta[0]/max(span,1e-6)]
        valid=all(math.isfinite(x) for x in delta) and span>1e-6 and not any(p[k]==[-1.,-1.] for k in (a,b))
        for j in range(1,8):
            u=j/8; center=[(1-u)*p[a][d]+u*p[b][d] for d in range(2)]
            result.append(dict(edge=edge,center=center,normal=normal,valid=valid,spacing=span/8))
    return result


def logit_stats(values):
    best=max(range(65),key=lambda k:values[k])
    margin=max(-700.,min(700.,values[best]-values[65]))
    score=1/(1+math.exp(-margin))
    probabilities=[math.exp(v-values[best]) for v in values[:65]]; total=math.fsum(probabilities)
    sigma=max(.5,math.sqrt(math.fsum(p*(k-best)**2 for k,p in enumerate(probabilities))/total))
    return best,score,sigma


def fit_line(points,radii):
    weights=[1/max(.5,r)**2 for r in radii]; total=math.fsum(weights)
    center=[math.fsum(w*p[d] for p,w in zip(points,weights))/total for d in range(2)]
    dx=[sub(p,center) for p in points]
    xx=math.fsum(w*d[0]*d[0] for w,d in zip(weights,dx))/total
    yy=math.fsum(w*d[1]*d[1] for w,d in zip(weights,dx))/total
    xy=math.fsum(w*d[0]*d[1] for w,d in zip(weights,dx))/total
    largest=(xx+yy+math.hypot(xx-yy,2*xy))/2
    if largest<=1e-12:return None
    angle=.5*math.atan2(2*xy,xx-yy); tangent=[math.cos(angle),math.sin(angle)]
    k=0 if abs(tangent[0])>=abs(tangent[1]) else 1
    if tangent[k]<0:tangent=[-x for x in tangent]
    normal=[-tangent[1],tangent[0]]; offset=dot(normal,center); ss=[dot(d,tangent) for d in dx]
    aa=math.fsum(w*s*s for w,s in zip(weights,ss)); bb=math.fsum(w*s for w,s in zip(weights,ss)); cc=total
    determinant=aa*cc-bb*bb
    if determinant<=1e-12:return None
    residual=[dot(p,normal)-offset for p in points]
    scale=max(1.,math.fsum((r/max(.5,rad))**2 for r,rad in zip(residual,radii))/len(points))
    covariance=[[cc/determinant*scale,-bb/determinant*scale],[-bb/determinant*scale,aa/determinant*scale]]
    return dict(normal=normal,offset=offset,tangent=tangent,center=center,
                support_interval_px=[min(ss),max(ss)],support_length_px=max(ss)-min(ss),
                parameter_covariance=covariance)


def build_lines(query,records):
    output=[]
    for edge in range(12):
        ids=[i for i,r in enumerate(records) if i//7==edge and r['selected_xy'] is not None]
        spacing=query[edge*7]['spacing']
        if len(ids)<3:continue
        points=[records[i]['selected_xy'] for i in ids]; radii=[records[i]['radius'] for i in ids]; hypotheses=[]
        for aa,bb in combinations(range(len(ids)),2):
            tangent=sub(points[bb],points[aa]); norm=length(tangent)
            if norm<=1e-6:continue
            normal=[-tangent[1]/norm,tangent[0]/norm]; offset=dot(normal,points[aa])
            residual=[abs(dot(p,normal)-offset) for p in points]
            selected=[j for j,r in enumerate(residual) if r<=radii[j]]
            if len(selected)<3 or max(ids[j]%7 for j in selected)-min(ids[j]%7 for j in selected)<2:continue
            coords=[dot(points[j],[x/norm for x in tangent]) for j in selected]; span=max(coords)-min(coords)
            if span<2*spacing*(1-64*EPS):continue
            key=(-len(selected),math.fsum((residual[j]/radii[j])**2 for j in selected)/len(selected),-span,ids[aa],ids[bb])
            hypotheses.append((key,selected))
        if not hypotheses:continue
        selected=min(hypotheses,key=lambda x:x[0])[1]
        line=fit_line([points[j] for j in selected],[radii[j] for j in selected])
        if line is None:continue
        selected=[j for j,p in enumerate(points) if abs(dot(p,line['normal'])-line['offset'])<=radii[j]]
        if len(selected)<3 or max(ids[j]%7 for j in selected)-min(ids[j]%7 for j in selected)<2:continue
        line=fit_line([points[j] for j in selected],[radii[j] for j in selected])
        if line is None or line['support_length_px']<2*spacing*(1-64*EPS):continue
        line.update(edge=edge,queries=[ids[j] for j in selected],spacing=spacing)
        output.append(line)
    return output


def intersection(a,b):
    x,y=a['normal']; u,v=b['normal']; det=x*v-y*u
    if abs(det)<=1e-6:return None
    xy=[(v*a['offset']-y*b['offset'])/det,(-u*a['offset']+x*b['offset'])/det]
    if not all(math.isfinite(c) for c in xy):return None
    variances=[]; gaps=[]
    for line in (a,b):
        s=dot(sub(xy,line['center']),line['tangent']); low,high=line['support_interval_px']; cov=line['parameter_covariance']
        variances.append(s*s*cov[0][0]+2*s*cov[0][1]+cov[1][1])
        gaps.append(max(low-s,s-high,0)/max(line['support_length_px'],1e-6))
    inv=[[v/det,-y/det],[-u/det,x/det]]
    cov=[[math.fsum(inv[i][k]*variances[k]*inv[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
    eigenmax=(cov[0][0]+cov[1][1]+math.hypot(cov[0][0]-cov[1][1],2*cov[0][1]))/2
    return dict(xy=xy,sigma=math.sqrt(max(0,eigenmax)),covariance=cov,
                extrapolation=max(gaps),spacing=min(a['spacing'],b['spacing']))


def dependencies(root):
    paths=dict(code=Path(__file__), ready_rows=REPAIR/'READY_SOURCE_TARGET_ROWS.jsonl.gz',
               ready_targets=REPAIR/'READY_PREPARED_TARGETS.npz', checkpoint_metadata=CORRECTED/'CHECKPOINT_METADATA.json',
               original_algorithm_protocol=V2_DOC/'CALIBRATION_PROTOCOL.json',
               original_algorithm=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/calibration.py',
               original_decoder=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py',
               original_model=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/model.py',
               protocol=root/'PROTOCOL.json',completion=root/'COMPLETION.json')
    for arm in ARMS:
        names=ROLE_FILES if arm=='IMAGE_ROLE' else ROLE_FILES[:-1]
        paths.update({arm+':'+name:root/arm/name for name in names})
    paths['ROLE:REUSE_RECEIPT.json']=root/'IMAGE_ROLE/REUSE_RECEIPT.json'
    paths.update({'original_ROLE:'+name:V2_DOC/name for name in ROLE_FILES})
    return paths


def provenance(root,audit):
    protocol,completion=read(root/'PROTOCOL.json'),read(root/'COMPLETION.json')
    audit.require(completion['complete'] and completion['inputs_before_after_equal'] and
                  completion['input_bindings']==protocol['inputs'], 'completed frozen source calibration')
    audit.bound(root/'PROTOCOL.json',completion['protocol'],'root source protocol')
    audit.require(protocol['indices']==INDICES and protocol['new_arms']==list(ARMS[:2]) and
                  protocol['reused_arm']=='IMAGE_ROLE', 'fixed source head scope')
    old=read(V2_DOC/'CALIBRATION_PROTOCOL.json')
    expected_policy={k:v for k,v in old.items() if k not in ('schema','inputs','arm')}
    audit.require(protocol['source_algorithm_policy']==expected_policy, 'exact unchanged source calibration policy')
    audit.bound(V2_DOC/'CALIBRATION_PROTOCOL.json',protocol['source_algorithm_protocol'],'original algorithm protocol')
    for key,path in (('ready_rows',REPAIR/'READY_SOURCE_TARGET_ROWS.jsonl.gz'),
                     ('ready_targets',REPAIR/'READY_PREPARED_TARGETS.npz'),
                     ('checkpoint_metadata',CORRECTED/'CHECKPOINT_METADATA.json'),
                     ('algorithm',REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/calibration.py'),
                     ('observations',REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py'),
                     ('original_model',REPO/'scripts/research/pallet_observation_refiner_20261009_v1/model.py')):
        audit.bound(path,protocol['inputs'][key],key)
    audit.require(completion['actual_new_head_forward_calls']==16 and completion['actual_new_head_image_exposures']==256
                  and completion['ROLE_new_head_forward_calls']==0, 'actual added head counts')
    for key in ('source_test_exposures','real_frames','new_training_updates','detector_calls','N3_calls','PnP_calls','rays','new_RGB','feature_recomputations'):
        audit.require(protocol[key]==completion[key]==0,'calibration forbidden scope '+key)
    metadata={r['arm']:r for r in read(CORRECTED/'CHECKPOINT_METADATA.json')['rows']}
    for arm in ARMS:
        info=completion['arms'][arm]; actual=completion['actual_counts'][arm]; calls=0 if arm=='IMAGE_ROLE' else 8
        audit.require(info['checkpoint']['sha256']==metadata[arm]['checkpoint']['sha256']==CHECKPOINT_SHAS[arm]
                      and metadata[arm]['steps']==3000 and metadata[arm]['parameters']==5890,'last arm checkpoint metadata')
        audit.require(actual['attempted']==actual['completed']==calls and actual['image_exposures']==16*calls
                      and info['new_head_forward_calls']==calls and info['new_head_image_exposures']==16*calls,'arm call count')
        directory=root/arm; cal=read(directory/'CALIBRATION.json'); execution=read(directory/'CALIBRATION_EXECUTION.json')
        audit.bound(directory/'CALIBRATION.json',info['calibration'],'deployed coefficients '+arm)
        audit.bound(directory/'CALIBRATION_EXECUTION.json',info['execution'],'head calibration execution '+arm)
        audit.require(execution['complete'] and execution['arm']==arm and execution['partition']=='calibration'
                      and execution['head_forward_calls']==8 and execution['head_image_exposures']==128,'per-arm historical/new execution')
        for key in ('source_test_exposures','real_frames','training_updates','detector_calls','PnP_calls','rays','new_RGB','feature_recomputations'):
            audit.require(execution[key]==0,'per-arm forbidden scope '+key)
        if arm=='IMAGE_ROLE':
            reuse=read(directory/'REUSE_RECEIPT.json'); audit.bound(directory/'REUSE_RECEIPT.json',info['reuse'],'ROLE reuse receipt')
            audit.require(reuse['complete'] and reuse['current_new_head_forward_calls']==reuse['current_new_image_exposures']==0
                          and reuse['historical_head_forward_calls']==8 and reuse['numerical_recalibration'] is False,'ROLE current/historical distinction')
            for name in ROLE_FILES:
                audit.bound(directory/name,protocol['inputs']['ROLE:'+name],'ROLE frozen bytes '+name)
                audit.bound(directory/name,binding(V2_DOC/name),'ROLE original bytes '+name)
                audit.bound(directory/name,reuse['files'][name],'ROLE reuse bound bytes '+name)
            audit.require(cal['protocol_sha256']==sha(V2_DOC/'CALIBRATION_PROTOCOL.json'),'ROLE protocol retained')
        else:
            own=read(directory/'CALIBRATION_PROTOCOL.json')
            audit.bound(directory/'CALIBRATION_PROTOCOL.json',execution['protocol'],'arm own protocol')
            audit.require(own['arm']==cal['arm']==cal['training_arm']==execution['training_arm']==arm and
                          own['indices']==INDICES and own['head_mode']==protocol['head_modes'][arm], 'arm mode/provenance')
            audit.require(own['source_algorithm_policy']==expected_policy and
                          cal['source_algorithm_protocol_sha256']==sha(V2_DOC/'CALIBRATION_PROTOCOL.json') and
                          cal['protocol_sha256']==sha(directory/'CALIBRATION_PROTOCOL.json'), 'own/original algorithm protocols')
            audit.require(cal['checkpoint']['sha256']==execution['checkpoint']['sha256']==CHECKPOINT_SHAS[arm], 'deployed head weight provenance')
    return protocol


def arm_check(root,arm,source,targets,audit):
    directory=root/arm; cal=read(directory/'CALIBRATION.json')
    raw=rows(directory/'CALIBRATION_ROWS.jsonl.gz'); querydiag=rows(directory/'CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz')
    stored_geometry=rows(directory/'CALIBRATION_GEOMETRY_ROWS.jsonl.gz'); scan=rows(directory/'CALIBRATION_THRESHOLD_SCAN.jsonl.gz')
    audit.require([r['index'] for r in raw]==[r['index'] for r in querydiag]==INDICES, 'raw/query CAL indices '+arm)
    for key,name in (('raw_rows','CALIBRATION_ROWS.jsonl.gz'),('threshold_scan','CALIBRATION_THRESHOLD_SCAN.jsonl.gz'),('geometry_rows','CALIBRATION_GEOMETRY_ROWS.jsonl.gz')):
        audit.bound(directory/name,cal[key],arm+' coefficient raw binding '+key)
    supported=sorted({q['edge'] for r in source for q in r['queries'] if q['proposed_target']=='POSITIVE'})
    audit.require(cal['supported_edges']==supported and cal['unsupported_edges']==sorted(set(range(12))-set(supported)),arm+' source coverage')
    buckets={}; accepted_errors=[]; ratios=[]; counts=Counter(); working=[]
    for sr,rr,qd in zip(source,raw,querydiag):
        index=sr['index']; h,w=sr['raw_hw']; q=geometry(sr['frozen_selected_points'])
        audit.require(sr['id']==rr['id']==qd['id'] and sr['family']==rr['family'] and rr['arm']==arm and
                      rr['partition']=='calibration' and rr['logits_shape']==[84,66],arm+' raw row identity')
        semantic=hashlib.sha256(json.dumps(sr,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        audit.require(semantic==rr['source_ready_row_semantic_sha256'],arm+' READY semantic binding')
        binary=zlib.decompress(base64.b64decode(rr['logits_fp32_zlib_base64']))
        audit.require(len(binary)==84*66*4 and hashlib.sha256(binary).hexdigest()==rr['raw_logits_sha256'],arm+' raw logit binding')
        z=struct.unpack('<5544f',binary); records=[]
        for j in range(84):
            audit.require(all(math.isfinite(v) for v in z[j*66:(j+1)*66]),arm+' finite logits')
            best,score,sigma=logit_stats(z[j*66:(j+1)*66]); audit.require(qd['candidate'][j]==best,arm+' first MAP bin')
            audit.close(score,qd['confidence'][j],'confidence',atol=1e-12,rtol=0)
            audit.close(sigma,qd['sigma_mode_px'][j],'conditional_sigma')
            # Membership uses these already verified recorded FP64 primitives.
            score,sigma=qd['confidence'][j],qd['sigma_mode_px'][j]
            xy=[q[j]['center'][d]+(best-32)*q[j]['normal'][d] for d in range(2)]
            inside=all(math.isfinite(c) for c in xy) and 0<=xy[0]<w and 0<=xy[1]<h
            valid=bool(targets['valid'][index][j]); positive=valid and targets['lo'][index][j]<65
            # Original NumPy int16 lo + float32 weight first produces float32;
            # subtracting its value from int64 argmax then promotes to float64.
            target=f32(targets['lo'][index][j]+targets['weight'][index][j])
            error=abs(best-target)
            eligible=valid and q[j]['valid'] and inside and j//7 in supported; correct=positive and error<=2
            accepted=eligible and cal['confidence']['enabled'] and score>=cal['confidence']['threshold']
            audit.require(qd['known_eligible'][j]==eligible and qd['correct_within2px'][j]==correct and
                          qd['accepted'][j]==accepted,arm+' known eligibility/correctness/adoption')
            if eligible:
                bucket=buckets.setdefault(score,[0,0]); bucket[0]+=1; bucket[1]+=int(correct)
            counts.update(valid=int(eligible),accepted=int(accepted),correct=int(accepted and correct),
                          accepted_POSITIVE=int(accepted and positive),accepted_NONE=int(accepted and not positive))
            if accepted and positive:accepted_errors.append(error);ratios.append(error/sigma)
            decoded=q[j]['valid'] and inside and j//7 in supported and cal['confidence']['enabled'] and score>=cal['confidence']['threshold']
            records.append(dict(selected_xy=xy if decoded else None,radius=sigma*cal['uncertainty']['query_scale']))
            counts['selected_queries']+=int(decoded); counts['unknown_IGNORE_accepted']+=int(decoded and not valid)
        working.append((sr,q,records));counts['source_frames']+=1
    expected_scan=[]; n=good=0
    for cutoff in sorted(buckets,reverse=True):
        if cutoff<.5:break
        add,success=buckets[cutoff]; n+=add; good+=success
        expected_scan.append(dict(threshold=cutoff,accepted=n,correct=good,incorrect=n-good,precision=good/n,Wilson_lower=wilson(good,n)))
    audit.require(len(expected_scan)==len(scan),arm+' full threshold scan size')
    for a,b in zip(expected_scan,scan):
        audit.require(all(a[k]==b[k] for k in ('threshold','accepted','correct','incorrect')),arm+' full threshold scan memberships')
        for k in ('precision','Wilson_lower'):audit.close(a[k],b[k],arm+' scan '+k,atol=1e-12,rtol=0)
    passing=[r for r in expected_scan if r['Wilson_lower']>=.95]; chosen=min(passing,key=lambda r:r['threshold']) if passing else None
    audit.require(cal['confidence']['enabled']==(chosen is not None),arm+' disabled/attained precision branch')
    if chosen:
        audit.require(cal['confidence']['threshold']==chosen['threshold'],arm+' least restrictive cutoff')
        for k in ('accepted','correct','incorrect'):audit.require(cal['confidence']['chosen'][k]==chosen[k],arm+' cutoff counts')
        audit.close(cal['confidence']['chosen']['Wilson_lower'],chosen['Wilson_lower'],arm+' chosen Wilson',atol=1e-12,rtol=0)
    else:audit.require(cal['confidence']['threshold'] is None and cal['confidence']['chosen'] is None,arm+' abstention coefficients')
    audit.require(cal['confidence']['known_query_counts']=={k:counts[k] for k in ('valid','accepted','correct','accepted_POSITIVE','accepted_NONE')},arm+' known counts')
    audit.close(cal['uncertainty']['query_scale'],max(1.,q95(ratios) or 1.),arm+' query scale')
    audit.close(cal['uncertainty']['query_error_q95_px'],q95(accepted_errors),arm+' query error q95')
    expected_geometry=[]
    for sr,q,records in working:
        lines=build_lines(q,records); lookup={l['edge']:l for l in lines};counts['scalar_reconstructed_lines']+=len(lines)
        for corner in range(8):
            incident=[e for e,pair in enumerate(EDGES) if corner in pair and e in lookup]
            for ea,eb in combinations(incident,2):
                candidate=intersection(lookup[ea],lookup[eb])
                if candidate is None:continue
                truths=[]; eligible=True
                for edge in (ea,eb):
                    queries=[sr['queries'][i] for i in lookup[edge]['queries']]
                    if any(x['proposed_target']!='POSITIVE' for x in queries):eligible=False;break
                    truth=fit_line([x['actual_target_uv'] for x in queries],[1.]*len(queries))
                    if truth is None:eligible=False;break
                    truth.update(edge=edge,spacing=lookup[edge]['spacing']);truths.append(truth)
                if not eligible:continue
                reference=intersection(*truths)
                if reference is None:continue
                h,w=sr['raw_hw']; point=reference['xy']
                if not(0<=point[0]<w and 0<=point[1]<h):continue
                error=length(sub(candidate['xy'],point)); gap=[]
                for edge in (ea,eb):
                    line=lookup[edge];s=dot(sub(point,line['center']),line['tangent']);lo,hi=line['support_interval_px']
                    gap.append(max(lo-s,s-hi,0.)/line['support_length_px'])
                expected_geometry.append(dict(index=sr['index'],id=sr['id'],corner=corner,edges=[ea,eb],
                    supporting_query_ids=[lookup[e]['queries'] for e in (ea,eb)],reference_xy=point,
                    predicted_xy=candidate['xy'],error_px=error,sigma_px=candidate['sigma'],
                    error_over_sigma=error/max(candidate['sigma'],1e-12),true_corner_support_extrapolation_ratio=max(gap)))
    audit.require(len(expected_geometry)==len(stored_geometry)==cal['geometry']['calibration_intersections'],arm+' complete geometry population')
    for expected,stored in zip(expected_geometry,stored_geometry):
        for key in ('index','id','corner','edges','supporting_query_ids'):audit.require(expected[key]==stored[key],arm+' geometry identity '+key)
        audit.require(stored['reference_from_certified_actual_wire_queries'] is True and
                      stored['physical_corner_ownership_independently_proved'] is False,arm+' source reference scope')
        for key in ('reference_xy','predicted_xy','error_px'):audit.close(expected[key],stored[key],arm+' geometry '+key,atol=1e-7,rtol=1e-9)
        for key in ('sigma_px','error_over_sigma','true_corner_support_extrapolation_ratio'):
            audit.close(expected[key],stored[key],arm+' geometry '+key,atol=1e-10,rtol=1e-9)
    enough=len(expected_geometry)>=10
    audit.require(cal['geometry']['enabled']==enough,arm+' ten-intersection minimum')
    audit.close(cal['uncertainty']['corner_scale'],max(1.,q95([r['error_over_sigma'] for r in expected_geometry]) or 1.) if enough else 1.,arm+' corner scale',atol=1e-10,rtol=1e-9)
    audit.close(cal['geometry']['max_extrapolation_ratio'],max(1/6,q95([r['true_corner_support_extrapolation_ratio'] for r in expected_geometry]) or 0.) if enough else 1/6,arm+' extrapolation cap',atol=1e-10,rtol=1e-9)
    if enough:audit.close(cal['geometry']['reference_error_q95_px'],q95([r['error_px'] for r in expected_geometry]),arm+' geometry error q95',atol=1e-7,rtol=1e-9)
    else:audit.require(cal['geometry']['reason']=='fewer than10supported calibration intersections',arm+' geometry abstention reason')
    return dict(counts=dict(counts),geometry_intersections=len(expected_geometry),geometry_enabled=enough,
        confidence_enabled=chosen is not None,least_restrictive_threshold=chosen['threshold'] if chosen else None,
        known_CAL_conditional_precision=counts['correct']/counts['accepted'] if counts['accepted'] else None,
        own_coefficients_independently_recalculated=True)


def output_guard(path):
    p=Path(path).absolute()
    if any(x.is_symlink() for x in (p,*p.parents)):raise AssertionError('symlink checker output/ancestor')
    out=p.resolve()
    if not(out.parent==DOC.resolve() or out.is_relative_to(PRIVATE.resolve())):raise AssertionError('checker output must be new DOC/private subtree')
    if out.exists():raise FileExistsError('preserve completed/interrupted checker receipt')
    if out.suffix!='.json' or out.name in ('PROTOCOL.json','COMPLETION.json'):raise AssertionError('new checker receipt basename required')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-root',default=str(DOC/'source_calibration'))
    parser.add_argument('--output',default=str(DOC/'SOURCE_CALIBRATION_CHECKS.json'))
    args=parser.parse_args(); output=output_guard(args.output); root=Path(args.calibration_root).resolve()
    audit=Audit();begin=time.monotonic();result=dict(complete=False,passed=False,verifier=binding(Path(__file__)),arms={})
    before={}
    try:
        paths=dependencies(root);before={k:binding(p) for k,p in paths.items()}
        provenance(root,audit)
        source=source_CAL(REPAIR/'READY_SOURCE_TARGET_ROWS.jsonl.gz',audit)
        targets=target_CAL(REPAIR/'READY_PREPARED_TARGETS.npz',audit)
        for arm in ARMS[:2]:result['arms'][arm]=arm_check(root,arm,source,targets,audit)
        result['arms']['IMAGE_ROLE']=dict(byte_exact_files=9,current_head_calls=0,historical_head_calls=8,
                                         numerical_recalibration=False)
        after={k:binding(p) for k,p in paths.items()};audit.require(before==after,'all public checker inputs preserved')
        result.update(complete=True,passed=True,inputs_before_after_equal=True)
    except Exception as error:
        result['failure']=dict(type=type(error).__name__,message=str(error))
    result.update(schema='independent_scalar_three_head_source_calibration_checks_v7',input_bindings=before,
        scalar_checks=audit.checks,max_absolute_differences=dict(audit.max_differences),
        elapsed_seconds=time.monotonic()-begin,
        new_head_forwards=0,new_detector_calls=0,new_N3_calls=0,new_PnP_calls=0,new_rays=0,
        new_training_updates=0,new_RGB=0,real_rows_read=0,source_test_rows_decoded=0,
        targets='NPZ members decompressed whole; only fixed CAL768:896 scalar positions decoded',
        floating_point_membership='recorded FP64 confidence/sigma validated independently against raw FP32 logits before cutoff/radius use; no bit-identical libm/BLAS claim',
        geometry_limits='WeightedTLS/support-line virtual intersections; no independent physicalcorner ownership, first-surface or RGB visibility validation',
        Wilson_limits='Correlated CAL queries and scanned threshold; known-label conditional precision, not independent transfer guarantee',
        execution_limits='Stored checkpoint/source execution bindings only; no private tensor, feature or GPU-forward authenticity replay')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SOURCE_CALIBRATION_CHECKS',result['passed'],audit.checks,flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
