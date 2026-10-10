"""Independent standard-library verification of recorded corner mechanisms.

Read public saved rows only. Reconstruct the 2x2 line intersection and error
decomposition with explicit inverse arithmetic, without NumPy or experiment
imports. This certifies proxy-relative arithmetic, not physical ownership.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
NAME='pallet_corner_mechanism_audit_20261010_v1'
BOUNDARY='pallet_boundary_corner_refiner_20261010_v2'
PRIMARY='N3_VALIDATED_ROLE'
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def require(value,message):
    if not value:raise ValueError(message)


def scalar(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def sub(a,b):return [x-y for x,y in zip(a,b)]


def same(a,b,label,atol=1e-7,rtol=1e-9):
    if isinstance(a,dict) and isinstance(b,dict):
        require(set(a)==set(b),label+': dictionary keys')
        for key in a:same(a[key],b[key],label+'/'+key,atol,rtol)
    elif isinstance(a,list) and isinstance(b,list):
        require(len(a)==len(b),label+': length')
        for x,y in zip(a,b):same(x,y,label,atol,rtol)
    elif scalar(a) or scalar(b):
        require(scalar(a) and scalar(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b)),label+': arithmetic mismatch')
    else:require(a==b,label+': value mismatch')


def solve(A,b):
    a,c=A[0];d,e=A[1];det=a*e-c*d;require(abs(det)>1e-12,'singular normal matrix')
    return [(e*b[0]-c*b[1])/det,(-d*b[0]+a*b[1])/det]


def condition(A):
    a,b=A[0];c,d=A[1];aa=a*a+c*c;cc=b*b+d*d;off=a*b+c*d
    delta=math.hypot(aa-cc,2*off);hi=(aa+cc+delta)/2;lo=(aa+cc-delta)/2
    require(lo>0,'singular normal matrix condition');return math.sqrt(hi/lo)


def quantile(values,q):
    if not values:return None
    v=sorted(values);t=(len(v)-1)*q;lo=int(math.floor(t));hi=int(math.ceil(t));return v[lo]+(v[hi]-v[lo])*(t-lo)


def describe(values):
    require(all(scalar(v) for v in values),'nonfinite distribution')
    n=len(values);mean=math.fsum(values)/n if n else None
    return dict(n=n,mean=mean,sample_variance=math.fsum((v-mean)**2 for v in values)/(n-1) if n>1 else None,
                median=quantile(values,.5),P90=quantile(values,.9),maximum=max(values) if n else None)


def summarize(rows):
    fields=dict(error_px='error_px',N3_error_px='N3_error_px',normal_displacement_norm_px='normal_displacement_norm_px',
                amplification='amplification',normal_matrix_condition='normal_matrix_condition',reference_error_over_radius='error_over_recorded_radius')
    out=dict(count=len(rows),human_states=dict(Counter(r['human_state'] for r in rows)))
    out.update({key:describe([r[field] for r in rows]) for key,field in fields.items()})
    out.update(delta_candidate_minus_N3_px=describe([r['error_px']-r['N3_error_px'] for r in rows]),
               max_line_support_rms_px=describe([max(r['line_support_rms_px']) for r in rows]),
               error_within_recorded_radius=sum(r['error_px']<=r['recorded_radius_px'] for r in rows),
               error_within_existing8px=sum(r['error_px']<=8. for r in rows),
               improves_N3=sum(r['error_px']<r['N3_error_px']-1e-9 for r in rows),
               harms_N3=sum(r['error_px']>r['N3_error_px']+1e-9 for r in rows))
    return out


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path)
    args=p.parse_args();root=args.root.resolve();doc=root/'_docs/experiments'/NAME;input_doc=root/'_docs/experiments'/BOUNDARY
    target=Path(args.output or doc/'MECHANISM_CHECKS.json').absolute()
    require(not target.exists() and not target.is_symlink(),'preserve existing output receipt');target=target.resolve()
    if target.is_relative_to(root):require(target.parent==doc and target.name.startswith('MECHANISM_CHECKS') and target.suffix=='.json','repository output must be an exclusive own mechanism receipt')
    for protected in (Path('/home/minjae/Documents/github/pallet-pose'),Path('/home/minjae/Documents/github/pallet-pose-handoff-20261006')):
        require(not target.is_relative_to(protected),'output overlaps original source/baseline')
    used={};checks=[];started=time.monotonic()

    def track(path):
        path=Path(path).resolve();require(path.is_relative_to(root) and path.is_file(),'missing public mechanism input')
        used[str(path.relative_to(root))]=dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size);return path
    def read(path):
        with track(path).open() as f:return json.load(f)
    def rows(path):
        with gzip.open(track(path),'rt') as f:return [json.loads(line) for line in f if line.strip()]
    def bound(item):
        path=track(root/item['path']);require(sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'bound mechanism input drift '+path.name)

    maximum_error=0.;computed=[];stored_count=0;ratios=[]
    try:
        protocol=read(doc/'PROTOCOL.json');result=read(doc/'RESULTS.json')
        require(protocol['schema']=='frozen_corner_mechanism_audit_v1' and result['schema']=='actual_frozen_corner_mechanism_results_v1' and result['complete'],'mechanism schema/completion')
        for item in list(protocol['inputs'].values())+[protocol['code'],result['protocol'],result['rows']]:bound(item)
        require(result['input_bindings_unchanged'] and all(result[k]==0 for k in ('actual_detector_calls','actual_head_calls','actual_PnP_calls','actual_new_training_updates','actual_RGB_generation','threshold_or_model_changes')),'mechanism execution scope differs')
        checks.append(dict(name='protocol_and_saved_input_bindings',status='PASS'))
        observations=rows(input_doc/'OBSERVATIONS.jsonl.gz');posthoc={r['id']:r for r in rows(input_doc/'POSTHOC_ROWS.jsonl.gz') if r['method']==PRIMARY}
        require(len(observations)==len(posthoc)==245 and {r['id'] for r in observations}==set(posthoc) and Counter(r['difficulty_label'] for r in posthoc.values())==dict(clean=153,moderate=92),'eligible245 primary identity')
        stored=rows(doc/'ROWS.jsonl.gz');by_key={(r['id'],r['corner']):r for r in stored};require(len(by_key)==len(stored)==482,'482 unique mechanism rows')
        unknown=0
        for observation in observations:
            frame=posthoc[observation['id']];lines={l['edge']:l for l in observation['lines']};evidence={c['id']:c for c in frame['observations']['boundary_corner_evidence']}
            for corner in observation['corners']:
                k=corner['id'];ev=evidence[k]
                if not ev['reference_available'] or ev['N3_error_px'] is None:unknown+=1;continue
                r=by_key[frame['id'],k];line=[lines[e] for e in corner['edges']]
                require(all(k in EDGES[e] for e in corner['edges']) and all(l['endpoints']==list(EDGES[l['edge']]) for l in line),'native corner/incident semantic edge graph')
                A=[l['normal'] for l in line];b=[l['offset'] for l in line]
                for n in A:same(norm(n),1.,'unit normal',1e-10,0.)
                ref=frame['reference_native_points_px'][k];xy=corner['xy'];n3=frame['fixed_N3_native_points_px'][k]
                displacement=[v-dot(n,ref) for v,n in zip(b,A)];reconstruction=solve(A,displacement);error_vector=sub(xy,ref)
                identity_error=max(abs(x-y) for x,y in zip(reconstruction,error_vector));maximum_error=max(maximum_error,identity_error);require(identity_error<1e-7,'2x2 exact identity fails')
                error=norm(error_vector);n3_error=norm(sub(n3,ref));normal_norm=norm(displacement)
                same(error,ev['error_px'],'saved proxy corner error',1e-9);same(n3_error,ev['N3_error_px'],'saved N3 proxy error',1e-9)
                recomputed=dict(r)
                updates=dict(id=frame['id'],session=frame['session'],difficulty=frame['difficulty_label'],corner=k,edges=corner['edges'],human_state=frame['human_states_native'][k],
                             reference_xy=ref,candidate_xy=xy,native_N3_xy=n3,error_px=error,N3_error_px=n3_error,normal_displacement_px=displacement,
                             normal_displacement_norm_px=normal_norm,amplification=error/normal_norm if normal_norm>1e-12 else 0.,
                             normal_matrix_condition=condition(A),absolute_normal_determinant=abs(A[0][0]*A[1][1]-A[0][1]*A[1][0]),
                             line_support_rms_px=[l['residual_rms_px'] for l in line],line_support_length_px=[l['support_length_px'] for l in line],line_query_ids=[l['queries'] for l in line],
                             extrapolation_ratios=corner['extrapolation_ratios'],recorded_radius_px=corner['radius_px'],error_over_recorded_radius=error/corner['radius_px'],
                             admitted=ev['admitted'],selected_for_hybrid=ev['selected_for_hybrid'],displayed=ev['displayed_as_boundary_observation'],physical_edge_ownership_proved=False)
                for key,value in updates.items():same(r[key],value,'mechanism row '+key)
                for l in line:
                    residual=[dot(p,l['normal'])-l['offset'] for p in l['support_points']]
                    same(l['residual_rms_px'],math.sqrt(math.fsum(v*v for v in residual)/len(residual)),'support RMS from coordinates')
                require(r['algebra_identity_error_px']<1e-7,'stored identity precision');recomputed.update(updates);computed.append(recomputed)
        require(len(computed)==482 and unknown==result['unknown_reference_corners']==0 and result['frames']==245 and result['computed_rows']==482,'computed population/unknown reference scope')
        stored_max=max(r['algebra_identity_error_px'] for r in stored);same(result['maximum_algebra_identity_error_px'],stored_max,'stored maximum identity error',1e-20,1e-12)
        checks.append(dict(name='all482_independent_2x2_inverse_and_native_graph',status='PASS',independent_maximum_identity_error_px=maximum_error,stored_maximum_identity_error_px=stored_max,physical_ownership_certified=False))
        scopes=dict(computed=computed,admitted=[r for r in computed if r['admitted']],selected_for_hybrid=[r for r in computed if r['selected_for_hybrid']],
                    displayed=[r for r in computed if r['displayed']],selected_improved=[r for r in computed if r['selected_for_hybrid'] and r['error_px']<r['N3_error_px']-1e-9],
                    selected_worsened=[r for r in computed if r['selected_for_hybrid'] and r['error_px']>r['N3_error_px']+1e-9])
        require(set(scopes)==set(result['scope']) and [len(v) for v in scopes.values()]==[482,447,423,395,186,237],'mechanism frozen scope counts')
        for name,values in scopes.items():same(result['scope'][name],summarize(values),'mechanism summary '+name)
        checks.append(dict(name='six_scope_moments_linear_quantiles_and_237_423_join',status='PASS',counts={k:len(v) for k,v in scopes.items()},sample_variance_ddof=1))
        cal=read(input_doc/'CALIBRATION.json');source=rows(input_doc/'CALIBRATION_GEOMETRY_ROWS.jsonl.gz')
        for r in source:
            error=norm(sub(r['predicted_xy'],r['reference_xy']));same(error,r['error_px'],'CAL saved virtual error',1e-8)
            require(r['reference_from_certified_actual_wire_queries'] and r['physical_corner_ownership_independently_proved'] is False,'CAL virtual ownership declaration')
            ratios.append(error/(r['sigma_px']*cal['uncertainty']['corner_scale']))
        same(result['source_virtual_corner_calibration'],dict(count=len(ratios),error_over_recorded_radius=describe(ratios),within_recorded_radius=sum(v<=1 for v in ratios),physical_corner_ownership_independently_proved=False),'source virtual radius coverage')
        require(len(ratios)==348 and sum(v<=1 for v in ratios)==331 and result['scope']['selected_for_hybrid']['error_within_recorded_radius']==188,'source331/348 versus selected188/423')
        checks.append(dict(name='source_virtual348_and_real_proxy423_radius_counts',status='PASS',source_within=331,source_n=348,real_proxy_within=188,real_proxy_n=423,physical_or_transfer_guarantee=False))
        expected=[]
        for name,A,b in [('orthogonal_displacement',[[1.,0.],[0.,1.]],[1.,1.]),('nearly_parallel_displacement',[[1.,0.],[math.cos(.01),math.sin(.01)]],[1.,-1.]),('perfect_support_biased_corner',[[1.,0.],[0.,1.]],[3.,4.])]:
            x=solve(A,b);require(max(abs(dot(n,x)-v) for n,v in zip(A,b))<1e-10,'toy line equation')
            expected.append(dict(name=name,line_normal_displacement_px=b,intersection_error_px=norm(x),support_residual_can_equal_zero=True,amplification=norm(x)/norm(b)))
        same(result['analytic_controls'],expected,'analytic control arithmetic')
        checks.append(dict(name='three_toy_math_controls',status='PASS',new_pose_or_model_calls=0))
        require(all(sha(root/item['path'])==item['sha256'] for item in used.values()),'public input changed while reviewing')
        checks.append(dict(name='all_input_bytes_preserved',status='PASS',files=len(used)))
    except Exception as exc:
        checks.append(dict(name='independent_mechanism_review',status='FAIL',error=type(exc).__name__+': '+str(exc)))
    passed=all(c['status']=='PASS' for c in checks)
    receipt=dict(schema='independent_stdlib_corner_mechanism_checks_v1',complete=passed,passed=passed,checks=checks,
                 verified_inputs=list(used.values()),verifier=dict(path=str(Path(__file__).relative_to(root)),sha256=sha(__file__),bytes=Path(__file__).stat().st_size),
                 actual_calls=dict(saved_arithmetic_invocations=1,detector=0,head=0,PnP=0,rays=0,optimizer_updates=0,private_GT_loader=0,benchmark=0),
                 limits=['Report ID: pallet_corner_mechanism_audit_20261010_v1; verifies saved actual rows and source/real reference arithmetic only.',
                         'Existing real reference is GEOMETRIC_PROXY; independent physical corner/edge ownership is not certified.',
                         'Source CAL reference is a virtual TLS intersection supported by certified wire queries, not independent physical corner ownership.',
                         'Association and exact algebra do not prove which physical boundary the model selected, domain-shift causality, or successful pose correction.',
                         'No new deployment acceptance threshold, model, training, images, masks or pose evaluation is created.'],
                 wall_seconds=time.monotonic()-started)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=passed,checks={c['name']:c['status'] for c in checks},output=str(target))))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
