"""Independent saved-result verification; numpy/stdlib only, no guard/scorer import.

Guard inputs contain prediction data only. Numeric reference files are opened
only after the complete238x12 prediction seal and every source binding pass.
No network, image, model, optimizer or PnP call exists in this module.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ARMS=('R0','R1','RA','RB','N0','N1','NA','NB','E0','E1','EA','EB')
PARTITIONS={'CAL':46,'TEST':49,'REMAINDER':143,'PRIMARY':192}
ERRORS=('translation_cm','rotation_deg','ADDsym_cm')
BOOTSTRAP_SEED=20260917
BOOTSTRAP_DRAWS=10000
REFERENCE_ANCHOR='_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/geometry_guided_edge_eval_v4_30/SCORE_SEAL.json'
REFERENCE_ANCHOR_SHA256='c2c85714f81dd90b78d46fd9e82541c9dc2ec1578e5e93c8f2209815ccf91d87'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def read_rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Checks:
    def __init__(self):self.count=0;self.failures=[]
    def __call__(self,passed,label,detail=None):
        self.count+=1
        if not bool(passed):self.failures.append(dict(check=label,detail=detail))
    def equal(self,a,b,label,tol=1e-9):
        if isinstance(a,dict):
            self(isinstance(b,dict),label+'/mapping')
            if isinstance(b,dict):
                for k,v in a.items():
                    self(k in b,label+'/'+k+'/present')
                    if k in b:self.equal(v,b[k],label+'/'+k,tol)
        elif isinstance(a,(list,tuple)):
            self(isinstance(b,(list,tuple)) and len(a)==len(b),label+'/length')
            if isinstance(b,(list,tuple)) and len(a)==len(b):
                for i,(x,y) in enumerate(zip(a,b)):self.equal(x,y,label+'/'+str(i),tol)
        elif isinstance(a,float) and isinstance(b,(int,float)):
            self(math.isclose(a,float(b),rel_tol=0,abs_tol=tol),label,dict(expected=a,actual=b))
        else:self(a==b,label,dict(expected=a,actual=b))


def rotation(value):
    R=np.asarray(value,dtype=float)
    if (R.shape!=(3,3) or not np.isfinite(R).all() or
        not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or
        not np.isclose(np.linalg.det(R),1.,atol=1e-7,rtol=0)):
        raise ValueError('Invalid proper rotation')
    return R


def transform(value):
    T=np.asarray(value,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.array_equal(T[3],[0,0,0,1]):
        raise ValueError('Invalid original reference transform')
    rotation(T[:3,:3]);return T


def signed_yaw(value):
    names={'YAW_0':0,'YAW_90':1,'YAW_180':2,'YAW_270':3}
    return np.linalg.matrix_power(np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]),names[value])


def reference_document(document):
    """Independent exact-copy/C2 validation, fixed candidate zero only."""
    if document.get('schema_version')!='real_pallet_gt_v2' or len(document.get('objects',[]))!=1:
        raise ValueError('Original one-object real_pallet_gt_v2 required')
    obj=document['objects'][0];cf=obj['camera_facing_pnp'];nominal=transform(cf['pose_transform'])
    xyz=[float(obj['physical_dimensions_m'][a]) for a in ('x','y','z')]
    if not np.isfinite(xyz).all() or min(xyz)<=0:raise ValueError('Invalid physical dimensions')
    records=obj.get('canonical_pose_candidates') or [];R=None
    if len(records)==2:
        try:
            TT=[transform(r['pose_transform']) for r in records]
            valid=all(np.array_equal(T[:3,3],nominal[:3,3]) and np.allclose(
                T[:3,:3],nominal[:3,:3]@signed_yaw(r['axis_assignment']),atol=1e-6,rtol=0)
                for T,r in zip(TT,records))
            first=int(records[0]['axis_assignment'].split('_')[1]);second=int(records[1]['axis_assignment'].split('_')[1])
            valid=valid and (second-first)%360==180 and np.array_equal(TT[1][:3,:3],TT[0][:3,:3]@signed_yaw('YAW_180'))
            if valid:R=TT[0][:3,:3]
        except (KeyError,ValueError,TypeError):pass
    confirmed=obj.get('canonical_pose')
    if confirmed is not None:
        T=transform(confirmed['pose_transform']);A=signed_yaw(confirmed['axis_assignment'])
        valid=(cf.get('axis_assignment_confirmed') is True and np.array_equal(T[:3,3],nominal[:3,3])
               and np.allclose(T[:3,:3],nominal[:3,:3]@A,atol=1e-6,rtol=0))
        R=T[:3,:3] if valid else None
    return dict(t=nominal[:3,3].tolist(),R=None if R is None else R.tolist(),xyz=xyz,order=2)


def metric(pose,reference):
    """Direct center, physical rotation and C2 vertex ADD; all failures retained."""
    if not pose['available']:
        return dict(pose_available=False,translation_cm=None,rotation_deg=None,ADDsym_cm=None,
                    success_le5=False,success_strict_lt5=False,reference_rotation_available=reference['R'] is not None)
    t=np.asarray(pose['centroid'],float);target=np.asarray(reference['t'],float)
    if t.shape!=(3,) or not np.isfinite(t).all():raise ValueError('Invalid available model center')
    T=float(np.linalg.norm(t-target)*100)
    if reference['R'] is None:r=add=None
    else:
        R=rotation(pose['R_physical']);G=rotation(reference['R']);x,y,z=np.asarray(reference['xyz'],float)/2
        xyz=np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])
        angles=[];adds=[]
        for k in range(reference['order']):
            angle=k*2*np.pi/reference['order'];c,s=np.cos(angle),np.sin(angle)
            S=np.array([[c,0,s],[0,1,0],[-s,0,c]]);Gk=G@S
            angles.append(float(np.degrees(np.arccos(np.clip((np.trace(Gk.T@R)-1)/2,-1,1)))))
            adds.append(float(np.linalg.norm(xyz@R.T+t-(xyz@Gk.T+target),axis=1).mean())*100)
        r=min(angles);add=min(adds)
    return dict(pose_available=True,translation_cm=T,rotation_deg=r,ADDsym_cm=add,
                success_le5=None if r is None else T<=5 and r<=5,
                success_strict_lt5=None if r is None else T<5 and r<5,
                reference_rotation_available=r is not None)


def fixed_references(source_root,reference_manifest,ordered_rows,checks):
    root=Path(source_root);m=read(reference_manifest);anchor=root/REFERENCE_ANCHOR
    checks(sha(anchor)==REFERENCE_ANCHOR_SHA256,'reference/original anchor')
    checks(m['reference_bindings']==read(anchor)['reference_bindings'],'reference/unchanged binding set')
    result={}
    for part,b in m['reference_bindings'].items():
        path=root/b['path'];checks(sha(path)==b['sha256'],'reference/'+part+'/sha')
        labels=read(path);wanted=[r['id'] for r in ordered_rows if r['partition']==part]
        if part=='REMAINDER':
            checks(labels['ids']==wanted,'reference/remainder exact IDs');refs={}
            for a in labels['annotations']:
                ap=root/a['annotation'];checks(sha(ap)==a['annotation_sha256'],'reference/'+a['id']+'/sha')
                refs[a['id']]=reference_document(read(ap))
            result[part]=refs
        else:result[part]={i:labels['rows'][i]['target']['truth'] for i in wanted}
    return result


def distribution(values):
    values=list(values);x=[float(v) for v in values if v is not None];n=len(x)
    if not all(math.isfinite(v) for v in x):raise ValueError('Nonfinite saved errors')
    mean=math.fsum(x)/n if n else None
    sd=math.sqrt(math.fsum((v-mean)**2 for v in x)/(n-1)) if n>1 else None
    return dict(n=n,denominator=len(values),missing=len(values)-n,mean=mean,sample_std=sd,max=max(x) if n else None)


def transitions(rows,new,before):
    groups={k:[] for k in ('failure_to_success','success_to_failure','both_success','both_failure','reference_unresolved')}
    for r in rows:
        a,b=[r['metrics'][arm]['success_le5'] for arm in (new,before)]
        key=('reference_unresolved' if a is None or b is None else 'failure_to_success' if a and not b else
             'success_to_failure' if b and not a else 'both_success' if a else 'both_failure')
        groups[key].append(r['id'])
    return dict(counts={k:len(v) for k,v in groups.items()},ids=groups,full_denominator=len(rows))


def recount_population(rows,contrasts):
    """Expand recording multiplicity to frames independently of grouped scorer."""
    units=sorted({r['cluster_id'] for r in rows});index={u:i for i,u in enumerate(units)}
    draws=np.random.default_rng(BOOTSTRAP_SEED).multinomial(len(units),[1/len(units)]*len(units),size=BOOTSTRAP_DRAWS).astype(np.uint16)
    frame_weights=draws[:,[index[r['cluster_id']] for r in rows]].astype(float)
    def interval(values):
        mask=np.asarray([v is not None for v in values]);x=np.asarray([v for v in values if v is not None],float)
        weights=frame_weights[:,mask];den=weights.sum(1);keep=den>0
        means=(weights[keep]@x)/den[keep];nunit=len({r['cluster_id'] for r,m in zip(rows,mask) if m})
        return dict(mean_difference=float(x.mean()) if len(x) else None,
                    CI95=np.quantile(means,[.025,.975]).tolist() if nunit>1 and len(means) else None,
                    common_available_n=int(mask.sum()),full_denominator=len(rows),excluded_unavailable_n=int((~mask).sum()),
                    eligible_clusters=nunit,valid_resamples=int(keep.sum()),empty_resamples=int((~keep).sum()),
                    improved=int((x<0).sum()),worsened=int((x>0).sum()),tied=int((x==0).sum()))
    arms={}
    for arm in ARMS:
        ms=[r['metrics'][arm] for r in rows];ss=[r['prediction_status'][arm] for r in rows]
        arms[arm]=dict(denominator=len(rows),successes_le5=sum(m['success_le5'] is True for m in ms),
            success_rate_le5=sum(m['success_le5'] is True for m in ms)/len(rows),
            reference_unresolved_n=sum(m['success_le5'] is None for m in ms),
            successes_strict_lt5=sum(m['success_strict_lt5'] is True for m in ms),
            pose_available_n=sum(m['pose_available'] for m in ms),no_pose_n=sum(not m['pose_available'] for m in ms),
            accepted_n=sum(s['accepted'] is True for s in ss),accepted_available_n=sum(s['accepted'] is not None for s in ss),
            fallback_n=sum(s['fallback'] is True for s in ss),fallback_available_n=sum(s['fallback'] is not None for s in ss),
            errors={k:distribution(m[k] for m in ms) for k in ERRORS},status_counts=dict(Counter(str(s['status']) for s in ss)))
    comparisons={}
    for new,before in contrasts:
        values=[None if any(r['metrics'][a]['success_le5'] is None for a in (new,before)) else
                int(r['metrics'][new]['success_le5'])-int(r['metrics'][before]['success_le5']) for r in rows]
        rate=interval(values);rate['gains']=rate.pop('worsened');rate['losses']=rate.pop('improved')
        if any(x is None for x in values):
            rate.update(conditional_known_label_mean_difference=rate['mean_difference'],mean_difference=None,CI95=None)
        comparisons[new+'-'+before]=dict(new=new,reference=before,transition=transitions(rows,new,before),success_rate=rate,
            continuous={k:interval([None if any(r['metrics'][a][k] is None for a in (new,before)) else
                       r['metrics'][new][k]-r['metrics'][before][k] for r in rows]) for k in ERRORS})
    return dict(rows=len(rows),arms=arms,comparisons=comparisons,
        bootstrap=dict(seed=BOOTSTRAP_SEED,resamples=BOOTSTRAP_DRAWS,units=len(units),clusters=units,
                       cluster_sizes=dict(Counter(r['cluster_id'] for r in rows)),session_sizes=dict(Counter(r['session'] for r in rows)),
                       draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest()))


LINE_SOURCES={'B_TARGET_BACKGROUND_LINES':('B','OBSERVED_TARGET_BACKGROUND_TRANSITION'),
              'I_INTERNAL_RGB_JUNCTION':('I','OBSERVED_INTERNAL_RGB_GRADIENT'),
              'B_ACTUAL_BOUNDARY_RECOVERY':('B','OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION')}


def _finite(value,shape=None):
    try:a=np.asarray(value,dtype=float)
    except (ValueError,TypeError):return None
    return a if (shape is None or a.shape==shape) and np.isfinite(a).all() else None


def _int(value):return isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_))


def projected_cuboid(observation,pose,K):
    xyz=np.asarray(observation['initial_geometry']['native_xyz'],float)
    camera=xyz@np.asarray(pose['R_cf'],float).T+np.asarray(pose['centroid'],float)
    xy=camera@np.asarray(K,float).T
    if not np.isfinite(xy).all() or np.any(camera[:,2]<=0) or np.any(xy[:,2]<=0):return None
    return xy[:,:2]/xy[:,2,None]


def ray_cost(points,weights,q,corner,neighbor):
    """Independent cross-product distance; no fitted observation line used."""
    a=q[corner];delta=q[neighbor]-a;length=float(np.hypot(delta[0],delta[1]))
    if not length>0:return None
    x=np.asarray(points,float)-a
    distance=np.abs(x[:,0]*delta[1]-x[:,1]*delta[0])/length
    # Same predeclared delta1.25, written without the production np.where form.
    loss=.5*np.minimum(distance,1.25)**2+1.25*np.maximum(distance-1.25,0.)
    numerator=math.fsum(float(w)*float(v) for w,v in zip(weights,loss))
    denominator=math.fsum(float(w) for w in weights)
    return numerator/denominator,numerator,denominator


def recompute_guard(observation,solved,coarse_pose,K):
    """Recompute two fixed rules; never import or call production guard.py.

    Method-validity mismatches are returned as a failed proposal rather than
    inferred from the production decision_audit or reason string.
    """
    states=observation.get('states');correction=observation.get('visible_correction') or {}
    solution=solved.get('solution') or {};candidate=solved.get('actual_pose') or {}
    out=dict(A=False,B=False,checks={},measurements=[],rays=[],line_cost=None)
    def c(name,value):out['checks'][name]=bool(value);return bool(value)
    if not c('states',isinstance(states,list) and len(states)==8 and set(states)<=set('SBIOTU')):return out
    prov=observation.get('provenance') or {};cp=correction.get('provenance') or {};pix=correction.get('image_pixel_sha256')
    authentic=(observation.get('schema')=='STATEWISE_RGB_OBSERVATIONS_V4' and
        correction.get('schema')=='visible_BI_measurements_boundary_recovery_v4_3b' and
        prov.get('GT_inputs') is False and prov.get('current_RGB_state_inference') is True and
        prov.get('S_image_queries')==0 and cp.get('S_queried') is False and
        cp.get('projected_coordinates_used_as_measurements') is False and
        cp.get('image_coordinates_frozen_before_joint_OT') is True and isinstance(pix,str) and
        len(pix)==64 and all(x in '0123456789abcdef' for x in pix))
    if not c('authentic_observation',authentic):return out
    if not c('accepted_solver',solved.get('accepted') is True and solution.get('accepted') is True):return out
    try:
        ext=np.asarray(coarse_pose['cf_extents'],float);a,b,d=ext/2
        xyz=np.asarray([[-a,-b,-d],[a,-b,-d],[a,b,-d],[-a,b,-d],[-a,-b,d],[a,-b,d],[a,b,d],[-a,b,d]])
        R0=rotation(coarse_pose['R_cf']);R1=rotation(candidate['R_cf']);P0=rotation(coarse_pose['R_physical']);P1=rotation(candidate['R_physical'])
        g=observation['initial_geometry'];t0=np.asarray(coarse_pose['centroid'],float);t1=np.asarray(candidate['centroid'],float)
        geometry=(ext.shape==(3,) and min(ext)>0 and np.isfinite(ext).all() and
            np.array_equal(ext,candidate['cf_extents']) and np.array_equal(xyz,g['native_xyz']) and
            np.array_equal(R0,g['R_cf']) and np.array_equal(t0,g['centroid']) and
            np.allclose(R0.T@P0,g['physical_registration_Q'],rtol=0,atol=1e-9) and
            np.allclose(R0.T@P0,R1.T@P1,rtol=0,atol=1e-9) and
            t0.shape==t1.shape==(3,) and np.isfinite(t0).all() and np.isfinite(t1).all() and
            candidate.get('available',True) is not False and coarse_pose.get('available',True) is not False and
            _finite(K,(3,3)) is not None)
    except (KeyError,ValueError,TypeError):geometry=False
    if not c('registered_available_geometry',geometry):return out
    original=_finite(observation.get('original_raw_points'));measured=observation.get('frozen_visible_measurements');records=correction.get('records')
    if not c('input_arrays',original is not None and original.shape in ((8,2),(9,2)) and isinstance(measured,list) and isinstance(records,list)):return out
    byid={}
    for r in records:
        k=r.get('native_corner') if isinstance(r,dict) else None
        if not _int(k) or not 0<=k<8 or k in byid or r.get('state')!=states[k]:
            c('unique_native_records',False);return out
        byid[int(k)]=r
    if not c('complete_native_records',set(byid)==set(range(8))):return out
    ids=[];moved=[]
    for m in measured:
        k=m.get('native_corner') if isinstance(m,dict) else None
        if not _int(k) or not 0<=k<8 or k in ids or states[k] not in ('B','I'):
            c('authentic_unique_BI',False);return out
        k=int(k);ids.append(k);xy=_finite(m.get('xy'),(2,));cov=_finite(m.get('covariance_px2'),(2,2))
        valid=(m.get('state')==states[k] and xy is not None and cov is not None and
               np.allclose(cov,cov.T,rtol=0,atol=1e-9) and np.linalg.eigvalsh(cov).min()>0 and
               m.get('image_measurement') is True and m.get('measurement_is_pose_projection') is False)
        if not c('measurement_'+str(k),valid):return out
        r=byid[k];source=m.get('source');spec=LINE_SOURCES.get(source)
        genuine=(spec is not None and spec[0]==states[k] and r.get('method')==source and
                 r.get('corrected') is True and r.get('pixel_queried') is True and
                 np.array_equal(r.get('raw_xy'),original[k]) and np.array_equal(r.get('measured_xy'),xy))
        movement=float(np.linalg.norm(xy-original[k]));actual=bool(genuine and movement>1e-8)
        out['measurements'].append(dict(native_corner=k,genuine=bool(genuine),movement_px=movement,moved=actual))
        if actual:moved.append(k)
    c('exact_BI_IDs',set(ids)=={i for i,s in enumerate(states) if s in ('B','I')})
    c('solver_input_IDs',solved.get('input_native_ids')==ids)
    c('minimum_four',len(ids)>=4);c('rank6',(solution.get('rank') or {}).get('visible',{}).get('rank')==6)
    ambiguity=solution.get('ambiguity') or {}
    for flag in ('detected','local_underdetermined','pose_family_from_two_BI'):c('ambiguity_'+flag,ambiguity.get(flag) is False)
    c('actual_current_RGB_movement',bool(moved))
    out['A']=all(out['checks'].values())
    if not out['A']:return out
    q0=projected_cuboid(observation,coarse_pose,K);q1=projected_cuboid(observation,candidate,K)
    if q0 is None or q1 is None:return out
    total0=[];total1=[];weightsums=[]
    for k in ids:
        r=byid[k];method=r.get('method');spec=LINE_SOURCES.get(method)
        for index,ray in enumerate(r.get('rays',[])):
            z=dict(native_corner=k,ray_index=index,status='excluded');out['rays'].append(z)
            if not isinstance(ray,dict) or ray.get('supported') is not True:continue
            j=ray.get('native_neighbor')
            if (spec is None or spec[0]!=states[k] or ray.get('measurement_source')!=spec[1] or
                ray.get('guide_source')!='initial_registered_model_direction_only' or r.get('pixel_queried') is not True):continue
            if (not _int(j) or not 0<=j<8 or j==k or ray.get('native_corner')!=k or ray.get('endpoints')!=[k,j] or
                ray.get('edge')!=k*8+j or np.count_nonzero(xyz[k]!=xyz[j])!=1):continue
            if method=='B_ACTUAL_BOUNDARY_RECOVERY' and (cp.get('recovery_RGB_support_required') is not True or
                ray.get('shared_native_edge')!=sorted([k,j]) or not _int(ray.get('rgb_supported_sample_count')) or
                ray['rgb_supported_sample_count']<1):continue
            points=_finite(ray.get('observed_points'));weights=_finite(ray.get('observation_confidence'))
            normal=_finite(ray.get('normal'),(2,));offset=_finite(ray.get('offset'),())
            if (points is None or points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or weights is None or
                weights.shape!=(len(points),) or np.any(weights<0) or weights.sum()<=0 or normal is None or
                np.linalg.norm(normal)<=0 or offset is None):continue
            old=ray_cost(points,weights,q0,k,j);new=ray_cost(points,weights,q1,k,j)
            if old is None or new is None:continue
            z.update(status='valid',native_neighbor=int(j),sample_count=len(points),weight_sum=old[2],
                     baseline_cost=old[0],candidate_cost=new[0],nonworsening=new[0]<=old[0]+1e-9)
            total0.append(old[1]);total1.append(new[1]);weightsums.append(old[2])
    valid=[r for r in out['rays'] if r['status']=='valid'];w=math.fsum(weightsums)
    mean0=math.fsum(total0)/w if w else None;mean1=math.fsum(total1)/w if w else None
    all_nonworse=all(r['nonworsening'] for r in valid);improved=bool(valid and mean0-mean1>1e-9)
    out['line_cost']=dict(valid_ray_count=len(valid),weight_sum=w,baseline_weighted_mean=mean0,
                          candidate_weighted_mean=mean1,strict_improvement=improved,all_rays_nonworsening=all_nonworse)
    out['B']=bool(valid and all_nonworse and improved)
    return out


def _cuboid_vertices(extents):
    a,b,c=np.asarray(extents,float)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def e_input_mapping_audit(folder,original,eframes,checks):
    eseal=read(folder/'e_inputs/INPUT_SEAL.json');contract=read(folder/'e_inputs/CONTRACT.json')
    checks(eseal['inputs_sha256']==sha(folder/'e_inputs/INPUTS.json'),'E/input bytes')
    checks(eseal['inputs_gzip_sha256']==sha(folder/'e_inputs/INPUTS.jsonl.gz'),'E/input gzip bytes')
    checks(eseal['contract_sha256']==sha(folder/'e_inputs/CONTRACT.json'),'E/input contract bytes')
    checks(eseal['GT_inputs'] is False and eseal['before_new_E_observation'] is True,'E/pre observation input seal')
    checks(eseal['ordered_ids']==[r['id'] for r in original],'E/exact input membership')
    permutations=Counter();maximum_error=0.
    for f,ef in zip(original,eframes):
        key=f['id']+'/E_mapping';checks(f['id']==ef['id'],key+'/ID')
        for b in ('RAW_YOLO','N3_SUBPIX'):checks(f['arms'][b]==ef['arms'][b],key+'/unchanged '+b)
        source=f['arms']['N3_SUBPIX'];target=ef['arms']['E_FROZEN'];p0=source['coarse_pose'];p1=target['coarse_pose']
        Q0=rotation(p0['R_cf']).T@rotation(p0['R_physical']);Q1=rotation(p1['R_cf']).T@rotation(p1['R_physical'])
        physical0=_cuboid_vertices(p0['cf_extents'])@Q0;physical1=_cuboid_vertices(p1['cf_extents'])@Q1
        known=_cuboid_vertices(f['xyz']);distance=np.linalg.norm(physical1[:,None,:]-physical0[None,:,:],axis=2)
        matches=distance<=1e-8
        checks(bool(np.all(matches.sum(0)==1) and np.all(matches.sum(1)==1)),key+'/unique physical vertex bijection')
        mapping=np.argmax(matches,axis=1).tolist();permutations[tuple(mapping)]+=1
        maximum_error=max(maximum_error,float(distance[matches].max()))
        for label,physical in [('N3',physical0),('E',physical1)]:
            same=np.linalg.norm(physical[:,None,:]-known[None,:,:],axis=2)<=1e-8
            checks(np.all(same.sum(0)==1) and np.all(same.sum(1)==1),key+'/'+label+'/known physical extents')
        meta=target['native_vertex_mapping']
        checks(meta['target_to_source_native_ids']==mapping and meta['source_to_target_native_ids']==np.argsort(mapping).tolist(),key+'/saved permutation')
        expected=np.asarray(source['native_points'])[mapping].tolist()
        if len(source['native_points'])==9:expected.append(source['native_points'][8])
        checks(target['native_points']==expected,key+'/coordinates exact source measurement permutation')
        checks(target['keypoints_conf']==[source['keypoints_conf'][k] for k in mapping]+source['keypoints_conf'][8:],key+'/confidence same permutation')
        checks(target['prediction_support']==[source['prediction_support'][k] for k in mapping]+source['prediction_support'][8:],key+'/support same permutation')
        checks(np.array_equal(meta['source_physical_vertex_xyz'],physical0) and np.array_equal(meta['target_physical_vertex_xyz'],physical1),key+'/physical coordinate archive')
    checks([dict(target_to_source_native_ids=list(p),frames=n) for p,n in sorted(permutations.items())]==contract['physical_vertex_permutations'],'E/complete permutation histogram')
    checks(math.isclose(maximum_error,contract['maximum_physical_vertex_error_m'],rel_tol=0,abs_tol=1e-14),'E/max physical mapping error')
    checks(sum(n for p,n in permutations.items() if p!=tuple(range(8)))==contract['nonidentity_permutation_frames'],'E/nonidentity frames retained')
    return dict(frames=238,nonidentity_frames=contract['nonidentity_permutation_frames'],maximum_physical_vertex_error_m=maximum_error)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _bound_file(binding,base,checks,label):
    path=Path(binding['path'])
    if not path.is_absolute():path=base/path
    checks(sha(path)==binding['sha256'],label+'/sha')
    return path


def prediction_audit(folder,prior,checks):
    """All prediction/hash/guard checks precede every new reference decode."""
    folder,prior=Path(folder),Path(prior);seal=read(folder/'PREDICTION_SEAL.json');protocol=read(folder/'METHOD_PROTOCOL.json')
    checks(seal['rows']==238 and seal['arm_rows']==2856 and seal['arms']==list(ARMS),'prediction/full2856')
    checks(seal.get('GT_inputs') is False and seal.get('before_new_CAL_scoring') is True and seal.get('all_rows_kept') is True,'prediction/pre-score declaration')
    for key,name in [('predictions_sha256','PREDICTIONS.jsonl.gz'),('guard_audits_sha256','GUARD_AUDITS.jsonl.gz'),
                     ('independent_bindings_sha256','INDEPENDENT_BINDINGS.json'),('protocol_sha256','METHOD_PROTOCOL.json')]:
        checks(seal[key]==sha(folder/name),'prediction/binding/'+name)
    code=Path(__file__).resolve().parents[3]
    for name,h in protocol['inference_source_sha256'].items():checks(sha(code/name)==h,'source/'+name)
    for name,h in protocol['immutable_inputs_sha256'].items():checks(sha(name)==h,'immutable/'+Path(name).name)
    rows=read_rows(folder/'PREDICTIONS.jsonl.gz');old=read_rows(prior/'run_v2/PREDICTIONS.jsonl.gz')
    original=read(prior/'inputs_v2/INPUTS.json')['frames'];eframes=read(folder/'e_inputs/INPUTS.json')['frames']
    e_mapping=e_input_mapping_audit(folder,original,eframes,checks)
    ids=[f['id'] for f in original];checks(len(ids)==len(set(ids))==238,'inventory/unique238')
    checks([r['id'] for r in rows]==ids==seal['ordered_ids'],'inventory/unchanged ordered238')
    checks([r['partition'] for r in rows]==['CAL']*46+['TEST']*49+['REMAINDER']*143,'inventory/CAL46 TEST49 R143 order')
    bindings=read(folder/'INDEPENDENT_BINDINGS.json');checks(bindings['ordered_ids']==ids,'guard/binding ordered238')
    checks([r['id'] for r in bindings['records']]==ids,'guard/binding record238')
    auditrows=read_rows(folder/'GUARD_AUDITS.jsonl.gz');audits={(r['id'],r['arm']):r for r in auditrows}
    checks(len(auditrows)==len(audits)==1428,'guard/unique1428 audits')
    summary=Counter();features=[]
    summary['E_nonidentity_physical_mapping_frames']=e_mapping['nonidentity_frames']
    for i,(r,bound,f,ef,previous) in enumerate(zip(rows,bindings['records'],original,eframes,old)):
        fid=r['id'];checks(set(r['predictions'])==set(ARMS),fid+'/all12')
        checks((r['session'],r['cluster_id'])==(f['session'],f['cluster_id'])==(ef['session'],ef['cluster_id']),fid+'/same source recording')
        for a in ('R0','R1','N0','N1'):checks(r['predictions'][a]==previous['predictions'][a],fid+'/'+a+'/exact old entry')
        for prefix,branch,frame in [('R','RAW_YOLO',f),('N','N3_SUBPIX',f),('E','E_FROZEN',ef)]:
            spec=bound['branches'][prefix];op=_bound_file(spec['observation'],folder,checks,fid+'/'+prefix+'/observation')
            sp=_bound_file(spec['solution'],folder,checks,fid+'/'+prefix+'/solve')
            oo,ss=read(op),read(sp);obs=oo['result']['observation'];solved=ss['result'];coarse=frame['arms'][branch]['coarse_pose'];K=frame['K']
            checks(oo['id']==ss['id']==fid and oo['branch']==ss['branch']==branch,fid+'/'+prefix+'/own input identity')
            checks(oo['result_sha256']==_digest(oo['result']) and ss['result_sha256']==_digest(solved),fid+'/'+prefix+'/payload digests')
            checks(ss['observation_binding']==oo['binding'],fid+'/'+prefix+'/solve observes same branch')
            checks(obs['original_coarse_pose']==coarse,fid+'/'+prefix+'/own coarse')
            checks(obs['original_raw_points']==frame['arms'][branch]['native_points'],fid+'/'+prefix+'/own real points')
            checks(obs['initial_geometry']['K']==K,fid+'/'+prefix+'/own K')
            hidden={k for k,s in enumerate(obs['states']) if s=='S'}
            checks(not hidden.intersection(obs['state_observation']['rgb_queried_ids']),fid+'/'+prefix+'/S no RGB')
            checks(not hidden.intersection(obs['mask_audit']['prompt_native_ids']),fid+'/'+prefix+'/S no mask prompt')
            checks(all(m['state'] in ('B','I') and m['native_corner'] not in hidden for m in obs['frozen_visible_measurements']),fid+'/'+prefix+'/onlyBI measurements')
            mask=oo['result']['mask_binding'];checks(sha(op.parent.parent/mask['path'])==mask['sha256'],fid+'/'+prefix+'/own mask bytes')
            if prefix=='E':
                checks(branch=='E_FROZEN' and oo['result']['operations'].get('record_id')==fid,fid+'/E/own state operation')
                checks(oo['result']['actual_calls'].get('n3_forward_calls')==0,fid+'/E/no new N3')
                checks(r['predictions']['E0']['actual_pose']==coarse,fid+'/E0/exact E')
                checks(r['predictions']['E1']['actual_pose']==solved['actual_pose'],fid+'/E1/own solved pose')
            computed=recompute_guard(obs,solved,coarse,K)
            for mode in ('A','B'):
                arm=prefix+mode;key=fid+'/'+arm;entry=r['predictions'][arm];saved=audits[(fid,arm)]
                checks(saved['observation_sha256']==spec['observation']['sha256'] and saved['solve_sha256']==spec['solution']['sha256'],key+'/audit input links')
                checks(saved['branch']==branch and saved['result']['mode']==mode,key+'/own arm and mode metadata')
                selected=computed[mode];summary[arm+'/'+('selected' if selected else 'retained')]+=1
                expected=solved['actual_pose'] if selected else coarse
                checks(entry['actual_pose']==expected,key+'/exact selected own pose')
                checks(entry['accepted'] is selected and entry['fallback'] is (not selected),key+'/replacement flags')
                checks(saved['result']['actual_pose']==entry['actual_pose'] and saved['result']['accepted'] is selected,key+'/saved audit decision')
                da=saved['result']['decision_audit']
                checks(da.get('GT_inputs') is False and da.get('new_PnP_calls')==0 and da.get('image_queries')==0,key+'/prediction-only guard')
                if mode=='B' and computed['line_cost'] is not None:
                    checks.equal(computed['line_cost'],da.get('line_cost'),key+'/independent line cost')
                    actual={(z['native_corner'],z['ray_index']):z for z in da['ray_records']}
                    for z in computed['rays']:
                        other=actual.get((z['native_corner'],z['ray_index']),{})
                        checks(z['status']==other.get('status'),key+'/ray status')
                        if z['status']=='valid':checks.equal(z,other,key+'/ray independent cost')
            features.append(dict(id=fid,branch=branch,A=computed['A'],B=computed['B'],checks=computed['checks'],line_cost=computed['line_cost']))
    checks(sum(r['partition']!='CAL' for r in rows)==192,'inventory/PRIMARY192')
    if checks.failures:raise ValueError('Prediction-only independent gate failed: '+repr(checks.failures[:3]))
    return rows,protocol,features,dict(summary)


def independent_selection(cal):
    if len(cal)!=46 or any(r['partition']!='CAL' for r in cal):raise ValueError('CAL46 selection only')
    def stats(arm):
        values=[r['metrics'][arm] for r in cal]
        return dict(success=sum(v['success_le5'] is True for v in values),
                    T=distribution(v['translation_cm'] for v in values)['mean'],R=distribution(v['rotation_deg'] for v in values)['mean'])
    by_init={};eligible={};allstats={a:stats(a) for a in ARMS}
    for prefix in ('R','N','E'):
        candidates=[prefix+'0']
        for mode in ('A','B'):
            arm=prefix+mode;cells=transitions(cal,arm,prefix+'0')['counts']
            ok=cells['success_to_failure']==0 and cells['failure_to_success']>=1
            eligible[arm]=dict(eligible=ok,recovery=cells['failure_to_success'],damage=cells['success_to_failure'])
            if ok:candidates.append(arm)
        by_init[prefix]=min(candidates,key=lambda a:(-allstats[a]['success'],eligible.get(a,{}).get('damage',0),
                                allstats[a]['T'],allstats[a]['R'],'0AB'.index(a[1])))
    chosen=min(by_init.values(),key=lambda a:(-allstats[a]['success'],'ENR'.index(a[0])))
    return dict(selected_arm=chosen,per_initialization=by_init,eligibility=eligible)


def _reported_subset(reconstructed,reported):
    """Map stable arithmetic fields to the production report's chosen labels."""
    result=dict(rows=reconstructed['rows'],arms={},comparisons={},bootstrap=reconstructed['bootstrap'])
    for arm,x in reconstructed['arms'].items():
        y=dict(denominator=x['denominator'],successes_le5=x['successes_le5'],successes_strict_lt5=x['successes_strict_lt5'],
               no_pose_n=x['no_pose_n'],unresolved_n=x['reference_unresolved_n'],accepted_n=x['accepted_n'],fallback_n=x['fallback_n'],
               acceptance_known_n=x['accepted_available_n'],errors=x['errors'],status_counts=x['status_counts'])
        result['arms'][arm]=y
    for key,x in reconstructed['comparisons'].items():
        result['comparisons'][key]={k:x[k] for k in ('transition','success_rate','continuous')}
    return result


def run(*,folder,prior,source_root,output,expected_prediction_seal_sha256,expected_score_seal_sha256):
    """Only closed saved artifacts are consumed; output is exclusive/new."""
    started=time.monotonic();cpu=time.process_time();folder=Path(folder);prior=Path(prior);output=Path(output)
    if output.exists():raise FileExistsError('Preserve every independent check attempt')
    if sha(folder/'PREDICTION_SEAL.json')!=expected_prediction_seal_sha256:raise ValueError('Exact prediction seal required')
    if sha(folder/'score/SCORE_SEAL.json')!=expected_score_seal_sha256:raise ValueError('Exact score seal required')
    checks=Checks();rows,protocol,features,guard_counts=prediction_audit(folder,prior,checks)
    # Above is the full prediction-only gate, before any reference coordinate IO.
    ss=read(folder/'score/SCORE_SEAL.json');selected=read(folder/'SELECTION_SEAL.json')
    checks(ss['prediction_seal_sha256']==expected_prediction_seal_sha256,'score/same full predictions')
    checks(ss['pre_PRIMARY_selection_seal_sha256']==sha(folder/'SELECTION_SEAL.json'),'score/prePRIMARY selection seal')
    checks(ss['rows']==238 and ss['arm_rows']==2856 and ss['all_rows_kept'] is True,'score/all2856')
    for name,h in ss['artifacts'].items():checks(sha(folder/'score'/name)==h,'score/artifact/'+name)
    checks(selected['prediction_seal_sha256']==expected_prediction_seal_sha256,'selection/same full predictions')
    checks(selected['method_protocol_sha256']==sha(folder/'METHOD_PROTOCOL.json'),'selection/fixed method')
    checks(selected['CAL_scored_rows_sha256']==sha(folder/'cal/SCORED_ROWS.json'),'selection/closed CAL score')
    checks(selected['ordered_CAL_ids']==[r['id'] for r in rows[:46]],'selection/exact CAL46')
    checks(selected['before_new_PRIMARY_scoring'] is True and selected['PRIMARY_used_for_selection'] is False,'selection/CAL-only declaration')
    checks(selected['selection_rule']==protocol['selection_rule'],'selection/predeclared rule')
    checks(selected['source_hashes']==protocol['inference_source_sha256'],'selection/unchanged inference sources')
    for name,h in [('guard.py',selected['guard_source_sha256']),('score.py',selected['score_source_sha256'])]:checks(sha(Path(__file__).with_name(name))==h,'selection/source '+name)
    if checks.failures:raise ValueError('Closed score/selection binding gate failed: '+repr(checks.failures[:3]))
    refs=fixed_references(source_root,prior/'SCORING_REFERENCE_MANIFEST_V2.json',rows,checks)
    source_scored=read(folder/'score/SCORED_ROWS.json');scored=source_scored['rows'];checks(source_scored['arm_rows']==2856,'score/2856 rows')
    checks([r['id'] for r in scored]==[r['id'] for r in rows],'score/ordered identities')
    reconstructed=[]
    for p,s in zip(rows,scored):
        values={a:metric(p['predictions'][a]['actual_pose'],refs[p['partition']][p['id']]) for a in ARMS}
        checks.equal(values,s['metrics'],p['id']+'/direct physical metrics')
        statuses={a:{k:p['predictions'][a].get(k) for k in ('accepted','fallback','status','reason')} for a in ARMS}
        checks.equal(statuses,s['prediction_status'],p['id']+'/prediction statuses')
        # Recount exact saved floating values after independent metric agreement.
        # This preserves the report's exact tie/boundary bins rather than
        # letting independent operation order turn a1e-14 tie into a sign.
        reconstructed.append(dict(id=p['id'],partition=p['partition'],session=p['session'],cluster_id=p['cluster_id'],metrics=s['metrics'],prediction_status=statuses))
    cal=reconstructed[:46];checks.equal(independent_selection(cal),selected,'selection/independent CAL-only rerun')
    checks.equal([r['metrics'] for r in cal],[r['metrics'] for r in read(folder/'cal/SCORED_ROWS.json')['rows']],'selection/CAL metric values')
    csvrows=list(csv.DictReader((folder/'score/ROW_METRICS.csv').open()));checks(len(csvrows)==2856,'CSV/all2856')
    csvmap={(r['id'],r['arm']):r for r in csvrows};checks(len(csvmap)==2856,'CSV/no duplicates')
    for r in reconstructed:
        for a in ARMS:
            v=csvmap[(r['id'],a)];m=r['metrics'][a]
            for k in ERRORS:
                val=None if v[k]=='' else float(v[k]);checks.equal(m[k],val,r['id']+'/'+a+'/CSV/'+k)
            for k in ('pose_available','success_le5','success_strict_lt5','reference_rotation_available'):
                val=None if v[k]=='' else v[k]=='True';checks(m[k] is val,r['id']+'/'+a+'/CSV/'+k)
    report=read(folder/'score/RESULTS.json');recount={}
    checks(report['selected_arm']==selected['selected_arm'],'report/same single CAL selected arm for all PRIMARY')
    checks(report['selection_seal_sha256']==sha(folder/'SELECTION_SEAL.json'),'report/frozen CAL choice binding')
    checks(report['default_changed'] is False and report['all_rows_and_failures_kept'] is True,'report/default and all failures preserved')
    for pop in ('CAL','TEST','REMAINDER','PRIMARY'):
        rr=[r for r in reconstructed if (r['partition']!='CAL' if pop=='PRIMARY' else r['partition']==pop)]
        checks(len(rr)==PARTITIONS[pop],pop+'/fixed denominator')
        contrasts=[tuple(k.split('-')) for k in report['populations'][pop]['comparisons']]
        recomputed=recount_population(rr,contrasts);recount[pop]=recomputed
        checks.equal(_reported_subset(recomputed,report['populations'][pop]),report['populations'][pop],pop+'/independent pooled recount')
        for arm in ARMS:
            m=report['populations'][pop]['arms'][arm]
            exact=sum(r['metrics'][arm]['translation_cm']==5. or r['metrics'][arm]['rotation_deg']==5. for r in rr)
            checks(exact==m['exact5_n'],pop+'/'+arm+'/exact boundary count')
    checks(sha(folder/'PREDICTION_SEAL.json')==expected_prediction_seal_sha256,'final/unchanged prediction seal')
    checks(sha(folder/'score/SCORE_SEAL.json')==expected_score_seal_sha256,'final/unchanged score seal')
    output.mkdir(parents=True)
    write(output/'GUARD_RECOUNT.json',dict(rows=features,counts=guard_counts))
    write(output/'RECOUNT.json',recount)
    receipt=dict(schema='INDEPENDENT_GUARDED_BI_FINAL_CHECK_V1',checks=checks.count,failure_count=len(checks.failures),failures=checks.failures,
        rows=238,arm_rows=2856,guard_decisions=1428,partitions=PARTITIONS,
        prediction_seal_sha256=expected_prediction_seal_sha256,score_seal_sha256=expected_score_seal_sha256,
        selection_seal_sha256=sha(folder/'SELECTION_SEAL.json'),protocol_sha256=sha(folder/'METHOD_PROTOCOL.json'),
        independent_source_sha256=sha(__file__),guard_counts=guard_counts,selected_arm=selected['selected_arm'],
        direct_reference_metric_recomputed=True,scorer_or_guard_or_PnP_or_model_imports=0,
        independent_reference_policy='fixed candidate0 after exact center-copy/C2 validation; no reference selection by prediction error',
        calibration_selection_recomputed_from_CAL46_only=True,no_new_training_or_image_or_optimizer_calls=True,
        hash_limit='Hashes verify saved source/artifact ancestry, not an absolute proof that a process never read any unrecorded file.',
        prior_development_exposure=True,new_independent_test=False,
        resources=dict(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,processes=1))
    write(output/'RECEIPT.json',receipt)
    write(output/'SEAL.json',dict(assets={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    if checks.failures:raise ValueError('Independent mismatches; all attempt outputs preserved')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','prior','source-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prediction-seal-sha256',required=True);p.add_argument('--score-seal-sha256',required=True)
    a=p.parse_args()
    print(json.dumps(run(folder=a.folder,prior=a.prior,source_root=a.source_root,output=a.output,
        expected_prediction_seal_sha256=a.prediction_seal_sha256,expected_score_seal_sha256=a.score_seal_sha256),indent=2))
