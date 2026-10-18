"""Synthetic diagnostic-only renderer/native/Q/finite-line verification.

No production projector, finite predicate, solver or feature builder is imported.
Raw renderer rotations and the original failed1e-10 roundtrip flags are retained.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,resource,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError('Unsupported JSON type: '+type(x).__name__)
def write(p,v):
    with Path(p).open('x')as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False,default=scalar);f.write('\n')


def rotation_vector_matrix(vector):
    v=np.asarray(vector,float);theta=float(np.linalg.norm(v))
    if theta==0:return np.eye(3)
    u=v/theta;x,y,z=u;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return math.cos(theta)*np.eye(3)+(1.-math.cos(theta))*np.outer(u,u)+math.sin(theta)*skew
def project(xyz,R,t,K):
    camera=np.asarray(xyz,float)@np.asarray(R,float).T+np.asarray(t,float);h=camera@np.asarray(K,float).T;z=h[:,2]
    den=np.where(np.abs(z)<1e-10,np.where(z<0,-1e-10,1e-10),z)
    return h[:,:2]/den[:,None],camera[:,2]
def permutation(first,second):
    a,b=np.asarray(first,float),np.asarray(second,float);distance=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=-1)
    choice=np.argmin(distance,axis=1);return choice,float(np.max(distance[np.arange(len(a)),choice])),len(set(choice.tolist()))==len(a)


def finite_intersection(query,xy,depth,pixel_scale,image_hw):
    xy=np.asarray(xy,float);a,b=xy[query['native_corner']],xy[query['native_neighbor']];n=np.asarray(query['outward_normal_xy'],float);c=np.asarray(query['center_xy'],float);v=b-a
    determinant=float(n[0]*v[1]-n[1]*v[0]);length=float(np.linalg.norm(v));eps=np.sqrt(np.finfo(float).eps);length_eps=eps*max(1.,math.hypot(*image_hw));leps=8.*eps
    if not np.isfinite(xy).all()or not np.isfinite([determinant,length]).all()or not np.isfinite(c).all()or not np.isfinite(n).all():status='NONFINITE_PROJECTION'
    elif depth is not None and(not np.isfinite(depth).all()or min(depth)<=0):status='NONPOSITIVE_DEPTH'
    elif length<=length_eps:status='DEGENERATE_PROJECTED_EDGE'
    elif abs(determinant)<=max(length_eps,eps*length):status='PARALLEL_NORMAL_AND_EDGE'
    else:status='SUPPORTED_FINITE_SEGMENT'
    coefficient=offset=None;intersection=None
    if status not in('NONFINITE_PROJECTION','DEGENERATE_PROJECTED_EDGE','PARALLEL_NORMAL_AND_EDGE')and determinant!=0:
        coefficient,normal_distance=np.linalg.solve(np.column_stack((v,-n)),c-a);coefficient=float(coefficient);offset=float(normal_distance/pixel_scale);intersection=a+coefficient*v
        if status=='SUPPORTED_FINITE_SEGMENT'and coefficient < -leps:status='BEFORE_FINITE_ENDPOINT'
        elif status=='SUPPORTED_FINITE_SEGMENT'and coefficient>1.+leps:status='AFTER_FINITE_ENDPOINT'
    return dict(geometry_status=status,lambda_segment=coefficient,offset_base_px=offset,intersection_xy=intersection,finite_support=status=='SUPPORTED_FINITE_SEGMENT')
def fixed_profiles(bundle):
    bycorner={}
    for group in(bundle or{}).get('groups',[]):
        if not group['supported']:continue
        profiles={r['support_id']:r for r in group['queries']if r['supported']}
        if profiles and group['native_corner']not in bycorner:bycorner[group['native_corner']]=[profiles[k]for k in sorted(profiles)]
    return[q for key in sorted(bycorner)for q in bycorner[key]]


def run(source_repo,archive,upstream,diagnostic,output):
    repo=Path(source_repo).resolve();archive=Path(archive).resolve();upstream=Path(upstream).resolve();diagnostic=Path(diagnostic).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();cpu=time.process_time();checks=Counter();failures=[];counts=Counter();maxima=Counter();annotationkeys=Counter();rawfails=[];metadata=[]
    def check(name,value,detail=None):
        checks[name]+=1
        if not bool(value):failures.append(dict(check=name,detail=detail))
    def near(a,b,absolute=1e-8):
        if a is None or b is None:return a is None and b is None
        return bool(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float)))<=absolute+1e-10*max(1.,float(np.max(np.abs(a)))))
    try:
        targetpath=upstream/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz';splitpath=upstream/'synthetic_v1/DATA_SPLIT_LOCK.json';rows_path=diagnostic/'B_REFERENCE_SUPPORT_ROWS.jsonl.gz';ds=read(diagnostic/'DIAGNOSTIC_SEAL.json')
        check('diagnostic_sealed',ds['complete']is True and ds['rows_sha256']==sha(rows_path)and ds['original_reference_sha256']==sha(targetpath)and ds['production_feature_or_candidate_insertion']is False)
        targets={r['id']:r for r in rows(targetpath)};split=read(splitpath);expected_split={r['id']:r['split']for r in split['all_rows']}
        sourcepath=repo/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json';check('original_source_manifest',sha(sourcepath)==split['original_source_sha256']);sourceby={r['id']:r for r in read(sourcepath)['records']}
        featurespath=archive/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz';ids=[r['id']for r in rows(featurespath)];index={fid:i for i,fid in enumerate(ids)};check('exact_original1003_IDs',set(ids)==set(targets)==set(expected_split)and len(ids)==1003)
        all_original_immutable={str(p):sha(p)for p in(targetpath,splitpath,sourcepath,featurespath,rows_path,diagnostic/'DIAGNOSTIC_SEAL.json')}
        for row in rows(rows_path):
            fid=row['id'];target=targets[fid];counts['frames']+=1;counts[row['split']+'/frames']+=1
            check('split_join',row['split']==target['split']==expected_split[fid],fid)
            check('reference_diagnostic_isolation',row['reference_used_only_in_this_diagnostic']is True and row['production_feature_or_candidate_insertion']is False)
            gt_xyz=np.asarray(target['native_GT_xyz']);K=np.asarray(target['K']);G=np.asarray(target['R_fixed']);t=np.asarray(target['t']);q_gt,z_gt=project(gt_xyz,G,t,K)
            renderer_gap=float(np.max(np.linalg.norm(q_gt-np.asarray(target['projected_GT_native_xy']),axis=1)));maxima['renderer_reprojection_max_px']=max(maxima['renderer_reprojection_max_px'],renderer_gap)
            check('original_renderer_reprojection_contract_lt0_05px',renderer_gap<.05,fid)
            folder=archive/'d1_synthetic_candidates_v1/frames'/f'{index[fid]:04d}';fs=read(folder/'FRAME_SEAL.json');bp=folder/'BOUNDARY_BUNDLE.jsonl.gz';check('original_fixed_bundle_hash',sha(bp)==fs['boundary_bundle_sha256'],fid);bundle=next(rows(bp));queries=fixed_profiles(bundle)
            counts['frames_with_B_profiles']+=int(bool(queries));counts['frames_without_B_profiles']+=int(not queries)
            check('registered_phase_denominator',len(row['registered_phases'])==row['symmetry_order']==target['symmetry_order'],fid)
            annotation=repo/sourceby[fid]['renderer_annotation_locator_provenance_only'];check('original_renderer_annotation_hash',sha(annotation)==target['renderer_annotation_sha256'],fid)
            annotation_value=read(annotation);object0=annotation_value['objects'][0]
            for key in object0:annotationkeys[key]+=1
            fields=[k for k in object0 if any(word in k.lower()for word in('material','mesh','visib','asset'))]
            metadata.append(dict(id=fid,annotation_sha256=target['renderer_annotation_sha256'],object_metadata_fields=fields,
                source_asset_is_identity_not_profile_owner='source_asset'in object0,material_owner_status='UNVERIFIED'))
            for phase in row['registered_phases']:
                counts['registered_phases']+=1;R=np.asarray(phase['reference_pose']['R_physical']);center=np.asarray(phase['reference_pose']['centroid']);angle=phase['phase_index']*2*np.pi/target['symmetry_order'];S=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
                check('original_raw_phase_rotation',np.array_equal(R,G@S)and np.array_equal(center,t),fid)
                check('both_registered_families',[f['family']for f in phase['families']]==['ORIGINAL_WD','SWAPPED_WD'])
                for family in phase['families']:
                    counts['phase_family_evaluations']+=1;name=family['family'];xyz=np.asarray(family['native_xyz']);Q=np.asarray(family['physical_registration_Q']);mean=xyz.mean(0);Rn=R@Q.T;tn=center-Rn@mean
                    check('native_origin_mean',np.array_equal(mean,family['native_origin_mean']),fid)
                    physical=(xyz-mean)@Q;perm,gap,unique=permutation(physical,gt_xyz);maxima['physical_vertex_set_max_m']=max(maxima['physical_vertex_set_max_m'],gap)
                    check('dynamic_GT_native_vertex_permutation',unique and gap<=1e-10,dict(id=fid,family=name,maximum_m=gap))
                    raw,z=project(xyz,Rn,tn,K);check('raw_GT_fixed_pose_projection',near(raw,family['raw_physical_projection_native']),fid)
                    phase_gt,_=project(gt_xyz,R,center,K);check('same_physical_GT_projection_set',near(raw,phase_gt[perm]),fid)
                    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3);p=np.asarray(family['native_pose_parameter_vector']);parameter_R=rotation_vector_matrix(p[:3]);parameter_t=p[3:]*scale
                    restored_R=parameter_R@Q;restored_center=parameter_R@mean+parameter_t;param_q,param_depth=project(xyz,parameter_R,parameter_t,K)
                    check('independent_parameterized_projection',near(param_q,family['reference_projected_native']),fid)
                    strict=bool(np.max(np.abs(restored_R-R))<=1e-10 and np.max(np.abs(restored_center-center))<=1e-10)
                    check('original_failed_roundtrip_flag_preserved',strict==family['physical_mapping_audit']['original_roundtrip_1e_10_pass'],fid)
                    if not strict:rawfails.append(dict(id=fid,phase=phase['phase_index'],family=name,R_max_abs=float(np.max(np.abs(restored_R-R))),center_max_abs=float(np.max(np.abs(restored_center-center))),original_tolerance=1e-10))
                    for representation,xy,depth,key in(('raw',raw,z,'raw_physical_finite_support'),('parameterized',param_q,param_depth,'finite_support')):
                        report=family[key];check('fixed_profile_universe',report['fixed_profile_count']==len(queries)and len(report['profiles'])==len(queries),fid)
                        supported=0
                        for query,prior in zip(queries,report['profiles']):
                            independent=finite_intersection(query,xy,depth,bundle['pixel_scale'],bundle['image_hw']);counts[representation+'/profiles']+=1;counts[representation+'/'+independent['geometry_status']]+=1
                            check('independent_2x2_finite_status',independent['geometry_status']==prior['geometry_status']and independent['finite_support']==prior['finite_support'],dict(id=fid,family=name,representation=representation,support_id=query['support_id']))
                            check('independent_2x2_lambda',near(independent['lambda_segment'],prior['lambda_segment']),fid)
                            check('independent_2x2_offset',near(independent['offset_base_px'],prior['offset_base_px']),fid)
                            check('independent_2x2_intersection',near(independent['intersection_xy'],prior['intersection_xy']),fid)
                            supported+=int(independent['finite_support'])
                        check('finite_supported_denominator',supported==report['supported_profiles'],fid)
                    if not queries:counts['phase_family_without_B_observation']+=1
        check('all_frame_phase_family_denominators',counts['frames']==1003 and counts['registered_phases']==sum(t['symmetry_order']for t in targets.values())and counts['phase_family_evaluations']==2*counts['registered_phases'])
        for path,digest in all_original_immutable.items():check('original_bytes_unchanged',sha(path)==digest,path)
        write(output/'ORIGINAL_STRICT_ROUNDTRIP_FAILURES.json',dict(original_tolerance=1e-10,failures=rawfails,count=len(rawfails),erased_or_tolerance_relaxed=False))
        write(output/'RENDERER_METADATA_AVAILABILITY.json',dict(frames=len(metadata),actual_object_keys_counts=dict(annotationkeys),rows=metadata,material_owner_status='UNVERIFIED',scope='Original renderer metadata presence only; an asset/visibility field is not an independently matched physical profile owner.'))
        result=dict(schema='RISK_INDEPENDENT_B_GEOMETRY_CHECK_V1',status='PASS'if not failures else'FAIL',checks=sum(checks.values()),checks_by_kind=dict(checks),failure_count=len(failures),failures=failures,counts=dict(counts),maximum_differences=dict(maxima),original_1e_10_roundtrip_failures_retained=len(rawfails),
            original_input_bindings=all_original_immutable,strict_failures_sha256=sha(output/'ORIGINAL_STRICT_ROUNDTRIP_FAILURES.json'),metadata_availability_sha256=sha(output/'RENDERER_METADATA_AVAILABILITY.json'),
            source_sha256=sha(__file__),production_geometry_or_finite_imports=0,reference_scope='Original synthetic renderer only; diagnostic output cannot enter production features/banks.',material_owner_status='UNVERIFIED',
            real_reference_reads=0,new_model_forward_calls=0,new_PnP_optimizer_solves=0,new_candidate_calls=0,
            cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,threads=1,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(output/'VERIFICATION.json',result);return result
    except BaseException as exc:
        write(output/'FAILED_ATTEMPT.json',dict(error=repr(exc),traceback=traceback.format_exc(),checks=dict(checks),failures=failures,wall_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-repo',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.source_repo,a.archive,a.upstream,a.diagnostic,a.output);print(json.dumps({k:r[k]for k in('status','checks','failure_count','counts','maximum_differences','original_1e_10_roundtrip_failures_retained','cost')}));raise SystemExit(0 if r['status']=='PASS'else 1)
