"""Public-only verification of the fixed IMAGE_ROLE all-lines C2 ablation.

Python >=3.9 and NumPy are required. This code reads only this checkout's
published files. It does not import experiment modules, OpenCV, PyTorch or
private source environments and performs no pose fit, model call or ray cast.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

NEW = '_docs/experiments/pallet_kp_supervision_gate_20261010_v1'
OLD = '_docs/experiments/pallet_observation_refiner_20261009_v1'
CAUSAL = '_docs/experiments/pallet_kp_difficulty_20261010_v1'
METHOD = 'IMAGE_ROLE_ORIGINAL_ALL_LINES'
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
LIMITS = [
    'Stored public arithmetic, identities and bindings are verified; no private evaluation truth is reread.',
    'Translation/rotation/ADDsym scalars are regrouped, not recomputed against private physical ground truth.',
    'Stored Jacobian ranks indicate local observability, not global uniqueness or correct observed boundary ownership.',
    'Triangle, surface and enclosure diagnostic rows are checked arithmetically; no mesh query or ray is executed.',
    'Full-source target arrays, labels and family splits are checked; train/calibration wire ownership is not independently reconstructed from the unpublished mesh.',
    'No inference, training, optimizer, PnP, rendering or deployment timing is repeated.',
]


def require(value, message):
    if not bool(value):
        raise AssertionError(message)


def close(a, b, label='', atol=1e-8, rtol=1e-9):
    if a is None or b is None:
        require(a is None and b is None, label + ': null mismatch')
    else:
        require(math.isfinite(float(a)) and math.isfinite(float(b)) and
                math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol),
                label + ': numeric mismatch ' + repr((a,b)))


def same(a, b, label='', atol=1e-7, rtol=1e-10):
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
            and np.allclose(aa, bb, atol=atol, rtol=rtol), label + ': array mismatch')


def percentile(values, q):
    if not len(values):
        return None
    values = sorted(float(x) for x in values)
    pos = (len(values)-1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo]) * (pos-lo)


def distribution(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(values)
    mean = math.fsum(values)/n if n else None
    variance = math.fsum((v-mean)**2 for v in values)/(n-1) if n>1 else None
    return dict(n=n, mean=mean, sample_variance=variance,
                sample_std=math.sqrt(variance) if variance is not None else None,
                median=percentile(values,.5), P90=percentile(values,.9),
                max=max(values) if n else None)


def compare_distribution(stored, values, label):
    calculated = distribution(values)
    require(stored['n'] == calculated['n'], label + ': denominator mismatch')
    for key in ('mean','sample_variance','sample_std','median','P90'):
        close(stored[key],calculated[key],label+'.'+key,atol=1e-7)
    if 'max' in stored:
        close(stored['max'],calculated['max'],label+'.max')
    return calculated


def semantic(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def cuboid(dimensions):
    a,b,c = np.asarray(dimensions,float)/2
    return np.asarray([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                       [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def projection(pose,K):
    camera = cuboid(pose['cf_extents']) @ np.asarray(pose['R_cf']).T + np.asarray(pose['centroid'])
    require(np.isfinite(camera).all() and (camera[:,2]>1e-9).all(),'nonfinite or nonpositive projection depth')
    homogeneous = camera @ np.asarray(K).T
    return homogeneous[:,:2]/homogeneous[:,2,None]


def hidden(pose):
    if not pose.get('available'):
        return []
    x = cuboid(pose['cf_extents'])
    camera = -np.asarray(pose['R_cf']).T @ np.asarray(pose['centroid'])
    rays = camera[None,:]-x
    cosine = np.max(np.sign(x)*rays/np.linalg.norm(rays,axis=1)[:,None],axis=1)
    return np.flatnonzero(cosine < -math.sin(math.radians(2))).tolist()


def box_interval(camera,point,lower,upper,epsilon=0):
    near,far=None,None
    for origin,target,lo,hi in zip(camera,point,lower,upper):
        direction=target-origin
        if direction==0:
            if origin<lo or origin>hi:return None
            continue
        a,b=(lo-origin)/direction,(hi-origin)/direction
        near=min(a,b) if near is None else max(near,min(a,b))
        far=max(a,b) if far is None else min(far,max(a,b))
    if near is None or near>far+epsilon or far<0 or near>1+epsilon:return None
    return near,far


def point_projection(point,R,t,K):
    camera=np.asarray(R,float)@np.asarray(point,float)+np.asarray(t,float)
    require(camera[2]>0,'sourcepoint positive cameraZ')
    p=np.asarray(K,float)@camera
    return p[:2]/p[2],camera[2]


def summary_quantiles(stored,values,label):
    require(len(values)>0,label+': empty values')
    for key,value in [('min',min(values)),('median',percentile(values,.5)),
                      ('P90',percentile(values,.9)),('P99',percentile(values,.99)),('max',max(values))]:
        close(stored[key],value,label+'.'+key,atol=1e-12)
    if 'n' in stored:require(stored['n']==len(values),label+': n mismatch')


def triangle_arithmetic(point,triangle):
    """Finite stored-triangle distance arithmetic; no spatial mesh query."""
    p=np.asarray(point,np.longdouble);a,b,c=np.asarray(triangle,np.longdouble)
    u,v=b-a,c-a;n=np.cross(u,v);nn=n@n
    uu,vv,uv=u@u,v@v,u@v;den=uu*vv-uv*uv
    require(nn>0 and den>0,'degenerate storedtriangle')
    plane=(p-a)@n/np.sqrt(nn);projected=p-n*((p-a)@n)/nn
    w=projected-a;beta=(vv*(w@u)-uv*(w@v))/den;gamma=(uu*(w@v)-uv*(w@u))/den
    bary=np.array([1-beta-gamma,beta,gamma])
    candidates=[]
    if np.all(bary>=0):candidates.append(projected)
    for first,second in ((a,b),(b,c),(c,a)):
        direction=second-first;position=np.clip((p-first)@direction/(direction@direction),0,1)
        candidates.append(first+position*direction)
    nearest=min(candidates,key=lambda q:float(np.linalg.norm(p-q)))
    return dict(distance=float(np.linalg.norm(p-nearest)),nearest=np.asarray(nearest,float),
                plane=float(plane),barycentric=np.asarray(bary,float),gram_condition=float(uu*vv/den),
                unitnormal=np.asarray(n/np.sqrt(nn),float))


class Review:
    def __init__(self,root,require_manifest=False):
        self.root=Path(root).resolve()
        self.require_manifest=require_manifest
        self.cache={}
        self.files_read=set()
        self.groups=[]
        self.frames={}
        self.predictions={}
        self.arms={}
        self.source={}

    def path(self,name):
        path=(self.root/name).resolve()
        require(path.is_relative_to(self.root),'read path escapes public checkout: '+str(name))
        return path

    def load(self,name,rows=False):
        if name not in self.cache:
            path=self.path(name)
            self.files_read.add(str(path.relative_to(self.root)))
            with (gzip.open if path.suffix=='.gz' else open)(path,'rt',encoding='utf-8') as stream:
                self.cache[name]=[json.loads(line) for line in stream if line.strip()] if rows else json.load(stream)
        return self.cache[name]

    def sha(self,name):
        path=self.path(name)
        self.files_read.add(str(path.relative_to(self.root)))
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        return digest.hexdigest(),path.stat().st_size

    def bind(self,item,name=None):
        path=name if name is not None else item['path']
        require(not Path(path).is_absolute(),'private binding cannot be opened: '+path)
        sha,size=self.sha(path)
        require(sha==item['sha256'] and size==item['bytes'],'binding mismatch: '+path)

    def provenance_bind(self,item):
        """Resolve historical absolute provenance only by explicit public map."""
        raw=item['path']
        if not Path(raw).is_absolute():
            self.bind(item)
            return
        basename=Path(raw).name
        mapping={
            'SOURCE_CEILING_ROWS.jsonl.gz':CAUSAL+'/SOURCE_CEILING_ROWS.jsonl.gz',
            'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz':CAUSAL+'/SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',
            'source_observation_gate.py':'scripts/research/pallet_kp_supervision_gate_20261010_v1/source_observation_gate.py',
        }
        if basename.startswith('SOURCE_OBSERVATION_GATE'):
            mapping[basename]=NEW+'/'+basename
        require(basename in mapping,'unmapped public provenance: '+basename)
        self.bind(item,mapping[basename])

    def group(self,name,function):
        try:
            details=function()
            self.groups.append(dict(name=name,passed=True,details=details))
        except Exception as error:
            self.groups.append(dict(name=name,passed=False,error=type(error).__name__+': '+str(error)))

    def protected(self):
        data=self.load(NEW+'/PRIOR_PUBLICATION_BINDINGS.json')
        entries=data['protected_files']
        require(len(entries)==189 and len({i['path'] for i in entries})==189,'expected189unique protected files')
        for item in entries:
            self.bind(item)
        protocol=self.load(NEW+'/PROTOCOL.json')
        require(protocol['new_training_updates']==0 and protocol['learning_budget']['remaining_formal_updates']==0,'new training budget claim changed')
        for item in protocol['inputs']+protocol['code_frozen_before_fit']:
            self.bind(item)
        relocation=self.load(NEW+'/PUBLICATION_RELOCATION.json')
        for item in relocation['files']:
            require(item['relocation_only'] and not item['content_changed'],'frozen provenance copy changed')
            self.bind(item)
        return dict(protected_files=189,parent_publication_commit=data['commit'],unchanged=True,
                    byte_exact_relocated_source_artifacts=len(relocation['files']),private_provenance_paths_not_opened=True)

    def geometry(self):
        inputs=self.load(OLD+'/INPUTS.json')
        self.frames={row['id']:row for row in inputs['frames']}
        require(len(self.frames)==319,'input319')
        sealed=self.load(NEW+'/GEOMETRY_SEALED.jsonl.gz',True)
        scored=self.load(NEW+'/PREDICTIONS.jsonl.gz',True)
        for rows in (sealed,scored):
            require(len(rows)==319 and {r['id'] for r in rows}==set(self.frames),'319exactIDs')
            require(len({r['id'] for r in rows})==319 and len({r['session'] for r in rows})==13,'duplicates or sessions')
        self.predictions={row['id']:row for row in scored}
        obs={row['id']:row for row in self.load(OLD+'/LEARNED_OBSERVATIONS.jsonl.gz',True) if row['method']=='IMAGE_ROLE'}
        require(set(obs)==set(self.frames),'originalIMAGE_ROLEpopulation')
        status=Counter();hidden_count=0;rank_counts=Counter();nfev=0;starts=0
        for row in sealed:
            fid=row['id'];f=self.frames[fid];p=self.predictions[fid];o=obs[fid];s=row['solver']
            require(row['method']==METHOD and row['session']==f['session'],'method/session mismatch')
            for key,value in row.items():
                require(p.get(key)==value,'sealed field altered during scoring: '+fid+':'+key)
            require(set(p)-set(row)=={'corner','pose'},'unexpected postseal fields')
            require(row['K']==f['K'] and row['xyz']==f['xyz'] and row['raw_hw']==f['raw_hw'],'input metadata mismatch')
            require(row['observations_semantic_sha256']==semantic(o),'observation identity mismatch')
            require(row['observation_raw_logits_sha256']==o['raw_logits_sha256'],'rawlogit identity mismatch')
            edges=[line['edge'] for line in o['lines']]
            require(len(edges)==len(set(edges)) and all(0<=e<12 for e in edges),'duplicate/invalidedge')
            for key in ('selected_line_edges','line_factor_edges'):
                require(s[key]==edges,'solverselectededge mismatch')
            require(row['selected_line_edges']==edges and row['selected_corner_ids']==[c['id'] for c in o['corners']],'selected observation mismatch')
            require(s['derived_corner_factors']==0 and not s['same_edge_point_line_double_count'],'pointline duplication')
            require(all(s[key]==[] for key in ('used','inliers','final_inliers','fit_input_ids','eligible')),'invented pointfit/inlier')
            require(not row['initial_or_hidden_image_points_used_in_fit'] and not row['no_match_points_filled_from_Base']
                    and not row['reprojections_reused_as_observations'] and not row['inference_GT_input'],'forbiddeninference input')
            H=hidden(row['initial_pose'])
            require(row['hidden_initial']==H and row['excluded']==H,'initialhidden mask mismatch')
            new=bool(s['available']);fallback=not new and bool(row['initial_pose']['available'])
            expected_status='NEW_POSE' if new else 'BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'
            require(row['new_pose_estimated']==new and row['fallback_used']==fallback and row['output_status']==expected_status,'output status mismatch')
            require(row['pose_available']==bool(row['actual_pose']['available']) and row['no_pose']==(not row['pose_available']),'availability mismatch')
            require(row['hidden_reprojected']==(new and bool(H)) and row['reprojected_ids']==(H if new else []),'reprojectionstatus')
            original=np.asarray(f['points']['BASE'],float);expected=original.copy()
            if new:
                require(row['actual_pose']==s,'new actualpose is not solverpose')
                R=np.asarray(s['R_cf']);same(R.T@R,np.eye(3),'rotationorthogonality');close(np.linalg.det(R),1,'rotationdet')
                modeled=projection(s,row['K']);same(modeled,s['projected'],'storedprojection')
                for c in o['corners']:
                    if c['id'] not in H:expected[c['id']]=c['xy']
                if H:expected[H]=modeled[H]
                require(row['hidden_after']==hidden(s),'finalhiddenmask')
                require(row['hidden_set_changed']==(set(H)!=set(row['hidden_after'])),'hiddenmaskchanged flag')
                hidden_count+=len(H)
                require(s['point_ids']==[] and s['consumed_edges']==[] and s['line_edges']==edges,'line-onlyfitfactor mismatch')
                geometric=s['model_geometry_observability'];matrix=np.asarray(geometric['analytic_jacobian'],float)
                require(matrix.shape==(2*len(edges),6) and np.isfinite(matrix).all(),'Jacobian shape/finite')
                scale=np.maximum(np.linalg.norm(matrix,axis=0),1e-300)
                same(scale,geometric['normalized_column_scale'],'Jacobiancolumnscale')
                singular=np.linalg.svd(matrix/scale,compute_uv=False)
                rank=int(np.sum(singular>singular[0]*1e-10))
                same(singular,geometric['singular_values'],'JacobianSVD')
                require(rank==geometric['rank']==6 and geometric['available'],'newpose ranknot6')
                require(geometric['edges']==edges and geometric['scalar_residuals']==2*len(edges),'modeledgecount')
                require(geometric['local_observability_only'] and not geometric['global_unique_pose_proven'],'falseglobaluniqueness')
                rank_counts[str(rank)]+=1
                dims=np.asarray(s['cf_extents']);xyz=np.asarray(row['xyz'])
                if np.allclose(dims,xyz,atol=1e-6,rtol=0):Q=np.eye(3)
                else:
                    same(dims,xyz[[2,1,0]],'registered dimensions')
                    Q=np.array([[0,0,1],[0,1,0],[-1,0,0]])
                same(s['R_physical'],R@Q,'physicalaxis mapping')
            else:
                require(row['actual_pose']==row['initial_pose'],'fallbackpose must equal initial')
                require(row['hidden_after']==[] and not row['hidden_set_changed'],'fallbackhiddenstate')
                if 'model_geometry_observability' in s:
                    g=s['model_geometry_observability'];require(not g['available'] and g['rank']<6,'rankdeficient rejection flag')
            same(row['native_points'],expected,'native output '+fid)
            same(row['native_points'][8],original[8],'center preservation')
            status[expected_status]+=1
            nfev+=sum(attempt.get('nfev',0) for attempt in s.get('attempts',[]))
            starts+=sum('nfev' in attempt for attempt in s.get('attempts',[]))
        execution=self.load(NEW+'/LINE_EXECUTION.json');initial_execution=self.load(NEW+'/LINE_EXECUTION_SEALED.json')
        require(execution['counts']==initial_execution['counts'],'execution counts alteredpostscore')
        require(execution['counts']['actual_paths']==319 and execution['counts']['optimizer_reported_nfev']==nfev
                and execution['counts']['optimizer_starts']==starts,'optimizercounts mismatch')
        require(execution['counts']['actual_optimizer_residual_evaluations']>=nfev,'residualeval ledger bound')
        require(all(execution[key]==0 for key in ('new_model_forwards','new_mesh_rays','new_training_updates','new_PnP_calls')),'unexpected new inference/training/rays')
        require(execution['geometry_sealed_before_GT'] and execution['score_after_seal'] and not execution['deployment_latency'],'execution scope mismatch')
        for item in execution['code']+[execution['geometry'],execution['scored']]:self.bind(item)
        return dict(frames=319,sessions=13,status=dict(status),hidden_coordinates_projected=hidden_count,
                    independent_stored_matrix_SVD=dict(rank_counts),optimizer_starts=starts,reported_nfev=nfev,
                    residual_evaluations_ledger_only=execution['counts']['actual_optimizer_residual_evaluations'])

    def source_gate(self):
        rows=self.load(NEW+'/SOURCE_OBSERVATION_GATE_ROWS.jsonl.gz',True)
        summary=self.load(NEW+'/SOURCE_OBSERVATION_GATE.json')
        protocol=self.load(NEW+'/SOURCE_OBSERVATION_GATE_PROTOCOL.json')
        receipt=self.load(NEW+'/SOURCE_OBSERVATION_GATE_RECEIPT.json')
        ceilings={r['id']:r for r in self.load(CAUSAL+'/SOURCE_CEILING_ROWS.jsonl.gz',True)}
        rays={r['id']:r for r in self.load(CAUSAL+'/SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',True)}
        require(len(rows)==128 and len({r['id'] for r in rows})==128 and {r['index'] for r in rows}==set(range(896,1024)),'source128population')
        paths=('current_corner_and_remaining_line','all_original_lines_once',
               'current_factor_topology_with_exact_source_lines','all_lines_once_with_exact_source_geometry')
        histograms={key:Counter() for key in paths};counts=Counter();errors=[]
        for row in rows:
            fid=row['id'];require(row['source_ceiling_row_sha256']==semantic(ceilings[fid]) and row['source_ray_row_sha256']==semantic(rays[fid]),'sourceinput semanticbinding')
            lines=row['ideal_selected_line_edges'];require(len(lines)==len(set(lines)),'sourceedge duplication')
            consumed=set(row['consumed_edges']);require(row['remaining_line_edges']==[e for e in lines if e not in consumed],'remainingedge partition')
            counts['frames']+=1;counts['in_frame_ge4_corners']+=len(row['original_in_frame_corner_ids'])>=4
            structural=len(lines)>=3 and len(row['three_dimensional_edge_direction_classes'])>=2
            counts['ge3_edges_two_directions']+=structural
            for key in paths:
                p=row['factor_paths'][key];s=p['normalized_singular_values'];a=p['analytical_normalized_singular_values']
                rank=sum(v>s[0]*1e-10 for v in s) if s else 0;arank=sum(v>a[0]*1e-10 for v in a) if a else 0
                require(rank==p['normalized_rank'] and arank==p['analytical_normalized_rank'],'sourcespectrumrank')
                histograms[key][str(rank)]+=1
                vector=p['source_residual_vector_px'];require(len(vector)==p['residual_dimensions'],'sourceresidual dimension')
                close(p['source_residual_rms_px'],math.sqrt(math.fsum(v*v for v in vector)/len(vector)) if vector else None,'sourceRMS')
                close(p['source_residual_max_abs_px'],max(map(abs,vector)) if vector else None,'sourcemax')
                require(p['local_linearization_only'] and not p['global_uniqueness_certified'],'sourceglobalclaim')
                errors.append(p['finite_difference_analytical_relative_Frobenius_error'])
            current=row['factor_paths'][paths[0]]['normalized_rank'];original=row['factor_paths'][paths[1]]['normalized_rank']
            exact=row['factor_paths'][paths[3]]['normalized_rank'];analytic=row['factor_paths'][paths[3]]['analytical_normalized_rank']
            counts['current_factor_finite_difference_rank6']+=current==6
            counts['all_original_lines_finite_difference_rank6']+=original==6
            counts['all_exact_source_lines_finite_difference_rank6']+=exact==6
            counts['all_exact_source_lines_analytical_rank6']+=analytic==6
            counts['exact_local_sensitivity_and_structural_gate']+=exact==6 and structural
            counts['line_exact_rank6_current_rank_lt6']+=exact==6 and current<6
            counts['all_original_lines_rank6_but_only_one_3D_direction']+=original==6 and len(row['three_dimensional_edge_direction_classes'])==1
            counts['existing_target_definition_valid_frames']+=row['accepted_line_queries_meet_existing_target_definition']
            counts['quantization_above_half_pixel_plus_roundoff_queries']+=row['quantization']['count_above_half_pixel_plus_frozen_roundoff']
            require(not row['gate']['supervised_repair_learning_ready'] and not row['gate']['pose_success_claim'] and not row['gate']['global_unique_pose_claim'],'sourceclaimscope')
        require(dict(counts)==summary['counts'],'source gate rawcounts mismatch')
        require({key:dict(value) for key,value in histograms.items()}==summary['factor_rank_histograms'],'source rankhist mismatch')
        close(max(errors),summary['max_finite_difference_analytical_relative_error'],'source derivativeerror max')
        require(summary['actual_calls']=={key:0 for key in summary['actual_calls']},'source newcallsscope')
        for item in protocol['inputs']+[protocol['code']]:self.provenance_bind(item)
        for key in ('protocol','code','rows','math_checks'):self.provenance_bind(summary[key])
        for before,after in zip(receipt['inputs_before'],receipt['inputs_after']):
            require(before==after,'sourceinput changed duringgate');self.provenance_bind(before)
        return dict(frames=128,rawcounts=dict(counts),exact_current_rank6=histograms[paths[2]]['6'],
                    exact_all_lines_rank6=histograms[paths[3]]['6'],source_boundary_ownership_and_NONE_unresolved=True,
                    stored_spectra_recount_only=True,no_derivative_or_ray_rerun=True)

    def statistics(self):
        metrics=self.load(NEW+'/METRICS.json')
        require(metrics['schema']=='original_ROLE_all_lines_metrics_v1','metrics schema')
        arms=('BASE','N3_SUBPIX','IMAGE_ROLE_POINT_LINE',METHOD)
        self.arms={arm:{} for arm in arms}
        for row in self.load(OLD+'/FIXED_CONTROLS.jsonl.gz',True):
            if row['method'] in ('BASE','N3_SUBPIX'):self.arms[row['method']][row['id']]=row
        for row in self.load(OLD+'/LEARNED_PREDICTIONS.jsonl.gz',True):
            if row['method']=='IMAGE_ROLE_POINT_LINE':self.arms[row['method']][row['id']]=row
        self.arms[METHOD]=self.predictions
        ids=list(self.frames)
        require(set(metrics['methods'])==set(arms),'four metric arms')
        fields={'translation_cm':('translation_cm',1,'cm'),'rotation_deg':('rotation_deg',1,'degree'),
                'ADDsym_cm':('ADDsym_m',100,'cm')}
        cells=0
        for arm,data in self.arms.items():
            require(set(data)==set(ids) and len({r['session'] for r in data.values()})==13,'rawarm319/13 '+arm)
            stored=metrics['methods'][arm]
            available=[fid for fid in ids if data[fid]['pose']['available']]
            new=[fid for fid in available if data[fid]['new_pose_estimated']]
            fallback=[fid for fid in ids if data[fid]['fallback_used']]
            failed=[fid for fid in ids if not data[fid]['pose']['available']]
            for key,value in [('total_frames',319),('pose_available',len(available)),('new_pose_estimated',len(new)),
                              ('fallback_used',len(fallback)),('no_pose',len(failed)),
                              ('hidden_reprojected',sum(bool(r['hidden_reprojected']) for r in data.values()))]:
                require(stored[key]==value,'operational count '+arm+'.'+key)
            for key,value in [('available_ids',available),('new_pose_ids',new),('fallback_ids',fallback),('no_pose_ids',failed)]:
                require(stored[key]==value,'operationalIDs '+arm+'.'+key)
            require(stored['output_status_counts']==dict(Counter(data[fid]['output_status'] for fid in ids)),'status histogram '+arm)
            require(stored['failure_reasons']==dict(Counter(data[fid].get('solver',{}).get('reason','historical_fixed')
                    for fid in ids if not data[fid]['new_pose_estimated'])),'failure reasons '+arm)
            for scope,eligible in [('operational',available),('new_pose',new)]:
                for metric,(field,factor,unit) in fields.items():
                    cell=stored['metrics'][scope][metric]
                    require(cell['unit']==unit and cell['ddof']==1,'metricunits/ddof')
                    compare_distribution(cell,[data[fid]['pose'][field]*factor for fid in eligible],arm+'.'+scope+'.'+metric)
                    cells+=1
        draws=self.load(CAUSAL+'/BOOTSTRAP_SESSION_DRAWS.json.gz')
        sessions=draws['sessions'];counts=np.asarray(draws['counts'],dtype='<u2')
        require(counts.shape==(10000,13) and np.all(counts.sum(1)==13),'frozenbootstrapmatrix')
        drawhash=hashlib.sha256(counts.tobytes(order='C')).hexdigest()
        require(drawhash==draws['serialized_raw_sha256']==metrics['bootstrap']['draw_sha256'],'frozendrawhash')
        require(sessions==metrics['bootstrap']['sessions'] and metrics['bootstrap']['new_draws_generated']==0,'drawsession/noRNG')
        require(dict(Counter(self.frames[fid]['session'] for fid in ids))==metrics['bootstrap']['session_frame_counts'],'sessioncounts')
        self.bind(metrics['bootstrap']['source'])
        drawcounts=counts.astype(float);CI_cells=0
        require(set(metrics['contrasts'])=={METHOD+'_minus_'+b for b in arms[:-1]},'three contrasts')
        for comparator in arms[:-1]:
            a,b=self.arms[METHOD],self.arms[comparator]
            for scope in ('common_operational','candidate_new_pose','both_new_pose'):
                eligible=[fid for fid in ids if a[fid]['pose']['available'] and b[fid]['pose']['available']
                          and (scope=='common_operational' or a[fid]['new_pose_estimated'])
                          and (scope!='both_new_pose' or b[fid]['new_pose_estimated'])]
                cell=metrics['contrasts'][METHOD+'_minus_'+comparator][scope]
                require(cell['common_ids']==eligible and cell['excluded_ids']==[fid for fid in ids if fid not in set(eligible)]
                        and cell['common_frames']==len(eligible) and cell['denominator']==319 and cell['scope']==scope,'paired population')
                sessioncounts=np.array([sum(self.frames[fid]['session']==session for fid in eligible) for session in sessions])
                denominator=drawcounts@sessioncounts;keep=denominator>0
                require(cell['bootstrap_nonempty_resamples']==int(keep.sum()),'nonempty bootstrap count')
                for metric,(field,factor,unit) in fields.items():
                    delta=[(a[fid]['pose'][field]-b[fid]['pose'][field])*factor for fid in eligible]
                    c=cell['metrics'][metric];compare_distribution(c,delta,'paired.'+comparator+'.'+scope+'.'+metric)
                    totals=np.array([math.fsum(value for fid,value in zip(eligible,delta)
                                               if self.frames[fid]['session']==session) for session in sessions])
                    samples=(drawcounts@totals)[keep]/denominator[keep]
                    confidence=[percentile(samples,.025),percentile(samples,.975)] if len(samples) else None
                    if confidence is None:require(c['CI95'] is None,'empty bootstrap CI')
                    else:same(c['CI95'],confidence,'independent frozenbootstrap CI',atol=1e-7)
                    require(c['improved_frames']==sum(v < -1e-9 for v in delta)
                            and c['worsened_frames']==sum(v > 1e-9 for v in delta)
                            and c['unchanged_frames']==sum(abs(v)<=1e-9 for v in delta),'paireddirections')
                    for label,data in [('new_marginals',a),('comparator_marginals',b)]:
                        compare_distribution(cell[label][metric],[data[fid]['pose'][field]*factor for fid in eligible],label)
                    CI_cells+=1
        self.files_read.add(NEW+'/METRICS.csv')
        with self.path(NEW+'/METRICS.csv').open(newline='',encoding='utf-8') as stream:csvrows=list(csv.DictReader(stream))
        require(len(csvrows)==24 and len({(r['method'],r['scope'],r['metric']) for r in csvrows})==24,'CSV24unique')
        for r in csvrows:
            summary=metrics['methods'][r['method']];cell=summary['metrics'][r['scope']][r['metric']]
            for key in ('n','mean','sample_variance','sample_std','median','P90','max'):
                close(float(r[key]) if r[key] else None,cell[key],'CSV.'+key)
            require(r['unit']==cell['unit'],'CSVunit')
            for key in ('total_frames','new_pose_estimated','fallback_used','no_pose'):require(int(r[key])==summary[key],'CSVstatus')
        t=metrics['methods'][METHOD]['metrics']['operational'];baseline=metrics['methods']['N3_SUBPIX']['metrics']['operational']
        contrast=metrics['contrasts'][METHOD+'_minus_N3_SUBPIX']['common_operational']['metrics']
        require(metrics['verdict']['both_operational_means_lower']==all(t[k]['mean']<baseline[k]['mean'] for k in ('translation_cm','rotation_deg')),'meanverdict')
        require(metrics['verdict']['both_paired_CI95_strictly_below_zero']==all(contrast[k]['CI95'][1]<0 for k in ('translation_cm','rotation_deg')),'CIverdict')
        receipt=self.load(NEW+'/STATISTICS_RECEIPT.json')
        for item in receipt['inputs']+receipt['outputs']+[receipt['code'],receipt['pure_math_dependency']]:self.bind(item)
        require(all(value==0 for value in receipt['actual_calls'].values()),'statisticsnotpurearithmetic')
        return dict(raw_rows=1276,arms=4,distributions=cells,CSV_rows=24,paired_CI_cells=CI_cells,
                    frozen_draws=10000,new_random_draws=0,draw_sha256=drawhash,
                    actual_mean_verdict=metrics['verdict']['both_operational_means_lower'])

    def posthoc(self):
        metrics=self.load(NEW+'/METRICS.json');records=self.load(NEW+'/POSTHOC_ROWS.jsonl.gz',True)
        require(len(records)==1276 and len({(r['method'],r['id']) for r in records})==1276,'posthoc1276unique')
        prior_rows=self.load(OLD+'/REAL_CORRESPONDENCE_ROWS.jsonl.gz',True)
        inherited={r['id']:r for r in prior_rows if r['method']=='IMAGE_ROLE'}
        base_reference={r['id']:r for r in prior_rows if r['method']=='BASE_NO_MASK_ROBUST'}
        grouped={arm:[] for arm in self.arms}
        for r in records:
            arm,fid=r['method'],r['id'];require(arm in self.arms and fid in self.frames,'posthocidentity')
            raw=self.arms[arm][fid];base=self.arms['BASE'][fid];old=inherited[fid]
            require(r['human_states_native']==old['human_states_native'] and
                    r['permutation_native_to_canonical']==old['permutation_native_to_canonical'],'frozenphase/humanstates')
            # Old learned-input validity includes unavailable NaN observations;
            # its mask is not the reference-only validity of these new rows.
            require(r['reference_valid_native_ids']==base_reference[fid]['reference_valid_native_ids'],'referencevalid BASE identity')
            require(r['reference_kind']=='GEOMETRIC_PROXY' and not r['physical_GT_independently_validated']
                    and r['reference_read_after_complete_geometry_and_score_seal'],'referenceprovenance')
            same(r['frozen_BASE_native_points'],base['native_points'][:8],'posthocBASEcoords')
            same(r['output_native_points'],raw['native_points'][:8],'posthocoutputcoords')
            require(r['pose']==raw['pose'],'posthocpose')
            for key in ('new_pose_estimated','fallback_used','no_pose','output_status'):require(r[key]==raw[key],'posthocstatus')
            before=np.asarray(r['frozen_BASE_native_points'],float);after=np.asarray(r['output_native_points'],float)
            reference=np.asarray(r['reference_native_points_px'],float)
            valid=set(r['reference_valid_native_ids']);paired=[]
            for i in range(8):
                refvalid=i in valid and np.isfinite(reference[i]).all() and not np.all(reference[i]==-1)
                bvalid=refvalid and np.isfinite(before[i]).all() and not np.all(before[i]==-1)
                avalid=refvalid and np.isfinite(after[i]).all() and not np.all(after[i]==-1)
                close(r['reference_error_initial_native_px'][i],float(np.linalg.norm(before[i]-reference[i])) if bvalid else None,'initialproxyerror')
                close(r['reference_error_output_native_px'][i],float(np.linalg.norm(after[i]-reference[i])) if avalid else None,'outputproxyerror')
                if bvalid and avalid:paired.append(i)
            require(r['paired_corner_valid_ids']==paired,'pairedcornervalid')
            H=set(raw.get('hidden_initial',[]));reproj=set(raw.get('reprojected_ids',[]));selected=set(raw.get('selected_corner_ids',[])) if arm==METHOD else set()
            direct={i for i,s in enumerate(r['human_states_native']) if s=='DIRECT_VISIBLE'}
            self_hidden={i for i,s in enumerate(r['human_states_native']) if s=='SELF_OCCLUDED'}
            known={i for i,s in enumerate(r['human_states_native']) if s!='UNANNOTATED'}
            changed={i for i in range(8) if np.isfinite(before[i]).all() and np.isfinite(after[i]).all()
                     and np.max(np.abs(before[i]-after[i]))>1e-9}
            updates={'initial_hidden_ids':H,'hidden_reprojected_ids':reproj,'false_excluded_direct_ids':H&direct,
                     'human_self_not_marked_hidden_ids':self_hidden-H,
                     'selected_nonhidden_output_corner_ids':selected-H if raw['new_pose_estimated'] else set(),
                     'native_changed_corner_ids':changed,'direct_visible_changed_ids':changed&direct,
                     'direct_visible_reprojected_ids':reproj&direct,
                     'direct_visible_observed_corner_updated_ids':(selected-H)&direct if raw['new_pose_estimated'] else set()}
            for key,value in updates.items():require(r[key]==sorted(value),'posthocpartition '+key)
            expected_wrong=bool((H^self_hidden)&known) if arm in (METHOD,'IMAGE_ROLE_POINT_LINE') else None
            require(r['mask_wrong_on_known']==expected_wrong,'maskrelation')
            for comparator in ('BASE','N3_SUBPIX','IMAGE_ROLE_POINT_LINE'):
                a,b=raw['pose'],self.arms[comparator][fid]['pose'];available=a['available'] and b['available']
                td=a['translation_cm']-b['translation_cm'] if available else None
                rd=a['rotation_deg']-b['rotation_deg'] if available else None
                outcome=r['paired_pose_outcomes'][comparator]
                close(outcome['translation_delta_cm'],td,'posthocTdelta');close(outcome['rotation_delta_deg'],rd,'posthocRdelta')
                expected='no_pose' if not a['available'] else 'fallback' if not raw['new_pose_estimated'] else 'both_improved' if td < -1e-9 and rd < -1e-9 else 'both_worsened' if td > 1e-9 and rd > 1e-9 else 'mixed_or_equal'
                require(outcome['classification']==expected,'posthocoutcomeclassification')
            require(r['line_factor_edges']==raw.get('selected_line_edges',[]),'posthoclineedges')
            geometry=raw.get('solver',{}).get('model_geometry_observability',{})
            require(r['actual_local_model_geometry_rank']==geometry.get('rank'),'posthocmodelrank')
            close(r['actual_local_model_geometry_condition'],geometry.get('condition_number'),'posthocmodelcondition')
            grouped[arm].append(r)
        categorycells=0
        for arm,rows in grouped.items():
            require(len(rows)==319,'posthocarm319')
            for scope in ('operational','new_pose'):
                pairs={};frames={}
                for r in rows:
                    if scope=='new_pose' and not r['new_pose_estimated']:continue
                    local={}
                    for i in r['paired_corner_valid_ids']:
                        groups=[r['human_states_native'][i]]
                        for key,name in [('hidden_reprojected_ids','ALGORITHM_REPROJECTED_IDS'),
                                         ('false_excluded_direct_ids','DIRECT_VISIBLE_FALSE_EXCLUDED'),
                                         ('direct_visible_observed_corner_updated_ids','DIRECT_VISIBLE_OBSERVED_CORNER_UPDATED')]:
                            if i in r[key]:groups.append(name)
                        pair=(r['reference_error_initial_native_px'][i],r['reference_error_output_native_px'][i])
                        for name in groups:
                            pairs.setdefault(name,[]).append(pair);local.setdefault(name,[]).append(pair)
                    for name,values in local.items():frames.setdefault(name,[]).append((math.fsum(a for a,b in values)/len(values),math.fsum(b for a,b in values)/len(values)))
                stored=metrics['visibility_damage'][arm][scope];require(set(stored)==set(pairs),'visibilitycategories')
                for name,values in pairs.items():
                    c=stored[name];a=[v[0] for v in values];b=[v[1] for v in values];delta=[v[1]-v[0] for v in values]
                    require(c['corners']==len(values) and c['frames']==len(frames[name]),'visibilitydenominators')
                    for key,value in [('before',a),('after',b),('paired_delta',delta),('before_frame_mean',[v[0] for v in frames[name]]),('after_frame_mean',[v[1] for v in frames[name]])]:
                        compare_distribution(c[key],value,'visibility.'+arm+'.'+scope+'.'+name+'.'+key);categorycells+=1
                    expected={'improved':sum(v < -1e-9 for v in delta),'worsened':sum(v > 1e-9 for v in delta),
                              'unchanged':sum(abs(v)<=1e-9 for v in delta),'good5_to_bad10':sum(x<5 and y>10 for x,y in values),
                              'bad20_to_good10':sum(x>20 and y<=10 for x,y in values)}
                    require(all(c[key]==value for key,value in expected.items()),'visibilitydamagecounts')
        for label,wrong in [('wrong_on_known',True),('matches_known_self_states',False)]:
            rows=[r for r in grouped[METHOD] if r['mask_wrong_on_known']==wrong];stored=metrics['mask_known_relation'][label]
            for key,value in [('frames',len(rows)),('new_pose',sum(r['new_pose_estimated'] for r in rows)),('fallback',sum(r['fallback_used'] for r in rows)),('no_pose',sum(r['no_pose'] for r in rows))]:
                require(stored[key]==value,'maskrelationcount')
            for c in ('BASE','N3_SUBPIX','IMAGE_ROLE_POINT_LINE'):
                require(stored['outcomes'][c]==dict(Counter(r['paired_pose_outcomes'][c]['classification'] for r in rows)),'maskposeoutcomes')
        return dict(rows=1276,public_proxy_coordinate_errors_recomputed=True,visibility_distributions=categorycells,
                    private_physical_ground_truth_verified=False,mask_pose_relation_recounted=True)

    def source_surface_diagnostics(self):
        triangle=self.load(NEW+'/TRIANGLE_OWNERSHIP_ROWS.jsonl.gz',True)
        audit=self.load(NEW+'/TRIANGLE_OWNERSHIP_AUDIT.json')
        summary={};values={}
        for r in triangle:
            c=summary.setdefault(r['category'],Counter());c['n']+=1
            d=r['source_to_witness_triangle_distance_m'];plane=r['source_to_witness_plane_signed_m'];tol=r['physical_tolerance_m'];facing=r['witness_facing_cosine']
            flags={'same_primitive':r['closest_primitive']==r['witness_primitive'],'share_geometric_edge':r['shared_geometric_vertices']>=2,
                   'source_on_triangle':d<=tol,'source_on_plane':abs(plane)<=tol,
                   'not_on_triangle_but_on_plane':d>tol and abs(plane)<=tol,
                   'not_on_triangle_not_on_plane':d>tol and abs(plane)>tol,
                   'facing_positive':facing>0,'facing_negative':facing<0,'triangle_owned_facing_positive':d<=tol and facing>0}
            for key,value in flags.items():c[key]+=value
            require(r['source_on_witness_triangle_within_original_tolerance']==(d<=tol)
                    and r['source_on_witness_plane_within_original_tolerance']==(abs(plane)<=tol)
                    and r['all_original_labels_unchanged'],'triangle diagnosticflags')
            for key,value in [('triangle_distance_m',d),('plane_abs_distance_m',abs(plane)),
                              ('normal_abs_dot',r['closest_witness_absolute_normal_dot']),
                              ('semantic_tangent_normal_abs',r['witness_normal_dot_semantic_edge_abs'])]:
                if value is not None:values.setdefault(r['category']+'_'+key,[]).append(value)
        require({key:dict(value) for key,value in summary.items()}==audit['summary'],'trianglerawsummary')
        for key,data in values.items():summary_quantiles(audit['metrics'][key],data,'triangle.'+key)
        require(self.sha(NEW+'/TRIANGLE_OWNERSHIP_ROWS.jsonl.gz')[0]==audit['rows_sha256'],'trianglerowhash')
        surfaces=self.load(NEW+'/FLOAT64_SURFACE_ROWS.jsonl.gz',True)
        surface=self.load(NEW+'/FLOAT64_SURFACE_VALIDATION.json')
        rays={r['id']:r for r in self.load(CAUSAL+'/SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',True)}
        require(len(surfaces)==6463 and len({(r['id'],r['query']) for r in surfaces})==6463,'surface6463unique')
        counts={};values={};framechecks={r['id']:r for r in surface['framechecks']}
        require(len(framechecks)==128,'sourcefloat64frame128')
        for r in surfaces:
            member=r['triangle_membership'];calculated=triangle_arithmetic(r['source_X'],r['actual_triangle'])
            close(member['distance_m'],calculated['distance'],'storedtriangledistance',atol=1e-12)
            close(member['plane_signed_m'],calculated['plane'],'storedtriangleplane',atol=1e-12)
            same(member['nearest_actual_point'],calculated['nearest'],'storedtrianglenearest',atol=1e-12)
            same(member['source_barycentric'],calculated['barycentric'],'storedtrianglebarycentric',atol=1e-9)
            close(member['gram_condition_proxy'],calculated['gram_condition'],'storedtriangleGram',atol=1e-8,rtol=1e-8)
            c=counts.setdefault(r['category'],Counter());c['n']+=1
            flags={'machine_triangle_membership':r['actual_surface_machine_membership'],
                   'old_tolerance_triangle_membership':r['actual_surface_old_tolerance_membership'],
                   'triangle_degenerate_or_unknown':member is None,'bbox_strict_terminal_entry':r['actual_bbox_strict_terminal_entry'],
                   'outward_bbox_machine_terminal_entry':r['outward_bbox_machine_terminal_entry'],
                   'at_least2actual_bbox_extreme_axes':len(r['source_at_bbox_extreme_axes'])>=2,
                   'surface_and_strict_visibility':r['candidate_actual_surface_and_strict_own_mesh_visibility'],
                   'surface_and_machine_visibility':r['candidate_actual_surface_and_machine_own_mesh_visibility']}
            for key,value in flags.items():c[key]+=value
            require(r['actual_surface_machine_membership']==(member['distance_m']<=r['machine_guard_m']),'machinemembershipflag')
            require(r['actual_surface_old_tolerance_membership']==(member['distance_m']<=1e-5*rays[r['id']]['physical_diagonal_m']),'oldsurface toleranceflag')
            require(not r['source_target_approved'] and r['RGB_boundary_identity_not_proven'] and r['original_labels_unchanged'],'surfacescope')
            f=framechecks[r['id']];camera=[Fraction(v) for v in f['exact_inverse_camera_rational']]
            bounds=box_interval(camera,[Fraction(float(x)) for x in r['source_X']],
                                [Fraction(float(x)) for x in f['measured_bbox_min']],
                                [Fraction(float(x)) for x in f['measured_bbox_max']])
            point=np.asarray(r['source_X']);lower=np.asarray(f['measured_bbox_min']);upper=np.asarray(f['measured_bbox_max']);tau=r['machine_guard_m']
            inside=bool(np.all(point>=lower) and np.all(point<=upper));inside_guard=bool(np.all(point>=lower-tau) and np.all(point<=upper+tau))
            require(inside==r['source_inside_actual_bbox_exact'] and inside_guard==r['source_inside_actual_bbox_machine_guard'],'surfacebboxmembership')
            require((bounds is not None and bounds[0]==1 and inside)==r['actual_bbox_strict_terminal_entry'],'exactboxentryflag')
            if bounds is not None:
                require(bounds[0]==Fraction(r['actual_bbox_slab_near_rational']),'stored exactboxentry')
            outward=box_interval(camera,[Fraction(float(x)) for x in point],
                                 [Fraction(float(x)) for x in f['outward_bbox_min']],
                                 [Fraction(float(x)) for x in f['outward_bbox_max']])
            require((outward is not None and abs(float(outward[0]-1))<=1e-12 and inside_guard)==r['outward_bbox_machine_terminal_entry'],'outwardboxentryflag')
            require(r['source_at_bbox_extreme_axes']==np.flatnonzero(np.minimum(abs(point-lower),abs(point-upper))<=tau).tolist(),'surfaceextremeaxes')
            for name,value in [('source_to_actual_triangle_distance_m',member['distance_m']),('triangle_gram_condition_proxy',member['gram_condition_proxy'])]:
                values.setdefault(r['category']+'_'+name,[]).append(value)
        require({key:dict(value) for key,value in counts.items()}==surface['counts'],'float64rawcounts')
        for key,data in values.items():summary_quantiles(surface['metrics'][key],data,'float64.'+key)
        require(self.sha(NEW+'/FLOAT64_SURFACE_ROWS.jsonl.gz')[0]==surface['rows_sha256'],'surfacehash')
        enclosure=self.load(NEW+'/ENCLOSURE_ROWS.jsonl.gz',True);certificate=self.load(NEW+'/ENCLOSURE_VALIDATION.json')
        rays={r['id']:r for r in self.load(CAUSAL+'/SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',True)}
        tally=Counter()
        for r in enclosure:
            frame=rays[r['id']];expected_camera=[-sum(frame['R'][k][j]*frame['t'][k] for k in range(3)) for j in range(3)]
            dims=frame['dimensions']
            for q in r['queries']:
                same(q['camera_object_frame'],expected_camera,'old enclosurecamera',atol=1e-12)
                bounds=box_interval(expected_camera,q['source_X'],[-d/2 for d in dims],[d/2 for d in dims],1e-12)
                if bounds is None:require(q['enclosure_segment_interval'] is None,'enclosureslabnull')
                else:same(q['enclosure_segment_interval'],bounds,'enclosureslab',atol=1e-12)
                inside=all(abs(x)<=d/2+1e-12*max(1,d) for x,d in zip(q['source_X'],dims))
                certified=bounds is not None and abs(bounds[0]-1)<=1e-12 and inside
                require(certified==q['own_mesh_visibility_certificate'] and inside==q['source_inside_enclosure'],'enclosurecertificateflags')
                require(q['actual_triangle_membership_not_tested_here'] and not q['target_changed'],'enclosurescope')
                witness=q['existing_fixed_offset_depth_witness'];tally['examined_original_infinite_NONE']+=1
                tally['own_mesh_visibility_certified']+=certified
                tally['certified_with_fixed_offset_depth_witness']+=certified and witness
                tally['certified_without_fixed_offset_depth_witness']+=certified and not witness
                tally['witness_without_enclosure_certificate']+=witness and not certified
        require(dict(tally)==certificate['counts'],'enclosurerawcounts')
        require(not certificate['complete_supervision_proven'] and not certificate['targets_changed'],'enclosure not fullsupervision')
        for data in (audit,surface,certificate):
            for key in data:
                if key.startswith('new_'):require(data[key]==0,'source diagnostic unexpectedcalls')
        return dict(triangle_rows=len(triangle),float64_rows=6463,enclosure_queries=tally['examined_original_infinite_NONE'],
                    stored_triangle_distance_and_exact_slab_arithmetic=True,
                    raw_summary_quantiles_checked=True,actual_mesh_not_loaded=True,target_approval=False)

    def wire_proposal(self):
        rows=self.load(NEW+'/CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz',True)
        result=self.load(NEW+'/CORRECTED_TARGET_PROPOSAL_VALIDATION.json')
        protocol=self.load(NEW+'/WIRE_TARGET_PROTOCOL.json')
        witnesses=self.load(NEW+'/WIRE_FACE_WITNESSES.json')
        faces={r['triangle_id']:r for r in witnesses['normalized_actual_faces']}
        wires={r['wire_id']:r for r in witnesses['normalized_actual_wires']}
        require(len(faces)==witnesses['face_count']==180 and len(wires)==witnesses['wire_count']==90,'wireface witnesscounts')
        require(len(rows)==6463 and len({(r['id'],r['query']) for r in rows})==6463,'proposal6463unique')
        require(self.sha(NEW+'/CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz')[0]==result['rows_sha256']==witnesses['proposal_rows_sha256'],'wireproposalrowbinding')
        require(self.sha(NEW+'/WIRE_TARGET_PROTOCOL.json')[0]==result['protocol_sha256'],'wireproposalprotocolbinding')
        require(witnesses['mesh_sha256']==protocol['inputs']['scene.usd.npz']['sha256'],'witnessmeshbinding')
        frames={r['id']:r for r in self.load(CAUSAL+'/SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',True)}
        ceilings={r['id']:r for r in self.load(CAUSAL+'/SOURCE_CEILING_ROWS.jsonl.gz',True)}
        framebounds={r['id']:r for r in result['framebounds']}
        expected=set()
        for fid,frame in frames.items():
            for q in frame['queries']:
                supported=q.get('actual_mesh_point_within_original_tolerance') and q.get('in_image') and q.get('supplied_visible_mask_support_3x3') and q.get('whole_edge_has_physical_sample')
                if q['cached_target']=='POSITIVE' or (supported and q['cached_target']=='NONE' and (not q['original_finite'] or q['original_hit_in_front_of_source'])):
                    expected.add((fid,q['query']))
        require(expected=={(r['id'],r['query']) for r in rows},'proposal fixedpopulation or IGNORE leakage')
        counts={};stages=0;labelchanges=0;allselected=0;crossvalues=[];depthgaps=[]
        for row in rows:
            fid=row['id'];frame=frames[fid];q=frame['queries'][row['query']]
            require(q['query']==row['query'] and q['edge']==row['edge'] and q['cached_target']==row['original_target'],'proposalquery identity')
            require(row['source_input_row_sha256']==semantic(frame),'proposalinputsemanticbinding')
            require(row['original_target']!='IGNORE' and not row['target_changed'] and not row['approved_for_training'],'proposaloldcacheandapprovalscope')
            category=row['category'];c=counts.setdefault(category,Counter());c['n']+=1;c['proposed_'+row['proposed_target']]+=1
            stages+=len(row['candidates']);labelchanges+=row['proposed_target']!=row['original_target']
            if category=='FRONT_SURFACE_NONE':
                require(row['proposed_target']=='NONE' and q['original_hit_in_front_of_source'] and row['candidates']==[],'frontNONE preserved')
                close(row['retained_front_depth_minus_source_Z_m'],q['ray_depth_minus_source_Z_m'][0],'frontNONEdepth')
                continue
            require(category in ('ORIGINAL_POSITIVE','INFINITE_NONE') and row['proposed_target']=='POSITIVE','selectedproposal category')
            if category=='INFINITE_NONE':require(q['cached_target']=='NONE' and not q['original_finite'],'infiniteNONE origin')
            s=row['selected'];allselected+=1
            require(s['wire_id']==row['selected_wire_id'] and s in row['candidates'] and s['validated_positive_proposal'],'selectedwireidentity')
            wire=wires[s['wire_id']];dims=np.asarray(frame['dimensions'])
            endpoints=np.asarray(wire['normalized_endpoint_coords'])*dims
            same([s['endpoint_A'],s['endpoint_B']],endpoints,'actualwire scaledendpoints',atol=1e-12)
            require(s['actual_endpoint_vertex_ids']==wire['original_representative_endpoint_vertex_ids'],'wirevertexIDidentity')
            faceids=s['noncoplanar_face_witness']['triangles']
            require(s['ownership']=='NONCOPLANAR_ACTUAL_SHARED_WIRE' and faceids==wire['ownership_face_ids']
                    and set(faceids)<=set(s['incident_triangle_ids']) and q['closest_point_primitive_id'] in s['incident_triangle_ids'],'actualsharedfaceidentity')
            normalized_endpoints=[tuple(p) for p in wire['normalized_endpoint_coords']]
            triangles=[];normals=[];conditions=[]
            for pid in faceids:
                f=faces[pid];coords=f['normalized_actual_vertex_coords']
                require(set(normalized_endpoints)<=set(tuple(p) for p in coords),'both actualface share weldedendpoints')
                T=np.asarray(coords)*dims;triangles.append(T)
                info=triangle_arithmetic(T[0],T);normals.append(info['unitnormal']);conditions.append(info['gram_condition'])
            cross=float(np.linalg.norm(np.cross(*normals)));bound=256*np.finfo(float).eps*max(1,*conditions)
            require(cross>bound,'actualfaces coplanar or unresolved')
            close(s['noncoplanar_face_witness']['normal_cross_norm'],cross,'facecross',atol=1e-12)
            close(s['noncoplanar_face_witness']['numerical_bound'],bound,'facebound',atol=1e-18,rtol=1e-8)
            crossvalues.append(cross)
            point=np.asarray(s['actual_point']);u=float(Fraction(s['wire_parameter_rational']))
            require(0<=u<=1,'physical wireparameter')
            close(u,s['wire_parameter'],'wireparameter',atol=1e-12)
            same(point,endpoints[0]*(1-u)+endpoints[1]*u,'point-on-wire',atol=1e-12)
            rationalpoint=[Fraction(v) for v in s['actual_point_rational']]
            same(point,[float(v) for v in rationalpoint],'actual rationalpoint',atol=1e-12)
            for T in triangles:require(triangle_arithmetic(point,T)['distance']<=1e-12,'actual point outsideclosedownertriangle')
            R,t,K=frame['R'],frame['t'],frame['K'];uv,z=point_projection(point,R,t,K)
            same(uv,s['actual_target_uv'],'actualwire projection',atol=1e-9);close(z,s['actual_camera_Z'],'sourcecameraZ',atol=1e-12)
            a,b=EDGES[row['edge']];v=(row['query']%7+1)/8
            initial=np.asarray(ceilings[fid]['frozen_selected_points']);center=initial[a]*(1-v)+initial[b]*v;normal=np.asarray(q['normal'])
            uvA,zA=point_projection(endpoints[0],R,t,K);uvB,zB=point_projection(endpoints[1],R,t,K)
            offset,fraction=np.linalg.solve(np.column_stack([normal,-(uvB-uvA)]),uvA-center)
            require(0<=fraction<=1 and abs(offset)<=32,'frozen normal-wire intersection eligibility')
            same(uv,center+offset*normal,'queryline intersection',atol=1e-8)
            up=(fraction/zB)/((1-fraction)/zA+fraction/zB);close(up,u,'perspectivewire parameter',atol=1e-9)
            delta=float((uv-center)@normal);close(delta,s['normal_offset_px'],'target offset',atol=1e-8)
            binvalue=delta+32;lo=max(0,min(64,math.floor(binvalue)));hi=min(64,lo+1);weight=binvalue-lo
            require(s['lo']==lo and s['hi']==hi and 0<=s['weight']<=1,'target neighboringbins')
            close(s['weight'],weight,'target interpolationweight',atol=1e-8)
            physical=float(np.linalg.norm(point-np.asarray(row['original_source_X'])))
            close(physical,s['source_to_actual_wire_point_m'],'source-to-wire distance',atol=1e-12)
            require(physical<=row['physical_tolerance_m'],'original physical tolerance')
            originalkernel=[round(float(x)-.5) for x in row['original_projected_position']]
            newkernel=[round(float(x)-.5) for x in uv]
            require(originalkernel==newkernel==s['original_and_corrected_mask_kernel'] and s['supplied_visible_mask_kernel_reused_unchanged']
                    and q['supplied_visible_mask_support_3x3'],'same originalmaskkernel')
            fb=framebounds[fid];camera=[Fraction(x) for x in fb['exact_camera_rational']]
            bounds=box_interval(camera,rationalpoint,[Fraction(float(x)) for x in fb['actual_bbox_min']],
                                [Fraction(float(x)) for x in fb['actual_bbox_max']])
            require(bounds is not None,'actual enclosure intersection')
            entry=max(Fraction(0),bounds[0]);require(entry==Fraction(s['enclosure_entry_rational']),'exact enclosure entry')
            gap=float((1-entry)*Fraction(float(z)))
            close(gap,s['worst_first_surface_camera_Z_gap_m'],'enclosure worstdepthgap',atol=1e-12)
            require(0<=gap<=row['original_depth_tolerance_m'],'unchanged original visibility tolerance')
            depthgaps.append(gap)
        require({key:dict(value) for key,value in counts.items()}==result['counts'],'wireproposal categorycounts')
        require(stages==result['stagecounts']['candidate_wire_checks'],'wirecandidatecounts')
        require(result['proposal_only'] and not result['source_targets_overwritten'] and not result['ready_for_learning'],'proposal not fullpreparation/learningapproval')
        require(labelchanges==2164 and allselected==4497,'proposalclass-change count')
        return dict(source_test_families=128,rows=6463,selected_actual_owned_wire_proposals=4497,
                    original_NONE_proposed_POSITIVE=labelchanges,front_NONE_preserved=1966,
                    actual_shared_face_coordinates_checked=True,normal_query_intersection_and_two_bin_targets=True,
                    minimum_cross_normal=min(crossvalues),maximum_enclosure_depth_gap_m=max(depthgaps),
                    mesh_cache_not_loaded=True,full_source_train_cal_preparation_checked=False,ready_for_learning=False,
                    target_changed_false_means_original_cache_unmodified=True)

    def full_source_preparation(self):
        result=self.load(NEW+'/FULL_SOURCE_PREPARATION.json')
        protocol=self.load(NEW+'/FULL_SOURCE_PREPARATION_PROTOCOL.json')
        rows=self.load(NEW+'/FULL_SOURCE_TARGET_ROWS.jsonl.gz',True)
        ceilings=self.load(CAUSAL+'/SOURCE_CEILING_ROWS.jsonl.gz',True)
        split=self.load(OLD+'/SOURCE_FAMILY_SPLIT.json')
        require(len(rows)==len(ceilings)==len(split['records'])==1024,'fullsource1024families')
        require(len({r['id'] for r in rows})==len({r['family'] for r in rows})==1024,'fullsource uniqueIDs/families')
        require([r['index'] for r in rows]==list(range(1024)),'fullsource frozen family ordering')
        require(result['family_count']==protocol['fixed_families']==split['family_count']==1024,'fullsource familymetadata')
        expected_splits={'train':768,'calibration':128,'source_test':128}
        require(result['splits']==protocol['split_counts']==split['counts']==expected_splits,'fullsource splitmetadata')
        require(result['complete_preparation'] and not result['ready_for_learning'] and not protocol['ready_for_learning'],
                'fullsource complete preparation is not complete no-match supervision')
        require(not result['source_cache_mutated'] and result['family_order_unchanged'] and result['no_real_GT_reads']
                and result['no_calibration_or_test_selection'] and result['augmentation']=='existing_original_only',
                'fullsource scope metadata')
        for key in ('new_rays','new_heads','new_detector','new_PnP','new_RGB','new_training_updates'):
            require(result[key]==protocol[key]==0,'fullsource unexpected '+key)
        for name,field in [('FULL_SOURCE_PREPARATION_PROTOCOL.json','protocol_sha256'),
                           ('FULL_SOURCE_TARGET_ROWS.jsonl.gz','rows_sha256'),
                           ('PREPARED_TARGETS.npz','prepared_target_arrays_sha256')]:
            require(self.sha(NEW+'/'+name)[0]==result[field],'fullsource '+name+' binding')
        script='scripts/research/pallet_kp_supervision_gate_20261010_v1/full_source_prepare.py'
        require(self.sha(script)[0]==result['code_sha256'],'fullsource codebinding')
        public_inputs={
            'actual_wire_proposal.py':'scripts/research/pallet_kp_supervision_gate_20261010_v1/actual_wire_proposal.py',
            'WIRE_TARGET_PROTOCOL.json':NEW+'/WIRE_TARGET_PROTOCOL.json',
            'CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz':NEW+'/CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz',
            'SOURCE_CEILING_ROWS.jsonl.gz':CAUSAL+'/SOURCE_CEILING_ROWS.jsonl.gz',
            'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz':CAUSAL+'/SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',
            'full_source_prepare.py':script,
        }
        for name,path in public_inputs.items():self.bind(protocol['inputs'][name],path)
        require(result['frozen_wire_math_sha256']==protocol['inputs']['actual_wire_proposal.py']['sha256'],'fullsource fixedwire binding')
        old_ceiling=self.load(CAUSAL+'/SOURCE_CEILING.json')
        old_bindings={Path(item['path']).name:item for item in old_ceiling['bindings']}
        old_order={Path(item['path']).name:item for item in self.load(OLD+'/REVIEW_LOSS_MASK_CHECK.json')['cache_bindings']}
        features=result['feature_compatibility']
        require(features['sha256']==protocol['inputs']['features.npy']['sha256']==old_bindings['features.npy']['sha256']
                and features['shape']==[1024,84,28,65] and features['dtype']=='float16' and not features['regenerated'],
                'fullsource unchanged feature binding')
        require(features['cache_manifest_sha256']==protocol['inputs']['CACHE_MANIFEST.json']['sha256']==old_bindings['CACHE_MANIFEST.json']['sha256'],
                'fullsource unchanged cache manifest binding')
        require(features['batch_order_sha256']==protocol['inputs']['order.npy']['sha256']==old_order['order.npy']['sha256'],
                'fullsource unchanged batch order binding')
        target_path=NEW+'/PREPARED_TARGETS.npz';self.files_read.add(target_path)
        with np.load(self.path(target_path),allow_pickle=False) as data:
            require(set(data.files)=={'lo','hi','weight','valid','source_index','partitions'},'preparedarray schema')
            arrays={key:data[key] for key in data.files}
        for key,dtype in [('lo','int16'),('hi','int16'),('weight','float32'),('valid','bool')]:
            require(arrays[key].shape==(1024,84) and str(arrays[key].dtype)==dtype,'preparedarray '+key+' shape/dtype')
        require(arrays['source_index'].shape==(1024,) and arrays['source_index'].dtype.kind in 'iu'
                and np.array_equal(arrays['source_index'],np.arange(1024)),'preparedarray familyorder')
        require(arrays['partitions'].shape==(1024,) and arrays['partitions'].dtype.kind=='U'
                and arrays['partitions'].tolist()==[r['partition'] for r in rows],'preparedarray splitorder')
        require(np.isfinite(arrays['weight']).all(),'preparedarray nonfinite weight')
        proposals={(r['id'],r['query']):r for r in self.load(NEW+'/CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz',True)}
        counters={p:Counter() for p in expected_splits};families=Counter();parity=0
        for index,(row,source,family) in enumerate(zip(rows,ceilings,split['records'])):
            require(source['index']==index and all(row[k]==source[k]==family[k] for k in ('id','family','partition')),
                    'fullsource original family identity/split join')
            part=row['partition'];families[part]+=1
            require(part==('train' if index<768 else 'calibration' if index<896 else 'source_test'),
                    'fullsource fixed partition range')
            require(row['source_ceiling_semantic_sha256']==semantic(source) and row['feature_cache_index']==index
                    and not row['source_cache_mutated'] and not row['source_features_regenerated']
                    and row['augmentation']=='existing_P0_original_only','fullsource input and feature identity')
            require(row['raw_hw']==source['raw_hw'] and row['frozen_selected_points']==source['frozen_selected_points'],
                    'fullsource unchanged observations')
            require(len(row['queries'])==84 and [q['query'] for q in row['queries']]==list(range(84)),
                    'fullsource queryIDs')
            for q in row['queries']:
                i=q['query'];old='IGNORE' if not source['targets']['valid'][i] else ('POSITIVE' if source['targets']['lo'][i]<65 else 'NONE')
                state=q['proposed_target'];require(state in ('POSITIVE','NONE','IGNORE'),'fullsource targetclass')
                require(q['original_target']==old and q['edge']==i//7 and q['proposal_label_changed']==(old!=state)
                        and not q['source_cache_mutated'] and not q['approved_for_training'],
                        'fullsource original target/class-change identity')
                require(q['unsupported_physical_edge']==(q['edge'] not in source['physical_edge_ids'])
                        and q['earlier_raw_depth_available']==(part=='source_test'),'fullsource support/depth metadata')
                if q['unsupported_physical_edge'] or old=='IGNORE':require(state=='IGNORE','fullsource unsupported/IGNORE leak')
                counters[part][state]+=1
                if old!=state:counters[part]['changed_'+old+'_to_'+state]+=1
                lo=int(arrays['lo'][index,i]);hi=int(arrays['hi'][index,i]);weight=float(arrays['weight'][index,i])
                require(bool(arrays['valid'][index,i])==(state!='IGNORE'),'preparedarray valid mask/class')
                if state=='POSITIVE':
                    require(lo==q['lo'] and hi==q['hi'] and 0<=lo<=64 and hi==min(64,lo+1)
                            and weight==float(np.float32(q['weight'])) and 0<=q['weight']<=1,'preparedarray positive two-bin target')
                    value=q['normal_offset_px']+32
                    require(-32<=q['normal_offset_px']<=32 and lo==min(64,max(0,math.floor(value))),
                            'fullsource normal-offset target bin')
                    close(q['weight'],value-lo,'fullsource target binweight',atol=1e-8)
                    require(q['ownership']=='NONCOPLANAR_ACTUAL_SHARED_WIRE' and len(q['actual_face_ids'])==2,
                            'fullsource stored ownership category')
                else:
                    require(lo==hi==65 and weight==0,'preparedarray NONE/IGNORE canonical value')
                    if state=='NONE':
                        require(part=='source_test' and old=='NONE' and q['retained_front_depth_minus_source_Z_m']<0,
                                'fullsource genuine NONE unavailable in train/calibration')
                        require(q['lo']==q['hi']==65 and q['weight']==0,'fullsource NONE record')
                original=proposals.get((row['id'],i))
                if original is not None:
                    require(state==original['proposed_target'],'fullsource frozen proposal class parity')
                    if state=='POSITIVE':
                        selected=original['selected']
                        require(q['actual_wire_id']==selected['wire_id'] and all(q[k]==selected[k]
                                for k in ('actual_point','actual_target_uv','lo','hi','weight')),
                                'fullsource frozen proposal exact selected-field parity')
                    parity+=1
        require(dict(families)==expected_splits,'fullsource raw splitcounts')
        require({key:dict(value) for key,value in counters.items()}==result['counts'],'fullsource raw target/change counts')
        require(parity==len(proposals)==6463 and result['frozen_source_test_parity']=={'compared':6463,'exact_match':6463},
                'fullsource frozen6463 exactparity')
        require(counters['train']['NONE']==counters['calibration']['NONE']==0 and counters['source_test']['NONE']==1966,
                'fullsource no-match readiness blocker')
        execution=result['execution']
        require(execution['scene_builds']==execution['closest_point_API_calls']==896
                and execution['closest_point_count']<=protocol['max_query_closest_points']==75264,
                'fullsource reported preparation call caps')
        return dict(families=1024,queries=1024*84,splits=dict(families),counts={p:dict(v) for p,v in counters.items()},
                    target_arrays_exactly_join_public_raw_labels=True,original_family_split_and_query_order=True,
                    frozen_source_test_exact_parity=parity,feature_and_batch_order_metadata_bindings_unchanged=True,
                    private_feature_arrays_or_order_not_opened=True,train_calibration_genuine_NONE=0,
                    ready_for_learning=False,full_train_calibration_geometry_ownership_not_independently_recomputed=True,
                    no_model_ray_pose_execution=True)

    def manifest(self):
        path=self.path(NEW+'/REVIEW_MANIFEST.json')
        if not path.exists():
            require(not self.require_manifest,'required REVIEW_MANIFEST.json is absent')
            return dict(present=False,required=False)
        data=self.load(NEW+'/REVIEW_MANIFEST.json');entries=data['files']
        require(len(entries)==len({i['path'] for i in entries}),'duplicate manifest path')
        for item in entries:
            require(Path(item['path']).name not in ('REVIEW_CHECKS.json','REVIEW_MANIFEST.json'),'hash cycle inmanifest')
            self.bind(item)
        return dict(present=True,files=len(entries),required=self.require_manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3],help='public checkout root')
    parser.add_argument('--output',type=Path,help='new JSON receipt; own default REVIEW_CHECKS.json may be regenerated')
    parser.add_argument('--require-manifest',action='store_true')
    args=parser.parse_args();root=args.root.resolve();own=root/NEW/'REVIEW_CHECKS.json'
    if own.resolve()!=own:
        parser.error('The own receipt path or its ancestors must not be symlinks')
    output=(args.output or own).resolve()
    if output.suffix!='.json' or (output.exists() and output!=own):
        parser.error('Refusing to overwrite a source, manifest, input or existing research artifact')
    if output.is_relative_to(root) and output!=own:
        parser.error('Inside the checkout only this phase REVIEW_CHECKS.json can be written')
    review=Review(root,args.require_manifest)
    for name,function in [('protected_prior_publication',review.protected),
                          ('sealed_geometry_projection_and_rank',review.geometry),
                          ('source_observation_gate',review.source_gate),
                          ('raw_pose_statistics_and_frozen_CI',review.statistics),
                          ('native_proxy_errors_visibility_and_mask_relation',review.posthoc),
                          ('source_triangle_surface_enclosure_arithmetic',review.source_surface_diagnostics),
                          ('actual_owned_wire_proposal_arithmetic',review.wire_proposal),
                          ('full_source_preparation_arrays_and_readiness',review.full_source_preparation),
                          ('publication_manifest',review.manifest)]:
        review.group(name,function)
    result=dict(schema='public_original_image_role_all_lines_review_v1',passed=all(g['passed'] for g in review.groups),
                checks=review.groups,limits=LIMITS,public_files_read=sorted(review.files_read),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions=dict(python=sys.version.split()[0],numpy=np.__version__),
                execution=dict(private_truth_reads=0,model_forwards=0,training_updates=0,pose_fits=0,PnP_calls=0,rays=0))
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_name(output.name+'.pending')
    if pending.exists():parser.error('Refusing to overwrite an existing pending receipt')
    with pending.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    pending.replace(output)
    print(json.dumps(dict(passed=result['passed'],groups=len(review.groups),receipt=str(output))))
    for group in review.groups:
        if not group['passed']:print(group['name']+': '+group['error'],file=sys.stderr)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
