"""Verify published source-supervision depth recovery without ray or model calls.

Run with Python >=3.9 and NumPy. --root denotes the public repository checkout,
not a private dataset. The verifier checks stored witness arithmetic and target
transitions. It does not independently authenticate first-surface engine output,
read private mesh/RGB/masks, or establish real pose improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

NEW = '_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
GATE = '_docs/experiments/pallet_kp_supervision_gate_20261010_v1'
CAUSAL = '_docs/experiments/pallet_kp_difficulty_20261010_v1'
OLD = '_docs/experiments/pallet_observation_refiner_20261009_v1'
CODE = 'scripts/research/pallet_kp_supervision_repair_20261010_v1'
LIMITS = [
    'Public hashes, frozen populations, stored numerical witnesses and target-array transitions are checked.',
    'The private actual mesh, source annotations, masks, feature/order arrays and evaluation truth are not opened.',
    'Before/after equality of 1,844 private input bindings is checked as recorded receipts, not by rereading those private files.',
    'First-surface authenticity and absence of an earlier mesh hit depend on the recorded engine execution; no rays are repeated.',
    'Source mixed-loss readiness is distinct from training authorization, real transfer, graph sufficiency and pose improvement.',
    'No detector/head inference, loss backward, optimizer update, pose fit, PnP or rendering is performed.',
    'The zero-update CPU receipt is joined to public targets and its recorded gradient fields checked; private features, head logits and backwards are not reproduced.',
]


def require(value, label):
    if not bool(value):
        raise AssertionError(label)


def close(a, b, label, atol=1e-10, rtol=1e-10):
    if a is None or b is None:
        require(a is None and b is None, label+': null mismatch')
        return
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a),float(b),abs_tol=atol,rel_tol=rtol),
            label+': numeric mismatch '+repr((a,b)))


def same(a, b, label, atol=1e-10):
    x,y=np.asarray(a,float),np.asarray(b,float)
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            and np.allclose(x,y,atol=atol,rtol=1e-10),label+': array mismatch')


def percentile(values, q):
    values=sorted(float(v) for v in values)
    if not values:return None
    position=(len(values)-1)*q;lo,hi=math.floor(position),math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


class Review:
    def __init__(self, root, require_manifest):
        self.root=root
        self.require_manifest=require_manifest
        self.cache={}
        self.files_read=set()
        self.groups=[]
        self.changed=set()

    def path(self, name):
        require(not Path(name).is_absolute(),'private/absolute read is forbidden: '+str(name))
        path=(self.root/name).resolve()
        require(path.is_relative_to(self.root),'read escapes public checkout: '+str(name))
        return path

    def load(self, name, rows=False):
        if name not in self.cache:
            path=self.path(name);self.files_read.add(str(path.relative_to(self.root)))
            with (gzip.open if path.suffix=='.gz' else open)(path,'rt',encoding='utf-8') as stream:
                self.cache[name]=[json.loads(line) for line in stream if line.strip()] if rows else json.load(stream)
        return self.cache[name]

    def sha(self, name):
        path=self.path(name);self.files_read.add(str(path.relative_to(self.root)))
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        return digest.hexdigest(),path.stat().st_size

    def bind(self, item, name=None):
        name=item['path'] if name is None else name
        sha,size=self.sha(name)
        require(sha==item['sha256'] and size==item['bytes'],'binding mismatch: '+name)

    def arrays(self, name):
        path=self.path(name);self.files_read.add(str(path.relative_to(self.root)))
        with np.load(path,allow_pickle=False) as data:
            require(set(data.files)=={'lo','hi','weight','valid','source_index','partitions'},'target-array schema')
            return {key:data[key] for key in data.files}

    def group(self, name, function):
        try:self.groups.append(dict(name=name,passed=True,details=function()))
        except Exception as error:self.groups.append(dict(name=name,passed=False,error=type(error).__name__+': '+str(error)))

    def prior(self):
        prior=self.load(NEW+'/PRIOR_PUBLICATION_BINDINGS.json');items=prior['files']
        require(len(items)==len({i['path'] for i in items})==257,'expected257 unique prior publication files')
        for item in items:self.bind(item)
        gate=self.load(GATE+'/REVIEW_MANIFEST.json')
        require(len(gate['files'])==65,'expected65 frozen prior-gate manifest bindings')
        for item in gate['files']:self.bind(item)
        relocation=self.load(NEW+'/PUBLICATION_RELOCATION.json')
        for item in relocation['files']:
            require(item['byte_exact'],'relocated execution artifact changed')
            self.bind(item)
        return dict(prior_files=257,prior_gate_manifest_files=65,parent_commit=prior['parent_commit'],
                    unchanged=True,byte_exact_execution_copies=len(relocation['files']),private_paths_not_opened=True)

    def bindings_and_plan(self):
        result=self.load(NEW+'/DEPTH_RECOVERY_VALIDATION.json')
        protocol=self.load(NEW+'/DEPTH_RECOVERY_V3_PROTOCOL.json')
        require(result['schema']=='missing_front_surface_depth_recovery_v3' and result['complete'], 'completed V3 schema')
        for name,key in [('DEPTH_RECOVERY_V3_PROTOCOL.json','protocol_sha256'),
                         ('RECOVERED_DEPTH_ROWS.jsonl.gz','raw_depth_rows_sha256'),
                         ('READY_SOURCE_TARGET_ROWS.jsonl.gz','ready_rows_sha256'),
                         ('READY_PREPARED_TARGETS.npz','ready_arrays_sha256')]:
            require(self.sha(NEW+'/'+name)[0]==result[key],name+' hash')
        require(self.sha(CODE+'/recover_front_none_v3.py')[0]==result['code_sha256'],'executed V3 code hash')
        require(result['inputs']==protocol['inputs'],'V3 inherited fixed input bindings')
        mapping={
            'recover_front_none.py':CODE+'/recover_front_none.py',
            'FULL_SOURCE_PREPARATION_PROTOCOL.json':GATE+'/FULL_SOURCE_PREPARATION_PROTOCOL.json',
            'FULL_SOURCE_PREPARATION.json':GATE+'/FULL_SOURCE_PREPARATION.json',
            'FULL_SOURCE_TARGET_ROWS.jsonl.gz':GATE+'/FULL_SOURCE_TARGET_ROWS.jsonl.gz',
            'PREPARED_TARGETS.npz':GATE+'/PREPARED_TARGETS.npz',
            'SOURCE_CEILING_ROWS.jsonl.gz':CAUSAL+'/SOURCE_CEILING_ROWS.jsonl.gz',
            'training.py':'scripts/research/pallet_observation_refiner_20261009_v1/training.py',
            'DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz':NEW+'/DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz',
        }
        for name,path in mapping.items():self.bind(protocol['inputs'][name],path)
        metadata=self.load(NEW+'/ACTUAL_MESH_METADATA.json')
        before=self.load(NEW+'/FIXED_INPUT_PROTECTION_BEFORE.json')
        after=self.load(NEW+'/FIXED_INPUT_PROTECTION_AFTER.json')
        require(len(protocol['fixed_inputs'])==result['fixed_input_protection']['files']==1844,'fixed1844 binding count')
        require(before['bindings']==after['bindings']==protocol['fixed_inputs'] and before['all_match_protocol']
                and after['all_match_protocol'] and after['all_match_before'],'private recorded fixed-input equality')
        require(before['new_rays']==0 and after['new_rays']==13757,'fixed receipt execution stages')
        require(self.sha(NEW+'/FIXED_INPUT_PROTECTION_BEFORE.json')[0]==result['fixed_input_protection']['before_receipt_sha256']
                and self.sha(NEW+'/FIXED_INPUT_PROTECTION_AFTER.json')[0]==result['fixed_input_protection']['after_receipt_sha256'],
                'fixed receipt hashes')
        self.bind(protocol['fixed_inputs']['ORIGINAL_ACTUAL_MESH/scene.usd.json'],NEW+'/ACTUAL_MESH_METADATA.json')
        require(metadata['triangles']==426540 and metadata['vertices']==413451,'actual mesh metadata counts')
        require(protocol['inputs']['scene.usd.npz']==protocol['fixed_inputs']['ORIGINAL_ACTUAL_MESH/scene.usd.npz'],
                'mesh NPZ metadata binding')
        for version,name in [('v1_history','DEPTH_RECOVERY_PROTOCOL.json'),('v2_history','DEPTH_RECOVERY_V2_PROTOCOL.json')]:
            require(self.sha(NEW+'/'+name)[0]==protocol[version]['protocol_sha256'],'planning history protocol hash')
            require(protocol[version][version[:2]+'_casts_executed']==0,'planning history cast count')
            script=CODE+('/recover_front_none.py' if version=='v1_history' else '/recover_front_none_v2.py')
            require(self.sha(script)[0]==protocol[version]['code_sha256'],'planning history code hash')
        require(protocol['created_before_first_cast'] and protocol['population_unchanged']
                and protocol['plan_reused_without_query_refreeze'],'fixed execution population')
        plan=self.load(NEW+'/DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz',True)
        prepared=self.load(GATE+'/FULL_SOURCE_TARGET_ROWS.jsonl.gz',True)
        sources={r['index']:r for r in self.load(CAUSAL+'/SOURCE_CEILING_ROWS.jsonl.gz',True)}
        require(len(plan)==896 and [f['index'] for f in plan]==list(range(896)),'fixed planned896 scene order')
        tally=Counter();planned=set()
        for frame in plan:
            index=frame['index'];original=prepared[index];source=sources[index]
            require(all(frame[key]==original[key] for key in ('id','index','partition','K','R','t','dimensions')),'plan source identity/geometry')
            require(frame['source_annotation']==source['source_annotation'],'plan annotation metadata identity')
            close(frame['depth_tolerance_m'],.001*np.linalg.norm(frame['dimensions']),'original depth tolerance')
            require(len({q['query'] for q in frame['queries']})==len(frame['queries']),'duplicate planned query')
            for q in frame['queries']:
                i=q['query'];prior=original['queries'][i]
                require(0<=i<84 and q['edge']==i//7 and q['edge'] in source['physical_edge_ids']
                        and prior['original_target']=='NONE' and prior['proposed_target']=='IGNORE', 'planned query target/support identity')
                planned.add((index,i));tally[frame['partition']]+=1
                require(0<=q['fraction']<=1 and abs(q['offset_px'])<=32,'planned segment/search eligibility')
                h,w=original['raw_hw'];x,y=q['position']
                require(0<=x<w and 0<=y<h and q['mask_kernel']==[round(x-.5),round(y-.5)],'planned ROI/mask kernel')
                camera=np.asarray(frame['R'])@np.asarray(q['source_X'])+np.asarray(frame['t'])
                homogeneous=np.asarray(frame['K'])@camera
                close(q['source_camera_Z'],camera[2],'planned target camera depth')
                require(camera[2]>0,'planned positive target depth')
                same(q['position'],homogeneous[:2]/homogeneous[2],'planned target projection',atol=1e-8)
        require(dict(tally)==protocol['planned_split_rays']=={'train':11783,'calibration':1974},'planned split ray counts')
        require(len(planned)==protocol['max_rays']==protocol['counts']['planned_rays']==13757
                and protocol['max_scene_builds']==896,'fixed ray/scene budget')
        residual=sum(q['original_target']=='NONE' and q['proposed_target']=='IGNORE' for r in prepared[:896] for q in r['queries'])
        require(residual==protocol['counts']['residual_original_NONE']==15844
                and residual-len(planned)==protocol['counts']['supplied_mask_absence_unchanged_IGNORE']+
                protocol['counts']['outside_segment_search_unchanged_IGNORE'],'unrecovered fixed population arithmetic')
        for key in ('new_RGB','new_heads','new_detector','new_PnP','new_training_updates'):
            require(protocol[key]==result[key]==0,'unexpected execution '+key)
        return dict(frozen_frames=896,planned_rays=13757,recorded_private_bindings=1844,
                    recorded_private_bindings_before_after_equal=True,private_bindings_not_reread=True,
                    unchanged_depth_tolerance=True,planning_V1_V2_rays=0,new_training_updates=0)

    def depth_witnesses(self):
        result=self.load(NEW+'/DEPTH_RECOVERY_VALIDATION.json')
        raw=self.load(NEW+'/RECOVERED_DEPTH_ROWS.jsonl.gz',True)
        plan=self.load(NEW+'/DEPTH_RECOVERY_PLAN_ROWS.jsonl.gz',True)
        triangle_count=self.load(NEW+'/ACTUAL_MESH_METADATA.json')['triangles']
        require(len(raw)==len(plan)==896,'raw896 scene frames')
        counts=Counter();differences=[];seen=set();self.changed=set()
        for sequence,(frame,frozen) in enumerate(zip(raw,plan),1):
            require({k:v for k,v in frame.items() if k not in ('queries','new_rays','scene_build_index')}==
                    {k:v for k,v in frozen.items() if k!='queries'},'raw/frozen frame identity')
            require(frame['scene_build_index']==sequence and frame['new_rays']==len(frame['queries'])==len(frozen['queries']),
                    'raw execution ordering/query count')
            R,t,K=np.asarray(frame['R']),np.asarray(frame['t']),np.asarray(frame['K'])
            origin=-R.T@t
            for q,locked in zip(frame['queries'],frozen['queries']):
                require(all(q[k]==v for k,v in locked.items()),'raw/frozen query identity')
                key=(frame['index'],q['query']);require(key not in seen,'duplicate raw query');seen.add(key)
                x,y=q['position'];direction=np.asarray([(x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],1.])@R
                ray=np.concatenate([origin,direction]).astype(np.float32).astype(float)
                require(np.array_equal(np.asarray(q['ray_float32']),ray),'exact original float32 ray arithmetic')
                depth=q['t_hit_original_ray_parameter'];finite=depth is not None and math.isfinite(depth)
                if finite:
                    require(depth==float(np.float32(depth)),'stored engine float32 depth')
                    world=ray[:3]+depth*ray[3:];camera_Z=float((R@world+t)[2])
                    same(q['actual_hit_point_world'],world,'stored hit-point arithmetic')
                    close(q['recomputed_hit_source_camera_Z'],camera_Z,'recomputed hit camera_Z')
                    close(q['recomputed_camera_Z_minus_original_ray_parameter_m'],camera_Z-depth,'camera vs ray parameter')
                    bary=np.asarray(q['primitive_uv'],float);normal=np.asarray(q['primitive_normal'],float);pid=q['primitive_id']
                    require(bary.shape==(2,) and normal.shape==(3,),'finite witness shapes')
                    guards=dict(finite_nonnegative_depth=depth>=0,
                                valid_primitive_index=isinstance(pid,int) and 0<=pid<triangle_count,
                                finite_closed_triangle_barycentrics=bool(np.isfinite(bary).all() and (bary>=0).all() and bary.sum()<=1),
                                finite_nonzero_normal=bool(np.isfinite(normal).all() and np.isfinite(np.linalg.norm(normal)) and np.linalg.norm(normal)>0),
                                finite_recomputed_camera_hit=bool(np.isfinite(world).all() and math.isfinite(camera_Z)))
                    require(q['witness_guards']==guards,'finite witness guards independently reconstructed')
                    legacy_delta=depth-q['source_camera_Z'];camera_delta=camera_Z-q['source_camera_Z']
                    close(q['original_ray_parameter_minus_target_camera_Z_m'],legacy_delta,'legacy front-depth delta')
                    close(q['recomputed_camera_Z_minus_target_camera_Z_m'],camera_delta,'camera front-depth delta')
                    legacy=legacy_delta < -frame['depth_tolerance_m'];camera=camera_delta < -frame['depth_tolerance_m']
                    negative=all(guards.values()) and legacy and camera
                    differences.append(camera_Z-depth)
                else:
                    require(depth is None and all(q[k] is None for k in ('actual_hit_point_world',
                            'recomputed_hit_source_camera_Z','recomputed_camera_Z_minus_original_ray_parameter_m',
                            'primitive_id','primitive_uv','primitive_normal','original_ray_parameter_minus_target_camera_Z_m',
                            'recomputed_camera_Z_minus_target_camera_Z_m')),'infinite witness serialized unknowns')
                    require(all(not q['witness_guards'][k] for k in ('finite_nonnegative_depth','valid_primitive_index',
                            'finite_nonzero_normal','finite_recomputed_camera_hit')),'infinite invalid guards')
                    # Infinite primitive barycentrics were not retained. Their stored
                    # guard is irrelevant once finite/primitive guards are false.
                    guards=q['witness_guards'];legacy=camera=negative=False
                require(q['legacy_parameter_front']==legacy and q['recomputed_camera_front']==camera
                        and q['proved_front_surface_NONE']==negative
                        and q['recovered_target']==('NONE' if negative else 'IGNORE'),'both-depth NONE decision')
                counts[frame['partition']+'_'+q['recovered_target']]+=1
                counts['finite' if finite else 'infinite']+=1
                if legacy!=camera:counts['legacy_vs_recomputed_front_disagreement']+=1
                for k,v in guards.items():
                    if not v:counts['invalid_'+k]+=1
                if negative:self.changed.add(key)
        require(len(seen)==result['new_rays']==13757 and len(self.changed)==13754,'raw witness decision totals')
        require(dict(counts)==result['counts'],'raw witness summary counts')
        require(result['scene_builds']==result['cast_rays_API_calls']==896,'reported actual scene/cast calls')
        stored=result['camera_vs_original_parameter_m']
        require(stored['count']==len(differences)==13754,'camera discrepancy count')
        for k,v in [('mean',math.fsum(differences)/len(differences)),('max_absolute',max(abs(v) for v in differences)),
                    ('median_absolute',percentile([abs(v) for v in differences],.5)),
                    ('p90_absolute',percentile([abs(v) for v in differences],.9))]:
            close(stored[k],v,'camera discrepancy '+k,atol=1e-13)
        return dict(witnesses=13757,finite_both_guard_NONE=13754,infinite_retained_IGNORE=3,
                    float32_ray_and_world_hit_camera_Z_recomputed=True,finite_primitive_bary_normal_guards=True,
                    infinite_bary_evidence_not_stored_and_not_needed=True,first_surface_engine_authenticity_not_independently_proven=True)

    def actual_face_arithmetic(self):
        faces=self.load(NEW+'/RECOVERED_FRONT_FACE_WITNESSES.json')
        result=self.load(NEW+'/RECOVERED_FRONT_FACE_VALIDATION.json')
        depth_result=self.load(NEW+'/FRONT_TRIANGLE_DEPTH_VALIDATION.json')
        mesh=self.load(NEW+'/ACTUAL_MESH_METADATA.json')
        protocol=self.load(NEW+'/DEPTH_RECOVERY_V3_PROTOCOL.json')
        table={r['primitive_id']:r for r in faces['triangles']}
        require(len(table)==len(faces['triangles'])==faces['unique_faces']==result['unique_faces']==5131,'actual face5131 uniqueIDs')
        require(faces['mesh_cache_sha256']==protocol['inputs']['scene.usd.npz']['sha256'],'actual face mesh binding')
        for name,field in [('RECOVERED_FRONT_FACE_WITNESSES.json','face_table_sha256'),
                           ('RECOVERED_FRONT_FACE_RECONSTRUCTION_ROWS.jsonl.gz','reconstruction_rows_sha256'),
                           ('RECOVERED_FRONT_FACE_PROTOCOL.json','protocol_sha256')]:
            require(self.sha(NEW+'/'+name)[0]==result[field],'actual face '+name+' hash')
        require(self.sha(CODE+'/export_recovered_front_faces.py')[0]==result['code_sha256'],'actual face export code binding')
        for name,field in [('FRONT_TRIANGLE_DEPTH_PROTOCOL.json','protocol_sha256'),('FRONT_TRIANGLE_DEPTH_ROWS.jsonl.gz','rows_sha256')]:
            require(self.sha(NEW+'/'+name)[0]==depth_result[field],'triangle depth '+name+' hash')
        require(self.sha(CODE+'/check_front_triangle_depth.py')[0]==depth_result['code_sha256'],'triangle depth code binding')
        require(faces['raw_depth_rows_sha256']==result['raw_depth_rows_sha256']==
                self.sha(NEW+'/RECOVERED_DEPTH_ROWS.jsonl.gz')[0],'actual face raw depth binding')
        for name in ('RECOVERED_FRONT_FACE_PROTOCOL.json','FRONT_TRIANGLE_DEPTH_PROTOCOL.json'):
            p=self.load(NEW+'/'+name)
            for input_name,item in p['inputs'].items():
                if input_name=='scene.usd.npz':require(item==protocol['inputs']['scene.usd.npz'],'actual face external mesh metadata')
                else:self.bind(item,(CODE if input_name.endswith('.py') else NEW)+'/'+input_name)
        vertices={}
        for pid,face in table.items():
            require(isinstance(pid,int) and 0<=pid<mesh['triangles'],'actual face primitive bounds')
            ids=face['original_vertex_ids'];coords=np.asarray(face['normalized_vertices'],float)
            require(len(ids)==len(set(ids))==3 and coords.shape==(3,3) and np.isfinite(coords).all(), 'actual face coordinate/vertex schema')
            for ident,point in zip(ids,face['normalized_vertices']):
                require(isinstance(ident,int) and 0<=ident<mesh['vertices'],'actual face vertex bounds')
                require(ident not in vertices or vertices[ident]==point,'shared actual vertex coordinate identity')
                vertices[ident]=point
        reconstruction=self.load(NEW+'/RECOVERED_FRONT_FACE_RECONSTRUCTION_ROWS.jsonl.gz',True)
        depths=self.load(NEW+'/FRONT_TRIANGLE_DEPTH_ROWS.jsonl.gz',True)
        raw=self.load(NEW+'/RECOVERED_DEPTH_ROWS.jsonl.gz',True)
        reconstructed={(f['index'],q['query']):(f,q) for f in reconstruction for q in f['queries']}
        depth_rows={(f['index'],q['query']):(f,q) for f in depths for q in f['queries']}
        require(len(reconstructed)==sum(len(f['queries']) for f in reconstruction)==13754
                and len(depth_rows)==sum(len(f['queries']) for f in depths)==13754
                and set(reconstructed)==set(depth_rows)==self.changed,'actual face all finite recovered query join')
        residuals=[];margins=[];camera_differences=[];counts=Counter();partitions=Counter()
        for frame in raw:
            R,t=np.asarray(frame['R']),np.asarray(frame['t'])
            for q in frame['queries']:
                key=(frame['index'],q['query'])
                if key not in self.changed:continue
                rf,rq=reconstructed[key];df,dq=depth_rows[key]
                require(all(f[k]==frame[k] for f in (rf,df) for k in ('id','index','partition')),'actual face/depth frame identity')
                pid=q['primitive_id'];require(rq['primitive_id']==dq['primitive_id']==pid and pid in table,'actual face primitive join')
                triangle=(np.asarray(table[pid]['normalized_vertices'])*np.asarray(frame['dimensions'])).astype(np.float32).astype(float)
                u,v=q['primitive_uv'];point=(1-u-v)*triangle[0]+u*triangle[1]+v*triangle[2]
                require(u>=0 and v>=0 and u+v<=1 and rq['closed_triangle_barycentric'],'actual closed-triangle barycentric surface')
                same(rq['barycentric_point_world'],point,'actual triangle barycentric point',atol=1e-12)
                same(dq['actual_triangle_barycentric_point_world'],point,'actual triangle depth surface point',atol=1e-12)
                same(rq['retained_ray_point_world'],q['actual_hit_point_world'],'actual face retained ray-point join',atol=1e-12)
                residual=float(np.linalg.norm(point-np.asarray(q['actual_hit_point_world'])))
                close(rq['world_point_residual_m'],residual,'actual triangle/ray residual',atol=1e-12);residuals.append(residual)
                require(rq['proved_front_surface_NONE']==q['proved_front_surface_NONE'],'actual face old front decision join')
                camera_Z=float((R@point+t)[2]);delta=camera_Z-q['source_camera_Z'];margin=-delta-frame['depth_tolerance_m']
                difference=camera_Z-q['recomputed_hit_source_camera_Z'];qualifies=math.isfinite(margin) and margin>0
                close(dq['actual_triangle_camera_Z'],camera_Z,'actual triangle camera_Z',atol=1e-12)
                close(dq['actual_triangle_minus_source_camera_Z_m'],delta,'actual triangle depth delta',atol=1e-12)
                close(dq['original_depth_tolerance_m'],frame['depth_tolerance_m'],'actual triangle unchanged tolerance')
                close(dq['strict_front_depth_margin_m'],margin,'actual triangle strict front margin',atol=1e-12)
                close(dq['triangle_camera_Z_minus_stored_ray_camera_Z_m'],difference,'actual triangle vs ray camera_Z',atol=1e-12)
                require(dq['qualifies_under_original_depth_tolerance']==qualifies,'actual triangle unchanged front-depth qualification')
                counts['qualifying' if qualifies else 'ambiguous']+=1
                counts[frame['partition']+('_qualifying' if qualifies else '_ambiguous')]+=1
                partitions[frame['partition']]+=1;margins.append(margin);camera_differences.append(difference)
        require(len(residuals)==faces['finite_queries']==result['finite_queries']==depth_result['finite_front_queries']==13754,
                'actual face complete finite witness population')
        require(dict(partitions)==result['partition_query_counts'],'actual face partition counts')
        for key,value in [('mean',math.fsum(residuals)/len(residuals)),('median',percentile(residuals,.5)),
                          ('p90',percentile(residuals,.9)),('maximum',max(residuals))]:
            close(result['world_point_residual_m'][key],value,'actual face residual '+key,atol=1e-12)
        require(dict(counts)==depth_result['counts'] and depth_result['all_qualify'] and not depth_result['ambiguous_rows'],
                'actual triangle raw qualification counts')
        close(depth_result['minimum_strict_front_margin_m'],min(margins),'actual triangle minimum front margin',atol=1e-12)
        for key,value in [('mean',math.fsum(margins)/len(margins)),('median',percentile(margins,.5)),('p90',percentile(margins,.9))]:
            close(depth_result['margin_m'][key],value,'actual triangle margin '+key,atol=1e-12)
        for key,value in [('max_absolute',max(abs(v) for v in camera_differences)),
                          ('median_absolute',percentile([abs(v) for v in camera_differences],.5)),
                          ('p90_absolute',percentile([abs(v) for v in camera_differences],.9))]:
            close(depth_result['triangle_camera_Z_minus_stored_ray_camera_Z_m'][key],value,'actual triangle/ray camera discrepancy '+key,atol=1e-12)
        require(not depth_result['source_targets_mutated'],'actual face arithmetic did not relabel')
        for data in (faces,result,depth_result):
            for k,v in data.items():
                if k.startswith('new_'):require(v==0,'actual face arithmetic unexpected '+k)
        return dict(actual_faces=5131,finite_queries=13754,closed_triangle_surface_points_reconstructed=True,
                    all_actual_triangle_camera_depths_front_at_original_tolerance=True,
                    minimum_front_margin_m=min(margins),maximum_ray_to_triangle_point_residual_m=max(residuals),
                    maximum_triangle_vs_ray_camera_depth_difference_m=max(abs(v) for v in camera_differences),
                    point_residual_is_not_zero=True,no_relabel_or_new_tolerance=True,first_surface_order_still_engine_provenance=True)

    def target_transitions(self):
        old=self.arrays(GATE+'/PREPARED_TARGETS.npz');new=self.arrays(NEW+'/READY_PREPARED_TARGETS.npz')
        original=self.load(GATE+'/FULL_SOURCE_TARGET_ROWS.jsonl.gz',True)
        ready=self.load(NEW+'/READY_SOURCE_TARGET_ROWS.jsonl.gz',True)
        result=self.load(NEW+'/DEPTH_RECOVERY_VALIDATION.json')
        require(len(original)==len(ready)==1024,'ready1024 rows')
        changed=np.zeros((1024,84),bool)
        for i,j in self.changed:changed[i,j]=True
        for key in old:
            require(old[key].dtype==new[key].dtype and old[key].shape==new[key].shape,'array dtype/shape preservation '+key)
        for key in ('lo','hi','weight','valid'):
            require(new[key].shape==(1024,84),'ready target shape')
            require(new[key][~changed].tobytes()==old[key][~changed].tobytes(),'non-recovered target bytes '+key)
            require(new[key][896:].tobytes()==old[key][896:].tobytes(),'source-test target bytes '+key)
            positive=old['valid']&(old['lo']<65)
            require(new[key][positive].tobytes()==old[key][positive].tobytes(),'protected POSITIVE target bytes '+key)
        for key in ('source_index','partitions'):
            require(new[key].tobytes()==old[key].tobytes(),'family/split target bytes '+key)
        require(np.array_equal(new['source_index'],np.arange(1024)),'ready frozen family index')
        require(np.array_equal(new['partitions'],np.asarray([r['partition'] for r in ready])),'ready frozen split index')
        for key in ('lo','hi'):require((new[key][changed]==65).all(),'recovered NONE array '+key)
        require((new['weight'][changed]==0).all() and new['valid'][changed].all() and not old['valid'][changed].any(),
                'recovered IGNORE to valid NONE array transition')
        counts={s:Counter() for s in ('train','calibration','source_test')};row_changes=0
        for index,(before,after) in enumerate(zip(original,ready)):
            require(before['index']==after['index']==index and {k:v for k,v in before.items() if k!='queries'}==
                    {k:v for k,v in after.items() if k!='queries'},'ready frame metadata unchanged')
            require(len(before['queries'])==len(after['queries'])==84,'ready query count')
            for i,(a,b) in enumerate(zip(before['queries'],after['queries'])):
                require(a['query']==b['query']==i and b['depth_recovery_changed_proposal']==bool(changed[index,i]),
                        'ready changed flag/witness join')
                stripped={k:v for k,v in b.items() if k!='depth_recovery_changed_proposal'}
                if changed[index,i]:
                    require(index<896 and a['original_target']=='NONE' and a['proposed_target']=='IGNORE','only original train/cal NONE may change')
                    expected=dict(a,proposed_target='NONE',proposal_label_changed=False,lo=65,hi=65,weight=0.,
                                  reason='Finite actual first surface precedes intended physical source point beyond unchanged original depth tolerance; new fixed recovery witness.',
                                  depth_recovery_binding=dict(frame_index=index,query=i))
                    require(stripped==expected,'ready recovered query exact mutation contract');row_changes+=1
                else:require(stripped==a,'ready non-recovered query record unchanged')
                state=b['proposed_target'];counts[after['partition']][state]+=1
                valid=bool(new['valid'][index,i]);lo=int(new['lo'][index,i]);hi=int(new['hi'][index,i]);weight=float(new['weight'][index,i])
                require(state in ('POSITIVE','NONE','IGNORE') and valid==(state!='IGNORE'),'ready raw target/valid join')
                if state=='POSITIVE':
                    require(lo==b['lo'] and hi==b['hi'] and 0<=lo<=64 and hi==min(64,lo+1)
                            and weight==float(np.float32(b['weight'])) and 0<=weight<=1,'ready positive bins/weights')
                else:require(lo==hi==65 and weight==0,'ready NONE/IGNORE canonical array')
                require(not b['source_cache_mutated'] and not b['approved_for_training'],'ready target proposal authorization scope')
        require(row_changes==13754 and {p:dict(c) for p,c in counts.items()}==result['target_counts'],'ready raw labels/counts')
        require(result['source_supervision_ready']==all(counts[s]['POSITIVE']>0 and counts[s]['NONE']>0 for s in counts),
                'mixed-loss source readiness gate')
        require(not result['source_cache_mutated'] and result['source_test_arrays_unchanged'] and all(result['array_invariants'].values()),
                'recorded source array invariants')
        return dict(families=1024,queries=86016,changed_IGNORE_to_NONE=13754,
                    target_counts={p:dict(c) for p,c in counts.items()},all_POSITIVE_nonrecovered_source_test_bytes_unchanged=True,
                    family_split_order_and_dtypes_unchanged=True,source_mixed_loss_ready=True,
                    training_authorization_or_real_pose_success_not_established=True)

    def fixed_retraining_plan(self):
        plan=self.load(NEW+'/RETRAINING_PROTOCOL.json');checks=self.load(NEW+'/ZERO_UPDATE_CPU_CHECKS.json')
        old=self.load(OLD+'/LEARNING_PROTOCOL.json');old_batch=self.load(OLD+'/REVIEW_LOSS_MASK_CHECK.json')
        require(plan['schema']=='fixed_source_supervision_repair_retraining_protocol_v1'
                and plan['status']=='PREPARED_AWAITING_ADDITIONAL_UPDATE_AUTHORIZATION','prepared fixed replay status')
        require(plan['original_formal_updates_completed']==9000 and plan['original_remaining_formal_updates']==0
                and plan['requested_additional_formal_updates']==9000 and plan['executed_additional_formal_updates']==0,
                'original consumed/additional unexecuted budget')
        for key in ('arms','model','parameters','seed','batch','optimizer','initial_state_sha256','batch_order_sha256'):
            require(plan[key]==old[key],'unchanged original replay configuration '+key)
        require(plan['updates_per_arm']==old['updates']==3000 and plan['parameters']==5890
                and plan['batch']==16 and plan['formal_RGB_exposures']==9000*16 and plan['throwaway_updates']==0,
                'fixed replay quantity')
        require(plan['feature_cache_reused_readonly'] and plan['fresh_feature_detector_calls']==0
                and plan['fresh_original_RGB']==0 and plan['readiness_does_not_establish_real_success'], 'fixed replay scope')
        self.bind(checks['protocol'])
        require(checks['complete'] and checks['passed'] and checks['fixed_inputs']==plan['fixed_inputs']
                and checks['after_input_hashes_equal'],'zero-update recorded completion/input identity')
        metadata=self.load(GATE+'/FULL_SOURCE_PREPARATION_PROTOCOL.json')['inputs']
        for key,item in plan['fixed_inputs'].items():
            if item['origin']=='public_repository':self.bind(item)
            elif key in ('features','order','cache_manifest'):
                name={'features':'features.npy','order':'order.npy','cache_manifest':'CACHE_MANIFEST.json'}[key]
                require(item['sha256']==metadata[name]['sha256'] and item['bytes']==metadata[name]['bytes'],
                        'unchanged private metadata binding '+key)
            elif key in ('repaired_targets','source_ready_validation'):
                name={'repaired_targets':'READY_PREPARED_TARGETS.npz','source_ready_validation':'DEPTH_RECOVERY_VALIDATION.json'}[key]
                self.bind(item,NEW+'/'+name)
            else:require(False,'unmapped replay input binding '+key)
        require(plan['batch_order_file_sha256']==metadata['order.npy']['sha256'],'replay original order file')
        ids=plan['first_batch_source_indices']
        require(ids==checks['first_batch_indices']==old_batch['family_indices'] and len(ids)==16
                and all(isinstance(i,int) and 0<=i<768 for i in ids),'same original frozen first train batch')
        arrays=self.arrays(NEW+'/READY_PREPARED_TARGETS.npz');rows=self.load(NEW+'/READY_SOURCE_TARGET_ROWS.jsonl.gz',True)
        require(checks['first_batch_ids']==[rows[i]['id'] for i in ids],'CPU first batch identity join')
        for key in ('lo','hi','weight','valid'):
            require(checks['first_batch_targets'][key]==arrays[key][ids].tolist(),'CPU first-batch target exact join '+key)
        calculated={}
        for partition,n in [('train',768),('calibration',128),('source_test',128)]:
            mask=arrays['partitions']==partition;valid=arrays['valid'][mask];lo=arrays['lo'][mask]
            calculated[partition]=dict(images=n,POSITIVE=int((valid&(lo<65)).sum()),NONE=int((valid&(lo==65)).sum()),IGNORE=int((~valid).sum()))
        require(plan['target_counts']==checks['target_counts']==calculated,'CPU/replay full target-count join')
        valid=arrays['valid'][ids];lo=arrays['lo'][ids]
        batch=dict(images=16,queries=1344,POSITIVE=int((valid&(lo<65)).sum()),NONE=int((valid&(lo==65)).sum()),IGNORE=int((~valid).sum()))
        require(checks['first_batch_target_counts']==batch,'CPU first batch target counts')
        close(checks['expected_uniform_logit_loss'],math.log(66),'analytic uniform soft-CE loss',atol=1e-12)
        close(checks['uniform_logit_loss'],math.log(66),'recorded float32 uniform soft-CE loss',atol=1e-6)
        heads=checks['head_checks'];require([r['arm'] for r in heads]==plan['arms'],'CPU all three fixed arms')
        for head in heads:
            require(head['parameters']==5890 and head['initial_state_sha256']==head['after_backward_state_sha256']==old['initial_state_sha256'],
                    'CPU recorded init/model unchanged by backward')
            require(math.isfinite(head['loss']) and head['ignored_logit_gradient_max']==0
                    and head['valid_query_logit_gradient_L1_min']>0 and 0<=head['analytic_CE_derivative_max_error']<1e-7,
                    'CPU recorded derivative/IGNORE fields')
            for label in ('POSITIVE','NONE'):
                scope=head['scoped_gradient'][label]
                require(scope['queries']==batch[label] and math.isfinite(scope['loss'])
                        and math.isfinite(scope['first_layer_gradient_norm']) and scope['first_layer_gradient_norm']>0,
                        'CPU recorded valid target gradient '+label)
            require(head['scoped_gradient']['ALL_IGNORED']=={'queries':0,'loss':0.,'first_layer_gradient_norm':0.},
                    'CPU recorded all-IGNORE zero gradient')
            require(all(v is not None and math.isfinite(v) and v>=0 for v in head['parameter_gradient_norms'].values()),
                    'CPU recorded finite parameter gradients')
            if head['arm']=='GEOMETRY_ONLY':require(head['image_channel_gradient_norm']==0,'CPU geometry-only image mask')
            if head['arm']=='IMAGE_NO_ROLE':require(head['role_channel_gradient_norm']==0,'CPU no-role mask')
            if head['arm']=='IMAGE_ROLE':require(head['image_channel_gradient_norm']>0 and head['role_channel_gradient_norm']>0,'CPU image/role gradients')
        execution=checks['execution']
        require(execution['device']=='cpu' and execution['head_forward_calls']==3 and execution['head_image_exposures']==48
                and execution['loss_only_uniform_evaluations']==1 and execution['autograd_grad_calls']==10
                and execution['backward_calls']==3,'recorded CPU execution quantities')
        for key in ('optimizer_instances','optimizer_steps','training_updates','model_weight_files_written','detector_forwards',
                    'rays','PnP','raw_RGB_generated','source_feature_files_changed'):
            require(execution[key]==0,'recorded zero-update execution '+key)
        return dict(requested_additional_updates=9000,executed_additional_updates=0,original_remaining_updates=0,
                    original_model_init_order_optimizer_metadata_unchanged=True,first_batch_targets_joined=1344,
                    first_batch_target_counts=batch,uniform_logit_CE_independently_calculated=True,
                    recorded_CPU_heads=3,recorded_CPU_backwards=3,recorded_optimizer_steps=0,
                    stored_gradient_fields_checked_not_reexecuted=True,private_features_or_model_logits_not_read=True,
                    training_authorization_still_required=True,real_pose_success_not_established=True)

    def manifest(self):
        path=self.path(NEW+'/REVIEW_MANIFEST.json')
        if not path.exists():
            require(not self.require_manifest,'required REVIEW_MANIFEST.json is absent')
            return dict(present=False,required=False)
        manifest=self.load(NEW+'/REVIEW_MANIFEST.json');items=manifest['files']
        require(len(items)==len({i['path'] for i in items}),'duplicate manifest file')
        for item in items:
            require(Path(item['path']).name not in ('REVIEW_CHECKS.json','REVIEW_MANIFEST.json'),'manifest hash cycle')
            self.bind(item)
        return dict(present=True,files=len(items),required=self.require_manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3],help='public checkout root')
    parser.add_argument('--output',type=Path,help='new JSON receipt; only own default receipt may be regenerated')
    parser.add_argument('--require-manifest',action='store_true')
    args=parser.parse_args();root=args.root.resolve();own=root/NEW/'REVIEW_CHECKS.json'
    if own.resolve()!=own:parser.error('The own receipt or its ancestors must not be symlinks')
    output=(args.output or own).resolve()
    if output.suffix!='.json' or (output.exists() and output!=own):
        parser.error('Refusing to overwrite source, input, manifest or existing research artifact')
    if output.is_relative_to(root) and output!=own:
        parser.error('Inside this checkout only this phase REVIEW_CHECKS.json can be written')
    review=Review(root,args.require_manifest)
    for name,function in [('protected_prior_publication',review.prior),
                          ('fixed_plan_and_recorded_input_bindings',review.bindings_and_plan),
                          ('front_surface_witness_arithmetic',review.depth_witnesses),
                          ('actual_face_surface_and_front_depth_arithmetic',review.actual_face_arithmetic),
                          ('full_target_transitions_and_mixed_loss_readiness',review.target_transitions),
                          ('fixed_retraining_plan_and_zero_update_receipt',review.fixed_retraining_plan),
                          ('publication_manifest',review.manifest)]:review.group(name,function)
    result=dict(schema='public_source_depth_recovery_review_v1',passed=all(g['passed'] for g in review.groups),
                checks=review.groups,limits=LIMITS,public_files_read=sorted(review.files_read),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions=dict(python=sys.version.split()[0],numpy=np.__version__),
                execution=dict(private_truth_reads=0,private_mesh_reads=0,model_forwards=0,training_updates=0,
                               loss_backwards=0,pose_fits=0,PnP_calls=0,rays=0))
    output.parent.mkdir(parents=True,exist_ok=True);pending=output.with_name(output.name+'.pending')
    if pending.exists():parser.error('Refusing to overwrite an existing pending receipt')
    with pending.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    pending.replace(output)
    print(json.dumps(dict(passed=result['passed'],groups=len(review.groups),receipt=str(output))))
    for group in review.groups:
        if not group['passed']:print(group['name']+': '+group['error'],file=sys.stderr)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
