"""Fixed96 geometry/query/GT jobs only. No image/renderer/model/fit imports."""
from pathlib import Path
from collections import Counter
import copy,hashlib,itertools,json,time
import numpy as np
from conservative_ray_scene import ConservativeRayScene,load_teacher
HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
PROTOCOL=HERE/'GEOMETRY_PROTOCOL_BEFORE_FULL96_CALCULATION.json'
FACES=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int64)
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v)for v in x]
    return x
def write(p,x):
    with Path(p).open('x')as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,x):
    with Path(p).open('x')as f:
        for row in x:f.write(json.dumps(clean(row),separators=(',',':'),allow_nan=False)+'\n')
def project(x,camera):
    c=np.asarray(x)@np.asarray(camera['R']).T+np.asarray(camera['t']);return(c@np.asarray(camera['K']).T)[...,:2]/c[...,2:3],c[...,2]
def affine(bounds,index):
    bounds=np.array(bounds);centre=bounds.mean(0);scale=np.ones(3)
    if index==1:scale[2]=(bounds[1,0]-bounds[0,0])/(bounds[1,2]-bounds[0,2])
    M=np.eye(4);M[:3,:3]=np.diag(scale);M[3,:3]=centre*(1-scale);return M
def transform(x,M):return np.asarray(x)@M[:3,:3]+M[3,:3]
def transformed_cues(cues,M):
    result=copy.deepcopy(cues)
    for cue in result:
        g=cue['geometry']
        for k in['endpoint_world','face_patch_quads_world','face_probe_points','edge_probe_points']:g[k]=transform(g[k],M)
        g['edge_vectors']=np.asarray(g['edge_vectors'])@M[:3,:3];g['edge_lengths']=np.linalg.norm(g['edge_vectors'],axis=1)
        normal=np.asarray(g['face_normals'])@np.linalg.inv(M[:3,:3]).T;g['face_normals']=normal/np.linalg.norm(normal,axis=1)[:,None]
    return result
def camera_for(config,reference):
    direction=np.array(reference['direction']);yaw=np.arctan2(direction[0],direction[2])+np.deg2rad(config['azimuth_offset_deg']);elev=np.arcsin(direction[1])+np.deg2rad(config['elevation_offset_deg'])
    d=np.array([np.cos(elev)*np.sin(yaw),np.sin(elev),np.cos(elev)*np.cos(yaw)]);centre=np.array(reference['target_world']);eye=centre+d*reference['distance'];z=-d
    x=np.cross(z,[0,1,0]);x/=np.linalg.norm(x);y=np.cross(z,x);R=np.stack([x,y,z])
    c=copy.deepcopy(reference);c.update(R=R,t=-R@eye,eye_world=eye,direction=d,azimuth_absolute_deg=float(np.rad2deg(yaw)),elevation_absolute_deg=float(np.rad2deg(elev)))
    return c
def foreign_for(camera,selected,direction):
    centre=np.array(selected['projected_support_disk']['center']);radius=selected['projected_support_disk']['radius'];half=radius+3.;front=.6*selected['all_patch_camera_z_min'];thick=.05*front
    K=np.array(camera['K']);R=np.array(camera['R']);t=np.array(camera['t']);fc=K.diagonal()[:2];c=np.r_[(centre-K[:2,2])/fc*front,front+thick/2];size=np.r_[np.full(2,2*half)/fc*front,thick]
    signs=np.array(list(itertools.product([-1.,1.],repeat=3)));cover_cam=c+signs*size/2;cover=(cover_cam-t)@R
    uv,z=project(cover,camera);d=np.array(direction,np.float64);norm2=float(d@d);assert abs(norm2-1)<1e-12
    s=(uv-centre)@d
    # One analytic maximum, not a placement search. Every projected vertex
    # moves by D*front/Z along d; entire convex projection clears disk by2px.
    displacement=float(np.max((radius+2.-s)*z/(front*norm2))+1e-9)
    shift_cam=np.r_[d*displacement*front/fc,0.];world_shift=shift_cam@R;sham=cover+world_shift
    suv,sz=project(sham,camera);clearance=float(np.min((suv-centre)@d)-radius)
    patch=np.array(selected['all_patch_projected_vertices']);rect=np.r_[centre-half,centre+half];margin=float(np.c_[patch-rect[:2],rect[2:]-patch].min())
    valid=bool(np.isfinite(cover).all()and np.isfinite(sham).all()and z.min()>0 and sz.min()>0 and margin>0 and clearance>=2-1e-9)
    return dict(cover_vertices_world=cover,sham_vertices_world=sham,triangles=FACES,front_camera_z=front,depth_thickness=thick,dimensions_camera=size,
        sham_world_translation=world_shift,sham_displacement_px=displacement,sham_direction_unit_xy=d,cover_front_bbox_xyxy=rect,
        cover_support_disk=dict(center=centre,radius=radius),full_projected_sham_clearance_px=clearance,cover_patch_margin_px=margin,
        placement_valid=valid,material_shared=True,shape_shared=True)
def main():
    start=time.monotonic();protocol=read(PROTOCOL)
    for b in protocol['bindings'].values():assert sha(b['path'])==b['sha256'],b['path']
    proof=read(protocol['bindings']['equivalence']['path']);assert proof['PASS']and proof['V97_baseline_rays']==296
    assert protocol['bindings']['builder']==bind(__file__)and not(HERE/'GEOMETRY96_SEAL.json').exists()
    config=lines(protocol['bindings']['root_config']['path']);assert len(config)==96 and [c['base_index']for c in config]==list(range(96))
    source=read(protocol['bindings']['source_cue_geometry']['path']);contract=read(protocol['bindings']['source_cue_contract']['path']);inventory=read(protocol['bindings']['inventory']['path']);oldrefs=read(contract['frozen_source_associations']['path'])['references']
    with np.load(protocol['bindings']['geometry']['path'],allow_pickle=False)as z:vertices=z['vertices_world_m'];tri=z['triangles_global'];part=z['triangle_asset_local_part_id']
    orientation=np.array([1 if m['USD_orientation']=='rightHanded'else-1 for m in inventory['per_mesh']])[part];two=np.array([m['double_sided']for m in inventory['per_mesh']])[part]
    refworld=np.array([r['virtual_reference_world']for r in oldrefs]);T=load_teacher();bounds=np.array(inventory['stage']['bounds_world_m']);groups={};worlds={};scenes={}
    for aspect in range(2):
        M=affine(bounds,aspect);world=transform(vertices,M);cues=transformed_cues(source['references'],M);groups[f'ASPECT_{aspect}']=dict(target_affine_world_row=M,references=cues)
        worlds[aspect]=world;scenes[aspect]=ConservativeRayScene(world[tri],part,orientation,two)
    groupfile=HERE/'TRANSFORMED_CUE_GEOMETRIES2_GT_ONLY.json';write(groupfile,dict(completed=True,contract=protocol['bindings']['source_cue_contract'],groups=groups))
    jobs=[];GT=[];inputs=[];counts=Counter();picked=[]
    write(HERE/'METADATA_RUN_STARTED.json',dict(completed=False,protocol=bind(PROTOCOL),builder=bind(__file__),RGB_model_fit_private_current_PnP=0))
    for cfg in config:
        i=cfg['base_index'];assert i==(((cfg['azimuth_index']*2+cfg['elevation_index'])*2+cfg['aspect_index'])*2+cfg['appearance_index'])
        aspect=cfg['aspect_index'];M=groups[f'ASPECT_{aspect}']['target_affine_world_row'];world=worlds[aspect];camera=camera_for(cfg,source['camera']);ref=transform(refworld,M);projection,z=project(ref,camera)
        L=float(np.clip(.25*np.ptp(projection,axis=0).max(),16,camera['size']));q=projection+np.array(cfg['query_jitter_xy']);cues=copy.deepcopy(groups[f'ASPECT_{aspect}']['references'])
        for k,cue in enumerate(cues):cue.update(original_q=q[k],L=L)
        before=[T.evaluate_endpoint(cue,camera,scenes[aspect])for cue in cues];direct=[k for k,r in enumerate(before)if r['state']=='D']
        keys={k:hashlib.sha256(f"V99_EXPECTED_CUE_TARGET_V1\n20261023\n{cfg['base_id']}\n{k}".encode()).hexdigest()for k in direct}
        selected=min(direct,key=lambda k:keys[k])if direct else None;foreign=None if selected is None else foreign_for(camera,before[selected],cfg['sham_direction_unit_xy'])
        depths=(world@np.array(camera['R']).T+np.array(camera['t']))[:,2]
        if foreign is not None:
            _,cz=project(foreign['cover_vertices_world'],camera);_,sz=project(foreign['sham_vertices_world'],camera);depths=np.r_[depths,cz,sz];foreign['sham_direction_index']=cfg['sham_direction_index']
        depthmin=float(depths.min());depthmax=float(depths.max());positive=bool(np.isfinite(depths).all()and depthmin>0)
        camera['near']=.5*depthmin;camera['far']=1.1*depthmax
        assert positive and 0<camera['near']<depthmin<=depthmax<camera['far']
        before=[T.evaluate_endpoint(cue,camera,scenes[aspect])for cue in cues]
        force=selected is None or not foreign['placement_valid'];reason='NO_DIRECT_CUE_AFTER_U'if selected is None else('FIXED_PLACEMENT_INFEASIBLE_AFTER_U'if force else'FIXED_TARGET_AND_FOREIGN_PLACEMENT_READY')
        stages={'BEFORE':before};combined_stats=[]
        for stage in['COVER','SHAM']:
            if force:after=[dict(r,state='U',reason=reason,underlying_target_only_state=r['state'])for r in before]
            else:
                fv=np.array(foreign['cover_vertices_world']if stage=='COVER'else foreign['sham_vertices_world']);ft=np.array(foreign['triangles'])
                combined=ConservativeRayScene(np.r_[world[tri],fv[ft]],np.r_[part,np.full(12,308)],np.r_[orientation,np.ones(12)],np.r_[two,np.zeros(12,bool)])
                after=[T.evaluate_endpoint(cue,camera,scenes[aspect],combined)for cue in cues];combined_stats.append(dict(stage=stage,rays=combined.ray_count,triangle_tests=combined.triangle_tests,unfiltered_triangle_tests=combined.full_triangle_tests))
            stages[stage]=after
        uvall,_=project(world,camera);camera_geometry_valid=bool(((uvall>=0)&(uvall<camera['size'])).all())
        job=dict(cfg,fold='TRAIN'if cfg['split']=='TRAIN'else'VAL',foreign_seed=cfg['foreign_material_seed'],camera=camera,
            target_affine_world_row=M,query8_xy=q,L=np.full(8,L),projected_virtual_reference8=projection,query_jitter_source='Exact root config; no reselection',
            cue_geometry=bind(groupfile),cue_geometry_key=f'ASPECT_{aspect}',selected_aux_index=selected,foreign_geometry=foreign,after_force_U=force,attempt_reason=reason,
            variants=['BEFORE','COVER','SHAM'],clip_bounds_camera_z=[depthmin,depthmax],target_all_vertices_in_frame=camera_geometry_valid,native8_mapping='UNKNOWN',human_gold=False)
        jobs.append(job);GT.append(dict(base_index=i,base_id=cfg['base_id'],split=cfg['split'],stages=stages,selected_aux_index=selected,direct_candidates=direct,selection_SHA_by_aux=keys,after_force_U=force,attempt_reason=reason,combined_broadphase=combined_stats))
        inputs.append(dict(base_index=i,base_id=cfg['base_id'],split=cfg['split'],raw_hw=[640,640],query8_xy=q,L=np.full(8,L),source_emulated_query_not_initializer_output=True,native8_mapping='UNKNOWN'))
        for stage,rows in stages.items():counts.update((cfg['split'],stage,r['state'])for r in rows)
        picked.append(None if selected is None else dict(base_index=i,aux=selected,states=[stages[s][selected]['state']for s in['BEFORE','COVER','SHAM']]))
        if(i+1)%8==0:print(json.dumps(dict(stage='GEOMETRY_METADATA',bases=i+1,total=96,seconds=time.monotonic()-start)),flush=True)
    gtfile=HERE/'EXPECTED_CUE_TARGETS96_GT_ONLY.jsonl';jsonl(gtfile,GT)
    gt_binding=bind(gtfile)
    for j in jobs:j.update(GT_only_targets=gt_binding,GT_row_index=j['base_index'])
    jsonl(HERE/'RENDER_JOBS96_GT_GUIDED.jsonl',jobs);jsonl(HERE/'QUERY_INPUTS96_INPUT_ONLY.jsonl',inputs)
    summary=dict(completed=True,base_scenes=96,RGB_expected=288,logical_query_occurrences=2304,TRAIN=64,VALIDATION=32,counts={'|'.join(k):v for k,v in counts.items()},selected_target_triplets=picked,
        no_direct_cases=sum(j['selected_aux_index']is None for j in jobs),infeasible_placements=sum(j['after_force_U']for j in jobs if j['selected_aux_index']is not None),camera_target_out_of_frame_cases=sum(not j['target_all_vertices_in_frame']for j in jobs),
        before_to_after_q_L_exact=True,all_original_transformed_associations_preserved=True,source_rays_target=sum(s.ray_count for s in scenes.values()),source_triangle_tests_target=sum(s.triangle_tests for s in scenes.values()),source_full_triangle_tests_target=sum(s.full_triangle_tests for s in scenes.values()),
        seconds=time.monotonic()-start,new_RGB_renderer_model_features_fit_private_current_PnP=0,training_admission=False)
    write(HERE/'GEOMETRY96_SUMMARY.json',summary)
    files=[bind(HERE/n)for n in['TRANSFORMED_CUE_GEOMETRIES2_GT_ONLY.json','EXPECTED_CUE_TARGETS96_GT_ONLY.jsonl','RENDER_JOBS96_GT_GUIDED.jsonl','QUERY_INPUTS96_INPUT_ONLY.jsonl','GEOMETRY96_SUMMARY.json']]
    write(HERE/'GEOMETRY96_SEAL.json',dict(completed=True,PASS=True,protocol=bind(PROTOCOL),root_config=protocol['bindings']['root_config'],renderer_jobs=bind(HERE/'RENDER_JOBS96_GT_GUIDED.jsonl'),GT_only_targets=bind(gtfile),shared_cue_geometry=bind(groupfile),query_inputs=bind(HERE/'QUERY_INPUTS96_INPUT_ONLY.jsonl'),files=files,new_RGB_model_fit_private_current_PnP=0))
    print(json.dumps(clean(dict(completed=True,summary=summary,seal=bind(HERE/'GEOMETRY96_SEAL.json')))),flush=True)
if __name__=='__main__':main()
