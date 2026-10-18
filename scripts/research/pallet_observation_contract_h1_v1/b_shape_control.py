"""Four newly constructed procedural shape/occluder diagnostic renders.

This is not the original dataset generator/scene replay. Primitive boxes only;
no external mesh, renderer, RGB, targets, trained model, or pose solver.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import resource,time
import numpy as np
from PIL import Image
from scripts.research.pallet_risk_information_v1.common import read,write,sha
from .b_mask_diagnostic import transition_summary,line_labels

CONFIG=dict(schema='PROCEDURAL_SHAPE_CONTROL_CONFIG_V1',image_hw=[480,640],supersampling=2,
    camera_eye_world_m=[.2,-3.,.42],camera_target_world_m=[0.,0.,.075],world_up=[0.,0.,1.],
    focal_px=520.,principal_xy_px=[319.5,239.5],target_WDH_m=[1.3,1.1,.15],
    deck_planks=7,deck_plank_width_m=.14,deck_z_m=[.125,.15],
    support_centers_x_m=[-.55,0.,.55],support_centers_y_m=[-.45,0.,.45],
    support_size_m=[.2,.2,.1],support_z_m=[.025,.125],bottom_z_m=[0.,.025],
    occluder_bounds_world_m=[[-.12,-.78,0.],[.12,-.64,.32]],
    light_direction=[1.,-1.,2.],background_RGB=[216,224,232],target_RGB=[90,145,115],occluder_RGB=[205,85,65],
    query_offsets_raw_px=list(range(-128,65)),
    query_definition='normal to projected front-left vertical bounding edge, centered at its projected midpoint; positive points outward',
    cases=[['PROCEDURAL_PALLET_WITH_OPENINGS',False],['PROCEDURAL_PALLET_WITH_OPENINGS',True],['SOLID_REGISTERED_CUBOID',False],['SOLID_REGISTERED_CUBOID',True]],
    rights='New code-generated primitive boxes/colors only, no third-party assets',
    source_scene_status='PROCEDURAL_COUNTEREXAMPLE_NOT_ORIGINAL_SCENE',original_scene_replay=0,new_D1_or_training_samples=0)


def camera(config):
    eye=np.asarray(config['camera_eye_world_m']);f=np.asarray(config['camera_target_world_m'])-eye;f=f/np.linalg.norm(f)
    right=np.cross(f,config['world_up']);right/=np.linalg.norm(right);up=np.cross(right,f)
    return np.stack((right,-up,f)),eye


def project(points,config):
    R,eye=camera(config);cam=(np.asarray(points)-eye)@R.T
    return cam[:,:2]/cam[:,2:]*config['focal_px']+config['principal_xy_px'],cam[:,2]


def boxes(shape,config):
    W,D,H=config['target_WDH_m']
    if shape=='SOLID_REGISTERED_CUBOID':return [(np.array([-W/2,-D/2,0.]),np.array([W/2,D/2,H]))]
    result=[];width=config['deck_plank_width_m'];xs=np.linspace(-W/2+width/2,W/2-width/2,config['deck_planks'])
    for x in xs:result.append((np.array([x-width/2,-D/2,config['deck_z_m'][0]]),np.array([x+width/2,D/2,config['deck_z_m'][1]])))
    for x in config['support_centers_x_m']:
        for y in config['support_centers_y_m']:
            result.append((np.array([x-.1,y-.1,config['support_z_m'][0]]),np.array([x+.1,y+.1,config['support_z_m'][1]])))
        result.append((np.array([x-.1,-D/2,config['bottom_z_m'][0]]),np.array([x+.1,D/2,config['bottom_z_m'][1]])))
    return result


def mesh(bound):
    low,high=bound
    points=np.array([[x,y,z]for x in [low[0],high[0]]for y in [low[1],high[1]]for z in [low[2],high[2]]])
    faces=[]
    for axis in range(3):
        for value in [low[axis],high[axis]]:
            ids=np.flatnonzero(points[:,axis]==value);center=points[ids].mean(0);other=[i for i in range(3)if i!=axis]
            order=np.argsort(np.arctan2(points[ids,other[1]]-center[other[1]],points[ids,other[0]]-center[other[0]]));ids=ids[order]
            faces.extend([(int(ids[0]),int(ids[1]),int(ids[2])),(int(ids[0]),int(ids[2]),int(ids[3]))])
    return points,faces


def conservative_owner(high_ids,hw,ss):
    H,W=hw;block=high_ids.reshape(H,ss,W,ss).transpose(0,2,1,3).reshape(H,W,ss*ss)
    first=block[:,:,0];same=(block==first[:,:,None]).all(2)
    return np.where(same,first,-1).astype(np.int8)


def render(shape,occlusion,config):
    H,W=config['image_hw'];ss=config['supersampling'];depth=np.full((H*ss,W*ss),np.inf)
    ids=np.zeros((H*ss,W*ss),np.int8);rgb=np.empty((H*ss,W*ss,3));rgb[:]=config['background_RGB']
    actors=[(1,b,config['target_RGB'])for b in boxes(shape,config)]
    if occlusion:actors.append((2,tuple(np.asarray(x)for x in config['occluder_bounds_world_m']),config['occluder_RGB']))
    light=np.asarray(config['light_direction']);light=light/np.linalg.norm(light)
    for owner,bound,color in actors:
        xyz,faces=mesh(bound);uv,z=project(xyz,config);uv=(uv+.5)*ss-.5
        if np.min(z)<=0:raise ValueError('Fixed scene must be wholly in front of camera')
        for face in faces:
            q=uv[list(face)];zz=z[list(face)]
            lo=np.maximum(np.floor(q.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(q.max(0)).astype(int),[W*ss-1,H*ss-1])
            if np.any(lo>hi):continue
            x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
            denominator=(q[1,1]-q[2,1])*(q[0,0]-q[2,0])+(q[2,0]-q[1,0])*(q[0,1]-q[2,1])
            if abs(denominator)<=np.finfo(float).eps:continue
            a=((q[1,1]-q[2,1])*(x-q[2,0])+(q[2,0]-q[1,0])*(y-q[2,1]))/denominator
            b=((q[2,1]-q[0,1])*(x-q[2,0])+(q[0,0]-q[2,0])*(y-q[2,1]))/denominator;c=1-a-b
            inside=(a>=-32*np.finfo(float).eps)&(b>=-32*np.finfo(float).eps)&(c>=-32*np.finfo(float).eps)
            inverse=a/zz[0]+b/zz[1]+c/zz[2]
            candidate=np.divide(1.,inverse,out=np.full_like(inverse,np.inf),where=inverse>0)
            sl=np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1];take=inside&(candidate<depth[sl]);depth[sl][take]=candidate[take];ids[sl][take]=owner
            normal=np.cross(xyz[face[1]]-xyz[face[0]],xyz[face[2]]-xyz[face[0]]);normal/=np.linalg.norm(normal)
            rgb[sl][take]=np.asarray(color)*(.55+.45*abs(float(normal@light)))
    owner=conservative_owner(ids,[H,W],ss)
    image=np.rint(rgb.reshape(H,ss,W,ss,3).mean((1,3))).astype(np.uint8)
    return image,owner,depth


def query(config):
    W,D,H=config['target_WDH_m'];edge=np.array([[-W/2,-D/2,0.],[-W/2,-D/2,H]])
    p,z=project(edge,config);t=p[1]-p[0];t/=np.linalg.norm(t);n=np.array([-t[1],t[0]])
    center=p.mean(0);boxxy,_=project(mesh((np.array([-W/2,-D/2,0.]),np.array([W/2,D/2,H])))[0],config)
    if n@(center-boxxy.mean(0))<0:n=-n
    xy=center+np.asarray(config['query_offsets_raw_px'])[:,None]*n
    return dict(center_xy=center.tolist(),normal_xy=n.tolist(),edge_xy=p.tolist(),sample_xy=xy.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','lock','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--render-GO',action='store_true');a=p.parse_args()
    lock=read(a.lock);config=read(a.config)
    if not a.render_GO or lock['config_sha256']!=sha(a.config)or lock['source_sha256']!=sha(__file__):raise ValueError('Pre-render fixed source/config lock and root GO required')
    if config!=CONFIG:raise ValueError('Frozen primitive control config changed')
    out=a.output;out.mkdir(parents=True,exist_ok=False);started,cpu=time.monotonic(),time.process_time();records=[]
    q=query(config)
    for i,(shape,occlusion)in enumerate(config['cases']):
        start,pcpu=time.monotonic(),time.process_time();image,owner,depth=render(shape,occlusion,config)
        name=f'{i:02d}_{shape}_occ{int(occlusion)}';Image.fromarray(image).save(out/(name+'_RGB.png'))
        # Label0 background,1 target,2 occluder; -1 denotes mixed AA samples.
        labels=np.full(len(q['sample_xy']),-2,np.int8);xy=np.floor(np.asarray(q['sample_xy'])+.5).astype(int)
        valid=(xy[:,0]>=0)&(xy[:,0]<config['image_hw'][1])&(xy[:,1]>=0)&(xy[:,1]<config['image_hw'][0]);raw=owner[xy[valid,1],xy[valid,0]]
        labels[valid]=np.where(raw==-1,-1,np.where(raw==1,1,0))
        np.savez_compressed(out/(name+'_MASKS.npz'),object_owner=owner,raw_owner_highres_ambiguous_count=np.sum(owner==-1))
        records.append(dict(case=name,shape=shape,occlusion=occlusion,query=q,ownership_labels=labels.tolist(),transitions=transition_summary(labels),
            owner_counts={str(k):int(np.sum(owner==k))for k in (-1,0,1,2)},
            wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-pcpu,actual_software_render_count=1))
    write(out/'PROCEDURAL_SHAPE_CONTROL.json',dict(schema='NEW_PROCEDURAL_SHAPE_CONTROL_V1',status='COMPLETE',source_scene_status='PROCEDURAL_COUNTEREXAMPLE_NOT_ORIGINAL_SCENE',
        original_scene_replay=0,actual_software_renders=4,independent_scene_configurations=1,crossed_conditions=4,config=config,cases=records,
        anti_aliasing_policy='All four supersamples must agree on instance owner; mixed pixels remain ambiguous, never assigned to target/face/edge',
        material_index_is_not_edge_ID=True,new_D1_samples=0,new_model_forward=0,new_training=0,new_pose_solve=0,
        rights='New procedural boxes/colors only; no original dataset RGB or external assets read',
        geometry_scope='Procedural pallet-like openings versus registered solid bounding cuboid; not recovered original pallet mesh'))
    write(out/'PROCEDURAL_RENDER_SEAL.json',dict(status='COMPLETE',source_sha256=sha(__file__),pre_render_lock_sha256=sha(a.lock),config_sha256=sha(a.config),
        artifacts={path.name:sha(path)for path in out.iterdir()if path.is_file()},wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,actual_software_renders=4,GPU_seconds=0,new_model_or_solver_calls=0))


if __name__=='__main__':main()
