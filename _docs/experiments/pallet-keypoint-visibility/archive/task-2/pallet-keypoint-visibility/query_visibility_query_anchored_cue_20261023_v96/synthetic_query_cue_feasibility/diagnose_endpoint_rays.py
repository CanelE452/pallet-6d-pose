"""Small independent source geometry rays; no renderer, models, or GT admission."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def main():
    scope=read(HERE/'SCOPE_BEFORE_SMALL_ASSOCIATION_CALCULATION.json');m=read(HERE/'SMALL_ASSOCIATION_MEASUREMENTS.json');inv=read(scope['inventory']['path']);protocol=read(scope['saved_appearance_protocol']['path'])
    with np.load(scope['geometry']['path'],allow_pickle=False)as z:
        vertices=z['vertices_world_m'];tri=z['triangles_global'];part=z['triangle_asset_local_part_id'];valid=z['triangle_valid_numeric'];toff=z['mesh_triangle_offsets']
    xyz=vertices[tri];a=xyz[:,0];e1=xyz[:,1]-a;e2=xyz[:,2]-a;orient=np.array([1 if x['USD_orientation']=='rightHanded'else-1 for x in inv['per_mesh']])[part]
    ds=np.array([x['double_sided']for x in inv['per_mesh']])[part];eye=np.array(protocol['camera']['eye_world']);out=[];total=0
    for r in m['references']:
        pid=r['association_part_id'];vertex=np.array(r['association_world_vertex']);radius=r['endpoint_patch_radius_world_proposal'];probes=[]
        for tid in range(toff[pid],toff[pid+1]):
            points=xyz[tid];match=np.flatnonzero(np.linalg.norm(points-vertex,axis=1)<1e-10)
            if not len(match):continue
            j=int(match[0]);p=points[j];v1=points[(j+1)%3];v2=points[(j+2)%3];alpha=min(.1,radius/(4*max(np.linalg.norm(v1-p),np.linalg.norm(v2-p))))
            candidate=(1-2*alpha)*p+alpha*v1+alpha*v2
            if not any(np.linalg.norm(candidate-x)<1e-10 for x in probes):probes.append(candidate)
        rec=[]
        for p in probes:
            d=p-eye;distance=float(np.linalg.norm(d));direction=d/distance;h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
            legal=valid&(np.abs(det)>1e-12)&(ds|(det*orient>0));safe=np.where(legal,det,1);s=eye-a;u=np.einsum('ij,ij->i',s,h)/safe;q=np.cross(s,e1);v=np.einsum('j,ij->i',direction,q)/safe;depth=np.einsum('ij,ij->i',e2,q)/safe
            hit=legal&(u>=0)&(v>=0)&(u+v<=1)&(depth>0);nearest=float(depth[hit].min())if hit.any()else None
            tol=64*np.finfo(np.float64).eps*max(1,distance);owners=[]if nearest is None else sorted(set(int(x)for x in part[hit&(np.abs(depth-nearest)<=tol)]))
            state='U'
            if len(owners)==1:
                if abs(nearest-distance)<=tol and owners[0]==pid:state='D_GEOMETRIC_PATCH_PROBE'
                elif nearest<distance-tol:state='S_GEOMETRIC_PATCH_PROBE'
            rec.append(dict(world_surface_probe=p.tolist(),camera_distance=distance,nearest_assembly_hit_distance=nearest,nearest_owners=owners,target_minus_hit_distance=None if nearest is None else distance-nearest,state=state));total+=1
        out.append(dict(aux_corner_index=r['aux_corner_index'],association_part_id=pid,probe_count=len(rec),probe_counts=dict(Counter(x['state']for x in rec)),probes=rec,patch_label_admitted=False))
    result=dict(completed=True,code=bind(__file__),policy=bind(HERE/'BOUNDED_RAY_DIAGNOSTIC_BEFORE_CALCULATION.json'),surface_rays=total,references=out,all308_parts_same_target_assembly_assumption=True,
        E_probe_cases=0,E_unavailable_reason='No external foreground in the one saved V3 RGB; do not synthesize an E appearance or merge V93 unrelated views.',new_renders_models_fit_private_current_PnP=0,labels_admitted=0)
    with (HERE/'FIXED_ENDPOINT_RAY_PROBES_DIAGNOSTIC.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'surface_rays':total,'references':[{k:r[k]for k in ['aux_corner_index','association_part_id','probe_count','probe_counts']}for r in out]},indent=2))
if __name__=='__main__':main()
