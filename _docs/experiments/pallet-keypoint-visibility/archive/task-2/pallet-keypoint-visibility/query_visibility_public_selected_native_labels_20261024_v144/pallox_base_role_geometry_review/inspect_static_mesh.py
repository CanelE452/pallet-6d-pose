"""Source-static mesh/topology diagnostics independent of predictions or GT errors."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_public_selected_native_labels_20261024_v144';H=V/'pallox_base_role_geometry_review'
mesh=P/'query_visibility_controlled_real_20261022_v66/public_pallet/data/assets/mesh/pallox.obj'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
policy=V/'ROOT_SOURCE_NATIVE_ALIGNMENT_SCOPE.json';assert bind(policy)['sha256']=='d2d356ca7325aff8918c0839b6ca3657bb4dd8c4a1adf96d119a52661af161ca'
assert bind(mesh)['sha256']=='2c44a016ec657955af2d195876af1234fab143dc1651203d13bad1c44c836e0f'
v=[];f=[];groups=[]
for line in mesh.open():
 if line.startswith('v '):v.append([float(x)for x in line.split()[1:]])
 elif line.startswith('f '):f.append([int(x.split('/')[0])-1 for x in line.split()[1:]])
 elif line.startswith(('g ','o ','usemtl ')):groups.append(line.strip())
v=np.asarray(v);f=np.asarray(f);xyz=v[:,:3];rgb=v[:,3:6];tri=xyz[f];color=rgb[f].mean(1)
lo=xyz.min(0);hi=xyz.max(0);height=hi[1]-xyz[:,1]
fig,ax=plt.subplots(2,2,figsize=(18,12),constrained_layout=True)
for a,(depthaxis,sgn,haxis,title) in zip(ax.ravel(),[(2,1,0,'View from +OBJ Z'),(2,-1,0,'View from -OBJ Z'),(0,1,2,'View from +OBJ X'),(0,-1,2,'View from -OBJ X')]):
 order=np.argsort(sgn*tri[:,:,depthaxis].mean(1));poly=np.stack([tri[:,:,haxis],hi[1]-tri[:,:,1]],axis=-1)
 a.add_collection(PolyCollection(poly[order],facecolors=np.clip(color[order],0,1),edgecolors='none',rasterized=True));a.autoscale();a.set_aspect('equal');a.set_title(title+' | original vertex RGB, depth-sorted triangles')
 a.set_xlabel('Raw OBJ '+('X'if haxis==0 else'Z')+' (m)');a.set_ylabel('Height above fullmesh min-height plane (m); UP=-OBJ Y');a.grid(alpha=.3)
 a.set_ylim(-.03,.87);a.set_yticks(np.arange(0,.9,.1))
fig.suptitle('Pallox static mesh: wholecrate and integrated lower support; NO query/pose fitting/cut selection',fontsize=16)
fig.savefig(H/'STATIC_MESH_VERTEX_RGB_FOUR_ORTHOGRAPHIC.png',dpi=150);plt.close(fig)
edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);g=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(v),len(v)));n,labels=connected_components(g,directed=False);sizes=np.bincount(labels)
components=[]
for c in np.argsort(sizes)[::-1][:12]:
 q=xyz[labels==c];components.append(dict(component=int(c),vertices=int(len(q)),face_count=int(np.all(labels[f]==c,axis=1).sum()),bounds_min=q.min(0).tolist(),bounds_max=q.max(0).tolist()))
# Fixed 25mm raw-Y bins: descriptive envelope only, never a selected boundary.
bins=[]
for a,b in zip(np.arange(-.425,.426,.025)[:-1],np.arange(-.425,.426,.025)[1:]):
 q=xyz[(xyz[:,1]>=a)&(xyz[:,1]<b)]
 if len(q):bins.append(dict(raw_Y_interval_m=[float(a),float(b)],height_above_fullbottom_interval_m=[float(hi[1]-b),float(hi[1]-a)],vertices=len(q),raw_XZ_min=q[:,[0,2]].min(0).tolist(),raw_XZ_max=q[:,[0,2]].max(0).tolist()))
area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
r=dict(completed=True,PASS=True,policy=bind(policy),code=bind(Path(__file__)),mesh=bind(mesh),vertices=len(v),faces=len(f),group_material_records=groups,raw_bounds_min=lo.tolist(),raw_bounds_max=hi.tolist(),connected_components=n,components=components,isolated_vertices=int((sizes==1).sum()),fixed25mm_vertical_envelopes=bins,total_triangle_area_m2=float(area.sum()),figure=bind(H/'STATIC_MESH_VERTEX_RGB_FOUR_ORTHOGRAPHIC.png'),no_query_prediction_reference_error_or_current_data_read=True,no_baseupper_boundary_selected=True,pose_fit_PnP_model_calls=0,V112_access=0)
with (H/'STATIC_MESH_FACTS.json').open('x')as stream:json.dump(r,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(completed=True,PASS=True,facts=bind(H/'STATIC_MESH_FACTS.json'),figure=r['figure'],components=n,major_components=[c['vertices']for c in components[:3]])))
