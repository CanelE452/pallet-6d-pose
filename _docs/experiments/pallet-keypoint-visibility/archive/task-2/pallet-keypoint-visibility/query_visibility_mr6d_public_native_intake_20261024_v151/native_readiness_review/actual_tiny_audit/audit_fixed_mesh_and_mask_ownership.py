"""Independent provided-geometry/raw-depth/mask audit, no model or fitted pose."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
H=Path(__file__).resolve().parent;V=H.parents[1]
S=V/'factual_sample_v1';SEAL=S/'TINY_SOURCE_DOWNLOAD_COMPLETE.json'
EXPECTED='5ba791ec8563224af01af1f177a1082431a225f544e86a3e670482dc08f2ad61'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def clean(x):
    if isinstance(x,dict):return{k:clean(v)for k,v in x.items()}
    if isinstance(x,(np.ndarray,list,tuple)):return[clean(v)for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(np.floating,)):return float(x)
    return x
def write(p,x):
    with Path(p).open('x')as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')
def point_triangle_distance(point,triangles):
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2];ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=(n*n).sum(1)
    assert(nn>0).all();signed=((point-a)*n).sum(1);projected=point-signed[:,None]*n/nn[:,None]
    ap=projected-a;aa=(ab*ab).sum(1);bb=(ab*ac).sum(1);cc=(ac*ac).sum(1);d=(ap*ab).sum(1);e=(ap*ac).sum(1);den=aa*cc-bb*bb
    u=(d*cc-e*bb)/den;v=(e*aa-d*bb)/den;inside=(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)
    edges=[]
    for x,y in((a,b),(b,c),(c,a)):
        xy=y-x;t=np.clip(((point-x)*xy).sum(1)/(xy*xy).sum(1),0,1);edges.append(np.linalg.norm(point-x-t[:,None]*xy,axis=1))
    distance=np.minimum.reduce(edges);distance[inside]=abs(signed[inside])/np.sqrt(nn[inside]);return float(distance.min())
def main():
    assert sha(SEAL)==EXPECTED;seal=read(SEAL);assert seal['completed']and seal['PASS'];files={r['relative_path']:r for r in seal['files']}
    assert len(files)==47
    for r in files.values():assert r['completed']and r['source_content_ID_verified']and sha(r['local_path'])==r['sha256']
    def image(rel):return np.array(Image.open(files[rel]['local_path']))
    mesh=files['models/obj_000001.ply'];text=Path(mesh['local_path']).read_text().splitlines();start=text.index('end_header')+1
    assert 'element vertex 424'in text and'element face 240'in text
    vertices=np.array([[float(x)for x in s.split()[:3]]for s in text[start:start+424]])
    face_lines=[[int(x)for x in s.split()]for s in text[start+424:]];assert len(face_lines)==240 and all(r[0]==3 for r in face_lines)
    faces=np.array([r[1:]for r in face_lines]);tri=vertices[faces]
    assert np.array_equal(vertices.min(0),[-400,-600,-72])and np.array_equal(vertices.max(0),[400,600,72])
    signs=np.array([[-1,-1,1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]])
    corners=signs*np.array([400,600,72]);corner_dist=[]
    unit=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]);assert point_triangle_distance(np.array([.2,.2,1.]),unit)==1.
    assert point_triangle_distance(np.array([0.,0.,0.]),unit)==0.
    for i,p in enumerate(corners):corner_dist.append(dict(fixed_Zup_index=i,point=p,nearest_mesh_vertex_distance=float(np.linalg.norm(vertices-p,axis=1).min()),nearest_triangle_surface_distance=point_triangle_distance(p,tri)))
    layer=[];planes={};area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    for z in np.unique(vertices[:,2]):
        m=(np.ptp(tri[:,:,2],axis=1)==0)&(tri[:,0,2]==z);bounds=np.unique(np.c_[tri[m,:,:2].min(1),tri[m,:,:2].max(1)],axis=0)
        planes[float(z)]=tri[m];layer.append(dict(z=z,horizontal_triangle_count=int(m.sum()),sum_triangle_area=area[m].sum(),XY_triangle_bounds=bounds))
    pairrows=[];fig,axs=plt.subplots(2,2,figsize=(16,9),constrained_layout=True)
    for row,frame in enumerate((5,9)):
        scene='000000';f=f'{frame:06d}';root=V/'provided_metadata/val'/scene;gt=read(root/'scene_gt.json')[str(frame)];camera=read(root/'scene_camera.json')[str(frame)]
        ids=[i for i,r in enumerate(gt)if r['obj_id']==1];assert ids==[0,1]
        masks=[image(f'val/{scene}/mask_visib/{f}_{i:06d}.png')>0 for i in ids];depth=image(f'val/{scene}/depth/{f}.png');rgb=image(f'val/{scene}/rgb/{f}.jpg')
        assert depth.dtype==np.uint16 and depth.shape==masks[0].shape==masks[1].shape==rgb.shape[:2]
        overlap=masks[0]&masks[1];zero=overlap&(depth==0);pos=overlap&(depth>0);values=depth[pos]*float(camera['depth_scale'])
        rotations=[np.asarray(gt[i]['cam_R_m2c']).reshape(3,3)for i in ids];translations=[np.asarray(gt[i]['cam_t_m2c']).reshape(3)for i in ids]
        relative=rotations[1]@rotations[0].T;angle=np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
        record=dict(scene=scene,frame=frame,GT_instance_indices=ids,overlap_pixels=int(overlap.sum()),raw_depth_zero=int(zero.sum()),raw_depth_positive=int(pos.sum()),zero_fraction=float(zero.sum()/overlap.sum()),positive_fraction=float(pos.sum()/overlap.sum()),
            depth_scale=float(camera['depth_scale']),positive_depth_scaled_min_median_max=np.quantile(values,[0,.5,1]),provided_R=rotations,provided_t=translations,translation_delta=translations[1]-translations[0],translation_distance=float(np.linalg.norm(translations[1]-translations[0])),relative_rotation_angle_deg=float(angle),
            observed='Most overlapping visible-mask pixels have positive depth; missing depth alone cannot explain them. Distinct GT poses are retained. No unique generation cause inferred.',RGB=bind(files[f'val/{scene}/rgb/{f}.jpg']['local_path']),depth=bind(files[f'val/{scene}/depth/{f}.png']['local_path']))
        pairrows.append(record)
        axs[row,0].imshow(rgb);axs[row,0].set_title(f'Raw source RGB: scene{scene} frame{frame} / two pallets')
        axs[row,1].imshow(rgb);rgba=np.zeros((*depth.shape,4),float);rgba[masks[0]]=[1,0,0,.5];rgba[masks[1]]=[0,.3,1,.5];rgba[overlap]=[1,0,1,.65];rgba[zero]=[1,1,0,1];axs[row,1].imshow(rgba)
        axs[row,1].set_title(f'Visible masks red/blue; overlap magenta {overlap.sum():,}\nOverlap depth0 yellow {zero.sum():,} ({zero.sum()/overlap.sum():.2%})')
        for ax in axs[row]:ax.axis('off')
    ownerfig=H/'VISIBLE_MASK_OVERLAP_RAW_DEPTH_DIAGNOSTIC.png';fig.savefig(ownerfig,dpi=120);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    for ax,z,title in((axs[0],72.,'+Z outer plane: five deck boards'),(axs[1],-72.,'-Z outer plane: three bottom runners')):
        ax.add_collection(PolyCollection(planes[z][:,:,:2],facecolors='#d2aa72',edgecolors='#55412b',linewidths=.7));ax.set_xlim(-450,450);ax.set_ylim(-650,650);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('object X [model units/mm convention]');ax.set_ylabel('object Y')
        for i,p in enumerate(corners):
            if p[2]==z:ax.plot(p[0],p[1],'ro',ms=5);ax.text(p[0],p[1],str(i),color='darkred',fontsize=12)
    axs[2].add_collection(PolyCollection(tri[:,:,[0,2]],facecolors='#d2aa7218',edgecolors='#55412b55',linewidths=.4));axs[2].set_xlim(-450,450);axs[2].set_ylim(-100,100);axs[2].set_aspect('equal');axs[2].set_title('Object X-Z structure / +Z is deck side');axs[2].set_xlabel('object X');axs[2].set_ylabel('object Z')
    for z in np.unique(vertices[:,2]):axs[2].axhline(z,ls='--',lw=.5,color='grey')
    fig.suptitle('Provided CAD geometry only. Labels are fixed canonical indices, not camera-dynamic native slots or semantic D/E.')
    meshfig=H/'MESH_ZUP_BOARD_LAYOUT_GEOMETRY_ONLY.png';fig.savefig(meshfig,dpi=140);plt.close(fig)
    result=dict(completed=True,PASS=True,download_seal=bind(SEAL),all47_source_files_independently_rehashed=True,mask_owner_overlap=pairrows,mesh=dict(file=bind(mesh['local_path']),vertices=424,triangles=240,bounds_min=vertices.min(0),bounds_max=vertices.max(0),unique_X=np.unique(vertices[:,0]),unique_Y=np.unique(vertices[:,1]),unique_Z=np.unique(vertices[:,2]),horizontal_layers=layer,canonical8_geometry_support=corner_dist),
        physical_UP_evidence='Provided mesh has five continuous outer deck boards at +Z72 versus three bottom runners at -Z72, with blocks/bearers between. Two raw source views show this deck-over-runner anatomy, and supplied rotations map +Z toward image-up. Supports source-asset +Z physical deck-up; not an initializer-channel or semantic-label certificate.',
        BOP_official_visibility=read(H/'BOP_OFFICIAL_CODE_PROVENANCE.json'),BOP_formula='bop19: (d_model-d_test <= delta OR d_test==0) AND d_model>0. It is not an exclusive multi-instance owner test; MR generation mode/delta are unverified.',
        limits=['The two overlaps are mostly positive depth, so missing-depth semantics alone are insufficient. No unique causal explanation is claimed.','Native corners happen to coincide with the provided nominal mesh surface; actual pallet dimensions and annotation precision remain uncertain by centimeters.','AABB/mesh membership and full/visible instance masks are geometry/proxy evidence, not automatic native semantic D/E/S gold.','Raw RGB is viewed/decoded for this diagnosis only. JPEG decoder equivalence/canonical model-input transport is not certified.'],
        source_viewed_raw_RGB=[str(files[f'val/000000/rgb/{f:06d}.jpg']['local_path'])for f in(5,9)],native_labels_created=0,new_model_or_R0_or_encoder_or_fit_or_PnP=0,current223_V112_private_reads=0)
    f=H/'RESULTS.json';write(f,result);write(H/'AUDIT_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),download_seal=bind(SEAL),results=bind(f),figures=[bind(ownerfig),bind(meshfig)],native_labels_created=0,model_fit_current_private_PnP=0))
    print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'AUDIT_COMPLETE.json'),corner_distances=clean(corner_dist),overlap=[{k:r[k]for k in('frame','overlap_pixels','raw_depth_zero','raw_depth_positive','translation_distance','relative_rotation_angle_deg')}for r in pairrows])),flush=True)
if __name__=='__main__':main()
