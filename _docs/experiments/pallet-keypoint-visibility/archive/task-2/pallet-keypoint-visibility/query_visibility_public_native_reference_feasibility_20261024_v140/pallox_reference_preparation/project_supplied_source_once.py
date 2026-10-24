"""Fixed-four public Pallox source geometry feasibility, root-entry only.

Raw OBJ frame + supplied object-to-camera T only. No fitted transform, native
label construction, model, private data, mask inference or coordinate repair.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys,time
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_PUBLIC_NATIVE_FEASIBILITY_POLICY_BEFORE_BODY_READS.json'
POLICY_SHA='59addd73ff75fbaf926c54d4640834327d3401fbd6a604a1097237372f816d5c'
SELECT=H/'FIXED_FOUR_FRAME_SELECTION_BEFORE_NEW_BODY_READS.json'
SELECT_SHA='c3a23c5f901a581291d9236901a5e23756c28f1a774b14b9a74c00406ca2f84b'
SRC=P/'query_visibility_controlled_real_20261022_v66/public_pallet'
RENDERER=SRC.parent/'source_audit/mesh_depth_render.py'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_BODY_READS.json';OUT=V/'pallox_reference_diagnostics_v1'
SIGNS=np.array([[-1,-1,1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]],np.int8)
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
FORBIDDEN=('model_forward','fit','current223','V112','PnP','pose_or_coordinate_optimization','q_L_correction','mask_to_semantic_labels','new_native_or_D_E_S_labels','replace_selected_frames')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def load_obj(p):
    vertices=[];faces=[]
    for line in Path(p).read_text().splitlines():
        if line.startswith('v '):vertices.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('f '):
            ix=[int(t.split('/')[0]) for t in line.split()[1:]];ix=[i-1 if i>0 else len(vertices)+i for i in ix]
            for k in range(1,len(ix)-1):faces.append([ix[0],ix[k],ix[k+1]])
    vv=np.asarray(vertices,np.float64);ff=np.asarray(faces,np.int64)
    assert vv.ndim==ff.ndim==2 and vv.shape[1]==ff.shape[1]==3 and np.isfinite(vv).all() and ff.min()>=0 and ff.max()<len(vv)
    return vv,ff
def project(points,K,T):
    camera=np.asarray(points)@T[:3,:3].T+T[:3,3];hom=camera@K.T
    with np.errstate(divide='ignore',invalid='ignore'):uv=hom[:,:2]/hom[:,2,None]
    return uv,camera[:,2]
def summaries(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(N=len(a),quantiles_0_25_50_75_95_100=np.quantile(a,[0,.25,.5,.75,.95,1]).tolist() if len(a) else None)
def json_points(x):return [[float(v) if np.isfinite(v) else None for v in row] for row in x]
def static_views(vertices,faces,corners,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig,axs=plt.subplots(1,3,figsize=(16,6));tri=vertices[faces]
    # Fixed +Z/+Y/+X orthographic viewing directions; no rotation/registration fit.
    for ax,(a,b,d) in zip(axs,((0,1,2),(0,2,1),(2,1,0))):
        order=np.argsort(tri[:,:,d].mean(1));normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(normal,axis=1);light=np.divide(abs(normal[:,d]),norm,out=np.zeros(len(norm)),where=norm>0)
        colors=np.c_[np.repeat((.35+.55*light)[:,None],3,axis=1),np.ones(len(norm))]
        ax.add_collection(PolyCollection(tri[order][:,:,[a,b]],facecolors=colors[order],edgecolors='none'))
        for i,j in EDGES:ax.plot(corners[[i,j],a],corners[[i,j],b],c='#0099bb',lw=.8)
        for k,point in enumerate(corners):ax.text(point[a],point[b],str(k),fontsize=8,color='#bb2200')
        ax.autoscale_view();ax.set_aspect('equal');ax.set_xlabel('XYZ'[a]+' (raw OBJ meters)');ax.set_ylabel('XYZ'[b]+' (raw OBJ meters)');ax.set_title('View from +'+ 'XYZ'[d])
    fig.suptitle('Pallox reconstructed mesh in unchanged raw object frame | cyan AABB V0..V7, native roles UNKNOWN')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
def font(size):
    from PIL import ImageFont
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def make_panel(rgb,depth,mesh,uv,z,K,T,corners,ident,path):
    from PIL import Image,ImageDraw
    import cv2
    raw=Image.fromarray(rgb);overlay=raw.copy();dr=ImageDraw.Draw(overlay)
    contours,_=cv2.findContours(np.isfinite(mesh).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour)>1:dr.line([tuple(map(int,p[0])) for p in contour]+[tuple(map(int,contour[0,0]))],fill=(255,200,0),width=1)
    for a,b in EDGES:
        if z[a]>0 and z[b]>0 and np.isfinite(uv[[a,b]]).all():dr.line([tuple(uv[a]),tuple(uv[b])],fill=(0,230,255),width=1)
    for i,(point,zz) in enumerate(zip(uv,z)):
        if zz>0 and np.isfinite(point).all():
            x,y=point;dr.ellipse((x-3,y-3,x+3,y+3),fill=(255,0,200));dr.text((x+4,y-8),'V'+str(i),font=font(13),fill=(255,0,200))
    axes=np.vstack((np.zeros(3),np.eye(3)*.25));axisuv,axisz=project(axes,K,T)
    if (axisz>0).all() and np.isfinite(axisuv).all():
        for i,(color,name) in enumerate(zip(((255,40,40),(40,255,40),(50,120,255)),('X','Y','Z')),1):
            dr.line([tuple(axisuv[0]),tuple(axisuv[i])],fill=color,width=3);dr.text(tuple(axisuv[i]),name,font=font(18),fill=color)
    observed=depth.astype(np.float64)/1000;valid=(depth>0)&np.isfinite(mesh);residual=np.where(valid,(observed-mesh)*1000,np.nan)
    # Fixed +/-100mm colors are display-only; no label/acceptance threshold.
    norm=np.clip(np.nan_to_num(residual,nan=0)/100,-1,1);colors=np.zeros((*depth.shape,3),np.uint8);colors[...,0]=np.where(norm>0,255,255*(1+norm));colors[...,2]=np.where(norm<0,255,255*(1-norm));colors[...,1]=255*(1-abs(norm));colors[~valid]=[30,30,30]
    canvas=Image.new('RGB',(1920,580),'white');d=ImageDraw.Draw(canvas);d.text((12,8),ident+' | SOURCE-ONLY supplied T, unchanged OBJ frame; no pose fit or native labels',font=font(23),fill='black')
    for x,label,img in ((0,'Original RGB (unchanged pixels)',raw),(640,'Mesh outline yellow / raw-object AABB V0..V7 cyan',overlay),(1280,'Observed minus mesh depth: blue closer / red farther',Image.fromarray(colors))):
        d.text((x+10,48),label,font=font(17),fill='black');canvas.paste(img,(x,76))
    d.text((12,558),'V0..V7 are fixed raw-XYZ AABB vertex codes, NOT native channels. Depth grid alignment and semantic support remain unverified.',font=font(17),fill='black');canvas.save(path)
    return valid,residual

def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(SELECT)==SELECT_SHA
    selected=read(SELECT);assert len(selected['selected'])==4 and selected['new_RGB_depth_mask_pose_K_mesh_body_reads']==0
    proofpoints=np.array([[0,0,1],[1,0,2]],float);uv,z=project(proofpoints,np.eye(3),np.eye(4));assert np.array_equal(uv,[[0,0],[.5,0]]) and np.array_equal(z,[1,2])
    ast.parse(Path(__file__).read_text())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),selection=bind(SELECT),renderer_code=bind(RENDERER),source_readme=bind(SRC/'data/README.md'),source_provenance=bind(SRC/'OFFICIAL_SOURCE_PROVENANCE.json'),output=str(OUT),
        projection='Raw OBJ vertices and fixed AABB corners transformed p_camera=T_supplied@p_object, then native integer pixel-center pinhole K. No recentering/orientation changes, inverse-pose choice or fitted registration.',
        virtual8='Raw-object XYZ AABB sign-codes V0..V7 only. Native up/front/channel mapping UNKNOWN pending source geometry/RGB evidence; all8 semantic slots U/unassigned.',
        images='Static3 fixed orthographic mesh views +4 fullscene rawRGB/mesh-AABB/depth-difference panels. No crop generation or mask body reads.',
        numeric_display_only=dict(depth_residual_color_limit_mm=100,axis_length_m=.25),prototype_invented_projection_check_PASS=True,source_mesh_pose_K_depth_RGB_model_body_reads=0)
    write(PROTOCOL,p);write(H/'ROOT_PALLOX_REFERENCE_ENTRY_TEMPLATE.json',dict(approved_public_pallox_source_projection=False,provided_T_only=True,**{k:False for k in FORBIDDEN},code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),selection=p['selection'],renderer_code=p['renderer_code']))
    print(json.dumps(dict(prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),template=bind(H/'ROOT_PALLOX_REFERENCE_ENTRY_TEMPLATE.json'))),flush=True)
def run(entry):
    e=read(entry);assert e['approved_public_pallox_source_projection'] and e['provided_T_only'] and all(e[k] is False for k in FORBIDDEN)
    assert sha(POLICY)==POLICY_SHA and sha(SELECT)==SELECT_SHA and not OUT.exists();p=read(PROTOCOL)
    for key,path in (('code',Path(__file__)),('policy',POLICY),('protocol',PROTOCOL),('selection',SELECT),('renderer_code',RENDERER)):assert e[key]==bind(path)
    assert p['code']==e['code'] and p['policy']==e['policy'] and p['selection']==e['selection'] and p['renderer_code']==e['renderer_code']
    selected=read(SELECT);OUT.mkdir();write(OUT/'BODY_INPUT_BINDINGS_BEFORE_READ.json',dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),selection=bind(SELECT),source_body_reads=0))
    os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache');os.environ['NUMBA_NUM_THREADS']='1';start=time.monotonic()
    vertices,faces=load_obj(check(selected['static_mesh_binding_inherited_without_body_read']));lo=vertices.min(0);hi=vertices.max(0);corners=lo+(SIGNS+1)/2*(hi-lo)
    assert len(vertices)==27598 and len(faces)==54744
    static=OUT/'RAW_OBJ_THREE_ORTHOGRAPHIC_VIEWS.png';static_views(vertices,faces,corners,static)
    spec=importlib.util.spec_from_file_location('v140_fixed_supplied_pose_renderer',RENDERER);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    from PIL import Image
    records=[]
    for r in selected['selected']:
        files=r['files'];T=np.loadtxt(check(files['pose']));K=np.loadtxt(check(files['K']));rgb=np.asarray(Image.open(check(files['image'])).convert('RGB'));depth=np.asarray(Image.open(check(files['depth'])))
        assert T.shape==(4,4) and K.shape==(3,3) and rgb.shape==(480,640,3) and rgb.dtype==np.uint8 and depth.shape==(480,640) and depth.dtype in (np.dtype('uint16'),np.dtype('int32'))
        assert depth.min()>=0 and depth.max()<=65535 and np.isfinite(T).all() and np.isfinite(K).all();depth=depth.astype(np.uint16)
        mesh,info=renderer.render_depth(vertices,faces,K,T,[480,640]);uv,z=project(corners,K,T)
        ident=r['sequence']+':'+r['stem'];panel=OUT/(r['sequence']+'_'+r['stem']+'_SUPPLIED_T_FULLSCENE.png');valid,residual=make_panel(rgb,depth,mesh,uv,z,K,T,corners,ident,panel)
        rawcopy=OUT/(r['sequence']+'_'+r['stem']+'_RAW_RGB.png');Image.fromarray(rgb).save(rawcopy)
        record=dict(id=ident,sequence=r['sequence'],stem=r['stem'],original_split=r['original_split'],selection_SHA256=r['selection_SHA256'],input_files_read={k:files[k] for k in ('image','depth','pose','K')},RGB_decoded_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),raw_RGB_copy=bind(rawcopy),panel=bind(panel),
            K=K.tolist(),supplied_object_to_camera_T=T.tolist(),T_rotation_orthogonality_max_abs=float(abs(T[:3,:3].T@T[:3,:3]-np.eye(3)).max()),T_rotation_determinant=float(np.linalg.det(T[:3,:3])),T_lastrow_max_abs=float(abs(T[3]-[0,0,0,1]).max()),
            raw_object_virtual_AABB8_xyz_m=corners.tolist(),raw_object_virtual_AABB8_signs=SIGNS.tolist(),virtual_vertex_codes=['V'+str(i) for i in range(8)],projected_virtual_vertices_xy=json_points(uv),projected_camera_Z_m=z.tolist(),virtual_vertex_continuous_inframe=((z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)).tolist(),
            native8_order='UNKNOWN_UNASSIGNED; V-index is a fixed raw-object AABB code, not initializer channel',native8_slots=[dict(native_slot=k,reference_xy=None,semantic_state='U',reason='No certified source up/front/native-role correspondence yet') for k in range(8)],
            renderer_facts=info,depth_units='observed uint16 millimeters, supplied mesh/pose meters',observed_valid_depth_pixels=int((depth>0).sum()),mesh_projected_pixels=int(np.isfinite(mesh).sum()),native_grid_common_support=int(valid.sum()),observed_minus_mesh_mm_conditional=summaries(residual[valid]),absolute_observed_minus_mesh_mm_conditional=summaries(abs(residual[valid])),
            depth_interpretation='Diagnostic common-grid assumption only. No semantic E/S/D/U is inferred from mask, depth gap, alpha or mesh.',model_predictions_or_GT_error_based_correspondence=False)
        records.append(record)
    summary=dict(completed=True,PASS=True,source_only=True,fully_solved=False,code=bind(__file__),entry=bind(entry),protocol=bind(PROTOCOL),policy=e['policy'],selection=e['selection'],mesh_binding=selected['static_mesh_binding_inherited_without_body_read'],mesh=dict(vertices=len(vertices),triangles=len(faces),raw_frame='Unmodified OBJ XYZ',units='meters per official README/source provenance',bounds_min=lo.tolist(),bounds_max=hi.tolist(),extent=(hi-lo).tolist(),AABB_center=((hi+lo)/2).tolist()),static_three_views=bind(static),fixed_four_records=records,
        axis_and_native_status='UNKNOWN pending actual source geometry/RGB review; no fitted/sorted-by-prediction mapping.',new_native_or_semantic_D_E_S_labels=0,mask_body_reads=0,model_forward_fit_current_V112_PnP_pose_or_q_optimization_calls=0,no_selected_frame_replacement=True,seconds=time.monotonic()-start)
    write(OUT/'SOURCE_GEOMETRY_FEASIBILITY.json',summary)
    html=['<!doctype html><meta charset="utf-8"><title>Pallox fixed source reference feasibility</title><style>body{font:17px sans-serif;max-width:1800px;margin:24px auto}img{max-width:100%}</style><h1>Pallox fixed source geometry</h1><p>Supplied T only. Raw AABB vertex codes are not native roles. No model/fit/native visibility labels.</p><img src="'+static.name+'">']
    for r in records:html.append('<h2>'+r['id']+'</h2><img src="'+Path(r['panel']['path']).name+'">')
    (OUT/'INDEX.html').write_text('\n'.join(html)+'\n');write(OUT/'PROJECTION_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),selection=e['selection'],results=bind(OUT/'SOURCE_GEOMETRY_FEASIBILITY.json'),static_views=bind(static),index=bind(OUT/'INDEX.html'),panels=[r['panel'] for r in records],source_only=True,native_correspondence='UNKNOWN',new_native_or_D_E_S_labels=0,model_current_V112_PnP_fit=0))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'PROJECTION_COMPLETE.json'),seconds=summary['seconds'])),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use metadata-only --prepare or explicit root --entry')
