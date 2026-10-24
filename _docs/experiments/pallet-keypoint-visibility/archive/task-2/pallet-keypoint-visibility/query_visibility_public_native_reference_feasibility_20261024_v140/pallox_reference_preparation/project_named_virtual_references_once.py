"""Named source-only CF virtual references under a frozen provisional axis map.
No visibility labels, model, pose fit, point snapping or initializer association.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_PUBLIC_NATIVE_FEASIBILITY_POLICY_BEFORE_BODY_READS.json';POLICY_SHA='59addd73ff75fbaf926c54d4640834327d3401fbd6a604a1097237372f816d5c'
AXIS=H/'AXIS_REGISTRATION_PROPOSAL_BEFORE_NAMED_NATIVE_LABELS.json';AXIS_SHA='74802a2f7663e75695a9ca1d44870357e3e7b5914c4dc28572fceab3fc3b756e'
FIRST=V/'pallox_reference_diagnostics_v1/PROJECTION_COMPLETE.json'
V74=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py';V74_SHA='4ddd134aa26da1d676628bd6987a161ab36f09328dfdaa89bedd213e397755ab'
SOURCECODE=H/'project_supplied_source_once.py';PROTOCOL=H/'NAMED_REFERENCE_PROTOCOL_BEFORE_COORDINATES.json';OUT=V/'pallox_named_virtual_reference_v1'
FORBIDDEN=('model_forward','fit','current223','V112','PnP','pose_or_coordinate_optimization','q_L_correction','new_D_E_S_labels','mesh_distance_to_visibility_labels','nearest_prediction_or_GT_error_permutation','frame_replacement')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def original_native_math():
    assert sha(V74)==V74_SHA;nodes=[];wanted={'FIXED_ZUP_SIGNS','VERTICAL_FIXED_FACES','NATIVE_NAMES','outward_normals','normal_facing_permutation'}
    for n in ast.parse(V74.read_text()).body:
        if isinstance(n,ast.FunctionDef) and n.name in wanted:nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets):nodes.append(n)
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(V74),'exec'),ns);assert wanted.issubset(ns);return ns
def project(p,K,T):
    camera=p@T[:3,:3].T+T[:3,3];hp=camera@K.T
    with np.errstate(divide='ignore',invalid='ignore'):xy=hp[:,:2]/hp[:,2,None]
    return xy,camera[:,2]
def triangle_distance(point,triangles):
    # Exact point-to-triangle minimum via face projection or three closed edges.
    A,B,C=np.moveaxis(triangles,1,0);ab=B-A;ac=C-A;ap=point-A
    nn=np.cross(ab,ac);nn2=np.einsum('ij,ij->i',nn,nn);dot=np.einsum('ij,ij->i',ap,nn)
    d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac);d20=np.einsum('ij,ij->i',ap,ab);d21=np.einsum('ij,ij->i',ap,ac);den=d00*d11-d01*d01
    good=(den>1e-24)&(nn2>1e-24);u=np.divide(d11*d20-d01*d21,den,out=np.full(len(den),np.nan),where=good);vv=np.divide(d00*d21-d01*d20,den,out=np.full(len(den),np.nan),where=good)
    inside=good&(u>=0)&(vv>=0)&(u+vv<=1);d2=np.full(len(den),np.inf);d2[inside]=dot[inside]**2/nn2[inside]
    for a,b in ((A,B),(B,C),(C,A)):
        edge=b-a;length=np.einsum('ij,ij->i',edge,edge);t=np.divide(np.einsum('ij,ij->i',point-a,edge),length,out=np.zeros(len(length)),where=length>0);closest=a+np.clip(t,0,1)[:,None]*edge;dd=np.einsum('ij,ij->i',point-closest,point-closest);d2=np.minimum(d2,dd)
    return float(np.sqrt(d2.min()))
def font(n):
    from PIL import ImageFont
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def panel(rgb,xy,z,ident,path):
    from PIL import Image,ImageDraw
    raw=Image.fromarray(rgb);overlay=raw.copy();d=ImageDraw.Draw(overlay);edges=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
    for a,b in edges:
        if z[a]>0 and z[b]>0 and np.isfinite(xy[[a,b]]).all():d.line([tuple(xy[a]),tuple(xy[b])],fill=(255,235,0),width=1)
    for k,(q,depth) in enumerate(zip(xy,z)):
        if depth>0 and np.isfinite(q).all():
            x,y=q;color=(0,255,255) if k<4 else (255,0,220);d.ellipse((x-3,y-3,x+3,y+3),fill=color);d.text((x+4,y-10),str(k),font=font(17),fill=color)
    canvas=Image.new('RGB',(1280,606),'white');dd=ImageDraw.Draw(canvas);dd.text((12,8),ident+' | provisional source-static UP=-OBJ Y | exact V74 CF virtual roles',font=font(22),fill='black');dd.text((12,39),'Raw RGB unchanged',font=font(17),fill='black');dd.text((652,39),'Named virtual references; cyan0..3 FRONT / magenta4..7 REAR',font=font(16),fill='black');canvas.paste(raw,(0,65));canvas.paste(overlay,(640,65));dd.text((12,550),'0 F-top-left;1 F-top-right;2 F-bottom-right;3 F-bottom-left;4..7 paired REAR. All semantic states U/unlabelled.',font=font(17),fill='black');dd.text((12,576),'Virtual cuboid role is separate from local structural support/surface membership. No model, pose fit, snapping or D/E labels.',font=font(16),fill='black');canvas.save(path)
def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(AXIS)==AXIS_SHA
    axis=read(AXIS);assert axis['new_named_native_coordinates_generated']==0 and axis['source_specific_registration']['rotation_determinant']==1
    native=original_native_math();assert len(native['NATIVE_NAMES'])==9
    tr=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]);assert triangle_distance(np.array([.2,.2,1]),tr)==1 and triangle_distance(np.array([2.,0,0]),tr)==1
    first=read(FIRST);assert first['completed'] and first['PASS'] and first['native_correspondence']=='UNKNOWN'
    results=read(check(first['results']));assert len(results['fixed_four_records'])==4
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),axis_proposal=bind(AXIS),source_projection_seal=bind(FIRST),source_results=first['results'],source_reader_code=bind(SOURCECODE),original_V74_code=bind(V74),output=str(OUT),
        role_rule='Exact AST original V74 outward_normals +normal_facing_permutation with original fixed signs/faces/native names. Registered world=(Xobj-Cx,Zobj-Cz,-(Yobj-Cy)); no fitted choice.',
        face_tie_guard='Report top2 vertical-normal cosine margin and top-front projected-X separation; keep all8 coordinates, ties explicit. No result-dependent convention choice.',
        source_scope='One visually justified provisional static registration of this reconstructed bin. Not official publisher axes, universal geometry certificate, native D/E gold or current transfer.',
        mesh_distance='Exact minimum Euclidean point-to-triangle distance of unchanged virtual references; diagnostic only, never snap, filter, correct pose or label visibility.',invented_triangle_distance_PASS=True,new_named_coordinates_generated=0,source_mesh_pose_K_RGB_model_body_reads_during_prepare=0)
    write(PROTOCOL,p);write(H/'ROOT_NAMED_REFERENCE_ENTRY_TEMPLATE.json',dict(approved_named_public_virtual_references=False,provided_T_only=True,**{k:False for k in FORBIDDEN},code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),axis_proposal=p['axis_proposal'],source_projection_seal=p['source_projection_seal'],original_V74_code=p['original_V74_code']))
    print(json.dumps(dict(prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),template=bind(H/'ROOT_NAMED_REFERENCE_ENTRY_TEMPLATE.json'))),flush=True)
def run(entry):
    e=read(entry);assert e['approved_named_public_virtual_references'] and e['provided_T_only'] and all(e[k] is False for k in FORBIDDEN)
    assert sha(POLICY)==POLICY_SHA and sha(AXIS)==AXIS_SHA and not OUT.exists();p=read(PROTOCOL)
    for key,path in (('code',Path(__file__)),('policy',POLICY),('protocol',PROTOCOL),('axis_proposal',AXIS),('source_projection_seal',FIRST),('original_V74_code',V74)):assert e[key]==bind(path)
    check(p['source_reader_code']);axis=read(AXIS);facts=read(check(p['source_results']));native=original_native_math();B=np.asarray(axis['source_specific_registration']['world_from_centered_object_rotation']);C=np.asarray(axis['source_specific_registration']['C_raw_object_AABB_center_m']);obj_from_world=np.asarray(axis['source_specific_registration']['object_from_world_4x4'])
    OUT.mkdir();write(OUT/'INPUT_BINDINGS_BEFORE_NAMED_COORDINATES.json',dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),axis_proposal=e['axis_proposal'],source_results=p['source_results'],named_coordinates_generated=0))
    spec=importlib.util.spec_from_file_location('v140_original_source_obj_reader',SOURCECODE);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader);vertices,faces=reader.load_obj(check(facts['mesh_binding']));tri=vertices[faces]
    extent=np.asarray(facts['mesh']['extent']);world_extent=extent[[0,2,1]];fixed=native['FIXED_ZUP_SIGNS'].astype(float)*world_extent/2;object_fixed=fixed@B+C
    distances=np.array([triangle_distance(q,tri) for q in object_fixed]);rows=[];frames=[]
    from PIL import Image
    for r in facts['fixed_four_records']:
        T=np.asarray(r['supplied_object_to_camera_T']);K=np.asarray(r['K']);xyfixed,zfixed=project(object_fixed,K,T);camera_obj=-T[:3,:3].T@T[:3,3];camera_world=(camera_obj-C)@B.T
        perm,cos=native['normal_facing_permutation'](fixed,xyfixed,camera_world);xy=xyfixed[perm];z=zfixed[perm];margin=float(np.sort(cos)[-1]-np.sort(cos)[-2]);front_image_x_gap=float(xy[1,0]-xy[0,0])
        # Algebraic registration only; source raw-T projection must stay identical.
        xyworld,zworld=project(fixed,K,T@obj_from_world);assert np.allclose(xyfixed,xyworld,rtol=1e-13,atol=1e-12,equal_nan=True) and np.allclose(zfixed,zworld,rtol=1e-13,atol=1e-12)
        assert sorted(perm.tolist())==list(range(8));dynamic=fixed[perm]
        for a,b in ((0,3),(1,2),(4,7),(5,6)):assert np.allclose(dynamic[a,:2],dynamic[b,:2]) and dynamic[a,2]>dynamic[b,2]
        for k in range(4):assert np.count_nonzero(abs(dynamic[k]-dynamic[k+4])>1e-12)==1 and dynamic[k,2]==dynamic[k+4,2]
        rgb=np.asarray(Image.open(check(r['input_files_read']['image'])).convert('RGB'));assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['RGB_decoded_sha256'];fig=OUT/(r['sequence']+'_'+r['stem']+'_NAMED_CF_VIRTUAL_REFERENCES.png');panel(rgb,xy,z,r['id'],fig)
        status='PROVISIONAL_STATIC_AXIS_ROLE_DEFINITION' if margin>1e-10 and front_image_x_gap>1e-10 else 'PROVISIONAL_WITH_FRONT_TIE_AMBIGUITY'
        frame=dict(id=r['id'],sequence=r['sequence'],stem=r['stem'],original_split=r['original_split'],source_input_files=r['input_files_read'],axis_registration=e['axis_proposal'],role_status=status,permutation_dynamic_to_registered_fixed=perm.tolist(),vertical_face_cosines_original_V74_order=cos.tolist(),front_normal_top2_margin=margin,front_top_image_X_separation_px=front_image_x_gap,registered_camera_position_world=camera_world.tolist(),figure=bind(fig),semantic_states=['U']*8)
        frames.append(frame)
        for k in range(8):
            valid=bool(z[k]>0 and np.isfinite(xy[k]).all());inframe=bool(valid and 0<=xy[k,0]<640 and 0<=xy[k,1]<480)
            rows.append(dict(id=r['id']+':native'+str(k),source_frame_id=r['id'],original_split=r['original_split'],native_slot=k,native_name=native['NATIVE_NAMES'][k],registered_fixed_corner_index=int(perm[k]),reference_object_xyz_m=object_fixed[perm[k]].tolist(),reference_registered_world_xyz_m=dynamic[k].tolist(),reference_xy=[float(a) if np.isfinite(a) else None for a in xy[k]],camera_Z_m=float(z[k]),continuous_projection_valid=valid,continuous_inframe=inframe,role_status=status,front_normal_top2_margin=margin,virtual_reference_to_reconstructed_mesh_min_triangle_distance_m=float(distances[perm[k]]),distance_is_diagnostic_not_visibility=True,semantic_state='U',semantic_annotation_status='UNLABELLED_PENDING_SOURCE_RGB_STRUCTURAL_SUPPORT_ADJUDICATION',initializer_association='NOT_RUN_OR_READ',human_gold=False))
    assert len(rows)==32 and all(sum(rr['source_frame_id']==f['id'] for rr in rows)==8 for f in frames)
    result=dict(completed=True,PASS=True,source_only=True,code=bind(__file__),policy=e['policy'],entry=bind(entry),protocol=bind(PROTOCOL),axis_proposal=e['axis_proposal'],source_projection_seal=e['source_projection_seal'],original_V74_code=e['original_V74_code'],registered_fixed_world_corners_xyz_m=fixed.tolist(),registered_fixed_object_corners_xyz_m=object_fixed.tolist(),fixed_corner_mesh_triangle_distance_m=distances.tolist(),world_extent_xyz_m=world_extent.tolist(),frames=frames,rows=rows,all4frames32slots_preserved=True,role_scope='Exact original CF convention conditional on a visually justified provisional source-static UP=-OBJY registration; not a publisher-native certificate.',semantic_state_counts={'U':32},new_D_E_S_labels=0,pose_query_or_model_optimization=0,model_fit_current223_V112_PnP_reads_or_calls=0)
    write(OUT/'NAMED_CF32_REFERENCES.json',result);html=['<!doctype html><meta charset="utf-8"><title>Pallox provisional CF references</title><h1>32 provisional CF virtual references</h1><p>Exact original role rule under frozen source-static UP=-OBJY registration. Visibility states remain U. No model or pose fit.</p>']
    for f in frames:html.append('<h2>'+f['id']+'</h2><img style="max-width:100%" src="'+Path(f['figure']['path']).name+'">')
    (OUT/'INDEX.html').write_text('\n'.join(html)+'\n');write(OUT/'NAMED_REFERENCE_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),policy=e['policy'],protocol=bind(PROTOCOL),entry=bind(entry),axis_proposal=e['axis_proposal'],source_projection_seal=e['source_projection_seal'],results=bind(OUT/'NAMED_CF32_REFERENCES.json'),figures=[f['figure'] for f in frames],index=bind(OUT/'INDEX.html'),all32slots=True,new_D_E_S_labels=0,source_only=True,model_fit_current_V112_PnP=0))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'NAMED_REFERENCE_COMPLETE.json'),front_face_margins=[f['front_normal_top2_margin'] for f in frames])),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use metadata-only --prepare or explicit root --entry')
