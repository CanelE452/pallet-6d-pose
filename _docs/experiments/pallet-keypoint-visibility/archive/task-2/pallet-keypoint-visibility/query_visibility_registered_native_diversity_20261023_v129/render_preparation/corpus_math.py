"""Pure deterministic configuration, projection, placement and material math."""
import hashlib,json
import numpy as np
OLD_DIM=np.array([1.2,.144,1.0],np.float64)
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],np.int8)
NAMES=['near_top_left','near_top_right','near_bottom_right','near_bottom_left','far_top_left','far_top_right','far_bottom_right','far_bottom_left']
ROLES=['near_top_same_side','near_top_opposite_side','near_bottom_same_side','far_top_same_side']
FOREIGN=['SAME_TARGET_MATERIAL','CARDBOARD_TAN','BACKGROUND_LIKE','HIGH_CONTRAST']
def hash_int(*parts):return int(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).hexdigest()[:16],16)
def uniform(lo,hi,*parts):return lo+(hi-lo)*(hash_int(*parts)/2**64)
def project(points,camera):
 p=np.asarray(points)@np.asarray(camera['R']).T+np.asarray(camera['t']);assert np.all(p[:,2]>0)
 return p[:,:2]/p[:,2,None]*np.array([camera['K'][0][0],camera['K'][1][1]])+np.array(camera['K'])[:2,2]
def quarter_mapping(anchors,k):
 A=np.linalg.matrix_power(np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]),k);rot=anchors@A.T;half=np.abs(anchors[0]);half=half[[2,1,0]]if k%2 else half;expected=SIGNS*half;perm=[]
 for p in expected:
  match=np.flatnonzero(np.all(rot==p,axis=1));assert len(match)==1;perm.append(int(match[0]))
 assert sorted(perm)==list(range(8));return A,perm
def camera_config(anchors,dimensions,quarter,pair,distance_scale):
 az,el=np.deg2rad(pair);distance=float(max(dimensions[0],dimensions[2])*distance_scale);eye_cf=distance*np.array([np.cos(el)*np.sin(az),-np.sin(el),-np.cos(el)*np.cos(az)])
 A,perm=quarter_mapping(anchors,quarter//90);eye=A.T@eye_cf;forward=-eye/np.linalg.norm(eye);right=np.cross(forward,[0.,-1.,0.]);right/=np.linalg.norm(right);down=np.cross(forward,right);R=np.stack([right,down,forward]);f=256/np.tan(np.deg2rad(25));K=np.array([[f,0,255.5],[0,f,255.5],[0,0,1.]])
 assert np.allclose(R@R.T,np.eye(3),atol=1e-14)and np.linalg.det(R)>0
 return dict(size=512,K=K.tolist(),R=R.tolist(),t=(-R@eye).tolist(),eye_world=eye.tolist(),near=.001,distance=distance),A,perm
def make_configs(policy,meshes):
 fixed=policy['fixed_factors'];rows=[]
 for fold,seeds in fixed['asset_seeds'].items():
  for asset_ordinal,asset in enumerate(seeds):
   for gi,geometry in enumerate(fixed['geometry_families']):
    for ai,(aspect,ratio)in enumerate(fixed['aspect_ratios'].items()):
     geomseed=hash_int(fold,asset,geometry,aspect,'geometry');z=uniform(.90,1.20,geomseed,'z');height=uniform(.12,.18,geomseed,'height');dims=np.array([ratio*z,height,z]);scale=dims/OLD_DIM;anchors=(SIGNS*OLD_DIM/2)*scale
     for qi,quarter in enumerate(fixed['signed_quarters']):
      for vi,pair in enumerate(fixed['camera_pairs_deg'][fold]):
       camera_index=2*qi+vi;cycle=(camera_index+gi+ai+asset_ordinal)%3;distance_scale=[2.2,3.2,4.2][cycle];camera,A,perm=camera_config(anchors,dims,quarter,pair,distance_scale)
       for mi,material in enumerate(fixed['materials']):
        b=len(rows);appearance=(camera_index+gi+2*ai+asset_ordinal)%4;role=(camera_index//2+gi+ai+asset_ordinal)%4;direction=(camera_index+3*gi+5*ai+asset_ordinal)%8;extent=(camera_index+gi+ai+asset_ordinal)%3
        side=1 if pair[0]>0 else 0;slots=[side,1-side,2 if side else 3,5 if side else 4];angle=np.deg2rad(45*direction)
        seeds2={purpose:hash_int(fold,asset,geometry,aspect,material if purpose=='material'else'',camera_index if purpose!='material'else'',purpose)for purpose in('material','light','background','foreign')}
        factors=dict(geometry_family=geometry,aspect=aspect,material=material,asset_seed=asset,camera_pair_deg=pair,signed_quarter=quarter,foreign_family=FOREIGN[appearance],target_role=ROLES[role],distance_scale=distance_scale)
        rows.append(dict(base_index=b,base_id='V129_BASE_%03d'%b,split=fold,factors=factors,geometry_group_id='%s_%d_%s_%s'%(fold,asset,geometry,aspect),target_layer_group_id='%s_%d_%s_%s_Q%d_V%d'%(fold,asset,geometry,aspect,quarter,vi),template=meshes[gi],template_index=gi,dimensions_xyz_m=dims.tolist(),positive_scale_xyz=scale.tolist(),raw_hw=[512,512],camera=camera,canonical_to_CF_rotation=A.tolist(),permCF_to_canonical=perm,native8_canonical_xyz=anchors.tolist(),native8_CF_xy=project(anchors,camera)[perm].tolist(),selected_CF_index=slots[role],selected_canonical_index=perm[slots[role]],sham_direction_unit_xy=[float(np.cos(angle)),float(np.sin(angle))],foreign_extent_factor=[1.25,2.,3.][extent],foreign_margin_px=[1.,3.,7.][extent],foreign_depth_fraction=[.45,.65,.80][extent],foreign_roll_deg=[-25.,0.,25.][extent],seeds=seeds2,configuration_uses_RGB_or_labels=False))
 assert len(rows)==192 and sum(r['split']=='TRAIN'for r in rows)==128
 return rows
def scaled_data(template,cfg):
 keys=('vertices','triangles','part_owner','face_axis','face_side','native_anchor3D','native_primary_part_id','native_incident_part_ids','native_patch_quads','native_edge_samples','native_witness_points');data={k:np.array(template[k],copy=True)for k in keys};scale=np.asarray(cfg['positive_scale_xyz'])
 for key in('vertices','native_anchor3D','native_patch_quads','native_edge_samples','native_witness_points'):data[key]=data[key]*scale
 assert np.array_equal(data['native_anchor3D'],np.asarray(cfg['native8_canonical_xyz']))
 for i,p in enumerate(data['native_primary_part_id']):assert np.any(np.all(data['vertices'][8*p:8*(p+1)]==data['native_anchor3D'][i],axis=1))
 return data
def foreign_placements(data,cfg):
 camera=cfg['camera'];R=np.asarray(camera['R']);t=np.asarray(camera['t']);K=np.asarray(camera['K']);points=data['native_patch_quads'][cfg['selected_canonical_index']].reshape(-1,3);pc=points@R.T+t;xy=project(points,camera);centre=(xy.min(0)+xy.max(0))/2
 angle=np.deg2rad(cfg['foreign_roll_deg']);B=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);local=(xy-centre)@B;half=np.maximum(np.abs(local).max(0)*cfg['foreign_extent_factor']+cfg['foreign_margin_px'],2.)
 frontpx=centre+np.array([[-1,-1],[1,-1],[-1,1],[1,1]])*half@B.T;z0=cfg['foreign_depth_fraction']*float(pc[:,2].min());thickness=min(.04*float(max(cfg['dimensions_xyz_m'][0],cfg['dimensions_xyz_m'][2])),.10*float(pc[:,2].min()));front=np.column_stack([(frontpx[:,0]-K[0,2])*z0/K[0,0],(frontpx[:,1]-K[1,2])*z0/K[1,1],np.full(4,z0)]);back=front.copy();back[:,2]+=thickness;covercam=np.vstack([front,back]);cover=(covercam-t)@R
 e=np.asarray(cfg['sham_direction_unit_xy']);proj=project(cover,camera);support_max=float((xy@e).max());distance=max(0.,float(np.max((support_max+1.-proj@e)*covercam[:,2]/K[0,0])));delta=np.r_[distance*e,0.];near=(covercam+delta-t)@R;far_delta=np.r_[(distance+128.*float(covercam[:,2].max())/K[0,0])*e,0.];far=(covercam+far_delta-t)@R
 near_clear=float((project(near,camera)@e).min()-support_max);far_added=float(((project(far,camera)-project(near,camera))@e).min());assert near_clear>=1.-1e-10 and far_added>=128.-1e-10
 return dict(COVER=cover,NEAR_SHAM=near,FAR_SHAM=far),dict(COVER=np.zeros(3),NEAR_SHAM=delta@R,FAR_SHAM=far_delta@R),dict(selected_CF_index=cfg['selected_CF_index'],selected_canonical_index=cfg['selected_canonical_index'],support_projected_bbox_xyxy=np.r_[xy.min(0),xy.max(0)].tolist(),physical_thickness=thickness,front_depth=z0,near_clearance_px=near_clear,far_additional_px=far_added,only_translation=True,no_RGB_or_label_search=True)
def material_rgb(point,material,seed):
 tone=.80+.35*((seed%1009)/1008);grain=.89+.07*np.sin((70+seed%53)*point[...,2]+2*np.sin(9*point[...,0]))+.04*np.sin(410*point[...,2]+6*point[...,0])
 if material=='PROCEDURAL_WOOD':return np.array([.60,.36,.16])*tone*grain[...,None]
 if material=='DARK_POLYMER':return np.array([.055,.075,.095])*tone*(.985+.015*np.sin(80*point[...,0])*np.sin(73*point[...,2]))[...,None]
 raise ValueError(material)
def shade(buffer,vertices,triangles,cfg,translation_world):
 cam=cfg['camera'];K=np.asarray(cam['K']);R=np.asarray(cam['R']);t=np.asarray(cam['t']);eye=np.asarray(cam['eye_world']);yy,xx=np.indices(buffer['depth'].shape);z=np.where(np.isfinite(buffer['depth']),buffer['depth'],0.);cp=np.stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z],-1);point=(cp-t)@R
 xyz=vertices[triangles];normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True);normal=normal[np.maximum(buffer['face_id'],0)];normal=np.where((np.sum(normal*(eye-point),-1)<0)[...,None],-normal,normal)
 seeds=cfg['seeds'];mat=cfg['factors']['material'];albedo=material_rgb(point,mat,seeds['material']);foreignpoint=point-np.asarray(translation_world);family=cfg['factors']['foreign_family'];bgbase=np.array([.57,.56,.52])if seeds['background']%2 else np.array([.52,.55,.58])
 if family=='SAME_TARGET_MATERIAL':foreign=material_rgb(foreignpoint,mat,seeds['material'])
 elif family=='CARDBOARD_TAN':foreign=np.array([.50,.35,.19])*(.96+.04*np.sin(37*foreignpoint[...,0]+51*foreignpoint[...,2]))[...,None]
 elif family=='BACKGROUND_LIKE':foreign=np.broadcast_to(bgbase,point.shape)
 elif family=='HIGH_CONTRAST':foreign=np.broadcast_to(np.array([.035,.16,.62])if seeds['foreign']%2 else np.array([.65,.045,.025]),point.shape)
 else:raise ValueError(family)
 albedo=np.where((buffer['part_id']==999)[...,None],foreign,albedo);light=np.array([-.5+((seeds['light']%101)/100),-1.,-.2-((seeds['light']%97)/96)]);light/=np.linalg.norm(light);ambient=[.30,.42,.58][seeds['light']%3];linear=albedo*(ambient+(1-ambient)*np.maximum(0,np.sum(normal*light,-1)))[...,None]
 background=bgbase*(.94+.035*(xx/512)+.025*np.sin(2*np.pi*(xx+yy)/512+(seeds['background']%31)))[...,None];linear=np.where((buffer['part_id']<0)[...,None],background,linear);linear=np.clip(linear,0,1);srgb=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055)
 return np.floor(255*np.clip(srgb,0,1)+.5).astype(np.uint8)
def CF_records(records,perm,data,camera):
 result=[]
 for i,p in enumerate(perm):
  active=np.array(records[p]['active_witness_indices']);xy=project(data['native_witness_points'][p][active],camera);patch=project(data['native_patch_quads'][p].reshape(-1,3),camera)
  result.append(dict(records[p],canonical_index=p,CF_index=i,canonical_name=records[p]['name'],name=NAMES[i],projected_active_witness_bbox_xyxy=np.r_[xy.min(0),xy.max(0)].tolist(),projected_all3face_patch_bbox_xyxy=np.r_[patch.min(0),patch.max(0)].tolist(),projected_support_scope='Continuous registered active endpoint/edge/face witness subset and all3 incident face quads, native512 coordinates; not raster mask or whole-part bbox.'))
 return result
