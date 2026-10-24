"""Saved triplet audit; own projection/crop and ray-plane/dominant-axis oracle.
No production teacher/render imports and no renderer/model/optimizer calls.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
OUT=V/'render_preparation/actual_matched_rgb_triplet';C=V/'cue_contract_preparation'
EPS=64*np.finfo(np.float64).eps
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def save(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def crop(rgb,q,L):
    a=(np.arange(96,dtype=np.float64)+.5)/96-.5;x,y=np.meshgrid(q[0]+a*L,q[1]+a*L)
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);fx=x-ix;fy=y-iy;h,w=rgb.shape[:2];z=np.zeros((96,96,3),np.float64)
    for dx,dy,wt in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        xx=ix+dx;yy=iy+dy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);z[ok]+=rgb[yy[ok],xx[ok]].astype(np.float64)*wt[ok,None]
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5);z=np.floor(np.clip(z,0,255)+.5).astype(np.uint8);z[~valid]=0
    return z.transpose(2,0,1),valid[None]
def projection(x,R,t,K):
    c=np.asarray(x)@R.T+t;return (c@K.T)[...,:2]/c[...,2:3],c[...,2]
class Oracle:
    def __init__(self,xyz,owners,orientation,two):
        self.xyz=xyz;self.owners=owners;self.orientation=orientation;self.two=two
        self.a=xyz[:,0];self.e1=xyz[:,1]-self.a;self.e2=xyz[:,2]-self.a;self.n=np.cross(self.e1,self.e2)
        drop=np.argmax(np.abs(self.n),axis=1);self.ix=(drop+1)%3;self.iy=(drop+2)%3;self.rows=np.arange(len(xyz))
        self.ex=self.e1[self.rows,self.ix];self.ey=self.e1[self.rows,self.iy];self.fx=self.e2[self.rows,self.ix];self.fy=self.e2[self.rows,self.iy];self.area=self.ex*self.fy-self.ey*self.fx
    def cast(self,eye,p,owner):
        direction=p-eye;distance=float(np.linalg.norm(direction));direction/=distance
        den=np.einsum('ij,j->i',self.n,direction);scale=np.linalg.norm(self.e1,axis=1)*np.linalg.norm(self.e2,axis=1)
        legal=(np.abs(den)>EPS*np.maximum(1,scale))&(self.two|(den*self.orientation<0))&(self.area!=0)
        depth=np.einsum('ij,ij->i',self.n,self.a-eye)/np.where(legal,den,1)
        point=eye+depth[:,None]*direction;d=point-self.a;dx=d[self.rows,self.ix];dy=d[self.rows,self.iy];area=np.where(legal,self.area,1)
        u=(dx*self.fy-dy*self.fx)/area;v=(self.ex*dy-self.ey*dx)/area
        hit=legal&(depth>0)&(u>=-EPS)&(v>=-EPS)&(u+v<=1+EPS)
        nearest=float(depth[hit].min())if hit.any()else None;tol=EPS*max(1,distance,nearest or 0)
        owners=[]if nearest is None else sorted(set(int(x)for x in self.owners[hit&(np.abs(depth-nearest)<=tol)]))
        state='U';margin=None if nearest is None else distance-nearest
        if len(owners)==1:
            if abs(margin)<=tol and owners[0]==owner:state='DIRECT'
            elif margin>tol:state='FOREIGN'if owners[0]>=308 else'SELF'
        return dict(state=state,owners=owners,margin=margin)
def reduce_state(source,combined,geometry,eye):
    face=np.array(geometry['face_probe_points']);normal=np.array(geometry['face_normals']);d=eye-face.mean(1);d/=np.linalg.norm(d,axis=1)[:,None]
    cosine=(d*normal).sum(1);eligible=np.flatnonzero(cosine>1e-10)
    witnesses={int(i):[0]+list(range(1+9*i,10+9*i))+sum([list(range(28+3*j,31+3*j))for j in geometry['face_edge_axes'][i]],[])for i in eligible}
    allstate=lambda records,indices,state:all(records[i]['state']==state for i in indices)
    direct=[i for i,w in witnesses.items()if allstate(source,w,'DIRECT')]
    if direct:
        if any(allstate(combined,witnesses[i],'DIRECT')for i in direct):return'D'
        if all(allstate(combined,witnesses[i],'FOREIGN')for i in direct):return'E'
        return'U'
    if witnesses and all(allstate(source,w,'SELF')for w in witnesses.values()):return'S'
    if not len(eligible)and np.all(cosine<-1e-10)and all(x['state']=='SELF'for x in source):return'S'
    return'U'
def main():
    assert sha(OUT/'RENDER_SEAL.json')=='fac353b66ea3435e9c118ae56951d84ecc1bc3857be748920e3cd90e07001a84'
    seal=read(OUT/'RENDER_SEAL.json');assert seal['completed']and seal['PASS_execution']
    checked=set()
    def check(b):
        k=(b['path'],b['sha256'])
        if k not in checked:assert sha(b['path'])==b['sha256'],b['path'];checked.add(k)
    for b in seal['files']:check(b)
    check(seal['protocol']);check(seal['result']);protocol=read(seal['protocol']['path']);result=read(seal['result']['path'])
    for k in ['code','math','generated','cue_descriptor','cue_geometry','cue_teacher_code','cue_contract','previous_appearance_protocol','previous_appearance_RGB','geometry','inventory','old_texture_camera_UV_code','geometry_raster_code','crop_code']:check(protocol[k])
    assert result['appearance_RGB_passes']==3 and result['training_admission']is False
    cue=read(protocol['cue_geometry']['path']);meta=read(protocol['cue_descriptor']['path']);teacher=read(OUT/'EXPECTED_LOCAL_CUE_GEOMETRY_GT_ONLY.json');refs=cue['references'];cam=cue['camera']
    assert cam==protocol['camera']==meta['camera']
    K=np.array(cam['K']);R=np.array(cam['R']);t=np.array(cam['t']);eye=np.array(cam['eye_world'])
    with np.load(OUT/'AUX_OWNERSHIP_GEOMETRY_ONLY.npz',allow_pickle=False)as z:
        cover=z['foreign_cover_vertices_world'];aside=z['foreign_sham_vertices_world'];faces=z['foreign_triangles'];owners=z['part_id'];rgbdepth=z['RGB_depth']
        assert np.array_equal(z['camera_K'],K)and np.array_equal(z['camera_R'],R)and np.array_equal(z['camera_t'],t)
        assert owners.shape==(3,640,640)and owners.dtype==np.int16 and z['ambiguity'].shape==(3,640,640)
    ext=protocol['external_geometry'];assert np.array_equal(cover,np.array(ext['cover_vertices_world']))and np.array_equal(aside,np.array(ext['sham_vertices_world']))and np.array_equal(faces,np.array(ext['triangles']))
    cc=cover@R.T+t;ss=aside@R.T+t;delta=aside-cover;tr=np.array(ext['sham_world_translation'])
    assert np.allclose(delta,tr,rtol=0,atol=4e-15)and np.allclose(cc[:,2],ss[:,2],rtol=0,atol=5e-15)
    assert np.allclose(np.ptp(cc,axis=0),ext['dimensions_camera'],rtol=0,atol=5e-15)
    normal=np.cross(cover[faces[:,1]]-cover[faces[:,0]],cover[faces[:,2]]-cover[faces[:,0]])
    assert np.all(np.einsum('ij,ij->i',normal,cover[faces].mean(1)-cover.mean(0))>0)
    # Own reconstruction from the declared supportdisk/frontdepth, not math helper.
    target=meta['fixed_target_aux7'];xy=np.array(target['projected_support_disk_center_xy']);half=target['radius_px']+3;front=.6*target['min_patch_camera_z'];thickness=.05*front
    centre=np.r_[(xy-K[:2,2])/np.diag(K)[:2]*front,front+thickness/2];dimensions=np.r_[np.full(2,2*half)/np.diag(K)[:2]*front,thickness]
    signs=np.array([[x,y,z]for x in[-1,1]for y in[-1,1]for z in[-1,1]])
    owncover=(centre+signs*dimensions/2-t)@R;shift_world=np.array([(2*half+2)*front/K[0,0],0,0])@R
    assert np.array_equal(owncover,cover)and np.allclose(owncover+shift_world,aside,rtol=0,atol=4e-15)
    quads=np.array(refs[7]['geometry']['face_patch_quads_world']).reshape(-1,3);p_uv,p_z=projection(quads,R,t,K);c_uv,c_z=projection(cover,R,t,K);s_uv,s_z=projection(aside,R,t,K)
    frontrect=np.r_[xy-half,xy+half];margins=np.c_[p_uv-frontrect[:2],frontrect[2:]-p_uv]
    assert margins.min()>0 and c_z.max()<p_z.min()
    horizontal_clear=float(s_uv[:,0].min()-p_uv[:,0].max());assert horizontal_clear>0
    with np.load(OUT/'MATCHED_RGB96_INPUT_ONLY.npz',allow_pickle=False)as z:
        crops=z['rgb96'];valid=z['crop_valid96'];q=z['query8_xy'];L=z['L'];stages=z['stage'].tolist()
    assert stages==['BEFORE','COVER','SHAM']and crops.shape==(3,8,3,96,96)and crops.dtype==np.uint8 and valid.dtype==bool
    assert np.array_equal(q,np.array([r['original_q']for r in refs]))and np.array_equal(L,np.array([r['L']for r in refs]))
    source_contract=read(protocol['cue_contract']['path']);check(source_contract['frozen_source_associations']);original=read(source_contract['frozen_source_associations']['path'])['references']
    for r,o in zip(refs,original):
        assert r['aux_corner_index']==o['aux_corner_index']and r['original_q']==o['original_aux_q']and r['L']==o['L']and r['part_id']==o['association_part_id']
        assert r['geometry']['endpoint_world']==o['association_world_vertex']
    rgb=[]
    for si,stage in enumerate(stages):
        a=np.array(Image.open(OUT/f'{stage}_RGB.png').convert('RGB'),copy=True);assert a.shape==(640,640,3);rgb.append(a)
        for k in range(8):
            a96,v=crop(a,q[k],L[k]);assert np.array_equal(a96,crops[si,k])and np.array_equal(v,valid[si,k])
            assert np.array_equal(a96.transpose(1,2,0),np.asarray(Image.open(OUT/f'{stage}_AUX{k}_RGB96_UNMARKED.png')))
    old=np.array(Image.open(protocol['previous_appearance_RGB']['path']).convert('RGB'),copy=True);difference=np.abs(rgb[0].astype(int)-old.astype(int))
    assert np.count_nonzero(difference)==result['previous_BEFORE_diff_channel_N']==2771
    # Independent camera-from-bounds check (no import of renderer camera code).
    inv=read(protocol['inventory']['path']);bounds=np.array(inv['stage']['bounds_world_m']);centre_world=bounds.mean(0);radius=np.linalg.norm(bounds[1]-bounds[0])/2;direction=np.array([1.25,.95,1.4]);direction/=np.linalg.norm(direction)
    own_eye=centre_world+direction*(1.3*radius/np.sin(np.deg2rad(25)));zz=(centre_world-own_eye);zz/=np.linalg.norm(zz);xx=np.cross(zz,[0,1,0]);xx/=np.linalg.norm(xx);yy=np.cross(zz,xx);ownR=np.stack([xx,yy,zz])
    assert np.allclose(own_eye,eye,atol=1e-12,rtol=0)and np.allclose(ownR,R,atol=1e-12,rtol=0)and np.allclose(-ownR@own_eye,t,atol=1e-12,rtol=0)
    with np.load(protocol['geometry']['path'],allow_pickle=False)as z:vertices=z['vertices_world_m'];tri=z['triangles_global'];pid=z['triangle_asset_local_part_id']
    orientation=np.array([1 if a['USD_orientation']=='rightHanded'else-1 for a in inv['per_mesh']])[pid];two=np.array([a['double_sided']for a in inv['per_mesh']])[pid]
    source_oracle=Oracle(vertices[tri],pid,orientation,two);oracles=[source_oracle,Oracle(np.r_[vertices[tri],cover[faces]],np.r_[pid,np.full(12,308)],np.r_[orientation,np.ones(12)],np.r_[two,np.zeros(12,bool)]),Oracle(np.r_[vertices[tri],aside[faces]],np.r_[pid,np.full(12,308)],np.r_[orientation,np.ones(12)],np.r_[two,np.zeros(12,bool)])]
    comparisons=[];disagreements=[];selected_states={};depthdelta=[]
    for k in [0,7]:
        g=refs[k]['geometry'];points=np.r_[np.array(g['endpoint_world'])[None],np.array(g['face_probe_points']).reshape(-1,3),np.array(g['edge_probe_points']).reshape(-1,3)]
        source=[source_oracle.cast(eye,p.copy(),refs[k]['part_id'])for p in points]
        pp,pz=projection(points,R,t,K);pixels=np.floor(pp+.5).astype(int)
        for si,stage in enumerate(stages):
            own=source if si==0 else[oracles[si].cast(eye,p.copy(),refs[k]['part_id'])for p in points]
            saved=teacher['stages'][stage][k]['combined_probe_states'];state=reduce_state(source,own,g,eye);assert state==teacher['stages'][stage][k]['underlying_geometric_state']
            if k==7:selected_states[stage]=state
            for j,(x,y)in enumerate(zip(own,saved)):
                if x['state']!=y['state']or x['owners']!=y['owners']:disagreements.append(dict(aux=k,stage=stage,probe=j,independent=x,saved=y))
            comparisons.append(dict(aux=k,stage=stage,state=state,probe_count=len(points)))
        if k==7:
            xx=pixels[:,0];yy=pixels[:,1];depthdelta=[float(np.max(np.abs(rgbdepth[1,yy,xx]-front))),float(np.max(np.abs(rgbdepth[0,yy,xx]-rgbdepth[2,yy,xx])))]
            # The original extra1e-4 exact-depth assertion failed and remains
            # archived. No larger tolerance substitutes for it. The installed
            # 24bit/F32 readback reconstruction is reported separately.
            assert np.all(owners[1,yy,xx]==308)
            assert np.array_equal(rgbdepth[0,yy,xx],rgbdepth[2,yy,xx])
    assert selected_states=={'BEFORE':'D','COVER':'E','SHAM':'D'}
    expected=['S','D','U','U','D','D','D','D']
    for stage in stages:
        actual=[x['state']for x in teacher['stages'][stage]];want=expected.copy()
        if stage=='COVER':want[7]='E'
        assert actual==want==result['states'][stage]
    numerical=dict(completed=True,PASS=not disagreements,code=bind(__file__),render_seal=bind(OUT/'RENDER_SEAL.json'),file_SHA_bindings_checked=len(checked),all8_qL_association_unchanged=True,all24_RGB96_valid_and_unmarked_PNG_exact=True,
        all_foreign_shape_material_depth_orientation_same_translation_only=True,cover_world_vertices_own_reconstruction_bitexact=True,cover_patch_front_margin_min_px=float(margins.min()),full_projected_sham_cube_patch_horizontal_clearance_px=horizontal_clear,
        full_cuboid_projected_bbox_cover=np.r_[c_uv.min(0),c_uv.max(0)].tolist(),full_cuboid_projected_bbox_sham=np.r_[s_uv.min(0),s_uv.max(0)].tolist(),
        independent_ray_plane_oracle_comparisons=comparisons,independent_probe_records=222,ray_owner_state_disagreements=disagreements,AUX7_states=selected_states,self0_separate=True,
        selected_patch_nearest_RGB_depth_cover_max_abs_vs_front=depthdelta[0],selected_patch_RGB_depth_before_sham_max_abs=depthdelta[1],
        posthoc_RGB_depth_exactness_check_pass=False,posthoc_depth_discrepancy_preserved=True,all_zero_numeric_discrepancies=False,depth_precision_diagnostic=bind(HERE/'DEPTH_READBACK_PRECISION_DIAGNOSTIC.json'),
        previousV96_BEFORE_drift=dict(different_pixels=int(np.any(difference>0,axis=2).sum()),different_channels=int(np.count_nonzero(difference)),max_abs_channel=int(difference.max()),mean_abs_channel=float(difference.mean()),cause='UNRESOLVED_NO_RETRY'),
        new_renderer_model_features_fit_private_current_PnP=0)
    save(HERE/'SAVED_NUMERICAL_REPLAY.json',numerical);print(json.dumps(numerical,indent=2))
if __name__=='__main__':main()
