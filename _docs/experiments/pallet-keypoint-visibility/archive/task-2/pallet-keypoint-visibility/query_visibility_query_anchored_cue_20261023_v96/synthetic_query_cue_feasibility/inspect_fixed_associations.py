"""Small saved-image/geometry association inspection, not a renderer or GT build."""
from pathlib import Path
import itertools,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def save(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def crop(rgb,q,L):
    t=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L;x,y=np.meshgrid(q[0]+t,q[1]+t)
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);dx=x-ix;dy=y-iy;h,w=rgb.shape[:2];z=np.zeros((96,96,3),np.float64)
    for ox,oy,wt in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        xx=ix+ox;yy=iy+oy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);z[ok]+=rgb[yy[ok],xx[ok]].astype(np.float64)*wt[ok,None]
    good=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5);z=np.floor(np.clip(z,0,255)+.5).astype(np.uint8);z[~good]=0;return z,good
def main():
    s=read(HERE/'SCOPE_BEFORE_SMALL_ASSOCIATION_CALCULATION.json')
    for k in ['geometry','inventory','saved_RGB','saved_owner','saved_appearance_protocol']:assert sha(s[k]['path'])==s[k]['sha256']
    inv=read(s['inventory']['path']);protocol=read(s['saved_appearance_protocol']['path']);cam=protocol['camera']
    with np.load(s['geometry']['path'],allow_pickle=False)as f:vertices=f['vertices_world_m'];offsets=f['mesh_vertex_offsets'];toff=f['mesh_triangle_offsets']
    with np.load(s['saved_owner']['path'],allow_pickle=False)as f:owner=f['asset_local_part_id'];ambiguity=f['ownership_ambiguity']
    bounds=np.array(inv['stage']['bounds_world_m']);corners=np.array([[bounds[int(a),0],bounds[int(b),1],bounds[int(c),2]]for a,b,c in itertools.product([0,1],repeat=3)])
    R=np.array(cam['R']);t=np.array(cam['t']);K=np.array(cam['K'])
    def proj(x):
        c=np.asarray(x)@R.T+t;return (c@K.T)[...,:2]/c[...,2:3]
    q=proj(corners);L=float(np.clip(.25*np.ptp(q,axis=0).max(),16,640));diag=float(np.linalg.norm(bounds[1]-bounds[0]))
    candidates=[i for i in range(308)if offsets[i+1]-offsets[i]==24 and toff[i+1]-toff[i]==12]
    rgb=np.array(Image.open(s['saved_RGB']['path']).convert('RGB'),copy=True)
    out=[];contact=Image.new('RGB',(1024,584),'white');draw=ImageDraw.Draw(contact);overview=Image.fromarray(rgb.copy());od=ImageDraw.Draw(overview)
    for k,(ref,qq)in enumerate(zip(corners,q)):
        ds=[];vv=[]
        for pid in candidates:
            p=vertices[offsets[pid]:offsets[pid+1]];d2=((p-ref)**2).sum(1);j=int(np.argmin(d2));ds.append(float(d2[j]));vv.append(p[j])
        min2=min(ds);tol=64*np.finfo(np.float64).eps*max(1,min2);ids=[candidates[j]for j,d in enumerate(ds)if d<=min2+tol]
        assert len(ids)==1,'Retain distinct-part tie explicitly rather than choose a visible alternative'
        pid=ids[0];idx=candidates.index(pid);point=vv[idx];pp=proj(point);cx,cy=np.floor(qq+.5).astype(int);px,py=np.floor(pp+.5).astype(int)
        h,w=owner.shape;patch=owner[max(0,py-4):min(h,py+5),max(0,px-4):min(w,px+5)]
        image,valid=crop(rgb,qq,L);path=HERE/f'corner_{k:02d}_RGB96_RAW.png';Image.fromarray(image).save(path)
        col,row=k%4,k//4;x=col*256;y=row*292;panel=Image.fromarray(image).resize((240,240),Image.Resampling.NEAREST);contact.paste(panel,(x+8,y+36))
        draw.text((x+8,y+4),f'AUX{k} -> part{pid}; q stays fixed',fill='black');draw.text((x+8,y+20),f'offset {np.linalg.norm(pp-qq):.2f}px; L={L:.1f}',fill='black')
        draw.ellipse((x+8+117,y+36+117,x+8+123,y+36+123),outline='red',width=2)
        local=(pp-qq)/L*240+120
        if np.all((local>=0)&(local<240)):
            xx=x+8+float(local[0]);yy=y+36+float(local[1]);draw.ellipse((xx-4,yy-4,xx+4,yy+4),outline='lime',width=2)
        od.ellipse((qq[0]-3,qq[1]-3,qq[0]+3,qq[1]+3),outline='red',width=2);od.text((qq[0]+4,qq[1]),str(k),fill='red')
        od.line([tuple(qq),tuple(pp)],fill='lime',width=1)
        out.append(dict(aux_corner_index=k,sign_xyz=[-1 if not z else 1 for z in list(itertools.product([0,1],repeat=3))[k]],virtual_reference_world=ref.tolist(),original_aux_q=qq.tolist(),L=L,
            association_part_id=pid,association_prim_path=inv['per_mesh'][pid]['prim_path'],association_world_vertex=point.tolist(),world_separation=float(np.sqrt(min2)),
            projected_structural_vertex=pp.tolist(),projected_q_to_cue_distance_px=float(np.linalg.norm(pp-qq)),q_center_owner=int(owner[cy,cx])if 0<=cx<w and 0<=cy<h else'OUT_OF_FRAME',
            cue_neighbourhood9x9_owner_counts={str(int(z)):int((patch==z).sum())for z in np.unique(patch)},associated_part_visible_pixels_in_saved_image=int((owner==pid).sum()),
            original_q_crop_valid_pixels=int(valid.sum()),associated_vertex_inside_original_crop=bool(np.all((pp-qq>=-.5*L)&(pp-qq<.5*L))),RGB96_RAW=bind(path),
            endpoint_patch_radius_world_proposal=.02*diag,endpoint_patch_VIS_DES='UNKNOWN_NOT_RASTERIZED',native8_mapping='UNKNOWN'))
    cp=HERE/'ALL8_FIXED_AUX_ASSOCIATION_CONTACT.png';contact.save(cp);op=HERE/'SAVED_RGB_WITH_SEPARATE_Q_AND_STRUCTURAL_VERTEX.png';overview.save(op)
    v93=P/'query_visibility_synthetic_part_teacher_20261023_v93';prior=read(v93/'raster_prototype/FIXED_PART152_SIX_CAMERAS/GEOMETRY_DIAGNOSTIC_RESULTS.json')
    save(HERE/'SMALL_ASSOCIATION_MEASUREMENTS.json',dict(completed=True,code=bind(__file__),scope=bind(HERE/'SCOPE_BEFORE_SMALL_ASSOCIATION_CALCULATION.json'),structural_box_candidate_part_ids=candidates,
        references=out,contact=bind(cp),overview=bind(op),prior_part152_geometry_summary_binding=bind(v93/'raster_prototype/FIXED_PART152_SIX_CAMERAS/GEOMETRY_DIAGNOSTIC_RESULTS.json'),
        new_renders=0,source_model_features_fit_current_PnP=0,labels_admitted=0))
    print(json.dumps({'completed':True,'candidate_parts':len(candidates),'L':L,'associations':[{k:r[k]for k in ['aux_corner_index','association_part_id','world_separation','projected_q_to_cue_distance_px','q_center_owner','associated_part_visible_pixels_in_saved_image','cue_neighbourhood9x9_owner_counts']}for r in out]},indent=2))
if __name__=='__main__':main()
