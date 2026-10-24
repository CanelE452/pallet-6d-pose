"""Pure opaque donor geometry, no source/current/model/file I/O."""
import numpy as np
from scipy.ndimage import distance_transform_edt

SAFETY_EPS=1e-6

def donor_geometry(mask):
    m=np.asarray(mask,bool)
    if m.ndim!=2 or not m.any():raise ValueError('Nonempty binary donor required')
    yy,xx=np.nonzero(m);box=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],int)
    x0,y0,x1,y1=box;crop=np.pad(m[y0:y1,x0:x1],1)
    edt=distance_transform_edt(crop);flat=int(edt.argmax());cy,cx=np.unravel_index(flat,edt.shape)
    d=float(edt[cy,cx]);assert d>=1
    return dict(mask=crop,box=box,center_xy=np.array([cx,cy],float),interior_distance=d)

def rotate(mask,rgb,center_xy,quarterturn):
    m=np.asarray(mask,bool);image=np.asarray(rgb,np.uint8);c=np.asarray(center_xy,float).copy()
    assert image.shape==(*m.shape,3)and quarterturn in range(4)
    for _ in range(quarterturn):
        w=m.shape[1];c=np.array([c[1],w-1-c[0]],float)
        m=np.rot90(m);image=np.rot90(image)
    return m,image,c

def compose(rgb,donor_rgb,donor_mask,donor_center_xy,interior_distance,reference_xy,rho_px):
    background=np.asarray(rgb,np.uint8);image=np.asarray(donor_rgb,np.uint8);m=np.asarray(donor_mask,bool)
    assert background.ndim==3 and background.shape[2]==3 and image.shape==(*m.shape,3)
    r=np.asarray(reference_xy,float);c=np.asarray(donor_center_xy,float)
    assert r.shape==c.shape==(2,)and np.isfinite(r).all()and np.isfinite(c).all()
    assert np.isfinite(rho_px)and rho_px>=4 and interior_distance>=1
    radius=float(rho_px)+2.;scale=radius/(float(interior_distance)-np.sqrt(2)/2-SAFETY_EPS)
    yy,xx=np.nonzero(m);lo=np.array([xx.min()-.5,yy.min()-.5]);hi=np.array([xx.max()+.5,yy.max()+.5])
    nominal=np.r_[r+(lo-c)*scale,r+(hi-c)*scale]
    h,w=background.shape[:2]
    x0=max(0,int(np.floor(nominal[0])));y0=max(0,int(np.floor(nominal[1])))
    x1=min(w,int(np.ceil(nominal[2]))+1);y1=min(h,int(np.ceil(nominal[3]))+1)
    output=background.copy();pasted=np.zeros((h,w),bool)
    if x1>x0 and y1>y0:
        px,py=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1))
        sx=c[0]+(px-r[0])/scale;sy=c[1]+(py-r[1])/scale
        nx=np.floor(sx+.5).astype(np.int64);ny=np.floor(sy+.5).astype(np.int64)
        valid=(nx>=0)&(nx<m.shape[1])&(ny>=0)&(ny<m.shape[0])
        opaque=np.zeros(nx.shape,bool);opaque[valid]=m[ny[valid],nx[valid]]
        ix=np.floor(sx).astype(np.int64);iy=np.floor(sy).astype(np.int64);dx=sx-ix;dy=sy-iy
        color=np.zeros((*sx.shape,3),float)
        for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
            x=ix+ax;y=iy+ay;v=(x>=0)&(x<m.shape[1])&(y>=0)&(y<m.shape[0])
            color[v]+=image[y[v],x[v]].astype(float)*weight[v,None]
        color=np.floor(np.clip(color,0,255)+.5).astype(np.uint8)
        output[y0:y1,x0:x1][opaque]=color[opaque];pasted[y0:y1,x0:x1]=opaque
    yy,xx=np.ogrid[:h,:w];disk=(xx-r[0])**2+(yy-r[1])**2<=radius**2
    covered=bool(np.all(pasted[disk]))
    return output,pasted,dict(scale=scale,radius_px=radius,donor_center_xy=c,reference_xy=r,
        donor_interior_distance=float(interior_distance),nominal_extent_xyxy=nominal,
        rendered_extent_xyxy=[x0,y0,x1,y1],disk_pixels=int(disk.sum()),disk_pasted_pixels=int(np.sum(pasted&disk)),
        disk_all_covered=covered,pasted_pixels=int(pasted.sum()),destination_out_of_bounds=bool((nominal[:2]<0).any()or nominal[2]>w-1 or nominal[3]>h-1))
