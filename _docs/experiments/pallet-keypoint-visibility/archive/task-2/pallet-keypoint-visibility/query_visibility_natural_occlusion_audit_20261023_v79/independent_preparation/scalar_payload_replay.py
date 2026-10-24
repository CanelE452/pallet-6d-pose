"""Independent scalar native mask/grid maps, following the frozen policy."""
import hashlib,math
import numpy as np
from independent_grid_math import grid_scalar,roi_inverse_float32_phase

def replay(f,v,q,L,n,cache):
 h,w=f.shape;cc=np.zeros((8,3,3),np.int32);vc=np.zeros((8,7),np.int32);unique=np.zeros((8,3),np.int32);fp=[[],[]];half=np.zeros(8,np.int32);algebra=np.zeros(8,np.int32);grids=[];natives=[];maps=[]
 for k in range(8):
  g=np.array(grid_scalar(q[k],L,n));grids.append(g);good=np.isfinite(q[k]).all()and not(q[k]==-1).all();sets=np.zeros((7,n,n),bool);full=np.zeros((n,n),bool);vis=full.copy();pixels=np.full((n,n,2),-10**8,int)
  for i in range(n):
   for j in range(n):
    x,y=g[i,j];finite=math.isfinite(x)and math.isfinite(y);nx=math.floor(x+.5)if finite else-10**8;ny=math.floor(y+.5)if finite else-10**8;pixels[i,j]=[nx,ny]
    cont=finite and 0<=x<w and 0<=y<h;nn=finite and 0<=nx<w and 0<=ny<h;native=cont and good and nn;cached=bool(cache[k,i,j]);sets[:,i,j]=[cont,nn,cont and good,native,cached,cached and nn,native and cached]
    if nn:
     full[i,j]=f[ny,nx];vis[i,j]=v[ny,nx]
     half[k]+=min(abs((x+.5)-round(x+.5)),abs((y+.5)-round(y+.5)))<=1e-6
     if n==9:
      ax=np.float64(q[k,0])+(np.float64(j)-4)*L/9;ay=np.float64(q[k,1])+(np.float64(i)-4)*L/9;algebra[k]+=(math.floor(ax+.5)!=nx or math.floor(ay+.5)!=ny)
    for si,di in enumerate([3,5,6]):
     if sets[di,i,j]:cc[k,si,0 if vis[i,j]else(1 if full[i,j]else 2)]+=1
  vc[k]=sets.sum((1,2));
  for si,di in enumerate([3,5,6]):unique[k,si]=len({(int(pixels[i,j,0]),int(pixels[i,j,1]))for i,j in zip(*np.nonzero(sets[di]))})
  for ri,di in enumerate([3,6]):
   bits=np.stack([sets[di],full&sets[di],vis&sets[di]]).ravel(order='C');fp[ri].append(hashlib.sha256(f'V79_MAP_V1|grid{n}'.encode()+np.packbits(bits,bitorder='little').tobytes()).hexdigest())
  natives.append(sets[3]);maps.append((full,vis,pixels))
 return dict(class_counts=cc,valid_counts=vc,unique_native_pixels=unique,gt_map_native_sha256=np.array(fp[0],dtype='S64'),gt_map_intersection_sha256=np.array(fp[1],dtype='S64'),near_NN_half_tie_counts=half,literal_vs_algebra_NN_changes=algebra),np.array(grids),np.array(natives),maps

def centers(f,v,q):
 h,w=f.shape;out=dict(q_center_continuous_valid=np.zeros(8,bool),q_center_NN_bounds_valid=np.zeros(8,bool),q_center_mask_valid=np.zeros(8,bool),q_center_M0=np.full(8,-1,np.int8),q_center_M4=np.full(8,-1,np.int8),q_center_support_state=np.full(8,-1,np.int8),q_round_raw3x3_valid=np.zeros((8,3,3),bool),q_round_raw3x3_M0=np.full((8,3,3),-1,np.int8),q_round_raw3x3_M4=np.full((8,3,3),-1,np.int8))
 for k,(x,y)in enumerate(q):
  good=np.isfinite([x,y]).all()and not(x==-1 and y==-1)
  if not good:continue
  nx=math.floor(x+.5);ny=math.floor(y+.5);cont=0<=x<w and 0<=y<h;nn=0<=nx<w and 0<=ny<h;valid=cont and nn
  out['q_center_continuous_valid'][k]=cont;out['q_center_NN_bounds_valid'][k]=nn;out['q_center_mask_valid'][k]=valid
  if valid:
   out['q_center_M0'][k]=f[ny,nx];out['q_center_M4'][k]=v[ny,nx];out['q_center_support_state'][k]=1 if v[ny,nx]else(0 if f[ny,nx]else 2)
  for i,dy in enumerate([-1,0,1]):
   for j,dx in enumerate([-1,0,1]):
    if 0<=nx+dx<w and 0<=ny+dy<h:
     out['q_round_raw3x3_valid'][k,i,j]=True;out['q_round_raw3x3_M0'][k,i,j]=f[ny+dy,nx+dx];out['q_round_raw3x3_M4'][k,i,j]=v[ny+dy,nx+dx]
 return out
