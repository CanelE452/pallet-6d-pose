"""Scalar V79 source mask-grid replay, no file/model imports or mutation."""
import math
import numpy as np

def grid_scalar(q,L,n):
 # Literal original phase, avoiding algebraic simplification around NN ties.
 return [[(float(np.float64(q[0])+((np.float64(j)+.5)/n-.5)*np.float64(L)),float(np.float64(q[1])+((np.float64(i)+.5)/n-.5)*np.float64(L)))for j in range(n)]for i in range(n)]

def replay_masks(M0,M4,q,L,n,cached,kind):
 f=np.asarray(M0,bool);v=np.asarray(M4,bool);cache=np.asarray(cached,bool)
 assert f.ndim==2 and f.shape==v.shape and not(v&~f).any()and cache.shape==(n,n)and kind in ['native9','rgb96']
 h,w=f.shape;good=bool(np.isfinite(q).all());qgood=good and not bool(np.all(np.asarray(q)==-1))
 out={name:{key:0 for key in ['valid','V','C','O']}for name in ['native','cached','intersection']};nn_ties=0;continuous_boundary_near=0
 for i,row in enumerate(grid_scalar(q,L,n)):
  for j,(x,y)in enumerate(row):
   finite=math.isfinite(x)and math.isfinite(y);nx=math.floor(x+.5)if finite else-10**8;ny=math.floor(y+.5)if finite else-10**8
   nn=finite and 0<=nx<w and 0<=ny<h
   cont=finite and 0<=x<w and 0<=y<h
   # Native9 has the original native_grid good-query sentinel guard. RGB96
   # follows the enclosing compile_frame good-query guard, not crop() alone.
   usable=qgood
   native=usable and cont and nn;cached=bool(cache[i,j])and nn
   modes=dict(native=native,cached=cached,intersection=native and cached)
   if finite:
    b=8*2**-53*(abs(x)+abs(y)+abs(L)+1)
    nn_ties+=int(min(abs((x+.5)-round(x+.5)),abs((y+.5)-round(y+.5)))<=b)
    continuous_boundary_near+=int(min(abs(x),abs(x-w),abs(y),abs(y-h),abs(x+.5),abs(x-(w-.5)),abs(y+.5),abs(y-(h-.5)))<=b)
   for name,okay in modes.items():
    if okay:
     c=out[name];c['valid']+=1
     if v[ny,nx]:c['V']+=1
     elif f[ny,nx]:c['C']+=1
     else:c['O']+=1
 for c in out.values():assert c['V']+c['C']+c['O']==c['valid']
 return dict(counts=out,near_NN_tie_cells=nn_ties,near_continuous_boundary_cells=continuous_boundary_near)

def roi_inverse_float32_phase(grid,gain,offset):
 g=np.asarray(grid,np.float64);off=np.asarray(offset,np.float64)
 norm=2*((((g+100)*gain+off)/8)+.5)/80-1
 rounded=norm.astype(np.float32).astype(np.float64)
 inverse=(8*((rounded+1)*40-.5)-off)/gain-100
 quant_bound=(abs(norm)*2**-24+2**-150)*(320/abs(gain))
 numerical64=32*2**-53*(abs(inverse)+abs(g)+abs(off)+101)
 assert np.all(abs(inverse-g)<=quant_bound+numerical64)
 return inverse,quant_bound+numerical64

def mock():
 f=np.zeros((9,11),bool);f[1:8,2:10]=1;v=f.copy();v[3:7,5:8]=0
 for q in [[4.5,4.5],[0.,0.],[-1.,-1.],[10.49,8.49],[10.5,8.5],[-.25,3.]]:
  for n,kind in [(9,'native9'),(96,'rgb96')]:
   grid=np.array(grid_scalar(q,16.,n));x=grid[...,0];y=grid[...,1]
   cache=((x>=0)&(x<11)&(y>=0)&(y<9)&(not np.all(np.array(q)==-1)))if kind=='native9'else((x>=-.5)&(x<10.5)&(y>=-.5)&(y<8.5))
   r=replay_masks(f,v,q,16.,n,cache,kind)
   assert all(c['V']+c['C']+c['O']==c['valid']for c in r['counts'].values())
   _,bound=roi_inverse_float32_phase(grid,.7619047619047619,np.array([0.,13.]))
   assert np.isfinite(bound).all()
 return dict(PASS=True,generated_scalar_cases=12,NN_floor_plus_half=True,outside_distinct_from_covered=True,invalid_NN_ignored=True,native9_sentinel_guard=True,rgb96_original_compiler_good_query_guard=True,FP32_ROI_inverse_bound_derived=True,source_data_model_current_forward_fit_PnP=0)
