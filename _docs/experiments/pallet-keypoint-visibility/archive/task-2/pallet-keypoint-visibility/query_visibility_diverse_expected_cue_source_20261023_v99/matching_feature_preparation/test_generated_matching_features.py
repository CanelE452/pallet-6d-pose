"""Generated arrays only; no source RGB, feature, weight or model constructor."""
from pathlib import Path
import argparse,ast,copy,importlib.util,json
import numpy as np
import produce_matching_features_once as p

def scalar_crop(rgb,q,L,n,iy,ix):
 h,w=rgb.shape[:2];xy=np.asarray(q)+((np.array([ix,iy])+.5)/n-.5)*L
 if not(-.5<=xy[0]<w-.5 and-.5<=xy[1]<h-.5):return np.zeros(3,np.uint8),False
 lo=np.floor(xy).astype(int);f=xy-lo;acc=np.zeros(3)
 for dy in(0,1):
  for dx in(0,1):
   x,y=lo+[dx,dy]
   if 0<=x<w and 0<=y<h:acc+=rgb[y,x]*((f[0]if dx else 1-f[0])*(f[1]if dy else 1-f[1]))
 return np.floor(np.clip(acc,0,255)+.5).astype(np.uint8),True

def test_all():
 math_path=p.P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
 assert p.sha(math_path)=='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
 m=p.module(math_path,'generated_v90_math');rng=np.random.default_rng(194)
 rgb=rng.integers(0,256,(640,640,3),dtype=np.uint8)
 plane=np.broadcast_to((np.arange(80)[None,:]+np.arange(80)[:,None]*.01)[None],(64,80,80)).astype(np.float16).copy()
 q=np.array([[320,320],[.1,.1],[639.7,639.7],[-20,40],[650,250],[0,639],[639,0],[122.125,446.875]],np.float64)
 L=np.array([112,16,80,96,72,99,64,123.25]);oldq=q.copy();oldL=L.copy();a=m.expected_affine([640,640])
 x,f=p.pack_frame(rgb,plane,q,L,m,a['gain'],a['offset'])
 assert set(x)==set(p.FIELDS)and x['p3_roi'].shape==(8,65,9,9)
 assert x['p3_roi'].dtype==np.float16 and x['rgb96'].dtype==np.uint8 and x['crop_valid96'].dtype==bool
 assert np.array_equal(q,oldq)and np.array_equal(L,oldL)
 assert np.isfinite(x['p3_roi']).all()and not x['p3_roi'][:,:64][np.broadcast_to(~x['p3_roi'][:,64:65].astype(bool),(8,64,9,9))].any()
 points=[(0,0),(47,48),(95,95)];n=0
 for k in range(8):
  for iy,ix in points:
   value,v=scalar_crop(rgb,q[k],L[k],96,iy,ix)
   assert np.array_equal(value,x['rgb96'][k,:,iy,ix])and v==x['crop_valid96'][k,0,iy,ix];n+=1
 # Scalar P3 oracle at the FP32 normalized-grid coordinates actually passed to
 # grid_sample. F16 comparison permits only rounding-sized feature difference.
 grid,valid=m.native_grid_explicit(q,L,[640,640]);idx=(grid+100)*a['gain']/8
 norm=(2*(idx+.5)/80-1).astype(np.float32);ridx=((norm.astype(np.float64)+1)*80-1)/2
 maxdiff=0
 for k in range(8):
  for iy,ix in[(0,0),(4,4),(8,8)]:
   pos=ridx[k,iy,ix];lo=np.floor(pos).astype(int);frac=pos-lo;v=0.
   if valid[k,iy,ix]:
    for dy in(0,1):
     for dx in(0,1):
      xx,yy=lo+[dx,dy]
      if 0<=xx<80 and 0<=yy<80:v+=float(plane[0,yy,xx])*((frac[0]if dx else 1-frac[0])*(frac[1]if dy else 1-frac[1]))
   maxdiff=max(maxdiff,abs(float(x['p3_roi'][k,0,iy,ix])-float(np.float16(v))))
 assert maxdiff<=.03125 # one FP16 ULP at this generated feature range
 assert not f['query_center_valid'][2]and not f['query_center_valid'][3]and not f['query_center_valid'][4]
 # Strict complete source order, including all stages and logical AUX slots.
 ims=[];qs=[];os=[]
 for bi in range(96):
  split='VALIDATION'if bi//8 in[2,5,8,11]else'TRAIN';base=f'B{bi:03d}';qs.append(dict(base_id=base,split=split,query8_xy=q.tolist(),L=L.tolist()))
  for si,st in enumerate(p.STAGES):
   fi=bi*3+si;im=dict(image_index=fi,base_index=bi,base_id=base,variant=st,split=split,raw_hw=[640,640],image_id=f'{base}_{st}');ims.append(im)
   for k in range(8):os.append(dict(occurrence_index=fi*8+k,image_index=fi,aux_corner_index=k,image_id=im['image_id'],base_index=bi,variant=st,split=split,query_xy=q[k].tolist(),L=float(L[k]),native8_mapping='UNKNOWN',source_emulated_not_initializer=True))
 summary=p.validate_metadata(ims,qs,os);assert summary==dict(frames=288,queries=2304,TRAIN=1536,VALIDATION=768)
 reject=0
 for what in('q','L','slot','order','split'):
  bad=copy.deepcopy(os)
  if what=='q':bad[1]['query_xy'][0]+=.1
  elif what=='L':bad[1]['L']+=1
  elif what=='slot':bad[1]['aux_corner_index']=2
  elif what=='order':bad[1]['occurrence_index']=3
  else:bad[1]['split']='VALIDATION'
  try:p.validate_metadata(ims,qs,bad)
  except AssertionError:reject+=1
  else:raise AssertionError('Malformed metadata accepted:'+what)
 tree=ast.parse(Path(p.__file__).read_text());run=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='run')
 assert isinstance(run.body[0],ast.Assign)and ast.unparse(run.body[0].value)=='authorize(entry)'
 read_calls=[ast.unparse(x)for x in ast.walk(run)if isinstance(x,ast.Call)and ast.unparse(x.func)in('rows','read','np.load','Image.open')]
 assert all('GT'not in x and 'gt'not in x for x in read_calls) # GT binding export is allowed; GT body reads are not.
 return dict(PASS=True,generated_RGB_scalar_positions=n,P3_scalar_positions=24,P3_max_F16_scalar_diff=maxdiff,all2304_metadata_and_rejects=reject,input_keys=list(p.FIELDS),q_L_unchanged=True,finite_OOB_retained=True,channel_valid_and_zero_mask=True,model_ctor_forward_source_RGB_features_weights_GT_current=0)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=test_all();r.update(completed=True,code=p.bind(p.__file__),test=p.bind(__file__));p.write(a.output,r);print(json.dumps(r))
